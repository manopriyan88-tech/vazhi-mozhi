"""Unit tests for MultiAgentOrchestrator pipeline."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from app.agents.orchestrator import (
    APITask,
    EndpointSpec,
    KitchenAPI,
    MultiAgentOrchestrator,
    Plan,
    SynthesizedResponse,
    TaskExecutionResult,
)
from app.services.http_executor import APIResponse
from app.services.resilience import ExecutionResult, ResilientHTTPExecutor


class TestMultiAgentOrchestrator(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying Planner, Parallel Wave Executor, and Synthesizer agents."""

    def setUp(self):
        self.weather_api = KitchenAPI(
            api_name="weather_service",
            base_url="https://api.weather.com",
            description="Global real-time weather and forecasts",
            api_key="mock_weather_key",
            auth_type="api_key_query",
            auth_param_name="appid",
            endpoints=[
                EndpointSpec(
                    path="/v1/current",
                    method="GET",
                    summary="Current weather by city",
                    parameters={"city": "string"},
                ),
                EndpointSpec(
                    path="/v1/forecast",
                    method="GET",
                    summary="5-day forecast by city",
                    parameters={"city": "string", "days": "integer"},
                ),
            ],
        )

        self.crypto_api = KitchenAPI(
            api_name="crypto_service",
            base_url="https://api.crypto.com",
            description="Cryptocurrency prices and market stats",
            api_key="mock_crypto_key",
            auth_type="bearer",
            endpoints=[
                EndpointSpec(
                    path="/v2/ticker",
                    method="GET",
                    summary="Current spot prices",
                    parameters={"symbol": "string"},
                )
            ],
        )

        self.kitchen = [self.weather_api, self.crypto_api]

    async def test_planner_agent(self):
        """Planner Agent correctly prompts Gemini and extracts Plan with waves."""
        mock_gemini = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "goal": "Get Tokyo weather and BTC price concurrently",
            "reasoning": "Both APIs are available in kitchen and independent, so execute in Wave 1.",
            "tasks": [
                {
                    "task_id": "task_1",
                    "api_name": "weather_service",
                    "endpoint": "/v1/current",
                    "method": "GET",
                    "description": "Fetch current Tokyo weather",
                    "params": {"city": "Tokyo"},
                    "wave": 1,
                    "depends_on": [],
                },
                {
                    "task_id": "task_2",
                    "api_name": "crypto_service",
                    "endpoint": "/v2/ticker",
                    "method": "GET",
                    "description": "Fetch Bitcoin spot price",
                    "params": {"symbol": "BTCUSD"},
                    "wave": 1,
                    "depends_on": [],
                },
            ],
        })
        mock_gemini.aio.models.generate_content = AsyncMock(return_value=mock_response)

        events = []
        async def on_event(ev):
            events.append(ev)

        orchestrator = MultiAgentOrchestrator(gemini_client=mock_gemini)
        plan = await orchestrator.run_planner(
            prompt="What is the weather in Tokyo and the price of BTC right now?",
            kitchen=self.kitchen,
            event_callback=on_event,
        )

        self.assertEqual(len(plan.tasks), 2)
        self.assertEqual(plan.tasks[0].api_name, "weather_service")
        self.assertEqual(plan.tasks[1].api_name, "crypto_service")
        self.assertEqual(plan.tasks[0].wave, 1)
        self.assertEqual(plan.tasks[1].wave, 1)

        # Check emitted events
        event_types = [e["type"] for e in events]
        self.assertIn("planner_started", event_types)
        self.assertIn("plan_generated", event_types)

    async def test_executor_agent_parallel_execution(self):
        """Executor Agent runs wave tasks concurrently using asyncio.gather."""
        mock_resilient_executor = MagicMock(spec=ResilientHTTPExecutor)

        # Mock results for 2 tasks
        async def mock_execute(request, tool_schema=None, event_callback=None):
            # Short sleep to test parallelism
            await asyncio.sleep(0.01)
            if "weather" in request.url:
                return ExecutionResult(
                    success=True,
                    status_code=200,
                    data={"temp_c": 18.5, "condition": "Sunny"},
                )
            else:
                return ExecutionResult(
                    success=True,
                    status_code=200,
                    data={"symbol": "BTCUSD", "price": 95400.0},
                )

        mock_resilient_executor.execute = AsyncMock(side_effect=mock_execute)

        plan = Plan(
            goal="Fetch weather and crypto in parallel",
            reasoning="Parallel Wave 1 execution",
            tasks=[
                APITask(
                    task_id="t1",
                    api_name="weather_service",
                    endpoint="/v1/current",
                    description="Get weather",
                    params={"city": "Tokyo"},
                    wave=1,
                ),
                APITask(
                    task_id="t2",
                    api_name="crypto_service",
                    endpoint="/v2/ticker",
                    description="Get BTC",
                    params={"symbol": "BTCUSD"},
                    wave=1,
                ),
            ],
        )

        events = []
        async def on_event(ev):
            events.append(ev)

        orchestrator = MultiAgentOrchestrator(resilient_executor=mock_resilient_executor)
        results = await orchestrator.run_executor(plan, self.kitchen, event_callback=on_event)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.success for r in results))
        self.assertEqual(results[0].data["temp_c"], 18.5)
        self.assertEqual(results[1].data["price"], 95400.0)

        # Check wave events
        event_types = [e["type"] for e in events]
        self.assertIn("execution_wave_started", event_types)
        self.assertIn("task_started", event_types)
        self.assertIn("task_completed", event_types)
        self.assertIn("execution_wave_completed", event_types)

    async def test_synthesizer_agent(self):
        """Synthesizer Agent combines live 3rd-party data into structured final response."""
        mock_gemini = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "summary": "Tokyo is sunny at 18.5°C while Bitcoin trades at $95,400.",
            "content": "### Live Market & Weather Report\n- **Tokyo Weather**: Currently 18.5°C and sunny.\n- **Bitcoin**: Trading actively at $95,400 USD.",
            "key_insights": ["Tokyo temperature: 18.5°C", "BTC price: $95,400"],
            "apis_used": ["weather_service", "crypto_service"],
        })
        mock_gemini.aio.models.generate_content = AsyncMock(return_value=mock_response)

        plan = Plan(goal="Compare weather and crypto", reasoning="", tasks=[])
        task_results = [
            TaskExecutionResult(
                task_id="t1",
                api_name="weather_service",
                endpoint="/v1/current",
                method="GET",
                success=True,
                status_code=200,
                data={"temp_c": 18.5, "condition": "Sunny"},
            ),
            TaskExecutionResult(
                task_id="t2",
                api_name="crypto_service",
                endpoint="/v2/ticker",
                method="GET",
                success=True,
                status_code=200,
                data={"symbol": "BTCUSD", "price": 95400.0},
            ),
        ]

        orchestrator = MultiAgentOrchestrator(gemini_client=mock_gemini)
        synthesized: SynthesizedResponse = await orchestrator.run_synthesizer(
            prompt="Give me Tokyo weather and BTC price",
            plan=plan,
            task_results=task_results,
            total_time=0.45,
        )

        self.assertIn("Tokyo is sunny", synthesized.summary)
        self.assertIn("95,400", synthesized.content)
        self.assertEqual(len(synthesized.key_insights), 2)
        self.assertEqual(synthesized.apis_used, ["weather_service", "crypto_service"])

    async def test_full_pipeline_orchestration(self):
        """Tests complete end-to-end 3-stage pipeline."""
        mock_gemini = MagicMock()

        # 1st call for Planner, 2nd call for Synthesizer
        planner_resp = MagicMock()
        planner_resp.text = json.dumps({
            "goal": "Get Tokyo weather",
            "reasoning": "Call weather API",
            "tasks": [
                {
                    "task_id": "task_1",
                    "api_name": "weather_service",
                    "endpoint": "/v1/current",
                    "method": "GET",
                    "description": "Tokyo weather",
                    "params": {"city": "Tokyo"},
                    "wave": 1,
                    "depends_on": [],
                }
            ],
        })

        synth_resp = MagicMock()
        synth_resp.text = json.dumps({
            "summary": "Tokyo current temperature is 21°C.",
            "content": "The weather in Tokyo is currently clear at 21°C.",
            "key_insights": ["21°C in Tokyo"],
            "apis_used": ["weather_service"],
        })

        mock_gemini.aio.models.generate_content = AsyncMock(side_effect=[planner_resp, synth_resp])

        mock_resilient_executor = MagicMock(spec=ResilientHTTPExecutor)
        mock_resilient_executor.execute = AsyncMock(
            return_value=ExecutionResult(
                success=True,
                status_code=200,
                data={"temp_c": 21, "condition": "Clear"},
            )
        )

        events = []
        async def on_event(ev):
            events.append(ev)

        orchestrator = MultiAgentOrchestrator(
            gemini_client=mock_gemini,
            resilient_executor=mock_resilient_executor,
        )

        final_response = await orchestrator.orchestrate(
            prompt="What's the weather in Tokyo?",
            kitchen=self.kitchen,
            event_callback=on_event,
        )

        self.assertEqual(final_response.summary, "Tokyo current temperature is 21°C.")
        self.assertEqual(len(final_response.task_results), 1)
        self.assertTrue(final_response.task_results[0].success)

        # Check full event trajectory
        event_types = [e["type"] for e in events]
        self.assertIn("orchestrator_started", event_types)
        self.assertIn("planner_started", event_types)
        self.assertIn("plan_generated", event_types)
        self.assertIn("execution_wave_started", event_types)
        self.assertIn("task_started", event_types)
        self.assertIn("task_completed", event_types)
        self.assertIn("execution_wave_completed", event_types)
        self.assertIn("synthesis_started", event_types)
        self.assertIn("synthesis_completed", event_types)
        self.assertIn("orchestrator_completed", event_types)


if __name__ == "__main__":
    unittest.main()
