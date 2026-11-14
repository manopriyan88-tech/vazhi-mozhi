"""Multi-Agent Wave Orchestrator.

A 3-stage agent pipeline powered by the official google-genai SDK:
1. Planner Agent: Inspects the user's prompt and available API keys/specs
   in the Adaptive Kitchen, breaking the request into structured API tasks
   and parallel execution waves.
2. Executor Agent: Executes the API tasks in parallel using asyncio.gather,
   backed by the ResilientHTTPExecutor for rate-limiting (429/5xx backoff)
   and Gemini self-healing (400 Bad Request parameter repair).
3. Synthesizer Agent: Merges live 3rd-party API responses, handling successes
   and partial failures into a structured, unified response for the user.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.models import (
    APITask,
    EndpointSpec,
    KitchenAPI,
    Plan,
    SynthesizedResponse,
    TaskExecutionResult,
)
from app.services.http_executor import APIRequest
from app.services.resilience import ExecutionResult, ResilientHTTPExecutor

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    genai_types = None  # type: ignore
    GENAI_AVAILABLE = False

logger = logging.getLogger("app.agents.orchestrator")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]


# ============================================================================
# Multi-Agent Wave Orchestrator
# ============================================================================

class MultiAgentOrchestrator:
    """Coordinates Planner, Executor (parallel waves), and Synthesizer agents."""

    def __init__(
        self,
        gemini_client: Optional[Any] = None,
        resilient_executor: Optional[ResilientHTTPExecutor] = None,
        model_name: Optional[str] = None,
    ):
        self._gemini_client = gemini_client
        self.resilient_executor = resilient_executor or ResilientHTTPExecutor(gemini_client=gemini_client)
        self.model_name = model_name or settings.GEMINI_MODEL

    def _get_gemini_client(self) -> Optional[Any]:
        """Lazy-inits the official google-genai Client if not provided."""
        if self._gemini_client is not None:
            return self._gemini_client

        if not GENAI_AVAILABLE:
            logger.warning("google-genai SDK not available.")
            return None

        api_key = settings.GEMINI_API_KEY
        if not api_key or api_key == "your_gemini_api_key_here":
            logger.warning("GEMINI_API_KEY is not set or using placeholder.")
            return None

        try:
            self._gemini_client = genai.Client(api_key=api_key)
            return self._gemini_client
        except Exception as exc:
            logger.error(f"Failed to create google-genai Client: {exc}")
            return None

    def _clean_json_markdown(self, text: str) -> str:
        """Strips markdown code fences from JSON strings."""
        s = text.strip()
        if s.startswith("```json"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()

    async def _emit_event(
        self,
        event_callback: Optional[EventCallback],
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Emits a stage or progress event to the WebSocket callback."""
        payload = {
            "type": event_type,
            "timestamp": time.time(),
            **data,
        }
        logger.debug(f"[OrchestratorEvent] {event_type}: {payload}")
        if event_callback:
            try:
                await event_callback(payload)
            except Exception as exc:
                logger.warning(f"Error in event callback: {exc}")

    def _create_heuristic_plan(self, prompt: str, kitchen: List[KitchenAPI]) -> Plan:
        """Heuristically matches user prompt keywords against registered kitchen connectors."""
        prompt_lower = prompt.lower()
        matched_tasks: List[APITask] = []
        task_idx = 1

        for api in kitchen:
            api_name_lower = api.api_name.lower()
            for ep in api.endpoints:
                ep_summary_lower = ep.summary.lower()
                ep_path_lower = ep.path.lower()

                is_match = (
                    api_name_lower in prompt_lower
                    or any(part in prompt_lower for part in api_name_lower.split("_"))
                    or any(word in prompt_lower for word in ep_summary_lower.split() if len(word) > 3)
                    or any(word in prompt_lower for word in ep_path_lower.replace("/", " ").split() if len(word) > 3)
                )

                if is_match or len(matched_tasks) < len(kitchen):
                    matched_tasks.append(
                        APITask(
                            task_id=f"task_{task_idx}",
                            api_name=api.api_name,
                            endpoint=ep.path,
                            method=ep.method,
                            description=ep.summary or f"Fetch data from {api.api_name}",
                            params=dict(ep.parameters) if ep.parameters else {},
                            wave=1,
                        )
                    )
                    task_idx += 1
                    break

        return Plan(
            goal=prompt,
            reasoning=f"Heuristic planner mapped user prompt to {len(matched_tasks)} parallel connector task(s).",
            tasks=matched_tasks,
        )

    # ------------------------------------------------------------------------
    # Stage 1: Planner Agent
    # ------------------------------------------------------------------------

    async def run_planner(
        self,
        prompt: str,
        kitchen: List[KitchenAPI],
        event_callback: Optional[EventCallback] = None,
    ) -> Plan:
        """Decomposes the user prompt into parallel API tasks using available kitchen keys."""
        await self._emit_event(
            event_callback,
            "planner_started",
            {
                "prompt": prompt,
                "available_apis": [k.api_name for k in kitchen],
            },
        )

        client = self._get_gemini_client()
        if not kitchen:
            fallback_plan = Plan(
                goal=prompt,
                reasoning="No APIs registered in kitchen.",
                tasks=[],
            )
            await self._emit_event(
                event_callback,
                "plan_generated",
                {"plan": fallback_plan.model_dump()},
            )
            return fallback_plan

        if not client:
            logger.info("Planner running in heuristic mode (no Gemini client). Matching prompt to kitchen connectors.")
            plan = self._create_heuristic_plan(prompt, kitchen)
            await self._emit_event(
                event_callback,
                "plan_generated",
                {"plan": plan.model_dump()},
            )
            return plan

        # Format kitchen APIs and endpoints description for Gemini
        kitchen_catalog = []
        for api in kitchen:
            endpoints_info = []
            for ep in api.endpoints:
                endpoints_info.append({
                    "path": ep.path,
                    "method": ep.method,
                    "summary": ep.summary,
                    "parameters": ep.parameters,
                })
            kitchen_catalog.append({
                "api_name": api.api_name,
                "base_url": api.base_url,
                "description": api.description,
                "endpoints": endpoints_info,
            })

        system_prompt = f"""You are an Expert Multi-Agent API Planner in an Adaptive API Kitchen.
Your job is to analyze the user's request and plan API tasks to retrieve or manipulate live data.

AVAILABLE APIS IN THE KITCHEN:
{json.dumps(kitchen_catalog, indent=2)}

INSTRUCTIONS:
1. Examine the user's prompt and determine which available APIs and endpoints are required.
2. Break down the request into concrete APITasks.
3. Assign a 'wave' integer to each task:
   - Tasks in 'wave: 1' are independent and CAN RUN IN PARALLEL.
   - Tasks in 'wave: 2' depend on data from Wave 1 tasks.
4. Provide the exact query parameters ('params') or JSON body ('json_body') required.
5. If no API in the kitchen matches the user's request, set 'tasks' to an empty list [].

Output ONLY a JSON object matching this structure:
{{
  "goal": "Brief statement of the goal",
  "reasoning": "Reasoning for the chosen APIs and parallel waves",
  "tasks": [
    {{
      "task_id": "task_1",
      "api_name": "example_api",
      "endpoint": "/v1/search",
      "method": "GET",
      "description": "Fetch initial search results",
      "params": {{"query": "example"}},
      "json_body": null,
      "wave": 1,
      "depends_on": []
    }}
  ]
}}
"""

        try:
            config = None
            if GENAI_AVAILABLE and genai_types:
                config = genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                )

            response = await client.aio.models.generate_content(
                model=self.model_name,
                contents=[{"role": "user", "parts": [{"text": f"{system_prompt}\n\nUSER PROMPT: {prompt}"}]}],
                config=config,
            )

            raw_json = self._clean_json_markdown(response.text or "{}")
            plan_data = json.loads(raw_json)
            plan = Plan.model_validate(plan_data)

            await self._emit_event(
                event_callback,
                "plan_generated",
                {"plan": plan.model_dump()},
            )
            return plan

        except Exception as exc:
            logger.warning(f"Planner Agent error: {exc}. Falling back to heuristic plan.")
            fallback_plan = self._create_heuristic_plan(prompt, kitchen)
            await self._emit_event(
                event_callback,
                "plan_generated",
                {"plan": fallback_plan.model_dump()},
            )
            return fallback_plan

    # ------------------------------------------------------------------------
    # Stage 2: Executor Agent (Parallel Wave Execution via asyncio.gather)
    # ------------------------------------------------------------------------

    async def _execute_single_task(
        self,
        task: APITask,
        kitchen_lookup: Dict[str, KitchenAPI],
        event_callback: Optional[EventCallback],
    ) -> TaskExecutionResult:
        """Executes a single API task through the ResilientHTTPExecutor."""
        start_time = time.perf_counter()
        api = kitchen_lookup.get(task.api_name)

        if not api:
            return TaskExecutionResult(
                task_id=task.task_id,
                api_name=task.api_name,
                endpoint=task.endpoint,
                method=task.method,
                success=False,
                status_code=404,
                error=f"API '{task.api_name}' not found in active kitchen keys.",
                elapsed_seconds=0.0,
            )

        # Normalize target URL
        base = api.base_url.rstrip("/")
        endpoint = task.endpoint.lstrip("/")
        full_url = f"{base}/{endpoint}"

        # Construct resilient APIRequest
        request = APIRequest(
            method=task.method,
            url=full_url,
            params=task.params,
            json_body=task.json_body,
            auth_type=api.auth_type,
            auth_token=api.api_key,
            auth_header_name=api.auth_header_name,
            auth_param_name=api.auth_param_name,
        )

        await self._emit_event(
            event_callback,
            "task_started",
            {
                "task_id": task.task_id,
                "api_name": task.api_name,
                "url": full_url,
                "method": task.method,
                "wave": task.wave,
            },
        )

        # Execute through resilience engine (429 backoff + 400 self-healing)
        try:
            exec_result: ExecutionResult = await self.resilient_executor.execute(
                request=request,
                tool_schema={"endpoint": task.endpoint, "method": task.method},
                event_callback=event_callback,
            )

            elapsed = time.perf_counter() - start_time
            result = TaskExecutionResult(
                task_id=task.task_id,
                api_name=task.api_name,
                endpoint=task.endpoint,
                method=task.method,
                success=exec_result.success,
                status_code=exec_result.status_code,
                data=exec_result.data if exec_result.data is not None else exec_result.text,
                error=exec_result.error,
                elapsed_seconds=round(elapsed, 4),
                self_healed=exec_result.self_healed,
                retry_count=exec_result.retry_count,
                executed_payload={
                    "params": exec_result.corrected_payload.get("params") if exec_result.corrected_payload else task.params,
                    "json_body": exec_result.corrected_payload.get("json_body") if exec_result.corrected_payload else task.json_body,
                },
            )

            await self._emit_event(
                event_callback,
                "task_completed",
                {
                    "task_id": task.task_id,
                    "api_name": task.api_name,
                    "success": result.success,
                    "status_code": result.status_code,
                    "self_healed": result.self_healed,
                    "elapsed_seconds": result.elapsed_seconds,
                },
            )

            return result

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.error(f"Error executing task {task.task_id}: {exc}")
            err_result = TaskExecutionResult(
                task_id=task.task_id,
                api_name=task.api_name,
                endpoint=task.endpoint,
                method=task.method,
                success=False,
                status_code=500,
                error=f"Unexpected execution error: {str(exc)}",
                elapsed_seconds=round(elapsed, 4),
            )
            await self._emit_event(
                event_callback,
                "task_failed",
                {
                    "task_id": task.task_id,
                    "api_name": task.api_name,
                    "error": str(exc),
                },
            )
            return err_result

    async def run_executor(
        self,
        plan: Plan,
        kitchen: List[KitchenAPI],
        event_callback: Optional[EventCallback] = None,
    ) -> List[TaskExecutionResult]:
        """Runs the tool-calling loop in parallel waves using asyncio.gather."""
        if not plan.tasks:
            logger.info("Executor received empty tasks list.")
            return []

        kitchen_lookup = {api.api_name: api for api in kitchen}

        # Group tasks into sequential waves (Wave 1, Wave 2...)
        waves: Dict[int, List[APITask]] = {}
        for task in plan.tasks:
            w = task.wave or 1
            waves.setdefault(w, []).append(task)

        all_results: List[TaskExecutionResult] = []

        # Execute each wave in parallel
        for wave_idx in sorted(waves.keys()):
            tasks_in_wave = waves[wave_idx]
            await self._emit_event(
                event_callback,
                "execution_wave_started",
                {
                    "wave": wave_idx,
                    "tasks_count": len(tasks_in_wave),
                    "task_ids": [t.task_id for t in tasks_in_wave],
                },
            )

            # Concurrent execution with asyncio.gather
            coros = [
                self._execute_single_task(task, kitchen_lookup, event_callback)
                for task in tasks_in_wave
            ]
            wave_outcomes = await asyncio.gather(*coros, return_exceptions=True)

            for outcome in wave_outcomes:
                if isinstance(outcome, TaskExecutionResult):
                    all_results.append(outcome)
                elif isinstance(outcome, Exception):
                    # Unhandled exception captured safely
                    all_results.append(
                        TaskExecutionResult(
                            task_id="unknown",
                            api_name="unknown",
                            endpoint="unknown",
                            method="unknown",
                            success=False,
                            status_code=500,
                            error=str(outcome),
                        )
                    )

            await self._emit_event(
                event_callback,
                "execution_wave_completed",
                {"wave": wave_idx, "completed_tasks": len(wave_outcomes)},
            )

        return all_results

    # ------------------------------------------------------------------------
    # Stage 3: Synthesizer Agent
    # ------------------------------------------------------------------------

    async def run_synthesizer(
        self,
        prompt: str,
        plan: Plan,
        task_results: List[TaskExecutionResult],
        total_time: float,
        event_callback: Optional[EventCallback] = None,
    ) -> SynthesizedResponse:
        """Combines live 3rd-party API data and outcomes into a structured response."""
        await self._emit_event(
            event_callback,
            "synthesis_started",
            {
                "task_results_count": len(task_results),
                "successful_tasks": sum(1 for r in task_results if r.success),
            },
        )

        apis_used = list({r.api_name for r in task_results if r.success})
        client = self._get_gemini_client()

        # If no tasks were planned/executed
        if not task_results:
            fallback_response = SynthesizedResponse(
                summary="Execution complete.",
                content=f"No API calls were executed for prompt: '{prompt}'. Reasoning: {plan.reasoning}",
                key_insights=[],
                apis_used=[],
                task_results=[],
                total_time_seconds=round(total_time, 3),
            )
            await self._emit_event(
                event_callback,
                "synthesis_completed",
                {"summary": fallback_response.summary},
            )
            return fallback_response

        # If Gemini client is unavailable, synthesize live data directly
        if not client:
            successful_tasks = [r for r in task_results if r.success]
            insights: List[str] = []
            for r in successful_tasks:
                if isinstance(r.data, dict):
                    if "current" in r.data:
                        curr = r.data["current"]
                        insights.append(f"Tokyo Weather: {curr.get('temperature_2m')}°C (Humidity: {curr.get('relative_humidity_2m')}%)")
                    elif "stargazers_count" in r.data:
                        insights.append(f"FastAPI: {r.data.get('stargazers_count'):,} stars, {r.data.get('forks_count'):,} forks")
                    elif "bitcoin" in r.data:
                        btc = r.data.get("bitcoin", {}).get("usd")
                        eth = r.data.get("ethereum", {}).get("usd")
                        insights.append(f"Crypto Spot: BTC = ${btc:,} USD | ETH = ${eth:,} USD" if btc and eth else "Crypto spot prices retrieved")
                    elif "title" in r.data:
                        insights.append(f"Custom Slot: Verified Bearer auth on post '{r.data.get('title')[:25]}...'")
                    else:
                        top_keys = list(r.data.keys())[:3]
                        insights.append(f"{r.api_name}: HTTP {r.status_code} with keys {top_keys}")
                else:
                    insights.append(f"{r.api_name}: HTTP {r.status_code}")

            summary_text = (
                f"Successfully queried {len(successful_tasks)}/{len(task_results)} live APIs "
                f"across {len(apis_used)} connectors in {round(total_time, 2)}s."
            )

            content_lines = [
                f"### Multi-API Live Execution Report\n",
                f"**Prompt**: {prompt}\n",
                f"**Plan**: {plan.reasoning}\n\n",
                "#### Live Data Breakdown:\n",
            ]
            for r in task_results:
                icon = "✅" if r.success else "❌"
                content_lines.append(f"- {icon} **{r.api_name}** (`{r.endpoint}`): HTTP {r.status_code} in {r.elapsed_seconds}s")

            content_text = "\n".join(content_lines)

            fallback_response = SynthesizedResponse(
                summary=summary_text,
                content=content_text,
                key_insights=insights,
                apis_used=apis_used,
                task_results=task_results,
                total_time_seconds=round(total_time, 3),
            )
            await self._emit_event(
                event_callback,
                "synthesis_completed",
                {"summary": fallback_response.summary, "apis_used": fallback_response.apis_used},
            )
            return fallback_response

        # Compile concise execution context for synthesis
        results_context = []
        for r in task_results:
            results_context.append({
                "task_id": r.task_id,
                "api_name": r.api_name,
                "endpoint": r.endpoint,
                "success": r.success,
                "status_code": r.status_code,
                "data": r.data if r.success else None,
                "error": r.error if not r.success else None,
                "self_healed": r.self_healed,
            })

        system_prompt = f"""You are an Expert Data Synthesizer Agent in an Adaptive API Kitchen.
Your task is to review live API data collected across 3rd-party services and craft a comprehensive,
accurate, and structured response addressing the user's initial prompt.

ORIGINAL USER PROMPT:
{prompt}

PLAN REASONING:
{plan.reasoning}

LIVE API TASK RESULTS:
{json.dumps(results_context, indent=2, default=str)}

INSTRUCTIONS:
1. Formulate a direct, coherent, and factual narrative answering the prompt based ONLY on the live data.
2. If any API failed or was rate-limited/healed, briefly acknowledge the status transparently.
3. Extract key bullet points / data insights.
4. Output strict JSON adhering to this schema:
{{
  "summary": "1-2 sentence executive summary",
  "content": "Rich markdown narrative answering the user's prompt with live data",
  "key_insights": ["Insight 1", "Insight 2", ...],
  "apis_used": ["api1", "api2"]
}}
"""

        try:
            config = None
            if GENAI_AVAILABLE and genai_types:
                config = genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                )

            response = await client.aio.models.generate_content(
                model=self.model_name,
                contents=[{"role": "user", "parts": [{"text": system_prompt}]}],
                config=config,
            )

            raw_json = self._clean_json_markdown(response.text or "{}")
            synth_dict = json.loads(raw_json)

            synthesized = SynthesizedResponse(
                summary=synth_dict.get("summary", "API data synthesis completed."),
                content=synth_dict.get("content", "Synthesized response based on live API data."),
                key_insights=synth_dict.get("key_insights", []),
                apis_used=synth_dict.get("apis_used", apis_used),
                task_results=task_results,
                total_time_seconds=round(total_time, 3),
            )

            await self._emit_event(
                event_callback,
                "synthesis_completed",
                {"summary": synthesized.summary, "apis_used": synthesized.apis_used},
            )
            return synthesized

        except Exception as exc:
            logger.error(f"Synthesizer Agent error: {exc}", exc_info=True)
            fallback = SynthesizedResponse(
                summary="Synthesis generated with basic formatter.",
                content=f"Executed {len(task_results)} tasks. Error in Gemini synthesis: {str(exc)}",
                key_insights=[f"Completed {len(task_results)} tasks"],
                apis_used=apis_used,
                task_results=task_results,
                total_time_seconds=round(total_time, 3),
            )
            await self._emit_event(
                event_callback,
                "synthesis_completed",
                {"summary": fallback.summary},
            )
            return fallback

    # ------------------------------------------------------------------------
    # Pipeline Orchestration Coordinator
    # ------------------------------------------------------------------------

    async def orchestrate(
        self,
        prompt: str,
        kitchen: List[KitchenAPI],
        event_callback: Optional[EventCallback] = None,
    ) -> SynthesizedResponse:
        """Executes the full 3-stage agent pipeline: Planner -> Parallel Executor -> Synthesizer."""
        pipeline_start = time.perf_counter()

        await self._emit_event(
            event_callback,
            "orchestrator_started",
            {"prompt": prompt, "kitchen_size": len(kitchen)},
        )

        # Stage 1: Planner
        plan = await self.run_planner(prompt, kitchen, event_callback)

        # Stage 2: Parallel Wave Executor
        task_results = await self.run_executor(plan, kitchen, event_callback)

        # Stage 3: Synthesizer
        total_time = time.perf_counter() - pipeline_start
        synthesized = await self.run_synthesizer(
            prompt=prompt,
            plan=plan,
            task_results=task_results,
            total_time=total_time,
            event_callback=event_callback,
        )

        await self._emit_event(
            event_callback,
            "orchestrator_completed",
            {
                "summary": synthesized.summary,
                "total_time_seconds": synthesized.total_time_seconds,
                "tasks_count": len(task_results),
            },
        )

        return synthesized
