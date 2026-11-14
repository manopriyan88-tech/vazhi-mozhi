"""Comprehensive test suite for Live WebSocket & SSE Event Streamer (app.routers.stream)."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.testclient import TestClient

from app.main import app
from app.core.models import (
    APITask,
    KitchenAPI,
    Plan,
    SynthesizedResponse,
    TaskExecutionResult,
)


@pytest.fixture
def client():
    """FastAPI TestClient."""
    return TestClient(app)


# Mock data helpers
def make_mock_synthesized_response():
    task_res = TaskExecutionResult(
        task_id="task_1",
        api_name="weather_service",
        endpoint="/v1/forecast",
        method="GET",
        success=True,
        status_code=200,
        data={"temperature": 21.5, "condition": "Sunny"},
        elapsed_seconds=0.12,
    )
    return SynthesizedResponse(
        summary="Weather data retrieved successfully.",
        content="Current temperature in Tokyo is 21.5°C with sunny conditions.",
        key_insights=["Tokyo temperature: 21.5°C"],
        apis_used=["weather_service"],
        task_results=[task_res],
        total_time_seconds=0.45,
    )


# ============================================================================
# 1. WebSocket Endpoint Tests (/ws/execute)
# ============================================================================

def test_websocket_connection_and_ping(client):
    """Verifies connecting to /ws/execute and receiving connection + ping/pong events."""
    with client.websocket_connect("/ws/execute") as websocket:
        # Initial greeting event
        greeting = websocket.receive_json()
        assert greeting["step"] == "connected"
        assert "Adaptive API Kitchen" in greeting["message"]

        # Ping / pong test
        websocket.send_json({"action": "ping"})
        pong = websocket.receive_json()
        assert pong["step"] == "pong"
        assert "timestamp" in pong


def test_websocket_missing_prompt_returns_error(client):
    """Verifies that sending a message without prompt returns a clean error event."""
    with client.websocket_connect("/ws/execute") as websocket:
        _ = websocket.receive_json()  # Connected event

        websocket.send_json({"session_id": "test_sess"})
        error_event = websocket.receive_json()
        assert error_event["step"] == "error"
        assert "Missing 'prompt'" in error_event["message"]


def test_websocket_live_execution_stream(client):
    """Verifies that running prompt over /ws/execute broadcasts real-time events.

    Specifically checks:
    - {'step': 'inspecting_key'}
    - {'step': 'calling_3rd_party_api', 'provider': ..., 'payload': ...}
    - {'step': 'completed', 'data': ...}
    """
    mock_synth = make_mock_synthesized_response()

    # Mock MultiAgentOrchestrator.orchestrate to simulate wave execution and callbacks
    async def mock_orchestrate(prompt, kitchen, event_callback):
        # 1. Emit planning
        if event_callback:
            await event_callback({
                "type": "planner_started",
                "prompt": prompt,
                "available_apis": [k.api_name for k in kitchen],
            })
            await event_callback({
                "type": "plan_generated",
                "plan": {"reasoning": "Plan generated", "tasks": [{"task_id": "task_1"}]},
            })
            await event_callback({
                "type": "execution_wave_started",
                "wave": 1,
                "tasks_count": 1,
                "task_ids": ["task_1"],
            })
            # 2. Emit task_started (translates to calling_3rd_party_api)
            await event_callback({
                "type": "task_started",
                "task_id": "task_1",
                "api_name": "weather_service",
                "url": "https://api.weather.com/v1/forecast",
                "method": "GET",
                "params": {"city": "Tokyo"},
                "wave": 1,
            })
            # 3. Emit task_completed
            await event_callback({
                "type": "task_completed",
                "task_id": "task_1",
                "api_name": "weather_service",
                "success": True,
                "status_code": 200,
                "elapsed_seconds": 0.12,
            })
            # 4. Emit synthesis
            await event_callback({
                "type": "synthesis_started",
                "task_results_count": 1,
                "successful_tasks": 1,
            })
            await event_callback({
                "type": "synthesis_completed",
                "summary": mock_synth.summary,
                "apis_used": mock_synth.apis_used,
            })
        return mock_synth

    with patch("app.agents.orchestrator.MultiAgentOrchestrator.orchestrate", side_effect=mock_orchestrate):
        with client.websocket_connect("/ws/execute") as websocket:
            _ = websocket.receive_json()  # Connected event

            # Send execution instruction
            websocket.send_json({
                "prompt": "What is the weather in Tokyo?",
                "session_id": "ws_session_test",
                "include_demo": True,
            })

            received_steps = []
            calling_api_event = None
            completed_event = None

            # Collect events until completed
            while True:
                msg = websocket.receive_json()
                step = msg.get("step")
                received_steps.append(step)

                if step == "calling_3rd_party_api":
                    calling_api_event = msg
                elif step == "completed":
                    completed_event = msg
                    break

            # Assert key required steps were broadcast
            assert "inspecting_key" in received_steps
            assert "calling_3rd_party_api" in received_steps
            assert "completed" in received_steps

            # Check calling_3rd_party_api structure
            assert calling_api_event is not None
            assert calling_api_event["provider"] == "weather_service"
            assert calling_api_event["payload"] == {"city": "Tokyo"}

            # Check completed structure
            assert completed_event is not None
            assert completed_event["data"]["summary"] == mock_synth.summary
            assert completed_event["data"]["total_time_seconds"] == 0.45


# ============================================================================
# 2. Server-Sent Events (SSE) Endpoint Tests (/api/stream)
# ============================================================================

def test_sse_stream_get(client):
    """Verifies that GET /api/stream returns text/event-stream chunks."""
    mock_synth = make_mock_synthesized_response()

    async def mock_orchestrate(prompt, kitchen, event_callback):
        if event_callback:
            await event_callback({
                "type": "task_started",
                "task_id": "task_1",
                "api_name": "crypto_service",
                "url": "https://api.coingecko.com/v1/simple/price",
                "method": "GET",
                "params": {"ids": "bitcoin", "vs_currencies": "usd"},
                "wave": 1,
            })
        return mock_synth

    with patch("app.agents.orchestrator.MultiAgentOrchestrator.orchestrate", side_effect=mock_orchestrate):
        response = client.get("/api/stream?prompt=Check+crypto+prices&session_id=sse_test")
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE lines
        raw_text = response.text
        lines = [line.strip() for line in raw_text.split("\n") if line.startswith("data: ")]
        assert len(lines) > 0

        parsed_events = [json.loads(line[6:]) for line in lines]
        steps = [evt["step"] for evt in parsed_events]

        assert "inspecting_key" in steps
        assert "calling_3rd_party_api" in steps
        assert "completed" in steps

        # Check calling_3rd_party_api event payload
        call_evt = next(e for e in parsed_events if e["step"] == "calling_3rd_party_api")
        assert call_evt["provider"] == "crypto_service"
        assert call_evt["payload"] == {"ids": "bitcoin", "vs_currencies": "usd"}

        # Check completed event
        comp_evt = next(e for e in parsed_events if e["step"] == "completed")
        assert comp_evt["data"]["summary"] == mock_synth.summary


def test_sse_stream_post(client):
    """Verifies that POST /api/stream streams events with JSON request payload."""
    mock_synth = make_mock_synthesized_response()

    with patch("app.agents.orchestrator.MultiAgentOrchestrator.orchestrate", new_callable=AsyncMock) as mock_orch:
        mock_orch.return_value = mock_synth

        payload = {
            "prompt": "Live test for SSE POST",
            "session_id": "post_sess",
            "include_demo": True,
        }

        response = client.post("/api/stream", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        raw_text = response.text
        lines = [line.strip() for line in raw_text.split("\n") if line.startswith("data: ")]
        parsed_events = [json.loads(line[6:]) for line in lines]
        steps = [evt["step"] for evt in parsed_events]

        assert "inspecting_key" in steps
        assert "completed" in steps
