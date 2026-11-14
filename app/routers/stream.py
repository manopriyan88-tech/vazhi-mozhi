"""Live WebSocket & SSE Event Streamer.

Provides:
- WebSocket endpoint: `/ws/execute`
- Server-Sent Events (SSE) endpoint: `/api/stream`

Broadcasts real-time JSON status events:
  {'step': 'inspecting_key', ...}
  {'step': 'calling_3rd_party_api', 'provider': '...', 'payload': ...}
  {'step': 'completed', 'data': ...}
along with rich intermediate stages so the frontend UI can render live terminal
logs and animated node execution graphs.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Set
from urllib.parse import urlparse

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.orchestrator import MultiAgentOrchestrator
from app.core.models import EndpointSpec, KitchenAPI, SynthesizedResponse
from app.services.key_vault import KeyRecord, key_vault
from app.services.kitchen_store import get_kitchen_store

logger = logging.getLogger("app.routers.stream")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

router = APIRouter(tags=["Live Streaming"])


class StreamExecutionRequest(BaseModel):
    """Payload to initiate an execution stream via HTTP SSE."""

    prompt: str = Field(..., min_length=1, description="Prompt or query for the multi-agent orchestrator")
    session_id: Optional[str] = Field(default="default", description="Session namespace holding API keys")
    include_demo: bool = Field(default=True, description="Whether to include pre-seeded public APIs if kitchen is empty")


class ConnectionManager:
    """Tracks and broadcasts events to active WebSocket connections."""

    def __init__(self):
        self._active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accepts and stores an active WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total active: {len(self._active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Removes a disconnected WebSocket client."""
        async with self._lock:
            self._active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total active: {len(self._active_connections)}")

    async def send_json(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        """Sends a JSON event to a specific WebSocket client."""
        try:
            await websocket.send_json(data)
        except Exception as exc:
            logger.warning(f"Failed to send JSON to WebSocket: {exc}")

    async def broadcast(self, data: Dict[str, Any]) -> None:
        """Broadcasts a JSON event to all connected clients."""
        async with self._lock:
            targets = list(self._active_connections)
        for ws in targets:
            try:
                await ws.send_json(data)
            except Exception:
                await self.disconnect(ws)


manager = ConnectionManager()


def key_record_to_kitchen_api(record: KeyRecord) -> KitchenAPI:
    """Converts an in-memory session KeyRecord into a KitchenAPI model for the orchestrator."""
    parsed = urlparse(record.endpoint_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path or "/"

    params: Dict[str, Any] = {}
    if record.function_declaration and "parameters" in record.function_declaration:
        props = record.function_declaration["parameters"].get("properties", {})
        if isinstance(props, dict):
            params = {k: v.get("description", "") if isinstance(v, dict) else "" for k, v in props.items()}

    endpoint_spec = EndpointSpec(
        path=path,
        method="GET",
        summary=record.function_declaration.get("description", f"Call {record.provider}"),
        parameters=params,
    )

    return KitchenAPI(
        api_name=record.provider,
        base_url=base_url,
        description=record.function_declaration.get("description", f"Connector for {record.provider}"),
        api_key=record.api_key,
        auth_type=record.auth_type,
        auth_header_name=record.auth_header_name,
        auth_param_name=record.auth_param_name,
        endpoints=[endpoint_spec],
        schema_dict=record.function_declaration,
    )


async def resolve_active_kitchen(
    session_id: str = "default",
    include_demo: bool = True,
) -> List[KitchenAPI]:
    """Gathers all available APIs from the KitchenStore and session KeyVault."""
    kitchen_map: Dict[str, KitchenAPI] = {}

    # 1. From in-memory KitchenStore
    store = get_kitchen_store()
    for api in store.get_all():
        kitchen_map[api.api_name.lower()] = api

    # 2. From session KeyVault
    records = await key_vault.list_keys(session_id=session_id)
    for rec in records:
        if rec.is_active or rec.api_key:
            k_api = key_record_to_kitchen_api(rec)
            kitchen_map[k_api.api_name.lower()] = k_api

    # 3. If empty and include_demo is True, auto-seed free public demo connectors
    if not kitchen_map and include_demo:
        demo_connectors = store.seed_demo_apis()
        for api in demo_connectors:
            kitchen_map[api.api_name.lower()] = api

    return list(kitchen_map.values())


class ExecutionEventBridge:
    """Translates and maps orchestrator execution steps into frontend status events.

    Ensures the exact steps requested:
      {'step': 'inspecting_key', ...}
      {'step': 'calling_3rd_party_api', 'provider': '...', 'payload': ...}
      {'step': 'completed', 'data': ...}
    along with terminal logs and animated graph execution details.
    """

    def __init__(self, emit_fn):
        self.emit_fn = emit_fn

    async def emit(self, step: str, **kwargs: Any) -> None:
        """Emits a structured JSON event with a 'step' key."""
        event = {
            "step": step,
            "timestamp": time.time(),
            "iso_time": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        await self.emit_fn(event)

    async def handle_orchestrator_event(self, raw_event: Dict[str, Any]) -> None:
        """Adapts internal events emitted by MultiAgentOrchestrator and ResilientHTTPExecutor."""
        event_type = raw_event.get("type", "")

        if event_type == "orchestrator_started":
            # Handled initially in execute_pipeline
            return

        elif event_type == "planner_started":
            await self.emit(
                step="planning_started",
                prompt=raw_event.get("prompt"),
                available_apis=raw_event.get("available_apis", []),
                node="planner",
                status="running",
                message=f"Planner agent decomposing prompt across {len(raw_event.get('available_apis', []))} APIs...",
            )

        elif event_type == "plan_generated":
            plan = raw_event.get("plan", {})
            tasks = plan.get("tasks", [])
            await self.emit(
                step="plan_generated",
                plan=plan,
                tasks_count=len(tasks),
                node="planner",
                status="success",
                message=f"Generated execution plan with {len(tasks)} tasks: {plan.get('reasoning', '')}",
            )

        elif event_type == "execution_wave_started":
            wave = raw_event.get("wave", 1)
            await self.emit(
                step="wave_started",
                wave=wave,
                tasks_count=raw_event.get("tasks_count", 0),
                task_ids=raw_event.get("task_ids", []),
                node="wave_executor",
                status="running",
                message=f"Executing Wave {wave} in parallel ({raw_event.get('tasks_count', 0)} concurrent tasks)...",
            )

        elif event_type == "task_started":
            provider = raw_event.get("api_name", "unknown")
            task_id = raw_event.get("task_id", "")
            endpoint = raw_event.get("url", "")
            method = raw_event.get("method", "GET")
            payload = raw_event.get("params") or raw_event.get("json_body") or {}

            # Explicit step: calling_3rd_party_api
            await self.emit(
                step="calling_3rd_party_api",
                provider=provider,
                task_id=task_id,
                endpoint=endpoint,
                method=method,
                payload=payload,
                node=f"task_{provider}",
                status="running",
                message=f"Dispatching HTTP {method} request to 3rd-party provider '{provider}'...",
            )

        elif event_type == "rate_limit_backoff":
            await self.emit(
                step="rate_limit_backoff",
                status_code=raw_event.get("status_code", 429),
                attempt=raw_event.get("attempt"),
                delay_seconds=raw_event.get("delay_seconds"),
                node="resilience",
                status="warning",
                message=(
                    f"HTTP 429 Rate Limit encountered. "
                    f"Backing off for {raw_event.get('delay_seconds')}s (attempt {raw_event.get('attempt')})..."
                ),
            )

        elif event_type == "self_healing_started":
            await self.emit(
                step="self_healing_started",
                status_code=raw_event.get("status_code"),
                error=raw_event.get("error"),
                endpoint=raw_event.get("endpoint"),
                node="resilience",
                status="repairing",
                message=f"HTTP {raw_event.get('status_code')} validation error detected. Gemini self-healing payload...",
            )

        elif event_type == "self_healing_success":
            await self.emit(
                step="self_healing_success",
                status_code=raw_event.get("status_code"),
                corrected_payload=raw_event.get("corrected_payload"),
                node="resilience",
                status="healed",
                message="Gemini successfully self-healed request parameters! Retried request succeeded.",
            )

        elif event_type == "task_completed":
            provider = raw_event.get("api_name", "unknown")
            success = raw_event.get("success", False)
            status_code = raw_event.get("status_code", 200)
            elapsed = raw_event.get("elapsed_seconds", 0.0)

            await self.emit(
                step="api_response_received",
                provider=provider,
                task_id=raw_event.get("task_id"),
                success=success,
                status_code=status_code,
                elapsed_seconds=elapsed,
                self_healed=raw_event.get("self_healed", False),
                node=f"task_{provider}",
                status="success" if success else "failed",
                message=f"Received response from '{provider}': HTTP {status_code} in {elapsed}s.",
            )

        elif event_type == "execution_wave_completed":
            wave = raw_event.get("wave", 1)
            await self.emit(
                step="wave_completed",
                wave=wave,
                completed_tasks=raw_event.get("completed_tasks", 0),
                node="wave_executor",
                status="success",
                message=f"Execution Wave {wave} completed successfully.",
            )

        elif event_type == "synthesis_started":
            await self.emit(
                step="synthesizing",
                task_results_count=raw_event.get("task_results_count", 0),
                successful_tasks=raw_event.get("successful_tasks", 0),
                node="synthesizer",
                status="running",
                message="Synthesizer agent compiling live 3rd-party data into unified structured response...",
            )

        elif event_type == "synthesis_completed":
            await self.emit(
                step="synthesis_completed",
                summary=raw_event.get("summary", ""),
                apis_used=raw_event.get("apis_used", []),
                node="synthesizer",
                status="success",
                message="Synthesis complete.",
            )


async def execute_stream_pipeline(
    prompt: str,
    session_id: str,
    include_demo: bool,
    emit_fn,
) -> SynthesizedResponse:
    """Coordinates key inspection, agent orchestration, and real-time event streaming."""
    bridge = ExecutionEventBridge(emit_fn=emit_fn)

    # 1. Step: inspecting_key
    await bridge.emit(
        step="inspecting_key",
        session_id=session_id,
        node="vault_inspector",
        status="running",
        message=f"Inspecting session vault credentials for session='{session_id}'...",
    )

    kitchen = await resolve_active_kitchen(session_id=session_id, include_demo=include_demo)

    await bridge.emit(
        step="keys_inspected",
        session_id=session_id,
        node="vault_inspector",
        status="success",
        active_connectors_count=len(kitchen),
        connectors=[k.api_name for k in kitchen],
        message=f"Discovered {len(kitchen)} active connector(s) in kitchen.",
    )

    # 2. Run Orchestrator with event bridge callback
    orchestrator = MultiAgentOrchestrator()

    try:
        synthesized: SynthesizedResponse = await orchestrator.orchestrate(
            prompt=prompt,
            kitchen=kitchen,
            event_callback=bridge.handle_orchestrator_event,
        )

        # 3. Step: completed
        await bridge.emit(
            step="completed",
            data=synthesized.model_dump(),
            summary=synthesized.summary,
            content=synthesized.content,
            key_insights=synthesized.key_insights,
            apis_used=synthesized.apis_used,
            total_time_seconds=synthesized.total_time_seconds,
            node="pipeline",
            status="completed",
            message=f"Pipeline completed successfully in {synthesized.total_time_seconds}s.",
        )

        return synthesized

    except Exception as exc:
        logger.error(f"Error in stream execution pipeline: {exc}", exc_info=True)
        await bridge.emit(
            step="error",
            message=f"Pipeline execution encountered an error: {str(exc)}",
            node="pipeline",
            status="failed",
        )
        raise


# ============================================================================
# 1. WebSocket Endpoint: /ws/execute
# ============================================================================

@router.websocket("/ws/execute")
async def websocket_execute(websocket: WebSocket) -> None:
    """Live bidirectional WebSocket endpoint to execute multi-agent prompts and stream events.

    Clients connect and send JSON messages:
      {
        "prompt": "Fetch weather in Tokyo and latest FastAPI repo stars",
        "session_id": "default",
        "include_demo": true
      }
    The endpoint broadcasts real-time status events:
      {'step': 'inspecting_key', ...}
      {'step': 'calling_3rd_party_api', 'provider': '...', 'payload': ...}
      {'step': 'completed', 'data': ...}
    """
    await manager.connect(websocket)

    # Send initial connection acknowledgment
    await manager.send_json(
        websocket,
        {
            "step": "connected",
            "timestamp": time.time(),
            "message": "Connected to Adaptive API Kitchen WebSocket execution stream.",
        },
    )

    try:
        while True:
            # Wait for client prompt/instruction
            raw_msg = await websocket.receive_text()
            try:
                data = json.loads(raw_msg)
            except Exception:
                data = {"prompt": raw_msg}

            # Handle ping/pong heartbeats
            if data.get("action") == "ping" or data.get("type") == "ping":
                await manager.send_json(websocket, {"step": "pong", "timestamp": time.time()})
                continue

            prompt = (data.get("prompt") or "").strip()
            session_id = (data.get("session_id") or "default").strip()
            include_demo = bool(data.get("include_demo", True))

            if not prompt:
                await manager.send_json(
                    websocket,
                    {
                        "step": "error",
                        "timestamp": time.time(),
                        "message": "Missing 'prompt' field in payload.",
                    },
                )
                continue

            # Emit wrapper targeting this connection
            async def ws_emit(event: Dict[str, Any]) -> None:
                await manager.send_json(websocket, event)

            # Run execution pipeline
            try:
                await execute_stream_pipeline(
                    prompt=prompt,
                    session_id=session_id,
                    include_demo=include_demo,
                    emit_fn=ws_emit,
                )
            except Exception as pipe_err:
                await manager.send_json(
                    websocket,
                    {
                        "step": "error",
                        "timestamp": time.time(),
                        "message": f"Execution failed: {str(pipe_err)}",
                    },
                )

    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/execute")
        await manager.disconnect(websocket)
    except Exception as exc:
        logger.warning(f"Unexpected WebSocket termination: {exc}")
        await manager.disconnect(websocket)


# ============================================================================
# 2. Server-Sent Events (SSE) Endpoint: /api/stream
# ============================================================================

async def sse_event_generator(
    prompt: str,
    session_id: str,
    include_demo: bool,
) -> AsyncGenerator[str, None]:
    """Generates Server-Sent Events (SSE) formatted text chunks as execution progresses."""
    queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

    async def sse_emit(event: Dict[str, Any]) -> None:
        await queue.put(event)

    async def worker() -> None:
        try:
            await execute_stream_pipeline(
                prompt=prompt,
                session_id=session_id,
                include_demo=include_demo,
                emit_fn=sse_emit,
            )
        except Exception:
            pass
        finally:
            # Sentinel to finish generator
            await queue.put(None)

    worker_task = asyncio.create_task(worker())

    try:
        while True:
            # Await event or timeout for heartbeat
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
                if event is None:
                    # Stream finished
                    break
                # Yield SSE chunk
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                # Keep-alive heartbeat comment
                yield ": ping\n\n"

    finally:
        if not worker_task.done():
            worker_task.cancel()


@router.get(
    "/api/stream",
    summary="Stream multi-agent execution events via Server-Sent Events (SSE)",
)
async def sse_stream_get(
    prompt: str = Query(..., min_length=1, description="User prompt to orchestrate"),
    session_id: str = Query(default="default", description="Session namespace"),
    include_demo: bool = Query(default=True, description="Whether to include pre-seeded public APIs"),
) -> StreamingResponse:
    """Streams real-time execution events over Server-Sent Events (SSE).

    Can be consumed directly by `new EventSource('/api/stream?prompt=...')`.
    Broadcasts real-time events:
      - `{'step': 'inspecting_key'}`
      - `{'step': 'calling_3rd_party_api', 'provider': '...', 'payload': ...}`
      - `{'step': 'completed', 'data': ...}`
    """
    return StreamingResponse(
        sse_event_generator(prompt=prompt, session_id=session_id, include_demo=include_demo),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/api/stream",
    summary="Stream multi-agent execution events via POST SSE",
)
async def sse_stream_post(payload: StreamExecutionRequest) -> StreamingResponse:
    """Streams real-time execution events over Server-Sent Events (SSE) from a POST payload."""
    return StreamingResponse(
        sse_event_generator(
            prompt=payload.prompt,
            session_id=payload.session_id or "default",
            include_demo=payload.include_demo,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
