"""Smart Rate-Limit, Retry & Self-Healing Engine.

Wraps the Universal HTTP Executor with:
1. Exponential backoff and jitter for HTTP 429 (Too Many Requests) and 5xx errors,
   respecting 'Retry-After' headers when present.
2. Self-healing mechanism via Google Gemini (official google-genai SDK):
   When a 3rd-party API returns HTTP 400 Bad Request or 422 Unprocessable Entity
   due to invalid parameters, the error message and schema are fed back to Gemini
   to automatically self-correct the payload and retry once without crashing.
3. Event callback hooks for real-time WebSocket streaming.
"""

from __future__ import annotations

import asyncio
import datetime
import email.utils
import json
import logging
import random
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

import httpx
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.http_executor import APIRequest, APIResponse, UniversalHTTPExecutor

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    genai_types = None  # type: ignore
    GENAI_AVAILABLE = False

logger = logging.getLogger("app.services.resilience")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)


class BackoffConfig(BaseModel):
    """Configuration for rate-limit and transient error backoff."""

    max_retries: int = Field(default=3, description="Maximum number of backoff retry attempts")
    base_delay: float = Field(default=1.0, description="Initial delay in seconds")
    max_delay: float = Field(default=30.0, description="Maximum ceiling delay in seconds")
    backoff_factor: float = Field(default=2.0, description="Multiplier applied per retry attempt")
    jitter: float = Field(default=0.5, description="Random jitter upper bound in seconds")
    retry_status_codes: Set[int] = Field(
        default={429, 500, 502, 503, 504},
        description="HTTP status codes that trigger backoff retries",
    )


class SelfHealingConfig(BaseModel):
    """Configuration for Gemini-powered payload self-healing."""

    enabled: bool = Field(default=True, description="Enable automatic payload self-healing")
    max_corrections: int = Field(default=1, description="Number of self-correction attempts (default: 1)")
    status_codes: Set[int] = Field(
        default={400, 422},
        description="HTTP status codes indicating invalid parameter payload to heal",
    )
    gemini_model: str = Field(default="gemini-2.5-flash", description="Gemini model for repairing payloads")


class ExecutionResult(BaseModel):
    """Complete structured result of an API execution through the resilience engine."""

    success: bool = Field(description="Whether the final execution succeeded (HTTP 2xx)")
    status_code: int = Field(description="Final HTTP status code received")
    data: Optional[Any] = Field(default=None, description="Parsed response JSON or structured data")
    text: str = Field(default="", description="Raw response text")
    error: Optional[str] = Field(default=None, description="Detailed error description if execution failed")
    rate_limited: bool = Field(default=False, description="Whether a 429 rate limit was encountered and handled")
    retry_count: int = Field(default=0, description="Number of backoff retries executed")
    self_healed: bool = Field(default=False, description="Whether the payload was successfully repaired by Gemini")
    self_healing_attempted: bool = Field(default=False, description="Whether self-healing was attempted")
    original_payload: Dict[str, Any] = Field(default_factory=dict, description="Original request parameters sent")
    corrected_payload: Optional[Dict[str, Any]] = Field(default=None, description="Repaired parameters if self-healed")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Audit log of attempts, delays, and responses")


# Type alias for async event callbacks (useful for WebSocket streaming)
EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]


class ResilientHTTPExecutor:
    """Wraps UniversalHTTPExecutor with exponential backoff and Gemini self-healing."""

    def __init__(
        self,
        executor: Optional[UniversalHTTPExecutor] = None,
        gemini_client: Optional[Any] = None,
        backoff_config: Optional[BackoffConfig] = None,
        self_healing_config: Optional[SelfHealingConfig] = None,
    ):
        self.executor = executor or UniversalHTTPExecutor()
        self.backoff_config = backoff_config or BackoffConfig()
        self.self_healing_config = self_healing_config or SelfHealingConfig()
        self._gemini_client = gemini_client

    def _get_gemini_client(self) -> Optional[Any]:
        """Initializes or returns the official google-genai Client."""
        if self._gemini_client is not None:
            return self._gemini_client

        if not GENAI_AVAILABLE:
            logger.warning("google-genai SDK is not installed. Self-healing is disabled.")
            return None

        api_key = settings.GEMINI_API_KEY
        if not api_key:
            logger.warning("GEMINI_API_KEY is not configured. Self-healing is disabled.")
            return None

        try:
            self._gemini_client = genai.Client(api_key=api_key)
            return self._gemini_client
        except Exception as exc:
            logger.error(f"Failed to initialize google-genai Client: {exc}")
            return None

    def parse_retry_after(self, retry_after_header: Optional[str]) -> Optional[float]:
        """Parses 'Retry-After' header value into seconds (float).

        Supports both integer/float seconds and HTTP-date formats.
        """
        if not retry_after_header:
            return None

        val = retry_after_header.strip()

        # 1. Check if numeric seconds
        try:
            seconds = float(val)
            if seconds >= 0:
                return seconds
        except ValueError:
            pass

        # 2. Check if HTTP-date (RFC 7231 / RFC 1123)
        try:
            target_dt = email.utils.parsedate_to_datetime(val)
            now_dt = datetime.datetime.now(datetime.timezone.utc)
            delta = (target_dt - now_dt).total_seconds()
            return max(0.0, delta)
        except Exception:
            pass

        return None

    def calculate_backoff(self, attempt: int, retry_after: Optional[str] = None) -> float:
        """Calculates backoff delay in seconds with exponential factor and jitter."""
        header_delay = self.parse_retry_after(retry_after)
        if header_delay is not None:
            # Add a slight jitter even when respecting server-specified Retry-After
            return min(self.backoff_config.max_delay, header_delay + random.uniform(0.1, 0.5))

        calculated = self.backoff_config.base_delay * (self.backoff_config.backoff_factor ** attempt)
        jitter = random.uniform(0, self.backoff_config.jitter)
        delay = min(self.backoff_config.max_delay, calculated + jitter)
        return round(delay, 3)

    async def _emit_event(
        self,
        event_callback: Optional[EventCallback],
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Emits an event to the optional async callback for WebSocket streaming."""
        payload = {
            "type": event_type,
            "timestamp": time.time(),
            **data,
        }
        logger.debug(f"[ResilienceEvent] {event_type}: {payload}")
        if event_callback:
            try:
                await event_callback(payload)
            except Exception as cb_err:
                logger.warning(f"Error in event callback: {cb_err}")

    async def _heal_payload_with_gemini(
        self,
        request: APIRequest,
        failed_response: APIResponse,
        tool_schema: Optional[Dict[str, Any]],
        event_callback: Optional[EventCallback],
    ) -> Optional[APIRequest]:
        """Asks Gemini to inspect the 400/422 validation error and repair the request payload."""
        gemini_client = self._get_gemini_client()
        if not gemini_client:
            logger.warning("Gemini client unavailable for self-healing. Skipping.")
            return None

        error_detail = failed_response.error or failed_response.text or "Unknown 400 Bad Request"
        original_params = request.params
        original_json = request.json_body
        original_data = request.data

        await self._emit_event(
            event_callback,
            "self_healing_started",
            {
                "status_code": failed_response.status_code,
                "error": error_detail,
                "endpoint": f"{request.method} {request.url}",
                "original_params": original_params,
                "original_json": original_json,
            },
        )

        prompt = f"""You are an automated API Self-Healing and Payload Repair Agent.
A 3rd-party API rejected an HTTP request with status code {failed_response.status_code}.

Endpoint: {request.method} {request.url}
API Error Response:
{error_detail}

Current Request Data Sent:
- Query Parameters: {json.dumps(original_params)}
- JSON Body: {json.dumps(original_json)}
- Form Data: {json.dumps(original_data)}

API / Tool Schema or Context:
{json.dumps(tool_schema) if tool_schema else "No explicit schema provided. Infer corrections from the API Error Response."}

Task:
Analyze why the 3rd-party API returned this validation/parameter error.
Fix the wrong or missing parameter(s) and provide the corrected parameters.
Ensure proper types (e.g., integer vs string, expected enum values, required keys).

Respond with a JSON object adhering to this exact schema:
{{
  "explanation": "Brief 1-sentence reason for what was corrected",
  "corrected_params": {{ ... }},  // All query parameters to send (or empty object if none)
  "corrected_json_body": {{ ... }} // All JSON body fields to send (or null if not a JSON request)
}}
"""

        try:
            config = None
            if GENAI_AVAILABLE and genai_types:
                config = genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                )

            model_name = self.self_healing_config.gemini_model or settings.GEMINI_MODEL

            # Call Gemini using async client
            response = await gemini_client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            raw_text = response.text or ""
            # Extract JSON from response text safely
            clean_text = raw_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            parsed = json.loads(clean_text)
            explanation = parsed.get("explanation", "Payload adjusted to resolve API parameter error.")
            new_params = parsed.get("corrected_params", original_params)
            new_json_body = parsed.get("corrected_json_body", original_json)

            # Build healed APIRequest copy
            healed_request = request.model_copy(
                update={
                    "params": new_params if isinstance(new_params, dict) else original_params,
                    "json_body": new_json_body if (new_json_body is not None and isinstance(new_json_body, (dict, list))) else original_json,
                }
            )

            await self._emit_event(
                event_callback,
                "self_healing_payload_generated",
                {
                    "explanation": explanation,
                    "new_params": healed_request.params,
                    "new_json_body": healed_request.json_body,
                },
            )

            return healed_request

        except Exception as exc:
            logger.error(f"Failed to generate self-healed payload from Gemini: {exc}")
            await self._emit_event(
                event_callback,
                "self_healing_failed",
                {"reason": f"Gemini error during payload repair: {str(exc)}"},
            )
            return None

    async def execute(
        self,
        request: APIRequest,
        tool_schema: Optional[Dict[str, Any]] = None,
        event_callback: Optional[EventCallback] = None,
    ) -> ExecutionResult:
        """Executes an API request with automatic rate-limit backoff and self-healing.

        - Automatically retries 429 and 5xx errors using exponential backoff.
        - Automatically feeds 400 Bad Request error messages back to Gemini,
          repairs the payload, and retries once without crashing.
        - Streams all state transitions to `event_callback` if provided.
        """
        history: List[Dict[str, Any]] = []
        current_request = request.model_copy()
        original_payload_snapshot = {
            "params": dict(request.params),
            "json_body": request.json_body,
            "data": request.data,
        }

        retry_count = 0
        rate_limited_encountered = False
        self_healing_attempted = False
        self_healed_success = False
        corrected_payload_snapshot: Optional[Dict[str, Any]] = None

        final_response: Optional[APIResponse] = None

        while True:
            # 1. Execute request via Universal HTTP Executor
            response = await self.executor.execute(current_request)
            final_response = response

            history_entry = {
                "attempt": retry_count + 1,
                "status_code": response.status_code,
                "is_success": response.is_success,
                "elapsed_seconds": response.elapsed_seconds,
                "error": response.error,
                "timestamp": time.time(),
            }
            history.append(history_entry)

            # 2. Case: Successful execution (HTTP 2xx)
            if response.is_success:
                if self_healing_attempted:
                    self_healed_success = True
                    await self._emit_event(
                        event_callback,
                        "self_healing_success",
                        {
                            "status_code": response.status_code,
                            "corrected_payload": corrected_payload_snapshot,
                        },
                    )

                await self._emit_event(
                    event_callback,
                    "execution_success",
                    {
                        "status_code": response.status_code,
                        "elapsed_seconds": response.elapsed_seconds,
                        "retry_count": retry_count,
                        "self_healed": self_healed_success,
                    },
                )

                return ExecutionResult(
                    success=True,
                    status_code=response.status_code,
                    data=response.data,
                    text=response.text,
                    error=None,
                    rate_limited=rate_limited_encountered,
                    retry_count=retry_count,
                    self_healed=self_healed_success,
                    self_healing_attempted=self_healing_attempted,
                    original_payload=original_payload_snapshot,
                    corrected_payload=corrected_payload_snapshot,
                    history=history,
                )

            # 3. Case: 429 Too Many Requests or 5xx Transient Server Error
            if (
                response.status_code in self.backoff_config.retry_status_codes
                or response.status_code == 0  # Network/timeout error
            ):
                if response.status_code == 429:
                    rate_limited_encountered = True

                if retry_count < self.backoff_config.max_retries:
                    retry_after_hdr = response.headers.get("retry-after") or response.headers.get("Retry-After")
                    delay = self.calculate_backoff(retry_count, retry_after_hdr)

                    event_name = "rate_limit_backoff" if response.status_code == 429 else "server_error_backoff"
                    await self._emit_event(
                        event_callback,
                        event_name,
                        {
                            "status_code": response.status_code,
                            "attempt": retry_count + 1,
                            "max_retries": self.backoff_config.max_retries,
                            "delay_seconds": delay,
                            "retry_after_header": retry_after_hdr,
                        },
                    )

                    await asyncio.sleep(delay)
                    retry_count += 1
                    continue
                else:
                    # Max retries exhausted
                    await self._emit_event(
                        event_callback,
                        "retries_exhausted",
                        {
                            "status_code": response.status_code,
                            "total_retries": retry_count,
                            "error": response.error,
                        },
                    )
                    break

            # 4. Case: 400 Bad Request or 422 Unprocessable Entity (Self-Healing Opportunity)
            if (
                self.self_healing_config.enabled
                and response.status_code in self.self_healing_config.status_codes
                and not self_healing_attempted
            ):
                self_healing_attempted = True

                # Call Gemini to heal the payload
                healed_request = await self._heal_payload_with_gemini(
                    request=current_request,
                    failed_response=response,
                    tool_schema=tool_schema,
                    event_callback=event_callback,
                )

                if healed_request:
                    current_request = healed_request
                    corrected_payload_snapshot = {
                        "params": dict(current_request.params),
                        "json_body": current_request.json_body,
                    }
                    # Retry once with the healed request
                    continue
                else:
                    # Self-healing could not produce a payload; exit cleanly without crashing
                    break

            # 5. Non-retryable status (e.g., 401 Unauthorized, 403 Forbidden, 404 Not Found, or 2nd 400 failure)
            break

        # If we exited the loop without returning success, return safe failed ExecutionResult without crashing
        final_status = final_response.status_code if final_response else 500
        final_error = final_response.error if final_response else "Unknown execution error"

        await self._emit_event(
            event_callback,
            "execution_failed",
            {
                "status_code": final_status,
                "error": final_error,
                "self_healing_attempted": self_healing_attempted,
            },
        )

        return ExecutionResult(
            success=False,
            status_code=final_status,
            data=final_response.data if final_response else None,
            text=final_response.text if final_response else "",
            error=final_error,
            rate_limited=rate_limited_encountered,
            retry_count=retry_count,
            self_healed=False,
            self_healing_attempted=self_healing_attempted,
            original_payload=original_payload_snapshot,
            corrected_payload=corrected_payload_snapshot,
            history=history,
        )


# Global default resilient executor instance
resilient_executor = ResilientHTTPExecutor()


async def resilient_execute(
    request: APIRequest,
    tool_schema: Optional[Dict[str, Any]] = None,
    event_callback: Optional[EventCallback] = None,
    gemini_client: Optional[Any] = None,
) -> ExecutionResult:
    """Convenience helper to execute an API request with rate-limit and self-healing resilience."""
    executor = ResilientHTTPExecutor(gemini_client=gemini_client) if gemini_client else resilient_executor
    return await executor.execute(request, tool_schema=tool_schema, event_callback=event_callback)
