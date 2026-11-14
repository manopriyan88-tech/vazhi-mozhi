"""Tests for Smart Rate-Limit, Retry & Self-Healing Engine."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from app.services.http_executor import APIRequest, APIResponse, UniversalHTTPExecutor
from app.services.resilience import (
    BackoffConfig,
    ExecutionResult,
    ResilientHTTPExecutor,
    SelfHealingConfig,
)


class TestResilienceEngine(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying backoff, retry, and Gemini self-healing."""

    def setUp(self):
        # Use low delays for fast unit tests
        self.fast_backoff = BackoffConfig(
            max_retries=3,
            base_delay=0.01,
            max_delay=0.05,
            backoff_factor=1.5,
            jitter=0.005,
        )

    def test_parse_retry_after_numeric(self):
        executor = ResilientHTTPExecutor(backoff_config=self.fast_backoff)
        self.assertEqual(executor.parse_retry_after("10"), 10.0)
        self.assertEqual(executor.parse_retry_after("2.5"), 2.5)
        self.assertIsNone(executor.parse_retry_after(None))
        self.assertIsNone(executor.parse_retry_after("invalid-date-string"))

    def test_calculate_backoff_values(self):
        executor = ResilientHTTPExecutor(backoff_config=self.fast_backoff)
        delay_0 = executor.calculate_backoff(0)
        delay_1 = executor.calculate_backoff(1)
        self.assertGreaterEqual(delay_0, 0.01)
        self.assertGreaterEqual(delay_1, delay_0)

    async def test_429_rate_limit_exponential_backoff_success(self):
        mock_http = MagicMock(spec=UniversalHTTPExecutor)
        # Sequence: 429 -> 429 -> 200 OK
        resp_429_1 = APIResponse(
            status_code=429,
            headers={"Retry-After": "0.01"},
            is_success=False,
            error="Rate limit exceeded",
        )
        resp_429_2 = APIResponse(
            status_code=429,
            headers={"Retry-After": "0.01"},
            is_success=False,
            error="Rate limit exceeded",
        )
        resp_200 = APIResponse(
            status_code=200,
            headers={},
            is_success=True,
            data={"result": "data"},
        )
        mock_http.execute = AsyncMock(side_effect=[resp_429_1, resp_429_2, resp_200])

        events = []
        async def on_event(ev):
            events.append(ev)

        resilient = ResilientHTTPExecutor(
            executor=mock_http,
            backoff_config=self.fast_backoff,
        )

        req = APIRequest(url="https://api.example.com/v1/resource")
        result = await resilient.execute(req, event_callback=on_event)

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.rate_limited)
        self.assertEqual(result.retry_count, 2)
        self.assertEqual(len(result.history), 3)

        # Check emitted WebSocket events
        backoff_events = [e for e in events if e["type"] == "rate_limit_backoff"]
        self.assertEqual(len(backoff_events), 2)
        self.assertEqual(events[-1]["type"], "execution_success")

    async def test_5xx_server_error_retry_and_exhaustion(self):
        mock_http = MagicMock(spec=UniversalHTTPExecutor)
        resp_503 = APIResponse(
            status_code=503,
            headers={},
            is_success=False,
            error="Service Unavailable",
        )
        # Always fail with 503
        mock_http.execute = AsyncMock(return_value=resp_503)

        resilient = ResilientHTTPExecutor(
            executor=mock_http,
            backoff_config=self.fast_backoff,
        )

        req = APIRequest(url="https://api.example.com/v1/flakey")
        result = await resilient.execute(req)

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 503)
        self.assertEqual(result.retry_count, 3)

    async def test_400_bad_request_self_healing_success(self):
        """Simulates 400 Bad Request with wrong parameter, Gemini self-corrects, and retry succeeds."""
        mock_http = MagicMock(spec=UniversalHTTPExecutor)

        # 1st call fails with 400 Bad Request
        resp_400 = APIResponse(
            status_code=400,
            is_success=False,
            error="Field 'limit' must be an integer, got string 'fifty'",
            data={"error": "Field 'limit' must be an integer, got string 'fifty'"},
        )
        # 2nd call (with corrected payload) succeeds with 200 OK
        resp_200 = APIResponse(
            status_code=200,
            is_success=True,
            data={"items": [1, 2, 3]},
        )
        mock_http.execute = AsyncMock(side_effect=[resp_400, resp_200])

        # Mock Gemini Client
        mock_gemini_client = MagicMock()
        mock_gen_response = MagicMock()
        mock_gen_response.text = '{"explanation": "Fixed limit parameter to integer 50", "corrected_params": {"limit": 50}}'
        mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_gen_response)

        events = []
        async def on_event(ev):
            events.append(ev)

        resilient = ResilientHTTPExecutor(
            executor=mock_http,
            gemini_client=mock_gemini_client,
            backoff_config=self.fast_backoff,
        )

        req = APIRequest(
            url="https://api.example.com/v1/items",
            params={"limit": "fifty"},
        )
        tool_schema = {
            "name": "get_items",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of items to fetch"}
                }
            }
        }

        result = await resilient.execute(req, tool_schema=tool_schema, event_callback=on_event)

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.self_healed)
        self.assertTrue(result.self_healing_attempted)
        self.assertEqual(result.corrected_payload, {"params": {"limit": 50}, "json_body": None})

        # Verify Gemini generate_content was called
        mock_gemini_client.aio.models.generate_content.assert_awaited_once()

        # Verify events emitted for WebSockets
        event_types = [e["type"] for e in events]
        self.assertIn("self_healing_started", event_types)
        self.assertIn("self_healing_payload_generated", event_types)
        self.assertIn("self_healing_success", event_types)
        self.assertIn("execution_success", event_types)

    async def test_400_bad_request_fails_without_crashing_when_gemini_correction_fails(self):
        """Simulates 400 Bad Request where self-correction still returns 400. Engine must not crash."""
        mock_http = MagicMock(spec=UniversalHTTPExecutor)

        resp_400_1 = APIResponse(
            status_code=400,
            is_success=False,
            error="Invalid category 'music'",
        )
        resp_400_2 = APIResponse(
            status_code=400,
            is_success=False,
            error="Category still invalid",
        )
        mock_http.execute = AsyncMock(side_effect=[resp_400_1, resp_400_2])

        mock_gemini_client = MagicMock()
        mock_gen_response = MagicMock()
        mock_gen_response.text = '{"explanation": "Attempted category correction", "corrected_params": {"category": "audio"}}'
        mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_gen_response)

        resilient = ResilientHTTPExecutor(
            executor=mock_http,
            gemini_client=mock_gemini_client,
            backoff_config=self.fast_backoff,
        )

        req = APIRequest(
            url="https://api.example.com/v1/search",
            params={"category": "music"},
        )

        # Must execute cleanly without raising an unhandled exception or crashing
        result = await resilient.execute(req)

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 400)
        self.assertFalse(result.self_healed)
        self.assertTrue(result.self_healing_attempted)
        self.assertEqual(result.error, "Category still invalid")

    async def test_gemini_api_crash_handled_gracefully(self):
        """If Gemini itself throws an exception during self-healing, the engine must not crash."""
        mock_http = MagicMock(spec=UniversalHTTPExecutor)
        resp_400 = APIResponse(
            status_code=400,
            is_success=False,
            error="Bad parameter",
        )
        mock_http.execute = AsyncMock(return_value=resp_400)

        mock_gemini_client = MagicMock()
        mock_gemini_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("Gemini Quota Exceeded"))

        resilient = ResilientHTTPExecutor(
            executor=mock_http,
            gemini_client=mock_gemini_client,
            backoff_config=self.fast_backoff,
        )

        req = APIRequest(url="https://api.example.com/v1/action", params={"bad": "val"})

        # Must return clean failed ExecutionResult without crashing
        result = await resilient.execute(req)
        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 400)
        self.assertFalse(result.self_healed)


if __name__ == "__main__":
    unittest.main()
