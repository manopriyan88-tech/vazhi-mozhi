"""
tests/test_executor.py

Comprehensive test suite for the Universal Async HTTP Executor (app.services.executor).
Tests function call interception, path/query/body parameter resolution,
3rd-party auth injection (Authorization: Bearer & x-api-key),
live async HTTP execution via httpx, and Gemini turn loop response formatting.
"""

import asyncio
import json
import os
import unittest
from typing import Any, Dict

import httpx
from google.genai import types

from app.services.executor import (
    APIKeyVault,
    AuthHeaderType,
    ExecutionResult,
    HttpMethod,
    RequestPlan,
    ToolDefinition,
    ToolRegistry,
    UniversalAsyncHTTPExecutor,
    default_executor,
    execute_for_gemini,
    execute_function_call,
    register_api_key,
    register_tool,
)


class TestFunctionCallInterception(unittest.TestCase):
    """Tests normalization and interception of various Gemini function call formats."""

    def test_extract_from_genai_function_call(self):
        fc = types.FunctionCall(name="github_get_user", args={"username": "octocat"}, id="call_123")
        name, args, call_id = UniversalAsyncHTTPExecutor.extract_function_call(fc)
        self.assertEqual(name, "github_get_user")
        self.assertEqual(args, {"username": "octocat"})
        self.assertEqual(call_id, "call_123")

    def test_extract_from_genai_part(self):
        fc = types.FunctionCall(name="weather_current", args={"city": "Tokyo"})
        part = types.Part(function_call=fc)
        name, args, call_id = UniversalAsyncHTTPExecutor.extract_function_call(part)
        self.assertEqual(name, "weather_current")
        self.assertEqual(args, {"city": "Tokyo"})
        self.assertIsNone(call_id)

    def test_extract_from_dict_standard(self):
        call_dict = {
            "name": "stripe_create_charge",
            "args": {"amount": 2000, "currency": "usd"},
            "id": "stripe_999",
        }
        name, args, call_id = UniversalAsyncHTTPExecutor.extract_function_call(call_dict)
        self.assertEqual(name, "stripe_create_charge")
        self.assertEqual(args["amount"], 2000)
        self.assertEqual(call_id, "stripe_999")

    def test_extract_from_dict_nested_function_call(self):
        call_dict = {
            "function_call": {
                "name": "post_message",
                "arguments": json.dumps({"channel": "general", "text": "hello"}),
                "id": "slack_1",
            }
        }
        name, args, call_id = UniversalAsyncHTTPExecutor.extract_function_call(call_dict)
        self.assertEqual(name, "post_message")
        self.assertEqual(args, {"channel": "general", "text": "hello"})
        self.assertEqual(call_id, "slack_1")

    def test_extract_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            UniversalAsyncHTTPExecutor.extract_function_call(12345)


class TestAuthInjection(unittest.TestCase):
    """Tests 3rd-party API key injection for Bearer and x-api-key schemes."""

    def setUp(self):
        self.vault = APIKeyVault()

    def test_bearer_injection(self):
        self.vault.register_key("github", "ghp_secrettoken123", auth_type=AuthHeaderType.BEARER)
        headers, params = self.vault.inject(
            headers={"Content-Type": "application/json"},
            params={},
            url="https://api.github.com/user",
            service_name="github",
        )
        self.assertEqual(headers["Authorization"], "Bearer ghp_secrettoken123")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_bearer_no_duplicate_prefix(self):
        self.vault.register_key("custom", "Bearer already_prefixed", auth_type=AuthHeaderType.BEARER)
        headers, params = self.vault.inject(
            headers={},
            params={},
            url="https://api.custom.com",
            service_name="custom",
        )
        self.assertEqual(headers["Authorization"], "Bearer already_prefixed")

    def test_x_api_key_injection(self):
        self.vault.register_key("anthropic", "sk-ant-api03-secret", auth_type=AuthHeaderType.X_API_KEY)
        headers, params = self.vault.inject(
            headers={},
            params={},
            url="https://api.anthropic.com/v1/messages",
            service_name="anthropic",
        )
        self.assertEqual(headers["x-api-key"], "sk-ant-api03-secret")
        self.assertNotIn("Authorization", headers)

    def test_known_service_heuristics_x_api_key(self):
        # Register without explicit auth_type; anthropic is known to use x-api-key
        self.vault.register_key("anthropic", "sk-ant-test")
        headers, _ = self.vault.inject(
            headers={},
            params={},
            url="https://api.anthropic.com/v1/models",
            service_name="anthropic",
        )
        self.assertEqual(headers["x-api-key"], "sk-ant-test")

    def test_known_service_heuristics_bearer(self):
        # GitHub is known to use Bearer
        self.vault.register_key("github", "ghp_test")
        headers, _ = self.vault.inject(
            headers={},
            params={},
            url="https://api.github.com/repos",
            service_name="github",
        )
        self.assertEqual(headers["Authorization"], "Bearer ghp_test")

    def test_placeholder_substitution_in_headers(self):
        self.vault.register_key("custom", "live_token_777")
        headers, _ = self.vault.inject(
            headers={"Authorization": "Bearer {API_KEY}", "x-api-key": "{API_KEY}"},
            params={},
            url="https://api.custom.com",
            service_name="custom",
        )
        self.assertEqual(headers["Authorization"], "Bearer live_token_777")

    def test_env_var_fallback_discovery(self):
        os.environ["STRIPE_API_KEY"] = "sk_live_stripe_secret_123"
        try:
            headers, _ = self.vault.inject(
                headers={},
                params={},
                url="https://api.stripe.com/v1/charges",
                service_name="stripe",
            )
            self.assertEqual(headers["Authorization"], "Bearer sk_live_stripe_secret_123")
        finally:
            os.environ.pop("STRIPE_API_KEY", None)

    def test_query_param_auth_injection(self):
        tool = ToolDefinition(
            name="weather_query",
            url="https://api.weatherapi.com/v1/current.json",
            auth_type=AuthHeaderType.QUERY,
            auth_query_param="key",
            api_key="weather_key_abc",
        )
        headers, params = self.vault.inject(
            headers={},
            params={"q": "Paris"},
            url=tool.url,
            tool_def=tool,
        )
        self.assertEqual(params["key"], "weather_key_abc")
        self.assertEqual(params["q"], "Paris")


class TestParameterAndUrlResolution(unittest.IsolatedAsyncioTestCase):
    """Tests URL path parameter interpolation and query/body mapping."""

    async def test_path_parameter_interpolation(self):
        registry = ToolRegistry()
        registry.register_endpoint(
            name="github_get_repo_issue",
            url="https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}",
            method="GET",
        )
        executor = UniversalAsyncHTTPExecutor(registry=registry)
        plan = await executor.plan_request(
            "github_get_repo_issue",
            {"owner": "google", "repo": "genai", "issue_number": 42, "state": "open"},
        )
        self.assertEqual(plan.url, "https://api.github.com/repos/google/genai/issues/42")
        self.assertEqual(plan.params, {"state": "open"})
        self.assertEqual(plan.method, "GET")

    async def test_post_body_mapping(self):
        registry = ToolRegistry()
        registry.register_endpoint(
            name="create_pet",
            url="https://petstore.swagger.io/v2/pet",
            method="POST",
        )
        executor = UniversalAsyncHTTPExecutor(registry=registry)
        plan = await executor.plan_request(
            "create_pet",
            {"name": "Doggie", "status": "available"},
        )
        self.assertEqual(plan.method, "POST")
        self.assertEqual(plan.json_data, {"name": "Doggie", "status": "available"})
        self.assertEqual(plan.params, {})

    async def test_explicit_body_param_mapping(self):
        registry = ToolRegistry()
        registry.register_endpoint(
            name="update_user",
            url="https://api.example.com/users/{id}",
            method="PUT",
            body_param="profile",
        )
        executor = UniversalAsyncHTTPExecutor(registry=registry)
        plan = await executor.plan_request(
            "update_user",
            {"id": "usr_100", "profile": {"age": 30, "city": "NYC"}},
        )
        self.assertEqual(plan.url, "https://api.example.com/users/usr_100")
        self.assertEqual(plan.json_data, {"age": 30, "city": "NYC"})

    async def test_generic_http_tool_execution(self):
        executor = UniversalAsyncHTTPExecutor()
        plan = await executor.plan_request(
            "http_request",
            {
                "url": "https://httpbin.org/post",
                "method": "POST",
                "json": {"message": "hello"},
                "headers": {"X-Custom": "test"},
            },
        )
        self.assertEqual(plan.method, "POST")
        self.assertEqual(plan.url, "https://httpbin.org/post")
        self.assertEqual(plan.json_data, {"message": "hello"})
        self.assertEqual(plan.headers["X-Custom"], "test")


class TestLiveAsyncHttpExecution(unittest.IsolatedAsyncioTestCase):
    """Tests live async HTTP execution with mock transport and httpx.AsyncClient."""

    async def test_successful_json_execution(self):
        # Mock transport returning JSON
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers.get("Authorization"), "Bearer my_secret_token")
            return httpx.Response(200, json={"id": 101, "name": "Rocket", "status": "active"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        vault = APIKeyVault()
        vault.register_key("test_service", "my_secret_token", auth_type=AuthHeaderType.BEARER)

        registry = ToolRegistry()
        registry.register_endpoint(
            name="get_rocket",
            url="https://api.test.com/rockets/{id}",
            method="GET",
            service_name="test_service",
        )

        executor = UniversalAsyncHTTPExecutor(client=client, key_vault=vault, registry=registry)

        call = types.FunctionCall(name="get_rocket", args={"id": 101}, id="call_rocket_1")
        result = await executor.execute_call(call)

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.data["name"], "Rocket")
        self.assertEqual(result.call_id, "call_rocket_1")

        # Verify Gemini loop response format
        gemini_dict = result.to_gemini_response()
        self.assertIsInstance(gemini_dict, dict)
        self.assertEqual(gemini_dict["name"], "Rocket")
        self.assertEqual(gemini_dict["status_code"], 200)

        # Verify Part conversion
        part = result.to_gemini_part()
        self.assertIsInstance(part, types.Part)
        self.assertEqual(part.function_response.name, "get_rocket")
        self.assertEqual(part.function_response.id, "call_rocket_1")

    async def test_x_api_key_header_execution(self):
        # Mock transport verifying x-api-key header
        def handler(request: httpx.Request) -> httpx.Response:
            auth_val = request.headers.get("x-api-key")
            self.assertEqual(auth_val, "anthropic_live_key_xyz")
            return httpx.Response(200, json={"content": [{"text": "Claude response"}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        vault = APIKeyVault()
        vault.register_key("anthropic", "anthropic_live_key_xyz", auth_type=AuthHeaderType.X_API_KEY)

        registry = ToolRegistry()
        registry.register_endpoint(
            name="anthropic_messages",
            url="https://api.anthropic.com/v1/messages",
            method="POST",
            auth_type=AuthHeaderType.X_API_KEY,
            service_name="anthropic",
        )

        executor = UniversalAsyncHTTPExecutor(client=client, key_vault=vault, registry=registry)
        result = await executor.execute_call({
            "name": "anthropic_messages",
            "args": {"model": "claude-3-haiku", "max_tokens": 100},
        })

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.data["content"][0]["text"], "Claude response")

    async def test_list_response_wrapping_for_gemini(self):
        # Gemini strictly requires FunctionResponse to be a dict; if the API returns a list,
        # it must be wrapped cleanly.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[{"id": 1, "login": "alice"}, {"id": 2, "login": "bob"}])

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        registry = ToolRegistry()
        registry.register_endpoint(name="list_users", url="https://api.example.com/users", method="GET")

        executor = UniversalAsyncHTTPExecutor(client=client, registry=registry)
        response_dict = await executor.execute("list_users")

        self.assertIsInstance(response_dict, dict)
        self.assertEqual(response_dict["count"], 2)
        self.assertEqual(len(response_dict["items"]), 2)
        self.assertEqual(response_dict["status_code"], 200)

    async def test_http_error_graceful_handling(self):
        # 404 Not Found returns structured JSON without crashing
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"message": "Repository not found", "error": "Not Found"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        registry = ToolRegistry()
        registry.register_endpoint(name="get_repo", url="https://api.example.com/repos/{repo}", method="GET")

        executor = UniversalAsyncHTTPExecutor(client=client, registry=registry)
        result = await executor.execute_call({"name": "get_repo", "args": {"repo": "nonexistent"}})

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 404)
        gemini_resp = result.to_gemini_response()
        self.assertIn("status_code", gemini_resp)
        self.assertEqual(gemini_resp["status_code"], 404)
        self.assertEqual(gemini_resp["message"], "Repository not found")

    async def test_unregistered_tool_returns_404_json(self):
        executor = UniversalAsyncHTTPExecutor()
        result = await executor.execute_call({"name": "unregistered_mystery_tool", "args": {}})

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 404)
        gemini_dict = result.to_gemini_response()
        self.assertEqual(gemini_dict["status_code"], 404)
        self.assertIn("error", gemini_dict)

    async def test_timeout_returns_504_json(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("Connection timed out after 30s")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        registry = ToolRegistry()
        registry.register_endpoint(name="slow_tool", url="https://slow.example.com/data")

        executor = UniversalAsyncHTTPExecutor(client=client, registry=registry)
        result = await executor.execute_call({"name": "slow_tool", "args": {}})

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 504)
        gemini_dict = result.to_gemini_response()
        self.assertEqual(gemini_dict["status_code"], 504)
        self.assertEqual(gemini_dict["error"], "GatewayTimeout")

    async def test_batch_execution(self):
        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "item1" in url_str:
                return httpx.Response(200, json={"item": "one"})
            return httpx.Response(200, json={"item": "two"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        registry = ToolRegistry()
        registry.register_endpoint(name="get_item", url="https://api.example.com/items/{item}")

        executor = UniversalAsyncHTTPExecutor(client=client, registry=registry)
        calls = [
            {"name": "get_item", "args": {"item": "item1"}, "id": "1"},
            {"name": "get_item", "args": {"item": "item2"}, "id": "2"},
        ]
        results = await executor.execute_batch(calls)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].data["item"], "one")
        self.assertEqual(results[1].data["item"], "two")

        parts = await executor.execute_batch_for_gemini(calls)
        self.assertEqual(len(parts), 2)
        self.assertIsInstance(parts[0], types.Part)
        self.assertEqual(parts[0].function_response.id, "1")
        self.assertEqual(parts[1].function_response.id, "2")


class TestSessionVaultIntegration(unittest.IsolatedAsyncioTestCase):
    """Tests seamless integration between key_vault (session vault) and UniversalAsyncHTTPExecutor."""

    async def test_session_vault_anthropic_x_api_key(self):
        from app.services.key_vault import key_vault

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers.get("x-api-key"), "sk-ant-test-key-555")
            return httpx.Response(200, json={"received_key": request.headers.get("x-api-key")})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        executor = UniversalAsyncHTTPExecutor(client=client)

        await key_vault.register_key(
            api_key="sk-ant-test-key-555",
            provider="Anthropic",
            endpoint_url="https://api.anthropic.com/v1/messages",
            session_id="sess_anthropic_test",
            skip_verification=True,
        )

        rec = await key_vault.get_key("sess_anthropic_test", "anthropic")
        tool_name = rec.function_declaration["name"]

        result = await executor.execute(tool_name, session_id="sess_anthropic_test")
        self.assertEqual(result["received_key"], "sk-ant-test-key-555")
        self.assertEqual(result["status_code"], 200)

    async def test_session_vault_github_bearer(self):
        from app.services.key_vault import key_vault

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers.get("Authorization"), "Bearer ghp_live_secret_444")
            return httpx.Response(200, json={"user": "octocat", "auth": request.headers.get("Authorization")})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        executor = UniversalAsyncHTTPExecutor(client=client)

        await key_vault.register_key(
            api_key="ghp_live_secret_444",
            provider="GitHub",
            endpoint_url="https://api.github.com/user",
            session_id="sess_gh_test",
            skip_verification=True,
        )

        rec = await key_vault.get_key("sess_gh_test", "github")
        tool_name = rec.function_declaration["name"]

        result = await executor.execute(tool_name, session_id="sess_gh_test")
        self.assertEqual(result["user"], "octocat")
        self.assertEqual(result["auth"], "Bearer ghp_live_secret_444")
        self.assertEqual(result["status_code"], 200)


if __name__ == "__main__":
    unittest.main()
