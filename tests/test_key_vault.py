"""Comprehensive test suite for the Dynamic 'API Kitchen' Key Vault & Inspector."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from starlette.testclient import TestClient

from app.main import app
from app.services.key_vault import (
    KeyVault,
    mask_api_key,
    sanitize_identifier,
)


@pytest.fixture
def clean_vault():
    """Provides a fresh isolated KeyVault instance."""
    return KeyVault()


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


# =========================================================================
# 1. API Key Masking & Identifier Sanitization Tests
# =========================================================================

def test_mask_api_key():
    assert mask_api_key("") == ""
    assert mask_api_key("abc") == "****"
    assert mask_api_key("1234") == "****"
    assert mask_api_key("12345678") == "12...78"
    assert mask_api_key("sk-proj-1234567890abcdef") == "sk-p...cdef"


def test_sanitize_identifier():
    assert sanitize_identifier("WeatherAPI") == "weatherapi"
    assert sanitize_identifier("Get Current Weather!") == "get_current_weather"
    assert sanitize_identifier("123_api_call") == "tool_123_api_call"
    assert sanitize_identifier("___weird___name___") == "weird_name"
    assert sanitize_identifier("") == "api_tool"


# =========================================================================
# 2. Key Vault In-Memory Session Storage & Isolation Tests
# =========================================================================

@pytest.mark.asyncio
async def test_vault_session_isolation(clean_vault):
    # Session 1: Register WeatherAPI
    r1 = await clean_vault.register_key(
        api_key="session1_weather_key_123456",
        provider="WeatherAPI",
        endpoint_url="https://api.weatherapi.com/v1/current.json",
        session_id="session-1",
        skip_verification=True,
    )
    assert r1.session_id == "session-1"
    assert r1.provider == "WeatherAPI"
    assert r1.masked_key == "sess...3456"

    # Session 2: Register WeatherAPI with different key
    r2 = await clean_vault.register_key(
        api_key="session2_weather_key_987654",
        provider="WeatherAPI",
        endpoint_url="https://api.weatherapi.com/v1/current.json",
        session_id="session-2",
        skip_verification=True,
    )
    assert r2.session_id == "session-2"
    assert r2.api_key == "session2_weather_key_987654"

    # Verify session 1 key was not overwritten by session 2
    s1_key = await clean_vault.get_key("session-1", "WeatherAPI")
    s2_key = await clean_vault.get_key("session-2", "WeatherAPI")
    assert s1_key is not None
    assert s2_key is not None
    assert s1_key.api_key == "session1_weather_key_123456"
    assert s2_key.api_key == "session2_weather_key_987654"


@pytest.mark.asyncio
async def test_vault_delete_and_clear(clean_vault):
    await clean_vault.register_key(
        api_key="key_github_12345",
        provider="GitHub",
        endpoint_url="https://api.github.com/user",
        session_id="test_sess",
        skip_verification=True,
    )
    await clean_vault.register_key(
        api_key="key_stripe_12345",
        provider="Stripe",
        endpoint_url="https://api.stripe.com/v1/charges",
        session_id="test_sess",
        skip_verification=True,
    )

    keys = await clean_vault.list_keys("test_sess")
    assert len(keys) == 2

    # Delete GitHub
    deleted = await clean_vault.delete_key("test_sess", "GitHub")
    assert deleted is True

    keys_after = await clean_vault.list_keys("test_sess")
    assert len(keys_after) == 1
    assert keys_after[0].provider == "Stripe"

    # Clear session
    cleared = await clean_vault.clear_session("test_sess")
    assert cleared == 1
    assert len(await clean_vault.list_keys("test_sess")) == 0


@pytest.mark.asyncio
async def test_key_record_create_api_request(clean_vault):
    record = await clean_vault.register_key(
        api_key="secret_weather_key_abc123",
        provider="WeatherAPI",
        endpoint_url="https://api.weatherapi.com/v1/current.json",
        session_id="test_req",
        skip_verification=True,
    )
    req = record.create_api_request(
        method="GET",
        params={"q": "Paris"},
    )
    assert req.method == "GET"
    assert req.url == "https://api.weatherapi.com/v1/current.json"
    assert req.auth_token == "secret_weather_key_abc123"
    assert req.auth_type == "api_key_query"
    assert req.auth_param_name == "key"
    assert req.params["q"] == "Paris"



# =========================================================================
# 3. Lightweight Async httpx Test Call Probing
# =========================================================================

@pytest.mark.asyncio
async def test_verify_key_200_success(clean_vault):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"current": {"temp_c": 22.5}}
    mock_resp.text = '{"current": {"temp_c": 22.5}}'

    with patch("httpx.AsyncClient.head", new_callable=AsyncMock) as mock_head:
        mock_head.return_value = mock_resp

        result = await clean_vault.verify_key_active(
            api_key="test_weather_key_123",
            endpoint_url="https://api.weatherapi.com/v1/current.json",
            provider="WeatherAPI",
        )

        assert result.is_active is True
        assert result.status == "verified"
        assert result.status_code == 200
        assert result.auth_method == "api_key_query"
        assert "verified successfully" in result.message.lower()


@pytest.mark.asyncio
async def test_verify_key_401_unauthorized(clean_vault):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"error": {"message": "API key is invalid"}}
    mock_resp.text = '{"error": {"message": "API key is invalid"}}'

    with patch("httpx.AsyncClient.head", new_callable=AsyncMock) as mock_head:
        mock_head.return_value = mock_resp

        result = await clean_vault.verify_key_active(
            api_key="bad_token_123",
            endpoint_url="https://api.github.com/user",
            provider="GitHub",
        )

        assert result.is_active is False
        assert result.status == "unauthorized"
        assert result.status_code == 401
        assert "401" in result.message


@pytest.mark.asyncio
async def test_verify_key_400_active_with_warning(clean_vault):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 400
    mock_resp.json.return_value = {"error": "parameter 'q' missing"}
    mock_resp.text = '{"error": "parameter \'q\' missing"}'

    with patch("httpx.AsyncClient.head", new_callable=AsyncMock) as mock_head:
        mock_head.return_value = mock_resp

        result = await clean_vault.verify_key_active(
            api_key="valid_token_but_no_params",
            endpoint_url="https://api.weatherapi.com/v1/current.json",
            provider="WeatherAPI",
        )

        # Authentication succeeded (not 401/403)
        assert result.is_active is True
        assert result.status == "active_with_warning"
        assert result.status_code == 400


@pytest.mark.asyncio
async def test_verify_key_timeout(clean_vault):
    with patch("httpx.AsyncClient.head", new_callable=AsyncMock) as mock_head:
        mock_head.side_effect = httpx.TimeoutException("Connection timed out")

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Connection timed out")

            result = await clean_vault.verify_key_active(
                api_key="valid_key",
                endpoint_url="https://api.unreachable.example/data",
                provider="CustomAPI",
            )

            assert result.is_active is False
            assert result.status == "unreachable"
            assert "timed out" in result.message.lower()


# =========================================================================
# 4. Gemini Function Declaration Generation Tests
# =========================================================================

@pytest.mark.asyncio
async def test_heuristic_schema_generation(clean_vault):
    schema = clean_vault._heuristic_generate_schema(
        provider="WeatherAPI",
        endpoint_url="https://api.weatherapi.com/v1/current.json?aqi=no",
    )

    assert "name" in schema
    assert "description" in schema
    assert "parameters" in schema
    assert schema["parameters"]["type"] == "OBJECT"
    assert "location" in schema["parameters"]["properties"] or "aqi" in schema["parameters"]["properties"]


@pytest.mark.asyncio
async def test_gemini_schema_generation_mock():
    mock_gemini_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
    {
        "name": "get_current_weather",
        "description": "Fetch real-time weather information for any global city or coordinates.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {
                    "type": "STRING",
                    "description": "City or geographical location"
                },
                "units": {
                    "type": "STRING",
                    "description": "Measurement units",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["city"]
        }
    }
    """
    mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    vault = KeyVault(gemini_client=mock_gemini_client)
    schema, source = await vault.generate_function_declaration(
        provider="WeatherAPI",
        endpoint_url="https://api.weatherapi.com/v1/current.json",
    )

    assert schema["name"] == "get_current_weather"
    assert schema["parameters"]["type"] == "OBJECT"
    assert "city" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["city"]
    assert source == "gemini-2.5-flash"


# =========================================================================
# 5. FastAPI Endpoints Integration Tests (/api/keys/register)
# =========================================================================

def test_api_keys_register_endpoint(client):
    # Test POST /api/keys/register with mocked verification probe
    with patch("app.services.key_vault.key_vault.verify_key_active", new_callable=AsyncMock) as mock_probe:
        from app.services.key_vault import VerificationResult
        mock_probe.return_value = VerificationResult(
            is_active=True,
            status="verified",
            status_code=200,
            latency_ms=120.5,
            auth_method="api_key_query",
            message="Key verified successfully.",
            sample_data={"temp": 24},
        )

        payload = {
            "api_key": "live_test_api_key_secret_12345",
            "provider": "WeatherAPI",
            "endpoint_url": "https://api.weatherapi.com/v1/current.json",
            "session_id": "integration-session",
        }

        response = client.post("/api/keys/register", json=payload)
        assert response.status_code == 201

        data = response.json()
        assert data["success"] is True
        assert data["provider"] == "WeatherAPI"
        assert data["session_id"] == "integration-session"
        assert data["is_active"] is True
        assert data["masked_key"] == "live...2345"
        # Secret key must NEVER be in response
        assert "live_test_api_key_secret_12345" not in response.text

        # Verify Function Declaration structure
        fn_decl = data["function_declaration"]
        assert "name" in fn_decl
        assert "description" in fn_decl
        assert "parameters" in fn_decl
        assert fn_decl["parameters"]["type"] == "OBJECT"

        # Test GET /api/keys to list the registered key
        list_resp = client.get("/api/keys?session_id=integration-session")
        assert list_resp.status_code == 200
        items = list_resp.json()
        assert len(items) == 1
        assert items[0]["provider"] == "WeatherAPI"
        assert items[0]["masked_key"] == "live...2345"

        # Test GET /api/keys/{provider}
        get_resp = client.get("/api/keys/WeatherAPI?session_id=integration-session")
        assert get_resp.status_code == 200
        assert get_resp.json()["provider"] == "WeatherAPI"

        # Test GET /api/keys/{provider}/schema
        schema_resp = client.get("/api/keys/WeatherAPI/schema?session_id=integration-session")
        assert schema_resp.status_code == 200
        assert "parameters" in schema_resp.json()

        # Test DELETE /api/keys/{provider}
        del_resp = client.delete("/api/keys/WeatherAPI?session_id=integration-session")
        assert del_resp.status_code == 200
        assert del_resp.json()["success"] is True

        # Verify it is deleted
        get_after = client.get("/api/keys/WeatherAPI?session_id=integration-session")
        assert get_after.status_code == 404
