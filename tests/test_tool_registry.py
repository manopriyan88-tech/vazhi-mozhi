"""Comprehensive test suite for Dynamic Tool & MCP Registry (app.services.tool_registry)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from google.genai import types

from app.services.key_vault import (
    AuthType,
    KeyRecord,
    KeyVault,
    VerificationResult,
)
from app.services.tool_registry import (
    MCPServerConfig,
    ToolBinding,
    ToolRegistry,
    ToolSourceType,
    get_tool_registry,
    json_schema_to_gemini_schema,
    resolve_json_ref,
    sanitize_identifier,
)


@pytest.fixture
def clean_vault():
    """Provides an isolated KeyVault instance."""
    return KeyVault()


@pytest.fixture
def clean_registry(clean_vault):
    """Provides a fresh isolated ToolRegistry instance bound to clean_vault."""
    return ToolRegistry(key_vault=clean_vault, http_timeout=5.0)


# =========================================================================
# 1. JSON Schema & Ref Resolver Tests
# =========================================================================

def test_resolve_json_ref():
    spec = {
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            }
        }
    }
    resolved = resolve_json_ref(spec, "#/components/schemas/User")
    assert resolved == {"type": "object", "properties": {"name": {"type": "string"}}}
    assert resolve_json_ref(spec, "#/components/schemas/Missing") == {}
    assert resolve_json_ref(spec, "invalid_ref") == {}


def test_json_schema_to_gemini_schema_types():
    raw_schema = {
        "type": "object",
        "description": "User profile schema",
        "properties": {
            "name": {"type": "string", "description": "Full name"},
            "age": {"type": "integer", "description": "Age in years"},
            "balance": {"type": "number", "description": "Account balance"},
            "is_vip": {"type": "boolean", "description": "VIP member status"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "User tags",
            },
            "status": {
                "type": "string",
                "enum": ["active", "suspended", "pending"],
            },
        },
        "required": ["name", "status"],
    }

    schema = json_schema_to_gemini_schema(raw_schema)
    assert isinstance(schema, types.Schema)
    assert schema.type == types.Type.OBJECT
    assert schema.description == "User profile schema"
    assert "name" in schema.properties
    assert schema.properties["name"].type == types.Type.STRING
    assert schema.properties["age"].type == types.Type.INTEGER
    assert schema.properties["balance"].type == types.Type.NUMBER
    assert schema.properties["is_vip"].type == types.Type.BOOLEAN
    assert schema.properties["tags"].type == types.Type.ARRAY
    assert schema.properties["tags"].items.type == types.Type.STRING
    assert schema.properties["status"].enum == ["active", "suspended", "pending"]
    assert "name" in schema.required
    assert "status" in schema.required


def test_json_schema_to_gemini_schema_with_ref():
    spec = {
        "components": {
            "schemas": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                    "required": ["city"],
                }
            }
        }
    }
    raw_schema = {
        "type": "object",
        "properties": {
            "address": {"$ref": "#/components/schemas/Address"},
        },
    }
    schema = json_schema_to_gemini_schema(raw_schema, root_spec=spec)
    assert isinstance(schema, types.Schema)
    assert "address" in schema.properties
    addr_schema = schema.properties["address"]
    assert addr_schema.type == types.Type.OBJECT
    assert "city" in addr_schema.properties
    assert addr_schema.properties["city"].type == types.Type.STRING


# =========================================================================
# 2. Key Vault Sync & Tool Declarations Generation
# =========================================================================

@pytest.mark.asyncio
async def test_sync_from_vault_basic(clean_vault, clean_registry):
    # Register an active WeatherAPI key
    await clean_vault.register_key(
        api_key="weather_secret_key_123",
        provider="WeatherAPI",
        endpoint_url="https://api.weatherapi.com/v1/current.json",
        session_id="test_session",
        auth_type="api_key_query",
        auth_param_name="key",
        skip_verification=True,
    )

    # Register an active GitHub key
    await clean_vault.register_key(
        api_key="ghp_octocat_secret_456",
        provider="GitHub",
        endpoint_url="https://api.github.com/user",
        session_id="test_session",
        auth_type="bearer",
        skip_verification=True,
    )

    # Sync from vault
    count = await clean_registry.sync_from_vault("test_session")
    assert count == 2

    # Check generated google-genai tools
    tools = clean_registry.get_tools("test_session")
    assert len(tools) == 1
    tool_obj = tools[0]
    assert isinstance(tool_obj, types.Tool)
    assert len(tool_obj.function_declarations) == 2

    fn_names = {fd.name for fd in tool_obj.function_declarations}
    assert any("weather" in name for name in fn_names)
    assert any("github" in name for name in fn_names)

    # Verify compatibility with GenerateContentConfig
    config = types.GenerateContentConfig(tools=tools)
    assert len(config.tools) == 1


@pytest.mark.asyncio
async def test_sync_from_vault_prunes_inactive(clean_vault, clean_registry):
    # Register 2 keys
    r1 = await clean_vault.register_key(
        api_key="stripe_key_1",
        provider="Stripe",
        endpoint_url="https://api.stripe.com/v1/charges",
        session_id="sess_prune",
        skip_verification=True,
    )
    r2 = await clean_vault.register_key(
        api_key="slack_key_1",
        provider="Slack",
        endpoint_url="https://slack.com/api/chat.postMessage",
        session_id="sess_prune",
        skip_verification=True,
    )

    count = await clean_registry.sync_from_vault("sess_prune")
    assert count == 2
    assert len(clean_registry.get_function_declarations("sess_prune")) == 2

    # Deactivate Slack key in vault
    r2.is_active = False

    # Sync again - Slack should be pruned
    new_count = await clean_registry.sync_from_vault("sess_prune")
    assert new_count == 1
    remaining_names = [fd.name for fd in clean_registry.get_function_declarations("sess_prune")]
    assert any("stripe" in name for name in remaining_names)
    assert not any("slack" in name for name in remaining_names)


# =========================================================================
# 3. OpenAPI 3.x Specification Parsing Tests
# =========================================================================

def test_parse_openapi_spec(clean_registry):
    sample_spec = {
        "openapi": "3.0.0",
        "info": {"title": "Petstore Subset", "version": "1.0.0"},
        "servers": [{"url": "https://api.petstore.com/v1"}],
        "paths": {
            "/pets/{petId}": {
                "get": {
                    "operationId": "getPetById",
                    "summary": "Find pet by ID",
                    "description": "Returns a single pet",
                    "parameters": [
                        {
                            "name": "petId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                            "description": "ID of pet to return",
                        },
                        {
                            "name": "include_history",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "boolean"},
                            "description": "Whether to include medical history",
                        },
                    ],
                },
                "delete": {
                    "operationId": "deletePet",
                    "summary": "Deletes a pet",
                    "parameters": [
                        {
                            "name": "petId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                        }
                    ],
                },
            },
            "/pets": {
                "post": {
                    "operationId": "createPet",
                    "summary": "Create a new pet",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string", "description": "Pet name"},
                                        "category": {
                                            "type": "string",
                                            "enum": ["dog", "cat", "bird"],
                                            "description": "Pet category",
                                        },
                                    },
                                    "required": ["name"],
                                }
                            }
                        },
                    },
                }
            },
        },
    }

    bindings = clean_registry.parse_openapi_spec(
        spec=sample_spec,
        provider="Petstore",
        base_url="https://api.petstore.com/v1",
        session_id="pet_session",
    )

    assert len(bindings) == 3
    binding_map = {b.name: b for b in bindings}

    # 1. getPetById
    get_binding = binding_map["petstore_getpetbyid"]
    assert get_binding.http_method == "GET"
    assert get_binding.endpoint_url == "https://api.petstore.com/v1/pets/{petId}"
    assert get_binding.param_locations["petId"] == "path"
    assert get_binding.param_locations["include_history"] == "query"

    fd_get = get_binding.to_function_declaration()
    assert isinstance(fd_get, types.FunctionDeclaration)
    assert fd_get.name == "petstore_getpetbyid"
    assert "petId" in fd_get.parameters.properties

    # 2. createPet
    post_binding = binding_map["petstore_createpet"]
    assert post_binding.http_method == "POST"
    assert post_binding.param_locations["name"] == "body"
    assert post_binding.param_locations["category"] == "body"

    fd_post = post_binding.to_function_declaration()
    assert "name" in fd_post.parameters.required
    assert fd_post.parameters.properties["category"].enum == ["dog", "cat", "bird"]


# =========================================================================
# 4. Dynamic MCP Server Registration & Tool Discovery
# =========================================================================

@pytest.mark.asyncio
async def test_register_mcp_server_native_tool(clean_registry):
    # Register remote MCP server without dynamic discovery probe
    mcp_config = await clean_registry.register_mcp_server(
        name="github-mcp",
        url="https://api.githubcopilot.com/mcp",
        headers={"Authorization": "Bearer mcp_secret_token"},
        session_id="mcp_session",
        discover_tools=False,
    )
    assert mcp_config.name == "github-mcp"
    assert mcp_config.is_active

    # Check google-genai tools output
    tools = clean_registry.get_tools("mcp_session")
    assert len(tools) == 1
    tool_obj = tools[0]
    assert tool_obj.mcp_servers is not None
    assert len(tool_obj.mcp_servers) == 1
    assert tool_obj.mcp_servers[0].name == "github-mcp"
    assert tool_obj.mcp_servers[0].streamable_http_transport.url == "https://api.githubcopilot.com/mcp"
    assert tool_obj.mcp_servers[0].streamable_http_transport.headers == {"Authorization": "Bearer mcp_secret_token"}


@pytest.mark.asyncio
async def test_mcp_dynamic_discovery_and_call(clean_registry):
    # Mock MCP server responding to 'tools/list' and 'tools/call'
    mock_tools_list_response = {
        "jsonrpc": "2.0",
        "id": "mcp_disc_filesystem",
        "result": {
            "tools": [
                {
                    "name": "read_file",
                    "description": "Reads contents of a file",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Absolute file path"}
                        },
                        "required": ["path"],
                    },
                }
            ]
        },
    }

    mock_tools_call_response = {
        "jsonrpc": "2.0",
        "id": "call_123",
        "result": {
            "content": [{"type": "text", "text": "file content: hello world"}],
            "isError": False,
        },
    }

    async def mock_post(url, json=None, headers=None, timeout=None):
        method = json.get("method")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        if method == "tools/list":
            mock_resp.json.return_value = mock_tools_list_response
        elif method == "tools/call":
            mock_resp.json.return_value = mock_tools_call_response
        return mock_resp

    with patch.object(clean_registry, "_get_http_client", new_callable=AsyncMock) as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=mock_post)
        mock_get_client.return_value = mock_client

        # Register server with discovery enabled
        config = await clean_registry.register_mcp_server(
            name="filesystem",
            url="http://localhost:8080/mcp",
            session_id="mcp_disc_session",
            discover_tools=True,
        )

        assert "mcp_filesystem_read_file" in config.discovered_tools
        binding = clean_registry.get_tool_binding("mcp_filesystem_read_file", "mcp_disc_session")
        assert binding is not None
        assert binding.source_type == ToolSourceType.MCP_SERVER

        # Execute MCP tool on demand
        exec_result = await clean_registry.execute_tool(
            name="mcp_filesystem_read_file",
            arguments={"path": "/workspace/test.txt"},
            session_id="mcp_disc_session",
        )
        assert exec_result["success"] is True
        assert "file content: hello world" in exec_result["data"]
        assert exec_result["provider"] == "filesystem"


# =========================================================================
# 5. On-Demand Execution with Dynamic Key Vault Credentials
# =========================================================================

@pytest.mark.asyncio
async def test_execute_tool_bearer_auth(clean_vault, clean_registry):
    # Register active key with Bearer auth
    await clean_vault.register_key(
        api_key="ghp_octocat_token_999",
        provider="GitHub",
        endpoint_url="https://api.github.com/repos/{owner}/{repo}",
        session_id="exec_sess",
        auth_type="bearer",
        skip_verification=True,
    )
    await clean_registry.sync_from_vault("exec_sess")

    tool_name = clean_registry.get_function_declarations("exec_sess")[0].name

    captured_request = {}

    async def mock_request(method, url, headers=None, params=None, json=None, timeout=None):
        captured_request["method"] = method
        captured_request["url"] = url
        captured_request["headers"] = headers
        captured_request["params"] = params
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"id": 12345, "name": "genai", "owner": "google"}
        return resp

    with patch.object(clean_registry, "_get_http_client", new_callable=AsyncMock) as mock_get_client:
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=mock_request)
        mock_get_client.return_value = mock_client

        result = await clean_registry.execute_tool(
            name=tool_name,
            arguments={"owner": "google", "repo": "genai"},
            session_id="exec_sess",
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        assert result["data"]["name"] == "genai"

        # Verify dynamic path parameter interpolation
        assert captured_request["url"] == "https://api.github.com/repos/google/genai"
        # Verify Bearer auth header injection
        assert captured_request["headers"]["Authorization"] == "Bearer ghp_octocat_token_999"


@pytest.mark.asyncio
async def test_execute_tool_api_key_query_auth(clean_vault, clean_registry):
    # Register active key with query parameter auth
    await clean_vault.register_key(
        api_key="weather_api_token_abc",
        provider="WeatherAPI",
        endpoint_url="https://api.weatherapi.com/v1/current.json",
        session_id="query_sess",
        auth_type="api_key_query",
        auth_param_name="key",
        skip_verification=True,
    )
    await clean_registry.sync_from_vault("query_sess")

    tool_name = clean_registry.get_function_declarations("query_sess")[0].name

    captured_request = {}

    async def mock_request(method, url, headers=None, params=None, json=None, timeout=None):
        captured_request["params"] = params
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"location": {"name": "London"}, "current": {"temp_c": 15.0}}
        return resp

    with patch.object(clean_registry, "_get_http_client", new_callable=AsyncMock) as mock_get_client:
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=mock_request)
        mock_get_client.return_value = mock_client

        result = await clean_registry.execute_tool(
            name=tool_name,
            arguments={"location": "London"},
            session_id="query_sess",
        )

        assert result["success"] is True
        assert captured_request["params"]["key"] == "weather_api_token_abc"
        assert captured_request["params"]["location"] == "London"


@pytest.mark.asyncio
async def test_execute_tool_rejects_inactive_key(clean_vault, clean_registry):
    # Register key then deactivate
    record = await clean_vault.register_key(
        api_key="stripe_revoked_key",
        provider="Stripe",
        endpoint_url="https://api.stripe.com/v1/charges",
        session_id="inactive_sess",
        skip_verification=True,
    )
    await clean_registry.sync_from_vault("inactive_sess")
    tool_name = clean_registry.get_function_declarations("inactive_sess")[0].name

    # Deactivate the key in Key Vault
    record.is_active = False

    # Execute should be rejected on demand
    result = await clean_registry.execute_tool(
        name=tool_name,
        arguments={"amount": 100},
        session_id="inactive_sess",
    )
    assert result["success"] is False
    assert result["status_code"] == 403
    assert "inactive or not found" in result["error"]


@pytest.mark.asyncio
async def test_execute_tool_custom_callable(clean_registry):
    # Register a local Python async function
    async def calculate_tax(amount: float, rate: float = 0.05) -> float:
        return amount * (1.0 + rate)

    clean_registry.register_callable(
        name="calc_tax",
        func=calculate_tax,
        description="Calculates amount with tax",
        session_id="callable_sess",
    )

    result = await clean_registry.execute_tool(
        name="calc_tax",
        arguments={"amount": 100.0, "rate": 0.10},
        session_id="callable_sess",
    )
    assert result["success"] is True
    assert round(result["data"], 2) == 110.0


# =========================================================================
# 6. ToolResponse Helper Tests
# =========================================================================

def test_to_tool_response(clean_registry):
    exec_result = {
        "success": True,
        "status_code": 200,
        "data": {"temp": 22},
        "tool": "weather_current",
        "provider": "WeatherAPI",
    }
    part = clean_registry.to_tool_response("weather_current", exec_result)
    assert isinstance(part, types.Part)
    assert part.function_response.name == "weather_current"
    assert part.function_response.response == exec_result


# =========================================================================
# 7. Inspection & Management Methods
# =========================================================================

def test_list_and_unregister_tools(clean_registry):
    clean_registry.register_callable(
        name="sample_fn",
        func=lambda x: x,
        description="Sample function",
        session_id="inspect_sess",
    )
    tools = clean_registry.list_tools("inspect_sess")
    assert len(tools) == 1
    assert tools[0]["name"] == "sample_fn"

    removed = clean_registry.unregister_tool("sample_fn", "inspect_sess")
    assert removed is True
    assert len(clean_registry.list_tools("inspect_sess")) == 0
