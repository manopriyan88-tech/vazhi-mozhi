"""Dynamic Tool & MCP Registry for Google Gemini.

Converts active 3rd-party API schemas from Key Vault and dynamic MCP server endpoints
into official google-genai SDK Tool declarations (types.Tool, types.FunctionDeclaration,
and types.McpServer). Enables Gemini to call any registered API key on demand with
automatic credential injection, parameter mapping, and HTTP/MCP execution.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, quote, urlencode, urlparse, urlunparse

import httpx
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.key_vault import (
    KeyRecord,
    KeyVault,
    key_vault as default_key_vault,
    mask_api_key,
    sanitize_identifier,
)

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    genai_types = None  # type: ignore
    GENAI_AVAILABLE = False

logger = logging.getLogger("app.services.tool_registry")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)


class ToolSourceType(str, Enum):
    """Source origin of a registered tool."""
    KEY_VAULT = "key_vault"
    OPENAPI = "openapi"
    MCP_SERVER = "mcp_server"
    CUSTOM_CALLABLE = "custom_callable"


def resolve_json_ref(root_spec: Dict[str, Any], ref_path: str) -> Dict[str, Any]:
    """Recursively resolves an internal JSON pointer ref (e.g. #/components/schemas/Pet)."""
    if not ref_path or not ref_path.startswith("#/"):
        return {}
    parts = ref_path.lstrip("#/").split("/")
    curr: Any = root_spec
    for part in parts:
        if isinstance(curr, dict) and part in curr:
            curr = curr[part]
        else:
            return {}
    return curr if isinstance(curr, dict) else {}


def json_schema_to_gemini_schema(
    schema_dict: Dict[str, Any],
    root_spec: Optional[Dict[str, Any]] = None,
    visited_refs: Optional[Set[str]] = None,
) -> Any:
    """Recursively converts a standard JSON Schema / OpenAPI schema dict to a google-genai types.Schema.

    Maps OpenAPI types ('string', 'integer', 'number', 'boolean', 'array', 'object')
    into google.genai.types.Type enum values while preserving descriptions, enums,
    properties, items, and required fields.
    """
    if visited_refs is None:
        visited_refs = set()

    # Resolve $ref if present
    if "$ref" in schema_dict and root_spec:
        ref_target = schema_dict["$ref"]
        if ref_target in visited_refs:
            # Prevent circular reference infinite loops
            return genai_types.Schema(type=genai_types.Type.OBJECT) if (GENAI_AVAILABLE and genai_types) else {"type": "OBJECT"}
        visited_refs.add(ref_target)
        resolved = resolve_json_ref(root_spec, ref_target)
        if resolved:
            schema_dict = {**resolved, **{k: v for k, v in schema_dict.items() if k != "$ref"}}

    raw_type = schema_dict.get("type", "object")
    if isinstance(raw_type, list):
        # e.g., ["string", "null"]
        non_null_types = [t for t in raw_type if t != "null"]
        raw_type = non_null_types[0] if non_null_types else "string"

    type_str = str(raw_type).lower()
    description = schema_dict.get("description")
    enum_values = schema_dict.get("enum")
    required_fields = schema_dict.get("required")

    if not GENAI_AVAILABLE or not genai_types:
        # Fallback dictionary schema representation
        gemini_type_str = {
            "string": "STRING",
            "integer": "INTEGER",
            "number": "NUMBER",
            "float": "NUMBER",
            "double": "NUMBER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT",
        }.get(type_str, "OBJECT")
        result: Dict[str, Any] = {"type": gemini_type_str}
        if description:
            result["description"] = description
        if enum_values:
            result["enum"] = [str(e) for e in enum_values]
        if required_fields:
            result["required"] = required_fields
        if "properties" in schema_dict and isinstance(schema_dict["properties"], dict):
            result["properties"] = {
                k: json_schema_to_gemini_schema(v, root_spec, visited_refs)
                for k, v in schema_dict["properties"].items()
            }
        if "items" in schema_dict and isinstance(schema_dict["items"], dict):
            result["items"] = json_schema_to_gemini_schema(schema_dict["items"], root_spec, visited_refs)
        return result

    # Using official google.genai types
    type_map = {
        "string": genai_types.Type.STRING,
        "integer": genai_types.Type.INTEGER,
        "int32": genai_types.Type.INTEGER,
        "int64": genai_types.Type.INTEGER,
        "number": genai_types.Type.NUMBER,
        "float": genai_types.Type.NUMBER,
        "double": genai_types.Type.NUMBER,
        "boolean": genai_types.Type.BOOLEAN,
        "array": genai_types.Type.ARRAY,
        "object": genai_types.Type.OBJECT,
    }
    gtype = type_map.get(type_str, genai_types.Type.OBJECT)

    props: Optional[Dict[str, Any]] = None
    if "properties" in schema_dict and isinstance(schema_dict["properties"], dict):
        props = {}
        for prop_name, prop_def in schema_dict["properties"].items():
            if isinstance(prop_def, dict):
                props[prop_name] = json_schema_to_gemini_schema(prop_def, root_spec, visited_refs)

    items_schema: Optional[Any] = None
    if "items" in schema_dict and isinstance(schema_dict["items"], dict):
        items_schema = json_schema_to_gemini_schema(schema_dict["items"], root_spec, visited_refs)

    clean_enums = [str(e) for e in enum_values] if enum_values else None
    clean_required = list(required_fields) if (required_fields and isinstance(required_fields, list)) else None

    return genai_types.Schema(
        type=gtype,
        description=description,
        properties=props,
        required=clean_required,
        enum=clean_enums,
        items=items_schema,
    )


class ToolBinding(BaseModel):
    """Runtime dispatch binding that connects a Gemini function declaration to an API execution target."""

    name: str = Field(description="Unique Gemini function name")
    provider: str = Field(description="Provider or service name")
    session_id: str = Field(default="default", description="Associated session namespace")
    source_type: ToolSourceType = Field(description="Source origin of this tool")
    description: str = Field(default="", description="Function summary and description")
    endpoint_url: str = Field(default="", description="Target endpoint URL or URL template with path variables")
    http_method: str = Field(default="GET", description="HTTP method (GET, POST, PUT, DELETE, etc.)")
    auth_type: str = Field(default="bearer", description="Auth delivery: bearer, api_key_header, api_key_query, custom")
    auth_header_name: str = Field(default="Authorization", description="Header name for auth token")
    auth_param_name: str = Field(default="api_key", description="Query param name for auth token")
    custom_headers: Dict[str, str] = Field(default_factory=dict, description="Custom static headers")
    param_locations: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps argument name to location ('path', 'query', 'header', 'body')",
    )
    declaration_dict: Dict[str, Any] = Field(description="Raw function declaration dictionary")
    mcp_server_name: Optional[str] = Field(default=None, description="Name of MCP server if source is MCP")
    mcp_tool_name: Optional[str] = Field(default=None, description="Original tool name on MCP server")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"arbitrary_types_allowed": True}

    def to_function_declaration(self) -> Any:
        """Converts stored declaration into an official google.genai.types.FunctionDeclaration."""
        if not GENAI_AVAILABLE or not genai_types:
            return self.declaration_dict
        try:
            # If parameters is a Schema object or dict, construct FunctionDeclaration
            params = self.declaration_dict.get("parameters")
            if isinstance(params, dict):
                # Convert parameters to genai_types.Schema
                schema_obj = json_schema_to_gemini_schema(params)
                return genai_types.FunctionDeclaration(
                    name=self.name,
                    description=self.description or self.declaration_dict.get("description", ""),
                    parameters=schema_obj,
                )
            return genai_types.FunctionDeclaration(**self.declaration_dict)
        except Exception as exc:
            logger.warning(f"Error constructing FunctionDeclaration for {self.name}: {exc}")
            # Fallback using raw dictionary
            return genai_types.FunctionDeclaration(
                name=self.name,
                description=self.description,
                parameters_json_schema=self.declaration_dict.get("parameters", {"type": "object"}),
            )


class MCPServerConfig(BaseModel):
    """Configuration for a dynamic Model Context Protocol (MCP) server."""

    name: str = Field(description="MCP server identifier")
    url: str = Field(description="Endpoint URL for MCP server (Streamable HTTP / SSE / POST)")
    headers: Dict[str, str] = Field(default_factory=dict, description="Headers including API keys / auth")
    session_id: str = Field(default="default", description="Associated session namespace")
    transport: str = Field(default="streamable_http", description="Transport type")
    timeout_seconds: float = Field(default=30.0, description="HTTP timeout for MCP operations")
    is_active: bool = Field(default=True, description="Whether server is active")
    discovered_tools: List[str] = Field(default_factory=list, description="Names of tools discovered on server")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_genai_mcp_server(self) -> Any:
        """Converts to google.genai types.McpServer with StreamableHttpTransport."""
        if not GENAI_AVAILABLE or not genai_types:
            return {"name": self.name, "url": self.url, "headers": self.headers}
        try:
            transport = genai_types.StreamableHttpTransport(
                url=self.url,
                headers=self.headers if self.headers else None,
                timeout=f"{int(self.timeout_seconds)}s",
            )
            return genai_types.McpServer(
                name=self.name,
                streamable_http_transport=transport,
            )
        except Exception as exc:
            logger.warning(f"Could not build types.McpServer for {self.name}: {exc}")
            return None


class ToolExecutionResult(BaseModel):
    """Result of an on-demand tool execution by Gemini."""

    success: bool = Field(description="Whether tool execution completed successfully")
    status_code: int = Field(default=0, description="HTTP status code (or 0 for non-HTTP calls)")
    data: Any = Field(default=None, description="Parsed response payload or output")
    latency_ms: float = Field(default=0.0, description="Call duration in milliseconds")
    tool: str = Field(description="Name of the executed tool")
    provider: str = Field(description="Target provider or MCP server")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")


class ToolRegistry:
    """Dynamic Tool & MCP Registry for Google Gemini.

    - Loads active 3rd-party API schemas from KeyVault and transforms them into
      types.Tool declarations compatible with google-genai SDK.
    - Supports dynamic loading and inspection of MCP servers and their exposed tools.
    - Executes registered API tools on demand, injecting live active API keys and credentials.
    """

    def __init__(
        self,
        key_vault: Optional[KeyVault] = None,
        http_timeout: float = 30.0,
    ) -> None:
        self.key_vault: KeyVault = key_vault or default_key_vault
        self.http_timeout: float = http_timeout
        # Storage: {session_id: {function_name: ToolBinding}}
        self._bindings: Dict[str, Dict[str, ToolBinding]] = {}
        # Storage: {session_id: {server_name: MCPServerConfig}}
        self._mcp_servers: Dict[str, Dict[str, MCPServerConfig]] = {}
        # Registered local Python callables: {session_id: {function_name: Callable}}
        self._callables: Dict[str, Dict[str, Callable]] = {}
        self._lock = asyncio.Lock()
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Retrieves or creates the reusable httpx.AsyncClient."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.http_timeout),
                follow_redirects=True,
                verify=True,
            )
        return self._http_client

    # -------------------------------------------------------------------------
    # OpenAPI Parsing & Conversion
    # -------------------------------------------------------------------------

    def parse_openapi_spec(
        self,
        spec: Dict[str, Any],
        provider: str,
        base_url: str = "",
        session_id: str = "default",
        auth_type: str = "bearer",
        auth_header_name: str = "Authorization",
        auth_param_name: str = "api_key",
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> List[ToolBinding]:
        """Parses an OpenAPI 3.x or Swagger 2.0 specification into Gemini ToolBindings."""
        bindings: List[ToolBinding] = []
        paths = spec.get("paths", {})
        if not isinstance(paths, dict):
            return bindings

        # Derive base URL from spec servers if not supplied
        if not base_url:
            servers = spec.get("servers", [])
            if servers and isinstance(servers, list) and isinstance(servers[0], dict):
                base_url = servers[0].get("url", "").rstrip("/")

        for path_pattern, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            path_level_params = path_item.get("parameters", [])

            for method in ("get", "post", "put", "delete", "patch", "head", "options"):
                if method not in path_item or not isinstance(path_item[method], dict):
                    continue

                operation = path_item[method]
                op_id = operation.get("operationId")

                if op_id:
                    fn_base = f"{provider}_{op_id}"
                else:
                    path_slug = re.sub(r"[^a-zA-Z0-9]+", "_", path_pattern.strip("/")).strip("_")
                    fn_base = f"{provider}_{method}_{path_slug}"

                func_name = sanitize_identifier(fn_base)
                summary = operation.get("summary", "")
                description = operation.get("description", "")
                doc = f"{summary}. {description}".strip(" .")
                if not doc:
                    doc = f"Executes {method.upper()} {path_pattern} on {provider}."

                properties: Dict[str, Any] = {}
                required: List[str] = []
                param_locations: Dict[str, str] = {}

                # Combine path-level and operation-level parameters
                combined_params = list(path_level_params) + list(operation.get("parameters", []))

                for p in combined_params:
                    if not isinstance(p, dict):
                        continue
                    p_name = p.get("name")
                    p_in = p.get("in", "query").lower()
                    if not p_name or p_name.lower() in ("api_key", "apikey", "authorization", "token"):
                        continue

                    p_schema = p.get("schema", {})
                    p_desc = p.get("description", f"Parameter '{p_name}' ({p_in})")
                    p_type = p_schema.get("type", "string")

                    properties[p_name] = {
                        "type": p_type,
                        "description": p_desc,
                    }
                    if "enum" in p_schema:
                        properties[p_name]["enum"] = p_schema["enum"]

                    if p.get("required", False) or p_in == "path":
                        required.append(p_name)

                    param_locations[p_name] = p_in

                # Parse requestBody (application/json)
                req_body = operation.get("requestBody", {})
                if isinstance(req_body, dict):
                    content = req_body.get("content", {})
                    json_media = content.get("application/json", {})
                    body_schema = json_media.get("schema", {})

                    if "$ref" in body_schema:
                        body_schema = resolve_json_ref(spec, body_schema["$ref"])

                    if isinstance(body_schema, dict):
                        body_props = body_schema.get("properties", {})
                        body_req = body_schema.get("required", [])

                        for b_name, b_def in body_props.items():
                            if isinstance(b_def, dict):
                                properties[b_name] = {
                                    "type": b_def.get("type", "string"),
                                    "description": b_def.get("description", f"Body property '{b_name}'"),
                                }
                                if "enum" in b_def:
                                    properties[b_name]["enum"] = b_def["enum"]
                                if b_name in body_req:
                                    required.append(b_name)
                                param_locations[b_name] = "body"

                decl_dict = {
                    "name": func_name,
                    "description": doc,
                    "parameters": {
                        "type": "OBJECT",
                        "properties": properties,
                        "required": required,
                    },
                }

                endpoint = f"{base_url.rstrip('/')}/{path_pattern.lstrip('/')}" if base_url else path_pattern

                binding = ToolBinding(
                    name=func_name,
                    provider=provider,
                    session_id=session_id,
                    source_type=ToolSourceType.OPENAPI,
                    description=doc,
                    endpoint_url=endpoint,
                    http_method=method.upper(),
                    auth_type=auth_type,
                    auth_header_name=auth_header_name,
                    auth_param_name=auth_param_name,
                    custom_headers=custom_headers or {},
                    param_locations=param_locations,
                    declaration_dict=decl_dict,
                )
                bindings.append(binding)

        return bindings

    # -------------------------------------------------------------------------
    # Key Vault Synchronization
    # -------------------------------------------------------------------------

    async def sync_from_vault(self, session_id: str = "default") -> int:
        """Syncs all currently active 3rd-party API schemas from Key Vault into the registry.

        Converts each active record's schema (or OpenAPI spec) into a Gemini FunctionDeclaration,
        binds it for on-demand execution, and removes tools for keys that became inactive or deleted.
        Returns the number of active tools registered.
        """
        session_clean = (session_id or "default").strip()
        active_records = await self.key_vault.get_active_keys(session_clean)

        async with self._lock:
            if session_clean not in self._bindings:
                self._bindings[session_clean] = {}

            # Collect existing tool names from Key Vault to prune deactivated ones
            existing_vault_tools = {
                name: b for name, b in self._bindings[session_clean].items()
                if b.source_type in (ToolSourceType.KEY_VAULT, ToolSourceType.OPENAPI)
            }

            active_providers = {r.provider.lower() for r in active_records}
            # Prune tools whose provider is no longer active
            for name, b in list(existing_vault_tools.items()):
                if b.provider.lower() not in active_providers:
                    del self._bindings[session_clean][name]
                    logger.info(f"Pruned deactivated tool '{name}' for provider '{b.provider}'")

            count = 0
            for record in active_records:
                # 1. If record contains a full OpenAPI specification, parse all endpoints
                if record.openapi_spec:
                    parsed_bindings = self.parse_openapi_spec(
                        spec=record.openapi_spec,
                        provider=record.provider,
                        base_url=record.endpoint_url,
                        session_id=session_clean,
                        auth_type=record.auth_type,
                        auth_header_name=record.auth_header_name,
                        auth_param_name=record.auth_param_name,
                        custom_headers=record.custom_headers,
                    )
                    for pb in parsed_bindings:
                        self._bindings[session_clean][pb.name] = pb
                        count += 1
                else:
                    # 2. Use stored function_declaration from KeyRecord
                    fn_decl = record.function_declaration
                    func_name = fn_decl.get("name", sanitize_identifier(f"{record.provider}_tool"))
                    desc = fn_decl.get("description", f"API tool for {record.provider}")

                    # Infer parameter locations
                    params_dict = fn_decl.get("parameters", {})
                    props = params_dict.get("properties", {}) if isinstance(params_dict, dict) else {}
                    param_locs: Dict[str, str] = {}
                    for p in props.keys():
                        if f"{{{p}}}" in record.endpoint_url:
                            param_locs[p] = "path"
                        elif record.http_method.upper() in ("POST", "PUT", "PATCH"):
                            param_locs[p] = "body"
                        else:
                            param_locs[p] = "query"

                    binding = ToolBinding(
                        name=func_name,
                        provider=record.provider,
                        session_id=session_clean,
                        source_type=ToolSourceType.KEY_VAULT,
                        description=desc,
                        endpoint_url=record.endpoint_url,
                        http_method=record.http_method,
                        auth_type=record.auth_type,
                        auth_header_name=record.auth_header_name,
                        auth_param_name=record.auth_param_name,
                        custom_headers=record.custom_headers,
                        param_locations=param_locs,
                        declaration_dict=fn_decl,
                    )
                    self._bindings[session_clean][func_name] = binding
                    count += 1

            logger.info(f"Synced {count} active tools from Key Vault for session='{session_clean}'")
            return count

    def sync_from_vault_sync(self, session_id: str = "default") -> int:
        """Synchronous helper for sync_from_vault."""
        try:
            loop = asyncio.get_running_loop()
            # If already in an async event loop, create task or run in executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(self.sync_from_vault(session_id))).result()
        except RuntimeError:
            return asyncio.run(self.sync_from_vault(session_id))

    # -------------------------------------------------------------------------
    # Dynamic MCP Server Management & Tool Discovery
    # -------------------------------------------------------------------------

    async def register_mcp_server(
        self,
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        session_id: str = "default",
        discover_tools: bool = True,
        timeout_seconds: float = 30.0,
    ) -> MCPServerConfig:
        """Registers an MCP server and optionally queries it dynamically for exposed tools.

        Stores the server configuration for Gemini native McpServer transport and binds
        discovered tools for dynamic execution.
        """
        session_clean = (session_id or "default").strip()
        config = MCPServerConfig(
            name=name,
            url=url,
            headers=headers or {},
            session_id=session_clean,
            timeout_seconds=timeout_seconds,
            is_active=True,
        )

        async with self._lock:
            if session_clean not in self._mcp_servers:
                self._mcp_servers[session_clean] = {}
            self._mcp_servers[session_clean][name] = config

        if discover_tools:
            try:
                discovered = await self.load_mcp_tools_dynamically(
                    server_name=name,
                    url=url,
                    headers=headers,
                    session_id=session_clean,
                    timeout_seconds=timeout_seconds,
                )
                config.discovered_tools = [d.name if hasattr(d, "name") else d.get("name") for d in discovered]
                logger.info(f"Discovered {len(discovered)} tools from MCP server '{name}'")
            except Exception as exc:
                logger.warning(f"Dynamic discovery on MCP server '{name}' encountered an error: {exc}")

        return config

    async def load_mcp_tools_dynamically(
        self,
        server_name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        session_id: str = "default",
        timeout_seconds: float = 15.0,
    ) -> List[Any]:
        """Probes an MCP server via JSON-RPC 'tools/list' and converts exposed tools to FunctionDeclarations."""
        session_clean = (session_id or "default").strip()
        req_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream, */*",
            "User-Agent": "AdaptiveKitchen-MCPClient/1.0",
        }
        if headers:
            req_headers.update(headers)

        payload = {
            "jsonrpc": "2.0",
            "id": f"mcp_disc_{sanitize_identifier(server_name)}",
            "method": "tools/list",
            "params": {},
        }

        client = await self._get_http_client()
        resp = await client.post(url, json=payload, headers=req_headers, timeout=timeout_seconds)

        if resp.status_code != 200:
            logger.warning(f"MCP server '{server_name}' returned HTTP {resp.status_code} during tool discovery.")
            return []

        data = resp.json()
        tools_list = data.get("result", {}).get("tools", [])
        declarations = []

        async with self._lock:
            if session_clean not in self._bindings:
                self._bindings[session_clean] = {}

            for mcp_tool in tools_list:
                raw_name = mcp_tool.get("name", "tool")
                func_name = sanitize_identifier(f"mcp_{server_name}_{raw_name}")
                desc = mcp_tool.get("description", f"MCP tool '{raw_name}' from server '{server_name}'")
                input_schema = mcp_tool.get("inputSchema", {"type": "object", "properties": {}})

                decl_dict = {
                    "name": func_name,
                    "description": desc,
                    "parameters": input_schema,
                }

                binding = ToolBinding(
                    name=func_name,
                    provider=server_name,
                    session_id=session_clean,
                    source_type=ToolSourceType.MCP_SERVER,
                    description=desc,
                    endpoint_url=url,
                    http_method="POST",
                    custom_headers=headers or {},
                    declaration_dict=decl_dict,
                    mcp_server_name=server_name,
                    mcp_tool_name=raw_name,
                )
                self._bindings[session_clean][func_name] = binding
                declarations.append(binding.to_function_declaration())

        return declarations

    def unregister_mcp_server(self, name: str, session_id: str = "default") -> bool:
        """Removes an MCP server and its associated tools."""
        session_clean = (session_id or "default").strip()
        server_dict = self._mcp_servers.get(session_clean, {})
        if name in server_dict:
            del server_dict[name]
            # Prune associated tools
            if session_clean in self._bindings:
                for fname, b in list(self._bindings[session_clean].items()):
                    if b.mcp_server_name == name:
                        del self._bindings[session_clean][fname]
            return True
        return False

    # -------------------------------------------------------------------------
    # Custom Callables Registration
    # -------------------------------------------------------------------------

    def register_callable(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        session_id: str = "default",
    ) -> ToolBinding:
        """Registers a custom Python async or sync callable as a Gemini tool."""
        session_clean = (session_id or "default").strip()
        clean_name = sanitize_identifier(name)

        if parameters is None:
            # Inspect signature
            sig = inspect.signature(func)
            props: Dict[str, Any] = {}
            req: List[str] = []
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                props[param_name] = {"type": "string", "description": f"Parameter '{param_name}'"}
                if param.default is inspect.Parameter.empty:
                    req.append(param_name)
            parameters = {"type": "OBJECT", "properties": props, "required": req}

        decl_dict = {
            "name": clean_name,
            "description": description,
            "parameters": parameters,
        }

        binding = ToolBinding(
            name=clean_name,
            provider="local",
            session_id=session_clean,
            source_type=ToolSourceType.CUSTOM_CALLABLE,
            description=description,
            declaration_dict=decl_dict,
        )

        if session_clean not in self._bindings:
            self._bindings[session_clean] = {}
        if session_clean not in self._callables:
            self._callables[session_clean] = {}

        self._bindings[session_clean][clean_name] = binding
        self._callables[session_clean][clean_name] = func
        return binding

    # -------------------------------------------------------------------------
    # Tool Generation Compatible with google-genai SDK
    # -------------------------------------------------------------------------

    def get_function_declarations(self, session_id: str = "default") -> List[Any]:
        """Returns all current types.FunctionDeclaration objects for the session."""
        session_clean = (session_id or "default").strip()
        bindings = self._bindings.get(session_clean, {})
        declarations: List[Any] = []
        for b in bindings.values():
            declarations.append(b.to_function_declaration())
        return declarations

    def get_mcp_servers(self, session_id: str = "default") -> List[Any]:
        """Returns all active types.McpServer declarations for the session."""
        session_clean = (session_id or "default").strip()
        servers = self._mcp_servers.get(session_clean, {})
        mcp_list: List[Any] = []
        for s in servers.values():
            if s.is_active:
                genai_mcp = s.to_genai_mcp_server()
                if genai_mcp is not None:
                    mcp_list.append(genai_mcp)
        return mcp_list

    def get_tools(
        self,
        session_id: str = "default",
        include_mcp: bool = True,
    ) -> List[Any]:
        """Returns a list of types.Tool declarations fully compatible with the google-genai SDK.

        Compatible directly with:
            config = types.GenerateContentConfig(tools=tool_registry.get_tools())
            response = client.models.generate_content(model=..., contents=..., config=config)
        """
        session_clean = (session_id or "default").strip()
        function_declarations = self.get_function_declarations(session_clean)
        mcp_servers = self.get_mcp_servers(session_clean) if include_mcp else []

        if not GENAI_AVAILABLE or not genai_types:
            # Fallback when google-genai is not imported
            tools = []
            if function_declarations:
                tools.append({"function_declarations": function_declarations})
            if mcp_servers:
                tools.append({"mcp_servers": mcp_servers})
            return tools

        tools: List[Any] = []

        # Return a unified types.Tool containing function declarations and/or MCP servers
        if function_declarations or mcp_servers:
            tool = genai_types.Tool(
                function_declarations=function_declarations if function_declarations else None,
                mcp_servers=mcp_servers if mcp_servers else None,
            )
            tools.append(tool)

        return tools

    # -------------------------------------------------------------------------
    # On-Demand Dynamic Execution
    # -------------------------------------------------------------------------

    async def execute_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """Executes a tool call on demand with live registered credentials.

        Gemini function calls specify the tool name and argument dictionary.
        This method:
        1. Resolves the tool binding.
        2. Retrieves the current API key / credentials dynamically from Key Vault on demand.
        3. Formats path, query, header, and body parameters.
        4. Injects authorization headers or query parameters.
        5. Executes the asynchronous HTTP request or MCP tool call.
        6. Returns structured response data suitable for Gemini's ToolResponse.
        """
        session_clean = (session_id or "default").strip()
        args = arguments or {}

        # Look up binding
        session_bindings = self._bindings.get(session_clean, {})
        binding = session_bindings.get(name)

        if not binding:
            return {
                "success": False,
                "status_code": 404,
                "error": f"Tool '{name}' is not registered in session '{session_clean}'.",
                "tool": name,
                "provider": "unknown",
            }

        start_time = asyncio.get_event_loop().time()

        # Handle Custom Callable
        if binding.source_type == ToolSourceType.CUSTOM_CALLABLE:
            func = self._callables.get(session_clean, {}).get(name)
            if not func:
                return {
                    "success": False,
                    "status_code": 500,
                    "error": f"Handler for callable tool '{name}' not found.",
                    "tool": name,
                    "provider": "local",
                }
            try:
                if inspect.iscoroutinefunction(func):
                    result_data = await func(**args)
                else:
                    result_data = func(**args)
                latency = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)
                return {
                    "success": True,
                    "status_code": 200,
                    "data": result_data,
                    "latency_ms": latency,
                    "tool": name,
                    "provider": "local",
                }
            except Exception as exc:
                latency = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)
                logger.error(f"Callable execution failed for {name}: {exc}", exc_info=True)
                return {
                    "success": False,
                    "status_code": 500,
                    "error": str(exc),
                    "latency_ms": latency,
                    "tool": name,
                    "provider": "local",
                }

        # Handle MCP Tool Call
        if binding.source_type == ToolSourceType.MCP_SERVER:
            mcp_server_name = binding.mcp_server_name or binding.provider
            mcp_tool_name = binding.mcp_tool_name or name
            server_config = self._mcp_servers.get(session_clean, {}).get(mcp_server_name)

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/plain, */*",
                "User-Agent": "AdaptiveKitchen-MCPClient/1.0",
            }
            if server_config and server_config.headers:
                headers.update(server_config.headers)

            payload = {
                "jsonrpc": "2.0",
                "id": f"call_{name}_{int(start_time * 1000)}",
                "method": "tools/call",
                "params": {
                    "name": mcp_tool_name,
                    "arguments": args,
                },
            }

            try:
                client = await self._get_http_client()
                resp = await client.post(
                    binding.endpoint_url,
                    json=payload,
                    headers=headers,
                    timeout=server_config.timeout_seconds if server_config else self.http_timeout,
                )
                latency = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)

                if resp.status_code >= 400:
                    return {
                        "success": False,
                        "status_code": resp.status_code,
                        "error": f"MCP server responded with HTTP {resp.status_code}: {resp.text[:300]}",
                        "latency_ms": latency,
                        "tool": name,
                        "provider": mcp_server_name,
                    }

                resp_data = resp.json()
                result = resp_data.get("result", {})
                is_error = result.get("isError", False)
                content = result.get("content", [])

                # Extract content texts
                texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
                output = "\n".join(texts) if texts else result

                return {
                    "success": not is_error,
                    "status_code": 200,
                    "data": output,
                    "latency_ms": latency,
                    "tool": name,
                    "provider": mcp_server_name,
                }
            except Exception as exc:
                latency = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)
                return {
                    "success": False,
                    "status_code": 500,
                    "error": f"MCP tool execution failed: {type(exc).__name__}: {str(exc)}",
                    "latency_ms": latency,
                    "tool": name,
                    "provider": mcp_server_name,
                }

        # Handle 3rd-Party API Call (Key Vault)
        key_record = await self.key_vault.get_key(session_clean, binding.provider)
        if not key_record or not key_record.is_active:
            return {
                "success": False,
                "status_code": 403,
                "error": f"API key for provider '{binding.provider}' is inactive or not found in Key Vault.",
                "tool": name,
                "provider": binding.provider,
            }

        # Prepare HTTP execution with dynamic live API key
        api_key = key_record.api_key
        headers = {
            "User-Agent": "AdaptiveKitchen-ToolExecutor/1.0",
            "Accept": "application/json, text/plain, */*",
        }
        if binding.custom_headers:
            headers.update(binding.custom_headers)
        if key_record.custom_headers:
            headers.update(key_record.custom_headers)

        # Inject Authentication credentials
        auth_type = key_record.auth_type.lower()
        query_params: Dict[str, Any] = {}

        if auth_type in ("bearer", "token"):
            headers[key_record.auth_header_name] = f"Bearer {api_key}".strip()
        elif auth_type in ("api_key_header", "header", "apikey"):
            headers[key_record.auth_header_name] = api_key
        elif auth_type in ("api_key_query", "query"):
            query_params[key_record.auth_param_name] = api_key
        elif auth_type == "custom":
            headers[key_record.auth_header_name] = api_key

        # URL path parameter interpolation
        endpoint = binding.endpoint_url or key_record.endpoint_url
        remaining_args = dict(args)

        path_vars = re.findall(r"\{([a-zA-Z0-9_]+)\}", endpoint)
        for var in path_vars:
            if var in remaining_args:
                val = str(remaining_args.pop(var))
                endpoint = endpoint.replace(f"{{{var}}}", quote(val, safe=""))

        # Partition arguments into query params vs JSON body
        body_data: Optional[Dict[str, Any]] = None
        method = binding.http_method.upper()

        if method in ("GET", "HEAD", "DELETE"):
            for k, v in remaining_args.items():
                query_params[k] = v
        else:
            # POST / PUT / PATCH
            body_dict: Dict[str, Any] = {}
            for k, v in remaining_args.items():
                loc = binding.param_locations.get(k)
                if loc == "query":
                    query_params[k] = v
                elif loc == "header":
                    headers[k] = str(v)
                else:
                    body_dict[k] = v
            body_data = body_dict if body_dict else None

        try:
            client = await self._get_http_client()
            resp = await client.request(
                method=method,
                url=endpoint,
                headers=headers,
                params=query_params if query_params else None,
                json=body_data,
                timeout=self.http_timeout,
            )
            latency = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)

            data_payload: Any = None
            try:
                data_payload = resp.json()
            except Exception:
                data_payload = resp.text

            is_ok = 200 <= resp.status_code < 300

            logger.info(
                f"Executed tool '{name}' ({binding.provider}) -> HTTP {resp.status_code} in {latency}ms"
            )

            return {
                "success": is_ok,
                "status_code": resp.status_code,
                "data": data_payload,
                "latency_ms": latency,
                "tool": name,
                "provider": binding.provider,
                "error": None if is_ok else f"Remote API returned HTTP {resp.status_code}",
            }

        except httpx.TimeoutException:
            latency = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)
            return {
                "success": False,
                "status_code": 504,
                "error": f"Tool execution timed out after {self.http_timeout}s.",
                "latency_ms": latency,
                "tool": name,
                "provider": binding.provider,
            }
        except Exception as exc:
            latency = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)
            logger.error(f"HTTP call failed for tool '{name}': {exc}", exc_info=True)
            return {
                "success": False,
                "status_code": 500,
                "error": f"HTTP execution failed: {type(exc).__name__}: {str(exc)}",
                "latency_ms": latency,
                "tool": name,
                "provider": binding.provider,
            }

    def execute_tool_sync(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """Synchronous helper for execute_tool."""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(self.execute_tool(name, arguments, session_id))).result()
        except RuntimeError:
            return asyncio.run(self.execute_tool(name, arguments, session_id))

    def to_tool_response(self, name: str, result: Dict[str, Any]) -> Any:
        """Converts an execution result dictionary into a google-genai types.Part function response."""
        if not GENAI_AVAILABLE or not genai_types:
            return {"name": name, "response": result}
        try:
            return genai_types.Part.from_function_response(name=name, response=result)
        except Exception as exc:
            logger.warning(f"Failed to create Part.from_function_response for {name}: {exc}")
            return {"name": name, "response": result}

    # -------------------------------------------------------------------------
    # Management & Inspection
    # -------------------------------------------------------------------------

    def list_tools(self, session_id: str = "default") -> List[Dict[str, Any]]:
        """Returns safe metadata summary of all registered tools in a session."""
        session_clean = (session_id or "default").strip()
        bindings = self._bindings.get(session_clean, {})
        summary = []
        for b in bindings.values():
            summary.append({
                "name": b.name,
                "provider": b.provider,
                "source_type": b.source_type.value,
                "description": b.description,
                "endpoint_url": b.endpoint_url,
                "http_method": b.http_method,
                "auth_type": b.auth_type,
                "param_locations": b.param_locations,
                "created_at": b.created_at.isoformat(),
            })
        return summary

    def get_tool_binding(self, name: str, session_id: str = "default") -> Optional[ToolBinding]:
        """Retrieves a specific ToolBinding."""
        session_clean = (session_id or "default").strip()
        return self._bindings.get(session_clean, {}).get(name)

    def unregister_tool(self, name: str, session_id: str = "default") -> bool:
        """Removes a tool from the registry."""
        session_clean = (session_id or "default").strip()
        if session_clean in self._bindings and name in self._bindings[session_clean]:
            del self._bindings[session_clean][name]
            if session_clean in self._callables and name in self._callables[session_clean]:
                del self._callables[session_clean][name]
            return True
        return False

    def clear(self, session_id: Optional[str] = None) -> None:
        """Clears registered tools for a session (or all sessions if None)."""
        if session_id:
            s_clean = session_id.strip()
            self._bindings.pop(s_clean, None)
            self._mcp_servers.pop(s_clean, None)
            self._callables.pop(s_clean, None)
        else:
            self._bindings.clear()
            self._mcp_servers.clear()
            self._callables.clear()

    async def close(self) -> None:
        """Closes the underlying HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()


# Dynamic Tool Registry aliases and global singleton
DynamicToolRegistry = ToolRegistry
tool_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Returns the shared ToolRegistry singleton."""
    return tool_registry
