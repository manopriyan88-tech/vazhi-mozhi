"""
app/services/executor.py

The Universal Async HTTP Executor for Gemini Function Calls.
Intercepts Gemini tool/function calls, injects matching 3rd-party credentials
(Authorization: Bearer or x-api-key), fires live asynchronous HTTP requests
using httpx.AsyncClient, and returns compliant JSON responses back to the Gemini loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field
from google.genai import types

from app.core.config import settings

# Optional import of session key vault and dynamic tool registry from sister services
try:
    from app.services.key_vault import KeyRecord, KeyVault, key_vault as global_session_vault
    KEY_VAULT_AVAILABLE = True
except ImportError:
    KeyRecord = None  # type: ignore
    KeyVault = None  # type: ignore
    global_session_vault = None  # type: ignore
    KEY_VAULT_AVAILABLE = False

try:
    from app.services.tool_registry import tool_registry as dynamic_tool_registry
    DYNAMIC_TOOL_REGISTRY_AVAILABLE = True
except ImportError:
    dynamic_tool_registry = None  # type: ignore
    DYNAMIC_TOOL_REGISTRY_AVAILABLE = False

logger = logging.getLogger("app.services.executor")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)


# ============================================================================
# Enums and Models
# ============================================================================

class AuthHeaderType(str, Enum):
    """Supported authentication header delivery schemes for 3rd-party APIs."""
    BEARER = "bearer"        # Authorization: Bearer <key>
    X_API_KEY = "x-api-key"  # x-api-key: <key>
    CUSTOM = "custom"        # <custom_header_name>: <key>
    QUERY = "query"          # ?<query_param>=<key>
    NONE = "none"            # No authentication injected


class HttpMethod(str, Enum):
    """Standard HTTP Methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class ToolDefinition(BaseModel):
    """
    Metadata defining a registered API endpoint callable by Gemini.
    Can be dynamically populated by OpenAPI parsers, key vault schemas, or manual registration.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str
    description: Optional[str] = ""
    method: str = "GET"
    url: str
    base_url: Optional[str] = None
    path: Optional[str] = None
    path_params: List[str] = Field(default_factory=list)
    query_params: List[str] = Field(default_factory=list)
    header_params: List[str] = Field(default_factory=list)
    body_param: Optional[str] = None
    auth_type: AuthHeaderType = AuthHeaderType.BEARER
    auth_header_name: Optional[str] = None
    auth_query_param: Optional[str] = None
    service_name: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    default_headers: Dict[str, str] = Field(default_factory=dict)
    timeout: Optional[float] = None


class RequestPlan(BaseModel):
    """Concrete HTTP request specification prepared for dispatch."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: str
    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    json_data: Optional[Any] = None
    data: Optional[Any] = None
    timeout: float = 30.0


class ExecutionResult(BaseModel):
    """Structured result of executing an HTTP tool call."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    function_name: str
    call_id: Optional[str] = None
    success: bool
    status_code: int
    data: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    url: str = ""
    method: str = ""
    request_headers: Dict[str, str] = Field(default_factory=dict)
    response_headers: Dict[str, str] = Field(default_factory=dict)

    def to_gemini_response(self) -> Dict[str, Any]:
        """
        Converts the response to a dictionary compatible with Gemini FunctionResponse.
        Gemini strictly requires `response` to be a dictionary (dict[str, Any]).
        If the API returned a list or scalar, it is wrapped gracefully.
        """
        if isinstance(self.data, dict):
            res = dict(self.data)
            if "status_code" not in res:
                res["status_code"] = self.status_code
            if not self.success and "error" not in res:
                res["error"] = self.error or True
            return res
        elif isinstance(self.data, list):
            return {
                "status_code": self.status_code,
                "items": self.data,
                "data": self.data,
                "count": len(self.data),
                "success": self.success,
            }
        else:
            return {
                "status_code": self.status_code,
                "result": self.data,
                "success": self.success,
                "error": self.error,
            }

    def to_gemini_part(self) -> types.Part:
        """
        Converts the result directly into a google.genai types.Part object
        ready for insertion into the Gemini turn loop.
        """
        resp_dict = self.to_gemini_response()
        if self.call_id:
            return types.Part(
                function_response=types.FunctionResponse(
                    name=self.function_name,
                    response=resp_dict,
                    id=self.call_id,
                )
            )
        return types.Part.from_function_response(
            name=self.function_name,
            response=resp_dict,
        )


# Alias for backward and cross-module compatibility
ToolExecutionResult = ExecutionResult


# ============================================================================
# 3rd-Party API Key Vault & Header Injector
# ============================================================================

class APIKeyVault:
    """
    Manages 3rd-party API credentials, service-to-domain mappings, and automatic
    header injection (Authorization: Bearer <key> or x-api-key: <key>).
    """

    KNOWN_X_API_KEY_SERVICES: Set[str] = {
        "anthropic",
        "aws",
        "coinmarketcap",
        "polygon",
        "claude",
        "postman",
        "weatherapi",
        "cohere",
    }

    KNOWN_BEARER_SERVICES: Set[str] = {
        "github",
        "openai",
        "stripe",
        "slack",
        "google",
        "gemini",
        "huggingface",
        "linear",
        "spotify",
        "notion",
        "gitlab",
    }

    def __init__(self, default_auth_type: AuthHeaderType = AuthHeaderType.BEARER):
        self._keys: Dict[str, Dict[str, Any]] = {}
        self.default_auth_type = default_auth_type

    def register_key(
        self,
        service_or_domain: str,
        api_key: str,
        auth_type: Optional[AuthHeaderType] = None,
        header_name: Optional[str] = None,
        query_param: Optional[str] = None,
    ) -> None:
        """
        Register or override an API key for a service or domain name.
        """
        identifier = service_or_domain.strip().lower()
        self._keys[identifier] = {
            "key": api_key,
            "auth_type": auth_type,
            "header_name": header_name,
            "query_param": query_param,
        }
        logger.debug("Registered API key for service/domain '%s'", identifier)

    set_key = register_key

    def remove_key(self, service_or_domain: str) -> bool:
        """Remove a registered key."""
        identifier = service_or_domain.strip().lower()
        return self._keys.pop(identifier, None) is not None

    def get_registered_key(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Get registered key data if present."""
        return self._keys.get(identifier.strip().lower())

    def _discover_env_key(self, service: str) -> Optional[str]:
        """
        Searches os.environ for matching API keys using common naming conventions:
        e.g., GITHUB_API_KEY, GITHUB_TOKEN, API_KEY_GITHUB, GITHUB_SECRET.
        """
        clean_name = re.sub(r"[^A-Za-z0-9]", "_", service).upper()
        candidates = [
            f"{clean_name}_API_KEY",
            f"{clean_name}_TOKEN",
            f"{clean_name}_SECRET",
            f"{clean_name}_KEY",
            f"API_KEY_{clean_name}",
            f"{clean_name}_ACCESS_TOKEN",
        ]
        for candidate in candidates:
            val = os.getenv(candidate)
            if val:
                return val.strip()

        # Check in settings if configured
        if clean_name == "GEMINI" and getattr(settings, "GEMINI_API_KEY", None):
            return settings.GEMINI_API_KEY

        return None

    def resolve_auth(
        self,
        url: str,
        service_name: Optional[str] = None,
        tool_def: Optional[ToolDefinition] = None,
        explicit_key: Optional[str] = None,
        explicit_auth_type: Optional[Union[str, AuthHeaderType]] = None,
    ) -> Tuple[Optional[str], AuthHeaderType, str, Optional[str]]:
        """
        Resolves the API key, auth type, header name, and optional query parameter
        based on the target URL, service name, tool definition, and environment.

        Returns:
            (api_key, auth_type, header_name, query_param_name)
        """
        domain = ""
        host_parts: List[str] = []
        if url:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                host_parts = [p for p in domain.split(".") if p not in ("api", "www", "com", "org", "net", "io", "dev")]
            except Exception:
                pass

        # 1. Check explicit key passed in call
        key: Optional[str] = explicit_key

        # 2. Check ToolDefinition explicit key or env
        if not key and tool_def:
            if tool_def.api_key:
                key = tool_def.api_key
            elif tool_def.api_key_env and os.getenv(tool_def.api_key_env):
                key = os.getenv(tool_def.api_key_env)

        # 3. Check registered keys in vault
        reg_info: Optional[Dict[str, Any]] = None
        candidates_to_check: List[str] = []
        if service_name:
            candidates_to_check.append(service_name.lower())
        if domain:
            candidates_to_check.append(domain)
        candidates_to_check.extend([p.lower() for p in host_parts])

        for cand in candidates_to_check:
            reg = self.get_registered_key(cand)
            if reg:
                reg_info = reg
                if not key:
                    key = reg.get("key")
                break

        # 4. Check environment variables
        if not key:
            for cand in candidates_to_check:
                env_key = self._discover_env_key(cand)
                if env_key:
                    key = env_key
                    break

        # 5. Determine auth header type & header name
        auth_type: Optional[AuthHeaderType] = None
        header_name: Optional[str] = None
        query_param: Optional[str] = None

        # Check explicit auth type passed
        if explicit_auth_type:
            if isinstance(explicit_auth_type, str):
                at_lower = explicit_auth_type.lower()
                if at_lower in ("x-api-key", "api_key", "api_key_header", "apikey"):
                    auth_type = AuthHeaderType.X_API_KEY
                elif at_lower in ("bearer", "token", "authorization"):
                    auth_type = AuthHeaderType.BEARER
                elif at_lower in ("query", "api_key_query"):
                    auth_type = AuthHeaderType.QUERY
                    query_param = "api_key"
                else:
                    try:
                        auth_type = AuthHeaderType(at_lower)
                    except ValueError:
                        auth_type = AuthHeaderType.BEARER
            else:
                auth_type = explicit_auth_type

        # Check ToolDefinition auth type
        elif tool_def and (tool_def.auth_type or tool_def.auth_header_name or tool_def.auth_query_param):
            if tool_def.auth_type:
                auth_type = tool_def.auth_type
            if tool_def.auth_header_name:
                header_name = tool_def.auth_header_name
                if "x-api-key" in header_name.lower():
                    auth_type = AuthHeaderType.X_API_KEY
                elif "authorization" in header_name.lower():
                    auth_type = AuthHeaderType.BEARER
                else:
                    auth_type = AuthHeaderType.CUSTOM
            if tool_def.auth_query_param and (tool_def.auth_type == AuthHeaderType.QUERY or not tool_def.auth_type):
                query_param = tool_def.auth_query_param
                auth_type = AuthHeaderType.QUERY

        # Check registered key info
        if not auth_type and reg_info and reg_info.get("auth_type"):
            auth_type = reg_info["auth_type"]
            if reg_info.get("header_name"):
                header_name = reg_info["header_name"]
            if reg_info.get("query_param"):
                query_param = reg_info["query_param"]
                auth_type = AuthHeaderType.QUERY

        # Check known service heuristics if still not determined
        if not auth_type:
            for cand in candidates_to_check:
                if cand in self.KNOWN_X_API_KEY_SERVICES:
                    auth_type = AuthHeaderType.X_API_KEY
                    break
                if cand in self.KNOWN_BEARER_SERVICES:
                    auth_type = AuthHeaderType.BEARER
                    break

        if not auth_type:
            auth_type = self.default_auth_type

        # Finalize header name based on auth_type
        if auth_type == AuthHeaderType.X_API_KEY:
            header_name = header_name if header_name and "authorization" not in header_name.lower() else "x-api-key"
        elif auth_type == AuthHeaderType.BEARER:
            header_name = header_name or "Authorization"
        elif not header_name:
            header_name = "Authorization"

        return key, auth_type, header_name, query_param

    def inject(
        self,
        headers: Dict[str, str],
        params: Dict[str, Any],
        url: str,
        service_name: Optional[str] = None,
        tool_def: Optional[ToolDefinition] = None,
        explicit_key: Optional[str] = None,
        explicit_auth_type: Optional[Union[str, AuthHeaderType]] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Injects the matching 3rd-party API key into the headers (Authorization: Bearer or x-api-key)
        or query parameters as appropriate.
        """
        headers = dict(headers)
        params = dict(params)

        key, auth_type, header_name, query_param = self.resolve_auth(
            url=url,
            service_name=service_name,
            tool_def=tool_def,
            explicit_key=explicit_key,
            explicit_auth_type=explicit_auth_type,
        )

        if not key or auth_type == AuthHeaderType.NONE:
            return headers, params

        # 1. Check for placeholder substitution in existing headers (e.g. {API_KEY} or Bearer {API_KEY})
        substituted = False
        for h_k, h_v in list(headers.items()):
            if isinstance(h_v, str) and ("{api_key}" in h_v.lower() or "{token}" in h_v.lower()):
                headers[h_k] = re.sub(r"\{api_key\}|\{token\}", key, h_v, flags=re.IGNORECASE)
                substituted = True

        if substituted:
            return headers, params

        # 2. Inject based on auth_type into headers
        if auth_type == AuthHeaderType.X_API_KEY:
            existing_key_header = next((k for k in headers if k.lower() == "x-api-key"), None)
            target_header = existing_key_header or header_name or "x-api-key"
            headers[target_header] = key
        elif auth_type == AuthHeaderType.BEARER:
            existing_auth_header = next((k for k in headers if k.lower() == "authorization"), None)
            target_header = existing_auth_header or "Authorization"
            token_val = key if key.startswith("Bearer ") else f"Bearer {key}"
            headers[target_header] = token_val
        elif auth_type == AuthHeaderType.CUSTOM:
            headers[header_name] = key
        elif auth_type == AuthHeaderType.QUERY and query_param:
            params[query_param] = key

        return headers, params


# ============================================================================
# Dynamic Tool Registry
# ============================================================================

class ToolRegistry:
    """
    In-memory registry of dynamic tools (OpenAPI endpoints or custom tools).
    Allows tools to be registered once and intercepted by function name.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        self._tools[tool.name] = tool
        logger.debug("Registered tool '%s' -> %s %s", tool.name, tool.method, tool.url)

    def register_endpoint(
        self,
        name: str,
        url: str,
        method: str = "GET",
        description: str = "",
        path_params: Optional[List[str]] = None,
        query_params: Optional[List[str]] = None,
        header_params: Optional[List[str]] = None,
        body_param: Optional[str] = None,
        auth_type: AuthHeaderType = AuthHeaderType.BEARER,
        auth_header_name: Optional[str] = None,
        service_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ToolDefinition:
        """Convenience method to register an endpoint."""
        tool = ToolDefinition(
            name=name,
            url=url,
            method=method.upper(),
            description=description,
            path_params=path_params or [],
            query_params=query_params or [],
            header_params=header_params or [],
            body_param=body_param,
            auth_type=auth_type,
            auth_header_name=auth_header_name,
            service_name=service_name,
            api_key=api_key,
            api_key_env=api_key_env,
            default_headers=default_headers or {},
            timeout=timeout,
        )
        self.register(tool)
        return tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve a tool definition by name."""
        return self._tools.get(name)

    def remove(self, name: str) -> bool:
        """Remove a tool by name."""
        return self._tools.pop(name, None) is not None

    def list_tools(self) -> List[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()


# ============================================================================
# Universal Async HTTP Executor
# ============================================================================

class UniversalAsyncHTTPExecutor:
    """
    The Universal Async HTTP Executor.
    Intercepts Gemini function calls, parses path/query/body parameters,
    injects 3rd-party credentials (Authorization: Bearer or x-api-key),
    fires live HTTP requests via httpx.AsyncClient, and returns JSON responses
    back to the Gemini loop.
    """

    GENERIC_HTTP_TOOL_NAMES: Set[str] = {
        "http_request",
        "execute_http_request",
        "api_request",
        "call_api",
        "rest_api_call",
        "fetch_api",
        "request",
    }

    def __init__(
        self,
        client: Optional[httpx.AsyncClient] = None,
        key_vault: Optional[APIKeyVault] = None,
        registry: Optional[ToolRegistry] = None,
        timeout: Optional[float] = None,
        default_headers: Optional[Dict[str, str]] = None,
        session_vault: Optional[Any] = None,
        on_request_start: Optional[Callable[[str, RequestPlan], Any]] = None,
        on_request_complete: Optional[Callable[[ExecutionResult], Any]] = None,
    ):
        self._client = client
        self.key_vault = key_vault or APIKeyVault()
        self.registry = registry or ToolRegistry()
        self.timeout = timeout or float(settings.TOOL_EXECUTION_TIMEOUT_SECONDS)
        self.default_headers = default_headers or {
            "User-Agent": "Adaptive-API-Kitchen/0.1.0",
            "Accept": "application/json, text/plain, */*",
        }
        self.session_vault = session_vault if session_vault is not None else global_session_vault
        self.on_request_start = on_request_start
        self.on_request_complete = on_request_complete

    async def get_client(self) -> httpx.AsyncClient:
        """Returns or lazily creates the internal httpx.AsyncClient."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
                headers=self.default_headers,
            )
        return self._client

    async def close(self) -> None:
        """Closes the underlying httpx.AsyncClient connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> UniversalAsyncHTTPExecutor:
        await self.get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # ------------------------------------------------------------------------
    # Function Call Extraction
    # ------------------------------------------------------------------------

    @staticmethod
    def extract_function_call(call: Any) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """
        Normalizes function call input from any supported format:
        - google.genai types.Part
        - google.genai types.FunctionCall
        - dict with 'name' and 'args' (or 'arguments')
        - dict with nested 'function_call'
        - object with .name and .args attributes

        Returns:
            (function_name, arguments_dict, call_id)
        """
        call_id = None

        # 1. Part object
        if hasattr(call, "function_call") and call.function_call is not None:
            fc = call.function_call
            name = getattr(fc, "name", "")
            raw_args = getattr(fc, "args", {}) or {}
            call_id = getattr(fc, "id", None)
            args = dict(raw_args) if isinstance(raw_args, dict) else {}
            return name, args, call_id

        # 2. FunctionCall object or duck-typed object
        if hasattr(call, "name") and hasattr(call, "args"):
            name = getattr(call, "name", "")
            raw_args = getattr(call, "args", {}) or {}
            call_id = getattr(call, "id", None)
            args = dict(raw_args) if isinstance(raw_args, dict) else {}
            return name, args, call_id

        # 3. Dictionary representation
        if isinstance(call, dict):
            if "function_call" in call and isinstance(call["function_call"], dict):
                inner = call["function_call"]
                name = inner.get("name", "")
                raw_args = inner.get("args") if inner.get("args") is not None else inner.get("arguments", {})
                call_id = inner.get("id")
            else:
                name = call.get("name", "")
                raw_args = call.get("args") if call.get("args") is not None else call.get("arguments", {})
                call_id = call.get("id")

            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {"raw_arguments": raw_args}
            elif isinstance(raw_args, dict):
                args = dict(raw_args)
            else:
                args = {}

            return name, args, call_id

        raise ValueError(f"Cannot extract function call from unsupported type: {type(call)}")

    # ------------------------------------------------------------------------
    # URL & Argument Resolution
    # ------------------------------------------------------------------------

    @staticmethod
    def _interpolate_url(url_template: str, arguments: Dict[str, Any]) -> Tuple[str, Set[str]]:
        """
        Safely substitutes path placeholders like {pet_id} or {owner} from arguments.
        Returns the interpolated URL and the set of consumed argument keys.
        """
        consumed: Set[str] = set()

        def replacer(match: re.Match) -> str:
            param = match.group(1)
            if param in arguments:
                consumed.add(param)
                return str(arguments[param])
            # Check case-insensitive match
            for k, v in arguments.items():
                if k.lower() == param.lower():
                    consumed.add(k)
                    return str(v)
            return match.group(0)

        interpolated = re.sub(r"\{([a-zA-Z0-9_\-]+)\}", replacer, url_template)
        return interpolated, consumed

    async def _resolve_session_vault_tool(
        self,
        function_name: str,
        session_id: str,
    ) -> Optional[ToolDefinition]:
        """
        Checks the KeyVault session storage for dynamically registered keys whose
        function declaration or provider matches function_name.
        """
        if not self.session_vault:
            return None

        try:
            records = await self.session_vault.list_keys(session_id)
            for r in records:
                fd_name = (r.function_declaration or {}).get("name", "")
                if fd_name and fd_name.lower() == function_name.lower():
                    # Map auth type
                    auth_type = AuthHeaderType.BEARER
                    query_param = None
                    if r.auth_type == "api_key_header" or (r.auth_header_name and "x-api-key" in r.auth_header_name.lower()):
                        auth_type = AuthHeaderType.X_API_KEY
                    elif r.auth_type == "api_key_query":
                        auth_type = AuthHeaderType.QUERY
                        query_param = r.auth_param_name

                    return ToolDefinition(
                        name=function_name,
                        url=r.endpoint_url,
                        method="GET",
                        description=r.function_declaration.get("description", ""),
                        auth_type=auth_type,
                        auth_header_name=r.auth_header_name,
                        auth_query_param=query_param,
                        service_name=r.provider,
                        api_key=r.api_key,
                    )
        except Exception as exc:
            logger.debug("Session vault lookup for %s: %s", function_name, exc)

        return None

    async def plan_request(
        self,
        function_name: str,
        arguments: Dict[str, Any],
        session_id: str = "default",
    ) -> RequestPlan:
        """
        Translates a function call and arguments into a concrete HTTP RequestPlan.
        Resolves registered endpoints, dynamic/universal HTTP arguments,
        URL parameter interpolation, and 3rd-party auth injection.
        """
        args = dict(arguments)
        tool_def = self.registry.get(function_name)

        # 1. Check dynamic tool registry bindings if available
        if not tool_def and dynamic_tool_registry:
            binding = dynamic_tool_registry.get_tool_binding(function_name, session_id=session_id)
            if binding:
                path_params = [k for k, loc in binding.param_locations.items() if loc == "path"]
                query_params = [k for k, loc in binding.param_locations.items() if loc == "query"]
                header_params = [k for k, loc in binding.param_locations.items() if loc == "header"]
                auth_type = AuthHeaderType.BEARER
                if binding.auth_type == "api_key_header" or (binding.auth_header_name and "x-api-key" in binding.auth_header_name.lower()):
                    auth_type = AuthHeaderType.X_API_KEY
                elif binding.auth_type == "api_key_query":
                    auth_type = AuthHeaderType.QUERY

                api_key = None
                if self.session_vault and binding.provider:
                    rec = await self.session_vault.get_key(session_id, binding.provider)
                    if rec:
                        api_key = rec.api_key

                tool_def = ToolDefinition(
                    name=function_name,
                    url=binding.endpoint_url,
                    method=binding.http_method,
                    description=binding.description,
                    path_params=path_params,
                    query_params=query_params,
                    header_params=header_params,
                    auth_type=auth_type,
                    auth_header_name=binding.auth_header_name,
                    auth_query_param=binding.auth_param_name if auth_type == AuthHeaderType.QUERY else None,
                    service_name=binding.provider,
                    api_key=api_key,
                    default_headers=dict(binding.custom_headers),
                )

        # 2. Check session vault if not found in local registry
        if not tool_def:
            tool_def = await self._resolve_session_vault_tool(function_name, session_id)

        # 3. Check if generic HTTP tool or un-registered call with explicit URL
        is_generic = (function_name in self.GENERIC_HTTP_TOOL_NAMES) or ("url" in args and not tool_def)

        if tool_def:
            method = tool_def.method.upper()
            raw_url = tool_def.url
            if tool_def.base_url and not (raw_url.startswith("http://") or raw_url.startswith("https://")):
                raw_url = f"{tool_def.base_url.rstrip('/')}/{raw_url.lstrip('/')}"

            # Interpolate path params
            url, consumed_keys = self._interpolate_url(raw_url, args)
            for k in tool_def.path_params:
                consumed_keys.add(k)

            # Build headers
            headers = dict(self.default_headers)
            headers.update(tool_def.default_headers)

            # Extract header parameters from args
            for h_param in tool_def.header_params:
                if h_param in args:
                    headers[h_param] = str(args[h_param])
                    consumed_keys.add(h_param)

            # Build query params
            params: Dict[str, Any] = {}
            for q_param in tool_def.query_params:
                if q_param in args:
                    params[q_param] = args[q_param]
                    consumed_keys.add(q_param)

            # Extract remaining arguments
            remaining_args = {k: v for k, v in args.items() if k not in consumed_keys}

            json_data: Optional[Any] = None
            data_body: Optional[Any] = None

            if method in {"POST", "PUT", "PATCH"}:
                if tool_def.body_param and tool_def.body_param in remaining_args:
                    json_data = remaining_args.pop(tool_def.body_param)
                elif "body" in remaining_args:
                    json_data = remaining_args.pop("body")
                elif "json" in remaining_args:
                    json_data = remaining_args.pop("json")
                elif "data" in remaining_args:
                    data_body = remaining_args.pop("data")
                elif remaining_args:
                    json_data = remaining_args
            else:
                # For GET/DELETE, remaining args default to query params
                params.update(remaining_args)

            # Detect service name
            service_name = tool_def.service_name
            if not service_name and "_" in function_name:
                service_name = function_name.split("_")[0]

            # Inject auth
            headers, params = self.key_vault.inject(
                headers=headers,
                params=params,
                url=url,
                service_name=service_name,
                tool_def=tool_def,
                explicit_key=args.get("api_key") or args.get("token"),
                explicit_auth_type=args.get("auth_type"),
            )

            req_timeout = tool_def.timeout or self.timeout
            return RequestPlan(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json_data=json_data,
                data=data_body,
                timeout=req_timeout,
            )

        elif is_generic:
            # Universal / Ad-hoc HTTP call
            raw_url = str(args.get("url", ""))
            if not raw_url:
                raise ValueError(f"Function call '{function_name}' has no URL specified.")

            method = str(args.get("method", "GET")).upper()
            url, _ = self._interpolate_url(raw_url, args)

            headers = dict(self.default_headers)
            if "headers" in args and isinstance(args["headers"], dict):
                headers.update(args["headers"])

            params = {}
            if "params" in args and isinstance(args["params"], dict):
                params.update(args["params"])
            elif "query" in args and isinstance(args["query"], dict):
                params.update(args["query"])

            json_data = args.get("json") or args.get("body") or args.get("payload")
            data_body = args.get("data") if json_data is None else None

            # Determine service name
            service_name = args.get("service") or args.get("service_name")
            if not service_name and "_" in function_name:
                prefix = function_name.split("_")[0]
                if prefix not in self.GENERIC_HTTP_TOOL_NAMES:
                    service_name = prefix

            headers, params = self.key_vault.inject(
                headers=headers,
                params=params,
                url=url,
                service_name=service_name,
                explicit_key=args.get("api_key") or args.get("token"),
                explicit_auth_type=args.get("auth_type"),
            )

            req_timeout = float(args.get("timeout", self.timeout))
            return RequestPlan(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json_data=json_data,
                data=data_body,
                timeout=req_timeout,
            )

        else:
            raise KeyError(
                f"Function '{function_name}' is not registered in ToolRegistry and contains no URL to execute."
            )

    # ------------------------------------------------------------------------
    # Request Execution & Response Handling
    # ------------------------------------------------------------------------

    @staticmethod
    def _mask_headers(headers: Dict[str, str]) -> Dict[str, str]:
        """Masks sensitive credentials before logging or returning metadata."""
        masked = dict(headers)
        sensitive = {"authorization", "x-api-key", "api-key", "apikey", "token", "secret"}
        for k, v in masked.items():
            if k.lower() in sensitive and isinstance(v, str):
                if v.lower().startswith("bearer "):
                    token = v[7:]
                    masked[k] = f"Bearer {token[:3]}...{token[-4:]}" if len(token) > 8 else "Bearer ***"
                elif len(v) > 8:
                    masked[k] = f"{v[:3]}...{v[-4:]}"
                else:
                    masked[k] = "***"
        return masked

    async def execute_call(
        self,
        call: Any,
        session_id: str = "default",
    ) -> ExecutionResult:
        """
        Executes a Gemini tool call end-to-end:
        1. Intercepts function name, arguments, and call_id.
        2. Plans request with parameter interpolation and auth header injection.
        3. Fires live asynchronous request via httpx.AsyncClient.
        4. Parses JSON response and returns structured ExecutionResult.
        """
        function_name = ""
        call_id = None
        arguments: Dict[str, Any] = {}

        try:
            function_name, arguments, call_id = self.extract_function_call(call)
        except Exception as e:
            return ExecutionResult(
                function_name=getattr(call, "name", "unknown"),
                call_id=getattr(call, "id", None),
                success=False,
                status_code=400,
                error=f"Invalid function call format: {e}",
                data={"error": "InvalidFunctionCall", "message": str(e)},
            )

        # Plan the request
        try:
            plan = await self.plan_request(function_name, arguments, session_id=session_id)
        except KeyError as e:
            logger.warning("Unregistered tool call: %s", e)
            return ExecutionResult(
                function_name=function_name,
                call_id=call_id,
                success=False,
                status_code=404,
                error=str(e),
                data={"error": "ToolNotFound", "message": str(e)},
            )
        except Exception as e:
            logger.error("Failed to plan request for %s: %s", function_name, e)
            return ExecutionResult(
                function_name=function_name,
                call_id=call_id,
                success=False,
                status_code=400,
                error=f"Failed to prepare HTTP request: {e}",
                data={"error": "BadRequestPlan", "message": str(e)},
            )

        # Fire live request
        client = await self.get_client()
        start_time = time.perf_counter()

        if self.on_request_start:
            try:
                res = self.on_request_start(function_name, plan)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.debug("on_request_start hook exception: %s", e)

        try:
            response = await client.request(
                method=plan.method,
                url=plan.url,
                params=plan.params or None,
                json=plan.json_data,
                data=plan.data,
                headers=plan.headers,
                timeout=plan.timeout,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Attempt JSON parsing
            try:
                parsed_data = response.json()
            except Exception:
                parsed_data = {
                    "text": response.text[:50000],
                    "content_type": response.headers.get("content-type", "text/plain"),
                }

            is_success = 200 <= response.status_code < 300
            error_message = None
            if not is_success:
                error_message = f"HTTP {response.status_code} {response.reason_phrase}"
                if isinstance(parsed_data, dict):
                    detail = parsed_data.get("message") or parsed_data.get("error") or parsed_data.get("detail")
                    if detail:
                        error_message = f"{error_message}: {detail}"

            result = ExecutionResult(
                function_name=function_name,
                call_id=call_id,
                success=is_success,
                status_code=response.status_code,
                data=parsed_data,
                error=error_message,
                execution_time_ms=elapsed_ms,
                url=str(response.url),
                method=plan.method,
                request_headers=self._mask_headers(plan.headers),
                response_headers=dict(response.headers),
            )

        except httpx.TimeoutException as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result = ExecutionResult(
                function_name=function_name,
                call_id=call_id,
                success=False,
                status_code=504,
                error=f"Request to {plan.url} timed out after {plan.timeout}s: {e}",
                execution_time_ms=elapsed_ms,
                url=plan.url,
                method=plan.method,
                request_headers=self._mask_headers(plan.headers),
                data={"error": "GatewayTimeout", "message": f"HTTP request timed out after {plan.timeout} seconds."},
            )

        except httpx.RequestError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result = ExecutionResult(
                function_name=function_name,
                call_id=call_id,
                success=False,
                status_code=502,
                error=f"Network error requesting {plan.url}: {e}",
                execution_time_ms=elapsed_ms,
                url=plan.url,
                method=plan.method,
                request_headers=self._mask_headers(plan.headers),
                data={"error": "BadGateway", "message": f"Network error during request: {e}"},
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result = ExecutionResult(
                function_name=function_name,
                call_id=call_id,
                success=False,
                status_code=500,
                error=f"Unexpected error executing HTTP request: {e}",
                execution_time_ms=elapsed_ms,
                url=plan.url,
                method=plan.method,
                request_headers=self._mask_headers(plan.headers),
                data={"error": "InternalExecutionError", "message": str(e)},
            )

        if self.on_request_complete:
            try:
                res = self.on_request_complete(result)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.debug("on_request_complete hook exception: %s", e)

        return result

    async def execute(
        self,
        call_or_name: Any,
        arguments: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Executes a function call and returns the real JSON response dictionary
        ready for the Gemini loop.
        """
        if isinstance(call_or_name, str):
            call_obj = {"name": call_or_name, "args": arguments or {}, "id": call_id}
        else:
            call_obj = call_or_name

        result = await self.execute_call(call_obj, session_id=session_id)
        return result.to_gemini_response()

    async def execute_for_gemini(
        self,
        call: Any,
        session_id: str = "default",
    ) -> types.Part:
        """
        Executes a function call and returns a google.genai types.Part object
        ready for direct appending to the Gemini conversation contents.
        """
        result = await self.execute_call(call, session_id=session_id)
        return result.to_gemini_part()

    async def execute_batch(
        self,
        calls: List[Any],
        session_id: str = "default",
    ) -> List[ExecutionResult]:
        """
        Concurrently executes multiple tool calls.
        """
        tasks = [self.execute_call(call, session_id=session_id) for call in calls]
        return await asyncio.gather(*tasks)

    async def execute_batch_for_gemini(
        self,
        calls: List[Any],
        session_id: str = "default",
    ) -> List[types.Part]:
        """
        Concurrently executes multiple tool calls and returns a list of google.genai types.Part.
        """
        results = await self.execute_batch(calls, session_id=session_id)
        return [res.to_gemini_part() for res in results]


# ============================================================================
# Global Singletons and Module-Level Helpers
# ============================================================================

default_key_vault = APIKeyVault()
default_registry = ToolRegistry()
default_executor = UniversalAsyncHTTPExecutor(
    key_vault=default_key_vault,
    registry=default_registry,
)


def register_tool(
    name: str,
    url: str,
    method: str = "GET",
    description: str = "",
    path_params: Optional[List[str]] = None,
    query_params: Optional[List[str]] = None,
    header_params: Optional[List[str]] = None,
    body_param: Optional[str] = None,
    auth_type: AuthHeaderType = AuthHeaderType.BEARER,
    auth_header_name: Optional[str] = None,
    service_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> ToolDefinition:
    """Register a tool in the global default registry."""
    return default_registry.register_endpoint(
        name=name,
        url=url,
        method=method,
        description=description,
        path_params=path_params,
        query_params=query_params,
        header_params=header_params,
        body_param=body_param,
        auth_type=auth_type,
        auth_header_name=auth_header_name,
        service_name=service_name,
        api_key=api_key,
        api_key_env=api_key_env,
        default_headers=default_headers,
        timeout=timeout,
    )


def register_api_key(
    service_or_domain: str,
    api_key: str,
    auth_type: Optional[AuthHeaderType] = None,
    header_name: Optional[str] = None,
    query_param: Optional[str] = None,
) -> None:
    """Register a 3rd-party API key in the global default key vault."""
    default_key_vault.register_key(
        service_or_domain=service_or_domain,
        api_key=api_key,
        auth_type=auth_type,
        header_name=header_name,
        query_param=query_param,
    )


async def execute_function_call(
    call_or_name: Any,
    arguments: Optional[Dict[str, Any]] = None,
    call_id: Optional[str] = None,
    session_id: str = "default",
) -> Dict[str, Any]:
    """Execute a function call using the default universal executor."""
    return await default_executor.execute(call_or_name, arguments, call_id, session_id=session_id)


async def execute_for_gemini(
    call: Any,
    session_id: str = "default",
) -> types.Part:
    """Execute a function call and return a Gemini Part using the default executor."""
    return await default_executor.execute_for_gemini(call, session_id=session_id)
