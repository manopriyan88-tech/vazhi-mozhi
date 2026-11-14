"""Dynamic 'API Kitchen' Key Vault & Schema Inspector.

Securely stores 3rd-party API credentials in an in-memory session vault,
verifies API key status via lightweight async httpx calls, and auto-generates
clean Gemini Function Declaration schemas (gemini-2.5-flash) for dynamic tool use.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import httpx
from pydantic import BaseModel, Field

from app.core.config import settings

class AuthType(str, Enum):
    """Authentication delivery mechanisms."""
    BEARER = "bearer"
    API_KEY_HEADER = "api_key_header"
    API_KEY_QUERY = "api_key_query"
    CUSTOM = "custom"
from app.services.http_executor import APIRequest

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    genai_types = None  # type: ignore
    GENAI_AVAILABLE = False

logger = logging.getLogger("app.services.key_vault")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)


def mask_api_key(api_key: str) -> str:
    """Masks an API key to safely log or expose in responses without leaking secrets.

    Examples:
        'sk-1234567890abcdef' -> 'sk-1...cdef'
        'secret_token'        -> 'secr...oken'
        'short'               -> 'sh...rt'
    """
    if not api_key:
        return ""
    key_clean = api_key.strip()
    length = len(key_clean)
    if length <= 4:
        return "****"
    if length <= 8:
        return f"{key_clean[:2]}...{key_clean[-2:]}"
    return f"{key_clean[:4]}...{key_clean[-4:]}"


def sanitize_identifier(name: str) -> str:
    """Sanitizes a string into a valid Gemini function declaration name (snake_case)."""
    # Replace non-alphanumeric chars with underscore
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean:
        clean = "api_tool"
    if clean[0].isdigit():
        clean = f"tool_{clean}"
    return clean[:64]


class VerificationResult(BaseModel):
    """Result of a lightweight async test call to verify API key validity."""

    is_active: bool = Field(description="Whether the API key is considered valid and active")
    status: str = Field(
        description="Status category: 'verified', 'active_with_warning', 'unauthorized', 'forbidden', 'unreachable', 'error'"
    )
    status_code: int = Field(default=0, description="HTTP status code received from test probe")
    latency_ms: float = Field(default=0.0, description="Probe latency in milliseconds")
    auth_method: str = Field(default="unknown", description="Authentication delivery method used during probe")
    message: str = Field(description="Human-readable explanation of verification outcome")
    sample_data: Optional[Any] = Field(default=None, description="Sample payload or headers returned by probe")


class KeyRecord(BaseModel):
    """In-memory representation of a registered 3rd-party API key."""

    session_id: str = Field(description="Session or user namespace holding this key")
    provider: str = Field(description="Provider name (e.g., WeatherAPI, GitHub, Stripe)")
    api_key: str = Field(description="Raw API key string (stored strictly in-memory)")
    masked_key: str = Field(description="Safely masked API key representation")
    endpoint_url: str = Field(description="Sample endpoint or documentation URL")
    http_method: str = Field(default="GET", description="Default HTTP method for endpoint calls")
    auth_type: str = Field(
        default="bearer",
        description="Auth delivery mechanism: 'bearer', 'api_key_header', 'api_key_query', or 'custom'",
    )
    auth_header_name: str = Field(default="Authorization", description="Header name when auth_type is header/bearer")
    auth_param_name: str = Field(default="api_key", description="Query param name when auth_type is query")
    custom_headers: Dict[str, str] = Field(default_factory=dict, description="Custom headers for service")
    openapi_spec: Optional[Dict[str, Any]] = Field(default=None, description="Optional full OpenAPI specification")
    is_active: bool = Field(default=False, description="Whether key passed verification probe")
    verification: VerificationResult = Field(description="Verification probe results")
    function_declaration: Dict[str, Any] = Field(
        description="Gemini Function Declaration JSON schema for this tool"
    )
    generator_source: str = Field(
        default="gemini-2.5-flash",
        description="Source of the schema ('gemini-2.5-flash' or 'heuristic_fallback')",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_safe_dict(self) -> Dict[str, Any]:
        """Returns safe dictionary representation with secrets masked."""
        data = self.model_dump(exclude={"api_key"})
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    def create_api_request(
        self,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Any] = None,
        path_suffix: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> APIRequest:
        """Constructs an authenticated APIRequest using this record's stored credentials."""
        target_url = self.endpoint_url
        if path_suffix:
            target_url = target_url.rstrip("/") + "/" + path_suffix.lstrip("/")

        return APIRequest(
            method=method.upper(),
            url=target_url,
            params=dict(params or {}),
            json_body=json_body,
            auth_type=self.auth_type,
            auth_token=self.api_key,
            auth_header_name=self.auth_header_name,
            auth_param_name=self.auth_param_name,
            timeout_seconds=timeout_seconds,
        )

    def to_gemini_tool(self) -> Any:
        """Converts stored function declaration into an official google.genai Tool object."""
        if not GENAI_AVAILABLE or not genai_types:
            return self.function_declaration
        try:
            fd = genai_types.FunctionDeclaration(**self.function_declaration)
            return genai_types.Tool(function_declarations=[fd])
        except Exception as exc:
            logger.warning(f"Could not convert function_declaration to genai Tool: {exc}")
            return self.function_declaration


class KeyVault:
    """In-memory session vault for 3rd-party API keys and dynamic Gemini tool schemas."""

    def __init__(self, gemini_client: Optional[Any] = None):
        # Vault storage: {session_id: {provider.lower(): KeyRecord}}
        self._vault: Dict[str, Dict[str, KeyRecord]] = {}
        self._lock = asyncio.Lock()
        self._gemini_client = gemini_client

    def _get_gemini_client(self, override_key: Optional[str] = None) -> Optional[Any]:
        """Resolves or instantiates the google-genai Client."""
        if self._gemini_client is not None:
            return self._gemini_client

        if not GENAI_AVAILABLE:
            logger.info("google-genai SDK not available; using heuristic schema generation.")
            return None

        api_key = override_key or settings.GEMINI_API_KEY
        if not api_key or api_key == "your_gemini_api_key_here":
            logger.info("No valid GEMINI_API_KEY provided; will use heuristic schema generation.")
            return None

        try:
            return genai.Client(api_key=api_key)
        except Exception as exc:
            logger.warning(f"Failed to initialize google-genai Client: {exc}")
            return None

    def _infer_auth_strategy(
        self,
        provider: str,
        endpoint_url: str,
        auth_type: Optional[str] = None,
        auth_header_name: Optional[str] = None,
        auth_param_name: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Infers the authentication type, header name, and query parameter name."""
        provider_lower = provider.strip().lower()
        url_lower = endpoint_url.lower()

        # If user explicitly provided auth_type, respect it
        if auth_type:
            at = auth_type.lower()
            if at in ("bearer", "token"):
                return ("bearer", auth_header_name or "Authorization", auth_param_name or "api_key")
            if at in ("api_key_header", "header", "apikey"):
                return ("api_key_header", auth_header_name or "x-api-key", auth_param_name or "api_key")
            if at in ("api_key_query", "query", "param"):
                return ("api_key_query", auth_header_name or "Authorization", auth_param_name or "key")

        # Provider-specific conventions
        if "weatherapi" in provider_lower or "weatherapi" in url_lower:
            return ("api_key_query", "Authorization", "key")
        if "openweathermap" in provider_lower or "openweathermap" in url_lower:
            return ("api_key_query", "Authorization", "appid")
        if "coingecko" in provider_lower:
            return ("api_key_header", "x-cg-demo-api-key", "x_cg_demo_api_key")
        if "anthropic" in provider_lower:
            return ("api_key_header", "x-api-key", "api_key")
        if "github" in provider_lower or "stripe" in provider_lower:
            return ("bearer", "Authorization", "api_key")

        # URL-based heuristics
        parsed = urlparse(endpoint_url)
        query_keys = [k.lower() for k in parse_qs(parsed.query).keys()]
        if "key" in query_keys:
            return ("api_key_query", "Authorization", "key")
        if "appid" in query_keys:
            return ("api_key_query", "Authorization", "appid")
        if "api_key" in query_keys or "apikey" in query_keys:
            return ("api_key_query", "Authorization", "api_key")

        # Default fallback: Bearer token
        return ("bearer", auth_header_name or "Authorization", auth_param_name or "api_key")

    async def verify_key_active(
        self,
        api_key: str,
        endpoint_url: str,
        provider: str,
        auth_type: Optional[str] = None,
        auth_header_name: Optional[str] = None,
        auth_param_name: Optional[str] = None,
        timeout_seconds: float = 8.0,
    ) -> VerificationResult:
        """Makes a lightweight async test call using httpx to probe key validity."""
        if not endpoint_url or not endpoint_url.startswith(("http://", "https://")):
            return VerificationResult(
                is_active=False,
                status="error",
                status_code=0,
                latency_ms=0.0,
                auth_method="none",
                message=f"Invalid endpoint URL: '{endpoint_url}'. Must start with http:// or https://",
            )

        strategy_auth_type, header_name, param_name = self._infer_auth_strategy(
            provider, endpoint_url, auth_type, auth_header_name, auth_param_name
        )

        headers: Dict[str, str] = {
            "User-Agent": "AdaptiveKitchen-KeyInspector/1.0",
            "Accept": "application/json, text/plain, */*",
        }
        params: Dict[str, str] = {}

        if strategy_auth_type == "bearer":
            headers[header_name] = f"Bearer {api_key}"
        elif strategy_auth_type == "api_key_header":
            headers[header_name] = api_key
        elif strategy_auth_type == "api_key_query":
            params[param_name] = api_key

        start_time = asyncio.get_event_loop().time()

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds),
                follow_redirects=True,
                verify=True,
            ) as client:
                # 1. Attempt lightweight HEAD request first
                try:
                    resp = await client.head(endpoint_url, headers=headers, params=params)
                    # Some APIs disallow HEAD (405 Method Not Allowed)
                    if resp.status_code == 405:
                        resp = await client.get(endpoint_url, headers=headers, params=params)
                except httpx.HTTPError:
                    # Fallback to GET on any initial HEAD glitch
                    resp = await client.get(endpoint_url, headers=headers, params=params)

                latency_ms = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)
                code = resp.status_code

                # Parse sample response data safely
                sample_data = None
                try:
                    sample_data = resp.json()
                    if isinstance(sample_data, dict) and len(sample_data) > 10:
                        # Keep only sample snippet
                        sample_data = {k: sample_data[k] for k in list(sample_data.keys())[:5]}
                except Exception:
                    sample_data = resp.text[:200] if resp.text else None

                # Interpret authentication response:
                # 2xx: Active & fully verified
                if 200 <= code < 300:
                    return VerificationResult(
                        is_active=True,
                        status="verified",
                        status_code=code,
                        latency_ms=latency_ms,
                        auth_method=strategy_auth_type,
                        message=f"Key verified successfully. Endpoint responded with HTTP {code}.",
                        sample_data=sample_data,
                    )

                # 401: Unauthorized - invalid key
                if code == 401:
                    return VerificationResult(
                        is_active=False,
                        status="unauthorized",
                        status_code=code,
                        latency_ms=latency_ms,
                        auth_method=strategy_auth_type,
                        message="API key rejected: HTTP 401 Unauthorized. Key is invalid or expired.",
                        sample_data=sample_data,
                    )

                # 403: Forbidden - permissions or blocked key
                if code == 403:
                    return VerificationResult(
                        is_active=False,
                        status="forbidden",
                        status_code=code,
                        latency_ms=latency_ms,
                        auth_method=strategy_auth_type,
                        message="API key rejected: HTTP 403 Forbidden. Key lacks required permissions or tier access.",
                        sample_data=sample_data,
                    )

                # 400 or 404 or 422: Authentication succeeded, but endpoint lacks required parameters or sub-resource
                if code in (400, 404, 422):
                    return VerificationResult(
                        is_active=True,
                        status="active_with_warning",
                        status_code=code,
                        latency_ms=latency_ms,
                        auth_method=strategy_auth_type,
                        message=(
                            f"Key accepted without 401/403 auth error (HTTP {code}). "
                            "Endpoint reached, though query/body parameters or specific resource ID were missing."
                        ),
                        sample_data=sample_data,
                    )

                # 429: Too Many Requests - Key is valid but currently rate limited
                if code == 429:
                    return VerificationResult(
                        is_active=True,
                        status="active_with_warning",
                        status_code=code,
                        latency_ms=latency_ms,
                        auth_method=strategy_auth_type,
                        message="API key is active, but the provider is currently returning HTTP 429 Rate Limit.",
                        sample_data=sample_data,
                    )

                # 5xx or others: Server error
                return VerificationResult(
                    is_active=False,
                    status="error",
                    status_code=code,
                    latency_ms=latency_ms,
                    auth_method=strategy_auth_type,
                    message=f"Endpoint returned HTTP {code}. Key state unverified due to remote server error.",
                    sample_data=sample_data,
                )

        except httpx.TimeoutException:
            latency_ms = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)
            return VerificationResult(
                is_active=False,
                status="unreachable",
                status_code=0,
                latency_ms=latency_ms,
                auth_method=strategy_auth_type,
                message=f"Request to endpoint '{endpoint_url}' timed out after {timeout_seconds}s.",
            )
        except Exception as exc:
            latency_ms = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)
            return VerificationResult(
                is_active=False,
                status="unreachable",
                status_code=0,
                latency_ms=latency_ms,
                auth_method=strategy_auth_type,
                message=f"Network error during probe: {type(exc).__name__}: {str(exc)}",
            )

    def _heuristic_generate_schema(
        self,
        provider: str,
        endpoint_url: str,
        sample_data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generates a clean Gemini Function Declaration schema using heuristic URL analysis.

        Used when Gemini API key is absent, offline, or as a rapid fallback.
        Conforms strictly to Gemini's FunctionDeclaration format.
        """
        parsed = urlparse(endpoint_url)
        path = parsed.path.strip("/")
        path_segments = [seg for seg in path.split("/") if seg and not seg.startswith("v1") and not seg.startswith("v2")]

        base_name = f"{provider}_{'_'.join(path_segments) if path_segments else 'call'}"
        func_name = sanitize_identifier(base_name)

        properties: Dict[str, Any] = {}
        required: List[str] = []

        # Analyze existing query parameters in the URL
        if parsed.query:
            query_dict = parse_qs(parsed.query)
            for key, val in query_dict.items():
                if key.lower() in ("key", "appid", "api_key", "token"):
                    continue
                properties[key] = {
                    "type": "STRING",
                    "description": f"Query parameter '{key}' for {provider} API (e.g. {val[0] if val else 'value'}).",
                }
                required.append(key)

        # Provider-specific default schemas
        provider_lower = provider.lower()
        if "weather" in provider_lower:
            if "q" not in properties and "city" not in properties:
                properties["location"] = {
                    "type": "STRING",
                    "description": "City name, zip code, or latitude/longitude coordinates (e.g., 'London', 'New York').",
                }
                required.append("location")
            if "units" not in properties:
                properties["units"] = {
                    "type": "STRING",
                    "description": "Units for temperature measurements.",
                    "enum": ["metric", "imperial", "standard"],
                }
        elif "crypto" in provider_lower or "coingecko" in provider_lower:
            if "ids" not in properties and "coin" not in properties:
                properties["coin_id"] = {
                    "type": "STRING",
                    "description": "Cryptocurrency identifier (e.g., 'bitcoin', 'ethereum').",
                }
                required.append("coin_id")
            if "vs_currency" not in properties:
                properties["vs_currency"] = {
                    "type": "STRING",
                    "description": "Target currency for price quote (e.g., 'usd', 'eur').",
                }
        elif "github" in provider_lower:
            if "username" not in properties:
                properties["username"] = {
                    "type": "STRING",
                    "description": "GitHub user or organization login name.",
                }
                required.append("username")
        else:
            if not properties:
                properties["query"] = {
                    "type": "STRING",
                    "description": f"Search query or input parameter for {provider} API.",
                }
                required.append("query")

        return {
            "name": func_name,
            "description": f"Executes an API request to {provider} ({endpoint_url}) to fetch or query data.",
            "parameters": {
                "type": "OBJECT",
                "properties": properties,
                "required": required,
            },
        }

    async def generate_function_declaration(
        self,
        provider: str,
        endpoint_url: str,
        sample_data: Optional[Any] = None,
        gemini_api_key: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Uses Gemini (gemini-2.5-flash) to auto-generate a clean JSON Function Declaration schema.

        Falls back gracefully to heuristic schema generation if Gemini is unreachable or unconfigured.
        Returns:
            Tuple of (schema_dict, generator_source)
        """
        gemini_client = self._get_gemini_client(override_key=gemini_api_key)

        if not gemini_client or not GENAI_AVAILABLE or not genai_types:
            logger.info("Using heuristic generator (Gemini client not initialized).")
            schema = self._heuristic_generate_schema(provider, endpoint_url, sample_data)
            return schema, "heuristic_fallback"

        model_name = settings.GEMINI_MODEL or "gemini-2.5-flash"

        sample_context = ""
        if sample_data:
            sample_preview = json.dumps(sample_data, default=str) if isinstance(sample_data, (dict, list)) else str(sample_data)
            sample_context = f"\nSample Endpoint Response / Probing Data:\n{sample_preview[:600]}\n"

        prompt = f"""You are an elite API Tool Architect for Google Gemini agents.
Generate a clean, production-grade Gemini Function Declaration JSON schema for this 3rd-party API:

Provider: {provider}
Endpoint URL: {endpoint_url}{sample_context}

The schema MUST strictly conform to the Google Gemini Tool FunctionDeclaration format:
{{
  "name": "lowercase_snake_case_name (max 63 chars, valid identifier)",
  "description": "Comprehensive explanation of what this tool does and when to call it",
  "parameters": {{
    "type": "OBJECT",
    "properties": {{
      "param_name": {{
        "type": "STRING" | "INTEGER" | "NUMBER" | "BOOLEAN" | "ARRAY",
        "description": "Clear explanation of this parameter",
        "enum": ["optional", "enum", "values"]
      }}
    }},
    "required": ["list", "of", "required", "parameters"]
  }}
}}

Guidelines:
1. 'name' must be descriptive, e.g., 'get_{provider.lower()}_data' or 'search_{provider.lower()}'.
2. Do NOT include authentication parameters (e.g., api_key, token, Authorization) in the schema properties, as auth is handled automatically by the backend vault.
3. Use standard uppercase Gemini Schema types: 'STRING', 'INTEGER', 'NUMBER', 'BOOLEAN', 'ARRAY', 'OBJECT'.
4. Return ONLY the raw JSON object. No Markdown fences, no explanation.
"""

        try:
            config = genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )

            response = await gemini_client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            raw_text = (response.text or "").strip()
            # Clean possible markdown wrap
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            schema = json.loads(raw_text)

            # Validate against google.genai types.FunctionDeclaration
            if "name" in schema and "parameters" in schema:
                # Sanitize name
                schema["name"] = sanitize_identifier(schema["name"])
                # Validate with genai SDK
                try:
                    genai_types.FunctionDeclaration(**schema)
                    logger.info(f"Successfully generated Gemini Function Declaration via {model_name}: {schema['name']}")
                    return schema, model_name
                except Exception as val_err:
                    logger.warning(f"Gemini output failed FunctionDeclaration validation ({val_err}), repairing...")
                    # Ensure parameters structure has type OBJECT
                    if not isinstance(schema.get("parameters"), dict):
                        schema["parameters"] = {"type": "OBJECT", "properties": {}}
                    if "type" not in schema["parameters"]:
                        schema["parameters"]["type"] = "OBJECT"
                    return schema, model_name

            # Fallback if parsed json didn't contain required keys
            fallback = self._heuristic_generate_schema(provider, endpoint_url, sample_data)
            return fallback, "heuristic_fallback"

        except Exception as exc:
            logger.warning(f"Gemini API call failed for schema generation: {exc}. Using heuristic fallback.")
            fallback = self._heuristic_generate_schema(provider, endpoint_url, sample_data)
            return fallback, "heuristic_fallback"

    async def register_key(
        self,
        api_key: str,
        provider: str,
        endpoint_url: str,
        session_id: str = "default",
        auth_type: Optional[str] = None,
        auth_header_name: Optional[str] = None,
        auth_param_name: Optional[str] = None,
        skip_verification: bool = False,
        gemini_api_key: Optional[str] = None,
        http_method: str = "GET",
        custom_headers: Optional[Dict[str, str]] = None,
        openapi_spec: Optional[Dict[str, Any]] = None,
    ) -> KeyRecord:
        """Full registration pipeline:

        1. Infers auth strategy.
        2. Probes API key with lightweight async httpx test call (unless skipped).
        3. Calls Gemini (gemini-2.5-flash) to auto-generate clean Function Declaration JSON schema.
        4. Stores key securely in in-memory session vault.
        """
        provider_clean = provider.strip()
        session_clean = (session_id or "default").strip()

        # Step 1: Infer auth delivery
        inferred_auth, header_name, param_name = self._infer_auth_strategy(
            provider_clean, endpoint_url, auth_type, auth_header_name, auth_param_name
        )

        # Step 2: Test call verification via httpx
        if skip_verification:
            verification = VerificationResult(
                is_active=True,
                status="unverified_skipped",
                status_code=0,
                latency_ms=0.0,
                auth_method=inferred_auth,
                message="Verification probe skipped by user request.",
            )
        else:
            verification = await self.verify_key_active(
                api_key=api_key,
                endpoint_url=endpoint_url,
                provider=provider_clean,
                auth_type=inferred_auth,
                auth_header_name=header_name,
                auth_param_name=param_name,
            )

        # Step 3: Auto-generate Gemini Function Declaration schema
        schema, generator_source = await self.generate_function_declaration(
            provider=provider_clean,
            endpoint_url=endpoint_url,
            sample_data=verification.sample_data,
            gemini_api_key=gemini_api_key,
        )

        # Step 4: Construct KeyRecord and securely store in vault
        masked = mask_api_key(api_key)
        record = KeyRecord(
            session_id=session_clean,
            provider=provider_clean,
            api_key=api_key,
            masked_key=masked,
            endpoint_url=endpoint_url,
            http_method=http_method,
            auth_type=inferred_auth,
            auth_header_name=header_name,
            auth_param_name=param_name,
            custom_headers=custom_headers or {},
            openapi_spec=openapi_spec,
            is_active=verification.is_active,
            verification=verification,
            function_declaration=schema,
            generator_source=generator_source,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        async with self._lock:
            if session_clean not in self._vault:
                self._vault[session_clean] = {}
            self._vault[session_clean][provider_clean.lower()] = record

        logger.info(
            f"Registered key for provider='{provider_clean}' in session='{session_clean}'. "
            f"Active={record.is_active}, Generator={generator_source}"
        )
        return record

    async def get_key(self, session_id: str, provider: str) -> Optional[KeyRecord]:
        """Retrieves a registered key record for a given session and provider."""
        async with self._lock:
            session_dict = self._vault.get((session_id or "default").strip(), {})
            return session_dict.get(provider.strip().lower())

    async def list_keys(self, session_id: str = "default") -> List[KeyRecord]:
        """Lists all registered keys in a session."""
        async with self._lock:
            session_dict = self._vault.get((session_id or "default").strip(), {})
            return list(session_dict.values())

    async def get_active_keys(self, session_id: str = "default") -> List[KeyRecord]:
        """Retrieves only verified active keys for a given session."""
        keys = await self.list_keys(session_id)
        return [k for k in keys if k.is_active]

    async def delete_key(self, session_id: str, provider: str) -> bool:
        """Deletes a key from a session vault."""
        async with self._lock:
            session_dict = self._vault.get((session_id or "default").strip())
            if session_dict and provider.strip().lower() in session_dict:
                del session_dict[provider.strip().lower()]
                return True
            return False

    async def clear_session(self, session_id: str) -> int:
        """Clears all keys for a session."""
        async with self._lock:
            session_clean = (session_id or "default").strip()
            if session_clean in self._vault:
                count = len(self._vault[session_clean])
                del self._vault[session_clean]
                return count
            return 0

    async def clear_all(self) -> None:
        """Clears the entire in-memory vault (all sessions)."""
        async with self._lock:
            self._vault.clear()

    async def get_gemini_tools(self, session_id: str = "default") -> List[Any]:
        """Retrieves all registered active tools in a session formatted for Gemini."""
        records = await self.list_keys(session_id)
        tools: List[Any] = []
        for r in records:
            if r.is_active and r.function_declaration:
                tools.append(r.to_gemini_tool())
        return tools


# Global default KeyVault instance
key_vault = KeyVault()
