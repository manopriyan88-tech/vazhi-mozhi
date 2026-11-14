"""Universal HTTP Executor for 3rd-party API requests.

Executes dynamic HTTP requests with flexible authentication, serialization,
and response parsing.
"""

from typing import Any, Dict, Optional, Union
import time
import httpx
from pydantic import BaseModel, Field


class APIRequest(BaseModel):
    """Specification of an outgoing HTTP request to a 3rd-party API."""

    method: str = Field(default="GET", description="HTTP method (GET, POST, PUT, PATCH, DELETE, etc.)")
    url: str = Field(description="Target URL endpoint")
    headers: Dict[str, str] = Field(default_factory=dict, description="Custom HTTP headers")
    params: Dict[str, Any] = Field(default_factory=dict, description="URL query parameters")
    json_body: Optional[Union[Dict[str, Any], list]] = Field(default=None, description="JSON payload")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Form data or urlencoded body")
    auth_type: Optional[str] = Field(default=None, description="Authentication type: 'bearer', 'api_key_header', 'api_key_query', 'basic', or None")
    auth_token: Optional[str] = Field(default=None, description="Token, API key, or credential secret")
    auth_header_name: str = Field(default="Authorization", description="Header name for API key or Bearer token")
    auth_param_name: str = Field(default="api_key", description="Query param name if API key is passed in URL query")
    auth_username: Optional[str] = Field(default=None, description="Username for Basic auth")
    auth_password: Optional[str] = Field(default=None, description="Password for Basic auth")
    timeout_seconds: float = Field(default=30.0, description="HTTP timeout in seconds")


class APIResponse(BaseModel):
    """Normalized response from an executed 3rd-party API call."""

    status_code: int = Field(description="HTTP status code")
    headers: Dict[str, str] = Field(default_factory=dict, description="Response headers")
    data: Optional[Any] = Field(default=None, description="Parsed JSON response payload if applicable")
    text: str = Field(default="", description="Raw response text")
    is_success: bool = Field(default=False, description="Whether status code is 2xx")
    elapsed_seconds: float = Field(default=0.0, description="Execution time in seconds")
    error: Optional[str] = Field(default=None, description="Error message if request failed")
    request_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of request parameters executed")


class UniversalHTTPExecutor:
    """Dispatches dynamic HTTP calls to external APIs with authentication and payload formatting."""

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self._client = client

    def _prepare_request(self, req: APIRequest) -> Dict[str, Any]:
        """Injects authentication and cleans headers and parameters."""
        headers = dict(req.headers)
        params = dict(req.params)
        auth = None

        if req.auth_type and req.auth_token:
            auth_type_lower = req.auth_type.lower()
            if auth_type_lower in ("bearer", "token"):
                headers[req.auth_header_name] = f"Bearer {req.auth_token}"
            elif auth_type_lower in ("api_key", "api_key_header", "apikey"):
                headers[req.auth_header_name] = req.auth_token
            elif auth_type_lower in ("query", "api_key_query"):
                params[req.auth_param_name] = req.auth_token

        if req.auth_type and req.auth_type.lower() == "basic" and req.auth_username:
            auth = (req.auth_username, req.auth_password or "")

        return {
            "method": req.method.upper(),
            "url": req.url,
            "headers": headers,
            "params": params,
            "json": req.json_body,
            "data": req.data,
            "auth": auth,
            "timeout": httpx.Timeout(req.timeout_seconds),
        }

    async def execute(self, request: APIRequest) -> APIResponse:
        """Executes the HTTP request asynchronously and returns a normalized APIResponse."""
        prep = self._prepare_request(request)
        start_time = time.perf_counter()

        request_summary = {
            "method": prep["method"],
            "url": prep["url"],
            "params": prep["params"],
            "has_json": prep["json"] is not None,
            "has_data": prep["data"] is not None,
        }

        try:
            if self._client:
                raw_response = await self._client.request(
                    method=prep["method"],
                    url=prep["url"],
                    headers=prep["headers"],
                    params=prep["params"],
                    json=prep["json"],
                    data=prep["data"],
                    auth=prep["auth"],
                    timeout=prep["timeout"],
                )
            else:
                async with httpx.AsyncClient() as client:
                    raw_response = await client.request(
                        method=prep["method"],
                        url=prep["url"],
                        headers=prep["headers"],
                        params=prep["params"],
                        json=prep["json"],
                        data=prep["data"],
                        auth=prep["auth"],
                        timeout=prep["timeout"],
                    )

            elapsed = time.perf_counter() - start_time

            # Attempt JSON decode
            parsed_data = None
            try:
                parsed_data = raw_response.json()
            except Exception:
                parsed_data = None

            is_success = raw_response.is_success
            error_message = None
            if not is_success:
                error_message = (
                    parsed_data.get("message") or parsed_data.get("error") or parsed_data.get("detail")
                    if isinstance(parsed_data, dict)
                    else raw_response.text[:1000]
                )

            return APIResponse(
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
                data=parsed_data,
                text=raw_response.text,
                is_success=is_success,
                elapsed_seconds=round(elapsed, 4),
                error=str(error_message) if error_message else None,
                request_summary=request_summary,
            )

        except httpx.HTTPError as exc:
            elapsed = time.perf_counter() - start_time
            return APIResponse(
                status_code=0,
                headers={},
                data=None,
                text=str(exc),
                is_success=False,
                elapsed_seconds=round(elapsed, 4),
                error=f"HTTP network error: {type(exc).__name__}: {str(exc)}",
                request_summary=request_summary,
            )
