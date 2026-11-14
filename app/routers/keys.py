"""Router for the Dynamic 'API Kitchen' Key Vault & Inspector.

Provides endpoints to register 3rd-party API keys, probe them via httpx,
and retrieve Gemini-generated Function Declaration schemas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.services.key_vault import KeyRecord, key_vault

router = APIRouter(tags=["API Kitchen Vault"])


class KeyRegisterRequest(BaseModel):
    """Request payload to register a 3rd-party API key."""

    api_key: str = Field(..., min_length=1, description="The 3rd-party API key or credential secret")
    provider: str = Field(..., min_length=1, description="Provider name (e.g. 'WeatherAPI', 'GitHub', 'CoinGecko')")
    endpoint_url: str = Field(..., min_length=4, description="Sample endpoint or documentation URL")
    session_id: Optional[str] = Field(default="default", description="Session or user namespace")
    auth_type: Optional[str] = Field(
        default=None,
        description="Auth delivery type: 'bearer', 'api_key_header', 'api_key_query', or None for auto-detection",
    )
    auth_header_name: Optional[str] = Field(
        default=None,
        description="Custom header name if using api_key_header or custom bearer",
    )
    auth_param_name: Optional[str] = Field(
        default=None,
        description="Custom query param name if using api_key_query",
    )
    skip_verification: bool = Field(
        default=False,
        description="Whether to skip probe call and register key directly",
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Optional Gemini API key override for schema generation",
    )


class KeyRegisterResponse(BaseModel):
    """Response returned upon successful key registration."""

    success: bool = Field(description="Whether registration was successfully processed")
    session_id: str = Field(description="Session namespace holding this key")
    provider: str = Field(description="Provider name")
    masked_key: str = Field(description="Safely masked representation of the API key")
    endpoint_url: str = Field(description="Sample endpoint URL")
    auth_type: str = Field(description="Resolved auth delivery type")
    is_active: bool = Field(description="Whether probe test confirmed key is active")
    verification: Dict[str, Any] = Field(description="Probe test details and status")
    function_declaration: Dict[str, Any] = Field(
        description="Clean JSON Function Declaration schema for Gemini"
    )
    generator_source: str = Field(
        description="Schema generator used ('gemini-2.5-flash' or 'heuristic_fallback')"
    )
    message: str = Field(description="Human-readable result summary")


class KeyListItem(BaseModel):
    """Masked key overview for listing in vault."""

    session_id: str
    provider: str
    masked_key: str
    endpoint_url: str
    auth_type: str
    is_active: bool
    generator_source: str
    function_declaration_name: str
    created_at: str
    updated_at: str


@router.post(
    "/register",
    response_model=KeyRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a 3rd-party API key, verify it via httpx, and generate Gemini tool schema",
)
async def register_key(payload: KeyRegisterRequest) -> KeyRegisterResponse:
    """Securely registers a 3rd-party API key into the in-memory session vault.

    - Performs a lightweight async probe with httpx to verify the key is active.
    - Uses Gemini (`gemini-2.5-flash`) to auto-generate a clean Function Declaration JSON schema.
    - Stores the record in memory with secure masking for output.
    """
    try:
        record: KeyRecord = await key_vault.register_key(
            api_key=payload.api_key,
            provider=payload.provider,
            endpoint_url=payload.endpoint_url,
            session_id=payload.session_id or "default",
            auth_type=payload.auth_type,
            auth_header_name=payload.auth_header_name,
            auth_param_name=payload.auth_param_name,
            skip_verification=payload.skip_verification,
            gemini_api_key=payload.gemini_api_key,
        )

        message = (
            f"Successfully registered key for '{record.provider}'. "
            f"Verification: {record.verification.status} (active={record.is_active}). "
            f"Gemini schema: '{record.function_declaration.get('name')}' via {record.generator_source}."
        )

        return KeyRegisterResponse(
            success=True,
            session_id=record.session_id,
            provider=record.provider,
            masked_key=record.masked_key,
            endpoint_url=record.endpoint_url,
            auth_type=record.auth_type,
            is_active=record.is_active,
            verification=record.verification.model_dump(),
            function_declaration=record.function_declaration,
            generator_source=record.generator_source,
            message=message,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register key for provider '{payload.provider}': {str(exc)}",
        )


@router.get(
    "",
    response_model=List[KeyListItem],
    summary="List all registered keys in a session vault (masked)",
)
@router.get(
    "/list",
    response_model=List[KeyListItem],
    summary="List all registered keys in a session vault (alias)",
)
async def list_keys(
    session_id: str = Query(default="default", description="Session namespace"),
) -> List[KeyListItem]:
    """Lists all registered keys in the in-memory session vault with keys safely masked."""
    records = await key_vault.list_keys(session_id=session_id)
    results = []
    for r in records:
        func_name = r.function_declaration.get("name", "unknown") if isinstance(r.function_declaration, dict) else "unknown"
        results.append(
            KeyListItem(
                session_id=r.session_id,
                provider=r.provider,
                masked_key=r.masked_key,
                endpoint_url=r.endpoint_url,
                auth_type=r.auth_type,
                is_active=r.is_active,
                generator_source=r.generator_source,
                function_declaration_name=func_name,
                created_at=r.created_at.isoformat(),
                updated_at=r.updated_at.isoformat(),
            )
        )
    return results


@router.get(
    "/tools",
    summary="Retrieve all Gemini-compatible tools for active keys in a session",
)
async def get_session_tools(
    session_id: str = Query(default="default", description="Session namespace"),
) -> Dict[str, Any]:
    """Returns all Gemini Tool objects for active keys in the session."""
    tools = await key_vault.get_gemini_tools(session_id=session_id)
    serialized = []
    for t in tools:
        if hasattr(t, "to_json_dict"):
            serialized.append(t.to_json_dict())
        elif hasattr(t, "model_dump"):
            serialized.append(t.model_dump())
        else:
            serialized.append(str(t))
    return {
        "session_id": session_id,
        "count": len(tools),
        "tools": serialized,
    }


@router.get(
    "/{provider}",
    summary="Get registered key metadata and Gemini schema for a provider",
)
async def get_key_details(
    provider: str,
    session_id: str = Query(default="default", description="Session namespace"),
) -> Dict[str, Any]:
    """Retrieves safe metadata and schema for a specific provider in a session."""
    record = await key_vault.get_key(session_id=session_id, provider=provider)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Key for provider '{provider}' not found in session '{session_id}'.",
        )
    return record.to_safe_dict()


@router.get(
    "/{provider}/schema",
    summary="Get Gemini Function Declaration schema for a provider",
)
async def get_key_schema(
    provider: str,
    session_id: str = Query(default="default", description="Session namespace"),
) -> Dict[str, Any]:
    """Retrieves the clean JSON Function Declaration schema for Gemini tool use."""
    record = await key_vault.get_key(session_id=session_id, provider=provider)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Key for provider '{provider}' not found in session '{session_id}'.",
        )
    return record.function_declaration


@router.delete(
    "/{provider}",
    summary="Remove a registered key from the session vault",
)
async def delete_key(
    provider: str,
    session_id: str = Query(default="default", description="Session namespace"),
) -> Dict[str, Any]:
    """Deletes a key from the in-memory session vault."""
    deleted = await key_vault.delete_key(session_id=session_id, provider=provider)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Key for provider '{provider}' not found in session '{session_id}'.",
        )
    return {
        "success": True,
        "message": f"Successfully removed key for provider '{provider}' from session '{session_id}'.",
    }


@router.post(
    "/clear",
    summary="Clear all keys from a session vault",
)
async def clear_session(
    session_id: str = Query(default="default", description="Session namespace"),
) -> Dict[str, Any]:
    """Clears all keys in the specified session."""
    count = await key_vault.clear_session(session_id=session_id)
    return {
        "success": True,
        "cleared_count": count,
        "message": f"Cleared {count} key(s) from session '{session_id}'.",
    }
