"""Shared domain and data models for the Adaptive API Kitchen & Orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class EndpointSpec(BaseModel):
    """Specification of an API endpoint available in the Kitchen."""

    path: str = Field(description="URL path or relative endpoint, e.g. '/v1/search'")
    method: str = Field(default="GET", description="HTTP method (GET, POST, PUT, DELETE, etc.)")
    summary: str = Field(default="", description="Short description of what the endpoint does")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Expected query/body parameters")


class KitchenAPI(BaseModel):
    """3rd-party API credentials, base URL, and endpoints registered in the Kitchen."""

    api_name: str = Field(description="Unique identifier for the API, e.g. 'weather_api', 'github'")
    base_url: str = Field(description="Base URL, e.g. 'https://api.weatherapi.com'")
    description: str = Field(default="", description="Summary of services this API provides")
    api_key: Optional[str] = Field(default=None, description="Secret token or API key")
    auth_type: str = Field(default="bearer", description="'bearer', 'api_key_header', 'api_key_query', or 'none'")
    auth_header_name: str = Field(default="Authorization", description="Header name if auth_type is api_key_header")
    auth_param_name: str = Field(default="key", description="Query param name if auth_type is api_key_query")
    endpoints: List[EndpointSpec] = Field(default_factory=list, description="List of accessible endpoints")
    schema_dict: Optional[Dict[str, Any]] = Field(default=None, description="Optional raw OpenAPI/JSON spec")


class APITask(BaseModel):
    """An individual API call task planned by the Planner Agent."""

    task_id: str = Field(description="Unique task identifier, e.g. 'task_1'")
    api_name: str = Field(description="Name of the kitchen API to call")
    endpoint: str = Field(description="Target endpoint path")
    method: str = Field(default="GET", description="HTTP method")
    description: str = Field(description="Explanation of why this task is being executed")
    params: Dict[str, Any] = Field(default_factory=dict, description="URL query parameters")
    json_body: Optional[Union[Dict[str, Any], list]] = Field(default=None, description="JSON body payload")
    wave: int = Field(default=1, description="Execution wave index (tasks in the same wave run in parallel)")
    depends_on: List[str] = Field(default_factory=list, description="IDs of tasks this task depends on")


class Plan(BaseModel):
    """Structured execution plan produced by the Planner Agent."""

    goal: str = Field(description="High-level interpretation of the user goal")
    reasoning: str = Field(description="Reasoning behind the chosen tasks and sequence")
    tasks: List[APITask] = Field(default_factory=list, description="List of API tasks to execute")


class TaskExecutionResult(BaseModel):
    """Outcome of an executed API task."""

    task_id: str
    api_name: str
    endpoint: str
    method: str
    success: bool
    status_code: int
    data: Optional[Any] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    self_healed: bool = False
    retry_count: int = 0
    executed_payload: Dict[str, Any] = Field(default_factory=dict)


class SynthesizedResponse(BaseModel):
    """Final coherent structured response synthesized from all API data."""

    summary: str = Field(description="Executive summary of the results")
    content: str = Field(description="Detailed narrative answer synthesizing live 3rd-party data")
    key_insights: List[str] = Field(default_factory=list, description="Key extracted data points")
    apis_used: List[str] = Field(default_factory=list, description="Names of APIs successfully queried")
    task_results: List[TaskExecutionResult] = Field(default_factory=list, description="Detailed per-task execution audit")
    total_time_seconds: float = Field(default=0.0, description="Total pipeline execution duration")
