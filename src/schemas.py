from typing import Any, List, Optional
from pydantic import BaseModel, Field


class SemanticSearchInput(BaseModel):
    query: str = Field(..., description="Natural language query describing desired property features or neighborhood vibe.")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top matching listings to retrieve.")


class SpatialStatsInput(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Target latitude in decimal degrees.")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Target longitude in decimal degrees.")
    radius_km: float = Field(default=2.0, gt=0.1, le=50.0, description="Search radius in kilometers.")


class ToolExecutionLog(BaseModel):
    step: int
    tool_name: str
    arguments: dict[str, Any]
    latency_ms: float
    output_summary: str


class AgentResponse(BaseModel):
    answer: str
    execution_trace: List[ToolExecutionLog]
    total_latency_ms: float
