import json
import time
from typing import List, Dict, Any
from openai import OpenAI

from src.schemas import (
    SemanticSearchInput,
    SpatialStatsInput,
    ToolExecutionLog,
    AgentResponse,
)
from src.embeddings_store import ListingVectorStore
from src.spatial_tools import SpatialMarketAnalyzer


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "semantic_property_search",
            "description": "Search property listings semantically using dense text embeddings (best for amenities, view, neighborhood vibe).",
            "parameters": SemanticSearchInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spatial_market_stats",
            "description": "Compute exact pricing statistics (mean, median, quartiles, count) within a kilometer radius of coordinates.",
            "parameters": SpatialStatsInput.model_json_schema(),
        },
    },
]


class GeospatialMarketAgent:
    """Tool-calling agent combining semantic vector retrieval with deterministic geospatial statistics."""

    def __init__(
        self,
        vector_store: ListingVectorStore,
        spatial_analyzer: SpatialMarketAnalyzer,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        max_steps: int = 5,
    ) -> None:
        self.vector_store = vector_store
        self.spatial_analyzer = spatial_analyzer
        self.client = client
        self.model = model
        self.max_steps = max_steps

    def _dispatch_tool(self, name: str, raw_args: str) -> str:
        parsed = json.loads(raw_args)
        if name == "semantic_property_search":
            validated = SemanticSearchInput(**parsed)
            res = self.vector_store.search(query=validated.query, top_k=validated.top_k)
            return json.dumps(res, ensure_ascii=False)
        elif name == "spatial_market_stats":
            validated = SpatialStatsInput(**parsed)
            res = self.spatial_analyzer.compute_radius_stats(
                latitude=validated.latitude,
                longitude=validated.longitude,
                radius_km=validated.radius_km,
            )
            return json.dumps(res, ensure_ascii=False)
        raise ValueError(f"Unknown tool requested: {name}")

    def run(self, user_query: str) -> AgentResponse:
        start_total = time.perf_counter()
        trace: List[ToolExecutionLog] = []

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a Geospatial & Market Intelligence Data Science Agent. "
                    "Never invent prices, coordinates, or statistics. Always call the provided tools "
                    "to retrieve grounded listings and compute exact spatial metrics before answering."
                ),
            },
            {"role": "user", "content": user_query},
        ]

        for step_idx in range(1, self.max_steps + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS_SPEC,
                tool_choice="auto",
                temperature=0.0,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                total_ms = round((time.perf_counter() - start_total) * 1000, 2)
                return AgentResponse(
                    answer=msg.content or "",
                    execution_trace=trace,
                    total_latency_ms=total_ms,
                )

            messages.append(msg.model_dump(exclude_none=True))

            for call in msg.tool_calls:
                t0 = time.perf_counter()
                tool_output = self._dispatch_tool(call.function.name, call.function.arguments)
                latency = round((time.perf_counter() - t0) * 1000, 2)

                trace.append(
                    ToolExecutionLog(
                        step=step_idx,
                        tool_name=call.function.name,
                        arguments=json.loads(call.function.arguments),
                        latency_ms=latency,
                        output_summary=tool_output[:200] + ("..." if len(tool_output) > 200 else ""),
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_output,
                    }
                )

        raise RuntimeError("Agent exceeded max_steps without producing a final response.")
