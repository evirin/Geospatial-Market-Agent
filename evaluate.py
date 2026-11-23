import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.embeddings_store import ListingVectorStore
from src.spatial_tools import SpatialMarketAnalyzer
from src.agent import GeospatialMarketAgent


def main() -> None:
    load_dotenv()
    df = pd.read_csv("data/athens_listings_2025.csv")
    print(f"Loaded {len(df):,} clean Athens 2025 listings.")

    vector_store = ListingVectorStore()
    vector_store.build_from_dataframe(df)
    spatial_analyzer = SpatialMarketAnalyzer(df)

    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

    agent = GeospatialMarketAgent(
        client=client,
        model=model,
        vector_store=vector_store,
        spatial_analyzer=spatial_analyzer,
    )

    user_prompt = (
        "Find the top 3 listings in Athens matching 'Acropolis view balcony fast wifi remote work', "
        "and calculate the exact mean and median nightly price within a 1.5 km radius of Koukaki "
        "(latitude 37.9689, longitude 23.7229)."
    )

    result = agent.run(user_prompt)

    print("\n=== FINAL ANSWER ===")
    print(result.answer)
    print("\n=== EXECUTION TRACE (OBSERVABILITY) ===")
    for step in result.execution_trace:
        print(f"Step {step.step} | Tool: {step.tool_name} | Latency: {step.latency_ms}ms | Args: {step.arguments}")
    print(f"Total Latency: {result.total_latency_ms}ms")


if __name__ == "__main__":
    main()