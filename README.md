# Geospatial & Market Intelligence Agent

An autonomous tool-calling AI agent that combines **dense semantic retrieval** (Hugging Face `sentence-transformers` + `FAISS`) with **deterministic geospatial and statistical computation** (vectorized Haversine distance & market pricing analytics) over **2,000 real short-term rental listings in Athens (2025)**.

Built with production engineering practices—strict Pydantic v2 schema validation, modular OOP architecture, per-step observability traces, and automated unit testing (`pytest`)—to eliminate numerical hallucinations in LLM-driven analytical workflows.

## Architecture

    User Query
       │
       ▼
    ┌──────────────────────────────────────────┐
    │  GeospatialMarketAgent (Tool-Calling)    │
    │  • Schema validation (Pydantic v2)       │
    │  • Step-by-step latency & trace logging  │
    └──────┬───────────────────────────┬───────┘
           │                           │
           ▼                           ▼
    ┌─────────────────────┐   ┌────────────────────────────┐
    │ Semantic Search     │   │ Spatial Market Analyzer    │
    │ • MiniLM-L6-v2      │   │ • Vectorized Haversine     │
    │ • FAISS FlatIP      │   │ • Mean / Median / IQR      │
    │ • Cosine Similarity │   │ • Radius-based filtering   │
    └─────────────────────┘   └────────────────────────────┘

## Key Engineering Decisions

* **Zero Numerical Hallucinations:** LLMs are strictly restricted to intent routing, query decomposition, and natural-language synthesis. All spatial distance filtering (great-circle Haversine) and market aggregations (mean, median, IQR, standard deviation) execute deterministically in vectorized NumPy/Pandas.
* **Exact Cosine Search via Inner Product:** Listing descriptions are L2-normalized at index build time (`all-MiniLM-L6-v2`), allowing `faiss.IndexFlatIP` to compute exact cosine similarity with sub-40ms query latency.
* **Schema Enforcement & Observability:** Every tool invocation and agent response is validated against Pydantic v2 contracts (`ToolExecutionLog`, `AgentResponse`), capturing per-step arguments and millisecond latency metrics.

## Project Structure

    geospatial_market_agent/
    ├── data/
    │   └── athens_listings_2025.csv   # 2,000 cleaned Athens 2025 listings
    ├── src/
    │   ├── schemas.py                 # Pydantic input/output & observability models
    │   ├── embeddings_store.py        # Dense text embeddings & FAISS vector index
    │   ├── spatial_tools.py           # Deterministic Haversine & statistical engine
    │   └── agent.py                   # Tool-calling orchestration loop with trace logging
    ├── tests/
    │   └── test_tools.py              # Automated unit tests (pytest)
    ├── evaluate.py                    # End-to-end agent evaluation & latency benchmark
    └── requirements.txt

## Quickstart

1. **Install dependencies:**

       python -m venv .venv
       source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
       pip install -r requirements.txt

2. **Run unit tests:**

       pytest -v

3. **Run end-to-end agent evaluation:**
   Create a `.env` file with your OpenAI-compatible endpoint (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`) and run:

       python evaluate.py

## Sample Benchmark Output (2,000 Athens Listings)

**Prompt:** *"Find the top 3 listings in Athens matching 'Acropolis view balcony fast wifi remote work', and calculate the exact mean and median nightly price within a 1.5 km radius of Koukaki (latitude 37.9689, longitude 23.7229)."*

**Deterministic Spatial Output (Koukaki 1.5 km radius):**

| Metric | Value (€) |
| :--- | :--- |
| Mean nightly price | **107.54** |
| Median nightly price | **75.00** |
| 25th percentile | 55.00 |
| 75th percentile | 117.00 |
| Standard deviation | 103.52 |

**Execution Trace:**

    === EXECUTION TRACE (OBSERVABILITY) ===
    Step 1 | Tool: semantic_property_search | Latency: 36.68ms | Args: {'query': 'Acropolis view balcony fast wifi remote work', 'top_k': 3}
    Step 2 | Tool: spatial_market_stats | Latency: 17.91ms | Args: {'latitude': 37.9689, 'longitude': 23.7229, 'radius_km': 1.5}
    Total Latency: 2392.78ms
