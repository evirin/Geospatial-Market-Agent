# Geospatial & Market Intelligence Agent

An autonomous tool-calling AI agent that combines **dense semantic retrieval** (Hugging Face `sentence-transformers` + `FAISS`) with **deterministic geospatial and statistical computation** (vectorized Haversine distance & market pricing analytics) over **2,000 real short-term rental listings in Athens (2025)**.

Built with production engineering practices—strict Pydantic v2 schema validation, modular OOP architecture, per-step observability traces, and automated unit testing (`pytest`)—to eliminate numerical hallucinations in LLM-driven analytical workflows.

## Architecture

```text
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