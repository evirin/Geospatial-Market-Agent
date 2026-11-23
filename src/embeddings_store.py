from pathlib import Path
from typing import List, Dict, Any
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class ListingVectorStore:
    """Manages dense text embeddings and FAISS similarity search for property listings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: List[Dict[str, Any]] = []

    def build_from_dataframe(self, df: pd.DataFrame, text_col: str = "description") -> None:
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in dataset.")

        clean_df = df.dropna(subset=[text_col]).reset_index(drop=True)
        texts = clean_df[text_col].astype(str).tolist()

        # Normalize embeddings so Inner Product equals Cosine Similarity
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = clean_df.to_dict(orient="records")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None or not self.metadata:
            raise RuntimeError("Vector index has not been initialized.")

        q_emb = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q_emb = np.asarray(q_emb, dtype="float32")

        scores, indices = self.index.search(q_emb, min(top_k, len(self.metadata)))
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["similarity_score"] = round(float(score), 4)
            results.append(item)
        return results
