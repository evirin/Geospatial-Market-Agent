import pandas as pd
import pytest
from src.spatial_tools import SpatialMarketAnalyzer, haversine_km
from src.embeddings_store import ListingVectorStore


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "latitude": 37.9715,
                "longitude": 23.7257,
                "price": 120.0,
                "description": "Sunlit loft with direct Acropolis view and private balcony in Koukaki.",
            },
            {
                "id": 2,
                "latitude": 37.9755,
                "longitude": 23.7348,
                "price": 80.0,
                "description": "Quiet minimalist studio near Syntagma metro station, ideal for remote work.",
            },
            {
                "id": 3,
                "latitude": 40.6401,
                "longitude": 22.9444,
                "price": 65.0,
                "description": "Cozy seaside apartment near White Tower in Thessaloniki.",
            },
        ]
    )


def test_spatial_stats_radius_filtering(sample_df: pd.DataFrame) -> None:
    analyzer = SpatialMarketAnalyzer(sample_df)
    stats = analyzer.compute_radius_stats(latitude=37.9720, longitude=23.7260, radius_km=2.0)

    assert stats["count"] == 2
    assert stats["mean_price_eur"] == 100.0
    assert stats["median_price_eur"] == 100.0


def test_semantic_search_ranking(sample_df: pd.DataFrame) -> None:
    store = ListingVectorStore()
    store.build_from_dataframe(sample_df)
    results = store.search(query="apartment with view of the Parthenon and terrace", top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == 1
    assert results[0]["similarity_score"] > 0.3