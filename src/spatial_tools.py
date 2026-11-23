import numpy as np
import pandas as pd
from typing import Dict, Any


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized Haversine distance calculation in kilometers."""
    earth_radius_km = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * earth_radius_km * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


class SpatialMarketAnalyzer:
    """Computes deterministic spatial pricing and macroeconomic statistics."""

    def __init__(self, df: pd.DataFrame) -> None:
        required = {"latitude", "longitude", "price"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        self.df = df.copy()

    def compute_radius_stats(self, latitude: float, longitude: float, radius_km: float) -> Dict[str, Any]:
        distances = haversine_km(
            latitude,
            longitude,
            self.df["latitude"].to_numpy(dtype=float),
            self.df["longitude"].to_numpy(dtype=float),
        )
        subset = self.df[distances <= radius_km]

        if subset.empty:
            return {
                "count": 0,
                "radius_km": radius_km,
                "message": "No listings found within the specified radius.",
            }

        prices = subset["price"].astype(float)
        stats: Dict[str, Any] = {
            "count": int(len(subset)),
            "radius_km": radius_km,
            "mean_price_eur": round(float(prices.mean()), 2),
            "median_price_eur": round(float(prices.median()), 2),
            "std_price_eur": round(float(prices.std(ddof=0)), 2),
            "p25_price_eur": round(float(np.percentile(prices, 25)), 2),
            "p75_price_eur": round(float(np.percentile(prices, 75)), 2),
        }

        if "macro_cpi_yoy" in subset.columns:
            stats["avg_macro_cpi_yoy"] = round(float(subset["macro_cpi_yoy"].mean()), 3)

        return stats
