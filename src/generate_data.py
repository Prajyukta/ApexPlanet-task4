"""Generate a reproducible synthetic retail transaction dataset."""
from pathlib import Path
from typing import Final
import numpy as np
import pandas as pd

SEED: Final[int] = 42

def generate_transactions(n_records: int = 2_500, seed: int = SEED) -> pd.DataFrame:
    """Create realistic transaction-level data for 2024-2025."""
    rng = np.random.default_rng(seed)
    customer_count = max(500, int(n_records * 0.28))
    customer_ids = [f"C{number:05d}" for number in range(1, customer_count + 1)]
    customer_risk = rng.beta(2.2, 8.0, customer_count)
    selected = rng.choice(customer_ids, size=n_records, replace=True)
    positions = np.array([int(customer_id[1:]) - 1 for customer_id in selected])
    start_day = pd.Timestamp("2024-01-01").value // 86_400_000_000_000
    end_day = pd.Timestamp("2025-12-31").value // 86_400_000_000_000
    dates = pd.to_datetime(rng.integers(start_day, end_day + 1, n_records), unit="D")
    regions = rng.choice(["North", "South", "East", "West"], n_records, p=[.27, .24, .22, .27])
    categories = rng.choice(["Electronics", "Home", "Fashion", "Beauty", "Sports"], n_records, p=[.24, .22, .25, .16, .13])
    category_base = {"Electronics": 155, "Home": 92, "Fashion": 68, "Beauty": 54, "Sports": 83}
    base_values = np.array([category_base[category] for category in categories])
    discounts = np.clip(rng.normal(12, 7, n_records), 0, 35)
    order_values = np.maximum(12, base_values * rng.lognormal(0, .34, n_records) * (1 - discounts / 180))
    delivery_days = np.clip(np.rint(rng.normal(3.7, 1.35, n_records)), 1, 9).astype(int)
    recency = np.clip(np.rint(rng.gamma(2.8, 23, n_records)), 1, 180).astype(int)
    frequency = np.clip(np.rint(rng.poisson(4.5, n_records) + 1), 1, 18).astype(int)
    churn_probability = np.clip(.05 + customer_risk[positions] * .65 + (recency > 75) * .12 - frequency * .008, .02, .92)
    churn = rng.binomial(1, churn_probability)
    ratings = np.clip(np.rint(4.2 - delivery_days * .11 - discounts * .008 - churn * .35 + rng.normal(0, .55, n_records)), 1, 5).astype(int)
    return pd.DataFrame({
        "Customer_ID": selected, "Order_Date": dates, "Region": regions, "Category": categories,
        "Order_Value": np.round(order_values, 2), "Discount_Pct": np.round(discounts, 2),
        "Delivery_Days": delivery_days, "Recency_Days": recency, "Frequency": frequency,
        "Churn": churn, "Customer_Rating": ratings,
    }).sort_values("Order_Date").reset_index(drop=True)

def save_transactions(output_path: str | Path, n_records: int = 2_500, seed: int = SEED) -> pd.DataFrame:
    """Generate and persist transactions, creating the parent directory."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_transactions(n_records, seed)
    data.to_csv(path, index=False)
    return data

if __name__ == "__main__":
    save_transactions(Path(__file__).parents[1] / "data/processed/customer_transactions_task4.csv")
