"""RFM feature engineering and K-Means segmentation."""
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def build_rfm(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions to one row per customer."""
    snapshot = pd.to_datetime(data["Order_Date"]).max() + pd.Timedelta(days=1)
    return data.assign(Order_Date=pd.to_datetime(data["Order_Date"])).groupby("Customer_ID").agg(Recency=("Order_Date", lambda dates: (snapshot - dates.max()).days), Frequency=("Order_Date", "count"), Monetary=("Order_Value", "sum")).reset_index()

def fit_segmentation(rfm: pd.DataFrame, figure_dir: str | Path, k: int | None = None) -> dict[str, Any]:
    """Evaluate k=2..8 and fit the selected K-Means model."""
    features = ["Recency", "Frequency", "Monetary"]; scaler = StandardScaler(); scaled = scaler.fit_transform(rfm[features]); ks = range(2, 9); inertias = []; silhouettes = []
    for candidate in ks:
        fitted = KMeans(n_clusters=candidate, n_init=20, random_state=42).fit(scaled); inertias.append(float(fitted.inertia_)); silhouettes.append(float(silhouette_score(scaled, fitted.labels_)))
    selected_k = k or int(list(ks)[int(np.argmax(silhouettes))]); model = KMeans(n_clusters=selected_k, n_init=20, random_state=42).fit(scaled); segmented = rfm.copy(); segmented["Cluster"] = model.labels_; projection = PCA(n_components=2, random_state=42).fit_transform(scaled)
    output_dir = Path(figure_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4)); axes[0].plot(list(ks), inertias, marker="o"); axes[0].set_title("Elbow / WCSS"); axes[1].plot(list(ks), silhouettes, marker="o", color="#ed553b"); axes[1].set_title("Silhouette scores"); fig.tight_layout(); fig.savefig(output_dir / "elbow_silhouette_curve.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 6)); axis.scatter(projection[:, 0], projection[:, 1], c=model.labels_, cmap="viridis", alpha=.75); axis.set_title(f"Customer RFM segments (k={selected_k})"); axis.set_xlabel("PC1"); axis.set_ylabel("PC2"); fig.tight_layout(); fig.savefig(output_dir / "pca_clusters_2d.png", dpi=150); plt.close(fig)
    return {"rfm": segmented, "model": model, "scaler": scaler, "selected_k": selected_k, "inertias": inertias, "silhouettes": silhouettes, "profile": segmented.groupby("Cluster")[features].mean().round(2)}
