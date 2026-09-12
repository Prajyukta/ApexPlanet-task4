"""Regression and churn classification workflows."""
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def run_predictive_models(data: pd.DataFrame, figure_dir: str | Path) -> dict[str, Any]:
    """Fit and score order-value regression and two churn classifiers."""
    figure_dir = Path(figure_dir); figure_dir.mkdir(parents=True, exist_ok=True); features = ["Recency_Days", "Frequency", "Discount_Pct", "Delivery_Days"]
    x_train, x_test, y_train, y_test = train_test_split(data[features], data["Order_Value"], test_size=.2, random_state=42); regression = LinearRegression().fit(x_train, y_train); predictions = regression.predict(x_test)
    regression_metrics = {"R2": float(r2_score(y_test, predictions)), "MAE": float(mean_absolute_error(y_test, predictions)), "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions)))}
    fig, axis = plt.subplots(figsize=(7, 4)); axis.scatter(y_test, predictions, alpha=.45); axis.set_xlabel("Actual"); axis.set_ylabel("Predicted"); axis.set_title("Order-value regression"); fig.tight_layout(); fig.savefig(figure_dir / "regression_residuals.png", dpi=150); plt.close(fig)
    x_train, x_test, y_train, y_test = train_test_split(data[features], data["Churn"], test_size=.2, stratify=data["Churn"], random_state=42); logistic = LogisticRegression(max_iter=2_000).fit(x_train, y_train); tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=42).fit(x_train, y_train)
    results: dict[str, Any] = {"regression": {"model": regression, "metrics": regression_metrics}, "classification": {}}
    for name, model in {"logistic": logistic, "decision_tree": tree}.items():
        predicted = model.predict(x_test); probabilities = model.predict_proba(x_test)[:, 1]
        results["classification"][name] = {"model": model, "confusion_matrix": confusion_matrix(y_test, predicted), "accuracy": float(accuracy_score(y_test, predicted)), "precision": float(precision_score(y_test, predicted, zero_division=0)), "recall": float(recall_score(y_test, predicted, zero_division=0)), "f1": float(f1_score(y_test, predicted, zero_division=0)), "roc_auc": float(roc_auc_score(y_test, probabilities)), "report": classification_report(y_test, predicted, zero_division=0), "probabilities": probabilities}
    fpr, tpr, _ = roc_curve(y_test, results["classification"]["logistic"]["probabilities"]); fig, axis = plt.subplots(figsize=(7, 5)); axis.plot(fpr, tpr, label=f"Logistic AUC={results['classification']['logistic']['roc_auc']:.3f}"); axis.plot([0, 1], [0, 1], "--", color="grey"); axis.set_title("ROC-AUC: churn classification"); axis.set_xlabel("False positive rate"); axis.set_ylabel("True positive rate"); axis.legend(); fig.tight_layout(); fig.savefig(figure_dir / "roc_auc_curve.png", dpi=150); plt.close(fig)
    coefficients = pd.Series(logistic.coef_[0], index=features).sort_values(); fig, axis = plt.subplots(figsize=(8, 4)); coefficients.plot.barh(ax=axis, color="#20639b"); axis.set_title("Logistic churn coefficients (log odds)"); fig.tight_layout(); fig.savefig(figure_dir / "feature_importance.png", dpi=150); plt.close(fig)
    results["classification"]["features"] = features; return results
