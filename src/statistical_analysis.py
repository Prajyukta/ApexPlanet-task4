"""Descriptive and inferential statistical analyses."""
from typing import Any
import pandas as pd
from scipy import stats

def descriptive_statistics(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return requested moments for continuous variables."""
    rows: dict[str, dict[str, float]] = {}
    for column in columns:
        values = data[column].dropna()
        rows[column] = {"Mean": values.mean(), "Median": values.median(), "Mode": values.mode().iloc[0], "Variance": values.var(), "Std_Dev": values.std(), "Skewness": stats.skew(values), "Kurtosis": stats.kurtosis(values)}
    return pd.DataFrame(rows).T

def confidence_interval(values: pd.Series, confidence: float = .95) -> tuple[float, float]:
    """Student-t confidence interval for a population mean."""
    clean = values.dropna().astype(float)
    margin = stats.t.ppf((1 + confidence) / 2, len(clean) - 1) * stats.sem(clean)
    return float(clean.mean() - margin), float(clean.mean() + margin)

def run_hypothesis_tests(data: pd.DataFrame) -> dict[str, Any]:
    """Run t-test, chi-square, ANOVA, and requested confidence intervals."""
    top_regions = data["Region"].value_counts().head(2).index.tolist()
    samples = [data.loc[data["Region"] == region, "Order_Value"] for region in top_regions]
    t_stat, t_p = stats.ttest_ind(*samples, equal_var=False)
    contingency = pd.crosstab(data["Region"], data["Churn"])
    chi_stat, chi_p, chi_dof, _ = stats.chi2_contingency(contingency)
    groups = [group["Order_Value"].to_numpy() for _, group in data.groupby("Category")]
    f_stat, anova_p = stats.f_oneway(*groups)
    return {
        "t_test": {"regions": top_regions, "statistic": float(t_stat), "p_value": float(t_p), "null": "Mean Order_Value is equal across the two top regions.", "alternative": "The two top-region means differ."},
        "chi_square": {"statistic": float(chi_stat), "p_value": float(chi_p), "dof": int(chi_dof), "null": "Region and Churn are independent.", "alternative": "Region and Churn are associated."},
        "anova": {"statistic": float(f_stat), "p_value": float(anova_p), "null": "All category mean Order_Value values are equal.", "alternative": "At least one category mean differs."},
        "ci_order_value": confidence_interval(data["Order_Value"]),
        "ci_net_revenue": confidence_interval(data["Order_Value"] * (1 - data["Discount_Pct"] / 100)),
    }
