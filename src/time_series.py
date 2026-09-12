"""Weekly sales diagnostics and ARIMA forecasting."""
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def prepare_weekly_sales(data: pd.DataFrame) -> pd.Series:
    """Aggregate dated transactions to a complete Monday-ending weekly index."""
    indexed = data.assign(Order_Date=pd.to_datetime(data["Order_Date"])).set_index("Order_Date")
    return indexed["Order_Value"].resample("W-MON").sum().asfreq("W-MON").ffill().rename("Weekly_Sales")

def adf_report(series: pd.Series) -> dict[str, Any]:
    """Return ADF test outputs in a serializable form."""
    result = adfuller(series.dropna(), autolag="AIC", result_object=False)
    return {"statistic": float(result[0]), "p_value": float(result[1]), "critical_values": {key: float(value) for key, value in result[4].items()}, "stationary": bool(result[1] < .05)}

def forecast_sales(weekly: pd.Series, horizon: int = 12, figure_path: str | Path | None = None) -> dict[str, Any]:
    """Fit ARIMA(1,1,1), score a holdout, and return a future forecast."""
    train, test = weekly.iloc[:-horizon], weekly.iloc[-horizon:]
    model = ARIMA(train, order=(1, 1, 1)).fit()
    prediction = model.get_forecast(steps=horizon)
    forecast, interval = prediction.predicted_mean, prediction.conf_int()
    metrics = {"MAE": float(mean_absolute_error(test, forecast)), "RMSE": float(np.sqrt(mean_squared_error(test, forecast))), "MAPE": float(np.mean(np.abs((test - forecast) / test)) * 100)}
    if figure_path:
        Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(12, 5)); train.plot(ax=axis, label="Train", color="#173f5f"); test.plot(ax=axis, label="Actual", color="#ed553b"); forecast.plot(ax=axis, label="12-week forecast", color="#20639b"); axis.fill_between(interval.index, interval.iloc[:, 0], interval.iloc[:, 1], color="#20639b", alpha=.18, label="95% CI"); axis.set_title("Weekly Sales Forecast"); axis.set_ylabel("Order value"); axis.legend(); fig.tight_layout(); fig.savefig(figure_path, dpi=150); plt.close(fig)
    return {"model": model, "forecast": forecast, "confidence_interval": interval, "metrics": metrics, "train": train, "test": test}

def save_decomposition(weekly: pd.Series, figure_path: str | Path) -> None:
    """Save additive seasonal decomposition using a 52-week period."""
    decomposition = seasonal_decompose(weekly, model="additive", period=52, extrapolate_trend="period")
    fig = decomposition.plot(); fig.set_size_inches(12, 8); fig.tight_layout(); Path(figure_path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(figure_path, dpi=150); plt.close(fig)
