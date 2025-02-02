from fastapi import FastAPI, Query
from typing import Dict, List
import datetime
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Generate synthetic historical stock data
def generate_historical_data() -> pd.DataFrame:
    now = datetime.datetime.utcnow()
    dates = [now - datetime.timedelta(days=i) for i in range(30)][::-1]
    prices = np.cumsum(np.random.randn(30) + 1) + 100
    return pd.DataFrame({"date": dates, "price": prices})

# Forecast stock prices using ARIMA model
def forecast_arima(historical_data: pd.DataFrame, period: int) -> Dict[str, List]:
    model = ARIMA(historical_data["price"], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=period)
    forecast_dates = [historical_data["date"].iloc[-1] + datetime.timedelta(days=i) for i in range(1, period+1)]
    
    combined_dates = historical_data["date"].tolist() + forecast_dates
    combined_prices = historical_data["price"].tolist() + forecast.tolist()
    
    return {"dates": [date.strftime('%Y-%m-%d') for date in combined_dates], "prices": combined_prices, "historicalLength": len(historical_data)}

# Forecast stock prices using Prophet model
def forecast_prophet(historical_data: pd.DataFrame, period: int) -> Dict[str, List]:
    df = historical_data.rename(columns={"date": "ds", "price": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    return {"dates": forecast["ds"].dt.strftime('%Y-%m-%d').tolist(), "prices": forecast["yhat"].tolist(), "historicalLength": len(historical_data)}

# Forecast stock prices using XGBoost model
def forecast_xgboost(historical_data: pd.DataFrame, period: int) -> Dict[str, List]:
    X = np.arange(len(historical_data)).reshape(-1, 1)
    y = historical_data["price"].values
    model = XGBRegressor()
    model.fit(X, y)
    future_X = np.arange(len(historical_data), len(historical_data) + period).reshape(-1, 1)
    forecast = model.predict(future_X)
    forecast_dates = [historical_data["date"].iloc[-1] + datetime.timedelta(days=i) for i in range(1, period+1)]
    combined_dates = historical_data["date"].tolist() + forecast_dates
    combined_prices = historical_data["price"].tolist() + forecast.tolist()
    return {"dates": [date.strftime('%Y-%m-%d') for date in combined_dates], "prices": combined_prices, "historicalLength": len(historical_data)}

@app.get("/historical")
def get_historical_data():
    historical_data = generate_historical_data()
    return historical_data.to_dict(orient="list")

@app.get("/predict")
def predict_stock(ticker: str = Query("TSLA"), method: str = Query("ARIMA"), period: int = Query(10)):
    historical_data = generate_historical_data()
    
    if method == "ARIMA":
        forecast_data = forecast_arima(historical_data, period)
    elif method == "Prophet":
        forecast_data = forecast_prophet(historical_data, period)
    elif method == "XGBoost":
        forecast_data = forecast_xgboost(historical_data, period)
    else:
        return {"error": "Unsupported forecasting method"}
    
    return {"ticker": ticker, "method": method, "period": period, "data": forecast_data}
