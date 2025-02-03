from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
import pandas as pd
import numpy as np
import datetime
import os
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Enable CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for frontend)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Directory containing the dataset files
DATASET_DIR = "./datasets"

# Load historical data for a specific stock ticker
def generate_historical_data(ticker: str) -> pd.DataFrame:
    file_path = os.path.join(DATASET_DIR, f"{ticker}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data file found for ticker: {ticker}")

    try:
        # Parse the Date column and use Close as the target
        historical_data = pd.read_csv(file_path, parse_dates=["Date"])[["Date", "Close"]]
    except ValueError as e:
        raise ValueError(f"Error reading {ticker}.csv: {e}. Ensure the file contains 'Date' and 'Close' columns.")

    # Ensure the data has the required columns
    if "Date" not in historical_data.columns or "Close" not in historical_data.columns:
        raise ValueError(f"{ticker}.csv must contain 'Date' and 'Close' columns.")

    # Rename 'Close' to 'price' for consistency
    historical_data = historical_data.rename(columns={"Close": "price"})
    return historical_data.sort_values("Date")

# Forecasting with ARIMA
def forecast_arima(historical_data: pd.DataFrame, period: int) -> Dict[str, List]:
    model = ARIMA(historical_data["price"], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=period)

    last_date = historical_data["Date"].iloc[-1]  # Last historical date
    last_price = historical_data["price"].iloc[-1]  # Last historical price

    forecast_dates = [last_date] + [last_date + datetime.timedelta(days=i) for i in range(1, period + 1)]
    forecast_prices = [last_price] + forecast.tolist()

    return {
        "dates": [date.strftime('%Y-%m-%d') for date in forecast_dates],
        "prices": forecast_prices
    }

# Forecasting with Prophet
def forecast_prophet(historical_data: pd.DataFrame, period: int) -> Dict[str, List]:
    df = historical_data.rename(columns={"Date": "ds", "price": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    return {
        "dates": forecast["ds"].dt.strftime('%Y-%m-%d').tolist(),
        "prices": forecast["yhat"].tolist()
    }

# Forecasting with XGBoost
def forecast_xgboost(historical_data: pd.DataFrame, period: int) -> Dict[str, List]:
    X = np.arange(len(historical_data)).reshape(-1, 1)
    y = historical_data["price"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor()
    model.fit(X_scaled, y)

    future_X = scaler.transform(np.arange(len(historical_data), len(historical_data) + period).reshape(-1, 1))
    forecast = model.predict(future_X)

    last_date = historical_data["Date"].iloc[-1]
    forecast_dates = [last_date + datetime.timedelta(days=i) for i in range(1, period + 1)]

    return {
        "dates": [date.strftime('%Y-%m-%d') for date in forecast_dates],
        "prices": forecast.tolist()
    }

# API endpoint to fetch historical data
@app.get("/historical")
def get_historical_data(ticker: str = Query("TSLA")):
    historical_data = generate_historical_data(ticker)
    return {
        "dates": historical_data["Date"].dt.strftime('%Y-%m-%d').tolist(),
        "prices": historical_data["price"].tolist()
    }

# API endpoint to predict stock prices
@app.get("/predict")
def predict_stock(ticker: str = Query("TSLA"), method: str = Query("ARIMA"), period: int = Query(10)):
    historical_data = generate_historical_data(ticker)

    if method == "ARIMA":
        forecast_data = forecast_arima(historical_data, period)
    elif method == "Prophet":
        forecast_data = forecast_prophet(historical_data, period)
    elif method == "XGBoost":
        forecast_data = forecast_xgboost(historical_data, period)
    else:
        return {"error": "Unsupported forecasting method"}

    combined_dates = historical_data["Date"].dt.strftime('%Y-%m-%d').tolist() + forecast_data["dates"][1:]
    combined_prices = historical_data["price"].tolist() + forecast_data["prices"][1:]

    return {
        "ticker": ticker,
        "method": method,
        "period": period,
        "dates": combined_dates,
        "prices": combined_prices,
        "historicalLength": len(historical_data)
    }
