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
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from typing import Any, Dict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input  # Import Input
from tensorflow.keras import regularizers  # For kernel regularization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
app = FastAPI()

# Enable CORS to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory containing the dataset files
DATASET_DIR = "./datasets"

# Load historical data for a specific stock ticker
def generate_historical_data(ticker: str) -> pd.DataFrame:
    file_path = os.path.join(DATASET_DIR, f"{ticker}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data file found for ticker: {ticker}")
    
    historical_data = pd.read_csv(file_path, parse_dates=["Date"])[["Date", "Close"]]
    historical_data = historical_data.rename(columns={"Close": "price"})
    return historical_data.sort_values("Date")



def forecast_hybrid_lstm_gru(historical_data: pd.DataFrame, period: int) -> Dict[str, List]:
    """
    Forecast stock prices using a hybrid LSTM-GRU model.

    Args:
    - historical_data: DataFrame with historical stock prices
    - period: Number of days to forecast

    Returns:
    - Dictionary containing forecasted dates and prices
    """
    # Extract price data and normalize
    data = historical_data[['price']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Sequence length for input to the model
    sequence_length = 60

    # Create sequences
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length - 1):
            x.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(x), np.array(y)

    # Split data into training (80%) and testing (20%)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Further split test data into validation (50%) and final test (50%)
    validation_size = int(len(test_data) * 0.5)
    validation_data = test_data[:validation_size]
    final_test_data = test_data[validation_size:]

    # Create sequences
    x_train, y_train = create_sequences(train_data, sequence_length)
    x_validation, y_validation = create_sequences(validation_data, sequence_length)

    # Reshape for input into LSTM/GRU
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_validation = x_validation.reshape((x_validation.shape[0], x_validation.shape[1], 1))

    # Define the hybrid model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        GRU(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)  # Predict the next close price
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

    # Configure early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Train the model
    model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(x_validation, y_validation),
        callbacks=[early_stopping],
        verbose=1
    )

    # Forecast future prices
    forecast_input = scaled_data[-sequence_length:]  # Last 60 days of data
    forecast_input = forecast_input.reshape((1, sequence_length, 1))

    forecasted_prices = []
    for _ in range(period):
        # Predict the next step
        forecast_price = model.predict(forecast_input)[0, 0]
        forecasted_prices.append(forecast_price)

        # Update the input with the predicted value
        forecast_input = np.roll(forecast_input, -1, axis=1)
        forecast_input[0, -1, 0] = forecast_price

    # Scale back the predictions to the original range
    forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1)).flatten()

    # Generate forecast dates
    last_date = historical_data["Date"].iloc[-1]
    forecast_dates = [last_date + datetime.timedelta(days=i) for i in range(1, period + 1)]

    return {
        "dates": [date.strftime('%Y-%m-%d') for date in forecast_dates],
        "prices": forecasted_prices
    }

# Forecasting with LSTM
def forecast_lstm(historical_data: pd.DataFrame, period: int) -> Dict[str, List]:
    """
    Forecast stock prices using an LSTM model.

    Args:
    - historical_data: DataFrame with historical stock prices
    - period: Number of days to forecast

    Returns:
    - Dictionary containing forecasted dates and prices
    """
    # Extract the price column and normalize it
    data = historical_data[['price']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Define the sequence length
    sequence_length = 10
    X_train, y_train = [], []

    # Create the sequences for training
    for i in range(len(scaled_data) - sequence_length):
        X_train.append(scaled_data[i:i + sequence_length])
        y_train.append(scaled_data[i + sequence_length])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Initialize the LSTM model
    regressor = Sequential()

    # Input Layer
    regressor.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

    # First LSTM layer with dropout
    regressor.add(LSTM(units=100, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))
    regressor.add(Dropout(rate=0.2))

    # Second LSTM layer with dropout
    regressor.add(LSTM(units=100, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))
    regressor.add(Dropout(rate=0.2))

    # Third LSTM layer with dropout
    regressor.add(LSTM(units=100, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))
    regressor.add(Dropout(rate=0.2))

    # Fourth LSTM layer with dropout
    regressor.add(LSTM(units=100, kernel_regularizer=regularizers.l2(0.001)))
    regressor.add(Dropout(rate=0.2))

    # Output layer
    regressor.add(Dense(units=1))

    # Compile the model
    regressor.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    regressor.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    # Forecast future prices
    predictions = []
    last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    last_date = historical_data['Date'].iloc[-1]

    for _ in range(period):
        next_pred = regressor.predict(last_sequence)[0][0]
        predictions.append(next_pred)
        next_pred_reshaped = np.array(next_pred).reshape(1, 1, 1)
        next_input = np.append(last_sequence[:, 1:, :], next_pred_reshaped, axis=1)
        last_sequence = next_input.reshape(1, sequence_length, 1)

    # Scale the predictions back to the original price range
    forecast_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()

    # Generate forecast dates
    forecast_dates = [last_date + datetime.timedelta(days=i) for i in range(1, period + 1)]

    return {
        "dates": [date.strftime('%Y-%m-%d') for date in forecast_dates],
        "prices": forecast_prices
    }

# Forecasting with GRU
def forecast_gru(historical_data: pd.DataFrame, period: int) -> Dict[str, List]:
    data = historical_data[['price']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    sequence_length = 10
    X_train, y_train = [], []
    for i in range(len(scaled_data) - sequence_length):
        X_train.append(scaled_data[i:i+sequence_length])
        y_train.append(scaled_data[i+sequence_length])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    
    predictions = []
    last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    last_date = historical_data['Date'].iloc[-1]
    
    for _ in range(period):
        next_pred = model.predict(last_sequence)[0][0]
        predictions.append(next_pred)
        next_pred_reshaped = np.array(next_pred).reshape(1, 1, 1)
        next_input = np.append(last_sequence[:, 1:, :], next_pred_reshaped, axis=1)
        last_sequence = next_input.reshape(1, sequence_length, 1)
    
    forecast_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()
    forecast_dates = [last_date + datetime.timedelta(days=i) for i in range(1, period + 1)]
    
    return {
        "dates": [date.strftime('%Y-%m-%d') for date in forecast_dates],
        "prices": forecast_prices
    }

# Forecasting with ARIMA
def forecast_arima(historical_data: pd.DataFrame, period: int) -> Dict[str, List]:
    model = ARIMA(historical_data["price"], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=period)
    
    last_date = historical_data["Date"].iloc[-1]
    forecast_dates = [last_date + datetime.timedelta(days=i) for i in range(1, period + 1)]
    
    return {
        "dates": [date.strftime('%Y-%m-%d') for date in forecast_dates],
        "prices": forecast.tolist()
    }

# Forecasting with XGBoost
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def forecast_xgboost(historical_data: pd.DataFrame, period: int) -> Dict[str, List]:
    """
    Forecast stock prices using the XGBoost model with performance metrics.

    Args:
    - historical_data: DataFrame with historical stock prices
    - period: Number of days to forecast

    Returns:
    - Dictionary containing forecasted dates, prices, and evaluation metrics
    """
    # Prepare the data
    data = historical_data[["price"]].values
    lag = 10  # Number of lag features for time series forecasting

    # Create lagged features for supervised learning
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i].flatten())
        y.append(data[i][0])  # Use the price value
    X, y = np.array(X), np.array(y)

    # Train/Test Split
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    # Initialize and train the XGBoost model
    model = XGBRegressor(
        n_estimators=225,
        max_depth=7,
        learning_rate=0.17,
        min_child_weight=1,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        gamma=0,
        random_state=100
    )
    model.fit(X_train, y_train)

    # Evaluate on the test set
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = 1 - np.sum((predictions - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    # F1 Score approximation (using MAE and RMSE)
    f1 = 2 * (mae * rmse) / (mae + rmse) if (mae + rmse) != 0 else 0

    # Forecast future prices
    last_sequence = data[-lag:].flatten()
    forecast_prices = []
    for _ in range(period):
        next_pred = model.predict(last_sequence.reshape(1, -1))[0]
        forecast_prices.append(next_pred)
        last_sequence = np.append(last_sequence[1:], next_pred)

    # Generate forecast dates
    last_date = historical_data["Date"].iloc[-1]
    forecast_dates = [last_date + datetime.timedelta(days=i) for i in range(1, period + 1)]

    return {
        "dates": [date.strftime('%Y-%m-%d') for date in forecast_dates],
        "prices": forecast_prices,
    }



def forecast_prophet(historical_data: pd.DataFrame, period: int) -> Dict[str, Any]:
    """
    Forecast stock prices using the Prophet model.
    Ensures the forecast starts seamlessly from the last historical value.

    Args:
    - historical_data: DataFrame with historical stock prices
    - period: Number of days to forecast

    Returns:
    - Dictionary containing forecasted dates, prices, and historical length
    """
    # Rename columns for Prophet
    df = historical_data.rename(columns={"Date": "ds", "price": "y"})

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df)

    # Create a dataframe for future predictions
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    # Filter only the forecast data beyond the last historical date
    forecasted_data = forecast[forecast['ds'] > df['ds'].max()]

    # Ensure continuity by appending the last historical value
    last_historical_value = df['y'].iloc[-1]
    forecasted_data.loc[forecasted_data.index[0], 'yhat'] = last_historical_value

    return {
        "dates": forecasted_data["ds"].dt.strftime('%Y-%m-%d').tolist(),
        "prices": forecasted_data["yhat"].tolist(),
        "historicalLength": len(df)  # Length of the historical data
    }
# API endpoint to predict stock prices
@app.get("/predict")
def predict_stock(
    ticker: str = Query(..., description="Ticker symbol of the stock"),
    method: str = Query("ARIMA", description="Forecasting method"),
    period: int = Query(1, description="Number of days to forecast")
):
    available_tickers = [file.split(".")[0] for file in os.listdir(DATASET_DIR) if file.endswith(".csv")]
    if ticker not in available_tickers:
        return {"error": f"Ticker {ticker} not found. Available tickers: {', '.join(available_tickers)}"}

    historical_data = generate_historical_data(ticker)

    try:
        if method == "ARIMA":
            forecast_data = forecast_arima(historical_data, period)
        elif method == "Prophet":
            forecast_data = forecast_prophet(historical_data, period)
        elif method == "LSTM":
            forecast_data = forecast_lstm(historical_data, period)
        elif method == "GRU":
            forecast_data = forecast_gru(historical_data, period)
        elif method == "XGBoost":
            forecast_data = forecast_xgboost(historical_data, period)
        elif method == "Hybrid":
            forecast_data = forecast_hybrid_lstm_gru(historical_data, period)
        else:
            return {"error": "Unsupported forecasting method"}
    except Exception as e:
        return {"error": str(e)}

    # Ensure all data is serializable
    combined_dates = historical_data["Date"].dt.strftime('%Y-%m-%d').tolist() + forecast_data["dates"]
    combined_prices = [float(price) for price in historical_data["price"].tolist()] + [
        float(price) for price in forecast_data["prices"]
    ]

    return {
        "ticker": ticker,
        "method": method,
        "period": period,
        "dates": combined_dates,
        "prices": combined_prices,
        "historicalLength": len(historical_data)
    }
