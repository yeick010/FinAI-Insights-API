from fastapi import FastAPI, HTTPException
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

app = FastAPI()

# Modello AI semplice per previsioni finanziarie
def train_model(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(data_scaled) - 10):
        X.append(data_scaled[i:i+10])
        y.append(data_scaled[i+10])
    
    X, y = np.array(X), np.array(y)
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(10, data.shape[1])),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(data.shape[1])
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=8, verbose=1)
    return model, scaler

@app.get("/predict/{ticker}")
def predict_stock(ticker: str):
    try:
        df = yf.download(ticker, period="1y", interval="1d")
        if df.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")
        
        data = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        model, scaler = train_model(data.values)
        
        last_10_days = data.values[-10:]
        last_10_days_scaled = scaler.transform(last_10_days)
        
        pred = model.predict(np.array([last_10_days_scaled]))
        pred_rescaled = scaler.inverse_transform(pred)
        
        return {"prediction": list(map(list, pred_rescaled))}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/{ticker}")
def get_market_data(ticker: str):
    try:
        df = yf.download(ticker, period="1mo", interval="1d")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().to_dict()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Welcome to FinAI Insights API! Use /predict/{ticker} for AI predictions"}
