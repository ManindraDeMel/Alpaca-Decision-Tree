import pandas as pd
import numpy as np

def preprocess_data(data):
    #data preprocessing logic here.
    data.fillna(0, inplace=True)
    data.replace([np.inf, -np.inf], 0, inplace=True)
    data = data.pct_change()

    return data

def calculate_technical_indicators(data, window=14):
    # Calculate technical indicators, such as RSI, MACD, etc.

    # RSI:
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    average_gain = up.rolling(window).mean()
    average_loss = abs(down.rolling(window).mean())

    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi
