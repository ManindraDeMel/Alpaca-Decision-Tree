import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from pandas_datareader import data as pdr
import yfinance as yf
import alpaca_trade_api as tradeapi

yf.pdr_override()

# Load environment variables from .env file
load_dotenv()

# Initialize Alpaca API
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets') 


# Fetch data from 2018-01-01 to 2023-05-01
df = pdr.get_data_yahoo("AAPL", start="2018-01-01", end="2023-05-01")

# Use shift to create the past 5 days' closing price columns
for i in range(1, 6):
    df[f'Close_{i}'] = df['Close'].shift(i)

# Drop the rows for which we don't have past 5 days' data
df = df.dropna()

# Predict the next day's closing price, so the target column will be the 'Close' column shifted up by 1
df['Target'] = df['Close'].shift(-1)

# Drop the rows for which we don't have target value
df = df.dropna()

# Create the feature matrix (X) and target array (y)
X = df.drop('Target', axis=1)
y = df['Target']

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict the target values for test set
y_pred = model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')

# Use the model to predict the next day's closing price
next_day_close = model.predict([X.iloc[-1]])

# Place an order based on the prediction
if next_day_close > df['Close'].iloc[-1]:
    api.submit_order(
        symbol='AAPL',
        qty=1,
        side='buy',
        type='market',
        time_in_force='gtc'
    )
else:
    api.submit_order(
        symbol='AAPL',
        qty=1,
        side='sell',
        type='market',
        time_in_force='gtc'
    )
