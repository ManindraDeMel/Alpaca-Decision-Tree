import os
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import alpaca_trade_api as tradeapi

# Load environment variables from .env file
load_dotenv()

# Initialize Alpaca API
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets') 

def fetch_data(symbol, days):
    data = api.get_barset(symbol, 'day', limit=days).df[symbol]

    df = pd.DataFrame()
    df['Open'] = data['open']
    df['High'] = data['high']
    df['Low'] = data['low']
    df['Close'] = data['close']

    for i in range(1, 6):
        df[f'Close_{i}'] = df['Close'].shift(i)

    df = df.dropna()

    df.loc[:, 'Target'] = df['Close'].shift(-1)
    df = df.dropna()

    return df

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_and_trade(model, X_last, last_close, symbol):
    next_day_close = model.predict([X_last])

    if next_day_close > last_close:
        api.submit_order(
            symbol=symbol,
            qty=1,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
    else:
        api.submit_order(
            symbol=symbol,
            qty=1,
            side='sell',
            type='market',
            time_in_force='gtc'
        )

def is_market_open():
    # Get the current time in Eastern Time
    now = datetime.now().astimezone(timezone('US/Eastern'))

    # Define market open and close hours
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    # Check if the current time is within market hours
    return market_open <= now <= market_close

def main():
    symbol = 'AAPL'
    days = 60
    test_size = 0.2
    wait_time = 60  # seconds to wait (1 min)

    while True:
        if is_market_open():
            df = fetch_data(symbol, days)
            X = df.drop('Target', axis=1)
            y = df['Target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model = train_model(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)

            print(f'Mean Absolute Error: {mae}')

            X_last = X.iloc[-1]
            last_close = df['Close'].iloc[-1]
            predict_and_trade(model, X_last, last_close, symbol)

        time.sleep(wait_time)

if __name__ == "__main__":
    main()
