import os
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import alpaca_trade_api as tradeapi
from alpaca_trade_api import TimeFrame
import pytz
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Load environment variables from .env file
load_dotenv()

# Initialize Alpaca API
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets') 

def fetch_data(symbol, days):
    # Get the current datetime in UTC
    now = datetime.now(timezone.utc)

    # Subtract 15 minutes from the current datetime
    end_date = now - timedelta(minutes=20)

    # Calculate the start date based on the end date and the number of days
    start_date = end_date - timedelta(days=days)

    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()


    data = api.get_bars(symbol, TimeFrame.Day, start=start_date_str, end=end_date_str, adjustment='raw').df

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

    action = ""
    if next_day_close > last_close:
        print(f"Buy: {symbol}")
        api.submit_order(
            symbol=symbol,
            qty=1,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
        action = "buy"
    else:
        print(f"Sell: {symbol}")
        api.submit_order(
            symbol=symbol,
            qty=1,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
        action = "sell"

    # Write to CSV file
    data = {
        'timestamp': [datetime.now().isoformat()],
        'symbol': [symbol],
        'predicted_price': [next_day_close[0]],  # Unpack from array
        'actual_price': [last_close],
        'action': [action]
    }

    df = pd.DataFrame(data)

    # If file does not exist, write with header
    if not os.path.isfile('trading_data.csv'):
        df.to_csv('trading_data.csv', index=False)
    else:  # else it exists so append without writing the header
        df.to_csv('trading_data.csv', mode='a', header=False, index=False)

def is_market_open():
    # Get the current time in Eastern Time
    now = datetime.now(pytz.timezone('US/Eastern'))

    # Define market open and close hours
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    # Check if the current time is within market hours
    return market_open <= now <= market_close

def plot_data(df):
    plt.plot(df['Close'])
    plt.title('Closing Prices Over Time')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

def main():
    print("Starting bot")
    symbol = 'AAPL'
    days = 120
    test_size = 0.2
    wait_time = 20  # seconds to wait (1 min)
    while True:        
        # print("\nChecking if market is open\n\n")
        # if is_market_open():
        # print("Market is open!")
        df = fetch_data(symbol, days)
        X = df.drop('Target', axis=1)
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = train_model(X_train, y_train)

        # Continue with model evaluation
        score = model.score(X_test, y_test)
        print(f"Model Score: {score}\n")

        # Get the last row of data (most recent)
        X_last = df.iloc[-1].drop('Target')

        # Get the most recent closing price
        last_close = df.iloc[-1]['Close']

        # Make a prediction and decide whether to trade
        predict_and_trade(model, X_last, last_close, symbol)
        # else:
        #     print("\nMarket is not open")
        # Wait for the defined period
        time.sleep(wait_time)
        
if __name__ == "__main__":
    main()
