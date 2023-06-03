import os
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
import alpaca_trade_api as tradeapi
from alpaca_trade_api import TimeFrame
from util import get_balance, decide_quantity, own_stock
import warnings
from train import train_model
warnings.filterwarnings('ignore')


# Load environment variables from .env file
load_dotenv()

# Initialize Alpaca API
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets') 

# Define the threshold for price changes (e.g. 1% change)
THRESHOLD = 0.01

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

def predict_and_trade(model, X_last, last_close, symbol):
    # Model Prediction
    next_day_close = model.predict([X_last])
    # next_day_close = [last_close * 0.9]
    print(f'next_day_close: {next_day_close}, last_close: {last_close}, buy_threshold: {last_close * (1 + THRESHOLD)}, sell_threshold: {last_close * (1 - THRESHOLD)}')

    # Balance Check
    cash = get_balance(api)

    # Decide the quantity for transaction
    quantity = decide_quantity(cash, last_close)
    
    action = ""
    if next_day_close > last_close * (1 + THRESHOLD):
        if cash > last_close * quantity and not own_stock(symbol, api):
            print(f"Buy {quantity} {symbol}")
            api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type='limit',
                time_in_force='day',
                extended_hours=True,
                limit_price=round(next_day_close[0], 2)
            )
            action = "buy"
        else:
            print("Holding (Insufficent funds)")
            action = "hold"
    elif next_day_close < last_close * (1 - THRESHOLD):
        if own_stock(symbol, api):
            print(f"Sell {quantity} {symbol}")
            api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type='limit',
                time_in_force='gtc',
                limit_price=round(last_close * (1 + THRESHOLD), 2)
            )
            action = "sell"
        else:
            print("No shares to sell")
    else:
        print("Hold")
        action = "hold"
        
    # Write prediction, actual price, and action to CSV file
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

def main():
    print("Starting bot\n################")
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
