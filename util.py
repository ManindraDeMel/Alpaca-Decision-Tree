from datetime import datetime
import pytz
def get_balance(api):
    # Get account information
    account = api.get_account()

    # Get the current cash balance
    cash = float(account.buying_power)

    return cash

def decide_quantity(cash, last_close):
    # We will use only 90% of cash for each transaction to avoid insufficient buying power error due to price fluctuations
    cash_for_this_transaction = cash * 0.9
    quantity = int(cash_for_this_transaction / last_close)

    return quantity

def is_market_open():
    # Get the current time in Eastern Time
    now = datetime.now(pytz.timezone('US/Eastern'))

    # Define market open and close hours
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    # Check if the current time is within market hours
    return market_open <= now <= market_close

def own_stock(symbol, api):
    positions = api.list_positions()
    for position in positions:
        if position.symbol == symbol:
            return True
    return False