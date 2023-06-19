# README for Stock Trading Bot

## Overview
This script creates a simple stock trading bot using Python and the Alpaca API. The bot makes predictions on the closing price of a specific stock (currently hard-coded to Apple Inc. ('AAPL')), using the past 120 days of trading data. The bot then makes a decision to buy, sell, or hold the stock based on these predictions. The decision-making process considers the current balance of the account and whether or not the account currently owns the stock. 

All predictions, actual prices, and actions (buy, sell, or hold) are stored in a CSV file for future analysis. 

This bot also has an interactive dashboard built with Dash, where it shows a live graph of the predicted price and actual price, and also displays the most recent action taken by the bot.

## Dependencies
- Python 3.7+
- pandas
- scikit-learn
- alpaca-trade-api
- python-dotenv
- dash

You can install these dependencies using pip:
```
pip install pandas scikit-learn alpaca-trade-api python-dotenv dash
```

## How to use

1. Clone this repository and navigate to the directory in your terminal.
2. Install all necessary dependencies.
3. Create a .env file in your project root and add your Alpaca API key and secret key to it:
    ```
    ALPACA_API_KEY="your_api_key"
    ALPACA_SECRET_KEY="your_secret_key"
    ```
4. To start the trading bot, run `python main.py`.
5. To view the trading dashboard, run `python app.py` and navigate to `localhost:8050` in your web browser.

## Code Structure

- `fetch_data(symbol, days)`: This function fetches the past `days` of trading data for the given `symbol` using the Alpaca API. The data is returned as a pandas DataFrame.
  
- `train_model(X_train, y_train)`: This function trains a random forest regression model using the provided training data.
  
- `predict_and_trade(model, X_last, last_close, symbol)`: This function makes a prediction using the provided model and the most recent data, then decides whether to trade the stock based on this prediction.
  
- `main()`: This function starts the bot, fetching the necessary data, training the model, making predictions, and deciding on trades in a continuous loop.
  
- `app.py`: This script creates an interactive dashboard using Dash, which shows the predicted and actual stock prices over time and displays the most recent action taken by the bot.

Please note that the helper functions are defined in the `util.py` file.

Please remember that this bot is for educational purposes only and should not be used for actual trading without thorough testing and modification. Trading stocks always comes with risk, and automated trading strategies can result in substantial losses.
