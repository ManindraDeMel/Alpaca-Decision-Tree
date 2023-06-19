import gym
from gym import spaces
import numpy as np
import alpaca_trade_api as tradeapi
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, alpaca_api_key, alpaca_secret_key, symbols):
        super(TradingEnv, self).__init__()

        # Connect to Alpaca API
        self.api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, base_url='https://paper-api.alpaca.markets') 

        self.symbols = symbols
        self.n_symbols = len(symbols)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_symbols,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_symbols,))

        # Initialize state
        self.current_step = 0
        self.get_data()

    def get_data(self):
        self.data = []
        for symbol in self.symbols:
            # Fetch data
            bars = self.api.get_barset(symbol, 'day', limit=100).df[symbol]
            self.data.append(bars)
        self.data = pd.concat(self.data, axis=1)

    def step(self, action):
        self.current_step += 1

        # Execute trading action
        for i, action in enumerate(action):
            if action > 0:  # Buy action
                self.api.submit_order(
                    symbol=self.symbols[i],
                    qty=1,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
            elif action < 0:  # Sell action
                self.api.submit_order(
                    symbol=self.symbols[i],
                    qty=1,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )

        # Calculate reward
        # This is a simplified example where the reward is the change in portfolio value
        # In a real-world application, you might want a more sophisticated reward function
        portfolio = self.api.get_account()
        reward = float(portfolio.portfolio_value) - float(portfolio.last_equity)

        # Get new observation data
        self.get_data()
        observation = self.data.iloc[self.current_step].values

        done = self.current_step >= len(self.data) - 1

        return observation, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.get_data()
        observation = self.data.iloc[self.current_step].values

        return observation

    def render(self):
        # In this simple example we're not implementing a renderer
        pass
