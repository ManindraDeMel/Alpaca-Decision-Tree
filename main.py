from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from trading_env import TradingEnv
import os

# Load environment variables
load_dotenv()
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
SYMBOLS = ['AAPL', 'MSFT']

# Initialize the environment
env = TradingEnv(ALPACA_API_KEY, ALPACA_SECRET_KEY, SYMBOLS)

# Initialize the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent for 10000 steps
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_trading_bot")

# Load the model
model = PPO.load("ppo_trading_bot")

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")
