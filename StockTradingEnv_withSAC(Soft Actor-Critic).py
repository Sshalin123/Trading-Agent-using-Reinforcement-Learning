# Key Changes in This Version:
# Action Space Changed:
# Instead of Buy/Sell/Hold (Discrete: {0,1,2}), actions are now continuous values between -1 and 1.
# -1 → Sell all
# 0 → Hold
# 1 → Buy as much as possible
# Fractional Buying & Selling:
# The agent can buy/sell a percentage of its balance/shares instead of all-or-nothing.
# Risk Control:
# Reward penalizes high volatility & large drawdowns









import gym
import numpy as np
import pandas as pd
from gym import spaces

class StockTradingEnv(gym.Env):
    """Stock Trading Environment for Reinforcement Learning using SAC/TD3"""

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        self.max_steps = len(df) - 1

        # Continuous action space: Range between -1 (sell all) to 1 (buy max)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation space includes price, holdings, balance, volatility, etc.
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        """Reset the environment for a new episode"""
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.portfolio_value = self.balance
        self.trades = []
        
        return self._next_observation()

    def _next_observation(self) -> np.ndarray:
        """Get the next market observation"""
        row = self.df.iloc[self.current_step]

        obs = np.array([
            row['Close'],            # Current price
            self.shares_held,        # Shares held
            self.balance,            # Available cash
            self.portfolio_value,    # Portfolio value
            row['Returns'],          # Market return
            row['Daily_Volatility']  # Market risk (volatility)
        ], dtype=np.float32)

        return obs

    def _take_action(self, action: float):
        """Execute trade based on action (-1 to 1)"""
        current_price = self.df.iloc[self.current_step]['Close']
        prev_portfolio_value = self.portfolio_value

        if action > 0:  # Buy shares
            amount_to_spend = action * self.balance  # Buy fractionally
            shares_to_buy = amount_to_spend // current_price
            if shares_to_buy > 0:
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price
                self.trades.append({'step': self.current_step, 'action': 'buy', 'shares': shares_to_buy, 'price': current_price})

        elif action < 0:  # Sell shares
            shares_to_sell = int(abs(action) * self.shares_held)  # Sell fractionally
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                self.trades.append({'step': self.current_step, 'action': 'sell', 'shares': shares_to_sell, 'price': current_price})

        # Update portfolio value
        self.portfolio_value = self.balance + (self.shares_held * current_price)

        # Reward function: Portfolio growth minus a volatility penalty
        portfolio_change = self.portfolio_value - prev_portfolio_value
        risk_penalty = self.df.iloc[self.current_step]['Daily_Volatility'] * 50  # Adjust penalty factor
        reward = portfolio_change - risk_penalty
        
        return reward

    def step(self, action: np.ndarray) -> tuple:
        """Execute a step in the environment"""
        if self.current_step >= self.max_steps:
            return self._next_observation(), 0, True, {}

        action_value = np.clip(action[0], -1, 1)  # Ensure action is within valid range
        reward = self._take_action(action_value)
        self.current_step += 1

        done = self.current_step >= self.max_steps
        obs = self._next_observation()

        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'shares_held': self.shares_held
        }

        return obs, reward, done, info

    def render(self, mode='human'):
        """Print the trading state"""
        current_price = self.df.iloc[self.current_step]['Close']
        print(f'Step: {self.current_step}')
        print(f'Price: {current_price:.2f}')
        print(f'Shares held: {self.shares_held} (Value: {self.shares_held * current_price:.2f})')
        print(f'Balance: {self.balance:.2f}')
        print(f'Portfolio value: {self.portfolio_value:.2f}')
