class StockTradingEnv(gym.Env):
    """Custom share trading env for RL"""
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        self.max_steps = len(df) - 1
        
        # Actions: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: current price, portfolio value, shares held, etc.
        self.observation_space = spaces.Box(
            low=0, high=np.inf, 
            shape=(6,),  # price, shares, cash, portfolio_value, returns, volatility
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset the env to initial state"""
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.portfolio_value = self.balance
        self.trades = []
        
        return self._next_observation()
    
    def _next_observation(self) -> np.ndarray:
        """Get the next observation from the env"""
        row = self.df.iloc[self.current_step]
        
        obs = np.array([
            row['Close'],  # current price of share
            self.shares_held,  # shares held
            self.balance,  # Current cash
            self.portfolio_value,  # total portfolio value
            row['Returns'],  # daily returns
            row['Daily_Volatility']  # daily volatility
        ])
        
        return obs
    
    def _take_action(self, action: int):
        """Execute the trade action"""
        current_price = self.df.iloc[self.current_step]['Close']
        action_type = action
        
        # Calculate portfolio value before taking action
        prev_portfolio_value = self.portfolio_value
        
        if action_type == 1:  # buy action
            max_shares = self.balance // current_price
            if max_shares > 0:
                self.shares_held += max_shares
                self.balance -= max_shares * current_price
                self.trades.append({'step': self.current_step, 'action': 'buy', 'shares': max_shares, 'price': current_price})
            # Buy as much as we can with current balance
        
        elif action_type == 2:  # sell action
            # Sell all shares
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.trades.append({'step': self.current_step, 'action': 'sell', 'shares': self.shares_held, 'price': current_price})
                self.shares_held = 0
        
        # Updation of portfolio value
        self.portfolio_value = self.balance + self.shares_held * current_price
        
        # calculating reward based on portfolio change
        reward = self.portfolio_value - prev_portfolio_value
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """execute one step in the env"""
        if self.current_step >= self.max_steps:
            return self._next_observation(), 0, True, {}
        
        reward = self._take_action(action)
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
        """render the env state"""
        portfolio_value = self.portfolio_value
        balance = self.balance
        shares_held = self.shares_held
        current_price = self.df.iloc[self.current_step]['Close']
        
        print(f'Step: {self.current_step}')
        print(f'Price: {current_price:.2f}')
        print(f'Shares held: {shares_held} (Value: {shares_held * current_price:.2f})')
        print(f'Balance: {balance:.2f}')
        print(f'Portfolio value: {portfolio_value:.2f}')
