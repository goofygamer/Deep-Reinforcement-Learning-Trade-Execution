import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np


class ExecutionEnv(gym.Env):
    """
    A custom Gymnasium environment for simulating the optimal execution of a
    large order in a market represented by 1-minute OHLCV bar data.

    The agent's goal is to sell a large initial inventory of an asset
    over a fixed time horizon.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_inventory=50.0, trade_horizon=60, lookback_window=30):
        """
        Args:
            df (pd.DataFrame): DataFrame with 1-min OHLCV data. Must include
                               'open', 'high', 'low', 'close', 'volume', 'vwap'.
            initial_inventory (float): The total quantity of the asset to be sold.
            trade_horizon (int): The number of minutes (steps) over which to sell.
            lookback_window (int): Number of past steps to use for market features.
        """
        super(ExecutionEnv, self).__init__()

        self.df                 = df
        self.initial_inventory  = initial_inventory
        self.trade_horizon      = trade_horizon
        self.lookback_window    = lookback_window

        