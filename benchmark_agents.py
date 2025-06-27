import numpy as np

class TWAPAgent:
    """A simple agent that implements the TWAP strategy."""
    def get_action(self, env):
        twap_quantity_per_step = env.initial_inventory / env.trade_horizon
        if env.remaining_inventory < twap_quantity_per_step:
            action = 1.0
        else:
            action = twap_quantity_per_step / env.remaining_inventory
        return np.array([action], dtype=np.float32)

class VWAPAgent:
    """An agent that implements a VWAP strategy based on a historical profile."""
    def __init__(self, df):
        self.vwap_profile = self._calculate_vwap_profile(df)

    def _calculate_vwap_profile(self, df):
        df['minute_of_day'] = df.index.hour * 60 + df.index.minute
        return df.groupby('minute_of_day')['volume'].mean()

    def get_action(self, env):
        current_tick = env.start_tick + env.current_step
        current_minute_of_day = env.df.index[current_tick].hour * 60 + env.df.index[current_tick].minute
        
        try:
            current_profile_volume = self.vwap_profile.loc[current_minute_of_day]
        except KeyError:
            current_profile_volume = self.vwap_profile.mean()

        total_profile_volume = 0
        for i in range(env.trade_horizon):
            tick = env.start_tick + i
            minute_of_day = env.df.index[tick].hour * 60 + env.df.index[tick].minute
            try:
                total_profile_volume += self.vwap_profile.loc[minute_of_day]
            except KeyError:
                total_profile_volume += self.vwap_profile.mean()
        
        if total_profile_volume > 0:
            proportion_to_sell = current_profile_volume / total_profile_volume
        else:
            proportion_to_sell = 1.0 / (env.trade_horizon - env.current_step) if env.trade_horizon > env.current_step else 1.0
            
        vwap_quantity_this_step = env.initial_inventory * proportion_to_sell

        if env.remaining_inventory < vwap_quantity_this_step:
            action = 1.0
        else:
            action = vwap_quantity_this_step / env.remaining_inventory
            
        return np.array([action], dtype=np.float32)