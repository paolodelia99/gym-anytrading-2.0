from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .trading_env import Actions, Positions
from .trading_env import TradingEnv


class FuturesEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, pos_size: float = 0.05,
                 risk_per_contract=1_000,
                 point_value=1_000,
                 initial_capital=1_000_000,
                 spread: float = 1.00):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.close_prices = df.close.values
        self.open_prices = df.open.values
        df = self._preprocess_df(df)
        super().__init__(df, window_size, initial_capital)

        self.close_idx = int(np.where(self.sc.feature_names_in_ == 'close')[0][0])
        self.open_idx = int(np.where(self.sc.feature_names_in_ == 'open')[0][0])
        self.pos_size = pos_size
        self.initial_capital = initial_capital
        self.risk_per_contract = risk_per_contract
        self.point_value = point_value
        self.spread = spread
        self.account_values = [self.initial_capital]
        self.long_ticks = []
        self.short_ticks = []
        self.close_ticks = []
        self.n_contracts = 0

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._total_profit = self.initial_capital
        self.long_ticks = []
        self.short_ticks = []
        self.close_ticks = []
        self.n_contracts = 0
        if return_info:
            return self._get_observation(), dict(
                total_reward=self._total_reward,
                total_profit=0,
                position=self._position.value
            )
        else:
            return self._get_observation()

    def _check_if_close_trade(self, action):
        if self._position == Positions.Short or self._position == Positions.Long:
            return action == Actions.Close.value
        else:
            return False

    def _check_if_open_trade(self, action):
        if self._position == Positions.NoPosition:
            return action == Actions.Sell.value or action == Actions.Buy.value

        return False

    def step(self, action: int):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._set_position(action, self._current_tick)

        self._update_position()
        self._action_history.append(action)
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position.value,
            action=action
        )
        self._update_history(info)

        if self._total_profit < 0:
            self._done = True

        return observation, step_reward, self._done, info

    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        prices = self.df.loc[:, 'close'].to_numpy()[start:end]
        df_ = self.df.copy(deep=True)
        signal_features = df_.to_numpy(dtype=np.float32)[start:end]
        return prices, signal_features

    def _preprocess_df(self, df):
        self.sc = MinMaxScaler(feature_range=(-1, 1))
        self.df_index = df.index
        df = pd.DataFrame(self.sc.fit_transform(df), columns=df.columns, index=df.index)
        df = FuturesEnv._add_position_states(df)
        self.df_cols = df.columns
        return df

    @staticmethod
    def _add_position_states(df):
        df = df.assign(short=np.zeros(len(df)))
        df = df.assign(no_pos=np.ones(len(df)))
        df = df.assign(long=np.zeros(len(df)))
        return df

    def _calculate_reward(self, action: int):
        action_ = action - 1
        prev_account_value = self.account_values[-1]

        if self.n_contracts == 0 and action_ != 0:
            self.n_contracts = int(np.floor((self._total_profit * self.pos_size) / self.risk_per_contract))
            c = self.n_contracts * 10
        else:
            c = self.n_contracts * 10

        d_returns = ((self.close_prices[self._current_tick] / self.open_prices[self._current_tick]) - 1)
        commission = c * np.abs(action_ - (self._action_history[-1] - 1)) * self.spread
        current_account_value = prev_account_value + action_ * c * d_returns - commission
        r = np.log(current_account_value / prev_account_value)
        self.account_values.append(current_account_value)
        self._total_profit = current_account_value

        return r

    def get_account_value(self):
        return self._total_profit

    def max_possible_profit(self):
        pass

    def _set_position(self, action, current_tick):
        if action == Actions.Buy.value:
            self._position = Positions.Long
            self.long_ticks.append(current_tick)
        elif action == Actions.Sell.value:
            self._position = Positions.Short
            self.short_ticks.append(current_tick)
        elif action == Actions.Close.value:
            if self._position == Positions.Short or self._position == Positions.Long:
                self.close_ticks.append(current_tick)
                self.n_contracts = 0
                self._position = Positions.NoPosition

    def render_all(self, mode='human'):
        plt.plot(self.close_prices)

        plt.plot(self.short_ticks, self.close_prices[self.short_ticks], 'ro', label='Long')
        plt.plot(self.long_ticks, self.close_prices[self.long_ticks], 'go', label='short')
        plt.plot(self.close_ticks, self.close_prices[self.close_ticks], 'bo', label='close')

        plt.legend()
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def _update_position(self):
        self._position_history.append(self._position)

        last_col_idx = self.signal_features.shape[1]

        if self._current_tick != self._end_tick:
            if self._position == Positions.Long:
                self.signal_features[self._current_tick, last_col_idx - 3:last_col_idx] = np.array([1, 0, 0])
            elif self._position == Positions.Short:
                self.signal_features[self._current_tick, last_col_idx - 3:last_col_idx] = np.array([0, 0, 1])
            elif self._position == Positions.NoPosition:
                self.signal_features[self._current_tick, last_col_idx - 3:last_col_idx] = np.array([0, 1, 0])
