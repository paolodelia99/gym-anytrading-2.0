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
                 trade_on_close: bool = True):
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
        self.long_ticks = []
        self.short_ticks = []
        self.trade_on_close = trade_on_close

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._total_profit = self.initial_capital
        self.long_ticks = []
        self.short_ticks = []
        if return_info:
            return self._get_observation(), dict(
                total_reward=self._total_reward,
                total_profit=0,
                position=self._position.value
            )
        else:
            return self._get_observation()

    def _check_if_close_trade(self, action):
        if self._position == Positions.Short:
            return action == Actions.Buy.value
        elif self._position == Positions.Long:
            return action == Actions.Sell.value
        else:
            return False

    def _check_if_open_trade(self, action):
        if self._position == Positions.NoPosition:
            return action == Actions.Sell.value or action == Actions.Buy.value

        return False

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward()
        self._total_reward += step_reward

        self._update_profit(action)

        if self._check_if_open_trade(action):
            if self.trade_on_close:
                self._last_trade_tick = self._current_tick
            else:
                self._last_trade_tick = self._current_tick + 1
            self._set_position(action, self._current_tick)
        elif self._check_if_close_trade(action):
            self._set_no_position(self._current_tick)

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

    def _get_trade_prices(self):
        current_price = self.close_prices[self._current_tick]
        if self.trade_on_close:
            last_trade_price = self.close_prices[self._last_trade_tick]
        else:
            last_trade_price = self.open_prices[self._last_trade_tick]

        return last_trade_price, current_price

    def _calculate_reward(self):
        last_trade_price, current_price = self._get_trade_prices()
        reward = 0

        if self._position == Positions.Long:
            reward = np.log(np.abs(current_price / last_trade_price))
        elif self._position == Positions.Short:
            reward = np.log(np.abs(last_trade_price / current_price))

        if current_price < 0 and last_trade_price > 0:
            reward = - reward

        return reward

    def _update_profit(self, action):

        if self._check_if_close_trade(action) or self._done:
            last_trade_price, current_price = self._get_trade_prices()

            n_contracts = np.floor((self._total_profit * self.pos_size) / self.risk_per_contract)

            if self._position == Positions.Long:
                pos_profit = (current_price - last_trade_price) * self.point_value * n_contracts
                self._total_profit += pos_profit
            elif self._position == Positions.Short:
                pos_profit = (last_trade_price - current_price) * self.point_value * n_contracts
                self._total_profit += pos_profit

    def get_account_value(self):
        return self._total_profit

    def max_possible_profit(self):
        pass

    def _set_position(self, action, current_tick):
        delay = 0 if self.trade_on_close else 1

        if action == Actions.Buy.value:
            self._position = Positions.Long
            self.long_ticks.append(current_tick + delay)
        elif action == Actions.Sell.value:
            self._position = Positions.Short
            self.short_ticks.append(current_tick + delay)

    def _set_no_position(self, current_tick):
        delay = 0 if self.trade_on_close else 1

        if self._position == Positions.Short:
            self.long_ticks.append(current_tick + delay)
        elif self._position == Positions.Long:
            self.short_ticks.append(current_tick + delay)

        self._position = Positions.NoPosition

    def get_trading_df(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        final_df = pd.DataFrame(
            self.sc.inverse_transform(np.concatenate([self.prices.reshape(-1, 1), self.signal_features], axis=1)),
            columns=self.df_cols,
            index=self.df_index[start:end]
        )
        final_df.loc[:, 'action'] = np.array(self._action_history)
        final_df.loc[:, 'total_profit'] = np.array(
            ((self.window_size + 1) * [self.initial_capital]) + self.history['total_profit'])
        return final_df

    def render_all(self, mode='human'):
        plt.plot(self.close_prices)

        plt.plot(self.short_ticks, self.close_prices[self.short_ticks], 'ro')
        plt.plot(self.long_ticks, self.close_prices[self.long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def _update_position(self):
        self._position_history.append(self._position)

        last_col_idx = self.signal_features.shape[1]

        if self._current_tick != self._end_tick:
            if self._position == Positions.Long:
                self.signal_features[self._current_tick + 1, last_col_idx - 3:last_col_idx] = np.array([1, 0, 0])
            elif self._position == Positions.Short:
                self.signal_features[self._current_tick + 1, last_col_idx - 3:last_col_idx] = np.array([0, 0, 1])
            elif self._position == Positions.NoPosition:
                self.signal_features[self._current_tick + 1, last_col_idx - 3:last_col_idx] = np.array([0, 1, 0])
