import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .trading_env import Actions, Positions
from .trading_env import TradingEnv


class FuturesEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, pos_size: float = 0.05,
                 risk_per_contract=1_000,
                 point_value=1_000, initial_capital=1_000_000):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size)

        self.sc = StandardScaler()
        self.df_cols = df.columns
        self.df_index = df.index
        self.df = pd.DataFrame(self.sc.fit_transform(df), columns=df.columns, index=df.index)
        self.prices, self.signal_features = self._process_data()
        self.close_idx = int(np.where(self.sc.feature_names_in_ == 'close')[0][0])
        self.pos_size = pos_size
        self.initial_capital = initial_capital
        self.risk_per_contract = risk_per_contract
        self.point_value = point_value
        self.long_ticks = []
        self.short_ticks = []


    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self.is_trade_open = False
        self._position = Positions.NoPosition
        self._position_history = (self.window_size * [None]) + [self._position]
        self._action_history = (self.window_size * [0]) + [Actions.Hold]
        self._total_reward = 0.
        self._total_profit = self.initial_capital
        self._first_rendering = True
        self.history = {}
        self.long_ticks = []
        self.short_ticks = []
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

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        if self._check_if_open_trade(action):
            self._last_trade_tick = self._current_tick
            self._set_position(action, self._current_tick)
        elif self._check_if_close_trade(action):
            self._set_no_position(self._current_tick)

        self._position_history.append(self._position)
        self._action_history.append(action)
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position.value
        )
        self._update_history(info)

        if self._total_profit < 0:
            self._done = True

        return observation, step_reward, self._done, info

    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        prices = self.df.loc[:, 'close'].to_numpy()[start:end]
        signal_features = self.df.drop(['close'], axis=1).to_numpy()[start:end]
        return prices, signal_features

    @staticmethod
    def close_rev_scaling(close, sc: StandardScaler, close_idx: int):
        return (close * np.sqrt(sc.var_[close_idx])) + sc.mean_[close_idx]

    def _calculate_reward(self, action):
        current_price = FuturesEnv.close_rev_scaling(self.prices[self._current_tick], self.sc, self.close_idx)
        last_trade_price = FuturesEnv.close_rev_scaling(self.prices[self._last_trade_tick], self.sc, self.close_idx)
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
            current_price = FuturesEnv.close_rev_scaling(self.prices[self._current_tick], self.sc, self.close_idx)
            last_trade_price = FuturesEnv.close_rev_scaling(self.prices[self._last_trade_tick], self.sc, self.close_idx)
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
        if action == Actions.Buy.value:
            self._position = Positions.Long
            self.long_ticks.append(current_tick)
        elif action == Actions.Sell.value:
            self._position = Positions.Short
            self.short_ticks.append(current_tick)

    def _set_no_position(self, current_tick):
        if self._position == Positions.Short:
            self.long_ticks.append(current_tick)
        elif self._position == Positions.Long:
            self.short_ticks.append(current_tick)

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
        final_df.loc[:, 'total_profit'] = np.array(((self.window_size + 1) * [self.initial_capital]) + self.history['total_profit'])
        return final_df

    def render_all(self, mode='human'):
        plt.plot(self.short_ticks, self.prices[self.short_ticks], 'ro')
        plt.plot(self.long_ticks, self.prices[self.long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )