import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


class MetricLogger(object):

    def __init__(self, save_dir: Path):
        self.save_log = save_dir / "log.csv"
        self.lastest_step = 0
        self.lastest_epsilon = 0
        self.lastest_episode = 0

        self.log_df = pd.DataFrame(columns=[
            'episode',
            'step',
            'epsilon',
            'mean_reward',
            'mean_loss',
            'mean_qvalue',
            'account_value',
            'timedelta',
            'time'
        ])

        # History metrics
        self.ep_rewards = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

        if Path(self.save_log).is_file():
            self.load_data()

    def load_data(self):
        self.log_df = pd.read_csv(self.save_log, index_col=0)

    def get_lastest_step(self):
        return self.lastest_step

    def get_lastest_epsilon(self):
        return self.lastest_epsilon

    def get_lastest_episode(self):
        return self.lastest_episode

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        """Mark end of episode"""
        self.ep_rewards.append(self.curr_ep_reward)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step, account_value):
        ep_reward = np.round(self.ep_rewards[-1:], 5)[0]
        ep_loss = np.round(self.ep_avg_losses[-1:], 5)[0]
        ep_q = np.round(self.ep_avg_qs[-1:], 5)[0]

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        item = {
            'episode': episode,
            'step': step,
            'epsilon': epsilon,
            'mean_reward': ep_reward,
            'mean_loss': ep_loss,
            'mean_qvalue': ep_q,
            'account_value': account_value,
            'timedelta': time_since_last_record,
            'time': datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        }

        print(item)

        self.log_df = self.log_df.append(item, ignore_index=True)
        self.log_df.to_csv(self.save_log)
