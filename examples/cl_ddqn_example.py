import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gym_anytrading.agents.DDQN_agent import DDQNTradingAgent
from gym_anytrading.envs.future_env import FuturesEnv

torch.manual_seed(0)
np.random.seed(0)


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


def add_cyclical_features(df):
    df['date'] = pd.to_datetime(df.index.copy(), format='%Y-%m-%d %H:%M:%S')
    df['hour_sin'] = df['date'].apply(lambda x: np.sin(x.hour * (2. * np.pi / 24)))
    df['hour_cos'] = df['date'].apply(lambda x: np.cos(x.hour * (2. * np.pi / 24)))
    df['day_sin'] = df['date'].apply(lambda x: np.sin(x.day * (2. * np.pi / 30)))
    df['day_cos'] = df['date'].apply(lambda x: np.cos(x.day * (2. * np.pi / 30)))
    df['month_sin'] = df['date'].apply(lambda x: np.sin(x.month * (2. * np.pi / 12)))
    df['month_cos'] = df['date'].apply(lambda x: np.cos(x.month * (2. * np.pi / 12)))
    df = df.drop('date', axis=1)
    return df


def train_loop(env, episodes: int, agent, logger):
    for episode in range(episodes):

        state = env.reset()

        while True:

            action = agent.act(state)

            next_state, reward, done, info = env.step(action)

            agent.cache(state, next_state, action, reward, done)

            q, loss = agent.learn()

            logger.log_step(reward, loss, q)

            state = next_state

            if done:
                print('info: ', info)
                break

        logger.log_episode()

        account_value = env.get_account_value()

        logger.record(
            episode=episode,
            epsilon=agent.exploration_rate,
            step=agent.curr_step,
            account_value=account_value)


if __name__ == '__main__':
    cl_df = pd.read_csv('data/CL1!_adj.csv', index_col=0, parse_dates=True)
    cl_df = cl_df.set_index('Date')
    cl_df = add_cyclical_features(cl_df)
    window_size = 115
    save_dir = Path("models/ddqn_cl_checkpoints") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    episodes = 150

    env = FuturesEnv(df=cl_df,
                     window_size=window_size,
                     frame_bound=(window_size, len(cl_df)))

    agent = DDQNTradingAgent(
        env.observation_space.shape[1],
        env.action_space.n,
        save_dir,
    )

    logger = MetricLogger(save_dir)
    train_loop(env, episodes, agent, logger)
