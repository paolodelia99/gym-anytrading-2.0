import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gym_anytrading2.agents.DDQN_agent import DDQNTradingAgent
from gym_anytrading2.envs.future_env import FuturesEnv
from utils import MetricLogger

torch.manual_seed(0)
np.random.seed(0)


def add_cyclical_features(df):
    df['date'] = pd.to_datetime(df.index.copy(), format='%Y-%m-%d')
    df['day_sin'] = df['date'].apply(lambda x: np.sin(x.day * (2. * np.pi / 30)))
    df['day_cos'] = df['date'].apply(lambda x: np.cos(x.day * (2. * np.pi / 30)))
    df['month_sin'] = df['date'].apply(lambda x: np.sin(x.month * (2. * np.pi / 12)))
    df['month_cos'] = df['date'].apply(lambda x: np.cos(x.month * (2. * np.pi / 12)))
    df = df.drop('date', axis=1)
    return df


def real_loop(env, agent):

    i = 0

    state = env.reset()

    while True:

        action = agent.act(state, eval_mode=True)

        next_state, reward, done, info = env.step(action)

        state = next_state

        if done:
            print('info: ', info)
            break

        print(f'step: {i}, reward: {reward}, account_value: {env.get_account_value()}, action: {action}, position: {info["position"]}')

        i += 1

    plt.cla()
    env.render_all()
    plt.title(f'Testing')
    plt.savefig('models/ddqn_cl_checkpoints_d/2022-06-14T23-15-05/cl_res.png', dpi=300)
    env.get_trading_df().to_csv('models/ddqn_cl_checkpoints_d/2022-06-14T23-15-05/testing_set.csv')


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
    parser = argparse.ArgumentParser()
    parser.add_argument('-te', '--test', action='store_true',
                        help='Test mode of the model')
    parser.add_argument('-p', '--path', type=str,
                        help='Path of the model')

    args = parser.parse_args()
    cl_df = pd.read_csv('./data/CL_daily_adj.csv', parse_dates=True)
    cl_df = cl_df.set_index('Date')
    cl_df = add_cyclical_features(cl_df)
    window_size = 21
    save_dir = Path("models/ddqn_cl_checkpoints_d") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S") if not args.test else args.path
    episodes = 400 if not args.test else 1

    if not args.test:
        save_dir.mkdir(parents=True)

    if not args.test:
        env = FuturesEnv(df=cl_df,
                         window_size=window_size,
                         frame_bound=(window_size, len(cl_df)))
    else:
        env = FuturesEnv(df=cl_df,
                         window_size=window_size,
                         frame_bound=(len(cl_df) - 753, len(cl_df)))

    agent = DDQNTradingAgent(
        env.observation_space.shape[1],
        env.action_space.n,
        save_dir,
    )

    if args.test:
        agent.load_model('models/ddqn_cl_checkpoints_d/2022-06-14T23-15-05/ddqn_net_6.chkpt', recurrent=True)

    if args.test:
        real_loop(env, agent)
    else:
        logger = MetricLogger(save_dir)
        train_loop(env, episodes, agent, logger)
