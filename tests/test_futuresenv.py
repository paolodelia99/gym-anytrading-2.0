import pytest

import numpy as np

import gym
from gym.utils.env_checker import check_env
import gym_anytrading2
from gym_anytrading2.datasets import CL_D
from gym_anytrading2.envs import FuturesEnv, Positions, Actions

seed = 42
np.random.seed(seed)


def test_env_init_conditions():
    env = gym.make('crude-oil-d-v0')
    env.reset(seed=seed)
    env.action_space.seed(seed)

    for i in range(0, 3):
        assert env.action_space.contains(i)

    assert env.observation_space.shape == (21, 20)
    assert env.observation_space.contains(np.ones((21, 20), dtype=np.float32))
    assert env.initial_capital == 1_000_000
    assert env.window_size == 21
    assert env.long_ticks == []
    assert env.short_ticks == []


def test_env_init_1():
    env = gym.make('crude-oil-d-v0', window_size=42, frame_bound=(42, len(CL_D)))
    obs = env.reset(seed=seed)

    assert obs.shape == (42, 20)


def test_env_step_basic():
    env = gym.make('crude-oil-d-v0')
    env.reset()
    env.action_space.seed(seed)

    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        assert type(done) is bool
        assert type(info) is dict
        assert obs.shape == (21, 20)
