from gym.envs.registration import register

from . import datasets

register(
    id='crude-oil-discrete-v0',
    entry_point='gym_anytrading2.envs:FuturesEnv',
    kwargs={
        'df': datasets.CL_D.copy(deep=True),
        'window_size': 21,
        'frame_bound': (21, len(datasets.CL_D))
    }
)

