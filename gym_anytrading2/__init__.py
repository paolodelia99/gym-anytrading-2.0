from gym.envs.registration import register
from copy import deepcopy

from . import datasets


register(
    id='crude-oil-d-v0',
    entry_point='gym_anytrading2.envs:FuturesEnv',
    kwargs={
        'df': deepcopy(datasets.CL_D),
        'window_size': 21,
        'frame_bound': (21, len(datasets.CL_D))
    }
)

