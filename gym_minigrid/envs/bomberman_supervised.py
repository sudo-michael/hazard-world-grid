from gym_minigrid.envs.bomberman import BombermanEnv, BombermanEnvS
from gym_minigrid.register import register
from gym_minigrid.minigrid import *

import numpy as np

from gym import spaces

IDX_TO_AVOID = {
    'lava': 9,
    'grass': 11,
    'water': 12
}

class BombermanSupervised(BombermanEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed)
        self.action_space = spaces.Dict({
            'action' : spaces.Discrete(4),
            'prediction' : spaces.Box(
                low=np.array([0.0, 0.0]), 
                high=np.array([np.inf, 1]),
                dtype=np.float32)
        })
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.avoid_obj is None:
            cost_mask = np.zeros((obs['image'].shape[0],obs['image'].shape[1]))
        else:
            avoid_obj_filter = np.zeros((obs['image'].shape[0], obs['image'].shape[1])) + IDX_TO_AVOID[self.avoid_obj]
            cost_mask = avoid_obj_filter == obs['image'][:,:,0]
        # cost_filter = np.expand_dims(cost_filter, axis=2)
        info['cost mask'] = cost_mask
        info['hc'] = [True, self.hc - 1]

        return obs, reward, done, info 

class BombermanSupervisedS(BombermanEnvS):
    def __init__(self, size=13, seed=None):
        super().__init__(size=size, seed=seed)
        self.action_space = spaces.Dict({
            'action' : spaces.Discrete(4),
            'prediction' : spaces.Box(
                low=np.array([0.0, 0.0]), 
                high=np.array([np.inf, 1]),
                dtype=np.float32
            )
        })
    
    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.avoid_obj is None:
            cost_mask = np.zeros((obs['image'].shape[0],obs['image'].shape[1]))
        else:
            avoid_obj_filter = np.zeros((obs['image'].shape[0], obs['image'].shape[1])) + IDX_TO_AVOID[self.avoid_obj]
            cost_mask = avoid_obj_filter == obs['image'][:,:,0]
        # cost_filter = np.expand_dims(cost_filter, axis=2)
        info['cost mask'] = cost_mask
        if self.avoid_obj is None:
            info['hc'] = [False, self.hc - 1]
        else:
            info['hc'] = [True, self.hc - 1]

        return obs, reward, done, info 
    
register(
    id='MiniGrid-BombSuper-S-v0',
    entry_point='gym_minigrid.envs:BombermanSupervisedS'
)
register(
    id='MiniGrid-BombSuper-v0',
    entry_point='gym_minigrid.envs:BombermanSupervised'
)

