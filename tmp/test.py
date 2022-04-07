import gym
import gym_minigrid
import numpy as np

import sys

sys.path.append("..")

from utils import encode_mission

# from gym_minigrid.wrappers import OneHotPartialObsWrapper

env_id = "MiniGrid-HazardWorld-B-v0"
env = gym.make(env_id)


class HazardWorldMissionWrapper(gym.core.ObservationWrapper):
    """
    Converts the mission of the observation field into an integer string.
    This wrapper only works for HazardWorld
    """

    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        print(f"{obs['mission']=}")
        enc_mission = encode_mission(obs["mission"])
        obs["mission"] = enc_mission
        return obs


class RemoveStateDimWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        img = obs["image"][:, :, 0:2]
        obs["image"] = img

        return obs


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id, size=7)
        env = HazardWorldMissionWrapper(env)
        env = RemoveStateDimWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# env = make_env(env_id, 0)()
# breakpoint()
envs = gym.vector.SyncVectorEnv([make_env(env_id, i) for i in range(2)])
obss = envs.reset()
# breakpoint()
# print("obs")
next_obs, rewards, dones, info = envs.step(np.array([0, 0]))
next_obs, rewards, dones, info = envs.step([0, 0])
obs = env.reset()
print(next_obs)
breakpoint()

