import gym
from gym_minigrid.wrappers import *

import argparse

parser = argparse.ArgumentParser()
import argparse

parser.add_argument(
    "--env-name",
    dest="env_name",
    help="gym environment to load",
    default="MiniGrid-HazardWorld-BA-v0",
)
parser.add_argument("--num_resets", default=200)
parser.add_argument("--num_frames", default=5000)
args = parser.parse_args()

env = gym.make(args.env_name)

obs = env.reset()
img = obs["image"]
mission = obs["mission"]

next_obs, reward, done, info = env.step(env.action_space.sample())

env.render()
