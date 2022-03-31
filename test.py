import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import gym
from gym_minigrid.wrappers import *

import numpy as np


import argparse

parser = argparse.ArgumentParser()
import argparse

parser.add_argument(
    "--env-name",
    dest="env_name",
    help="gym environment to load",
    default="MiniGrid-HazardWorld-B-v0",
)
parser.add_argument("--num_resets", default=200)
parser.add_argument("--num_frames", default=5000)
args = parser.parse_args()

env = gym.make(args.env_name)

obs = env.reset()
mask_label = torch.zeros((7, 7))
breakpoint()
# img = obs["image"]  # (7, 7, 3)
# mission = obs["mission"]
# mission = [w.lower() for w in mission]
# hc = obs["hc"]
# # img = torch.from_numpy(img)


# class CMM(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#         self.transform = transforms.Compose(
#             [transforms.ToPILImage(), transforms.ToTensor()]
#         )

#         self.conv1 = nn.Conv2d(3, 10, 3)
#         self.dense1 = nn.Linear(250, 64)
#         self.dense2 = nn.Linear(64, 49)

#     def forward(self, img):
#         # img = np.array([transforms.ToPILImage()(img_) for img_ in img])
#         img = torch.from_numpy(img)
#         print(img.shape)
#         img = self.transform(img)
#         img = torch.tanh(self.conv1(img))
#         img = img.flatten()
#         print(img.shape)
#         img = self.dense1(img)
#         img = self.dense2(img)
#         img = img.reshape(7, 7)
#         return img


# cmm = CMM()

# # batch_img = np.stack([img, img], axis=0)
# c_img = cmm(img)


# mission = obs["mission"]

# next_obs, reward, done, info = env.step(env.action_space.sample())


# import json

# files = ["sequential.json", "relational.json", "budgetary.json"]
# from collections import Counter

# cnt = Counter()
# m = 0
# c_cnt = 0
# for fil in files:
#     with open(fil) as f:
#         data = json.load(f)
#     print(data.keys())
#     for k, constraints in data.items():
#         # print(k[-1])
#         for c in constraints:
#             c_cnt += 1
#             c = c.split()
#             c = [w.lower() for w in c]
#             cnt += Counter(c)
#             m = max(m, len(c))
# print(m)
# print(len(cnt))
