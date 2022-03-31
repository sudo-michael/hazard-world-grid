from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import gym

import gym_minigrid

from torch.utils.data import Dataset, DataLoader
import pandas as pd

from TextLSTM import ConstraintThresholdModule


class ConstraintMaskModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(8, 10, 3)
        self.mlp = nn.Sequential(nn.Linear(250, 64), nn.ReLU(), nn.Linear(64, 49),)

    def forward(self, image_obs, instruction_embedding):
        # [batch, h, w, c]
        instruction_embedding = instruction_embedding.reshape(
            -1, 1, 1, instruction_embedding.shape[-1]
        )  # [batch, dim, 1, 1, hidden_size]
        instruction_embedding = instruction_embedding.repeat(1, 7, 7, 1)
        ctm = torch.cat((image_obs, instruction_embedding), dim=3)
        ctm = torch.permute(ctm, (0, 3, 1, 2))
        ctm = self.conv1(ctm)
        ctm = torch.permute(ctm, (0, 1, 2, 3))
        ctm = torch.flatten(ctm, start_dim=1)
        ctm = self.mlp(ctm).reshape(-1, 7, 7)
        return ctm


if "__main__" in __name__:
    env = gym.make("MiniGrid-HazardWorld-B-v0")

    ctm_net = ConstraintThresholdModule()
    ctm_net.load_state_dict(torch.load("ctm.pt"))
    for param in ctm_net.parameters():
        param.requires_grad = False

    cmm = ConstraintMaskModule()

    env_id = "MiniGrid-HazardWorld-B-v0"
    env = gym.make(env_id)
    csv = f"{env_id}_cmm_test_data.csv"
    df = pd.read_csv(csv)

