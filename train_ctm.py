import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn

# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import random

import gym_minigrid
from TextLSTM import ConstraintThresholdModule
from utils import encode_mission

SEED = 0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


class HazardWorld_B(Dataset):
    def __init__(self, csv_file) -> None:
        self.dataset = pd.read_csv(csv_file, index_col=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        mission, hc = self.dataset.iloc[idx]
        mission = encode_mission(mission)
        return torch.LongTensor(mission), torch.tensor(hc, dtype=torch.float)


env_id = "MiniGrid-HazardWorld-B-v0"
env = gym.make(env_id)
BATCH_SIZE = 128
training_data = HazardWorld_B(f"{env_id}_train_data.csv")
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = HazardWorld_B(f"{env_id}_test_data.csv")
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

net = ConstraintThresholdModule().cuda()
optim = torch.optim.Adam(net.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()

# writer = SummaryWriter()

total_loss = 0.0
for epoch in range(8):
    total_loss = 0.0
    for i_batch, (missions, hcs) in tqdm(enumerate(train_dataloader)):
        missions = missions.cuda()
        hcs = hcs.cuda()

        optim.zero_grad()
        pred_hc, _ = net(missions)
        loss = loss_fn(pred_hc, hcs.view(-1, 1))
        total_loss = loss.item()
        loss.backward()
        optim.step()

    print(f"{epoch=}, Loss: {total_loss / (len(train_dataloader) * BATCH_SIZE)}")

    corrects = 0
    over = 0
    under = 0
    for i_batch, (missions, hcs) in tqdm(enumerate(test_dataloader)):
        missions = missions.cuda()
        hcs = hcs.cuda()
        with torch.no_grad():
            pred_hc, _ = net(missions)
            pred_hc = torch.floor(pred_hc).int().view(1, -1)
            pred_hc = torch.maximum(pred_hc, torch.zeros_like(pred_hc))
        correct = (hcs == pred_hc).sum().cpu()
        corrects += correct.item()
        overs = (hcs < pred_hc).sum().cpu()
        over += overs.item()
        unders = (hcs > pred_hc).sum().cpu()
        under += unders.item()

    print(f"test: {corrects=} %{corrects / (BATCH_SIZE * len(test_dataloader))}")
    print(f"test: {over=} %{over / (BATCH_SIZE * len(test_dataloader))}")
    print(f"test: {under=} %{under / (BATCH_SIZE * len(test_dataloader))}")
    print(f"{hcs=}")
    print(f"{pred_hc=}")

torch.save(net.state_dict(), "ctm.pt")
