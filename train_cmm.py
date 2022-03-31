import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import random

import gym_minigrid
from TextLSTM import ConstraintThresholdModule
from cmm import ConstraintMaskModule
from utils import encode_mission, from_np_array

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
        mission, img, cm = self.dataset.iloc[idx]
        img = from_np_array(img).reshape(7, 7, 3)
        cm = from_np_array(cm).reshape(7, 7)
        mission = encode_mission(mission)
        return (
            torch.LongTensor(mission),
            torch.tensor(img, dtype=torch.float),
            torch.tensor(cm, dtype=torch.float),
        )


env_id = "MiniGrid-HazardWorld-B-v0"
env = gym.make(env_id)
BATCH_SIZE = 256
training_data = HazardWorld_B(f"{env_id}_cmm_train_data_1.csv")
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = HazardWorld_B(f"{env_id}_cmm_test_data_1.csv")
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

ctm_net = ConstraintThresholdModule().cuda()
ctm_net.load_state_dict(torch.load("ctm.pt"))
# for param in ctm_net.parameters():
#     param.requires_grad = False


device = torch.device("cuda")

cmm_net = ConstraintMaskModule().cuda()
optim = torch.optim.Adam(cmm_net.parameters(), lr=3e-5)
pos_weight = torch.ones(1, device=device) * 3
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

writer = SummaryWriter()

total_loss = 0.0
for epoch in range(100):
    total_loss = 0.0
    for i_batch, (missions, imgs, cms) in tqdm(
        enumerate(train_dataloader), total=len(train_dataloader)
    ):
        missions = missions.cuda()
        imgs = imgs.cuda()
        cms = cms.cuda()

        optim.zero_grad()
        with torch.no_grad():
            _, h_n = ctm_net(missions)
        pred_cms = cmm_net(imgs, h_n)
        loss = loss_fn(pred_cms, cms)
        total_loss += loss.item()
        loss.backward()
        optim.step()

    print(f"{epoch=}, Loss: {total_loss / (len(train_dataloader) * BATCH_SIZE)}")
    writer.add_scalar(
        "train/loss", total_loss / (len(train_dataloader) * BATCH_SIZE), epoch
    )
    print((torch.sigmoid(pred_cms[0]) >= 0.5).int())
    print(cms[0])

    corrects = 0
    totals = 0
    for i_batch, (missions, imgs, cms) in tqdm(
        enumerate(test_dataloader), total=len(test_dataloader)
    ):
        missions = missions.cuda()
        imgs = imgs.cuda()
        cms = cms.cuda()

        with torch.no_grad():
            _, h_n = ctm_net(missions)
        pred_cms = cmm_net(imgs, h_n)
        pred_cms = torch.sigmoid(pred_cms) >= 0.5
        correct = (pred_cms == cms).sum().cpu()
        corrects += correct.item()
        totals += torch.numel(cms)

    print(f"test: {corrects=} %{corrects / (totals)}")
    writer.add_scalar("test/correct", correct / totals, epoch)
    # print(f"test: {over=} %{over / (BATCH_SIZE * len(test_dataloader))}")
    # print(f"test: {under=} %{under / (BATCH_SIZE * len(test_dataloader))}")
    # print(f"{hcs=}")
    # print(f"{pred_hc=}")

# t = 0
# for i_batch, (missions, imgs, cms) in tqdm(
#     enumerate(train_dataloader), total=len(train_dataloader)
# ):
#     t += (cms > 0).sum().cpu() / (49)

# print(t / 20_000)
