from importlib.metadata import requires
from tkinter.tix import TEXT
import torch
import torch.nn as nn
import numpy as np
import gym

import gym_minigrid

from torch.utils.data import Dataset, DataLoader
import pandas as pd

from utils import STOI


class ConstraintThresholdModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = nn.Embedding(len(STOI), 5)
        self.lstm = nn.LSTM(input_size=5, hidden_size=5)
        self.linear = nn.Sequential(
            nn.Linear(5, 10), nn.Tanh(), nn.Linear(10, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, X):
        X = self.C(X)  # X = (batch_size, words, embeding_size)
        X = X.transpose(0, 1)  # X = (words, batch_size, embeding_size)
        _, (h_n, _) = self.lstm(X)  # h_n = (1, batch_size, final_hidden_cell)
        pred_hc = self.linear(h_n[-1])
        return pred_hc, h_n[-1]


if __name__ in "__main__":
    env = gym.make("MiniGrid-HazardWorld-B-v0")

    net = ConstraintThresholdModule()
    optim = torch.optim.Adam(net.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    obs = env.reset()
    mission = obs["mission"]
    hc = obs["hc"]

    mission = encode_mission(mission)
    missions = [mission]
    hc_targets = [[hc]]
    missions = torch.LongTensor(missions)
    hc_targets = torch.Tensor(hc_targets)

    for _ in range(1):
        optim.zero_grad()
        print(missions)
        print(missions.shape)
        print(missions.type())
        pred_hc = net(missions)
        loss = loss_fn(pred_hc, hc_targets)
        print(loss)
        loss.backward()
        optim.step()

    class HazardWorld_B(Dataset):
        def __init__(self, csv_file) -> None:
            self.dataset = pd.read_csv(csv_file, index_col=0)

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            mission, hc = self.dataset.iloc[idx]
            mission = encode_mission(mission)
            return torch.LongTensor(mission), hc

    env_id = "MiniGrid-HazardWorld-B-v0"
    env = gym.make(env_id)
    training_data = HazardWorld_B(f"{env_id}_train_data.csv")
    train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)

    net = ConstraintThresholdModule()
    optim = torch.optim.Adam(net.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    for i_batch, (missions, hcs) in enumerate(train_dataloader):
        optim.zero_grad()
        print(missions)
        print(missions.shape)
        print(missions.type())
        pred_hc = net(missions)
        print(pred_hc)
        print(hcs.view(-1, 1))
        loss = loss_fn(pred_hc, hcs.view(-1, 1))
        print(loss)
        loss.backward()
        optim.step()
        break

    # for i in range(1):
#         mission = encode_mission(mission)

#         pred_hc = net(mission)
#         print(pred_hc)
#         print(pred_hc.shape)

#         target = torch.LongTensor([obs["hc"]]).view(1, 1)
#         print(target)
#         print(target.shape)
#         # pred_hc = ctm(mission)

#         optim.zero_grad()
#         loss = loss_fn(pred_hc, target)
#         loss.backward()
#         optim.step()

#         print(f"{i=} loss:{loss.cpu().item()}")


# # class TextLSTM(LasagnePowered, Serializable):
# #     def __init__(self, output_dim, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
# #                  output_W_init=LI.GlorotUniform(), output_b_init=LI.Constant(0.),
# #                  name=None, input_var=None, input_layer=None, batch_norm=False,input_shape=None):

# #         Serializable.quick_init(self, locals())

# #         if name is None:
# #             prefix = ""
# #         else:
# #             prefix = name + "_"

# #         # We now build the LSTM layer which takes l_in as the input layer
# #         # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients.

# #         # version v1: 04/22
# #         '''
# #         if input_layer is None:
# #             l_in = L.InputLayer(shape=(None, None, text_dim_glove_size),input_var=input_var)
# #         else:
# #             l_in = input_layer

# #         self._layers = [l_in]
# #         '''

# #         # version v2: 04/22
# #         l_in = L.InputLayer(shape=(None,) + input_shape,input_var=TT.imatrix())
# #         self._layers = [l_in]


# #         obj_embedding_size = 10
# #         network_text = L.EmbeddingLayer(l_in, input_size=650, output_size=obj_embedding_size,name="text_emb",W=LI.Normal(1.0))
# #         self._layers.append(network_text)
# #         # size of network_text is (None, 8, 10) #(batch_size, sequence_length, num_inputs)
# #         network_text = L.LSTMLayer(
# #             network_text, num_units=obj_embedding_size, #grad_clipping=100,
# #             nonlinearity=LN.tanh,
# #             #precompute_input=True,
# #             #unroll_scan=True,
# #             name="text_lstm",
# #             only_return_final=True)# output is (None, 10)
# #         self._layers.append(network_text)
