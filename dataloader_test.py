import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

env_id = "MiniGrid-HazardWorld-B-v0"
df = pd.read_csv(f"{env_id}_data.csv", index_col=0)


class HazardWorld_B(Dataset):
    def __init__(self, csv_file) -> None:
        self.dataset = pd.read_csv(csv_file, index_col=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        mission, hc = self.dataset.iloc[idx]
        return mission, hc


training_data = HazardWorld_B(f"{env_id}_train_data.csv")
test_data = HazardWorld_B(f"{env_id}_train_data.csv")
train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)


for i_batch, sample_batched in enumerate(train_dataloader):
    print(i_batch)
    print(sample_batched)
