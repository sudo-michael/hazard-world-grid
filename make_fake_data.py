import numpy as np
import gym
import gym_minigrid
import pandas as pd

from utils import create_constraint_mask

env_id = "MiniGrid-HazardWorld-B-v0"
env = gym.make(env_id)


imgs = []
missions = []
cms = []
for _ in range(8_000):
    obs = env.reset()
    img = obs["image"].flatten()
    mission = obs["mission"]
    cm = create_constraint_mask(obs, env).astype(int)
    cm = cm.flatten()
    img = img.flatten()
    imgs.append(img)
    missions.append(mission)
    cms.append(cm)
df = pd.DataFrame(data={"mission": missions, "img": imgs, "cm": cms})
df.to_csv(f"{env_id}_cmm_train_data_1.csv")

imgs = []
missions = []
cms = []
for _ in range(2_000):
    obs = env.reset()
    img = obs["image"].flatten()
    mission = obs["mission"]
    cm = create_constraint_mask(obs, env).astype(int)
    cm = cm.flatten()
    img = img.flatten()
    imgs.append(img)
    missions.append(mission)
    cms.append(cm)
df = pd.DataFrame(data={"mission": missions, "img": imgs, "cm": cms})
df.to_csv(f"{env_id}_cmm_test_data_1.csv")

# missions = []
# hcs = []
# for _ in range(80_000):
#     obs = env.reset()
#     missions.append(obs["mission"])
#     hcs.append(obs["hc"])

# df = pd.DataFrame(data={"mission": missions, "hc": hcs})
# df.to_csv(f"{env_id}_train_data.csv")

# missions = []
# hcs = []
# for _ in range(20_000):
#     obs = env.reset()
#     missions.append(obs["mission"])
#     hcs.append(obs["hc"])

# df = pd.DataFrame(data={"mission": missions, "hc": hcs})
# df.to_csv(f"{env_id}_test_data.csv")

