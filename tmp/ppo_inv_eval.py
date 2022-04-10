# docs and experiment results can be found at
# https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse
from http.client import ImproperConnectionState
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


import gym_minigrid

import sys

sys.path.append("..")

from utils import encode_mission, create_constraint_mask


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="983",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--path-to-model", type=str, default=None,
        help="path to model")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MiniGrid-HazardWorld-B-v0",
        help="the id of the environment")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=200,
        help="the number of steps to run in each environment per policy rollout")
    args = parser.parse_args()
    # fmt: on
    return args


class HazardWorldMissionWrapper(gym.core.ObservationWrapper):
    """
    Converts the mission of the observation field into an integer string.
    This wrapper only works for HazardWorld
    """

    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # print(f"{obs['mission']=}")
        enc_mission = encode_mission(obs["mission"])
        obs["mission"] = enc_mission
        return obs


class RemoveStateDimWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space_img = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 2),
            dtype="uint8",
        )

        self.observation_space = gym.spaces.Dict(
            {
                "image": self.observation_space_img,
                "mission": self.observation_space_text,
                "violations": self.observation_space_violations,
                "hc": self.observation_space_hc,
            }
        )

    def observation(self, obs):
        img = obs["image"][:, :, 0:2]
        obs["image"] = img

        return obs


from collections import deque


class RecordEpisodeStatisticsV(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None
        # self.violation_returns = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.violations_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        # self.violation_returns = np.zeros(self.num_envs, dtype=np.float32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        # self.violation_returns += observations["violations"] - observations["hc"]
        if not self.is_vector_env:
            infos = [infos]
            dones = [dones]
        else:
            infos = list(infos)  # Convert infos to mutable type
        for i in range(len(dones)):
            if dones[i]:
                infos[i] = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                violation_return = observations["violations"][i] - observations["hc"][i]

                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "v": violation_return,
                    "t": round(time.perf_counter() - self.t0, 6),
                }
                infos[i]["episode"] = episode_info
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.violations_queue.append(violation_return)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        if self.is_vector_env:
            infos = tuple(infos)
        return (
            observations,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, size=11)
        if args.capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = HazardWorldMissionWrapper(env)
        env = RemoveStateDimWrapper(env)
        env = gym.wrappers.TimeLimit(env, 200)
        env = RecordEpisodeStatisticsV(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# class Agent(nn.Module):
#     def __init__(self, envs):
#         super(Agent, self).__init__()
#         self.critic = nn.Sequential(
#             layer_init(
#                 nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
#             ),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 1), std=1.0),
#         )
#         self.actor = nn.Sequential(
#             layer_init(
#                 nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
#             ),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
#         )

#     def get_value(self, x):
#         return self.critic(x)

#     def get_action_and_value(self, x, action=None):
#         logits = self.actor(x)
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(2, 5, 3, stride=1)),
            nn.Tanh(),
            layer_init(nn.Conv2d(5, 6, 3, stride=1)),
            nn.Tanh(),
            nn.Flatten(),
            layer_init(nn.Linear(54, 64)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1)

    def get_value(self, x):
        # [batch, h, w, c]
        x = torch.permute(x, (0, 3, 1, 2))
        # [batch, c, h, w]

        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None, invalid_action_masks=None):
        # [batch, h, w, c]
        x = torch.permute(x, (0, 3, 1, 2))
        # [batch, c, h, w]
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)

        adjusted_logits = torch.where(
            invalid_action_masks, logits, torch.tensor(-1e8, device=device)
        )
        # probs = Categorical(logits=logits)
        probs = Categorical(logits=adjusted_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    def get_invalid_action_masks(envs, obs_image):
        invalid_action_masks = torch.zeros(
            (len(envs.envs), 4), dtype=torch.bool, device=device
        )

        for i, env in enumerate(envs.envs):
            cmm, idx_to_avoid = create_constraint_mask(obs_image[i], env)

            # print(cmm)
            # print(obs['image'][i][:, :, 0])

            obj_in_front = obs_image[i][3, -2, 0]
            # print(obj_in_front)
            invalid_action_mask = torch.tensor([1.0, 1.0, 1.0, 1.0,], device=device)

            # only pick up items if it's infront
            if obj_in_front not in [5, 6, 7]:
                invalid_action_mask[3] = 0.0

            # don't violate constraint, 2 == wall
            if obj_in_front in [2, idx_to_avoid]:
                invalid_action_mask[2] = 0.0

            # target_logits = torch.tensor([1., 1., 1., 1.,] , requires_grad=True)
            invalid_action_mask = invalid_action_mask.type(torch.BoolTensor)
            invalid_action_masks[i] = invalid_action_mask

        return invalid_action_masks

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.path_to_model))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    EPISODES = 10
    step = 0
    for episode in range(EPISODES):
        obs = envs.reset()
        obs_img = torch.Tensor(obs["image"]).to(device)
        print(f"{episode=}")
        while True:

            # ALGO LOGIC: action logic
            with torch.no_grad():
                am = get_invalid_action_masks(envs, obs_img)
                action, logprob, _, value = agent.get_action_and_value(
                    obs_img, invalid_action_masks=am
                )
            # if action.item() == 2:
            #     import code; code.interact(local=locals())

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())

            obs = next_obs
            obs_img = torch.Tensor(obs["image"]).to(device)
            # # next_obs, reward, done, info = envs.step([2])
            # print(f'{action.item()=}')
            # print(f'{envs.envs[0].agent_dir}')
            # print(f'{envs.envs[0].agent_pos}')
            # import time
            # time.sleep(1)
            # envs.envs[0].render()
            # envs.render()
            # rewards[step] = torch.tensor(reward).to(device).view(-1)

            # next_obs_image = torch.Tensor(next_obs["image"]).to(device)
            # next_obs_mission = torch.Tensor(next_obs["mission"]).to(device)
            # next_obs_hc = torch.Tensor(next_obs["hc"]).to(device)
            # next_obs_violations = torch.Tensor(next_obs["violations"]).to(device)
            # next_done = torch.Tensor(done).to(device)
            for item in info:
                if "episode" in item.keys():
                    print(
                        f"global_step={global_step}, episodic_return={item['episode']['r']}, violations={max(item['episode']['v'], 0)}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", item["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/violations", item["episode"]["v"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", item["episode"]["l"], global_step
                    )
                    break

            if done[0]:
                break

    envs.close()
    writer.close()
