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
    parser.add_argument("--save-freq", type=int, default=100,
        help="save every x policy updates")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MiniGrid-HazardWorld-B-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=200,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=2,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
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
        env = gym.make(env_id, size=13)
        if args.capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = HazardWorldMissionWrapper(env)
        env = gym.wrappers.TimeLimit(env, 200)
        env = RemoveStateDimWrapper(env)
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
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs_image = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space["image"].shape
    ).to(device)
    obs_mission = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space["mission"].shape
    ).to(device)
    obs_hc = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space["hc"].shape
    ).to(device)
    obs_violations = torch.zeros(
        (args.num_steps, args.num_envs)
        + envs.single_observation_space["violations"].shape
    ).to(device)

    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    invalid_action_masks = torch.zeros(
        (args.num_steps, args.num_envs) + (4,), dtype=torch.bool,
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_obs_image = torch.Tensor(next_obs["image"]).to(device)
    next_obs_mission = torch.Tensor(next_obs["mission"]).to(device)
    next_obs_hc = torch.Tensor(next_obs["hc"]).to(device)
    next_obs_violations = torch.Tensor(next_obs["violations"]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    print(f"{num_updates=}")
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            # obs[step] = next_obs
            obs_image[step] = next_obs_image
            obs_mission[step] = next_obs_mission
            obs_hc[step] = next_obs_hc
            obs_violations[step] = next_obs_violations
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                invalid_action_masks[step] = get_invalid_action_masks(
                    envs, next_obs_image
                )
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs_image, invalid_action_masks=invalid_action_masks[step]
                )
                values[step] = value.flatten()
            # if action.item() == 2:
            #     import code; code.interact(local=locals())
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            # # next_obs, reward, done, info = envs.step([2])
            # print(f'{action.item()=}')
            # print(f'{envs.envs[0].agent_dir}')
            # print(f'{envs.envs[0].agent_pos}')
            # import time
            # time.sleep(1)
            # envs.envs[0].render()
            # envs.render()
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            next_obs_image = torch.Tensor(next_obs["image"]).to(device)
            next_obs_mission = torch.Tensor(next_obs["mission"]).to(device)
            next_obs_hc = torch.Tensor(next_obs["hc"]).to(device)
            next_obs_violations = torch.Tensor(next_obs["violations"]).to(device)
            next_done = torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(
                        f"global_step={global_step}, episodic_return={item['episode']['r']}, violations={item['episode']['v']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", item["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/violations", max(item["episode"]["v"], 0), global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", item["episode"]["l"], global_step
                    )
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs_image).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_obs_image = obs_image.reshape(
            (-1,) + envs.single_observation_space["image"].shape
        )
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_invalid_action_masks = invalid_action_masks.reshape((-1,) + (4,))

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs_image[mb_inds],
                    action=b_actions.long()[mb_inds],
                    invalid_action_masks=b_invalid_action_masks[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        if (update - 1) % args.save_freq == 0 or update == num_updates:
            t_dir = f"runs/{run_name}/models"
            if not os.path.exists(t_dir):
                os.makedirs(t_dir)
            torch.save(agent.state_dict(), f"{t_dir}/agent.pt")
            torch.save(agent.state_dict(), f"{t_dir}/{global_step}.pt")

    envs.close()
    writer.close()

    writer = SummaryWriter(f"runs/{run_name}")
