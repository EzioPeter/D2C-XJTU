import torch
from torch import nn, Tensor
from typing import Union, Sequence
import numpy as np
from gym.spaces import Box, Space
import torch.nn.functional as F
from gym.spaces import Discrete



class FullyConnectedNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, arch='256-256', orthogonal_init=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init

        d = input_dim
        modules = []
        hidden_sizes = [int(h) for h in arch.split('-')]

        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            if orthogonal_init:
                nn.init.orthogonal_(fc.weight, gain=np.sqrt(2))
                nn.init.constant_(fc.bias, 0.0)
            modules.append(fc)
            modules.append(nn.ReLU())
            d = hidden_size

        last_fc = nn.Linear(d, output_dim)
        if orthogonal_init:
            nn.init.orthogonal_(last_fc.weight, gain=1.0)
        else:
            nn.init.xavier_uniform_(last_fc.weight, gain=1.0)

        nn.init.constant_(last_fc.bias, 0.0)
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        return self.network(input_tensor)
    
class Qnetwork(nn.Module):
    def __init__(self, observation_space, action_space, fc_layer_params=(), device="cpu", orthogonal_init=False, bins_per_dim=5):
        super().__init__()
        if isinstance(observation_space, int):
            obs_dim = observation_space
        else:
            obs_dim = int(np.prod(observation_space.shape))

        self.device = device
        self.is_discrete = isinstance(action_space, Discrete)

        if self.is_discrete:
            
            self.action_dim = action_space.n
            self.discretized_actions = None
        elif isinstance(action_space, Box):
            
            self.low = action_space.low
            self.high = action_space.high
            self.bins_per_dim = bins_per_dim
            
            grids = [np.linspace(l, h, bins_per_dim) for l, h in zip(self.low, self.high)]
            mesh = np.meshgrid(*grids)
            self.discretized_actions = np.stack([m.flatten() for m in mesh], axis=-1)
            self.action_dim = len(self.discretized_actions)
        else:
            raise ValueError("Unsupported action space type.")

        arch = "-".join(map(str, fc_layer_params))
        self.network = FullyConnectedNetwork(obs_dim, self.action_dim, arch, orthogonal_init)

    def forward(self, state):
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        x = x.view(x.size(0), -1) if x.ndim > 2 else x
        return self.network(x)

class EpsilonGreedyPolicy(object):
    def __init__(self, q_network, action_dim, epsilon, device="cpu"):
        self.q_network = q_network
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.device = device

    def __call__(self, observation):
        if np.random.rand() < self.epsilon:
            a_idx = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.q_network(state)
                a_idx = torch.argmax(q_values, dim=1).item()

        if getattr(self.q_network, "discretized_actions", None) is not None:
            return self.q_network.discretized_actions[a_idx]
        else:
            return a_idx
            
class StepSampler(object):

    def __init__(self, env, max_traj_length=1000, epsilon = 0.1, dis=None,  device="cuda"):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._dis = dis
        self.device = device
        # if self._dis:
        #    self.d_sa = dis[0]
        #    self.d_sas = dis[1]
        #    self.clip_dynamics_ratio_min = dis[2]
        #    self.clip_dynamics_ratio_max = dis[3]
        # self._current_observation = self.env.reset(seed=42)
        self._current_observation = self.env.reset()

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None, joint_noise_std=0.):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation
            if isinstance(observation, torch.Tensor):
                observation = observation.cpu().numpy()
            action = policy(observation)
            if getattr(policy.q_network, "discretized_actions", None) is not None:
                if isinstance(action, (int, np.integer)):
                    action = policy.q_network.discretized_actions[action]
            if isinstance(action, np.ndarray):
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
            next_observation, reward, done, _ = self.env.step(action)
            observations.append(observation)
            actions.append(action)
            if isinstance(next_observation, torch.Tensor):
                next_observation = next_observation.cpu().numpy()
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                # self._current_observation = self.env.reset(seed=42)
                self._current_observation = self.env.reset()

        # if self._dis:
        #    sim_real_dynamics_ratio = self.sim_real_dynamics_ratio(observations, actions, next_observations)
        #    in_dynamics = (sim_real_dynamics_ratio < self.clip_dynamics_ratio_max) & (sim_real_dynamics_ratio > self.clip_dynamics_ratio_min)
        #    in_dynamics_index = [i for i, x in enumerate(in_dynamics) if x]
        #    observations = [observations[i] for i in range(len(observations)) if i in in_dynamics_index]
        #    actions = [actions[i] for i in range(len(actions)) if i in in_dynamics_index]
        #    rewards = [rewards[i] for i in range(len(rewards)) if i in in_dynamics_index]
        #    next_observations = [next_observations[i] for i in range(len(next_observations)) if i in in_dynamics_index]
        #     dones = [dones[i] for i in range(len(dones)) if i in in_dynamics_index]
            
        if replay_buffer is not None:
            replay_buffer.append_traj(
                observations, actions, rewards, next_observations, dones
            )
        
        return dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )

    @property
    def env(self):
        return self._env
    
    # def sim_real_dynamics_ratio(self, observations, actions, next_observations):
    #    observations = torch.FloatTensor(observations).to(self.device)
    #    actions = torch.FloatTensor(actions).to(self.device)
    #    next_observations = torch.FloatTensor(next_observations).to(self.device)
        
    #    sa_logits = self.d_sa(observations, actions)
    #    sa_prob = F.softmax(sa_logits, dim=1)
    #    adv_logits = self.d_sas(observations, actions, next_observations)
    #    sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

    #    with torch.no_grad():
    #        ratio = torch.clamp((sas_prob[:, 1] * sa_prob[:, 0]) / (sas_prob[:, 0] * sa_prob[:, 1]), min=self.clip_dynamics_ratio_min, max=self.clip_dynamics_ratio_max)

    #    return ratio

class TrajSampler(object):

    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env

    def sample(self, policy, n_trajs, deterministic=False, replay_buffer=None):
        trajs = []
        for _ in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []

            # observation = self.env.reset(seed=42)
            observation = self.env.reset()
            if isinstance(observation, torch.Tensor):
                observation = observation.cpu().numpy()

            for _ in range(self.max_traj_length):
                action = policy(observation)
                next_observation, reward, done, _ = self.env.step(action)
                if isinstance(next_observation, torch.Tensor):
                    next_observation = next_observation.cpu().numpy()

                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_observations.append(next_observation)

                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        observation, action, reward, next_observation, done
                    )

                observation = next_observation

                if done:
                    break

            trajs.append(dict(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(actions, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
            ))

        return trajs
