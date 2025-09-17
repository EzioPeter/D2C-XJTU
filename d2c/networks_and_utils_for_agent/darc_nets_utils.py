import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from typing import Tuple, List, Union, Type, Optional, Sequence
from gym.spaces import Box, Space

def extend_and_repeat(tensor, dim, repeat):
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)

def atanh(z):
    return 0.5 * (torch.log(1 + z) - torch.log(1 - z))

def soft_target_update(network, target_network, soft_target_update_rate):
    target_network_params = {k: v for k, v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (
            (1 - soft_target_update_rate) * target_network_params[k].data
            + soft_target_update_rate * v.data
        )


def multiple_action_q_function(forward):
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values
    return wrapped


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
            nn.init.orthogonal_(last_fc.weight, gain=1e-2)
        else:
            nn.init.xavier_uniform_(last_fc.weight, gain=1e-2)

        nn.init.constant_(last_fc.bias, 0.0)
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        return self.network(input_tensor)


class ReparameterizedTanhGaussian(nn.Module):

    def __init__(self, log_std_min=-20.0, log_std_max=2.0, eps=1e-6, no_tanh=False):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh
        self.eps = eps

    def log_prob(self, mean, log_std, sample):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(torch.clamp(sample, -1 + self.eps, 1 - self.eps)), dim=-1)
    

    def forward(self, mean, log_std, deterministic=False):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(
            action_distribution.log_prob(action_sample), dim=-1
        )

        return action_sample, log_prob


class ActorNetwork(nn.Module):

    def __init__(
            self, 
            observation_space: Union[Box, Space],
            action_space: Union[Box, Space],
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
            # arch='256-256',
            log_std_multiplier=1.0, 
            log_std_offset=-1.0,
            orthogonal_init=False, 
            no_tanh=False
    ):
        super().__init__()
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.arch = "-".join(map(str, fc_layer_params))
        self.device = device
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = FullyConnectedNetwork(
            self.observation_dim, 2 * self.action_dim, self.arch, orthogonal_init
        )
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        assert torch.isnan(observations).sum() == 0, print(observations)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        assert torch.isnan(mean).sum() == 0, print(mean)
        assert torch.isnan(log_std).sum() == 0, print(log_std)
        return self.tanh_gaussian(mean, log_std, deterministic)


class SamplerPolicy(object):

    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def __call__(self, observations, deterministic=False):
        with torch.no_grad():
            observations = torch.tensor(
                observations, dtype=torch.float32, device=self.device
            )
            actions, _ = self.policy(observations, deterministic)
            actions = actions.cpu().numpy()
        return actions


class FullyConnectedQFunction(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256', orthogonal_init=False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.network = FullyConnectedNetwork(
            observation_dim + action_dim, 1, arch, orthogonal_init
        )

    @multiple_action_q_function
    def forward(self, observations, actions):
        input_tensor = torch.cat([observations, actions], dim=-1)
        return torch.squeeze(self.network(input_tensor), dim=-1)


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32)
        )

    def forward(self):
        return self.constant

class StepSampler(object):

    def __init__(self, env, max_traj_length=1000, dis=None, device="cuda"):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._dis = dis
        self.device = device
        if self._dis:
            self.d_sa = dis[0]
            self.d_sas = dis[1]
            self.clip_dynamics_ratio_min = dis[2]
            self.clip_dynamics_ratio_max = dis[3]
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
            action = policy(
                np.expand_dims(observation, 0), deterministic=deterministic
            )[0, :]
            if joint_noise_std > 0.:
                next_observation, reward, done, _ = self.env.step(action + np.random.randn(action.shape[0],) * joint_noise_std)

            else:
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

        if self._dis:
            sim_real_dynamics_ratio = self.sim_real_dynamics_ratio(observations, actions, next_observations)
            in_dynamics = (sim_real_dynamics_ratio < self.clip_dynamics_ratio_max) & (sim_real_dynamics_ratio > self.clip_dynamics_ratio_min)
            in_dynamics_index = [i for i, x in enumerate(in_dynamics) if x]
            observations = [observations[i] for i in range(len(observations)) if i in in_dynamics_index]
            actions = [actions[i] for i in range(len(actions)) if i in in_dynamics_index]
            rewards = [rewards[i] for i in range(len(rewards)) if i in in_dynamics_index]
            next_observations = [next_observations[i] for i in range(len(next_observations)) if i in in_dynamics_index]
            dones = [dones[i] for i in range(len(dones)) if i in in_dynamics_index]
            
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
    
    def sim_real_dynamics_ratio(self, observations, actions, next_observations):
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        next_observations = torch.FloatTensor(next_observations).to(self.device)
        
        sa_logits = self.d_sa(observations, actions)
        sa_prob = F.softmax(sa_logits, dim=1)
        adv_logits = self.d_sas(observations, actions, next_observations)
        sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        with torch.no_grad():
            ratio = torch.clamp((sas_prob[:, 1] * sa_prob[:, 0]) / (sas_prob[:, 0] * sa_prob[:, 1]), min=self.clip_dynamics_ratio_min, max=self.clip_dynamics_ratio_max)

        return ratio

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
                action = policy(
                    np.expand_dims(observation, 0), deterministic=deterministic
                )[0, :]
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

    @property
    def env(self):
        return self._env
    
class Discriminator(nn.Module):
    def __init__(self, num_input, num_hidden, num_output=2, device="cuda", scale=2, dropout=False):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_output)
        self.output_scale = scale
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if self.dropout:
            x = F.relu(self.dropout_layer(self.fc1(x)))
            x = F.relu(self.dropout_layer(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        output = self.output_scale * torch.tanh(self.fc3(x))
        return output

class ConcatDiscriminator(Discriminator):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)

class RatioEstimator(nn.Module):
    def __init__(self, num_input, num_hidden, num_output=2, device="cuda", scale=1, dropout=False, output_activation=None):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_output)
        self.output_scale = scale
        self.output_activation = output_activation
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if self.dropout:
            x = F.relu(self.dropout_layer(self.fc1(x)))
            x = F.relu(self.dropout_layer(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        if self.output_activation:
            output = self.output_scale * self.output_activation(self.fc3(x))
        else:
            output = self.output_scale * self.fc3(x)
        
        return output
    
class ConcatRatioEstimator(RatioEstimator):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)
