import collections
import copy
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from d2c.utils import utils
from d2c.models.base import BaseAgent, BaseAgentModule
from d2c.networks_and_utils_for_agent.dqn_nets_utils import Qnetwork, EpsilonGreedyPolicy
from gym.spaces import Box, Discrete
import numpy as np
from d2c.envs import LeaEnv, BaseEnv
from typing import Union, Optional, List, Tuple, Dict, Sequence, Any, Iterator
from easydict import EasyDict
from d2c.utils.replaybuffer import ReplayBuffer

class DQNAgent(BaseAgent):

    def __init__(
        self,
        env: BaseEnv,
        model_params: Union[Dict, EasyDict, Any],
        optimizers: Union[Dict, EasyDict, Any],
        train_data: ReplayBuffer,
        weight_decays: float = 0.0,
        update_freq: int = 1,
        update_rate: float = 0.005,
        discount: float = 0.99,
        empty_dataset: Optional[ReplayBuffer] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        gamma=0.98,
        epsilon_start=1.0,
        epsilon_end=0.05,
        exploration_fraction=0.4,
        batch_size=256,
        max_traj_length=500,
        env_seed=1,
        learning_starts=10000,
        train_frequency=10,
        target_update_freq=500,
        alpha=0,
    ):
        self._env = env
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        
        if hasattr(self._action_space, "high"):  # Box
            self._a_max = torch.tensor(self._action_space.high, device=device, dtype=torch.float32)
            self._a_min = torch.tensor(self._action_space.low, device=device, dtype=torch.float32)
            self._a_dim = self._action_space.shape[0]
        elif hasattr(self._action_space, "n"):  # Discrete
            self._a_max = torch.tensor(float(self._action_space.n - 1), device=device)
            self._a_min = torch.tensor(0.0, device=device)
            self._a_dim = self._action_space.n
        else:
            raise ValueError(f"Unsupported action space type: {type(self._action_space)}")
        
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.exploration_fraction = exploration_fraction
        self.epsilon = epsilon_start
        self._batch_size = batch_size
        self._env_seed = env_seed
        self._learning_starts = learning_starts
        self._train_frequency = train_frequency
        self._target_update_freq = target_update_freq
        self.alpha = alpha
        self._model_params = model_params
        self._optimizers = optimizers
        self._batch_size = batch_size
        self._weight_decays = weight_decays
        self._train_data = train_data
        self._update_freq = update_freq
        self._update_rate = update_rate
        self._discount = discount
        self._empty_dataset = empty_dataset
        self._device = device
        self._modules = self._get_modules()
        self._build_agent()        
        self._state_mean = None
        self._state_std = None
        self._normalize_state = False

        self._update_counter = 0


    def _get_modules(self) -> utils.Flags:
        model_params_q, n_q_fns = self._model_params.q
        n_q_fns = 1

        def q_net_factory():
            bins_per_dim = getattr(self._model_params, "q_bins_per_dim", 5)
            return Qnetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_q,
                device=self._device,
                bins_per_dim=bins_per_dim,
            )

        return utils.Flags(q_net_factory=q_net_factory, n_q_fns=n_q_fns, device=self._device)

    def _build_fns(self):
        self._agent_module = AgentModule(modules=self._modules)
        self._q_fns = self._agent_module.q_nets
        self._q_target_fns = self._agent_module.q_target_nets
        self._q_fn = self._q_fns[0]
        self._q_target_fn = self._q_target_fns[0]

        if isinstance(self._action_space, Discrete):
            action_dim = self._action_space.n
        elif isinstance(self._action_space, Box):
            bins_per_dim = getattr(self._model_params, "q_bins_per_dim", 5)
            action_dim = bins_per_dim ** len(self._action_space.shape)
        else:
            raise ValueError(f"Unsupported action space type: {type(self._action_space)}")

        self._sampler_policy = EpsilonGreedyPolicy(
            q_network=self._q_fn,
            action_dim=action_dim,
            epsilon=self.epsilon,
            device=self._device,
        )

    def _build_optimizers(self):
        opts = self._optimizers
        self._q_optimizer = utils.get_optimizer(opts.q[0])(
            parameters=self._q_fn.parameters(),
            lr=opts.q[1],
            weight_decay=self._weight_decays,
        )


    def update_epsilon(self, step, total_train_steps):
        duration = int(self.exploration_fraction * total_train_steps)
        slope = (self.epsilon_end - self.epsilon_start) / duration
        self.epsilon = max(slope * step + self.epsilon_start, self.epsilon_end)
        self._sampler_policy.epsilon = self.epsilon

    def _normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        if not self._normalize_state:
            return states
        if self._state_mean is None:
            self._state_mean = states.mean(dim=0)
            self._state_std = states.std(dim=0) + 1e-6
        else:
            self._state_mean = 0.99 * self._state_mean + 0.01 * states.mean(dim=0)
            self._state_std = 0.99 * self._state_std + 0.01 * (states.std(dim=0) + 1e-6)
        return (states - self._state_mean) / self._state_std

    def _build_q_loss(self, batch):
        states = torch.as_tensor(batch['s1'], device=self._device, dtype=torch.float32)
        actions = torch.as_tensor(batch['a1'], device=self._device, dtype=torch.long)
        rewards = torch.as_tensor(batch['reward'], device=self._device, dtype=torch.float32)
        next_states = torch.as_tensor(batch['s2'], device=self._device, dtype=torch.float32)
        dones = torch.as_tensor(batch['dsc'], device=self._device, dtype=torch.float32)

        states = self._normalize_states(states)
        next_states = self._normalize_states(next_states)

        with torch.no_grad():
            next_actions = self._q_fn(next_states).argmax(1, keepdim=True)
            target_q_values = self._q_target_fn(next_states).gather(1, next_actions)
            td_target = rewards.flatten() + self.gamma * target_q_values.flatten() * (1 - dones.flatten())


        old_val = self._q_fn(states).gather(1, actions.long()).squeeze()
        q_loss = F.mse_loss(td_target, old_val)

        info = collections.OrderedDict()
        info['Q_loss'] = q_loss.detach().mean()
        info['average_q_pred'] = old_val.detach().mean()
        info['average_q_target'] = td_target.detach().mean()
        return q_loss, info

    def update_targets(self, tau=1.0):
        for t, s in zip(self._q_target_fn.parameters(), self._q_fn.parameters()):
            t.data.copy_(tau * s.data + (1.0 - tau) * t.data)

    def _optimize_step(self, batch):
        self._update_counter += 1
        q_loss, q_info = self._build_q_loss(batch)

        self._q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._q_fn.parameters(), max_norm=1.0)
        self._q_optimizer.step()

        if self._update_counter % self._target_update_freq == 0:
            self.update_targets(tau=1.0)

        return q_info

    def _build_test_policies(self) -> None:
        self._test_policies['main'] = EpsilonGreedyPolicy(
            q_network=self._q_fn,
            action_dim=self._sampler_policy.action_dim,
            epsilon=0.0,
            device=self._device,
        )

    def save(self, ckpt_name: str) -> None:
        torch.save(self._agent_module.state_dict(), ckpt_name + '.pth')
        torch.save(self._q_fn.state_dict(), ckpt_name + '_q.pth')

    def restore(self, ckpt_name: str) -> None:
        self._agent_module.load_state_dict(torch.load(ckpt_name + '.pth'))
        self._q_fn.load_state_dict(torch.load(ckpt_name + '_q.pth', map_location=self._device))


class AgentModule(BaseAgentModule):
    def _build_modules(self) -> None:
        device = self._net_modules.device
        self._q_nets = nn.ModuleList()
        for _ in range(self._net_modules.n_q_fns):
            self._q_nets.append(self._net_modules.q_net_factory().to(device))
        self._q_target_nets = copy.deepcopy(self._q_nets)

    @property
    def q_nets(self) -> nn.ModuleList:
        return self._q_nets

    @property
    def q_target_nets(self) -> nn.ModuleList:
        return self._q_target_nets
