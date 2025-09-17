"""
Implementation of SAC (Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor)
Paper: https://arxiv.org/abs/1801.01290.pdf

Refer to https://github.com/t6-thu/H2Oplus/blob/master/SimpleSAC/sac.py, the following revised content is rewritten by Jiayi Xie, Xi'an Jiaotong University.
"""
import collections

import copy
from ml_collections import ConfigDict

import torch
from tqdm import trange
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Union, Tuple, Any, Sequence, Dict, Iterator
from d2c.models.base import BaseAgent, BaseAgentModule
from d2c.utils import networks, utils, policies
from d2c.networks_and_utils_for_agent.sac_nets_utils import SamplerPolicy, ActorNetwork

class SACAgent(BaseAgent):
    """
    DARC Agent for cross-domain online reinforcement learning.
    """
    def __init__(
            self,
            update_actor_freq: int = 1,
            rollout_sim_freq: int = 1000,
            rollout_sim_num: int = 1000,            
            reward_scale: float = 1.0,
            alpha_multiplier: float = 1.0,
            automatic_entropy_tuning: bool = True,
            log_alpha_init_value: float = 0.0,
            backup_entropy: bool = True,
            target_entropy: float = 0.0,
            target_update_period: int = 1,
            batch_size: int = 256,
            joint_noise_std: float = 0.0,
            max_traj_length: int = 1000,
            env_seed: int = 42,
            **kwargs: Any,
    ) -> None:
        self._update_actor_freq = update_actor_freq
        self._rollout_sim_freq = rollout_sim_freq
        self._rollout_sim_num = rollout_sim_num
        self._reward_scale = reward_scale
        self._alpha_multiplier = alpha_multiplier
        self._automatic_entropy_tuning = automatic_entropy_tuning
        self._log_alpha_init_value = log_alpha_init_value
        self._backup_entropy = backup_entropy
        self._target_entropy = target_entropy
        self._target_update_period = target_update_period
        self._joint_noise_std = joint_noise_std
        self._max_traj_length = max_traj_length
        self._p_info = collections.OrderedDict()
        super(SACAgent, self).__init__(**kwargs)
        self._batch_size = batch_size
        self._env_seed = env_seed
        self._target_entropy = -np.prod(self._action_space.shape[0]).item()

    def _get_modules(self) -> utils.Flags:
        model_params_q, n_q_fns = self._model_params.q
        model_params_p = self._model_params.p[0]

        def q_net_factory():
            return networks.CriticNetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_q,
                device=self._device,
            )

        def p_net_factory():
            return ActorNetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_p,
                device=self._device,
            )

        def log_alpha_net_factory():
            return networks.Scalar(
                init_value=self._log_alpha_init_value,
                device=self._device
            )

        modules = utils.Flags(
            p_net_factory=p_net_factory,
            q_net_factory=q_net_factory,
            n_q_fns=n_q_fns,
            log_alpha_net_factory=log_alpha_net_factory,
            device=self._device,
            automatic_entropy_tuning=self._automatic_entropy_tuning,
        )

        return modules
    
    def _build_fns(self) -> None:
        self._agent_module = AgentModule(modules=self._modules)
        self._q_fns = self._agent_module.q_nets
        self._q_target_fns = self._agent_module.q_target_nets
        self._p_fn = self._agent_module.p_net
        self._p_target_fn = self._agent_module.p_target_net
        self._sampler_policy = SamplerPolicy(self._p_fn, self._device)
        if self._automatic_entropy_tuning:
            self._log_alpha_fn = self._agent_module.log_alpha_net

    def _init_vars(self) -> None:
        pass

    def _build_optimizers(self) -> None:
        opts = self._optimizers
        self._p_optimizer = utils.get_optimizer(opts.p[0])(
            parameters=self._p_fn.parameters(),
            lr=opts.p[1],
            weight_decay=self._weight_decays,
        )
        self._q_optimizer = utils.get_optimizer(opts.q[0])(
            parameters=list(self._q_fns[0].parameters())+list(self._q_fns[1].parameters()),
            lr=opts.q[1],
            weight_decay=self._weight_decays,
        )
        if self._automatic_entropy_tuning:
            self._alpha_optimizer = utils.get_optimizer(opts.alpha[0])(
                parameters=self._log_alpha_fn.parameters(),
                lr=opts.alpha[1],
                weight_decay=self._weight_decays,
            )
        else:
            self.log_alpha = None
    
    def _build_alpha_loss(self, batch: Dict) -> Tuple:
        states = batch['s1']
        actions = batch['a1']
        rewards = batch['reward']
        next_states = batch['s2']
        dsc = batch['dsc']

        log_pi = self.log_pi

        if self._automatic_entropy_tuning:
            alpha_loss = -(self._log_alpha_fn() * (log_pi + self._target_entropy).detach()).mean()
            self.alpha = self._log_alpha_fn().exp() * self._alpha_multiplier
        else:
            alpha_loss = states.new_tensor(0.0)
            self.alpha = states.new_tensor(self._alpha_multiplier)

        info = collections.OrderedDict()
        info['alpha'] = self.alpha
        if self._automatic_entropy_tuning:
            info['alpha_loss'] = alpha_loss
            return alpha_loss, info
        else:
            return 0, info

    def _build_q_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']
        actions = batch['a1']
        rewards = batch['reward']
        next_states = batch['s2']
        dsc = batch['dsc']
        
        qf1_pred = self._q_fns[0](states, actions)
        qf2_pred = self._q_fns[1](states, actions)

        new_next_actions, next_log_pi = self._p_fn(next_states)

        target_q_values = torch.min(
            self._q_target_fns[0](next_states, new_next_actions),
            self._q_target_fns[1](next_states, new_next_actions),
        )

        if self._backup_entropy:
            target_q_values = target_q_values - self.alpha * next_log_pi

        q_target = self._reward_scale * rewards + dsc * self._discount * target_q_values

        qf1_loss = F.mse_loss(qf1_pred, q_target.detach())
        qf2_loss = F.mse_loss(qf2_pred, q_target.detach())
        qf_loss = qf1_loss + qf2_loss

        info = collections.OrderedDict()
        info['Q1_loss'] = qf1_loss.detach().mean()
        info['Q2_loss'] = qf2_loss.detach().mean()
        info['Q_loss'] = qf_loss.detach().mean()
        info['average_qf1'] = qf1_pred.detach().mean()
        info['average_qf2'] = qf2_pred.detach().mean()
        info['average_target_q'] = target_q_values.detach().mean()
        
        return qf_loss, info
    
    def _build_p_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']
        actions = batch['a1']
        rewards = batch['reward']
        next_states = batch['s2']
        dsc = batch['dsc']

        new_actions = self.new_actions
        log_pi = self.log_pi

        q_new_actions = torch.min(
            self._q_fns[0](states, new_actions),
            self._q_fns[1](states, new_actions),
        )
        p_loss = (self.alpha * log_pi - q_new_actions).mean()

        info = collections.OrderedDict()
        info['actor_loss'] = p_loss.detach().mean()
        info['log_pi'] = log_pi.detach().mean()
        return p_loss, info

    def _get_train_batch(self) -> Dict:
        """Sample two batches of transitions from real dataset and sim replay buffer respectively."""
        # periodically rollout transitions from sim env
        if self._global_step % self._rollout_sim_freq == 0:
            with torch.no_grad():
                self._traj_steps = 0
                # self._current_state = self._env.reset(seed=self._env_seed) # use it for debug
                self._current_state = self._env.reset()
                for _ in trange(self._rollout_sim_num):
                    self._traj_steps += 1
                    state = self._current_state
                    action = self._sampler_policy(state)
                    if self._joint_noise_std > 0:
                        next_state, reward, done, __ = self._env.step(
                            action + np.random.randn(action.shape[0], ) * self._joint_noise_std)
                    else:
                        next_state, reward, done, __ = self._env.step(action)

                    self._empty_dataset.add(state=state, action=action, next_state=next_state, next_action=0,
                                            reward=reward, done=done)
                    self._current_state = next_state

                    if done or self._traj_steps >= self._max_traj_length:
                        self._traj_steps = 0
                        self._current_state = self._env.reset()

        _sim_batch = self._empty_dataset.sample_batch(self._batch_size)
        
        return _sim_batch

    def _optimize_step(self, batch: Dict) -> Dict:
        info = collections.OrderedDict()

        self.new_actions, self.log_pi = self._p_fn(batch['s1'])
        alpha_loss, alpha_info = self._build_alpha_loss(batch)
        if self._global_step % self._update_actor_freq == 0:
            p_loss, self._p_info = self._build_p_loss(batch)
        q_loss, q_info = self._build_q_loss(batch)

        if self._automatic_entropy_tuning:
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        if self._global_step % self._update_actor_freq == 0:
            self._p_optimizer.zero_grad()
            p_loss.backward()
            self._p_optimizer.step()

        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()

        if self._global_step % self._target_update_period == 0:
            self._update_target_fns(self._q_fns, self._q_target_fns)
            self._update_target_fns(self._p_fn, self._p_target_fn)
        
        info.update(alpha_info)
        info.update(q_info)
        info.update(self._p_info)
        return info
    
    def _build_test_policies(self) -> None:
        policy = self._sampler_policy
        self._test_policies['main'] = policy
    
    def save(self, ckpt_name: str) -> None:
        pass

    def restore(self, ckpt_name: str) -> None:
        pass


class AgentModule(BaseAgentModule):
    def _build_modules(self) -> None:
        device = self._net_modules.device
        automatic_entropy_tuning = self._net_modules.automatic_entropy_tuning
        self._p_net = self._net_modules.p_net_factory().to(device)
        self._q_nets = nn.ModuleList()
        n_q_fns = self._net_modules.n_q_fns
        for _ in range(n_q_fns):
            self._q_nets.append(self._net_modules.q_net_factory().to(device))
        self._q_target_nets = copy.deepcopy(self._q_nets)    
        self._p_target_net = copy.deepcopy(self._p_net)
        if automatic_entropy_tuning:
            self._log_alpha_net = self._net_modules.log_alpha_net_factory().to(device)
        
    @property
    def q_nets(self) -> nn.ModuleList:
        return self._q_nets
    
    @property
    def q_target_nets(self) -> nn.ModuleList:
        return self._q_target_nets
    
    @property
    def p_net(self) -> nn.Module:
        return self._p_net
    
    @property
    def p_target_net(self) -> nn.Module:
        return self._p_target_net
    
    @property
    def log_alpha_net(self) -> nn.Module:
        return self._log_alpha_net