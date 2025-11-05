import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../")))
from d2c.utils.config import ConfigBuilder, update_config
from d2c.utils.utils import abs_file_path
from example.benchmark.config.app_config import app_config
import copy
import logging
import json5
import numpy as np
from easydict import EasyDict
from typing import Union, Optional, Dict, Any, Tuple, Generator, Callable, List
from d2c.utils.utils import Flags
from d2c.envs import benchmark_env
from d2c.models.base import BaseAgent
import torch

# class DiscreteBaseAgent(BaseAgent):
#    def __init__(self, env, model_params, optimizers, train_data, batch_size = 64, weight_decays = 0, update_freq = 1, update_rate = 0.005, discount = 0.99, empty_dataset = None, device = None):
#        super().__init__(env, model_params, optimizers, train_data, batch_size, weight_decays, update_freq, update_rate, discount, empty_dataset, device)
#        if hasattr(self._action_space, "high"):  # Box
#            self._a_max = torch.tensor(self._action_space.high, device=device, dtype=torch.float32)
#            self._a_min = torch.tensor(self._action_space.low, device=device, dtype=torch.float32)
#            self._a_dim = self._action_space.shape[0]
#        elif hasattr(self._action_space, "n"):  # Discrete
#            self._a_max = torch.tensor(float(self._action_space.n - 1), device=device)
#            self._a_min = torch.tensor(0.0, device=device)
#            self._a_dim = self._action_space.n
#        else:
#            raise ValueError(f"Unsupported action space type: {type(self._action_space)}")


class DiscreteConfigBuilder(ConfigBuilder):
    def __init__(self, app_config, model_config_path, work_abs_dir, command_args = None, experiment_type = 'benchmark'):
        super().__init__(app_config, model_config_path, work_abs_dir, command_args, experiment_type)
    
    def _update_model_cfg(self) -> None:
        def _convert_numpy(obj):

            if isinstance(obj, dict):
                return {k: _convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert_numpy(x) for x in obj]
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        self._model_cfg = update_config(self._model_cfg_path, self._command_args)
        self._env_info = self._get_env_info()
        self._update_env_info()
        self._update_model_dir()

        logging.debug('-' * 20 + " The config of this experiment " + '-' * 20)

        _m_cfg = copy.deepcopy(self._model_cfg)
        _m_cfg = _convert_numpy(_m_cfg)   
        logging.debug(json5.dumps(_m_cfg, indent=2, ensure_ascii=False))
    def _get_env_space(
        self,
        benchmark_name: str,
        data_source: str,
        env_name: str,
        **kwargs: Any
    ) -> Tuple:
        env_class = benchmark_env(benchmark_name=benchmark_name)
        environment_space = env_class.make_env_space(
            data_source=data_source,
            env_name=env_name,
            **kwargs,
        )
        if isinstance(environment_space, tuple):
            observation_space, action_space = environment_space
        else:
            observation_space = getattr(environment_space, 'observation', None)
            action_space = getattr(environment_space, 'action', None)

        try:
            state_dim = np.prod(observation_space.shape)
            state_min = getattr(observation_space, 'low', -np.inf)
            state_max = getattr(observation_space, 'high', np.inf)
            state_info = (state_dim, state_min, state_max)
        except Exception:
            state_info = (1, -np.inf, np.inf)

        if hasattr(action_space, 'shape') and len(action_space.shape) > 0:
            a_dim = int(action_space.shape[0])
            a_min = getattr(action_space, 'low', -1.0)
            a_max = getattr(action_space, 'high', 1.0)
        elif hasattr(action_space, 'n'):
            a_dim = 1
            a_min = 0
            a_max = action_space.n - 1
        else:
            raise TypeError(f"Unsupported action space type: {type(action_space)}")

        action_info = (a_dim, a_min, a_max)

        return state_info, action_info

    def _update_env_info(self) -> None:
        # Update the env basic_info
        self._update_env_basic_info()
        if self._exp_type == 'benchmark':
            # update env parameters
            data_file_path = os.path.join(
                self._work_abs_dir,
                'data',
                self._env_ext.benchmark_name,
                self._env_ext.data_source,
                self._env_ext.data_name
            )
            # Update the external env info
            temp_dict = dict([('score_norm_min', self._env_info.norm_min),
                              ('score_norm_max', self._env_info.norm_max),
                              ('data_file_path', data_file_path)])
            for k, v in temp_dict.items():
                if self._model_cfg.env.external[k] is None:
                    self._model_cfg.env.external[k] = v

def make_config(command_args=None):
    work_abs_dir = abs_file_path(__file__, '../../example/benchmark')
    model_config_path = os.path.join(work_abs_dir, 'config', 'model_config.json5')
    cfg_builder = DiscreteConfigBuilder(
        app_config=app_config,
        model_config_path=model_config_path,
        work_abs_dir=work_abs_dir,
        command_args=command_args,
    )
    return cfg_builder.build_config()

import d2c.envs.base as base_env
import logging


def patch_dqn_for_discrete():
    """Add discrete supports for DQN"""
    _patch_env_base()
    logging.info("[dqn_patch_utils] DQN discrete action patch applied.")


def _patch_env_base():
    if getattr(base_env.BaseEnv, "_patched_for_discrete", False):
        return
    old_init = base_env.BaseEnv.__init__

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        if hasattr(self._action_space, "n"):  
            self._a_dim = self._action_space.n
        elif hasattr(self._action_space, "shape"):
            self._a_dim = self._action_space.shape[0]
        else:
            raise ValueError(f"[dqn_patch_utils] Unknown action space type: {type(self._action_space)}")

    base_env.BaseEnv.__init__ = new_init
    base_env.BaseEnv._patched_for_discrete = True

