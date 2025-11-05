"""An implementation of the Env for standard Gym / Gymnasium environments."""

import numpy as np
import gym
from gym.spaces import Space
from typing import Tuple, Any, Union
from d2c.envs import BaseEnv
from d2c.utils.wrappers import wrapped_norm_obs_env
from types import SimpleNamespace 

class GymEnv(BaseEnv):

    def __init__(
        self,
        env_name: str,
        obs_shift: np.ndarray = None,
        obs_scale: np.ndarray = None,
    ) -> None:
        self._env_name = env_name
        self._obs_shift = obs_shift
        self._obs_scale = obs_scale
        self._load_model()
        super(GymEnv, self).__init__()

    def _load_model(self):
        gym_env = gym.make(self._env_name) 
        self._env = wrapped_norm_obs_env(
            gym_env=gym_env,
            shift=self._obs_shift,
            scale=self._obs_scale,
        )

    def _set_action_space(self) -> Space:
        self.action_space = self._env.action_space
        return self.action_space

    def _set_observation_space(self) -> Space:
        self.observation_space = self._env.observation_space
        return self.observation_space

    def step(self, a: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Run one step of environment dynamics."""
        obs, reward, done, info = self._env.step(a)
        done = done   
        return obs, reward, done, info

    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """Reset the environment."""
        obs = self._env.reset(**kwargs)
        return obs
    
    @staticmethod
    def make_env_space(env_name: str, **kwargs):
        """Return observation and action space info for the env."""
        env = gym.make(env_name)
        obs_space = env.observation_space
        act_space = env.action_space

        environment_space = SimpleNamespace(
            observation=obs_space,
            action=act_space
        )
        env.close()
        return environment_space