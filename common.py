import gym
import numpy as np
import torch
import os
__all__ = ["make_env", "create_folders", "get_frame_skip_and_timestep"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Make environment using its name
def make_env(env_name, seed, time_change_factor, env_timestep, frameskip, delayed_env):
    env = gym.make(env_name)
    if delayed_env:
        env = Float64ToFloat32(env)
        env = RealTimeWrapper(env, env_name)
        env.env.env._max_episode_steps = 1000 * time_change_factor
        env.env.env.frame_skip = int(frameskip)
        env.env.env.env.frame_skip = int(frameskip)

    env.seed(seed)
    env.delayed = delayed_env
    env.action_space.seed(seed)
    env.model.opt.timestep = env_timestep
    # env.env.env.frame_skip = int(frameskip)
    env.env.frame_skip = int(frameskip)
    env.frame_skip = int(frameskip)

    return env


class RealTimeWrapper(gym.Wrapper):
    def __init__(self, env, env_name):
        super().__init__(env)
        self.observation_space = gym.spaces.Tuple((env.observation_space, env.action_space))
        # self.initial_action = env.action_space.sample()
        assert isinstance(env.action_space, gym.spaces.Box)
        self.initial_action = env.action_space.high * 0
        self.previous_action = self.initial_action

    def reset(self):
        self.previous_action = self.initial_action
        return np.concatenate((super().reset(), self.previous_action), axis=0)

    def step(self, action):
        observation, reward, done, info = super().step(self.previous_action)
        self.previous_action = action
        return np.concatenate((observation, action), axis=0), reward, done, info


class Float64ToFloat32(gym.ObservationWrapper):
  """Converts np.float64 arrays in the observations to np.float32 arrays."""


  def observation(self, observation):
    observation = deepmap({np.ndarray: float64_to_float32}, observation)
    return observation

  def step(self, action):
    s, r, d, info = super().step(action)
    return s, r, d, info


def deepmap(f, m):
  """Apply functions to the leaves of a dictionary or list, depending type of the leaf value.
  Example: deepmap({torch.Tensor: lambda t: t.detach()}, x)."""
  for cls in f:
    if isinstance(m, cls):
      return f[cls](m)
  if isinstance(m, Sequence):
    return type(m)(deepmap(f, x) for x in m)
  elif isinstance(m, Mapping):
    return type(m)((k, deepmap(f, m[k])) for k in m)
  else:
    raise AttributeError()


def float64_to_float32(x):
    return np.asarray(x, np.float32) if x.dtype == np.float64 else x


def create_folders():
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")
