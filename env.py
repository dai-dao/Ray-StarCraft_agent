import numpy as np
import gym
from gym.spaces import Discrete, Box
from gym.envs.registration import EnvSpec

import ray
from ray.tune.registry import register_env
from ray.rllib import ppo

from pysc2.env import sc2_env
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.lib.actions import TYPES as ACTION_TYPES
# Workaround for pysc2 flags
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['tests.py'])

from a3c.preprocessor import is_spatial_action


def env_creator(env_config):
    import gym
    return gym.make("PongNoFrameskip-v4")  # or return your own custom env


def actions_to_pysc2(actions, size):
    """Convert agent action representation to FunctionCall representation."""
    height, width = size
    fn_id, arg_ids = actions
    actions_list = []
    for n in range(fn_id.shape[0]):
        a_0 = fn_id[n]
        a_l = []
        for arg_type in FUNCTIONS._func_list[a_0].args:
            arg_id = arg_ids[arg_type.name][n]
            if is_spatial_action[arg_type.name]:
                arg = [arg_id % width, arg_id // height]
            else:
                arg = [arg_id]
            a_l.append(arg)
        action = FunctionCall(a_0, a_l)
        actions_list.append(action)
    return actions_list


class StarCraft(gym.Env):
    def __init__(self, config):
        self.config = config
        self.size_px = (config['res'], config['res'])
        env_args = dict(
            map_name = config['map'],
            step_mul = config['step_mul'],
            game_steps_per_episode = 0,
            screen_size_px = self.size_px,
            minimap_size_px = self.size_px)
        self.env = sc2_env.SC2Env(**env_args)
        self._spec = EnvSpec("Sc2-{}-v0".format(config['map']))

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        pysc2_action = actions_to_pysc2(action, self.size_px)[0]
        ob_raw = self.env.step([pysc2_action])[0]
        return ob_raw, ob_raw.reward, ob_raw.last(), {}