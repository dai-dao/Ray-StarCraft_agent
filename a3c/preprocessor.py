import numpy as np
from collections import namedtuple
from pysc2.lib import actions
from pysc2.lib import features
from ray.rllib.models.preprocessors import Preprocessor

from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.lib.actions import TYPES as ACTION_TYPES

NUM_FUNCTIONS = len(actions.FUNCTIONS)
NUM_PLAYERS = features.SCREEN_FEATURES.player_id.scale


is_spatial_action = {}
for name, arg_type in actions.TYPES._asdict().items():
    # HACK: we should infer the point type automatically
    is_spatial_action[arg_type.name] = name in ['minimap', 'screen', 'screen2']


def stack_ndarray_dicts(lst, axis=0):
    """ Concatenate ndarray values from list of dicts
    along new axis."""
    res = {}
    for k in lst[0].keys():
        res[k] = np.stack([d[k] for d in lst], axis=axis)
    return res


def flatten_first_dims_dict(x):
    return {k: flatten_first_dims(v) for k, v in x.items()}


def flatten_first_dims(x):
    new_shape = [x.shape[0] * x.shape[1]] + list(x.shape[2:])
    return x.reshape(*new_shape)


def stack_and_flatten_actions(lst, axis=0):
    fn_id_list, arg_dict_list = zip(*lst)
    fn_id = np.stack(fn_id_list, axis=axis)
    fn_id = flatten_first_dims(fn_id)
    arg_ids = stack_ndarray_dicts(arg_dict_list, axis=axis)
    arg_ids = flatten_first_dims_dict(arg_ids)
    return (fn_id, arg_ids)


def mask_unused_argument_samples(actions):
    """ Replace sampled argument id by -1 for all arguments not used
        in a steps action (in-place).
    """
    fn_id, arg_ids = actions
    for n in range(fn_id.shape[0]):
        a_0 = fn_id[n]
        unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[a_0].args)
        for arg_type in unused_types:
            arg_ids[arg_type.name][n] = -1
    return (fn_id, arg_ids)


SVSpec = namedtuple('SVSpec', ['type', 'index', 'scale'])
screen_specs_sv = [SVSpec(features.FeatureType.SCALAR, 0 , 256.),
                   SVSpec(features.FeatureType.CATEGORICAL, 1, 4),
                   SVSpec(features.FeatureType.CATEGORICAL, 2, 2),
                   SVSpec(features.FeatureType.CATEGORICAL, 3, 2),
                   SVSpec(features.FeatureType.CATEGORICAL, 5, 5),
                   SVSpec(features.FeatureType.CATEGORICAL, 6, 1850),
                   SVSpec(features.FeatureType.SCALAR, 14, 16.),
                   SVSpec(features.FeatureType.SCALAR, 15, 256.)]


minimap_specs_sv = [SVSpec(features.FeatureType.SCALAR, 0 , 256.),
                    SVSpec(features.FeatureType.CATEGORICAL, 1 , 4),
                    SVSpec(features.FeatureType.CATEGORICAL, 2 , 2),
                    SVSpec(features.FeatureType.CATEGORICAL, 5 , 5)]


flat_specs_sv = [SVSpec(features.FeatureType.SCALAR, 0, 1.),
                 SVSpec(features.FeatureType.SCALAR, 1, 1.),
                 SVSpec(features.FeatureType.SCALAR, 2, 1.),
                 SVSpec(features.FeatureType.SCALAR, 3, 1.),
                 SVSpec(features.FeatureType.SCALAR, 4, 1.),
                 SVSpec(features.FeatureType.SCALAR, 5, 1.),
                 SVSpec(features.FeatureType.SCALAR, 6, 1.),
                 SVSpec(features.FeatureType.SCALAR, 7, 1.),
                 SVSpec(features.FeatureType.SCALAR, 8, 1.),
                 SVSpec(features.FeatureType.SCALAR, 9, 1.),
                 SVSpec(features.FeatureType.SCALAR, 10, 1.)]


class StarCraftPreprocessor(Preprocessor):
    """Compute network inputs from pysc2 observations.
        See https://github.com/deepmind/pysc2/blob/master/docs/environment.md
        for the semantics of the available observations.
    """   
    def __init__(self, ob_space, options):
        self.screen_channels = len(features.SCREEN_FEATURES)
        self.minimap_channels = len(features.MINIMAP_FEATURES)
        self.flat_channels = len(flat_specs_sv)
        self.available_actions_channels = NUM_FUNCTIONS
        self.shape = (10, 10) # RAY API 


    def get_input_channels(self):
        """Get static channel dimensions of network inputs."""
        return {
            'screen': self.screen_channels,
            'minimap': self.minimap_channels,
            'flat': self.flat_channels,
            'available_actions': self.available_actions_channels}


    # RAY API
    def transform(self, ob):
        return stack_ndarray_dicts([self._preprocess_obs_sv(ob.observation)])     


    def preprocess_obs(self, obs_list):
        return stack_ndarray_dicts(
            [self._preprocess_obs(o.observation) for o in obs_list])


    def _preprocess_obs_sv(self, obs):
            chosen_screen_index = [0, 1, 2, 3, 5, 6, 14, 15]
            chosen_minimap_index = [0, 1, 2, 5]
            screen = np.stack([obs['screen'][i] for i in chosen_screen_index], axis=0)
            minimap = np.stack([obs['minimap'][i] for i in chosen_minimap_index], axis=0)
            flat = np.concatenate([obs['player']])
            available_actions = np.zeros(NUM_FUNCTIONS, dtype=np.float32)
            available_actions[obs['available_actions']] = 1 
            return {'screen': screen,
                    'minimap': minimap,
                    'flat': flat, 
                    'available_actions': available_actions}


    def _preprocess_obs(self, obs):
        """Compute screen, minimap and flat network inputs from raw observations.
        """
        available_actions = np.zeros(NUM_FUNCTIONS, dtype=np.float32)
        available_actions[obs['available_actions']] = 1

        screen = self._preprocess_spatial(obs['screen'])
        minimap = self._preprocess_spatial(obs['minimap'])

        flat = np.concatenate([
            obs['player']])
            # TODO available_actions, control groups, cargo, multi select, build queue

        return {
            'screen': screen,
            'minimap': minimap,
            'flat': flat,
            'available_actions': available_actions}


    def _preprocess_spatial(self, spatial):
        return spatial


