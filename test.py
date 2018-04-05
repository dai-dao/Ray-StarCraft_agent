import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.tune.registry import register_env
from ray.tune.config_parser import make_parser, resources_to_json
from ray.tune import register_trainable, run_experiments

from a3c.preprocessor import StarCraftPreprocessor
from env import StarCraft
from a3c.a3c import A3CAgent

import argparse
import torch
import sys


parser = make_parser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Train a reinforcement learning agent.",
    epilog="EXAMPLE_USAGE")


# See also the base parser definition in ray/tune/config_parser.py
parser.add_argument(
    "--redis-address", default=None, type=str,
    help="The Redis address of the cluster.")
parser.add_argument(
    "--num-cpus", default=2, type=int,
    help="Number of CPUs to allocate to Ray.")
parser.add_argument(
    "--num-gpus", default=1, type=int,
    help="Number of GPUs to allocate to Ray.")
parser.add_argument(
    "--experiment-name", default="default", type=str,
    help="Name of the subdirectory under `local_dir` to put results in.")
parser.add_argument(
    "--env", default=None, type=str, help="The gym environment to use.")


ModelCatalog.register_custom_preprocessor("sc_prep", StarCraftPreprocessor)
register_env("sc2", lambda config: StarCraft(config))
register_trainable("SC_A3C", A3CAgent)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    experiments = {
                'experiment_name': {
                    "run" : 'SC_A3C',
                    "env" : 'sc2',
                    "trial_resources" : resources_to_json(args.trial_resources),
                    "config": dict(args.config, env=args.env),
                }
            }
    ray.init(redis_address=args.redis_address, num_gpus=1, num_cpus=args.num_cpus)
    run_experiments(experiments)
    