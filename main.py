import multiprocessing
import sys

import ray
from ray.tune import register_trainable, run_experiments

from rllib_agent import RLLibAgent


if __name__ == '__main__':
    register_trainable('Agent', RLLibAgent)
    smoke_test = len(sys.argv) > 1 and sys.argv[1] == "--smoke"
    if smoke_test:
        ray.init(num_gpus=1)
    else:
        ray.init(redis_address="localhost:6379")

    run_experiments({
        "rainbow-apex-pong": {
            "run": "Agent",
            "env": "PongNoFrameskip-v4",
            "trial_resources": {
                "cpu": 4,
                "gpu": 1,
            },
            "config": {
                "num_workers": 4,
                "apex": True,
                "lr": .0001,
                "gamma": 0.99,
                "sample_batch_size": 50,
                "max_weight_sync_delay": 400,
                "train_batch_size": 512,
                "num_replay_buffer_shards": 1,
                "learning_starts": 1000 if smoke_test else 50000,
                "buffer_size": 50000 if smoke_test else 2000000,
                "target_network_update_freq": 1000 if smoke_test else 50000,
                "timesteps_per_iteration": 1000 if smoke_test else 25000,
            },
        },
    })