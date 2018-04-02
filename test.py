import ray
from ray.rllib.optimizers.local_sync_replay import LocalSyncReplayOptimizer
from ray.rllib.optimizers.apex_optimizer import ApexOptimizer

import gym

from evaluator import AgentEvaluator
from rllib_agent import DEFAULT_CONFIG as rainbow_config
from ray.rllib.utils.atari_wrappers import wrap_deepmind


if __name__ == '__main__':
    ray.init()
    env_creator = lambda env_config: gym.make("PongNoFrameskip-v4")
    # optimizer = LocalSyncReplayOptimizer.make(
    #     AgentEvaluator, [rainbow_config, env_creator], 0, {})
## Uncomment to use the Ape-X optimizer with 8 workers
    optimizer = ApexOptimizer.make(
        AgentEvaluator, [rainbow_config, env_creator], 3, {})
    last_target_update = 0
    i = 0
    while optimizer.num_steps_sampled < 100000:
        i += 1
        print("== optimizer step {} ==".format(i))
        optimizer.step()
        if optimizer.num_steps_sampled - last_target_update > 1000:
            last_target_update = optimizer.num_steps_sampled
            optimizer.local_evaluator.update_target()
        print("optimizer stats", optimizer.stats())
        print("local evaluator stats", optimizer.local_evaluator.stats())
        for ev in optimizer.remote_evaluators[:1]:
            print("remote evaluator stats", ray.get(ev.stats.remote()))
        print()