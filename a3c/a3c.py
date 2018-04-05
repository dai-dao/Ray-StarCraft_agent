from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import os
import torch

import ray
from ray.rllib.agent import Agent
from ray.rllib.optimizers import AsyncOptimizer
from ray.tune.result import TrainingResult

from a3c.evaluator import A3CEvaluator, RemoteA3CEvaluator, GPURemoteA3CEvaluator


DEFAULT_CONFIG = {
    # Number of workers (excluding master)
    "num_workers": 1,
    # Size of rollout batch
    "batch_size": 32,
    # Discount factor of MDP
    "gamma": 0.99,
    # GAE(gamma) parameter
    "lambda": 1.0,
    # Max global norm for each gradient calculated by worker
    "grad_clip": 40.0,
    # Learning rate
    "lr": 0.0001,
    # Value Function Loss coefficient
    "value_loss_weight": 0.5,
    # Entropy coefficient
    "entropy_weight": -0.01,
    # Whether to place workers on GPUs
    "use_gpu_for_workers": False,
    # Model and preprocessor options
    "model": {
            "custom_preprocessor": "sc_prep",
        },
    # Arguments to pass to the rllib optimizer
    "optimizer": {
        # Number of gradients applied for each `train` step
        "grads_per_step": 100,
    },
    # Arguments to pass to the env creator
    "env_config": {
        'res' : 64,
        'map' : 'MoveToBeacon',
        'step_mul' : 8
    },
}


class A3CAgent(Agent):
    _agent_name = "SC_A3C"
    _default_config = DEFAULT_CONFIG
    _allow_unknown_subkeys = ["model", "optimizer", "env_config"]


    def _init(self):
        self.local_evaluator = A3CEvaluator(self.registry, \
                                            self.env_creator, self.config, \
                                            self.logdir, start_sampler=False)
        if self.config["use_gpu_for_workers"]:
            remote_cls = GPURemoteA3CEvaluator
        else:
            remote_cls = RemoteA3CEvaluator
        self.remote_evaluators = [remote_cls.remote(self.registry, 
                                    self.env_creator, self.config, self.logdir) \
                                    for i in range(self.config["num_workers"])]
        self.optimizer = AsyncOptimizer(self.config["optimizer"], 
                                        self.local_evaluator, self.remote_evaluators)


    def _train(self):
        self.optimizer.step()
        res = self._fetch_metrics_from_remote_evaluators()
        return res


    def _fetch_metrics_from_remote_evaluators(self):
        episode_rewards = []
        episode_lengths = []
        metric_lists = [a.get_completed_rollout_metrics.remote()
                        for a in self.remote_evaluators]
        for metrics in metric_lists:
            for episode in ray.get(metrics):
                episode_lengths.append(episode.episode_length)
                episode_rewards.append(episode.episode_reward)
        avg_reward = (
            np.mean(episode_rewards) if episode_rewards else float('nan'))
        avg_length = (
            np.mean(episode_lengths) if episode_lengths else float('nan'))
        timesteps = np.sum(episode_lengths) if episode_lengths else 0
        result = TrainingResult(
            episode_reward_mean=avg_reward,
            episode_len_mean=avg_length,
            timesteps_this_iter=timesteps,
            info={})
        return result


    def _stop(self):
        # workaround for https://github.com/ray-project/ray/issues/1516
        for ev in self.remote_evaluators:
            ev.__ray_terminate__.remote(ev._ray_actor_id.id())


    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(
            checkpoint_dir, "checkpoint-{}".format(self.iteration))
        agent_state = ray.get(
            [a.save.remote() for a in self.remote_evaluators])
        extra_data = {
            "remote_state": agent_state,
            "local_state": self.local_evaluator.save()}
        pickle.dump(extra_data, open(checkpoint_path + ".extra_data", "wb"))
        return checkpoint_path


    def _restore(self, checkpoint_path):
        extra_data = pickle.load(open(checkpoint_path + ".extra_data", "rb"))
        ray.get(
            [a.restore.remote(o) for a, o in zip(
                self.remote_evaluators, extra_data["remote_state"])])
        self.local_evaluator.restore(extra_data["local_state"])


    def compute_action(self, observation):
        action, info = self.local_evaluator.policy.compute(observation)
        return action