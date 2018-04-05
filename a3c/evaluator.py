import pickle

import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.optimizers import PolicyEvaluator
from ray.rllib.a3c.common import get_policy_cls
from ray.rllib.utils.filter import get_filter, Filter
from ray.rllib.utils.sampler import AsyncSampler
from ray.rllib.utils.process_rollout import process_rollout

from a3c.shared_torch_policy import SharedTorchPolicy


class MyNoFilter(Filter):
    is_concurrent = True

    def __init__(self, *args):
        pass

    def __call__(self, x, update=True):
        return x

    def apply_changes(self, other, *args, **kwargs):
        pass

    def copy(self):
        return self

    def sync(self, other):
        pass

    def clear_buffer(self):
        pass

    def as_serializable(self):
        return self


class A3CEvaluator(PolicyEvaluator):
    def __init__(self, registry, env_creator, config, logdir, start_sampler=True):
        self.env = ModelCatalog.get_preprocessor_as_wrapper(
            registry, env_creator(config["env_config"]), config["model"])
        self.config = config
        self.policy = SharedTorchPolicy(config)
        # Technically not needed when not remote
        self.filter = MyNoFilter()
        # Observation sampler
        self.sampler = AsyncSampler(self.env, self.policy, \
                                    self.filter, config["batch_size"])
        # Misc
        if start_sampler and self.sampler.async:
            self.sampler.start()
        self.logdir = logdir        


    def sample(self):
        rollout = self.sampler.get_data()
        samples = process_rollout(rollout, self.filter, gamma=self.config["gamma"], \
                                  lambda_=self.config["lambda"], use_gae=True)
        return samples


    def get_completed_rollout_metrics(self):
        """ Returns metrics on previously completed rollouts.
            Calling this clears the queue of completed rollout metrics.
        """
        return self.sampler.get_metrics()


    def compute_gradients(self, samples):
        gradient, info = self.policy.compute_gradients(samples)
        return gradient, {}


    def apply_gradients(self, grads):
        self.policy.apply_gradients(grads)


    def get_weights(self):
        return self.policy.get_weights()


    def set_weights(self, params):
        self.policy.set_weights(params)


    def save(self):
        weights = self.get_weights()
        return pickle.dumps({"weights": weights})


    def restore(self, objs):
        objs = pickle.loads(objs)
        self.set_weights(objs["weights"])


RemoteA3CEvaluator = ray.remote(A3CEvaluator)
GPURemoteA3CEvaluator = ray.remote(num_gpus=1)(A3CEvaluator)