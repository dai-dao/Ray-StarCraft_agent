import six.moves.queue as queue
import threading
from collections import namedtuple
import numpy as np
from ray.rllib.utils.sampler import CompletedRollout, _env_runner


class MyAsyncSampler(threading.Thread):
    """This class interacts with the environment and tells it what to do.
    Note that batch_size is only a unit of measure here. Batches can
    accumulate and the gradient can be calculated on up to 5 batches."""
    async = True

    def __init__(self, env, policy, obs_filter,
                 num_local_steps, horizon=None):
        assert getattr(obs_filter, "is_concurrent", False), (
            "Observation Filter must support concurrent updates.")
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.metrics_queue = queue.Queue()
        self.num_local_steps = num_local_steps
        self.horizon = horizon
        self.env = env
        self.policy = policy
        self._obs_filter = obs_filter
        self.started = False
        self.daemon = True

    def run(self):
        self.started = True
        try:
            self._run()
        except BaseException as e:
            self.queue.put(e)
            raise e

    def _run(self):
        rollout_provider = _env_runner(
            self.env, self.policy, self.num_local_steps,
            self.horizon, self._obs_filter)
        while True:
            # The timeout variable exists because apparently, if one worker
            # dies, the other workers won't die with it, unless the timeout is
            # set to some large number. This is an empirical observation.
            item = next(rollout_provider)
            if isinstance(item, CompletedRollout):
                self.metrics_queue.put(item)
            else:
                self.queue.put(item, timeout=600.0)

    def get_data(self):
        """Gets currently accumulated data.
        Returns:
            rollout (PartialRollout): trajectory data (unprocessed)
        """
        assert self.started, "Sampler never started running!"
        rollout = self.queue.get(timeout=600.0)
        if isinstance(rollout, BaseException):
            raise rollout
        while not rollout.is_terminal():
            try:
                part = self.queue.get_nowait()
                if isinstance(part, BaseException):
                    raise part
                rollout.extend(part)
            except queue.Empty:
                break
        return rollout

    def get_metrics(self):
        completed = []
        while True:
            try:
                completed.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return completed
