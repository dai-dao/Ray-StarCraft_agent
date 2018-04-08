from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from a3c.base_policy import Policy
from torch.autograd import Variable
from threading import Lock


class TorchPolicy(Policy):
    def __init__(self, config, name="local", summarize=True):
        # RAY API
        self.local_steps = 0
        self.config = config
        self.summarize = summarize
        torch.set_num_threads(2)
        self.lock = Lock()


    def apply_gradients(self, grads):
        self.optimizer.zero_grad()
        for g, p in zip(grads, self._model.get_trainable_params()):
            p.grad = Variable(torch.from_numpy(g))
        self.optimizer.step()


    def get_weights(self):
        out = {}
        params = self._model.get_trainable_params(with_id=True)
        for k, v in params.items():
            out[k] = v.state_dict()
        return out


    def set_weights(self, weights):
        with self.lock:
            params = self._model.get_trainable_params(with_id=True)
            for k, v in params.items():
                v.load_state_dict(weights[k])


    def compute_gradients(self, samples):
        with self.lock:
            self._backward(samples)
            # Note that return values are just references;
            # calling zero_grad will modify the values
            return [p.grad.data.cpu().numpy() for p in self._model.get_trainable_params()], {}


    def model_update(self, batch):
        """
        Implements compute + apply
        """
        with self.lock:
            self._backward(batch)
            self.optimizer.step()


    def _backward(self, batch):
        raise NotImplementedError