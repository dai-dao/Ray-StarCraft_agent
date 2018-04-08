from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical

from a3c.model import FullyConv
from a3c.torch_policy import TorchPolicy
from a3c.preprocessor import stack_ndarray_dicts, flatten_first_dims_dict, \
                            stack_and_flatten_actions, flatten_first_dims, \
                            mask_unused_argument_samples


def convert_batch(trajectory):
    obs = list(trajectory['observations'])
    obs = flatten_first_dims_dict(stack_ndarray_dicts(obs))
    actions = stack_and_flatten_actions(list(trajectory['actions']))
    returns = np.reshape(trajectory['value_targets'], (-1, 1))
    advs = np.reshape(trajectory['advantages'], (-1, 1))
    return obs, actions, advs, returns


class SharedTorchPolicy(TorchPolicy):
    other_output = ["vf_preds"]
    is_recurrent = False

    def __init__(self, config, **kwargs):
        super(SharedTorchPolicy, self).__init__(config, **kwargs)
        # Config network
        self.dtype = torch.FloatTensor
        self.atype = torch.LongTensor
        self._model = FullyConv(config, supervised=True)
        if torch.cuda.is_available():
            print('CUDA-enabled')
            self.dtype = torch.cuda.FloatTensor
            self.atype = torch.cuda.LongTensor
            self._model.cuda()
        # Optimizer
        self.optimizer = torch.optim.Adam(self._model.get_trainable_params(), 
                                          lr=self.config["lr"])


    def _make_var(self, input_dict):
        new_dict = {}
        for k, v in input_dict.items():
            new_dict[k] = Variable(self.dtype(v))
        return new_dict

    
    def _sample_actions(self, available_actions, policy):
        fn_pi, arg_pis = policy
        fn_pi = self._mask_unavailable_actions(available_actions, fn_pi)  
        # Sample actions
        # Avoid the case where the sampled action is NOT available
        while True:
            fn_samples = self._sample(fn_pi)
            if (available_actions.gather(1, fn_samples.unsqueeze(1)) == 1).all():
                fn_samples = fn_samples.data.cpu().numpy()
                break
        arg_samples = dict()
        for arg_type, arg_pi in arg_pis.items():
            arg_samples[arg_type] = self._sample(arg_pi).data.cpu().numpy()
        return fn_samples, arg_samples


    def _mask_unavailable_actions(self, available_actions, fn_pi):
        fn_pi = fn_pi * available_actions
        fn_pi = fn_pi / fn_pi.sum(1, keepdim=True)
        return fn_pi

    
    def _sample(self, probs):
        dist = Categorical(probs=probs)
        return dist.sample()


    def compute(self, ob, *args):
        """Should take in a SINGLE ob"""
        with self.lock:
            ob_var = self._make_var(ob)
            policy, value = self._model.forward(ob_var["screen"], ob_var["minimap"], \
                                                ob_var["flat"])
            available_actions = ob_var["available_actions"]
            samples = self._sample_actions(available_actions, policy)
            samples = mask_unused_argument_samples(samples)
            return samples, {"vf_preds" : value.cpu().data.numpy()[0]}


    def compute_logits(self, ob, *args):
        with self.lock:
            ob_var = self._make_var(ob)
            policy, _ = self._model.forward(ob_var["screen"], ob_var["minimap"], \
                                            ob_var["flat"])
            return policy

        
    def value(self, ob, *args):
        with self.lock:
            ob_var = self._make_var(ob)
            _, value = self._model.forward(ob_var["screen"], ob_var["minimap"], \
                                            ob_var["flat"])
            return value.cpu().data.numpy()[0]
        
    
    def _evaluate(self, obs, actions):
        """Passes in multiple obs."""
        obs_var = self._make_var(obs)
        policy, values = self._model.forward(obs_var["screen"], \
                                            obs_var["minimap"], obs_var["flat"])
        actions_var = self._make_var_actions(actions)
        available_actions = obs_var["available_actions"]
        log_probs = self._compute_policy_log_probs(available_actions, policy, actions_var)
        entropy = self._compute_policy_entropy(available_actions, policy, actions_var)
        return values, log_probs, entropy


    def _backward(self, batch):
        obs, actions, advs, returns = convert_batch(batch)
        values, ac_logprobs, entropy = self._evaluate(obs, actions)
        advs_var = Variable(self.dtype(advs))
        returns_var = Variable(self.dtype(returns))
        # Loss calculation
        policy_loss = -(advs_var * ac_logprobs).mean()
        value_loss = (returns_var - values).pow(2).mean()
        loss = policy_loss + value_loss * self.config['value_loss_weight'] \
                - entropy * self.config['entropy_weight']

        print('LOSS', loss)
        
        # Backward and clip grad
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self._model.get_trainable_params(), \
                                      self.config["grad_clip"])

    
    def _make_var_actions(self, actions):
        n_id, arg_ids = actions 
        args_var = {}
        fn_id_var = Variable(self.atype(n_id))
        for k, v in arg_ids.items():
            args_var[k] = Variable(self.atype(v))
        return fn_id_var, args_var


    def _compute_policy_log_probs(self, available_actions, policy, actions_var):
        def logclip(x):
            return torch.log(torch.clamp(x, 1e-12, 1.0))

        def compute_log_probs(probs, labels):
            new_labels = labels.clone()
            new_labels[new_labels < 0] = 0
            selected_probs = probs.gather(1, new_labels.unsqueeze(1))
            out = logclip(selected_probs)
            # Log of 0 will be 0
            # out[selected_probs == 0] = 0
            return out.view(-1)

        fn_id, arg_ids = actions_var
        fn_pi, arg_pis = policy
        fn_pi = self._mask_unavailable_actions(available_actions, fn_pi)
        fn_log_prob = compute_log_probs(fn_pi, fn_id)

        log_prob = fn_log_prob
        for arg_type in arg_ids.keys():
            arg_id = arg_ids[arg_type]
            arg_pi = arg_pis[arg_type]
            arg_log_prob = compute_log_probs(arg_pi, arg_id)

            arg_id_masked = arg_id.clone()
            arg_id_masked[arg_id_masked != -1] = 1
            arg_id_masked[arg_id_masked == -1] = 0
            arg_log_prob = arg_log_prob * arg_id_masked.float()
            log_prob = log_prob + arg_log_prob
        return log_prob


    def _compute_policy_entropy(self, available_actions, policy, actions_var):
        def logclip(x):
            return torch.log(torch.clamp(x, 1e-12, 1.0))
        
        def compute_entropy(probs):
            return -(logclip(probs) * probs).sum(-1)
        
        _, arg_ids = actions_var    
        fn_pi, arg_pis = policy
        fn_pi = self._mask_unavailable_actions(available_actions, fn_pi)

        entropy = compute_entropy(fn_pi).mean()
        for arg_type in arg_ids.keys():
            arg_id = arg_ids[arg_type]
            arg_pi = arg_pis[arg_type]

            batch_mask = arg_id.clone()
            batch_mask[batch_mask != -1] = 1
            batch_mask[batch_mask == -1] = 0
            # Reference: https://discuss.pytorch.org/t/how-to-use-condition-flow/644/4
            if (batch_mask == 0).all():
                arg_entropy = (compute_entropy(arg_pi) * 0.0).sum()
            else:
                arg_entropy = (compute_entropy(arg_pi) * batch_mask.float()).sum() / batch_mask.float().sum()
            entropy = entropy + arg_entropy
        return entropy