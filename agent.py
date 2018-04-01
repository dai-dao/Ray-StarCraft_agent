# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from model import DQN


class Agent():
  def __init__(self, args, action_space):
    self.action_space = action_space
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, args.atoms)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (args.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.priority_exponent = args.priority_exponent
    self.max_gradient_norm = args.max_gradient_norm

    self.policy_net = DQN(args, self.action_space)
    if args.model and os.path.isfile(args.model):
      self.policy_net.load_state_dict(torch.load(args.model))
    self.policy_net.train()

    self.target_net = DQN(args, self.action_space)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.policy_net.parameters(), lr=args.lr, eps=args.adam_eps)
    if args.cuda:
      self.policy_net.cuda()
      self.target_net.cuda()
      self.support = self.support.cuda()

  # Resets noisy weights in all linear layers (of policy net only)
  def reset_noise(self):
    self.policy_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    return (self.policy_net(state.unsqueeze(0)).data * self.support).sum(2).max(1)[1][0]

  # Acts with an ε-greedy policy
  def act_e_greedy(self, state, epsilon=0.001):
    return random.randrange(self.action_space) if random.random() < epsilon else self.act(state)

  def _batch_loss(self, sample_batch):
    batch_size = sample_batch.count
    states, actions, returns, next_states, nonterminals = (
        Variable(torch.from_numpy(np.array(sample_batch["obs"]))),
        torch.from_numpy(np.array(sample_batch["actions"])),
        torch.from_numpy(np.array(sample_batch["rewards"])).float(),
        Variable(torch.from_numpy(np.array(sample_batch["new_obs"]))),
        torch.from_numpy(
            np.ones_like(sample_batch["dones"]) - sample_batch["dones"]
        ).unsqueeze(1).float())

    if torch.cuda.is_available():
        states = states.cuda()
        actions = actions.cuda()
        returns = returns.cuda()
        next_states = next_states.cuda()
        nonterminals = nonterminals.cuda()

    return self._loss(states, actions, returns, next_states, nonterminals)

  def grad(self, sample_batch):
    loss = self._batch_loss(sample_batch)
    self.policy_net.zero_grad()
    (sample_batch["weights"] * loss).mean().backward()
    nn.utils.clip_grad_norm(self.policy_net.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
    return [p.grad.data.cpu().numpy() for p in self.policy_net.parameters()], loss.abs().data.cpu().numpy()

  def apply_grad(self, grads):
    if type(grads) is tuple:
        grads, _ = grads  # drop td_error
    self.optimiser.zero_grad()
    for g, p in zip(grads, self.policy_net.parameters()):
        p.grad = Variable(torch.from_numpy(g))
    self.optimiser.step()

  def compute_apply(self, sample_batch):
    loss = self._batch_loss(sample_batch)
    self.policy_net.zero_grad()
    weights = Variable(torch.from_numpy(sample_batch["weights"]).float())
    if torch.cuda.is_available():
        weights = weights.cuda()
    (weights * loss).mean().backward()
    nn.utils.clip_grad_norm(self.policy_net.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
    self.optimiser.step()
    return loss.abs().data.cpu().numpy()

  def compute_td_error(self, sample_batch):
    loss = self._batch_loss(sample_batch)
    return loss.abs().data.cpu().numpy()

  def _loss(self, states, actions, returns, next_states, nonterminals):
    batch_size = len(actions)

    # Calculate current state probabilities (note that policy net noise reset between updates anyway)
    ps = self.policy_net(states)  # Probabilities p(s_t, ·; θpolicy)
    ps_a = ps[range(batch_size), actions]  # p(s_t, a_t; θpolicy)

    # Calculate nth next state probabilities
    self.policy_net.reset_noise()  # Sample new noise for action selection
    pns = self.policy_net(next_states).data  # Probabilities p(s_t+n, ·; θpolicy)
    dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θpolicy))
    argmax_indices_ns = dns.sum(2).max(1)[1]  # Perform argmax action selection using policy network: argmax_a[(z, p(s_t+n, a; θpolicy))]
    self.target_net.reset_noise()  # Sample new target net noise
    pns = self.target_net(next_states).data  # Probabilities p(s_t+n, ·; θtarget)
    pns_a = pns[range(batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θpolicy))]; θtarget)
    pns_a *= nonterminals  # Set p = 0 for terminal nth next states as all possible expected returns = expected reward at final transition

    # Compute Tz (Bellman operator T applied to z)
    Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
    Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
    # Compute L2 projection of Tz onto fixed support z
    b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
    l, u = b.floor().long(), b.ceil().long()

    # Distribute probability of Tz
    m = states.data.new(batch_size, self.atoms).zero_()
    offset = torch.linspace(0, ((batch_size - 1) * self.atoms), batch_size).long().unsqueeze(1).expand(batch_size, self.atoms).type_as(actions)
    m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
    m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(Variable(m) * ps_a.log(), 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    return loss

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
    loss = self._loss(states, actions, returns, next_states, nonterminals)
    self.policy_net.zero_grad()
    (weights * loss).mean().backward()  # Importance weight losses
    nn.utils.clip_grad_norm(self.policy_net.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
    self.optimiser.step()
    mem.update_priorities(idxs, loss.data.pow(self.priority_exponent))  # Update priorities of sampled transitions

  def update_target_net(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())

  def save(self, path):
    torch.save(self.policy_net.state_dict(), os.path.join(path, 'model.pth'))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    return (self.policy_net(state.unsqueeze(0)).data * self.support).sum(2).max(1)[0][0]

  def train(self):
    self.policy_net.train()

  def eval(self):
    self.policy_net.eval()