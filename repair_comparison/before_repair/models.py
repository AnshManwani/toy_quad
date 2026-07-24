"""Before repair: policy code that did not provide a REINFORCE gradient."""

import torch
from torch import nn
from torch.distributions import Normal


class TactilePolicy(nn.Module):
  def __init__(self, observation_dim=8, action_dim=4, hidden_dim=64):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(observation_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, action_dim))
    self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))

  def sample(self, observation):
    distribution = Normal(self.net(observation), self.log_std.exp())
    # Bug in this non-differentiable environment: this cancels the
    # score-function gradient used by the REINFORCE loss below.
    latent_action = distribution.rsample()
    return torch.tanh(latent_action), distribution.log_prob(latent_action).sum(-1)
