"""Small MLP policy and tactile RND module."""

import torch
from torch import nn
from torch.distributions import Normal


class TactilePolicy(nn.Module):
  """Two-layer MLP policy; intentionally no recurrence, vision, or attention."""

  def __init__(self, observation_dim=8, action_dim=4, hidden_dim=64):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(observation_dim, hidden_dim), nn.Tanh(),
      nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
      nn.Linear(hidden_dim, action_dim))
    self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))

  def sample(self, observation):
    distribution = Normal(self.net(observation), self.log_std.exp())
    # REINFORCE needs a score-function gradient.  ``rsample`` makes the
    # sampled latent action part of the gradient graph, which cancels that
    # gradient in ``log_prob`` for this non-differentiable environment.
    latent_action = distribution.sample()
    return torch.tanh(latent_action), distribution.log_prob(latent_action).sum(-1)


class TactileRND(nn.Module):
  """Predict a fixed tactile encoder; prediction error is the intrinsic reward."""

  def __init__(self, tactile_dim=4, feature_dim=32):
    super().__init__()
    self.target = nn.Sequential(nn.Linear(tactile_dim, feature_dim), nn.ReLU(),
                                nn.Linear(feature_dim, feature_dim))
    self.predictor = nn.Sequential(nn.Linear(tactile_dim, feature_dim), nn.ReLU(),
                                   nn.Linear(feature_dim, feature_dim))
    for parameter in self.target.parameters():
      parameter.requires_grad = False

  def error(self, tactile):
    return (self.predictor(tactile) - self.target(tactile)).square().mean(-1)
