"""Before repair: REINFORCE trainer without deterministic evaluation."""

import numpy as np
import torch

from tactile_quad.env import TactileQuadrupedEnv
from tactile_quad.models import TactilePolicy


def compute_returns(rewards, gamma):
  returns = np.zeros(len(rewards), dtype=np.float32)
  running = 0.0
  for t in reversed(range(len(rewards))):
    running = rewards[t] + gamma * running
    returns[t] = running
  return returns


def train(episodes, seed, gamma):
  torch.manual_seed(seed)
  env = TactileQuadrupedEnv(seed=seed)
  policy = TactilePolicy()
  optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

  for episode in range(1, episodes + 1):
    observation, done = env.reset(), False
    log_probs, rewards = [], []
    while not done:
      obs_tensor = torch.as_tensor(observation).unsqueeze(0)
      action, log_prob = policy.sample(obs_tensor)
      observation, reward, done = env.step(action.detach().squeeze(0).numpy())
      log_probs.append(log_prob.squeeze(0))
      rewards.append(reward)

    returns = torch.as_tensor(compute_returns(rewards, gamma))
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)
    policy_loss = -(torch.stack(log_probs) * returns).mean()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
