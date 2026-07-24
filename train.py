"""Train a tactile MLP policy with a supervised forward-locomotion reward."""

import argparse
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


def train(episodes, seed, gamma, log_path=None):
  if log_path:
    import os
    import csv
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
      if write_header:
        csv.writer(f).writerow(['episode', 'return', 'steps'])

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

    if log_path:
      import csv
      with open(log_path, 'a', newline='') as f:
        csv.writer(f).writerow([episode, sum(rewards), len(rewards)])

    if episode == 1 or episode % 25 == 0:
      print(f"episode={episode:4d} return={sum(rewards):7.2f} "
            f"steps={len(rewards):3d}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--episodes", type=int, default=300)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--gamma", type=float, default=0.99)
  parser.add_argument("--log-path", type=str, default=None)
  train(**vars(parser.parse_args()))
