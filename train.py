"""Train a tactile MLP with an intrinsic RND reward only."""

import argparse
import numpy as np
import torch

from tactile_quad.env import TactileQuadrupedEnv
from tactile_quad.models import TactilePolicy, TactileRND


def train(episodes, seed):
  torch.manual_seed(seed)
  env = TactileQuadrupedEnv(seed=seed)
  policy = TactilePolicy()
  rnd = TactileRND()
  policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
  rnd_optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-3)

  for episode in range(1, episodes + 1):
    observation, done = env.reset(), False
    log_probs, rewards, tactile_batch = [], [], []
    while not done:
      obs_tensor = torch.as_tensor(observation).unsqueeze(0)
      action, log_prob = policy.sample(obs_tensor)
      observation, done = env.step(action.detach().squeeze(0).numpy())
      tactile = torch.as_tensor(observation[-4:]).unsqueeze(0)
      with torch.no_grad():
        reward = rnd.error(tactile).item()
      log_probs.append(log_prob.squeeze(0))
      rewards.append(reward)
      tactile_batch.append(tactile)

    # Keep the novelty model learning, so familiar tactile states lose reward.
    tactile_batch = torch.cat(tactile_batch)
    rnd_loss = rnd.error(tactile_batch).mean()
    rnd_optimizer.zero_grad()
    rnd_loss.backward()
    rnd_optimizer.step()

    returns = torch.as_tensor(rewards, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)
    policy_loss = -(torch.stack(log_probs) * returns).mean()
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    if episode == 1 or episode % 25 == 0:
      print(f"episode={episode:4d} novelty={np.mean(rewards):.4f} "
            f"rnd_loss={rnd_loss.item():.4f}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--episodes", type=int, default=300)
  parser.add_argument("--seed", type=int, default=0)
  train(**vars(parser.parse_args()))
