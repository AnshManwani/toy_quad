"""A tiny kinematic quadruped-like environment with four tactile readings."""

import numpy as np


class TactileQuadrupedEnv:
  """Toy environment used to demonstrate tactile-only locomotion learning.

  Observation: [height, pitch, forward_velocity, pitch_velocity, contacts(4)].
  Action: one stance command in [-1, 1] for each leg.
  Reward: forward-velocity progress minus pitch-instability and action-effort
    penalties, plus a fall penalty. Computed only from proprioception and the
    four tactile contact forces -- no vision, no ground-truth pose.
  """

  observation_dim = 8
  action_dim = 4

  def __init__(self, horizon=200, seed=0):
    self.horizon = horizon
    self.rng = np.random.default_rng(seed)
    self.reset()

  def reset(self):
    self.step_count = 0
    self.height = 0.32
    self.pitch = 0.0
    self.velocity = 0.0
    self.pitch_velocity = 0.0
    self.phase = self.rng.uniform(0, 2 * np.pi, size=4)
    return self._observation()

  def step(self, action):
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    self.step_count += 1
    self.phase = (self.phase + np.array([0.25, 0.25, 0.25, 0.25]) +
                  0.08 * action) % (2 * np.pi)
    contacts = self._contacts(action)

    front_support = contacts[:2].sum()
    rear_support = contacts[2:].sum()
    support = contacts.mean()
    self.velocity = 0.93 * self.velocity + 0.06 * support * action.mean()
    self.pitch_velocity = 0.88 * self.pitch_velocity + 0.025 * (front_support - rear_support)
    self.pitch = np.clip(self.pitch + self.pitch_velocity, -0.7, 0.7)
    self.height = np.clip(0.28 + 0.09 * support - 0.04 * abs(self.pitch), 0.05, 0.45)
    fell = self.height < 0.10
    done = self.step_count >= self.horizon or fell
    reward = self._reward(action, fell)
    return self._observation(contacts), reward, done

  def _reward(self, action, fell):
    forward_reward = self.velocity
    stability_penalty = 0.4 * self.pitch ** 2
    effort_penalty = 0.02 * float(np.mean(np.square(action)))
    fall_penalty = 5.0 if fell else 0.0
    return float(forward_reward - stability_penalty - effort_penalty - fall_penalty)

  def _contacts(self, action):
    # Positive values represent normalized normal force at each foot.
    gait = np.maximum(0.0, np.sin(self.phase))
    force = gait * (0.45 + 0.55 * (action + 1.0) / 2.0)
    return np.clip(force + self.rng.normal(0, 0.015, 4), 0.0, 1.0)

  def _observation(self, contacts=None):
    if contacts is None:
      contacts = self._contacts(np.zeros(4))
    return np.array([self.height, self.pitch, self.velocity,
                     self.pitch_velocity, *contacts], dtype=np.float32)
