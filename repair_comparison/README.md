# Before and after: policy-gradient repair

These are the two source files that matter for the zero-forward-motion bug.
They are copies for comparison; the live runnable version remains at the
project root.

| Version | `models.py` action draw | `train.py` feedback |
| --- | --- | --- |
| `before_repair` | `distribution.rsample()` | noisy rollout return only |
| `after_repair` | `distribution.sample()` | noisy rollout plus deterministic policy evaluation |

The environment is non-differentiable because actions are converted to NumPy
before `env.step`. REINFORCE therefore needs the score-function gradient from
`distribution.sample()`. With `rsample()`, the sampled action remains in the
autograd graph and its contribution cancels the log-probability gradient.

The runnable repaired version is `../tactile_quad/models.py` and `../train.py`.
