# Tactile Quadruped RL (minimal)

A from-scratch, deliberately small example of tactile-only quadruped RL. It
has no simulator dependency, camera, transformer, PPO framework, or robot I/O.

The toy quadruped has four legs. The policy receives four body/proprioceptive
values and four normalized foot-contact forces -- no vision, no ground-truth
pose -- and outputs four continuous leg commands. It is trained with a
supervised (extrinsic) reward: forward-velocity progress minus penalties for
pitch instability, control effort, and falling. `tactile_quad/models.py` also
ships a Random Network Distillation (`TactileRND`) module, unused by
`train.py` for now -- it's there for a follow-up exercise that adds an
intrinsic novelty bonus on top of this baseline.

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python train.py --episodes 300
```

## Logging and Plotting

You can log the per-episode return and steps to a CSV file during training:

```bash
python train.py --episodes 300 --log-path runs/baseline.csv
```

To visualize the training progress, use the included plotting script. It can overlay multiple runs for comparison:

```bash
python plot_runs.py runs/baseline.csv --out plot.png
```

## View the toy quadruped

```bash
python viewer.py
```

This opens an explanatory schematic rather than a physical robot renderer.
The left panel shows the toy body's pitch and its four labelled legs. A bright
foot on the ground is carrying more contact force (stance); a faint raised foot
is swinging. The right panel is a top view of the four tactile sensors:
`FL`/`FR` are front-left/front-right and `RL`/`RR` are rear-left/rear-right;
each circle's number and colour are its normalized force from 0 to 1. It uses a
simple alternating gait by default; use random commands with `python viewer.py
--random`. Close the window or press Ctrl+C to stop it.

If you are on a headless machine or see `FigureCanvasAgg is non-interactive`,
render the animation to a GIF instead:

```bash
python viewer.py --save toy_quadruped.gif
```

This is a learning scaffold, not a controller for physical hardware. Replace
`TactileQuadrupedEnv` with a simulator or robot interface only after adding
appropriate limits, safety checks, and supervised validation.

## Layout

- `tactile_quad/env.py`: tiny contact-based quadruped toy environment
- `tactile_quad/models.py`: MLP Gaussian policy and RND novelty model
- `train.py`: compact REINFORCE training loop using intrinsic reward only
- `viewer.py`: Matplotlib visual viewer for the toy environment
