"""A clear schematic viewer for the toy tactile quadruped environment.

This is deliberately an explanatory drawing, not a physics renderer.  It
turns the environment's four tactile readings into visible feet: bright,
ground-level feet are carrying more load; faint, raised feet are swinging.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon

from tactile_quad.env import TactileQuadrupedEnv


LEG_NAMES = ("FL", "FR", "RL", "RR")
# The environment orders contacts as front-left, front-right, rear-left,
# rear-right.  The values below place them at distinct body corners.
LEG_LAYOUT = (
    ("FL", 0.32, -0.09),
    ("FR", 0.32, 0.09),
    ("RL", -0.32, -0.09),
    ("RR", -0.32, 0.09),
)


class ToyViewer:
  def __init__(self, random_actions=False, seed=0):
    self.env = TactileQuadrupedEnv(seed=seed)
    self.observation = self.env.reset()
    self.random_actions = random_actions
    self.time = 0
    self.last_action = np.zeros(4, dtype=np.float32)

    self.figure, (self.world_ax, self.tactile_ax) = plt.subplots(
        1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.45, 1]})
    self.figure.suptitle("Toy tactile quadruped: contact forces made visible",
                          fontsize=14, fontweight="bold")
    self.figure.subplots_adjust(top=0.82, wspace=0.3)

    self.world_ax.set_xlim(-0.95, 0.95)
    self.world_ax.set_ylim(-0.12, 0.72)
    self.world_ax.set_aspect("equal")
    self.world_ax.set_title("Side view — arrow points forward →", pad=12)
    self.world_ax.axis("off")
    self.world_ax.axhline(0, color="#334155", linewidth=2)
    self.world_ax.text(0.91, -0.07, "ground", ha="right", va="top",
                       color="#475569", fontsize=9)
    self.body = Polygon(np.zeros((4, 2)), closed=True, facecolor="#4f83cc",
                        edgecolor="#173f73", linewidth=2, zorder=3)
    self.world_ax.add_patch(self.body)
    self.legs = [self.world_ax.plot([], [], color="#64748b", linewidth=4,
                                    solid_capstyle="round", zorder=2)[0]
                 for _ in LEG_NAMES]
    self.feet = [Circle((0, 0), 0.035, zorder=4) for _ in LEG_NAMES]
    for foot in self.feet:
      self.world_ax.add_patch(foot)
    self.leg_labels = [self.world_ax.text(0, 0, name, ha="center", va="bottom",
                                          fontsize=9, fontweight="bold", zorder=5)
                       for name in LEG_NAMES]
    self.status = self.world_ax.text(-0.92, 0.65, "", fontsize=10,
                                     color="#1e293b")
    self.world_ax.text(-0.92, 0.58,
                       "Bright foot = more pressure / stance\n"
                       "Faint raised foot = swing",
                       fontsize=9, color="#475569", va="top")

    self.tactile_ax.set_title("Tactile sensors: viewed from above", pad=12)
    self.tactile_ax.set_xlim(-1.25, 1.25)
    self.tactile_ax.set_ylim(-1.15, 1.15)
    self.tactile_ax.set_aspect("equal")
    self.tactile_ax.axis("off")
    self.tactile_ax.add_patch(Polygon([(-0.42, -0.62), (0.42, -0.62),
                                       (0.58, 0.62), (-0.58, 0.62)],
                                      closed=True, facecolor="#dbeafe",
                                      edgecolor="#173f73", linewidth=2))
    self.tactile_ax.annotate("FORWARD", xy=(0, 1.06), xytext=(0, 0.78),
                             ha="center", fontsize=9, color="#475569",
                             arrowprops={"arrowstyle": "->", "color": "#475569"})
    sensor_positions = ((-0.85, 0.67), (0.85, 0.67),
                        (-0.85, -0.67), (0.85, -0.67))
    self.sensor_pads = []
    self.sensor_text = []
    for name, position in zip(LEG_NAMES, sensor_positions):
      pad = Circle(position, 0.23, edgecolor="#334155", linewidth=1.5)
      self.tactile_ax.add_patch(pad)
      self.sensor_pads.append(pad)
      self.sensor_text.append(self.tactile_ax.text(
          *position, name, ha="center", va="center", fontsize=10,
          fontweight="bold"))
    self.force_text = self.tactile_ax.text(0, -1.05, "", ha="center",
                                            fontsize=10, color="#1e293b")

  def _action(self):
    if self.random_actions:
      return self.env.rng.uniform(-1, 1, size=4)
    # A simple diagonal alternating gait: FL/RR then FR/RL.
    return np.array([1, -1, -1, 1], dtype=np.float32) * np.sin(self.time * 0.18)

  @staticmethod
  def _rotate(points, angle, center):
    """Rotate an array of (x, y) points around center."""
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    return (points - center) @ rotation.T + center

  def update(self, _):
    self.time += 1
    self.last_action = self._action()
    self.observation, done = self.env.step(self.last_action)
    if done:
      self.observation = self.env.reset()

    height, pitch, velocity, _, *contacts = self.observation
    contacts = np.asarray(contacts)
    center = np.array([0.0, height])
    body_corners = np.array([[-0.48, height - 0.10], [0.40, height - 0.10],
                             [0.50, height + 0.10], [-0.48, height + 0.10]])
    self.body.set_xy(self._rotate(body_corners, pitch, center))

    for index, ((name, root_x, side), leg, foot, label, contact) in enumerate(
        zip(LEG_LAYOUT, self.legs, self.feet, self.leg_labels, contacts)):
      root = self._rotate(np.array([[root_x, height - 0.09]]), pitch, center)[0]
      # In a stance the foot reaches the ground.  In swing it is visibly raised.
      foot_y = 0.018 if contact > 0.12 else 0.13
      foot_x = root[0] + 0.10 * side + 0.075 * np.sin(self.env.phase[index])
      color = plt.cm.viridis(0.12 + 0.88 * contact)
      leg.set_data([root[0], foot_x], [root[1], foot_y])
      leg.set_color(color)
      leg.set_linewidth(3 + 3 * contact)
      foot.center = (foot_x, foot_y)
      foot.set_facecolor(color)
      foot.set_edgecolor("#172554")
      foot.set_alpha(0.35 + 0.65 * contact)
      label.set_position((foot_x, foot_y + 0.045))
      label.set_color("#172554" if contact > 0.12 else "#64748b")

    for pad, label, contact in zip(self.sensor_pads, self.sensor_text, contacts):
      pad.set_facecolor(plt.cm.viridis(0.12 + 0.88 * contact))
      pad.set_alpha(0.25 + 0.75 * contact)
      label.set_text(f"{label.get_text().split(chr(10))[0]}\n{contact:.2f}")
      label.set_color("white" if contact > 0.52 else "#172554")

    stance = ", ".join(name for name, force in zip(LEG_NAMES, contacts)
                       if force > 0.12) or "none"
    self.status.set_text(f"height {height:.2f}   pitch {np.degrees(pitch):+.0f}°"
                         f"   speed {velocity:+.2f}")
    self.force_text.set_text(f"Numbers are normalized contact force (0–1).  Stance: {stance}")
    return [self.body, *self.legs, *self.feet, *self.leg_labels,
            *self.sensor_pads, *self.sensor_text, self.status, self.force_text]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--random", action="store_true",
                      help="use random commands instead of the demo gait")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--save", metavar="PATH",
                      help="save a GIF instead of opening a GUI window")
  parser.add_argument("--frames", type=int, default=200,
                      help="number of frames when using --save (default: 200)")
  args = parser.parse_args()
  viewer = ToyViewer(random_actions=args.random, seed=args.seed)
  animation_kwargs = dict(interval=50, blit=False, cache_frame_data=False)
  if args.save:
    animation_kwargs["frames"] = args.frames
  animation = FuncAnimation(viewer.figure, viewer.update, **animation_kwargs)
  if args.save:
    animation.save(args.save, writer="pillow", fps=20,
                   savefig_kwargs={"facecolor": "white"})
    print("Saved animation to {}".format(args.save))
  else:
    backend = plt.get_backend().lower()
    if "agg" in backend:
      raise RuntimeError(
        "No interactive display is available. Use --save toy_quadruped.gif "
        "to render a GIF instead.")
    plt.show()
  return animation


if __name__ == "__main__":
  main()
