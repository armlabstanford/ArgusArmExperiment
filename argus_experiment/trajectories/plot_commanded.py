"""
Plot one period of the commanded EE trajectory: sawtooth and sinusoid, as two
separate minimal figures (axis names only, no ticks/grid/title).

Both waves are given the same period T by choosing each one's speed:
    sawtooth  : period = 4A / v        ->  v = 4A / T   (constant speed)
    sinusoid  : period = 2*pi*A / v    ->  v = 2*pi*A / T (peak speed)

Waypoints come from the same builders run_trajectories.py executes
(build_linear_sawtooth / build_linear_sinusoidal), so the curves are exactly
what gets sent to IK. Both figures share axis limits so they line up.

Usage:
    python argus_experiment/trajectories/plot_commanded.py
    python argus_experiment/trajectories/plot_commanded.py --period 6 --end-dwell 0.5
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_trajectories as rt

RED = "#D62828"   # sawtooth
BLUE = "#1F5FBF"  # sinusoid
INK = "#1f1f1e"


def displacement(waypoints, unit):
    return np.array([wp[:3, 3] @ unit for wp in waypoints])  # center is identity -> offset from center


def save_minimal(t, y, color, t_max, amp, out):
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=300)
    ax.plot(t, y, color=color, lw=3.75, solid_capstyle="butt", solid_joinstyle="round")  # clipped at the axes so it starts exactly on the y axis
    ax.set_xlim(0, t_max)
    ax.set_ylim(-amp * 1.15, amp * 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(1.5)
    ax.set_xlabel("Time", fontsize=16, color=INK, labelpad=8)
    ax.set_ylabel("Displacement", fontsize=16, color=INK, labelpad=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--period", type=float, default=4.0, help="s, shared period of both waves")
    p.add_argument("--amplitude", type=float, default=rt.LINEAR_AMPLITUDE, help="m, half peak-to-peak")
    p.add_argument("--end-dwell", type=float, default=0.0, help="s hold at each end of travel")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--outdir", default=str(Path(__file__).parent / "figures"))
    a = p.parse_args()

    A, T = a.amplitude, a.period
    v_saw, v_sin = 4 * A / T, 2 * np.pi * A / T
    unit, center = rt.AXES["x"], np.eye(4)
    saw = displacement(rt.build_linear_sawtooth(center, unit, A, v_saw, 1, a.dt, a.end_dwell), unit)
    sin = displacement(rt.build_linear_sinusoidal(center, unit, A, v_sin, 1, a.dt, a.end_dwell), unit)
    sin = np.append(sin, sin[0])  # builder omits the t=T sample (next period's start); close the loop
    t_saw, t_sin = np.arange(len(saw)) * a.dt, np.arange(len(sin)) * a.dt
    t_max = max(t_saw[-1], t_sin[-1])

    print(f"period {T:g} s, amplitude ±{A * 100:g} cm: sawtooth {v_saw:.4f} m/s constant, "
          f"sinusoid {v_sin:.4f} m/s peak (spans {t_saw[-1]:.2f} s / {t_sin[-1]:.2f} s)")
    out = Path(a.outdir)
    save_minimal(t_saw, saw, RED, t_max, A, out / "commanded_sawtooth.png")
    save_minimal(t_sin, sin, BLUE, t_max, A, out / "commanded_sinusoid.png")


if __name__ == "__main__":
    main()
