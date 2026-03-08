# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math

import matplotlib.pyplot as plt
import numpy as np


# --- parameter ranges from SoftplusDecayGate defaults ---
A_init_min, A_init_max = 0, 16
dt_init_min, dt_init_max = 1e-3, 1e-3
dt_init_floor = 1e-4


def inv_softplus(x):
    """Inverse of softplus, matching the PyTorch convention used in decay_gate.py."""
    return x + np.log(-np.expm1(-x))


def decay_gate(x, A, dt_bias):
    """e^{-A * ln(1 + e^{dt_bias + x})}"""
    return np.exp(-A * np.log1p(np.exp(dt_bias + x)))


np.random.seed(17)
n_samples = 8

A_samples = np.random.uniform(A_init_min, A_init_max, size=n_samples)
dt_samples = np.exp(np.random.uniform(math.log(dt_init_min), math.log(dt_init_max), size=n_samples))
dt_samples = np.clip(dt_samples, a_min=dt_init_floor, a_max=None)
dt_bias_samples = inv_softplus(dt_samples)

x = np.linspace(-2.5, 15, 1000)

# Darken tab20 so lines are more visible
# colors = plt.cm.tab20(np.linspace(0, 1, n_samples))
# colors[:, :3] *= 0.7  # darken RGB, keep alpha

sort_idx = np.argsort(A_samples)
A_samples = A_samples[sort_idx]
dt_samples = dt_samples[sort_idx]
dt_bias_samples = dt_bias_samples[sort_idx]

cmap = plt.cm.viridis
colors = [cmap(i / (n_samples - 1)) for i in range(n_samples)]

fig, ax = plt.subplots(figsize=(8, 5))
for i in range(n_samples):
    y = decay_gate(x, A_samples[i], dt_bias_samples[i])
    ax.plot(x, y, color=colors[i], label=f"$\\alpha_n$={A_samples[i]:.1f}")

ax.set_xlabel("$x_t$", fontsize=16)
ax.set_ylabel("forget gate ($f_t$)", fontsize=16)
ax.legend(loc="upper right")
ax.tick_params(axis="both", labelsize=16)
ax.grid(True, alpha=0.3)
# ax.set_ylim(-0.05, 1.05)

fig.tight_layout()
plt.savefig("decay_gate_plot.png", dpi=150, bbox_inches="tight")

# plt.show()
plt.savefig("decay_gate.svg", format="svg")
