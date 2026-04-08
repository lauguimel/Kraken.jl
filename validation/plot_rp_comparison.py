#!/usr/bin/env python3
"""Plot Rayleigh-Plateau comparison: 3 gas models vs theory."""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "results" / "rp_comparison"
OUT_DIR  = DATA_DIR

models = ["smooth", "ghost", "phasefield"]
colors = {"smooth": "C0", "ghost": "C1", "phasefield": "C2"}
markers = {"smooth": "o", "ghost": "s", "phasefield": "^"}

# --- Read all r_min trajectories ---
def load_rmin(model, lam):
    f = DATA_DIR / f"rmin_{model}_lambda{lam}.dat"
    data = np.loadtxt(f, comments="#")
    return data[:, 0], data[:, 1]  # t, r_min/R0

# --- 1. r_min(t) for all models at λ/R0 = 9.0 (most unstable mode) ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
for m in models:
    t, r = load_rmin(m, "9.0")
    ax.plot(t, r, marker=markers[m], color=colors[m], label=m, markersize=4, lw=1.5)
ax.set_xlabel("step")
ax.set_ylabel(r"$r_{\min}/R_0$")
ax.set_title(r"RP thinning at $\lambda/R_0 = 9$ (most unstable)")
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)

# --- 2. r_min(t) at one stable wavelength to verify decay ---
ax = axes[1]
for m in models:
    t, r = load_rmin(m, "5.0")
    ax.plot(t, r, marker=markers[m], color=colors[m], label=m, markersize=4, lw=1.5)
ax.set_xlabel("step")
ax.set_ylabel(r"$r_{\min}/R_0$")
ax.set_title(r"Stable mode $\lambda/R_0 = 5$ ($kR_0 > 1$)")
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0.9, 1.02)

plt.tight_layout()
plt.savefig(OUT_DIR / "rmin_trajectories.png", dpi=140)
print(f"Saved: {OUT_DIR / 'rmin_trajectories.png'}")

# --- 3. r_min(t) for all wavelengths, one panel per model ---
lambdas = ["4.5", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, m in zip(axes, models):
    cmap = plt.cm.viridis(np.linspace(0, 1, len(lambdas)))
    for i, lam in enumerate(lambdas):
        t, r = load_rmin(m, lam)
        ax.plot(t, r, color=cmap[i], lw=1.5, label=rf"$\lambda/R_0={lam}$")
    ax.set_xlabel("step")
    ax.set_title(f"{m}")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    if m == models[0]:
        ax.set_ylabel(r"$r_{\min}/R_0$")
    if m == models[-1]:
        ax.legend(fontsize=8, loc="lower left")

plt.tight_layout()
plt.savefig(OUT_DIR / "rmin_all_wavelengths.png", dpi=140)
print(f"Saved: {OUT_DIR / 'rmin_all_wavelengths.png'}")

# --- 4. Dispersion relation: ω vs kR0 ---
disp = np.loadtxt(DATA_DIR / "dispersion.dat", comments="#")
disp_th = np.loadtxt(DATA_DIR / "dispersion_theory.dat", comments="#")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(disp_th[:, 0], disp_th[:, 1], "k-", lw=2, label="Rayleigh (inviscid)")

for i, m in enumerate(models):
    ax.plot(disp[:, 0], disp[:, 2 + i], marker=markers[m], color=colors[m],
            ls="none", markersize=8, label=m)

ax.axvline(1.0, color="gray", ls="--", alpha=0.5, label=r"$kR_0=1$ (cutoff)")
ax.set_xlabel(r"$kR_0$")
ax.set_ylabel(r"$\omega$ (lattice units)")
ax.set_title("Rayleigh-Plateau dispersion relation")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 1.5)
plt.tight_layout()
plt.savefig(OUT_DIR / "dispersion.png", dpi=140)
print(f"Saved: {OUT_DIR / 'dispersion.png'}")

print("Done.")
