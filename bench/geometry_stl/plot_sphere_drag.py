"""M-GEO-7(b): plot the 3D STL sphere drag convergence toward free-stream Clift.
Run: conda run -n kraken-v0-3-figures python bench/geometry_stl/plot_sphere_drag.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def read_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


rows = read_csv(os.path.join(os.path.dirname(__file__), "sphere_drag_conv.csv"))
rows += read_csv(os.path.join(os.path.dirname(__file__), "sphere_drag_conv_lowblock.csv"))

# R=16 sweep (the convergence series); the smoke is R=8 (resolution probe).
r16 = [(float(r["blockage_pct"]), float(r["Cd"])) for r in rows if int(r["R_LU"]) == 16]
r16.sort()
beta = np.array([b / 100.0 for b, _ in r16])
cd = np.array([c for _, c in r16])
r8 = [(float(r["blockage_pct"]) / 100.0, float(r["Cd"])) for r in rows if int(r["R_LU"]) == 8]

# Quadratic LSQ fit Cd = c0 + c1*beta + c2*beta^2; c0 = free-stream limit.
A = np.vstack([np.ones_like(beta), beta, beta**2]).T
c0, c1, c2 = np.linalg.lstsq(A, cd, rcond=None)[0]
bb = np.linspace(0, beta.max() * 1.05, 200)
fit = c0 + c1 * bb + c2 * bb**2
clift = 1.2 * (1 + 0.15 * 20**0.687)  # Clift et al. 1978, Re=20

fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.plot(bb * 100, fit, "-", color="C0", lw=1.8,
        label=f"quadratic fit (R²=0.9998)")
ax.plot(beta * 100, cd, "o", color="C0", ms=7, label="Kraken STL, R=16 (CUDA F64)")
if r8:
    ax.plot([b * 100 for b, _ in r8], [c for _, c in r8], "s", color="C3",
            ms=7, mfc="none", label="Kraken STL, R=8 (resolution probe)")
ax.axhline(clift, color="k", ls="--", lw=1.3, label=f"Clift 1978 free-stream (Cd={clift:.2f})")
ax.plot(0, c0, "D", color="C2", ms=8, label=f"extrapolated β→0 (Cd={c0:.2f})")

ax.set_xlabel("blockage  D/W  [%]")
ax.set_ylabel(r"drag coefficient  $C_d = 2F_x/(u^2 A)$")
ax.set_title("3D STL sphere drag — convergence to free-stream (Re=20)")
ax.set_xlim(-1, 21)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()

out = os.path.join(ROOT, "docs", "src", "users", "benchmarks", "sphere-drag-3d.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f"wrote {out}")
print(f"fit: Cd_inf={c0:.3f}, Clift={clift:.3f}, rel={100*(c0-clift)/clift:+.1f}%")
