# rheoTool lid-driven cavity reference — Oldroyd-BLog

Project-local copy of rheoTool's `rheoFoam/Cavity/Oldroyd-BLog` tutorial
(of90 source tree at `~/Documents/Recherche/Codes CFD/rheotool/`),
for the closed axis-aligned log-FV comparison.

## Parameters

From `constant/constitutiveProperties` and the rheoTool user guide §5.1.4:

- Square cavity, side `L = 1`, mesh `127 x 127` uniform.
- `rho = 0.01`, `etaS = etaP = 0.5`, `lambda = 1`.
- `beta = etaS / (etaS + etaP) = 0.5`, `eta_0 = 1`.
- Reference scales: length `L`, time `L/U`, velocity `U = 1`,
  stress `eta_0 * U / L`.
- `Re = rho * U * L / eta_0 = 0.01` (creeping with finite inertia).
- `De = lambda * U / L = 1`.

## Boundary conditions

- `movingLid` (top): smoothly ramped velocity
  `U_lid(x, t) = 8 * U * [1 + tanh(8 * (t - 0.5))] * x^2 * (1 - x)^2`.
- `fixedWalls` (left, right, bottom): no-slip `(0, 0, 0)`.
- Stress: linearly extrapolated; pressure: zero normal gradient.

## Reference data

User guide Fig. 5.4 plots, against Fattal & Kupferman 2005:

- (a) `u(x=0.5, y)` at `t = 8`
- (b) `theta_xy(x, y=0.75)` at `t = 8`
- (c) volume-averaged kinetic energy `E_k(t)` from `t = 0` to `t = 8`

The sampleDict in `system/` already extracts (a) and (b) automatically.
The kineticEnergy `coded` functionObject in `controlDict` writes `kinEner.txt`
with the kinetic and elastic energy time series.

## Tweaks vs upstream tutorial

- `writeInterval` reduced from `2` to `1` to get snapshots at `t = 1, ..., 8`.
- `endTime` reduced from `10` to `8` (comparison target is `t = 8`; the
  user guide kinetic-energy plot shows steady behaviour by `t ~ 4-5`).
- `adjustTimeStep` toggled from `off` to `on`. The stock tutorial fixed
  `dt = 2e-4` which gave Courant max around `3e-5` (7000x below cap),
  bloating wall-clock to roughly six hours. Adaptive timestep with
  `maxCo = 0.2` and `maxDeltaT = 5e-3` (raised from `1e-3`) keeps the
  lid-ramp dynamics resolved while letting `dt` grow to its Courant
  limit (~1.5e-3 at peak `U_lid`).

Everything else (127x127 uniform mesh, constitutive params, BCs,
sampleDict, kineticEnergy function object) is unchanged.

## Run

Wall-clock estimate: ~10-15 min on Apple Silicon Docker
(16 k cells uniform, `maxCo = 0.2`, sparse coupled solver).

Interactive wrapper (drops you into a shell with the right environment):

```sh
"/Users/guillaume/Documents/Recherche/Codes CFD/rheotool/container/v1.2/openfoam9-rheotool.sh" -d bench/rheotool/cavity_oldroydb_log_re001_de1_b05
./Allclean
./Allrun
```

Non-interactive helper (Docker):

```sh
./run_docker.sh
```

Background daemon (recommended for the t=10 run):

```sh
nohup ./run_docker.sh > run.log 2>&1 &
disown
tail -f run.log
```

## Outputs after a clean run

- `0/`, `1/`, `2/`, ..., `10/` — field snapshots (U, p, tau, theta).
- `postProcessing/sampleDict/8/lineVert_x0.5_U_tau_theta.xy` (or per-field
  raw files) — vertical sample at `x = 0.5`, time `t = 8`.
- `postProcessing/sampleDict/8/lineHorz_y0.75_U_tau_theta.xy` — horizontal
  sample at `y = 0.75`, time `t = 8`.
- `kinEner.txt` — `(time, E_k_kinetic, E_k_elastic)` for the function
  object's `writeInterval = 20` timestep cadence.

## Cross-reference

Used by `bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool.jl`
(to be added in Track 2) to compare profiles against Kraken's
`run_viscoelastic_logfv_cavity_coupled_2d`.
