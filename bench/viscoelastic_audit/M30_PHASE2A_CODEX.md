# M30 Phase 2a - Codex interpBB patch

Setup: standalone D2Q9/BGK Couette annulus, 64x64 LU, center `(32.5,32.5)`, `R_in=10`, `R_out=25`, `tau=1`, Float64 CPU, 5000 steps. `f` is initialized from `feq(rho0+p_an/cs2,u_an)` on `R_in<r<R_out`. Both cylinder walls use the tested BC (`halfway` or Bouzidi-FL `interp`); the moving inner-wall correction is sampled at the true circular cut point for both runs, so the comparison isolates interpolation weights from rotor velocity sampling. No Kraken `src/` code is used.

## Q1-Q3 metrics

Pressure metrics subtract the spatial mean of `p_LBM-p_an` before the max norm.

| omega | BC | u max rel | p all rel | p wall rel | drag | abs(T) | torque rel err |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.001 | halfway | 3.966956e-02 | 5.187967e+00 | 1.287452e+01 | 2.927523e-16 | 2.534381e-01 | 1.646523e-02 |
| 0.001 | interp | 4.412771e-03 | 8.723525e-01 | 2.164840e+00 | 1.570399e-15 | 2.484905e-01 | 3.377930e-03 |
| 0.005 | halfway | 3.947746e-02 | 1.086766e+00 | 2.696931e+00 | 1.175810e-15 | 1.269757e+00 | 1.852432e-02 |
| 0.005 | interp | 5.657086e-03 | 2.857472e-01 | 7.091135e-01 | 8.142490e-16 | 1.246911e+00 | 1.980195e-04 |

Q1: both runs remain stable and approach the analytical Couette velocity; interpBB is much closer in velocity and offset-removed pressure. Low-Mach pressure is the noisiest because the analytical pressure signal is small.

Q2: inner wall-band pressure error drops from `12.8745 -> 2.16484` at `omega=0.001` and `2.69693 -> 0.709114` at `omega=0.005`.

Q3: drag cancels to roundoff (`~1e-15`). Torque magnitudes match `T_an` far better with interpBB; signs are opposite the natural analytical convention, as expected for Ladd MEA.

## Q4 verdict

| omega | wall pressure improvement | torque error improvement | verdict |
|---:|---:|---:|---|
| 0.001 | 83.185% | 79.484% | GO |
| 0.005 | 73.707% | 98.931% | GO |

**GO.** Bouzidi-FL interpBB clears the >=30% bar for both wall-pressure error and torque error on both omega configurations, with no instability and no drag symmetry regression.
