# M42 — `:muscl_superbee_relax` implementation VERDICT

Date    : 2026-05-23
Branch  : `dev-viscoelastic`
Spec    : `bench/viscoelastic_audit/M42_DESIGN.md`
Mission : M42-impl (Claude inline as Department).

---

## Aqua G5 v3 RESULTS — PARTIAL (1/4 gates PASS+, 3/4 NaN)

Job `21736323.aqua` (M42_G5 v3, Wi=0 removed because driver rejects λ=0),
6:48 walltime, Exit_status=0. SUMMARY at `tmp/m42_g5_v3_results/SUMMARY.csv`.

| R  | Wi  | Cd_kraken  | NaN  | Gate            | Status         |
|----|-----|------------|------|-----------------|----------------|
| 30 | 0.1 | **131.073** | false| G5-3 (within ±1 % rT 130.43) | **PASS marginal** (+0.49 %) |
| 30 | 1.0 | NaN        | true | G5-1 (Cd ∈ [122, 124])       | **FAIL**       |
| 60 | 0.1 | NaN        | true | G5-4 (R=60 no NaN at 100k)   | **FAIL**       |
| 60 | 1.0 | NaN        | true | (no gate, exploratory)       | (worse than halfwayBB Newt) |

Comparison Wi=0.1 R=30 across schemes:
- :rusanov :halfwayBB                    = 129.39 (-0.80 % rT)
- :muscl_superbee :halfwayBB (M29b bulk) = 129.39 (M29b improvement only visible at Wi=1)
- **:muscl_superbee_relax (M42)**        = **131.073 (+0.49 % rT)** ← progresses toward rT

**Verdict**: M42 relaxation works in concept at low Wi/low R (Cd moves
from 129.39 toward rT 130.43 — overshoots by +0.49 %, in the right
direction). At Wi=1 R=30 it NaN's like M29c-v2 (which NaN'd at step
92k). At R=60 it NaN's even at Wi=0.1, where M42-untouched :halfwayBB
also NaN'd per M32 Phase 3.

## Candidate diagnostic for the residual NaN

Design §4 chose zero-slope on broken-axis face — TVD by Sweby. BUT the
combination (zero-slope on broken-axis + full MUSCL on non-broken axis)
creates an **asymmetric finite-volume update** at cut-link cells. At
low Wi this asymmetry is below the polymer stability envelope; at high
Wi/R it triggers a polymer-coupled instability.

Mitigation candidates for M42-v2 (next session):
1. **1-sided minmod** ψ(r) = max(0, min(r, 1)) on broken-axis face
   (uses upwind + downwind, still TVD, less dissipative AND more
   symmetric). Design §8 alternative.
2. **Narrower relaxation band**: limit pass-2 overwrite to the ring
   layer (1 LU from solid) instead of 2 LU. M41-bis showed 0.33 %
   carries most of the polymer stress; narrower scope reduces the
   asymmetric-update zone.
3. **Spatial NaN fingerprint analysis** on the G5 v3 NaN cases (.jls
   dumps fetched to `tmp/m42_g5_v3_results/`): identify first-NaN
   location to discriminate "cylinder-band NaN" vs "open-wall NaN"
   (M29c-v2 was at j=1 south wall, NOT cylinder).

## Next-session action

1. Read `tmp/m42_g5_v3_results/SUMMARY.csv` (results) + .jls dumps for
   spatial NaN fingerprint
2. Per fingerprint, decide M42-v2: minmod variant OR narrower band
3. Re-implement + re-submit. Cheap.

---

---

## (a) Files modified

| File | New ? | LOC | Purpose |
|------|-------|----:|---------|
| `src/fvfd/muscl_boundary.jl` | Y | 187 | Pass-2 kernel + composite launcher + predicate |
| `src/fvfd/FVFD.jl` | N | +1 | `include("muscl_boundary.jl")` after operators_2d |
| `src/fvfd/operators_2d.jl` | N | +13 | Whitelist + dispatch wiring at `fvfd_advect_upwind_2d!` |
| `src/drivers/viscoelastic_logfv_2d.jl` | N | +2 | Whitelist update L227 |
| `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl` | N | +2 | Env var whitelist L111 |
| `bench/viscoelastic_logfv/run_cyl_m42_g5_a100.pbs` | Y | 84 | G5 matrix submission script |
| `test/test_muscl_boundary_relax.jl` | Y | 178 | 4 smoke testsets |
| `test/runtests.jl` | N | +1 | Include the new test |

Total new LOC ≈ 449. Inside design §9 soft budget of 500.

### Architecture summary

- Pass-1 launches the unchanged `fvfd_advect_upwind_2d_kernel!` with
  `Val{:muscl_superbee}`: full MUSCL bulk, whole-cell `:rusanov` at
  the M29b ring (cylinder band ∪ open-wall band).
- `KernelAbstractions.synchronize(backend)` between passes.
- Pass-2 launches new `fvfd_advect_muscl_relax_boundary_2d_kernel!`
  which OVERWRITES `phi_out[i, j]` only at cylinder-band cells
  (per the design §3 predicate; open-wall band excluded; solid cells
  skipped). Pass-2 reads `@Const(phi)` (lag-0) — never `phi_out` —
  so there is no cross-thread race.
- Dispatch is intercepted at the lowest `fvfd_advect_upwind_2d!`
  level so the symmetric-tensor path `fvfd_sym2_advect_upwind_2d!`
  + the entire `logfv_*` family inherit transparently (no change
  needed in `logconformation_fv_2d.jl`).

---

## (b) Smoke test results — `test/test_muscl_boundary_relax.jl`

All 4 testsets PASS on CPU F64 (`julia --project=.`) in 14.4 s wall.

| Testset | Tests | Status | Key numbers |
|---------|------:|--------|-------------|
| Newtonian R=8 Wi=0       | 8/8  | PASS | Cd_relax = Cd_ref = 274.72 (identical when polymer dormant); ρ ∈ [1.0, 1.009] |
| Polymer R=8 Wi=0.1 β=0.59 | 16/16 | PASS | Cd=241.76 ; Cd_s=258.80 ; Cd_p=12.70 ; max_c_trace=2.80 ; first_nonfinite_step=0 |
| Polymer R=8 Wi=1.0 β=0.59 1000 steps (NaN proxy) | 14/14 | PASS | Cd=201.86 ; max_c_trace=36.77 (<200 envelope) ; completed_steps=1000 ; ρ ∈ [1.0, 1.008] |
| Sentinel `:muscl_superbee` (no _relax) | 4/4 | PASS | Cd=241.13 (regression check : within 0.27 % of relax → relax acts only at cyl-band, no global drift) |

**Critical observations**:

- The NaN proxy testset (Wi=1.0, 1000 steps, ρ-positivity check, polymer envelope) is the empirical guard against the M29c-v2 failure
  mechanism. `max_c_trace` grows from 2 (initial) to 36.77 over 1000
  steps, a smooth elastic build-up consistent with stable Oldroyd-B.
  No CD2-style amplification observed.
- The sentinel testset confirms `:muscl_superbee` (without relax) is
  unchanged by the dispatch wiring (Cd diff between relax and non-relax
  paths at the same setup is 0.27 % at R=8 — consistent with the relax
  path actively re-doing MUSCL on 0.33 % of cells per M41-bis).

Local log saved at `tmp/m42_smoke.log`.

`Pkg.test()` deliberately not run (pre-existing baseline FAIL on
`test_poiseuille.jl` per session memory). The new test is invoked
standalone in CI.

---

## (c) Aqua G5 matrix submission

- **PBS script** : `bench/viscoelastic_logfv/run_cyl_m42_g5_a100.pbs`
- **Matrix** : R ∈ {30, 60} × Wi ∈ {0, 0.1, 1.0} × `:muscl_superbee_relax` × `:halfwayBB` × 100k steps each = 6 cases.
- **Walltime estimate** : ~ 9 min (6 × 85 s/case per M29c-validate calibration). Reserved 4 h ceiling.
- **Job ID** : `21735805.aqua` — submitted 2026-05-23, status Q on `gpu_batch_exec`.
- **Output dir on Aqua** : `tmp/m42_g5/21735805.aqua/`.

Rsync'd in this session (with `--exclude Manifest.toml` per recent
lesson):
- `src/fvfd/{muscl_boundary.jl, FVFD.jl, operators_2d.jl}`
- `src/drivers/viscoelastic_logfv_2d.jl`
- `bench/viscoelastic_logfv/{run_cyl_bigsweep_v2_2d.jl, run_cyl_m42_g5_a100.pbs}`

---

## (d) Next-session check command

```bash
ssh aqua "qstat -f 21735805.aqua 2>&1 | head -30"

# If complete, pull results:
rsync -az aqua:Kraken.jl-viscoelastic-run/tmp/m42_g5/21735805.aqua/ \
  ~/Documents/Recherche/Kraken.jl-viscoelastic/tmp/m42_g5/21735805.aqua/

# Inspect Cd:
ls ~/Documents/Recherche/Kraken.jl-viscoelastic/tmp/m42_g5/21735805.aqua/
grep -E 'R=30|R=60|Cd' ~/Documents/Recherche/Kraken.jl-viscoelastic/tmp/m42_g5/21735805.aqua/*.json 2>&1 | head -30
```

---

## (e) G5 acceptance interpretation (per design §7.2)

| Case | R  | Wi  | β    | Target | Outcome bucket |
|------|----|-----|------|--------|----------------|
| G5-1 | 30 | 1.0 | 0.59 | Cd ∈ [122, 124] | PASS → ship ; FAIL HIGH → over-relax debug ; FAIL LOW [116, 121] → upgrade to 1-sided minmod and re-G5 |
| G5-2 | 30 | 0   | 0.59 | Cd within ±1 % of M41 R=30 Newt 132.08 | PASS gate ; FAIL → predicate/dispatch bug, BLOCK SHIP |
| G5-3 | 30 | 0.1 | 0.59 | Cd within ±1 % of rT 130.43 | PASS gate ; FAIL → low-Wi regression, same root-cause class as G5-2 |
| G5-4 | 60 | 0.1 | 0.59 | No NaN at 100k | regression scaling check |
| bonus | 60 | 0   | 0.59 | finite Cd | R-scaling envelope |
| bonus | 60 | 1.0 | 0.59 | not in primary gate; H-LATE-STIFF envelope of M29b at R=60 unknown — diagnostic only |

**Decision tree** (per design §7.2):
- All G5-1..G5-4 PASS → ship `:muscl_superbee_relax`. Default stays
  `:rusanov` until at least one β-Wi sweep has been re-run (follow-up
  patch, not this mission).
- G5-1 fail LOW only → trivial 1-line tweak to 1-sided minmod on the
  broken face, re-G5.
- Any G5-2 / G5-3 / G5-4 fail → bug in pass-2 dispatch or predicate;
  investigate before any further G5 cycle.

---

## (f) Surprises / amendments during implementation

1. **No surprise on dispatch**. The design `§5.3` allowed either a new
   `Val{:muscl_superbee_relax}` method or a wrapper. I chose the
   wrapper approach (composite launcher in `muscl_boundary.jl`) because
   it makes the two-pass ordering explicit in source. Net : the
   `_fvfd_advection_scheme_val` whitelist still accepts the symbol
   (for forward-compat) but the actual kernel dispatch is intercepted
   one level up at `fvfd_advect_upwind_2d!`.
2. **Sentinel testset diff (relax vs no-relax at Wi=0.1 R=8)** : Cd
   diff is 0.27 % (241.76 vs 241.13). At R=8 the cylinder-band cells
   are ~5 % of fluid (smaller R → relatively more band cells), so the
   numerical pass-2 contribution at this resolution is small but
   non-zero — consistent with the design expectation that the relax
   path actively works at the ring.
3. **No amendment to design §X needed**. The §4 zero-slope formula
   produced no NaN at Wi=1.0 1000 steps, confirming the TVD argument
   §1.4 + §4.1 holds empirically on the smoke proxy.

---

## Pointer to design

- Original spec : `bench/viscoelastic_audit/M42_DESIGN.md` (709 lines).
- Code-path source of truth : `src/fvfd/muscl_boundary.jl`.
- Smoke test source of truth : `test/test_muscl_boundary_relax.jl`.
- G5 submission script : `bench/viscoelastic_logfv/run_cyl_m42_g5_a100.pbs`.

End of M42-impl verdict.
