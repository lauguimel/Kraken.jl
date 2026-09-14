# Next session prompt — Kraken cylinder Cd, M32 closure → M33 polymer scheme upgrade

Copy-paste below to start a fresh session.

---

Continue work on branch `dev-viscoelastic` of Kraken.jl
(worktree `~/Documents/Recherche/Kraken.jl-viscoelastic`).

Resume via orchestrator. Open `~/.claude/skills/orchestrator/SKILL.md` first.
The Boss role continues — Departments + Engineers absorb the detail;
Boss stays strategic.

## State at handoff (2026-05-21 evening)

### What landed this session (6 commits pushed)

```
fabaf7e8 feat(viscoelastic): M32 Phase 3 closes M28-M32 cluster — G3 PASS, gap Wi=1 R-invariant = polymer scheme locus
35543402 fix(viscoelastic): M32 methodology overhaul — rT/Kraken setup audit + canonical matrix prep
f737cddd feat(viscoelastic): M30 Phase 1 (BSD/R/Wi sweeps) + 2a (Bouzidi-FL analytical bench)
034b30dd fix(viscoelastic): M31 frame audit — Phase 0c/M29c-wallstress wall-decomp was 1 LU off
46bb9ad2 feat(viscoelastic): M30 Phase 0 — KRAKEN_SAVE_FIELDS persists rho; rheoTool p(theta) front-pole locus 93.6%
1059ab10 docs(viscoelastic): M29c rolled back; wall-decomp inverts M28/M29 attribution
```

### Production reference (Aqua F64 CUDA + rT Docker shrunk L=15R)

Canonical setup : R=30 (or 40), β=0.59, Re=1, BSD=1.0, `:rusanov`, embedded OFF (qwall), L_up=L_down=15·R, H=4·R, blockage 0.5.

| case | Kraken (Aqua F64) | rT shrunk | gap % |
|---|---|---|---|
| **Newtonian R=30** | **132.08** | **132.37** | **−0.22 %** ✓ G3 PASS |
| Wi=0.1 R=30 β=0.59 | 129.39 | 130.43 | −0.80 % |
| Wi=0.1 R=40 β=0.59 | 129.49 | 130.43 | −0.72 % |
| **Wi=1.0 R=30 β=0.59** | **111.55** | **120.38** | **−7.34 %** ← gap to close |
| **Wi=1.0 R=40 β=0.59** | **111.29** | **120.38** | **−7.55 %** |
| Newtonian R=60 β=1.0 | **132.68** | n/a | stable |
| Wi ≥ 0.1 R=60 β=0.59 | NaN | n/a | polymer-coupled NaN |

The Wi=1 gap is **R-invariant** between R=30 and R=40 → not resolution, not L_down truncation (rT shrunk Wi=1 Cd = 120.38 == non-shrunk Cd = 120.40), not Cd normalisation (Kraken classical Cd ≡ rT Hulsen K bit-for-bit at this setup per C1 audit), not Newtonian baseline (G3 passes at −0.22 %).

**The gap is structurally in the polymer scheme** : `:rusanov` 1st-order upwind on log-conformation Ψ advection over-dissipates the polymer wake stress. The M28 verdict (2026-05-19) was directionally correct ; it was abandoned 2026-05-19 evening because of a frame artifact in the wall decomposition (M29c-wallstress `:phys` frame "matched" Cd_polymer), corrected by M31 (2026-05-20) but not propagated to M30 Phase 2 attribution. The multi-Wi cross-code matrix M32 Phase 3 (this session) is the definitive empirical signal.

### What's parked

- **M30 Phase 2b Bouzidi-FL src/ port** : landed in `src/kernels/dsl/bricks.jl` and `src/kernels/li_bb_2d_v2.jl` behind `wall_bc=:bouzidi_fl` kwarg (default `:halfwayBB` byte-identical regression). NaN'd at Wi=1 100k step 36500 (and Newtonian β=1 also NaN'd at 40000 → not polymer-coupled, it's the BC port itself). Adversarial Claude+Codex audit identified **lag-1 read on `x_ff` in q ≤ 0.5 branch** (production reads `f_in` instead of `f_out`). Proposed fix : two-pass kernel split. Parked because Bouzidi-FL addresses BC pole K/rT 0.59/0.16 (Wi-invariant) which is **sub-dominant in Cd_total at Wi=1** — the wake polymer signal dominates.

  See `bench/viscoelastic_audit/M30_PHASE2B_AUDIT_VERDICT.md` for the lag-bug specification.

## M33 — Polymer scheme upgrade — PLANNED (open this on fresh session)

The mandate `M33` entry specifies it. TL;DR :

- **Mandate** : replace `:rusanov` with a TVD/higher-order scheme on Ψ-advection that closes the −7.3 % Wi=1 gap without NaN at 100k.
- **Best candidate** : `:muscl_superbee` (already in `src/fvfd/operators_2d.jl` from M29b) + the **two-pass fix from M30 Phase 2b audit** (split single-pass kernel into two kernel launches to guarantee lag-0 reads). M29c-v2 NaN at step 92k was lag-induced ; with the two-pass fix it should converge.
- **Alternatives** (defer if MUSCL-superbee suffices) : CUBISTA NVD (rT's scheme, closer apples-to-apples but bigger port), WENO5 (higher order, more bookkeeping).
- **Acceptance** : Cd Kraken R=30 Wi=1 β=0.59 ∈ [115, 122] (closes ≥80 % of the gap), AND R=30 Wi=0.1 stays within 1 % of rT (no regression), AND R=40 Wi=1 reproduces, AND no NaN at 100k Metal F32.
- **Runner** : Codex via `kraken-codex-pilot`. Validation cascade : Pkg.test bit-identical on default `:rusanov`, then matrix re-run on Aqua F64 CUDA against the canonical cases above.

## Reference docs (next Boss reads first)

1. **`.orchestrator/mandate.md` §5** — M32 Phase 3 entry has the definitive matrix table ; M33 entry has the upgrade spec. Read these BEFORE any action.
2. **`.orchestrator/memory/boss.md`** 2026-05-21 entry — meta-lesson : *always run cross-code multi-(Wi, R) matrix FIRST when comparing to a reference code*. Don't waste another session on volume vs wall vs frame side-quests — this is the most expensive meta-lesson of the May session.
3. **`.orchestrator/memory/engineer.md`** 2026-05-21 entries — R=60 polymer-coupled NaN envelope, L_down=60→15R Cd-invariance.
4. **`bench/viscoelastic_audit/M30_PHASE2B_AUDIT_VERDICT.md`** — the lag-bug spec for the two-pass fix (load-bearing for M33).
5. **`bench/viscoelastic_audit/M32_PHASE1_SETUP_AUDIT_VERDICT.md`** — full Kraken vs rT setup comparison + Hulsen K convention reference.
6. Global auto-memory `~/.claude/projects/-Users-guillaume-Documents-Recherche-Kraken-jl/memory/MEMORY.md` — pointers to `feedback_cd_wall_vs_volume`, `feedback_wall_ring_idx_frame`, `feedback_adversarial_codex_claude`, etc.

## Reusable infrastructure (now landed)

- **Canonical Kraken bench** : `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl` (supports `KRAKEN_R_LIST`, `KRAKEN_WI_LIST`, `KRAKEN_BETA_LIST` env vars, `KRAKEN_SAVE_FIELDS=1` persists `(ux, uy, tauxx, tauxy, tauyy, rho, is_solid, ...)` per case as .jls).
- **Aqua F64 PBS templates** : `run_cyl_m32_matrix_a100.pbs` (4-case matrix), `run_cyl_m32_newtonian_sanity_a100.pbs`. Edit `KRAKEN_*_LIST` to retarget.
- **rT shrunk cases** (canonical L=15R) : `bench/rheotool/cylinder_{newtonian_re1, wi0.1, wi1.0}_shrunk15R/` + `bench/rheotool/run_shrunk_matrix.sh` dispatcher (Docker local, ~2-3 h for the 3 cases).
- **Wall-ring `:idx`-frame integration harness** : `bench/scratch/m30_centering_audit/run_centering_audit.jl` (cite this in any future wall-decomp brief, NOT the locked `:phys` `m29c_wallstress` or `m30_kraken_p_profile`).
- **Phase 2a analytical Bouzidi-FL reference** (standalone D2Q9 SRT Couette) : `bench/scratch/m30_phase2a_interpBB_{claude,codex}/m30_phase2a*.jl`. Useful for any future BC port validation pattern.

## Don'ts for the next Boss

- **DO NOT** restart from single-point wall decomposition. The matrix (Newtonian gate + 2 R × 2 Wi minimum) is the entry point. Cross-code parity at multiple Wi is the only signal that disambiguates BC vs polymer vs setup.
- **DO NOT** try to port Bouzidi-FL Phase 2b "for completeness" before M33. It addresses a Wi-invariant sub-dominant contribution. Park stays parked until M33 closes the polymer scheme question.
- **DO NOT** read the raw Department transcripts (`.jsonl`) — use the verdict markdowns. The 6 commits this session have ~12 verdict files in `bench/viscoelastic_audit/M30_*`, `M31_*`, `M32_*` — those are the canonical reference.
- **DO NOT** trust `Cd_total` scalar match alone (cf. Wi=0.1 matches at 0.8 % but K/rT poles are 0.628/0.184 — cancellation pattern masks structural BC error). Use the matrix.

## Aqua workflow reminder

- SSH alias `aqua` is configured in `~/.ssh/config`.
- Remote path : `aqua:Kraken.jl-viscoelastic-run/`.
- Sync : `rsync -az --delete --exclude .git/ --exclude tmp/ --exclude bench/scratch/ --exclude .engineer_brief*.md --exclude .engineer_logs/ ~/Documents/Recherche/Kraken.jl-viscoelastic/ aqua:Kraken.jl-viscoelastic-run/`.
- Submit : `ssh aqua 'cd ~/Kraken.jl-viscoelastic-run && qsub bench/viscoelastic_logfv/<pbs>'`.
- The session's M32 jobs ran in 2-15 min each on A100 F64 CUDA ; budget walltime 1h for single case, 4h for 4-case matrix.

End of prompt.
