# M31 frame audit — adversarial verdict (Claude + Codex synthesis)

Date: 2026-05-20
Engines: Claude (Anthropic Opus 4.7) + Codex (OpenAI gpt-5)
Step 1 (Claude solo): `bench/viscoelastic_audit/M31_FRAME_AUDIT_CLAUDE.md`
Step 2 (Codex solo): `bench/viscoelastic_audit/M31_FRAME_AUDIT_CODEX.md`
Codex log: `.engineer_logs/M31-audit-codex_20260520_115708.log`

The two engines reached **OPPOSITE verdicts on Q4 in step 1**, and after careful
synthesis, **Claude REVERSES position to align with Codex.** The verdict on
Q4–Q6 is now the Codex finding. The reasoning is laid out below.

---

## Per-question comparison and synthesis

### Q1 — Cd_kraken final-assembly line
- Claude: `src/drivers/viscoelastic_logfv_2d.jl:591–604` (final reduction),
  `:515–521` (per-step accumulation),
  `_run_viscoelastic_logfv_step_channel_coupled_2d`.
- Codex: same lines, additionally cites `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl:312–315`
  to show that `SUMMARY.csv`'s `Cd_kraken` column is literally `result.Cd`.
- **Agreement: YES** (Codex's extra `SUMMARY.csv` traceback is value-add).

### Q2 — Formulas & frames per component
- Claude and Codex AGREE on all formulas:
  - `Cd_s` is a frame-INDEPENDENT cut-link MEA sum (no centre referenced).
  - `Cd_p` is a q-wall surface quadrature with `xw = (i−1) + q_w·c_q`,
    `cx = drag_cx = L_up·R`, integrated as `Σ τ·n·ds`.
  - `Cd_bsd` delegates to the same `compute_polymeric_drag_2d` machinery with
    the same `cx, cy`.
  - `Cd_kraken = Cd_s + Cd_p − Cd_bsd`.
- **Agreement: YES**.

### Q3 — Rasterisation `(i−1, j−1)` convention
- Both confirm `xf = i−1, yf = j−1` at line 277 of `src/kernels/li_bb_2d.jl`,
  and that the physical centre `(cx_phys, cy_phys) = (450, 59.5)` sits at
  raw-index coordinates `(cx_phys + 1, cy_phys + 1) = (451, 60.5)`.
- **Agreement: YES**.

### Q4 — Physically correct frame
- **Claude step-1 vote**: A (`:phys`), rationale: driver uses physical-frame
  formula and centre `cx_phys`, that's also the centre rheoTool uses.
- **Codex vote**: B (`:idx` in M30's nomenclature), rationale: M30 audit script
  uses `dx = i − cx_lu`, so the M30 `:idx` mode (with `cx_lu = cx_phys + 1`)
  reproduces the physically-correct formula `(i−1) − cx_phys`. M30's `:phys`
  mode (with `cx_lu = cx_phys`) gives `dx = i − cx_phys`, which is OFF BY 1 LU.
- **Disagreement: YES at step 1.**

**Synthesis (verifying against the actual audit script)**:
`bench/scratch/m30_centering_audit/run_centering_audit.jl:151–152` reads
```julia
dx = Float64(i) - cx_lu
dy = Float64(j) - cy_lu
```
With `cx_lu = cx_phys` (the `:phys` mode), this is `dx = i − cx_phys` — but the
wall-cell physical coordinate is `(i−1)`, NOT `i`. So `:phys` mixes raw-index
`i` with the physical centre `cx_phys`, yielding a 1-LU systematic offset of
the moment arm origin. With `cx_lu = cx_phys + 1` (the `:idx` mode), this
becomes `dx = i − (cx_phys + 1) = (i−1) − cx_phys`, which IS the physical-frame
formula identical to what the driver uses internally.

**Conclusion**: M30's label `:phys` is a misnomer. Codex is correct.
**The physically defensible frame for the M30 cell-ring audit is `:idx` (Option B).**

- **Final agreement on Q4: YES, both engines now vote B (`:idx`)** — Claude
  reverses position after the synthesis.

### Q5 — Cd_polymer in correct frame vs rheoTool
With Q4 = B (`:idx`), the relevant numbers are:

| source | Cd_polymer | gap vs rT 13.45 |
|---|---|---|
| rheoTool exact-disk (M29c-wallstress)               | **13.45** | 0       |
| M30 ring `:phys` (mis-frame, Phase 0c value)        | 13.46     | +0.07 % |
| M30 ring `:idx`  (correct frame, kernel-consistent) | **10.82** | **−19.5 %** |
| Driver-internal `Cd_p` stored in M30 snapshot       | 11.49     | −14.6 % |
| Aqua F64 M29b Cd_p (M29c-wallstress)                | 13.40 in `:phys` frame | (matches in mis-frame only) |

The driver-internal `Cd_p = 11.49` is computed in the **physically-correct
frame** (via `compute_polymeric_drag_2d` with `cx = cx_phys` and `xw = i−1+q_w·c_q`),
yet differs from the M30 `:idx` ring value 10.82 by ~6 %. That residual is
*quadrature difference* (driver does q-wall cut-link surface quadrature on
fluid cells; M30 does cell-ring traction integral) — both are physical-frame
numbers and both are LOW vs rT (14.6 % and 19.5 % respectively).

**Verdict: the M29c-wallstress claim "M29b polymer matches rheoTool to 0.05"
is FALSIFIED.** That match was a coincidence of mis-framed ring sampling and
the staircase asymmetry. **Kraken's polymer wall drag is genuinely
under-predicted by 15–20 %.**

### Q6 — Cd_kraken bug or convention?
- Claude step-1: "internal-consistency convention" (predicated on a
  wrong Q4 vote).
- Codex: NOT a smoking-gun driver bug — but the **M29c-wallstress
  conclusion is wrong, and the Cd_pressure decomposition published in
  Phase 0c was mis-framed.**

**Synthesis: the driver itself (`compute_polymeric_drag_2d`) IS in the
correct physical frame**: it uses `xw = (i−1) + q_w·c_q` and `cx = cx_phys`.
The driver's stored `Cd_p = 11.49` and `Cd_kraken = 111.09` are physically
defensible numbers. **There is NO load-bearing frame bug in the production
driver.**

The load-bearing problem is in the **post-processing audit scripts**
(M30/Phase 0c) that mixed raw-index `i` with a physical centre. The
mis-framed ring totals (111.20 `:phys`) "match" the driver `Cd_kraken` (111.09)
only because of a chain of coincidences in the cancellation
pressure+solvent+polymer. The correctly-framed `:idx` ring (108.63) is the
trustworthy reference for the cell-ring decomposition.

**Implication**: M28–M30 stored `Cd_kraken` values are valid. But the
**Cd_pressure ≈ 76.6 / Cd_polymer ≈ 13.46 component split published in M30
Phase 0c and used by M29c-wallstress is mis-framed**. The correct
decomposition is **Cd_pressure ≈ 76.6, Cd_solvent ≈ 21.2, Cd_polymer ≈ 10.8**
in the `:idx`/physically-correct ring, leaving a total ring of 108.6 which is
~2 % below the driver MEA total because of the quadrature difference noted
above.

### Codex side-finding (worth surfacing)
Codex flagged a numerical inconsistency in
`bench/viscoelastic_audit/M30_CENTERING_AUDIT_VERDICT.md:55–56`: the text
claims "both ring totals reconcile with `Cd_kraken` within ~0.1 %", but the
`:idx` total 108.63 vs stored 111.09 is **−2.21 %**, not 0.1 %. The 0.1 %
applies only to the `:phys` mis-frame. This sentence in M30 should be
corrected.

---

## Final verdict template

```
### Q1 — Cd_kraken final-assembly line
- Both: src/drivers/viscoelastic_logfv_2d.jl:591–604, function
  _run_viscoelastic_logfv_step_channel_coupled_2d; per-step at :515–521.
- Agreement: YES.

### Q2 — formulas & frames
- Cd_s frame-independent cut-link MEA sum.
- Cd_p, Cd_bsd q-wall surface quadrature with cx=cx_phys, xw=(i−1)+q_w·c_q
  → CORRECT physical frame in the driver.
- Cd_kraken = Cd_s + Cd_p − Cd_bsd.

### Q3 — rasterisation (i−1, j−1) convention
- Confirmed: line 277 of src/kernels/li_bb_2d.jl, both engines agree.

### Q4 — physically correct frame
- Claude step-1: A (:phys). Codex: B (:idx).
- Synthesis: M30's :phys mode uses dx = i − cx_phys with i as raw index → 1-LU
  mis-frame. M30's :idx mode uses dx = i − (cx_phys + 1) = (i−1) − cx_phys =
  driver-correct physical formula. Codex is right.
- Final: B (:idx). Agreement after synthesis: YES.

### Q5 — Cd_polymer in correct frame vs rheoTool
- M30 ring (:idx, correct) = 10.82  vs rT = 13.45 → −19.5 %.
- Driver stored Cd_p        = 11.49  vs rT = 13.45 → −14.6 %.
- The M29c "matches to 0.05" claim is FALSIFIED. Polymer is under-predicted
  by 15–20 %.

### Q6 — Cd_kraken bug or convention?
- Driver code is in the correct physical frame; Cd_kraken stored is valid.
- M30/Phase 0c POST-PROCESSING ring decomposition was mis-framed.
- Verdict: NOT a load-bearing driver bug; IT IS a load-bearing audit-script bug
  that invalidates the M29c-wallstress polymer-match conclusion and the Phase
  0c component decomposition.

### Confidence: HIGH after synthesis.

### Boss decision implication
- H1 (LBM ρ-BC) ranking from Phase 0c needs RE-RANKING. The +9.13 pt
  Cd_pressure gap is real, but the polymer was ALSO genuinely off by 15–20 %,
  so the decomposition of the total Cd_kraken vs rT gap is:
    Cd_pressure deficit  ≈ +9 to +10 pts (Kraken low)
    Cd_polymer  deficit  ≈ +2.6 pts       (Kraken low, was previously 0)
    Cd_solvent           ≈ −0.6 to −1.4 pts (Kraken high, small)
  H1 still primary, but H2 (polymer wall stencil / Cd_polymer accuracy) is
  RE-RANKED UP from "demoted" to "secondary/co-primary" — it is no longer the
  case that polymer wall stress agrees with rheoTool.
- Commit Phase 0c verdict + centering audit + M31 in ONE batch: yes, because
  M31 retroactively reinterprets both. They should ship together with a
  prominent note that the Phase 0c "Cd_p_phys = 13.46" was mis-framed and the
  corrected value is Cd_p_idx = 10.82 (or driver-internal 11.49).

### Memory candidates
1. `feedback_audit_script_frame_consistency` — When raster uses x_phys = i−1,
   ALL ring/quadrature post-processing must use `dx = (i−1) − cx` and NOT
   `dx = i − cx`. Mixing raw indices with a physical centre is a 1-LU bias
   that does NOT cancel — on R=30 it gave a 24 % shift of Cd_polymer. M30 was
   bitten by this; the bug was caught by adversarial Claude+Codex on M31.
2. `feedback_polymer_wall_drag_underpredicted` — Kraken `viscoelastic_logfv`
   under-predicts the polymer wall drag on R=30 Wi=1 β=0.59 by 15–20 %
   relative to rheoTool. Previously thought to match (M29c-wallstress) due to
   a mis-framed audit; the genuine number is Cd_p ≈ 10.8 (ring) / 11.5
   (driver) vs rT 13.45.
3. `feedback_adversarial_cross_engine_revealed_phys_frame_bug` — Adversarial
   Claude+Codex on Q4 disagreed at step 1; the synthesis (auditing the M30
   harness source directly) revealed Codex was right and Claude was wrong.
   This is the third demonstrated win of cross-engine adversarial protocol;
   Claude-on-Claude in step 1 would have confirmed the mis-frame and shipped
   the wrong conclusion.
```

## Files

- `bench/viscoelastic_audit/M31_FRAME_AUDIT_CLAUDE.md` (Claude step-1 solo)
- `bench/viscoelastic_audit/M31_FRAME_AUDIT_CODEX.md` (Codex step-2 solo)
- `bench/viscoelastic_audit/M31_FRAME_AUDIT_VERDICT.md` (this synthesis)
- `.engineer_logs/M31-audit-codex_20260520_115708.log` (Codex execution log)
- `.engineer_brief_M31_audit_codex.md` (Codex brief, ephemeral)

## Process note

Codex flagged that an `rg` command incidentally matched
`bench/viscoelastic_audit/M31_FRAME_AUDIT_CLAUDE.md` during source search and
"printed snippets". Codex states it did not intentionally use the file
further. The conclusions in Codex's file are sourced from primary code +
M30/M29 verdicts, and the two engines initially disagreed on Q4 — which is
strong evidence of independence. The verdict is robust to this leak.
