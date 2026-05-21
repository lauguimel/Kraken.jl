## M30 Phase 2b port audit — Codex solo

### Q1 — production code locations
- `bricks.jl`: `_bouzidi_fl_post_value(qw::T,...has_ff::Bool) where {T}` L404-L419; `ApplyBouzidiFLPostCollide` struct/args/phase/apply L421-L550.
- `li_bb_2d_v2.jl`: `_TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_SPEC` L56-L61; `wall_bc` dispatch guard/`Val` L123-L134; `Val{:bouzidi_fl}` hook L152-L164.

### Q2 — Phase 2a q-convention
- Claude ref: header L29-L31 says `|x_b-x_w|/|x_b-x_f|`, but actual precompute uses distance `x_f -> x_w`: L164-L170 and positive `s` in `x_f+s*c` L173-L194; BFL L452-L469.
- Codex ref: `cut_fraction` solves `x+s*c` L82-L98; stores q/xw L125-L131; BFL L182-L189.
- AGREE: yes in code: both use fluid-node-to-wall `s`, not `1-s`.

### Q3 — production q-convention vs Phase 2a
- Match. `q_wall` is fraction from fluid node L257-L260; `xf=i-1` L277; solves `x_f+t*c_q` L292-L305. It is not `1-q`.

### Q4 — branch logic + lag
- q <= 0.5 branch: algebra matches L407-L410, but `x_ff` read diverges.
- q > 0.5 branch: algebra matches L415-L417.
- Lag on x_f: lag-0 (`f_out` snapshots L430-L437).
- Lag on x_ff: lag-1 (`f_in[...]`, e.g. L448,L462,L476,L490,L504,L518,L532,L546).
- Lag on x_qb: lag-0 (`f_out` snapshots L430-L437).
- Phase 2a refs use current `fpost` for all three: Claude L452-L469; Codex L183-L189. Lag-1 on `x_ff` is a defect vs canonical.

### Q5 — moving-wall correction term
- Present/scaled: deltas L441,L455,L469,L483,L497,L511,L525,L539; q> scales by `inv_two_qw` L415-L417. Cylinder `uwx/uwy` are zero L289-L293, so term vanishes. `rho_w=ρ_out[i,j]` L427 is stale: `WriteMoments` occurs later L656-L658 in spec order L56-L60.

### Q6 — root cause hypothesis
- Pick: B. q convention ok; q=0 is guarded; no cross-cell `f_out` read race; stationary wall removes Q5. Most likely: single-pass lag/storage mismatch (`x_ff` lag-1, post-only BFL in pull-collide storage).
- Confidence: medium-high.
- Proposed minimal fix: split collision and BFL boundary so `x_f`, `x_ff`, `qbar`, and `ρ` are current, or rework BFL as a pull-storage pre-phase brick.
