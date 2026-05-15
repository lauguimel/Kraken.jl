# FVFD Resource Bilan

Date: 2026-05-15

Context: Kraken-E architecture (see
[kraken_e_fvfd_interface_plan_2026-05-15.md](kraken_e_fvfd_interface_plan_2026-05-15.md))
introduces a neutral FVFD operator core. This document records the provenance
of the FVFD code, its current branch status, the extraction route chosen, and
the ownership rule for future FVFD operator changes.

## Source

Worktree:

```text
/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic
```

Branch at extraction time: `dev-viscoelastic`

Branch HEAD at extraction: (recorded at start of S1, see commit)

Git status of FVFD files at S0:

```text
?? src/fvfd/FVFD.jl
?? src/fvfd/specs.jl
?? src/fvfd/lowering_2d.jl
?? src/fvfd/operators_2d.jl
?? test/test_fvfd_operators_2d.jl
?? docs/design/fvfd_operator_library.md
```

All six files are **untracked** on `dev-viscoelastic` at S0. They have never
been committed. The branch HEAD references `src/fvfd/FVFD.jl` indirectly via
`src/kernels/logconformation_fv_2d.jl` (a consumer), but the FVFD core itself
lives only in the dirty worktree.

Implication: a blind `git merge dev-viscoelastic` from any branch would not
import the FVFD core, because it is not part of the tracked tree. Extraction
must be explicit (copy from worktree) or staged (first commit on
dev-viscoelastic, then cherry-pick).

## Files in scope

Extracted into the FVFD core (owner: `dev/fvfd-core`):

```text
src/fvfd/FVFD.jl             5 lines    module entry
src/fvfd/specs.jl           98 lines    BC + field BC + geometry specs
src/fvfd/lowering_2d.jl    695 lines    spec → backend array lowering
src/fvfd/operators_2d.jl  1061 lines    gradient/divergence/advection/traction
test/test_fvfd_operators_2d.jl 1054 lines    operator canaries (CPU/Metal/CUDA)
docs/design/fvfd_operator_library.md         operator library contract
```

Not extracted (consumer code, stays on `dev-viscoelastic`):

```text
src/kernels/logconformation_fv_2d.jl    1276 lines    viscoelastic consumer
```

Reason: this file depends on the FVFD core but is viscoelastic-specific. It
will be rebased on top of `dev/fvfd-core` when `dev-viscoelastic` next
synchronizes (post-S1).

Not extracted (out of scope entirely):

```text
bench/   hpc/   RheoTool integrations
non-FVFD viscoelastic drivers
benchmark output and logs
```

## Extraction route

Selected: hybrid R2/R4 (see plan §7).

```text
1. Create dev/fvfd-core from main (clean, no slbm-paper or dev-viscoelastic
   baggage).
2. Copy the six FVFD files listed above into dev/fvfd-core worktree.
3. Adapt module entry (FVFD.jl), Kraken.jl includes/exports as needed to
   make src/fvfd/FVFD.jl loadable and tests runnable.
4. Run test/test_fvfd_operators_2d.jl on CPU. Metal/CUDA optional if local
   hardware permits.
5. Commit atomically with provenance message.
6. Notify dev-viscoelastic: the next rebase or merge of dev-viscoelastic
   on dev/fvfd-core makes dev/fvfd-core the canonical owner. From that
   point onward, no FVFD operator change happens on dev-viscoelastic.
```

Rejected alternatives:

- **R0 do nothing**: would duplicate FVFD work in Kraken-E.
- **R1 copy without provenance**: silent duplication, untraceable.
- **R3 standalone package**: API not stable, premature freeze.
- **R2 pure (commit first on dev-viscoelastic)**: the dirty worktree contains
  more than FVFD (benchmark logs, viscoelastic experiments) and the user has
  not requested a cleanup pass on dev-viscoelastic at this time. The hybrid
  R2/R4 captures the same provenance without forcing a dev-viscoelastic
  cleanup as a blocker.

## Ownership rules

After S1 is green:

1. **Owner**: `dev/fvfd-core`. All FVFD operator changes (gradients,
   divergence, advection, traction, BC lowering, embedded geometry, new
   stabilizers) happen on this branch first.

2. **Consumers**:

   - `dev/kraken-e-fvfd-blocks` (LBM bulk + block runtime + FVFD/LBM interface)
   - `dev-viscoelastic` (viscoelastic consumer, log-conformation, polymer
     coupling)
   - any future v0.2/v0.3 architecture branch

   Consumers rebase or merge `dev/fvfd-core` and never independently modify
   FVFD operators. If a consumer needs a new operator or a fix, the change
   goes through `dev/fvfd-core`.

3. **Standalone package**: deferred. Re-evaluate after the Kraken-E phase 1
   ladder (S2..S7) is green AND after 3D operators land. Premature
   extraction would freeze the wrong abstractions (see plan §7 R3).

## Verification commands

At end of S1, must produce:

```bash
git -C /Users/guillaume/Documents/Recherche/Kraken.jl-fvfd-core log --oneline -5
# expects: feat(fvfd): extract operator library from dev-viscoelastic

cd /Users/guillaume/Documents/Recherche/Kraken.jl-fvfd-core
julia --project=. -e 'using Pkg; Pkg.test(test_args=["fvfd"])'
# or, if tests aren't gated by an arg:
julia --project=. -e 'using Test; include("test/test_fvfd_operators_2d.jl")'
# expects: all FVFD canaries green on CPU
```

## Coordination notice for dev-viscoelastic

To be added on `dev-viscoelastic` after S1:

```text
docs/agent/fvfd_consumer_notice_2026-MM-DD.md

Content:
- FVFD operator core now owned by dev/fvfd-core.
- This branch consumes FVFD via rebase/merge of dev/fvfd-core.
- Do not modify src/fvfd/ files on this branch.
- Operator fixes/extensions go through dev/fvfd-core first.
```

Reminder for the orchestrator: this notice must be added when
`dev-viscoelastic` is next touched, not during S1. S1 stays scoped to
`dev/fvfd-core` extraction.
