# Cylinder Cd HPC DoE Plan - 2026-05-18

## A. Parameter manifests

Comma-valued `qsub -v` values are passed with literal quotes (`\"...\"`) so
the PBS variable-list parser receives one assignment per `KRAKEN_*` key.

Tier 1: domain and embedded-boundary discriminator, 12 cases.

```bash
# NEW — requires extension: KRAKEN_L_UP_LIST (see D)
# NEW — requires extension: KRAKEN_L_DOWN_LIST (see D)
# NEW — requires extension: KRAKEN_EMBEDDED_GRADIENT (see D)
# NEW — requires extension: KRAKEN_EMBEDDED_ADVECTION (see D)
# NEW — requires extension: KRAKEN_EMBEDDED_FORCE (see D)
# NEW — requires extension: KRAKEN_EMBEDDED_DRAG (see D)
# NEW — requires extension: KRAKEN_EMBEDDED_GEOMETRY (see D)
qsub -v \
KRAKEN_BACKEND=cuda,\
KRAKEN_FT=float64,\
KRAKEN_BETA_LIST=0.5,\
KRAKEN_WI_LIST=0.1,\
KRAKEN_RE_LIST=1.0,\
KRAKEN_R_LIST=30,\
KRAKEN_BSD_LIST=1.0,\
KRAKEN_U_MEAN=0.005,\
KRAKEN_MAX_STEPS_BASE=100000,\
KRAKEN_AVG_WINDOW_FRAC=0.2,\
KRAKEN_L_UP_LIST=\"4,15,20\",\
KRAKEN_L_DOWN_LIST=\"8,15,60\",\
KRAKEN_EMBEDDED_GRADIENT=\"0,1,1,1\",\
KRAKEN_EMBEDDED_ADVECTION=\"0,1,0,1\",\
KRAKEN_EMBEDDED_FORCE=\"0,1,0,1\",\
KRAKEN_EMBEDDED_DRAG=\"0,1,1,1\",\
KRAKEN_EMBEDDED_GEOMETRY=\"qwall,qwall,circle,circle\",\
KRAKEN_OUTPUT_DIR=results/viscoelastic_logfv/cyl_doe_tier1 \
bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs
```

Tier 2: physical sweep at the promoted configuration. The concrete command
below uses the reference-domain/circle candidate; if Tier 1 selects another
configuration, change only the seven new knobs.

```bash
# NEW — requires extension: KRAKEN_L_UP_LIST, KRAKEN_L_DOWN_LIST (see D)
# NEW — requires extension: KRAKEN_EMBEDDED_* and KRAKEN_EMBEDDED_GEOMETRY (see D)
qsub -v \
KRAKEN_BACKEND=cuda,\
KRAKEN_FT=float64,\
KRAKEN_BETA_LIST=\"0.3,0.5,0.7\",\
KRAKEN_WI_LIST=\"0.1,0.2,0.3\",\
KRAKEN_RE_LIST=\"0.1,1.0\",\
KRAKEN_R_LIST=50,\
KRAKEN_BSD_LIST=1.0,\
KRAKEN_U_MEAN=0.005,\
KRAKEN_MAX_STEPS_BASE=100000,\
KRAKEN_AVG_WINDOW_FRAC=0.2,\
KRAKEN_L_UP_LIST=20,\
KRAKEN_L_DOWN_LIST=60,\
KRAKEN_EMBEDDED_GRADIENT=1,\
KRAKEN_EMBEDDED_ADVECTION=1,\
KRAKEN_EMBEDDED_FORCE=1,\
KRAKEN_EMBEDDED_DRAG=1,\
KRAKEN_EMBEDDED_GEOMETRY=circle,\
KRAKEN_OUTPUT_DIR=results/viscoelastic_logfv/cyl_doe_tier2 \
bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs
```

Tier 3: production mesh convergence at `beta=0.5`, `Re_R=1`.

```bash
# NEW — requires extension: KRAKEN_L_UP_LIST, KRAKEN_L_DOWN_LIST (see D)
# NEW — requires extension: KRAKEN_EMBEDDED_* and KRAKEN_EMBEDDED_GEOMETRY (see D)
qsub -v \
KRAKEN_BACKEND=cuda,\
KRAKEN_FT=float64,\
KRAKEN_BETA_LIST=0.5,\
KRAKEN_WI_LIST=\"0.1,0.2,0.3\",\
KRAKEN_RE_LIST=1.0,\
KRAKEN_R_LIST=\"30,50,80,100\",\
KRAKEN_BSD_LIST=1.0,\
KRAKEN_U_MEAN=0.005,\
KRAKEN_MAX_STEPS_BASE=100000,\
KRAKEN_AVG_WINDOW_FRAC=0.2,\
KRAKEN_L_UP_LIST=20,\
KRAKEN_L_DOWN_LIST=60,\
KRAKEN_EMBEDDED_GRADIENT=1,\
KRAKEN_EMBEDDED_ADVECTION=1,\
KRAKEN_EMBEDDED_FORCE=1,\
KRAKEN_EMBEDDED_DRAG=1,\
KRAKEN_EMBEDDED_GEOMETRY=circle,\
KRAKEN_OUTPUT_DIR=results/viscoelastic_logfv/cyl_doe_tier3 \
bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs
```

## B. Three-tier sweep plan

Cost model: `t(R) ~= 15 min * (R/30)^2` on A100 F64, from the project
baseline of about 15 min per `R=30` case at `max_steps=100000`.

Tier 1 purpose: separate the historical domain mismatch from embedded-boundary
operator choices. Domains are zipped pairs `(4,8)`, `(15,15)`, `(20,60)`.
The rheoTool block mesh spans `x in [-20R,+60R]`; its `y=0..2R`
half-channel implies full height `H=4R`, matching the driver harness.
Embedded modes are zipped tuples: all-off/qwall, all-on/qwall,
grad+drag/circle, all-on/circle. Count: `1*1*1*1*1*3*4 = 12`.
Per-case time: 0.25 h at `R=30`. Total: about 3.0 h. Stop when the
best Cd agreement and stability choice is unambiguous: domain shift changes
Cd by less than 1% between `(15,15)` and `(20,60)`, or one domain/embedded
mode is clearly outside the rheoTool `R=30` Cd band.

Tier 2 purpose: physical sweep once Tier 1 fixes the baseline configuration.
Use `Wi={0.1,0.2,0.3}`, `Re_R={0.1,1.0}`, `beta={0.3,0.5,0.7}`, `R=50`,
`bsd_fraction=1`. Count: `3*2*3 = 18`. Per-case time: about 0.69 h at
`R=50`. Total: about 12.5 h. Stop when Cd, `min_det_C`, and `nan_flag`
identify the stable publishable range; failed cases must fail by recorded
non-finiteness, not by silent missing output.

Tier 3 purpose: mesh convergence at production `Re_R=1`, `beta=0.5`, and
`Wi={0.1,0.2,0.3}`. Use `R={30,50,80,100}` at the Tier 1 baseline.
Count: `3*4 = 12`. Per-case times: 0.25 h, 0.69 h, 1.78 h, 2.78 h.
Total: about 16.5 h. Stop when `R=80` to `R=100` changes Cd by less than
1% for finite cases, or when the first reproducible stability boundary is
located with `first_nonfinite_step` and `min_det_C`.

## C. Preflight script

File 2 is `bench/viscoelastic_audit/cylinder_doe_preflight.sh`. It checks
that the PBS template, Julia harness, rheoTool mesh dictionary, and this DoE
plan exist; greps the current harness for every env var used in the manifests;
prints Tier ETA estimates; and lists files that need plumbing.

Expected footprint before applying D:

```text
[PASS] file bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs
[PASS] file bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl
[PASS] file bench/rheotool/cylinder_wi0.1/system/blockMeshDict
[PASS] DoE plan is non-empty
[PASS] KRAKEN_BETA_LIST
[PASS] KRAKEN_WI_LIST
[MISS] KRAKEN_L_UP_LIST
[MISS] KRAKEN_EMBEDDED_GEOMETRY
Tier 1: 12 cases, ETA ~3.0h
Tier 2: 18 cases, ETA ~12.5h
Tier 3: 12 cases, ETA ~16.5h
Files needing modification:
- bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl
- bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs
Missing env vars: KRAKEN_L_UP_LIST ...
```

## D. Bench-harness extensions

The new list semantics are intentional: `KRAKEN_L_UP_LIST` and
`KRAKEN_L_DOWN_LIST` are zipped into domain pairs, and the five embedded
lists are zipped into embedded-mode tuples. Those two zipped dimensions are
then crossed with the existing physical lists.

Line actions against the current harness:

- `KRAKEN_L_UP_LIST`: ADD header after line 13, ADD parser constant after
  line 63, REPLACE hard-coded `L_UP` use at lines 67/174/205, ADD CSV/row
  fields at lines 72/189.
- `KRAKEN_L_DOWN_LIST`: same action as `KRAKEN_L_UP_LIST` for `L_DOWN`.
- `KRAKEN_EMBEDDED_GRADIENT`: ADD header/parser after lines 13/63, REPLACE
  hard-coded driver kwarg at line 215, ADD CSV/row fields at lines 72/189.
- `KRAKEN_EMBEDDED_ADVECTION`: ADD header/parser after lines 13/63, REPLACE
  hard-coded driver kwarg at line 216, ADD CSV/row fields at lines 72/189.
- `KRAKEN_EMBEDDED_FORCE`: ADD header/parser after lines 13/63, REPLACE
  hard-coded driver kwarg at line 216, ADD CSV/row fields at lines 72/189.
- `KRAKEN_EMBEDDED_DRAG`: ADD header/parser after lines 13/63, REPLACE
  hard-coded driver kwarg at line 216, ADD CSV/row fields at lines 72/189.
- `KRAKEN_EMBEDDED_GEOMETRY`: ADD header/parser after lines 13/63, REPLACE
  hard-coded driver kwarg at line 215, ADD CSV/row fields at lines 72/189.

PBS actions for all seven new env vars: ADD override comments after current
line 30, ADD `export KRAKEN_*=...` defaults after line 50, and ADD echo
diagnostics after line 65.

Harness diff:

```diff
--- a/bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl
+++ b/bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl
@@ -13,6 +13,13 @@
 #   KRAKEN_BSD_LIST         "1.0"                (user directive: use 1.0, not 0.75)
+#   KRAKEN_L_UP_LIST        "15.0"               (zipped with KRAKEN_L_DOWN_LIST)
+#   KRAKEN_L_DOWN_LIST      "15.0"
+#   KRAKEN_EMBEDDED_GRADIENT  "0"                (zipped embedded-mode tuple)
+#   KRAKEN_EMBEDDED_ADVECTION "0"
+#   KRAKEN_EMBEDDED_FORCE     "0"
+#   KRAKEN_EMBEDDED_DRAG      "0"
+#   KRAKEN_EMBEDDED_GEOMETRY  "qwall"            (qwall | circle)
 #   KRAKEN_U_MEAN           "0.005"              (controls Re via Re_R = u_mean*R/nu_total)
@@ -58,13 +65,47 @@
 parse_int_list(name, default) =
     [parse(Int, strip(x)) for x in split(get(ENV, name, default), ",")]
+parse_symbol_list(name, default) =
+    [Symbol(strip(x)) for x in split(get(ENV, name, default), ",")]
+
+function parse_bool_token(x)
+    s = lowercase(strip(x))
+    s in ("1", "true", "t", "yes", "y", "on") && return true
+    s in ("0", "false", "f", "no", "n", "off") && return false
+    throw(ArgumentError("invalid boolean token: $(x)"))
+end
+
+parse_bool_list(name, default) =
+    [parse_bool_token(x) for x in split(get(ENV, name, default), ",")]
+
+function zip_equal(name, lists...)
+    n = length(first(lists))
+    all(l -> length(l) == n, lists) ||
+        throw(ArgumentError("$(name) lists must have equal length"))
+    return collect(zip(lists...))
+end
 
 const BETA_LIST       = parse_list("KRAKEN_BETA_LIST",       "0.3,0.5,0.7")
 const WI_LIST         = parse_list("KRAKEN_WI_LIST",         "0.1,0.3,0.5")
 const RE_LIST         = parse_list("KRAKEN_RE_LIST",         "0.1,1.0")
 const R_LIST          = parse_int_list("KRAKEN_R_LIST",      "30,50,80")
 const BSD_LIST        = parse_list("KRAKEN_BSD_LIST",        "0.0,0.5,1.0")
+const L_UP_LIST       = parse_list("KRAKEN_L_UP_LIST",       "15.0")
+const L_DOWN_LIST     = parse_list("KRAKEN_L_DOWN_LIST",     "15.0")
+const EMBEDDED_GRADIENT_LIST  = parse_bool_list("KRAKEN_EMBEDDED_GRADIENT",  "0")
+const EMBEDDED_ADVECTION_LIST = parse_bool_list("KRAKEN_EMBEDDED_ADVECTION", "0")
+const EMBEDDED_FORCE_LIST     = parse_bool_list("KRAKEN_EMBEDDED_FORCE",     "0")
+const EMBEDDED_DRAG_LIST      = parse_bool_list("KRAKEN_EMBEDDED_DRAG",      "0")
+const EMBEDDED_GEOMETRY_LIST  = parse_symbol_list("KRAKEN_EMBEDDED_GEOMETRY", "qwall")
+all(g -> g in (:qwall, :circle), EMBEDDED_GEOMETRY_LIST) ||
+    throw(ArgumentError("KRAKEN_EMBEDDED_GEOMETRY values must be qwall or circle"))
+const GEOM_CONFIGS = zip_equal("KRAKEN_L_UP_LIST/KRAKEN_L_DOWN_LIST",
+                                 L_UP_LIST, L_DOWN_LIST)
+const EMBEDDED_CONFIGS = zip_equal("KRAKEN_EMBEDDED_*",
+    EMBEDDED_GRADIENT_LIST, EMBEDDED_ADVECTION_LIST, EMBEDDED_FORCE_LIST,
+    EMBEDDED_DRAG_LIST, EMBEDDED_GEOMETRY_LIST)
 const U_MEAN          = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.005"))
 const MAX_STEPS_BASE  = parse(Int,     get(ENV, "KRAKEN_MAX_STEPS_BASE", "100000"))
 const AVG_WINDOW_FRAC = parse(Float64, get(ENV, "KRAKEN_AVG_WINDOW_FRAC", "0.2"))
-const L_UP, L_DOWN    = 15.0, 15.0
 const JOB_ID          = get(ENV, "PBS_JOBID", "manual")
@@ -72,8 +113,10 @@
 const CSV_COLUMNS = [
     :timestamp, :backend, :FT, :R, :Wi, :Re_R, :beta, :bsd_fraction,
+    :L_up, :L_down, :embedded_gradient, :embedded_advection,
+    :embedded_force, :embedded_drag, :embedded_geometry,
     :u_mean, :nu_total, :nu_s, :nu_p, :lambda, :max_steps, :avg_window,
@@ -167,20 +210,26 @@
-function case_tag(beta, wi, re, R, bsd)
+function case_tag(beta, wi, re, R, bsd, L_up, L_down, eg, ea, ef, ed, geom)
     fmt(x) = replace(@sprintf("%.4g", x), "." => "p", "-" => "m")
-    return "beta$(fmt(beta))_wi$(fmt(wi))_re$(fmt(re))_R$(R)_bsd$(fmt(bsd))"
+    b(x) = x ? "1" : "0"
+    return "beta$(fmt(beta))_wi$(fmt(wi))_re$(fmt(re))_R$(R)_bsd$(fmt(bsd))" *
+           "_Lup$(fmt(L_up))_Ldn$(fmt(L_down))_eg$(b(eg))_ea$(b(ea))" *
+           "_ef$(b(ef))_ed$(b(ed))_geom$(geom)"
 end
 
-function run_case(beta, wi, re_target, R, bsd, summary_path)
+function run_case(beta, wi, re_target, R, bsd, domain_cfg, embedded_cfg, summary_path)
+    L_up, L_down = domain_cfg
+    embedded_gradient, embedded_advection, embedded_force, embedded_drag,
+        embedded_geometry = embedded_cfg
     H = 4 * R
-    Nx = ceil(Int, (L_UP + L_DOWN) * R)
+    Nx = ceil(Int, (L_up + L_down) * R)
@@ -183,13 +232,18 @@
-    tag = case_tag(beta, wi, re_target, R, bsd)
+    tag = case_tag(beta, wi, re_target, R, bsd, L_up, L_down,
+                   embedded_gradient, embedded_advection, embedded_force,
+                   embedded_drag, embedded_geometry)
@@ -189,6 +243,10 @@
         :R => R, :Wi => wi, :Re_R => re_target, :beta => beta,
         :bsd_fraction => bsd, :u_mean => U_MEAN,
+        :L_up => L_up, :L_down => L_down,
+        :embedded_gradient => Int(embedded_gradient),
+        :embedded_advection => Int(embedded_advection),
+        :embedded_force => Int(embedded_force), :embedded_drag => Int(embedded_drag),
+        :embedded_geometry => string(embedded_geometry),
@@ -205,16 +263,17 @@
-            radius=R, H=H, L_up=L_UP, L_down=L_DOWN,
+            radius=R, H=H, L_up=L_up, L_down=L_down,
@@ -215,8 +274,9 @@
-            embedded_geometry=:qwall, embedded_gradient=false,
-            embedded_advection=false, embedded_force=false, embedded_drag=false,
+            embedded_geometry=embedded_geometry, embedded_gradient=embedded_gradient,
+            embedded_advection=embedded_advection, embedded_force=embedded_force,
+            embedded_drag=embedded_drag,
@@ -279,14 +339,16 @@
     println("beta=$BETA_LIST | Wi=$WI_LIST | Re=$RE_LIST | R=$R_LIST | bsd=$BSD_LIST")
+    println("domains=$GEOM_CONFIGS | embedded=$EMBEDDED_CONFIGS")
@@ -283,11 +345,12 @@
     n_total = length(BETA_LIST) * length(WI_LIST) * length(RE_LIST) *
-              length(R_LIST) * length(BSD_LIST)
+              length(R_LIST) * length(BSD_LIST) *
+              length(GEOM_CONFIGS) * length(EMBEDDED_CONFIGS)
@@ -288,10 +351,11 @@
     for beta in BETA_LIST, wi in WI_LIST, re in RE_LIST,
-        R in R_LIST, bsd in BSD_LIST
+        R in R_LIST, bsd in BSD_LIST, domain_cfg in GEOM_CONFIGS,
+        embedded_cfg in EMBEDDED_CONFIGS
@@ -293,7 +357,7 @@
-        run_case(beta, wi, re, R, bsd, summary_path)
+        run_case(beta, wi, re, R, bsd, domain_cfg, embedded_cfg, summary_path)
```

PBS diff:

```diff
--- a/bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs
+++ b/bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs
@@ -30,6 +30,13 @@
 #   KRAKEN_BSD_LIST="0.0,0.5,1.0"
+#   KRAKEN_L_UP_LIST="15.0"
+#   KRAKEN_L_DOWN_LIST="15.0"
+#   KRAKEN_EMBEDDED_GRADIENT="0"
+#   KRAKEN_EMBEDDED_ADVECTION="0"
+#   KRAKEN_EMBEDDED_FORCE="0"
+#   KRAKEN_EMBEDDED_DRAG="0"
+#   KRAKEN_EMBEDDED_GEOMETRY="qwall"
@@ -50,6 +57,13 @@
 export KRAKEN_BSD_LIST="${KRAKEN_BSD_LIST:-0.0,0.5,1.0}"
+export KRAKEN_L_UP_LIST="${KRAKEN_L_UP_LIST:-15.0}"
+export KRAKEN_L_DOWN_LIST="${KRAKEN_L_DOWN_LIST:-15.0}"
+export KRAKEN_EMBEDDED_GRADIENT="${KRAKEN_EMBEDDED_GRADIENT:-0}"
+export KRAKEN_EMBEDDED_ADVECTION="${KRAKEN_EMBEDDED_ADVECTION:-0}"
+export KRAKEN_EMBEDDED_FORCE="${KRAKEN_EMBEDDED_FORCE:-0}"
+export KRAKEN_EMBEDDED_DRAG="${KRAKEN_EMBEDDED_DRAG:-0}"
+export KRAKEN_EMBEDDED_GEOMETRY="${KRAKEN_EMBEDDED_GEOMETRY:-qwall}"
@@ -65,6 +79,8 @@
 echo "Sweeps: beta=$KRAKEN_BETA_LIST | Wi=$KRAKEN_WI_LIST | Re=$KRAKEN_RE_LIST | R=$KRAKEN_R_LIST | bsd=$KRAKEN_BSD_LIST"
+echo "Domains: L_up=$KRAKEN_L_UP_LIST | L_down=$KRAKEN_L_DOWN_LIST"
+echo "Embedded: grad=$KRAKEN_EMBEDDED_GRADIENT adv=$KRAKEN_EMBEDDED_ADVECTION force=$KRAKEN_EMBEDDED_FORCE drag=$KRAKEN_EMBEDDED_DRAG geom=$KRAKEN_EMBEDDED_GEOMETRY"
```

One-line behavior checks after applying the diff:

```bash
grep -n 'KRAKEN_L_UP_LIST' bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs
grep -n 'KRAKEN_L_DOWN_LIST' bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs
grep -n 'KRAKEN_EMBEDDED_GRADIENT' bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs
grep -n 'KRAKEN_EMBEDDED_ADVECTION' bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs
grep -n 'KRAKEN_EMBEDDED_FORCE' bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs
grep -n 'KRAKEN_EMBEDDED_DRAG' bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs
grep -n 'KRAKEN_EMBEDDED_GEOMETRY' bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs
```

Stealth knob: `KRAKEN_SKIP_PKG` is read by the PBS script but is not part of
the case manifest; it changes package setup only.

## E. Run-by-run output schema

Add these exact `Symbol` entries to `CSV_COLUMNS` after `:bsd_fraction`:

```julia
:L_up, :L_down, :embedded_gradient, :embedded_advection,
:embedded_force, :embedded_drag, :embedded_geometry,
```

Add these row assignments when building `row`:

```julia
:L_up => L_up,
:L_down => L_down,
:embedded_gradient => Int(embedded_gradient),
:embedded_advection => Int(embedded_advection),
:embedded_force => Int(embedded_force),
:embedded_drag => Int(embedded_drag),
:embedded_geometry => string(embedded_geometry),
```

These columns are enough to reconstruct every new env-var choice for each
`SUMMARY.csv` row. The existing `R`, `Wi`, `Re_R`, `beta`, `bsd_fraction`,
`max_steps`, `avg_window`, `backend`, and `FT` columns already cover the
remaining manifest knobs.

## F. Comparison criteria

1. Runs to isolate baseline: Tier 1 must decide the promoted
   domain/embedded configuration in 12 cases or fewer.
2. Walltime to first publishable plot: Tier 1 plus Tier 2 must finish within
   about 15.5 A100 hours using the stated cost model.
3. Preflight robustness: fraction of manifest env-var plumbing errors caught
   before submission; target is 100% for the seven new knobs.

## G. Honest gaps

- This is not a statistically powered factorial design.
- It does not estimate interaction terms with ANOVA.
- It does not build a response-surface model for Cd.
- It does not optimize run placement by D-optimal or related methodology.
- It intentionally accepts confounding inside the compact embedded-mode tuple.
- It treats the A100 walltime model as first-order and does not model queue or
  I/O variability.
