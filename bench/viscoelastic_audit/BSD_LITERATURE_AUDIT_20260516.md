# BSD literature audit — rheoTool & Liu 2025 vs Kraken (2026-05-16)

This audit compares the discrete BSD/stabilisation coupling used by rheoTool,
the local stress-coupling strategy in Liu et al. (2025), and Kraken's current
`:fd` BSD path.
The focus is the low-Weissenberg Newtonian-additive limit identified in the
M10 audit, where operator identity matters as much as continuum algebra.

## rheoTool BSD discrete scheme

The rheoTool implementation point is
`/Users/guillaume/Desktop/rheoTool/of90/src/libs/constitutiveEquations/constitutiveEqs/constitutiveEq/constitutiveEq.C:72-132`.
The relevant dispatch is the `constitutiveEq::divTau(const volVectorField& U)`
switch over `stabOption_` at
`/Users/guillaume/Desktop/rheoTool/of90/src/libs/constitutiveEquations/constitutiveEqs/constitutiveEq/constitutiveEq.C:88-127`.

The three modes are implemented as follows in the uncoupled branch
(`solveCoupled_ == false`):

- `soNone`: explicit `fvc::div(tau()/rho(), "div(tau)")` plus implicit
  `fvm::laplacian(etaS()/rho(), U, "laplacian(eta,U)")`, from
  `/Users/guillaume/Desktop/rheoTool/of90/src/libs/constitutiveEquations/constitutiveEqs/constitutiveEq/constitutiveEq.C:90-99`.
- `soBSD`: explicit `fvc::div(tau()/rho(), "div(tau)")`, explicit subtraction
  `- fvc::laplacian(etaP()/rho(), U, "laplacian(eta,U)")`, and implicit
  `fvm::laplacian((etaP()+etaS())/rho(), U, "laplacian(eta,U)")`, from
  `/Users/guillaume/Desktop/rheoTool/of90/src/libs/constitutiveEquations/constitutiveEqs/constitutiveEq/constitutiveEq.C:101-112`.
- `soCoupling`: explicit `fvc::div(tau()/rho(), "div(tau)")`, explicit
  subtraction `- fvc::div((etaP()/rho())*fvc::grad(U), "div(grad(U))")`, and
  implicit `fvm::laplacian((etaP()+etaS())/rho(), U, "laplacian(eta,U)")`,
  from `/Users/guillaume/Desktop/rheoTool/of90/src/libs/constitutiveEquations/constitutiveEqs/constitutiveEq/constitutiveEq.C:114-126`.

The exact C++ lines for the three stabilisation modes are:

```cpp
90      case soNone : // none
91      return
92      (
93        solveCoupled_
94        ?
95           fvm::laplacian(etaS()/rho(), U, "laplacian(eta,U)")
96        :
97           fvc::div(tau()/rho(), "div(tau)")
98         + fvm::laplacian(etaS()/rho(), U, "laplacian(eta,U)")
99      );
100
101     case soBSD : // BSD
102     return
103     (
104       solveCoupled_
105       ?
106        - fvc::laplacian(etaP()/rho(), U, "laplacian(eta,U)")
107        + fvm::laplacian( (etaP()+ etaS())/rho(), U, "laplacian(eta,U)")
108       :
109          fvc::div(tau()/rho(), "div(tau)")
110        - fvc::laplacian(etaP()/rho(), U, "laplacian(eta,U)")
111        + fvm::laplacian( (etaP()+ etaS())/rho(), U, "laplacian(eta,U)")
112     );
113
114     case soCoupling : // coupling
115     return
116     (
117       solveCoupled_
118       ?
119        - fvc::div((etaP()/rho())*fvc::grad(U),"div(grad(U))")
120        + fvm::laplacian( (etaP() + etaS())/rho(), U, "laplacian(eta,U)")
121
122       :
123          fvc::div(tau()/rho(), "div(tau)")
124        - fvc::div((etaP()/rho())*fvc::grad(U),"div(grad(U))")
125        + fvm::laplacian( (etaP() + etaS())/rho(), U, "laplacian(eta,U)")
126     );
127
```

The local cavity case chooses the improved BSD/coupling branch:
`stabilization coupling` is set in
`bench/rheotool/cavity_oldroydb_log_re001_de1_b05/constant/constitutiveProperties:16-27`.
That means this case uses the `soCoupling` branch above, not classical
`soBSD`.

The same-stencil enforcement is visible at the OpenFOAM dictionary level.
The cavity `fvSchemes` file declares `div(tau) Gauss linear` and
`div(grad(U)) Gauss linear` in adjacent entries at
`bench/rheotool/cavity_oldroydb_log_re001_de1_b05/system/fvSchemes:30-35`.
The same file also sets `grad(U) Gauss linear` at
`bench/rheotool/cavity_oldroydb_log_re001_de1_b05/system/fvSchemes:21-28`.

Therefore rheoTool's improved BSD subtraction uses `fvc::div` of a tensor
for both the polymer force and the promoted-polymer-viscosity subtraction:
`fvc::div(tau()/rho(), "div(tau)")` at
`/Users/guillaume/Desktop/rheoTool/of90/src/libs/constitutiveEquations/constitutiveEqs/constitutiveEq/constitutiveEq.C:123`
and `fvc::div((etaP()/rho())*fvc::grad(U), "div(grad(U))")` at
`/Users/guillaume/Desktop/rheoTool/of90/src/libs/constitutiveEquations/constitutiveEqs/constitutiveEq/constitutiveEq.C:124`.
This is the discrete point that differs from Kraken's current narrow-laplacian
`:fd` subtraction, whose kernel subtracts `zeta_nu_p * lap_ux` and
`zeta_nu_p * lap_uy` at `src/fvfd/operators_2d.jl:886-911`.

## rheoTool pipeline order

The segregated rheoFoam branch is the `else` block at
`/Users/guillaume/Desktop/rheoTool/of90/src/solvers/rheoFoam/rheoFoam.C:147-154`.
Within each inner iteration, it includes `UEqn.H`, then `pEqn.H`, then calls
`constEq.correct()`, exactly in that order at
`/Users/guillaume/Desktop/rheoTool/of90/src/solvers/rheoFoam/rheoFoam.C:149-153`.

The per-timestep and per-inner-iteration order is:

1. Mesh and MRF update are handled before the equation solves through
   `fvModels.preUpdateMesh()`, `mesh.update()`, and `MRF.update()` at
   `/Users/guillaume/Desktop/rheoTool/of90/src/solvers/rheoFoam/rheoFoam.C:101-123`.
2. `UEqn` is assembled by `UEqn.H`; the right-hand side includes
   `constEq.divTau(U)` at
   `/Users/guillaume/Desktop/rheoTool/of90/src/solvers/rheoFoam/UEqn.H:5-13`.
3. The momentum equation is solved by
   `spSolverU->solve(UEqn == -fvc::grad(p))` at
   `/Users/guillaume/Desktop/rheoTool/of90/src/solvers/rheoFoam/UEqn.H:17-23`.
4. The pressure-correction equation is included immediately after `UEqn.H` at
   `/Users/guillaume/Desktop/rheoTool/of90/src/solvers/rheoFoam/rheoFoam.C:149-150`.
5. `constEq.correct()` is called after `pEqn.H` at
   `/Users/guillaume/Desktop/rheoTool/of90/src/solvers/rheoFoam/rheoFoam.C:152-153`.

For Oldroyd-B log, `constEq.correct()` solves the log-conformation equation in
`theta_` through an `fvSymmTensorMatrix` at
`/Users/guillaume/Desktop/rheoTool/of90/src/libs/constitutiveEquations/constitutiveEqs/Oldroyd-B/Oldroyd-BLog/Oldroyd_BLog.C:140-167`.
It then diagonalises `theta_` at
`/Users/guillaume/Desktop/rheoTool/of90/src/libs/constitutiveEquations/constitutiveEqs/Oldroyd-B/Oldroyd-BLog/Oldroyd_BLog.C:169-171`
and reconstructs `tau_` from the eigen representation at
`/Users/guillaume/Desktop/rheoTool/of90/src/libs/constitutiveEquations/constitutiveEqs/Oldroyd-B/Oldroyd-BLog/Oldroyd_BLog.C:173-178`.

The `tau` consumed by `constEq.divTau(U)` in `UEqn.H` is therefore the stress
already present before the current momentum and pressure solves. This follows
from `constEq.divTau(U)` being assembled at
`/Users/guillaume/Desktop/rheoTool/of90/src/solvers/rheoFoam/UEqn.H:5-13`,
while the only visible Oldroyd-B log stress update in the segregated branch,
`constEq.correct()`, happens later at
`/Users/guillaume/Desktop/rheoTool/of90/src/solvers/rheoFoam/rheoFoam.C:149-153`.

The BSD gradient capture is not a separately stored field in this source path.
It is evaluated inside `constitutiveEq::divTau(U)` as
`fvc::div((etaP()/rho())*fvc::grad(U), "div(grad(U))")` at
`/Users/guillaume/Desktop/rheoTool/of90/src/libs/constitutiveEquations/constitutiveEqs/constitutiveEq/constitutiveEq.C:119-125`.
Because `UEqn.H` passes the current pre-solve `U` into `constEq.divTau(U)` at
`/Users/guillaume/Desktop/rheoTool/of90/src/solvers/rheoFoam/UEqn.H:5-13`,
the `grad(U)` used by this BSD term is the SIMPLE-iteration velocity field
available during assembly, not the freshly solved velocity from
`/Users/guillaume/Desktop/rheoTool/of90/src/solvers/rheoFoam/UEqn.H:21`.

## Liu 2025 BSD scheme

Liu et al. do not introduce a BSD subtraction term in the hydrodynamic model.
They describe their hydrodynamic TRT-RLB extension as a local discretization
that directly integrates the stress tensor without spatial derivatives at
`bench/viscoelastic_audit/liu_2025.txt:343-350`.

They explicitly contrast this with traditional LBM viscoelastic coupling:
traditional methods compute `Fp,α = ∂β ταβ` and insert it through forcing
schemes, while their approach incorporates `ταβ` without spatial derivatives
at `bench/viscoelastic_audit/liu_2025.txt:351-354`.

Their mesoscopic hydrodynamic evolution equation includes three source pieces,
`Gi Δt`, `Fi Δt`, and `Ti Δt`, in Eq. 12 at
`bench/viscoelastic_audit/liu_2025.txt:355-377`.
The viscoelastic source is Eq. 22:

```text
T_i = - w_i H_{i,αβ} τ_{αβ} / (2 c_s^4 τ_{s,1} Δt)
```

This formula is printed at `bench/viscoelastic_audit/liu_2025.txt:558-564`.
Because the source is proportional to the second-order Hermite tensor
`H_{i,αβ}` and the stress tensor `τ_{αβ}`, it is a Hermite-moment injection
onto the second-order moment of the hydrodynamic distribution. This is the
mechanism Liu uses instead of computing `∂β ταβ`.

The force term `Fi` remains a separate first-order Hermite expansion in Eq. 23,
shown at `bench/viscoelastic_audit/liu_2025.txt:565-577`.
The cubic-defect correction `Gi` is separately defined in Eq. 19 and Eq. 20/21
at `bench/viscoelastic_audit/liu_2025.txt:533-555`.
Thus the stress term `Ti`, the external/body-force term `Fi`, and the
cubic-defect correction `Gi` are distinct terms in Liu's hydrodynamic update.

The macroscopic density and velocity recovery use moment relations in Eq. 14:
the extracted text shows `sum_i f_i = ρ` and
`sum_i e_{iα} f_i = ρu_α - Δt F_α / 2` at
`bench/viscoelastic_audit/liu_2025.txt:399-415`.
Equivalently, the velocity uses a `+ Δt F / 2` half-step force correction when
solved for `u`. This matches the half-force convention noted for Kraken's Guo
brick in the M10 audit at
`bench/viscoelastic_audit/BSD_GUO_WI0_AUDIT_20260516.md:129-137`.

On the conformation side, Liu solves a CDE for the independent components of
the conformation tensor, with Eq. 24 and source Eq. 25 shown at
`bench/viscoelastic_audit/liu_2025.txt:580-602`.
The improved regularized LB evolution for `g_i` is Eq. 26 at
`bench/viscoelastic_audit/liu_2025.txt:604-639`.
The off-equilibrium moments used by the CDE regularization are defined in
Eqs. 28-30 at `bench/viscoelastic_audit/liu_2025.txt:652-692`.

The CDE-side coupling is the auxiliary term `G̃_i`.
Liu state that this term avoids velocity-gradient calculations in favor of
nearly negligible density gradients at
`bench/viscoelastic_audit/liu_2025.txt:696-702`.
The auxiliary term is defined in Eq. 31 at
`bench/viscoelastic_audit/liu_2025.txt:703-722`, and the incompressible
reduction Eq. 32 couples it to the NSE force `F_α` at
`bench/viscoelastic_audit/liu_2025.txt:723-735`.

The CDE forcing term `F̃_i` is Eq. 33 at
`bench/viscoelastic_audit/liu_2025.txt:736-746`, and the transport coefficient
relation `κ = (τ_{p,1} - 1/2) Δt c_s^2` is Eq. 34 at
`bench/viscoelastic_audit/liu_2025.txt:747-759`.
Liu's super-convergence discussion attributes accuracy to the Eq. 32
hydrodynamic force coupling and to direct stress incorporation without spatial
discretization in Eq. 22 at `bench/viscoelastic_audit/liu_2025.txt:1162-1174`.
Their conclusion repeats that direct stress-tensor incorporation removes the
need for stress-gradient calculations and that Eq. 32 tightens NSE-CDE
coupling at `bench/viscoelastic_audit/liu_2025.txt:2893-2905`.

There is therefore no narrow-versus-wide Laplacian dichotomy in Liu's method:
the stress is not converted to a post-hoc body force by differentiating `τ`,
and no BSD Laplacian is subtracted afterward. This conclusion is anchored to
the paper's explicit rejection of `∂β ταβ` forcing at
`bench/viscoelastic_audit/liu_2025.txt:351-354` and the direct Eq. 22
stress source at `bench/viscoelastic_audit/liu_2025.txt:558-564`.

## Wi → 0 treatment

For rheoTool, the Oldroyd-B log update contains a relaxation term with
`(1.0/lambda) * innerP(eigVecs_, (inv(eigVals_) - Itensor), false)` at
`/Users/guillaume/Desktop/rheoTool/of90/src/libs/constitutiveEquations/constitutiveEqs/Oldroyd-B/Oldroyd-BLog/Oldroyd_BLog.C:146-163`.
The same `correct()` method reconstructs
`tau_ = (etaP/lambda) * symm(innerP(eigVecs_, eigVals_, false) - Itensor)` at
`/Users/guillaume/Desktop/rheoTool/of90/src/libs/constitutiveEquations/constitutiveEqs/Oldroyd-B/Oldroyd-BLog/Oldroyd_BLog.C:173-178`.
In the low-Wi limit this is the usual Oldroyd-B relaxation toward
`tau ≈ 2 etaP D`.

Under that limit, the rheoTool `soCoupling` branch becomes:

```text
fvc::div(2 etaP D)
- fvc::div(etaP grad(U))
+ fvm::laplacian((etaP + etaS), U)
```

The explicit stress-divergence term and explicit BSD subtraction are both
`fvc::div` tensor divergences in the `soCoupling` branch at
`/Users/guillaume/Desktop/rheoTool/of90/src/libs/constitutiveEquations/constitutiveEqs/constitutiveEq/constitutiveEq.C:123-125`.
The cavity dictionary assigns both named divergence schemes to `Gauss linear`
at `bench/rheotool/cavity_oldroydb_log_re001_de1_b05/system/fvSchemes:30-35`.
For incompressible flow and constant `etaP`,
`div(2 etaP D) = div(etaP (grad U + grad U^T)) = div(etaP grad U)` after the
`grad(div U)` term vanishes.
Because rheoTool uses the same `fvc::div` operator name family and the same
`Gauss linear` dictionary stencil for both pieces, the `soCoupling`
low-Wi cancellation is discretely exact at the operator level for this case.

The M10 Kraken audit states the same continuum identity, then shows where
Kraken breaks it discretely: `div(2 * nu_p * D) = nu_p * nabla^2 u` is exact
in continuum, but Kraken's discrete operators are not the same at
`bench/viscoelastic_audit/BSD_GUO_WI0_AUDIT_20260516.md:161-166`.
M10 derives that Kraken's `div(tau)` path produces a wide spacing-`2dx`
second difference at `bench/viscoelastic_audit/BSD_GUO_WI0_AUDIT_20260516.md:175-197`.
It then shows that the current FD-BSD branch subtracts a narrow operator at
`bench/viscoelastic_audit/BSD_GUO_WI0_AUDIT_20260516.md:199-217`.
The M10 conclusion is that the polymer Newtonian contribution is applied
through the wide stencil rather than the same narrow operator used by the LBM
viscosity, leaving an `O(nu_p * dx^2)` broken cancellation at
`bench/viscoelastic_audit/BSD_GUO_WI0_AUDIT_20260516.md:219-229`.

For Liu, no explicit Wi→0 BSD cancellation is discussed in the paper text
reviewed here.
The low-Wi behavior note says prior studies included `Wi = 0.3` and `Wi = 0.6`,
but those low-Weissenberg cases closely resemble Newtonian behavior and pose
minimal methodic challenges, so only `Wi = 5` is presented at
`bench/viscoelastic_audit/liu_2025.txt:1183-1186`.

The relevant numerical consequence is structural: Liu's stress enters as the
local Hermite source `T_i` at `bench/viscoelastic_audit/liu_2025.txt:558-564`,
and Liu explicitly avoids computing `Fp,α = ∂β ταβ` at
`bench/viscoelastic_audit/liu_2025.txt:351-354`.
Therefore there is no low-Wi operator mismatch between an explicit
stress-divergence stencil and a BSD Laplacian subtraction. The Newtonian limit
emerges without requiring a BSD-style same-stencil cancellation.

## Comparison to Kraken

| Aspect | rheoTool soCoupling | Liu 2025 | Kraken (bsd_kind=:fd) | Consistent at Wi→0? |
|---|---|---|---|---|
| Stress incorporation | explicit `fvc::div(τ/ρ)`, FV face flux (`constitutiveEq.C:123`) | Hermite-moment source `T_i` (local, no derivative; `liu_2025.txt:558-564`) | explicit FD central `div(τ)` through `fvfd_tensor_divergence_2d!` (`src/fvfd/operators_2d.jl:633-659`) | rheoTool ✓ / Liu ✓ / Kraken ✗ |
| BSD subtraction | `-fvc::div(etaP*grad(U)/ρ)` with same FV `div` family as `div(τ)` (`constitutiveEq.C:123-125`) | none; direct stress injection makes it unnecessary (`liu_2025.txt:351-354`) | `-ζ*ν_p*∇²u` via narrow second-derivative helpers (`src/fvfd/operators_2d.jl:886-911`) | rheoTool ✓ / Kraken ✗ |
| Implicit LBM/solver viscosity | `(etaP+etaS)/rho` in `fvm::laplacian` (`constitutiveEq.C:119-125`) | solvent relaxation through `τ_{s,1}` in TRT-RLB Eq. 12 and Eq. 22 (`liu_2025.txt:355-377`, `:558-564`) | `ν_s + ζ*ν_p` passed to fused TRT step after BSD correction (`src/drivers/viscoelastic_logfv_2d.jl:1145-1149`; M10 `:139-148`) | n/a |
| `grad(U)` source for BSD | SIMPLE pre-solve `U` passed into `constEq.divTau(U)` (`UEqn.H:5-13`; `constitutiveEq.C:119-125`) | n/a | current one-step `ux, uy` passed into `logfv_bsd_correct_force_bc_aware_2d!` (`src/drivers/viscoelastic_logfv_2d.jl:1119-1122`) | minor |
| Same-stencil enforcement | yes: `div(tau)` and `div(grad(U))` are both `Gauss linear` (`fvSchemes:30-35`) | n/a: no explicit stress divergence (`liu_2025.txt:351-354`) | NO: `Lap_wide` from `div(2νD)` differs from `Lap_narrow` from `∇²u` (M10 `:175-229`) | rheoTool ✓ / Kraken ✗ |
| Wi→0 Newtonian-additive limit | discretely exact cancellation in `soCoupling` for constant `etaP` and incompressible flow (`constitutiveEq.C:123-125`; `fvSchemes:30-35`) | trivially exact with respect to BSD because Eq. 22 is local (`liu_2025.txt:558-564`) | breaks at `O(ν_p*dx²*∂⁴u)` by M10's wide/narrow mismatch (`BSD_GUO_WI0_AUDIT_20260516.md:219-229`) | rheoTool ✓ / Liu ✓ / Kraken ✗ |

Kraken's current sequence is verified in
`src/drivers/viscoelastic_logfv_2d.jl:1098-1148`: stress is reconstructed from
log-conformation at `:1098-1102`, `F_poly = div(tau)` is computed at
`:1104-1108`, the BSD branch is selected at `:1110-1123`, and the fused
TRT+LI-BB+Guo field step consumes `fx_total, fy_total` at `:1145-1149`.
The current `:fd` branch calls `logfv_bsd_correct_force_bc_aware_2d!` at
`src/drivers/viscoelastic_logfv_2d.jl:1119-1122`.
That wrapper delegates to `fvfd_bsd_force_2d!` at
`src/kernels/logconformation_fv_2d.jl:710-718`, whose kernel subtracts
`zeta_nu_p * lap_ux` and `zeta_nu_p * lap_uy` at
`src/fvfd/operators_2d.jl:886-911`.

## Recommendation

Primary recommendation: adopt rheoTool's `soCoupling` pattern in Kraken's
`:fd` path. Replace the current narrow-laplacian body-force correction at
`src/drivers/viscoelastic_logfv_2d.jl:1119-1122` with a second
`fvfd_tensor_divergence_2d!` call on a BSD stress tensor.

The existing helper already builds the required tensor:
`logfv_bsd_stress_from_gradient_2d!` writes
`tau_bsd_xx = 2*zeta_nu_p*dudx`,
`tau_bsd_xy = zeta_nu_p*(dudy+dvdx)`, and
`tau_bsd_yy = 2*zeta_nu_p*dvdy` at
`src/kernels/logconformation_fv_2d.jl:678-708`.
M10 already describes the intended patch shape: build
`tau_BSD = 2*zeta*nu_p*D`, pass it through the same
`fvfd_tensor_divergence_2d!` operator used by `F_poly`, then subtract that
same-stencil force from `F_poly` at
`bench/viscoelastic_audit/BSD_GUO_WI0_AUDIT_20260516.md:271-300`.

The brief identifies a later M11 attempt of this path as destabilising. That
destabilisation should be treated as a separate stability problem, not as
evidence against the operator identity. Existing cavity notes already keep the
lid-corner region live as a risk: one report mentions a forbidden
`ζ = 1.0` extrapolation because of a lid-corner crash at
`bench/viscoelastic_logfv/CAVITY_BSD_M4B_VERDICT_20260516.md:51-57`, and
another leaves the M2 corner-kernel artifact inconclusive at
`bench/viscoelastic_logfv/CAVITY_M6B_CONFIRM_VERDICT_20260516.md:68-70`.
The literature evidence in this audit supports the rheoTool same-stencil
operator cancellation; the stability fix should focus on wall and lid-corner
coupling rather than reverting to a narrow BSD Laplacian.

Secondary recommendation: keep `bsd_kind=:kinetic` as the closest existing
Kraken analogue to Liu's locality philosophy, but do not overclaim it as Liu's
scheme. The kinetic path extracts `Pi^{neq}` from D2Q9 populations at
`src/kernels/bsd_kinetic.jl:3-12` and assembles a BSD force from the
non-equilibrium moment at `src/kernels/bsd_kinetic.jl:54-82`.
The public driver selects that path when `bsd_kind === :kinetic` at
`src/drivers/viscoelastic_logfv_2d.jl:1110-1117`.

Liu goes further than Kraken's kinetic BSD path: Liu does not compute a BSD
correction and does not compute `div(tau)` for hydrodynamic coupling. The
stress enters directly through the local Hermite source Eq. 22 at
`bench/viscoelastic_audit/liu_2025.txt:558-564`, after the paper explicitly
rejects the traditional `∂βτ_αβ` force route at
`bench/viscoelastic_audit/liu_2025.txt:351-354`.
A future Kraken `bsd_kind=:hermite` variant that mirrors Liu's `T_i` is a
viable longer-term refactor, but it requires LBM moment plumbing beyond this
M12 documentation scope.
