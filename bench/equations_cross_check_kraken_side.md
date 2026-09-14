# Kraken Viscoelastic Kernel Audit — Equations Cross-Check

Extraction of exact formulas from source code for comparison with Liu et al. 2025 (arxiv 2508.16997).

## 1. Hermite source term

### 2D (src/kernels/collide_viscoelastic_source_2d.jl:47-83)

**Source term pre-factor:**
```julia
pre = -ω * T(9.0/2.0) / (one(T) - ω / T(2))
```

**Per-direction Hermite contributions to f_q:**
```
q=1:  T1 = pre * (4/9) * (-cs2·(txx + tyy))
q=2:  T2 = pre * (1/9) * ((1-cs2)·txx - cs2·tyy)
q=3:  T3 = pre * (1/9) * (-cs2·txx + (1-cs2)·tyy)
q=4:  T4 = T2 (same as q=2)
q=5:  T5 = T3 (same as q=3)
q=6:  T6 = pre * (1/36) * ((1-cs2)·txx + (1-cs2)·tyy + 2·txy)
q=7:  T7 = pre * (1/36) * ((1-cs2)·txx + (1-cs2)·tyy - 2·txy)
q=8:  T8 = pre * (1/36) * ((1-cs2)·txx + (1-cs2)·tyy + 2·txy)
q=9:  T9 = pre * (1/36) * ((1-cs2)·txx + (1-cs2)·tyy - 2·txy)
```

with cs2 = 1/3. Added as: `f[i,j,q]_new = f[i,j,q]_post + T_q` (src/kernels/collide_viscoelastic_source_2d.jl:85-93)

### 3D (src/kernels/viscoelastic_3d.jl:26-67)

**Source term pre-factor:**
```julia
pre = -s_plus * T(9.0/2.0) / (one(T) - s_plus / T(2))
```

**Common weight combinations (cs² = 1/3, a = 2/3):**
```julia
s_xyz = txx + tyy + tzz
diag_x = a*txx - cs2*tyy - cs2*tzz           # axial x
diag_y = -cs2*txx + a*tyy - cs2*tzz          # axial y
diag_z = -cs2*txx - cs2*tyy + a*tzz          # axial z
edge_xy_diag = a*(txx + tyy) - cs2*tzz
edge_xz_diag = a*(txx + tzz) - cs2*tyy
edge_yz_diag = a*(tyy + tzz) - cs2*txx
```

**Per-direction additions:**
```
q=1:       T1  = pre * (1/3) * (-cs2·s_xyz)
q=2,3:     T_x = pre * (1/18) * diag_x
q=4,5:     T_y = pre * (1/18) * diag_y
q=6,7:     T_z = pre * (1/18) * diag_z
q=8,11:    Txy_p = pre * (1/36) * (edge_xy_diag + 2·txy)
q=9,10:    Txy_m = pre * (1/36) * (edge_xy_diag - 2·txy)
q=12,15:   Txz_p = pre * (1/36) * (edge_xz_diag + 2·txz)
q=13,14:   Txz_m = pre * (1/36) * (edge_xz_diag - 2·txz)
q=16,19:   Tyz_p = pre * (1/36) * (edge_yz_diag + 2·tyz)
q=17,18:   Tyz_m = pre * (1/36) * (edge_yz_diag - 2·tyz)
```

Added as: `f[i,j,k,q] += T_q` (src/kernels/viscoelastic_3d.jl:58-67)

---

## 2. Equilibrium distribution for conformation tensor field

### 2D (D2Q9) (src/kernels/equilibrium_helpers.jl:8-50)

Standard D2Q9 equilibrium with `ρ → φ` (scalar conformation component):

```julia
feq_2d(Val(1), φ, u, v, usq) = (4/9)·φ·(1 - 1.5·usq)
feq_2d(Val(2), φ, u, v, usq) = (1/9)·φ·(1 + 3·u + 4.5·u² - 1.5·usq)
feq_2d(Val(3), φ, u, v, usq) = (1/9)·φ·(1 + 3·v + 4.5·v² - 1.5·usq)
feq_2d(Val(4), φ, u, v, usq) = (1/9)·φ·(1 - 3·u + 4.5·u² - 1.5·usq)
feq_2d(Val(5), φ, u, v, usq) = (1/9)·φ·(1 - 3·v + 4.5·v² - 1.5·usq)
feq_2d(Val(6), φ, u, v, usq) = (1/36)·φ·(1 + 3·(u+v) + 4.5·(u+v)² - 1.5·usq)
feq_2d(Val(7), φ, u, v, usq) = (1/36)·φ·(1 + 3·(-u+v) + 4.5·(-u+v)² - 1.5·usq)
feq_2d(Val(8), φ, u, v, usq) = (1/36)·φ·(1 + 3·(-u-v) + 4.5·(-u-v)² - 1.5·usq)
feq_2d(Val(9), φ, u, v, usq) = (1/36)·φ·(1 + 3·(u-v) + 4.5·(u-v)² - 1.5·usq)
```

Used for initialization and CNEBB reconstruction (src/kernels/conformation_lbm_2d.jl:158-166, 323-331)

### 3D (D3Q19) (src/kernels/equilibrium_helpers_3d.jl:26-100)

Standard D3Q19 equilibrium with `ρ → φ`:

```julia
feq_3d(Val(1), φ, u, v, w, usq) = (1/3)·φ·(1 - 1.5·usq)
feq_3d(Val(2), φ, u, v, w, usq) = (1/18)·φ·(1 + 3·u + 4.5·u² - 1.5·usq)
feq_3d(Val(3), φ, u, v, w, usq) = (1/18)·φ·(1 - 3·u + 4.5·u² - 1.5·usq)
feq_3d(Val(4), φ, u, v, w, usq) = (1/18)·φ·(1 + 3·v + 4.5·v² - 1.5·usq)
feq_3d(Val(5), φ, u, v, w, usq) = (1/18)·φ·(1 - 3·v + 4.5·v² - 1.5·usq)
feq_3d(Val(6), φ, u, v, w, usq) = (1/18)·φ·(1 + 3·w + 4.5·w² - 1.5·usq)
feq_3d(Val(7), φ, u, v, w, usq) = (1/18)·φ·(1 - 3·w + 4.5·w² - 1.5·usq)
[q=8–19: edge and corner velocities with appropriate (u±v), (u±w), (v±w) combinations, weight (1/36)]
```

**Lattice:** D3Q19 (19 discrete velocities)

Used for initialization and CNEBB reconstruction (src/kernels/conformation_lbm_3d.jl:228-246, 316-326)

---

## 3. CNEBB (Conservative Non-Equilibrium Bounce-Back) for walls

### 2D (src/kernels/conformation_lbm_2d.jl:286-358)

**Step 1: Conservative moment recovery**
```julia
φ = g_post[i,j,1]  # rest population (always valid)
for q in 2:9:
    if source_neighbor_is_solid:
        φ += g_pre[i,j,opposite(q)]       # recover streamed-out population
    else:
        φ += g_post[i,j,q]                # use valid post-stream
```

**Step 2: Equilibrium reconstruction at u_wall = 0**
```julia
ge_q[q] = feq_2d(Val(q), φ, 0, 0, 0)    # for all q=1..9
```

**Step 3: Unknown population reconstruction (Eq. 39)**
```julia
if source_q_neighbor_is_solid:
    g_post[i,j,q] = ge[q] + (g_post[i,j,opposite(q)] - ge[opposite(q)])
```

**Step 4: Update macroscopic**
```julia
C_field[i,j] = φ
```

Applied at: src/kernels/conformation_lbm_2d.jl:286-357

### 3D (src/kernels/conformation_lbm_3d.jl:289-341)

**Identical structure to 2D, adapted for D3Q19:**

1. Conservative φ recovery using pre/post + opposite indices
2. Equilibrium at u_wall = (0,0,0) using `feq_3d`
3. NEBB reconstruction: `g_post[i,j,k,q] = ge_q + (g_post[i,j,k,opp(q)] - ge_opp(q))`
4. Macroscopic update: `C_field[i,j,k] = φ`

**D3Q19 opposite lookup:** `_opp_q3(q) = (1, 3, 2, 5, 4, 7, 6, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16)[q]`

Applied at: src/kernels/conformation_lbm_3d.jl:289-341

---

## 4. Drag integration on LI-BB

### 2D (src/drivers/cylinder_libb.jl:27-42, 68-114)

**Standard halfway-BB MEA:**
```julia
F_x = Σ_{q=2}^{9} 2·c_qx·f_q(i,j)    where q_wall[i,j,q] > 0
F_y = Σ_{q=2}^{9} 2·c_qy·f_q(i,j)    where q_wall[i,j,q] > 0
```
(src/drivers/cylinder_libb.jl:33-40)

**Mei-Luo-Shyy 2002 with LI-BB Bouzidi:**
```julia
F_link = c_q · (f_q_pre + f_q_bouzidi)    per cut link q

where: f_q_bouzidi = {
    2·q_w·f_q + (1 - 2·q_w)·f_q_upstream + δ    (q_w ≤ 0.5)
    (1/(2q_w))·f_q + (1 - 1/(2q_w))·f_qbar + (1/(2q_w))·δ    (q_w > 0.5)
}

δ = -6·w_q·c_q·u_wall
```
(src/drivers/cylinder_libb.jl:98-108)

### 3D (src/drivers/cylinder_libb.jl and src/drivers/viscoelastic_3d.jl)

**Standard MEA (no Mei interpolation):**
```julia
F_x = Σ_{cut-link} f_q(i,j,k)     # summation over flagged D3Q19 links at solid boundary
F_y = Σ_{cut-link} f_q(i,j,k)
F_z = Σ_{cut-link} f_q(i,j,k)
```

Used in driver (src/drivers/viscoelastic_3d.jl:234)

---

## 5. Cd normalisation

### 2D (src/drivers/cylinder_libb.jl:249-253, src/drivers/viscoelastic.jl:599-604)

**Reference velocity (Schäfer-Turek convention):**
```julia
# Parabolic inlet:
u_ref = (2/3) · u_max    [src/drivers/cylinder_libb.jl:205]

# Uniform inlet:
u_ref = u_in              [src/drivers/cylinder_libb.jl:205]
```

**Cd calculation:**
```julia
D = 2 · radius
Cd = 2.0 · Fx / (u_ref² · D)     [src/drivers/cylinder_libb.jl:252]
```

For viscoelastic (with Hermite source):
```julia
Cd = 2.0 · Fx_s / (u_ref² · D)   [src/drivers/viscoelastic.jl:604]
```

where `u_ref = u_mean = (2/3)·u_in` for parabolic inlet (src/drivers/viscoelastic.jl:426)

### 3D (src/drivers/viscoelastic_3d.jl:309-313)

**Reference velocity:**
```julia
u_ref = (4/9)·u_in           (inlet=:parabolic)
u_ref = (2/3)·u_in           (inlet=:parabolic_y, default)
u_ref = u_in                 (inlet=:uniform)
```
[src/drivers/viscoelastic_3d.jl:85-87]

**Cd calculation:**
```julia
A = π · radius²              # frontal area for sphere
Cd = 2.0 · Fx / (u_ref² · A)  [src/drivers/viscoelastic_3d.jl:313]
```

---

## 6. TRT two-relaxation-time rates for conformation field

### 2D (src/kernels/conformation_lbm_2d.jl:133-144)

**TRT rate relation via magic parameter:**
```julia
tau_minus = magic / (tau_plus - 0.5) + 0.5     [line 138]
```

Default: `magic = 0.25` (from function signature, line 134)

**Inverse relaxation frequencies (ω):**
```julia
ωp = 1 / tau_plus     [line 77]
ωm = 1 / tau_minus    [line 78]
```

**Collision:** symmetric and anti-symmetric parts collided separately
```julia
post_q = g_q − ωp·(gp − ep) − ωm·(gm − em)
post_opp = g_opp − ωp·(gp − ep) + ωm·(gm − em)
```
where `gp = (g_q + g_opp)/2`, `gm = (g_q − g_opp)/2`, and similarly for equilibria
(src/kernels/conformation_lbm_2d.jl:88-117)

### 3D (src/kernels/conformation_lbm_3d.jl:206)

Identical TRT relation:
```julia
tau_minus = magic / (tau_plus - 0.5) + 0.5    [line 206]
```

Default: `magic = 0.25` (function signature, line 202)

**Collision:** 9 opposite-pair collisions for D3Q19 (src/kernels/conformation_lbm_3d.jl:136-186)
```julia
post_q = g_q − ωp·(gp − ep) − ωm·(gm − em) + w_q·S
post_opp = g_opp − ωp·(gp − ep) + ωm·(gm − em) + w_opp·S
```

---

## 7. Conformation inlet/outlet reset for 3D

### 3D Inlet (src/kernels/conformation_lbm_3d.jl:365-409)

**Imposed values at i=1 (inlet plane):**
```julia
C_xx_inlet[j,k]   # prescribed per (j,k)
C_xy_inlet[j,k]   # prescribed per (j,k)
C_xz_inlet[j,k]   # prescribed per (j,k)
C_yy_inlet[j,k]   # prescribed per (j,k)
C_yz_inlet[j,k]   # prescribed per (j,k)
C_zz_inlet[j,k]   # prescribed per (j,k)
u_profile[j,k]    # prescribed per (j,k)
v, w = 0          # transverse velocity zero
```

**Action:** Reset each component g_[:,:,q] at (i=1, j, k) to equilibrium:
```julia
g[1,j,k,q] = feq_3d(Val(q), C_αβ_inlet[j,k], u_profile[j,k], 0, 0, usq)
```
[src/kernels/conformation_lbm_3d.jl:374-392]

**Analytical values from driver (src/drivers/viscoelastic_3d.jl:126-136):**
```julia
C_xx_inlet[j,k] = 1 + 2·(λ·∂u/∂y)²
C_xy_inlet[j,k] = λ·∂u/∂y
C_xz_inlet[j,k] = 0
C_yy_inlet[j,k] = 1
C_yz_inlet[j,k] = 0
C_zz_inlet[j,k] = 1
u_profile[j,k] = {
    (4/9)·u_in·y·(Hy−y)·z·(Hz−z)/(Hy²·Hz²)       (parabolic, fully 3D)
    (4/9)·u_in·y·(Hy−y)/Hy²                      (parabolic_y, y-only)
    u_in                                          (uniform)
}
```
where ∂u/∂y computed from the velocity profile derivative.

### 3D Outlet (src/kernels/conformation_lbm_3d.jl:411-431)

**Zero-gradient extrapolation at i=Nx:**
```julia
g[Nx, j, k, q] = g[Nx-1, j, k, q]    for all q=1..19
```
[src/kernels/conformation_lbm_3d.jl:415]

