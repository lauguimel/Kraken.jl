# Departure-Aware Bounce-Back for SLBM — Pedagogical Guide

## 0. LBM fundamentals: populations, velocities, equilibrium

### What is a "population" f_q?

In LBM, each grid node carries **9 scalar values** `f_1, f_2, ..., f_9`. Each value
represents a density of fictitious particles moving in a specific direction. Together
they encode the macroscopic fluid state (density ρ, velocity u).

The index `q` (from 1 to 9) labels the **direction**. Each direction has:

- A **lattice velocity** `c_q = (cx_q, cy_q)` — a vecteur unitaire discret qui donne
  la direction et la norme du déplacement des particules fictives à chaque pas de temps.
  `cx_q` est la composante en x, `cy_q` en y. Les valeurs possibles sont -1, 0, +1.
  Exemple: pour E (q=2), `c_2 = (+1, 0)` → les particules "E" se déplacent d'un nœud
  vers la droite à chaque pas de temps.

- Un **poids** `w_q` — la fraction de densité portée par cette direction à l'équilibre
  quand le fluide est au repos (u=0). C'est une constante du schéma D2Q9, pas une
  variable. Les poids ne sont PAS égaux car les directions ne sont pas géométriquement
  équivalentes : les diagonales parcourent √2 nœuds vs 1 pour les axes. Les poids
  sont l'unique solution du système qui rend la physique **isotrope** (pas de direction
  privilégiée) : `Σ w_q·cx²= Σ w_q·cy² = 1/3`, `Σ w_q·cx²·cy² = 1/9`, etc.
  Résultat : repos=4/9, axes=1/9, diagonales=1/36.

- Un **opposé** `opp(q)` — la direction inverse. `c_opp(q) = -c_q`. Exemple: l'opposé
  de E (+1,0) est W (-1,0).

### D2Q9 lattice: the 9 directions

```
     NW(7)   N(3)   NE(6)         q=7: (-1,+1)  q=3: (0,+1)  q=6: (+1,+1)
       ↖      ↑      ↗                    ↖          ↑          ↗
  W(4) ←  rest(1) → E(2)         q=4: (-1, 0)  q=1: (0, 0)  q=2: (+1, 0)
       ↙      ↓      ↘                    ↙          ↓          ↘
     SW(8)   S(5)   SE(9)         q=8: (-1,-1)  q=5: (0,-1)  q=9: (+1,-1)
```

Complete table:

| q | Name | cx_q | cy_q | w_q  | opp(q) |
|---|------|------|------|------|--------|
| 1 | rest | 0    | 0    | 4/9  | 1      |
| 2 | E    | +1   | 0    | 1/9  | 4 (W)  |
| 3 | N    | 0    | +1   | 1/9  | 5 (S)  |
| 4 | W    | -1   | 0    | 1/9  | 2 (E)  |
| 5 | S    | 0    | -1   | 1/9  | 3 (N)  |
| 6 | NE   | +1   | +1   | 1/36 | 8 (SW) |
| 7 | NW   | -1   | +1   | 1/36 | 9 (SE) |
| 8 | SW   | -1   | -1   | 1/36 | 6 (NE) |
| 9 | SE   | +1   | -1   | 1/36 | 7 (NW) |

Key properties:
- `(cx_q, cy_q)` tells you where direction q "points to"
- `w_q = w_opp(q)` (weights are symmetric: E and W have the same weight)
- `cx_q = -cx_opp(q)` and `cy_q = -cy_opp(q)` (opposite directions cancel)

### From populations to macroscopic quantities

The density and velocity at a node come from summing the populations:

```
ρ  = f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 + f_9

ρ·ux = f_2 - f_4 + f_6 - f_7 - f_8 + f_9    (sum of cx_q · f_q)

ρ·uy = f_3 - f_5 + f_6 + f_7 - f_8 - f_9    (sum of cy_q · f_q)
```

### The equilibrium distribution

At "rest" (no gradients, no forces), the populations take their equilibrium values:

```
feq_q(ρ, ux, uy) = w_q · ρ · [1 + 3·(cx_q·ux + cy_q·uy)
                                 + 4.5·(cx_q·ux + cy_q·uy)²
                                 - 1.5·(ux² + uy²)]
```

For `u = 0`: `feq_q = w_q · ρ`. All the velocity dependence vanishes.
For small `u`: the linear term `3·(c_q · u)` dominates. The quadratic terms are O(u²).

### The LBM cycle: stream → collide → (boundary conditions)

Each time step:
1. **Stream**: each population "moves" by one lattice spacing in its direction.
   `f_q` at node `(i,j)` came from node `(i - cx_q, j - cy_q)` at the previous step.
   The point `(i - cx_q, j - cy_q)` is the **departure point**.
2. **Collide**: relax populations toward equilibrium.
   `f_q ← f_q - ω·(f_q - feq_q)`  where ω ∈ (0, 2) controls viscosity.
3. **Boundary conditions**: enforce walls, inlets, etc.

---

## 1. Bounce-back on a Cartesian mesh

### Setup: north wall

```
j_n+1  (ghost)    ·  ·  ·  ·  ·      ← ghost row (fictitious)
       - - - - - - - - - - - - -      ← physical wall at j_n + 0.5
j_n    (boundary)  ·  ·  ·  ·  ·      ← last physical row
j_n-1  (interior)  ·  ·  ·  ·  ·
```

### Which populations need BB at the north wall?

In standard LBM, the streaming step moves population q from its departure to the node.
The departure of q at `(i, j_n)` is:

```
departure(q) = (i - cx_q, j_n - cy_q)
```

For the **southward** populations (S, SW, SE), the departures are:
- S (cy=-1): departure at `j_n + 1` → **in the ghost** → needs BB
- SW (cx=-1, cy=-1): departure at `j_n + 1` → ghost → BB
- SE (cx=+1, cy=-1): departure at `j_n + 1` → ghost → BB

The BB set on a Cartesian north wall is always **{S, SW, SE}**.

BB replaces these with their opposite:
```
f_S  = f_N    (the northward pop at the same node)
f_SW = f_NE
f_SE = f_NW
```

For a **stationary wall** (u_wall = 0): at steady state, `u = 0` everywhere, so
`f_q = feq_q(1, 0, 0) = w_q` at every node. Since `w_q = w_opp(q)`:
```
f_S = f_N = 1/9,  f_SW = f_NE = 1/36,  f_SE = f_NW = 1/36
```
BB does nothing — it replaces a value with the same value. Trivially exact. ✓

For a **moving wall** at velocity `u_w` in x (Ladd 1994 correction):
```
f_q = f_opp(q) - 2·w_opp(q)·ρ·(cx_opp(q)·u_w)·3
```
The second term is the **momentum correction**: it adds the wall's momentum to the
reflected population. Without it, the wall would be stationary.

On Cartesian at `θ=0°`, this gives **1e-16** (machine precision) for Couette flow.

### Why does the BB set matter for moving walls?

For a stationary wall, f = w_q everywhere → reflection gives the same value regardless
of which set we choose. For a moving wall, f varies spatially → reflecting the WRONG
population gives the WRONG value. The BB set must be correct.

---

## 2. SLBM: streaming by interpolation

### Why SLBM?

On a Cartesian mesh, the standard LBM streaming is trivial: population q at `(i,j)`
came from `(i - cx_q, j - cy_q)`, which is always a grid node. Just copy the value.

On a **curvilinear** or **oblique** mesh, the physical lattice velocities `(cx_q, cy_q)`
don't map to integer displacements in computational space. The departure point is at a
**fractional** position → we need **interpolation**.

### Departure points on an oblique mesh

Consider a parallelogram mesh with skew angle θ:
```
Physical coordinates:  X[i,j] = i-1,  Y[i,j] = (j-1) + (i-1)·tan(θ)
```

The physical velocity `(cx_q, cy_q)` maps to computational displacements via the
inverse Jacobian of the coordinate transform:
```
Δξ_q = cx_q                          (ξ ~ x, no distortion)
Δη_q = cy_q - cx_q · tan(θ)          (η ~ y, but skewed by the mesh)
```

The departure point in computational space is:
```
i_dep = i - Δξ_q
j_dep = j - Δη_q
```

The key: **Δη_q depends on both cy AND cx** through the mesh angle θ.
On a Cartesian mesh (θ=0): `Δη_q = cy_q` (standard integer displacement).
On an oblique mesh: Δη_q is generally NOT an integer.

### The SLBM streaming step

Instead of copying from a grid neighbor, we **interpolate** the field at the
(fractional) departure point:

```
f_streamed_q(i, j) = interpolate( f_previous[..., q], at (i_dep, j_dep) )
```

Bilinear interpolation uses 4 surrounding nodes (2×2 stencil).
Biquadratic uses 9 surrounding nodes (3×3 stencil) → higher accuracy.

### Example: E population at θ=20°

E has `(cx, cy) = (+1, 0)`. On a 20° mesh:
```
Δξ_E = 1
Δη_E = 0 - 1·tan(20°) = -0.364
```
So the departure of E at `(i, j)` is at `(i-1, j+0.364)` — a fractional η position.
The SLBM interpolates f_E at this point using surrounding grid values.

The crucial term is `Δη_q`: the **contravariant** η-velocity of population q.

### Example at θ = 30°

| Pop | cx | cy | Δη = cy - cx·tan(30°) | j_dep at j_n | In ghost? |
|-----|----|----|----------------------|--------------|-----------|
| E   | +1 |  0 | 0 - 0.577 = **-0.577** | j_n + 0.577 | YES (>j_n+0.5) |
| N   |  0 | +1 | 1 - 0 = +1.000 | j_n - 1 | no |
| NE  | +1 | +1 | 1 - 0.577 = +0.423 | j_n - 0.423 | no |
| S   |  0 | -1 | -1 - 0 = **-1.000** | j_n + 1 | YES |
| SW  | -1 | -1 | -1 + 0.577 = **-0.423** | j_n + 0.423 | NO (< j_n+0.5) |
| SE  | +1 | -1 | -1 - 0.577 = **-1.577** | j_n + 1.577 | YES |
| W   | -1 |  0 | 0 + 0.577 = +0.577 | j_n - 0.577 | no |
| NW  | -1 | +1 | 1 + 0.577 = +1.577 | j_n - 1.577 | no |

**Standard BB set at 30°:** {S, SW, SE} (always the same) ← WRONG!
- SW: departure at j_n + 0.423 → in the fluid, NOT in the ghost
- E: departure at j_n + 0.577 → in the ghost, but NOT reflected!

**DA-BB set at 30°:** {E, S, SE} (based on actual departures) ← CORRECT!

### The DA-BB rule

Reflect population q at the north wall if and only if:
```
j_dep[i, j_n, q] > j_n + 0.5     (departure crosses into ghost)
```

This is the threshold at the wall midpoint (half-way between boundary row and ghost row).

---

## 3. Why DA-BB gives machine precision for stationary walls

For a stationary wall (u = 0), the steady-state solution is:
```
f_q(i, j) = feq_q(ρ=1, ux=0, uy=0) = w_q   for all q, i, j
```

Every population equals its weight constant. Therefore:
- `f_q = w_q` at every node
- `f_opp(q) = w_opp(q) = w_q` (D2Q9 weights are symmetric)
- BB: `f_q ← f_opp(q) = w_q` → unchanged

**It doesn't matter which populations we reflect**, because the reflection doesn't change anything.
The DA-BB set could be {S, SW, SE} or {E, S, SE} or anything — the result is the same.

**The machine precision for stationary walls is NOT a property of DA-BB.** It's a property
of the trivial solution u=0. Both standard BB and DA-BB give machine precision.

So where did the "10⁻⁵ error" on oblique meshes come from in earlier tests? It was from
the **ghost fill** or **interpolation setup**, not from the BB set selection. The DA-BB
session 4 investigation ALSO fixed ng=2 and linear ghost extrapolation as prerequisites.

---

## 4. The moving wall problem — SOLVED

### The trap: wrong wall velocity direction

On a parallelogram mesh with `Y[i,j] = (j-1) + (i-1)·tan(θ)`, the walls at
`j = const` are NOT horizontal — they are **tilted** at angle θ:

```
θ = 20° mesh:

    j=Ny  ·----·----·----·----·    ← wall tilted at 20°
         /    /    /    /    /       wall tangent = (1, tan 20°)
    j=1  ·----·----·----·----·    ← wall tilted at 20°
```

The wall tangent direction is `(1, tan θ)`. If we impose `u_wall = (u_w, 0)` — purely
horizontal — this velocity has a **component normal to the tilted wall**:

```
u_normal = u_wall · n_wall = u_w · (-sin θ) ≠ 0
```

This non-zero normal velocity creates a **pressure perturbation** that manifests as a
density error `δρ ≈ u_w · tan(θ) / (Ny-1)` at the wall. This is O(Ma · θ), and its
effect on velocity is O(Ma²) — exactly the residual we observed.

**This is NOT a numerical error — it is the correct LBM response** to imposing a
non-tangential velocity on a tilted wall.

### The fix: make u_wall tangent to the wall

The correct Couette flow between tilted walls has the wall velocity **tangent** to the wall:

```
u_wall = (u_w, u_w · tan(θ))
```

The Couette profile becomes:
```
s = (j - 1) / (Ny - 1)
ux(j) = s · u_w
uy(j) = s · u_w · tan(θ)
```

The Ladd correction uses the **full dot product** with both velocity components:
```
f_q = f_opp(q) - 6 · w_opp(q) · ρ · (cx_opp · u_wx + cy_opp · u_wy)
```

### Result: machine precision at all angles

| θ | After SLBM interp | After 200 steps |
|---|-------------------|-----------------|
| 0° | 4.5e-17 | 1.3e-16 |
| 20° | 4.2e-17 | 2.2e-16 |
| 45° | 5.3e-17 | 3.1e-16 |

**Machine precision recovered everywhere.** The DA-BB + standard Ladd + hybrid_z ghost
were correct all along. The "error" was in the test setup.

### Lesson learned

On a curvilinear mesh, always verify that the wall velocity is **geometrically
consistent** with the wall orientation. A "horizontal" velocity `(u_w, 0)` is only
tangent to horizontal walls. On tilted walls, the tangent velocity has both components.

For production code with arbitrary curvilinear meshes, the wall velocity should be
projected onto the wall tangent direction using the mesh metric at the wall.

---

## 5. Summary of ingredients

The complete SLBM moving wall boundary condition requires:

1. **DA-BB set selection**: reflect `q` iff `j_dep[i, j_n, q] > j_n + 0.5`
2. **Standard Ladd correction**: `f_q = f_opp(q) - 6·w_opp·ρ·(c_opp · u_wall)`
3. **Tangent wall velocity**: `u_wall` must be tangent to the wall surface
4. **Hybrid_z ghost fill**: feq(extrapolated moments) + zeroth-order fneq (for stability at large angles)

---

## 6. Notation and key equations

### D2Q9 lattice
```
q:   1    2    3    4    5    6    7    8    9
cx:  0   +1    0   -1    0   +1   -1   -1   +1
cy:  0    0   +1    0   -1   +1   +1   -1   -1
w:  4/9  1/9  1/9  1/9  1/9 1/36 1/36 1/36 1/36
opp: 1    4    5    2    3    8    9    6    7
```

### SLBM departure (parallelogram mesh, angle θ)
```
Δξ_q = cx_q
Δη_q = cy_q - cx_q · tan(θ)

i_dep = i - Δξ_q
j_dep = j - Δη_q
```

### DA-BB criterion (north wall at j_n)
```
Reflect q  ⟺  j_dep[i, j_n, q] > j_n + 0.5
           ⟺  j_n + |Δη_q| > j_n + 0.5     (for Δη_q < 0)
           ⟺  |Δη_q| > 0.5
```

### Ladd moving wall correction
```
f_q = f_opp(q) - 2·w_opp(q)·ρ·(c_opp(q) · u_wall)·3
```

### Hybrid_z ghost fill (feq from extrapolated moments + zeroth-order fneq)
```
ρ_ghost = 2·ρ_boundary - ρ_interior
u_ghost = 2·u_boundary - u_interior
feq_ghost = feq(ρ_ghost, u_ghost)
fneq_ghost = f_boundary - feq(ρ_boundary, u_boundary)    ← copy from wall
f_ghost = feq_ghost + fneq_ghost
```
