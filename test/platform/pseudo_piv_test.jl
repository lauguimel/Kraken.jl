# Pseudo-PIV generator tests (P1-P7). CPU, a few seconds.
using Test
using Random
using Statistics
import Kraken
const PIV = Kraken

# Independent (brute-force) reference helpers --------------------------------
poiseuille(Nx, Ny, m; umax = 0.03) =
    [4 * umax * ((j - 0.5) / m / (Ny / m)) * (1 - (j - 0.5) / m / (Ny / m)) for i in 1:Nx, j in 1:Ny]

# distance between the square window (centre, half side h) and a solid cell
# (side 1/m) at (cx, cy), x periodic with period Lx
function rect_dist(xa, yb, h, cx, cy, m, Lx)
    dxr = abs(cx - xa); dxr = min(dxr, Lx - dxr)
    dx = max(0.0, dxr - (h + 0.5 / m)); dy = max(0.0, abs(cy - yb) - (h + 0.5 / m))
    return hypot(dx, dy)
end

function brute_window_dist(solid, m, xa, yb, h, Ly; y_walls = true)
    Nx, Ny = size(solid); Lx = Nx / m
    d = Inf
    if y_walls
        d = min(d, yb - h, Ly - (yb + h))
    end
    for j in 1:Ny, i in 1:Nx
        solid[i, j] || continue
        d = min(d, rect_dist(xa, yb, h, (i - 0.5) / m, (j - 0.5) / m, m, Lx))
    end
    return d
end

function obstacle_like_solid(m)
    Nx, Ny = 8m, 4m
    s = falses(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        x = (i - 0.5) / m; y = (j - 0.5) / m
        (x - 4)^2 + y^2 < 1.5^2 && (s[i, j] = true)
        (x - 4)^2 + (y - 4)^2 < 1.5^2 && (s[i, j] = true)
    end
    return s
end

bits(A::AbstractArray{Float32}) = reinterpret(UInt32, vec(collect(A)))
bits(A::AbstractArray{Float64}) = reinterpret(UInt64, vec(collect(A)))

@testset "pseudo_piv" begin

    @testset "P1 window average" begin
        m = 8; Nx, Ny = 64, 32; nu0 = 0.34
        ulat = poiseuille(Nx, Ny, m)
        vlat = 0.3 .* ulat .+ 0.001 .* [sin(i) * cos(j) for i in 1:Nx, j in 1:Ny]
        solid = falses(Nx, Ny); solid[10:14, 5:9] .= true; solid[40, 1:3] .= true
        # general window (0.5 = 4 nodes, overlap 0.5), no effect other than the average
        st = PIV.PseudoPIVSettings(window = 0.5, overlap = 0.5, lost_band = 0.0, y_walls = false,
                                   sigma_rel = 0.0, dropout = 0.0)
        f = PIV.pseudo_piv(ulat, vlat, solid, m, nu0, st; rng = Xoshiro(1))
        conv = m / nu0
        maxerr = 0.0; nchecked = 0; h = 0.25
        for b in eachindex(f.y), a in eachindex(f.x)
            sx = 0.0; sy = 0.0; n = 0
            for j in 1:Ny, i in 1:Nx
                x = (i - 0.5) / m; y = (j - 0.5) / m
                # periodic image of the node nearest to the window centre
                xs = x + Nx / m * round((f.x[a] - x) / (Nx / m))
                if f.x[a] - h <= xs < f.x[a] + h && f.y[b] - h <= y < f.y[b] + h && !solid[i, j]
                    sx += ulat[i, j] * conv; sy += vlat[i, j] * conv; n += 1
                end
            end
            if n == 0
                @test !f.valid[a, b]
            else
                @test f.valid[a, b]
                maxerr = max(maxerr, abs(f.ux[a, b] - sx / n), abs(f.uy[a, b] - sy / n)); nchecked += 1
            end
        end
        @test nchecked > 400
        @test maxerr < 1e-12
        # window = 1 node: output = input
        st1 = PIV.PseudoPIVSettings(window = 1 / m, overlap = 0.0, lost_band = 0.0, y_walls = false,
                                    sigma_rel = 0.0, dropout = 0.0)
        g = PIV.pseudo_piv(ulat, vlat, falses(Nx, Ny), m, nu0, st1; rng = Xoshiro(1))
        @test size(g.ux) == (Nx, Ny)
        @test maximum(abs.(g.ux .- ulat .* conv)) < 1e-12
        @test maximum(abs.(g.uy .- vlat .* conv)) < 1e-12
        @test all(g.valid)
        @test g.x[1] ≈ 0.5 / m && g.y[end] ≈ 4 - 0.5 / m
    end

    @testset "P2 lost band" begin
        for (name, solid, m) in (("half-disks", obstacle_like_solid(16), 16),
                                 ("rectangle", begin
                                      s = falses(128, 64); s[48:80, 24:40] .= true; s
                                  end, 16))
            Nx, Ny = size(solid)
            st = PIV.PseudoPIVSettings(window = 0.25, overlap = 0.5, sigma_rel = 0.0, dropout = 0.0)
            band = PIV.piv_lost_band_value(st)
            @test band == 0.25
            u = ones(Nx, Ny)
            f = PIV.pseudo_piv(u, u, solid, m, 1.0, st; rng = Xoshiro(2))
            h = st.window / 2
            nbad = 0; nlost = 0
            for b in eachindex(f.y), a in eachindex(f.x)
                d = brute_window_dist(solid, m, f.x[a], f.y[b], h, Ny / m)
                expect_lost = d < band - 1e-9
                if f.valid[a, b] == expect_lost     # valid iff not lost (no no-fluid case here)
                    nbad += 1
                end
                nlost += expect_lost
            end
            @test nbad == 0
            @test f.counts.lost_band == nlost
            @test f.counts.total == f.counts.lost_band + f.counts.no_fluid + f.counts.dropout + f.counts.valid
            @test f.counts.dropout == 0
            @test f.counts.valid > 0 && f.counts.lost_band > 0
            # y_walls = false: wall windows come back where no solid is near
            st2 = PIV.PseudoPIVSettings(window = 0.25, overlap = 0.5, sigma_rel = 0.0, dropout = 0.0, y_walls = false)
            f2 = PIV.pseudo_piv(u, u, solid, m, 1.0, st2; rng = Xoshiro(2))
            @test f2.counts.lost_band <= f.counts.lost_band
        end
        # window with no fluid node and no band: counted as no_fluid, not as band
        solid = falses(64, 32); solid[1:8, :] .= true
        st = PIV.PseudoPIVSettings(window = 0.5, overlap = 0.0, lost_band = 0.0, y_walls = false,
                                   sigma_rel = 0.0, dropout = 0.0)
        f = PIV.pseudo_piv(ones(64, 32), ones(64, 32), solid, 8, 1.0, st; rng = Xoshiro(3))
        @test f.counts.lost_band == 0 && f.counts.no_fluid > 0
        @test f.counts.valid == f.counts.total - f.counts.no_fluid
        # periodic x: a solid block at the left edge also removes windows near the right edge
        solid = falses(64, 32); solid[1:4, 10:20] .= true
        st = PIV.PseudoPIVSettings(window = 0.5, overlap = 0.5, lost_band = 0.5, y_walls = false,
                                   sigma_rel = 0.0, dropout = 0.0)
        f = PIV.pseudo_piv(ones(64, 32), ones(64, 32), solid, 8, 1.0, st; rng = Xoshiro(3))
        a_right = length(f.x)   # last window column, x = 7.875, right edge 8.125 -> touches the periodic image
        @test any(.!f.valid[a_right, :])
    end

    @testset "P3 noise" begin
        n1, n2 = 250, 400                       # 1e5 vectors
        ux = zeros(n1, n2); uy = zeros(n1, n2)
        u_ref = 5.0
        PIV.piv_add_noise!(Xoshiro(11), ux, uy, 0.02, 0.0, u_ref)
        σ = 0.02 * u_ref; N = length(ux)
        @test abs(std(ux) / σ - 1) < 0.03 && abs(std(uy) / σ - 1) < 0.03
        @test abs(mean(ux)) <= 3σ / sqrt(N) && abs(mean(uy)) <= 3σ / sqrt(N)
        @test abs(cor(vec(ux), vec(uy))) <= 0.01
        # relative then absolute: independent, variances add
        ux .= 0; uy .= 0
        PIV.piv_add_noise!(Xoshiro(12), ux, uy, 0.02, 0.05, u_ref)
        σt = hypot(σ, 0.05)
        @test abs(std(ux) / σt - 1) < 0.03 && abs(std(uy) / σt - 1) < 0.03
        @test abs(cor(vec(ux), vec(uy))) <= 0.01
        # zero sigma: untouched
        ux .= 1; uy .= 2
        PIV.piv_add_noise!(Xoshiro(13), ux, uy, 0.0, 0.0, u_ref)
        @test all(ux .== 1) && all(uy .== 2)
    end

    @testset "P4 dropout and outliers" begin
        valid = trues(250, 400)
        nd = PIV.piv_drop!(Xoshiro(21), valid, 0.03)
        @test abs(nd / length(valid) - 0.03) <= 0.005
        @test nd == count(.!valid)
        # dropout only touches valid vectors
        v2 = falses(250, 400); @test PIV.piv_drop!(Xoshiro(22), v2, 0.5) == 0
        r = Xoshiro(23)
        ux = 1 .+ rand(r, 250, 400); uy = rand(r, 250, 400) .- 0.5
        ux0 = copy(ux); uy0 = copy(uy)
        ok = trues(size(ux))
        flag = PIV.piv_add_outliers!(Xoshiro(24), ux, uy, ok, 0.05, 3.0)
        @test abs(count(flag) / length(flag) - 0.05) <= 0.005
        n_new = hypot.(ux, uy); n_old = hypot.(ux0, uy0)
        @test all(n_new[flag] .>= 3.0 .* n_old[flag] .* (1 - 1e-12))
        @test all(ux[.!flag] .== ux0[.!flag]) && all(uy[.!flag] .== uy0[.!flag])
        # random direction: not all along the original one
        @test abs(cor(ux[flag], ux0[flag])) < 0.2
        # invalid vectors never become outliers
        bad = falses(size(ux)); f2 = PIV.piv_add_outliers!(Xoshiro(25), copy(ux), copy(uy), bad, 1.0, 3.0)
        @test count(f2) == 0
    end

    @testset "P5 reference-grid projection and npy" begin
        m = 32; Nx, Ny = 256, 128
        u = fill(0.01, Nx, Ny)
        st = PIV.PseudoPIVSettings(sigma_rel = 0.0, dropout = 0.0, lost_band = 0.0, y_walls = false)
        f = PIV.pseudo_piv(u, 0.5 .* u, falses(Nx, Ny), m, 0.34, st; rng = Xoshiro(31))
        cx = 0.01 * m / 0.34
        ux, uy, val = PIV.to_reference_grid(f)
        @test size(ux) == (256, 128) && size(uy) == (256, 128)
        @test eltype(ux) == Float32 && eltype(uy) == Float32
        @test all(isfinite, ux[val.x]) && all(isnan, ux[.!val.x])
        @test all(isfinite, uy[val.y]) && all(isnan, uy[.!val.y])
        @test all(abs.(ux[val.x] .- Float32(cx)) .<= 1f-5 * Float32(cx))
        @test all(abs.(uy[val.y] .- Float32(0.5cx)) .<= 1f-5 * Float32(cx))
        @test count(val.x) == f.counts.valid && count(val.y) == f.counts.valid
        # nearest staggered node: |node position - vector position| <= dx/2 (MAC)
        dx = 8 / 256
        okpos = true
        for j in 1:128, i in 1:256
            if val.x[i, j]
                xn = i * dx; yn = (j - 0.5) * dx
                okpos &= minimum(abs.(mod.(f.x .- xn .+ 4, 8) .- 4)) <= dx / 2 + 1e-9
                okpos &= minimum(abs.(f.y .- yn)) <= dx / 2 + 1e-9
            end
        end
        @test okpos
        # alternative convention changes only where values land
        ux2, uy2, val2 = PIV.to_reference_grid(f; stagger = :center)
        @test count(val2.x) == count(val.x)
        @test_throws ArgumentError PIV.to_reference_grid(f; stagger = :nope)
        # .npy round trip, bit for bit (NaN included), C order, header multiple of 64
        dir = mktempdir()
        PIV.write_reference_npy(dir, ux, uy, val)
        rx = PIV.read_npy(joinpath(dir, "reference_velocity_x.npy"))
        ry = PIV.read_npy(joinpath(dir, "reference_velocity_y.npy"))
        @test eltype(rx) == Float32 && size(rx) == (256, 128)
        @test bits(rx) == bits(ux) && bits(ry) == bits(uy)
        @test PIV.read_npy(joinpath(dir, "observation_mask_x.npy")) == UInt8.(val.x)
        raw = read(joinpath(dir, "reference_velocity_x.npy"))
        hl = Int(raw[9]) + 256 * Int(raw[10])
        @test (10 + hl) % 64 == 0 && raw[1:6] == UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59] && raw[7] == 1
        @test occursin("'fortran_order': False", String(raw[11:10+hl]))
        A = Float64[1 2 3; 4 5 6]               # numpy a[0, 2] == 3
        PIV.write_npy(joinpath(dir, "a.npy"), A)
        @test PIV.read_npy(joinpath(dir, "a.npy")) == A
        # x periodicity: a vector at x = Lx - eps and one at x = eps
        g = PIV.pseudo_piv(u, u, falses(Nx, Ny), m, 0.34, st; rng = Xoshiro(31))
        b0 = 10
        g.valid[end, b0] = false     # its uy would share column 256 with the x = Lx - eps vector
        g.ux[1, b0] = 111.0; g.uy[1, b0] = 222.0; g.x[1] = 8 - 1e-6
        g.ux[2, b0] = 333.0; g.uy[2, b0] = 444.0; g.x[2] = 1e-6
        jx = floor(Int, g.y[b0] / dx - 0.5 + 1 + 0.5); jy = floor(Int, g.y[b0] / dx - 1 + 1 + 0.5)
        ux3, uy3, _ = PIV.to_reference_grid(g)             # :mac
        # value 111 (x = Lx - eps) and 333 (x = eps) both land on ux column 256: averaged
        @test ux3[256, jx] ≈ (111 + 333) / 2
        @test uy3[256, jy] == 222f0           # uy at x = Lx - eps -> column 256
        @test uy3[1, jy] == 444f0             # uy at x = eps -> column 1
        ux4, uy4, _ = PIV.to_reference_grid(g; stagger = :center)
        jc = floor(Int, g.y[b0] / dx - 0.5 + 1 + 0.5)
        @test ux4[256, jc] == 111f0 && ux4[1, jc] == 333f0
    end

    @testset "P5b non-default domain 6x3" begin
        m = 16; Nx, Ny = 96, 48                       # Lx = 6, Ly = 3
        u = fill(0.02, Nx, Ny)
        st = PIV.PseudoPIVSettings(sigma_rel = 0.0, dropout = 0.0, lost_band = 0.0, y_walls = false)
        f = PIV.pseudo_piv(u, u, falses(Nx, Ny), m, 0.34, st; rng = Xoshiro(41))
        @test (f.Lx, f.Ly) == (6.0, 3.0)
        ux, uy, val = PIV.to_reference_grid(f; nx = 192, ny = 96)
        dx = 6 / 192
        @test count(val.x) == f.counts.valid
        okpos = true
        for j in 1:96, i in 1:192
            if val.x[i, j]
                xn = i * dx; yn = (j - 0.5) * dx
                okpos &= minimum(abs.(mod.(f.x .- xn .+ 3, 6) .- 3)) <= dx / 2 + 1e-9
                okpos &= minimum(abs.(f.y .- yn)) <= dx / 2 + 1e-9
            end
        end
        @test okpos
        @test_throws ArgumentError PIV.to_reference_grid(f; nx = 192, ny = 96, Lx = 8.0, Ly = 4.0)
    end

    @testset "P6 reproducibility" begin
        m = 16; Nx, Ny = 128, 64
        solid = obstacle_like_solid(m)
        ulat = poiseuille(Nx, Ny, m); vlat = 0.1 .* ulat
        st = PIV.PseudoPIVSettings(outlier_frac = 0.02)
        a = PIV.pseudo_piv(ulat, vlat, solid, m, 0.34, st; rng = Xoshiro(st.seed))
        b = PIV.pseudo_piv(ulat, vlat, solid, m, 0.34, st; rng = Xoshiro(st.seed))
        c = PIV.pseudo_piv(ulat, vlat, solid, m, 0.34, st; rng = Xoshiro(st.seed + 1))
        @test bits(a.ux) == bits(b.ux) && bits(a.uy) == bits(b.uy)
        @test a.valid == b.valid && a.outlier == b.outlier && a.counts == b.counts
        @test bits(a.ux) != bits(c.ux)
        @test a.valid != c.valid
        @test a.counts.outliers > 0 && a.counts.dropout > 0
        @test a.counts.total == a.counts.lost_band + a.counts.no_fluid + a.counts.dropout + a.counts.valid
        @test 0.5 * st.sigma_rel * a.u_ref < a.sigma_eff < 2 * st.sigma_rel * a.u_ref
    end

    @testset "P7 observation mask" begin
        m = 16; Nx, Ny = 128, 64
        solid = obstacle_like_solid(m)
        ulat = poiseuille(Nx, Ny, m)
        f = PIV.pseudo_piv(ulat, 0.0 .* ulat, solid, m, 0.34, PIV.PseudoPIVSettings(); rng = Xoshiro(7))
        @test count(PIV.piv_mask(f.valid)) == f.counts.valid
        @test PIV.piv_mask(f) == f.valid
        ux, uy, val = PIV.to_reference_grid(f; nx = 256, ny = 128)
        @test count(val.x) == f.counts.valid == count(val.y)
        @test count(isfinite, ux) == count(val.x)
        @test PIV.piv_mask(val) == (val.x .| val.y)
        @test PIV.piv_mask(val) isa BitMatrix
    end

    @testset "settings validation" begin
        @test_throws ArgumentError PIV.pseudo_piv(ones(8, 8), ones(8, 8), falses(8, 8), 2, 1.0,
                                                  PIV.PseudoPIVSettings(overlap = 1.0); rng = Xoshiro(1))
        @test_throws ArgumentError PIV.pseudo_piv(ones(8, 8), ones(8, 8), falses(8, 8), 2, 1.0,
                                                  PIV.PseudoPIVSettings(dropout = 1.5); rng = Xoshiro(1))
    end
end
