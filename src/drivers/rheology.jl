# --- Phase-field static droplet (Laplace law validation) ---

"""
    run_static_droplet_phasefield_2d(; N, R, W_pf, σ, ρ_l, ρ_g, ν, τ_g, max_steps, backend, FT)

Static droplet validation for the phase-field two-phase model.
Measures pressure jump Δp and compares to Laplace law: Δp = σ/R.

Uses Allen-Cahn for interface tracking and pressure-based MRT for momentum.
Fully periodic boundary conditions.
"""
function run_static_droplet_phasefield_2d(;
        N=100, R=25, W_pf=4.0, σ=0.01,
        ρ_l=1.0, ρ_g=0.001, ν=0.1,
        τ_g=0.7, max_steps=5000,
        backend=KernelAbstractions.CPU(), FT=Float64)

    Nx, Ny = N, N
    β, κ = phasefield_params(σ, W_pf)

    @info "Phase-field static droplet" N R W_pf σ ρ_l ρ_g ν τ_g β κ

    # Initialize φ: circular droplet centered in domain
    φ_cpu = zeros(FT, Nx, Ny)
    ux_cpu = zeros(FT, Nx, Ny)
    uy_cpu = zeros(FT, Nx, Ny)
    cx = FT(Nx) / FT(2) + FT(0.5)
    cy = FT(Ny) / FT(2) + FT(0.5)
    for j in 1:Ny, i in 1:Nx
        x = FT(i) - FT(0.5)
        y = FT(j) - FT(0.5)
        r = sqrt((x - cx)^2 + (y - cy)^2)
        φ_cpu[i,j] = -tanh((r - FT(R)) / FT(W_pf))
    end

    # Initialize distributions
    f_cpu = init_pressure_equilibrium(φ_cpu, ux_cpu, uy_cpu, ρ_l, ρ_g, FT)
    g_cpu = init_phasefield_equilibrium(φ_cpu, ux_cpu, uy_cpu, FT)

    # GPU arrays
    f_in  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    f_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_in  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    φ     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    μ_pf  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    p     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ux    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    uy    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Ax    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Ay    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)

    copyto!(f_in, f_cpu); copyto!(f_out, f_cpu)
    copyto!(g_in, g_cpu); copyto!(g_out, g_cpu)
    copyto!(φ, φ_cpu)

    # Main loop
    for step in 1:max_steps
        # 1. Stream (fully periodic)
        stream_fully_periodic_2d!(f_out, f_in, Nx, Ny)
        stream_fully_periodic_2d!(g_out, g_in, Nx, Ny)

        # 2. Compute φ from g
        compute_phi_2d!(φ, g_out)

        # 3. Chemical potential and surface tension force (with ramp for stability)
        ramp = min(FT(step) / FT(500), one(FT))
        compute_chemical_potential_2d!(μ_pf, φ, β, κ)
        compute_phasefield_force_2d!(Fx, Fy, μ_pf, φ)
        if ramp < one(FT)
            Fx .*= ramp; Fy .*= ramp
        end

        # 4. Macroscopic (pressure-based)
        compute_macroscopic_phasefield_2d!(p, ux, uy, f_out, φ, Fx, Fy;
                                           ρ_l=ρ_l, ρ_g=ρ_g)

        # 5. Allen-Cahn collision (conservative form)
        compute_antidiffusion_flux_2d!(Ax, Ay, φ)
        collide_allen_cahn_2d!(g_out, ux, uy, Ax, Ay; τ_g=τ_g, W=W_pf)

        # 6. Pressure-based MRT collision
        collide_pressure_phasefield_mrt_2d!(f_out, φ, Fx, Fy, is_solid;
                                             ρ_l=ρ_l, ρ_g=ρ_g, ν_l=ν, ν_g=ν)

        # 7. Swap
        f_in, f_out = f_out, f_in
        g_in, g_out = g_out, g_in
    end

    # Final macroscopic
    compute_phi_2d!(φ, g_in)
    compute_chemical_potential_2d!(μ_pf, φ, β, κ)
    compute_phasefield_force_2d!(Fx, Fy, μ_pf, φ)
    compute_macroscopic_phasefield_2d!(p, ux, uy, f_in, φ, Fx, Fy;
                                       ρ_l=ρ_l, ρ_g=ρ_g)

    p_cpu = Array(p)
    φ_cpu_out = Array(φ)

    # Measure pressure jump: average inside vs outside
    center_i = Nx ÷ 2 + 1
    center_j = Ny ÷ 2 + 1
    n_in = 0; sum_in = FT(0)
    n_out = 0; sum_out = FT(0)
    for j in 1:Ny, i in 1:Nx
        if φ_cpu_out[i,j] > FT(0.5)
            sum_in += p_cpu[i,j]; n_in += 1
        elseif φ_cpu_out[i,j] < -FT(0.5)
            sum_out += p_cpu[i,j]; n_out += 1
        end
    end
    p_inside = n_in > 0 ? sum_in / n_in : p_cpu[center_i, center_j]
    p_outside = n_out > 0 ? sum_out / n_out : p_cpu[1, 1]
    Δp = p_inside - p_outside
    Δp_exact = FT(σ) / FT(R)

    @info "Laplace law" Δp Δp_exact error_pct=abs(Δp - Δp_exact) / Δp_exact * 100

    return (p=p_cpu, φ=φ_cpu_out, ux=Array(ux), uy=Array(uy),
            Δp=Δp, Δp_exact=Δp_exact,
            params=(; N, R, W_pf, σ, ρ_l, ρ_g, ν, β, κ))
end

# --- Phase-field CIJ jet (axisymmetric) ---

"""
    run_cij_jet_phasefield_2d(; Re, We, δ, R0, u_lb, ρ_ratio, μ_ratio, W_pf, τ_g, ...)

Axisymmetric CIJ jet simulation using phase-field (Allen-Cahn + pressure-based MRT).
Supports density ratios up to 1000:1 (vs ρ_ratio ≈ 10 for VOF-based driver).

Two D2Q9 distributions: f (pressure/velocity), g (Allen-Cahn order parameter φ).
"""
function run_cij_jet_phasefield_2d(;
        Re=200, We=600, δ=0.02,
        R0=40, u_lb=0.04,
        domain_ratio=80, nr_ratio=3,
        ρ_ratio=1000.0, μ_ratio=10.0,
        W_pf=4.0, τ_g=0.7,
        init_length=4, max_steps=200_000,
        output_interval=2000,
        output_dir="cij_jet_pf",
        backend=KernelAbstractions.CPU(), FT=Float64)

    # --- Derive LBM parameters ---
    ρ_l = FT(1.0)
    ρ_g = ρ_l / FT(ρ_ratio)
    ν_l = FT(u_lb) * FT(R0) / FT(Re)
    ν_g = ρ_l * ν_l / (ρ_g * FT(μ_ratio))
    σ_lb = ρ_l * FT(u_lb)^2 * FT(R0) / FT(We)

    τ_l = FT(3) * ν_l + FT(0.5)
    τ_g_visc = FT(3) * ν_g + FT(0.5)

    β, κ = phasefield_params(σ_lb, W_pf)

    @info "CIJ jet (phase-field)" Re We δ R0 u_lb ν_l ν_g σ_lb τ_l τ_g_visc ρ_l ρ_g β κ W_pf τ_g

    if τ_l < FT(0.505) || τ_g_visc < FT(0.505)
        @warn "Relaxation time close to 0.5" τ_l τ_g_visc
    end

    # --- Domain ---
    Nz = Int(domain_ratio * R0)
    Nr = Int(nr_ratio * R0)
    Nx, Ny = Nz, Nr

    # --- Allocate arrays ---
    f_in  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    f_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_in  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    φ     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    μ_pf  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    p     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ux    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    uy    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx_st = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_st = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Ax    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Ay    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_gpu = KernelAbstractions.zeros(backend, FT, Nx, Ny)  # VOF from φ for axisym correction
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)

    # Perturbation frequency
    T_period = FT(7) * FT(R0) / FT(u_lb)
    f_stim = one(FT) / T_period

    # Inlet profiles (spatial shape)
    W_smooth = FT(3)
    inlet_profile_cpu = zeros(FT, Ny)
    inlet_phi_cpu = zeros(FT, Ny)
    for j in 1:Ny
        r = FT(j) - FT(0.5)
        envelope = FT(0.5) * (one(FT) - tanh((r - FT(R0)) / W_smooth))
        inlet_profile_cpu[j] = envelope
        inlet_phi_cpu[j] = -tanh((r - FT(R0)) / FT(W_pf))
    end
    inlet_profile = KernelAbstractions.zeros(backend, FT, Ny)
    copyto!(inlet_profile, inlet_profile_cpu)
    inlet_phi_gpu = KernelAbstractions.zeros(backend, FT, Ny)
    copyto!(inlet_phi_gpu, inlet_phi_cpu)

    ux_inlet = KernelAbstractions.zeros(backend, FT, Ny)
    uy_inlet = KernelAbstractions.zeros(backend, FT, Ny)

    # --- Initialize: flat jet of length init_length·R0 ---
    φ_cpu = zeros(FT, Nx, Ny)
    ux_cpu = zeros(FT, Nx, Ny)
    L_init = FT(init_length * R0)
    for j in 1:Ny, i in 1:Nx
        r = FT(j) - FT(0.5)
        z = FT(i) - FT(0.5)
        φ_r = -tanh((r - FT(R0)) / FT(W_pf))
        φ_z = tanh((L_init - z) / FT(W_pf))
        φ_cpu[i,j] = min(φ_r, φ_z)
        ux_cpu[i,j] = max(FT(0), (one(FT) + φ_cpu[i,j]) / FT(2)) * FT(u_lb)
    end

    f_cpu = init_pressure_equilibrium(φ_cpu, ux_cpu, zeros(FT, Nx, Ny), ρ_l, ρ_g, FT)
    g_cpu = init_phasefield_equilibrium(φ_cpu, ux_cpu, zeros(FT, Nx, Ny), FT)

    copyto!(f_in, f_cpu); copyto!(f_out, f_cpu)
    copyto!(g_in, g_cpu); copyto!(g_out, g_cpu)
    copyto!(φ, φ_cpu)

    # C field for VOF output compatibility
    C_out = zeros(FT, Nx, Ny)

    # --- Output setup ---
    mkpath(output_dir)
    pvd = create_pvd(joinpath(output_dir, "cij_jet_pf"))

    interface_snapshots = Dict{Int, Vector{NTuple{2,FT}}}()
    breakup_detected = false
    breakup_step = 0

    @info "CIJ jet (phase-field) simulation" Nz Nr T_period max_steps output_dir

    # --- Main loop ---
    for step in 1:max_steps

        # 1. Stream (axisym: specular j=1, wall j=Ny, bounce-back i=1/Nx)
        stream_axisym_inlet_2d!(f_out, f_in, Nx, Ny)
        stream_axisym_inlet_2d!(g_out, g_in, Nx, Ny)

        # 2. Inlet BC for f: pulsed Zou-He velocity
        u_t = FT(u_lb) * (one(FT) + FT(δ) * sin(FT(2π) * f_stim * FT(step)))
        ux_inlet .= inlet_profile .* u_t
        apply_zou_he_west_spatial_2d!(f_out, ux_inlet, uy_inlet, Nx, Ny)

        # 3. Inlet BC for g: equilibrium at prescribed φ and velocity
        set_phasefield_west_2d!(g_out, inlet_phi_gpu, ux_inlet)

        # 4. Outlet BC
        apply_zou_he_pressure_east_2d!(f_out, Nx, Ny; ρ_out=1.0)
        extrapolate_phasefield_east_2d!(g_out, Nx, Ny)

        # 5. Compute φ from g
        compute_phi_2d!(φ, g_out)

        # 6. Chemical potential + azimuthal correction + force
        compute_chemical_potential_2d!(μ_pf, φ, β, κ)
        add_azimuthal_chemical_potential_2d!(μ_pf, φ, κ, Ny)
        compute_phasefield_force_2d!(Fx_st, Fy_st, μ_pf, φ)

        # 7. Macroscopic (pressure-based)
        compute_macroscopic_phasefield_2d!(p, ux, uy, f_out, φ, Fx_st, Fy_st;
                                           ρ_l=ρ_l, ρ_g=ρ_g)

        # 8. Axisymmetric viscous correction: ν/r · ∂uz/∂r
        compute_vof_from_phi_2d!(C_gpu, φ)
        add_axisym_viscous_correction_2d!(Fx_st, ux, C_gpu, ν_l, ν_g, Ny)

        # 9. Allen-Cahn collision (conservative) + azimuthal correction
        compute_antidiffusion_flux_2d!(Ax, Ay, φ)
        collide_allen_cahn_2d!(g_out, ux, uy, Ax, Ay; τ_g=τ_g, W=W_pf)
        add_azimuthal_allen_cahn_source_2d!(g_out, φ; τ_g=τ_g)

        # 10. Pressure-based MRT collision
        collide_pressure_phasefield_mrt_2d!(f_out, φ, Fx_st, Fy_st, is_solid;
                                             ρ_l=ρ_l, ρ_g=ρ_g, ν_l=ν_l, ν_g=ν_g)

        # 11. Swap
        f_in, f_out = f_out, f_in
        g_in, g_out = g_out, g_in

        # --- Output and diagnostics ---
        if step % output_interval == 0
            φ_out = Array(φ)
            C_out .= (φ_out .+ one(FT)) ./ FT(2)
            ux_out = Array(ux)

            # Extract interface contour (C = 0.5 → φ = 0)
            interface_pts = NTuple{2,FT}[]
            for i in 1:Nx, j in 1:Ny-1
                if φ_out[i,j] * φ_out[i,j+1] < 0
                    frac = -φ_out[i,j] / (φ_out[i,j+1] - φ_out[i,j])
                    push!(interface_pts, (FT(i) - FT(0.5), (FT(j) - FT(0.5)) + frac))
                end
            end
            interface_snapshots[step] = interface_pts

            # Breakup detection
            if !breakup_detected && step > 5 * Int(round(T_period))
                for i in Int(round(5*R0)):Nx-1
                    max_c = maximum(C_out[i, 1:min(2*R0, Ny)])
                    if max_c < FT(0.01)
                        has_upstream = any(C_out[max(1,i-5*R0):i-1, 1] .> FT(0.5))
                        has_downstream = i + 5 <= Nx && any(C_out[i+1:min(i+5*R0,Nx), 1] .> FT(0.5))
                        if has_upstream && has_downstream
                            breakup_detected = true
                            breakup_step = step
                            @info "Breakup detected" step z=FT(i)-FT(0.5) t_phys=step*u_lb/R0
                            break
                        end
                    end
                end
            end

            # VTK output
            t_phys = FT(step) * FT(u_lb) / FT(R0)
            write_vtk_to_pvd(pvd,
                joinpath(output_dir, "cij_jet_pf_$(lpad(step, 7, '0'))"),
                Nx, Ny, 1.0,
                Dict("C" => C_out, "phi" => φ_out, "p" => Array(p),
                     "ux" => ux_out, "uy" => Array(uy)),
                Float64(t_phys))

            p_out = Array(p)
            sum_C = sum(C_out)
            @info "Step $step / $max_steps" t_phys n_interface=length(interface_pts) sum_C p_range=extrema(p_out) ux_range=extrema(ux_out) breakup=breakup_detected
        end

        if breakup_detected && step > breakup_step + 3 * Int(round(T_period))
            @info "Stopping: 3 periods after breakup" step
            break
        end
    end

    vtk_save(pvd)

    # Final macroscopic
    compute_phi_2d!(φ, g_in)
    compute_chemical_potential_2d!(μ_pf, φ, β, κ)
    compute_phasefield_force_2d!(Fx_st, Fy_st, μ_pf, φ)
    compute_macroscopic_phasefield_2d!(p, ux, uy, f_in, φ, Fx_st, Fy_st;
                                       ρ_l=ρ_l, ρ_g=ρ_g)

    φ_final = Array(φ)
    C_final = (φ_final .+ one(FT)) ./ FT(2)

    return (p=Array(p), uz=Array(ux), ur=Array(uy), C=C_final, φ=φ_final,
            interfaces=interface_snapshots,
            breakup_detected=breakup_detected, breakup_step=breakup_step,
            params=(; Re, We, δ, R0, u_lb, ν_l, ν_g, σ_lb, ρ_l, ρ_g, Nz=Nx, Nr=Ny,
                    T_period, W_pf, τ_g, β, κ))
end
