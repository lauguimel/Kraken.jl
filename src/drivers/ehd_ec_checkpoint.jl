# EC client of the shared state/checkpoint contract (Issues #22 and #26).
# No collision, boundary, Poisson or time-integration algorithm is changed here.

const _EC_SCHEMA = 1
const _EC_IDENTITY = (:Nx, :Ny, :C, :M, :Ma_E, :alpha, :delta_U, :gamma,
    :phi_tol, :phi_max_iter, :phi_substeps, :phi_scheme, :charge_scheme,
    :ns_scheme, :force_projection, :velocity_stop, :history_interval)
const _EC_SYMBOLS = (:phi_scheme, :charge_scheme, :ns_scheme, :force_projection)

_ec_encode(v::Symbol) = String(v)
_ec_encode(v) = v
_ec_backend(b) = string(typeof(b))
function _ec_identity(c, FT)
    d = Dict{String,Any}(String(k) => _ec_encode(getproperty(c, k)) for k in _EC_IDENTITY)
    d["FT"] = string(FT)
    return d
end

"""
    snapshot(s::ECState) -> StateSnapshot

Copy carried EC state to host arrays. Use `export_state`/`save_checkpoint` for
boundary and finiteness guards. Direct potential retains its staggered time level.
"""
function snapshot(s::ECState{FT}) where {FT}
    KernelAbstractions.synchronize(s.backend)
    fields = Dict{String,Array}(String(k) => copy(Array(getproperty(s, k)))
        for k in (:f_in, :q_f_in, :Fx_prev, :Fy_prev))
    potential = s.config.phi_scheme === :lbm ? :phi_f_in : :phi
    fields[String(potential)] = copy(Array(getproperty(s, potential)))
    return StateSnapshot(solver="ehd_ec", schema_version=_EC_SCHEMA, cycle=s.cycle,
        fields=fields,
        series=Dict("cycle" => copy(s.cycle_history), "umax" => copy(s.umax_history),
            "parameter_cycle" => copy(s.parameter_cycles), "parameter_T" => copy(s.parameter_values)),
        scalars=Dict("phi_iters_last" => s.phi_iters_last,
            "phi_rel_last" => s.phi_rel_last, "q_rel_last" => s.q_rel_last),
        identity=_ec_identity(s.config, FT),
        run_control=Dict("backend" => _ec_backend(s.backend),
            "perturb_amplitude" => s.config.perturb_amplitude,
            "perturb_mode" => s.config.perturb_mode),
        parameters=Dict("T" => s.config.T),
        derived=Dict(String(k) => v for (k, v) in pairs(s.p)))
end

function _ec_keys(d, expected, label)
    for k in expected
        haskey(d, k) || throw(CheckpointError("$label/$k: missing"))
    end
    for k in keys(d)
        k in expected || throw(CheckpointError("$label/$k: unexpected key"))
    end
end

function _ec_precision(snap)
    v = get(snap.identity, "FT", nothing)
    v == "Float64" && return Float64
    v == "Float32" && return Float32
    throw(CheckpointError("identity/FT: expected Float32 or Float64, found $(repr(v))"))
end

function _ec_array(d, key, type, dims, label)
    haskey(d, key) || throw(CheckpointError("$label/$key: missing"))
    a = d[key]
    (eltype(a) === type && size(a) == dims) ||
        throw(CheckpointError("$label/$key: expected $type with size $dims, found $(eltype(a)) with size $(size(a))"))
end

"""
    validate_snapshot(ECState, snap)

Reject missing, incompatible or non-finite carried state before device allocation.
"""
function validate_snapshot(::Type{ECState}, snap::StateSnapshot)
    validate_content(snap)
    check_finite(snap)
    _ec_keys(snap.identity, [String.(_EC_IDENTITY)..., "FT"], "identity")
    FT = _ec_precision(snap)
    d = snap.identity
    for (k, lo) in (("Nx", 4), ("Ny", 8), ("history_interval", 1), ("phi_max_iter", 1))
        d[k] isa Int && d[k] >= lo || throw(CheckpointError("identity/$k: expected Int >= $lo"))
    end
    for k in ("C", "M", "Ma_E", "alpha", "delta_U", "gamma", "phi_tol", "velocity_stop")
        d[k] isa Real && isfinite(d[k]) && d[k] > 0 ||
            throw(CheckpointError("identity/$k: expected a finite positive value"))
    end
    for (k, values) in (("phi_scheme", ("lbm", "direct")),
                        ("charge_scheme", ("srt", "regularized")),
                        ("ns_scheme", ("bgk", "mrt")),
                        ("force_projection", ("none", "xy", "y")))
        d[k] in values || throw(CheckpointError("identity/$k: unsupported value $(repr(d[k]))"))
    end
    sub = d["phi_substeps"]
    (sub === nothing || (sub isa Int && sub > 0)) ||
        throw(CheckpointError("identity/phi_substeps: expected nothing or a positive Int"))
    nx, ny = d["Nx"], d["Ny"]
    potential = d["phi_scheme"] == "lbm" ? "phi_f_in" : "phi"
    _ec_keys(snap.fields, ("f_in", "q_f_in", "Fx_prev", "Fy_prev", potential), "fields")
    for k in ("f_in", "q_f_in", "Fx_prev", "Fy_prev", potential)
        dims = k in ("f_in", "q_f_in", "phi_f_in") ? (nx, ny, 9) : (nx, ny)
        _ec_array(snap.fields, k, FT, dims, "fields")
    end
    _ec_keys(snap.series, ("cycle", "umax", "parameter_cycle", "parameter_T"), "series")
    n = length(snap.series["cycle"])
    _ec_array(snap.series, "cycle", Int, (n,), "series")
    _ec_array(snap.series, "umax", FT, (n,), "series")
    h = snap.series["cycle"]
    (all(c -> 0 < c <= snap.cycle, h) && all(>(0), diff(h))) ||
        throw(CheckpointError("series/cycle: must increase strictly within the completed cycles"))
    m = length(snap.series["parameter_cycle"])
    _ec_array(snap.series, "parameter_cycle", Int, (m,), "series")
    _ec_array(snap.series, "parameter_T", Float64, (m,), "series")
    pc, pt = snap.series["parameter_cycle"], snap.series["parameter_T"]
    (m > 0 && first(pc) == 0 && all(c -> 0 <= c <= snap.cycle, pc) && issorted(pc) && all(>(0), pt)) ||
        throw(CheckpointError("series/parameter_cycle: invalid parameter-change history"))
    _ec_keys(snap.parameters, ("T",), "parameters")
    t = snap.parameters["T"]
    (t isa Real && isfinite(t) && t > 0 && Float64(t) == last(pt)) ||
        throw(CheckpointError("parameters/T: invalid value or inconsistent parameter history"))
    _ec_keys(snap.scalars, ("phi_iters_last", "phi_rel_last", "q_rel_last"), "scalars")
    v = snap.scalars["phi_iters_last"]
    v isa Int && v >= 0 || throw(CheckpointError("scalars/phi_iters_last: expected a non-negative Int"))
    for k in ("phi_rel_last", "q_rel_last")
        v = snap.scalars[k]
        (v isa FT && !isnan(v) && v >= 0) ||
            throw(CheckpointError("scalars/$k: expected non-negative $FT (Inf allowed before sampling)"))
    end
    return nothing
end

function _ec_checked_params(c, FT)
    p = _ehd_ec_lattice_params(c.Ny, c.C, c.M, c.T, c.Ma_E, c.alpha, c.delta_U, c.gamma; FT=FT)
    (all(isfinite, values(p)) && p.nu > 0 && p.tau > FT(0.5) &&
        zero(FT) < p.omega < FT(2) && p.tau_q > FT(0.5)) ||
        throw(ArgumentError("T update/restore: non-finite or unrepresentable lattice relaxation parameters"))
    return p
end

"""
    restore_state(ECState, snap; backend=CPU(), kwargs...) -> ECState

Restore the saved configuration and state. Optional identity keywords are checked
against the snapshot; they cannot change the simulation. Change T afterwards with
`update_parameter!`. Backend migration is refused in this first increment.
"""
function restore_state(::Type{ECState}, snap::StateSnapshot;
                       backend=KernelAbstractions.CPU(), kwargs...)
    expected = copy(snap.identity)
    # Key-set validation prevents an unknown identity entry from comparing to itself.
    _ec_keys(expected, [String.(_EC_IDENTITY)..., "FT"], "identity")
    for (k, v) in kwargs
        String(k) in keys(expected) ||
            throw(CheckpointError("restore/$k: only identity checks are accepted; change T with update_parameter!"))
        expected[String(k)] = k === :FT ? string(v) : _ec_encode(v)
    end
    snap = check_compatible(ECState, snap; solver="ehd_ec", schema_version=_EC_SCHEMA, identity=expected)
    validate_snapshot(ECState, snap)
    get(snap.run_control, "backend", nothing) == _ec_backend(backend) ||
        throw(CheckpointError("run_control/backend: cross-backend restoration is not supported"))
    FT = _ec_precision(snap)
    cfg = Dict{Symbol,Any}(k => (k in _EC_SYMBOLS ? Symbol(snap.identity[String(k)]) : snap.identity[String(k)])
                          for k in _EC_IDENTITY)
    cfg[:T] = snap.parameters["T"]
    cfg[:perturb_amplitude] = get(snap.run_control, "perturb_amplitude", 1e-4)
    cfg[:perturb_mode] = get(snap.run_control, "perturb_mode", 1)
    c = (; cfg...)
    p = try
        _ec_checked_params(c, FT)
    catch e
        throw(CheckpointError("derived: " * sprint(showerror, e)))
    end
    _ec_keys(snap.derived, String.(keys(p)), "derived")
    for (k, v) in pairs(p)
        isequal(v, snap.derived[String(k)]) && typeof(v) === typeof(snap.derived[String(k)]) ||
            throw(CheckpointError("derived/$k: stored lattice mapping differs from recomputation"))
    end
    # Only now allocate device arrays / rebuild the Poisson factorization.
    s = init_state(ECState; c..., backend, FT)
    for (k, a) in snap.fields
        copyto!(getproperty(s, Symbol(k)), a)
    end
    s.cycle = snap.cycle
    s.cycle_history = copy(snap.series["cycle"])
    s.umax_history = copy(snap.series["umax"])
    s.parameter_cycles = copy(snap.series["parameter_cycle"])
    s.parameter_values = copy(snap.series["parameter_T"])
    s.phi_iters_last = snap.scalars["phi_iters_last"]
    s.phi_rel_last = snap.scalars["phi_rel_last"]
    s.q_rel_last = snap.scalars["q_rel_last"]
    # Current force equals previous force on a completed cycle (also zero at 0).
    copyto!(s.Fx, s.Fx_prev)
    copyto!(s.Fy, s.Fy_prev)
    compute_ehd_scalar_2d!(s.qfield, s.q_f_in)
    if c.phi_scheme === :lbm
        compute_ehd_scalar_2d!(s.phi, s.phi_f_in)
        compute_electric_field_2d!(s.Ex, s.Ey, s.phi_f_in, s.p.tau_U)
    else
        compute_electric_field_fd_2d!(s.Ex, s.Ey, s.phi, :neumann, c.Nx, c.Ny)
    end
    KernelAbstractions.synchronize(backend)
    return s
end

updatable_parameters(::Type{ECState}) = ParameterSpace([:T], [0.0], [Inf])
updatable_parameters(::Type{<:ECState}) = updatable_parameters(ECState)

"""
    update_parameter!(s::ECState, :T, value) -> s

Update T without resetting dynamic state or time. Only nu, tau, omega and T_check
may change in the existing mapping. Rejected updates leave the state untouched.
"""
function update_parameter!(s::ECState{FT}, name::Symbol, value) where {FT}
    check_updatable(ECState, name, value)
    require_boundary(s, "update_parameter!")
    isfinite(value) && value > 0 || throw(ArgumentError("T must be finite and positive"))
    t = try
        convert(typeof(s.config.T), value)
    catch
        throw(ArgumentError("T must be representable in the original configuration type $(typeof(s.config.T))"))
    end
    isfinite(t) && t > 0 && t == value || throw(ArgumentError("T conversion would lose the requested value"))
    isequal(t, s.config.T) && return s
    c = merge(s.config, (; T=t))
    p = _ec_checked_params(c, FT)
    for k in keys(p)
        k in (:nu, :tau, :omega, :T_check) && continue
        isequal(getproperty(p, k), getproperty(s.p, k)) ||
            throw(ArgumentError("T update would also change $k; state left untouched"))
    end
    # Allocate the new provenance vectors before mutating any state field.
    cycles = [s.parameter_cycles; s.cycle]
    values = [s.parameter_values; Float64(t)]
    s.config = c
    s.p = p
    s.parameter_cycles = cycles
    s.parameter_values = values
    return s
end
