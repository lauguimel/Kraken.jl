struct Issue
    severity::Symbol
    code::Symbol
    message::String
end

struct PlanValidationError <: Exception
    plan::Any
    issues::Vector{Issue}
end

struct NotImplementedError <: Exception
    message::String
end

Base.showerror(io::IO, err::NotImplementedError) =
    print(io, "NotImplementedError: ", err.message)

function Base.showerror(io::IO, err::PlanValidationError)
    codes = join(string.(getfield.(err.issues, :code)), ", ")
    print(io, "PlanValidationError: blocking units issues: ", codes)
end

fatal_issue(code::Symbol, message::AbstractString) =
    Issue(:fatal, code, String(message))
error_issue(code::Symbol, message::AbstractString) =
    Issue(:error, code, String(message))
warn_issue(code::Symbol, message::AbstractString) =
    Issue(:warn, code, String(message))
info_issue(code::Symbol, message::AbstractString) =
    Issue(:info, code, String(message))

const _SEVERITY_RANK = Dict(:fatal => 1, :error => 2, :warn => 3, :info => 4)

_issue_rank(i::Issue) = (get(_SEVERITY_RANK, i.severity, 99), String(i.code))
sort_issues(issues::Vector{Issue}) = sort(issues; by=_issue_rank)
blocking_issues(issues::Vector{Issue}) =
    [i for i in issues if i.severity === :fatal || i.severity === :error]
issue_codes(issues::Vector{Issue}) = sort([i.code for i in issues])

function emit_warning_logs(issues::Vector{Issue})
    for issue in issues
        issue.severity === :warn && @warn issue.message code=issue.code
    end
    return nothing
end

function phase_stub_error(sym::Symbol)
    return NotImplementedError("physics :$sym is a Phase-2 stub (see units-v1.md §7)")
end
