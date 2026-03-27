"""
    DiagnosticsLogger

Mutable struct holding an IO handle, file path, and column names for CSV diagnostics output.
"""
mutable struct DiagnosticsLogger
    io::IO
    filepath::String
    columns::Vector{String}
end

"""
    open_diagnostics(path, columns) -> DiagnosticsLogger

Create a CSV diagnostics file at `path` with the given column names as header.
Returns a `DiagnosticsLogger` that can be used with `log_diagnostics!` and `close_diagnostics!`.
"""
function open_diagnostics(path::String, columns::Vector{String})
    io = open(path, "w")
    println(io, join(columns, ","))
    flush(io)
    return DiagnosticsLogger(io, path, columns)
end

"""
    log_diagnostics!(logger, values...)

Append one row of comma-separated values to the diagnostics CSV and flush.
The number of values must match the number of columns.
"""
function log_diagnostics!(logger::DiagnosticsLogger, values...)
    ncol = length(logger.columns)
    nval = length(values)
    nval == ncol || error("Expected $ncol values, got $nval")
    println(logger.io, join(string.(values), ","))
    flush(logger.io)
    return nothing
end

"""
    close_diagnostics!(logger)

Close the IO handle of the diagnostics logger.
"""
function close_diagnostics!(logger::DiagnosticsLogger)
    close(logger.io)
    return nothing
end
