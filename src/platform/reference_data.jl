# Optional reference data of the inverse-rheology track (not distributed with
# Kraken): root directory from the environment variable KRAKEN_REFERENCE_DATA.
# Names are not exported.

"""
    reference_data_dir() -> Union{String,Nothing}

Root of the optional reference data of the inverse-rheology track (fixtures,
reference fields and tables, not distributed with Kraken), from the environment
variable `KRAKEN_REFERENCE_DATA`; `nothing` when unset or empty. Tests that
need it are skipped without it.
"""
function reference_data_dir()
    d = get(ENV, "KRAKEN_REFERENCE_DATA", "")
    return isempty(d) ? nothing : d
end

"`joinpath(reference_data_dir(), parts...)`, or `nothing` without reference data."
reference_data_path(parts::AbstractString...) =
    (d = reference_data_dir(); d === nothing ? nothing : joinpath(d, parts...))
