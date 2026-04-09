"""
    krk_download.jl

Helper to expose `.krk` configuration files as downloadable assets in
the built documentation.
"""

"""
    krk_download(krk_path; build_dir="docs/build/assets") -> String

Copy a `.krk` configuration file into the documentation build assets
directory and return a markdown download badge ready to embed inside a
Literate.jl page.

# Arguments
- `krk_path::String`: path to the `.krk` file, relative to the project root
  or absolute.
- `build_dir::String="docs/build/assets"`: directory where the file is
  copied. Created if it does not exist.

# Returns
A markdown link string of the form `[Download name.krk](assets/name.krk)`.

# Examples
```julia
badge = krk_download("examples/cavity.krk")
println(badge)
```
"""
function krk_download(krk_path::String; build_dir::String="docs/build/assets")
    isfile(krk_path) || error("krk_download: file not found: $(krk_path)")
    mkpath(build_dir)
    fname = basename(krk_path)
    dest = joinpath(build_dir, fname)
    cp(krk_path, dest; force=true)
    return "[Download `$(fname)`](assets/$(fname))"
end
