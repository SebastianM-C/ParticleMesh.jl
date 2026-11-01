# Readers for Smilei's diagnostics (layout of Smilei v5.1).
#
#   scalars.txt   written by src/Diagnostic/DiagnosticScalar.cpp: header lines "# <k> <name>",
#                 then "#", then one "#" line with all column names; then one row per output,
#                 whitespace separated, first column the time in units of 1/ω_r.
#   Fields<n>.h5  written by src/Diagnostic/DiagnosticFields*.cpp: group "/data", one subgroup per
#                 output named by the iteration number zero-padded to 10 digits, with the openPMD
#                 attributes `time` (iteration × timestep, normalized) and `dt`; one dataset per
#                 field ("Ex", "Rho", ...) with `gridSpacing`, `gridGlobalOffset` and `position`.
#                 In 1D a dataset holds cells + 1 points (default subgrid); for periodic
#                 boundaries the last repeats the first. In 2D (DiagnosticFields2D.cpp) a
#                 dataset has the C-order dimensions [Nx + 1, Ny + 1] (x slowest, openPMD
#                 `dataOrder = "C"`, `axisLabels = ["x", "y"]`), and `gridSpacing`,
#                 `gridGlobalOffset` and `position` hold one entry per axis, in x, y order.
#
# Schemas (also in scripts/crosscode/README.md):
#   read_scalars(dir) -> (; time::Vector{Float64}, columns::Dict{String,Vector{Float64}})
#   read_fields(dir; file = "Fields0.h5", names = nothing, timestep = nothing)
#       -> (; iterations::Vector{Int}, time::Vector{Float64}, x, fields)
#       1D: x::Dict{String,Vector{Float64}}, fields::Dict{String,Matrix{Float64}};
#           fields[name] is npoints × noutputs and x[name] the positions of its points.
#       2D: x::Dict{String,NTuple{2,Vector{Float64}}}, fields::Dict{String,Array{Float64,3}};
#           fields[name] is npoints_x × npoints_y × noutputs (x first) and x[name] the
#           positions (x, y) of its points along each axis.

module SmileiRead

using HDF5

export read_scalars, read_fields

function read_scalars(path::AbstractString)
    file = isdir(path) ? joinpath(path, "scalars.txt") : path
    names = SubString{String}[]
    rows = Vector{Vector{Float64}}()
    last_header = ""
    for line in eachline(file)
        s = strip(line)
        isempty(s) && continue
        if startswith(s, "#")
            last_header = s
            continue
        end
        if isempty(names)
            names = split(strip(last_header[2:end]))
        end
        vals = parse.(Float64, split(s))
        length(vals) == length(names) ||
            error("scalars.txt: row with $(length(vals)) values, header has $(length(names)) names")
        push!(rows, vals)
    end
    isempty(names) && error("scalars.txt: no data rows in $file")
    names[1] == "time" || error("scalars.txt: first column is $(names[1]), expected time")
    columns = Dict(String(n) => [r[i] for r in rows] for (i, n) in enumerate(names))
    return (; time = columns["time"], columns)
end

"""
Time of an output in normalized units: the iteration attribute `time`, which Smilei writes as
iteration × timestep in units of 1/ω_r; without it, iteration × `timestep`.
"""
function _iteration_time(g, it, timestep)
    a = attrs(g)
    haskey(a, "time") && return Float64(_scalar(a["time"]))
    timestep === nothing && error("output $it has no time attribute; pass timestep")
    return it * timestep
end

_scalar(x) = x isa AbstractArray ? only(x) : x

"""
Positions of the points of a dataset from its openPMD attributes,
x = gridGlobalOffset + (i + position) gridSpacing for i = 0, 1, … . Smilei records `position`
0.5 for dual fields (Ex in 1D) and 0 for primal ones. This is the openPMD convention as written;
where Smilei's first dual point sits relative to it was not verified in the source.
"""
function _positions(ds, n)
    a = attrs(ds)
    dx = haskey(a, "gridSpacing") ? Float64(_scalar(a["gridSpacing"])) : NaN
    x0 = haskey(a, "gridGlobalOffset") ? Float64(_scalar(a["gridGlobalOffset"])) : 0.0
    s = haskey(a, "position") ? Float64(_scalar(a["position"])) : 0.0
    return collect(x0 .+ ((0:n-1) .+ s) .* dx)
end

"Per-axis positions of a 2D dataset with `n = (nx, ny)` points, from its per-axis openPMD attributes."
function _positions2(ds, n)
    a = attrs(ds)
    axis(key, d, default) = haskey(a, key) ? Float64(a[key][d]) : default
    return ntuple(d -> collect(axis("gridGlobalOffset", d, 0.0) .+
        ((0:n[d]-1) .+ axis("position", d, 0.0)) .* axis("gridSpacing", d, NaN)), 2)
end

# A dataset in x-first order: HDF5.jl shows the C-order dimensions [Nx + 1, Ny + 1] reversed.
_xfirst(v::AbstractVector) = v
_xfirst(v::AbstractMatrix) = permutedims(v)

function read_fields(path::AbstractString; file = "Fields0.h5", names = nothing, timestep = nothing)
    fname = isdir(path) ? joinpath(path, file) : path
    h5open(fname, "r") do f
        data = f["data"]
        keys_ = sort(collect(keys(data)); by = k -> parse(Int, k))
        isempty(keys_) && error("$fname: no outputs in /data")
        iterations = [parse(Int, k) for k in keys_]
        first_group = data[keys_[1]]
        fieldnames = names === nothing ? sort(collect(keys(first_group))) : collect(String, names)
        nd = ndims(first_group[first(fieldnames)])
        nd in (1, 2) || error("$fname: $nd-dimensional fields are not read")
        time = Vector{Float64}(undef, length(keys_))
        if nd == 1
            x = Dict(n => _positions(first_group[n], length(first_group[n])) for n in fieldnames)
            fields = Dict(n => Matrix{Float64}(undef, length(x[n]), length(keys_)) for n in fieldnames)
        else
            npts = Dict(n => reverse(size(first_group[n])) for n in fieldnames)
            x = Dict(n => _positions2(first_group[n], npts[n]) for n in fieldnames)
            fields = Dict(n => Array{Float64,3}(undef, npts[n]..., length(keys_)) for n in fieldnames)
        end
        for (j, k) in enumerate(keys_)
            g = data[k]
            time[j] = _iteration_time(g, iterations[j], timestep)
            for n in fieldnames
                selectdim(fields[n], nd + 1, j) .= _xfirst(read(g[n]))
            end
        end
        return (; iterations, time, x, fields)
    end
end

end
