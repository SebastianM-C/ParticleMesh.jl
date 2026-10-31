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
#                 boundaries the last repeats the first.
#
# Schemas (also in scripts/crosscode/README.md):
#   read_scalars(dir) -> (; time::Vector{Float64}, columns::Dict{String,Vector{Float64}})
#   read_fields(dir; file = "Fields0.h5", names = nothing, timestep = nothing)
#       -> (; iterations::Vector{Int}, time::Vector{Float64}, x::Dict{String,Vector{Float64}},
#             fields::Dict{String,Matrix{Float64}})
#       fields[name] is npoints × noutputs and x[name] the positions of its points.

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

function read_fields(path::AbstractString; file = "Fields0.h5", names = nothing, timestep = nothing)
    fname = isdir(path) ? joinpath(path, file) : path
    h5open(fname, "r") do f
        data = f["data"]
        keys_ = sort(collect(keys(data)); by = k -> parse(Int, k))
        isempty(keys_) && error("$fname: no outputs in /data")
        iterations = [parse(Int, k) for k in keys_]
        first_group = data[keys_[1]]
        fieldnames = names === nothing ? sort(collect(keys(first_group))) : collect(String, names)
        x = Dict(n => _positions(first_group[n], length(first_group[n])) for n in fieldnames)
        fields = Dict(n => Matrix{Float64}(undef, length(x[n]), length(keys_)) for n in fieldnames)
        time = Vector{Float64}(undef, length(keys_))
        for (j, k) in enumerate(keys_)
            g = data[k]
            time[j] = _iteration_time(g, iterations[j], timestep)
            for n in fieldnames
                fields[n][:, j] = read(g[n])
            end
        end
        return (; iterations, time, x, fields)
    end
end

end
