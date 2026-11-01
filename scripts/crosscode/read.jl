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
#                 Smilei writes `position = 0.5` on dual axes (DiagnosticFields.cpp,
#                 stagger = 0.5·isDual) but starts the data at local index `oversize`
#                 (DiagnosticFields2D.cpp), which on a dual axis is x = −Δ/2: point i of a dual
#                 axis sits at (i − ½)Δ, not at (i + ½)Δ as the openPMD reading of the attribute
#                 gives. The 2D reader places it at gridGlobalOffset + (i − position)·gridSpacing;
#                 Gauss's law on the dumps closes to roundoff only with that convention
#                 (reports/crosscode-smilei-2d/review.md, B1).
#
# Schemas (also in scripts/crosscode/README.md):
#   read_scalars(dir) -> (; time::Vector{Float64}, columns::Dict{String,Vector{Float64}})
#   read_fields(dir; file = "Fields0.h5", names = nothing, timestep = nothing)
#       -> (; iterations::Vector{Int}, time::Vector{Float64}, x, fields)
#       1D: x::Dict{String,Vector{Float64}}, fields::Dict{String,Matrix{Float64}};
#           fields[name] is npoints × noutputs and x[name] the positions of its points.
#       2D: x::Dict{String,NTuple{2,Vector{Float64}}}, fields::Dict{String,Array{Float64,3}};
#           fields[name] is npoints_x × npoints_y × noutputs (x first) and x[name] the
#           positions (x, y) of its points along each axis, a dual axis starting at −Δ/2.

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

"""
Per-axis positions of a 2D dataset with `n = (nx, ny)` points, in Smilei's layout:
x = gridGlobalOffset + (i − position) gridSpacing for i = 0, 1, …, so the first point of a
dual axis (`position = 0.5`) is at −Δ/2 (see the file header).
"""
function _positions2(ds, n)
    a = attrs(ds)
    axis(key, d, default) = haskey(a, key) ? Float64(a[key][d]) : default
    return ntuple(d -> collect(axis("gridGlobalOffset", d, 0.0) .+
        ((0:n[d]-1) .- axis("position", d, 0.0)) .* axis("gridSpacing", d, NaN)), 2)
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

# Readers for PIConGPU's output (PIConGPU 0.8.0, the openPMD plugin with the HDF5 backend and
# group-based iteration encoding, `--openPMD.infix NULL`), into the schemas above.
#
#   openPMD/simData.h5   written by plugins/openPMD/openPMDWriter.x.cpp: group "/data", one
#                        subgroup per iteration (attributes `time`, `dt`, `timeUnitSI`, from
#                        plugins/common/openPMDWriteMeta.hpp), with "fields/E/{x,y,z}",
#                        "fields/B/{x,y,z}" and "fields/<species>_all_chargeDensity" (the
#                        file's `meshesPath` is "fields/"). A 2D
#                        dataset is C-order F[y][x] (axisLabels ["y", "x"]), which HDF5.jl shows
#                        as an Nx × Ny array, x first; it holds the cells only (no repeated
#                        periodic point). `gridSpacing` and `gridGlobalOffset` follow axisLabels
#                        (y first), while `position` is written x first (the writer loops over
#                        the dimensions d = 0, 1 = x, y without reversing; E_x at (0.5, 0), E_y at
#                        (0, 0.5), B_x at (0, 0.5), B_y at (0.5, 0), the charge density at (0, 0),
#                        fields/YeeCell.hpp). Values times `unitSI` are SI.
#   fields_energy.dat    plugins/EnergyFields.x.cpp: "#step total Bx By Bz Ex Ey Ez" in J.
#   <species>_energy_all.dat  plugins/EnergyParticles.x.cpp: "#step Ekin_Joule E_Joule" in J;
#                        Ekin_Joule is p²/2m below γ = 1.005 (algorithms/KinEnergy.hpp), E_Joule
#                        the exact Σ γmc², from which the reader forms the kinetic energy.
#   units.toml           written by scripts/crosscode/picongpu.jl: the SI value of each unit of
#                        A1 for the run, and the J per unit of the energy scalars.
#
# Schemas: `read_picongpu_scalars(dir)` returns (; time, columns) with the column names of
# Smilei's scalars.txt (`Uelm`, `Uelm_Ex`, …, `Uelm_Bx_m`, … for B, `Ukin`, `Utot`), in the
# units of A1 (energies per unit depth in 2D, per unit transverse area in quasi-1D, as Smilei
# 2D and 1D); `read_picongpu_fields(dir; names, yaverage)` returns (; iterations, time, x,
# fields) as `read_fields` does, with names "Ex", "Ey", "Ez", "Bx", "By", "Bz", "Rho" (the sum
# of the species' charge densities) and "Rho_<species>". The reader appends the periodic copy
# of the first point on each axis, so that a field has cells + 1 points per axis as Smilei's do
# and the analysis drops it the same way. With `yaverage = true` (the quasi-1D runs) each field
# is averaged over y and returned in the 1D schema: points × outputs, x[name] a vector.

module PIConGPURead

using HDF5
using TOML
using Statistics: mean

export read_picongpu_scalars, read_picongpu_fields, picongpu_output, yaverage

"The run directory's units.toml, looked up in `dir` and its parent (a seed directory's run)."
function picongpu_units(dir)
    for d in (dir, dirname(abspath(dir)))
        f = joinpath(d, "units.toml")
        isfile(f) && return TOML.parsefile(f)
    end
    error("no units.toml in $dir or its parent")
end

"Whether `dir` holds PIConGPU output (openPMD/simData.h5 or fields_energy.dat)."
picongpu_output(dir) = isfile(joinpath(dir, "openPMD", "simData.h5")) || isfile(joinpath(dir, "fields_energy.dat"))

function read_dat(file)
    rows = Vector{Vector{Float64}}()
    for line in eachline(file)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        push!(rows, parse.(Float64, split(s)))
    end
    isempty(rows) && error("$file: no data rows")
    return rows
end

"""
The rest energy of each mobile species in the units of the energy scalars: Σ_s n_s m_s × the box
length (quasi-1D, per unit transverse area) or the box area (2D, per unit depth), from the run's
case.toml (in `dir` or its parent). The loaded weights sum to n·V exactly for an integer-mode
cosine profile, so this is Σ w m c² of the load.
"""
function rest_energy(dir, u)
    f = isfile(joinpath(dir, "case.toml")) ? joinpath(dir, "case.toml") : joinpath(dirname(abspath(dir)), "case.toml")
    case = TOML.parsefile(f)
    L = case["grid"]["length"]
    V = u["quasi1d"] ? Float64(L isa AbstractVector ? L[1] : L) : prod(Float64.(L))
    return Dict(s["name"] => Float64(s["density"]) * Float64(s["mass"]) * V for s in case["species"] if s["mobile"])
end

"""
    read_picongpu_scalars(dir) -> (; time, columns)

PIConGPU's energy files in the units of A1 and Smilei's column names.

- `Uelm` is the 16-digit total of `fields_energy.dat`; its components are printed with 6
  significant digits, so `Uelm_Ex` is formed as `Uelm` minus the other five components (an
  absolute error of 5 × 10⁻⁷ of those, which are small next to Ex in these cases), and the
  printed component is kept as `Uelm_Ex_printed`.
- `Ukin` is the exact relativistic kinetic energy Σ (γ − 1) m c², taken as the `E_Joule` column
  (Σ γ m c², exact) minus the rest energy of the load (`rest_energy`). PIConGPU's own `Ekin_Joule`
  column switches to p²/2m below γ = 1.005 (`algorithms/KinEnergy.hpp`, `GAMMA_THRESH` in
  `param/speciesConstants.param`); it is kept as `Ukin_threshold`. In FP32 runs the subtraction
  carries the float rounding of the total energy, about 10⁻⁵ of the kinetic energy.
"""
function read_picongpu_scalars(dir)
    u = picongpu_units(dir)
    J = u["energy"]
    rows = read_dat(joinpath(dir, "fields_energy.dat"))
    steps = [Int(r[1]) for r in rows]
    cols = Dict{String,Vector{Float64}}()
    for (k, n) in enumerate(("Uelm", "Uelm_Bx_m", "Uelm_By_m", "Uelm_Bz_m", "Uelm_Ex_printed", "Uelm_Ey", "Uelm_Ez"))
        cols[n] = [r[k+1] / J for r in rows]
    end
    cols["Uelm_Ex"] = cols["Uelm"] .- cols["Uelm_Ey"] .- cols["Uelm_Ez"] .- cols["Uelm_Bx_m"] .- cols["Uelm_By_m"] .- cols["Uelm_Bz_m"]
    rest = rest_energy(dir, u)
    ukin = zeros(length(steps))
    uthr = zeros(length(steps))
    for f in readdir(dir)
        m = match(r"^(.+)_energy_all\.dat$", f)
        m === nothing && continue
        sp = m.captures[1]
        r = read_dat(joinpath(dir, f))
        Int.(first.(r)) == steps || error("$f: steps differ from fields_energy.dat")
        haskey(rest, sp) || error("$f: species $sp is not a mobile species of the case")
        e = [x[3] / J - rest[sp] for x in r]
        cols["Ukin_" * sp] = e
        cols["Ukin_threshold_" * sp] = [x[2] / J for x in r]
        ukin .+= e
        uthr .+= cols["Ukin_threshold_" * sp]
    end
    cols["Ukin"] = ukin
    cols["Ukin_threshold"] = uthr
    cols["Utot"] = ukin .+ cols["Uelm"]
    cols["time"] = steps .* u["dt"]
    return (; time = cols["time"], columns = cols)
end

# Append the periodic copy of the first point along each axis.
periodic_extend(a::AbstractMatrix) = (b = vcat(a, a[1:1, :]); hcat(b, b[:, 1:1]))

"""
Positions of the N + 1 points of each axis of a PIConGPU mesh record component: offset + (i +
position) spacing in c/ω₀, with spacing and offset read in axisLabels order (y first) and
position x first (see the header).
"""
function positions(mesh, comp, n, u)
    a = attrs(mesh)
    labels = collect(String, a["axisLabels"])
    spacing = Float64.(a["gridSpacing"])
    offset = Float64.(a["gridGlobalOffset"])
    unit = Float64(a["gridUnitSI"]) / u["length"]
    pos = Float64.(attrs(comp)["position"])
    ax(lbl) = findfirst(==(lbl), labels)
    return ntuple(d -> collect((offset[ax(("x", "y")[d])] .+ ((0:n[d]) .+ pos[d]) .* spacing[ax(("x", "y")[d])]) .* unit), 2)
end

const COMPONENTS = Dict("Ex" => ("E", "x"), "Ey" => ("E", "y"), "Ez" => ("E", "z"),
                        "Bx" => ("B", "x"), "By" => ("B", "y"), "Bz" => ("B", "z"))

function read_picongpu_fields(dir; names = nothing, yaverage = false, file = joinpath("openPMD", "simData.h5"))
    u = picongpu_units(dir)
    h5open(joinpath(dir, file), "r") do f
        data = f["data"]
        keys_ = sort(collect(keys(data)); by = k -> parse(Int, k))
        iterations = parse.(Int, keys_)
        mp = rstrip(haskey(attrs(f), "meshesPath") ? attrs(f)["meshesPath"] : "meshes/", '/')
        g0 = data[keys_[1]][mp]
        species = sort([m.captures[1] for m in (match(r"^(.+)_all_chargeDensity$", k) for k in keys(g0)) if m !== nothing])
        available = vcat(sort(collect(keys(COMPONENTS))), "Rho", "Rho_" .* species)
        wanted = names === nothing ? available : collect(String, names)
        for n in wanted
            n in available || error("$n is not in $(joinpath(dir, file)) (available: $available)")
        end
        record(g, n) = n in keys(COMPONENTS) ? (g[COMPONENTS[n][1]], g[COMPONENTS[n][1]][COMPONENTS[n][2]]) :
            (g[replace(n, "Rho_" => "") * "_all_chargeDensity"], g[replace(n, "Rho_" => "") * "_all_chargeDensity"])
        scale(n) = startswith(n, "E") ? 1 / u["efield"] : startswith(n, "B") ? 1 / u["bfield"] : 1 / u["charge_density"]
        function value(g, n)
            if n == "Rho"
                return sum(value(g, "Rho_" * s) for s in species)
            end
            mesh, comp = record(g, n)
            return Float64.(read(comp)) .* (Float64(attrs(comp)["unitSI"]) * scale(n))
        end
        first_ = Dict(n => value(g0, n) for n in wanted)
        N = size(first_[first(wanted)])
        x = Dict{String,Any}()
        for n in wanted
            src = n == "Rho" ? "Rho_" * first(species) : n
            mesh, comp = record(g0, src)
            x[n] = positions(mesh, comp, N, u)
        end
        nt = length(keys_)
        fields = Dict(n => Array{Float64,3}(undef, N[1] + 1, N[2] + 1, nt) for n in wanted)
        time = Vector{Float64}(undef, nt)
        for (j, k) in enumerate(keys_)
            g = data[k]
            a = attrs(g)
            t_attr = Float64(a["time"]) * Float64(a["timeUnitSI"]) / u["time"]
            time[j] = iterations[j] * u["dt"]
            isapprox(t_attr, time[j]; rtol = 1e-5, atol = 1e-6 * u["dt"]) ||
                error("iteration $k: time attribute $t_attr is not $(iterations[j]) × dt")
            m = g[mp]
            for n in wanted
                fields[n][:, :, j] .= periodic_extend(j == 1 ? first_[n] : value(m, n))
            end
        end
        if yaverage
            return (; iterations, time, x = Dict(n => x[n][1] for n in wanted),
                      fields = Dict(n => PIConGPURead.yaverage(fields[n]) for n in wanted))
        end
        return (; iterations, time, x, fields)
    end
end

"""
    yaverage(f) -> Matrix

The y-average of a 2D field in the reader's layout (points_x × points_y × outputs, the last
y point the periodic copy of the first, which is left out), as points_x × outputs: the 1D
field of a quasi-1D run.
"""
yaverage(f::AbstractArray{<:Real,3}) = dropdims(mean(view(f, :, 1:size(f, 2)-1, :); dims = 2); dims = 2)

end
