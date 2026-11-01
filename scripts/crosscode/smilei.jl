# Smilei namelist generator for the cases in scripts/crosscode/cases/ (Smilei v5.1): schema 1
# (1D, `1Dcartesian`) and schema 2 (2D periodic, `2Dcartesian`).
#
#   julia +1.13 --project=scripts scripts/crosscode/smilei.jl prepare <case> [tag]
#
# writes scripts/runs/<tag>/ with smilei.py (the namelist), case.toml (a copy of the case),
# run.sh (what the container executes) and manifest.toml. The default tag is
# "smilei-<case>-<first 8 hex of the case hash>".

module SmileiDeck

using TOML

include(joinpath(@__DIR__, "manifest.jl"))
using .RunManifest

export load_case, case_view, smilei_parameters, namelist, prepare_run

const CROSSCODE = @__DIR__
const SCRIPTS = dirname(CROSSCODE)
const ROOT = dirname(SCRIPTS)

"""
    load_case(name_or_path) -> Dict

Read a case file: `cases/<name>.toml`, or a path. The `schema` key selects the version: 1 (a 1D
grid, scalar `[grid]` keys) or 2 (a 2D periodic grid, per-axis keys); see
`scripts/crosscode/README.md`. Checks the keys the generator reads and returns the parsed file
unchanged, with the key `_path` added.
"""
function load_case(x::AbstractString)
    path = isfile(x) ? x : joinpath(CROSSCODE, "cases", x * ".toml")
    case = TOML.parsefile(path)
    schema = get(case, "schema", 0)
    for k in ("name", "grid", "time", "numerics", "species", "diagnostics")
        haskey(case, k) || error("$path: missing [$k]")
    end
    if schema == 1
        case["grid"]["boundaries"] == "periodic" || error("$path: only periodic boundaries")
    elseif schema == 2
        validate_schema2(case, path)
    else
        error("$path: schema must be 1 or 2")
    end
    case["_path"] = path
    return case
end

"The wavevector 2π (m₁/L₁, m₂/L₂, …) of an integer mode in a periodic box."
wavevector(mode, lengths) = Tuple(2π .* mode ./ lengths)

isint(x) = x isa Integer
intvec(x, n) = x isa AbstractVector && length(x) == n && all(isint, x)
realvec(x, n) = x isa AbstractVector && length(x) == n && all(v -> v isa Real, x)

function validate_schema2(case, path)
    g = case["grid"]
    N = 2
    realvec(get(g, "length", nothing), N) && all(>(0), g["length"]) ||
        error("$path: [grid] length must be $N positive lengths, one per axis")
    intvec(get(g, "cells", nothing), N) && all(>(0), g["cells"]) ||
        error("$path: [grid] cells must be $N positive integers, one per axis")
    b = get(g, "boundaries", nothing)
    b isa AbstractVector && length(b) == N ||
        error("$path: [grid] boundaries must be a list with one entry per axis")
    all(==("periodic"), b) || error("$path: only periodic boundaries (got $b)")
    lengths = Tuple(Float64.(g["length"]))
    wv = get(case, "wavevector", nothing)
    for s in case["species"]
        name = get(s, "name", "?")
        intvec(get(s, "particles_per_cell", nothing), N) && all(>(0), s["particles_per_cell"]) ||
            error("$path: species $name: particles_per_cell must be $N positive integers, one per axis")
        realvec(get(s, "drift", [0.0, 0.0, 0.0]), 3) ||
            error("$path: species $name: drift must be [v_x, v_y, v_z] in schema 2")
        sum(abs2, get(s, "drift", [0.0, 0.0, 0.0])) < 1 ||
            error("$path: species $name: the drift speed must be below c")
        haskey(s, "transverse_drift") &&
            error("$path: species $name: transverse_drift is schema 1; schema 2 takes drift = [v_x, v_y, v_z]")
        pert = get(s, "perturbation", nothing)
        pert === nothing && continue
        intvec(get(pert, "mode", nothing), N) ||
            error("$path: species $name: the perturbation mode must be $N integers [m_x, m_y]")
        any(!iszero, pert["mode"]) || error("$path: species $name: the perturbation mode is zero")
        wv === nothing && error("$path: a perturbed species needs the [wavevector] it is meant to excite")
        Tuple(pert["mode"]) == Tuple(wv["mode"]) ||
            error("$path: species $name: perturbation mode $(pert["mode"]) is not the stated wavevector mode $(wv["mode"])")
    end
    if wv !== nothing
        intvec(get(wv, "mode", nothing), N) || error("$path: [wavevector] mode must be $N integers")
        # The stated wavevector must be a lattice vector of the box: k_d L_d / 2π an integer
        # (the mode) on every axis, so that the mode is periodic in the box.
        k, θ = Float64(wv["magnitude"]), Float64(wv["angle"])
        kvec = (k * cos(θ), k * sin(θ))
        m = kvec .* lengths ./ 2π
        all(abs.(m .- round.(m)) .< 1e-9) && Tuple(round.(Int, m)) == Tuple(wv["mode"]) ||
            error("$path: the stated wavevector (|k| = $k, angle = $θ) is not periodic in the box " *
                  "$lengths: k L/2π = $m, mode $(wv["mode"])")
    end
    return nothing
end

"""
    case_view(case) -> NamedTuple

The per-axis view of a case of either schema: `ndims`, `lengths`, `cells` (tuples, one entry
per axis) and per species `name`, `density`, `particles_per_cell` (per axis), `drift`
(v_x, v_y, v_z), `mode` (per axis, or `nothing` without a perturbation) and `amplitude` (the
relative amplitude). A schema-1 file is a 1D grid: its scalar `drift` is v_x and its
`transverse_drift = [v_y, v_z]` the other two components.
"""
function case_view(case)
    g = case["grid"]
    two = case["schema"] == 2
    lengths = two ? Tuple(Float64.(g["length"])) : (Float64(g["length"]),)
    cells = two ? Tuple(Int.(g["cells"])) : (Int(g["cells"]),)
    species = map(case["species"]) do s
        pert = get(s, "perturbation", nothing)
        drift = two ? Tuple(Float64.(get(s, "drift", [0.0, 0.0, 0.0]))) :
            (Float64(get(s, "drift", 0.0)), Float64.(get(s, "transverse_drift", [0.0, 0.0]))...)
        (; name = s["name"], density = Float64(s["density"]),
           particles_per_cell = two ? Tuple(Int.(s["particles_per_cell"])) : (Int(s["particles_per_cell"]),),
           drift,
           mode = pert === nothing ? nothing : two ? Tuple(Int.(pert["mode"])) : (Int(pert["mode"]),),
           amplitude = pert === nothing ? 0.0 : Float64(pert["amplitude"]))
    end
    return (; ndims = length(cells), lengths, cells, species)
end

"""
    smilei_parameters(case) -> NamedTuple

The unit conversion from a case in the units of A1 to Smilei's normalized units: schema 1 by
`smilei_parameters_1d`, schema 2 by `smilei_parameters_2d`.
"""
smilei_parameters(case) = case["schema"] == 2 ? smilei_parameters_2d(case) : smilei_parameters_1d(case)

"""
    smilei_parameters_1d(case) -> NamedTuple

The unit conversion from a schema-1 case in the units of A1 (time 1/ω₀, length c/ω₀, velocity c,
density n_c, temperature m_e c²) to Smilei's normalized units with the reference frequency
ω_r = ω₀ (Smilei normalizes time by 1/ω_r, length by c/ω_r, density by n_r = ε₀ m_e ω_r²/e²,
which is n_c for ω_r = ω₀, temperature by m_e c², velocity by c). This is the only place
where a case quantity becomes a Smilei quantity; `namelist` only formats its result.

- cell length Δx = length / cells, number of cells as given;
- time step as given; number of steps round(t_end / dt), so the simulated time is
  steps × dt (Smilei truncates simulation_time/timestep, so the step count is passed instead);
- per species: number density in n_r equal to the case density; the perturbation
  n(x) (1 + α cos(2π m x / L)) becomes Smilei's `cosine` profile with absolute amplitude
  α·n and `xnumber = m` over `xlength = L`; temperature T/(m_e c²) = (m/m_e) v_th² for
  v_th = √(T/m); mean velocity the drift; immobile species frozen for longer than the run;
- patches: the largest power of 2 that divides the cells and leaves at least 8 cells per
  patch (Smilei needs a power of 2 and more than 2·oversize + 1 = 5 cells per patch at
  interpolation order 2), capped at 32;
- diagnostics cadence in steps, as given.
"""
function smilei_parameters_1d(case)
    g, t, num = case["grid"], case["time"], case["numerics"]
    L = Float64(g["length"])
    N = Int(g["cells"])
    dx = L / N
    dt = Float64(t["dt"])
    steps = round(Int, Float64(t["t_end"]) / dt)
    order = Int(num["shape_order"])
    order == 2 || error("Smilei supports interpolation orders 2 and 4; the cases use 2 (A2)")
    get(num, "filter_passes", 0) == 0 || error("current filtering is not generated yet")
    get(num, "precision", "Float64") == "Float64" || error("Smilei runs in double precision")
    dt < dx || @warn "dt ≥ Δx violates the CFL condition of Smilei's 1D Yee solver"
    patches = 1
    while patches < 32 && N % (2patches) == 0 && N ÷ (2patches) >= 8
        patches *= 2
    end
    species = map(case["species"]) do s
        n = Float64(s["density"])
        m = Float64(s["mass"])
        vth = Float64(get(s, "thermal_velocity", 0.0))
        pert = get(s, "perturbation", nothing)
        velocity = s["velocity"]
        velocity in ("maxwellian", "cold") || error("unknown velocity loading $velocity")
        velocity == "maxwellian" && vth <= 0 && error("$(s["name"]): maxwellian needs thermal_velocity > 0")
        any(!iszero, get(s, "transverse_drift", (0, 0))) &&
            error("$(s["name"]): transverse_drift is not passed to Smilei yet (cross-code stage 2)")
        (; name = s["name"],
           charge = Float64(s["charge"]),
           mass = m,
           density = n,
           cosine = pert === nothing ? nothing :
               (; base = n, xamplitude = Float64(pert["amplitude"]) * n,
                  xlength = L, xnumber = Int(pert["mode"])),
           particles_per_cell = Int(s["particles_per_cell"]),
           position = s["position"],
           momentum = velocity == "maxwellian" ? "maxwell-juettner" : "cold",
           temperature = m * vth^2,
           mean_velocity = Float64(get(s, "drift", 0.0)),
           time_frozen = s["mobile"] ? 0.0 : 2 * steps * dt)
    end
    d = case["diagnostics"]
    return (; geometry = "1Dcartesian", interpolation_order = order,
              cell_length = dx, number_of_cells = N, grid_length = N * dx,
              timestep = dt, number_of_timesteps = steps, simulation_time = steps * dt,
              number_of_patches = patches,
              random_seed = Int(num["random_seed"]),
              species,
              scalars_every = Int(d["scalars_every"]), fields_every = Int(d["fields_every"]),
              fields = collect(String, d["fields"]))
end

"""
    patches_2d(cells; min_cells = 8, max_patches = 16) -> (px, py)

Patches per axis for a 2D grid: powers of 2 that divide the cells of their axis and leave at
least `min_cells` cells per patch (Smilei, Params.cpp: a power of 2 per axis that divides the
cells, and more than 2·oversize + 1 = 5 cells per patch at interpolation order 2), grown one
axis at a time, the axis with more cells per patch first (x on a tie), up to `max_patches` in
total; one OpenMP thread per patch.
"""
function patches_2d(cells; min_cells = 8, max_patches = 16)
    p = [1, 1]
    while prod(p) < max_patches
        grow = [d for d in 1:2 if cells[d] % (2p[d]) == 0 && cells[d] ÷ (2p[d]) >= min_cells]
        isempty(grow) && break
        d = argmax(d -> (cells[d] ÷ p[d], -d), grow)
        p[d] *= 2
    end
    return (p[1], p[2])
end

"""
    smilei_parameters_2d(case) -> NamedTuple

The unit conversion of a schema-2 case (2D periodic, `2Dcartesian`), with the same units and
rules as `smilei_parameters_1d`, per axis:

- cell lengths (Δx, Δy) = lengths ./ cells, numbers of cells as given; Smilei's 2D Yee
  solver is stable for Δt < 1/√(1/Δx² + 1/Δy²) (Params.cpp, `dtCFL`);
- per species: the perturbation n(x, y) = n (1 + α cos(2π (m_x x/L_x + m_y y/L_y))) becomes
  `profile = (; base = n, amplitude = α·n, mode, lengths)`, written as a Python function of
  (x, y) (Smilei's `cosine` profile is separable and cannot express an oblique mode);
  particles per cell per axis become `regular_number` with `particles_per_cell` their
  product; the drift [v_x, v_y, v_z] becomes `mean_velocity`;
- patches per axis by `patches_2d`, one thread per patch (`threads`).
"""
function smilei_parameters_2d(case)
    g, t, num = case["grid"], case["time"], case["numerics"]
    L = Tuple(Float64.(g["length"]))
    N = Tuple(Int.(g["cells"]))
    dx = L ./ N
    dt = Float64(t["dt"])
    steps = round(Int, Float64(t["t_end"]) / dt)
    order = Int(num["shape_order"])
    order == 2 || error("Smilei supports interpolation orders 2 and 4; the cases use 2 (A2)")
    get(num, "filter_passes", 0) == 0 || error("current filtering is not generated yet")
    get(num, "precision", "Float64") == "Float64" || error("Smilei runs in double precision")
    dt < 1 / sqrt(sum(abs2, 1 ./ dx)) || @warn "dt violates the CFL condition of Smilei's 2D Yee solver"
    patches = patches_2d(N)
    species = map(case["species"]) do s
        n = Float64(s["density"])
        m = Float64(s["mass"])
        vth = Float64(get(s, "thermal_velocity", 0.0))
        pert = get(s, "perturbation", nothing)
        velocity = s["velocity"]
        velocity in ("maxwellian", "cold") || error("unknown velocity loading $velocity")
        velocity == "maxwellian" && vth <= 0 && error("$(s["name"]): maxwellian needs thermal_velocity > 0")
        s["position"] == "regular" || error("$(s["name"]): only regular positions are generated in 2D")
        ppc = Tuple(Int.(s["particles_per_cell"]))
        (; name = s["name"],
           charge = Float64(s["charge"]),
           mass = m,
           density = n,
           profile = pert === nothing ? nothing :
               (; base = n, amplitude = Float64(pert["amplitude"]) * n,
                  mode = Tuple(Int.(pert["mode"])), lengths = L),
           particles_per_cell = prod(ppc),
           regular_number = ppc,
           position = s["position"],
           momentum = velocity == "maxwellian" ? "maxwell-juettner" : "cold",
           temperature = m * vth^2,
           mean_velocity = Tuple(Float64.(get(s, "drift", [0.0, 0.0, 0.0]))),
           time_frozen = s["mobile"] ? 0.0 : 2 * steps * dt)
    end
    d = case["diagnostics"]
    return (; geometry = "2Dcartesian", interpolation_order = order,
              cell_length = dx, number_of_cells = N, grid_length = N .* dx,
              timestep = dt, number_of_timesteps = steps, simulation_time = steps * dt,
              number_of_patches = patches, threads = prod(patches),
              random_seed = Int(num["random_seed"]),
              species,
              scalars_every = Int(d["scalars_every"]), fields_every = Int(d["fields_every"]),
              fields = collect(String, d["fields"]))
end

# Julia's shortest round-trip form; Python's float() parses it to the same double.
pyfloat(x::Real) = (isfinite(x) || error("non-finite value in namelist"); string(Float64(x)))
pyfloat(x::Integer) = string(x)

"""
    namelist(case) -> String

A complete Smilei namelist for the case, from `smilei_parameters`: `1Dcartesian` for schema 1,
`2Dcartesian` for schema 2.
"""
namelist(case) = case["schema"] == 2 ? namelist_2d(case) : namelist_1d(case)

function namelist_1d(case)
    p = smilei_parameters_1d(case)
    io = IOBuffer()
    println(io, "# Smilei namelist generated by scripts/crosscode/smilei.jl from case \"", case["name"], "\".")
    println(io, "# Do not edit; regenerate from the case file. Units: Smilei normalized, reference frequency = ω₀ (A1).")
    println(io)
    println(io, """
    Main(
        geometry = "$(p.geometry)",
        interpolation_order = $(p.interpolation_order),
        cell_length = [$(pyfloat(p.cell_length))],
        number_of_cells = [$(p.number_of_cells)],
        timestep = $(pyfloat(p.timestep)),
        number_of_timesteps = $(p.number_of_timesteps),
        number_of_patches = [$(p.number_of_patches)],
        EM_boundary_conditions = [["periodic"]],
        solve_poisson = True,
        random_seed = $(p.random_seed),
        print_every = $(max(1, p.number_of_timesteps ÷ 20)),
    )
    """)
    for s in p.species
        density = s.cosine === nothing ? pyfloat(s.density) :
            "cosine($(pyfloat(s.cosine.base)), xamplitude=$(pyfloat(s.cosine.xamplitude)), " *
            "xvacuum=0., xlength=$(pyfloat(s.cosine.xlength)), xphi=0., xnumber=$(s.cosine.xnumber))"
        temperature = s.momentum == "cold" ? "" : "\n    temperature = [$(pyfloat(s.temperature))],"
        println(io, """
        Species(
            name = "$(s.name)",
            position_initialization = "$(s.position)",
            momentum_initialization = "$(s.momentum)",$temperature
            particles_per_cell = $(s.particles_per_cell),
            mass = $(pyfloat(s.mass)),
            charge = $(pyfloat(s.charge)),
            number_density = $density,
            mean_velocity = [$(pyfloat(s.mean_velocity)), 0., 0.],
            time_frozen = $(pyfloat(s.time_frozen)),
            boundary_conditions = [["periodic"]],
        )
        """)
    end
    println(io, """
    DiagScalar(
        every = $(p.scalars_every),
        precision = 10,
    )

    DiagFields(
        every = $(p.fields_every),
        fields = [$(join(repr.(p.fields), ", "))],
        datatype = "double",
    )""")
    return String(take!(io))
end

pyvec(v) = "[" * join(pyfloat.(v), ", ") * "]"

# The Python name of a species' density profile.
profile_name(s) = "density_" * replace(s.name, r"[^A-Za-z0-9_]" => "_")

"""
The density profile n(x, y) = base + amplitude cos(2π (m_x x/L_x + m_y y/L_y)) as a Python
function. It uses numpy, so that Smilei evaluates it on whole arrays (Profile.cpp tries numpy
arguments first; a function of floats also works, one point at a time).
"""
function profile_def(s)
    pr = s.profile
    return """
    def $(profile_name(s))(x, y):
        return $(pyfloat(pr.base)) + $(pyfloat(pr.amplitude)) * np.cos(2.0 * np.pi * ($(pr.mode[1]) * x / $(pyfloat(pr.lengths[1])) + $(pr.mode[2]) * y / $(pyfloat(pr.lengths[2]))))
    """
end

function namelist_2d(case)
    p = smilei_parameters_2d(case)
    io = IOBuffer()
    println(io, "# Smilei namelist generated by scripts/crosscode/smilei.jl from case \"", case["name"], "\".")
    println(io, "# Do not edit; regenerate from the case file. Units: Smilei normalized, reference frequency = ω₀ (A1).")
    println(io)
    println(io, "import numpy as np")
    println(io)
    println(io, """
    Main(
        geometry = "$(p.geometry)",
        interpolation_order = $(p.interpolation_order),
        cell_length = $(pyvec(p.cell_length)),
        number_of_cells = $(pyvec(p.number_of_cells)),
        timestep = $(pyfloat(p.timestep)),
        number_of_timesteps = $(p.number_of_timesteps),
        number_of_patches = $(pyvec(p.number_of_patches)),
        EM_boundary_conditions = [["periodic"], ["periodic"]],
        solve_poisson = True,
        random_seed = $(p.random_seed),
        print_every = $(max(1, p.number_of_timesteps ÷ 20)),
    )
    """)
    for s in p.species
        s.profile === nothing || println(io, profile_def(s))
        density = s.profile === nothing ? pyfloat(s.density) : profile_name(s)
        temperature = s.momentum == "cold" ? "" : "\n    temperature = [$(pyfloat(s.temperature))],"
        println(io, """
        Species(
            name = "$(s.name)",
            position_initialization = "$(s.position)",
            regular_number = $(pyvec(s.regular_number)),
            momentum_initialization = "$(s.momentum)",$temperature
            particles_per_cell = $(s.particles_per_cell),
            mass = $(pyfloat(s.mass)),
            charge = $(pyfloat(s.charge)),
            number_density = $density,
            mean_velocity = $(pyvec(s.mean_velocity)),
            time_frozen = $(pyfloat(s.time_frozen)),
            boundary_conditions = [["periodic"], ["periodic"]],
        )
        """)
    end
    println(io, """
    DiagScalar(
        every = $(p.scalars_every),
        precision = 10,
    )

    DiagFields(
        every = $(p.fields_every),
        fields = [$(join(repr.(p.fields), ", "))],
        datatype = "double",
    )""")
    return String(take!(io))
end

"""
The script the container runs (runner: `bash /run-dir/run.sh`; `run-smilei.sh` mounts the same
paths). Smilei writes its output into the working directory; the log goes to smilei.log.
"""
function run_script(p)
    threads = hasproperty(p, :threads) ? p.threads : p.number_of_patches
    return """
    #!/usr/bin/env bash
    # Generated by scripts/crosscode/smilei.jl. Runs inside the Smilei container with the run
    # directory at /run-dir (PM_RUN_DIR) and the worktree read-only at /worktree.
    set -euo pipefail
    cd "\${PM_RUN_DIR:-/run-dir}"
    export OMP_NUM_THREADS=$threads
    export OMP_SCHEDULE=dynamic
    {
      echo "# smilei binary: \$(command -v smilei)"
      echo "# smilei version: \$(cat /opt/smilei/VERSION 2>/dev/null || echo unknown)"
      echo "# threads: \$OMP_NUM_THREADS, cpus: \$(nproc)"
      echo "# cpu model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
    } > container.txt
    echo "# start: \$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> container.txt
    smilei smilei.py > smilei.log 2>&1
    echo "# end: \$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> container.txt
    """
end

"""
    prepare_run(case_name; tag) -> dir

Write the run directory under scripts/runs/<tag>/ and its manifest.
"""
function prepare_run(name; tag = nothing, runs = joinpath(SCRIPTS, "runs"))
    case = load_case(name)
    h = file_hash(case["_path"])
    tag = something(tag, "smilei-$(case["name"])-$(h[1:8])")
    dir = joinpath(runs, tag)
    mkpath(dir)
    cp(case["_path"], joinpath(dir, "case.toml"); force = true)
    write(joinpath(dir, "smilei.py"), namelist(case))
    write(joinpath(dir, "run.sh"), run_script(smilei_parameters(case)))
    chmod(joinpath(dir, "run.sh"), 0o755)
    write_manifest(dir; root = ROOT, inputs = ["case.toml", "smilei.py", "run.sh"],
        extra = Dict{String,Any}(
            "run" => tag, "code" => "smilei", "case" => case["name"],
            "case_source" => relpath(case["_path"], ROOT),
            "generator" => "scripts/crosscode/smilei.jl",
            "generator_hash" => "sha256:" * file_hash(@__FILE__),
            "image" => Dict{String,Any}(
                "recipe" => "scripts/crosscode/containers/smilei.Containerfile",
                "recipe_hash" => "sha256:" * file_hash(joinpath(CROSSCODE, "containers", "smilei.Containerfile")),
                "id" => "pending: written by the analyze step from the runner log")))
    return dir
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .SmileiDeck
    cmd = get(ARGS, 1, "")
    if cmd == "prepare" && length(ARGS) >= 2
        dir = prepare_run(ARGS[2]; tag = get(ARGS, 3, nothing))
        println(dir)
    elseif cmd == "namelist" && length(ARGS) >= 2
        print(namelist(load_case(ARGS[2])))
    else
        println(stderr, "usage: smilei.jl prepare <case> [tag] | namelist <case>")
        exit(2)
    end
end
