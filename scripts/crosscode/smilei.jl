# Smilei namelist generator for the 1D cases in scripts/crosscode/cases/ (Smilei v5.1).
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

export load_case, smilei_parameters, namelist, prepare_run

const CROSSCODE = @__DIR__
const SCRIPTS = dirname(CROSSCODE)
const ROOT = dirname(SCRIPTS)

"""
    load_case(name_or_path) -> Dict

Read a case file: `cases/<name>.toml`, or a path. Checks the schema version and the keys the
generator reads.
"""
function load_case(x::AbstractString)
    path = isfile(x) ? x : joinpath(CROSSCODE, "cases", x * ".toml")
    case = TOML.parsefile(path)
    get(case, "schema", 0) == 1 || error("$path: schema must be 1")
    for k in ("name", "grid", "time", "numerics", "species", "diagnostics")
        haskey(case, k) || error("$path: missing [$k]")
    end
    case["grid"]["boundaries"] == "periodic" || error("$path: only periodic boundaries")
    case["_path"] = path
    return case
end

"""
    smilei_parameters(case) -> NamedTuple

The unit conversion from a case in the units of A1 (time 1/ω₀, length c/ω₀, velocity c,
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
function smilei_parameters(case)
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

# Julia's shortest round-trip form; Python's float() parses it to the same double.
pyfloat(x::Real) = (isfinite(x) || error("non-finite value in namelist"); string(Float64(x)))
pyfloat(x::Integer) = string(x)

"""
    namelist(case) -> String

A complete Smilei namelist for the case, from `smilei_parameters`.
"""
function namelist(case)
    p = smilei_parameters(case)
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

"""
The script the container runs (runner: `bash /run-dir/run.sh`; `run-smilei.sh` mounts the same
paths). Smilei writes its output into the working directory; the log goes to smilei.log.
"""
function run_script(p)
    threads = p.number_of_patches
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
