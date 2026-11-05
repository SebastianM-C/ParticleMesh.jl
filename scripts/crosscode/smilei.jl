# Smilei namelist generator for the cases in scripts/crosscode/cases/ (Smilei v5.1): schema 1
# (1D, `1Dcartesian`) and schema 2 (2D periodic, `2Dcartesian`).
#
#   julia +1.13 --project=scripts scripts/crosscode/smilei.jl prepare <case> [tag] [--budget <s>] [--gpu] [--no-fastmath] [--patches <px>x<py>] [--load-check] [--first-seeds <n>]
#
# writes scripts/runs/<tag>/ with namelist.py (the namelist), case.toml (a copy of the case),
# run.sh (what the container executes) and manifest.toml. A case with a `seeds` list gets one
# subdirectory seed-<seed>/ per seed, each with its namelist and a copy of the case, and one
# run.sh that runs them in turn. The default tag is "smilei-<case>-<first 8 hex of the case hash>".
# The input is not called smilei.py because Smilei writes its expanded namelist to smilei.py in
# the working directory (#256).
#
# --gpu (schema 2 only, WP-030) prepares a run of the smilei-gpu image: `gpu_computing = True`,
# one patch by default (Smilei's GPU guideline, Understand/GPU_offloading.rst), the binary with
# the guide's fastmath flags unless --no-fastmath. --patches sets the patch layout of either
# build. --load-check prepares a one-step run of the case's first seed that dumps the t = 0 load
# of a corner of the box (DiagTrackParticles) and the fields at t = 0, to compare two builds'
# loads (scripts/crosscode/loadcheck.jl); its tag is "<code>-loadcheck-<case>-<layout>-<hash>".

module SmileiDeck

using TOML
using SHA: sha256

include(joinpath(@__DIR__, "manifest.jl"))
using .RunManifest

export load_case, case_view, case_seeds, smilei_parameters, namelist, prepare_run, cli_options

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
    if haskey(case["numerics"], "seeds")
        seeds = case["numerics"]["seeds"]
        (seeds isa AbstractVector && !isempty(seeds) && all(s -> s isa Integer && 0 <= s < 2^31, seeds)) ||
            error("$path: seeds must be a non-empty list of integers in [0, 2^31)")
        allunique(seeds) || error("$path: seeds must be distinct")
        haskey(case["numerics"], "random_seed") && error("$path: give either seeds or random_seed")
    else
        haskey(case["numerics"], "random_seed") || error("$path: missing random_seed")
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
    case_seeds(case) -> Vector{Int}

The seeds of a case: the `[numerics]` list `seeds` when present (one run per seed), otherwise
the single `random_seed` (schema 1 without the list).
"""
case_seeds(case) = collect(Int, get(() -> [case["numerics"]["random_seed"]], case["numerics"], "seeds"))

"""
    smilei_parameters(case) -> NamedTuple

The unit conversion from a case in the units of A1 to Smilei's normalized units: schema 1 by
`smilei_parameters_1d`, schema 2 by `smilei_parameters_2d`.
"""
function smilei_parameters(case; gpu = false, patches = nothing)
    case["schema"] == 2 && return smilei_parameters_2d(case; gpu, patches)
    (gpu || patches !== nothing) &&
        error("GPU runs and patch layouts are generated for schema-2 (2D) cases only (Smilei's GPU port covers 2D and 3D Cartesian)")
    return smilei_parameters_1d(case)
end

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
- the binomial current filter's pass count (`filter_passes`), emitted as `CurrentFilter`;
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
    passes = filter_passes(num; gpu = false)
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
              number_of_patches = patches, filter_passes = passes,
              random_seed = first(case_seeds(case)),
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
    check_patches(p, cells) -> p

A patch layout Smilei accepts at interpolation order 2 (Params.cpp): per axis a power of 2 that
divides the cells, with more than 2·oversize + 1 = 5 cells per patch.
"""
function check_patches(p, cells)
    length(p) == length(cells) || error("patches $p: one count per axis")
    for (n, c) in zip(p, cells)
        n >= 1 && ispow2(n) && c % n == 0 && c ÷ n > 5 ||
            error("patches $p for cells $cells: each must be a power of 2 dividing its axis with more than 5 cells per patch")
    end
    return p
end

"""
    smilei_parameters_2d(case; gpu = false, patches = nothing) -> NamedTuple

The unit conversion of a schema-2 case (2D periodic, `2Dcartesian`), with the same units and
rules as `smilei_parameters_1d`, per axis:

- cell lengths (Δx, Δy) = lengths ./ cells, numbers of cells as given; Smilei's 2D Yee
  solver is stable for Δt < 1/√(1/Δx² + 1/Δy²) (Params.cpp, `dtCFL`);
- per species: the perturbation n(x, y) = n (1 + α cos(2π (m_x x/L_x + m_y y/L_y))) becomes
  `profile = (; base = n, amplitude = α·n, mode, lengths)`, written as a Python function of
  (x, y) (Smilei's `cosine` profile is separable and cannot express an oblique mode);
  particles per cell per axis become `regular_number` with `particles_per_cell` their
  product; the drift [v_x, v_y, v_z] becomes `mean_velocity`;
- patches per axis by `patches_2d`, one thread per patch (`threads`);
- `gpu = true` (the smilei-gpu image, WP-030): `gpu_computing = True`, one MPI rank without
  OpenMP (`make config="gpu_nvidia"` implies `noopenmp`), so `threads = 1`, and one patch
  unless `patches` says otherwise (Smilei's guideline: one patch per GPU);
- `patches = (px, py)` overrides the layout of either build, checked as `patches_2d`'s rule;
- the filter's pass count as in 1D (`filter_passes`), refused with `gpu = true`.
"""
function smilei_parameters_2d(case; gpu = false, patches = nothing)
    g, t, num = case["grid"], case["time"], case["numerics"]
    L = Tuple(Float64.(g["length"]))
    N = Tuple(Int.(g["cells"]))
    dx = L ./ N
    dt = Float64(t["dt"])
    steps = round(Int, Float64(t["t_end"]) / dt)
    order = Int(num["shape_order"])
    order == 2 || error("Smilei supports interpolation orders 2 and 4; the cases use 2 (A2)")
    passes = filter_passes(num; gpu)
    get(num, "precision", "Float64") == "Float64" || error("Smilei runs in double precision")
    dt < 1 / sqrt(sum(abs2, 1 ./ dx)) || @warn "dt violates the CFL condition of Smilei's 2D Yee solver"
    patches = patches === nothing ? (gpu ? (1, 1) : patches_2d(N)) : check_patches(Tuple(Int.(patches)), N)
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
              number_of_patches = patches, threads = gpu ? 1 : prod(patches), gpu_computing = gpu,
              filter_passes = passes,
              random_seed = first(case_seeds(case)),
              species,
              scalars_every = Int(d["scalars_every"]), fields_every = Int(d["fields_every"]),
              fields = collect(String, d["fields"]))
end

"""
    filter_passes(numerics; gpu) -> Int

The case's binomial pass count (`filter_passes`, default 0), a non-negative integer. Smilei's
`CurrentFilter(model = "binomial", passes = [p])` is the same operator as ParticleMesh's
`BinomialFilter(p)` (v5.1 source: `Params.cpp` copies a one-element `passes` to every axis;
`ElectroMagn1D/2D::binomialCurrentFilter` sweeps J[i] ← (J[i] + J[i+1])/2 up and
J[i] ← (J[i] + J[i−1])/2 down, which is (1, 2, 1)/4, along x then y, on Jx, Jy and Jz on their own
extents, once per pass, with a guard exchange after each pass; `VectorPatch::solveMaxwell` runs
it after the current is summed and before the Yee update). The filter has no device code, so a
GPU run (`gpu = true`) with passes is refused.
"""
function filter_passes(num; gpu)
    p = get(num, "filter_passes", 0)
    p isa Integer && p >= 0 || error("filter_passes must be a non-negative integer, got $(repr(p))")
    gpu && p > 0 && error("Smilei's binomial current filter has no GPU code path (v5.1); run a filtered case on the CPU build")
    return Int(p)
end

"""
    current_filter_block(p) -> String

The namelist's `CurrentFilter` block for `p.filter_passes` passes along every axis; empty for
zero passes, so that an unfiltered namelist stays as it was.
"""
current_filter_block(p) = p.filter_passes == 0 ? "" : """
    CurrentFilter(
        model = "binomial",
        passes = [$(p.filter_passes)],
    )

    """

# Julia's shortest round-trip form; Python's float() parses it to the same double.
pyfloat(x::Real) = (isfinite(x) || error("non-finite value in namelist"); string(Float64(x)))
pyfloat(x::Integer) = string(x)

"""
    namelist(case; seed = first(case_seeds(case)), gpu = false, patches = nothing, load_check = false) -> String

A complete Smilei namelist for the case, from `smilei_parameters`: `1Dcartesian` for schema 1,
`2Dcartesian` for schema 2, with `random_seed = seed`. The seed is the only line that differs
between the namelists of one case (T-181). Schema 2 also takes the GPU build and a patch layout
(`smilei_parameters_2d`) and `load_check` (`load_check_blocks`).
"""
function namelist(case; seed = first(case_seeds(case)), gpu = false, patches = nothing, load_check = false)
    case["schema"] == 2 && return namelist_2d(case; seed, gpu, patches, load_check)
    (gpu || patches !== nothing || load_check) &&
        error("GPU runs, patch layouts and load checks are generated for schema-2 (2D) cases only")
    return namelist_1d(case; seed)
end

function namelist_1d(case; seed = first(case_seeds(case)))
    p = merge(smilei_parameters_1d(case), (; random_seed = Int(seed)))
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
    print(io, current_filter_block(p))
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

"""
    smilei_simulation_time(p) -> Float64

The `simulation_time` passed to Smilei for `p.number_of_timesteps` steps: (n + ½) Δt. Smilei
reads only `simulation_time` (`number_of_timesteps` sets it to n·Δt when it is absent,
pyinit.py) and takes n_time = (int)(simulation_time/timestep) (Params.cpp), which truncates
n·Δt/Δt to n − 1 when the quotient rounds below n (3391 steps of landau-oblique-2d ran as 3390).
The half step makes the truncation give n; Smilei then resets simulation_time to n·Δt.
"""
smilei_simulation_time(p) = (p.number_of_timesteps + 0.5) * p.timestep

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

"""
    load_check_steps

The steps of a load-check run: one, so that the run stops right after its t = 0 diagnostics.
"""
const load_check_steps = 1

"""
    load_check_blocks(p) -> String

The diagnostics a load-check namelist adds (WP-030): every mobile species' particles in the
lower-left eighth of the box per axis (1/64 of them; x < L_x/8 and y < L_y/8, whole patches of
any layout up to 8 per axis) at t = 0 only, with positions, momenta and weights
(`TrackParticlesDisordered_<species>.h5`). With the case's scalars and the fields dumped at
t = 0, `scripts/crosscode/loadcheck.jl` compares two builds' loads particle by particle.
"""
function load_check_blocks(p)
    io = IOBuffer()
    xl, yl = p.grid_length ./ 8
    println(io, """

    def pm_load_check_corner(particles):
        return (particles.x < $(pyfloat(xl))) * (particles.y < $(pyfloat(yl)))
    """)
    for s in p.species
        s.time_frozen > 0 && continue
        println(io, """
        DiagTrackParticles(
            species = "$(s.name)",
            every = $(10 * (load_check_steps + 1)),
            filter = pm_load_check_corner,
            attributes = ["x", "y", "px", "py", "pz", "w"],
        )
        """)
    end
    return String(take!(io))
end

function namelist_2d(case; seed = first(case_seeds(case)), gpu = false, patches = nothing, load_check = false)
    p = merge(smilei_parameters_2d(case; gpu, patches), (; random_seed = Int(seed)))
    load_check && (p = merge(p, (; number_of_timesteps = load_check_steps, fields_every = 10 * (load_check_steps + 1),
                                   scalars_every = 1)))
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
        simulation_time = $(pyfloat(smilei_simulation_time(p))),
        number_of_timesteps = $(p.number_of_timesteps),
        number_of_patches = $(pyvec(p.number_of_patches)),
        EM_boundary_conditions = [["periodic"], ["periodic"]],
        solve_poisson = True,
        random_seed = $(p.random_seed),
        print_every = $(max(1, p.number_of_timesteps ÷ 20)),$(p.gpu_computing ? "\n    gpu_computing = True," : "")
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
    print(io, current_filter_block(p))
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
    load_check && print(io, load_check_blocks(p))
    return String(take!(io))
end

"""
    run_header(p; binary, gpu, indent) -> String

The lines a run writes to its container.txt before Smilei starts: the binary, the image's
version, threads and CPU, the patch layout; a GPU run also the build, the GPU and driver, and
the GPU's utilization, memory in use and compute processes as nvidia-smi sees them inside the
container just before the start (so a timing says whether the GPU was otherwise idle).
"""
function run_header(p; binary, gpu)
    nvq(q) = "\$(nvidia-smi --query-$q --format=csv,noheader 2>/dev/null | paste -sd ';' - || echo unknown)"
    lines = ["echo \"# build: $(gpu ? "gpu" : "cpu")\"",
             "echo \"# smilei binary: \$(command -v $binary)\"",
             "echo \"# smilei version: \$(cat /opt/smilei/VERSION 2>/dev/null || echo unknown)\"",
             "echo \"# threads: \$OMP_NUM_THREADS, cpus: \$(nproc)\"",
             "echo \"# cpu model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')\"",
             "echo \"# patches: $(join(p.number_of_patches, "x"))\""]
    gpu && append!(lines, ["echo \"# fastmath: $(binary == "smilei" ? "yes" : "no")\"",
                           "echo \"# mpi: \$(grep -h '^export ' \"\$mpi_env\" | sed 's/^export //' | paste -sd ' ' -)\"",
                           "echo \"# gpu: $(nvq("gpu=name,driver_version"))\"",
                           "echo \"# gpu before: $(nvq("gpu=utilization.gpu,memory.used"))\"",
                           "echo \"# gpu processes before: $(nvq("compute-apps=pid,process_name,used_memory"))\""])
    return join(lines, "\n")
end

"""
The lines of a GPU run.sh that set up MPI: the image's /opt/smilei/mpi_env.sh, the messaging
layer and transports its build verified with a singleton MPI start (smilei-gpu.Containerfile),
or the file `PM_MPI_ENV` names.
"""
mpi_env_lines() = "mpi_env=\"\${PM_MPI_ENV:-/opt/smilei/mpi_env.sh}\"\n. \"\$mpi_env\""

"The line a GPU run appends to container.txt after Smilei ends (empty for a CPU run)."
gpu_after(gpu) = gpu ? "echo \"# gpu after: \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | paste -sd ';' - || echo unknown)\" >> container.txt" : ""

indent(text, n) = join((isempty(l) ? l : " "^n * l for l in split(text, '\n')), '\n')

"""
    run_script(p; seeds = nothing, budget = nothing, gpu = false, fastmath = true) -> String

The script the container runs (runner: `bash /run-dir/run.sh`; `run-smilei.sh` mounts the same
paths). Smilei writes its output into the working directory; the log goes to smilei.log. A GPU
run (the smilei-gpu image) runs `smilei`, built with the guide's fastmath flags, or
`smilei-nofastmath`, with one MPI rank and no OpenMP, and writes the image's toolchain
(`pm-versions`) to versions.toml in the top directory. `seeds` and `budget` as in
`seeded_run_script`.
"""
function run_script(p; seeds = nothing, budget = nothing, gpu = false, fastmath = true)
    threads = hasproperty(p, :threads) ? p.threads : p.number_of_patches
    binary = gpu && !fastmath ? "smilei-nofastmath" : "smilei"
    seeds === nothing || return seeded_run_script(p, threads, seeds; budget, binary, gpu)
    return """
    #!/usr/bin/env bash
    # Generated by scripts/crosscode/smilei.jl. Runs inside the Smilei $(gpu ? "GPU " : "")container with the run
    # directory at /run-dir (PM_RUN_DIR) and the worktree read-only at /worktree.
    set -euo pipefail
    cd "\${PM_RUN_DIR:-/run-dir}"
    export OMP_NUM_THREADS=$threads
    export OMP_SCHEDULE=dynamic
    $(gpu ? mpi_env_lines() : "")
    $(gpu ? "pm-versions > versions.toml" : "")
    {
    $(indent(run_header(p; binary, gpu), 2))
    } > container.txt
    echo "# start: \$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> container.txt
    $binary namelist.py > smilei.log 2>&1
    echo "# end: \$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> container.txt
    $(gpu_after(gpu))
    """
end

"""
The run.sh of a seeded case: each seed-<seed>/ in turn, each with its own container.txt and log.
A seed whose container.txt already has its `# end:` line is skipped, so running the directory
again continues it. With `budget` (seconds), the script starts no new seed once the job has
run that long, and says which seeds remain; the next job on the directory runs them.
"""
function seeded_run_script(p, threads, seeds; budget = nothing, binary = "smilei", gpu = false)
    list = join(string.(seeds), " ")
    return """
    #!/usr/bin/env bash
    # Generated by scripts/crosscode/smilei.jl. Runs inside the Smilei $(gpu ? "GPU " : "")container with the run
    # directory at /run-dir (PM_RUN_DIR) and the worktree read-only at /worktree; one Smilei run
    # per seed, in turn, each in its seed-<seed>/ subdirectory. Seeds that finished in an earlier
    # job are skipped$(budget === nothing ? "" : "; no new seed starts after $budget s of this job").
    set -euo pipefail
    top="\${PM_RUN_DIR:-/run-dir}"
    export OMP_NUM_THREADS=$threads
    export OMP_SCHEDULE=dynamic
    $(gpu ? mpi_env_lines() : "")
    $(gpu ? "pm-versions > \"\$top/versions.toml\"" : "")
    job_start=\$(date +%s)
    budget=$(something(budget, ""))
    remaining=""
    for seed in $list; do
      cd "\$top/seed-\$seed"
      grep -q '^# end:' container.txt 2>/dev/null && continue
      if [ -n "\$budget" ] && [ \$(( \$(date +%s) - job_start )) -ge "\$budget" ]; then
        remaining="\$remaining \$seed"; continue
      fi
      {
    $(indent(run_header(p; binary, gpu), 4))
        echo "# seed: \$seed"
      } > container.txt
      echo "# start: \$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> container.txt
      $binary namelist.py > smilei.log 2>&1
      echo "# end: \$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> container.txt
      $(gpu_after(gpu))
    done
    if [ -n "\$remaining" ]; then echo "# seeds remaining after the budget:\$remaining"; else echo "# all seeds done"; fi
    """
end

"""
    recipe_tag_hash(family) -> String

The first 16 hex of SHA-256 over the recipe and the files in `<family>/` in sorted path order,
the tag rule of runs outside the factory (scripts/crosscode/containers/README.md).
"""
function recipe_tag_hash(family)
    cdir = joinpath(CROSSCODE, "containers")
    files = ["$family.Containerfile"]
    fdir = joinpath(cdir, family)
    if isdir(fdir)
        for (root, _, fs) in walkdir(fdir), f in fs
            push!(files, relpath(joinpath(root, f), cdir))
        end
    end
    sort!(files)
    return bytes2hex(sha256(reduce(vcat, read.(joinpath.(cdir, files)))))[1:16]
end

"""
    cli_options(args, flags) -> (positional, options)

Split command-line arguments into the positional ones, in order, and the value of each flag in
`flags` (`--name value`, anywhere in the list); a flag without a value is an error.
"""
function cli_options(args, flags)
    pos, opts = String[], Dict{String,String}()
    i = 1
    while i <= length(args)
        if args[i] in flags
            i < length(args) || error("$(args[i]) needs a value")
            opts[args[i]] = args[i+1]
            i += 2
        else
            push!(pos, args[i])
            i += 1
        end
    end
    return pos, opts
end

"""
    prepare_run(case_name; tag, budget, gpu = false, fastmath = true, patches = nothing, load_check = false) -> dir

Write the run directory under scripts/runs/<tag>/ and its manifest; for a case with a `seeds`
list, one subdirectory seed-<seed>/ per seed with its namelist and a copy of the case, and a
run.sh that skips finished seeds and, with `budget` (seconds), starts no seed after that long.

`gpu` prepares a run of the smilei-gpu image (schema 2), `fastmath = false` its binary without
the guide's fastmath flags, `patches = (px, py)` a patch layout other than the default of the
build. `load_check` prepares the one-step run of the case's first seed that dumps the t = 0
load (`load_check_blocks`), as a single run. The default tag names the build, the layout when
it is not the CPU default, and the fastmath choice of a GPU run:
smilei[-gpu][-loadcheck]-<case>[-p<px>x<py>][-nofastmath][-s<n>]-<first 8 hex of the case hash>.
`first_seeds = n` runs only the first n seeds of a seeded case (a paired sub-ensemble; the
manifest's `seeds` lists them, and the analysis takes the seeds from there).
"""
function prepare_run(name; tag = nothing, runs = joinpath(SCRIPTS, "runs"), budget = nothing, gpu = false, fastmath = true,
                     patches = nothing, load_check = false, first_seeds = nothing)
    case = load_case(name)
    gpu || fastmath || error("fastmath = false selects the GPU image's binary without fastmath; it needs gpu = true")
    p = smilei_parameters(case; gpu, patches)
    h = file_hash(case["_path"])
    layout = join(p.number_of_patches, "x")
    family = gpu ? "smilei-gpu" : "smilei"
    variant = (gpu || patches !== nothing || load_check ? "-p$layout" : "") * (fastmath ? "" : "-nofastmath") *
              (first_seeds === nothing ? "" : "-s$first_seeds")
    tag = something(tag, "$family$(load_check ? "-loadcheck" : "")-$(case["name"])$variant-$(h[1:8])")
    dir = joinpath(runs, tag)
    mkpath(dir)
    cp(case["_path"], joinpath(dir, "case.toml"); force = true)
    seeded = haskey(case["numerics"], "seeds") && !load_check
    seeds = load_check ? [first(case_seeds(case))] : case_seeds(case)
    if first_seeds !== nothing
        seeded && 1 <= first_seeds <= length(seeds) || error("first_seeds = $first_seeds: a seeded case has 1 to $(length(seeds)) seeds")
        seeds = seeds[1:first_seeds]
    end
    if seeded
        for s in seeds
            sd = joinpath(dir, "seed-$s")
            mkpath(sd)
            cp(case["_path"], joinpath(sd, "case.toml"); force = true)
            write(joinpath(sd, "namelist.py"), namelist(case; seed = s, gpu, patches))
        end
        inputs = ["case.toml"; ["seed-$s/namelist.py" for s in seeds]; "run.sh"]
    else
        write(joinpath(dir, "namelist.py"), namelist(case; seed = only(seeds), gpu, patches, load_check))
        inputs = ["case.toml", "namelist.py", "run.sh"]
    end
    write(joinpath(dir, "run.sh"), run_script(p; seeds = seeded ? seeds : nothing, budget, gpu, fastmath))
    chmod(joinpath(dir, "run.sh"), 0o755)
    recipe = joinpath(CROSSCODE, "containers", "$family.Containerfile")
    write_manifest(dir; root = ROOT, inputs,
        extra = Dict{String,Any}("seeds" => seeds,
            "run" => tag, "code" => "smilei", "case" => case["name"],
            "build" => gpu ? "gpu" : "cpu", "patches" => [p.number_of_patches...],
            "seed_subset" => first_seeds === nothing ? "all of the case's seeds" : "the first $first_seeds of the case's $(length(case_seeds(case))) seeds",
            "fastmath" => gpu ? fastmath : "not applicable (CPU build)", "load_check" => load_check,
            "case_source" => relpath(case["_path"], ROOT),
            "generator" => "scripts/crosscode/smilei.jl",
            "generator_hash" => "sha256:" * file_hash(@__FILE__),
            "image" => Dict{String,Any}(
                "family" => family,
                "recipe" => relpath(recipe, ROOT),
                "recipe_hash" => "sha256:" * file_hash(recipe),
                "outside_factory_tag" => "particlemesh-$family:" * recipe_tag_hash(family),
                "id" => "pending: written by the analyze step from the runner log")))
    return dir
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .SmileiDeck
    cmd = get(ARGS, 1, "")
    args = collect(ARGS[2:end])
    flag(f) = (i = findfirst(==(f), args); i === nothing ? false : (deleteat!(args, i); true))
    gpu, nofast, load_check = flag("--gpu"), flag("--no-fastmath"), flag("--load-check")
    rest, opts = cli_options(args, ("--budget", "--patches", "--first-seeds"))
    first_seeds = haskey(opts, "--first-seeds") ? parse(Int, opts["--first-seeds"]) : nothing
    budget = haskey(opts, "--budget") ? parse(Int, opts["--budget"]) : nothing
    patches = haskey(opts, "--patches") ? Tuple(parse.(Int, split(opts["--patches"], "x"))) : nothing
    if cmd == "prepare" && length(rest) >= 1
        dir = prepare_run(rest[1]; tag = get(rest, 2, nothing), budget, gpu, fastmath = !nofast, patches, load_check, first_seeds)
        println(dir)
    elseif cmd == "namelist" && length(rest) >= 1
        print(namelist(load_case(rest[1]); gpu, patches, load_check))
    else
        println(stderr, "usage: smilei.jl prepare <case> [tag] [--budget <seconds>] [--gpu] [--no-fastmath] [--patches <px>x<py>] [--load-check] [--first-seeds <n>] | namelist <case> [--gpu] [--patches <px>x<py>] [--load-check]")
        exit(2)
    end
end
