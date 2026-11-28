# The like-for-like 3D GPU benchmark of ParticleMesh against PIConGPU 0.8.0 (WP-090, A5, A7).
#
#   julia +1.13 --project=scripts scripts/bench3d/bench3d.jl prepare <case.toml> <tag>     # scripts/runs/bench3d-<tag>/
#   julia +1.13 --project=scripts scripts/bench3d/bench3d.jl run-pm <dir> [cpu|cuda]       # a pm-* directory into timing.json
#   julia +1.13 --project=scripts scripts/bench3d/bench3d.jl read-picongpu <dir>           # a picongpu-* directory's log into timing.json
#   julia +1.13 --project=scripts scripts/bench3d/bench3d.jl compare <runs dir> <out dir>  # comparison.json, the figure and its sidecar
#   julia +1.13 --project=scripts scripts/bench3d/bench3d.jl table <comparison.json>       # the report's tables
#
# Two kinds of run, both on one GPU and one rank:
#
# - `uniform`: a warm electron plasma in a periodic box, no laser, no window, no output in the
#   loop. It isolates the step: push, current, field solve. The case's `[[uniform]]` entries are
#   the sizes, particles per cell and precisions; ParticleMesh runs each in Reference mode (sort and
#   segmented reduction) and in Fast mode (atomics), PIConGPU once.
# - `lwfa`: WP-082's 3D ionization-injection case (`scripts/lwfa/cases/ionization-3d.toml`) as
#   ionization.jl builds it for each code, with the dopant (`plasma`) and without it
#   (`plasma-nodopant`), in the case's precision, without the outputs: the application number.
#   PIConGPU's moving window needs a second, hidden local domain ahead of the window
#   (scripts/lwfa/picongpu.jl), so it computes more cells than ParticleMesh; the timing records
#   both boxes.
#
# How the codes are matched in `uniform` (each line a choice, the case holds the numbers):
#
# - Algorithms: Yee, Esirkepov current, quadratic shape (PIConGPU's TSC) for deposition and gather,
#   Boris, no current filter, the element type of the entry in both (PIConGPU's
#   `precision32Bit`/`precision64Bit` for every operation, sqrt, exp and trigonometric included).
# - The load: a quiet lattice of `ppc` particles per cell in both; the momenta thermal, v_th per
#   component, Maxwell–Jüttner in ParticleMesh and PIConGPU's non-relativistic Maxwellian in
#   momentum at T = mₑv_th² (`Temperature`), which agree to O(v_th²). Electrons only: neither code
#   solves Poisson's equation, the immobile neutralizing ions enter nothing.
# - Diagnostics: none in the loop. ParticleMesh records no history (`record = ()`); PIConGPU runs
#   its field and particle energy plugins at period `steps`, at step 0 and the last, which are the
#   cross-check that both ran the same plasma (the kinetic energy at the start, the field energy
#   the thermal noise grows).
# - Timing. The steady time per step is the time from the first timing point at or after
#   `warmup` steps to the last, over the steps between: ParticleMesh synchronizes the device every
#   `block` steps, PIConGPU prints its elapsed time every 5 % of the run (`picongpu.log`), and the
#   steps between their first points differ by at most one block. Start-up is recorded apart:
#   ParticleMesh's package load, the state's construction and initialization, and its first step
#   (Julia compiles the step there); PIConGPU's `initialization time` and its compile (`build.sh`,
#   once per binary, from container.txt).
# - The push rate is particles × steps per second over the steady steps; ParticleMesh's particle
#   count is its alive count, PIConGPU's the cells times the lattice (its quiet start fills every
#   cell, `QuietImpl`, and a periodic box loses none).

const T_START = time()
isdefined(@__MODULE__, :pm_ion_problem) || include(joinpath(@__DIR__, "..", "lwfa", "ionization.jl"))   # once, when the LWFA tests have it already

const BENCH = @__DIR__
const PIC_BENCH_SUPERCELL = (8, 8, 4)

# --- the case -------------------------------------------------------------------------------------

"""
    uniform_entry(case, e) -> NamedTuple

One `[[uniform]]` entry with the case's common numbers: `name`, `cells` (3), the lattice `ppc` (3),
`float` ("Float32" or "Float64"), the cubic cell `dx`, Δt = courant/√(Σ 1/Δ²), the density, v_th,
`steps`, `warmup`, `block`, the particles `cells × ppc` and the cells.
"""
function uniform_entry(case, e)
    U = case["uniform_defaults"]
    cells = Tuple(Int.(e["cells"]))
    ppc = Tuple(Int.(e["ppc"]))
    length(cells) == 3 && length(ppc) == 3 || error("$(e["name"]): cells and ppc take three integers")
    float = get(e, "float", U["float"])
    float in ("Float32", "Float64") || error("$(e["name"]): float = $float")
    dx = Float64(U["dx"])
    dt = Float64(U["courant"]) / sqrt(3 / dx^2)
    steps, warmup, block = Int(U["steps"]), Int(U["warmup"]), Int(U["block"])
    0 < warmup < steps && steps % block == 0 && warmup % block == 0 || error("steps, warmup and block: 0 < warmup < steps, both multiples of block")
    all(cells[k] % PIC_BENCH_SUPERCELL[k] == 0 for k in 1:3) || error("$(e["name"]): cells $cells are not multiples of PIConGPU's supercell $PIC_BENCH_SUPERCELL")
    return (; name = String(e["name"]), cells, ppc, float, dx, dt, density = Float64(U["density"]), vth = Float64(U["thermal_velocity"]),
              steps, warmup, block, particles = prod(cells) * prod(ppc), ncells = prod(cells),
              lambda0 = Float64(case["lambda0_um"]) * 1e-6)
end

uniform_entries(case) = [uniform_entry(case, e) for e in case["uniform"]]

"The ParticleMesh problem of a `uniform` entry in element type T: electrons on the lattice, thermal, periodic."
function uniform_problem(u; T = u.float == "Float32" ? Float32 : Float64)
    L = ntuple(k -> T(u.cells[k] * u.dx), 3)
    v = T(u.vth)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(u.density), zero(T), (1, 0, 0), u.ppc,
                              (zero(T), zero(T), zero(T)), (v, v, v), 1)
    return EMProblem("bench3d-$(u.name)", L, u.cells, T(u.dt), u.steps, (e,), 1, (1, 0, 0))
end

"ParticleMesh's numerics of the benchmark: quadratic, Esirkepov, Boris, Yee, no filter, in `mode`."
bench_numerics(mode) = EMNumerics(; shape = Quadratic, mode, pusher = Boris(), solver = Yee(), current = Esirkepov())

mode_named(m) = m == "reference" ? Reference() : m == "fast" ? Fast() : error("mode $m: reference or fast")

# --- ParticleMesh ---------------------------------------------------------------------------------

"Host field energy ½ΣE² ΔV and ½ΣB² ΔV over the interior, and the kinetic energy Σ m w (γ − 1) ΔV of the alive particles (A1 units, mₑc² n_c (c/ω₀)³)."
function host_energies(st)
    g, n, h = guards(st.grid), ncells(st.grid), cellsize(st.grid)
    I = CartesianIndices(ntuple(d -> (g + 1):(g + n[d]), length(n)))
    dV = prod(Float64.(h))
    W(c) = 0.5 * dV * sum(abs2, Float64.(Array(getfield(st.fields, c)))[I])
    K = 0.0
    for s in st.species
        alive = Array(s.status) .== ParticleMeshCore.ALIVE
        u = map(a -> Float64.(Array(a))[alive], s.u)
        w = Float64.(Array(s.w))[alive]
        K += Float64(s.m) * dV * sum(w .* (sqrt.(1 .+ u[1] .^ 2 .+ u[2] .^ 2 .+ u[3] .^ 2) .- 1))   # a weight is a density (n_c), as the recorder's K
    end
    return Dict{String,Any}("WE" => sum(W, (:Ex, :Ey, :Ez)), "WB" => sum(W, (:Bx, :By, :Bz)), "K" => K)
end

sync(be) = KernelAbstractions.synchronize(be)

"""
    timed_steps!(st, p, be, steps, block) -> (; first_step, marks)

Step 1 alone (it compiles), then steps 2:steps with the device synchronized every `block` steps;
`marks` the (step, seconds since the end of step 1) pairs, (1, 0) first.
"""
function timed_steps!(st, p, be, steps, block; first = 1)
    sync(be)
    t = time_ns()
    step!(st, p, first)
    sync(be)
    t1 = time_ns()
    marks = Tuple{Int,Float64}[(first, 0.0)]
    for k in (first + 1):(first + steps - 1)
        step!(st, p, k)
        if (k - first + 1) % block == 0
            sync(be)
            push!(marks, (k, (time_ns() - t1) / 1e9))
        end
    end
    sync(be)
    marks[end][1] == first + steps - 1 || push!(marks, (first + steps - 1, (time_ns() - t1) / 1e9))
    return (; first_step = (t1 - t) / 1e9, marks)
end

"The steady seconds per step from timing points (step, seconds): from the first point at or after step `warmup` to the last."
function steady_step(marks, warmup)
    i = findfirst(m -> m[1] >= warmup, marks)
    (i === nothing || i == length(marks)) && error("no two timing points at or after step $warmup")
    a, b = marks[i], marks[end]
    return (; seconds = (b[2] - a[2]) / (b[1] - a[1]), from = a[1], to = b[1])
end

function gpu_label(be)
    be isa CPU && return "none"
    return string(@eval CUDACore.name(CUDACore.device()))
end

"Run a ParticleMesh `uniform` directory (its case.toml, entry.toml) into timing.json and the manifest."
function run_pm_uniform(dir; backend = CPU())
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    ent = TOML.parsefile(joinpath(dir, "entry.toml"))
    u = uniform_entry(case, ent)
    mode = ent["mode"]
    t_build = time()
    prob = uniform_problem(u)
    st = EMState(prob, bench_numerics(mode_named(mode)); record = (), backend)
    p = params(prob)
    initialize!(st, p)
    sync(backend)
    np = alive_count(only(st.species))
    e0 = host_energies(st)
    t_ready = time()
    r = timed_steps!(st, p, backend, u.steps, u.block)
    s = steady_step(r.marks, u.warmup)
    e1 = host_energies(st)
    out = Dict{String,Any}("code" => "ParticleMesh", "kind" => "uniform", "entry" => u.name, "mode" => mode, "float" => u.float,
        "cells" => collect(u.cells), "ppc" => collect(u.ppc), "particles" => np, "computed_cells" => u.ncells, "steps" => u.steps,
        "load_seconds" => t_build - T_START, "setup_seconds" => t_ready - t_build, "first_step_seconds" => r.first_step,
        "marks" => [[m[1], m[2]] for m in r.marks],
        "loop_seconds" => r.first_step + r.marks[end][2],
        "steady" => Dict("seconds_per_step" => s.seconds, "from" => s.from, "to" => s.to,
                         "ns_per_particle_step" => 1e9 * s.seconds / np, "ns_per_cell_step" => 1e9 * s.seconds / u.ncells),
        "energy_start" => e0, "energy_end" => e1, "backend" => backend_label(backend), "gpu" => gpu_label(backend),
        "cpu" => Sys.cpu_info()[1].model, "threads" => Threads.nthreads(), "julia" => string(VERSION))
    write_timing(dir, out)
    println("pm $(basename(dir)): $(round(1e3 * s.seconds; digits = 3)) ms per step, $(round(out["steady"]["ns_per_particle_step"]; sigdigits = 4)) ns per particle-step, ",
            "first step $(round(r.first_step; digits = 1)) s")
    return out
end

"""
Run a ParticleMesh `lwfa` directory (case.toml, kind.txt, entry.toml) into timing.json: the
problem and numerics of `run_pm_ion` in the entry's mode and element type, every step without an
output, and at the end the field energies and the born electrons' weight in the window
(`pm_ion_output`), the check against WP-082's run of the same kind.
"""
function run_pm_lwfa(dir; backend = CPU())
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    ent = TOML.parsefile(joinpath(dir, "entry.toml"))
    kind = strip(read(joinpath(dir, "kind.txt"), String))
    T = ent["float"] == "Float32" ? Float32 : Float64
    t_build = time()
    prob = pm_ion_problem(case; T, plasma = kind != "vacuum", dopant = kind != "plasma-nodopant", envelope = laser_envelope(case))
    st = EMState(prob, pm_ion_numerics(case; mode = mode_named(ent["mode"])); record = (), backend)
    p = params(prob)
    initialize!(st, p)
    sync(backend)
    t_ready = time()
    d = ion_derived(case)
    block = Int(ent["block"])
    r = timed_steps!(st, p, backend, prob.nsteps, block)
    s = steady_step(r.marks, Int(ent["warmup"]))
    o = pm_ion_output(st, case, prob.nsteps)
    check = Dict{String,Any}("energy_end" => host_energies(st), "Ey_absmax" => o["Ey_absmax"])
    haskey(o, "born") && (check["born_weight"] = o["born"]["weight"])
    haskey(o, "plasma") && (check["plasma_weight"] = o["plasma"]["weight"])
    ncell = prod(ncells(st.grid))
    out = Dict{String,Any}("code" => "ParticleMesh", "kind" => "lwfa", "entry" => kind, "mode" => ent["mode"], "float" => ent["float"],
        "cells" => collect(ncells(st.grid)), "computed_cells" => ncell, "steps" => prob.nsteps,
        "load_seconds" => t_build - T_START, "setup_seconds" => t_ready - t_build, "first_step_seconds" => r.first_step,
        "marks" => [[m[1], m[2]] for m in r.marks], "loop_seconds" => r.first_step + r.marks[end][2],
        "steady" => Dict("seconds_per_step" => s.seconds, "from" => s.from, "to" => s.to, "ns_per_cell_step" => 1e9 * s.seconds / ncell),
        "check" => check, "backend" => backend_label(backend), "gpu" => gpu_label(backend),
        "cpu" => Sys.cpu_info()[1].model, "threads" => Threads.nthreads(), "julia" => string(VERSION))
    write_timing(dir, out)
    println("pm $(basename(dir)): loop $(round(out["loop_seconds"]; digits = 1)) s, steady $(round(1e3 * s.seconds; digits = 2)) ms per step")
    return out
end

function write_timing(dir, out)
    open(io -> JSON.print(io, out, 2), joinpath(dir, "timing.json"), "w")
    write_manifest(dir; root = REPO, inputs = filter(f -> isfile(joinpath(dir, f)), ["case.toml", "entry.toml", "kind.txt"]),
                   extra = Dict{String,Any}("run_id" => basename(dirname(dir)) * "/" * basename(dir), "code" => out["code"],
                                            "work_package" => "WP-090", "generator" => "scripts/bench3d/bench3d.jl",
                                            "outputs" => Dict("timing.json" => "sha256:" * RunManifest.file_hash(joinpath(dir, "timing.json"))),
                                            (k => out[k] for k in ("backend", "gpu", "cpu", "threads", "julia", "float", "mode") if haskey(out, k))...))
end

# --- PIConGPU: the uniform deck ---------------------------------------------------------------------

"""
    picongpu_uniform_param_files(u) -> Dict{String,String}

The .param files of a `uniform` entry over PIConGPU's `Empty` example, in SI from
`reference_units` (A1's conversion lives only here and in the readers): DIM3, the precision, the
cubic cell and Δt, a homogeneous density on the quiet lattice, the temperature, TSC, Esirkepov,
Boris, Yee, supercells of 8 × 8 × 4; no particle output.
"""
function picongpu_uniform_param_files(u)
    cf, H = PIConGPUDeck.cfloat, PIConGPUDeck.HEADER * "/* WP-090 benchmark \"$(u.name)\", scripts/bench3d/bench3d.jl */\n"
    U = reference_units(u.lambda0)
    bits = u.float == "Float64" ? "precision64Bit" : "precision32Bit"
    T_keV = u.vth^2 * U.energy / PIConGPUDeck.QE_SI / 1e3
    sx, sy, sz = PIC_BENCH_SUPERCELL
    px, py, pz = u.ppc
    f = Dict{String,String}()
    f["dimension.param"] = H * """
    #pragma once
    #define SIMDIM DIM3
    namespace picongpu
    {
        constexpr uint32_t simDim = SIMDIM;
    } // namespace picongpu
    """
    f["precision.param"] = H * """
    #pragma once
    #include <picongpu/simulation_types.hpp>
    namespace picongpu
    {
        namespace precisionPIConGPU = $bits;
        namespace precisionSqrt = precisionPIConGPU;
        namespace precisionExp = precisionPIConGPU;
        namespace precisionTrigonometric = precisionPIConGPU;
    } // namespace picongpu
    #include "picongpu/unitless/precision.unitless"
    """
    f["simulation.param"] = H * """
    #pragma once
    namespace picongpu
    {
        namespace SI
        {
            constexpr float_64 DELTA_T_SI = $(cf(u.dt * U.time));
            constexpr float_64 CELL_WIDTH_SI = $(cf(u.dx * U.length));
            constexpr float_64 CELL_HEIGHT_SI = $(cf(u.dx * U.length));
            constexpr float_64 CELL_DEPTH_SI = $(cf(u.dx * U.length));
            /* n_c at the reference wavelength $(cf(u.lambda0)) m; the profile is relative to it */
            constexpr float_64 BASE_DENSITY_SI = $(cf(U.density));
        } // namespace SI
        constexpr uint32_t TYPICAL_PARTICLES_PER_CELL = $(prod(u.ppc))u;
    } // namespace picongpu
    """
    f["memory.param"] = H * """
    #pragma once
    #include <pmacc/mappings/kernel/MappingDescription.hpp>
    #include <pmacc/math/Vector.hpp>
    #include <array>
    namespace picongpu
    {
        constexpr size_t reservedGpuMemorySize = 350 * 1024 * 1024;
        namespace mCT = pmacc::math::CT;
        using SuperCellSize = typename mCT::shrinkTo<mCT::Int<$sx, $sy, $sz>, simDim>::type;
        static constexpr uint32_t numFrameSlots = pmacc::math::CT::volume<SuperCellSize>::type::value;
        using MappingDesc = MappingDescription<simDim, SuperCellSize>;
        using GuardSize = typename mCT::shrinkTo<mCT::Int<1, 1, 1>, simDim>::type;
        struct DefaultExchangeMemCfg
        {
            static constexpr uint32_t BYTES_EXCHANGE_X = 16 * 1024 * 1024;
            static constexpr uint32_t BYTES_EXCHANGE_Y = 16 * 1024 * 1024;
            static constexpr uint32_t BYTES_EXCHANGE_Z = 16 * 1024 * 1024;
            static constexpr uint32_t BYTES_EDGES = 1 * 1024 * 1024;
            static constexpr uint32_t BYTES_CORNER = 256 * 1024;
            using REF_LOCAL_DOM_SIZE = mCT::Int<0, 0, 0>;
            const std::array<float_X, 3> DIR_SCALING_FACTOR = {{0.0, 0.0, 0.0}};
        };
        constexpr uint32_t fieldTmpNumSlots = 1;
        constexpr bool fieldTmpSupportGatherCommunication = true;
    } // namespace picongpu
    """
    f["species.param"] = H * """
    #pragma once
    #include "picongpu/algorithms/AssignedTrilinearInterpolation.hpp"
    #include "picongpu/algorithms/FieldToParticleInterpolation.hpp"
    #include "picongpu/fields/currentDeposition/Solver.def"
    #include "picongpu/particles/shapes.hpp"
    namespace picongpu
    {
        /* quadratic shape (TSC), Esirkepov current, Boris pusher, as ParticleMesh's bench_numerics */
        using UsedParticleShape = particles::shapes::TSC;
        using UsedField2Particle = FieldToParticleInterpolation<UsedParticleShape, AssignedTrilinearInterpolation>;
        using UsedParticleCurrentSolver = currentSolver::Esirkepov<UsedParticleShape>;
        using UsedParticlePusher = particles::pusher::Boris;
    } // namespace picongpu
    """
    f["fieldSolver.param"] = H * """
    #pragma once
    #include "picongpu/fields/MaxwellSolver/Solvers.def"
    namespace picongpu
    {
        namespace fields
        {
            using Solver = maxwellSolver::Yee;
        } // namespace fields
    } // namespace picongpu
    """
    f["random.param"] = H * """
    #pragma once
    #include "picongpu/random/seed/Seed.hpp"
    #include <pmacc/random/methods/methods.hpp>
    namespace picongpu
    {
        namespace random
        {
            using Generator = pmacc::random::methods::XorMin<>;
            /* the seed is read from PIC_SEED at start-up */
            using SeedGenerator = seed::FromEnvironment;
        } // namespace random
    } // namespace picongpu
    """
    f["speciesDefinition.param"] = H * """
    #pragma once
    #include "picongpu/defines.hpp"
    #include "picongpu/particles/Particles.hpp"
    #include <pmacc/identifier/value_identifier.hpp>
    #include <pmacc/meta/String.hpp>
    #include <pmacc/meta/conversion/MakeSeq.hpp>
    #include <pmacc/particles/Identifier.hpp>
    #include <pmacc/particles/traits/FilterByFlag.hpp>
    namespace picongpu
    {
        using DefaultParticleAttributes = MakeSeq_t<position<position_pic>, momentum, weighting>;
        value_identifier(float_X, MassRatio_e, 1.0);
        value_identifier(float_X, ChargeRatio_e, 1.0);
        using ParticleFlags_e = MakeSeq_t<
            particlePusher<UsedParticlePusher>,
            shape<UsedParticleShape>,
            interpolation<UsedField2Particle>,
            current<UsedParticleCurrentSolver>,
            massRatio<MassRatio_e>,
            chargeRatio<ChargeRatio_e>>;
        using PIC_e = Particles<PMACC_CSTRING("e"), ParticleFlags_e, DefaultParticleAttributes>;
        using VectorAllSpecies = MakeSeq_t<PIC_e>;
    } // namespace picongpu
    """
    f["density.param"] = H * """
    #pragma once
    #include "picongpu/particles/densityProfiles/profiles.def"
    namespace picongpu
    {
        namespace densityProfiles
        {
            /* n = DENSITY n_c everywhere */
            constexpr float_64 DENSITY_e = $(cf(u.density));
            struct ProfileFunctor_e
            {
                HDINLINE float_X operator()(const floatD_64& position_SI, const float3_64& cellSize_SI)
                {
                    return static_cast<float_X>(DENSITY_e);
                }
            };
            using Profile_e = FreeFormulaImpl<ProfileFunctor_e>;
        } // namespace densityProfiles
    } // namespace picongpu
    """
    f["particle.param"] = H * """
    #pragma once
    #include "picongpu/particles/filter/filter.def"
    #include "picongpu/particles/manipulators/manipulators.def"
    #include "picongpu/particles/startPosition/functors.def"
    #include <pmacc/math/operation.hpp>
    namespace picongpu
    {
        namespace particles
        {
            /* QuietImpl drops particles below the weighting floor; none is near it here */
            constexpr float_X MIN_WEIGHTING = 1.0e-20;
            namespace startPosition
            {
                struct QuietParam_e
                {
                    using numParticlesPerDimension = mCT::shrinkTo<mCT::Int<$px, $py, $pz>, simDim>::type;
                };
                using Quiet_e = QuietImpl<QuietParam_e>;
            } // namespace startPosition
            namespace manipulators
            {
                struct TemperatureParam_e
                {
                    /* T = mₑ v_th² in keV, v_th = $(cf(u.vth)) c per component */
                    static constexpr float_64 temperature = $(cf(T_keV));
                };
                using AddTemperature_e = unary::Temperature<TemperatureParam_e>;
            } // namespace manipulators
        } // namespace particles
    } // namespace picongpu
    """
    f["speciesInitialization.param"] = H * """
    #pragma once
    #include "picongpu/particles/InitFunctors.hpp"
    namespace picongpu
    {
        namespace particles
        {
            using InitPipeline = pmacc::mp_list<
                CreateDensity<densityProfiles::Profile_e, startPosition::Quiet_e, PIC_e>,
                Manipulate<manipulators::AddTemperature_e, PIC_e>>;
        } // namespace particles
    } // namespace picongpu
    """
    f["fileOutput.param"] = H * """
    #pragma once
    #include <pmacc/meta/conversion/MakeSeq.hpp>
    #include "picongpu/fields/Fields.def"
    namespace picongpu
    {
        /* the benchmark writes no openPMD output; the lists only have to exist */
        using FieldTmpSolvers = MakeSeq_t<>;
        using NativeFileOutputFields = MakeSeq_t<FieldE, FieldB>;
        using FileOutputFields = MakeSeq_t<NativeFileOutputFields, FieldTmpSolvers>;
        using FileOutputParticles = MakeSeq_t<>;
    } // namespace picongpu
    """
    return f
end

"The command line of a `uniform` entry: one device, the grid, periodic on all sides, no current filter, the energies at step 0 and the last."
picongpu_uniform_arguments(u) = join(["-d 1 1 1", "-g $(u.cells[1]) $(u.cells[2]) $(u.cells[3])", "-s $(u.steps)", "--periodic 1 1 1",
                                      "--currentInterpolation none", "--fields_energy.period $(u.steps)",
                                      "--e_energy.period $(u.steps)", "--e_energy.filter all"], " ")

"""
    picongpu_bench_lwfa_arguments(args, steps) -> String

WP-082's PIConGPU command line without its outputs: the openPMD dumps and the energy histogram
dropped, the field energies at step 0 and the last; the rest (the grid, the ranks, the window, the
absorber) unchanged.
"""
function picongpu_bench_lwfa_arguments(args, steps)
    w = split(args)
    out, i = String[], 1
    while i <= length(w)
        if startswith(w[i], "--openPMD.") || startswith(w[i], "--$(PIC_SPECIES)_energyHistogram.") || w[i] == "--fields_energy.period"
            i += 2
        else
            push!(out, w[i]); i += 1
        end
    end
    push!(out, "--fields_energy.period", string(steps))
    return join(out, " ")
end

const STAMP = "\$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"

"build.sh: compile `Empty` with the directory's params (inside the PIConGPU container, the directory at /run-dir), timed in container.txt."
function picongpu_build_script()
    return """
    #!/usr/bin/env bash
    # Generated by scripts/bench3d/bench3d.jl. Runs inside the PIConGPU container with the run directory at /run-dir (PM_RUN_DIR).
    set -euo pipefail
    cd "\${PM_RUN_DIR:-/run-dir}"
    pm-versions > versions.toml
    {
      echo "# picongpu: \$PICONGPU_TAG \$PICONGPU_COMMIT, backend \$PIC_BACKEND"
      echo "# cpus: \$(nproc)"
      echo "# cpu model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
      echo "# compile start: $STAMP"
    } > container.txt
    rm -rf setup ./*.dat
    pic-create "\$PIC_EXAMPLES/Empty" setup > build.log 2>&1
    cp params/*.param setup/include/picongpu/param/
    (cd setup && pic-build -j "\$(nproc)") >> build.log 2>&1 || { tail -n 60 build.log; exit 1; }
    echo "# compile end: $STAMP" >> container.txt
    """
end

"run.sh: run the compiled binary on the GPU with `ranks` ranks, timed in container.txt; picongpu.log holds its own timing."
function picongpu_bench_run_script(args; ranks = 1)
    return """
    #!/usr/bin/env bash
    # Generated by scripts/bench3d/bench3d.jl. Runs inside the PIConGPU container with the run directory at /run-dir (PM_RUN_DIR)
    # and the GPU bound in, after build.sh; $ranks rank(s) on the one GPU.
    set -euo pipefail
    cd "\${PM_RUN_DIR:-/run-dir}"
    rm -f ./*.dat
    echo "# gpu at run: \$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo unknown)" >> container.txt
    echo "# start: $STAMP" >> container.txt
    PIC_SEED=1 mpiexec -n $ranks setup/bin/picongpu $args > picongpu.log 2>&1 || { tail -n 60 picongpu.log; exit 1; }
    echo "# end: $STAMP" >> container.txt
    """
end

# --- PIConGPU: the log --------------------------------------------------------------------------

"Seconds of a PIConGPU duration as it prints them: `1h 2min 3sec 456msec` with any of the parts."
function pic_duration(s)
    t = 0.0
    for (re, f) in ((r"(\d+)\s*h\b", 3600.0), (r"(\d+)\s*min\b", 60.0), (r"(\d+)\s*sec\b", 1.0), (r"(\d+)\s*msec\b", 1e-3))
        m = match(re, s)
        m === nothing || (t += f * parse(Int, m.captures[1]))
    end
    return t
end

"""
    read_picongpu_log(text) -> NamedTuple

PIConGPU's own timing from its log (`simulation/control/Simulation.hpp`): the `initialization
time`, the `calculation  simulation time` (the loop), the `full simulation time` (the part after
`=`, in seconds), and the progress points (step, elapsed seconds) of its `N % = step | time
elapsed: …` lines.
"""
function read_picongpu_log(text)
    secs(key) = (m = match(Regex(key * raw":.*=\s*([0-9.]+)\s*sec"), text); m === nothing ? nothing : parse(Float64, m.captures[1]))
    marks = Tuple{Int,Float64}[]
    for m in eachmatch(r"^\s*\d+\s*%\s*=\s*(\d+)\s*\|\s*time elapsed:\s*([^|]+)\|"m, text)
        push!(marks, (parse(Int, m.captures[1]), pic_duration(m.captures[2])))
    end
    return (; init = secs("initialization time"), loop = secs(raw"calculation\s+simulation time"), full = secs("full simulation time"), marks)
end

"The rows of a PIConGPU energy plugin's .dat file: the step and the values (J)."
function read_pic_dat(file)
    rows = Vector{Float64}[]
    for l in eachline(file)
        s = strip(l)
        (isempty(s) || startswith(s, "#")) && continue
        push!(rows, parse.(Float64, split(s)))
    end
    return rows
end

"The compile seconds from container.txt's compile stamps, or nothing."
function container_seconds(dir, a, b)
    f = joinpath(dir, "container.txt")
    isfile(f) || return nothing
    lines = readlines(f)
    stamp(q) = (l = filter(x -> startswith(x, q), lines); isempty(l) ? nothing : DateTime(last(l)[length(q) + 1:end][1:23], "yyyy-mm-ddTHH:MM:SS.sss"))
    x, y = stamp(a), stamp(b)
    x === nothing || y === nothing ? nothing : Dates.value(y - x) / 1000
end

"Read a PIConGPU directory (entry.toml, picongpu.log, the energy .dat files) into timing.json."
function read_picongpu_bench(dir)
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    ent = TOML.parsefile(joinpath(dir, "entry.toml"))
    log = read_picongpu_log(read(joinpath(dir, "picongpu.log"), String))
    kind = ent["kind"]
    out = Dict{String,Any}("code" => "PIConGPU", "kind" => kind, "float" => ent["float"], "init_seconds" => log.init, "loop_seconds" => log.loop,
                           "full_seconds" => log.full, "compile_seconds" => container_seconds(dir, "# compile start: ", "# compile end: "),
                           "wall_seconds" => container_seconds(dir, "# start: ", "# end: "), "marks" => [[m[1], m[2]] for m in log.marks],
                           "ranks" => Int(ent["ranks"]))
    if kind == "uniform"
        u = uniform_entry(case, ent)
        s = steady_step(log.marks, u.warmup)
        U = reference_units(u.lambda0)
        J = U.energy * U.density * U.length^3                # J per mₑc² n_c (c/ω₀)³
        F, K = read_pic_dat(joinpath(dir, "fields_energy.dat")), read_pic_dat(joinpath(dir, "e_energy_all.dat"))
        energy(k) = (i = findfirst(r -> Int(r[1]) == k, F); j = findfirst(r -> Int(r[1]) == k, K);
                     i === nothing || j === nothing ? nothing :
                     Dict{String,Any}("WE" => sum(F[i][6:8]) / J, "WB" => sum(F[i][3:5]) / J, "K" => K[j][2] / J))
        merge!(out, Dict{String,Any}("entry" => u.name, "cells" => collect(u.cells), "ppc" => collect(u.ppc), "particles" => u.particles,
            "computed_cells" => u.ncells, "steps" => u.steps,
            "steady" => Dict("seconds_per_step" => s.seconds, "from" => s.from, "to" => s.to,
                             "ns_per_particle_step" => 1e9 * s.seconds / u.particles, "ns_per_cell_step" => 1e9 * s.seconds / u.ncells),
            "energy_start" => energy(0), "energy_end" => energy(u.steps)))
    else
        lay = picongpu_layout(picongpu_ion_lab(case))
        d = ion_derived(case)
        s = steady_step(log.marks, Int(ent["warmup"]))
        ncell = prod(lay.cells) * d.nz
        F = read_pic_dat(joinpath(dir, "fields_energy.dat"))
        U = reference_units(Float64(case["lambda0_um"]) * 1e-6)
        J = U.energy * U.density * U.length^3
        merge!(out, Dict{String,Any}("entry" => strip(read(joinpath(dir, "kind.txt"), String)), "cells" => [lay.cells[2], lay.cells[1], d.nz],
            "window_cells" => [lay.window, d.ny, d.nz], "computed_cells" => ncell, "steps" => d.steps,
            "steady" => Dict("seconds_per_step" => s.seconds, "from" => s.from, "to" => s.to, "ns_per_cell_step" => 1e9 * s.seconds / ncell),
            "check" => Dict{String,Any}("energy_end" => (i = findfirst(r -> Int(r[1]) == d.steps, F);
                                         i === nothing ? nothing : Dict("WE" => sum(F[i][6:8]) / J, "WB" => sum(F[i][3:5]) / J)))))
    end
    vf = joinpath(dir, "versions.toml")
    isfile(vf) && (out["versions"] = TOML.parsefile(vf))
    cf = joinpath(dir, "container.txt")
    if isfile(cf)
        for l in readlines(cf)
            startswith(l, "# gpu at run: ") && (out["gpu"] = l[length("# gpu at run: ") + 1:end])
            startswith(l, "# cpu model: ") && (out["cpu"] = l[length("# cpu model: ") + 1:end])
        end
    end
    open(io -> JSON.print(io, out, 2), joinpath(dir, "timing.json"), "w")
    return out
end

# --- prepare ------------------------------------------------------------------------------------

"""
    prepare_bench(case_path, tag) -> root

scripts/runs/bench3d-<tag>/ with one directory per run: `pm-<entry>-<mode>` and `picongpu-<entry>`
for every `[[uniform]]` entry, `pm-lwfa-<kind>-<mode>` and `picongpu-lwfa-<kind>` for every kind of
`[lwfa]`, each with case.toml, entry.toml (and kind.txt), PIConGPU's params/, build.sh and run.sh,
and its manifest; `runs.txt` lists them in the order to run.
"""
function prepare_bench(case_path, tag)
    case = TOML.parsefile(case_path)
    root = joinpath(REPO, "scripts", "runs", "bench3d-$tag")
    ispath(root) && error("$root exists")
    names = String[]
    function make(name, c, entry; kind = nothing, pic = nothing, case_file = case_path)
        dir = joinpath(root, name)
        mkpath(dir)
        cp(case_file, joinpath(dir, "case.toml"))
        open(io -> TOML.print(io, entry; sorted = true), joinpath(dir, "entry.toml"), "w")
        inputs = ["case.toml", "entry.toml"]
        kind === nothing || (write(joinpath(dir, "kind.txt"), kind * "\n"); push!(inputs, "kind.txt"))
        if pic !== nothing
            mkpath(joinpath(dir, "params"))
            for (n, t) in pic.files
                write(joinpath(dir, "params", n), t)
            end
            write(joinpath(dir, "build.sh"), pic.build)
            write(joinpath(dir, "run.sh"), pic.run)
            chmod(joinpath(dir, "build.sh"), 0o755); chmod(joinpath(dir, "run.sh"), 0o755)
            write(joinpath(dir, "arguments.txt"), pic.args * "\n")
            append!(inputs, ["build.sh", "run.sh", "arguments.txt", ("params/" .* sort(collect(keys(pic.files))))...])
        end
        write_manifest(dir; root = REPO, inputs,
                       extra = Dict{String,Any}("run_id" => "bench3d-$tag/$name", "code" => c, "work_package" => "WP-090",
                                                "generator" => "scripts/bench3d/bench3d.jl"))
        push!(names, name)
    end
    for e in case["uniform"]
        u = uniform_entry(case, e)
        base = Dict{String,Any}(k => v for (k, v) in e)
        base["kind"] = "uniform"
        for m in case["uniform_defaults"]["modes"]
            make("pm-$(u.name)-$m", "ParticleMesh", merge(base, Dict{String,Any}("mode" => m)))
        end
        args = picongpu_uniform_arguments(u)
        make("picongpu-$(u.name)", "PIConGPU 0.8.0", merge(base, Dict{String,Any}("ranks" => 1, "float" => u.float)),
             pic = (; files = picongpu_uniform_param_files(u), build = picongpu_build_script(), run = picongpu_bench_run_script(args), args))
    end
    if haskey(case, "lwfa")
        Lw = case["lwfa"]
        lcase_path = joinpath(REPO, Lw["case"])
        lcase = TOML.parsefile(lcase_path)
        d = ion_derived(lcase)
        for kind in Lw["kinds"]
            common = Dict{String,Any}("kind" => "lwfa", "float" => Lw["float"], "warmup" => Int(Lw["warmup"]), "block" => Int(Lw["block"]))
            for m in Lw["modes"]
                make("pm-lwfa-$kind-$m", "ParticleMesh", merge(common, Dict{String,Any}("mode" => m)); kind, case_file = lcase_path)
            end
            lab = picongpu_ion_lab(lcase)
            p = picongpu_parameters(lab; plasma = kind != "vacuum", precision = Lw["float"])
            files = picongpu_ion_param_files(lcase; kind, seed = 1, precision = Lw["float"])
            args = picongpu_bench_lwfa_arguments(picongpu_ion_arguments(lcase, p), d.steps)
            # no particle dump, so WP-082's shadowed WriteSpecies.hpp (picongpu.jl) is not needed
            make("picongpu-lwfa-$kind", "PIConGPU 0.8.0", merge(common, Dict{String,Any}("ranks" => p.layout.devices)); kind, case_file = lcase_path,
                 pic = (; files, build = picongpu_build_script(), run = picongpu_bench_run_script(args; ranks = p.layout.devices), args))
        end
    end
    write(joinpath(root, "runs.txt"), join(names, "\n") * "\n")
    return root
end

# --- compare ------------------------------------------------------------------------------------

"""
    compare_bench(runs, out) -> comparison

Every timing.json under `runs` (a directory of run directories) into `out/comparison.json`: per
uniform entry and per LWFA kind the rows of each code and mode, and the ratios of PIConGPU's steady
time per step to each ParticleMesh mode's; the figure `steps.png` with its sidecar.
"""
function compare_bench(runs, out)
    mkpath(out)
    rows = Dict{String,Any}[]
    for d in sort(readdir(runs; join = true))
        f = joinpath(d, "timing.json")
        isfile(f) || continue
        r = JSON.parsefile(f)
        r["run"] = basename(d)
        delete!(r, "marks")
        push!(rows, r)
    end
    isempty(rows) && error("no timing.json under $runs")
    groups = Dict{String,Any}()
    for r in rows
        key = r["kind"] * "/" * r["entry"]
        g = get!(groups, key, Dict{String,Any}("kind" => r["kind"], "entry" => r["entry"], "rows" => Dict{String,Any}()))
        label = r["code"] == "PIConGPU" ? "picongpu" : "pm-" * r["mode"]
        g["rows"][label] = r
    end
    for g in values(groups)
        pic = get(g["rows"], "picongpu", nothing)
        pic === nothing && continue
        g["picongpu_over_pm"] = Dict(l => pic["steady"]["seconds_per_step"] / r["steady"]["seconds_per_step"]
                                     for (l, r) in g["rows"] if l != "picongpu")
    end
    c = Dict{String,Any}("groups" => groups, "runs" => basename(abspath(runs)))
    open(io -> JSON.print(io, c, 2), joinpath(out, "comparison.json"), "w")
    plot_bench(out, c)
    write_manifest(out; root = REPO, extra = Dict{String,Any}("work_package" => "WP-090", "generator" => "scripts/bench3d/bench3d.jl",
                   "inputs" => Dict(r["run"] * "/timing.json" => "sha256:" * RunManifest.file_hash(joinpath(runs, r["run"], "timing.json")) for r in rows)))
    return c
end

const BENCH_LABELS = ("pm-reference" => "ParticleMesh, Reference", "pm-fast" => "ParticleMesh, Fast", "picongpu" => "PIConGPU 0.8.0")

"Groups in the order of the report: the uniform entries by particle count, then the LWFA kinds."
bench_order(c) = sort(collect(values(c["groups"])); by = g -> (g["kind"] != "uniform", g["kind"] == "uniform" ?
                       (g["rows"][first(keys(g["rows"]))]["float"], get(g["rows"][first(keys(g["rows"]))], "particles", 0)) : ("", 0), g["entry"]))

function plot_bench(out, c)
    gs = bench_order(c)
    fig = Figure(size = (1000, 460))
    ax1 = Axis(fig[1, 1]; ylabel = "ns per particle-step (steady)", title = "Uniform warm plasma, periodic, one GPU",
               xticks = (1:count(g -> g["kind"] == "uniform", gs), [g["entry"] for g in gs if g["kind"] == "uniform"]), xticklabelrotation = π / 5)
    ax2 = Axis(fig[1, 2]; ylabel = "ms per step (steady)", title = "WP-082's 3D LWFA case",
               xticks = (1:count(g -> g["kind"] == "lwfa", gs), [g["entry"] for g in gs if g["kind"] == "lwfa"]))
    side = Dict{String,Any}()
    colors = Makie.wong_colors()
    for (ax, kind, val, scale) in ((ax1, "uniform", "ns_per_particle_step", 1.0), (ax2, "lwfa", "seconds_per_step", 1e3))
        sub = [g for g in gs if g["kind"] == kind]
        for (j, (l, name)) in enumerate(BENCH_LABELS)
            xs, ys = Float64[], Float64[]
            for (i, g) in enumerate(sub)
                r = get(g["rows"], l, nothing)
                r === nothing && continue
                push!(xs, i + (j - 2) * 0.27); push!(ys, scale * r["steady"][val])
            end
            isempty(xs) && continue
            barplot!(ax, xs, ys; width = 0.25, color = colors[j], label = name)
            side["$(kind)_$l"] = Dict("x" => xs, "y" => ys, "entries" => [g["entry"] for g in sub if haskey(g["rows"], l)])
        end
    end
    axislegend(ax1; position = :lt)
    save_plot(out, "steps", fig, side)
end

fmt(x; d = 3) = x === nothing ? "–" : string(round(x; sigdigits = d))

"The report's tables from a comparison: the uniform entries, the LWFA kinds, and start-up."
function bench_table(c)
    io = IOBuffer()
    gs = bench_order(c)
    println(io, "| uniform | precision | particles | code | ms per step | ns per particle-step | ns per cell-step | PIConGPU / this |")
    println(io, "|---|---|---|---|---|---|---|---|")
    for g in gs
        g["kind"] == "uniform" || continue
        for (l, name) in BENCH_LABELS
            r = get(g["rows"], l, nothing)
            r === nothing && continue
            s = r["steady"]
            ratio = l == "picongpu" ? "" : fmt(get(get(g, "picongpu_over_pm", Dict()), l, nothing))
            println(io, "| $(g["entry"]) | $(r["float"]) | $(r["particles"]) | $name | $(fmt(1e3 * s["seconds_per_step"])) | ",
                    "$(fmt(s["ns_per_particle_step"])) | $(fmt(s["ns_per_cell_step"])) | $ratio |")
        end
    end
    println(io)
    println(io, "| LWFA | code | cells computed | ms per step (steady) | loop s | ns per computed cell-step | PIConGPU / this |")
    println(io, "|---|---|---|---|---|---|---|")
    for g in gs
        g["kind"] == "lwfa" || continue
        for (l, name) in BENCH_LABELS
            r = get(g["rows"], l, nothing)
            r === nothing && continue
            ratio = l == "picongpu" ? "" : fmt(get(get(g, "picongpu_over_pm", Dict()), l, nothing))
            println(io, "| $(g["entry"]) | $name | $(join(r["cells"], " × ")) | $(fmt(1e3 * r["steady"]["seconds_per_step"])) | $(fmt(r["loop_seconds"])) | ",
                    "$(fmt(r["steady"]["ns_per_cell_step"])) | $ratio |")
        end
    end
    println(io)
    println(io, "| start-up (s) | code | load | set-up and initialization | first step | PIConGPU compile |")
    println(io, "|---|---|---|---|---|---|")
    for g in gs, (l, name) in BENCH_LABELS
        r = get(g["rows"], l, nothing)
        r === nothing && continue
        if l == "picongpu"
            println(io, "| $(g["kind"]) $(g["entry"]) | $name | – | $(fmt(r["init_seconds"])) | – | $(fmt(r["compile_seconds"])) |")
        else
            println(io, "| $(g["kind"]) $(g["entry"]) | $name | $(fmt(r["load_seconds"])) | $(fmt(r["setup_seconds"])) | $(fmt(r["first_step_seconds"])) | – |")
        end
    end
    return String(take!(io))
end

# --- command line ---------------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    cmd = get(ARGS, 1, "")
    if cmd == "prepare"
        println(prepare_bench(ARGS[2], ARGS[3]))
    elseif cmd == "run-pm"
        be = backend_named(get(ARGS, 3, "cpu"))
        dir = ARGS[2]
        kind = TOML.parsefile(joinpath(dir, "entry.toml"))["kind"]
        Base.invokelatest(kind == "uniform" ? run_pm_uniform : run_pm_lwfa, dir; backend = be)
    elseif cmd == "read-picongpu"
        read_picongpu_bench(ARGS[2])
    elseif cmd == "compare"
        compare_bench(ARGS[2], ARGS[3])
    elseif cmd == "table"
        print(bench_table(JSON.parsefile(ARGS[2])))
    else
        error("usage: bench3d.jl prepare <case.toml> <tag> | run-pm <dir> [cpu|cuda] | read-picongpu <dir> | compare <runs> <out> | table <comparison.json>")
    end
end
