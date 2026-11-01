# PIConGPU deck generator for the cases in scripts/crosscode/cases/ (PIConGPU 0.8.0, the image
# of containers/picongpu.Containerfile): schema 1 (1D) as a quasi-1D 2D run, schema 2 (2D periodic).
#
#   julia +1.13 --project=scripts scripts/crosscode/picongpu.jl prepare <case> [tag] [--precision Float32]
#
# writes scripts/runs/<tag>/ with params/*.param (the setup, copied over PIConGPU's `Empty`
# example inside the container), case.toml, units.toml (the SI units of the run, which the reader
# needs), run.sh (compile once, then run every seed) and manifest.toml. A case with a `seeds`
# list gets one subdirectory seed-<seed>/ per seed, all run by the one binary with PIC_SEED set.
# The default tag is "picongpu-<case>-<fp64|fp32>-<first 8 hex of the case hash>".
#
# Every PIConGPU-specific choice below was read from the source at the pinned release (tag
# 0.8.0, commit 6bb3470e); the list with file names is in scripts/crosscode/README.md
# ("PIConGPU specifics").

module PIConGPUDeck

using TOML

include(joinpath(@__DIR__, "smilei.jl"))
using .SmileiDeck: SmileiDeck, case_view, case_seeds
using .SmileiDeck.RunManifest: file_hash, write_manifest

export picongpu_parameters, param_files, run_arguments, run_script, prepare_run, reference_units

const CROSSCODE = @__DIR__
const SCRIPTS = dirname(CROSSCODE)
const ROOT = dirname(SCRIPTS)

# PIConGPU 0.8.0, include/picongpu/param/physicalConstants.param (CODATA 2022 values there).
const C_SI = 2.99792458e8
const MU0_SI = 1.25663706127e-6
const EPS0_SI = 1.0 / MU0_SI / C_SI / C_SI
const ME_SI = 9.1093837139e-31
const QE_SI = 1.602176634e-19     # |e|; PIConGPU's ELECTRON_CHARGE_SI is −this

"The default reference wavelength (WP-017): 0.8 µm."
const LAMBDA0_DEFAULT = 0.8e-6

"""
    reference_units(λ₀) -> NamedTuple

The SI values of A1's units for the reference frequency ω₀ = 2πc/λ₀, with PIConGPU's constants:
time 1/ω₀, length c/ω₀, density n_c = ε₀ m_e ω₀²/e², electric field m_e c ω₀/e, magnetic field
m_e ω₀/e, energy m_e c².
"""
function reference_units(λ0)
    ω0 = 2π * C_SI / λ0
    return (; lambda0 = Float64(λ0), omega0 = ω0, time = 1 / ω0, length = C_SI / ω0,
              density = EPS0_SI * ME_SI * ω0^2 / QE_SI^2, efield = ME_SI * C_SI * ω0 / QE_SI,
              bfield = ME_SI * ω0 / QE_SI, energy = ME_SI * C_SI^2)
end

"The quadratic B-spline (TSC) assignment function of PIConGPU (particles/shapes/TSC.hpp)."
function tsc(x)
    a = abs(x)
    a < 0.5 && return 0.75 - x * x
    a < 1.5 && return 0.5 * (1.5 - a)^2
    return 0.0
end

"""
    deposit_form_factor(n, kΔ) -> Float64

The factor by which TSC deposition of a quiet lattice of `n` particles per cell (in-cell
positions (j + ½)/n, QuietImpl.hpp) with weights sampled at the cell centres
(FreeFormulaImpl.hpp) multiplies a mode cos(k x) of the density at the nodes:
K̂ = (1/n) Σ_j Σ_d S(d + u_j) e^{ikΔ(d + ½)}, d the cell offset from the node. It is real
because the lattice is symmetric about the cell centre.
"""
function deposit_form_factor(n, kΔ)
    s = 0.0
    for j in 0:n-1, d in -2:1
        u = (j + 0.5) / n
        s += tsc(d + u) * cos(kΔ * (d + 0.5))
    end
    return s / n
end

"""
    transverse_cells(case) -> Int

The transverse cells of the quasi-1D box of a schema-1 case: the smallest count ≥ 7 that
divides the per-cell count of every mobile species. Seven is the floor of PIConGPU's domain
rules with one-cell supercells along y and three guard supercells (memory.param below): the
domain needs at least 2·guard + 1 supercells per axis (pmacc MappingDescription.hpp), and the
Esirkepov stencil of TSC needs a guard of 3 cells (Esirkepov2D.hpp).
"""
function transverse_cells(case)
    counts = [Int(s["particles_per_cell"]) for s in case["species"] if s["mobile"]]
    for n in 7:64
        all(c -> c % n == 0, counts) && return n
    end
    error("no transverse cell count in 7:64 divides the per-cell counts $counts")
end

"The ratio Δy/Δx of the quasi-1D box: Δy = 8Δx keeps 0.95Δx inside the 2D Courant limit (0.992Δx)."
const TRANSVERSE_RATIO = 8

valid_name(n) = occursin(r"^[A-Za-z][A-Za-z0-9_]*$", n) || error("species name $n is not a C++ identifier")

"""
    picongpu_parameters(case; precision = case's, lambda0 = 0.8e-6) -> NamedTuple

The unit conversion from a case in the units of A1 to PIConGPU's SI input, the only place where
a case quantity becomes a PIConGPU quantity; the writers only format its result.

- units: `reference_units(lambda0)`; the reference wavelength is recorded in the manifest;
- grid: schema 2 as given; schema 1 as a quasi-1D 2D box, periodic in y, with
  `transverse_cells(case)` cells of TRANSVERSE_RATIO·Δx along y (`transverse`); cell sizes
  in SI (Δx, Δy, depth), the depth equal to Δx (it cancels in 2D);
- time step as given in SI, round(t_end/dt) steps;
- per species: charge ratio to the electron charge (−q), mass ratio, density in SI, the regular
  lattice per cell (schema 1: the 1D count spread over the transverse cells, all along x), the
  profile (base, relative amplitude, mode, cells), the temperature in keV (T = m v_th² m_e c²),
  the drift as γ and a direction, and whether it is mobile;
- precision "Float64" or "Float32" (PIConGPU's precision64Bit or precision32Bit);
- supercells and guards (memory.param), diagnostics cadence, the seeds.
"""
function picongpu_parameters(case; precision = nothing, lambda0 = nothing)
    num, t, d = case["numerics"], case["time"], case["diagnostics"]
    v = case_view(case)
    units = reference_units(something(lambda0, LAMBDA0_DEFAULT))
    prec = something(precision, get(num, "precision", "Float64"))
    prec in ("Float64", "Float32") || error("precision must be Float64 or Float32, got $prec")
    Int(num["shape_order"]) == 2 || error("the cases use quadratic shapes (TSC in PIConGPU)")
    num["current_deposition"] == "esirkepov" || error("only Esirkepov deposition is generated")
    get(num, "filter_passes", 0) == 0 || error("current filtering is not generated")
    dt = Float64(t["dt"])
    steps = round(Int, Float64(t["t_end"]) / dt)
    quasi1d = v.ndims == 1
    if quasi1d
        Ny = transverse_cells(case)
        cells = (v.cells[1], Ny)
        Δ = (v.lengths[1] / v.cells[1], TRANSVERSE_RATIO * v.lengths[1] / v.cells[1])
        transverse = (; cells = Ny, ratio = TRANSVERSE_RATIO)
        supercell, guard = (16, 1), (1, 3)
    else
        cells = v.cells
        Δ = v.lengths ./ v.cells
        transverse = nothing
        supercell, guard = (8, 8), (1, 1)
    end
    all(cells[k] % supercell[k] == 0 && cells[k] ÷ supercell[k] >= 2guard[k] + 1 for k in 1:2) ||
        error("cells $cells do not fit supercells $supercell with guards $guard")
    dt < 1 / sqrt(sum(abs2, 1 ./ Δ)) || error("dt = $dt violates the 2D Courant limit of the Yee solver")
    L = units.length
    cell_size_SI = (Δ[1] * L, Δ[2] * L, Δ[1] * L)
    species = map(zip(case["species"], v.species)) do (s, sv)
        valid_name(s["name"])
        s["position"] == "regular" || error("$(s["name"]): only regular (Quiet) positions are generated")
        vel = s["velocity"]
        vel in ("maxwellian", "cold") || error("unknown velocity loading $vel")
        m = Float64(s["mass"])
        vth = Float64(get(s, "thermal_velocity", 0.0))
        vel == "maxwellian" && vth <= 0 && error("$(s["name"]): maxwellian needs thermal_velocity > 0")
        if quasi1d
            n1 = sv.particles_per_cell[1]
            lattice = (max(1, n1 ÷ cells[2]), 1)
            s["mobile"] && lattice[1] * cells[2] != n1 &&
                error("$(s["name"]): $n1 particles per cell do not spread over $(cells[2]) transverse cells")
        else
            lattice = sv.particles_per_cell
        end
        mode = sv.mode === nothing ? nothing : (quasi1d ? (sv.mode[1], 0) : sv.mode)
        profile = sv.mode === nothing ? nothing :
            (; base = sv.density, amplitude = sv.amplitude, mode, cells)
        vd = sqrt(sum(abs2, sv.drift))
        vd < 1 || error("$(s["name"]): drift ≥ c")
        drift = vd == 0 ? nothing : (; gamma = 1 / sqrt(1 - vd^2), direction = sv.drift ./ vd)
        (; name = s["name"], charge_ratio = -Float64(s["charge"]), mass_ratio = m,
           density = sv.density, density_SI = sv.density * units.density, profile, lattice,
           per_column = quasi1d ? lattice[1] * lattice[2] * cells[2] : nothing,
           temperature_keV = vel == "maxwellian" ? m * vth^2 * units.energy / QE_SI / 1e3 : 0.0,
           drift, mobile = Bool(s["mobile"]))
    end
    return (; units, lambda0 = units.lambda0, precision = prec, quasi1d, transverse, cells,
              cell_size = Δ, cell_size_SI, dt, dt_SI = dt * units.time, steps,
              supercell, guard, species,
              scalars_every = Int(d["scalars_every"]), fields_every = Int(d["fields_every"]),
              seeds = case_seeds(case), seeded = haskey(num, "seeds"), case_name = case["name"])
end

"""
    picongpu_density(p, s, i, j) -> Float64

The density in m⁻³ that the generated functor of species `s` returns for cell (i, j) (0-based),
evaluated as PIConGPU evaluates it: FreeFormulaImpl passes the cell centre (i + ½, j + ½) times
the cell size in SI, and the functor forms the phase from position/cellSize. Float64 arithmetic
in the order of the generated C++.
"""
function picongpu_density(p, s, i, j)
    pr = s.profile
    pr === nothing && return p.units.density * s.density
    csx, csy = p.cell_size_SI[1], p.cell_size_SI[2]
    px, py = (i + 0.5) * csx, (j + 0.5) * csy
    phase = 2.0 * π * (Float64(pr.mode[1]) * (px / csx) / Float64(pr.cells[1]) +
                       Float64(pr.mode[2]) * (py / csy) / Float64(pr.cells[2]))
    return p.units.density * (pr.base * (1.0 + pr.amplitude * cos(phase)))
end

"""
    background_field(p) -> NamedTuple

The electrostatic field of the charge PIConGPU loads at t = 0, which PIConGPU does not compute
(it has no Poisson solve). The perturbed density of each species, sampled at cell centres and
loaded on a quiet lattice with TSC, deposits at the nodes ρ₁ = ρ̂ cos(k·x_node) exactly, with
ρ̂ = Σ_s q_s n_s α_s K̂_x K̂_y (`deposit_form_factor` per axis), and the field of the Yee grid that
satisfies the discrete Gauss law is E_d = (ρ̂/K²) κ_d sin(k·x_E_d), κ_d = (2/Δ_d) sin(k_dΔ_d/2),
K² = Σ κ_d², at E_x's points (i + ½, j) and E_y's (i, j + ½). Returns the normalized amplitudes
`ax`, `ay` (m_e c ω₀/e), the mode and cells, and functions `Ex(i, j)`, `Ey(i, j)`; `nothing`
amplitudes when no species is perturbed. The generator passes it as FieldBackgroundE, which
the pusher sees and the field solver does not (simulation/stage/FieldBackground.hpp): the total
field then obeys Maxwell's equations with ∇·E = ρ from t = 0, as after Smilei's Poisson solve.
"""
function background_field(p)
    perturbed = [s for s in p.species if s.profile !== nothing && s.profile.amplitude != 0]
    if isempty(perturbed)
        z(i, j) = 0.0
        return (; ax = 0.0, ay = 0.0, mode = (0, 0), cells = p.cells, Ex = z, Ey = z)
    end
    mode = only(unique(s.profile.mode for s in perturbed))
    N, Δ = p.cells, p.cell_size
    kΔ = 2π .* mode ./ N
    ρ̂ = sum(-s.charge_ratio * s.density * s.profile.amplitude *
            deposit_form_factor(s.lattice[1], kΔ[1]) * deposit_form_factor(s.lattice[2], kΔ[2]) for s in perturbed)
    κ = 2 .* sin.(kΔ ./ 2) ./ Δ
    Φ = ρ̂ / sum(abs2, κ)
    ax, ay = Φ * κ[1], Φ * κ[2]
    θ(x, y) = 2π * (mode[1] * x / N[1] + mode[2] * y / N[2])
    return (; ax, ay, mode, cells = N, Ex = (i, j) -> ax * sin(θ(i + 0.5, j)), Ey = (i, j) -> ay * sin(θ(i, j + 0.5)))
end

# Shortest round-trip decimal form, which C++ parses to the same double.
cfloat(x::Real) = (isfinite(x) || error("non-finite value"); s = string(Float64(x)); occursin(r"[.eE]", s) ? s : s * ".0")

const HEADER = "/* Generated by scripts/crosscode/picongpu.jl; do not edit, regenerate from the case file. */\n"

"""
    param_files(case; precision, lambda0) -> Dict{String,String}

The .param files of the setup, file name => content, from `picongpu_parameters`. They replace
PIConGPU's defaults of the same names in the `Empty` example; every other default stays.
"""
function param_files(case; precision = nothing, lambda0 = nothing)
    p = picongpu_parameters(case; precision, lambda0)
    f = Dict{String,String}()
    f["dimension.param"] = HEADER * """
    #pragma once
    #define SIMDIM DIM2
    namespace picongpu
    {
        constexpr uint32_t simDim = SIMDIM;
    } // namespace picongpu
    """
    bits = p.precision == "Float64" ? "precision64Bit" : "precision32Bit"
    f["precision.param"] = HEADER * """
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
    typical = maximum(prod(s.lattice) for s in p.species if s.mobile)
    f["simulation.param"] = HEADER * """
    #pragma once
    namespace picongpu
    {
        namespace SI
        {
            constexpr float_64 DELTA_T_SI = $(cfloat(p.dt_SI));
            constexpr float_64 CELL_WIDTH_SI = $(cfloat(p.cell_size_SI[1]));
            constexpr float_64 CELL_HEIGHT_SI = $(cfloat(p.cell_size_SI[2]));
            constexpr float_64 CELL_DEPTH_SI = $(cfloat(p.cell_size_SI[3]));
            /* n_c at the reference wavelength $(cfloat(p.lambda0)) m; species profiles are relative to it */
            constexpr float_64 BASE_DENSITY_SI = $(cfloat(p.units.density));
        } // namespace SI
        constexpr uint32_t TYPICAL_PARTICLES_PER_CELL = $(typical)u;
    } // namespace picongpu
    """
    sx, sy = p.supercell
    gx, gy = p.guard
    f["memory.param"] = HEADER * """
    #pragma once
    #include <pmacc/mappings/kernel/MappingDescription.hpp>
    #include <pmacc/math/Vector.hpp>
    #include <array>
    namespace picongpu
    {
        constexpr size_t reservedGpuMemorySize = 350 * 1024 * 1024;
        namespace mCT = pmacc::math::CT;
        using SuperCellSize = typename mCT::shrinkTo<mCT::Int<$sx, $sy, 1>, simDim>::type;
        static constexpr uint32_t numFrameSlots = pmacc::math::CT::volume<SuperCellSize>::type::value;
        using MappingDesc = MappingDescription<simDim, SuperCellSize>;
        using GuardSize = typename mCT::shrinkTo<mCT::Int<$gx, $gy, 1>, simDim>::type;
        struct DefaultExchangeMemCfg
        {
            static constexpr uint32_t BYTES_EXCHANGE_X = 16 * 1024 * 1024;
            static constexpr uint32_t BYTES_EXCHANGE_Y = 16 * 1024 * 1024;
            static constexpr uint32_t BYTES_EXCHANGE_Z = 1 * 1024 * 1024;
            static constexpr uint32_t BYTES_EDGES = 1 * 1024 * 1024;
            static constexpr uint32_t BYTES_CORNER = 256 * 1024;
            using REF_LOCAL_DOM_SIZE = mCT::Int<0, 0, 0>;
            const std::array<float_X, 3> DIR_SCALING_FACTOR = {{0.0, 0.0, 0.0}};
        };
        constexpr uint32_t fieldTmpNumSlots = 1;
        constexpr bool fieldTmpSupportGatherCommunication = true;
    } // namespace picongpu
    """
    f["species.param"] = HEADER * """
    #pragma once
    #include "picongpu/algorithms/AssignedTrilinearInterpolation.hpp"
    #include "picongpu/algorithms/FieldToParticleInterpolation.hpp"
    #include "picongpu/fields/currentDeposition/Solver.def"
    #include "picongpu/particles/shapes.hpp"
    namespace picongpu
    {
        /* A5 matching: quadratic shape (TSC), Esirkepov current, Boris pusher */
        using UsedParticleShape = particles::shapes::TSC;
        using UsedField2Particle = FieldToParticleInterpolation<UsedParticleShape, AssignedTrilinearInterpolation>;
        using UsedParticleCurrentSolver = currentSolver::Esirkepov<UsedParticleShape>;
        using UsedParticlePusher = particles::pusher::Boris;
    } // namespace picongpu
    """
    f["fieldSolver.param"] = HEADER * """
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
    f["random.param"] = HEADER * """
    #pragma once
    #include "picongpu/random/seed/Seed.hpp"
    #include <pmacc/random/methods/methods.hpp>
    namespace picongpu
    {
        namespace random
        {
            using Generator = pmacc::random::methods::XorMin<>;
            /* the seed is read from PIC_SEED at start-up, so one binary runs every seed */
            using SeedGenerator = seed::FromEnvironment;
        } // namespace random
    } // namespace picongpu
    """
    io = IOBuffer()
    print(io, HEADER, """
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
    """)
    for s in p.species
        n = s.name
        flags = s.mobile ?
            "particlePusher<UsedParticlePusher>,\n        shape<UsedParticleShape>,\n        interpolation<UsedField2Particle>,\n        current<UsedParticleCurrentSolver>,\n        " :
            "shape<UsedParticleShape>,\n        interpolation<UsedField2Particle>,\n        "
        print(io, """
            value_identifier(float_X, MassRatio_$n, $(cfloat(s.mass_ratio)));
            value_identifier(float_X, ChargeRatio_$n, $(cfloat(s.charge_ratio)));
            /* $(s.mobile ? "mobile" : "immobile: no pusher and no current deposition") */
            using ParticleFlags_$n = MakeSeq_t<
                $(flags)massRatio<MassRatio_$n>,
                chargeRatio<ChargeRatio_$n>>;
            using PIC_$n = Particles<PMACC_CSTRING("$n"), ParticleFlags_$n, DefaultParticleAttributes>;
        """)
    end
    print(io, "    using VectorAllSpecies = MakeSeq_t<", join(("PIC_" * s.name for s in p.species), ", "), ">;\n} // namespace picongpu\n")
    f["speciesDefinition.param"] = String(take!(io))

    io = IOBuffer()
    print(io, HEADER, """
    #pragma once
    #include "picongpu/particles/densityProfiles/profiles.def"
    namespace picongpu
    {
        namespace densityProfiles
        {
    """)
    for s in p.species
        n, pr = s.name, s.profile
        mx, my = pr === nothing ? (0, 0) : pr.mode
        amp = pr === nothing ? 0.0 : pr.amplitude
        print(io, """
                /* n(x, y) = BASE n_c (1 + AMPLITUDE cos(2π (MX i/NX + MY j/NY))) at the cell centre (i + ½, j + ½) */
                constexpr float_64 BASE_$n = $(cfloat(s.density));
                constexpr float_64 AMPLITUDE_$n = $(cfloat(amp));
                constexpr float_64 MX_$n = $mx;
                constexpr float_64 MY_$n = $my;
                constexpr float_64 NX_$n = $(p.cells[1]);
                constexpr float_64 NY_$n = $(p.cells[2]);
                struct ProfileFunctor_$n
                {
                    HDINLINE float_X operator()(const floatD_64& position_SI, const float3_64& cellSize_SI)
                    {
                        const float_64 phase = 2.0 * PI
                            * (MX_$n * (position_SI.x() / cellSize_SI.x()) / NX_$n
                               + MY_$n * (position_SI.y() / cellSize_SI.y()) / NY_$n);
                        return static_cast<float_X>(BASE_$n * (1.0 + AMPLITUDE_$n * math::cos(phase)));
                    }
                };
                using Profile_$n = FreeFormulaImpl<ProfileFunctor_$n>;
        """)
    end
    print(io, "    } // namespace densityProfiles\n} // namespace picongpu\n")
    f["density.param"] = String(take!(io))

    io = IOBuffer()
    print(io, HEADER, """
    #pragma once
    #include "picongpu/particles/filter/filter.def"
    #include "picongpu/particles/manipulators/manipulators.def"
    #include "picongpu/particles/startPosition/functors.def"
    #include <pmacc/math/operation.hpp>
    namespace picongpu
    {
        namespace particles
        {
            /* weightings are far below PIConGPU's default floor of 10 at these densities and cells;
             * QuietImpl would drop particles below it */
            constexpr float_X MIN_WEIGHTING = 1.0e-20;
            namespace startPosition
            {
    """)
    for s in p.species
        print(io, """
                    struct QuietParam_$(s.name)
                    {
                        using numParticlesPerDimension = mCT::shrinkTo<mCT::Int<$(s.lattice[1]), $(s.lattice[2]), 1>, simDim>::type;
                    };
                    using Quiet_$(s.name) = QuietImpl<QuietParam_$(s.name)>;
        """)
    end
    print(io, "        } // namespace startPosition\n        namespace manipulators\n        {\n")
    for s in p.species
        if s.temperature_keV > 0
            print(io, """
                        struct TemperatureParam_$(s.name)
                        {
                            /* T = m v_th^2 in keV; non-relativistic Maxwellian in momentum (Temperature.hpp) */
                            static constexpr float_64 temperature = $(cfloat(s.temperature_keV));
                        };
                        using AddTemperature_$(s.name) = unary::Temperature<TemperatureParam_$(s.name)>;
            """)
        end
        if s.drift !== nothing
            dx, dy, dz = s.drift.direction
            print(io, """
                        struct DriftParam_$(s.name)
                        {
                            static constexpr float_64 gamma = $(cfloat(s.drift.gamma));
                            static constexpr auto driftDirection = float3_X($(cfloat(dx)), $(cfloat(dy)), $(cfloat(dz)));
                        };
                        using AssignDrift_$(s.name) = unary::Drift<DriftParam_$(s.name), pmacc::math::operation::Assign>;
            """)
        end
    end
    print(io, "        } // namespace manipulators\n    } // namespace particles\n} // namespace picongpu\n")
    f["particle.param"] = String(take!(io))

    steps = String[]
    for s in p.species
        push!(steps, "CreateDensity<densityProfiles::Profile_$(s.name), startPosition::Quiet_$(s.name), PIC_$(s.name)>")
        s.drift === nothing || push!(steps, "Manipulate<manipulators::AssignDrift_$(s.name), PIC_$(s.name)>")
        s.temperature_keV > 0 && push!(steps, "Manipulate<manipulators::AddTemperature_$(s.name), PIC_$(s.name)>")
    end
    f["speciesInitialization.param"] = HEADER * """
    #pragma once
    #include "picongpu/particles/InitFunctors.hpp"
    namespace picongpu
    {
        namespace particles
        {
            using InitPipeline = pmacc::mp_list<
                $(join(steps, ",\n            "))>;
        } // namespace particles
    } // namespace picongpu
    """

    bg = background_field(p)
    on = bg.ax != 0 || bg.ay != 0
    f["fieldBackground.param"] = HEADER * """
    #pragma once
    namespace picongpu
    {
        /* The electrostatic field of the charge loaded at t = 0 (PIConGPU solves no Poisson
         * equation): E_x = AX sin(2π(MX (i + ½)/NX + MY j/NY)) at (i + ½, j), E_y = AY sin(2π(MX i/NX +
         * MY (j + ½)/NY)) at (i, j + ½), the discrete solution for the deposited charge
         * (scripts/crosscode/picongpu.jl, background_field). Static, seen by the pusher; the
         * grid field starts at zero and the total field obeys ∇·E = ρ. */
        class FieldBackgroundE
        {
        public:
            static constexpr bool InfluenceParticlePusher = $(on ? "true" : "false");
            static constexpr float_64 AX_SI = $(cfloat(bg.ax * p.units.efield));
            static constexpr float_64 AY_SI = $(cfloat(bg.ay * p.units.efield));
            static constexpr float_64 MX = $(bg.mode[1]);
            static constexpr float_64 MY = $(bg.mode[2]);
            static constexpr float_64 NX = $(p.cells[1]);
            static constexpr float_64 NY = $(p.cells[2]);
            PMACC_ALIGN(m_unitField, const float3_64);
            HINLINE FieldBackgroundE(const float3_64 unitField) : m_unitField(unitField)
            {
            }
            HDINLINE FieldBackgroundE(const FieldBackgroundE&) = default;
            HDINLINE float3_X operator()(const DataSpace<simDim>& cellIdx, const uint32_t currentStep) const
            {
                const float_64 i = cellIdx.x();
                const float_64 j = cellIdx.y();
                const float_64 ex = AX_SI * math::sin(2.0 * PI * (MX * (i + 0.5) / NX + MY * j / NY));
                const float_64 ey = AY_SI * math::sin(2.0 * PI * (MX * i / NX + MY * (j + 0.5) / NY));
                return float3_X(
                    static_cast<float_X>(ex / m_unitField[0]),
                    static_cast<float_X>(ey / m_unitField[1]),
                    0.0_X);
            }
        };
        class FieldBackgroundB
        {
        public:
            static constexpr bool InfluenceParticlePusher = false;
            PMACC_ALIGN(m_unitField, const float3_64);
            HINLINE FieldBackgroundB(const float3_64 unitField) : m_unitField(unitField)
            {
            }
            HDINLINE FieldBackgroundB(const FieldBackgroundB&) = default;
            HDINLINE float3_X operator()(const DataSpace<simDim>& cellIdx, const uint32_t currentStep) const
            {
                return float3_X(0.0_X, 0.0_X, 0.0_X);
            }
        };
        class FieldBackgroundJ
        {
        public:
            static constexpr bool activated = false;
            PMACC_ALIGN(m_unitField, const float3_64);
            HINLINE FieldBackgroundJ(const float3_64 unitField) : m_unitField(unitField)
            {
            }
            HDINLINE FieldBackgroundJ(const FieldBackgroundJ&) = default;
            HDINLINE float3_X operator()(const DataSpace<simDim>& cellIdx, const uint32_t currentStep) const
            {
                return float3_X(0.0_X, 0.0_X, 0.0_X);
            }
        };
    } // namespace picongpu
    """
    f["fileOutput.param"] = HEADER * """
    #pragma once
    #include <pmacc/meta/conversion/MakeSeq.hpp>
    #include "picongpu/fields/Fields.def"
    #include "picongpu/particles/filter/filter.def"
    #include "picongpu/particles/particleToGrid/CombinedDerive.def"
    #include "picongpu/particles/particleToGrid/ComputeGridValuePerFrame.def"
    namespace picongpu
    {
        namespace deriveField = particles::particleToGrid;
        /* E, B and the charge density of every species (no energy densities, no particles) */
        using ChargeDensity_Seq
            = deriveField::CreateEligible_t<VectorAllSpecies, deriveField::derivedAttributes::ChargeDensity>;
        using FieldTmpSolvers = MakeSeq_t<ChargeDensity_Seq>;
        using NativeFileOutputFields = MakeSeq_t<FieldE, FieldB>;
        using FileOutputFields = MakeSeq_t<NativeFileOutputFields, FieldTmpSolvers>;
        using FileOutputParticles = MakeSeq_t<>;
    } // namespace picongpu
    """
    return f
end

"""
    run_arguments(p) -> String

The command-line options of the run: one device, the grid, the step count, periodic boundaries,
no current interpolation (filter), the duplicated field buffers of the background (so that the
subtraction after the push restores the grid field bit for bit), openPMD field dumps (E, B and
the charge densities, HDF5, all iterations in one file), the field energies per component, the
kinetic energy of each mobile species, and the charge-conservation check.
"""
function run_arguments(p)
    a = ["-d 1 1", "-g $(p.cells[1]) $(p.cells[2])", "-s $(p.steps)", "--periodic 1 1",
         "--currentInterpolation none", "--fieldBackground.duplicateFields",
         "--openPMD.period $(p.fields_every)", "--openPMD.file simData", "--openPMD.ext h5",
         "--openPMD.infix NULL", "--openPMD.source fields_all",
         "--fields_energy.period $(p.scalars_every)",
         "--chargeConservation.period $(p.fields_every)"]
    for s in p.species
        s.mobile && push!(a, "--$(s.name)_energy.period $(p.scalars_every)", "--$(s.name)_energy.filter all")
    end
    return join(a, " ") * " "
end

"""
The script the container runs (runner: `bash /run-dir/run.sh`, the GPU bound in): record the
toolchain and the GPU, compile the setup once (`pic-create` of `Empty`, the generated params
copied over it, `pic-build`), then run each seed (or the single run) in its directory with
PIC_SEED set, timing compile and runs separately in container.txt.
"""
function run_script(p)
    dirs = p.seeded ? ["seed-$s" for s in p.seeds] : ["."]
    list = join(("$d:$s" for (d, s) in zip(dirs, p.seeds)), " ")
    stamp = "\$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
    return """
    #!/usr/bin/env bash
    # Generated by scripts/crosscode/picongpu.jl. Runs inside the PIConGPU container with the run
    # directory at /run-dir (PM_RUN_DIR), the worktree read-only at /worktree and the GPU bound in.
    set -euo pipefail
    top="\${PM_RUN_DIR:-/run-dir}"
    cd "\$top"
    pm-versions > versions.toml
    {
      echo "# picongpu: \$PICONGPU_TAG \$PICONGPU_COMMIT, backend \$PIC_BACKEND"
      echo "# gpu: \$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo unknown)"
      echo "# cpus: \$(nproc)"
      echo "# cpu model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
      echo "# compile start: $stamp"
    } > container.txt
    rm -rf setup
    pic-create "\$PIC_EXAMPLES/Empty" setup > build.log 2>&1
    cp params/*.param setup/include/picongpu/param/
    (cd setup && pic-build -j 16) >> build.log 2>&1 || { tail -n 60 build.log; exit 1; }
    echo "# compile end: $stamp" >> container.txt
    for entry in $list; do
      d=\${entry%%:*}; seed=\${entry##*:}
      cd "\$top/\$d"
      rm -rf openPMD ./*.dat
      echo "# run \$d seed \$seed start: $stamp" >> "\$top/container.txt"
      PIC_SEED=\$seed mpiexec -n 1 "\$top/setup/bin/picongpu" $(run_arguments(p))> picongpu.log 2>&1 \\
        || { tail -n 60 picongpu.log; exit 1; }
      echo "# run \$d seed \$seed end: $stamp" >> "\$top/container.txt"
    done
    """
end

"The units.toml of a run: what the reader needs to bring PIConGPU's SI output to the units of A1."
function units_record(p)
    depth = p.cell_size[1]                       # CELL_DEPTH_SI / (c/ω₀)
    transverse = p.quasi1d ? p.cells[2] * p.cell_size[2] : 1.0
    return Dict{String,Any}(
        "lambda0" => p.lambda0, "omega0" => p.units.omega0, "time" => p.units.time,
        "length" => p.units.length, "density" => p.units.density, "efield" => p.units.efield,
        "bfield" => p.units.bfield, "charge_density" => QE_SI * p.units.density,
        "energy" => p.units.energy * p.units.density * p.units.length^3 * depth * transverse,
        "energy_note" => "J per unit of the energy scalars: n_c m_e c² (c/ω₀)³ × depth" *
            (p.quasi1d ? " × transverse length (per unit area, as in 1D)" : " (per unit depth, as in 2D)"),
        "quasi1d" => p.quasi1d, "cells" => collect(p.cells), "cell_size" => collect(p.cell_size),
        "dt" => p.dt, "steps" => p.steps, "precision" => p.precision,
        "background" => (bg = background_field(p); Dict{String,Any}("ax" => bg.ax, "ay" => bg.ay, "mode" => collect(bg.mode))))
end

"""
    prepare_run(case_name; tag, precision, lambda0) -> dir

Write the run directory under scripts/runs/<tag>/ and its manifest.
"""
function prepare_run(name; tag = nothing, precision = nothing, lambda0 = nothing, runs = joinpath(SCRIPTS, "runs"))
    case = SmileiDeck.load_case(name)
    p = picongpu_parameters(case; precision, lambda0)
    h = file_hash(case["_path"])
    tag = something(tag, "picongpu-$(case["name"])-$(lowercase(replace(p.precision, "Float" => "fp")))-$(h[1:8])")
    dir = joinpath(runs, tag)
    mkpath(joinpath(dir, "params"))
    cp(case["_path"], joinpath(dir, "case.toml"); force = true)
    files = param_files(case; precision, lambda0)
    for (n, text) in files
        write(joinpath(dir, "params", n), text)
    end
    if p.seeded
        for s in p.seeds
            mkpath(joinpath(dir, "seed-$s"))
            cp(case["_path"], joinpath(dir, "seed-$s", "case.toml"); force = true)
        end
    end
    open(io -> TOML.print(io, units_record(p); sorted = true), joinpath(dir, "units.toml"), "w")
    write(joinpath(dir, "run.sh"), run_script(p))
    chmod(joinpath(dir, "run.sh"), 0o755)
    recipe = joinpath(CROSSCODE, "containers", "picongpu.Containerfile")
    write_manifest(dir; root = ROOT,
        inputs = ["case.toml", "units.toml", "run.sh", ("params/" .* sort(collect(keys(files))))...],
        extra = Dict{String,Any}("seeds" => p.seeds, "run" => tag, "code" => "picongpu",
            "case" => case["name"], "case_source" => relpath(case["_path"], ROOT),
            "precision" => p.precision, "reference_wavelength_m" => p.lambda0,
            "quasi1d" => p.quasi1d, "cells" => collect(p.cells),
            "transverse" => p.transverse === nothing ? "none" : Dict("cells" => p.transverse.cells, "ratio" => p.transverse.ratio),
            "arguments" => strip(run_arguments(p)),
            "generator" => "scripts/crosscode/picongpu.jl",
            "generator_hash" => "sha256:" * file_hash(@__FILE__),
            "image" => Dict{String,Any}(
                "recipe" => "scripts/crosscode/containers/picongpu.Containerfile",
                "recipe_hash" => "sha256:" * file_hash(recipe),
                "id" => "pending: written by the analyze step from the runner log")))
    return dir
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .PIConGPUDeck
    cmd = get(ARGS, 1, "")
    rest = ARGS[2:end]
    i = findfirst(==("--precision"), rest)
    prec = i === nothing ? nothing : rest[i+1]
    i === nothing || deleteat!(rest, i:i+1)
    if cmd == "prepare" && length(rest) >= 1
        println(prepare_run(rest[1]; tag = get(rest, 2, nothing), precision = prec))
    elseif cmd == "params" && length(rest) >= 1
        for (n, t) in sort(collect(param_files(PIConGPUDeck.SmileiDeck.load_case(rest[1]); precision = prec)))
            println("=== ", n); print(t)
        end
    else
        println(stderr, "usage: picongpu.jl prepare <case> [tag] [--precision Float32] | params <case>")
        exit(2)
    end
end
