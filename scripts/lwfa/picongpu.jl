# The PIConGPU 0.8.0 column of the LWFA case (included by scripts/lwfa/lwfa.jl): the deck, the
# run script, and the reader into lwfa.jl's summary format. Every PIConGPU choice below was read
# in the source at the image's pin (tag 0.8.0, commit 6bb3470e); paths are under include/picongpu/.
#
# - Axes. PIConGPU's moving window slides only along y (`simulation/control/MovingWindow.hpp`), so
#   the case's propagation axis x is PIConGPU's y and the case's transverse y is PIConGPU's x. The
#   swap keeps every Yee point where it was: the case's E_x (longitudinal, dual along the
#   propagation axis) is PIConGPU's E_y at (i, j + ½), the case's E_y (transverse, the laser's
#   polarization) is PIConGPU's E_x at (i + ½, j), the charge density sits on the nodes in both.
#   The reader swaps them back, so a PIConGPU summary has the case's names and orientation.
# - The lab lattice. As in lwfa.jl, a node at lab x = m·Δx for an integer m. PIConGPU's node of
#   total cell index j (counted over every slide) is m = j − j0, j0 = HUYGENS + 1: x = 0 is the
#   first node in front of the Huygens surface. The openPMD dumps hold the moving window only, with
#   its total offset in `gridGlobalOffset` (window offset plus slides × local domain,
#   `plugins/openPMD/openPMDWriter.x.cpp`), which the reader turns into m.
# - The window. PIConGPU keeps a hidden row of local domains ahead of the window, so the window
#   is (devices along y − 1) local domains and needs at least two (`Simulation.hpp` refuses one).
#   The run puts all of them on the one GPU (`-r`, ranks per device). The window's back moves at
#   c from the step where a virtual photon, started `ceil(W(1 − movePoint))` cells into the box,
#   passes the window (`MovingWindow::getCurrentSlideInfo`); `--windowMovePoint` is chosen so that
#   the back is at lab x = c(t − start), the case's window. The data slides by a whole local
#   domain at a time, always behind the window's back.
# - The laser. A `GaussianPulse` incident field on the YMin side (total-field/scattered-field,
#   `fields/incidentField/`), Huygens surface at (HUYGENS + 0.75)Δ (`Functors.hpp`, `getOrigin`),
#   a quarter cell behind lab x = 0. Its field envelope is exp(−(t'/(2·PULSE_DURATION))²) with t'
#   counted from PULSE_INIT·PULSE_DURATION/2 (`GaussianPulse.hpp`), so PULSE_DURATION = τ/2 and
#   PULSE_INIT puts the envelope's peak on x = 0 at t = delay, as in the other two codes; it is
#   zero before t' = 0 there, e^−16 of the peak, where ParticleMesh's antenna starts too. W0 is the
#   1/e field radius at the focus (the case's waist), the 2D amplitude falls as √(w0/w) and the
#   Gouy phase is halved in 2D. The focus is on the axis (`Origin::Center` along x). The transverse
#   axis is periodic, so the Huygens surface is made contiguous across it
#   (`MakePeriodicTransversalHuygensSurfaceContiguous`).
# - The plasma. Electrons only: PIConGPU solves no Poisson equation and immobile ions deposit no
#   current, so frozen ions would enter nothing but the charge-density output; the reader reads the
#   electron density alone, as Smilei's `Rho_electron`. The density is a `FreeFormula` evaluated at
#   the cell centre (`FreeFormulaImpl.hpp`), the load a quiet lattice (`QuietImpl`), cold. The
#   window's new cells are loaded by the same pipeline after each slide. Below the case's three
#   load rules (lwfa.jl's header) this is a third: the density at the cell centre.
# - Boundaries. Periodic transversally (`--periodic 1 0`); along y PIConGPU's exponential absorber
#   (`fieldAbsorber.param` default, 12 cells), which PIConGPU switches off at the front while the
#   window moves (`fields/absorber/exponential/Exponential.hpp`). The Huygens surface must sit
#   inside the absorber's offset; HUYGENS = 16 leaves 4 cells.
# - Numerics. Yee, Esirkepov, TSC, Boris, no current filter (`--currentInterpolation none`),
#   Float64 by default (`precision64Bit`), supercells of 8 × 8.
# - Diagnostics. openPMD (HDF5) dumps of E and the electron charge density every `fields_every`
#   steps; the field energies; the electron energy histogram (`plugins/BinEnergyParticles.x.cpp`,
#   in keV, real electrons per bin, with one underflow and one overflow column). The histogram
#   covers the whole simulated domain, hidden row included, while ParticleMesh's and Smilei's
#   cover their windows; the hidden row holds cold plasma only, which adds to the first bins.
#   Its energy is the kinetic energy of `algorithms/KinEnergy.hpp`, p²/2m below γ = 1.005.

include(joinpath(REPO, "scripts", "crosscode", "picongpu.jl"))
using .PIConGPUDeck: reference_units

const PIC_KINDS = ("picongpu-vacuum", "picongpu-plasma")

"Cells from the back of PIConGPU's box to its Huygens surface (the incident field's POSITION)."
const HUYGENS = 16
"PIConGPU's default exponential absorber thickness (`param/fieldAbsorber.param`), which the Huygens surface must clear."
const PIC_ABSORBER = 12
const PIC_SUPERCELL = 8
const PIC_SPECIES = "e"
"Local domains along y unless the case's `grid.picongpu_devices` says otherwise: a window of four, one hidden."
const PIC_DEVICES_DEFAULT = 5

"""
    picongpu_layout(case) -> NamedTuple

PIConGPU's box for the case: `devices` local domains along y (the case's `grid.picongpu_devices`,
default 5), all of `local_y` cells, the window `window` = (devices − 1)·local_y cells long, at
least the case's window plus the j0 cells behind x = 0; `start_cell` and `move_point` place the
window's back at lab x = c(t − start).
"""
function picongpu_layout(case)
    d = derived(case)
    g = case["grid"]
    g["transverse"] == "periodic" || error("only a periodic transverse axis is generated for PIConGPU")
    HUYGENS >= PIC_ABSORBER + 1 || error("the Huygens surface must clear the absorber")
    j0 = HUYGENS + 1
    devices = Int(get(g, "picongpu_devices", PIC_DEVICES_DEFAULT))
    devices >= 2 || error("PIConGPU's moving window needs at least two local domains along y, got $devices")
    d.ny % PIC_SUPERCELL == 0 || error("the transverse cells ($(d.ny)) must be a multiple of the supercell $PIC_SUPERCELL")
    local_y = cld(cld(d.nx + j0, devices - 1), PIC_SUPERCELL) * PIC_SUPERCELL
    window = (devices - 1) * local_y
    lag = round(Int, case["window"]["start"] / d.dx)          # c = 1: cells the light front travels before the window moves
    start_cell = window + j0 - lag                           # the virtual photon's start, ceil(W(1 − movePoint))
    0 <= start_cell <= window || error("window start $(case["window"]["start"]) does not fit a window of $window cells")
    move_point = 1 - (start_cell - 0.5) / window             # ceil(W(1 − movePoint)) = start_cell, safe from rounding
    return (; j0, devices, local_y, window, cells = (d.ny, devices * local_y), lag, start_cell, move_point)
end

"""
    picongpu_window_back(case, k) -> Int

The lab m of the first node of PIConGPU's window in the output of step k, from
`MovingWindow::getCurrentSlideInfo` and `getWindow` (the offset is the virtual photon's position
after step k): before the window moves, the box's first node, m = −j0.
"""
function picongpu_window_back(case, k)
    d, lay = derived(case), picongpu_layout(case)
    way = d.dt / d.dx                                        # cells per step at c = 1
    first_move = ceil(Int, (lay.window - lay.start_cell) / way) - 1
    k < first_move && return -lay.j0
    return floor(Int, way * (k + 1)) + lay.start_cell - lay.window - lay.j0
end

"""
    picongpu_density(case, j) -> Float64

The density in n_c that the generated functor returns for total cell j along y (0-based, every
slide counted), evaluated as PIConGPU evaluates it: at the cell centre, lab x = (j + ½ − j0)Δx.
"""
function picongpu_density(case, j)
    d, lay, P = derived(case), picongpu_layout(case), case["plasma"]
    x = ((j + 0.5) - lay.j0) * d.dx
    s = (x - Float64(P["vacuum"])) / Float64(P["ramp"])
    s = s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s)
    r = sin(0.5 * π * s)
    return Float64(P["density"]) * r * r
end

"""
    picongpu_parameters(case; plasma, precision) -> NamedTuple

The case as PIConGPU input: the units of A1 in SI (`reference_units` at the case's λ₀, with
PIConGPU's constants), cell sizes (x transverse, y longitudinal, the depth equal to Δx of the
case), the step, the laser's SI parameters, the energy histogram in keV, the layout.
"""
function picongpu_parameters(case; plasma = true, precision = "Float64")
    precision in ("Float64", "Float32") || error("precision must be Float64 or Float32, got $precision")
    d, lay = derived(case), picongpu_layout(case)
    L, P, D = case["laser"], case["plasma"], case["diagnostics"]
    polarization_angle(case)
    P["ramp"] > 0 || error("the density ramp must have a positive length")
    u = reference_units(Float64(case["lambda0_um"]) * 1e-6)
    cell = (d.dy, d.dx)                                      # PIConGPU (x, y) = the case's (y, x)
    σ = Float64(L["duration"]) / 2                           # PULSE_DURATION: σ of the intensity, field e-folding τ = 2σ
    t_surface = Float64(L["delay"]) - 0.25 * d.dx            # the peak on the Huygens surface, a quarter cell behind x = 0
    t_surface > 0 || error("the laser delay must exceed a quarter cell")
    laser = (; wavelength_SI = u.lambda0, amplitude_SI = Float64(L["a0"]) * u.efield, pulse_duration_SI = σ * u.time,
             pulse_init = 2 * t_surface / σ, w0_SI = Float64(L["waist"]) * u.length,
             focus_y_SI = (lay.j0 * d.dx + Float64(L["focus"])) * u.length)
    s = D["spectrum_ekin"]
    keV = u.energy / PIConGPUDeck.QE_SI / 1e3                # m_e c² in keV, with PIConGPU's constants
    ppc = Int.(P["particles_per_cell"])
    return (; units = u, precision, plasma, layout = lay, cells = lay.cells, cell, dt = d.dt,
              cell_SI = (cell[1] * u.length, cell[2] * u.length, d.dx * u.length), dt_SI = d.dt * u.time,
              steps = d.steps, every = d.every, laser, density = Float64(P["density"]),
              vacuum = Float64(P["vacuum"]), ramp = Float64(P["ramp"]), dx = d.dx,
              lattice = (ppc[2], ppc[1]),                    # numParticlesPerDimension (x, y) = (transverse, longitudinal)
              histogram = (; bins = Int(s[3]), min_keV = Float64(s[1]) * keV, max_keV = Float64(s[2]) * keV))
end

"""
    picongpu_param_files(case; plasma, precision) -> Dict{String,String}

The .param files of the setup, copied over PIConGPU's `Empty` example; every other default stays.
"""
function picongpu_param_files(case; plasma = true, precision = "Float64")
    p = picongpu_parameters(case; plasma, precision)
    cf, H = PIConGPUDeck.cfloat, PIConGPUDeck.HEADER * "/* LWFA case \"$(case["name"])\" ($(plasma ? "plasma" : "vacuum")), scripts/lwfa/picongpu.jl */\n"
    f = Dict{String,String}()
    f["dimension.param"] = H * """
    #pragma once
    #define SIMDIM DIM2
    namespace picongpu
    {
        constexpr uint32_t simDim = SIMDIM;
    } // namespace picongpu
    """
    bits = p.precision == "Float64" ? "precision64Bit" : "precision32Bit"
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
            constexpr float_64 DELTA_T_SI = $(cf(p.dt_SI));
            /* x is the case's transverse axis, y its propagation axis */
            constexpr float_64 CELL_WIDTH_SI = $(cf(p.cell_SI[1]));
            constexpr float_64 CELL_HEIGHT_SI = $(cf(p.cell_SI[2]));
            constexpr float_64 CELL_DEPTH_SI = $(cf(p.cell_SI[3]));
            /* n_c at the reference wavelength $(cf(p.units.lambda0)) m; the profile is relative to it */
            constexpr float_64 BASE_DENSITY_SI = $(cf(p.units.density));
        } // namespace SI
        constexpr uint32_t TYPICAL_PARTICLES_PER_CELL = $(prod(p.lattice))u;
    } // namespace picongpu
    """
    f["memory.param"] = H * """
    #pragma once
    #include <pmacc/mappings/kernel/MappingDescription.hpp>
    #include <pmacc/math/Vector.hpp>
    #include <array>
    namespace picongpu
    {
        /* per rank; all ranks share the one GPU (-r) */
        constexpr size_t reservedGpuMemorySize = 350 * 1024 * 1024;
        namespace mCT = pmacc::math::CT;
        using SuperCellSize = typename mCT::shrinkTo<mCT::Int<$PIC_SUPERCELL, $PIC_SUPERCELL, 1>, simDim>::type;
        static constexpr uint32_t numFrameSlots = pmacc::math::CT::volume<SuperCellSize>::type::value;
        using MappingDesc = MappingDescription<simDim, SuperCellSize>;
        using GuardSize = typename mCT::shrinkTo<mCT::Int<1, 1, 1>, simDim>::type;
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
    f["species.param"] = H * """
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
    n = PIC_SPECIES
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
        /* electrons only: the ions are immobile and PIConGPU solves no Poisson equation */
        value_identifier(float_X, MassRatio_$n, 1.0);
        value_identifier(float_X, ChargeRatio_$n, 1.0);
        using ParticleFlags_$n = MakeSeq_t<
            particlePusher<UsedParticlePusher>,
            shape<UsedParticleShape>,
            interpolation<UsedField2Particle>,
            current<UsedParticleCurrentSolver>,
            massRatio<MassRatio_$n>,
            chargeRatio<ChargeRatio_$n>>;
        using PIC_$n = Particles<PMACC_CSTRING("$n"), ParticleFlags_$n, DefaultParticleAttributes>;
        using VectorAllSpecies = MakeSeq_t<PIC_$n>;
    } // namespace picongpu
    """
    f["density.param"] = H * """
    #pragma once
    #include "picongpu/particles/densityProfiles/profiles.def"
    namespace picongpu
    {
        namespace densityProfiles
        {
            /* n = DENSITY n_c sin²(π/2 clamp((x − VACUUM)/RAMP, 0, 1)) at lab x = (j + ½ − J0) DX, the cell
             * centre of total cell j along y (every slide counted); lengths in c/ω₀ */
            constexpr float_64 DENSITY = $(cf(p.density));
            constexpr float_64 VACUUM = $(cf(p.vacuum));
            constexpr float_64 RAMP = $(cf(p.ramp));
            constexpr float_64 DX = $(cf(p.dx));
            constexpr float_64 J0 = $(cf(p.layout.j0));
            struct ProfileFunctor_$n
            {
                HDINLINE float_X operator()(const floatD_64& position_SI, const float3_64& cellSize_SI)
                {
                    const float_64 x = (position_SI.y() / cellSize_SI.y() - J0) * DX;
                    float_64 s = (x - VACUUM) / RAMP;
                    s = s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s);
                    const float_64 r = math::sin(0.5 * PI * s);
                    return static_cast<float_X>(DENSITY * r * r);
                }
            };
            using Profile_$n = FreeFormulaImpl<ProfileFunctor_$n>;
        } // namespace densityProfiles
    } // namespace picongpu
    """
    f["particle.param"] = H * """
    #pragma once
    #include "picongpu/particles/filter/filter.def"
    #include "picongpu/particles/manipulators/manipulators.def"
    #include "picongpu/particles/startPosition/functors.def"
    namespace picongpu
    {
        namespace particles
        {
            /* the ramp's first cells carry weightings far below PIConGPU's default floor of 10, which
             * QuietImpl would drop; only empty cells are dropped */
            constexpr float_X MIN_WEIGHTING = 1.0e-20;
            namespace startPosition
            {
                struct QuietParam_$n
                {
                    using numParticlesPerDimension = mCT::shrinkTo<mCT::Int<$(p.lattice[1]), $(p.lattice[2]), 1>, simDim>::type;
                };
                using Quiet_$n = QuietImpl<QuietParam_$n>;
            } // namespace startPosition
        } // namespace particles
    } // namespace picongpu
    """
    init = plasma ? "CreateDensity<densityProfiles::Profile_$n, startPosition::Quiet_$n, PIC_$n>" : ""
    f["speciesInitialization.param"] = H * """
    #pragma once
    #include "picongpu/particles/InitFunctors.hpp"
    namespace picongpu
    {
        namespace particles
        {
            /* $(plasma ? "a cold electron load" : "the laser in vacuum: no particles") */
            using InitPipeline = pmacc::mp_list<$init>;
        } // namespace particles
    } // namespace picongpu
    """
    l = p.laser
    f["incidentField.param"] = H * """
    #pragma once
    #include "picongpu/fields/incidentField/profiles/profiles.def"
    namespace picongpu
    {
        namespace fields
        {
            namespace incidentField
            {
                /* The case's Gaussian beam along +y, E along x (the case's transverse E_y). The envelope's
                 * peak crosses the Huygens surface, a quarter cell behind lab x = 0, at PULSE_INIT ·
                 * PULSE_DURATION / 2 = delay − Δ/4 (GaussianPulse.hpp). */
                struct LwfaLaserParam : public profiles::BaseParam
                {
                    static constexpr float_64 WAVE_LENGTH_SI = $(cf(l.wavelength_SI));
                    /* a0 · m_e c ω₀ / e */
                    static constexpr float_64 AMPLITUDE_SI = $(cf(l.amplitude_SI));
                    /* σ of the intensity: τ/2 */
                    static constexpr float_64 PULSE_DURATION_SI = $(cf(l.pulse_duration_SI));
                    static constexpr float_X LASER_PHASE = 0.0_X;
                    static constexpr float_64 DIRECTION_X = 0.0;
                    static constexpr float_64 DIRECTION_Y = 1.0;
                    static constexpr float_64 DIRECTION_Z = 0.0;
                    /* on the axis (the box's centre along x); along y the lab focus plus J0 cells */
                    static constexpr float_64 FOCUS_POSITION_X_SI = 0.0;
                    static constexpr float_64 FOCUS_POSITION_Y_SI = $(cf(l.focus_y_SI));
                    static constexpr float_64 FOCUS_POSITION_Z_SI = 0.0;
                    static constexpr Origin FOCUS_ORIGIN_X = Origin::Center;
                    static constexpr Origin FOCUS_ORIGIN_Y = Origin::Zero;
                    static constexpr Origin FOCUS_ORIGIN_Z = Origin::Center;
                    static constexpr PolarisationType Polarisation = PolarisationType::Linear;
                    static constexpr float_64 POLARISATION_DIRECTION_X = 1.0;
                    static constexpr float_64 POLARISATION_DIRECTION_Y = 0.0;
                    static constexpr float_64 POLARISATION_DIRECTION_Z = 0.0;
                    /* 1/e field radius at the focus: the case's waist */
                    static constexpr float_64 W0_SI = $(cf(l.w0_SI));
                    static constexpr float_64 PULSE_INIT = $(cf(l.pulse_init));
                    static constexpr uint32_t numModes = 0;
                    static constexpr auto laguerreModes = floatN_X<numModes + 1>(1.0);
                    static constexpr auto laguerrePhases = floatN_X<numModes + 1>(0.0);
                };
                using LwfaLaser = profiles::GaussianPulse<LwfaLaserParam>;
                namespace profiles
                {
                    /* the transverse axis is periodic */
                    template<>
                    struct MakePeriodicTransversalHuygensSurfaceContiguous<LwfaLaser>
                    {
                        static constexpr bool value = true;
                    };
                } // namespace profiles
                using XMin = profiles::None;
                using XMax = profiles::None;
                using YMin = LwfaLaser;
                using YMax = profiles::None;
                using ZMin = profiles::None;
                using ZMax = profiles::None;
                constexpr int32_t POSITION[3][2] = {
                    {$HUYGENS, -$HUYGENS}, // x (periodic: the surface spans the axis)
                    {$HUYGENS, -$HUYGENS}, // y: the laser's plane at ($HUYGENS + 0.75) cells
                    {$HUYGENS, -$HUYGENS} // z
                };
            } // namespace incidentField
        } // namespace fields
    } // namespace picongpu
    """
    f["fileOutput.param"] = H * """
    #pragma once
    #include <pmacc/meta/conversion/MakeSeq.hpp>
    #include "picongpu/fields/Fields.def"
    #include "picongpu/particles/filter/filter.def"
    #include "picongpu/particles/particleToGrid/CombinedDerive.def"
    #include "picongpu/particles/particleToGrid/ComputeGridValuePerFrame.def"
    namespace picongpu
    {
        namespace deriveField = particles::particleToGrid;
        /* E and the electron charge density (B is not compared; it would double the dumps) */
        using ChargeDensity_Seq
            = deriveField::CreateEligible_t<VectorAllSpecies, deriveField::derivedAttributes::ChargeDensity>;
        using FieldTmpSolvers = MakeSeq_t<ChargeDensity_Seq>;
        using NativeFileOutputFields = MakeSeq_t<FieldE>;
        using FileOutputFields = MakeSeq_t<NativeFileOutputFields, FieldTmpSolvers>;
        using FileOutputParticles = MakeSeq_t<>;
    } // namespace picongpu
    """
    return f
end

"""
    picongpu_arguments(p) -> String

The command line: the devices along y on one GPU, the grid, the steps, periodic along x only, the
moving window with its move point, no grid adjustment (the layout is exact), the exponential
absorber, no current filter, the openPMD dumps, the field energies and (plasma) the histogram.
"""
function picongpu_arguments(p)
    lay, h = p.layout, p.histogram
    a = ["-d 1 $(lay.devices)", "-r $(lay.devices)", "-g $(p.cells[1]) $(p.cells[2])", "-s $(p.steps)",
         "--periodic 1 0", "-m", "--windowMovePoint $(PIConGPUDeck.cfloat(lay.move_point))", "--autoAdjustGrid 0",
         "--fieldAbsorber exponential", "--currentInterpolation none",
         "--openPMD.period $(p.every)", "--openPMD.file simData", "--openPMD.ext h5", "--openPMD.infix NULL",
         "--openPMD.source fields_all", "--fields_energy.period $(p.every)"]
    p.plasma && push!(a, "--$(PIC_SPECIES)_energyHistogram.period $(p.every)", "--$(PIC_SPECIES)_energyHistogram.filter all",
                      "--$(PIC_SPECIES)_energyHistogram.binCount $(h.bins)",
                      "--$(PIC_SPECIES)_energyHistogram.minEnergy $(PIConGPUDeck.cfloat(h.min_keV))",
                      "--$(PIC_SPECIES)_energyHistogram.maxEnergy $(PIConGPUDeck.cfloat(h.max_keV))")
    return join(a, " ")
end

"The run.sh of a PIConGPU directory: the toolchain, compile `Empty` with the generated params, run on the one GPU."
function picongpu_run_script(p)
    stamp = "\$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
    return """
    #!/usr/bin/env bash
    # Generated by scripts/lwfa/lwfa.jl (scripts/lwfa/picongpu.jl). Runs inside the PIConGPU container
    # with the run directory at /run-dir (PM_RUN_DIR) and the GPU bound in; $(p.layout.devices) ranks share it.
    set -euo pipefail
    cd "\${PM_RUN_DIR:-/run-dir}"
    pm-versions > versions.toml
    {
      echo "# picongpu: \$PICONGPU_TAG \$PICONGPU_COMMIT, backend \$PIC_BACKEND"
      echo "# gpu: \$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo unknown)"
      echo "# cpus: \$(nproc)"
      echo "# cpu model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
      echo "# compile start: $stamp"
    } > container.txt
    rm -rf setup openPMD ./*.dat
    pic-create "\$PIC_EXAMPLES/Empty" setup > build.log 2>&1
    cp params/*.param setup/include/picongpu/param/
    (cd setup && pic-build -j 16) >> build.log 2>&1 || { tail -n 60 build.log; exit 1; }
    echo "# compile end: $stamp" >> container.txt
    echo "# start: $stamp" >> container.txt
    mpiexec -n $(p.layout.devices) setup/bin/picongpu $(picongpu_arguments(p)) > picongpu.log 2>&1 \\
      || { tail -n 60 picongpu.log; exit 1; }
    echo "# end: $stamp" >> container.txt
    """
end

"Write a PIConGPU run directory (params/, run.sh, case.toml, kind.txt, the manifest) into `dir`."
function prepare_picongpu(dir, case_path, case; plasma, tag, kind)
    p = picongpu_parameters(case; plasma)
    files = picongpu_param_files(case; plasma)
    mkpath(joinpath(dir, "params"))
    for (n, text) in files
        write(joinpath(dir, "params", n), text)
    end
    write(joinpath(dir, "run.sh"), picongpu_run_script(p))
    chmod(joinpath(dir, "run.sh"), 0o755)
    lay = p.layout
    write_manifest(dir; root = REPO, inputs = ["case.toml", "kind.txt", "run.sh", ("params/" .* sort(collect(keys(files))))...],
                   extra = Dict{String,Any}("run_id" => "lwfa-$tag/$kind", "code" => "PIConGPU 0.8.0", "case" => case["name"],
                                            "generator" => "scripts/lwfa/lwfa.jl", "precision" => p.precision,
                                            "reference_wavelength_m" => p.units.lambda0, "arguments" => picongpu_arguments(p),
                                            "layout" => Dict{String,Any}("devices_y" => lay.devices, "local_y" => lay.local_y,
                                                                         "window_y" => lay.window, "cells" => collect(lay.cells),
                                                                         "j0" => lay.j0, "move_point" => lay.move_point)))
    return dir
end

# --- the reader ---------------------------------------------------------------------------------

"""
    read_energy_histogram(file) -> Dict{Int, NamedTuple}

`<species>_energyHistogram_<filter>.dat`: per row the step, the real particles below the
minimum, one column per bin, above the maximum, and the total (BinEnergyParticles.x.cpp).
"""
function read_energy_histogram(file)
    out = Dict{Int,Any}()
    for line in eachline(file)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        v = parse.(Float64, split(s))
        out[round(Int, v[1])] = (; under = v[2], bins = v[3:(end - 2)], over = v[end - 1], total = v[end])
    end
    return out
end

"""
Read PIConGPU's dumps in `dir` into the summary format of `pm_output`, with the case's axes and
names: `Ex_axis` from PIConGPU's E_y on the axis column, `Ey_axis` from its E_x on the two
columns around the axis, `rho_axis` from the electron charge density, each with the lab m of its
first point, and the spectrum in electrons per unit depth in the units of A1 (n_c (c/ω₀)²).
"""
function summarize_picongpu(dir)
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    d, lay = derived(case), picongpu_layout(case)
    plasma = strip(read(joinpath(dir, "kind.txt"), String)) == "plasma"
    p = picongpu_parameters(case; plasma)
    u = p.units
    steps = output_steps(case)
    maps_at = Set(steps[Int.(case["diagnostics"]["map_outputs"])])
    hist = plasma ? read_energy_histogram(joinpath(dir, "$(PIC_SPECIES)_energyHistogram_all.dat")) : Dict{Int,Any}()
    per_weight = u.density * u.length^2 * p.cell_SI[3]       # real electrons per unit of A1's weight per unit depth
    outs = Any[]
    h5open(joinpath(dir, "openPMD", "simData.h5"), "r") do f
        data = f["data"]
        mp = rstrip(haskey(HDF5.attributes(f), "meshesPath") ? read_attribute(f, "meshesPath") : "meshes/", '/')
        for key in sort(collect(keys(data)); by = k -> parse(Int, k))
            it = parse(Int, key)
            m = data[key][mp]
            E = m["E"]
            labels = collect(String, read_attribute(E, "axisLabels"))
            labels == ["y", "x"] || error("iteration $it: axisLabels $labels, expected [y, x]")
            off = Float64.(read_attribute(E, "gridGlobalOffset")) .* Float64(read_attribute(E, "gridUnitSI")) ./ u.length
            jw = round(Int, off[1] / d.dx)                    # the window's first total cell along y
            abs(jw * d.dx - off[1]) <= 1e-6 * d.dx || error("iteration $it: gridGlobalOffset $(off[1]) is not a whole number of cells")
            iszero(off[2]) || error("iteration $it: a transverse offset $(off[2])")
            component(c) = permutedims(Float64.(read(E[c])) .* (Float64(read_attribute(E[c], "unitSI")) / u.efield))   # (y, x) = the case's (x, y)
            Ex, Ey = component("y"), component("x")                   # the case's longitudinal and transverse E
            size(Ex) == (lay.window, d.ny) || error("iteration $it: E is $(size(Ex)), expected the window $((lay.window, d.ny))")
            m0 = jw - lay.j0                                  # node j of the window (from 0) ↔ m = j + m0
            ja = d.jaxis + 1
            out = Dict{String,Any}("step" => it, "t" => it * d.dt, "window_offset" => jw,
                                   "Ex_axis" => Ex[:, ja], "Ex_m0" => m0 + 1,                   # E_y at (j + ½): x = (m − ½)Δx with m = j + 1 + m0
                                   "Ey_axis" => (Ey[:, ja - 1] .+ Ey[:, ja]) ./ 2, "Ey_m0" => m0,   # E_x at y_c ∓ Δy/2
                                   "Ey_absmax" => maximum(abs, Ey))
            out["Ey_envelope"] = running_envelope(out["Ey_axis"], 16)
            if plasma
                R = m["$(PIC_SPECIES)_all_chargeDensity"]
                ρ = permutedims(Float64.(read(R)) .* (Float64(read_attribute(R, "unitSI")) / (PIConGPUDeck.QE_SI * u.density)))
                out["rho_axis"], out["rho_m0"] = ρ[:, ja], m0
                h = get(hist, it, nothing)
                if h !== nothing
                    length(h.bins) == p.histogram.bins || error("iteration $it: $(length(h.bins)) histogram bins, expected $(p.histogram.bins)")
                    out["spectrum"] = h.bins ./ per_weight
                    out["spectrum_under"], out["spectrum_over"] = h.under / per_weight, h.over / per_weight
                else
                    out["spectrum"] = Float64[]
                end
                it in maps_at && (out["map"] = (rho = ρ, Ey = Ey, m0 = m0))
            end
            push!(outs, out)
        end
    end
    secs = try
        lines = readlines(joinpath(dir, "container.txt"))
        stamp(q) = DateTime(first(filter(l -> startswith(l, q), lines))[length(q) + 1:end][1:23], "yyyy-mm-ddTHH:MM:SS.sss")
        Dates.value(stamp("# end: ") - stamp("# start: ")) / 1000
    catch
        nothing
    end
    write_summary(dir, outs; code = "PIConGPU", extra = Dict{String,Any}("seconds" => secs, "spectrum_edges" => spectrum_edges(case),
                  "spectrum_domain" => "the whole simulated domain, hidden row included",
                  "devices_y" => lay.devices, "window_cells" => lay.window))
    return dir
end
