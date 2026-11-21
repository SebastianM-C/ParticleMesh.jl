# Tests of the ionization case's driver (scripts/lwfa/ionization.jl, WP-081), included by runtests.jl.

include(joinpath(@__DIR__, "..", "ionization.jl"))
include(joinpath(REPO, "scripts", "optim", "born2d_case.jl"))
using ParticleMeshElectromagnetic: WindowClock

const ION_CASE_FILE = joinpath(@__DIR__, "..", "cases", "ionization-2d.toml")

@testset "T-1120 the ionization case file is born2d_case.jl's problem: with the axis envelope field by field the same, with the case's wavefront envelope only the laser differs [A5, WP-081]" begin
    case = TOML.parsefile(ION_CASE_FILE)
    d = ion_derived(case)
    a, b = pm_ion_problem(case; envelope = AxisTime()), born_problem()
    for f in fieldnames(typeof(a))
        f in (:name, :nsteps) && continue
        @test repr(getfield(a, f)) == repr(getfield(b, f))
    end
    # the name is the case's, and the steps end on the last output: the whole number of outputs
    # nearest to t_end, which born_problem reaches at ceil(t_end/Δt)
    @test a.name == case["name"] && a.nsteps == d.steps && b.nsteps == ceil(Int, Float64(case["time"]["t_end"]) / d.dt)
    @test abs(a.nsteps - b.nsteps) < d.every
    # the outputs every `every` steps from 0
    @test d.outputs == collect(0:d.every:d.steps) && length(d.outputs) == Int(case["time"]["outputs"]) + 1
    w = pm_ion_problem(case)
    @test only(w.lasers).envelope isa WavefrontTime
    for f in fieldnames(typeof(a))
        f === :lasers && continue
        @test repr(getfield(w, f)) == repr(getfield(a, f))
    end
    # vacuum and the plasma without the dopant
    @test isempty(pm_ion_problem(case; plasma = false).species) && pm_ion_problem(case; plasma = false).ionization === nothing
    @test pm_ion_problem(case; dopant = false).ionization === nothing && length(pm_ion_problem(case; dopant = false).species) == 1
end

@testset "T-1121 the three codes' ADK prefactors: Smilei's is the l* = n* − 1 coefficient, PIConGPU's the Stirling form at every field, and the levels change only by it [A5, WP-081]" begin
    case = TOML.parsefile(ION_CASE_FILE)
    g = ionization_energy(:N, 1)
    for z in (6, 7)
        e = ionization_energy(:N, z)
        # Smilei's P = 2^{2n*}(2l + 1)/(n*Γ(2n*))·I is C²·I with l* = n* − 1, so ParticleMesh's over it is the ratio of the C²
        cg = adk_numbers(e, z; wavelength = 0.8e-6, ground_eV = g).C2
        cl = adk_numbers(e, z; wavelength = 0.8e-6, ground_eV = g, lstar = :level).C2
        @test rate_ratio("pm", e, z; ground_eV = g) ≈ cg / cl rtol = 1e-9          # the hartree in eV against atomic.jl's constants
        @test rate_ratio("smilei", e, z; ground_eV = g) == 1.0
        # PIConGPU's rate (AlgorithmADK.hpp, atomic units) over Smilei's at fields from 0.05 to 2 a.u., in logarithms
        # (the rates underflow at the weak fields): one number
        I = e / 27.211386245988
        n = z / sqrt(2I)
        logpic(E) = log(E) + 2n * log(4 * exp(1) * z^3 / (E * n^4)) - log(8π * z) - 2 * z^3 / (3 * n^3 * E)
        F = 2 * (2I)^1.5
        logsmi(E) = 2n * log(2) - log(n * gamma(2n)) + log(I) + (2n - 1) * log(F / E) - F / (3E)
        r = [exp(logpic(E) - logsmi(E)) for E in (0.05, 0.2, 0.7, 2.0)]
        @test all(x -> isapprox(x, rate_ratio("picongpu", e, z; ground_eV = g); rtol = 1e-9), r)
        # the K shell's prefactors are within a factor of two of each other: no convention is off by a power
        @test 0.5 < rate_ratio("pm", e, z; ground_eV = g) < 2 && 0.5 < rate_ratio("picongpu", e, z; ground_eV = g) < 2
    end
    pm = case_levels(Float64, case)
    for rates in ("smilei", "picongpu")
        other = case_levels(Float64, case; rates)
        for (z, lp, lo) in zip(Int.(case["dopant"]["levels"]), pm, other)
            e = ionization_energy(:N, z)
            @test lo.prefactor / lp.prefactor ≈ rate_ratio(rates, e, z; ground_eV = g) / rate_ratio("pm", e, z; ground_eV = g) rtol = 1e-12
            @test lo.field == lp.field && lo.power == lp.power && lo.cutoff == lp.cutoff
        end
    end
end

@testset "T-1122 the Smilei namelist of the ionization case: the dopant as N⁵⁺ ionized by the tunnel model into the born species, the run's seed, the diagnostics' numbering [A5, WP-081]" begin
    case = TOML.parsefile(ION_CASE_FILE)
    d = ion_derived(case)
    nl = smilei_ion_namelist(case; seed = 3)
    @test occursin("number_of_cells = [$(d.nx), $(d.ny)]", nl) && occursin("random_seed = 3,", nl)
    @test occursin("ionization_model = \"tunnel\"", nl) && occursin("ionization_electrons = \"born\"", nl)
    @test occursin("atomic_number = 7,", nl) && occursin("charge = 5.0,", nl) && occursin("time_frozen = 1.0e9,", nl)
    @test occursin("regular_number = [16, 16]", nl) && occursin("particles_per_cell = 256,", nl)
    @test occursin("MovingWindow(time_start = 108.0, velocity_x = 1.0)", nl)
    @test occursin("\"silver-muller\"", nl) && !occursin("periodic", nl)
    # binning 0 the charge states over N⁵⁺ to N⁷⁺, then three per species
    @test occursin("axes = [[\"charge\", 4.5, 7.5, 3]]", nl)
    @test count("DiagParticleBinning(", nl) == 7 && occursin("DiagTrackParticles(species = \"born\"", nl)
    nd = smilei_ion_namelist(case; dopant = false)
    @test !occursin("nitrogen", nd) && !occursin("\"born\"", nd) && count("DiagParticleBinning(", nd) == 3
    @test smilei_binning_index("electron", false) == 0 && smilei_binning_index("born", true) == 1 && smilei_binning_index("electron", true) == 4
    vac = smilei_ion_namelist(case; plasma = false)
    @test !occursin("Species(", vac) && occursin("LaserGaussian2D(", vac)
    # the dopant's profile is DopantSection's: 0 before the section, 1 on its plateau, 0 after it
    s0, s1, sr = Float64.(case["dopant"]["section"])
    @test dopant_section_factor(case, s0) == 0 && dopant_section_factor(case, s0 + sr) ≈ 1 && dopant_section_factor(case, (s0 + s1) / 2) == 1
    @test abs(dopant_section_factor(case, s1 + sr)) < 1e-15 && dopant_section_factor(case, s0 + sr / 2) ≈ 0.5
end

@testset "T-1123 the PIConGPU deck of the ionization case: lab x from the antenna plane, a window long enough to start on time, N⁵⁺ ionized by ADKCircPol into eb, frozen, PML on the open sides [A5, WP-081]" begin
    case = TOML.parsefile(ION_CASE_FILE)
    d = ion_derived(case)
    lab = picongpu_ion_lab(case)
    @test lab["window"]["start"] == Float64(case["window"]["start"]) + d.xa
    @test lab["laser"]["focus"] == Float64(case["laser"]["focus"]) - d.xa && lab["laser"]["delay"] == case["laser"]["delay"]
    @test lab["plasma"]["vacuum"] == Float64(case["plasma"]["vacuum"]) - d.xa
    # PIConGPU delays the window by at most its length: the window covers the cells the front crosses before it moves
    lay = picongpu_layout(lab)
    # (picongpu_layout refuses a start that does not fit: the virtual photon starts window + j0 − lag cells into it)
    @test lab["grid"]["cells"][1] >= d.nx - d.na && lay.lag == round(Int, lab["window"]["start"] / d.dx)
    @test 0 <= lay.window + lay.j0 - lay.lag <= lay.window && 0 < lay.move_point < 1
    # with the case's own window the start would not fit
    short = deepcopy(lab); short["grid"]["cells"][1] = d.nx - d.na
    @test_throws ErrorException picongpu_layout(short)
    f = picongpu_ion_param_files(case; kind = "plasma", seed = 3)
    sp = f["speciesDefinition.param"]
    @test occursin("ADKCircPol<PIC_eb, particles::ionization::current::None>", sp)
    @test occursin("particlePusher<particles::pusher::Probe>", sp) && occursin("Nitrogen_t", sp)
    @test occursin("using VectorAllSpecies = MakeSeq_t<PIC_e, PIC_eb, PIC_N>;", sp)
    # the dopant carries no current: its flags have none
    flagsN = split(split(sp, "using ParticleFlags_N")[2], "using PIC_N")[1]
    @test !occursin("current<", replace(flagsN, "current::None" => ""))
    @test occursin("ChargeState<5u>", f["speciesInitialization.param"]) && occursin("seed::Value<3>", f["random.param"])
    @test occursin("mCT::Int<16, 16, 1>", f["particle.param"]) && occursin("MIN_WEIGHTING = 1.0e-20", f["particle.param"])
    # the section in lab x
    s0 = Float64(case["dopant"]["section"][1]) - d.xa
    @test occursin("S0 = $(PIConGPUDeck.cfloat(s0));", f["density.param"])
    nd = picongpu_ion_param_files(case; kind = "plasma-nodopant", seed = 1)
    @test occursin("using VectorAllSpecies = MakeSeq_t<PIC_e>;", nd["speciesDefinition.param"]) && !occursin("ChargeState", nd["speciesInitialization.param"])
    p = picongpu_parameters(lab; plasma = true)
    args = picongpu_arguments(p)
    @test occursin("--periodic 0 0", args) && occursin("--fieldAbsorber pml", args)
    # an open case's Huygens surface does not wrap around the transverse axis
    @test !occursin("MakePeriodicTransversalHuygensSurfaceContiguous", f["incidentField.param"])
    @test occursin("MakePeriodicTransversalHuygensSurfaceContiguous", picongpu_param_files(TOML.parsefile(joinpath(@__DIR__, "..", "cases", "lwfa-2d.toml")))["incidentField.param"])
end

@testset "T-1124 the window shifts the comparison reads every code's particles against are ParticleMesh's: window_shifts at every step of the case [A5, WP-081]" begin
    case = TOML.parsefile(ION_CASE_FILE)
    d = ion_derived(case)
    c = WindowClock(Float64(case["window"]["start"]), d.dt, d.dx)
    @test all(k -> pm_window_shifts(case, k) == window_shifts(c, k), 0:d.steps)
    @test pm_window_shifts(case, 0) == 0 && pm_window_shifts(case, d.steps) > 0
end
