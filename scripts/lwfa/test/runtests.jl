# Tests of the LWFA case's generator and readers (scripts/lwfa/lwfa.jl).
#   julia +1.13 --project=scripts scripts/lwfa/test/runtests.jl

using Test
include(joinpath(@__DIR__, "..", "lwfa.jl"))
using ParticleMeshElectromagnetic: ramp_factor

# A5's case and its coarse-grid reference, the input of reports/lwfa-2d-a2.md.
const CASE_FILES = [joinpath(@__DIR__, "..", "cases", f) for f in ("lwfa-2d.toml", "lwfa-2d-a2.toml")]

@testset "T-510 the Smilei namelist carries the case: grid, step, window, the laser's parameters and the density profile [A1, A5]" for file in CASE_FILES
    CASE = TOML.parsefile(file)
    d = derived(CASE)
    nl = smilei_namelist(CASE; plasma = true)
    num(key) = parse(Float64, match(Regex("\\b$key = ([-0-9.e]+)"), nl).captures[1])
    @test occursin("number_of_cells = [$(d.nx), $(d.ny)]", nl)
    @test num("timestep") == d.dt
    @test d.dt == CASE["time"]["courant"] / sqrt(1 / d.dx^2 + 1 / d.dy^2)
    @test floor(Int, num("simulation_time") / d.dt) == d.steps        # Smilei takes floor(simulation_time/timestep) steps
    @test num("time_start") == CASE["window"]["start"]
    # the laser as smilei_laser places it (T-698 checks the pulse these give): the focus counted from the
    # ghost node two cells behind xmin, the envelope's centre and the phase offset
    las = smilei_laser(CASE)
    @test las.back == 2 * d.dx && las.focus == CASE["laser"]["focus"] + 2 * d.dx
    @test occursin("focus = [$(repr(las.focus)), $(repr(d.yc))]", nl)
    @test num("center") == las.center && num("duration") == 2 * las.center && num("phase_offset") == las.phase_offset
    # the amplitude given to Smilei is the case's over what its boundary condition launches per unit (T-710)
    @test num("a0") == las.a0 == CASE["laser"]["a0"] / smilei_launch_factor(d)
    @test 1.001 < smilei_launch_factor(d) < 1.005
    # tgaussian (v5.1): exp(−(t − center)²/σ), σ = (fwhm/2)²/ln 2: the envelope exp(−(t − center)²/τ²)
    fwhm, τ = num("fwhm"), CASE["laser"]["duration"]
    σ = (fwhm / 2)^2 / log(2)
    @test all(t -> isapprox(exp(-(t - las.center)^2 / σ), exp(-(t - las.center)^2 / τ^2); rtol = 1e-12), 0:13:208)
    # the profile, evaluated as the namelist's Python writes it, against ParticleMesh's at x + x_a
    P = CASE["plasma"]
    smilei_n(x) = P["density"] * sin(0.5 * π * clamp((x - P["vacuum"]) / P["ramp"], 0.0, 1.0))^2
    prob = pm_problem(CASE)
    for x in (0.0, 259.0, 261.5, 290.0, 319.9, 400.0)
        @test smilei_n(x) ≈ P["density"] * ramp_factor(x + d.xa, prob.profile.vacuum, prob.profile.ramp) atol = 1e-15
    end
    @test occursin("time_frozen = frozen", nl) && occursin("(\"ion\", 1.0, 1836.15267343, ", nl)
    # the window's spectrum and its two strips, cell by cell, at the back and at the front
    strip = smilei_strip(CASE)
    @test strip == 2 * (d.nx ÷ Int(CASE["grid"]["smilei_patches"][1]))
    @test count("DiagParticleBinning(", nl) == 3
    @test occursin("axes = [[\"moving_x\", 0.0, $(repr(strip * d.dx)), $strip], [\"ekin\"", nl)
    @test occursin("axes = [[\"moving_x\", $(repr((d.nx - strip) * d.dx)), $(repr(d.nx * d.dx)), $strip], [\"ekin\"", nl)
    vac = smilei_namelist(CASE; plasma = false)
    @test !occursin("Species(", vac) && occursin("fields = [\"Ex\", \"Ey\"]", vac) && !occursin("DiagParticleBinning", vac)
end

@testset "T-511 the ParticleMesh problem is the case with the antenna offset: plane on node antenna_cells + 1, focus and profile shifted [A1, A3, A5]" for file in CASE_FILES
    CASE = TOML.parsefile(file)
    d = derived(CASE)
    prob = pm_problem(CASE)
    @test prob.cells == (d.nx + d.na, d.ny)
    @test prob.dt == d.dt && prob.nsteps == d.steps
    l = only(prob.lasers)
    @test l.position == d.xa && l.focus == CASE["laser"]["focus"] + d.xa && l.center == d.yc
    @test prob.profile.vacuum == CASE["plasma"]["vacuum"] + d.xa
    st = EMState(pm_problem(CASE; plasma = false), EMNumerics())
    @test only(st.aux.antennas).ia == d.na + 1                     # node i at (i − 1)Δx: the plane x = x_a
    @test window_shifts(st, 0) == 0
    k = ceil(Int, (CASE["window"]["start"] + 100) / d.dt)          # 100/ω₀ after the window starts
    @test window_shifts(st, k) == floor(Int, (k * d.dt - CASE["window"]["start"]) / d.dx)
end

@testset "T-512 lineouts are matched on the lab lattice they share [A5]" begin
    a, b = collect(1.0:10.0), collect(101.0:120.0)          # a at m = 5 … 14, b at m = 0 … 19
    c = common_points(a, 5, b, 0)
    @test c.m == collect(5:14) && c.a == a && c.b == b[6:15]
    c = common_points(a, 5, b, 0; mlo = 7, mhi = 9)
    @test c.m == [7, 8, 9] && c.a == [3.0, 4.0, 5.0] && c.b == [108.0, 109.0, 110.0]
    @test isempty(common_points(a, 100, b, 0).m)
    @test relrms([1.0, 2.0], [1.0, 2.0]) == 0
    o = Dict("Ey_m0" => -3, "Ey_envelope" => [0.0, 0.1, 2.0, 0.5])
    @test peak_m(o) == -1
    @test running_envelope([0.0, 1.0, -3.0, 0.5, 0.0], 1) == [1.0, 3.0, 3.0, 3.0, 0.5]
    # half a wavelength: 24 points at λ₀/48 (lwfa-2d), 16 at λ₀/32 (lwfa-2d-a2)
    @test [envelope_halfwidth(TOML.parsefile(f)) for f in CASE_FILES] == [24, 16]
end

@testset "T-513 Smilei's dumps are read on the lab lattice: x_moved, the dual and primal axes, the axis rows [A5]" for file in CASE_FILES
    CASE = TOML.parsefile(file)
    d = derived(CASE)
    mktempdir() do dir
        cp(file, joinpath(dir, "case.toml"))
        patch = d.nx ÷ Int(CASE["grid"]["smilei_patches"][1])     # Smilei moves its window by a patch
        bins = Int(CASE["diagnostics"]["spectrum_ekin"][3])
        write(joinpath(dir, "kind.txt"), "plasma\n")
        nx, ny = d.nx, d.ny
        steps = output_steps(CASE)
        h5open(joinpath(dir, "Fields0.h5"), "w") do f
            data = create_group(f, "data")
            for (j, it) in enumerate(steps)
                g = create_group(data, lpad(it, 10, '0'))
                HDF5.attributes(g)["x_moved"] = patch * d.dx * (j - 1)
                # C order [y][x]: HDF5.jl writes a Julia (ny', nx') array as x-first after the reader's permutedims
                Ex = [Float64(i) + 1000 * jj for jj in 0:ny, i in 0:(nx + 1)]           # dual in x
                Ey = [Float64(i) + 1000 * jj for jj in 0:(ny + 1), i in 0:nx]           # dual in y
                ρ = [-Float64(i) - 1000 * jj for jj in 0:ny, i in 0:nx]
                g["Ex"], g["Ey"], g["Rho_electron"] = Ex, Ey, ρ
            end
        end
        h5open(joinpath(dir, "ParticleBinning0.h5"), "w") do f
            for it in steps
                f["timestep" * lpad(it, 8, '0')] = fill(Float64(it), bins)
            end
        end
        write(joinpath(dir, "container.txt"), "# start: 2026-11-05T10:00:00.000000000Z\n# end: 2026-11-05T10:01:30.500000000Z\n")
        summarize_smilei(dir)
        s = JSON.parsefile(joinpath(dir, "summary.json"))
        o = s["outputs"][3]
        @test o["Ex_m0"] == 2patch && o["rho_m0"] == 2patch && o["Ey_m0"] == 2patch           # output 3: two patches moved
        @test o["Ex_axis"][1:3] == [0.0, 1.0, 2.0] .+ 1000 * d.jaxis                  # row jaxis (from 0) is y = y_c
        @test o["rho_axis"][1] == -1000.0 * d.jaxis
        @test o["Ey_axis"][1] == 1000.0 * d.jaxis + 500                                # rows jaxis, jaxis + 1 at y_c ∓ Δy/2
        @test o["spectrum"] == fill(Float64(steps[3]), bins)
        @test s["seconds"] ≈ 90.5
        maps = h5open(joinpath(dir, "maps.h5"), "r") do f
            [k for k in keys(f)]
        end
        @test sort(maps) == sort(["step$(steps[j])" for j in Int.(CASE["diagnostics"]["map_outputs"])])
    end
end

# --- PIConGPU (scripts/lwfa/picongpu.jl) -------------------------------------------------------------

cnum(text, key) = parse(Float64, match(Regex("\\b$key = ([-+0-9.eE]+)"), text).captures[1])

@testset "T-514 the PIConGPU deck carries the case with the axes swapped: cells, step, the laser's envelope, focus and polarization [A1, A5]" for file in CASE_FILES
    CASE = TOML.parsefile(file)
    d = derived(CASE)
    u = reference_units(CASE["lambda0_um"] * 1e-6)
    f = picongpu_param_files(CASE; plasma = true)
    sim, las = f["simulation.param"], f["incidentField.param"]
    @test cnum(sim, "CELL_WIDTH_SI") == d.dy * u.length          # PIConGPU x: the case's transverse axis
    @test cnum(sim, "CELL_HEIGHT_SI") == d.dx * u.length         # PIConGPU y: the propagation axis
    @test cnum(sim, "DELTA_T_SI") == d.dt * u.time
    @test cnum(sim, "BASE_DENSITY_SI") == u.density
    @test occursin("Solver = maxwellSolver::Yee", f["fieldSolver.param"]) && occursin("precision64Bit", f["precision.param"])
    @test occursin("mCT::Int<$(CASE["plasma"]["particles_per_cell"][2]), $(CASE["plasma"]["particles_per_cell"][1]), 1>", f["particle.param"])
    @test occursin("DIRECTION_Y = 1.0", las) && occursin("POLARISATION_DIRECTION_X = 1.0", las) && occursin("using YMin = LwfaLaser", las)
    @test occursin("{$HUYGENS, -$HUYGENS}, // y", las)
    L = CASE["laser"]
    @test cnum(las, "AMPLITUDE_SI") == L["a0"] * u.efield
    @test cnum(las, "W0_SI") == L["waist"] * u.length
    lay = picongpu_layout(CASE)
    @test cnum(las, "FOCUS_POSITION_Y_SI") ≈ (lay.j0 * d.dx + L["focus"]) * u.length rtol = 1e-15
    # GaussianPulse.hpp: envelope exp(−((t + TIME_SHIFT)/(2σ))²), TIME_SHIFT = −PULSE_INIT·σ/2, at the Huygens
    # surface (HUYGENS + 0.75)Δ; on lab x = 0, a quarter cell later, it must be the case's exp(−(t − delay)²/τ²)
    σ = cnum(las, "PULSE_DURATION_SI") / u.time
    init = cnum(las, "PULSE_INIT")
    pic(t) = exp(-((t - 0.25 * d.dx) - init * σ / 2)^2 / (2σ)^2)
    @test all(t -> isapprox(pic(t), exp(-(t - L["delay"])^2 / L["duration"]^2); rtol = 1e-12), 0:13:260)
    args = picongpu_arguments(picongpu_parameters(CASE))
    @test occursin("-g $(d.ny) $(lay.cells[2])", args) && occursin("-s $(d.steps)", args) && occursin("--periodic 1 0", args)
    @test occursin("-d 1 $(lay.devices) -r $(lay.devices)", args) && occursin("--currentInterpolation none", args)
    @test occursin("--e_energyHistogram.binCount $(CASE["diagnostics"]["spectrum_ekin"][3])", args)
    keV = 1e3 * PIConGPUDeck.QE_SI
    @test parse(Float64, match(r"maxEnergy ([0-9.e+]+)", args).captures[1]) * keV ≈ CASE["diagnostics"]["spectrum_ekin"][2] * u.energy rtol = 1e-14
    vac = picongpu_param_files(CASE; plasma = false)
    @test occursin("InitPipeline = pmacc::mp_list<>", vac["speciesInitialization.param"])
    @test !occursin("energyHistogram", picongpu_arguments(picongpu_parameters(CASE; plasma = false)))
    # the plasma run asks openPMD for the species too; the deck lists no particles, so none are written (picongpu.jl's header)
    @test occursin("using FileOutputParticles = MakeSeq_t<>;", f["fileOutput.param"])
    @test occursin("--openPMD.source species_all,fields_all ", args)
    @test occursin("--openPMD.source fields_all ", picongpu_arguments(picongpu_parameters(CASE; plasma = false)))
end

@testset "T-515 PIConGPU's moving window: a box of whole supercells, the window's back at lab x = c(t − start) [A5]" for file in CASE_FILES
    CASE = TOML.parsefile(file)
    d = derived(CASE)
    lay = picongpu_layout(CASE)
    @test lay.window == (lay.devices - 1) * lay.local_y && lay.cells[2] == lay.devices * lay.local_y
    @test lay.local_y % PIC_SUPERCELL == 0 && d.ny % PIC_SUPERCELL == 0
    @test lay.window >= d.nx + lay.j0                                     # it holds the case's window and the cells behind x = 0
    @test ceil(Int, lay.window * (1 - lay.move_point)) == lay.start_cell  # MovingWindow.hpp's virtual photon start
    @test HUYGENS >= PIC_ABSORBER + 1
    @test picongpu_window_back(CASE, 0) == -lay.j0
    start = CASE["window"]["start"]
    for k in round.(Int, range(start / d.dt + 50, d.steps; length = 40))
        @test abs(picongpu_window_back(CASE, k) * d.dx - (k * d.dt - start)) <= 2 * d.dx
    end
    # the window never moves backwards, and the front of the case's window stays inside it
    backs = [picongpu_window_back(CASE, k) for k in 0:d.steps]
    @test all(>=(0), diff(backs))
    @test all(k -> picongpu_window_back(CASE, k) + lay.window - 1 >= max(0.0, k * d.dt - start) / d.dx + d.nx - 2, 0:97:d.steps)
end

@testset "T-516 PIConGPU's density is the case's profile at the cell centres of the lab lattice [A3, A5]" for file in CASE_FILES
    CASE = TOML.parsefile(file)
    d = derived(CASE)
    lay = picongpu_layout(CASE)
    P = CASE["plasma"]
    dens = picongpu_param_files(CASE)["density.param"]
    @test cnum(dens, "DENSITY") == P["density"] && cnum(dens, "VACUUM") == P["vacuum"] && cnum(dens, "RAMP") == P["ramp"]
    @test cnum(dens, "DX") == d.dx && cnum(dens, "J0") == lay.j0
    prob = pm_problem(CASE)
    # cells before, across and after the ramp (x = 260 to 320), and past the window
    for j in (0, lay.j0, (lay.j0 .+ round.(Int, (250, 275, 300, 330) ./ d.dx))..., 4 * d.nx)
        x = (j + 0.5 - lay.j0) * d.dx                                         # lab x of the cell centre
        @test picongpu_density(CASE, j) ≈ P["density"] * ramp_factor(x + d.xa, prob.profile.vacuum, prob.profile.ramp) atol = 1e-15
    end
end

@testset "T-517 PIConGPU's dumps are read with the axes swapped back: the window offset, E_y as E_x, E_x as E_y, the axis columns [A5]" for file in CASE_FILES
    CASE = TOML.parsefile(file)
    d = derived(CASE)
    lay = picongpu_layout(CASE)
    u = reference_units(CASE["lambda0_um"] * 1e-6)
    W, N = lay.window, d.ny
    mktempdir() do dir
        cp(file, joinpath(dir, "case.toml"))
        bins = Int(CASE["diagnostics"]["spectrum_ekin"][3])
        write(joinpath(dir, "kind.txt"), "plasma\n")
        steps = output_steps(CASE)
        mkpath(joinpath(dir, "openPMD"))
        h5open(joinpath(dir, "openPMD", "simData.h5"), "w") do f
            HDF5.attributes(f)["meshesPath"] = "fields/"
            data = create_group(f, "data")
            for (q, it) in enumerate(steps)
                fl = create_group(create_group(data, string(it)), "fields")
                E = create_group(fl, "E")
                HDF5.attributes(E)["axisLabels"] = ["y", "x"]
                HDF5.attributes(E)["gridUnitSI"] = u.length
                HDF5.attributes(E)["gridGlobalOffset"] = [10.0 * (q - 1) * d.dx, 0.0]     # y first, in units of gridUnitSI
                # HDF5.jl writes a Julia (x, y) array as C order [y][x]
                E["x"] = [1000.0 * i + j for i in 0:(N - 1), j in 0:(W - 1)]
                E["y"] = [-1000.0 * i - j for i in 0:(N - 1), j in 0:(W - 1)]
                HDF5.attributes(E["x"])["unitSI"] = u.efield                              # the reader's scale is then exactly 1
                HDF5.attributes(E["y"])["unitSI"] = u.efield
                fl["e_all_chargeDensity"] = [Float64(j + 2000 * i) for i in 0:(N - 1), j in 0:(W - 1)]
                HDF5.attributes(fl["e_all_chargeDensity"])["unitSI"] = PIConGPUDeck.QE_SI * u.density
            end
        end
        per_weight = u.density * u.length^2 * d.dx * u.length
        open(joinpath(dir, "e_energyHistogram_all.dat"), "w") do io
            println(io, "#step <0 header")
            for it in steps
                println(io, join([it; 7.0; fill(Float64(it), bins); 3.0; 99.0], " "))
            end
        end
        summarize_picongpu(dir)
        s = JSON.parsefile(joinpath(dir, "summary.json"))
        @test s["code"] == "PIConGPU"
        o = s["outputs"][3]
        jw = 20
        @test o["window_offset"] == jw
        @test o["Ey_m0"] == jw - lay.j0 && o["rho_m0"] == jw - lay.j0 && o["Ex_m0"] == jw - lay.j0 + 1
        @test o["Ex_axis"][1:3] == -1000.0 * d.jaxis .- [0.0, 1.0, 2.0]            # PIConGPU's E_y on column jaxis (from 0)
        @test o["Ey_axis"][1] == 1000.0 * d.jaxis - 500                             # its E_x on columns jaxis − 1, jaxis
        @test o["Ey_absmax"] == 1000.0 * (N - 1) + W - 1
        @test o["rho_axis"][5] == 4.0 + 2000 * d.jaxis
        @test o["spectrum"] ≈ fill(steps[3] / per_weight, bins) rtol = 1e-14
        @test o["spectrum_under"] ≈ 7 / per_weight rtol = 1e-14
        # no electron dump in this run: the spectrum is the plugin's, over the whole simulated domain
        @test o["spectrum"] == o["spectrum_plugin"] && !haskey(o, "spectrum_region")
        @test occursin("whole simulated domain", s["spectrum_covers"]) && isempty(s["spectrum_failed"])
        maps = h5open(joinpath(dir, "maps.h5"), "r") do f
            Dict(k => size(read(f[k]["rho"])) for k in keys(f))
        end
        @test sort(collect(keys(maps))) == sort(["step$(steps[j])" for j in Int.(CASE["diagnostics"]["map_outputs"])])
        # (x, y) as the other codes' maps, at the case's stride from the first lattice column divisible by it
        sx, sy = Int.(CASE["diagnostics"]["map_stride"])
        m0s = [o["rho_m0"] for o in s["outputs"][Int.(CASE["diagnostics"]["map_outputs"])]]
        @test sort(collect(values(maps))) == sort([(length((mod(-m0, sx) + 1):sx:W), cld(N, sy)) for m0 in m0s])
    end
end

# The window offsets (total cells, `gridGlobalOffset`) PIConGPU 0.8.0 wrote at the case's outputs on the
# RTX 4080 SUPER: lwfa-2d in run lwfa-a5pic (commit 515e4c0), lwfa-2d-a2 in run lwfa-pic2 (commit eafe193).
const PIC_WINDOW_OFFSETS = Dict(
    "lwfa-2d.toml" => [0, 0, 0, 354, 1118, 1882, 2646, 3410, 4174, 4938, 5702, 6466, 7230, 7994],
    "lwfa-2d-a2.toml" => [0, 0, 0, 242, 752, 1262, 1771, 2281, 2791, 3300, 3810, 4320, 4829, 5339])

@testset "T-518 picongpu_window_back is the window PIConGPU wrote, to the cell, at every output [A5]" for file in CASE_FILES
    case = TOML.parsefile(file)
    j0 = picongpu_layout(case).j0
    @test [picongpu_window_back(case, k) + j0 for k in output_steps(case)] == PIC_WINDOW_OFFSETS[basename(file)]
end

# --- the particles ParticleMesh loads: zero weights in the vacuum, the weight unit (lwfa.jl's pm_spectrum) -----

# A small window problem of the case: 96 × 16 cells and 8 antenna columns, 4 c/ω₀ of vacuum and a
# 5 c/ω₀ ramp ahead of the injection plane, so that the first load holds vacuum, ramp and plateau.
function small_window_case()
    case = TOML.parsefile(CASE_FILES[1])
    case["grid"]["cells"], case["grid"]["antenna_cells"] = [96, 16], 8
    case["plasma"]["vacuum"], case["plasma"]["ramp"] = 4.0, 5.0
    case["window"]["start"] = 260.0                       # the antenna, 8 columns in, is passed after delay + 6·duration
    return case
end

function small_window_state(case)
    prob = pm_problem(case)
    st = EMState(prob, EMNumerics(; mode = Reference()))
    initialize!(st, prob)
    return st, prob
end

@testset "T-690 ParticleMesh's ekin_max is taken over the electrons of positive weight: the window's vacuum is loaded with zero weights [A3, A5]" begin
    case = small_window_case()
    d = derived(case)
    st, prob = small_window_state(case)
    s = only(st.species)
    alive = findall(==(ParticleMeshCore.ALIVE), s.status)
    x = [(s.cell[1][i] - 1 + s.offset[1][i]) * d.dx for i in alive]       # in the box's coordinate, x_a behind the lab's
    w = s.w[alive]
    # the loader fills every plasma column, the vacuum's with zero weight (#362's cause)
    ncol = maximum(s.cell[1][alive])
    @test length(alive) == ncol * d.ny * prod(case["plasma"]["particles_per_cell"])
    @test count(iszero, w) == count(<=(prob.profile.vacuum), x) > 0
    @test all(>(0), w[x .> prob.profile.vacuum])
    edges = spectrum_edges(case)
    cold = pm_spectrum(s, edges)
    @test cold.ekin_max == 0 && cold.alive == length(alive)
    @test sum(cold.hist) ≈ sum(w) rtol = length(w) * eps()    # two summation orders
    # a zero-weight particle that quivers is no electron: neither the maximum nor the histogram sees it
    ghost, electron = alive[findfirst(iszero, w)], alive[findlast(>(0), w)]
    s.u[2][ghost] = 3.0
    quiver = pm_spectrum(s, edges)
    @test quiver.ekin_max == 0 && quiver.hist == cold.hist
    s.u[1][electron] = 0.75                                                  # γ − 1 = 0.25
    hot = pm_spectrum(s, edges)
    @test hot.ekin_max ≈ 0.25 rtol = 1e-15
    @test hot.hist[2] == s.w[electron]                                       # bins of 0.2: [0.2, 0.4)
    @test sum(hot.hist) ≈ sum(w) rtol = length(w) * eps()
end

# --- the comparison of any pair of codes, the spectra in one unit (lwfa.jl's compare) ----------------

@testset "T-691 the lattice's analytic signal gives a carrier's amplitude and phase, whatever the phase [A5]" begin
    d = derived(TOML.parsefile(CASE_FILES[1]))
    q = quarter_wave(d)
    @test [quarter_wave(derived(TOML.parsefile(f))) for f in CASE_FILES] == [12, 8]
    m0, n = -7, 400
    x = (m0 .+ (0:(n - 1))) .* d.dx
    for φ in (0.0, 0.7, 1.98, -2.5)
        s = carrier_signal(1.3 .* cos.(x .+ φ), m0, q)
        @test s.m0 == m0 + 3q && length(s.z) == n - 6q
        xs = (s.m0 .+ (0:(length(s.z) - 1))) .* d.dx
        @test maximum(abs, abs.(s.z) .- 1.3) < 1e-13                         # a quarter wavelength is q points exactly
        @test maximum(abs, s.z .- 1.3 .* cis.(xs .+ φ)) < 1e-12
    end
    # under a Gaussian envelope of half width τ the amplitude is right to fourth order, at every carrier
    # phase: the two differences leave −(3/8)·(qΔx)⁴·A⁗ of the quadrature, and |A⁗| ≤ 12/τ⁴
    τ, x0 = 26.0, x[n ÷ 2]
    A = exp.(-(x .- x0) .^ 2 ./ τ^2)
    for φ in (0.0, 0.7, 1.98)
        s = carrier_signal(A .* cos.(x .+ φ), m0, q)
        @test maximum(abs, abs.(s.z) .- A[(3q + 1):(n - 3q)]) < 4.5 * (q * d.dx / τ)^4
    end
    @test 4.5 * (q * d.dx / τ)^4 < 1e-4
    # the phase of a minus that of b, on lineouts that start at different points of the one lattice
    xb = (m0 + 5 .+ (0:(n - 1))) .* d.dx
    a = carrier_signal(A .* cos.(x .+ 0.3), m0, q)
    b = carrier_signal(exp.(-(xb .- x0) .^ 2 ./ τ^2) .* cos.(xb .+ 1.5), m0 + 5, q)
    @test carrier_phase(a, b) ≈ 0.3 - 1.5 atol = 1e-4
    @test carrier_phase(b, a) ≈ 1.5 - 0.3 atol = 1e-4
    @test carrier_phase(a, carrier_signal(zeros(n), m0, q)) === nothing
    @test carrier_phase(a, carrier_signal(A, m0 + 2n, q)) === nothing        # no common point
end

# A synthetic output on the lab lattice: a laser of the case's duration with its envelope peak at
# x0 and carrier phase φ, behind it a wake −w·sin(k_p(x − x0)) and its density, and a narrow dip of
# depth 1 in E_x three plasma wavelengths behind the peak.
function synthetic_output(case, step; m0, n, x0, φ = 0.0, a0 = 1.0, w = 0.05, plasma = true, bunch = 1.0e-3)
    d = derived(case)
    kp = sqrt(case["plasma"]["density"])
    τ = case["laser"]["duration"]
    x = (m0 .+ (0:(n - 1))) .* d.dx
    Ey = @. a0 * exp(-(x - x0)^2 / τ^2) * cos(x + φ)
    o = Dict{String,Any}("step" => step, "t" => step * d.dt, "Ey_axis" => Ey, "Ey_m0" => m0, "Ey_absmax" => maximum(abs, Ey),
                         "Ey_envelope" => running_envelope(Ey, envelope_halfwidth(case)))
    plasma || return o
    xe = x .+ d.dx / 2                                                       # E_x's points, first at m = m0 + 1
    o["Ex_axis"] = @. ifelse(xe < x0, -w * sin(kp * (xe - x0)), 0.0) - exp(-(xe - (x0 - 6π / kp))^2 / 4)
    o["Ex_m0"] = m0 + 1
    o["rho_axis"] = @. -case["plasma"]["density"] * (1 + ifelse(x < x0, 0.5 * cos(kp * (x - x0)), 0.0))
    o["rho_m0"] = m0
    h = zeros(Int(case["diagnostics"]["spectrum_ekin"][3]))
    h[1], h[6], h[40] = 5.0, 0.02, bunch                                     # the cold plasma, a warm bin, the bunch
    o["spectrum"] = h
    return o
end

@testset "T-692 compare_rows takes any two codes: pairs in the order a, b; the carrier phase apart from the amplitude, the wake on the anchor's window and its whole-cell shift [A5, A7]" begin
    case = TOML.parsefile(CASE_FILES[1])
    d = derived(case)
    λp = 2π / sqrt(case["plasma"]["density"])
    n, x0 = 3200, 300.0
    mk(; kw...) = [synthetic_output(case, 829; m0 = 10, n, kw...)]
    a = mk(; x0)
    b = mk(; x0, φ = 1.2, a0 = 1.1)                                          # the same pulse, another carrier phase, 10 % higher
    c = mk(; x0 = x0 + 3 * d.dx)                                             # the same pulse and wake three cells ahead
    ab, ba = only(compare_rows(a, b, "plasma", d, λp)), only(compare_rows(b, a, "plasma", d, λp))
    @test ab["Ey_absmax"] == [a[1]["Ey_absmax"], b[1]["Ey_absmax"]] == reverse(ba["Ey_absmax"])
    @test ab["centroid_x"] == reverse(ba["centroid_x"]) && ab["peak_x"] == reverse(ba["peak_x"])
    @test ab["carrier_phase"] ≈ -1.2 atol = 1e-4
    @test ba["carrier_phase"] ≈ 1.2 atol = 1e-4
    @test ab["amplitude_max"] ≈ [1.0, 1.1] rtol = 1e-4                       # the peak, whatever crest the lattice samples
    # which the sampled maximum is not: below the crest by up to 1 − cos(Δx/2), 0.21 % at λ₀/48
    low = 1 - ab["Ey_absmax"][2] / 1.1
    @test 3 * abs(ab["amplitude_max"][2] / 1.1 - 1) < low <= 1 - cos(d.dx / 2) + 1e-5
    # relative to b's rms: a against b is 1 − 1/1.1, b against a is 1.1 − 1, in the amplitude; the
    # running-max envelope adds the carrier phase's crest positions to it
    @test ab["amplitude_relrms"] ≈ 1 - 1 / 1.1 rtol = 1e-3
    @test ba["amplitude_relrms"] ≈ 0.1 rtol = 1e-3
    # a carrier phase alone does not reach the amplitude (the quadrature's fourth-order term is what is left)
    for φ in (1.2, 1.979, π / 2)
        same = only(compare_rows(a, mk(; x0, φ), "plasma", d, λp))
        @test same["amplitude_relrms"] < 4.5 * (quarter_wave(d) * d.dx / case["laser"]["duration"])^4
        @test same["envelope_relrms"] > 0.02
    end
    # the two differences measured between Smilei and the other codes on this case (reports/lwfa-2d-a5.md §3):
    # a lag of two cells is δ/τ in the amplitude (a Gaussian against itself displaced), and with a carrier
    # phase of 1.979 rad the running-max envelope reads 5 %
    lag = only(compare_rows(a, mk(; x0 = x0 - 2 * d.dx), "plasma", d, λp))
    @test lag["amplitude_relrms"] ≈ 2 * d.dx / case["laser"]["duration"] rtol = 1e-2
    @test lag["envelope_relrms"] ≈ lag["amplitude_relrms"] rtol = 0.1
    both = only(compare_rows(a, mk(; x0 = x0 - 2 * d.dx, φ = 1.979), "plasma", d, λp))
    @test both["envelope_relrms"] ≈ 0.050 rtol = 2e-2
    @test both["amplitude_relrms"] ≈ 2 * d.dx / case["laser"]["duration"] rtol = 1e-2
    # c against a: the centroid 3 cells ahead; the wake differs by 2·sin(k_p·3Δx/2) relative rms (a sine
    # over whole periods against itself shifted), and the shift of 3 cells removes the difference
    ac = only(compare_rows(a, c, "plasma", d, λp))
    @test ac["centroid_x"][2] - ac["centroid_x"][1] ≈ 3 * d.dx rtol = 1e-9
    @test ac["wake_Ex_relrms"] ≈ 2 * sin(3 * d.dx * 2π / λp / 2) rtol = 2e-2
    @test ac["wake_Ex_shift"] == 3 && ac["wake_Ex_relrms_shifted"] < 1e-12
    @test only(compare_rows(c, a, "plasma", d, λp))["wake_Ex_shift"] == -3
    @test ac["wake_Ex_min"][1] ≈ -0.05 rtol = 1e-3                           # one and a half wavelengths behind a's peak: the wake, not the dip
    @test WAKE_LENGTH == 1.5
    # the window moved back by the anchor's offset: 1.6 wavelengths further on, it reaches the dip
    @test only(compare_rows(a, c, "plasma", d, λp; anchor_offset = -round(Int, 1.6λp / d.dx)))["wake_Ex_min"][1] < -0.9
    @test only(compare_rows(a, c, "plasma", d, λp; anchor_offset = -24))["wake_Ex_min"][1] ≈ -0.05 rtol = 1e-3
    @test ac["wake_rho_relrms"] > 0
    # the anchor places the window: two wavelengths further back it holds the dip three wavelengths behind
    back = mk(; x0 = x0 - 2λp)
    anchored = only(compare_rows(a, c, "plasma", d, λp; anchor = back))
    @test anchored["wake_Ex_min"][1] < -0.9 && anchored["Ey_absmax"] == ac["Ey_absmax"]
    # the window's length: three and a half wavelengths from a's own peak reach the dip too, one does not
    @test only(compare_rows(a, c, "plasma", d, λp; wake_length = 3.5))["wake_Ex_min"][1] < -0.9
    @test only(compare_rows(a, c, "plasma", d, λp; wake_length = 1))["wake_Ex_min"][1] ≈ -0.05 rtol = 1e-3
    # no wake, no wake's measures: E_x below 10⁻³ of the wave-breaking field in either code leaves the keys out
    quiet = mk(; x0, w = 0.5e-3 * 2π / λp)
    @test !haskey(only(compare_rows(a, quiet, "plasma", d, λp; wake_length = 1)), "wake_Ex_relrms")
    @test !haskey(only(compare_rows(quiet, a, "plasma", d, λp; wake_length = 1)), "wake_Ex_shift")
    @test haskey(only(compare_rows(a, mk(; x0, w = 2.0e-3 * 2π / λp), "plasma", d, λp; wake_length = 1)), "wake_Ex_relrms")
    @test !haskey(ac, "ekin_max_pm")
    a[1]["ekin_max"] = 7.0
    @test only(compare_rows(a, c, "plasma", d, λp))["ekin_max_pm"] == 7.0
    vac = only(compare_rows(mk(; x0, plasma = false), mk(; x0, plasma = false, φ = 0.4), "vacuum", d, λp))
    @test !haskey(vac, "wake_Ex_relrms")
    @test vac["carrier_phase"] ≈ -0.4 atol = 1e-4
    # an output only one code holds is no row
    @test isempty(compare_rows(a, [synthetic_output(case, 1658; m0 = 10, n, x0)], "plasma", d, λp))
end

@testset "T-693 the spectra in one unit: ParticleMesh's weights times Δx·Δy are ∫n dA, thresholds and the highest occupied bin [A3, A5]" begin
    case = small_window_case()
    d = derived(case)
    st, prob = small_window_state(case)
    s = only(st.species)
    alive = s.status .== ParticleMeshCore.ALIVE
    # the load's total against the profile's integral over the loaded columns: n0·L_y·(∫ sin² over the ramp + the plateau)
    P = case["plasma"]
    front = maximum(s.cell[1][alive]) * d.dx - d.xa                          # lab x of the loaded front
    r = P["ramp"]
    exact = P["density"] * d.ny * d.dy * (r / 2 + (front - P["vacuum"] - r))
    @test front > P["vacuum"] + r
    @test sum(s.w[alive]) * spectrum_scale("pm", d) ≈ exact rtol = 1e-4       # the load is a midpoint rule of the ramp
    @test spectrum_scale("smilei", d) == 1 && spectrum_scale("picongpu", d) == 1
    # one bunch of 1e-3 in the bin [7.8, 8.0), seen through each code's unit; PIConGPU's domain holds more cold plasma
    big = TOML.parsefile(CASE_FILES[1])
    D = derived(big)
    out(code, cold) = (o = synthetic_output(big, 829; m0 = 0, n = 400, x0 = 20.0); o["spectrum"][1] = cold; o["spectrum"] ./= spectrum_scale(code, D); o)
    S = Dict("pm-plasma" => Dict("outputs" => [out("pm", 5.0)], "spectrum_edges" => spectrum_edges(big)),
             "smilei-plasma" => Dict("outputs" => [out("smilei", 5.0)]), "picongpu-plasma" => Dict("outputs" => [out("picongpu", 11.0)]))
    S["picongpu-plasma"]["outputs"][1]["spectrum_over"] = 0.0
    sp = spectrum_rows(S, D, ["pm", "smilei", "picongpu"])
    @test sp["thresholds"] ≈ [1.0, 5.0, 10.0, 20.0, 40.0] atol = 1e-12
    rows = [only(sp["rows"][c]) for c in ("pm", "smilei", "picongpu")]
    @test [r["total"] for r in rows] ≈ [5.021, 5.021, 11.021] rtol = 1e-12
    for r in rows
        @test r["above"] ≈ [0.021, 1e-3, 0.0, 0.0, 0.0] atol = 1e-15          # the warm bin [1.0, 1.2) and the bunch; the bunch alone above 5
        @test r["top"] ≈ 8.0 atol = 1e-12                                     # bin 40 of 0.2: [7.8, 8.0)
    end
    @test rows[3]["over"] == 0 && !haskey(rows[1], "over")
    # summaries without a spectrum_region: each histogram covers what its code's does
    @test sp["region"] == "each code's own" && occursin("whole simulated domain", sp["covers"]["picongpu"])
    @test all(r -> r["region"] === nothing, rows)
    empty = spectrum_rows(Dict("pm-plasma" => Dict("outputs" => [Dict("step" => 0, "t" => 0.0, "spectrum" => zeros(500))], "spectrum_edges" => spectrum_edges(big))), D, ["pm"])
    @test only(empty["rows"]["pm"])["top"] === nothing
    # a case whose histogram ends at 20 mₑc² has no threshold beyond it
    a2 = TOML.parsefile(CASE_FILES[2])
    S2 = Dict("pm-plasma" => Dict("outputs" => [synthetic_output(a2, 553; m0 = 0, n = 400, x0 = 20.0)], "spectrum_edges" => spectrum_edges(a2)))
    @test spectrum_rows(S2, derived(a2), ["pm"])["thresholds"] ≈ [1.0, 5.0, 10.0] atol = 1e-12
end

@testset "T-694 compare writes all three pairs, keeps the two-code keys, and copes with a run that has no map [A5, A7]" begin
    file = CASE_FILES[2]
    case = TOML.parsefile(file)
    d = derived(case)
    steps = output_steps(case)
    kmap = steps[Int.(case["diagnostics"]["map_outputs"])][end]
    # three codes: Smilei's carrier 1.98 rad off and its pulse two cells behind, PIConGPU 5 % higher
    variant = Dict("pm" => (; φ = 0.0, lag = 0, a0 = 1.0, bunch = 1.0e-3), "smilei" => (; φ = -1.98, lag = 2, a0 = 1.0, bunch = 1.1e-3),
                   "picongpu" => (; φ = 0.0, lag = 0, a0 = 1.05, bunch = 1.2e-3))
    function write_code(root, code; with_map = false)
        v = variant[code]
        for medium in ("vacuum", "plasma")
            dir = joinpath(root, "$code-$medium")
            mkpath(dir)
            cp(file, joinpath(dir, "case.toml"); force = true)
            outs = map(steps) do k
                t = k * d.dt
                m0 = max(0, floor(Int, (t - case["window"]["start"]) / d.dx)) + (code == "smilei" ? 3 : 0)
                o = synthetic_output(case, k; m0, n = d.nx, x0 = t - case["laser"]["delay"] - v.lag * d.dx, φ = v.φ, a0 = v.a0,
                                     plasma = medium == "plasma", bunch = v.bunch)
                medium == "plasma" && (o["spectrum"] ./= spectrum_scale(code, d); o["spectrum_region"] = collect(spectrum_region(case, k)))
                with_map && medium == "plasma" && k == kmap && (o["map"] = (rho = fill(-0.01, d.nx, d.ny), Ey = zeros(d.nx, d.ny), m0 = m0))
                o
            end
            write_summary(dir, outs; code, stride = Int.(case["diagnostics"]["map_stride"]),
                          extra = Dict{String,Any}("seconds" => 1.5, "spectrum_edges" => spectrum_edges(case)))
            with_map || rm(joinpath(dir, "maps.h5"))
        end
    end
    mktempdir() do root
        runs = joinpath(root, "runs")
        write_code(runs, "pm"); write_code(runs, "smilei")
        two = compare(runs, joinpath(root, "two"))
        @test two["codes"] == ["pm", "smilei"] && two["media"] == ["vacuum", "plasma"]
        @test two["pairs"] == Dict("vacuum" => ["pm", "smilei"], "plasma" => ["pm", "smilei"])
        @test !any(k -> occursin("picongpu", k), keys(two))
        @test !isfile(joinpath(root, "two", "density_maps.png"))             # no run has a map: no figure, no error
        write_code(runs, "picongpu"; with_map = true)
        out = joinpath(root, "three")
        res = compare(runs, out)
        @test res["codes"] == ["pm", "smilei", "picongpu"]
        @test res["pairs"]["smilei_picongpu_plasma"] == ["smilei", "picongpu"] && res["pairs"]["picongpu_vacuum"] == ["pm", "picongpu"]
        # the keys a two-code comparison had are the same rows
        @test res["vacuum"] == two["vacuum"] && res["plasma"] == two["plasma"]
        on_disk = JSON.parsefile(joinpath(out, "comparison.json"))
        for key in ("vacuum", "plasma", "picongpu_vacuum", "picongpu_plasma", "smilei_picongpu_vacuum", "smilei_picongpu_plasma")
            @test length(on_disk[key]) == length(steps)
        end
        late = findfirst(>=(500.0), steps .* d.dt)                           # the pulse well inside every window
        sp, pp, ps = res["smilei_picongpu_vacuum"][late], res["picongpu_vacuum"][late], res["vacuum"][late]
        @test pp["Ey_absmax"][2] / pp["Ey_absmax"][1] ≈ 1.05 rtol = 1e-12    # [ParticleMesh, PIConGPU], the same crests
        @test sp["Ey_absmax"] == [ps["Ey_absmax"][2], pp["Ey_absmax"][2]]    # [Smilei, PIConGPU]
        @test sp["carrier_phase"] ≈ -1.98 atol = 1e-3                        # the synthetic lag moves the envelope, not the carrier
        @test ps["carrier_phase"] ≈ 1.98 atol = 1e-3
        @test abs(pp["carrier_phase"]) < 1e-6
        @test sp["centroid_x"][1] - sp["centroid_x"][2] ≈ -2 * d.dx rtol = 1e-6
        @test pp["amplitude_relrms"] ≈ 1 - 1 / 1.05 rtol = 1e-6
        @test pp["amplitude_max"][2] / pp["amplitude_max"][1] ≈ 1.05 rtol = 1e-6
        # no wake before the pulse is in the window: those rows have no wake keys, here or in the sensitivity rows
        @test !haskey(res["plasma"][1], "wake_Ex_relrms") && haskey(res["plasma"][late], "wake_Ex_relrms")
        @test res["wake"]["length"] == 1.5 && res["wake"]["anchor"] == "pm" && res["wake"]["floor"] ≈ 1e-3 * sqrt(case["plasma"]["density"])
        @test sort(collect(keys(res["wake_sensitivity"]))) == ["1.0", "2.0", "2.5"]
        # the window moved by the anchor's step, half a laser wavelength of this case's grid, either way
        hw = envelope_halfwidth(case)
        @test sort(collect(keys(res["wake_anchor_sensitivity"]))) == sort(["$(-hw)", "$hw"])
        moved = res["wake_anchor_sensitivity"]["$hw"]["smilei_picongpu_plasma"]
        @test length(moved) == length(steps) && moved[late]["wake_Ex_shift"] == 2 && haskey(moved[late], "wake_Ex_min")
        # the maps off the axis: only PIConGPU's run has one here, so no pair has a row's entry
        @test res["maps"]["field"] == "Ey" && length(res["maps"]["bands"]) == 3 && res["maps"]["bands"][1] == [0.0, nothing]
        @test all(r -> !haskey(r, "picongpu_plasma") && haskey(r, "t"), res["maps"]["rows"])
        sens = res["wake_sensitivity"]["1.0"]["smilei_picongpu_plasma"]
        @test length(sens) == length(steps) && !haskey(sens[1], "wake_Ex_relrms") && sens[late]["wake_Ex_shift"] == 2
        @test sens[late]["wake_Ex_relrms"] ≈ res["smilei_picongpu_plasma"][late]["wake_Ex_relrms"] rtol = 0.05   # a sine: any whole number of wavelengths
        # the wakes on ParticleMesh's window: Smilei's two cells behind PIConGPU's
        w = res["smilei_picongpu_plasma"][late]
        @test w["wake_Ex_shift"] == 2 && w["wake_Ex_relrms_shifted"] < 1e-12
        @test res["picongpu_plasma"][late]["wake_Ex_shift"] == 0
        # the spectra: each code's bunch in the one unit, whatever its histogram's own
        @test res["spectrum"]["thresholds"] ≈ [1.0, 5.0, 10.0] atol = 1e-12                    # this case's histogram ends at 20
        @test [res["spectrum"]["rows"][c][late]["above"][1] for c in res["codes"]] ≈ [1.0e-3, 1.1e-3, 1.2e-3] rtol = 1e-12
        @test all(c -> res["spectrum"]["rows"][c][late]["top"] ≈ 4.0, res["codes"])   # bin 40 of 0.1
        # every summary's histogram is over the one region, and the bunch is each code's highest group
        @test res["spectrum"]["region"] == "common" && all(==("spectrum_region"), values(res["spectrum"]["covers"]))
        @test [res["spectrum"]["rows"][c][late]["bunch"]["weight"] for c in res["codes"]] ≈ [1.0e-3, 1.1e-3, 1.2e-3] rtol = 1e-12
        @test res["spectrum"]["macro_weight"] ≈ case["plasma"]["density"] * d.dx * d.dy / 4 rtol = 1e-14
        maps = JSON.parsefile(joinpath(out, "density_maps.json"))
        @test maps["codes"] == ["picongpu"] && maps["no_map"] == ["pm", "smilei"]
        # the map was written at the case's stride on the lattice columns divisible by it, and is drawn as it is
        sx, sy = Int.(case["diagnostics"]["map_stride"])
        @test all(x -> isapprox(mod(x / (sx * d.dx) + 0.5, 1), 0.5; atol = 1e-9), maps["x"]) && length(maps["y"]) == cld(d.ny, sy)
        @test all(v -> v ≈ 1, reduce(vcat, maps["PIConGPU"]))
        laser = JSON.parsefile(joinpath(out, "laser_amplitude.json"))
        @test laser["picongpu-vacuum"]["amplitude_max"][late] / laser["pm-vacuum"]["amplitude_max"][late] ≈ 1.05 rtol = 1e-6
        for name in ("laser_amplitude", "wake_axis", "spectrum", "pair_distances", "density_maps")
            @test isfile(joinpath(out, "$name.png")) && isfile(joinpath(out, "$name.json"))
        end
        spec = JSON.parsefile(joinpath(out, "spectrum.json"))
        @test [spec["pm"][40], spec["smilei"][40], spec["picongpu"][40]] ≈ [1.0e-3, 1.1e-3, 1.2e-3] rtol = 1e-12
        man = TOML.parsefile(joinpath(out, "manifest.toml"))
        @test sort(collect(keys(man["summaries"]))) == sort(["$c-$m" for c in res["codes"] for m in ("vacuum", "plasma")])
        @test man["summaries"]["pm-plasma"] == "sha256:" * RunManifest.file_hash(joinpath(runs, "pm-plasma", "summary.json"))
        # vacuum runs alone (the check with the focus on the injection plane): the laser's pairs, no wake, no spectrum
        vruns = mkpath(joinpath(root, "vacuum-runs"))
        for c in res["codes"]
            cp(joinpath(runs, "$c-vacuum"), joinpath(vruns, "$c-vacuum"))
        end
        vac = compare(vruns, joinpath(root, "vacuum"))
        @test vac["media"] == ["vacuum"] && vac["codes"] == res["codes"]
        @test vac["vacuum"] == res["vacuum"] && vac["smilei_picongpu_vacuum"] == res["smilei_picongpu_vacuum"]
        @test !haskey(vac, "plasma") && !haskey(vac, "spectrum") && !haskey(vac, "wake_sensitivity")
        @test isfile(joinpath(root, "vacuum", "pair_distances.png")) && isfile(joinpath(root, "vacuum", "laser_amplitude.png"))
        @test !isfile(joinpath(root, "vacuum", "wake_axis.png")) && !isfile(joinpath(root, "vacuum", "spectrum.png"))
    end
end

# --- the three lasers are one pulse (lwfa.jl's header, "The laser") ----------------------------------

# Smilei's laser on its injection node, as its source evaluates the namelist's LaserGaussian2D at
# normal incidence (pyprofiles.py:507 to 563, Laser.cpp:460): the B_z amplitude
# tgaussian(t − φ)·a0·√w̃·exp(−(w̃/waist)²·y²)·sin(t − φ), φ(y) = coeff·y² − phase_offset,
# w̃ = (1 + (f/Z_r)²)^(−1/2), coeff = −f·w̃²/(2Z_r²), with f the namelist's focus[0] and y from the axis.
function smilei_node_field(nl, y, t)
    num(key) = parse(Float64, match(Regex("\\b$key = ([-0-9.e]+)"), nl).captures[1])
    f = parse(Float64, match(r"focus = \[([-0-9.e]+),", nl).captures[1])
    a0, waist, fwhm, center, start, duration, po = num("a0"), num("waist"), num("fwhm"), num("center"), num("start"), num("duration"), num("phase_offset")
    Zr = waist^2 / 2
    w = sqrt(1 / (1 + (f / Zr)^2))
    φ = -f * w^2 / (2 * Zr^2) * y^2 - po
    te = t - φ
    envelope = start <= te <= start + duration ? exp(-(te - center)^2 / ((fwhm / 2)^2 / log(2))) : 0.0
    return envelope * a0 * sqrt(w) * exp(-(w / waist)^2 * y^2) * sin(t - φ)
end

@testset "T-698 the Smilei namelist's laser is the case's wavefront pulse on its injection node, two cells behind x = 0 [A1, A5]" for file in CASE_FILES
    case = TOML.parsefile(file)
    d, L = derived(case), case["laser"]
    nl = smilei_namelist(case; plasma = false)
    las = smilei_laser(case)
    # ParticleMesh's pulse with the envelope along the wavefront, whatever the case file selects
    l = GaussianLaser(; a0 = L["a0"], waist = L["waist"], duration = L["duration"], delay = L["delay"], focus = L["focus"],
                      position = 0.0, center = 0.0, envelope = WavefrontTime())
    # on the node at x = −2Δx the pulse of the lab plane x = 0 is 2Δx earlier and 2Δx further from the focus
    ξ, worst = -(L["focus"] + las.back), 0.0
    for y in (0.0, 3.3, -17.0, 40.0, 75.0), t in (L["delay"] .+ (-60.0, -26.0, -3.1, 0.0, 0.4, 11.0, 50.0))
        # what the namelist prescribes, times what the boundary condition launches of it
        worst = max(worst, abs(smilei_node_field(nl, y, t) * smilei_launch_factor(d) - laser_field(l, ξ, y, t + las.back - L["delay"])[1]))
    end
    @test worst <= 1e-12 * L["a0"]
    # the envelope starts as low as the other codes': the cut of tgaussian is where the envelope is e^−16 or less
    @test las.center >= L["delay"] && las.center - L["delay"] < 2π
    @test smilei_node_field(nl, 0.0, 0.0) == 0 || abs(smilei_node_field(nl, 0.0, 0.0)) <= exp(-15.5) * L["a0"]
    # what the first namelist launched (focus from xmin, center = delay, no phase offset): the same envelope on
    # the node, so 2Δx late on x = 0, and the carrier sin(t): against the case's cos(t − delay − ½atan(f/Z_r))
    # and that lag, a phase of delay + ½atan(f/Z_r) − π/2 − 2Δx, the 1.979 rad of the first runs of lwfa-2d
    Zr = L["waist"]^2 / 2
    first_phase = mod2pi(L["delay"] + atan(L["focus"] / Zr) / 2 - π / 2 - las.back)
    basename(file) == "lwfa-2d.toml" && @test 1.9785 <= first_phase <= 1.9790
end

@testset "T-699 the case selects the pulse: lwfa-2d runs the wavefront envelope in ParticleMesh, a case without the key the default; the focus-0 case is lwfa-2d but for the focus [A1, A5]" begin
    a5, a2 = TOML.parsefile(CASE_FILES[1]), TOML.parsefile(CASE_FILES[2])
    @test a5["laser"]["envelope"] == "wavefront" && !haskey(a2["laser"], "envelope")
    @test only(pm_problem(a5; plasma = false).lasers).envelope === WavefrontTime()
    @test only(pm_problem(a2; plasma = false).lasers).envelope === AxisTime()
    @test laser_envelope(a5) === WavefrontTime() && laser_envelope(a2) === AxisTime()
    bad = deepcopy(a5)
    bad["laser"]["envelope"] = "curved"
    @test_throws ErrorException laser_envelope(bad)
    f0 = TOML.parsefile(joinpath(@__DIR__, "..", "cases", "lwfa-2d-focus0.toml"))
    @test f0["laser"]["focus"] == 0 && f0["name"] == "lwfa-2d-focus0"
    same = deepcopy(f0)
    same["laser"]["focus"], same["name"], same["description"] = a5["laser"]["focus"], a5["name"], a5["description"]
    @test same == a5
    # with the focus on the plane the wavefront is flat there: the two envelopes are one field on ParticleMesh's plane
    l0 = only(pm_problem(f0; plasma = false).lasers)
    la = GaussianLaser(; a0 = l0.a0, waist = l0.waist, duration = l0.duration, delay = l0.delay, focus = l0.focus, position = l0.position, center = l0.center)
    @test l0.envelope === WavefrontTime() && l0.position == l0.focus
    @test all(all(isapprox.(laser_field(l0, 0.0, y, s), laser_field(la, 0.0, y, s); rtol = 1e-13, atol = 1e-15)) for y in (0.0, 9.0, -40.0), s in (-30.0, 0.0, 12.5))
    # the generators take it: Smilei's focus is its two ghost cells, PIConGPU's the first node in front of its Huygens surface
    @test smilei_laser(f0).focus == 2 * derived(f0).dx
    u = reference_units(f0["lambda0_um"] * 1e-6)
    @test cnum(picongpu_param_files(f0; plasma = false)["incidentField.param"], "FOCUS_POSITION_Y_SI") ≈ picongpu_layout(f0).j0 * derived(f0).dx * u.length rtol = 1e-15
end

# --- the windows, the region of the spectra and the bunch ------------------------------------------

# What the first runs of lwfa-2d recorded at their 14 outputs (runs/lwfa-a5-1 at 9d3be28, and lwfa-a5pic):
# ParticleMesh's shifts, Smilei's x_moved in cells, and the region the three windows then share.
const PM_SHIFTS_A5 = [0, 0, 0, 336, 1100, 1864, 2628, 3392, 4156, 4920, 5684, 6448, 7212, 7976]
const SMILEI_MOVED_A5 = [0, 0, 0, 360, 1116, 1872, 2664, 3420, 4176, 4932, 5688, 6480, 7236, 7992]

@testset "T-700 the three windows and the region they share: the formulas are what the first runs recorded [A5]" begin
    case = TOML.parsefile(CASE_FILES[1])
    d, lay = derived(case), picongpu_layout(case)
    steps = output_steps(case)
    @test [pm_window_back(case, k) for k in steps] == PM_SHIFTS_A5
    @test [smilei_window_back(case, k) for k in steps] == SMILEI_MOVED_A5
    pic = [picongpu_window_back(case, k) for k in steps]
    @test pic == PIC_WINDOW_OFFSETS["lwfa-2d.toml"] .- lay.j0
    regions = [spectrum_region(case, k) for k in steps]
    # the back is Smilei's (the foremost), the front ParticleMesh's injection column
    @test [r[1] for r in regions] == max.(PM_SHIFTS_A5, SMILEI_MOVED_A5, pic) == SMILEI_MOVED_A5
    @test [r[2] for r in regions] == PM_SHIFTS_A5 .+ d.nx .- 3
    @test regions[end] == (7992, 10277)
    @test all(r -> r[1] >= 0 && d.nx - 2 * smilei_strip(case) < r[2] - r[1] <= d.nx, regions)
    # ParticleMesh's load ends on its injection column: PM_FRONT_FREE columns from the window's front
    st = EMState(pm_problem(case; plasma = false), EMNumerics())
    @test ncells(st.grid)[1] - ParticleMeshElectromagnetic.injection_column(st.aux.window) == PM_FRONT_FREE
    # Smilei's window: every iteration moves it at most once, and never back
    moved = [smilei_window_back(case, k) for k in 0:d.steps]
    patch = d.nx ÷ Int(case["grid"]["smilei_patches"][1])
    @test all(x -> x in (0, patch), diff(moved))
    @test all(k -> moved[k + 1] * d.dx >= (k - 0.5) * d.dt - case["window"]["start"] > (moved[k + 1] - patch) * d.dx || moved[k + 1] == 0, 0:97:d.steps)
end

@testset "T-701 ParticleMesh's spectrum over the region: the columns asked for, and the load's integral over them [A3, A5]" begin
    case = small_window_case()
    d = derived(case)
    st, prob = small_window_state(case)
    s = only(st.species)
    alive = s.status .== ParticleMeshCore.ALIVE
    edges = spectrum_edges(case)
    whole = pm_spectrum(s, edges)
    part = pm_spectrum(s, edges; columns = (40, 70))
    in_columns = alive .& (40 .<= s.cell[1] .< 70)
    @test sum(part.hist) ≈ sum(s.w[in_columns]) rtol = 1e-12
    @test sum(part.hist) < sum(whole.hist) && part.alive == whole.alive
    # the columns 40 to 69 are the lab cells 31 to 60 (8 antenna columns): x from 31Δx to 61Δx, in the ramp (4 to 9)
    P = case["plasma"]
    ramp(x) = (σ = clamp((x - P["vacuum"]) / P["ramp"], 0, 1); P["ramp"] * (σ / 2 - sin(π * σ) / (2π)) + max(x - P["vacuum"] - P["ramp"], 0))
    @test 31 * d.dx > P["vacuum"] && 61 * d.dx < P["vacuum"] + P["ramp"]
    @test sum(part.hist) * spectrum_scale("pm", d) ≈ P["density"] * d.ny * d.dy * (ramp(61 * d.dx) - ramp(31 * d.dx)) rtol = 1e-4
    # the maximum is the region's: an electron given energy outside the columns does not count
    out = findfirst(i -> alive[i] && s.cell[1][i] >= 80 && s.w[i] > 0, eachindex(alive))
    s.u[1][out] = 0.75
    @test pm_spectrum(s, edges).ekin_max ≈ 0.25 rtol = 1e-15
    @test pm_spectrum(s, edges; columns = (40, 70)).ekin_max == 0
end

@testset "T-702 the bunch is the highest group of the spectrum, above its widest gap: strays above a threshold are not in it [A5, A7]" begin
    edges = collect(range(0.0, 100.0; length = 501))
    macro_weight = 1.7e-4
    h = zeros(500)
    h[1:12] .= 3.0                                       # the bulk, up to 2.4
    h[70], h[85] = 4.8e-6, 2.8e-6                        # strays near 14 and 17
    bunch = [262, 263, 287:300..., 303, 305:336...]      # 52.2 to 67.2, with empty bins inside (up to 23 of them)
    h[bunch] .= 2.0e-6
    h[330:336] .= 1.6e-4                                 # most of its weight in a few heavy bins at the top
    g = bunch_group(h, edges; macro_weight)
    @test g["from"] ≈ 52.2 && g["to"] ≈ 67.2 && g["bins"] == length(bunch)
    @test g["weight"] ≈ sum(h[bunch]) rtol = 1e-14
    @test g["gap"] ≈ (262 - 85 - 1) * 0.2 rtol = 1e-12  # the empty run below it, 35.2 wide, against its own 15
    @test g["light_bins"] == length(bunch) - 7           # all but the heavy bins hold less than half a plateau macro-particle
    @test !haskey(bunch_group(h, edges), "light_bins")
    # two strays above 20 mₑc², as PIConGPU had: the threshold counts them, the group does not
    strays = copy(h)
    strays[101], strays[113] = 4.64e-6, 4.64e-6
    gs = bunch_group(strays, edges; macro_weight)
    @test gs["weight"] == g["weight"] && gs["from"] == g["from"]
    @test sum(strays[101:end]) - gs["weight"] ≈ 2 * 4.64e-6 rtol = 1e-10
    @test gs["gap"] ≈ (262 - 113 - 1) * 0.2 rtol = 1e-12
    # through spectrum_rows: `above` 20 has the strays, `bunch` not
    D = derived(TOML.parsefile(CASE_FILES[1]))
    S = Dict("pm-plasma" => Dict("outputs" => [Dict("step" => 1, "t" => 1.0, "spectrum" => strays ./ spectrum_scale("pm", D))], "spectrum_edges" => edges))
    r = only(spectrum_rows(S, D, ["pm"]; macro_weight)["rows"]["pm"])
    @test r["bunch"]["weight"] ≈ g["weight"] rtol = 1e-12
    @test r["above"][4] ≈ g["weight"] + 2 * 4.64e-6 rtol = 1e-12
    # no group when nothing is empty between occupied bins; with equal gaps, the highest
    @test bunch_group([1.0, 2.0, 3.0, 0.0, 0.0], collect(0.0:1.0:5.0)) === nothing
    @test bunch_group(zeros(5), collect(0.0:1.0:5.0)) === nothing
    tie = bunch_group([1.0, 0.0, 1.0, 0.0, 2.0, 2.0], collect(0.0:1.0:6.0))
    @test tie["from"] == 4.0 && tie["weight"] == 4.0 && tie["gap"] == 1.0
end

# --- the readers of the region's spectra -------------------------------------------------------------

@testset "T-703 Smilei's spectrum over the region: the window's histogram less the cells of its strips outside it, and the window's when a strip does not reach [A5]" begin
    file = CASE_FILES[1]
    case = TOML.parsefile(file)
    d = derived(case)
    bins, strip = Int(case["diagnostics"]["spectrum_ekin"][3]), smilei_strip(case)
    steps = output_steps(case)[[1, 4, 9, 14]]
    # a spectrum per cell j of the window (from 0): j + 1 in the first bin, 1 in the seventh
    percell(j) = (h = zeros(bins); h[1] = j + 1; h[7] = 1; h)
    window = sum(percell(j) for j in 0:(d.nx - 1))
    function write_run(dir, moved)
        cp(file, joinpath(dir, "case.toml"))
        write(joinpath(dir, "kind.txt"), "plasma\n")
        h5open(joinpath(dir, "Fields0.h5"), "w") do f
            data = create_group(f, "data")
            for it in steps
                g = create_group(data, lpad(it, 10, '0'))
                HDF5.attributes(g)["x_moved"] = moved(it) * d.dx
                g["Ex"], g["Ey"], g["Rho_electron"] = zeros(d.ny + 1, d.nx + 2), zeros(d.ny + 2, d.nx + 1), zeros(d.ny + 1, d.nx + 1)
            end
        end
        name(it) = "timestep" * lpad(it, 8, '0')
        h5open(joinpath(dir, "ParticleBinning0.h5"), "w") do f
            foreach(it -> f[name(it)] = window, steps)
        end
        # Smilei writes [moving_x][ekin] in C order: (ekin, moving_x) for HDF5.jl
        h5open(joinpath(dir, "ParticleBinning1.h5"), "w") do f
            foreach(it -> f[name(it)] = reduce(hcat, [percell(j) for j in 0:(strip - 1)]), steps)
        end
        h5open(joinpath(dir, "ParticleBinning2.h5"), "w") do f
            foreach(it -> f[name(it)] = reduce(hcat, [percell(j) for j in (d.nx - strip):(d.nx - 1)]), steps)
        end
    end
    mktempdir() do dir
        write_run(dir, it -> smilei_window_back(case, it))
        summarize_smilei(dir)
        s = JSON.parsefile(joinpath(dir, "summary.json"))
        @test s["spectrum_covers"] == "region" && isempty(s["spectrum_failed"])
        for (o, it) in zip(s["outputs"], steps)
            lo, hi = spectrum_region(case, it)
            mx = smilei_window_back(case, it)
            @test o["spectrum_region"] == [lo, hi] && o["spectrum_window"] == window
            @test o["spectrum"] == sum(percell(j) for j in (lo - mx):(hi - mx - 1))
        end
        # at the end Smilei's window leads ParticleMesh's by 16 cells: 19 cells of its front are outside, none of its back
        last_out = s["outputs"][end]
        @test last_out["spectrum"][7] == d.nx - 19 && last_out["spectrum_window"][7] == d.nx
    end
    # a window that is not where smilei_window_back puts it, by more than the strips reach: the summary is
    # written with the window's spectrum and says so, instead of failing
    mktempdir() do dir
        write_run(dir, it -> smilei_window_back(case, it) + (it == steps[end] ? 3 * strip : 0))
        summarize_smilei(dir)
        s = JSON.parsefile(joinpath(dir, "summary.json"))
        @test length(s["spectrum_failed"]) == 1 && occursin("step $(steps[end])", only(s["spectrum_failed"]))
        @test s["outputs"][end]["spectrum"] == window && !haskey(s["outputs"][end], "spectrum_region")
        @test haskey(s["outputs"][end - 1], "spectrum_region")
    end
    @test_throws ErrorException strip_matrix(zeros(3, 4), strip, bins)
    @test strip_matrix(zeros(bins, strip), strip, bins) == zeros(strip, bins)
end

# A PIConGPU run directory with an electron dump (openPMD as PIConGPU 0.8.0 writes it: the species'
# `weighting`, the macro-particles' `momentum`, `positionOffset` in cells) and the plugin's histogram, 7
# real electrons in every bin. One electron per cell of the window along y: weighting 1000 + its
# window cell, at rest but for every 50th, which has u_y = 3 (γ − 1 = √10 − 1 = 2.16, bin 22 of 0.1),
# and the last cell's, u_x = u_z = 2.02 (γ − 1 = 2.03: bin 21). `shift(jw)` is what positionOffset adds
# to the window cell when the window's first total cell is jw.
function write_picongpu_dump(dir, file, shift)
    case = TOML.parsefile(file)
    d, lay = derived(case), picongpu_layout(case)
    u = reference_units(case["lambda0_um"] * 1e-6)
    W, N = lay.window, d.ny
    bins = Int(case["diagnostics"]["spectrum_ekin"][3])
    steps = output_steps(case)
    mc = PIConGPUDeck.ME_SI * PIConGPUDeck.C_SI
    weighting = 1000.0 .+ (0:(W - 1))
    uy = [j % 50 == 0 ? 3.0 : 0.0 for j in 0:(W - 1)]
    ux = [j == W - 1 ? 2.02 : 0.0 for j in 0:(W - 1)]
    cp(file, joinpath(dir, "case.toml"))
    write(joinpath(dir, "kind.txt"), "plasma\n")
    mkpath(joinpath(dir, "openPMD"))
    h5open(joinpath(dir, "openPMD", "simData.h5"), "w") do f
        HDF5.attributes(f)["meshesPath"] = "fields/"
        HDF5.attributes(f)["particlesPath"] = "particles/"
        data = create_group(f, "data")
        for it in steps
            jw = picongpu_window_back(case, it) + lay.j0                      # the window's first total cell
            itg = create_group(data, string(it))
            fl = create_group(itg, "fields")
            E = create_group(fl, "E")
            HDF5.attributes(E)["axisLabels"] = ["y", "x"]
            HDF5.attributes(E)["gridUnitSI"] = u.length
            HDF5.attributes(E)["gridGlobalOffset"] = [jw * d.dx, 0.0]
            E["x"], E["y"] = zeros(N, W), zeros(N, W)
            HDF5.attributes(E["x"])["unitSI"] = u.efield
            HDF5.attributes(E["y"])["unitSI"] = u.efield
            fl["e_all_chargeDensity"] = zeros(N, W)
            HDF5.attributes(fl["e_all_chargeDensity"])["unitSI"] = PIConGPUDeck.QE_SI * u.density
            e = create_group(create_group(itg, "particles"), "e")
            e["weighting"] = collect(weighting)
            mom = create_group(e, "momentum")
            HDF5.attributes(mom)["macroWeighted"] = UInt32(1)
            HDF5.attributes(mom)["weightingPower"] = 1.0
            for (c, v) in (("x", ux), ("y", uy), ("z", ux))
                mom[c] = v .* weighting .* (mc / 2.5)                         # the macro-particle's momentum, in units of 2.5
                HDF5.attributes(mom[c])["unitSI"] = 2.5
            end
            off = create_group(e, "positionOffset")
            off["y"] = Int32.((0:(W - 1)) .+ shift(jw))
            HDF5.attributes(off["y"])["unitSI"] = d.dx * u.length
            off["x"] = zeros(Int32, W)
        end
    end
    open(joinpath(dir, "e_energyHistogram_all.dat"), "w") do io
        println(io, "#step <0 header")
        foreach(it -> println(io, join([it; 0.0; fill(7.0, bins); 0.0; 7.0 * bins], " ")), steps)
    end
    return (; case, d, lay, W, bins, steps, weighting, per_weight = u.density * u.length^2 * d.dx * u.length)
end

@testset "T-704 PIConGPU's spectrum over the region from its electron dump, whichever way positionOffset counts [A5]" for convention in ("the window", "the domain's data", "every slide")
    mktempdir() do dir
        local_y = picongpu_layout(TOML.parsefile(CASE_FILES[2])).local_y
        shift = convention == "the window" ? (jw -> 0) : convention == "every slide" ? (jw -> jw) : (jw -> jw - (jw ÷ local_y) * local_y)
        r = write_picongpu_dump(dir, CASE_FILES[2], shift)
        (; case, lay, W, bins, steps, weighting, per_weight) = r
        summarize_picongpu(dir)
        s = JSON.parsefile(joinpath(dir, "summary.json"))
        @test s["spectrum_covers"] == "region" && isempty(s["spectrum_failed"])
        for (o, it) in zip(s["outputs"], steps)
            lo, hi = spectrum_region(case, it)
            back = picongpu_window_back(case, it)
            cells = [j for j in 0:(W - 1) if lo <= j + back < hi]                 # the window cells of the region
            @test o["spectrum_region"] == [lo, hi] && 0 < length(cells) < W
            @test o["spectrum_plugin"] ≈ fill(7 / per_weight, bins) rtol = 1e-14
            @test o["spectrum"][22] ≈ sum(weighting[j + 1] for j in cells if j % 50 == 0) / per_weight rtol = 1e-12
            @test o["spectrum"][1] ≈ sum(weighting[j + 1] for j in cells if j % 50 != 0 && j != W - 1) / per_weight rtol = 1e-12
            @test o["spectrum_window"][21] ≈ weighting[end] / per_weight rtol = 1e-12   # the window's last cell, ahead of the region
            @test o["spectrum"][21] == 0
            @test sum(o["spectrum_window"]) ≈ sum(weighting) / per_weight rtol = 1e-12
            # the three ways of counting are one while the window has not moved a local domain; after that each is told
            back + lay.j0 > lay.local_y && @test o["position_offset"] == convention
        end
        @test count(it -> picongpu_window_back(case, it) + lay.j0 > lay.local_y, steps) >= 5
        # the maps come at the case's stride, as the other codes'
        h5open(joinpath(dir, "maps.h5"), "r") do f
            g = f[first(keys(f))]
            @test read_attribute(g, "stride") == Int.(case["diagnostics"]["map_stride"]) && read_attribute(g, "m0") % Int(case["diagnostics"]["map_stride"][1]) == 0
        end
    end
end

@testset "T-705 a PIConGPU electron dump that cannot be placed leaves the plugin's histogram as the spectrum, and the summary says so [A5]" begin
    case = TOML.parsefile(CASE_FILES[2])
    lay = picongpu_layout(case)
    u = reference_units(case["lambda0_um"] * 1e-6)
    mktempdir() do dir
        r = write_picongpu_dump(dir, CASE_FILES[2], jw -> 10 * lay.window)   # offsets that end nowhere near the window's front
        summarize_picongpu(dir)
        s = JSON.parsefile(joinpath(dir, "summary.json"))
        @test length(s["spectrum_failed"]) == length(r.steps) && occursin("positionOffset", s["spectrum_failed"][1])
        @test occursin("plugin", s["spectrum_covers"])
        o = s["outputs"][end]
        @test o["spectrum"] == o["spectrum_plugin"] && !haskey(o, "spectrum_region") && !haskey(o, "spectrum_window")
    end
    # picongpu_electrons by itself: a record that is not macro-weighted is taken as it is, a missing unitSI as 1
    mktempdir() do dir
        h5open(joinpath(dir, "p.h5"), "w") do f
            e = create_group(f, "e")
            e["weighting"] = [4.0, 4.0]
            mom = create_group(e, "momentum")
            HDF5.attributes(mom)["macroWeighted"] = UInt32(0)
            mom["y"] = [0.0, 3.0 * PIConGPUDeck.ME_SI * PIConGPUDeck.C_SI]
            off = create_group(e, "positionOffset")
            off["y"] = Int32[0, lay.window - 1]
            el = picongpu_electrons(e, case, u, 0)
            @test el.ekin ≈ [0.0, sqrt(10) - 1] && el.m == [-lay.j0, lay.window - 1 - lay.j0] && el.weight == [4.0, 4.0]
            @test picongpu_histogram(el, [0.0, 1.0, 3.0], -lay.j0, 0) == [4.0, 0.0]
            @test picongpu_histogram(el, [0.0, 1.0, 3.0], typemin(Int), typemax(Int)) == [4.0, 4.0]
            # offsets that end 500 cells beyond the window's front in every reading are refused
            far = create_group(f, "far")
            far["weighting"] = [4.0, 4.0]
            create_group(far, "momentum")["y"] = [0.0, 0.0]
            create_group(far, "positionOffset")["y"] = Int32[0, lay.window + 499]
            @test_throws ErrorException picongpu_electrons(far, case, u, 1000)
        end
    end
end

# --- the maps and a ParticleMesh run end to end ------------------------------------------------------

@testset "T-706 write_summary keeps the maps at the case's stride on the lattice columns the codes share [A5]" begin
    mktempdir() do dir
        rho = [100.0 * i + j for i in 1:40, j in 1:9]                      # 40 columns from m = 7, 9 rows
        outs = [Dict{String,Any}("step" => 5, "t" => 0.6, "map" => (rho = rho, Ey = -rho, m0 = 7)),
                Dict{String,Any}("step" => 6, "t" => 0.7)]
        write_summary(dir, outs; code = "x", stride = [6, 3])
        h5open(joinpath(dir, "maps.h5"), "r") do f
            @test collect(keys(f)) == ["step5"]
            g = f["step5"]
            @test read_attribute(g, "m0") == 12 && read_attribute(g, "stride") == [6, 3]   # the first column with m divisible by 6
            @test read(g["rho"]) == rho[6:6:end, 1:3:end] && read(g["Ey"]) == -rho[6:6:end, 1:3:end]
        end
        @test !haskey(JSON.parsefile(joinpath(dir, "summary.json"))["outputs"][1], "map")
        # without a stride the whole map, from its own m0
        write_summary(dir, outs; code = "x")
        h5open(joinpath(dir, "maps.h5"), "r") do f
            @test read_attribute(f["step5"], "m0") == 7 && read(f["step5"]["rho"]) == rho
        end
    end
end

# A run small enough for a test: 240 × 16 cells, a short weak pulse (a0 = 0.3, so that the plasma stays
# where it was loaded to a few percent), the window moving from t = 24, the plasma from x = 4 with a
# ramp of 5; 400 steps, an output every 100.
function small_run_case()
    case = TOML.parsefile(CASE_FILES[1])
    case["name"] = "lwfa-small"
    case["grid"]["cells"], case["grid"]["antenna_cells"], case["grid"]["smilei_patches"] = [240, 16], 8, [10, 2]
    case["laser"]["a0"], case["laser"]["waist"], case["laser"]["duration"], case["laser"]["delay"], case["laser"]["focus"] = 0.3, 4.0, 2.0, 12.0, 10.0
    case["window"]["start"] = 24.0
    case["plasma"]["vacuum"], case["plasma"]["ramp"] = 4.0, 5.0
    case["time"]["steps"], case["time"]["fields_every"] = 400, 100
    case["diagnostics"]["map_outputs"] = [2, 5]
    return case
end

@testset "T-707 run-pm end to end on a small case: the region's spectrum against the profile, the window's next to it, the maps at their stride [A3, A5]" begin
    case = small_run_case()
    d = derived(case)
    mktempdir() do dir
        open(io -> TOML.print(io, case), joinpath(dir, "case.toml"), "w")
        write(joinpath(dir, "kind.txt"), "plasma\n")
        run_pm(dir)
        s = JSON.parsefile(joinpath(dir, "summary.json"))
        @test s["code"] == "ParticleMesh" && s["spectrum_covers"] == "region" && s["envelope"] == "wavefront" && s["lost"] == [0]
        @test s["backend"] == "KernelAbstractions CPU"
        steps = output_steps(case)
        @test [o["step"] for o in s["outputs"]] == steps && [o["shifts"] for o in s["outputs"]] == [pm_window_back(case, k) for k in steps]
        @test s["outputs"][end]["shifts"] > 100                              # the window has moved
        P = case["plasma"]
        ramp(x) = (σ = clamp((x - P["vacuum"]) / P["ramp"], 0, 1); P["ramp"] * (σ / 2 - sin(π * σ) / (2π)) + max(x - P["vacuum"] - P["ramp"], 0))
        for (o, k) in zip(s["outputs"], steps)
            lo, hi = spectrum_region(case, k)
            @test o["spectrum_region"] == [lo, hi]
            # charge is conserved cell by cell only at the start; later the region's total is the profile's
            # integral to within what the laser has moved across its ends
            total, exact = sum(o["spectrum"]) * spectrum_scale("pm", d), P["density"] * d.ny * d.dy * (ramp(hi * d.dx) - ramp(lo * d.dx))
            @test total ≈ exact rtol = (k == 0 ? 1e-4 : 0.02)
            @test sum(o["spectrum"]) <= sum(o["spectrum_window"]) * (1 + 1e-12)
            @test all(o["spectrum"] .<= o["spectrum_window"] .* (1 + 1e-12) .+ 1e-300)
        end
        # the region is narrower than the window where another code's window starts further on
        @test sum(s["outputs"][end]["spectrum"]) < sum(s["outputs"][end]["spectrum_window"])
        @test s["outputs"][end]["ekin_max"] > 0                              # the laser has been through the plasma
        h5open(joinpath(dir, "maps.h5"), "r") do f
            @test sort(collect(keys(f))) == sort(["step$(steps[2])", "step$(steps[5])"])
            g = f["step$(steps[5])"]
            sx, sy = Int.(case["diagnostics"]["map_stride"])
            @test read_attribute(g, "stride") == [sx, sy] && read_attribute(g, "m0") % sx == 0
            @test size(read(g["rho"])) == (length(mod(-(s["outputs"][5]["rho_m0"]), sx) + 1:sx:(d.nx + d.na)), cld(d.ny, sy))
        end
        man = TOML.parsefile(joinpath(dir, "manifest.toml"))
        @test man["backend"] == "KernelAbstractions CPU" && haskey(man, "run_seconds")
    end
    @test backend_named("cpu") isa CPU
    @test_throws ErrorException backend_named("metal")
end

# --- one pair of runs (lwfa.jl's compare_pair) ------------------------------------------------------

@testset "T-708 compare_pair: two runs of one case with compare's measures, the lineouts' largest difference, each run's spectrum [A4, A5]" begin
    file = CASE_FILES[2]
    case = TOML.parsefile(file)
    d = derived(case)
    steps = output_steps(case)[[1, 6, 9, 14]]
    function write_run(dir, medium; scale = 1.0, ahead = 0, code = "ParticleMesh", bunch = 1.0e-3)
        mkpath(dir)
        cp(file, joinpath(dir, "case.toml"))
        write(joinpath(dir, "kind.txt"), medium * "\n")
        outs = map(steps) do k
            t = k * d.dt
            m0 = max(0, floor(Int, (t - case["window"]["start"]) / d.dx))
            o = synthetic_output(case, k; m0, n = d.nx, x0 = t - case["laser"]["delay"], a0 = scale, plasma = medium == "plasma", bunch)
            if medium == "plasma"
                o["Ex_m0"] += ahead                                          # the same wake, `ahead` cells on
                o["spectrum"] ./= spectrum_scale(code == "ParticleMesh" ? "pm" : "smilei", d)
            end
            o
        end
        write_summary(dir, outs; code, extra = Dict{String,Any}("seconds" => 2.5, "backend" => "test", "spectrum_edges" => spectrum_edges(case)))
        return dir
    end
    mktempdir() do root
        a = write_run(joinpath(root, "a"), "plasma")
        same = compare_pair(a, write_run(joinpath(root, "same"), "plasma"), joinpath(root, "out-same"))
        @test same["medium"] == "plasma" && same["a"]["code"] == "ParticleMesh" && same["a"]["backend"] == "test" && same["a"]["seconds"] == 2.5
        @test length(same["rows"]) == length(steps)
        late = same["rows"][end]
        @test late["amplitude_relrms"] == 0 && late["carrier_phase"] == 0 && late["wake_Ex_relrms"] == 0 && late["wake_Ex_shift"] == 0
        @test late["Ey_axis_maxdiff"] == 0 && late["Ex_axis_maxdiff"] == 0 && late["rho_axis_maxdiff"] == 0
        @test !haskey(same["rows"][1], "wake_Ex_relrms")                        # no wake at t = 0
        # b: the laser a part in 10⁶ higher, the wake one cell ahead, a bunch 2 % heavier, in Smilei's unit
        b = write_run(joinpath(root, "b"), "plasma"; scale = 1 + 1e-6, ahead = 1, code = "Smilei", bunch = 1.02e-3)
        out = joinpath(root, "out")
        res = compare_pair(a, b, out)
        late = res["rows"][end]
        @test late["amplitude_relrms"] ≈ 1e-6 rtol = 1e-3
        @test late["Ey_axis_maxdiff"] ≈ 1e-6 * late["Ey_absmax"][1] rtol = 1e-3
        @test late["wake_Ex_shift"] == 1 && late["wake_Ex_relrms_shifted"] < 1e-12 && late["wake_Ex_relrms"] > 1e-3
        @test late["Ex_axis_maxdiff"] > 1e-4 && late["rho_axis_maxdiff"] == 0
        @test res["spectrum"]["a"][end]["bunch"]["weight"] ≈ 1.0e-3 rtol = 1e-12
        @test res["spectrum"]["b"][end]["bunch"]["weight"] ≈ 1.02e-3 rtol = 1e-12  # each in the one unit, whatever its code's
        @test res["wake"]["anchor"] == "a" && res["wake"]["length"] == WAKE_LENGTH
        on_disk = JSON.parsefile(joinpath(out, "pair.json"))
        @test on_disk["rows"][end]["wake_Ex_shift"] == 1 && on_disk["b"]["code"] == "Smilei"
        man = TOML.parsefile(joinpath(out, "manifest.toml"))
        @test man["summaries"]["b"] == "sha256:" * RunManifest.file_hash(joinpath(b, "summary.json"))
        # a vacuum pair has the laser's measures only; runs of different media or cases are refused
        va, vb = write_run(joinpath(root, "va"), "vacuum"), write_run(joinpath(root, "vb"), "vacuum"; scale = 1.5)
        vac = compare_pair(va, vb, joinpath(root, "out-vac"))
        @test vac["medium"] == "vacuum" && !haskey(vac, "spectrum") && !haskey(vac["rows"][end], "Ex_axis_maxdiff")
        @test vac["rows"][end]["amplitude_relrms"] ≈ 1 / 3 rtol = 1e-9
        @test_throws ErrorException compare_pair(a, va, joinpath(root, "out-x"))
        other = write_run(joinpath(root, "other"), "plasma")
        cp(CASE_FILES[1], joinpath(other, "case.toml"); force = true)
        @test_throws ErrorException compare_pair(a, other, joinpath(root, "out-y"))
    end
end

# --- Smilei's launched amplitude, and the maps off the axis ------------------------------------------

# Smilei's Silver–Müller update at xmin as its source has it (ElectroMagnBC2D_SM.cpp:63 to 66 and 245:
# factor = 1/(k0 + Δt/Δx) with k0 = 1 at normal incidence, B_z[0] = −2·factor·E_y[0] −
# (1 − Δt/Δx)·factor·B_z[1] + 4·factor·b, b the laser's profile at the new B's time), on a 1D Yee grid in
# vacuum: E_y on the nodes i·Δx, B_z on the dual points (j − ½)·Δx, E advanced to n·Δt and B to
# (n + ½)·Δt as Smilei does. Returns the amplitude of E_y at a node 300 cells in for b = sin(t), by a
# least-squares fit of a·cos t + b·sin t over 30 periods after the start has passed.
function silver_muller_amplitude(dx, dt; probe = 300, t0 = 400.0, periods = 30)
    steps = ceil(Int, (t0 + periods * 2π) / dt)
    n = ceil(Int, steps * dt / dx) + probe + 50          # the far end is never reached
    r = dt / dx
    factor = 1 / (1 + r)
    α, β, γ = 2 * factor, -(1 - r) * factor, 4 * factor
    Ey, Bz = zeros(n + 1), zeros(n + 2)
    cc = ss = cs = ec = es = 0.0
    for it in 1:steps
        tdual = (it + 0.5) * dt
        @views @. Ey = Ey - r * (Bz[2:end] - Bz[1:(end - 1)])
        @views @. Bz[2:(end - 1)] = Bz[2:(end - 1)] - r * (Ey[2:end] - Ey[1:(end - 1)])
        Bz[1] = -α * Ey[1] + β * Bz[2] + γ * (1 - exp(-(tdual / 50)^4)) * sin(tdual)
        t = it * dt
        if t >= t0
            c, s = cos(t), sin(t)
            cc += c * c; ss += s * s; cs += c * s; ec += Ey[probe + 1] * c; es += Ey[probe + 1] * s
        end
    end
    det = cc * ss - cs^2
    return hypot((ec * ss - es * cs) / det, (es * cc - ec * cs) / det)
end

@testset "T-710 Smilei's Silver–Müller condition launches 2/(cos(kΔx/2) + cos(Δt/2)) of the amplitude it is given: its update run on a 1D Yee grid [A5]" begin
    for file in CASE_FILES
        d = derived(TOML.parsefile(file))
        @test silver_muller_amplitude(d.dx, d.dt) ≈ smilei_launch_factor(d) rtol = 1e-10
    end
    a5 = derived(TOML.parsefile(CASE_FILES[1]))
    @test smilei_launch_factor(a5) ≈ 1.001984 atol = 1e-6                    # λ₀/48: 0.198 %
    @test smilei_launch_factor(derived(TOML.parsefile(CASE_FILES[2]))) ≈ 1.004474 atol = 1e-6   # λ₀/32: 0.447 %
    # other cells and steps, where the two cosines weigh differently; it is not 1/cos(Δt/2)
    for (dx, dt) in ((0.3, 0.25), (0.3, 0.1), (2π / 48, 0.05))
        @test silver_muller_amplitude(dx, dt) ≈ smilei_launch_factor((; dx, dt)) rtol = 1e-10
    end
    @test abs(smilei_launch_factor(a5) - 1 / cos(a5.dt / 2)) > 1e-4
    @test abs(smilei_launch_factor((; dx = 0.3, dt = 0.1)) - 1 / cos(0.05)) > 5e-3
    # to second order (Δx² + Δt²)/16 above 1
    @test smilei_launch_factor(a5) - 1 ≈ (a5.dx^2 + a5.dt^2) / 16 rtol = 2e-3
end

@testset "T-711 the maps off the axis: the relative rms difference of E_y by band of the distance from the axis, on the columns two maps share [A5, A7]" begin
    case = TOML.parsefile(CASE_FILES[1])
    d = derived(case)
    w0 = case["laser"]["waist"]
    stride = Int.(case["diagnostics"]["map_stride"])
    k = output_steps(case)[Int.(case["diagnostics"]["map_outputs"])][1]
    nrow = cld(d.ny, stride[2])
    y(j) = (1 + (j - 1) * stride[2] - 0.5) * d.dy - d.yc                     # E_y's rows of a map kept at the stride
    # a field that is 1 on the axis band, 2 on the flank, 3 in the wing; b differs by 1 %, 5 % and 20 % there
    level(v) = 0.9w0 <= abs(v) < 1.6w0 ? 2.0 : 2.15w0 <= abs(v) < 2.85w0 ? 3.0 : 1.0
    excess(v) = 0.9w0 <= abs(v) < 1.6w0 ? 0.05 : 2.15w0 <= abs(v) < 2.85w0 ? 0.20 : 0.01
    A = [level(y(j)) * cos(0.3 * i) for i in 1:50, j in 1:nrow]
    B = [(1 + excess(y(j))) * level(y(j)) * cos(0.3 * i) for i in 1:50, j in 1:nrow]
    bands = [(lo * w0, hi * w0) for (lo, hi) in MAP_BANDS]
    r = map_relrms(A, B, d, d.yc, stride, bands)
    @test r[2] ≈ 0.05 / 1.05 rtol = 1e-12                                     # relative to b's rms
    @test r[3] ≈ 0.20 / 1.20 rtol = 1e-12
    @test 0.01 / 1.01 < r[1] < 0.20 / 1.20                                    # everything: between the bands'
    @test map_relrms(A, A, d, d.yc, stride, bands) == [0.0, 0.0, 0.0]
    @test map_relrms(A, B, d, d.yc, stride, [(10.0 * w0, 11.0 * w0)]) == [nothing]   # a band beyond the box
    @test_throws ErrorException map_relrms(A, B[1:40, :], d, d.yc, stride, bands)
    # through the files: two codes' maps that start on different columns are compared on the columns they share
    mktempdir() do runs
        full(scale, m0, n) = (m = m0 .+ (0:(n - 1)); (rho = zeros(n, d.ny), Ey = [scale(( jj - 0.5) * d.dy - d.yc) * cos(0.05 * mm) for mm in m, jj in 1:d.ny], m0 = m0))
        one(v) = level(v)
        other(v) = (1 + excess(v)) * level(v)
        # the sign of Smilei's map is wrong in its first three columns, as a field at the back of a window may be
        edge(mm, v) = mm < 31 + 3 * stride[1] ? -other(v) : other(v)
        for (code, scale, m0) in (("pm", one, 7), ("smilei", other, 31), ("picongpu", other, 31))
            dir = joinpath(runs, "$code-plasma")
            mkpath(dir)
            code == "picongpu" && continue                                    # no map for this one
            M = full(scale, m0, 600)
            code == "smilei" && (M = (; M..., Ey = [edge(mm, (jj - 0.5) * d.dy - d.yc) * cos(0.05 * mm) for mm in m0 .+ (0:599), jj in 1:d.ny]))
            write_summary(dir, [Dict{String,Any}("step" => k, "t" => k * d.dt, "map" => M)]; code, stride)
        end
        rows = map_rows(runs, case, d, [p for p in PAIRS])
        @test rows["field"] == "Ey" && rows["bands"] == [[0.0, nothing], [0.9w0, 1.6w0], [2.15w0, 2.85w0]]
        first_row = rows["rows"][1]
        @test first_row["step"] == k && first_row["t"] == k * d.dt
        @test first_row["plasma"][2] ≈ 0.05 / 1.05 rtol = 1e-12                # ParticleMesh against Smilei
        @test first_row["plasma"][3] ≈ 0.20 / 1.20 rtol = 1e-12
        # the back of the shared columns, a laser wavelength of cells, is left out of that and kept in this
        @test rows["back_margin"] == round(Int, 2π / d.dx) && rows["back_margin"] >= 3 * stride[1]
        @test first_row["plasma_with_back"][2] > 2 * first_row["plasma"][2]
        @test first_row["plasma_with_back"][3] > first_row["plasma"][3]
        @test !haskey(first_row, "picongpu_plasma") && !haskey(first_row, "smilei_picongpu_plasma")
        @test length(rows["rows"]) == length(case["diagnostics"]["map_outputs"]) && !haskey(rows["rows"][2], "plasma")
    end
end
