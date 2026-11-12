# Tests of the LWFA case's generator and readers (scripts/lwfa/lwfa.jl).
#   julia +1.13 --project=scripts scripts/lwfa/test/runtests.jl

using Test
include(joinpath(@__DIR__, "..", "lwfa.jl"))
using ParticleMeshElectromagnetic: ramp_factor

# A5's case and its coarse-grid reference, the input of reports/lwfa-2d-a2.md.
const CASE_FILES = [joinpath(@__DIR__, "..", "cases", f) for f in ("lwfa-2d.toml", "lwfa-2d-a2.toml")]

@testset "T-510 the Smilei namelist carries the case: grid, step, window, the laser's envelope and the density profile [A1, A5]" for file in CASE_FILES
    CASE = TOML.parsefile(file)
    d = derived(CASE)
    nl = smilei_namelist(CASE; plasma = true)
    num(key) = parse(Float64, match(Regex("\\b$key = ([-0-9.e]+)"), nl).captures[1])
    @test occursin("number_of_cells = [$(d.nx), $(d.ny)]", nl)
    @test num("timestep") == d.dt
    @test d.dt == CASE["time"]["courant"] / sqrt(1 / d.dx^2 + 1 / d.dy^2)
    @test floor(Int, num("simulation_time") / d.dt) == d.steps        # Smilei takes floor(simulation_time/timestep) steps
    @test num("time_start") == CASE["window"]["start"]
    @test occursin("focus = [$(repr(Float64(CASE["laser"]["focus"]))), $(repr(d.yc))]", nl)
    # tgaussian (v5.1): exp(−(t − center)²/σ), σ = (fwhm/2)²/ln 2: the same envelope as exp(−(t − delay)²/τ²)
    fwhm, τ, delay = num("fwhm"), CASE["laser"]["duration"], CASE["laser"]["delay"]
    σ = (fwhm / 2)^2 / log(2)
    @test all(t -> isapprox(exp(-(t - delay)^2 / σ), exp(-(t - delay)^2 / τ^2); rtol = 1e-12), 0:13:208)
    @test num("center") == delay && num("duration") == 2 * delay
    # the profile, evaluated as the namelist's Python writes it, against ParticleMesh's at x + x_a
    P = CASE["plasma"]
    smilei_n(x) = P["density"] * sin(0.5 * π * clamp((x - P["vacuum"]) / P["ramp"], 0.0, 1.0))^2
    prob = pm_problem(CASE)
    for x in (0.0, 259.0, 261.5, 290.0, 319.9, 400.0)
        @test smilei_n(x) ≈ P["density"] * ramp_factor(x + d.xa, prob.profile.vacuum, prob.profile.ramp) atol = 1e-15
    end
    @test occursin("time_frozen = frozen", nl) && occursin("(\"ion\", 1.0, 1836.15267343, ", nl)
    vac = smilei_namelist(CASE; plasma = false)
    @test !occursin("Species(", vac) && occursin("fields = [\"Ex\", \"Ey\"]", vac)
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
        maps = h5open(joinpath(dir, "maps.h5"), "r") do f
            Dict(k => size(read(f[k]["rho"])) for k in keys(f))
        end
        @test sort(collect(keys(maps))) == sort(["step$(steps[j])" for j in Int.(CASE["diagnostics"]["map_outputs"])])
        @test all(==((W, N)), values(maps))                                          # (x, y) as the other codes' maps
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
        @test s.m0 == m0 + q && length(s.z) == n - 2q
        xs = (s.m0 .+ (0:(length(s.z) - 1))) .* d.dx
        @test maximum(abs, abs.(s.z) .- 1.3) < 1e-13                         # a quarter wavelength is q points exactly
        @test maximum(abs, s.z .- 1.3 .* cis.(xs .+ φ)) < 1e-12
    end
    # under a Gaussian envelope of half width τ the amplitude is right to (qΔx/τ)², at every carrier phase
    τ, x0 = 26.0, x[n ÷ 2]
    A = exp.(-(x .- x0) .^ 2 ./ τ^2)
    for φ in (0.0, 1.98)
        s = carrier_signal(A .* cos.(x .+ φ), m0, q)
        @test maximum(abs, abs.(s.z) .- A[(q + 1):(n - q)]) < (q * d.dx / τ)^2
    end
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
    # relative to b's rms: a against b is 1 − 1/1.1, b against a is 1.1 − 1, in the amplitude; the
    # running-max envelope adds the carrier phase's crest positions to it
    @test ab["amplitude_relrms"] ≈ 1 - 1 / 1.1 rtol = 2e-2
    @test ba["amplitude_relrms"] ≈ 0.1 rtol = 2e-2
    same = only(compare_rows(a, mk(; x0, φ = 1.2), "plasma", d, λp))
    @test same["amplitude_relrms"] < (quarter_wave(d) * d.dx / case["laser"]["duration"])^2
    @test same["envelope_relrms"] > 5 * same["amplitude_relrms"]
    # the two differences measured between Smilei and the other codes on this case (reports/lwfa-2d-a5.md §3):
    # a lag of two cells is δ/τ in the amplitude (a Gaussian against itself displaced), and with a carrier
    # phase of 1.979 rad the running-max envelope reads 5 %
    lag = only(compare_rows(a, mk(; x0 = x0 - 2 * d.dx), "plasma", d, λp))
    @test lag["amplitude_relrms"] ≈ 2 * d.dx / case["laser"]["duration"] rtol = 1e-2
    @test lag["envelope_relrms"] ≈ lag["amplitude_relrms"] rtol = 0.1
    both = only(compare_rows(a, mk(; x0 = x0 - 2 * d.dx, φ = 1.979), "plasma", d, λp))
    @test both["envelope_relrms"] ≈ 0.050 rtol = 2e-2
    @test both["amplitude_relrms"] ≈ 2 * d.dx / case["laser"]["duration"] rtol = 3e-2
    # c against a: the centroid 3 cells ahead; the wake differs by 2·sin(k_p·3Δx/2) relative rms (a sine
    # over whole periods against itself shifted), and the shift of 3 cells removes the difference
    ac = only(compare_rows(a, c, "plasma", d, λp))
    @test ac["centroid_x"][2] - ac["centroid_x"][1] ≈ 3 * d.dx rtol = 1e-9
    @test ac["wake_Ex_relrms"] ≈ 2 * sin(3 * d.dx * 2π / λp / 2) rtol = 2e-2
    @test ac["wake_Ex_shift"] == 3 && ac["wake_Ex_relrms_shifted"] < 1e-12
    @test only(compare_rows(c, a, "plasma", d, λp))["wake_Ex_shift"] == -3
    @test ac["wake_Ex_min"][1] ≈ -0.05 rtol = 1e-3                           # two wavelengths behind a's peak: the wake, not the dip
    @test ac["wake_rho_relrms"] > 0
    # the anchor places the window: 1.5 wavelengths further back it holds the dip three wavelengths behind
    back = mk(; x0 = x0 - 1.5λp)
    anchored = only(compare_rows(a, c, "plasma", d, λp; anchor = back))
    @test anchored["wake_Ex_min"][1] < -0.9 && anchored["Ey_absmax"] == ac["Ey_absmax"]
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
                medium == "plasma" && (o["spectrum"] ./= spectrum_scale(code, d))
                with_map && medium == "plasma" && k == kmap && (o["map"] = (rho = fill(-0.01, d.nx, d.ny), Ey = zeros(d.nx, d.ny), m0 = m0))
                o
            end
            write_summary(dir, outs; code, extra = Dict{String,Any}("seconds" => 1.5, "spectrum_edges" => spectrum_edges(case)))
            with_map || rm(joinpath(dir, "maps.h5"))
        end
    end
    mktempdir() do root
        runs = joinpath(root, "runs")
        write_code(runs, "pm"); write_code(runs, "smilei")
        two = compare(runs, joinpath(root, "two"))
        @test two["codes"] == ["pm", "smilei"]
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
        # the wakes on ParticleMesh's window: Smilei's two cells behind PIConGPU's
        w = res["smilei_picongpu_plasma"][late]
        @test w["wake_Ex_shift"] == 2 && w["wake_Ex_relrms_shifted"] < 1e-12
        @test res["picongpu_plasma"][late]["wake_Ex_shift"] == 0
        # the spectra: each code's bunch in the one unit, whatever its histogram's own
        @test res["spectrum"]["thresholds"] ≈ [1.0, 5.0, 10.0] atol = 1e-12                    # this case's histogram ends at 20
        @test [res["spectrum"]["rows"][c][late]["above"][1] for c in res["codes"]] ≈ [1.0e-3, 1.1e-3, 1.2e-3] rtol = 1e-12
        @test all(c -> res["spectrum"]["rows"][c][late]["top"] ≈ 4.0, res["codes"])   # bin 40 of 0.1
        maps = JSON.parsefile(joinpath(out, "density_maps.json"))
        @test maps["codes"] == ["picongpu"] && maps["no_map"] == ["pm", "smilei"]
        for name in ("laser_amplitude", "wake_axis", "spectrum", "pair_distances", "density_maps")
            @test isfile(joinpath(out, "$name.png")) && isfile(joinpath(out, "$name.json"))
        end
        spec = JSON.parsefile(joinpath(out, "spectrum.json"))
        @test [spec["pm"][40], spec["smilei"][40], spec["picongpu"][40]] ≈ [1.0e-3, 1.1e-3, 1.2e-3] rtol = 1e-12
        man = TOML.parsefile(joinpath(out, "manifest.toml"))
        @test sort(collect(keys(man["summaries"]))) == sort(["$c-$m" for c in res["codes"] for m in ("vacuum", "plasma")])
        @test man["summaries"]["pm-plasma"] == "sha256:" * RunManifest.file_hash(joinpath(runs, "pm-plasma", "summary.json"))
    end
end
