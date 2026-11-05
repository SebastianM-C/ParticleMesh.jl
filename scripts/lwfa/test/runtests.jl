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
