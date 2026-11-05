# Tests of the LWFA case's generator and readers (scripts/lwfa/lwfa.jl).
#   julia +1.13 --project=scripts scripts/lwfa/test/runtests.jl

using Test
include(joinpath(@__DIR__, "..", "lwfa.jl"))
using ParticleMeshElectromagnetic: ramp_factor

const CASE = TOML.parsefile(joinpath(@__DIR__, "..", "cases", "lwfa-2d-a2.toml"))

@testset "T-510 the Smilei namelist carries the case: grid, step, window, the laser's envelope and the density profile [A1, A5]" begin
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

@testset "T-511 the ParticleMesh problem is the case with the antenna offset: plane on node antenna_cells + 1, focus and profile shifted [A1, A3, A5]" begin
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
    k = 2000
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
end

@testset "T-513 Smilei's dumps are read on the lab lattice: x_moved, the dual and primal axes, the axis rows [A5]" begin
    d = derived(CASE)
    mktempdir() do dir
        cp(joinpath(@__DIR__, "..", "cases", "lwfa-2d-a2.toml"), joinpath(dir, "case.toml"))
        write(joinpath(dir, "kind.txt"), "plasma\n")
        nx, ny = d.nx, d.ny
        steps = output_steps(CASE)
        h5open(joinpath(dir, "Fields0.h5"), "w") do f
            data = create_group(f, "data")
            for (j, it) in enumerate(steps)
                g = create_group(data, lpad(it, 10, '0'))
                HDF5.attributes(g)["x_moved"] = 24 * d.dx * (j - 1)
                # C order [y][x]: HDF5.jl writes a Julia (ny', nx') array as x-first after the reader's permutedims
                Ex = [Float64(i) + 1000 * jj for jj in 0:ny, i in 0:(nx + 1)]           # dual in x
                Ey = [Float64(i) + 1000 * jj for jj in 0:(ny + 1), i in 0:nx]           # dual in y
                ρ = [-Float64(i) - 1000 * jj for jj in 0:ny, i in 0:nx]
                g["Ex"], g["Ey"], g["Rho_electron"] = Ex, Ey, ρ
            end
        end
        h5open(joinpath(dir, "ParticleBinning0.h5"), "w") do f
            for it in steps
                f["timestep" * lpad(it, 8, '0')] = fill(Float64(it), 200)
            end
        end
        write(joinpath(dir, "container.txt"), "# start: 2026-11-05T10:00:00.000000000Z\n# end: 2026-11-05T10:01:30.500000000Z\n")
        summarize_smilei(dir)
        s = JSON.parsefile(joinpath(dir, "summary.json"))
        o = s["outputs"][3]
        @test o["Ex_m0"] == 48 && o["rho_m0"] == 48 && o["Ey_m0"] == 48
        @test o["Ex_axis"][1:3] == [0.0, 1.0, 2.0] .+ 1000 * d.jaxis                  # row jaxis (from 0) is y = y_c
        @test o["rho_axis"][1] == -1000.0 * d.jaxis
        @test o["Ey_axis"][1] == 1000.0 * d.jaxis + 500                                # rows jaxis, jaxis + 1 at y_c ∓ Δy/2
        @test o["spectrum"] == fill(Float64(steps[3]), 200)
        @test s["seconds"] ≈ 90.5
        maps = h5open(joinpath(dir, "maps.h5"), "r") do f
            [k for k in keys(f)]
        end
        @test sort(maps) == sort(["step$(steps[j])" for j in Int.(CASE["diagnostics"]["map_outputs"])])
    end
end
