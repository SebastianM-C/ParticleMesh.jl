# A 2D LWFA case in ParticleMesh, Smilei and PIConGPU, matched through one case file (A5).
#
#   julia +1.13 --project=scripts scripts/lwfa/lwfa.jl prepare <case.toml> <tag> [vacuum]  # scripts/runs/lwfa-<tag>/ and box-request.toml
#   julia +1.13 --project=scripts scripts/lwfa/lwfa.jl run-pm <dir> [cpu|cuda]             # ParticleMesh into <dir>/summary.json, maps.h5
#   julia +1.13 --project=scripts scripts/lwfa/lwfa.jl summarize-smilei <dir>              # Smilei's dumps into the same format
#   julia +1.13 --project=scripts scripts/lwfa/lwfa.jl summarize-picongpu <dir>            # PIConGPU's dumps into the same format
#   julia +1.13 --project=scripts scripts/lwfa/lwfa.jl compare <runs dir> <out dir>        # the comparison, figures and sidecars
#
# A prepared run has six directories: `smilei-vacuum`, `smilei-plasma`, `pm-vacuum`, `pm-plasma`,
# `picongpu-vacuum`, `picongpu-plasma` (with `vacuum`, the three vacuum ones). The vacuum runs check
# that the codes launch the same pulse (A5's "the injected field should be compared in vacuum
# before any plasma run", `questions/wp-017-lwfa-case.md` §3); the plasma runs are the comparison.
# The CPU box request covers ParticleMesh and Smilei; the PIConGPU directories go to the GPU runner
# (`command: picongpu scripts/runs/lwfa-<tag>/picongpu-<kind>`), and `compare` takes PIConGPU in
# when its summaries exist. How PIConGPU is matched, its axes swapped (its window moves along y),
# is in scripts/lwfa/picongpu.jl.
#
# How the codes are matched (each line a choice, the case file holds the numbers):
#
# - Coordinates. The case's x is the lab coordinate with x = 0 on the laser's injection plane:
#   Smilei's xmin at t = 0, and ParticleMesh's antenna plane, `antenna_cells` columns into its box.
#   All use the same Δx, Δy and Δt, so the grids are one lattice in the lab frame: a primal node
#   sits at x = m·Δx, a dual one at (m − ½)·Δx, for an integer m that the summaries record. The
#   comparison matches nodes by m; nothing is interpolated.
#   ParticleMesh: interior node i of the window after s shifts is at m = i − 1 + s − antenna_cells.
#   Smilei: primal index i (from 0) at m = i + x_moved/Δx, its dual index i at the same m (the
#   Fields dumps' `x_moved` attribute; `gridGlobalOffset` is 0, `Params/OpenPMDparams.cpp`).
# - The window. ParticleMesh moves by one cell once c·(t − start) has passed it (it lags the
#   continuous front by less than a cell); Smilei v5.1 moves by a whole patch at the end of an
#   iteration whose half-step time has c·(t − start) beyond what it has moved (`SimWindow::isMoving`,
#   `MovWindow/SimWindow.cpp:93`; the diagnostics of the iteration run before, `Smilei.cpp:663`,
#   `667`), so its back leads by less than a patch (`smilei_window_back`).
# - The laser: one pulse on the lab plane x = 0, the case's `[laser]` with `envelope = "wavefront"`,
#
#       E_y(y, t) = a0·|u(y)|·exp(−σ²/τ²)·cos(σ + ½·atan(ξ/z_R)),   σ = t − delay − (y − y_c)²/(2R),
#
#   ξ = −focus, z_R = waist²/2, R = ξ·(1 + z_R²/ξ²), u the 2D paraxial beam: a cosine carrier counted
#   from the envelope's peak, the 2D Gouy phase, and envelope and carrier on the same time σ along
#   the curved wavefront. What each code does by itself, read in its source:
#   * ParticleMesh, `GaussianLaser` (`lib/ParticleMeshElectromagnetic/src/antenna.jl`): this field
#     with `envelope = WavefrontTime()`. Its default, `AxisTime()`, has the envelope in t − delay
#     alone, the same at every y; that pulse arrives on the axis later, by up to w²/(4|R|) = 0.41 c/ω₀
#     for A5's case (T-696), which the first runs of this case showed (reports/lwfa-2d-a5.md).
#   * PIConGPU 0.8.0, `GaussianPulse`
#     (`include/picongpu/fields/incidentField/profiles/GaussianPulse.hpp`): the carrier is
#     cos(ω·time − k·r + ξ_G + LASER_PHASE) (lines 242, 276 to 280), time counted from the
#     envelope's centre (`TIME_SHIFT`, 221); the Gouy phase ξ_G = atan(−focusPos/z_R), halved in 2D
#     (265 to 267); r = ½·ρ²·R_inv the wavefront's delay (262, 270), and the envelope is taken at
#     time − r/c (284, 286). So it is this pulse as it is.
#   * Smilei v5.1, `LaserGaussian2D` (`Python/pyprofiles.py:507`) through `LaserProfileSeparable`
#     (`ElectroMagn/Laser.cpp:460`): amplitude = envelope(t − φ(y)/ω)·space(y)·sin(ω·t − φ(y)), with
#     φ(y) = −(y − y_c)²/(2|R|) − phase_offset at normal incidence (`pyprofiles.py:531` to `536`, `563`).
#     So the envelope is along the wavefront too, but the carrier is a sine of the time since the
#     start of the run, there is no Gouy phase, and `phase_offset` moves envelope and carrier
#     together. And the Silver–Müller condition applies the laser on the outermost ghost node
#     (`ElectroMagnBC2D_SM.cpp:28` to `31`: index 0 of arrays that hold `oversize` =
#     interpolation_order = 2 ghost cells, `Params.cpp:1217`), two cells behind xmin: the pulse
#     reaches x = 0 two cells late and its focus lies two cells short. The first runs measured
#     both: Smilei's pulse 0.26 c/ω₀ = 2Δx behind, its carrier 1.979 rad from the other codes'
#     (delay + ½·atan(focus/z_R) − π/2 − 2Δx = 1.9787 mod 2π).
#     `smilei_laser` therefore gives the namelist `focus + 2Δx`, an envelope `center` equal to
#     π/2 − ½·atan(ξ′/z_R) mod 2π (which sets the carrier's phase at the envelope's peak) and the
#     `phase_offset` that puts that peak on the ghost node at t = delay − 2Δx (T-698).
#   `tgaussian(start, duration, fwhm, center)` is exp(−(t − center)²/σ) with σ = (fwhm/2)²/ln 2
#   (`pyprofiles.py`), cut to zero outside [start, start + duration]; the cut is at e^−16 of the
#   peak, where ParticleMesh's antenna and PIConGPU's pulse (zero before its start) also begin.
#   A case without `envelope = "wavefront"` runs ParticleMesh's default pulse, which the other two
#   codes do not have; `prepare` says so.
#   The comparison measures the laser by the centroid of E_y² and by the carrier-free amplitude and
#   phase of `carrier_signal`; the running max of |E_y| over one wavelength (`envelope_relrms`)
#   snaps to the crests, so a carrier phase alone moves it by percents, and the box maximum of
#   |E_y| is sampled on the lattice, up to 1 − cos(Δx/2) below the crest (0.21 % at λ₀/48).
# - The plasma. Electrons of n0·sin²((x − vacuum)/ramp·π/2) (then n0), regular, 2 × 2 per cell,
#   cold; ions immobile (ParticleMesh's neutralizing background, Smilei's frozen ions on the same
#   positions). Smilei's regular load sits at 0.975·(k + ½)/n of a cell
#   (`ParticleCreator::createPosition`), ParticleMesh's at (k + ½)/n; Smilei evaluates the density
#   where its creator says, ParticleMesh at each particle (its quiet start), PIConGPU at the cell
#   centre. Small, known differences of the loads; they weigh most where the density is lowest, at
#   the foot of the ramp.
# - The spectra. One region for the three histograms, the lattice cells every code's plasma covers
#   at the output (`spectrum_region`: from the foremost window back, Smilei's, to ParticleMesh's
#   injection column), in one unit (`spectrum_scale`). ParticleMesh bins its particles of that
#   region; Smilei's window histogram has the strips outside it taken off (two more
#   DiagParticleBinning with a `moving_x` axis); PIConGPU's is binned from its particle dump.
# - Numerics. Yee, Esirkepov, quadratic shapes, Boris, no current filter, Float64, Reference mode
#   in ParticleMesh. Transverse boundaries periodic in all codes (the box is 7.2 waists wide);
#   along x Smilei has Silver–Müller boundaries, ParticleMesh its field-free outside, which
#   reflects what reaches the back edge (`questions/wp-028-moving-window.md` §3); the comparison
#   stays away from the back.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
const ParticleMeshCore = ParticleMesh.ParticleMeshCore
using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: window_shifts, window_charge!
using .ParticleMeshCore: guards, ncells, field
using HDF5

const KINDS = ("smilei-vacuum", "smilei-plasma", "pm-vacuum", "pm-plasma")

# --- the case -----------------------------------------------------------------------------------

"The numbers every part derives from the case: cells, steps, Δt, the antenna offset, the axis."
function derived(case)
    g, t = case["grid"], case["time"]
    dx, dy = Float64(g["dx"]), Float64(g["dy"])
    nx, ny = Int.(g["cells"])
    na = Int(g["antenna_cells"])
    dt = Float64(t["courant"]) / sqrt(1 / dx^2 + 1 / dy^2)
    steps, every = Int(t["steps"]), Int(t["fields_every"])
    iseven(ny) || error("the axis must be a node row: an even number of transverse cells, got $ny")
    return (; dx, dy, nx, ny, na, xa = na * dx, dt, steps, every, yc = ny * dy / 2, jaxis = ny ÷ 2)
end

"The output steps: every `fields_every` steps from 0."
output_steps(case) = (d = derived(case); collect(0:d.every:d.steps))

polarization_angle(case) = case["laser"]["polarization"] == "y" ? 0.0 :
    error("polarization $(case["laser"]["polarization"]): only \"y\" is matched")

"The case's `laser.envelope`: \"wavefront\" (the pulse of all three codes) or \"axis\" (ParticleMesh's default, when the key is absent)."
function laser_envelope(case)
    e = get(case["laser"], "envelope", "axis")
    return e == "wavefront" ? WavefrontTime() : e == "axis" ? AxisTime() : error("laser.envelope = $e: \"wavefront\" or \"axis\"")
end

# --- the windows and the region of the spectra -----------------------------------------------------

"Columns at the front of ParticleMesh's window that hold no plasma: its load ends on the injection column, n₁ − 3 for quadratic shapes and no filter (`window_injection_column`; T-701 reads it from a state)."
const PM_FRONT_FREE = 3

"The lattice cell m of the first cell of the case's window in ParticleMesh after step k: its shifts, `max(0, floor((k·Δt − start)/Δx))`."
pm_window_back(case, k) = (d = derived(case); max(0, floor(Int, (k * d.dt - Float64(case["window"]["start"])) / d.dx)))

"""
    smilei_window_back(case, k) -> Int

The lattice cell m of the first cell of Smilei's window in the output of step k: whole patches,
one more at the end of every iteration j whose half-step time (j + ½)·Δt has c·(t − start) beyond
what the window has moved, the output of step k coming before the move of iteration k
(header, "The window").
"""
function smilei_window_back(case, k)
    d = derived(case)
    patch = d.nx ÷ Int(case["grid"]["smilei_patches"][1])
    return patch * max(0, ceil(Int, ((k - 0.5) * d.dt - Float64(case["window"]["start"])) / (patch * d.dx)))
end

"""
    spectrum_region(case, k) -> (lo, hi)

The lattice cells lo ≤ m < hi (cell m spans x from m·Δx to (m + 1)·Δx) whose plasma all three
codes hold in the output of step k: from the foremost of the three windows' backs to the first of
their fronts, ParticleMesh's ending on its injection column. The histograms of the three codes
are taken over these cells, so that their totals and their low energies compare.
"""
function spectrum_region(case, k)
    d = derived(case)
    pm, sm, pic = pm_window_back(case, k), smilei_window_back(case, k), picongpu_window_back(case, k)
    return (max(pm, sm, pic), min(pm + d.nx - PM_FRONT_FREE, sm + d.nx, pic + picongpu_layout(case).window))
end

# --- ParticleMesh -------------------------------------------------------------------------------

"The case as a ParticleMesh `EMProblem{2}` (`plasma = false`: the laser in vacuum)."
function pm_problem(case; plasma = true, T = Float64)
    d = derived(case)
    L, P = case["laser"], case["plasma"]
    laser = GaussianLaser(; a0 = T(L["a0"]), waist = T(L["waist"]), duration = T(L["duration"]), delay = T(L["delay"]),
                          focus = T(L["focus"] + d.xa), position = T(d.xa), center = T(d.yc),
                          polarization = T(polarization_angle(case)), envelope = laser_envelope(case))
    z = (zero(T), zero(T), zero(T))
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(P["density"]), zero(T), (1, 0),
                              Tuple(Int.(P["particles_per_cell"])), z, z, 1)
    n = (d.nx + d.na, d.ny)
    return EMProblem(String(case["name"]), (T(n[1] * d.dx), T(n[2] * d.dy)), n, T(d.dt), d.steps, plasma ? (e,) : (),
                     Int(P["seed"]), (1, 0); lasers = (laser,), boundaries = (:window, :periodic),
                     window = MovingWindow(; start = T(case["window"]["start"])),
                     profile = DensityRamp(; vacuum = T(P["vacuum"] + d.xa), ramp = T(P["ramp"])))
end

# A running max of |v| over ±h points: the envelope of a carrier with h about half a wavelength.
running_envelope(v, h) = [maximum(abs, @view v[max(1, i - h):min(end, i + h)]) for i in eachindex(v)]

"The envelope's half width in points: half a laser wavelength, π/Δx (16 at λ₀/32, 24 at λ₀/48)."
envelope_halfwidth(case) = round(Int, π / derived(case).dx)

# The kinetic-energy histogram of the alive electrons, weighted by the particle weights, and the
# largest kinetic energy among the electrons of positive weight. The density enters through the
# weights (A3), so the loader fills the window's vacuum and the foot of the ramp with particles of
# zero weight; the laser makes them quiver, and they are no electrons (#362). `alive` counts the
# slots, those included. With `columns = (lo, hi)` only the particles of the window's cell columns
# lo ≤ i < hi enter the histogram and the maximum.
function pm_spectrum(s, edges; columns = nothing)
    alive = Array(s.status) .== ParticleMeshCore.ALIVE
    keep = columns === nothing ? alive : alive .& (columns[1] .<= Array(s.cell[1]) .< columns[2])
    u1, u2, u3, w = Array(s.u[1])[keep], Array(s.u[2])[keep], Array(s.u[3])[keep], Array(s.w)[keep]
    ek = @. sqrt(1 + u1^2 + u2^2 + u3^2) - 1
    h = zeros(length(edges) - 1)
    for (e, wi) in zip(ek, w)
        b = searchsortedlast(edges, e)
        1 <= b < length(edges) && (h[b] += wi)
    end
    return (; hist = h, ekin_max = maximum(ek[w .> 0]; init = 0.0), alive = count(alive))
end

spectrum_edges(case) = (s = case["diagnostics"]["spectrum_ekin"]; collect(range(Float64(s[1]), Float64(s[2]); length = Int(s[3]) + 1)))

"One output of a ParticleMesh state at step k: the axis lineouts with their lattice offsets, the box's max |E_y|, the spectra, the map."
function pm_output(st, case, k, ρbuf, keep_map)
    d = derived(case)
    s = window_shifts(st, k)
    g, n = guards(st.grid), ncells(st.grid)
    I, J = (g + 1):(g + n[1]), (g + 1):(g + n[2])
    Ex, Ey = Array(st.fields.Ex)[I, J], Array(st.fields.Ey)[I, J]
    m0 = s - d.na                         # node i ↔ m = i − 1 + m0; Ex index i ↔ m = i + m0 (x = (m − ½)Δx)
    ja = d.jaxis + 1                      # node row of the axis, y = (ja − 1)·Δy = y_c
    out = Dict{String,Any}("step" => k, "t" => k * d.dt, "shifts" => s,
                           "Ex_axis" => Ex[:, ja], "Ex_m0" => 1 + m0,
                           "Ey_axis" => (Ey[:, ja - 1] .+ Ey[:, ja]) ./ 2, "Ey_m0" => m0,      # Ey at y = (j − ½)Δy: rows around the axis
                           "Ey_absmax" => maximum(abs, Ey))
    out["Ey_envelope"] = running_envelope(out["Ey_axis"], envelope_halfwidth(case))
    if !isempty(st.species)
        window_charge!(ρbuf, st)
        ρ = Array(ρbuf)[I, J]
        out["rho_axis"], out["rho_m0"] = ρ[:, ja], m0
        edges = spectrum_edges(case)
        lo, hi = spectrum_region(case, k)
        sp = pm_spectrum(st.species[1], edges; columns = (lo - m0 + 1, hi - m0 + 1))    # cell column i ↔ lattice cell m = i − 1 + m0
        out["spectrum"], out["spectrum_region"], out["ekin_max"], out["alive"] = sp.hist, [lo, hi], sp.ekin_max, sp.alive
        out["spectrum_window"] = pm_spectrum(st.species[1], edges).hist
        keep_map && (out["map"] = (rho = ρ, Ey = Ey, m0 = m0))
    end
    return out
end

"The KernelAbstractions backend of a name: `cpu`, or `cuda` (CUDACore is loaded then, not before)."
function backend_named(name)
    name == "cpu" && return CPU()
    name == "cuda" || error("backend $name: cpu or cuda")
    @eval using CUDACore: CUDACore
    return @eval CUDACore.CUDABackend()
end

backend_label(be) = be isa CPU ? "KernelAbstractions CPU" : string(nameof(typeof(be)))

"""
Run the ParticleMesh column of the case in `dir` (its `case.toml` and `kind.txt`) into
`summary.json` and `maps.h5`, in Reference mode on `backend` (the CPU by default; the command
line's `cuda` runs it on the GPU, the diagnostics on host copies).
"""
function run_pm(dir; backend = CPU())
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    plasma = strip(read(joinpath(dir, "kind.txt"), String)) == "plasma"
    prob = pm_problem(case; plasma)
    st = EMState(prob, EMNumerics(; mode = Reference()); backend)
    initialize!(st, prob)
    p = params(prob)
    ρbuf = field(backend, Float64, st.grid)
    steps = output_steps(case)
    maps_at = Set(steps[Int.(case["diagnostics"]["map_outputs"])])
    outs = Any[pm_output(st, case, 0, ρbuf, 0 in maps_at)]
    t0 = time()
    for k in 1:prob.nsteps
        step!(st, p, k)
        if k in steps
            push!(outs, pm_output(st, case, k, ρbuf, k in maps_at))
            println("pm $(basename(dir)): step $k of $(prob.nsteps), $(round(time() - t0; digits = 1)) s")
            flush(stdout)
        end
    end
    elapsed = time() - t0
    lost = isempty(st.species) ? Int[] : collect(ParticleMeshElectromagnetic.window_lost(st))
    hardware = Dict{String,Any}("threads" => Threads.nthreads(), "cpu" => Sys.cpu_info()[1].model, "backend" => backend_label(backend))
    backend isa CPU || (hardware["gpu"] = @eval CUDACore.name(CUDACore.device()))
    write_summary(dir, outs; code = "ParticleMesh", stride = Int.(case["diagnostics"]["map_stride"]),
                  extra = merge(hardware, Dict{String,Any}("seconds" => elapsed, "lost" => lost, "nsteps" => prob.nsteps,
                                                           "spectrum_edges" => spectrum_edges(case), "spectrum_covers" => "region",
                                                           "envelope" => get(case["laser"], "envelope", "axis"))))
    write_manifest(dir; root = REPO, extra = merge(hardware, Dict{String,Any}("run_seconds" => elapsed, "julia" => string(VERSION))))
    return dir
end

# --- Smilei ---------------------------------------------------------------------------------------

pyf(x) = repr(Float64(x))

"Ghost cells of Smilei's fields at interpolation order 2 (`oversize`, `Params.cpp:1217`): its laser is applied that many cells behind xmin (`ElectroMagnBC2D_SM.cpp:28`)."
const SMILEI_GHOST = 2

"""
    smilei_laser(case) -> (; focus, center, duration, fwhm, phase_offset, back)

The parameters of Smilei's `LaserGaussian2D` and its `tgaussian` that put the case's pulse on the
lab plane x = 0 (header, "The laser"). Smilei's amplitude on its injection node, `back` =
2Δx behind xmin, is tgaussian(t′ − center)·space(y)·sin(t′) with t′ = t + phase_offset +
(y − y_c)²/(2|R|): `focus` is counted from that node, `center` = π/2 − ½·atan(focus/z_R) + 2πn makes
the sine the case's cosine with its Gouy phase at the envelope's peak (the least n with center ≥
the case's delay, so that the envelope starts as low as the other codes'), and `phase_offset` =
center − (delay − back) puts the peak on the node at t = delay − back.
"""
function smilei_laser(case)
    d, L = derived(case), case["laser"]
    back = SMILEI_GHOST * d.dx
    focus = Float64(L["focus"]) + back
    zR = Float64(L["waist"])^2 / 2
    c0 = π / 2 - atan(focus / zR) / 2
    delay = Float64(L["delay"])
    center = c0 + 2π * ceil((delay - c0) / (2π))
    return (; focus, center, duration = 2 * center, fwhm = 2 * Float64(L["duration"]) * sqrt(log(2)),
              phase_offset = center - (delay - back), back)
end

"The cells of the strips at the back and the front of Smilei's window whose spectra its two `moving_x` diagnostics resolve cell by cell: two patches."
smilei_strip(case) = 2 * (derived(case).nx ÷ Int(case["grid"]["smilei_patches"][1]))

"The Smilei v5.1 namelist of the case (`plasma = false`: the laser in vacuum)."
function smilei_namelist(case; plasma = true)
    d = derived(case)
    L, P, D = case["laser"], case["plasma"], case["diagnostics"]
    ppc = Int.(P["particles_per_cell"])
    las = smilei_laser(case)
    patches = Int.(case["grid"]["smilei_patches"])
    io = IOBuffer()
    println(io, """
    # Smilei namelist generated by scripts/lwfa/lwfa.jl from case "$(case["name"])" ($(plasma ? "plasma" : "vacuum")).
    # Do not edit; regenerate from the case file. Units: Smilei normalized, reference frequency = ω₀ (A1).
    import math
    import numpy as np

    Main(
        geometry = "2Dcartesian",
        interpolation_order = 2,
        cell_length = [$(pyf(d.dx)), $(pyf(d.dy))],
        number_of_cells = [$(d.nx), $(d.ny)],
        timestep = $(pyf(d.dt)),
        simulation_time = $(pyf((d.steps + 0.5) * d.dt)),
        number_of_patches = [$(patches[1]), $(patches[2])],
        EM_boundary_conditions = [["silver-muller"], ["periodic"]],
        solve_poisson = False,
        random_seed = $(Int(P["seed"])),
        print_every = 500,
    )

    MovingWindow(
        time_start = $(pyf(case["window"]["start"])),
        velocity_x = 1.0,
    )

    # The laser is applied $(SMILEI_GHOST) cells behind xmin, as sin(t) of the run's time: focus, center and
    # phase_offset put the case's pulse (a cosine from the envelope's peak, with the Gouy phase) on x = 0
    # (smilei_laser in scripts/lwfa/lwfa.jl).
    LaserGaussian2D(
        box_side = "xmin",
        a0 = $(pyf(L["a0"])),
        omega = 1.0,
        focus = [$(pyf(las.focus)), $(pyf(d.yc))],
        waist = $(pyf(L["waist"])),
        polarization_phi = $(pyf(polarization_angle(case))),
        time_envelope = tgaussian(start = 0.0, duration = $(pyf(las.duration)), fwhm = $(pyf(las.fwhm)), center = $(pyf(las.center))),
        phase_offset = $(pyf(las.phase_offset)),
    )

    DiagScalar(every = 50, precision = 10)
    """)
    if plasma
        strip = smilei_strip(case)
        ekin = "[\"ekin\", $(pyf(D["spectrum_ekin"][1])), $(pyf(D["spectrum_ekin"][2])), $(Int(D["spectrum_ekin"][3]))]"
        println(io, """
        def plasma_density(x, y):
            s = np.clip((x - $(pyf(P["vacuum"]))) / $(pyf(P["ramp"])), 0.0, 1.0)
            return $(pyf(P["density"])) * np.sin(0.5 * math.pi * s)**2

        for name, charge, mass, frozen in (("electron", -1.0, 1.0, 0.0), ("ion", 1.0, 1836.15267343, $(pyf(4 * d.steps * d.dt)))):
            Species(
                name = name,
                position_initialization = "regular",
                regular_number = [$(ppc[1]), $(ppc[2])],
                momentum_initialization = "cold",
                particles_per_cell = $(prod(ppc)),
                mass = mass,
                charge = charge,
                number_density = plasma_density,
                time_frozen = frozen,
                boundary_conditions = [["remove", "remove"], ["periodic", "periodic"]],
            )

        # 0: the window's spectrum; 1 and 2: the same cell by cell over the $(strip) cells at its back and at its
        # front, which the reader takes off outside the region all three codes hold (spectrum_region)
        DiagParticleBinning(
            deposited_quantity = "weight",
            every = $(d.every),
            species = ["electron"],
            axes = [$ekin],
        )
        DiagParticleBinning(
            deposited_quantity = "weight",
            every = $(d.every),
            species = ["electron"],
            axes = [["moving_x", 0.0, $(pyf(strip * d.dx)), $strip], $ekin],
        )
        DiagParticleBinning(
            deposited_quantity = "weight",
            every = $(d.every),
            species = ["electron"],
            axes = [["moving_x", $(pyf((d.nx - strip) * d.dx)), $(pyf(d.nx * d.dx)), $strip], $ekin],
        )
        """)
    end
    println(io, """
    DiagFields(
        every = $(d.every),
        fields = [$(plasma ? "\"Ex\", \"Ey\", \"Rho_electron\"" : "\"Ex\", \"Ey\"")],
        datatype = "double",
    )""")
    return String(take!(io))
end

const SMILEI_RUN_SH = """
#!/usr/bin/env bash
# Generated by scripts/lwfa/lwfa.jl. Runs inside the Smilei container with the run directory at
# /run-dir (PM_RUN_DIR) and the worktree read-only at /worktree.
set -euo pipefail
cd "\${PM_RUN_DIR:-/run-dir}"
export OMP_NUM_THREADS=16
export OMP_SCHEDULE=dynamic
{
  echo "# smilei binary: \$(command -v smilei)"
  echo "# smilei version: \$(cat /opt/smilei/VERSION 2>/dev/null || echo unknown)"
  echo "# threads: \$OMP_NUM_THREADS, cpus: \$(nproc)"
  echo "# cpu model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
} > container.txt
echo "# start: \$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> container.txt
smilei namelist.py > smilei.log 2>&1
echo "# end: \$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> container.txt
"""

_xfirst(a::AbstractMatrix) = permutedims(a)

# The datasets `timestep<step>` of a Smilei ParticleBinning file, by step.
function smilei_binning(path)
    out = Dict{Int,Array{Float64}}()
    h5open(path, "r") do f
        for k in keys(f)
            startswith(k, "timestep") || continue
            out[parse(Int, k[9:end])] = Float64.(read(f[k]))
        end
    end
    return out
end

# A strip histogram as (cell, energy bin): Smilei writes [moving_x][ekin] in C order, which HDF5.jl reads as (ekin, moving_x).
function strip_matrix(a, strip, bins)
    size(a) == (bins, strip) && return permutedims(a)
    size(a) == (strip, bins) && strip != bins && return a
    error("a strip histogram of size $(size(a)), expected $bins energy bins by $strip cells")
end

"""
    smilei_region_spectrum(window, back, front, case, it, mx) -> hist

Smilei's spectrum over `spectrum_region(case, it)`: the window's histogram minus the cells of
its back strip behind the region and of its front strip ahead of it (`back`, `front` as
(cell, energy bin), the strips of `smilei_strip` cells at the two ends of the window, whose first
cell is the lattice cell `mx`).
"""
function smilei_region_spectrum(window, back, front, case, it, mx)
    d, strip = derived(case), smilei_strip(case)
    lo, hi = spectrum_region(case, it)
    nb, nf = lo - mx, mx + d.nx - hi                      # cells of the window behind and ahead of the region
    0 <= nb <= strip && 0 <= nf <= strip || error("step $it: the region $lo to $hi leaves $nb and $nf cells of Smilei's window (first cell $mx) outside, beyond its strips of $strip")
    h = copy(window)
    for j in 1:nb
        h .-= @view back[j, :]
    end
    for j in (strip - nf + 1):strip
        h .-= @view front[j, :]
    end
    return h
end

"Read Smilei's dumps in `dir` into the summary format of `pm_output`."
function summarize_smilei(dir)
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    d = derived(case)
    plasma = strip(read(joinpath(dir, "kind.txt"), String)) == "plasma"
    steps = output_steps(case)
    maps_at = Set(steps[Int.(case["diagnostics"]["map_outputs"])])
    outs = Any[]
    bins = Int(case["diagnostics"]["spectrum_ekin"][3])
    spectra = plasma ? smilei_binning(joinpath(dir, "ParticleBinning0.h5")) : Dict{Int,Array{Float64}}()
    # the strips: absent in a run prepared before they existed, and then the spectrum is the window's
    strips = plasma && all(i -> isfile(joinpath(dir, "ParticleBinning$i.h5")), (1, 2)) ?
        (smilei_binning(joinpath(dir, "ParticleBinning1.h5")), smilei_binning(joinpath(dir, "ParticleBinning2.h5"))) : nothing
    covers = strips === nothing ? "window" : "region"
    failed = String[]
    h5open(joinpath(dir, "Fields0.h5"), "r") do f
        data = f["data"]
        for key in sort(collect(keys(data)); by = k -> parse(Int, k))
            it = parse(Int, key)
            grp = data[key]
            xm = Float64(read_attribute(grp, "x_moved"))
            mx = round(Int, xm / d.dx)
            abs(mx * d.dx - xm) <= 1e-6 * d.dx || error("x_moved = $xm is not a whole number of cells")
            Ex, Ey = _xfirst(read(grp["Ex"])), _xfirst(read(grp["Ey"]))   # Ex (nx + 2, ny + 1) dual in x; Ey (nx + 1, ny + 2) dual in y
            ja = d.jaxis + 1                                                # primal row of the axis (index jaxis from 0)
            out = Dict{String,Any}("step" => it, "t" => it * d.dt, "x_moved" => xm,
                                   "Ex_axis" => Ex[:, ja], "Ex_m0" => mx,  # dual index i (from 0) ↔ m = i + mx, at x = (m − ½)Δx
                                   "Ey_axis" => (Ey[:, ja] .+ Ey[:, ja + 1]) ./ 2, "Ey_m0" => mx,   # dual rows jaxis, jaxis + 1 at y_c ∓ Δy/2
                                   "Ey_absmax" => maximum(abs, Ey))
            out["Ey_envelope"] = running_envelope(out["Ey_axis"], envelope_halfwidth(case))
            if plasma
                ρ = _xfirst(read(grp["Rho_electron"]))                      # (nx + 1, ny + 1), primal
                out["rho_axis"], out["rho_m0"] = ρ[:, ja], mx
                window = haskey(spectra, it) ? vec(spectra[it]) : Float64[]
                out["spectrum_window"] = window
                out["spectrum"] = window
                if strips !== nothing && !isempty(window)
                    # a strip that cannot be read or does not reach the region must not cost the summary: the
                    # window's spectrum stays, without a `spectrum_region`, and the summary says why
                    try
                        s = smilei_strip(case)
                        out["spectrum"] = smilei_region_spectrum(window, strip_matrix(strips[1][it], s, bins), strip_matrix(strips[2][it], s, bins), case, it, mx)
                        out["spectrum_region"] = collect(spectrum_region(case, it))
                    catch err
                        push!(failed, "step $it: " * sprint(showerror, err))
                    end
                end
                it in maps_at && (out["map"] = (rho = ρ[1:(end - 1), 1:(end - 1)], Ey = Ey[1:(end - 1), 2:(end - 1)], m0 = mx))
            end
            push!(outs, out)
        end
    end
    secs = try
        lines = readlines(joinpath(dir, "container.txt"))
        stamp(p) = DateTime(first(filter(l -> startswith(l, p), lines))[length(p) + 1:end][1:23], "yyyy-mm-ddTHH:MM:SS.sss")
        Dates.value(stamp("# end: ") - stamp("# start: ")) / 1000
    catch
        nothing
    end
    write_summary(dir, outs; code = "Smilei", stride = Int.(case["diagnostics"]["map_stride"]),
                  extra = Dict{String,Any}("seconds" => secs, "spectrum_edges" => spectrum_edges(case),
                                           "spectrum_covers" => isempty(failed) ? covers : "window at the outputs named in spectrum_failed",
                                           "spectrum_failed" => failed))
    return dir
end

# --- summaries ------------------------------------------------------------------------------------

"""
Write the outputs as `summary.json` (lineouts, scalars, spectra) and the 2D maps as `maps.h5`.
A map keeps every `stride[1]`-th column, those of lattice m divisible by it (so the maps of the
three codes share their columns), and every `stride[2]`-th row from the first; its `m0` is the m of
its first kept column, its `stride` attribute the two strides. At the case's `map_stride` the
three maps of a run are a few megabytes, and the CPU box returns them.
"""
function write_summary(dir, outs; code, extra = Dict{String,Any}(), stride = (1, 1))
    h5open(joinpath(dir, "maps.h5"), "w") do f
        for o in outs
            haskey(o, "map") || continue
            g = create_group(f, "step$(o["step"])")
            mp = o["map"]
            i0 = mod(-mp.m0, stride[1]) + 1
            g["rho"], g["Ey"] = mp.rho[i0:stride[1]:end, 1:stride[2]:end], mp.Ey[i0:stride[1]:end, 1:stride[2]:end]
            HDF5.attributes(g)["m0"] = mp.m0 + i0 - 1
            HDF5.attributes(g)["stride"] = collect(Int, stride)
            HDF5.attributes(g)["t"] = o["t"]
        end
    end
    rows = [Dict{String,Any}(k => v for (k, v) in o if k != "map") for o in outs]
    open(io -> JSON.print(io, merge(Dict{String,Any}("code" => code, "outputs" => rows), extra)), joinpath(dir, "summary.json"), "w")
    return dir
end

# --- prepare --------------------------------------------------------------------------------------

"""
    prepare(case_path, tag; media = ("vacuum", "plasma")) -> root

Write the run directories of `case_path` under scripts/runs/lwfa-<tag>/ (the three codes, each
medium of `media`) and the CPU box request of ParticleMesh and Smilei.
"""
function prepare(case_path, tag; media = ("vacuum", "plasma"))
    case = TOML.parsefile(case_path)
    root = joinpath(REPO, "scripts", "runs", "lwfa-$tag")
    ispath(root) && error("$root exists")
    laser_envelope(case) isa WavefrontTime ||
        @warn "case $(case["name"]) has no laser.envelope = \"wavefront\": ParticleMesh runs its default pulse (the envelope in t − delay alone), Smilei and PIConGPU theirs along the wavefront"
    wanted(kinds) = [k for k in kinds if any(m -> endswith(k, m), media)]
    kinds = wanted(KINDS)
    for kind in kinds
        dir = joinpath(root, kind)
        mkpath(dir)
        cp(case_path, joinpath(dir, "case.toml"))
        plasma = endswith(kind, "plasma")
        write(joinpath(dir, "kind.txt"), plasma ? "plasma\n" : "vacuum\n")
        inputs = ["case.toml", "kind.txt"]
        if startswith(kind, "smilei")
            write(joinpath(dir, "namelist.py"), smilei_namelist(case; plasma))
            write(joinpath(dir, "run.sh"), SMILEI_RUN_SH)
            chmod(joinpath(dir, "run.sh"), 0o755)
            append!(inputs, ["namelist.py", "run.sh"])
        end
        write_manifest(dir; root = REPO, inputs,
                       extra = Dict{String,Any}("run_id" => "lwfa-$tag/$kind", "code" => startswith(kind, "smilei") ? "Smilei v5.1" : "ParticleMesh",
                                                "case" => case["name"], "generator" => "scripts/lwfa/lwfa.jl"))
    end
    for kind in wanted(PIC_KINDS)
        dir = joinpath(root, kind)
        mkpath(dir)
        cp(case_path, joinpath(dir, "case.toml"))
        plasma = endswith(kind, "plasma")
        write(joinpath(dir, "kind.txt"), plasma ? "plasma\n" : "vacuum\n")
        prepare_picongpu(dir, case_path, case; plasma, tag, kind)
    end
    rel(k) = "scripts/runs/lwfa-$tag/$k"
    req = IOBuffer()
    println(req, "# CPU box request of the LWFA case $(case["name"]) (scripts/lwfa/lwfa.jl prepare)")
    println(req, "timeout = 7200\ncategory = \"crosscode\"")
    println(req, "outputs = [", join(repr.(rel.(kinds)), ", "), "]")
    # ParticleMesh first: the box stops at the first failing step, and a Smilei namelist that
    # Smilei refuses should not cost the ParticleMesh column
    julia_step(args...) = println(req, "\n[[steps]]\nkind = \"julia\"\nargs = [", join(repr.(["--project=scripts", "scripts/lwfa/lwfa.jl", args...]), ", "), "]")
    for k in kinds
        startswith(k, "pm") && julia_step("run-pm", rel(k))
    end
    for k in kinds
        startswith(k, "smilei") || continue
        println(req, "\n[[steps]]\nkind = \"smilei\"\ndir = \"$(rel(k))\"")
        julia_step("summarize-smilei", rel(k))
    end
    write(joinpath(root, "box-request.toml"), String(take!(req)))
    return root
end

# --- compare --------------------------------------------------------------------------------------

load_summary(dir) = JSON.parsefile(joinpath(dir, "summary.json"))

# The values of two lineouts on the lattice points m both hold, in the range [mlo, mhi].
function common_points(a, a0, b, b0; mlo = typemin(Int), mhi = typemax(Int))
    ma, mb = a0 .+ (0:(length(a) - 1)), b0 .+ (0:(length(b) - 1))
    lo, hi = max(first(ma), first(mb), mlo), min(last(ma), last(mb), mhi)
    lo <= hi || return (m = Int[], a = Float64[], b = Float64[])
    m = collect(lo:hi)
    return (; m, a = Float64.(a[m .- a0 .+ 1]), b = Float64.(b[m .- b0 .+ 1]))
end

relrms(a, b) = sqrt(sum(abs2, a .- b) / max(sum(abs2, b), eps()))

# The m of the envelope peak (the laser's centre on the axis, to within the carrier: the running
# max snaps to a crest); used to place the wake window.
peak_m(o) = o["Ey_m0"] + argmax(Float64.(o["Ey_envelope"])) - 1

# The energy centroid of E_y² on the axis, in c/ω₀: the laser's position, independent of the
# carrier phase.
function centroid_x(o, dx)
    e = Float64.(o["Ey_axis"]) .^ 2
    m = o["Ey_m0"] .+ (0:(length(e) - 1))
    return sum(e) > 0 ? sum(m .* e) / sum(e) * dx : nothing
end

"A quarter of the laser wavelength in points, π/(2Δx): 12 at λ₀/48, 8 at λ₀/32."
quarter_wave(d) = round(Int, π / (2 * d.dx))

"""
    carrier_signal(v, m0, q) -> (; z, m0)

The analytic signal of a lineout `v` of the laser field (first point at lattice m `m0`) from the
lattice itself, `q` points being a quarter wavelength: for E = A·cos(x − t + φ),
z(m) = E(m) + i·(9·(E(m − q) − E(m + q)) + E(m − 3q) − E(m + 3q))/16 = A·exp(i(x − t + φ)).
`abs(z)` is the amplitude whatever the carrier phase is, and `angle(z)` the phase, which the
running-max envelope cannot separate (it snaps to the crests). The two differences carry
(A(m + q) + A(m − q))/2 and (A(m + 3q) + A(m − 3q))/2 for A, and their weights cancel the
envelope's curvature: what is left of the carrier in `abs(z)` is of order (qΔx/τ)⁴ for an
envelope of width τ (10⁻⁵ for A5's laser), and of the order of the lattice's phase-velocity error
(10⁻⁴ at λ₀/48) where the wavelength is not 4q points. The first and last `3q` points have no
neighbours and are left out, so the returned `m0` is `m0 + 3q`.
"""
function carrier_signal(v, m0, q)
    n = length(v)
    return (; z = ComplexF64[complex(v[i], (9 * (v[i - q] - v[i + q]) + v[i - 3q] - v[i + 3q]) / 16) for i in (3q + 1):(n - 3q)], m0 = m0 + 3q)
end

# The carrier phase of `a` minus that of `b` (rad, E ∝ cos(x − t + φ)), amplitude weighted over the points both hold.
function carrier_phase(za, zb)
    lo, hi = max(za.m0, zb.m0), min(za.m0 + length(za.z), zb.m0 + length(zb.z)) - 1
    lo <= hi || return nothing
    m = lo:hi
    s = sum(za.z[m .- za.m0 .+ 1] .* conj.(zb.z[m .- zb.m0 .+ 1]))
    return iszero(s) ? nothing : angle(s)
end

# The whole-cell shift s, |s| ≤ smax, for which b(m + s) is closest to a(m) on the lattice range
# [mlo, mhi], and the relative rms difference there: s > 0 says b's wake is s cells ahead of a's.
function best_shift(a, a0, b, b0; mlo, mhi, smax)
    best = (shift = 0, relrms = Inf)
    for s in -smax:smax
        c = common_points(a, a0, b, b0 - s; mlo, mhi)
        isempty(c.m) && continue
        r = relrms(c.a, c.b)
        r < best.relrms && (best = (shift = s, relrms = r))
    end
    return best
end

"The least |E_x| in the window, in units of the cold wave-breaking field mₑcω_p/e, for which an output has a wake to compare."
const WAKE_FLOOR = 1.0e-3

"""
    compare_rows(a, b, medium, d, λp; anchor = a, wake_length = 2) -> rows

The comparison of the outputs of code `a` with those of code `b` in one medium, a row per output
both hold; every pair of numbers is [a, b], and every relative rms difference is relative to b's
rms. The laser: the centroid of E_y², the carrier-free amplitude on the axis (its maximum,
`amplitude_max`, and the relative rms difference, `amplitude_relrms`), the carrier phase of a
minus b, and for reference the box maximum of |E_y| (sampled on the lattice: up to
1 − cos(Δx/2) below a crest), the running-max envelope's peak and its relative rms difference
(`envelope_relrms`, which a carrier phase between the codes inflates). In plasma, the wake on the
axis from the laser's peak back `wake_length` plasma wavelengths: E_x (relative rms difference,
extrema, and the same after the whole-cell shift of b that fits best) and ρ_e. The wake's window is
placed at the envelope peak of `anchor`'s output, so that with one anchor several pairs are
compared on the same lattice points; an output has the wake's keys only when |E_x| reaches
`WAKE_FLOOR` of the wave-breaking field in the window in both codes (before the laser is in the
plasma E_x is roundoff, and its relative difference means nothing). `wake_rho_relrms` is kept for
the record and not a measure to count: ρ_e is a few narrow spikes, whose cell-wide displacements
and the anchor's choice move it by tens of percent, and the window holds the pulse, under which
ρ_e follows the carrier.
"""
function compare_rows(a, b, medium, d, λp; anchor = a, wake_length = 2)
    bystep = Dict(o["step"] => o for o in b)
    anchors = Dict(o["step"] => o for o in anchor)
    q = quarter_wave(d)
    floor_Ex = WAKE_FLOOR * 2π / λp
    rows = Dict{String,Any}[]
    for o in a
        s = get(bystep, o["step"], nothing)
        s === nothing && continue
        env = common_points(o["Ey_envelope"], o["Ey_m0"], s["Ey_envelope"], s["Ey_m0"])
        za, zb = carrier_signal(Float64.(o["Ey_axis"]), o["Ey_m0"], q), carrier_signal(Float64.(s["Ey_axis"]), s["Ey_m0"], q)
        amp = common_points(abs.(za.z), za.m0, abs.(zb.z), zb.m0)
        r = Dict{String,Any}("step" => o["step"], "t" => o["t"],
                             "Ey_absmax" => [o["Ey_absmax"], s["Ey_absmax"]],
                             "peak_x" => [peak_m(o) * d.dx, peak_m(s) * d.dx],
                             "centroid_x" => Any[centroid_x(o, d.dx), centroid_x(s, d.dx)],
                             "envelope_relrms" => isempty(env.m) ? nothing : relrms(env.a, env.b),
                             "amplitude_max" => [maximum(abs, za.z; init = 0.0), maximum(abs, zb.z; init = 0.0)],
                             "amplitude_relrms" => isempty(amp.m) ? nothing : relrms(amp.a, amp.b),
                             "carrier_phase" => carrier_phase(za, zb))
        if medium == "plasma"
            # the wake: from the laser's peak back wake_length plasma wavelengths, on the axis
            mp = peak_m(get(anchors, o["step"], o))
            mlo = mp - round(Int, wake_length * λp / d.dx)
            wake = common_points(o["Ex_axis"], o["Ex_m0"], s["Ex_axis"], s["Ex_m0"]; mlo, mhi = mp)
            rho = common_points(o["rho_axis"], o["rho_m0"], s["rho_axis"], s["rho_m0"]; mlo, mhi = mp)
            if !isempty(wake.m) && min(maximum(abs, wake.a), maximum(abs, wake.b)) >= floor_Ex
                r["wake_Ex_relrms"] = relrms(wake.a, wake.b)
                r["wake_Ex_min"] = [minimum(wake.a), minimum(wake.b)]
                r["wake_Ex_max"] = [maximum(wake.a), maximum(wake.b)]
                r["wake_rho_relrms"] = relrms(rho.a, rho.b)
                fit = best_shift(o["Ex_axis"], o["Ex_m0"], s["Ex_axis"], s["Ex_m0"]; mlo, mhi = mp, smax = q)
                r["wake_Ex_shift"] = fit.shift
                r["wake_Ex_relrms_shifted"] = fit.relrms
            end
            # ParticleMesh's own maximum, as its summary has it: over every alive particle in summaries
            # written before #362 was fixed, zero weights included; `spectrum` below has the comparable one
            haskey(o, "ekin_max") && (r["ekin_max_pm"] = o["ekin_max"])
        end
        push!(rows, r)
    end
    return rows
end

"The codes whose summaries are under `runs`: ParticleMesh and Smilei always, PIConGPU when its summaries exist for every medium there."
function present_codes(runs)
    codes = ["pm", "smilei"]
    media = present_media(runs)
    all(m -> isfile(joinpath(runs, "picongpu-$m", "summary.json")), media) && push!(codes, "picongpu")
    return codes
end

"The media of the runs under `runs`: vacuum, and plasma when ParticleMesh's plasma run is there."
present_media(runs) = isdir(joinpath(runs, "pm-plasma")) ? ["vacuum", "plasma"] : ["vacuum"]

# The pairs `compare` writes, each [a, b], and the prefix of their keys in comparison.json.
const PAIRS = (("", "pm", "smilei"), ("picongpu_", "pm", "picongpu"), ("smilei_picongpu_", "smilei", "picongpu"))

# --- the spectra in one unit ----------------------------------------------------------------------

# The factor that brings a code's histogram to electrons per unit depth in A1's units, n_c·(c/ω₀)².
# ParticleMesh's weights are density per particle of a cell (`quiet_particle_2d`: n/ppc), so their
# sum over a region is ∫n dA/(Δx·Δy) (T-693); Smilei's ParticleBinning of "weight" and the PIConGPU
# reader give ∫n dA already (at t = 0 each total of the first runs is the profile's integral over
# the code's domain, reports/lwfa-2d-a5.md).
spectrum_scale(code, d) = code == "pm" ? d.dx * d.dy : 1.0

# The region each code's histogram covers when it is not `spectrum_region` (summaries without a
# `spectrum_region`: the runs before it existed, or a reader that fell back).
const SPECTRUM_COVERS = Dict(
    "pm" => "its window: the case's window and the antenna_cells columns behind it",
    "smilei" => "its window, whose back leads the case's by less than a patch",
    "picongpu" => "its whole simulated domain: the local domains it holds, which reach behind the window's back and ahead of its front")

"""
    bunch_group(h, edges; macro_weight = nothing) -> Dict or nothing

The highest group of a spectrum `h`: the occupied bins above the widest run of empty bins between
occupied ones (the highest such run when several are as wide). A trapped bunch that gains energy
leaves the rest of the spectrum behind, so the widest gap is below it; a threshold in energy
instead also counts whatever stray electron lies above it. `from` and `to` are the group's energy
range, `weight` its sum, `bins` its occupied bins (a lower bound of its macro-particles), `gap` the
width of the empty run below it (how well it is separated; compare it with to − from), and, with
the weight of one plateau macro-particle, `light_bins` the bins that hold less than half of it
(particles loaded where the density is low, at the foot of the ramp). `nothing` when no bin is
empty between occupied ones.
"""
function bunch_group(h, edges; macro_weight = nothing)
    occ = findall(>(0), h)
    length(occ) >= 2 || return nothing
    gaps = diff(occ) .- 1
    g = maximum(gaps)
    g > 0 || return nothing
    grp = occ[(findlast(==(g), gaps) + 1):end]
    r = Dict{String,Any}("from" => edges[first(grp)], "to" => edges[last(grp) + 1], "weight" => sum(h[grp]), "bins" => length(grp),
                         "gap" => g * (edges[2] - edges[1]))
    macro_weight === nothing || (r["light_bins"] = count(<(macro_weight / 2), h[grp]))
    return r
end

"""
    spectrum_rows(S, d, codes; macro_weight = nothing, thresholds) -> Dict

The electron spectra of the plasma runs in one unit (electrons per unit depth, n_c·(c/ω₀)²): per
code and output the total, the weight above each threshold (mₑc², moved to the nearest bin edge),
`top`, the upper edge of the highest occupied bin (the maximum energy that compares across the
codes: each has it from its own weighted histogram), and `bunch`, the highest energy group
(`bunch_group`). `region` is "common" when every summary's histograms are over the same
`spectrum_region` at every output; otherwise each histogram covers what `covers` says, and totals
and low energies compare only where those agree.
"""
function spectrum_rows(S, d, codes; macro_weight = nothing, thresholds = (1.0, 5.0, 10.0, 20.0, 40.0))
    edges = Float64.(S["pm-plasma"]["spectrum_edges"])
    cut = unique(argmin(abs.(edges .- t)) for t in thresholds if t < last(edges))
    rows = Dict{String,Any}()
    for c in codes
        f = spectrum_scale(c, d)
        rows[c] = map(S["$c-plasma"]["outputs"]) do o
            h = Float64.(get(o, "spectrum", Float64[])) .* f
            top = findlast(>(0), h)
            r = Dict{String,Any}("step" => o["step"], "t" => o["t"], "total" => sum(h),
                                 "above" => isempty(h) ? nothing : [sum(@view h[i:end]) for i in cut],
                                 "top" => top === nothing ? nothing : edges[top + 1],
                                 "bunch" => isempty(h) ? nothing : bunch_group(h, edges; macro_weight),
                                 "region" => get(o, "spectrum_region", nothing))
            haskey(o, "spectrum_over") && (r["over"] = o["spectrum_over"] * f)
            r
        end
    end
    regions = [[r["region"] for r in rows[c]] for c in codes]
    common = all(r -> !any(isnothing, r) && r == regions[1], regions)
    return Dict{String,Any}("unit" => "electrons per unit depth, n_c (c/ω₀)²", "thresholds" => edges[cut],
                            "macro_weight" => macro_weight, "region" => common ? "common" : "each code's own",
                            "covers" => Dict(c => any(isnothing, [r["region"] for r in rows[c]]) ? SPECTRUM_COVERS[c] : "spectrum_region" for c in codes),
                            "rows" => rows)
end

"The lengths of the wake's window, in plasma wavelengths, at which `compare` repeats the wake's measures next to its own 2."
const WAKE_LENGTHS = (1.0, 1.5, 2.5)

"""
Compare the summaries under `runs` (pm-*, smilei-*, and picongpu-* when present; the vacuum runs
alone when there is no pm-plasma) and write the figures, sidecars and `comparison.json` into
`out`. `comparison.json` keeps ParticleMesh against Smilei under `vacuum` and `plasma`,
ParticleMesh against PIConGPU under `picongpu_vacuum` and `picongpu_plasma`, and Smilei against
PIConGPU under `smilei_picongpu_vacuum` and `smilei_picongpu_plasma` (`pairs` names the two codes
of each key); the rows are `compare_rows`', every wake window two plasma wavelengths long and
placed at ParticleMesh's envelope peak (`wake`). `wake_sensitivity` repeats the wake's E_x
measures for the windows of `WAKE_LENGTHS`, and `spectrum` holds the spectra in one unit
(`spectrum_rows`).
"""
function compare(runs, out)
    mkpath(out)
    case = TOML.parsefile(joinpath(runs, "pm-vacuum", "case.toml"))
    d = derived(case)
    λp = 2π / sqrt(case["plasma"]["density"])
    codes, media = present_codes(runs), present_media(runs)
    kinds = ["$c-$m" for c in codes for m in media]
    S = Dict(k => load_summary(joinpath(runs, k)) for k in kinds)
    # the envelope again from Ey_axis with the case's half width, for summaries written with a fixed 16 points
    h = envelope_halfwidth(case)
    for k in kinds, o in S[k]["outputs"]
        o["Ey_envelope"] = running_envelope(Float64.(o["Ey_axis"]), h)
    end
    result = Dict{String,Any}("case" => case["name"], "dx" => d.dx, "dt" => d.dt, "lambda_p" => λp, "codes" => codes, "media" => media)
    pairs = [p for p in PAIRS if p[2] in codes && p[3] in codes]
    result["pairs"] = Dict(prefix * medium => [a, b] for (prefix, a, b) in pairs for medium in media)
    rows(a, b, medium; kw...) = compare_rows(S["$a-$medium"]["outputs"], S["$b-$medium"]["outputs"], medium, d, λp; anchor = S["pm-$medium"]["outputs"], kw...)
    for medium in media, (prefix, a, b) in pairs
        result[prefix * medium] = rows(a, b, medium)
    end
    if "plasma" in media
        result["wake"] = Dict{String,Any}("length" => 2.0, "anchor" => "pm", "floor" => WAKE_FLOOR * 2π / λp)
        wake_keys = ("step", "t", "wake_Ex_relrms", "wake_Ex_shift", "wake_Ex_relrms_shifted")
        result["wake_sensitivity"] = Dict(string(len) => Dict(prefix * "plasma" => [Dict(k => r[k] for k in wake_keys if haskey(r, k)) for r in rows(a, b, "plasma"; wake_length = len)]
                                                              for (prefix, a, b) in pairs) for len in WAKE_LENGTHS)
        P = case["plasma"]
        result["spectrum"] = spectrum_rows(S, d, codes; macro_weight = P["density"] * d.dx * d.dy / prod(Int.(P["particles_per_cell"])))
    end
    result["seconds"] = Dict(k => get(S[k], "seconds", nothing) for k in kinds)
    open(io -> JSON.print(io, result, 2), joinpath(out, "comparison.json"), "w")
    plot_laser(out, S, d, codes, media)
    plot_pairs(out, result, pairs, media)
    if "plasma" in media
        plot_wake(out, S, case, d, λp, codes)
        plot_maps(out, runs, case, d, codes)
        plot_spectra(out, S, d, codes)
    end
    write_manifest(out; root = REPO,
                   extra = Dict{String,Any}("runs" => relpath(abspath(runs), REPO), "case" => case["name"],
                                            "summaries" => Dict(k => "sha256:" * RunManifest.file_hash(joinpath(runs, k, "summary.json")) for k in kinds)))
    return result
end

const COLORS = (pm = "#1f77b4", smilei = "#d62728", picongpu = "#2ca02c")
const NAMES = Dict("pm" => "ParticleMesh", "smilei" => "Smilei", "picongpu" => "PIConGPU")
const STYLES = Dict("pm" => :solid, "smilei" => :dash, "picongpu" => :dot)

# The laser's amplitude against time: the maximum of the carrier-free amplitude on the axis
# (`carrier_signal`), which does not depend on where the lattice samples the crests.
function plot_laser(out, S, d, codes, media)
    fig = Figure(size = (550 * length(media), 420))
    side = Dict{String,Any}()
    q = quarter_wave(d)
    for (col, medium) in enumerate(media)
        ax = Axis(fig[1, col]; xlabel = "t (1/ω₀)", ylabel = "max of the amplitude on the axis (mₑcω₀/e)", title = "laser amplitude, $medium")
        for key in codes
            o = S["$key-$medium"]["outputs"]
            t = Float64[x["t"] for x in o]
            a = Float64[maximum(abs, carrier_signal(Float64.(x["Ey_axis"]), x["Ey_m0"], q).z; init = 0.0) for x in o]
            scatterlines!(ax, t, a; label = NAMES[key], color = getfield(COLORS, Symbol(key)))
            side["$key-$medium"] = Dict("t" => t, "amplitude_max" => a, "Ey_absmax" => Float64[x["Ey_absmax"] for x in o])
        end
        axislegend(ax; position = :lt)
    end
    save_plot(out, "laser_amplitude", fig, side)
end

function plot_wake(out, S, case, d, λp, codes)
    pm = S["pm-plasma"]["outputs"]
    others = [c for c in codes if c != "pm"]
    bystep = Dict(c => Dict(o["step"] => o for o in S["$c-plasma"]["outputs"]) for c in others)
    picks = pm[Int.(get(case["diagnostics"], "wake_outputs", [6, 9, 13]))]   # in the plasma: t ≈ 500, 800, 1200 for A5's case
    fig = Figure(size = (1100, 300 * length(picks)))
    side = Dict{String,Any}()
    for (row, o) in enumerate(picks)
        mp = peak_m(o)
        lo, hi = mp - round(Int, 2.5λp / d.dx), mp + round(Int, 0.5λp / d.dx)
        for (col, (key, m0key, lab)) in enumerate((("Ex_axis", "Ex_m0", "E_x on axis"), ("rho_axis", "rho_m0", "ρ_e on axis (n_c e)")))
            ax = Axis(fig[row, col]; xlabel = "x (c/ω₀), lab frame", ylabel = lab, title = "t = $(round(o["t"]; digits = 1))")
            # the points every code holds, so that the curves share one x range
            l, h = lo, hi
            for c in others
                s = bystep[c][o["step"]]
                m = common_points(o[key], o[m0key], s[key], s[m0key]; mlo = l, mhi = h).m
                isempty(m) || ((l, h) = (first(m), last(m)))
            end
            entry = Dict{String,Any}()
            for c in codes
                s = c == "pm" ? o : bystep[c][o["step"]]
                v = common_points(s[key], s[m0key], o[key], o[m0key]; mlo = l, mhi = h)
                ξ = key == "Ex_axis" ? (v.m .- 0.5) .* d.dx : v.m .* d.dx
                lines!(ax, ξ, v.a; label = NAMES[c], color = getfield(COLORS, Symbol(c)), linestyle = STYLES[c])
                entry["x"] = ξ
                entry[c] = v.a
            end
            row == 1 && axislegend(ax; position = :lb)
            side["$(key)_step$(o["step"])"] = entry
        end
    end
    save_plot(out, "wake_axis", fig, side)
end

# The density maps of the codes that have one (a run's map can be missing: the CPU box does not
# return a maps.h5 above its size limit, which the maps written before `write_summary` kept the
# case's stride were), at the case's `map_stride`, on the lattice columns the maps share.
function plot_maps(out, runs, case, d, codes)
    stride = Int.(case["diagnostics"]["map_stride"])
    steps = output_steps(case)[Int.(case["diagnostics"]["map_outputs"])]
    k = last(steps)
    has_map(code) = (p = joinpath(runs, "$code-plasma", "maps.h5"); isfile(p) && h5open(f -> haskey(f, "step$k"), p, "r"))
    read_map(code) = h5open(joinpath(runs, "$code-plasma", "maps.h5"), "r") do f
        g = f["step$k"]
        (rho = read(g["rho"]), m0 = Int(read_attribute(g, "m0")), stride = haskey(HDF5.attributes(g), "stride") ? Int.(read_attribute(g, "stride")) : [1, 1])
    end
    missing_maps = [c for c in codes if !has_map(c)]
    codes = [c for c in codes if has_map(c)]
    isempty(codes) && return nothing
    maps = [read_map(c) for c in codes]
    kept = maps[1].stride                                    # the stride the maps were written with: 1, or the case's
    all(a -> a.stride == kept && a.m0 % kept[1] == 0, maps) && all(stride .% kept .== 0) ||
        error("the maps were written with strides $([a.stride for a in maps]) from m0 = $([a.m0 for a in maps]); the case's stride is $stride")
    step = stride .÷ kept
    lo, hi = maximum(a.m0 ÷ kept[1] for a in maps), minimum(a.m0 ÷ kept[1] + size(a.rho, 1) for a in maps) - 1   # in kept columns
    sel = 1:step[1]:length(lo:hi)
    ys = 1:step[2]:minimum(size(a.rho, 2) for a in maps)
    x, y = collect(lo:hi)[sel] .* (kept[1] * d.dx), (collect(ys) .- 1) .* (kept[2] * d.dy)
    n0 = case["plasma"]["density"]
    fig = Figure(size = (1100, 350 * length(codes)))
    side = Dict{String,Any}("x" => x, "y" => y, "t" => k * d.dt, "codes" => codes, "no_map" => missing_maps)
    for (row, (a, c)) in enumerate(zip(maps, codes))
        M = a.rho[((lo:hi) .- a.m0 ÷ kept[1] .+ 1)[sel], ys]
        ax = Axis(fig[row, 1]; xlabel = "x (c/ω₀), lab frame", ylabel = "y (c/ω₀)", title = "$(NAMES[c]): n_e/n₀ at t = $(round(k * d.dt; digits = 1))")
        hm = heatmap!(ax, x, y, -M ./ n0; colorrange = (0, 3), colormap = :viridis)
        Colorbar(fig[row, 2], hm)
        side[NAMES[c]] = -M ./ n0
    end
    save_plot(out, "density_maps", fig, side)
end

# The spectra at the end in one unit (`spectrum_scale`), not each divided by its own sum. They
# compare bin by bin when the histograms are over the common region (`spectrum_rows`'s `region`);
# otherwise where the regions agree, at the energies only the wake's electrons reach.
function plot_spectra(out, S, d, codes)
    edges = Float64.(S["pm-plasma"]["spectrum_edges"])
    mids = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
    fig = Figure(size = (900, 420))
    ax = Axis(fig[1, 1]; xlabel = "E_kin (mₑc²)", ylabel = "electrons per unit depth and bin (n_c (c/ω₀)²)", yscale = log10,
              title = "electron spectrum at the end (bins of $(round(edges[2] - edges[1]; digits = 3)) mₑc²)")
    last_out(c) = last(S["$c-plasma"]["outputs"])
    side = Dict{String,Any}("ekin" => mids, "unit" => "electrons per unit depth and bin, n_c (c/ω₀)²",
                            "covers" => Dict(c => haskey(last_out(c), "spectrum_region") ? "spectrum_region $(last_out(c)["spectrum_region"])" : SPECTRUM_COVERS[c] for c in codes))
    markers = Dict("pm" => :circle, "smilei" => :utriangle, "picongpu" => :xcross)
    for key in codes
        h = Float64.(last_out(key)["spectrum"]) .* spectrum_scale(key, d)
        isempty(h) && continue
        ok = h .> 0
        any(ok) && scatter!(ax, mids[ok], h[ok]; label = NAMES[key], color = getfield(COLORS, Symbol(key)), marker = markers[key], markersize = 9)
        side[key] = h
    end
    axislegend(ax)
    save_plot(out, "spectrum", fig, side)
end

# The distances of the pairs against time, the numbers of comparison.json's rows: the laser
# (carrier-free amplitude, centroid) in each medium and the wake in plasma (E_x as it is and after
# the best whole-cell shift), at the outputs where the measure exists.
function plot_pairs(out, result, pairs, media)
    colors = Dict("pm-smilei" => "#d62728", "pm-picongpu" => "#2ca02c", "smilei-picongpu" => "#7f7f7f")
    markers = Dict("pm-smilei" => :circle, "pm-picongpu" => :utriangle, "smilei-picongpu" => :rect)
    panels = [(m, key, title, ylabel) for (m, key, title, ylabel) in
              (("vacuum", "amplitude_relrms", "laser amplitude on the axis, vacuum", "relative rms difference"),
               ("vacuum", "centroid_x", "laser centroid on the axis, vacuum", "a − b (c/ω₀)"),
               ("vacuum", "carrier_phase", "carrier phase, vacuum", "a − b (rad)"),
               ("plasma", "centroid_x", "laser centroid on the axis, plasma", "a − b (c/ω₀)"),
               ("plasma", "wake_Ex_relrms", "wake E_x on the axis, plasma", "relative rms difference"),
               ("plasma", "wake_Ex_relrms_shifted", "wake E_x after the best whole-cell shift", "relative rms difference")) if m in media]
    fig = Figure(size = (1300, 400 * cld(length(panels), 3)))
    side = Dict{String,Any}()
    local ax
    for (i, (medium, key, title, ylabel)) in enumerate(panels)
        ax = Axis(fig[(i - 1) ÷ 3 + 1, (i - 1) % 3 + 1]; xlabel = "t (1/ω₀)", ylabel, title)
        for (prefix, a, b) in pairs
            t, v = Float64[], Float64[]
            for r in result[prefix * medium]
                x = get(r, key, nothing)
                x isa AbstractVector && (x = any(isnothing, x) ? nothing : x[1] - x[2])
                x === nothing && continue
                push!(t, r["t"]); push!(v, x)
            end
            name = "$a-$b"
            scatterlines!(ax, t, v; label = "$(NAMES[a]) − $(NAMES[b])", color = colors[name], marker = markers[name])
            side["$(medium)_$(key)_$name"] = Dict("t" => t, "value" => v)
        end
    end
    Legend(fig[cld(length(panels), 3) + 1, 1:3], ax; orientation = :horizontal, framevisible = false)
    save_plot(out, "pair_distances", fig, side)
end

# --- PIConGPU -------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "picongpu.jl"))

# --- command line ---------------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    cmd = get(ARGS, 1, "")
    if cmd == "prepare"
        media = get(ARGS, 4, "") == "vacuum" ? ("vacuum",) : length(ARGS) >= 4 ? error("prepare <case> <tag> [vacuum], got $(ARGS[4])") : ("vacuum", "plasma")
        println(prepare(ARGS[2], ARGS[3]; media))
    elseif cmd == "run-pm"
        be = backend_named(get(ARGS, 3, "cpu"))
        Base.invokelatest(run_pm, ARGS[2]; backend = be)        # CUDACore's methods are newer than this call
    elseif cmd == "summarize-smilei"
        summarize_smilei(ARGS[2])
    elseif cmd == "summarize-picongpu"
        summarize_picongpu(ARGS[2])
    elseif cmd == "compare"
        compare(ARGS[2], ARGS[3])
    else
        error("usage: lwfa.jl prepare <case> <tag> [vacuum] | run-pm <dir> [cpu|cuda] | summarize-smilei <dir> | summarize-picongpu <dir> | compare <runs> <out>")
    end
end
