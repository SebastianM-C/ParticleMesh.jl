# The ionization-injection case of WP-064 and WP-065 in ParticleMesh, Smilei and PIConGPU, from one
# case file (WP-081; `scripts/lwfa/cases/ionization-2d.toml`), and its 3D variant (WP-082).
#
#   julia +1.13 --project=scripts scripts/lwfa/ionization.jl prepare <case.toml> <tag> [codes …]  # scripts/runs/ion-<tag>/
#   julia +1.13 --project=scripts scripts/lwfa/ionization.jl run-pm <dir> [cpu|cuda]             # ParticleMesh into <dir>/summary.json
#   julia +1.13 --project=scripts scripts/lwfa/ionization.jl summarize-smilei <dir>              # Smilei's diagnostics into the same format
#   julia +1.13 --project=scripts scripts/lwfa/ionization.jl summarize-picongpu <dir>            # PIConGPU's dumps into the same format
#   julia +1.13 --project=scripts scripts/lwfa/ionization.jl compare <runs dir> <out dir>        # the comparison, figures and sidecars
#   julia +1.13 --project=scripts scripts/lwfa/ionization.jl table <comparison.json>             # its table for the report (ion_table)
#
# A prepared run has one directory per code, medium and variant: `pm-vacuum`, `pm-plasma`, the
# ParticleMesh runs with the other codes' rate conventions `pm-plasma-rates-smilei` and
# `pm-plasma-rates-picongpu`, and the Monte Carlo codes' plasma runs once per seed
# (`smilei-plasma-s<seed>`, `picongpu-plasma-s<seed>`) next to their vacuum runs.
#
# How the codes are matched (each line a choice; the case file holds the numbers):
#
# - Coordinates. x is the box coordinate of `scripts/optim/born2d_case.jl`: 0 at the back of
#   ParticleMesh's window at t = 0, which is Smilei's xmin. A lattice node sits at x = m·Δx in
#   every code; ParticleMesh's node i (1-based, interior) after s window shifts is m = i − 1 + s,
#   Smilei's primal index i (from 0) is m = i + x_moved/Δx. PIConGPU's are in picongpu.jl's terms
#   with its lab x = 0 on the antenna plane, m_box = m_lab + antenna_cells.
# - The laser is the case's `[laser]` on the antenna plane x_a = antenna_cells·Δx, with the
#   envelope along the curved wavefront (`envelope = "wavefront"`, ParticleMesh's WavefrontTime),
#   the pulse Smilei and PIConGPU launch (scripts/lwfa/lwfa.jl's header, "The laser"). Smilei's
#   laser enters at its xmin, two ghost cells behind x = 0, with the envelope's peak there at
#   t = delay − x_a − 2Δx, so that it crosses the antenna plane at t = delay as ParticleMesh's;
#   the launch factor and the narrower waist are lwfa.jl's (`smilei_launch_factor`, T-710).
# - The plasma: cold electrons on a 2 × 2 lattice (Smilei's `regular`), density 0 up to `vacuum`,
#   a sin² ramp, then the plateau; ions immobile and left out (no code solves Poisson's equation
#   here). The dopant: nitrogen loaded as N⁵⁺, density `dopant.density` times the DopantSection's
#   factor along x, on the 2·half_rows rows around the axis, frozen; its K shell ionizes to N⁷⁺.
# - Ionization. The one physics in which the codes are not the same scheme:
#   * ParticleMesh (`FieldIonization`, WP-059): macro-atoms on a 1 × 1 lattice that release the
#     expected fraction of an electron every step, ADK with FBPIC's l* (from the neutral's first
#     ionization energy, `scripts/ionization/atomic.jl`), no ionization current, no random numbers.
#   * Smilei v5.1 (`ionization_model = "tunnel"`, `Ionization/IonizationTunnel.cpp`): Monte Carlo
#     per macro-ion and step, ADK with l* = n* − 1 (`beta_tunnel` = 2^{2n*}(2l + 1)/(n*Γ(2n*))·I),
#     several levels in one step by the cumulative probability, and the ionization current.
#   * PIConGPU 0.8.0 (`ADKCircPol`, `particles/ionization/byField/ADK/AlgorithmADK.hpp`): Monte
#     Carlo, probability rate·Δt, at most one level per step, the ADK form of Ammosov et al. with
#     Stirling's approximation of the Γ functions, D = (4eZ³/(E n*⁴))^{n*}; the linear-polarization
#     variant `ADKLinPol` multiplies the instantaneous rate by the cycle-average factor
#     √(3n*³E/(πZ³)), which a resolved laser field must not get, so the deck takes `ADKCircPol`,
#     the instantaneous rate. No ionization current (`None`).
#   All three take the same exponent (field 2(2I)^{3/2}, power 2n* − 1) and differ in the
#   prefactor only (`rate_ratio`); `pm-plasma-rates-<code>` runs ParticleMesh with the other
#   code's prefactor, so that a code's distance splits into its rate convention's share and the
#   rest. The Monte Carlo codes run several seeds; their spread is the comparison's noise.
# - Boundaries. Open in y: ParticleMesh's PML (`pml_cells`), Smilei's Silver–Müller, PIConGPU's
#   PML. Along x the window's back: ParticleMesh's PML, Smilei's Silver–Müller, PIConGPU's
#   absorber (picongpu.jl). Particles are removed at every open face.
# - Numerics. Yee, Esirkepov (ParticleMesh's flux deposition), quadratic shapes, Boris, no
#   filter, Float64; ParticleMesh in Reference mode with the plain gather (Smilei and PIConGPU
#   have no compensated gather).
# - What is compared (`compare`): the pulse in vacuum on the axis (peak envelope, its position,
#   the carrier phase), the wake's E_x on the axis behind the pulse, the dopant's charge states
#   in the window over time, the born electrons' weight, the objectives of WP-064 (born energy
#   with the position weight `fade`) and WP-065 (every electron with u_x weighted by `rise`), and
#   the born electrons' kinetic-energy spectrum.

isdefined(@__MODULE__, :smilei_namelist) || include(joinpath(@__DIR__, "lwfa.jl"))     # once, when the tests have it already
include(joinpath(REPO, "scripts", "ionization", "atomic.jl"))
using SpecialFunctions: gamma
using ParticleMeshElectromagnetic: ionization_state, ionization_electrons, ionization_lost, ionization_frame, window_back_band

# --- the case -------------------------------------------------------------------------------------

"""
    ion_derived(case) -> NamedTuple

The numbers every part derives from an ionization case: cell sizes and counts (`nz` and `dz` at
N = 3), the antenna plane `xa`, Δt, the outputs every `every` steps from 0, `every` the nearest
whole number to ceil(t_end/Δt)/outputs, the steps `every·outputs` (the last output, the read-out),
the axis and the dopant's rows.
"""
function ion_derived(case)
    case["schema"] == "lwfa-ionization-1" || error("not an ionization case: schema $(case["schema"])")
    g, t = case["grid"], case["time"]
    cells = Int.(g["cells"])
    N = length(cells)
    N in (2, 3) || error("an ionization case is 2D or 3D, got $(length(cells)) cell counts")
    dx, dy = Float64(g["dx"]), Float64(g["dy"])
    dz = N == 3 ? Float64(get(g, "dz", dy)) : dy
    h = N == 2 ? (dx, dy) : (dx, dy, dz)
    dt = Float64(t["courant"]) / sqrt(sum(1 ./ h .^ 2))
    # outputs every `every` steps from 0, the last one the run's end: Smilei's time selections are periodic
    nout = Int(t["outputs"])
    every = round(Int, ceil(Int, Float64(t["t_end"]) / dt) / nout)
    steps = every * nout
    outs = collect(0:every:steps)
    all(iseven, cells[2:end]) || error("the axis must be a node row: an even number of transverse cells, got $cells")
    na = Int(g["antenna_cells"])
    D = case["dopant"]
    return (; N, h, dx, dy, dz, nx = cells[1], ny = cells[2], nz = N == 3 ? cells[3] : 1, cells = Tuple(cells), na, xa = na * dx,
            dt, steps, every, outputs = outs, yc = cells[2] * dy / 2, zc = N == 3 ? cells[3] * dz / 2 : 0.0, jaxis = cells[2] ÷ 2,
            kaxis = N == 3 ? cells[3] ÷ 2 : 0, rows = Int(D["half_rows"]) * dy, rows_z = N == 3 ? Int(D["half_rows"]) * dz : 0.0)
end

"The dopant's longitudinal factor of the case at box x: `DopantSection`'s, r(x − start)·(1 − r(x − stop)), r the sin² rise over `ramp`."
function dopant_section_factor(case, x)
    s0, s1, r = Float64.(case["dopant"]["section"])
    rise(u) = (v = clamp(u / r, 0.0, 1.0); sin(π * v / 2)^2)
    return rise(x - s0) * (1 - rise(x - s1))
end

# --- rate conventions -----------------------------------------------------------------------------

"""
    rate_ratio(convention, energy_eV, charge) -> Float64

The prefactor of a code's ADK rate over Smilei's for one level, at the same field: all three
codes take W = P·(F/E)^{2n* − 1}·exp(−F/(3E)) with F = 2(2I)^{3/2} (atomic units) and differ in P
(header, "Ionization"). Smilei: P = 2^{2n*}(2l + 1)/(n*Γ(2n*))·I, l = 0 for the K shell;
`"pm"`: ParticleMesh's default, C² with FBPIC's l* (`adk_numbers`); `"picongpu"`: PIConGPU's
E·D²/(8πZ) with D = (4eZ³/(E n*⁴))^{n*}, which is P·F^{2n* − 1} = (4eZ³/n*⁴)^{2n*}/(8πZ).
"""
function rate_ratio(convention, energy_eV, charge; ground_eV)
    I = energy_eV / 27.211386245988                       # Hartree
    Z = charge
    n = Z / sqrt(2I)
    smilei = 2^(2n) / (n * gamma(2n)) * I
    convention == "smilei" && return 1.0
    if convention == "pm"
        a = adk_numbers(energy_eV, charge; wavelength = 0.8e-6, ground_eV)
        return a.C2 * I / smilei
    elseif convention == "picongpu"
        F = 2 * (2I)^1.5
        return (4 * exp(1) * Z^3 / n^4)^(2n) / (8π * Z) / F^(2n - 1) / smilei
    end
    error("rate convention $convention: pm, smilei or picongpu")
end

"""
    case_levels(T, case; rates = "pm") -> Tuple of ADKLevel

ParticleMesh's levels of the case's dopant with the rate convention of `rates` (`"pm"`,
`"smilei"` or `"picongpu"`): ParticleMesh's own levels with the prefactor scaled by
`rate_ratio(rates)/rate_ratio("pm")`, the field and the power unchanged.
"""
function case_levels(::Type{T}, case; rates = "pm") where {T}
    D = case["dopant"]
    el = Symbol(D["element"])
    λ = Float64(case["lambda0_um"]) * 1e-6
    return Tuple(map(Int.(D["levels"])) do z
        l = adk_level(Float64, el, z; wavelength = λ, cutoff = Float64(D["cutoff"]))
        rates == "pm" && return ParticleMeshElectromagnetic.level_as(T, l)
        e, g = ionization_energy(el, z), ionization_energy(el, 1)
        f = rate_ratio(rates, e, z; ground_eV = g) / rate_ratio("pm", e, z; ground_eV = g)
        ADKLevel(T(l.prefactor * f), T(l.field), T(l.power); cutoff = T(l.cutoff))
    end)
end

# --- ParticleMesh ---------------------------------------------------------------------------------

"""
    pm_ion_problem(case; T = Float64, plasma = true, dopant = true, rates = "pm", envelope) -> EMProblem

The case as a ParticleMesh problem in 2D or 3D: born2d_case.jl's `born_problem` written from the
case file (T-1120 holds the two to the same numbers), with the laser's envelope from the case
(`envelope` overrides it) and the rate convention `rates` (`case_levels`). `plasma = false` is
the laser in vacuum; `dopant = false` the plasma without the dopant.
"""
function pm_ion_problem(case; T = Float64, plasma = true, dopant = true, rates = "pm", envelope = laser_envelope(case))
    d = ion_derived(case)
    L, P, D = case["laser"], case["plasma"], case["dopant"]
    N = d.N
    z3 = (zero(T), zero(T), zero(T))
    center = N == 2 ? T(d.yc) : (T(d.yc), T(d.zc))
    pulse = GaussianLaser(; a0 = T(L["a0"]), waist = T(L["waist"]), duration = T(L["duration"]), delay = T(L["delay"]),
                          focus = T(L["focus"]), position = d.na * (T(2) * T(π) / T(round(Int, 2π / d.dx))), center,
                          polarization = T(polarization_angle(case)), envelope)
    ppc = Tuple(Int.(P["particles_per_cell"]))
    length(ppc) == N || error("plasma.particles_per_cell needs $N entries, got $ppc")
    kmode = ntuple(i -> i == 1 ? 1 : 0, N)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(P["density"]), zero(T), kmode, ppc, z3, z3, 1)
    ion = nothing
    if plasma && dopant
        rows = (d.yc - d.rows, d.yc + d.rows)
        s = Float64.(D["section"])
        ion = FieldIonization(; density = T(D["density"]), levels = case_levels(T, case; rates), capacity = Int(get(D, "capacity", 400_000)),
                              every = 1, ppc = Tuple(Int.(D["particles_per_cell"])),
                              rows = N == 2 ? rows : (rows, (d.zc - d.rows_z, d.zc + d.rows_z)),
                              section = DopantSection(; start = T(s[1]), stop = T(s[2]), ramp = T(s[3])))
    end
    h = T.(d.h)
    box = ntuple(i -> d.cells[i] * h[i], N)
    boundaries = N == 2 ? (:window, :open) : (:window, :open, :open)
    return EMProblem(String(case["name"]), box, d.cells, T(d.dt), d.steps, plasma ? (e,) : (), Int(P["seed"]), kmode;
                     lasers = (pulse,), boundaries, window = MovingWindow(; start = T(case["window"]["start"])),
                     profile = DensityRamp(; vacuum = T(P["vacuum"]), ramp = T(P["ramp"])), ionization = ion)
end

"The case's numerics in ParticleMesh: Reference mode, the PMLs of the case, the plain gather."
pm_ion_numerics(case; mode = Reference()) =
    EMNumerics(; mode, absorber = PML(; cells = Int(case["grid"]["pml_cells"])), back_absorber = PML(; cells = Int(case["grid"]["back_pml_cells"])))

"The kinetic-energy histogram of weights `w` at energies `g` over the case's edges, as plain numbers."
function ekin_histogram(g, w, edges)
    hist = zeros(length(edges) - 1)
    for (gi, wi) in zip(g, w)
        b = searchsortedlast(edges, gi)
        1 <= b < length(edges) && (hist[b] += wi)
    end
    return hist
end

ion_spectrum_edges(case) = (s = case["diagnostics"]["spectrum_ekin"]; collect(range(Float64(s[1]), Float64(s[2]); length = Int(s[3]) + 1)))

"""
    electron_measures(x, u, w, case, area) -> Dict

The measures of one electron population, from plain arrays: window x (c/ω₀), momenta u = γv (a
tuple of arrays), weights (n_c per particle), and the cell area (volume at N = 3): the weight,
WP-064's objective Σ w(γ − 1)S(x) (`born_energy`, `fade`), WP-065's Σ w(γ − 1)S(x)R(u_x)
(`rise`), the weight above `u_fast`, the largest u_x, and the spectrum ahead of fade[1].
"""
function electron_measures(x, u, w, case, area)
    D = case["diagnostics"]
    fade, rise = Float64.(D["fade"]), Float64.(D["rise"])
    g = sqrt.(1 .+ sum(a -> a .^ 2, u)) .- 1
    S = smooth_rise.(x, fade[1], fade[2])
    R = smooth_rise.(u[1], rise[1], rise[2])
    ahead = x .>= fade[1]
    return Dict{String,Any}("weight" => area * sum(w; init = 0.0), "J_born" => area * sum(w .* g .* S; init = 0.0),
                            "J_trapped" => area * sum(w .* g .* S .* R; init = 0.0),
                            "fast_weight" => area * sum(w[u[1] .> Float64(D["u_fast"])]; init = 0.0),
                            "ux_max" => isempty(u[1]) ? 0.0 : maximum(u[1]),
                            "spectrum" => area .* ekin_histogram(g[ahead], w[ahead], ion_spectrum_edges(case)))
end

"The C² step of born2d_case.jl (`smooth_rise`): 0 at x ≤ lo, 1 at x ≥ hi."
function smooth_rise(x, lo, hi)
    s = clamp((x - lo) / (hi - lo), zero(x), one(x))
    return s * s * s * (10 - 15 * s + 6 * s * s)
end

# The alive particles of non-zero weight of a species as plain arrays: window x, momenta, weights.
function pm_particles(s, h1)
    f(a) = Float64.(Array(a))
    alive = (Array(s.status) .== ParticleMeshCore.ALIVE) .& (f(s.w) .> 0)
    x = (Array(s.cell[1])[alive] .- 1 .+ f(s.offset[1])[alive]) .* h1
    return x, map(a -> f(a)[alive], s.u), f(s.w)[alive]
end

"""
    pm_ion_output(st, case, k) -> Dict

One output of a ParticleMesh run of an ionization case at step k: the window's shifts, E_x and
E_y on the axis (E_y averaged over the two rows around it, and at N = 3 over the four columns),
the largest |E_y|; with a plasma the electron measures of the plasma's and of the born
electrons (`electron_measures`), the dopant's weight in each charge state, and the born
electrons themselves (window x, u_x, weight) at the last output.
"""
function pm_ion_output(st, case, k)
    d = ion_derived(case)
    s = window_shifts(st, k)
    g, n = guards(st.grid), ncells(st.grid)
    I = (g + 1):(g + n[1])
    J = (g + 1):(g + n[2])
    ja = d.jaxis + 1
    if d.N == 2
        Ex, Ey = Array(st.fields.Ex)[I, J], Array(st.fields.Ey)[I, J]
        ex, ey = Ex[:, ja], (Ey[:, ja - 1] .+ Ey[:, ja]) ./ 2
    else
        K = (g + 1):(g + n[3])
        ka = d.kaxis + 1
        Ex, Ey = Array(st.fields.Ex)[I, J, K], Array(st.fields.Ey)[I, J, K]
        ex, ey = Ex[:, ja, ka], (Ey[:, ja - 1, ka] .+ Ey[:, ja, ka]) ./ 2         # E_y is dual along y, primal along z
    end
    out = Dict{String,Any}("step" => k, "t" => k * d.dt, "shifts" => s, "m0" => s,       # node i ↔ m = i − 1 + m0; E_x index i ↔ m = i − 1 + Ex_m0 at x = (m − ½)Δx
                           "Ex_axis" => ex, "Ex_m0" => s + 1, "Ey_axis" => ey, "Ey_absmax" => maximum(abs, Ey))
    out["Ey_envelope"] = running_envelope(ey, round(Int, π / d.dx))
    isempty(st.species) && return out
    area = prod(d.h)
    x, u, w = pm_particles(st.species[1], d.dx)
    out["plasma"] = electron_measures(x, u, w, case, area)
    if ionization_frame(st) !== nothing
        e = ionization_electrons(st)
        x, u, w = pm_particles(e, d.dx)
        out["born"] = electron_measures(x, u, w, case, area)
        a = ionization_state(st)
        b = a.bound
        wa = a.w .* a.alive
        L = length(b)
        pops = [sum(wa .* b[j]) for j in 1:L]
        push!(pops, sum(wa) - sum(pops))
        out["charge_states"] = area .* pops                     # N⁵⁺, N⁶⁺, N⁷⁺ (n_c·(c/ω₀)^N)
        out["charge_state_names"] = [Int(case["dopant"]["charge"]) + j - 1 for j in 1:(L + 1)]
        out["lost"] = ionization_lost(st)
        k == d.steps && (out["born_particles"] = Dict("x" => x, "ux" => u[1], "w" => w))
    end
    return out
end

"""
Run the ParticleMesh column of the ionization case in `dir` (its `case.toml` and `kind.txt`:
`vacuum`, `plasma`, `plasma-rates-<code>`, `plasma-nodopant`, `plasma-axis`) into `summary.json`,
on `backend`, in Float64 (`float = Float32` for a 3D run on the GPU).
"""
function run_pm_ion(dir; backend = CPU(), float = Float64)
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    kind = strip(read(joinpath(dir, "kind.txt"), String))
    plasma = kind != "vacuum"
    fieldof = match(r"^plasma-rates-(smilei|picongpu)-a0$", kind)
    rates = fieldof !== nothing ? String(fieldof.captures[1]) :
            startswith(kind, "plasma-rates-") ? String(kind[(length("plasma-rates-") + 1):end]) : "pm"
    if fieldof !== nothing                         # that code's rate and its launched peak field (the case's [comparison])
        case["laser"]["a0"] = Float64(case["laser"]["a0"]) * Float64(case["comparison"]["$(rates)_field_ratio"])
    end
    envelope = kind == "plasma-axis" ? AxisTime() : laser_envelope(case)
    prob = pm_ion_problem(case; T = float, plasma, dopant = kind != "plasma-nodopant", rates, envelope)
    st = EMState(prob, pm_ion_numerics(case); record = (), backend)
    p = params(prob)
    initialize!(st, p)
    d = ion_derived(case)
    outs = Any[pm_ion_output(st, case, 0)]
    t0 = time()
    for k in 1:prob.nsteps
        step!(st, p, k)
        if k in d.outputs
            push!(outs, pm_ion_output(st, case, k))
            println("pm $(basename(dir)): step $k of $(prob.nsteps), $(round(time() - t0; digits = 1)) s")
            flush(stdout)
        end
    end
    elapsed = time() - t0
    ionization_frame(st) === nothing || ionization_lost(st) == 0 || error("$(ionization_lost(st)) births found no free slot: raise dopant.capacity")
    hardware = Dict{String,Any}("threads" => Threads.nthreads(), "cpu" => Sys.cpu_info()[1].model, "backend" => backend_label(backend),
                                "float" => string(float))
    backend isa CPU || (hardware["gpu"] = @eval CUDACore.name(CUDACore.device()))
    open(io -> JSON.print(io, merge(Dict{String,Any}("code" => "ParticleMesh", "kind" => kind, "rates" => rates, "outputs" => outs,
                                                     "seconds" => elapsed, "nsteps" => prob.nsteps,
                                                     "spectrum_edges" => ion_spectrum_edges(case)), hardware)),
         joinpath(dir, "summary.json"), "w")
    write_manifest(dir; root = REPO, extra = merge(hardware, Dict{String,Any}("run_seconds" => elapsed, "julia" => string(VERSION))))
    return dir
end

# --- Smilei ---------------------------------------------------------------------------------------

"""
    smilei_ion_laser(case) -> NamedTuple

The parameters of Smilei's `LaserGaussian2D` or `LaserGaussian3D` and its `tgaussian` that put
the case's pulse on the antenna plane x_a at t = delay (header, "The laser"; lwfa.jl's
`smilei_laser` with the box's coordinates): the laser enters `SMILEI_GHOST` cells behind xmin,
so its peak is on that node at t = delay − x_a − back, the focus is counted from that node, and
`center` gives the carrier the case's Gouy phase at it (halved in 2D). `a0` is divided by the
boundary condition's launch factor and carries the narrower waist √(w₀² − 1)'s factor (w₀/w in
2D, (w₀/w)² in 3D: the transverse spectrum's 1 − k⊥²/4 widens it back to w₀ along each axis).
"""
function smilei_ion_laser(case)
    d, L = ion_derived(case), case["laser"]
    back = SMILEI_GHOST * d.dx
    focus = Float64(L["focus"]) + back
    w0 = Float64(L["waist"])
    w0 > 1 || error("the waist must exceed c/ω₀ for the correction of Smilei's boundary condition")
    waist = sqrt(w0^2 - 1)
    zR = waist^2 / 2
    gouy = d.N == 2 ? atan(focus / zR) / 2 : atan(focus / zR)
    c0 = π / 2 - gouy
    delay = Float64(L["delay"]) - d.xa
    center = c0 + 2π * ceil((delay - c0) / (2π))
    return (; a0 = Float64(L["a0"]) / smilei_launch_factor(d) * (w0 / waist)^(d.N - 1), waist, focus, center, duration = 2 * center,
              fwhm = 2 * Float64(L["duration"]) * sqrt(log(2)), phase_offset = center - (delay - back), back)
end

"""
    smilei_bins(case) -> NamedTuple

The bins of Smilei's histograms of the electrons: window x from 0 to the window's length in
cells of Δx, and u_x from −`ux_lo` to `ux_hi` in `ux_bins`; the objectives are sums over them
(`smilei_objectives`). The spectrum is the case's.
"""
smilei_bins(case) = (d = ion_derived(case); (; xlo = 0.0, xhi = d.nx * d.dx, nxb = d.nx, ux_lo = -10.0, ux_hi = 70.0, ux_bins = 320))

"""
    smilei_ion_namelist(case; plasma = true, seed = 1, dopant = true) -> String

The Smilei v5.1 namelist of the ionization case (2D or 3D): the plasma's electrons, the born
electrons `born` and the nitrogen `nitrogen` (loaded as N⁵⁺ on `mc_particles_per_cell`, frozen,
`ionization_model = "tunnel"`); fields on the axis rows every output; per output the histograms
of `born` and `electron` over (window x, u_x) of the weight and of weight·(γ − 1), the born
electrons' spectrum ahead of fade[1], and the nitrogen's weight per charge state.
"""
function smilei_ion_namelist(case; plasma = true, seed = 1, dopant = true)
    d = ion_derived(case)
    P, D, G = case["plasma"], case["dopant"], case["diagnostics"]
    las = smilei_ion_laser(case)
    patches = Int.(case["grid"]["smilei_patches"])
    every = "[0, $(d.steps), $(d.every)]"
    b = smilei_bins(case)
    geo = d.N == 2 ? "2Dcartesian" : "3Dcartesian"
    open_bc = d.N == 2 ? "[[\"silver-muller\"], [\"silver-muller\"]]" : "[[\"silver-muller\"], [\"silver-muller\"], [\"silver-muller\"]]"
    part_bc = d.N == 2 ? "[[\"remove\", \"remove\"], [\"remove\", \"remove\"]]" : "[[\"remove\", \"remove\"], [\"remove\", \"remove\"], [\"remove\", \"remove\"]]"
    coords = d.N == 2 ? "x, y" : "x, y, z"
    io = IOBuffer()
    println(io, """
    # Smilei namelist generated by scripts/lwfa/ionization.jl from case "$(case["name"])" ($(plasma ? "plasma, seed $seed" : "vacuum")$(dopant ? "" : ", no dopant")).
    # Do not edit; regenerate from the case file. Units: Smilei normalized, reference frequency = ω₀ (A1).
    import math
    import numpy as np

    Main(
        geometry = "$geo",
        interpolation_order = 2,
        cell_length = [$(join(pyf.(d.h), ", "))],
        number_of_cells = [$(join(d.cells, ", "))],
        timestep = $(pyf(d.dt)),
        simulation_time = $(pyf((d.steps + 0.5) * d.dt)),
        number_of_patches = [$(join(patches, ", "))],
        EM_boundary_conditions = $open_bc,
        solve_poisson = False,
        reference_angular_frequency_SI = $(pyf(2π * 299792458.0 / (Float64(case["lambda0_um"]) * 1e-6))),
        random_seed = $seed,
        print_every = 200,
    )

    MovingWindow(time_start = $(pyf(case["window"]["start"])), velocity_x = 1.0)
    """)
    if d.N == 2
        println(io, """
        LaserGaussian2D(
            box_side = "xmin",
            a0 = $(pyf(las.a0)),
            omega = 1.0,
            focus = [$(pyf(las.focus)), $(pyf(d.yc))],
            waist = $(pyf(las.waist)),
            polarization_phi = $(pyf(polarization_angle(case))),
            time_envelope = tgaussian(start = 0.0, duration = $(pyf(las.duration)), fwhm = $(pyf(las.fwhm)), center = $(pyf(las.center))),
            phase_offset = $(pyf(las.phase_offset)),
        )""")
    else
        println(io, """
        LaserGaussian3D(
            box_side = "xmin",
            a0 = $(pyf(las.a0)),
            omega = 1.0,
            focus = [$(pyf(las.focus)), $(pyf(d.yc)), $(pyf(d.zc))],
            waist = $(pyf(las.waist)),
            polarization_phi = $(pyf(polarization_angle(case))),
            time_envelope = tgaussian(start = 0.0, duration = $(pyf(las.duration)), fwhm = $(pyf(las.fwhm)), center = $(pyf(las.center))),
            phase_offset = $(pyf(las.phase_offset)),
        )""")
    end
    println(io, "\nDiagScalar(every = $every, precision = 12)\n")
    # the fields on the axis: two rows (2D) or a 3 × 3 pencil (3D) of the whole grid around it
    sub = d.N == 2 ? "subgrid = np.s_[:, $(d.jaxis - 1):$(d.jaxis + 3)]," :
          "subgrid = np.s_[:, $(d.jaxis - 1):$(d.jaxis + 3), $(d.kaxis - 1):$(d.kaxis + 3)],"
    println(io, """
    DiagFields(
        every = $every,
        fields = ["Ex", "Ey"],
        $sub
        datatype = "double",
    )
    """)
    plasma || return String(take!(io))
    ppc = Int.(P["particles_per_cell"])
    mc = Int.(D["mc_particles_per_cell"])
    s0, s1, sr = Float64.(D["section"])
    trans = d.N == 2 ? "(np.abs(y - $(pyf(d.yc))) < $(pyf(d.rows)))" :
            "(np.abs(y - $(pyf(d.yc))) < $(pyf(d.rows))) * (np.abs(z - $(pyf(d.zc))) < $(pyf(d.rows_z)))"
    println(io, """
    def plasma_density($coords):
        s = np.clip((x - $(pyf(P["vacuum"]))) / $(pyf(P["ramp"])), 0.0, 1.0)
        return $(pyf(P["density"])) * np.sin(0.5 * math.pi * s)**2

    def rise(s):
        return np.sin(0.5 * math.pi * np.clip(s / $(pyf(sr)), 0.0, 1.0))**2

    # The dopant's DopantSection along x on the rows around the axis. The rows are those whose centre
    # lies inside the band; Smilei evaluates the profile at each particle's place, which puts a
    # particle near a band edge on the side of its own place (the band's edges are on cell faces).
    def dopant_density($coords):
        return $(pyf(D["density"])) * rise(x - $(pyf(s0))) * (1.0 - rise(x - $(pyf(s1)))) * $trans

    Species(
        name = "electron",
        position_initialization = "regular",
        regular_number = [$(join(ppc, ", "))],
        momentum_initialization = "cold",
        particles_per_cell = $(prod(ppc)),
        mass = 1.0, charge = -1.0,
        number_density = plasma_density,
        boundary_conditions = $part_bc,
    )
    """)
    if dopant
        println(io, """
        Species(
            name = "born",
            position_initialization = "regular",
            momentum_initialization = "cold",
            particles_per_cell = 0,
            mass = 1.0, charge = -1.0,
            number_density = 0.0,
            boundary_conditions = $part_bc,
        )

        Species(
            name = "nitrogen",
            position_initialization = "regular",
            regular_number = [$(join(mc, ", "))],
            momentum_initialization = "cold",
            particles_per_cell = $(prod(mc)),
            atomic_number = 7,
            charge = $(pyf(D["charge"])),
            mass = $(pyf(14 * 1836.15267343)),
            number_density = dopant_density,
            time_frozen = 1.0e9,
            ionization_model = "tunnel",
            ionization_electrons = "born",
            boundary_conditions = $part_bc,
        )

        # 0: the nitrogen's weight per charge state
        DiagParticleBinning(deposited_quantity = "weight", every = $every, species = ["nitrogen"],
                            axes = [["charge", $(pyf(Int(D["charge"]) - 0.5)), $(pyf(Int(D["levels"][end]) + 0.5)), $(length(D["levels"]) + 1)]])
        """)
    end
    s = G["spectrum_ekin"]
    for sp in (dopant ? ("born", "electron") : ("electron",))
        base = smilei_binning_index(sp, dopant)
        println(io, """
        # $base, $(base + 1): $sp over (window x, u_x), the weight and weight·(γ − 1); $(base + 2): its spectrum ahead of fade[1]
        DiagParticleBinning(deposited_quantity = "weight", every = $every, species = ["$sp"],
                            axes = [["moving_x", $(pyf(b.xlo)), $(pyf(b.xhi)), $(b.nxb)], ["px", $(pyf(b.ux_lo)), $(pyf(b.ux_hi)), $(b.ux_bins)]])
        DiagParticleBinning(deposited_quantity = "weight_ekin", every = $every, species = ["$sp"],
                            axes = [["moving_x", $(pyf(b.xlo)), $(pyf(b.xhi)), $(b.nxb)], ["px", $(pyf(b.ux_lo)), $(pyf(b.ux_hi)), $(b.ux_bins)]])
        DiagParticleBinning(deposited_quantity = "weight", every = $every, species = ["$sp"],
                            axes = [["moving_x", $(pyf(Float64(G["fade"][1]))), $(pyf(b.xhi)), 1], ["ekin", $(pyf(Float64(s[1]))), $(pyf(Float64(s[2]))), $(Int(s[3]))]])
        """)
    end
    dopant && println(io, "DiagTrackParticles(species = \"born\", every = [$(d.steps), $(d.steps), 1], attributes = [\"x\", \"px\", \"py\", \"pz\", \"w\"])")
    return String(take!(io))
end

"The number of the first of a species' three `DiagParticleBinning` in the namelist: the nitrogen's charge states are 0 when there is a dopant."
smilei_binning_index(sp, dopant) = dopant ? (sp == "born" ? 1 : 4) : 0

const SMILEI_ION_RUN_SH = """
#!/usr/bin/env bash
# Generated by scripts/lwfa/ionization.jl. Runs Smilei in the run directory (inside the Smilei
# container, or with a native build: SMILEI points at the binary); OMP_NUM_THREADS from the caller.
set -euo pipefail
cd "\${PM_RUN_DIR:-\$(dirname "\$0")}"
export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-16}
export OMP_SCHEDULE=dynamic
SMILEI=\${SMILEI:-smilei}
{
  echo "# smilei binary: \$(command -v \$SMILEI)"
  echo "# threads: \$OMP_NUM_THREADS, cpus: \$(nproc)"
  echo "# cpu model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
} > container.txt
echo "# start: \$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> container.txt
\$SMILEI namelist.py > smilei.log 2>&1
echo "# end: \$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> container.txt
"""

# A (window x, u_x) histogram of Smilei as (x bin, u_x bin): HDF5.jl reads its C-order [x][u_x] as (u_x, x).
function xu_matrix(a, b)
    size(a) == (b.ux_bins, b.nxb) && return permutedims(a)
    size(a) == (b.nxb, b.ux_bins) && return a
    error("a (moving_x, px) histogram of size $(size(a)), expected $(b.nxb) × $(b.ux_bins)")
end

"""
    smilei_objectives(W, WE, case) -> Dict

The electron measures of `electron_measures` from Smilei's histograms over (window x, u_x) of the
weight `W` and of weight·(γ − 1) `WE`, the position and trapping weights taken at the bins'
centres (bins of Δx in x and of 0.25 in u_x, against weights that change over 12 c/ω₀ and 2 in
u_x). Smilei's weight is n·(cell volume)/ppc (`ParticleCreator::createWeight`), so a sum of
weights is ParticleMesh's Δx·Δy·Σw.
"""
function smilei_objectives(W, WE, case; shift = 0.0)
    b, G = smilei_bins(case), case["diagnostics"]
    fade, rise = Float64.(G["fade"]), Float64.(G["rise"])
    xc = range(b.xlo, b.xhi; length = b.nxb + 1)
    xc = (xc[1:(end - 1)] .+ xc[2:end]) ./ 2 .+ shift
    uc = range(b.ux_lo, b.ux_hi; length = b.ux_bins + 1)
    uc = (uc[1:(end - 1)] .+ uc[2:end]) ./ 2
    S = smooth_rise.(xc, fade[1], fade[2])
    R = smooth_rise.(uc, rise[1], rise[2])
    occupied = findall(>(0), vec(sum(W; dims = 1)))
    return Dict{String,Any}("weight" => sum(W), "J_born" => sum(WE .* S), "J_trapped" => sum(WE .* (S * R')),
                            "fast_weight" => sum(W[:, uc .> Float64(G["u_fast"])]),
                            "ux_max" => isempty(occupied) ? 0.0 : uc[last(occupied)] + step(range(b.ux_lo, b.ux_hi; length = b.ux_bins + 1)) / 2)
end

"ParticleMesh's window shifts after `k` steps of the case (`window_shifts`): max(0, ⌊(kΔt − start)/Δx⌋)."
pm_window_shifts(case, k) = (d = ion_derived(case); max(0, floor(Int, (k * d.dt - Float64(case["window"]["start"])) / d.dx)))

"Smilei's tracked particles of `species` at its last dump in `dir`: x less `x_back` (ParticleMesh's window back), u_x and the weight."
function smilei_tracked(dir, species, x_back)
    path = joinpath(dir, "TrackParticlesDisordered_$species.h5")
    isfile(path) || return nothing
    h5open(path, "r") do f
        data = f["data"]
        key = last(sort(collect(keys(data)); by = k -> parse(Int, k)))       # zero-padded step numbers
        it = parse(Int, key)
        p = data[key]["particles"][species]
        x = Float64.(read(p["position"]["x"]))
        return Dict("x" => x .- x_back, "ux" => Float64.(read(p["momentum"]["x"])), "w" => Float64.(read(p["weight"])), "step" => it)
    end
end

"Read Smilei's diagnostics of the ionization case in `dir` into the summary format of `pm_ion_output`."
function summarize_smilei_ion(dir)
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    d = ion_derived(case)
    kind = strip(read(joinpath(dir, "kind.txt"), String))
    plasma = kind != "vacuum"
    dopant = plasma && kind != "plasma-nodopant"
    b = smilei_bins(case)
    pb(i) = smilei_binning(joinpath(dir, "ParticleBinning$i.h5"))
    bins = plasma ? Dict(i => pb(i) for i in 0:(dopant ? 6 : 2)) : Dict()
    outs = Any[]
    last_moved, last_pm = 0.0, 0
    h5open(joinpath(dir, "Fields0.h5"), "r") do f
        data = f["data"]
        for key in sort(collect(keys(data)); by = k -> parse(Int, k))
            it = parse(Int, key)
            grp = data[key]
            xm = Float64(read_attribute(grp, "x_moved"))
            last_moved = xm
            mx = round(Int, xm / d.dx)
            abs(mx * d.dx - xm) <= 1e-6 * d.dx || error("x_moved = $xm is not a whole number of cells")
            # Smilei moves its window by whole patches, ParticleMesh by cells: a particle at Smilei's
            # window x sits at x + (mx − s)Δx in ParticleMesh's window after s(k) shifts
            spm = pm_window_shifts(case, it)
            last_pm = spm
            Ex, Ey = read(grp["Ex"]), read(grp["Ey"])               # C order [x][y]([z]): reversed axes in HDF5.jl
            Ex, Ey = permutedims(Ex, ndims(Ex):-1:1), permutedims(Ey, ndims(Ey):-1:1)
            # the subgrid starts at index jaxis − 1 (and kaxis − 1): the axis row is its 2nd, E_y's dual rows
            # around it its 2nd and 3rd
            ex = d.N == 2 ? Ex[:, 2] : Ex[:, 2, 2]
            ey = d.N == 2 ? (Ey[:, 2] .+ Ey[:, 3]) ./ 2 : (Ey[:, 2, 2] .+ Ey[:, 3, 2]) ./ 2
            out = Dict{String,Any}("step" => it, "t" => it * d.dt, "x_moved" => xm, "m0" => mx,
                                   "Ex_axis" => ex, "Ex_m0" => mx,            # dual index i (from 0) ↔ m = i + mx, at x = (m − ½)Δx
                                   "Ey_axis" => ey, "Ey_absmax" => maximum(abs, Ey))
            out["Ey_envelope"] = running_envelope(ey, round(Int, π / d.dx))
            if plasma
                for (name, sp) in (("plasma", "electron"), ("born", "born"))
                    (sp == "born" && !dopant) && continue
                    i = smilei_binning_index(sp, dopant)
                    W, WE = xu_matrix(bins[i][it], b), xu_matrix(bins[i + 1][it], b)
                    m = smilei_objectives(W, WE, case; shift = (mx - spm) * d.dx)
                    m["spectrum"] = vec(bins[i + 2][it])
                    out[name] = m
                end
                if dopant
                    out["charge_states"] = vec(bins[0][it])
                    out["charge_state_names"] = [Int(case["dopant"]["charge"]) + j - 1 for j in 1:(length(case["dopant"]["levels"]) + 1)]
                end
            end
            push!(outs, out)
        end
    end
    if dopant
        tr = smilei_tracked(dir, "born", last_pm * ion_derived(case).dx)
        tr === nothing || (outs[end]["born_particles"] = Dict(k => tr[k] for k in ("x", "ux", "w")))
    end
    secs = try
        lines = readlines(joinpath(dir, "container.txt"))
        stamp(p) = DateTime(first(filter(l -> startswith(l, p), lines))[length(p) + 1:end][1:23], "yyyy-mm-ddTHH:MM:SS.sss")
        Dates.value(stamp("# end: ") - stamp("# start: ")) / 1000
    catch
        nothing
    end
    seed = tryparse(Int, something(match(r"-s(\d+)$", basename(dir)), (captures = ["1"],)).captures[1])
    open(io -> JSON.print(io, Dict{String,Any}("code" => "Smilei", "kind" => kind, "rates" => "smilei", "seed" => seed, "outputs" => outs,
                                               "seconds" => secs, "nsteps" => d.steps, "spectrum_edges" => ion_spectrum_edges(case))),
         joinpath(dir, "summary.json"), "w")
    return dir
end

# --- PIConGPU -------------------------------------------------------------------------------------

"""
    picongpu_ion_lab(case) -> Dict

The ionization case as a case of lwfa.jl's PIConGPU deck (scripts/lwfa/picongpu.jl), whose x is
the lab coordinate with 0 on the antenna plane: lab x = box x − x_a. The laser keeps its delay
(the envelope's peak on the antenna plane at t = delay) and has its focus, the plasma its vacuum,
at box x − x_a. The window's back is at lab c(t − start) in that deck; at `start` + x_a here, so
that it is at box c(t − start), ParticleMesh's. PIConGPU delays the window's motion by at most
its own length (`--windowMovePoint` in [0, 1]), so the lab window is lengthened at the front to
the cells the light front crosses before the window moves; the plasma ahead of ParticleMesh's
window front is reached by nothing that is compared. A 3D case gives its (x, y) plane, and
`picongpu_ion_3d!` adds the third axis to the deck.
"""
function picongpu_ion_lab(case)
    d = ion_derived(case)
    g, L, P = case["grid"], case["laser"], case["plasma"]
    start = Float64(case["window"]["start"]) + d.xa
    j0 = HUYGENS + 1
    lag = round(Int, start / d.dx)
    laser = Dict{String,Any}(k => v for (k, v) in L)
    laser["focus"] = Float64(L["focus"]) - d.xa
    return Dict{String,Any}(
        "schema" => "lwfa-proposal-1", "name" => case["name"], "lambda0_um" => case["lambda0_um"],
        "grid" => Dict{String,Any}("dx" => d.dx, "dy" => d.dy, "cells" => [max(d.nx - d.na, lag - 2j0), d.ny], "antenna_cells" => d.na,
                                   "transverse" => g["transverse"], "picongpu_devices" => get(g, "picongpu_devices", PIC_DEVICES_DEFAULT)),
        "time" => Dict{String,Any}("courant" => case["time"]["courant"], "steps" => d.steps, "fields_every" => d.every),
        "laser" => laser, "window" => Dict{String,Any}("start" => start),
        "plasma" => Dict{String,Any}("density" => P["density"], "vacuum" => Float64(P["vacuum"]) - d.xa, "ramp" => P["ramp"],
                                     "particles_per_cell" => P["particles_per_cell"][1:2], "seed" => P["seed"]),
        "diagnostics" => Dict{String,Any}("spectrum_ekin" => case["diagnostics"]["spectrum_ekin"], "map_stride" => [1, 1], "map_outputs" => Int[]))
end

# the PIConGPU names of the born electrons and of the dopant
const PIC_BORN, PIC_DOPANT = "eb", "N"

"""
    picongpu_ion_param_files(case; kind, seed) -> Dict{String,String}

picongpu.jl's .param files of `picongpu_ion_lab(case)` with the species of the ionization case
in place of its electrons: the plasma's electrons `e`, the born electrons `eb`, and the dopant
`N`, loaded as N⁵⁺ (`ChargeState`, `boundElectrons` = 2) on a 16 × 16 quiet lattice, static (the
probe pusher without a push) and without current, ionized by `ADKCircPol` into `eb` without an
ionization current. The seed of the random numbers is the run's.
"""
function picongpu_ion_param_files(case; kind, seed)
    lab = picongpu_ion_lab(case)
    d0 = ion_derived(case)
    plasma, dopant = kind != "vacuum", kind ∉ ("vacuum", "plasma-nodopant")
    f = picongpu_param_files(lab; plasma)
    plasma || return d0.N == 3 ? picongpu_ion_3d!(f, case) : f
    d, D = ion_derived(case), case["dopant"]
    D["element"] == "N" && Int(D["charge"]) == 5 && Int.(D["levels"]) == [6, 7] || error("the PIConGPU deck loads N⁵⁺ and ionizes the K shell")
    p = picongpu_parameters(lab; plasma)
    cf, H = PIConGPUDeck.cfloat, PIConGPUDeck.HEADER * "/* ionization case \"$(case["name"])\" ($kind), scripts/lwfa/ionization.jl */\n"
    e, eb, N = PIC_SPECIES, PIC_BORN, PIC_DOPANT
    ions = dopant ? ", PIC_$eb, PIC_$N" : ""
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
        using IonAttributes = MakeSeq_t<DefaultParticleAttributes, boundElectrons>;
        /* the plasma's electrons and the born electrons: the same particles in two species */
        value_identifier(float_X, MassRatio_$e, 1.0);
        value_identifier(float_X, ChargeRatio_$e, 1.0);
        using ParticleFlags_$e = MakeSeq_t<
            particlePusher<UsedParticlePusher>,
            shape<UsedParticleShape>,
            interpolation<UsedField2Particle>,
            current<UsedParticleCurrentSolver>,
            massRatio<MassRatio_$e>,
            chargeRatio<ChargeRatio_$e>>;
        using PIC_$e = Particles<PMACC_CSTRING("$e"), ParticleFlags_$e, DefaultParticleAttributes>;
        using PIC_$eb = Particles<PMACC_CSTRING("$eb"), ParticleFlags_$e, DefaultParticleAttributes>;
        /* the dopant: static (the probe pusher, no push), no current; the mass and charge of N-14 */
        value_identifier(float_X, MassRatio_$N, $(cf(14 * 1836.15267343)));
        value_identifier(float_X, ChargeRatio_$N, -7.0);
        using ParticleFlags_$N = MakeSeq_t<
            particlePusher<particles::pusher::Probe>,
            shape<UsedParticleShape>,
            interpolation<UsedField2Particle>,
            massRatio<MassRatio_$N>,
            chargeRatio<ChargeRatio_$N>,
            atomicNumbers<ionization::atomicNumbers::Nitrogen_t>,
            ionizationEnergies<ionization::energies::AU::Nitrogen_t>,
            ionizers<MakeSeq_t<particles::ionization::ADKCircPol<PIC_$eb, particles::ionization::current::None>>>>;
        using PIC_$N = Particles<PMACC_CSTRING("$N"), ParticleFlags_$N, IonAttributes>;
        using VectorAllSpecies = MakeSeq_t<PIC_$e$ions>;
    } // namespace picongpu
    """
    s0, s1, sr = Float64.(D["section"]) .- (d.xa, d.xa, 0.0)
    f["density.param"] = H * """
    #pragma once
    #include "picongpu/particles/densityProfiles/profiles.def"
    namespace picongpu
    {
        namespace densityProfiles
        {
            /* lengths in c/ω₀, densities in n_c, at the cell centre: lab x = (j + ½ − J0) DX along y, and
             * (i + ½ − NY/2) cells from the axis along x */
            constexpr float_64 DX = $(cf(p.dx));
            constexpr float_64 J0 = $(cf(p.layout.j0));
            /* the plasma: DENSITY sin²(π/2 clamp((x − VACUUM)/RAMP, 0, 1)) */
            constexpr float_64 DENSITY = $(cf(p.density));
            constexpr float_64 VACUUM = $(cf(p.vacuum));
            constexpr float_64 RAMP = $(cf(p.ramp));
            struct ProfileFunctor_$e
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
            using Profile_$e = FreeFormulaImpl<ProfileFunctor_$e>;
            /* the dopant: DOPANT r(x − S0)(1 − r(x − S1)), r(u) = sin²(π/2 clamp(u/SR, 0, 1)), on the rows whose
             * centre is less than HALF_ROWS cells from the axis */
            constexpr float_64 DOPANT = $(cf(Float64(D["density"])));
            constexpr float_64 S0 = $(cf(s0));
            constexpr float_64 S1 = $(cf(s1));
            constexpr float_64 SR = $(cf(sr));
            constexpr float_64 AXIS = $(cf(d.ny / 2));
            constexpr float_64 HALF_ROWS = $(cf(Float64(D["half_rows"])));
            struct ProfileFunctor_$N
            {
                HDINLINE float_64 rise(float_64 u) const
                {
                    float_64 v = u / SR;
                    v = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
                    const float_64 r = math::sin(0.5 * PI * v);
                    return r * r;
                }
                HDINLINE float_X operator()(const floatD_64& position_SI, const float3_64& cellSize_SI)
                {
                    const float_64 x = (position_SI.y() / cellSize_SI.y() - J0) * DX;
                    const float_64 r = position_SI.x() / cellSize_SI.x() - AXIS;
                    if(r <= -HALF_ROWS || r >= HALF_ROWS)
                        return 0.0_X;
                    return static_cast<float_X>(DOPANT * rise(x - S0) * (1.0 - rise(x - S1)));
                }
            };
            using Profile_$N = FreeFormulaImpl<ProfileFunctor_$N>;
        } // namespace densityProfiles
    } // namespace picongpu
    """
    mc = Int.(D["mc_particles_per_cell"])
    f["particle.param"] = H * """
    #pragma once
    #include "picongpu/particles/filter/filter.def"
    #include "picongpu/particles/manipulators/manipulators.def"
    #include "picongpu/particles/startPosition/functors.def"
    namespace picongpu
    {
        namespace particles
        {
            /* the ramp's first cells and the dopant carry weightings far below PIConGPU's default floor of
             * 10, which QuietImpl would drop; only empty cells are dropped */
            constexpr float_X MIN_WEIGHTING = 1.0e-20;
            namespace startPosition
            {
                struct QuietParam_$e
                {
                    using numParticlesPerDimension = mCT::shrinkTo<mCT::Int<$(p.lattice[1]), $(p.lattice[2]), 1>, simDim>::type;
                };
                using Quiet_$e = QuietImpl<QuietParam_$e>;
                struct QuietParam_$N
                {
                    using numParticlesPerDimension = mCT::shrinkTo<mCT::Int<$(mc[2]), $(mc[1]), 1>, simDim>::type;
                };
                using Quiet_$N = QuietImpl<QuietParam_$N>;
            } // namespace startPosition
        } // namespace particles
    } // namespace picongpu
    """
    load = "CreateDensity<densityProfiles::Profile_$e, startPosition::Quiet_$e, PIC_$e>"
    dopant && (load *= ",\n            CreateDensity<densityProfiles::Profile_$N, startPosition::Quiet_$N, PIC_$N>,\n            Manipulate<manipulators::unary::ChargeState<$(Int(D["charge"]))u>, PIC_$N>")
    f["speciesInitialization.param"] = H * """
    #pragma once
    #include "picongpu/particles/InitFunctors.hpp"
    namespace picongpu
    {
        namespace particles
        {
            /* cold electrons$(dopant ? "; the dopant as N⁵⁺ (the pipeline runs again on the local domain a slide resets, so the charge state is set on new ions only)" : "") */
            using InitPipeline = pmacc::mp_list<
                $load>;
        } // namespace particles
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
            /* the run's seed */
            using SeedGenerator = seed::Value<$(Int(seed))>;
        } // namespace random
    } // namespace picongpu
    """
    return d.N == 3 ? picongpu_ion_3d!(f, case) : f
end

"""
    picongpu_ion_3d!(files, case) -> files

The 2D deck of a 3D case's (x, y) plane made 3D: `SIMDIM` DIM3, the 3D time step (the 2D deck's
Δt has no 1/Δz² term) and the cell depth Δz, supercells of 8 × 8 × 4, the quiet lattices with the
case's third count (the plasma's `particles_per_cell[3]`, the dopant's `mc_particles_per_cell[3]`),
and the dopant's rows bounded along z as along x, PIConGPU's z being the case's z. Every edit
must find its line.
"""
function picongpu_ion_3d!(f, case)
    d, D, P = ion_derived(case), case["dopant"], case["plasma"]
    u = reference_units(Float64(case["lambda0_um"]) * 1e-6)
    cf = PIConGPUDeck.cfloat
    function edit!(name, pat, new)
        haskey(f, name) || return
        occursin(pat, f[name]) || error("$name: no line matching $pat to make the deck 3D")
        f[name] = replace(f[name], pat => new)
    end
    edit!("dimension.param", "#define SIMDIM DIM2", "#define SIMDIM DIM3")
    edit!("simulation.param", r"DELTA_T_SI = [^;]*;", "DELTA_T_SI = $(cf(d.dt * u.time));")
    edit!("simulation.param", r"CELL_DEPTH_SI = [^;]*;", "CELL_DEPTH_SI = $(cf(d.dz * u.length));")
    edit!("memory.param", "mCT::Int<$PIC_SUPERCELL, $PIC_SUPERCELL, 1>", "mCT::Int<$PIC_SUPERCELL, $PIC_SUPERCELL, 4>")
    ppc, mc = Int.(P["particles_per_cell"]), Int.(D["mc_particles_per_cell"])
    edit!("simulation.param", r"TYPICAL_PARTICLES_PER_CELL = \d+u;", "TYPICAL_PARTICLES_PER_CELL = $(prod(ppc))u;")
    edit!("particle.param", "mCT::Int<$(ppc[2]), $(ppc[1]), 1>", "mCT::Int<$(ppc[2]), $(ppc[1]), $(ppc[3])>")
    occursin("ProfileFunctor_$PIC_DOPANT", get(f, "density.param", "")) || return f      # no dopant: the rest is the dopant's
    edit!("particle.param", "mCT::Int<$(mc[2]), $(mc[1]), 1>", "mCT::Int<$(mc[2]), $(mc[1]), $(mc[3])>")
    edit!("density.param", r"\n( *)constexpr float_64 HALF_ROWS",
          SubstitutionString("\n\\1constexpr float_64 AXIS_Z = $(cf(d.nz / 2));\n\\1constexpr float_64 HALF_ROWS"))
    edit!("density.param", r"\n( *)if\(r <= -HALF_ROWS \|\| r >= HALF_ROWS\)",
          SubstitutionString("\n\\1const float_64 rz = position_SI.z() / cellSize_SI.z() - AXIS_Z;\n\\1if(r <= -HALF_ROWS || r >= HALF_ROWS || rz <= -HALF_ROWS || rz >= HALF_ROWS)"))
    return f
end

"The command line of a 3D case: one device along z, the cells along z, open along z (`picongpu_arguments` writes the 2D one)."
function picongpu_ion_arguments(case, p)
    a = picongpu_arguments(p)
    ion_derived(case).N == 2 && return a
    lay = p.layout
    for (old, new) in ("-d 1 $(lay.devices) " => "-d 1 $(lay.devices) 1 ", "-g $(p.cells[1]) $(p.cells[2]) " => "-g $(p.cells[1]) $(p.cells[2]) $(ion_derived(case).nz) ",
                       "--periodic 0 0 " => "--periodic 0 0 0 ",
                       # one file per output, so a rank killed at shutdown costs at most the last one (a 3D run hung in the
                       # close of its single file once, which left all of that file's outputs unreadable)
                       "--openPMD.infix NULL " => "--openPMD.infix _%06T ")
        occursin(old, a) || error("PIConGPU's arguments: no \"$old\" to make 3D")
        a = replace(a, old => new)
    end
    return a
end

"Write a PIConGPU directory of the ionization case into `dir` (params/, run.sh); the input files for the manifest."
function prepare_picongpu_ion(dir, case; kind, seed, tag)
    lab = picongpu_ion_lab(case)
    p = picongpu_parameters(lab; plasma = kind != "vacuum")
    files = picongpu_ion_param_files(case; kind, seed)
    mkpath(joinpath(dir, "params"))
    for (n, text) in files
        write(joinpath(dir, "params", n), text)
    end
    args = picongpu_ion_arguments(case, p)
    sh = picongpu_run_script(p)
    occursin(picongpu_arguments(p), sh) || error("run.sh does not hold PIConGPU's arguments")
    write(joinpath(dir, "run.sh"), replace(sh, picongpu_arguments(p) => args))
    chmod(joinpath(dir, "run.sh"), 0o755)
    write(joinpath(dir, "arguments.txt"), args * "\n")
    return ["run.sh", "arguments.txt", ("params/" .* sort(collect(keys(files))))...]
end

# A record of an openPMD particle species: a dataset, or a constant component (attributes `value` and `shape`).
function pic_record(obj)
    obj isa HDF5.Dataset && return Float64.(vec(read(obj)))
    a = HDF5.attributes(obj)
    haskey(a, "value") || error("record $(HDF5.name(obj)): neither a dataset nor a constant component")
    return fill(Float64(read(a["value"])), prod(Int.(read(a["shape"]))))
end

"""
    picongpu_ion_particles(sp, case, units, jw, shift) -> NamedTuple

The particles of one species of a PIConGPU dump in the ionization case's terms: box x, the
momenta (u_x, u_y, u_z) along the case's axes (PIConGPU's y, x, z), the weight in the units of A1
(real particles over n_c (c/ω₀)² per unit depth, the depth being Δx, in 2D; over n_c (c/ω₀)³ in
3D) and, for the dopant, the bound electrons. `shift` turns `positionOffset` into the total cell (`picongpu_electrons`).
"""
function picongpu_ion_particles(sp, case, u, shift)
    d, lab = ion_derived(case), picongpu_ion_lab(case)
    lay = picongpu_layout(lab)
    wr = pic_record(sp["weighting"])
    mom = sp["momentum"]
    per = Int(pic_attribute(mom, "macroWeighted", 1)) == 1 ? wr .^ Float64(pic_attribute(mom, "weightingPower", 1.0)) : ones(length(wr))
    comp(c) = haskey(mom, c) ? pic_record(mom[c]) .* (Float64(pic_attribute(mom[c], "unitSI", 1.0)) / (PIConGPUDeck.ME_SI * PIConGPUDeck.C_SI)) ./ per : zeros(length(wr))
    cellsize = d.dx * u.length
    off, pos = sp["positionOffset"]["y"], sp["position"]["y"]
    c = pic_record(off) .* (Float64(pic_attribute(off, "unitSI", cellsize)) / cellsize) .+
        pic_record(pos) .* (Float64(pic_attribute(pos, "unitSI", cellsize)) / cellsize)
    x = (c .+ (shift - lay.j0)) .* d.dx .+ d.xa
    w = d.N == 2 ? wr ./ (u.density * u.length^2 * cellsize) : wr ./ (u.density * u.length^3)
    be = haskey(sp, "boundElectrons") ? pic_record(sp["boundElectrons"]) : nothing
    return (; x, u = (comp("y"), comp("x"), comp("z")), w, bound = be)
end

"""
    picongpu_dump_files(dir) -> Vector{Pair{String,Vector{Int}}}

PIConGPU's openPMD files in `dir/openPMD`, each with its iterations in order: the one file
`simData.h5` of the 2D deck (`--openPMD.infix NULL`), or the 3D deck's one file per output,
`simData_<iteration>.h5` (`picongpu_ion_arguments`), ordered by iteration.
"""
function picongpu_dump_files(dir)
    o = joinpath(dir, "openPMD")
    single = joinpath(o, "simData.h5")
    isfile(single) && return [single => sort(h5open(f -> [parse(Int, k) for k in keys(f["data"])], single, "r"))]
    per = sort([(parse(Int, m.captures[1]), joinpath(o, n)) for n in readdir(o) for m in (match(r"^simData_(\d+)\.h5$", n),) if m !== nothing])
    isempty(per) && error("no openPMD files in $o")
    return [path => [it] for (it, path) in per]
end

"Read PIConGPU's dumps of the ionization case in `dir` into the summary format of `pm_ion_output`."
function summarize_picongpu_ion(dir)
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    d, lab = ion_derived(case), picongpu_ion_lab(case)
    lay = picongpu_layout(lab)
    kind = strip(read(joinpath(dir, "kind.txt"), String))
    plasma, dopant = kind != "vacuum", kind ∉ ("vacuum", "plasma-nodopant")
    u = reference_units(Float64(case["lambda0_um"]) * 1e-6)
    outs = Any[]
    offset_reading = nothing
    files = picongpu_dump_files(dir)
    last_it = maximum(it for (_, its) in files for it in its)
    unreadable = Int[]
    for (path, its) in files, it in its
      try
      h5open(path, "r") do f
        data = f["data"]
        mp = rstrip(haskey(HDF5.attributes(f), "meshesPath") ? read_attribute(f, "meshesPath") : "meshes/", '/')
        pp = rstrip(pic_attribute(f, "particlesPath", "particles/"), '/')
        let key = string(it)
            E = data[key][mp]["E"]
            labels = collect(String, read_attribute(E, "axisLabels"))
            labels == (d.N == 2 ? ["y", "x"] : ["z", "y", "x"]) || error("iteration $it: axisLabels $labels for a $(d.N)D case")
            off = Float64.(read_attribute(E, "gridGlobalOffset")) .* Float64(read_attribute(E, "gridUnitSI")) ./ u.length
            oy = off[findfirst(==("y"), labels)]
            jw = round(Int, oy / d.dx)
            abs(jw * d.dx - oy) <= 1e-6 * d.dx || error("iteration $it: gridGlobalOffset $oy is not a whole number of cells")
            # HDF5.jl reads C order reversed, PIConGPU's (x, y[, z]); the case's (x, y[, z]) swaps the first two
            component(c) = permutedims(Float64.(read(E[c])) .* (Float64(read_attribute(E[c], "unitSI")) / u.efield), d.N == 2 ? (2, 1) : (2, 1, 3))
            Ex, Ey = component("y"), component("x")
            m0 = jw - lay.j0 + d.na                            # the window's node j (from 0) ↔ box m = j + m0
            ja, ka = d.jaxis + 1, d.kaxis + 1
            spm = pm_window_shifts(case, it)
            ex = d.N == 2 ? Ex[:, ja] : Ex[:, ja, ka]
            ey = d.N == 2 ? (Ey[:, ja - 1] .+ Ey[:, ja]) ./ 2 : (Ey[:, ja - 1, ka] .+ Ey[:, ja, ka]) ./ 2
            out = Dict{String,Any}("step" => it, "t" => it * d.dt, "window_offset" => jw, "m0" => m0, "pm_shifts" => spm,
                                   "Ex_axis" => ex, "Ex_m0" => m0 + 1, "Ey_axis" => ey, "Ey_absmax" => maximum(abs, Ey))
            out["Ey_envelope"] = running_envelope(ey, round(Int, π / d.dx))
            if plasma
                g = data[key][pp]
                # how positionOffset counts is read off the plasma's electrons, which fill the window to its front
                el = picongpu_electrons(g[PIC_SPECIES], lab, u, jw)
                shifts = Dict("the window" => jw, "the domain's data" => (jw ÷ lay.local_y) * lay.local_y, "every slide" => 0)
                el.offset == "none" || (offset_reading = el.offset)
                shift = shifts[something(offset_reading, "the window")]
                # ParticleMesh's window: box x from s(k)Δx, nx cells
                inwin(x) = (x .>= spm * d.dx) .& (x .< (spm + d.nx) * d.dx)
                for (name, sp) in (("plasma", PIC_SPECIES), ("born", PIC_BORN))
                    (sp == PIC_BORN && !dopant) && continue
                    if !haskey(g, sp)
                        out[name] = electron_measures(Float64[], (Float64[], Float64[], Float64[]), Float64[], case, 1.0)
                        continue
                    end
                    q = picongpu_ion_particles(g[sp], case, u, shift)
                    k = inwin(q.x)
                    out[name] = electron_measures(q.x[k] .- spm * d.dx, map(a -> a[k], q.u), q.w[k], case, 1.0)
                    if sp == PIC_BORN && it == last_it
                        out["born_particles"] = Dict("x" => q.x[k] .- spm * d.dx, "ux" => q.u[1][k], "w" => q.w[k])
                    end
                end
                if dopant
                    cs = zeros(3)
                    if haskey(g, PIC_DOPANT)
                        q = picongpu_ion_particles(g[PIC_DOPANT], case, u, shift)
                        k = inwin(q.x)
                        for (b, w) in zip(q.bound[k], q.w[k])
                            cs[3 - round(Int, b)] += w                 # 2 bound: N⁵⁺, 1: N⁶⁺, 0: N⁷⁺
                        end
                    end
                    out["charge_states"] = cs
                    out["charge_state_names"] = [5, 6, 7]
                end
            end
            push!(outs, out)
        end
      end
      catch e
        # an output whose HDF5 metadata PIConGPU left broken (the 3D vacuum deck's outputs before the window
        # moves, on an RTX 5090) is left out and listed; the comparison pairs the codes' outputs by step
        e isa HDF5.API.H5Error || rethrow()
        push!(unreadable, it)
        @warn "PIConGPU output $it in $path is unreadable, left out" exception = (e, catch_backtrace())
      end
    end
    isempty(outs) && error("no readable PIConGPU output in $dir")
    secs = try
        lines = readlines(joinpath(dir, "container.txt"))
        stamp(q) = DateTime(first(filter(l -> startswith(l, q), lines))[length(q) + 1:end][1:23], "yyyy-mm-ddTHH:MM:SS.sss")
        Dates.value(stamp("# end: ") - stamp("# start: ")) / 1000
    catch
        nothing
    end
    seed = tryparse(Int, something(match(r"-s(\d+)$", basename(dir)), (captures = ["1"],)).captures[1])
    open(io -> JSON.print(io, Dict{String,Any}("code" => "PIConGPU", "kind" => kind, "rates" => "picongpu", "seed" => seed, "outputs" => outs,
                                               "seconds" => secs, "nsteps" => d.steps, "spectrum_edges" => ion_spectrum_edges(case),
                                               "position_offset" => offset_reading, "window_cells" => lay.window,
                                               "unreadable_outputs" => unreadable)),
         joinpath(dir, "summary.json"), "w")
    return dir
end

# --- prepare --------------------------------------------------------------------------------------

"The ParticleMesh kinds of a prepared run: the case, the other codes' rate conventions, each with that code's launched peak field, WP-064's pulse, no dopant."
const PM_ION_KINDS = ("vacuum", "plasma", "plasma-rates-smilei", "plasma-rates-picongpu", "plasma-rates-smilei-a0", "plasma-rates-picongpu-a0", "plasma-axis",
                      "plasma-nodopant")

"Whether the case can run `kind`: a run at another code's peak field needs the ratio the vacuum runs measured (the case's [comparison])."
function ion_kind_available(case, kind)
    m = match(r"^plasma-rates-(smilei|picongpu)-a0$", kind)
    return m === nothing || haskey(get(case, "comparison", Dict()), "$(m.captures[1])_field_ratio")
end

"""
    prepare_ion(case_path, tag; codes = ("pm", "smilei", "picongpu"), seeds = 1:4) -> root

Write the run directories of the ionization case under scripts/runs/ion-<tag>/: ParticleMesh's
`PM_ION_KINDS`, and each Monte Carlo code's vacuum run, plasma run without the dopant and plasma
runs for `seeds`.
"""
function prepare_ion(case_path, tag; codes = ("pm", "smilei", "picongpu"), seeds = 1:4)
    case = TOML.parsefile(case_path)
    ion_derived(case)
    root = joinpath(REPO, "scripts", "runs", "ion-$tag")
    ispath(root) && error("$root exists")
    dirs = String[]
    function make(name, kind, code; seed = 1)
        dir = joinpath(root, name)
        mkpath(dir)
        cp(case_path, joinpath(dir, "case.toml"))
        write(joinpath(dir, "kind.txt"), kind * "\n")
        inputs = ["case.toml", "kind.txt"]
        if code == "smilei"
            write(joinpath(dir, "namelist.py"), smilei_ion_namelist(case; plasma = kind != "vacuum", seed, dopant = kind != "plasma-nodopant"))
            write(joinpath(dir, "run.sh"), SMILEI_ION_RUN_SH)
            chmod(joinpath(dir, "run.sh"), 0o755)
            append!(inputs, ["namelist.py", "run.sh"])
        elseif code == "picongpu"
            append!(inputs, prepare_picongpu_ion(dir, case; kind, seed, tag))
        end
        write_manifest(dir; root = REPO, inputs,
                       extra = Dict{String,Any}("run_id" => "ion-$tag/$name", "code" => Dict("pm" => "ParticleMesh", "smilei" => "Smilei v5.1", "picongpu" => "PIConGPU 0.8.0")[code],
                                                "case" => case["name"], "kind" => kind, "seed" => seed, "generator" => "scripts/lwfa/ionization.jl"))
        push!(dirs, dir)
    end
    "pm" in codes && foreach(k -> make("pm-$k", k, "pm"), [k for k in PM_ION_KINDS if ion_kind_available(case, k)])
    for code in ("smilei", "picongpu")
        code in codes || continue
        make("$code-vacuum", "vacuum", code)
        make("$code-plasma-nodopant", "plasma-nodopant", code)
        foreach(s -> make("$code-plasma-s$s", "plasma", code; seed = s), seeds)
    end
    return root
end

# --- compare ----------------------------------------------------------------------------------

"The measures of `compare_ion`'s table, per species group of an output."
const ION_MEASURES = ("weight", "J_born", "J_trapped", "fast_weight", "ux_max")

# mean, sample standard deviation, minimum and maximum of numbers
function ion_stats(v)
    v = Float64.(v)
    n = length(v)
    m = sum(v) / n
    return Dict{String,Any}("mean" => m, "std" => n > 1 ? sqrt(sum(abs2, v .- m) / (n - 1)) : 0.0, "min" => minimum(v), "max" => maximum(v), "n" => n)
end

"The groups of runs under `runs` that `compare_ion` compares: ParticleMesh's kinds one each, a Monte Carlo code's seeds together."
function ion_groups(runs)
    G = Dict{String,Vector{String}}()
    for name in sort(readdir(runs))
        isfile(joinpath(runs, name, "summary.json")) || continue
        key = replace(name, r"-s\d+$" => "")
        push!(get!(G, key, String[]), joinpath(runs, name))
    end
    return G
end

"""
    ion_series(summaries, group, measure) -> (t, mean, min, max)

A measure of the born (`group = "born"`) or plasma electrons against time over the runs of one
group: per output the mean, minimum and maximum over the seeds (one run: all three the same).
"""
function ion_series(S, group, measure)
    outs = [s["outputs"] for s in S]
    n = minimum(length, outs)
    t = Float64[outs[1][k]["t"] for k in 1:n]
    v = [Float64[haskey(o[k], group) ? Float64(o[k][group][measure]) : NaN for o in outs] for k in 1:n]
    return t, [sum(x) / length(x) for x in v], minimum.(v), maximum.(v)
end

"""
    lineout_distance(a, a0, b, b0, region) -> Float64

The relative rms distance of two axis lineouts on the box nodes they share inside `region`
(a range of m): `a[j]` (j from 0) sits on m = j + a0, `b[j]` on m = j + b0; the rms of a − b
over the larger of the two rms values, NaN when they share no node or both vanish there.
"""
function lineout_distance(a, a0, b, b0, region)
    lo = max(a0, b0, first(region))
    hi = min(a0 + length(a) - 1, b0 + length(b) - 1, last(region))
    hi < lo && return NaN
    A = Float64[a[m - a0 + 1] for m in lo:hi]
    B = Float64[b[m - b0 + 1] for m in lo:hi]
    r = max(sqrt(sum(abs2, A) / length(A)), sqrt(sum(abs2, B) / length(B)))
    return r == 0 ? NaN : sqrt(sum(abs2, A .- B) / length(A)) / r
end

"""
    ion_lineout_distances(S, case) -> Dict

The codes' axis lineouts against each other, pair by pair, in vacuum (E_y) and in the plasma
without the dopant (E_y and the wake's E_x), seed 1 of each Monte Carlo code: per output the
`lineout_distance` on ParticleMesh's window less its back PML and as much again, and the largest
over the outputs, E_y's from t = 50 (the pulse has entered), E_x's from the time the pulse's peak
reaches the plasma, delay + vacuum − x_a (before it, E_x is the pulse's small longitudinal field).
"""
function ion_lineout_distances(S, case)
    d = ion_derived(case)
    cut = 2 * Int(case["grid"]["back_pml_cells"])
    t_wake = Float64(case["laser"]["delay"]) + Float64(case["plasma"]["vacuum"]) - d.xa
    res = Dict{String,Any}()
    for (medium, kind) in (("vacuum", "vacuum"), ("plasma", "plasma-nodopant"))
        codes = [c for c in ("pm", "smilei", "picongpu") if haskey(S, "$c-$kind")]
        for (i, a) in enumerate(codes), b in codes[(i + 1):end]
            oa, ob = S["$a-$kind"][1]["outputs"], S["$b-$kind"][1]["outputs"]
            # outputs paired by step: a code may lack some (`summarize_picongpu_ion`'s unreadable outputs)
            byb = Dict(Int(o["step"]) => o for o in ob)
            pairs = [(x, byb[Int(x["step"])]) for x in oa if haskey(byb, Int(x["step"]))]
            isempty(pairs) && error("$a-$kind and $b-$kind: no output at a common step")
            for (field, key, m0) in (("Ey", "Ey_axis", "m0"), ("Ex", "Ex_axis", "Ex_m0"))
                (medium == "vacuum" && field == "Ex") && continue
                t, v = Float64[], Float64[]
                for (xa, xb) in pairs
                    tk = Float64(xa["t"])
                    tk < (field == "Ex" ? t_wake : 50.0) && continue
                    s = pm_window_shifts(case, Int(xa["step"]))
                    r = lineout_distance(xa[key], Int(xa[m0]), xb[key], Int(xb[m0]), (s + cut):(s + d.nx - 1))
                    isnan(r) && continue                      # no shared node, or no field on either
                    push!(t, tk)
                    push!(v, r)
                end
                res["$medium $field $a / $b"] = Dict("t" => t, "relrms" => v, "max" => isempty(v) ? nothing : maximum(v))
            end
        end
    end
    res["t_wake"] = t_wake
    return res
end

const ION_TABLE_ROWS = (("pm-plasma", "ParticleMesh"), ("pm-plasma-rates-smilei", "ParticleMesh, Smilei's rate"),
                        ("pm-plasma-rates-smilei-a0", "ParticleMesh, Smilei's rate and peak field"),
                        ("pm-plasma-rates-picongpu", "ParticleMesh, PIConGPU's rate"),
                        ("pm-plasma-rates-picongpu-a0", "ParticleMesh, PIConGPU's rate and peak field"), ("pm-plasma-axis", "ParticleMesh, WP-064's pulse"),
                        ("smilei-plasma", "Smilei"), ("picongpu-plasma", "PIConGPU"))

"""
    ion_table(comparison) -> String

The comparison's final results as Markdown: per run (a Monte Carlo code's seeds as mean ± sample
standard deviation) the born electrons' weight, J_born and J_trapped at the read-out, N⁶⁺ at its
peak and J_born over each Monte Carlo code's mean; then the largest lineout distances by pair.
"""
function ion_table(c)
    F = c["final"]
    t = round(F["pm-plasma"]["t"]; digits = 1)
    mcs = [k for k in ("smilei-plasma", "picongpu-plasma") if haskey(F, k)]
    fmt(x) = string(round(x; sigdigits = 3))
    cell(st, scale) = st["n"] > 1 ? "$(fmt(st["mean"] * scale)) ± $(fmt(st["std"] * scale))" : fmt(st["mean"] * scale)
    io = IOBuffer()
    head = ["born weight (10⁻³)", "J_born (10⁻²)", "J_trapped (10⁻²)", "N⁶⁺ at its peak (10⁻³)"]
    append!(head, ["J_born / $(Dict("smilei-plasma" => "Smilei", "picongpu-plasma" => "PIConGPU")[k])" for k in mcs])
    println(io, "| at t = $t | ", join(head, " | "), " |")
    println(io, "|---|", repeat("---|", length(head)))
    for (k, label) in ION_TABLE_ROWS
        haskey(F, k) || continue
        e = F[k]
        n = length(e["runs"])
        row = [cell(e["born"]["weight"], 1e3), cell(e["born"]["J_born"], 1e2), cell(e["born"]["J_trapped"], 1e2),
               cell(e["charge_states"]["populations"][2], 1e3)]
        append!(row, [fmt(e["born"]["J_born"]["mean"] / F[r]["born"]["J_born"]["mean"]) for r in mcs])
        println(io, "| ", label, n > 1 ? " ($n seeds)" : "", " | ", join(row, " | "), " |")
    end
    L = get(c, "lineouts", Dict())
    if !isempty(L)
        names = Dict("pm" => "ParticleMesh", "smilei" => "Smilei", "picongpu" => "PIConGPU")
        pairs = unique([join(split(k)[3:end], " ") for k in keys(L) if k != "t_wake"])
        order = ["pm / smilei", "pm / picongpu", "smilei / picongpu"]
        pairs = [p for p in order if p in pairs]
        println(io)
        println(io, "| largest relative rms distance on the axis (%) | ", join([join(map(x -> names[x], split(p, " / ")), "–") for p in pairs], " | "), " |")
        println(io, "|---|", repeat("---|", length(pairs)))
        for (label, key) in (("vacuum, E_y", "vacuum Ey"), ("plasma without the dopant, E_y", "plasma Ey"), ("plasma without the dopant, wake E_x from t = $(round(Int, L["t_wake"]))", "plasma Ex"))
            println(io, "| ", label, " | ", join([haskey(L, "$key $p") && L["$key $p"]["max"] !== nothing ? fmt(100 * L["$key $p"]["max"]) : "–" for p in pairs], " | "), " |")
        end
    end
    return String(take!(io))
end

"""
    compare_ion(runs, out) -> Dict

Compare the summaries under `runs` (`prepare_ion`'s directories) and write `comparison.json`,
the figures with their sidecars, and a manifest into `out`. `final` holds, per group of runs
(ParticleMesh's kinds, each Monte Carlo code's seeds together), the measures of the born and the
plasma electrons at the read-out and the dopant's charge states at the output where N⁶⁺ peaks;
`ratios` the born measures of each ParticleMesh run over each Monte Carlo code's seed mean, with
the seeds' relative standard deviation next to them; `laser` the largest |E_y| in the window per
output and group; `lineouts` the codes' axis lineouts against each other (`ion_lineout_distances`);
`seconds` the runtimes as the summaries have them.
"""
function compare_ion(runs, out)
    mkpath(out)
    G = ion_groups(runs)
    S = Dict(k => [JSON.parsefile(joinpath(d, "summary.json")) for d in v] for (k, v) in G)
    case = TOML.parsefile(joinpath(first(G[first(sort(collect(keys(G))))]), "case.toml"))
    d = ion_derived(case)
    final = Dict{String,Any}()
    for (k, ss) in S
        e = Dict{String,Any}("runs" => [basename(x) for x in G[k]], "t" => last(ss[1]["outputs"])["t"])
        for group in ("born", "plasma")
            haskey(last(ss[1]["outputs"]), group) || continue
            e[group] = Dict(m => ion_stats([Float64(last(s["outputs"])[group][m]) for s in ss]) for m in ION_MEASURES)
        end
        if haskey(last(ss[1]["outputs"]), "charge_states")
            # the output where N⁶⁺ peaks in the first run: the dopant is in the window and half ionized
            o1 = ss[1]["outputs"]
            kpk = argmax([Float64(o["charge_states"][2]) for o in o1])
            cs = [Float64.(s["outputs"][kpk]["charge_states"]) for s in ss]
            e["charge_states"] = Dict("t" => o1[kpk]["t"], "names" => [5, 6, 7],
                                      "populations" => [ion_stats([c[i] for c in cs]) for i in 1:3])
        end
        final[k] = e
    end
    ratios = Dict{String,Any}()
    mc = [k for k in keys(final) if !startswith(k, "pm-") && haskey(final[k], "born")]
    for k in keys(final), ref in mc
        startswith(k, "pm-") && haskey(final[k], "born") || continue
        q(a, b) = b == 0 ? nothing : a / b               # nothing when the reference has no born electrons
        ratios["$k / $ref"] = Dict(m => Dict("ratio" => q(final[k]["born"][m]["mean"], final[ref]["born"][m]["mean"]),
                                             "ref_rel_std" => q(final[ref]["born"][m]["std"], final[ref]["born"][m]["mean"]),
                                             "ref_seeds" => final[ref]["born"][m]["n"])
                                   for m in ("weight", "J_born", "J_trapped"))
    end
    laser = Dict(k => Dict("t" => Float64[o["t"] for o in ss[1]["outputs"]],
                           "Ey_absmax" => [ion_stats([Float64(s["outputs"][i]["Ey_absmax"]) for s in ss])["mean"] for i in eachindex(ss[1]["outputs"])])
                 for (k, ss) in S)
    result = Dict{String,Any}("case" => case["name"], "dx" => d.dx, "dt" => d.dt, "steps" => d.steps, "groups" => Dict(k => basename.(v) for (k, v) in G),
                              "final" => final, "ratios" => ratios, "laser" => laser, "lineouts" => ion_lineout_distances(S, case),
                              "seconds" => Dict(basename(x) => get(JSON.parsefile(joinpath(x, "summary.json")), "seconds", nothing) for v in values(G) for x in v))
    open(io -> JSON.print(io, result, 2), joinpath(out, "comparison.json"), "w")
    plot_ion_objectives(out, S; N = d.N)
    plot_ion_charge_states(out, S; N = d.N)
    plot_ion_spectra(out, S; N = d.N)
    plot_ion_phase_space(out, S)
    plot_ion_wake(out, S, d)
    write_manifest(out; root = REPO,
                   extra = Dict{String,Any}("runs" => relpath(abspath(runs), REPO), "case" => case["name"],
                                            "summaries" => Dict(basename(x) => "sha256:" * RunManifest.file_hash(joinpath(x, "summary.json")) for v in values(G) for x in v)))
    return result
end

const ION_COLORS = Dict("pm-plasma" => "#1f77b4", "pm-plasma-rates-smilei" => "#9467bd", "pm-plasma-rates-picongpu" => "#17becf",
                        "pm-plasma-rates-picongpu-a0" => "#bcbd22", "pm-plasma-rates-smilei-a0" => "#e377c2",
                        "pm-plasma-axis" => "#7f7f7f", "smilei-plasma" => "#d62728", "picongpu-plasma" => "#2ca02c")
const ION_LABELS = Dict("pm-plasma" => "ParticleMesh", "pm-plasma-rates-smilei" => "ParticleMesh, Smilei's rate",
                        "pm-plasma-rates-picongpu" => "ParticleMesh, PIConGPU's rate", "pm-plasma-axis" => "ParticleMesh, WP-064's pulse",
                        "pm-plasma-rates-picongpu-a0" => "ParticleMesh, PIConGPU's rate and peak field",
                        "pm-plasma-rates-smilei-a0" => "ParticleMesh, Smilei's rate and peak field",
                        "smilei-plasma" => "Smilei (seeds)", "picongpu-plasma" => "PIConGPU (seeds)")
ion_dopant_groups(S) = [k for k in ("pm-plasma", "pm-plasma-rates-smilei", "pm-plasma-rates-smilei-a0", "pm-plasma-rates-picongpu",
                                    "pm-plasma-rates-picongpu-a0", "pm-plasma-axis",
                                    "smilei-plasma", "picongpu-plasma") if haskey(S, k)]

"The measure a weight is counted in: an area in 2D (per unit depth), a volume in 3D."
ion_measure_unit(N) = N == 2 ? "(c/ω₀)²" : "(c/ω₀)³"

# The born electrons' weight and the two objectives against time; a Monte Carlo code's band spans its seeds.
function plot_ion_objectives(out, S; N = 2)
    fig = Figure(size = (1500, 420))
    side = Dict{String,Any}()
    for (col, (m, lab)) in enumerate((("weight", "born electrons (n_c $(ion_measure_unit(N)))"), ("J_born", "J_born, WP-064 (mₑc² n_c $(ion_measure_unit(N)))"),
                                      ("J_trapped", "J_trapped, WP-065 (mₑc² n_c $(ion_measure_unit(N)))")))
        ax = Axis(fig[1, col]; xlabel = "t (1/ω₀)", ylabel = lab)
        for k in ion_dopant_groups(S)
            t, mean, lo, hi = ion_series(S[k], "born", m)
            length(S[k]) > 1 && band!(ax, t, lo, hi; color = (ION_COLORS[k], 0.25))
            lines!(ax, t, mean; color = ION_COLORS[k], label = ION_LABELS[k], linestyle = startswith(k, "pm-") ? :solid : :dash)
            side["$(m)_$k"] = Dict("t" => t, "mean" => mean, "min" => lo, "max" => hi)
        end
        col == 1 && axislegend(ax; position = :lt, labelsize = 11)
    end
    save_plot(out, "born_objectives", fig, side)
end

function plot_ion_charge_states(out, S; N = 2)
    fig = Figure(size = (1000, 420))
    side = Dict{String,Any}()
    for (col, (i, lab)) in enumerate(((2, "N⁶⁺ in the window"), (3, "N⁷⁺ in the window")))
        ax = Axis(fig[1, col]; xlabel = "t (1/ω₀)", ylabel = "$lab (n_c $(ion_measure_unit(N)))")
        for k in ion_dopant_groups(S)
            outs = [s["outputs"] for s in S[k]]
            t = Float64[o["t"] for o in outs[1]]
            v = [sum(Float64(o[j]["charge_states"][i]) for o in outs) / length(outs) for j in eachindex(t)]
            scatterlines!(ax, t, v; color = ION_COLORS[k], label = ION_LABELS[k], markersize = 6)
            side["$(i + 4)_$k"] = Dict("t" => t, "mean" => v)
        end
        col == 1 && axislegend(ax; position = :lt, labelsize = 11)
    end
    save_plot(out, "charge_states", fig, side)
end

function plot_ion_spectra(out, S; N = 2)
    k0 = first(ion_dopant_groups(S))
    edges = Float64.(S[k0][1]["spectrum_edges"])
    mids = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
    fig = Figure(size = (900, 420))
    ax = Axis(fig[1, 1]; xlabel = "E_kin (mₑc²)", ylabel = "born electrons per bin (n_c $(ion_measure_unit(N)))",
              title = "born electrons ahead of the fade at the read-out (bins of $(round(edges[2] - edges[1]; digits = 3)) mₑc²)")
    side = Dict{String,Any}("ekin" => mids)
    for k in ion_dopant_groups(S)
        h = sum(Float64.(last(s["outputs"])["born"]["spectrum"]) for s in S[k]) ./ length(S[k])
        stairs!(ax, edges, vcat(h, h[end]); step = :post, color = ION_COLORS[k], label = ION_LABELS[k])
        side[k] = h
    end
    axislegend(ax; position = :lt, labelsize = 11)
    save_plot(out, "born_spectrum", fig, side)
end

# The born electrons of the first run of each group at the read-out: window x against u_x.
function plot_ion_phase_space(out, S)
    ks = [k for k in ion_dopant_groups(S) if haskey(last(S[k][1]["outputs"]), "born_particles") &&
                                             any(>(0), Float64.(last(S[k][1]["outputs"])["born_particles"]["w"]))]
    isempty(ks) && return nothing                           # no run kept a born electron
    fig = Figure(size = (450 * length(ks), 400))
    side = Dict{String,Any}()
    for (col, k) in enumerate(ks)
        p = last(S[k][1]["outputs"])["born_particles"]
        x, ux, w = Float64.(p["x"]), Float64.(p["ux"]), Float64.(p["w"])
        ax = Axis(fig[1, col]; xlabel = "window x (c/ω₀)", ylabel = "u_x", title = ION_LABELS[k])
        keep = w .> 0
        scatter!(ax, x[keep], ux[keep]; markersize = 2, color = (ION_COLORS[k], 0.5))
        side[k] = Dict("particles" => count(keep), "x_range" => isempty(x) ? nothing : extrema(x), "ux_range" => isempty(ux) ? nothing : extrema(ux))
    end
    save_plot(out, "born_phase_space", fig, side)
end

# E_x on the axis against box x at three outputs, every code's run without the dopant (one seed: the plasma's load is not random).
function plot_ion_wake(out, S, d)
    ks = [k for k in ("pm-plasma-nodopant", "smilei-plasma-nodopant", "picongpu-plasma-nodopant") if haskey(S, k)]
    isempty(ks) && return
    n = length(S[ks[1]][1]["outputs"])
    picks = unique(clamp.((n - 4, n - 2, n), 1, n))
    fig = Figure(size = (1100, 280 * length(picks)))
    side = Dict{String,Any}()
    col = Dict("pm-plasma-nodopant" => "#1f77b4", "smilei-plasma-nodopant" => "#d62728", "picongpu-plasma-nodopant" => "#2ca02c")
    for (row, j) in enumerate(picks)
        ax = Axis(fig[row, 1]; xlabel = "x (c/ω₀), box", ylabel = "E_x on the axis", title = "t = $(round(S[ks[1]][1]["outputs"][j]["t"]; digits = 1))")
        for k in ks
            o = S[k][1]["outputs"][j]
            ex = Float64.(o["Ex_axis"])
            x = ((0:(length(ex) - 1)) .+ Int(o["Ex_m0"]) .- 0.5) .* d.dx
            lines!(ax, x, ex; color = col[k], label = Dict("pm" => "ParticleMesh", "smilei" => "Smilei", "picongpu" => "PIConGPU")[split(k, "-")[1]])
            side["$(k)_t$(o["step"])"] = Dict("x0" => first(x), "dx" => d.dx, "Ex" => ex)
        end
        row == 1 && axislegend(ax; position = :lb)
    end
    save_plot(out, "wake_axis", fig, side)
end

# --- command line ---------------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    cmd = get(ARGS, 1, "")
    if cmd == "prepare"
        codes = Tuple(w for w in ARGS[4:end] if w in ("pm", "smilei", "picongpu"))
        println(prepare_ion(ARGS[2], ARGS[3]; codes = isempty(codes) ? ("pm", "smilei", "picongpu") : codes))
    elseif cmd == "run-pm"
        be = backend_named(get(ARGS, 3, "cpu"))
        float = get(ARGS, 4, "Float64") == "Float32" ? Float32 : Float64
        Base.invokelatest(run_pm_ion, ARGS[2]; backend = be, float)
    elseif cmd == "summarize-smilei"
        summarize_smilei_ion(ARGS[2])
    elseif cmd == "summarize-picongpu"
        summarize_picongpu_ion(ARGS[2])
    elseif cmd == "compare"
        compare_ion(ARGS[2], ARGS[3])
    elseif cmd == "table"
        print(ion_table(JSON.parsefile(ARGS[2])))
    else
        error("usage: ionization.jl prepare <case> <tag> [pm|smilei|picongpu …] | run-pm <dir> [cpu|cuda] [Float64|Float32] | summarize-smilei <dir> | summarize-picongpu <dir> | compare <runs> <out> | table <comparison.json>")
    end
end
