# What sets the grid and the particles per cell of the LWFA case with field ionization: the case of
# WP-065 (trapped2d_case.jl: WP-064's ionization case at a fixed laser energy, with the first
# bucket's trapped electrons as the objective J) run over the cell sizes along x and y, the
# particles per cell of the plasma and of the dopant, the element type, the deposition mode, the
# shape order, the Courant number, the ionization rate law and the birth cadence, each run with
# the same read-outs, and with ForwardDiff duals in the four parameters of the optimization
# (duration, waist, focus, density) where a job asks for the gradient. No optimizer runs here.
#
#   julia +1.13 --project=scripts -t 4 scripts/resolution/gridppc.jl campaign <name> <dir> [cpu|cuda]
#   julia +1.13 --project=scripts -t 4 scripts/resolution/gridppc.jl job <dir> <job JSON> [cpu|cuda]
#   julia +1.13 --project=scripts scripts/resolution/gridppc.jl list <name>
#
# A campaign is a list of independent jobs run one after the other in this process (so that a
# backend compiles each kernel once); each writes `<id>.json` into the directory, a job whose JSON
# is there is not run again, and a job that fails writes `<id>.err` and the campaign goes on.
# While a file `PAUSE` exists in the directory the campaign waits between jobs (for timings of
# other work on a shared GPU). `frames` jobs also write `<id>.frames.bin` (Float32 maps).
#
# The numbers are measurements of single runs, not independently reviewed.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "..", "optim", "trapped2d_case.jl"))
using ParticleMeshElectromagnetic: window_charge!, window_shifts, ionization_state, ionization_electrons, ionization_lost

const BACKEND_NAME = let b = filter(a -> a in ("cpu", "cuda"), ARGS)
    isempty(b) ? "cpu" : b[end]
end
if BACKEND_NAME == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
    @eval const BE = CUDABackend()
else
    const BE = CPU()
end

"The directions of a gradient job, positions in x = (duration, waist, focus, delay, density)."
const GRAD_DIRS = (1, 2, 3, 5)
const GRAD_NAMES = ("duration", "waist", "focus", "density")

"The defaults of a job; a job's own keys replace them."
const JOB_DEFAULTS = Dict{String,Any}("ppl" => 16, "ny" => 1, "ppc" => [2, 2], "ion_ppc" => [1, 1], "T" => "f64", "mode" => "reference",
                                      "shape" => 2, "cfl" => 95, "law" => "adk", "every" => 1, "a0E" => 1.9, "x" => collect(TCASE.x0),
                                      "grad" => false, "series" => true, "frames" => 0, "a0_scale" => 1.0, "t_end" => BCASE.t_end,
                                      "dopant_fraction" => TCASE.dopant_fraction, "rep" => 0)

eltype_named(s) = s == "f64" ? Float64 : s == "f32" ? Float32 : error("element type $s: f64 or f32")

"The two K-shell levels of nitrogen by the job's law: `adk` (the case's), `tonglin` (with the barrier-suppression factor), `adk_tl` (Tong and Lin's conventions without the factor)."
function study_levels(::Type{R}, law) where {R}
    kw = (; wavelength = BCASE.wavelength, cutoff = BCASE.cutoff)
    law == "adk" && return (adk_level(R, :N, 6; kw...), adk_level(R, :N, 7; kw...))
    law == "tonglin" && return (tong_lin_level(R, :N, 6; kw...), tong_lin_level(R, :N, 7; kw...))
    law == "adk_tl" && return (adk_level(R, :N, 6; lstar = 0, electrons = 2, kw...), adk_level(R, :N, 7; lstar = 0, electrons = 1, kw...))
    error("law $law: adk, tonglin or adk_tl")
end

"""
    study_problem(x, a0E; T, ppl, ny, ppc, ion_ppc, cfl, law, every, capacity, t_end, a0_scale, dopant_fraction)

The case of `trapped_problem` at x = (duration, waist, focus, delay, density) and the energy budget
of `a0E`, with what that function fixes left open: Δx = λ₀/`ppl` for any `ppl` (Δy = π/(4·`ny`)
whatever `ppl` is), the plasma's and the dopant's particles per cell, the time step as `cfl`
hundredths of the Courant limit, the rate law, and the element type `T` (the problem's is that of
the parameters, which are converted to `T` or are duals of it). At the defaults it is
`trapped_problem`'s problem.
"""
function study_problem(x, a0E; T = Float64, ppl::Integer, ny::Integer = 1, ppc = (2, 2), ion_ppc = (1, 1), cfl::Integer = 95, law = "adk",
                       every::Integer = 1, capacity::Integer, t_end = BCASE.t_end, a0_scale = 1.0, dopant_fraction = TCASE.dopant_fraction)
    duration, waist, focus, delay, density = x
    a0 = a0_for_energy(duration, waist; energy = budget(a0E)) * T(a0_scale)
    dopant = T(dopant_fraction) * density
    R = promote_type(typeof(a0), typeof(dopant), typeof(duration), typeof(density), typeof(waist), typeof(delay), typeof(focus), T)
    z3 = (zero(R), zero(R), zero(R))
    h = 2 * R(π) / ppl
    hy = R(π) / (4 * ny)
    cells = (BCASE.length_wavelengths * ppl, BCASE.rows * ny)
    dt = R(cfl) / 100 / sqrt(1 / h^2 + 1 / hy^2)
    nsteps = ceil(Int, t_end / Float64(value_of(dt)))
    hy64 = Float64(value_of(hy))
    yc = cells[2] * hy64 / 2
    pulse = GaussianLaser(; a0 = R(a0), waist = R(waist), duration = R(duration), delay = R(delay), focus = R(focus),
                          position = max(12, 192 ÷ ppl) * (2 * R(π) / 32), center = cells[2] * hy / 2)
    e = SpeciesDescription{R}(:electron, -one(R), one(R), R(density), zero(R), (1, 0), Tuple(ppc), z3, z3, 1)
    ion = FieldIonization(; density = R(dopant), levels = study_levels(R, law), capacity, every, ppc = Tuple(ion_ppc),
                          rows = (yc - BCASE.half_rows * ny * hy64, yc + BCASE.half_rows * ny * hy64),
                          section = DopantSection(; start = R(BCASE.section.start), stop = R(BCASE.section.stop), ramp = R(BCASE.section.ramp)))
    return EMProblem("gridppc", (cells[1] * h, cells[2] * hy), cells, dt, nsteps, (e,), 1, (1, 0);
                     lasers = (pulse,), boundaries = (:window, :open), window = MovingWindow(; start = R(BCASE.start)),
                     profile = DensityRamp(; vacuum = R(BCASE.vacuum), ramp = R(BCASE.ramp)), ionization = ion)
end

"The case's numerics (`born_numerics`) with the job's shape order and deposition mode."
study_numerics(ppl, ny; shape = 2, mode = "reference") =
    EMNumerics(; shape = shape == 2 ? Quadratic : shape == 1 ? Linear : error("shape order $shape: 1 or 2"),
               mode = mode == "reference" ? Reference() : mode == "fast" ? Fast() : error("mode $mode: reference or fast"),
               absorber = PML(; cells = BCASE.pml_y * ny), back_absorber = PML(; cells = max(4, round(Int, BCASE.pml_x_wavelengths * ppl))))

"The electron slots a job asks for: the dopant lattice of the whole window (a birth step offers one slot per atom slot) and room for the electrons alive."
function study_capacity(job)
    ppl, ny, ip = Int(job["ppl"]), Int(job["ny"]), Int.(job["ion_ppc"])
    atoms = BCASE.length_wavelengths * ppl * 2 * BCASE.half_rows * ny * prod(ip)
    return haskey(job, "capacity") ? Int(job["capacity"]) : atoms + max(400_000, round(Int, 6 * atoms * (ppl / 16) / Int(job["every"])))
end

# --- read-outs -----------------------------------------------------------------------------------

f64(a) = Float64.(value_of.(Array(a)))

"The on-axis row of a field over the window's columns, on the host as Float64."
function axis_line(st, a)
    g, n = guards(st.grid), ncells(st.grid)
    return f64(view(a, (g + 1):(g + n[1]), g + 1 + n[2] ÷ 2))
end

"""
    envelope(v, lo, hi) -> Vector

The modulus of the analytic signal of `v` restricted to the Fourier modes `lo:hi` of the line (a
plain discrete Fourier sum over that band): the envelope of a carrier whose wavenumber lies in the
band, with what lies outside it (the wake, the grid's noise) left out.
"""
function envelope(v::Vector{Float64}, lo::Integer, hi::Integer)
    n = length(v)
    tab = [cis(2π * m / n) for m in 0:(n - 1)]
    F = [sum(v[i + 1] * conj(tab[(k * i) % n + 1]) for i in 0:(n - 1)) for k in lo:hi]
    return [2 * abs(sum(F[q] * tab[((lo + q - 1) * i) % n + 1] for q in eachindex(F))) / n for i in 0:(n - 1)]
end

"The place and the value of the largest entry of `v`, refined by the parabola through it and its neighbours (in entries, 1-based)."
function parabolic_peak(v)
    i = argmax(v)
    (i == 1 || i == length(v)) && return Float64(i), v[i]
    a, b, c = v[i - 1], v[i], v[i + 1]
    d = a - 2b + c
    d == 0 && return Float64(i), b
    s = (a - c) / (2d)
    return i + s, b - (a - c) * s / 4
end

"""
    axis_readout(st, k, ppl) -> Dict

The on-axis fields after step `k`, in window coordinates (c/ω₀, the first column at h): the
laser's envelope (the analytic signal of E_y over the wavenumbers 0.3 to 3 ω₀/c) with its peak,
the place of the peak, its energy centroid and rms length and ∫ env² dx; the wake's E_x averaged
over a wavelength, with its zero crossings behind the pulse (the nearest first) and its extrema
over the 40 c/ω₀ behind the pulse; and the lab x of the window's first column.
"""
function axis_readout(st, k, ppl)
    h = Float64(value_of(cellsize(st.grid)[1]))
    n = ncells(st.grid)[1]
    Ey, Ex = axis_line(st, st.fields.Ey), axis_line(st, st.fields.Ex)
    L = n * h
    env = envelope(Ey, max(1, round(Int, 0.3 * L / 2π)), round(Int, 3 * L / 2π))
    ip, a_peak = parabolic_peak(env)
    e2 = env .^ 2
    U = sum(e2) * h
    xs = (1:n) .* h
    xc = sum(xs .* e2) * h / U
    len = sqrt(sum((xs .- xc) .^ 2 .* e2) * h / U)
    half = ppl ÷ 2
    sm = [mean(@view Ex[max(1, i - half):min(n, i + half)]) for i in 1:n]
    ic = clamp(round(Int, xc / h), 2, n)
    zeros_x = Float64[]
    for i in ic:-1:2
        if sm[i - 1] * sm[i] < 0
            push!(zeros_x, (i - 1 + sm[i - 1] / (sm[i - 1] - sm[i])) * h)
            length(zeros_x) == 4 && break
        end
    end
    lo = max(1, ic - round(Int, 40 / h))
    return Dict{String,Any}("a_peak" => a_peak, "x_peak" => ip * h, "x_centroid" => xc, "length" => len, "U" => U, "Ey_max" => maximum(abs, Ey),
                            "Ex_zeros" => zeros_x, "Ex_min" => minimum(@view sm[lo:ic]), "Ex_max" => maximum(@view sm[lo:ic]),
                            "x0" => window_shifts(st, k) * h)
end

"The born electrons alive, as host arrays of plain numbers: window x, y, the momenta and the weight."
function born_arrays(st)
    e = ionization_electrons(st)
    h = Float64.(value_of.(cellsize(st.grid)))
    alive = Array(e.status) .== ALIVE
    x = (Array(e.cell[1])[alive] .- 1 .+ f64(e.offset[1])[alive]) .* h[1]
    y = (Array(e.cell[2])[alive] .- 1 .+ f64(e.offset[2])[alive]) .* h[2]
    return (; x, y, ux = f64(e.u[1])[alive], uy = f64(e.u[2])[alive], uz = f64(e.u[3])[alive], w = f64(e.w)[alive])
end

"""
    bunch_readout(st) -> Dict

The trapped born electrons of the first bucket as a bunch (the weights w·S·R of the objective):
macro-particles alive, those with u_x above the trapping weight's upper edge, the effective number
(Σq)²/Σq² of the objective's summands q = w(γ−1)·S·R and of the weights, the share of J the
heaviest hundredth of the particles carries, the mean and rms of γ − 1, the rms of y and u_y about
their means and the normalized rms emittance in y, and the weighted spectrum over γ − 1 in bins
of 1 from 0 to 60.
"""
function bunch_readout(st)
    b = born_arrays(st)
    area = prod(Float64.(value_of.(cellsize(st.grid))))
    g = sqrt.(1 .+ b.ux .^ 2 .+ b.uy .^ 2 .+ b.uz .^ 2) .- 1
    s = b.w .* smooth_rise.(b.x, TCASE.fade[1], TCASE.fade[2]) .* smooth_rise.(b.ux, TCASE.rise[1], TCASE.rise[2])
    q = s .* g
    W, Q = sum(s; init = 0.0), sum(q; init = 0.0)
    out = Dict{String,Any}("alive" => length(b.w), "fast" => count(>(TCASE.rise[2]), b.ux), "weight" => area * W,
                           "weight_alive" => area * sum(b.w; init = 0.0))
    spec = zeros(60)
    for i in eachindex(g)
        j = floor(Int, g[i]) + 1
        1 <= j <= 60 && (spec[j] += area * s[i])
    end
    out["spectrum"] = spec
    W > 0 || return out
    m(v) = sum(s .* v) / W
    ey, euy, eg = m(b.y), m(b.uy), m(g)
    vy, vuy, cyu = m((b.y .- ey) .^ 2), m((b.uy .- euy) .^ 2), m((b.y .- ey) .* (b.uy .- euy))
    qs = sort(q; rev = true)
    top = max(1, length(qs) ÷ 100)
    merge!(out, Dict{String,Any}("neff_J" => Q^2 / sum(abs2, q), "neff_w" => W^2 / sum(abs2, s), "top1_share" => sum(@view qs[1:top]) / Q,
                                 "mean_energy" => eg, "rms_energy" => sqrt(m((g .- eg) .^ 2)), "rms_y" => sqrt(vy), "rms_uy" => sqrt(vuy),
                                 "emittance_y" => sqrt(max(0.0, vy * vuy - cyu^2)), "ux_max" => maximum(b.ux), "mean_x" => m(b.x)))
    return out
end

"""
    wake_energy(st) -> W

A second objective with no ionization in it, for comparison: the energy of the longitudinal field
Δx·Δy·Σ E_x²/2 over the first bucket's box (window x from 24 to 44, within 8 c/ω₀ of the axis),
summed on the host in storage order.
"""
function wake_energy(st)
    g, n = guards(st.grid), ncells(st.grid)
    h = cellsize(st.grid)
    h64 = Float64.(value_of.(h))
    i1, i2 = round(Int, 24 / h64[1]), round(Int, 44 / h64[1])
    jc, dj = n[2] ÷ 2, round(Int, 8 / h64[2])
    Ex = Array(view(st.fields.Ex, (g + i1):(g + i2), (g + 1 + jc - dj):(g + jc + dj)))
    return h[1] * h[2] * foldl(+, vec(Ex .* Ex)) / 2
end

"The plasma's kinetic energy Δx·Δy·Σ w(γ−1) ahead of window x = `from` (the plasma the pulse has not reached): what the grid alone has heated."
function ahead_energy(st, from)
    h = Float64.(value_of.(cellsize(st.grid)))
    return h[1] * h[2] * sum(st.species; init = 0.0) do s
        x, g, w = plasma_arrays(s, h[1], from)
        sum(w .* g; init = 0.0)
    end
end

"One sample of a run's time series after step `k`."
function sample(st, k, dt, ppl)
    out = axis_readout(st, k, ppl)
    b = born_arrays(st)
    area = prod(Float64.(value_of.(cellsize(st.grid))))
    a = ionization_state(st)
    parts = trapped_parts(st)
    merge!(out, Dict{String,Any}("t" => k * dt, "step" => k, "born_alive" => length(b.w), "born_weight" => area * sum(b.w; init = 0.0),
                                 "born_fast_weight" => area * sum(b.w[b.ux .> TCASE.rise[2]]; init = 0.0),
                                 "released_in_window" => area * sum(f64(a.w) .* f64(a.emitted); init = 0.0),
                                 "J" => Float64(value_of(parts.born + parts.plasma)), "J_plasma" => Float64(value_of(parts.plasma)),
                                 "ux_max" => maximum(b.ux; init = 0.0),
                                 "ahead_energy" => ahead_energy(st, min(out["x_centroid"] + 20, ncells(st.grid)[1] * Float64(value_of(cellsize(st.grid)[1])) - 4))))
    return out
end

"Maps of a frame on a fixed raster of `nx × ny` points over the window (nearest cell): E_y, E_x and the electron density over the plasma's (the plasma's and the born electrons' charge with the dopant ions')."
function frame_maps(st, ρbuf, density, nx, ny)
    g, n = guards(st.grid), ncells(st.grid)
    I = [g + clamp(round(Int, (i - 0.5) * n[1] / nx + 0.5), 1, n[1]) for i in 1:nx]
    J = [g + clamp(round(Int, (j - 0.5) * n[2] / ny + 0.5), 1, n[2]) for j in 1:ny]
    window_charge!(ρbuf, st)
    return Float32.(Array(st.fields.Ey)[I, J]), Float32.(Array(st.fields.Ex)[I, J]), Float32.(-Array(ρbuf)[I, J] ./ density)
end

# --- a job -----------------------------------------------------------------------------------------

"""
    run_job(job, dir) -> Dict

One run of the case as the job describes it (`JOB_DEFAULTS`). A plain job gives J, the summaries,
the bunch, the wake's energy, the on-axis lines at the read-out and, with `series`, the time
series every 2.5 c/ω₀; a `grad` job runs with duals in duration, waist, focus and density and
gives J with its gradient and the wake energy with its own.
"""
function run_job(job0, dir)
    job = merge(JOB_DEFAULTS, job0)
    T = eltype_named(job["T"])
    ppl, ny = Int(job["ppl"]), Int(job["ny"])
    x64, a0E = Float64.(job["x"]), Float64(job["a0E"])
    grad = job["grad"] == true
    x = if grad
        ntuple(i -> (q = findfirst(==(i), GRAD_DIRS); q === nothing ? T(x64[i]) : ForwardDiff.Dual{Nothing}(T(x64[i]), ntuple(j -> T(j == q), length(GRAD_DIRS)))), 5)
    else
        Tuple(T.(x64))
    end
    capacity = study_capacity(job)
    prob = study_problem(x, T(a0E); T, ppl, ny, ppc = Int.(job["ppc"]), ion_ppc = Int.(job["ion_ppc"]), cfl = Int(job["cfl"]), law = job["law"],
                         every = Int(job["every"]), capacity, t_end = Float64(job["t_end"]), a0_scale = Float64(job["a0_scale"]),
                         dopant_fraction = Float64(job["dopant_fraction"]))
    numerics = study_numerics(ppl, ny; shape = Int(job["shape"]), mode = job["mode"])
    t0 = time()
    st = EMState(prob, numerics; backend = BE, record = ())
    p = params(prob)
    initialize!(st, p)
    dt = Float64(value_of(prob.dt))
    series = Dict{String,Any}[]
    nframes = Int(job["frames"])
    frame_steps = nframes > 0 ? unique(round.(Int, range(prob.nsteps / nframes, prob.nsteps; length = nframes))) : Int[]
    sample_steps = job["series"] == true && !grad ? unique(round.(Int, (2.5:2.5:Float64(job["t_end"])) ./ dt)) : Int[]
    stops = sort(unique(vcat(frame_steps, sample_steps, prob.nsteps)))
    frame_io = nframes > 0 ? open(joinpath(dir, job["id"] * ".frames.bin"), "w") : nothing
    ρbuf = nframes > 0 ? field(BE, eltype(st.fields.Ex), st.grid) : nothing
    frames_meta = Dict{String,Any}[]
    FX, FY = 448, 128
    k = 0
    for stop in stops
        stop > k && run!(st, p, stop - k; first = k + 1)
        k = stop
        stop in sample_steps && push!(series, sample(st, k, dt, ppl))
        if stop in frame_steps
            Ey, Ex, ne = frame_maps(st, ρbuf, x64[5], FX, FY)
            write(frame_io, Ey, Ex, ne)
            b = born_arrays(st)
            keep = length(b.w) <= 4000 ? collect(eachindex(b.w)) : collect(1:(length(b.w) ÷ 4000):length(b.w))
            push!(frames_meta, Dict{String,Any}("t" => k * dt, "x0" => window_shifts(st, k) * Float64(value_of(cellsize(st.grid)[1])),
                                                "born_x" => round.(b.x[keep]; digits = 2), "born_y" => round.(b.y[keep]; digits = 2),
                                                "born_ux" => round.(b.ux[keep]; digits = 2), "born_w" => b.w[keep], "born_alive" => length(b.w)))
        end
    end
    frame_io === nothing || close(frame_io)
    seconds = time() - t0
    lost = ionization_lost(st)
    J = trapped_energy(st)
    W = wake_energy(st)
    parts = trapped_parts(st)
    h = Float64.(value_of.(cellsize(st.grid)))
    out = Dict{String,Any}(job)
    merge!(out, Dict{String,Any}(string(k) => v for (k, v) in pairs(trapped_summary(st))))
    merge!(out, Dict{String,Any}("J" => Float64(value_of(J)), "W" => Float64(value_of(W)), "J_born_label" => Float64(value_of(born_energy(st))),
                                 "seconds" => seconds, "threads" => Threads.nthreads(), "backend" => BACKEND_NAME, "cells" => collect(ncells(st.grid)),
                                 "dx" => h[1], "dy" => h[2], "dt" => dt, "nsteps" => prob.nsteps, "capacity" => capacity, "ionization_lost" => lost,
                                 "a0" => Float64(value_of(a0_for_energy(x[1], x[2]; energy = budget(a0E)))) * Float64(job["a0_scale"]),
                                 "plasma_slots" => sum(s -> length(s.w), st.species), "bunch" => bunch_readout(st),
                                 "atoms" => length(ionization_state(st).w)))
    if grad
        out["gradient"] = Float64.(collect(ForwardDiff.partials(J)))
        out["gradient_born"] = Float64.(collect(ForwardDiff.partials(parts.born)))
        out["gradient_plasma"] = Float64.(collect(ForwardDiff.partials(parts.plasma)))
        out["gradient_W"] = Float64.(collect(ForwardDiff.partials(W)))
        out["gradient_names"] = collect(GRAD_NAMES)
    else
        out["axis"] = axis_readout(st, prob.nsteps, ppl)
        nx = ncells(st.grid)[1]
        stride = max(1, nx ÷ 896)
        out["line_x"] = round.(collect((1:stride:nx) .* h[1]); digits = 4)
        out["line_Ey"] = Float32.(axis_line(st, st.fields.Ey)[1:stride:nx])
        out["line_Ex"] = Float32.(axis_line(st, st.fields.Ex)[1:stride:nx])
        isempty(series) || (out["series"] = series)
        isempty(frames_meta) || (out["frames"] = frames_meta; out["frame_raster"] = [FX, FY])
    end
    return out
end

# --- campaigns -------------------------------------------------------------------------------------

jid(parts...) = join(string.(parts), "-")
tag(v) = join(string.(v), "x")

"A plain job with its time series and the gradient job of the same configuration."
function pair!(jobs, id; kw...)
    base = Dict{String,Any}(string(k) => v for (k, v) in kw)
    push!(jobs, merge(base, Dict{String,Any}("id" => id)))
    push!(jobs, merge(base, Dict{String,Any}("id" => id * "-grad", "grad" => true)))
    return jobs
end
plain!(jobs, id; kw...) = push!(jobs, merge(Dict{String,Any}(string(k) => v for (k, v) in kw), Dict{String,Any}("id" => id)))

"The jobs of a campaign, the cheap ones first."
function campaign_jobs(name)
    jobs = Dict{String,Any}[]
    if name == "smoke"
        pair!(jobs, "smoke-r16x1"; ppl = 16, ny = 1)
    elseif name == "smoke8"
        pair!(jobs, "smoke8"; ppl = 8, ny = 1, t_end = 150.0)
        plain!(jobs, "smoke8-f32"; ppl = 8, ny = 1, t_end = 150.0, T = "f32")
        plain!(jobs, "smoke8-fast"; ppl = 8, ny = 1, t_end = 150.0, mode = "fast")
        plain!(jobs, "smoke8-linear"; ppl = 8, ny = 1, t_end = 150.0, shape = 1)
        plain!(jobs, "smoke8-tonglin"; ppl = 8, ny = 1, t_end = 150.0, law = "tonglin")
        plain!(jobs, "smoke8-ppc"; ppl = 8, ny = 1, t_end = 150.0, ppc = [3, 3], ion_ppc = [2, 2], cfl = 70, every = 2)
        plain!(jobs, "smoke8-frames"; ppl = 8, ny = 1, t_end = 150.0, frames = 4)
        push!(jobs, Dict{String,Any}("id" => "smoke8-f32-grad", "ppl" => 8, "ny" => 1, "t_end" => 150.0, "T" => "f32", "grad" => true))
    elseif name == "grid"
        # the cell sizes along x and y, each run plain and with duals
        for ppl in (16, 24, 32, 48, 64, 96), ny in (1, 2, 4)
            pair!(jobs, jid("grid", "r$(ppl)x$(ny)"); ppl, ny)
        end
        for (ppl, ny) in ((32, 8), (48, 3), (48, 6), (64, 8), (96, 6), (128, 4))
            pair!(jobs, jid("grid", "r$(ppl)x$(ny)"); ppl, ny)
        end
        # the same along the energy budget: the three budgets of WP-065 on the diagonal Δy = 2Δx
        for a0E in (2.175, 2.4), (ppl, ny) in ((16, 1), (32, 2), (48, 3), (64, 4))
            pair!(jobs, jid("grid", "E$a0E", "r$(ppl)x$(ny)"); ppl, ny, a0E)
        end
    elseif name == "ppc"
        for (ppl, ny) in ((16, 1), (32, 2), (48, 3)), ppc in ((1, 1), (2, 1), (1, 2), (2, 2), (3, 3), (4, 4), (6, 6), (8, 8))
            ppl == 48 && prod(ppc) > 16 && continue
            pair!(jobs, jid("ppc", "r$(ppl)x$(ny)", "p$(tag(ppc))"); ppl, ny, ppc = collect(ppc))
        end
        for (ppl, ny) in ((16, 1), (32, 2), (48, 3)), ip in ((2, 1), (1, 2), (2, 2), (4, 4)), ppc in ((2, 2), (4, 4))
            pair!(jobs, jid("ppc", "r$(ppl)x$(ny)", "p$(tag(ppc))", "i$(tag(ip))"); ppl, ny, ppc = collect(ppc), ion_ppc = collect(ip))
        end
    elseif name == "types"
        for (ppl, ny) in ((16, 1), (32, 2), (48, 3), (64, 4))
            pair!(jobs, jid("types", "r$(ppl)x$(ny)", "f32"); ppl, ny, T = "f32")
            for rep in 1:3, T in ("f32", "f64")
                plain!(jobs, jid("types", "r$(ppl)x$(ny)", T, "fast", rep); ppl, ny, T, mode = "fast", rep, series = false)
            end
            plain!(jobs, jid("types", "r$(ppl)x$(ny)", "f64", "again"); ppl, ny, rep = 1, series = false)
            plain!(jobs, jid("types", "r$(ppl)x$(ny)", "f32", "again"); ppl, ny, T = "f32", rep = 1, series = false)
            push!(jobs, Dict{String,Any}("id" => jid("types", "r$(ppl)x$(ny)", "f32", "fast", "grad"), "ppl" => ppl, "ny" => ny, "T" => "f32",
                                         "mode" => "fast", "grad" => true))
        end
        for ppc in ((1, 1), (4, 4), (8, 8))
            pair!(jobs, jid("types", "r32x2", "p$(tag(ppc))", "f32"); ppl = 32, ny = 2, T = "f32", ppc = collect(ppc))
        end
    elseif name == "numerics"
        for (ppl, ny) in ((16, 1), (32, 2), (48, 3), (64, 4))
            pair!(jobs, jid("num", "r$(ppl)x$(ny)", "linear"); ppl, ny, shape = 1)
            for cfl in (50, 70, 85, 99)
                pair!(jobs, jid("num", "r$(ppl)x$(ny)", "cfl$cfl"); ppl, ny, cfl)
            end
            for law in ("tonglin", "adk_tl")
                pair!(jobs, jid("num", "r$(ppl)x$(ny)", law); ppl, ny, law)
            end
        end
        for every in (2, 4, 8, 16)
            pair!(jobs, jid("num", "r32x2", "every$every"); ppl = 32, ny = 2, every)
        end
        for every in (2, 4)
            pair!(jobs, jid("num", "r64x4", "every$every"); ppl = 64, ny = 4, every)
        end
    elseif name == "rough"
        # 33 plain runs a relative 2⁻¹⁰ apart in one parameter, with the gradient at the middle and the ends
        cfgs = [("r16x1-p2x2", (; ppl = 16, ny = 1)), ("r32x2-p1x1", (; ppl = 32, ny = 2, ppc = [1, 1])), ("r32x2-p2x2", (; ppl = 32, ny = 2)),
                ("r32x2-p4x4", (; ppl = 32, ny = 2, ppc = [4, 4])), ("r32x2-p2x2-i2x2", (; ppl = 32, ny = 2, ion_ppc = [2, 2])),
                ("r32x2-p2x2-f32", (; ppl = 32, ny = 2, T = "f32")), ("r48x3-p2x2", (; ppl = 48, ny = 3)),
                ("r32x2-p2x2-tonglin", (; ppl = 32, ny = 2, law = "tonglin"))]
        for (name_, cfg) in cfgs, (dir, di) in (("duration", 1), ("density", 5)), s in -16:16
            x = collect(TCASE.x0)
            x[di] *= 1 + s * 2.0^-10
            plain!(jobs, jid("rough", name_, dir, s); cfg..., x, series = false, direction = dir, offset = s)
            s in (-16, 0, 16) && plain!(jobs, jid("rough", name_, dir, s, "grad"); cfg..., x, grad = true, direction = dir, offset = s)
        end
    elseif name == "fd"
        # central differences over steps 2⁻ᵏ in duration and density, against the gradient of the grid and ppc campaigns
        cfgs = [("r32x2-p1x1", (; ppl = 32, ny = 2, ppc = [1, 1])), ("r32x2-p4x4", (; ppl = 32, ny = 2, ppc = [4, 4])),
                ("r32x2-p2x2-i2x2", (; ppl = 32, ny = 2, ion_ppc = [2, 2])), ("r48x3-p2x2", (; ppl = 48, ny = 3)), ("r64x4-p2x2", (; ppl = 64, ny = 4)),
                ("r32x2-p2x2-f32", (; ppl = 32, ny = 2, T = "f32"))]
        for (name_, cfg) in cfgs, (dir, di) in (("duration", 1), ("density", 5)), pw in (3, 5, 7, 9, 11, 13, 15, 17), sgn in (1, -1)
            x = collect(TCASE.x0)
            x[di] *= 1 + sgn * 2.0^-pw
            plain!(jobs, jid("fd", name_, dir, pw, sgn > 0 ? "p" : "m"); cfg..., x, series = false, direction = dir, power = pw, sign = sgn)
        end
    elseif name == "frames"
        for (ppl, ny) in ((16, 1), (32, 2), (64, 4))
            plain!(jobs, jid("frames", "r$(ppl)x$(ny)"); ppl, ny, frames = 60)
        end
    else
        error("unknown campaign $name")
    end
    cost(j) = (q = merge(JOB_DEFAULTS, j); q["ppl"]^2 * q["ny"] * prod(q["ppc"]) * (q["grad"] == true ? 4 : 1) / q["cfl"])
    return sort(jobs; by = cost)
end

function run_campaign(name, dir)
    mkpath(dir)
    jobs = campaign_jobs(name)
    gpu = BACKEND_NAME == "cuda" ? CUDACore.name(CUDACore.device()) : nothing
    extra = gpu === nothing ? Dict{String,Any}() : Dict{String,Any}("gpu" => gpu, "cudacore" => string(pkgversion(CUDACore)))
    isfile(joinpath(dir, "manifest.toml")) ||
        run_dir("gridppc-$name", @__FILE__; prefix = "resolution", work_package = "none (study for a report)", dir,
                backend = BACKEND_NAME == "cuda" ? "KernelAbstractions CUDA (CUDACore)" : "KernelAbstractions CPU", extra,
                extra_inputs = [joinpath(@__DIR__, "..", "optim", "trapped2d_case.jl"), joinpath(@__DIR__, "..", "optim", "born2d_case.jl"),
                                joinpath(@__DIR__, "..", "ionization", "atomic.jl")],
                params = Dict{String,Any}("campaign" => name, "jobs" => length(jobs), "defaults" => JOB_DEFAULTS))
    open(io -> JSON.print(io, jobs, 1), joinpath(dir, "jobs.json"), "w")
    todo = filter(j -> !isfile(joinpath(dir, j["id"] * ".json")), jobs)
    println("campaign $name on $BACKEND_NAME: $(length(todo)) of $(length(jobs)) jobs to run, into $dir"); flush(stdout)
    t0 = time()
    for (i, j) in enumerate(todo)
        while isfile(joinpath(dir, "PAUSE")) || isfile(joinpath(dirname(dir), "PAUSE"))
            sleep(10)
        end
        try
            out = run_job(j, dir)
            open(io -> JSON.print(io, out), joinpath(dir, j["id"] * ".json"), "w")
            println("done $(j["id"]) ($i of $(length(todo))): J = $(out["J"]), lost $(out["ionization_lost"]), $(round(out["seconds"]; digits = 1)) s, total $(round(time() - t0; digits = 0)) s")
        catch err
            open(io -> showerror(io, err, catch_backtrace()), joinpath(dir, j["id"] * ".err"), "w")
            println("FAILED $(j["id"]): $(sprint(showerror, err)[1:min(end, 300)])")
        end
        flush(stdout)
        BACKEND_NAME == "cuda" && CUDACore.reclaim()
    end
    println("campaign $name finished in $(round(time() - t0; digits = 0)) s")
    return dir
end

if length(ARGS) >= 3 && ARGS[1] == "campaign"
    run_campaign(ARGS[2], abspath(ARGS[3]))
elseif length(ARGS) >= 3 && ARGS[1] == "job"
    job = JSON.parse(ARGS[3])
    out = run_job(job, abspath(ARGS[2]))
    open(io -> JSON.print(io, out), joinpath(ARGS[2], job["id"] * ".json"), "w")
    println(job["id"], ": J = ", out["J"], " in ", round(out["seconds"]; digits = 1), " s")
elseif length(ARGS) >= 2 && ARGS[1] == "list"
    js = campaign_jobs(ARGS[2])
    println(length(js), " jobs")
    foreach(j -> println(j["id"]), js)
end
