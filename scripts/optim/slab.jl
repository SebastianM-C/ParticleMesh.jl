# WP-051: a wake objective that does not depend on when it is read, and the 1D cold-fluid wake
# it is compared with. Include after wake2d_case.jl.
#
# Behind the pulse the cold plasma oscillates in place: at a fixed lab position x the on-axis E_x
# rings at ω_p with the amplitude the pulse left there. So the objective samples each plasma column
# of a fixed lab-frame slab x ∈ [x1, x2] at a fixed delay s = t − x behind the vacuum light front,
# with a Hann weight over s ∈ [s1, s2], and averages over the slab: J² = 2⟨E_x²⟩/E₀², the
# mean squared wake amplitude over the slab. Once the run is long enough to have sampled every
# column, more steps add nothing, so J is the same at every read-out time by construction. The
# Hann window of 1.5 plasma periods has a spectral zero at 2ω_p, the frequency of E_x²'s ripple,
# so the time average needs no whole number of periods.

using ParticleMeshElectromagnetic: window_shifts

"""
A lab-frame read-out: plasma columns x ∈ [x1, x2] (c/ω₀), each sampled at delays s = t − x ∈ [s1, s2]
with weight sin²(π(s − s1)/(s2 − s1)).
"""
struct SlabWindow
    x1::Float64
    x2::Float64
    s1::Float64
    s2::Float64
end

"""
The windows of the study. `main` is the objective: the plateau from its start at 72 over four
plasma wavelengths, read 72 to 102 behind the light front (the pulse's peak passes at s = 51.6, so
the drive of τ ≤ 9 has fallen below e⁻⁸ at s1 in vacuum, and s2 stays clear of the window's back
band, which reaches s = 108 − band). The others test the read-out: `early` a quarter period
earlier, `short` one plasma period long (no spectral zero at 2ω_p), `half` the first half of the slab.
"""
const WINDOWS = (main = SlabWindow(72, 152, 72, 102), early = SlabWindow(72, 152, 67, 97),
                 short = SlabWindow(72, 152, 72, 92), half = SlabWindow(72, 112, 72, 102))

hann(u) = 0 < u < 1 ? sinpi(u)^2 : 0.0

"The last time any window samples."
last_sample(windows) = maximum(w -> w.x2 + w.s2, windows)

"""
    wake_observe(duration, focus; windows = WINDOWS, box_times = (), kw...) -> (; slab, coverage, box, nsteps, dt, dx)

One run of the case (`kw` go to [`wake_problem`](@ref)), long enough for every window and box
time. `slab` holds, per window, J = √(2·Σw E_x²/Σw)/E₀ over the window's samples; `coverage`, per
window, the smallest and largest Σw of a slab column relative to the ideal (s2 − s1)/(2Δt) (a
column the window's back band or the run's end cut short shows as a ratio below 1; the run refuses
a ratio off by more than 1 %). `box` holds WP-045's objective, the RMS over the whole window past
its back band, at each time in `box_times`, linearly interpolated between the two steps around it
(exact at a step).
"""
function wake_observe(duration, focus; windows = WINDOWS, box_times = (), kw...)
    T = promote_type(typeof(duration), typeof(focus))
    p0 = wake_problem(duration, focus; nsteps = 1, kw...)
    dt = Float64(value_of(p0.dt))
    tend = max(last_sample(windows), isempty(box_times) ? 0.0 : maximum(box_times))
    nsteps = ceil(Int, tend / dt - 1e-9) + 1
    prob = wake_problem(duration, focus; nsteps, kw...)
    st = EMState(prob, EMNumerics(; mode = Reference()))
    initialize!(st, prob)
    p = params(prob)
    g, n = guards(st.grid), ncells(st.grid)
    dx = Float64(value_of(cellsize(st.grid)[1]))
    band = window_back_band(st)
    row = g + 1 + n[2] ÷ 2
    E0 = sqrt(T(CASE.density))
    ks = Dict{Int,Vector{Tuple{Int,Float64}}}()         # step → (box time index, weight)
    for (i, tb) in enumerate(box_times)
        r = tb / dt
        kl = floor(Int, r + 1e-9)
        w = r - kl
        push!(get!(ks, kl, Tuple{Int,Float64}[]), (i, abs(w) < 1e-9 ? 1.0 : 1 - w))
        abs(w) < 1e-9 || push!(get!(ks, kl + 1, Tuple{Int,Float64}[]), (i, w))
    end
    nw = length(windows)
    num = [zero(T) for _ in 1:nw]                         # Σw·E_x² (`rms`)
    nosc = [zero(T) for _ in 1:nw]                        # Σw·(E_x² + (∂ₜE_x/ω_p)²) (`osc`)
    den = zeros(nw)
    cov = [Dict{Int,Float64}() for _ in 1:nw]            # lab column → Σw
    box = [zero(T) for _ in box_times]
    ωp = sqrt(CASE.density)
    tfirst = minimum(w -> w.x1 + w.s1, windows) - 2dt
    prev, shprev = nothing, 0
    for k in 1:nsteps
        step!(st, p, k)
        t = k * dt
        sh = window_shifts(st, k)
        bk = get(ks, k, nothing)
        t >= tfirst || bk !== nothing || continue
        ex = Array(@view st.fields.Ex[(g + band + 1):(g + n[1]), row])
        if bk !== nothing
            j = sqrt(2 * sum(abs2, ex) / length(ex)) / E0
            for (i, w) in bk
                box[i] += w * j
            end
        end
        if prev !== nothing
            tm = t - dt / 2                                     # E_x and its time derivative between the two steps
            for (iw, w) in enumerate(windows)
                tm - w.s2 < w.x2 && tm - w.s1 > w.x1 || continue
                for jj in eachindex(ex)
                    c = band + jj + sh                          # global column of this E_x
                    jp = jj + sh - shprev                       # the same column at the step before
                    1 <= jp <= length(prev) || continue
                    x = (c - 0.5) * dx
                    w.x1 <= x <= w.x2 || continue
                    wt = hann((tm - x - w.s1) / (w.s2 - w.s1))
                    wt > 0 || continue
                    em, ed = (ex[jj] + prev[jp]) / 2, (ex[jj] - prev[jp]) / dt
                    num[iw] += wt * abs2(em)
                    nosc[iw] += wt * (abs2(em) + abs2(ed / ωp))
                    den[iw] += wt
                    cov[iw][c] = get(cov[iw], c, 0.0) + wt
                end
            end
        end
        prev, shprev = ex, sh
    end
    slab = map((a, b) -> sqrt(2 * a / b) / E0, num, den)
    osc = map((a, b) -> sqrt(a / b) / E0, nosc, den)
    coverage = map(enumerate(windows)) do (iw, w)
        ideal = (w.s2 - w.s1) / (2dt)
        cols = [c for c in ceil(Int, w.x1 / dx + 0.5):floor(Int, w.x2 / dx + 0.5)]
        r = [get(cov[iw], c, 0.0) / ideal for c in cols]
        (minimum(r), maximum(r))
    end
    for (iw, c) in enumerate(coverage)
        abs(c[1] - 1) < 0.01 && abs(c[2] - 1) < 0.01 ||
            error("window $(windows[iw]) is not fully sampled: column weight ratios $(c) (back band $band columns, $nsteps steps)")
    end
    names = keys(windows) isa Tuple ? keys(windows) : Tuple(1:nw)
    nt(v) = NamedTuple{Tuple(Symbol.(names))}(Tuple(v))
    return (; slab = nt(slab), osc = nt(osc), coverage, box, nsteps, dt, dx)
end

"""
The resolution at which WP-051 found the optimum converged: Δx = λ₀/32, Δy = 2Δx = λ₀/16
(keywords of [`wake_problem`](@ref)). The optimum moves by an estimated 0.05 in τ and 0.5 in focus
under further refinement; J itself is about 1.7 % below its limit (reports/wp-051-converged-optimization.md).
"""
const CONVERGED = (; ppl = 32, dy_cells = 2)

"The objective of WP-051: J over the `main` window."
slab_amplitude(duration, focus; kw...) = wake_observe(duration, focus; windows = (main = WINDOWS.main,), kw...).slab.main

# --- 1D theory ---------------------------------------------------------------------------------

"""
    wake1d(a0, τ; kp = √n₀, β = √(1 − n₀), t1, t2, du = 0.01) -> (; J, Emax, u, E)

The 1D cold-fluid wake (immobile ions) of a linearly polarized pulse with field envelope
a0·exp(−t²/τ²) in time at the antenna, carried unchanged at the group velocity β = √(1 − n₀) (in
n_c units): in u = β·t − x, the delay behind the pulse's peak in length, the cycle-averaged drive
is ⟨a²⟩ = (a0²/(2β))·exp(−2u²/(βτ)²), the 1/β the WKB gain of a² from vacuum into the plasma (the
energy flux η·E² with refractive index η = β is conserved), and the wake potential obeys the
quasi-static equation for a wake of phase velocity β (γ² = 1/(1 − β²)),
φ'' = k_p²γ²·(β/√(1 − (1 + ⟨a²⟩)/(γ²(1 + φ)²)) − 1), from φ = φ' = 0 ahead of the pulse, with
E_x/E₀ = φ'/k_p. At β = 1 the bracket is its limit (1 + ⟨a²⟩)/(2(1 + φ)²) − 1/2 and the pulse
enters with a0 unchanged. RK4 in u. `J` is the slab objective's definition, √(2⟨E²⟩) under a
Hann weight over times t1 … t2 after the peak, `Emax` the largest |E| behind the pulse. In the
linear limit J = (√(2π)/8)·a0²·k_pτ·exp(−k_p²τ²/8)/β² (derived here), largest at k_pτ = 2 for
every β: the time profile at a fixed x does not depend on β.
"""
function wake1d(a0, τ; kp = sqrt(CASE.density), β = sqrt(1 - CASE.density), t1 = 72 - (CASE.delay - 6 * 2π / 16), t2 = t1 + 30, du = 0.01)
    gain = β < 1 ? 1 / β : 1.0
    a2(u) = a0^2 / 2 * gain * exp(-2 * u^2 / (β * τ)^2)
    γ2 = β < 1 ? 1 / (1 - β^2) : Inf
    rhs(u, φ) = β < 1 ? kp^2 * γ2 * (β / sqrt(1 - (1 + a2(u)) / (γ2 * (1 + φ)^2)) - 1) : kp^2 / 2 * ((1 + a2(u)) / (1 + φ)^2 - 1)
    f(u, y) = (y[2], rhs(u, y[1]))
    u1, u2 = β * t1, β * t2
    u = -6β * τ
    y = (0.0, 0.0)
    us, Es = Float64[], Float64[]
    num = den = 0.0
    Emax = 0.0
    while u < u2
        k1 = f(u, y)
        k2 = f(u + du / 2, y .+ du / 2 .* k1)
        k3 = f(u + du / 2, y .+ du / 2 .* k2)
        k4 = f(u + du, y .+ du .* k3)
        y = y .+ du / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        u += du
        E = y[2] / kp
        u > 2β * τ && (Emax = max(Emax, abs(E)))
        w = hann((u - u1) / (u2 - u1))
        num += w * E^2
        den += w
        push!(us, u)
        push!(Es, E)
    end
    return (; J = sqrt(2 * num / den), Emax, u = us, E = Es)
end

"""
    golden_max(f, lo, hi; tol = 1e-4) -> (x, f(x))

The maximum of a unimodal `f` on [lo, hi] by golden-section search.
"""
function golden_max(f, lo, hi; tol = 1e-4)
    r = (sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - r * (b - a), a + r * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol
        if fc > fd
            b, d, fd = d, c, fc
            c = b - r * (b - a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + r * (b - a)
            fd = f(d)
        end
    end
    x = (a + b) / 2
    return x, f(x)
end

"""
    wake1d_dispersive(a0, τ; x1 = 72, x2 = 152, start = 68, n0 = CASE.density) -> J

The linear 1D wake with the plasma's full dispersion k(ω) = √(ω² − n₀): the pulse of
[`wake1d`](@ref) enters a uniform plasma at `start` (the ramp's middle) with the WKB transmission
η(ω)^(-1/2), spreads and chirps as it goes, and drives (∂ₜ² + ω_p²)E = −ω_p²·∂ₓ⟨a²⟩/2 at each x.
The wake amplitude behind the pulse at depth z = x − start is (derived here, for the envelope's
spectrum Â(ω) = a0·τ·√π·exp(−(ω − 1)²τ²/4))
E/E₀ = (1/8π)·|∫ Â(ω + ω_p)·Â(ω)·(η(ω + ω_p)·η(ω))^(-1/2)·Δk·exp(i·Δk·z) dω|, Δk = k(ω + ω_p) − k(ω),
and J is its RMS over the slab x1 … x2, as the slab objective averages. Without dispersion
(k linear in ω) it is the linear limit of [`wake1d`](@ref) at every z.
"""
function wake1d_dispersive(a0, τ; x1 = 72.0, x2 = 152.0, start = 68.0, n0 = CASE.density, nz = 41, dω = 0.002)
    ωp = sqrt(n0)
    k(ω) = sqrt(ω^2 - n0)
    Â(ω) = a0 * τ * sqrt(π) * exp(-(ω - 1)^2 * τ^2 / 4)
    ωs = (max(ωp + 1e-6, 1 - ωp - 12 / τ)):dω:(1 + 12 / τ)
    zs = range(x1 - start, x2 - start; length = nz)
    terms = [(Â(ω + ωp) * Â(ω) * sqrt(k(ω + ωp) / (ω + ωp) * k(ω) / ω)^(-1) * (k(ω + ωp) - k(ω)), k(ω + ωp) - k(ω)) for ω in ωs]
    A(z) = abs(sum(c * cis(dk * z) for (c, dk) in terms) * dω) / (8π)
    return sqrt(sum(z -> A(z)^2, zs) / nz)
end
