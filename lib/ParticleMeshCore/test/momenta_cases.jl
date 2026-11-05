# WP-033: the shared loads, the quadrature references and the digests of momenta.jl (T-530 to
# T-538), and the digest lines that threads_worker.jl repeats with 1 and 4 threads (T-536).
# Needs setup.jl, helpers.jl (CosineDensity), digest.jl and engine2d_cases.jl (ObliqueDensity).

"`n` particles (a multiple of 128) of `dist` from a uniform N = 1 load: 128 per cell, the in-cell index and the cell in the counter."
function momenta_sample(::Type{T}, dist, n; seed = 11, stream = 3, be = CPU(), workgroupsize = nothing,
                        exhausted = nothing) where {T}
    ppc = 128
    grid = PeriodicGrid(n ÷ ppc, basetype(T)(1), 2)
    s = Species(backend, T, n; q = -1, m = 1)
    on(be, load_quiet!, s, grid, CosineDensity(), (one(T), zero(T), zero(T)), ppc, dist; seed, stream, workgroupsize, exhausted)
    return s
end

"The momenta of a species as Float64 triples of their values (a `Dual`'s value)."
momenta64(s) = [Float64.(value_of.((s.u[1][i], s.u[2][i], s.u[3][i]))) for i in 1:alive_count(s)]

"γ − 1 of a momentum, without the cancellation of √(1 + u²) − 1."
kinetic(u) = (u2 = sum(abs2, u); u2 / (1 + sqrt(1 + u2)))

"""
    juttner_quadrature(θ; nodes = 20001) -> (; ε, cdf, mean_ε, mean_u2)

The rest-frame Maxwell–Jüttner distribution by quadrature in Float64, independent of the sampler:
the density of the kinetic energy ε = γ − 1 is ∝ (1 + ε)√(ε(ε + 2)) e^(−ε/θ) (from u² e^(−γ/θ) du
with u = √(ε(ε + 2))), integrated in t with ε = θt², t ∈ [0, √80], by the trapezoid rule (the
integrand in t is an even function times e^(−t²), so the rule converges spectrally). Returns the
node energies, the cumulative distribution there, ⟨ε⟩ and ⟨u²⟩.
"""
function juttner_quadrature(θ; nodes = 20001)
    t = range(0, sqrt(80.0); length = nodes)
    h = step(t)
    ε = [θ * x^2 for x in t]
    f = [(1 + e) * sqrt(e * (e + 2)) * exp(-x^2) * 2θ * x for (e, x) in zip(ε, t)]
    cdf = zeros(nodes)
    for i in 2:nodes
        cdf[i] = cdf[i - 1] + h * (f[i - 1] + f[i]) / 2
    end
    Z = cdf[end]
    w(g) = h * (sum(g.(ε) .* f) - (g(ε[1]) * f[1] + g(ε[end]) * f[end]) / 2) / Z
    return (; ε, cdf = cdf ./ Z, mean_ε = w(identity), mean_u2 = w(e -> e * (e + 2)))
end

"""
    boosted_quadrature(θ, β; nodes = 1201) -> (; upar, γ, uperp2)

The lab-frame moments of a Maxwell–Jüttner plasma drifting with `β` (a 3-vector), by direct 2D
quadrature in Float64 of the lab-frame density per d³u, ∝ exp(−Γ(γ − b·u∥)/θ) (the scalar
distribution function exp(−u·U/θ) with the plasma's four-velocity U = Γ(1, β)), over
u∥ (along β) and u⊥ ≥ 0 with the weight u⊥: ⟨u∥⟩, ⟨γ⟩ and ⟨u⊥²⟩ (both transverse components
together). The box covers rest-frame energies up to 40θ boosted (e^(−40) beyond); trapezoid rule.
"""
function boosted_quadrature(θ, β; nodes = 1201)
    b = sqrt(sum(abs2, β))
    Γ = 1 / sqrt(1 - b^2)
    εmax = 40θ
    U = sqrt(εmax * (εmax + 2))
    γU = 1 + εmax
    lo, hi = Γ * (-U + b * γU), Γ * (U + b * γU)
    lo = min(lo, Γ * (-U + b))                     # the backward edge of the boosted sphere
    up = range(lo, hi; length = nodes)
    ut = range(0, U; length = nodes)
    tw(i) = (i == 1 || i == nodes) ? 0.5 : 1.0
    Z = M1 = M2 = M3 = 0.0
    for (i, a) in enumerate(up), (j, c) in enumerate(ut)
        γ = sqrt(1 + a^2 + c^2)
        w = tw(i) * tw(j) * c * exp(-(Γ * (γ - b * a) - 1) / θ)
        Z += w
        M1 += w * a
        M2 += w * γ
        M3 += w * c^2
    end
    return (; upar = M1 / Z, γ = M2 / Z, uperp2 = M3 / Z)
end

"The one-sample Kolmogorov–Smirnov statistic √N·D of the samples `x` against the cdf tabulated at the sorted `nodes`."
function ks_statistic(x, nodes, cdf)
    xs = sort(x)
    N = length(xs)
    D = 0.0
    for (i, v) in enumerate(xs)
        k = clamp(searchsortedlast(nodes, v), 1, length(nodes) - 1)
        F = cdf[k] + (cdf[k + 1] - cdf[k]) * (v - nodes[k]) / (nodes[k + 1] - nodes[k])
        D = max(D, i / N - F, F - (i - 1) / N)
    end
    return sqrt(N) * D
end

"The two-sample Kolmogorov–Smirnov statistic √(NM/(N + M))·D."
function ks_two_sample(x, y)
    a, b = sort(x), sort(y)
    N, M = length(a), length(b)
    i = j = 0
    D = 0.0
    while i < N && j < M
        v = min(a[i + 1], b[j + 1])
        while i < N && a[i + 1] <= v
            i += 1
        end
        while j < M && b[j + 1] <= v
            j += 1
        end
        D = max(D, abs(i / N - j / M))
    end
    return sqrt(N * M / (N + M)) * D
end

"The standard score of the sample mean of `x` against `μ`."
zscore(x, μ) = (n = length(x); m = sum(x) / n; (m - μ) / sqrt(sum(v -> (v - m)^2, x) / (n - 1) / n))

# The digest set of T-536: a Maxwell–Jüttner load at N = 1 and N = 2 (a drift along an oblique
# direction, θ on both sides of the switch of methods) and a Gaussian-momentum load, per type.
const MOMENTA_CASES = ((0.04, (0.3, -0.5, 0.2)), (2.5, (0.0, 0.0, 0.9)))
const MOMENTA_WG = (nothing, 7, 64)

function momenta_digests(be = CPU(); wgs = MOMENTA_WG)
    lines = String[]
    for T in TYPES, (θ, β) in MOMENTA_CASES, w in wgs
        dist = MaxwellJuttner(T(θ), T.(β))
        s = momenta_sample(T, dist, 1024; be, workgroupsize = w)
        push!(lines, "mj1|$(type_label(T))|$θ|$(w === nothing ? "default" : w)|" *
                     string(fnv_digest(s.u[1], s.u[2], s.u[3], s.cell[1], s.offset[1], s.w, s.status); base = 16))
        g = PeriodicGrid((8, 6), (basetype(T)(1 // 4), basetype(T)(1 // 3)), 2)
        s2 = Species(backend, T, 48 * 6; q = -1, m = 1, N = 2)
        on(be, load_quiet!, s2, g, ObliqueDensity(), oblique_params(T, g), (3, 2), dist; seed = 5, stream = 2, workgroupsize = w)
        push!(lines, "mj2|$(type_label(T))|$θ|$(w === nothing ? "default" : w)|" *
                     string(fnv_digest(s2.u[1], s2.u[2], s2.u[3], s2.cell[1], s2.cell[2], s2.offset[1], s2.offset[2], s2.w);
                            base = 16))
    end
    for T in TYPES, w in wgs
        s = momenta_sample(T, GaussianMomentum(T.((0.2, 0.0, -0.1)), T.((0.3, 0.1, 0.05))), 1024; be, workgroupsize = w)
        push!(lines, "gm1|$(type_label(T))|-|$(w === nothing ? "default" : w)|" *
                     string(fnv_digest(s.u[1], s.u[2], s.u[3]); base = 16))
    end
    return lines
end
