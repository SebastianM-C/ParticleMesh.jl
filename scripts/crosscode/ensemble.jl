# Seed ensembles of a noise-limited case (WP-012): the complex mode coefficient of a field, the
# damping rate of the ensemble mean, per-seed rates at the ensemble's maxima, the jackknife over
# seeds, and the aggregation of per-seed rates that refuses a failed seed instead of dropping it.
# Include analysis.jl (module Observables) before this file.

module Ensemble

using Statistics: mean, median, std
using ..Observables: damping_from_peaks, linear_fit, refine_peak

export mode_coefficient, ensemble_energy, ensemble_damping, fit_at_peaks, aggregate_seeds,
       SeedRefused, cross_seed_correlation, seed_scatter

"""
    mode_coefficient(f, m; periodic_duplicate = true) -> Vector{ComplexF64}

Complex coefficient (2/N) Σ_j f_j exp(-2πi m j / N) of Fourier mode m of each column of f, so
that f_j = A cos(2π m j / N + φ) gives A e^{iφ}; `abs` of it is `mode_amplitude`. When the last
point repeats the first (Smilei's periodic dumps), it is dropped.
"""
function mode_coefficient(f::AbstractMatrix, m::Integer; periodic_duplicate = true)
    N = size(f, 1) - (periodic_duplicate ? 1 : 0)
    phase = cis.(-2π * m .* (0:N-1) ./ N)
    return [2 * sum(view(f, 1:N, c) .* phase) / N for c in axes(f, 2)]
end

"""
    ensemble_energy(coeffs) -> Vector

|⟨Ê⟩|² of the seed mean of the complex mode coefficients `coeffs` (one vector per seed, on a
common time grid). The coherent part keeps its amplitude; independent noise of the seeds adds in
energy, so its contribution falls as 1/M.
"""
ensemble_energy(coeffs) = abs2.(reduce(+, coeffs) ./ length(coeffs))

"""
    fit_at_peaks(t, U, used) -> NamedTuple

Damping rate of one series U at given sample indices `used` (the maxima the ensemble fit took):
each point is refined by `refine_peak` on log U, then γ = -slope/2 of a least-squares line. The
fit is `ok` when there are at least three points and every U[i-1:i+1] is positive and finite;
otherwise `reason` says why and γ is NaN.
"""
function fit_at_peaks(t, U, used)
    fail(reason) = (; gamma = NaN, gamma_stderr = NaN, n = length(used), ok = false, reason)
    length(used) >= 3 || return fail("fewer than three maxima in the ensemble fit")
    all(i -> 1 < i < length(U) && all(u -> isfinite(u) && u > 0, U[i-1:i+1]), used) ||
        return fail("non-positive or non-finite energy at a fit point")
    ly = log.(U)
    refined = [refine_peak(t, ly, i) for i in used]
    fit = linear_fit(first.(refined), last.(refined))
    return (; gamma = -fit.slope / 2, gamma_stderr = fit.slope_stderr / 2, n = length(used),
              ok = isfinite(fit.slope), reason = isfinite(fit.slope) ? "" : "non-finite slope")
end

"""
    ensemble_damping(t, coeffs; fit_start, floor_factor) -> NamedTuple

The pre-registered ensemble observable: `damping_from_peaks` on `ensemble_energy(coeffs)` with
the case's `fit_start` and `floor_factor` (floor: median of the last 20 % of the ensemble
energy). Also returned:

- `per_seed`: `fit_at_peaks` of each seed's |Ê_s|² at the maxima the ensemble fit used;
- `single_seed`: `damping_from_peaks` of each seed's |Ê_s|² on its own (the single-run rule);
- `jackknife`: the ensemble fit with each seed left out, and the jackknife standard error
  √((M-1)/M Σ (γ₋ₛ - γ̄₋)²), NaN when M < 3 or any leave-one-out fit fails;
- `M`, the floor, the fit points, and `ok` (the ensemble fit took at least three maxima).
"""
function ensemble_damping(t, coeffs; fit_start, floor_factor)
    M = length(coeffs)
    M >= 1 || error("ensemble_damping: no seeds")
    U = ensemble_energy(coeffs)
    r = damping_from_peaks(t, U; fit_start, floor_factor)
    ok = isfinite(r.gamma)
    seedU = [abs2.(c) for c in coeffs]
    per_seed = ok ? [fit_at_peaks(t, u, r.used) for u in seedU] :
        [(; gamma = NaN, gamma_stderr = NaN, n = 0, ok = false, reason = "the ensemble fit failed") for _ in 1:M]
    single_seed = [damping_from_peaks(t, u; fit_start, floor_factor) for u in seedU]
    loo = M >= 3 ? [damping_from_peaks(t, ensemble_energy(coeffs[setdiff(1:M, s)]); fit_start, floor_factor).gamma
                    for s in 1:M] : fill(NaN, M)
    jk = all(isfinite, loo) && M >= 3 ? sqrt((M - 1) / M * sum(abs2, loo .- mean(loo))) : NaN
    return (; gamma = r.gamma, gamma_stderr = r.gamma_stderr, omega = r.omega, floor = r.floor,
              t_fit = r.t_fit, U_fit = r.U_fit, fit = r.fit, used = r.used, ok, M, U,
              per_seed, single_seed, jackknife = (; gamma = loo, stderr = jk))
end

"Raised by `aggregate_seeds` when a seed's fit failed its criteria; names every failed seed and why."
struct SeedRefused <: Exception
    seeds::Vector{Int}
    reasons::Vector{String}
end
Base.showerror(io::IO, e::SeedRefused) =
    print(io, "SeedRefused: ", join(["seed $s ($r)" for (s, r) in zip(e.seeds, e.reasons)], ", "))

"""
    aggregate_seeds(seeds, fits) -> (; mean, std, stderr, M, seeds, gammas)

Mean, sample standard deviation and standard error of the mean (std/√M) of the per-seed rates
`fits[i].gamma`. A fit counts only when `fits[i].ok` is true (for `damping_from_peaks` results,
which carry no `ok`, when γ is finite); if any seed fails, the aggregation throws `SeedRefused`
listing all failed seeds instead of averaging the others.
"""
function aggregate_seeds(seeds, fits)
    length(seeds) == length(fits) || error("aggregate_seeds: one fit per seed")
    isok(f) = hasproperty(f, :ok) ? f.ok : isfinite(f.gamma)
    why(f) = hasproperty(f, :reason) ? f.reason :
        "fewer than three maxima above the floor after fit_start ($(length(f.t_fit)) used)"
    bad = findall(!isok, fits)
    isempty(bad) || throw(SeedRefused(collect(Int, seeds[bad]), [why(fits[i]) for i in bad]))
    g = [f.gamma for f in fits]
    M = length(g)
    s = M > 1 ? std(g) : NaN
    return (; mean = mean(g), std = s, stderr = s / sqrt(M), M, seeds = collect(seeds), gammas = g)
end

"""
    cross_seed_correlation(coeffs, window) -> NamedTuple

Normalized complex correlation |⟨a_s, a_r⟩| / (‖a_s‖ ‖a_r‖) of the mode coefficients of every
pair of seeds over the sample indices `window` (late times, where the noise dominates), as the
check that the seeds' noise is independent: its maximum and mean over the pairs.
"""
function cross_seed_correlation(coeffs, window)
    M = length(coeffs)
    c = Float64[]
    for s in 1:M, r in s+1:M
        a, b = coeffs[s][window], coeffs[r][window]
        push!(c, abs(sum(conj.(a) .* b)) / sqrt(sum(abs2, a) * sum(abs2, b)))
    end
    return (; max = isempty(c) ? NaN : maximum(c), mean = isempty(c) ? NaN : mean(c), pairs = length(c))
end

"""
    seed_scatter(coeffs, window) -> NamedTuple

The noise measured from the seeds' scatter about their mean, which the coherent signal does not
enter: `noise_of_mean`, the mean over the sample indices `window` of Σₛ |Ê_s - ⟨Ê⟩|² / (M(M-1))
(the variance of the seed mean), and `noise_per_seed` = M × that. As an independence check
that the signal does not enter either, `difference_correlation`: the normalized correlations
|⟨d_a, d_b⟩| / (‖d_a‖ ‖d_b‖) of the differences d = Ê_s - Ê_r of disjoint seed pairs (1, 2),
(3, 4), ..., over `window`; near 0 for independent noise, 1 for a shared one.
"""
function seed_scatter(coeffs, window)
    M = length(coeffs)
    M >= 2 || error("seed_scatter: need two seeds")
    m = reduce(+, coeffs) ./ M
    v = [sum(abs2(c[i] - m[i]) for c in coeffs) / (M * (M - 1)) for i in window]
    d = [coeffs[2k-1][window] .- coeffs[2k][window] for k in 1:M÷2]
    corr = Float64[]
    for a in 1:length(d), b in a+1:length(d)
        push!(corr, abs(sum(conj.(d[a]) .* d[b])) / sqrt(sum(abs2, d[a]) * sum(abs2, d[b])))
    end
    return (; noise_of_mean = mean(v), noise_per_seed = M * mean(v),
              difference_correlation = (; max = isempty(corr) ? NaN : maximum(corr),
                                          mean = isempty(corr) ? NaN : mean(corr), pairs = length(corr)))
end

end
