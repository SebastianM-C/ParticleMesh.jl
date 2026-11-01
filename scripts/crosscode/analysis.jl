# Scalar observables of the 1D cases from time series: the damping rate from the envelope of
# the field energy, the growth rate from its exponential phase, and the mode amplitude of a
# field on the periodic grid. Rates are those of the field amplitude (the energy goes as e^{2γt}).

module Observables

using Statistics: mean, median

export linear_fit, local_maxima, damping_from_peaks, growth_from_window, mode_amplitude, checked_fit

"""
    linear_fit(x, y) -> (; slope, intercept, slope_stderr, n)

Ordinary least squares, with the standard error of the slope from the residuals (n > 2).
"""
function linear_fit(x, y)
    n = length(x)
    n >= 2 || error("linear_fit: need at least two points")
    x̄, ȳ = mean(x), mean(y)
    sxx = sum(abs2, x .- x̄)
    slope = sum((x .- x̄) .* (y .- ȳ)) / sxx
    intercept = ȳ - slope * x̄
    res = y .- (intercept .+ slope .* x)
    stderr = n > 2 ? sqrt(sum(abs2, res) / (n - 2) / sxx) : NaN
    return (; slope, intercept, slope_stderr = stderr, n)
end

"Indices i with y[i-1] < y[i] ≥ y[i+1]."
local_maxima(y) = [i for i in 2:length(y)-1 if y[i-1] < y[i] >= y[i+1]]

"""
    damping_from_peaks(t, U; fit_start, floor_factor) -> NamedTuple

Damping rate γ of the field amplitude from the local maxima of the field energy U(t) of a
standing Langmuir wave (two maxima per wave period). The noise floor is the median of U over
the last 20 % of the series; the fit uses the maxima at t ≥ fit_start that lie above
floor_factor × floor, up to the first one that is larger than its predecessor. γ = -slope/2
of log U at the maxima; the frequency is π over the mean spacing of the maxima used. `used`
holds the sample indices of the maxima the fit took (all candidates when it failed).
"""
function damping_from_peaks(t, U; fit_start, floor_factor)
    n = length(U)
    floor = median(U[max(1, n - n ÷ 5):end])
    peaks = [i for i in local_maxima(U) if t[i] >= fit_start]
    used = Int[]
    for i in peaks
        U[i] > floor_factor * floor || break
        !isempty(used) && U[i] > U[used[end]] && break
        push!(used, i)
    end
    length(used) >= 3 || return (; gamma = NaN, gamma_stderr = NaN, omega = NaN,
        floor, t_fit = Float64[], U_fit = Float64[], fit = nothing, used)
    refined = [refine_peak(t, log.(U), i) for i in used]
    tp, lp = first.(refined), last.(refined)
    fit = linear_fit(tp, lp)
    omega = π / mean(diff(tp))
    return (; gamma = -fit.slope / 2, gamma_stderr = fit.slope_stderr / 2, omega, floor,
              t_fit = tp, U_fit = exp.(lp), fit, used)
end

"""
    refine_peak(t, y, i) -> (t_peak, y_peak)

Vertex of the parabola through (t, y) at i-1, i, i+1 (uniform or not), so that the maxima of
a sampled series are not quantized to the output cadence.
"""
function refine_peak(t, y, i)
    t1, t2, t3 = t[i-1], t[i], t[i+1]
    y1, y2, y3 = y[i-1], y[i], y[i+1]
    d = (t1 - t2) * (t1 - t3) * (t2 - t3)
    a = (t3 * (y2 - y1) + t2 * (y1 - y3) + t1 * (y3 - y2)) / d
    b = (t3^2 * (y1 - y2) + t2^2 * (y3 - y1) + t1^2 * (y2 - y3)) / d
    c = (t2 * t3 * (t2 - t3) * y1 + t3 * t1 * (t3 - t1) * y2 + t1 * t2 * (t1 - t2) * y3) / d
    a < 0 || return (t2, y2)
    tv = -b / (2a)
    return (tv, c - b^2 / (4a))
end

"""
    growth_from_window(t, U; window) -> NamedTuple

Growth rate γ of the field amplitude from a least-squares fit of log U over the contiguous
stretch before saturation where window[1] ≤ U / U_sat ≤ window[2]; γ = slope/2. Saturation is
the first local maximum of U that reaches `saturation_fraction` of the global maximum, so that
a later, higher nonlinear bounce does not move the window past the linear phase.
"""
function growth_from_window(t, U; window, saturation_fraction = 0.1)
    Uglobal = maximum(U)
    peaks = [i for i in local_maxima(U) if U[i] >= saturation_fraction * Uglobal]
    imax = isempty(peaks) ? argmax(U) : first(peaks)
    Umax = U[imax]
    lo, hi = window
    inwin = [lo <= U[i] / Umax <= hi for i in 1:imax]
    # the last contiguous run of points inside the window before the maximum
    last_in = findlast(inwin)
    last_in === nothing && return (; gamma = NaN, gamma_stderr = NaN, t_fit = Float64[],
        U_fit = Float64[], fit = nothing, t_max = t[imax], U_max = Umax)
    first_in = last_in
    while first_in > 1 && inwin[first_in-1]
        first_in -= 1
    end
    idx = first_in:last_in
    length(idx) >= 3 || return (; gamma = NaN, gamma_stderr = NaN, t_fit = t[idx],
        U_fit = U[idx], fit = nothing, t_max = t[imax], U_max = Umax)
    fit = linear_fit(t[idx], log.(U[idx]))
    return (; gamma = fit.slope / 2, gamma_stderr = fit.slope_stderr / 2, t_fit = t[idx],
              U_fit = U[idx], fit, t_max = t[imax], U_max = Umax)
end

"""
    mode_amplitude(f, m; periodic_duplicate = true) -> Vector

Amplitude |f̂_m| of Fourier mode m (exp(2πi m j / N) over the N distinct points) of each
column of f, normalized so that f_j = A cos(2π m j / N + φ) gives A. When the last point
repeats the first (Smilei's periodic dumps), it is dropped.
"""
function mode_amplitude(f::AbstractMatrix, m::Integer; periodic_duplicate = true)
    N = size(f, 1) - (periodic_duplicate ? 1 : 0)
    j = 0:N-1
    phase = cis.(-2π * m .* j ./ N)
    return [2 * abs(sum(view(f, 1:N, c) .* phase)) / N for c in axes(f, 2)]
end

"""
    checked_fit(r; min_points = 5) -> (; usable, gamma, gamma_stderr, points, flag)

A rate fit (`damping_from_peaks` or `growth_from_window`) as it may be reported: `usable` when
γ is finite and the fit took at least `min_points` points, and then `gamma` and `gamma_stderr`
are its values; otherwise both are `nothing` and `flag` says why, so that a fit on too few
points (a coarse output cadence crossing the window in a few samples) is not reported as a
number.
"""
function checked_fit(r; min_points = 5)
    n = length(r.t_fit)
    flag = !isfinite(r.gamma) ? "refused: the fit failed ($n points)" :
        n < min_points ? "refused: $n points in the fit window, fewer than $min_points" : ""
    usable = isempty(flag)
    return (; usable, gamma = usable ? r.gamma : nothing,
              gamma_stderr = usable ? r.gamma_stderr : nothing, points = n, flag)
end

end
