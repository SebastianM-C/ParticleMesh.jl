# WP-067 and WP-076: the states and sampled modes of the compensated gather's tests, gather.jl's
# (N = 1 and 2) and em3d_gather.jl's (N = 3).

using ParticleMeshElectromagnetic: fields_at, kick_kernel!, FIELD_LAYOUT

const GATHER_NAMES = (:Ex, :Ey, :Ez, :Bx, :By, :Bz)

"""
The mean of a B-spline gather of order `p` (0, 1 or 2) over `M` evenly spaced offsets per cell (M even), of a
sampled mode of wavenumber `k` (x = kΔx/2): the shape's transfer at k and at its aliases
k + 2πlM/Δx, which those offsets cannot tell from k, Σ_l (sin x/(x + πlM))^(p+1), in closed form
with Σ_l (z + πl)^−2 = csc²z and Σ_l (z + πl)^−3 = csc²z·cot z (z = x/M). In BigFloat.
"""
function gather_lattice_mean(p, x, M)
    x, z = big(x), big(x) / M
    p == 0 && return sin(x) / M / sin(z)      # the nearest grid point (WP-070): |(1/M)Σ_m e^(2ix·m/M)|
    p == 1 && return sin(x)^2 / M^2 / sin(z)^2
    p == 2 && return sin(x)^3 / M^3 * cos(z) / sin(z)^3
    error("order $p")
end

"""
A state of `N` = 1, 2 or 3 resolved axes, 12 cells per laser wavelength along x over two
wavelengths (24 cells; at N = 2 six rows of 0.7, at N = 3 six rows of 0.7 by six layers of 0.6),
a cold species that the test does not push, the `gather`.
"""
function gather_state(::Type{T}, N, shape, gather; be = CPU()) where {T}
    h = 2 * T(π) / 12
    if N == 1
        sp = (SpeciesDescription{T}(:e, -one(T), one(T), one(T), zero(T), 1, 4, zero3(T), zero3(T), 1),)
        prob = EMProblem("gather-1d", 24h, 24, 2h / 5, 4, sp, 1, 1)
    elseif N == 2
        sp = (SpeciesDescription{T}(:e, -one(T), one(T), one(T), zero(T), (1, 0), (2, 2), zero3(T), zero3(T), 1),)
        prob = EMProblem("gather-2d", (24h, 6 * T(7) / 10), (24, 6), 2h / 5, 4, sp, 1, (1, 0))
    else
        sp = (SpeciesDescription{T}(:e, -one(T), one(T), one(T), zero(T), (1, 0, 0), (1, 1, 1), zero3(T), zero3(T), 1),)
        prob = EMProblem("gather-3d", (24h, 6 * T(7) / 10, 6 * T(6) / 10), (24, 6, 6), 2h / 5, 4, sp, 1, (1, 0, 0))
    end
    return EMState(prob, EMNumerics(; shape, gather); backend = be)
end

"The mode cos(k·x_i + φ) along axis 1 of a field of `grid`, sampled at the interior array index i (x_i = (i − 1)Δx), each partial of a dual its own phase; guards the periodic images."
mode_field!(a, grid, k, φ, ::Type{T}) where {T} = oblique_field!(a, grid, (k,), φ, T)

"""
The plane wave cos(Σ_d k_d·x_d + φ) on a field of `grid`, sampled at the interior array indices
(x_d = (i_d − 1)Δx_d), the wavenumbers `ks` along the first `length(ks)` axes (none along the
rest), each partial of a dual its own phase; guards the periodic images (so each k_d times the
box's length along d a multiple of 2π beyond axis 1).
"""
function oblique_field!(a, grid, ks, φ, ::Type{T}) where {T}
    g, n = guards(grid), ncells(grid)
    hs = map(h -> Float64(value_of(h)), cellsize(grid))
    A = zeros(T, size(a))
    for I in CartesianIndices(A)
        all(d -> g < I[d] <= g + n[d], 1:ndims(A)) || continue
        θ = sum(d -> ks[d] * (I[d] - g - 1) * hs[d], eachindex(ks)) + φ
        A[I] = T <: Dual ? mkdual(cos(θ), sin(θ), cos(θ + 1) / 2, -sin(θ + 2)) : T(cos(θ))
    end
    copyto!(a, A)
    fill_guards!(a, grid)
    return a
end
