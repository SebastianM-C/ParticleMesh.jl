# Shared by check.jl, bench2d.jl and bench3d.jl: synthetic states on N axes, host-side references
# (ρ, the fold, the continuity residual) and the timer.
include(joinpath(@__DIR__, "proto.jl"))
using Random
using ForwardDiff: ForwardDiff, Dual

"The out-of-plane quantity of the 2D checks: q·w·u_z."
struct MomentumZ end
@inline (::MomentumZ)(s, i) = @inbounds s.q * s.w[i] * s.u[3][i]

lift(::Type{T}, x, rng) where {T<:AbstractFloat} = T(x)
lift(::Type{Dual{G,V,P}}, x, rng) where {G,V,P} = Dual{G}(V(x), ntuple(_ -> V(randn(rng)) / 8, Val(P))...)
base_type(::Type{T}) where {T<:AbstractFloat} = T
base_type(::Type{Dual{G,V,P}}) where {G,V,P} = V

mag(x::Real) = abs(x)
mag(x::Dual) = max(abs(ForwardDiff.value(x)), maximum(abs, ForwardDiff.partials(x); init = zero(ForwardDiff.valtype(x))))
maxmag(a) = maximum(mag, a; init = 0.0)

old_keys(N) = (ntuple(d -> Symbol(:old_cell_, d), N), ntuple(d -> Symbol(:old_offset_, d), N))
flux_extras_nd(::Type{T}, N) where {T} = N == 2 ? PMC.flux_extras(T, 2) : flux_extras_3d(T)

"""
A species of `ppc` slots per cell in cell order (so it is sorted by cell), a fraction `fill` of
them alive at random, with random offsets, momenta and weights; the old position is the current
one.
"""
function synthetic_species(backend, ::Type{T}, grid::PMC.PeriodicGrid{N}; ppc, fill = 1.0, seed = 1) where {T,N}
    rng = Xoshiro(seed)
    V = base_type(T)
    ncell = prod(grid.n)
    cap = ncell * ppc
    s = PMC.Species(backend, T, cap; q = -1, m = 1, N, extras = flux_extras_nd(T, N))
    cell = ntuple(_ -> Vector{Int32}(undef, cap), N)
    for L in 1:ncell, k in 1:ppc
        c = PMC.cartesian_cell(L, grid.n)
        foreach(d -> cell[d][(L - 1) * ppc + k] = c[d], 1:N)
    end
    offset = ntuple(_ -> [lift(T, rand(rng, V), rng) for _ in 1:cap], N)
    status = [rand(rng) < fill ? PMC.ALIVE : PMC.FREE for _ in 1:cap]
    kc, ko = old_keys(N)
    foreach((a, b) -> copyto!(a, b), s.cell, cell)
    foreach((a, b) -> copyto!(a, b), s.offset, offset)
    foreach(a -> copyto!(a, [lift(T, randn(rng, V), rng) for _ in 1:cap]), s.u)
    copyto!(s.w, [lift(T, (1 + rand(rng, V)) / ppc, rng) for _ in 1:cap])
    copyto!(s.status, status)
    foreach((k, b) -> copyto!(s.extras[k], b), kc, cell)
    foreach((k, b) -> copyto!(s.extras[k], b), ko, offset)
    return s
end

"Move every alive particle by a random displacement below `dmax` cells per axis (host side); the old position goes to the extras. A fraction `kill` dies."
function move!(s::PMC.Species{N,T}, grid; dmax = 0.5, kill = 0.0, seed = 2) where {N,T}
    rng = Xoshiro(seed)
    V = base_type(T)
    n = Int32.(grid.n)
    cell, offset, status = map(Array, s.cell), map(Array, s.offset), Array(s.status)
    new_cell, new_offset = map(copy, cell), map(copy, offset)
    for i in eachindex(status)
        status[i] == PMC.ALIVE || continue
        for d in 1:N
            Δ = T(V(dmax) * (2 * rand(rng, V) - 1))
            new_cell[d][i], new_offset[d][i] = PMC.advance(cell[d][i], offset[d][i], Δ, n[d])
        end
        rand(rng) < kill && (status[i] = PMC.FREE)
    end
    kc, ko = old_keys(N)
    foreach((k, b) -> copyto!(s.extras[k], b), kc, cell)
    foreach((k, b) -> copyto!(s.extras[k], b), ko, offset)
    foreach((a, b) -> copyto!(a, b), s.cell, new_cell)
    foreach((a, b) -> copyto!(a, b), s.offset, new_offset)
    copyto!(s.status, status)
    return s
end

"A copy of a species on its backend."
function copy_species(s::PMC.Species{N,T}) where {N,T}
    c = PMC.Species(get_backend(s), T, PMC.capacity(s); q = s.q, m = s.m, N, extras = map(eltype, s.extras))
    restore!(c, s)
    return c
end
function restore!(dst, src)
    foreach(copyto!, species_arrays(dst), species_arrays(src))
    copyto!(dst.count, src.count)
    return dst
end

"""
The species moved by half a cell per axis (new and old position), on the host: its cells are
the nearest nodes of the original, and a deposition at `Center()` on it is the deposition at
`Node()` of the original, binned by nearest node.
"""
function half_cell_shifted(s::PMC.Species{N,T}, grid) where {N,T}
    c = copy_species(s)
    h = one(T) / 2
    kc, ko = old_keys(N)
    for (cells, offsets) in ((c.cell, c.offset), (map(k -> c.extras[k], kc), map(k -> c.extras[k], ko)))
        for d in 1:N
            cell, δ = Array(cells[d]), Array(offsets[d])
            for i in eachindex(cell)
                y = δ[i] + h
                up = y >= one(T)
                δ[i] = up ? y - one(T) : y
                cell[i] = mod1(cell[i] + Int32(up), Int32(grid.n[d]))
            end
            copyto!(cells[d], cell)
            copyto!(offsets[d], δ)
        end
    end
    return c
end

"The fields with their guards folded into the interior, on the host."
folded_host(J, grid) = map(a -> fold_host!(Array(a), grid), J)

# --- host references -----------------------------------------------------------------------------

"ρ of the alive particles at the current (`old = false`) or the old position, with guards, on the host."
function rho_host(s::PMC.Species{N,T}, shape, st, grid; old = false) where {N,T}
    kc, ko = old_keys(N)
    cell = old ? map(k -> Array(s.extras[k]), kc) : map(Array, s.cell)
    offset = old ? map(k -> Array(s.extras[k]), ko) : map(Array, s.offset)
    status, w = Array(s.status), Array(s.w)
    ρ = zeros(T, (grid.n .+ 2grid.g)...)
    for i in eachindex(w)
        status[i] == PMC.ALIVE || continue
        fw = ntuple(d -> PMC.shape_weights(shape, st[d], offset[d][i]), N)
        for K in CartesianIndices(ntuple(d -> 1:length(fw[d][2]), N))
            idx = ntuple(d -> cell[d][i] + fw[d][1] + grid.g + K[d] - 1, N)
            ρ[idx...] += s.q * w[i] * prod(ntuple(d -> fw[d][2][K[d]], N))
        end
    end
    return ρ
end

"Fold the guards of a host array into the interior, axis by axis (the engine's `fold_guards!` is for matrices)."
function fold_host!(a::Array{T,N}, grid) where {T,N}
    g, n = grid.g, grid.n
    for d in 1:N
        selectdim(a, d, (n[d] + 1):(n[d] + g)) .+= selectdim(a, d, 1:g)
        selectdim(a, d, (g + 1):(2g)) .+= selectdim(a, d, (n[d] + g + 1):(n[d] + 2g))
        selectdim(a, d, 1:g) .= zero(T)
        selectdim(a, d, (n[d] + g + 1):(n[d] + 2g)) .= zero(T)
    end
    return a
end

"The largest residual of discrete continuity on the interior nodes, relative to the largest |ρ_new - ρ_old|/dt."
function continuity_residual(J, s::PMC.Species{N}, shape, st, grid, dt) where {N}
    ρ0 = fold_host!(rho_host(s, shape, st, grid; old = true), grid)
    ρ1 = fold_host!(rho_host(s, shape, st, grid), grid)
    inner = ntuple(d -> (grid.g + 1):(grid.g + grid.n[d]), N)
    r = (ρ1[inner...] .- ρ0[inner...]) ./ dt
    scale = maxmag(r)
    for d in 1:N
        Jd = fold_host!(Array(J[d]), grid)[inner...]
        r .+= (Jd .- circshift(Jd, ntuple(a -> a == d ? 1 : 0, N))) ./ grid.dx[d]
    end
    return maxmag(r) / scale
end

"The largest difference of two sets of fields, relative to the largest entry of the second."
field_difference(a, b) = maximum(map((x, y) -> maxmag(Array(x) .- Array(y)), a, b)) / maximum(map(y -> maxmag(Array(y)), b))

zero_fields(backend, ::Type{T}, grid::PMC.PeriodicGrid{N}) where {T,N} =
    ntuple(_ -> KernelAbstractions.zeros(backend, T, (grid.n .+ 2grid.g)...), N == 2 ? 3 : N)
zero!(J) = foreach(a -> fill!(a, zero(eltype(a))), J)

# --- timing --------------------------------------------------------------------------------------

"Seconds per call of `f()` on the device: the least of `reps` calls after one warm-up call; `setup()` runs before each call, untimed."
function least(f, backend; reps = 9, setup = () -> nothing)
    setup(); f(); KernelAbstractions.synchronize(backend)
    return minimum(1:reps) do _
        setup(); KernelAbstractions.synchronize(backend)
        t0 = time_ns()
        f(); KernelAbstractions.synchronize(backend)
        (time_ns() - t0) / 1e9
    end
end
