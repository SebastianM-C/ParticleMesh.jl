"The values of `load_quiet!`'s `density_at`; anything else is an `ArgumentError` (a host keyword, §0.4)."
check_density_at(a::Symbol) = a in (:particle, :cell_centre) ? a :
    throw(ArgumentError("density_at must be :particle or :cell_centre, got :$a"))

# The kernel argument of a checked `density_at` (a small Union, so the launch is union-split, not dispatched).
density_val(a::Symbol) = a === :cell_centre ? Val(:cell_centre) : Val(:particle)

# The in-cell offset at which a particle's density is evaluated: its own, or the cell centre.
@inline density_offset(::Val{:particle}, δ) = δ
@inline density_offset(::Val{:cell_centre}, δ) = one(δ) / 2

@kernel function load_quiet_kernel!(s, density, p, dx, ppc::Int32, first::Int32, vdrift, vth, key, at)
    i = @index(Global, Linear)
    T = eltype(s.w)
    c = Int32((i - 1) ÷ ppc + 1)
    k = Int32((i - 1) % ppc + 1)
    slot = first - Int32(1) + Int32(i)
    δ = (T(k) - one(T) / 2) / T(ppc)
    x = (T(c - Int32(1)) + density_offset(at, δ)) * dx
    # counter: (global cell, 0, 0, in-cell index k in the high 16 bits | draw index in the low 16 bits); `first` is not part of it
    ctr = (UInt32(c), UInt32(0), UInt32(0), UInt32(k) << 16)
    r = philox4x32(ctr, key)
    z1, z2 = box_muller(T, r[1], r[2])
    z3, _ = box_muller(T, r[3], r[4])
    @inbounds begin
        s.cell[1][slot] = c
        s.offset[1][slot] = δ
        s.u[1][slot] = vdrift[1] + vth[1] * z1
        s.u[2][slot] = vdrift[2] + vth[2] * z2
        s.u[3][slot] = vdrift[3] + vth[3] * z3
        s.w[slot] = density(x, p) / T(ppc)
        s.status[slot] = ALIVE
    end
end

"""
    load_quiet!(species, grid, density, p, ppc; vdrift, vth, seed, stream, first = 1, density_at = :particle, workgroupsize = nothing)

Quiet-start loading: `ppc` particles per cell on the lattice `δₖ = (k - 1/2)/ppc`, written into
slots `first:first-1+n·ppc`. The weight of a particle at `x` is `density(x, p)/ppc` (A3: the
parameters `p` enter through the weights only; `density` is an isbits callable). With
`density_at = :cell_centre` the density is evaluated at the centre of the particle's cell
instead, `density((c - 1/2)·dx, p)/ppc`, so the loaded profile is piecewise constant per cell, as
Smilei and PIConGPU sample it; positions and draws are the same either way, and the default
`:particle` keeps the weights bit for bit. Anything else is an `ArgumentError`. Each velocity
component is `vdrift[d] + vth[d]·z` with `z` standard normal from Box–Muller over Philox4x32-10
draws, keyed by `(seed, stream)` with the counter `(cell, 0, 0, k << 16)` (A4), so a seed means
the same thing on every backend and thread count. Sets the alive count to the last loaded slot.
"""
function load_quiet!(s::Species{1,T}, grid::PeriodicGrid{1}, density, p, ppc::Integer;
                     vdrift = (zero(T), zero(T), zero(T)), vth = (zero(T), zero(T), zero(T)),
                     seed::Integer, stream::Integer, first::Integer = 1, density_at::Symbol = :particle,
                     workgroupsize = nothing) where {T}
    check_density_at(density_at)
    n = ncells(grid)
    0 < ppc < 2^16 || throw(ArgumentError("ppc must be in 1:65535 for the counter layout"))
    last = first - 1 + n * ppc
    last <= capacity(s) || throw(ArgumentError("capacity $(capacity(s)) < $last particles"))
    key = (UInt32(seed), UInt32(stream))
    launcher(load_quiet_kernel!, get_backend(s), workgroupsize)(s, density, p, cellsize(grid), Int32(ppc), Int32(first),
                                                                T.(vdrift), T.(vth), key, density_val(density_at); ndrange = n * ppc)
    fill!(s.count, Int32(last))
    return s
end

# N = 2: the per-axis product lattice. Particle `i` is the in-cell index `k` (axis 1 fastest)
# of linear cell `L`, so the loaded slots are already in sorted order.
@kernel function load_quiet_2d_kernel!(s, density, p, dx, n1::Int32, ppc::NTuple{2,Int32}, first::Int32, vdrift, vth, key, at)
    i = @index(Global, Linear)
    T = eltype(s.w)
    K = ppc[1] * ppc[2]
    L = Int32((i - 1) ÷ K + 1)
    k = Int32((i - 1) % K + 1)
    c1 = (L - Int32(1)) % n1 + Int32(1)
    c2 = (L - Int32(1)) ÷ n1 + Int32(1)
    k1 = (k - Int32(1)) % ppc[1] + Int32(1)
    k2 = (k - Int32(1)) ÷ ppc[1] + Int32(1)
    slot = first - Int32(1) + Int32(i)
    δ1 = (T(k1) - one(T) / 2) / T(ppc[1])
    δ2 = (T(k2) - one(T) / 2) / T(ppc[2])
    x = ((T(c1 - Int32(1)) + density_offset(at, δ1)) * dx[1], (T(c2 - Int32(1)) + density_offset(at, δ2)) * dx[2])
    # counter: (cell₁, cell₂, 0, in-cell index k in the high 16 bits | draw index in the low 16 bits)
    ctr = (UInt32(c1), UInt32(c2), UInt32(0), UInt32(k) << 16)
    r = philox4x32(ctr, key)
    z1, z2 = box_muller(T, r[1], r[2])
    z3, _ = box_muller(T, r[3], r[4])
    @inbounds begin
        s.cell[1][slot] = c1
        s.cell[2][slot] = c2
        s.offset[1][slot] = δ1
        s.offset[2][slot] = δ2
        s.u[1][slot] = vdrift[1] + vth[1] * z1
        s.u[2][slot] = vdrift[2] + vth[2] * z2
        s.u[3][slot] = vdrift[3] + vth[3] * z3
        s.w[slot] = density(x, p) / T(K)
        s.status[slot] = ALIVE
    end
end

"""
    load_quiet!(species::Species{2}, grid::PeriodicGrid{2}, density, p, ppc::NTuple{2}; vdrift, vth, seed, stream, first = 1, density_at = :particle, workgroupsize = nothing)

The 2D quiet start: `prod(ppc)` particles per cell on the per-axis product lattice
`δ_d = (k_d - 1/2)/ppc_d`, written cell by cell in column-major linear order (axis 1 fastest)
and within a cell by the in-cell index `k = k₁ + ppc₁(k₂ - 1)`, also axis 1 fastest
(`prod(ppc) < 2^16`). The weight is `density(x, p)/prod(ppc)` with `x` an `NTuple{2}` position, so
the parameters enter through the weights only (A3); the Philox counter is
`(cell₁, cell₂, 0, k << 16)` (A4). `density_at = :cell_centre` evaluates the density at the cell
centre `((c_d - 1/2)·dx_d)` instead, as in 1D. Sets the alive count to the last loaded slot.
"""
function load_quiet!(s::Species{2,T}, grid::PeriodicGrid{2}, density, p, ppc::NTuple{2,Integer};
                     vdrift = (zero(T), zero(T), zero(T)), vth = (zero(T), zero(T), zero(T)),
                     seed::Integer, stream::Integer, first::Integer = 1, density_at::Symbol = :particle,
                     workgroupsize = nothing) where {T}
    check_density_at(density_at)
    n = ncells(grid)
    all(>(0), ppc) && prod(ppc) < 2^16 || throw(ArgumentError("ppc = $ppc: each ≥ 1 and prod(ppc) < 2^16 for the counter layout"))
    total = prod(n) * prod(ppc)
    last = first - 1 + total
    last <= capacity(s) || throw(ArgumentError("capacity $(capacity(s)) < $last particles"))
    key = (UInt32(seed), UInt32(stream))
    launcher(load_quiet_2d_kernel!, get_backend(s), workgroupsize)(s, density, p, cellsize(grid), Int32(n[1]), Int32.(ppc),
                                                                   Int32(first), T.(vdrift), T.(vth), key, density_val(density_at);
                                                                   ndrange = total)
    fill!(s.count, Int32(last))
    return s
end
