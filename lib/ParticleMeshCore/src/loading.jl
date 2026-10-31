@kernel function load_quiet_kernel!(s, density, p, dx, ppc::Int32, first::Int32, vdrift, vth, key)
    i = @index(Global, Linear)
    T = eltype(s.w)
    c = Int32((i - 1) ÷ ppc + 1)
    k = Int32((i - 1) % ppc + 1)
    slot = first - Int32(1) + Int32(i)
    δ = (T(k) - one(T) / 2) / T(ppc)
    x = (T(c - Int32(1)) + δ) * dx
    # counter: (global cell, 0, 0, slot in the high 16 bits | draw index in the low 16 bits)
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
    load_quiet!(species, grid, density, p, ppc; vdrift, vth, seed, stream, first = 1)

Quiet-start loading: `ppc` particles per cell on the lattice `δₖ = (k - 1/2)/ppc`, written into
slots `first:first-1+n·ppc`. The weight of a particle at `x` is `density(x, p)/ppc` (A3: the
parameters `p` enter through the weights only; `density` is an isbits callable). Each velocity
component is `vdrift[d] + vth[d]·z` with `z` standard normal from Box–Muller over Philox4x32-10
draws, keyed by `(seed, stream)` with the counter `(cell, 0, 0, k << 16)` (A4), so a seed means
the same thing on every backend and thread count. Sets the alive count to the last loaded slot.
"""
function load_quiet!(s::Species{1,T}, grid::PeriodicGrid{1}, density, p, ppc::Integer;
                     vdrift = (zero(T), zero(T), zero(T)), vth = (zero(T), zero(T), zero(T)),
                     seed::Integer, stream::Integer, first::Integer = 1) where {T}
    n = ncells(grid)
    0 < ppc < 2^16 || throw(ArgumentError("ppc must be in 1:65535 for the counter layout"))
    last = first - 1 + n * ppc
    last <= capacity(s) || throw(ArgumentError("capacity $(capacity(s)) < $last particles"))
    key = (UInt32(seed), UInt32(stream))
    load_quiet_kernel!(get_backend(s))(s, density, p, cellsize(grid), Int32(ppc), Int32(first),
                                       T.(vdrift), T.(vth), key; ndrange = n * ppc)
    fill!(s.count, Int32(last))
    return s
end
