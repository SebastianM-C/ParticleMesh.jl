# The pullbacks of the transfers (WP-077): the reverse-mode adjoints of `deposit!`, `gather!`,
# `deposit_flux!` (N = 1 and 2) and the sort, written by hand as plain kernels. Each adds the adjoint
# of its transfer into a shadow species (and, for the gather, a shadow field), from the adjoint of
# what the transfer wrote. The extension on EnzymeCore registers them as Enzyme's reverse rules,
# so that Enzyme runs these kernels instead of differentiating the transfers' own kernels.

"""
    pullback_quantity!(s̄, quantity, s, i, q̄)

The device-level pullback of a deposited quantity: add the adjoint of `q̄·quantity(s, i)` with
respect to the arrays of species `s` into the shadow species `s̄` (a species of the same layout
holding adjoints) at slot `i`. The engine defines it for `ChargeWeight()` (`s̄.w[i] += q·q̄`); a
module that deposits a quantity of its own defines a method for it next to the quantity, so that
the reverse rules of `deposit!` and `deposit_flux!` can take it (WP-077). It writes slot `i`
only, so the pullback kernels need no atomics for it.
"""
function pullback_quantity! end

@inline function pullback_quantity!(s̄, ::ChargeWeight, s, i, q̄)
    @inbounds s̄.w[i] += s.q * q̄
    return nothing
end

"Whether `quantity` has a `pullback_quantity!` method for these species types (a host check before a pullback launches)."
has_pullback(quantity, s̄, s) = hasmethod(pullback_quantity!, Tuple{typeof(s̄),typeof(quantity),typeof(s),Int,eltype(s)})

function check_pullback(quantity, s̄, s)
    has_pullback(quantity, s̄, s) ||
        throw(ArgumentError("no pullback_quantity! method for the quantity $(typeof(quantity)): define one next to the quantity " *
                            "for the reverse rules of deposit! and deposit_flux! (WP-077)"))
    return nothing
end

as_staggers(st::Stagger) = (st,)
as_staggers(st::Tuple) = st

"""
    gather_slopes_at(a, c::NTuple{N,Int32}, δ::NTuple{N,T}, shape, stagger::NTuple{N}, g) -> NTuple{N,T}

The derivatives of `gather_at(a, c, δ, shape, stagger, g)` with respect to each offset `δ_d`: the
field read with the tensor-product weights in which axis `d`'s weights are replaced by their
`shape_slopes`.
"""
@inline function gather_slopes_at(a, c::NTuple{N,Int32}, δ::NTuple{N,T}, shape::Shape{P}, stagger::NTuple{N,Stagger},
                                  g::Int32) where {N,T,P}
    fw = shape_weights(shape, stagger, δ)
    dw = ntuple(d -> shape_slopes(shape, stagger[d], δ[d]), Val(N))
    acc = ntuple(_ -> zero(T), Val(N))
    @inbounds for K in CartesianIndices(ntuple(_ -> Base.OneTo(P + 1), Val(N)))
        k = Tuple(K)
        v = a[ntuple(d -> c[d] + fw[d][1] + g + Int32(k[d] - 1), Val(N))...]
        t = ntuple(d -> prod(ntuple(e -> e == d ? dw[e][2][k[e]] : fw[e][2][k[e]], Val(N))) * v, Val(N))
        acc = map(+, acc, t)
    end
    return acc
end

@inline function add_offsets!(s̄, i, v::NTuple{N}) where {N}
    ntuple(d -> (@inbounds s̄.offset[d][i] += v[d]; nothing), Val(N))
    return nothing
end

@kernel function deposit_pullback_kernel!(s̄, @Const(ρ̄), s, quantity, shape, stagger::NTuple{N,Stagger}, g::Int32) where {N}
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        c = particle_cell(s, i, Val(N))
        δ = particle_offset(s, i, Val(N))
        pullback_quantity!(s̄, quantity, s, i, gather_at(ρ̄, c, δ, shape, stagger, g))
        qv = quantity(s, i)
        add_offsets!(s̄, i, map(x -> qv * x, gather_slopes_at(ρ̄, c, δ, shape, stagger, g)))
    end
end

"""
    deposit_pullback!(s̄, ρ̄, species, shape, stagger, grid; quantity = ChargeWeight(), workgroupsize = nothing)

The pullback of `deposit!(mode, ρ, species, shape, stagger, grid, ws; quantity)` in either mode:
with `ρ̄` the adjoint of `ρ` (guards included, as the deposit writes them), add into the shadow
species `s̄`, for every alive particle `i`, the adjoint of its quantity (`pullback_quantity!` of
`ρ̄` gathered at the particle, the transpose of the deposit) and of its offsets (the quantity
times `ρ̄` gathered with the shape's slopes along each axis). `ρ̄` is left as it is: the deposit
adds into `ρ`. One work item per particle writing its own slots, so the result does not depend on
the launch shape or the thread count. N = 1, 2 and 3.

Guards: the pullback reads `ρ̄` where `Fast()` (and `Reference()` with `CellBins`) writes. With
`NodeBins`, a `Reference()` deposit writes some contributions into the guard on the other side,
the periodic image of the same node, so the pullback is its transpose only for an adjoint that is
equal on a guard and its image, as the adjoint of [`fold_guards!`](@ref) is (a step folds after
depositing); a loss that reads the unfolded guards directly gets the cell-bin transpose.
"""
function deposit_pullback!(s̄::Species{N}, ρ̄::AbstractArray, s::Species{N}, shape::Shape, stagger, grid::PeriodicGrid{N};
                           quantity = ChargeWeight(), workgroupsize = nothing) where {N}
    check_pullback(quantity, s̄, s)
    launcher(deposit_pullback_kernel!, get_backend(s), workgroupsize)(s̄, ρ̄, s, quantity, shape, as_staggers(stagger), Int32(guards(grid));
                                                                     ndrange = capacity(s))
    return s̄
end

@kernel function gather_pullback_kernel!(ā, s̄, ō, @Const(a), s, shape::Shape{P}, stagger::NTuple{N,Stagger}, g::Int32) where {N,P}
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        o = ō[i]
        ō[i] = zero(o)
        c = particle_cell(s, i, Val(N))
        δ = particle_offset(s, i, Val(N))
        if !(ā isa Nothing)
            fw = shape_weights(shape, stagger, δ)
            for K in CartesianIndices(ntuple(_ -> Base.OneTo(P + 1), Val(N)))
                k = Tuple(K)
                idx = CartesianIndex(ntuple(d -> Int(c[d] + fw[d][1] + g) + k[d] - 1, Val(N)))
                Atomix.@atomic ā[idx] += o * tensor_weight(fw, k)
            end
        end
        s̄ isa Nothing || add_offsets!(s̄, i, map(x -> o * x, gather_slopes_at(a, c, δ, shape, stagger, g)))
    end
end

"""
    gather_pullback!(ā, s̄, ō, a, species, shape, stagger, grid; workgroupsize = nothing)

The pullback of `gather!(out, a, species, shape, stagger, grid)`: with `ō` the adjoint of `out`,
for every alive particle `i` add `ō[i]` times its shape weights into the shadow field `ā` (the
transpose of the gather, a deposit into the guards, by atomics), add `ō[i]` times `a` gathered
with the shape's slopes into the offsets of the shadow species `s̄`, and zero `ō[i]`, which the
gather overwrote. Either `ā` or `s̄` may be `nothing` (no adjoint wanted). The atomic adds make
`ā` depend on their order by roundoff. N = 1, 2 and 3.
"""
function gather_pullback!(ā, s̄, ō::AbstractVector, a::AbstractArray, s::Species{N}, shape::Shape, stagger, grid::PeriodicGrid{N};
                          workgroupsize = nothing) where {N}
    launcher(gather_pullback_kernel!, get_backend(s), workgroupsize)(ā, s̄, ō, a, s, shape, as_staggers(stagger), Int32(guards(grid));
                                                                    ndrange = capacity(s))
    return ō
end

"`Σ_k u[k]·a[b + k - 1]`, in increasing `k`."
@inline function lin1(u::NTuple{M,Any}, a, b::Int32, ::Type{T}) where {M,T}
    acc = zero(T)
    @inbounds for k in 1:M
        acc += u[k] * a[b + Int32(k - 1)]
    end
    return acc
end

"`Σ_{k1,k2} u[k1]·v[k2]·a[b1 + k1 - 1, b2 + k2 - 1]`, each row of axis 1 summed first."
@inline function lin2(u::NTuple{M1,Any}, v::NTuple{M2,Any}, a, b1::Int32, b2::Int32, ::Type{T}) where {M1,M2,T}
    acc = zero(T)
    @inbounds for k2 in 1:M2
        row = zero(T)
        for k1 in 1:M1
            row += u[k1] * a[b1 + Int32(k1 - 1), b2 + Int32(k2 - 1)]
        end
        acc += v[k2] * row
    end
    return acc
end

@inline front_prefix(t) = Base.front(prefix_sums(t))
@inline halve(t) = map(x -> x / 2, t)

# N = 1, in the particle's frame: Jx = -(q·w·sx)·P on the P + 1 nodes of the prefix sums of
# ΔS = Wn - Wo, Jy = quantity_y·A and Jz = quantity_z·A with A = (Wo + Wn)/2 on P + 2.
@kernel function flux_pullback_1d_kernel!(s̄, @Const(Ēx), @Const(Ēy), @Const(Ēz), s, shape, st, n::Int32, g::Int32, sx,
                                          quantity_y, quantity_z)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        T = eltype(s.w)
        c, δ, c0, δ0 = s.cell[1][i], s.offset[1][i], s.extras.old_cell[i], s.extras.old_offset[i]
        f, Wo, Wn = flux_axis_frame(shape, st, c, δ, c0, δ0, n)
        dWo, dWn = flux_axis_slopes(shape, st, c, δ, c0, δ0, n)
        A = map((a, b) -> (a + b) / 2, Wo, Wn)
        b = c + f + g
        X, Y, Z = lin1(front_prefix(map(-, Wn, Wo)), Ēx, b, T), lin1(A, Ēy, b, T), lin1(A, Ēz, b, T)
        qw = s.q * s.w[i]
        qy, qz = quantity_y(s, i), quantity_z(s, i)
        kx = -(qw * sx)
        s̄.w[i] += s.q * -(sx * X)
        pullback_quantity!(s̄, quantity_y, s, i, Y)
        pullback_quantity!(s̄, quantity_z, s, i, Z)
        s̄.offset[1][i] += kx * lin1(front_prefix(dWn), Ēx, b, T) + qy * lin1(halve(dWn), Ēy, b, T) + qz * lin1(halve(dWn), Ēz, b, T)
        s̄.extras.old_offset[i] += -(kx * lin1(front_prefix(dWo), Ēx, b, T)) + qy * lin1(halve(dWo), Ēy, b, T) +
                                  qz * lin1(halve(dWo), Ēz, b, T)
    end
end

# N = 2, in the particle's frame: Jx = -(q·w·sx₁)·P₁ ⊗ A₂, Jy = -(q·w·sx₂)·A₁ ⊗ P₂ and
# Jz = quantity_z·(A₁ ⊗ A₂ + ΔS₁ ⊗ ΔS₂/12) (`flux_stencil`). The derivatives with respect to the
# new offsets δ_d take ∂Wn_d = dWn_d (so ∂A_d = dWn_d/2, ∂ΔS_d = dWn_d), those with respect to the
# old offsets ∂Wo_d = dWo_d (∂A_d = dWo_d/2, ∂ΔS_d = -dWo_d).
@kernel function flux_pullback_2d_kernel!(s̄, @Const(Ēx), @Const(Ēy), @Const(Ēz), s, shape, st, n::NTuple{2,Int32}, g::Int32, sx,
                                          quantity_z)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        T = eltype(s.w)
        c = particle_cell(s, i, Val(2))
        δ = particle_offset(s, i, Val(2))
        c0 = old_cell_2d(s, i)
        δ0 = old_offset_2d(s, i)
        f1, Wo1, Wn1 = flux_axis_frame(shape, st[1], c[1], δ[1], c0[1], δ0[1], n[1])
        f2, Wo2, Wn2 = flux_axis_frame(shape, st[2], c[2], δ[2], c0[2], δ0[2], n[2])
        dWo1, dWn1 = flux_axis_slopes(shape, st[1], c[1], δ[1], c0[1], δ0[1], n[1])
        dWo2, dWn2 = flux_axis_slopes(shape, st[2], c[2], δ[2], c0[2], δ0[2], n[2])
        D1, D2 = map(-, Wn1, Wo1), map(-, Wn2, Wo2)
        A1, A2 = map((a, b) -> (a + b) / 2, Wo1, Wn1), map((a, b) -> (a + b) / 2, Wo2, Wn2)
        p1, p2 = front_prefix(D1), front_prefix(D2)
        b1, b2 = c[1] + f1 + g, c[2] + f2 + g
        X = lin2(p1, A2, Ēx, b1, b2, T)
        Y = lin2(A1, p2, Ēy, b1, b2, T)
        Z = lin2(A1, A2, Ēz, b1, b2, T) + lin2(D1, D2, Ēz, b1, b2, T) / 12
        # the derivatives of X, Y, Z with respect to δ₁, δ₂ (n) and δ0₁, δ0₂ (o)
        Xn1 = lin2(front_prefix(dWn1), A2, Ēx, b1, b2, T)
        Xo1 = -lin2(front_prefix(dWo1), A2, Ēx, b1, b2, T)
        Xn2 = lin2(p1, dWn2, Ēx, b1, b2, T) / 2
        Xo2 = lin2(p1, dWo2, Ēx, b1, b2, T) / 2
        Yn1 = lin2(dWn1, p2, Ēy, b1, b2, T) / 2
        Yo1 = lin2(dWo1, p2, Ēy, b1, b2, T) / 2
        Yn2 = lin2(A1, front_prefix(dWn2), Ēy, b1, b2, T)
        Yo2 = -lin2(A1, front_prefix(dWo2), Ēy, b1, b2, T)
        Zn1 = lin2(dWn1, A2, Ēz, b1, b2, T) / 2 + lin2(dWn1, D2, Ēz, b1, b2, T) / 12
        Zo1 = lin2(dWo1, A2, Ēz, b1, b2, T) / 2 - lin2(dWo1, D2, Ēz, b1, b2, T) / 12
        Zn2 = lin2(A1, dWn2, Ēz, b1, b2, T) / 2 + lin2(D1, dWn2, Ēz, b1, b2, T) / 12
        Zo2 = lin2(A1, dWo2, Ēz, b1, b2, T) / 2 - lin2(D1, dWo2, Ēz, b1, b2, T) / 12
        qw = s.q * s.w[i]
        qz = quantity_z(s, i)
        kx, ky = -(qw * sx[1]), -(qw * sx[2])
        s̄.w[i] += s.q * (-(sx[1] * X) - sx[2] * Y)
        pullback_quantity!(s̄, quantity_z, s, i, Z)
        s̄.offset[1][i] += kx * Xn1 + ky * Yn1 + qz * Zn1
        s̄.offset[2][i] += kx * Xn2 + ky * Yn2 + qz * Zn2
        s̄.extras.old_offset_1[i] += kx * Xo1 + ky * Yo1 + qz * Zo1
        s̄.extras.old_offset_2[i] += kx * Xo2 + ky * Yo2 + qz * Zo2
    end
end

"""
    flux_pullback!(s̄, J̄x, J̄y, J̄z, species, shape, stagger, grid; dt, quantity_z, quantity_y, workgroupsize = nothing)

The pullback of `deposit_flux!(mode, Jx, Jy, Jz, species, shape, stagger, grid, ws; dt, quantity_y,
quantity_z)` at N = 1 and N = 2, in either mode: with `J̄x`, `J̄y`, `J̄z` the adjoints of the three
fields (guards included), add into the shadow species `s̄`, for every alive particle, the adjoints
of its weight, of the deposited transverse quantities (through `pullback_quantity!`; `quantity_y`
at N = 1 only), of its current offsets and of its old offsets (`s̄`'s extras of `flux_extras`).
It reads `J̄` with Esirkepov's weights along the move in the particle's frame (the frame of the
`Fast()` path, whose left-out column has zero derivative, as its weight changes sum to zero) and
with their derivatives (`flux_axis_slopes`). The `J̄` are left as they are: the deposition adds.
One work item per particle writing its own slots. The cells and the old cells are integers and
take no adjoint; the derivative is that of the stencil within the cells the particle has
(piecewise smooth in the positions, as the shape functions are).

Guards: the pullback reads `J̄` where `Fast()` (and `Reference()` with `CellBins`) writes. With
`NodeBins`, a `Reference()` deposit writes some contributions into the guard on the other side,
the periodic image of the same node, so the pullback is its transpose only for an adjoint that is
equal on a guard and its image, as the adjoint of [`fold_guards!`](@ref) is (a step folds after
depositing); a loss that reads the unfolded guards directly gets the cell-bin transpose.
"""
function flux_pullback!(s̄::Species{1}, J̄x, J̄y, J̄z, s::Species{1}, shape::Shape, stagger::Stagger, grid::PeriodicGrid{1};
                        dt, quantity_y, quantity_z, workgroupsize = nothing)
    check_pullback(quantity_y, s̄, s)
    check_pullback(quantity_z, s̄, s)
    sx = cellsize(grid) / dt
    launcher(flux_pullback_1d_kernel!, get_backend(s), workgroupsize)(s̄, J̄x, J̄y, J̄z, s, shape, stagger, Int32(ncells(grid)),
                                                                     Int32(guards(grid)), sx, quantity_y, quantity_z;
                                                                     ndrange = capacity(s))
    return s̄
end

function flux_pullback!(s̄::Species{2}, J̄x, J̄y, J̄z, s::Species{2}, shape::Shape, stagger::NTuple{2,Stagger}, grid::PeriodicGrid{2};
                        dt, quantity_z, quantity_y = NoQuantity(), workgroupsize = nothing)
    quantity_y isa NoQuantity || throw(ArgumentError("deposit_flux! at N = 2 takes no quantity_y: Jy is in-plane"))
    check_pullback(quantity_z, s̄, s)
    sx = map(h -> h / dt, cellsize(grid))
    launcher(flux_pullback_2d_kernel!, get_backend(s), workgroupsize)(s̄, J̄x, J̄y, J̄z, s, shape, stagger, Int32.(ncells(grid)),
                                                                     Int32(guards(grid)), sx, quantity_z; ndrange = capacity(s))
    return s̄
end

@kernel function scatter_perm_kernel!(dst, @Const(src), @Const(perm))
    i = @index(Global, Linear)
    @inbounds dst[perm[i]] = src[i]
end

"""
    sort_pullback!(s̄, perm; workgroupsize = nothing)

The pullback of a sort that gathered every array of a species by `perm` (slot `i` took what slot
`perm[i]` held; `ws.perm` after `sort_by_cell!` or `resort_by_cell!`): every floating-point array
of the shadow species `s̄`, the extras included, is scattered back, `ā[perm[i]] = ā′[i]`, the
inverse permutation (the transpose of a permutation). Integer arrays carry no adjoint and stay.
"""
function sort_pullback!(s̄::Species, perm::AbstractVector{<:Integer}; workgroupsize = nothing)
    arrays = Iterators.filter(a -> eltype(a) <: AbstractFloat, species_arrays(s̄))
    isempty(arrays) && return s̄
    tmp = similar(first(arrays))
    backend = get_backend(perm)
    for a in arrays
        b = eltype(a) === eltype(tmp) ? tmp : similar(a)
        launcher(scatter_perm_kernel!, backend, workgroupsize)(b, a, perm; ndrange = length(perm))
        copyto!(a, b)
    end
    return s̄
end
