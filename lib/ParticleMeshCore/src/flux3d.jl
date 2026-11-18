# Charge-conserving flux deposition (Esirkepov's scheme) for N = 3: the per-particle stencil of
# one component and its two paths, atomic Fast() and per-bin-patch Reference(), one component
# per launch.

has_flux_extras(s::Species{3}) =
    all(k -> haskey(s.extras, k), (:old_cell_1, :old_cell_2, :old_cell_3, :old_offset_1, :old_offset_2, :old_offset_3))

@inline old_cell_3d(s, i) = @inbounds (s.extras.old_cell_1[i], s.extras.old_cell_2[i], s.extras.old_cell_3[i])
@inline old_offset_3d(s, i) = @inbounds (s.extras.old_offset_1[i], s.extras.old_offset_2[i], s.extras.old_offset_3[i])

# The two axes other than `C`, in increasing order.
@inline transverse_axes(::Val{1}) = (2, 3)
@inline transverse_axes(::Val{2}) = (1, 3)
@inline transverse_axes(::Val{3}) = (1, 2)

"""
    flux_parts_3d(Val(C), s, i, shape, stagger::NTuple{3}, n, sx, frame) -> (lo, (p, Aa, Ab, Da, Eb))

What component `C` of particle `i`'s 3D flux is made of, on the nodes `cell .+ lo .+ m .- 1` of
`frame` (`CellFrame()`, `BinFrame(bins)` or `ParticleFrame()`, as at N = 2): with `S0_d`, `S1_d` the
old and new 1D shape weights of axis `d` on the frame's nodes, `ΔS_d = S1_d - S0_d` and
`A_d = (S0_d + S1_d)/2`, `p` is `-(q·w·sx_C)` times the prefix sums of `ΔS_C` in increasing node
order, and on the two other axes `a < b` it is `A_a`, `A_b`, `ΔS_a` and `ΔS_b/12`. The value of
the component on node `(m_C; m_a, m_b)` is `p[m_C]·(A_a[m_a]·A_b[m_b] + ΔS_a[m_a]·(ΔS_b[m_b]/12))`
(`flux_values_3d`).
"""
@inline function flux_parts_3d(vC::Val{C}, s, i, shape::Shape, st::NTuple{3,Stagger}, n::NTuple{3,Int32}, sx::NTuple{3,Any},
                               frame) where {C}
    c = particle_cell(s, i, Val(3))
    δ = particle_offset(s, i, Val(3))
    c0 = old_cell_3d(s, i)
    δ0 = old_offset_3d(s, i)
    f1 = axis_frame(frame, shape, st[1], c[1], δ[1], c0[1], δ0[1], n[1], 1)
    f2 = axis_frame(frame, shape, st[2], c[2], δ[2], c0[2], δ0[2], n[2], 2)
    f3 = axis_frame(frame, shape, st[3], c[3], δ[3], c0[3], δ0[3], n[3], 3)
    fr = (f1, f2, f3)
    lo = (f1[1], f2[1], f3[1])
    a, b = transverse_axes(vC)
    _, Woa, Wna = fr[a]
    _, Wob, Wnb = fr[b]
    _, Woc, Wnc = fr[C]
    @inbounds k = -(s.q * s.w[i] * sx[C])
    p = map(x -> k * x, prefix_sums(map(-, Wnc, Woc)))
    Aa = map((x, y) -> (x + y) / 2, Woa, Wna)
    Ab = map((x, y) -> (x + y) / 2, Wob, Wnb)
    Da = map(-, Wna, Woa)
    Eb = map((x, y) -> (x - y) / 12, Wnb, Wob)
    return (lo, (p, Aa, Ab, Da, Eb))
end

"""
    flux_dims_3d(Val(C), M::NTuple{3,Int}) -> NTuple{3,Int}

The node counts per axis of component `C` on a frame of `M` nodes per axis: `M`, one fewer
along the component's own axis. The prefix sum of all the weight changes on the last node is zero
up to roundoff and has no node of its own, as in PIConGPU, WarpX and Smilei.
"""
flux_dims_3d(::Val{C}, M::NTuple{3,Int}) where {C} = ntuple(d -> M[d] - (d == C), 3)

"""
    flux_values_3d(Val(C), parts, Val(K0), Val(Ks)) -> NTuple{Ks}

The values `K0 + 1 : K0 + Ks` of component `C` from its `flux_parts_3d`, column-major over the
component's nodes (`flux_dims_3d`, axis 1 fastest), each `p[m_C]·(A_a[m_a]·A_b[m_b] + ΔS_a[m_a]·Eb[m_b])`,
as straight-line code with every transverse product formed once. One expression per value, the
same in every frame and every split into planes, so a particle's value on a node is the same bits
whichever launch forms it (T-944).
"""
@generated function flux_values_3d(::Val{C}, parts::Tuple{NTuple{Mc,Any},NTuple{Ma,Any},NTuple{Mb,Any},NTuple{Ma,Any},NTuple{Mb,Any}},
                                   ::Val{K0}, ::Val{Ks}) where {C,Mc,Ma,Mb,K0,Ks}
    a, b = transverse_axes(Val(C))
    M = ntuple(d -> d == C ? Mc : d == a ? Ma : Mb, 3)
    dims = flux_dims_3d(Val(C), M)
    ci = CartesianIndices(dims)
    seen = Dict{Tuple{Int,Int},Symbol}()
    body = Expr[]
    entries = Expr[]
    for l in (K0 + 1):(K0 + Ks)
        m = Tuple(ci[l])
        key = (m[a], m[b])
        t = get!(seen, key) do
            sym = Symbol(:t_, m[a], :_, m[b])
            push!(body, :($sym = Aa[$(m[a])] * Ab[$(m[b])] + Da[$(m[a])] * Eb[$(m[b])]))
            sym
        end
        push!(entries, :(p[$(m[C])] * $t))
    end
    return quote
        $(Expr(:meta, :inline))
        p, Aa, Ab, Da, Eb = parts
        @inbounds begin
            $(body...)
            ($(entries...),)
        end
    end
end

# Atomic adds of the column-major block `v` of dims `D` at the field index `b`. Relaxed, as at N = 2.
@inline function atomic_add_block_3d!(J, v::NTuple{K,Any}, b::NTuple{3,Int}, ::Val{D}) where {K,D}
    ci = CartesianIndices(D)
    for k in 1:K
        m = Tuple(ci[k])
        @inbounds Atomix.@atomic :monotonic J[CartesianIndex(b[1] + m[1] - 1, b[2] + m[2] - 1, b[3] + m[3] - 1)] += v[k]
    end
    return nothing
end

@inline function atomic_component_3d!(J, vC::Val{C}, s, i, shape::Shape{P}, st, n, g::Int32, sx) where {C,P}
    lo, parts = flux_parts_3d(vC, s, i, shape, st, n, sx, ParticleFrame())
    D = flux_dims_3d(vC, (P + 2, P + 2, P + 2))
    v = flux_values_3d(vC, parts, Val(0), Val(prod(D)))
    c = particle_cell(s, i, Val(3))
    atomic_add_block_3d!(J, v, ntuple(d -> Int(c[d] + lo[d] + g), 3), Val(D))
    return nothing
end

# One work item per particle, in the particle's frame: 3(P + 1)(P + 2)² atomic adds (144 at order 2).
@kernel function flux_atomic_3d_kernel!(Jx, Jy, Jz, s, shape, st, n::NTuple{3,Int32}, g::Int32, sx)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        atomic_component_3d!(Jx, Val(1), s, i, shape, st, n, g, sx)
        atomic_component_3d!(Jy, Val(2), s, i, shape, st, n, g, sx)
        atomic_component_3d!(Jz, Val(3), s, i, shape, st, n, g, sx)
    end
end

"""
    deposit_flux!(mode, Jx, Jy, Jz, species::Species{3}, shape, stagger::NTuple{3,Stagger}, grid, ws;
                  dt, planes = nothing, workgroupsize = nothing)

The charge-conserving current of one step at N = 3 (Esirkepov's scheme), added into the three
fields `Jx`, `Jy`, `Jz` (3D arrays with ρ's layout and guards). `stagger` is ρ's, and every shape
weight is evaluated at it; `J_d[j]` is the face half a cell along axis `d` from ρ's node `j`, at
ρ's index `j` (for ρ at `(Node(), Node(), Node())`, the module's layout, Jx at `(Center(), Node(),
Node())` and cyclically). Per particle, with `S0_d`, `S1_d` the old and new 1D weights of axis `d`,
`ΔS_d = S1_d - S0_d`, `A_d = (S0_d + S1_d)/2` and `P_d` the prefix sum of `ΔS_d` in increasing node
order,

    Jx = -(q·w·Δx₁/dt)·P_1 ⊗ (A_2 ⊗ A_3 + ΔS_2 ⊗ ΔS_3/12)

and cyclically (the transverse factor of `Jy` over axes 1 and 3, of `Jz` over 1 and 2), which
expands to Esirkepov's `ΔS_1·(S0_2·S0_3 + (ΔS_2·S0_3 + S0_2·ΔS_3)/2 + ΔS_2·ΔS_3/3)`. Every
component is in space, so there is no module quantity: a `quantity_y` or `quantity_z` is an
`ArgumentError`. The old position comes from the extras of `flux_extras(T, 3)`, which the module
stores before its push and the sort carries.

With ρ from `deposit!` with the same shape and stagger before and after the step, discrete
continuity `(ρ_new - ρ_old)/dt + Σ_d (J_d[j] - J_d[j - e_d])/Δx_d = 0` holds at every node to
roundoff (T-940); `Σ J_d = Σ q·w·Δ_d·Δx_d/dt` and a particle at rest deposits exactly zero
(T-941); a species that does not move along axis 3 deposits exactly zero `Jz` and, summed along
axis 3, the 2D flux, and one that moves along axis 3 only, exactly zero `Jx`, `Jy` and the 1D flux
(T-942); one particle's current is Esirkepov's by hand (T-943). Reference and Fast agree per node
to roundoff (T-944); the flux is linear in the weights, with exact `Dual` partials and the primal
of a `Dual` run the Float32 run bit for bit (T-948); order 2 is C¹ in the displacement through a
crossing of a face, an edge and the corner (T-947); allocation does not grow with the particles or
the grid (T-949).

Preconditions: every particle moved less than one cell per axis since the old position was
stored, seams, edges and corners included (not detected); at least 3 cells per axis; at least
`maximum(flux_guard_width(shape, stagger))` guard cells (checked). The guard convention is
`deposit!`'s (fold each field).

`Fast()` adds with atomics, one work item per particle in its own frame (`ParticleFrame()`,
`3(P + 1)(P + 2)²` adds, 144 at order 2), and falls back to `Reference()` for element types without
them (`effective_mode`), sorting first. `Reference()` needs `sort_by_cell!` after the move and a
workspace built with `Workspace(species, grid; flux = (shape, stagger))`, and is bitwise across
reruns, thread counts, launch shapes and `planes` (T-945). It runs one component at a time:
one work item per bin (`ws.bins`: cells, or `NodeBins`) sums the component's values of its
particles in slot order, from zero, in registers, and writes them once into the bin's patch; one
work item per node then adds the patches of the bins whose frame holds it, in increasing
column-major linear index, from zero. A component has `(M₁ - 1)·M₂·M₃` values per bin (cyclically),
`M_d` the flux stencil width of the bin's frame (`flux_bin_reach`): 100 at order 2 on nodes by
`NodeBins` of that shape and stagger, 180 by cell. A wide element type does not keep them all in
registers, so the per-bin kernel takes the planes of the frame's third axis in groups of `planes`
per launch, each launch forming the same values of the same particles in the same order (the bits
do not depend on the split); `planes = nothing` takes as many as fit `FLUX_PATCH_BYTES` (all of
them for Float32 and Float64 at order 2 by `NodeBins`, two of five for `Dual{Float32,3}`).
"""
function deposit_flux!(mode, Jx::AbstractArray{T,3}, Jy::AbstractArray{T,3}, Jz::AbstractArray{T,3}, s::Species{3},
                       shape::Shape, stagger::NTuple{3,Stagger}, grid::PeriodicGrid{3}, ws::Workspace;
                       dt, quantity_y = NoQuantity(), quantity_z = NoQuantity(), planes = nothing, workgroupsize = nothing) where {T}
    quantity_y isa NoQuantity && quantity_z isa NoQuantity ||
        throw(ArgumentError("deposit_flux! at N = 3 takes no quantity_y or quantity_z: every component is in space, from Esirkepov's decomposition"))
    has_flux_extras(s) ||
        throw(ArgumentError("deposit_flux! at N = 3 reads the old position from the extras of flux_extras(T, 3): $(keys(flux_extras(T, 3)))"))
    gw = maximum(flux_guard_width(shape, stagger))
    guards(grid) >= gw ||
        throw(ArgumentError("deposit_flux! needs $gw guard cells for this shape and stagger, the grid has $(guards(grid))"))
    all(>=(3), ncells(grid)) || throw(ArgumentError("deposit_flux! needs at least 3 cells per axis"))
    planes === nothing || planes isa Integer && planes >= 1 ||
        throw(ArgumentError("planes: a positive integer or nothing, got $planes"))
    sx = map(h -> h / dt, cellsize(grid))
    fallback_sort!(mode, T, s, grid, ws, workgroupsize)
    _deposit_flux_3d!(effective_mode(mode, T), (Jx, Jy, Jz), s, shape, stagger, grid, ws, sx, planes, workgroupsize)
end

function _deposit_flux_3d!(::Fast, J, s, shape, st, grid, ws, sx, planes, workgroupsize)
    launcher(flux_atomic_3d_kernel!, get_backend(s), workgroupsize)(J..., s, shape, st, Int32.(ncells(grid)), Int32(guards(grid)), sx;
                                                                   ndrange = capacity(s))
    return J
end

# The Reference() path.

"""
    FLUX_PATCH_BYTES

The bytes of accumulators one work item of the 3D `Reference()` flux patch holds by default: the
planes of a launch are as many as fit (at least one). 800 bytes: on an RTX 4090 at 64³ cells and
8 particles per cell, by `NodeBins` at order 2 (100 values per component), all five planes in one
launch were the fastest for Float32 and Float64 (400 and 800 bytes) and two planes per launch for
`Dual{Float32,3}` (800 bytes; all five, 1600 bytes, took 2.9 times as long), in the prototype of the
deposition study (`scripts/deposition/proto.jl` on the branch `claude/project-thread-5h7lt1`; one
run, not reviewed). Not
measured on the CPU backend.
"""
const FLUX_PATCH_BYTES = 800

"The planes of the frame's third axis per launch of the patch kernel: `planes`, or as many as fit `FLUX_PATCH_BYTES`."
flux_planes(::Type{T}, plane::Int, total::Int, ::Nothing) where {T} = clamp(FLUX_PATCH_BYTES ÷ (sizeof(T) * plane), 1, total)
flux_planes(::Type{T}, plane::Int, total::Int, planes::Integer) where {T} = clamp(Int(planes), 1, total)

"The per-axis stencil width of the 3D flux in the frame of a bin, as plain integers."
@inline flux_widths(shape::Shape, st::NTuple{3,Stagger}, bins = CellBins()) =
    ntuple(d -> _valof(flux_bin_width(bins, shape, st[d], d)), 3)

"""
    flux_patch_values(shape, stagger::NTuple{3,Stagger}, bins = CellBins())

Values of `T` per linear bin in the flux patch of `Workspace(species, grid; flux = (shape,
stagger), bins)` at N = 3: the largest component's, `(M₁ - 1)·M₂·M₃` and cyclically, which the
three components take in turn (100 at order 2 on nodes by `NodeBins` of that shape and stagger,
180 by cell).
"""
function flux_patch_values(shape::Shape, st::NTuple{3,Stagger}, bins = CellBins())
    M = flux_widths(shape, st, bins)          # each component written out: a `Val` of a loop variable would not infer
    return max(prod(flux_dims_3d(Val(1), M)), prod(flux_dims_3d(Val(2), M)), prod(flux_dims_3d(Val(3), M)))
end

# One work item per linear bin: its particles in slot order, the values `K0 + 1 : K0 + Ks` of
# component `C` summed from zero in a register tuple and written once into the bin's patch. An
# empty bin writes zeros.
@kernel function flux_patch_3d_kernel!(patch, s, @Const(seg_start), @Const(seg_stop), shape, st, frame, n::NTuple{3,Int32}, sx, layout,
                                       vC::Val{C}, vK0::Val{K0}, vKs::Val{Ks}) where {C,K0,Ks}
    L = Int32(@index(Global, Linear))
    T = eltype(patch)
    acc = ntuple(_ -> zero(T), Val(Ks))
    @inbounds for i in seg_start[L]:seg_stop[L]
        _, parts = flux_parts_3d(vC, s, i, shape, st, n, sx, frame)
        acc = tuple_add(acc, flux_values_3d(vC, parts, vK0, vKs))
    end
    @inbounds for k in 1:Ks
        patch[patch_index(layout, K0 + k, L)...] = acc[k]
    end
end

# One work item per node (guards included): the patches of the bins whose frame holds it, in
# increasing column-major linear index, summed from zero. `lo` is the frame's first node relative
# to its bin and `D` the component's node counts per axis.
@kernel function flux_reduce_3d_kernel!(J, @Const(patch), layout, n::NTuple{3,Int32}, g::Int32, lo::NTuple{3,Int32}, D::NTuple{3,Int32})
    I = @index(Global, Cartesian)
    T = eltype(J)
    j1, j2, j3 = Int32(I[1]) - g - lo[1], Int32(I[2]) - g - lo[2], Int32(I[3]) - g - lo[3]
    acc = zero(T)
    @inbounds for c3 in max(Int32(1), j3 - D[3] + Int32(1)):min(n[3], j3), c2 in max(Int32(1), j2 - D[2] + Int32(1)):min(n[2], j2),
                  c1 in max(Int32(1), j1 - D[1] + Int32(1)):min(n[1], j1)
        L = c1 + ((c2 - Int32(1)) + (c3 - Int32(1)) * n[2]) * n[1]
        k = (j1 - c1) + ((j2 - c2) + (j3 - c3) * D[2]) * D[1] + Int32(1)
        acc += patch[patch_index(layout, k, L)...]
    end
    @inbounds J[I] += acc
end

function _deposit_flux_3d!(::Reference, J, s, shape, st, grid, ws, sx, planes, workgroupsize)
    T = eltype(s)
    check_workspace(ws, s, grid)
    ws.flux == (shape, st) ||
        throw(ArgumentError("the Reference() deposit_flux! at N = 3 needs Workspace(species, grid; flux = (shape, stagger)) for this " *
                            "shape and stagger, $((shape, st)); the workspace was built with flux = $(ws.flux)"))
    need = flux_patch_values(shape, st, ws.bins)
    size(ws.flux_patch) == flux_patch_size(ws.flux_layout, need, prod(ncells(grid))) ||
        throw(ArgumentError("the Reference() deposit_flux! at N = 3 needs Workspace(species, grid; flux = (shape, stagger)) for this " *
                            "shape and stagger: $need values per bin in the layout $(ws.flux_layout), the workspace's patch array " *
                            "is $(size(ws.flux_patch))"))
    backend = get_backend(s)
    n = Int32.(ncells(grid))
    g = Int32(guards(grid))
    M = flux_widths(shape, st, ws.bins)
    r = ntuple(d -> flux_bin_reach(ws.bins, shape, st[d], d), 3)
    check_bin_guards(grid, r)
    lo = ntuple(d -> r[d][1], 3)
    frame = ws.bins isa CellBins ? CellFrame() : BinFrame(ws.bins)
    for (C, Jc) in zip((Val(1), Val(2), Val(3)), J)
        D = flux_dims_3d(C, M)
        plane = D[1] * D[2]
        step = flux_planes(T, plane, D[3], planes)
        for a in 1:step:D[3]
            count = min(step, D[3] - a + 1)
            item_launcher(flux_patch_3d_kernel!, backend, workgroupsize)(ws.flux_patch, s, ws.seg_start, ws.seg_stop, shape, st, frame, n, sx,
                                                                         ws.flux_layout, C, Val((a - 1) * plane), Val(count * plane);
                                                                         ndrange = prod(n))
        end
        launcher(flux_reduce_3d_kernel!, backend, workgroupsize)(Jc, ws.flux_patch, ws.flux_layout, n, g, lo, Int32.(D); ndrange = size(Jc))
    end
    return J
end
