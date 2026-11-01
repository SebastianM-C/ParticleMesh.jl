# CPU timing of the 2D charge-conserving flux deposition (WP-020). No thresholds: the throughput
# budget of A7 is on the GPU.
#
# Part 1, `prototypes` (before the Reference mechanism is fixed): prototype 2D Reference flux
# kernels written here, one per candidate mechanism, all on the engine's per-particle 2D
# `flux_stencil` (so a particle's contributions are the same bits in every mechanism):
#   recompute     one work item per node; the cells whose flux stencil reaches it in increasing
#                 linear index, within a cell its particles in slot order, each particle's whole
#                 stencil recomputed for every node it touches (the N = 1 design, #93);
#   scratch       one work item per particle writes its contributions into a per-particle scratch
#                 buffer, then one work item per node reduces them in the order of `recompute`
#                 (bitwise equal to it);
#   patch         one work item per cell sums its particles in slot order into a fixed-size patch
#                 per component, then one work item per node adds the patches of the cells that
#                 reach it in increasing linear cell order; four variants: the patch accumulated
#                 in global memory (`patch_global`) or in a register tuple written once
#                 (`patch_tuple`, bitwise equal to `patch_global`), each with the patch stored
#                 stencil-major (`[k, cell]`) or cell-major (`[cell, k]`, suffix `_cm`);
#   *_nn          `recompute` and `patch_tuple` with the particles binned by their nearest node
#                 (`floor(x + 1/2)` per axis, host-built binning) instead of by cell, at order 2
#                 with ρ at (Node, Node) only, where it shrinks a node's reach from 6 to 5 cells
#                 per axis; measured only (the sort key is contract 9's).
# next to the engine's 2D Fast flux and the engine's 2D `deposit!` (Reference and Fast) on the
# same particles. Run once per thread count:
#   julia +1.13 --project=scripts -t 1 scripts/engine2d/flux_timing.jl prototypes
#   julia +1.13 --project=scripts -t 32 scripts/engine2d/flux_timing.jl prototypes
#
# Part 2, `ported` (after the mechanism is in the engine): the engine's 2D `deposit_flux!` in both
# modes against the engine's 2D `deposit!` on the same particles and the engine's 1D Reference and
# Fast flux at the same particle count, and the scratch bytes per particle and per cell of every
# measured mechanism per element type:
#   julia +1.13 --project=scripts -t 1 scripts/engine2d/flux_timing.jl ported
#   julia +1.13 --project=scripts -t 32 scripts/engine2d/flux_timing.jl ported
include(joinpath(@__DIR__, "common.jl"))
const PMC = ParticleMesh.ParticleMeshCore
using .PMC: shape_weights, pick, ALIVE, Stagger, Shape, Species, Workspace, sort_by_cell!, deposit!, deposit_flux!, flux_extras,
            flux_guard_width, Reference, Fast, ChargeWeight, PeriodicGrid, field, capacity, Node, Center
using KernelAbstractions: @kernel, @index, @Const
using ForwardDiff: Dual

const SIZES = ((16, 12), (64, 48))          # cells per axis
const PPC = 64                               # particles per cell, offsets drawn at random
const MAXSTEP = 1 / 2                        # displacements in (-1/2, 1/2) cells per axis
const STAGGER_CASES = (("Node,Node", (Node(), Node())), ("Center,Center", (Center(), Center())))

"q·w·u₂, the module's quantity for Jz in these runs (u₂ is drawn in [-1, 1))."
struct QZ end
@inline (::QZ)(s, i) = s.q * s.w[i] * s.u[2][i]

# ---------------------------------------------------------------------------------------------
# Stencils: the engine's (frame: the particle's cell) and the nearest-node one (frame: its bin)

"The engine's 2D flux stencil in the cell frame."
struct CellStencil{SH,ST,SX,Q}
    shape::SH
    st::ST
    n::NTuple{2,Int32}
    sx::SX
    qz::Q
end
@inline (k::CellStencil)(s, i) = PMC.flux_stencil(s, i, k.shape, k.st, k.n, k.sx, k.qz)

"""
One axis of an order-2 flux stencil at a `Node()` stagger in the frame of the particle's
nearest node `b = c + floor(δ + 1/2)` (wrapped as the bin), on the nodes `b - 2:b + 2`.
"""
@inline function axis_nn(shape::Shape{2}, c::Int32, δ, c0::Int32, δ0, n::Int32)
    f1, w1 = shape_weights(shape, Node(), δ)                 # nodes c + f1 + 0:2, nearest c + f1 + 1
    f0, w0 = shape_weights(shape, Node(), δ0)
    b1 = c + f1 + Int32(1)
    b0 = c0 + f0 + Int32(1)
    e = mod(b0 - b1 + Int32(1), n) - Int32(1)
    Wn = ntuple(m -> pick(w1, Int32(m - 2)), Val(5))         # node b1 - 2 + m - 1 is weight index m - 2
    Wo = ntuple(m -> pick(w0, Int32(m - 2) - e), Val(5))
    return (Wo, Wn)
end

"The 2D flux stencil of order 2 at (Node, Node) in the nearest-node frame, the engine's formulas."
struct BinStencil{SH,SX,Q}
    shape::SH
    n::NTuple{2,Int32}
    sx::SX
    qz::Q
end
@inline function (k::BinStencil)(s, i)
    @inbounds begin
        Wo1, Wn1 = axis_nn(k.shape, s.cell[1][i], s.offset[1][i], s.extras.old_cell_1[i], s.extras.old_offset_1[i], k.n[1])
        Wo2, Wn2 = axis_nn(k.shape, s.cell[2][i], s.offset[2][i], s.extras.old_cell_2[i], s.extras.old_offset_2[i], k.n[2])
        qw = s.q * s.w[i]
    end
    D1, D2 = map(-, Wn1, Wo1), map(-, Wn2, Wo2)
    A1, A2 = map((a, b) -> (a + b) / 2, Wo1, Wn1), map((a, b) -> (a + b) / 2, Wo2, Wn2)
    P1, P2 = PMC.prefix_sums(D1), PMC.prefix_sums(D2)
    qws1, qws2 = qw * k.sx[1], qw * k.sx[2]
    qz = k.qz(s, i)
    T = typeof(qw)
    jx = PMC.outer((p, a) -> -qws1 * (p * a), P1, A2)
    jy = PMC.outer((a, p) -> -qws2 * (a * p), A1, P2)
    jz = PMC.outer2(((a1, d1), (a2, d2)) -> qz * (a1 * a2 + d1 * d2 / T(12)), (A1, D1), (A2, D2))
    return ((Int32(-2), Int32(-2)), (5, 5), (jx, jy, jz))
end

# ---------------------------------------------------------------------------------------------
# Prototype kernels (part 1). `lo`, `hi` are the per-axis node reach relative to a segment's
# cell (or bin), `M` the stencil width per axis. No float literal; `workgroupsize` is the default.

@kernel function proto_recompute!(Jx, Jy, Jz, s, @Const(seg_start), @Const(seg_stop), stencil, n::NTuple{2,Int32}, g::Int32,
                                  lo::NTuple{2,Int32}, hi::NTuple{2,Int32})
    I = @index(Global, Cartesian)
    T = eltype(Jx)
    j1, j2 = Int32(I[1]) - g, Int32(I[2]) - g
    ax, ay, az = zero(T), zero(T), zero(T)
    @inbounds for c2 in max(Int32(1), j2 - hi[2]):min(n[2], j2 - lo[2]), c1 in max(Int32(1), j1 - hi[1]):min(n[1], j1 - lo[1])
        L = c1 + (c2 - Int32(1)) * n[1]
        for i in seg_start[L]:seg_stop[L]
            l, M, (jx, jy, jz) = stencil(s, i)
            k = (j1 - c1 - l[1]) + (j2 - c2 - l[2]) * Int32(M[1]) + Int32(1)
            ax += jx[k]
            ay += jy[k]
            az += jz[k]
        end
    end
    @inbounds begin
        Jx[I] += ax
        Jy[I] += ay
        Jz[I] += az
    end
end

@kernel function proto_scratch_fill!(scr, s, stencil)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        _, _, (jx, jy, jz) = stencil(s, i)
        K = length(jx)
        for k in 1:K
            scr[k, i] = jx[k]
            scr[K + k, i] = jy[k]
            scr[2K + k, i] = jz[k]
        end
    end
end

@kernel function proto_scratch_reduce!(Jx, Jy, Jz, @Const(scr), @Const(seg_start), @Const(seg_stop), n::NTuple{2,Int32}, g::Int32,
                                       lo::NTuple{2,Int32}, hi::NTuple{2,Int32}, M::NTuple{2,Int32})
    I = @index(Global, Cartesian)
    T = eltype(Jx)
    K = M[1] * M[2]
    j1, j2 = Int32(I[1]) - g, Int32(I[2]) - g
    ax, ay, az = zero(T), zero(T), zero(T)
    @inbounds for c2 in max(Int32(1), j2 - hi[2]):min(n[2], j2 - lo[2]), c1 in max(Int32(1), j1 - hi[1]):min(n[1], j1 - lo[1])
        L = c1 + (c2 - Int32(1)) * n[1]
        k = (j1 - c1 - lo[1]) + (j2 - c2 - lo[2]) * M[1] + Int32(1)
        for i in seg_start[L]:seg_stop[L]
            ax += scr[k, i]
            ay += scr[K + k, i]
            az += scr[2K + k, i]
        end
    end
    @inbounds begin
        Jx[I] += ax
        Jy[I] += ay
        Jz[I] += az
    end
end

@inline pidx(::Val{false}, kk, L) = (kk, L)        # stencil-major: patch[k, cell]
@inline pidx(::Val{true}, kk, L) = (L, kk)         # cell-major: patch[cell, k]

@kernel function proto_patch_global!(patch, s, @Const(seg_start), @Const(seg_stop), stencil, ::Val{K}, cm) where {K}
    L = Int32(@index(Global, Linear))
    T = eltype(patch)
    @inbounds for kk in 1:(3K)
        patch[pidx(cm, kk, L)...] = zero(T)
    end
    @inbounds for i in seg_start[L]:seg_stop[L]
        _, _, (jx, jy, jz) = stencil(s, i)
        for k in 1:K
            patch[pidx(cm, k, L)...] += jx[k]
            patch[pidx(cm, K + k, L)...] += jy[k]
            patch[pidx(cm, 2K + k, L)...] += jz[k]
        end
    end
end

@kernel function proto_patch_tuple!(patch, s, @Const(seg_start), @Const(seg_stop), stencil, ::Val{K}, cm) where {K}
    L = Int32(@index(Global, Linear))
    T = eltype(patch)
    ax = ntuple(_ -> zero(T), Val(K))
    ay, az = ax, ax
    @inbounds for i in seg_start[L]:seg_stop[L]
        _, _, (jx, jy, jz) = stencil(s, i)
        ax = ntuple(k -> ax[k] + jx[k], Val(K))
        ay = ntuple(k -> ay[k] + jy[k], Val(K))
        az = ntuple(k -> az[k] + jz[k], Val(K))
    end
    @inbounds for k in 1:K
        patch[pidx(cm, k, L)...] = ax[k]
        patch[pidx(cm, K + k, L)...] = ay[k]
        patch[pidx(cm, 2K + k, L)...] = az[k]
    end
end

@kernel function proto_patch_reduce!(Jx, Jy, Jz, @Const(patch), n::NTuple{2,Int32}, g::Int32, lo::NTuple{2,Int32},
                                     hi::NTuple{2,Int32}, M::NTuple{2,Int32}, cm)
    I = @index(Global, Cartesian)
    T = eltype(Jx)
    K = M[1] * M[2]
    j1, j2 = Int32(I[1]) - g, Int32(I[2]) - g
    ax, ay, az = zero(T), zero(T), zero(T)
    @inbounds for c2 in max(Int32(1), j2 - hi[2]):min(n[2], j2 - lo[2]), c1 in max(Int32(1), j1 - hi[1]):min(n[1], j1 - lo[1])
        L = c1 + (c2 - Int32(1)) * n[1]
        k = (j1 - c1 - lo[1]) + (j2 - c2 - lo[2]) * M[1] + Int32(1)
        ax += patch[pidx(cm, k, L)...]
        ay += patch[pidx(cm, K + k, L)...]
        az += patch[pidx(cm, 2K + k, L)...]
    end
    @inbounds begin
        Jx[I] += ax
        Jy[I] += ay
        Jz[I] += az
    end
end

# ---------------------------------------------------------------------------------------------
# Particles

"""
    moved_sorted_2d(T, n, ppc; seed) -> (s, grid, ws)

`ppc` particles per cell at random offsets with weights in [1/2, 3/2) and u₂ in [-1, 1), the old
position stored in the extras of `flux_extras`, every particle then moved by a displacement in
(-MAXSTEP, MAXSTEP) cells per axis through the engine's `advance`, and sorted by the engine. The
grid has the guards of the widest flux stencil (order 2 at `Node()`), cells of 1 × 1.
"""
function moved_sorted_2d(::Type{T}, n, ppc; seed = 1, flux = nothing) where {T}
    ncell = prod(n)
    grid = PeriodicGrid(n, (one(T), one(T)), 3)
    s = Species(CPU(), T, ncell * ppc; q = -1, m = 1, N = 2, extras = flux_extras(T, 2))
    state = UInt64(seed)
    nextu() = (state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(state >> 11) / 2.0^53)
    nn = Int32.(n)
    for L in 1:ncell, k in 1:ppc
        i = (L - 1) * ppc + k
        c = (Int32(mod1(L, n[1])), Int32((L - 1) ÷ n[1] + 1))
        δ = (T(nextu()), T(nextu()))
        s.extras.old_cell_1[i], s.extras.old_cell_2[i] = c
        s.extras.old_offset_1[i], s.extras.old_offset_2[i] = δ
        s.w[i] = T(1 / 2 + nextu())
        s.u[2][i] = T(2nextu() - 1)
        s.status[i] = ALIVE
        c1, δ1 = PMC.advance(c, δ, (T(MAXSTEP * (2nextu() - 1)), T(MAXSTEP * (2nextu() - 1))), nn)
        s.cell[1][i], s.cell[2][i] = c1
        s.offset[1][i], s.offset[2][i] = δ1
    end
    fill!(s.count, Int32(capacity(s)))
    ws = flux === nothing ? Workspace(s, grid) : Workspace(s, grid; flux)
    sort_by_cell!(s, grid, ws)
    return s, grid, ws
end

"""
    nn_binned(s, n) -> (sb, seg_start, seg_stop)

A copy of the sorted species `s` reordered by its nearest-node bin per axis (`c + floor(δ + 1/2)`,
wrapped), then slot, with the per-bin segment bounds, built on the host (order 2 at `Node()`).
"""
function nn_binned(s, n)
    T = eltype(s)
    np = capacity(s)
    bin(c, δ, m) = mod1(c + Int(floor(δ + one(T) / 2)), m)
    Lb = [bin(s.cell[1][i], s.offset[1][i], n[1]) + (bin(s.cell[2][i], s.offset[2][i], n[2]) - 1) * n[1] for i in 1:np]
    perm = sortperm(collect(zip(Lb, 1:np)))
    sb = Species(CPU(), T, np; q = s.q, m = s.m, N = 2, extras = flux_extras(T, 2))
    for (dst, src) in ((sb.cell[1], s.cell[1]), (sb.cell[2], s.cell[2]), (sb.offset[1], s.offset[1]), (sb.offset[2], s.offset[2]),
                       (sb.w, s.w), (sb.u[2], s.u[2]), (sb.status, s.status), (sb.extras.old_cell_1, s.extras.old_cell_1),
                       (sb.extras.old_cell_2, s.extras.old_cell_2), (sb.extras.old_offset_1, s.extras.old_offset_1),
                       (sb.extras.old_offset_2, s.extras.old_offset_2))
        dst .= src[perm]
    end
    fill!(sb.count, Int32(np))
    seg_start, seg_stop = ones(Int32, prod(n)), zeros(Int32, prod(n))
    Ls = Lb[perm]
    for i in 1:np
        (i == 1 || Ls[i - 1] != Ls[i]) && (seg_start[Ls[i]] = i)
        (i == np || Ls[i + 1] != Ls[i]) && (seg_stop[Ls[i]] = i)
    end
    return sb, seg_start, seg_stop
end

# ---------------------------------------------------------------------------------------------
# Timing

"""
    timed_adaptive(f; reps = 7, target = 0.1) -> (; median, min, max)

Seconds per call of `f()`: one warm-up call, then `reps` samples, each the mean over enough calls
to take about `target` seconds (at least one), so the slow mechanisms do not dominate the run.
"""
function timed_adaptive(f; reps = 7, target = 0.1)
    t0 = @elapsed f()
    inner = clamp(round(Int, target / max(t0, 1e-9)), 1, 50)
    samples = map(1:reps) do _
        (@elapsed for _ in 1:inner
             f()
         end) / inner
    end
    return (; median = median(samples), min = minimum(samples), max = maximum(samples), inner)
end

valof(::Val{V}) where {V} = V
bitsof(J) = reinterpret(UInt8, vec(collect(J)))
same(a, b) = all(bitsof(x) == bitsof(y) for (x, y) in zip(a, b))
maxrel(a, b) = maximum(maximum(abs.(x .- y)) / maximum(abs, y) for (x, y) in zip(a, b))
zero3!(J) = foreach(x -> fill!(x, zero(eltype(x))), J)

function row(T, n, np, P, label, method, r; extra...)
    return Dict{String,Any}("T" => string(T), "cells" => collect(n), "particles" => np, "order" => P, "stagger" => label,
                            "method" => method, "ns_per_particle" => 1e9 * r.median / np, "ns_per_particle_min" => 1e9 * r.min / np,
                            "ns_per_particle_max" => 1e9 * r.max / np, "calls_per_sample" => r.inner, extra...)
end

function prototype_rows()
    rows = Dict{String,Any}[]
    be = CPU()
    for T in (Float32, Float64), n in SIZES, P in (1, 2), (label, st) in STAGGER_CASES
        shape = Shape{P}()
        s, grid, ws = moved_sorted_2d(T, n, PPC)
        np = capacity(s)
        g = Int32(PMC.guards(grid))
        nn = Int32.(n)
        dt = T(1 // 4)
        sx = map(h -> h / dt, PMC.cellsize(grid))
        stencil = CellStencil(shape, st, nn, sx, QZ())
        r = map(x -> PMC.flux_reach(shape, x), st)
        lo, hi = (r[1][1], r[2][1]), (r[1][2], r[2][2])
        M = Int32.(map(x -> valof(PMC.flux_width(shape, x)), st))
        K = Int(M[1] * M[2])
        dims = size(field(be, T, grid))
        J() = (KernelAbstractions.zeros(be, T, dims...), KernelAbstractions.zeros(be, T, dims...), KernelAbstractions.zeros(be, T, dims...))
        results = Dict{String,Any}()
        outs = Dict{String,Any}()
        # recompute
        Jr = J()
        run_r() = proto_recompute!(be)(Jr..., s, ws.seg_start, ws.seg_stop, stencil, nn, g, lo, hi; ndrange = dims)
        run_r(); outs["recompute"] = map(copy, Jr)
        results["recompute"] = timed_adaptive(run_r)
        # scratch
        scr = KernelAbstractions.zeros(be, T, 3K, np)
        Js = J()
        run_s() = (proto_scratch_fill!(be)(scr, s, stencil; ndrange = np);
                   proto_scratch_reduce!(be)(Js..., scr, ws.seg_start, ws.seg_stop, nn, g, lo, hi, M; ndrange = dims))
        run_s(); outs["scratch"] = map(copy, Js)
        results["scratch"] = timed_adaptive(run_s)
        # patches
        for (name, kern, cm) in (("patch_global", proto_patch_global!, Val(false)), ("patch_tuple", proto_patch_tuple!, Val(false)),
                                 ("patch_global_cm", proto_patch_global!, Val(true)), ("patch_tuple_cm", proto_patch_tuple!, Val(true)))
            patch = cm === Val(true) ? KernelAbstractions.zeros(be, T, prod(n), 3K) : KernelAbstractions.zeros(be, T, 3K, prod(n))
            Jp = J()
            run_p() = (kern(be)(patch, s, ws.seg_start, ws.seg_stop, stencil, Val(K), cm; ndrange = prod(n));
                       proto_patch_reduce!(be)(Jp..., patch, nn, g, lo, hi, M, cm; ndrange = dims))
            run_p(); outs[name] = map(copy, Jp)
            results[name] = timed_adaptive(run_p)
        end
        # the engine's Fast flux and deposit! on the same particles
        JF = J()
        run_F() = deposit_flux!(Fast(), JF..., s, shape, st, grid, ws; dt, quantity_z = QZ())
        run_F(); outs["fast"] = map(copy, JF)
        results["fast"] = timed_adaptive(run_F)
        ρ = field(be, T, grid)
        results["deposit_ref"] = timed_adaptive(() -> deposit!(Reference(), ρ, s, shape, st, grid, ws))
        results["deposit_fast"] = timed_adaptive(() -> deposit!(Fast(), ρ, s, shape, st, grid, ws))
        # nearest-node binning, order 2 at (Node, Node)
        if P == 2 && st == (Node(), Node())
            sb, bs, bt = nn_binned(s, n)
            bst = BinStencil(shape, nn, sx, QZ())
            lob, hib = (Int32(-2), Int32(-2)), (Int32(2), Int32(2))
            Jn = J()
            run_n() = proto_recompute!(be)(Jn..., sb, bs, bt, bst, nn, g, lob, hib; ndrange = dims)
            run_n(); outs["recompute_nn"] = map(copy, Jn)
            results["recompute_nn"] = timed_adaptive(run_n)
            patch = KernelAbstractions.zeros(be, T, 75, prod(n))
            Jq = J()
            run_q() = (proto_patch_tuple!(be)(patch, sb, bs, bt, bst, Val(25), Val(false); ndrange = prod(n));
                       proto_patch_reduce!(be)(Jq..., patch, nn, g, lob, hib, (Int32(5), Int32(5)), Val(false); ndrange = dims))
            run_q(); outs["patch_tuple_nn"] = map(copy, Jq)
            results["patch_tuple_nn"] = timed_adaptive(run_q)
        end
        ref = outs["recompute"]
        checks = Dict{String,Any}(
            "scratch_bitwise_recompute" => same(outs["scratch"], ref),
            "patches_bitwise_equal" => all(same(outs[k], outs["patch_global"]) for k in ("patch_tuple", "patch_global_cm", "patch_tuple_cm")),
            "patch_bitwise_recompute" => same(outs["patch_global"], ref),
            "patch_maxrel_recompute" => maxrel(outs["patch_global"], ref),
            "fast_maxrel_recompute" => maxrel(outs["fast"], ref))
        haskey(outs, "recompute_nn") && (checks["nn_maxrel_recompute"] = maxrel(outs["recompute_nn"], ref);
                                         checks["patch_nn_maxrel_recompute"] = maxrel(outs["patch_tuple_nn"], ref))
        for (k, v) in results
            push!(rows, row(T, n, np, P, label, k, v; checks...))
        end
        println(T, " ", n, " P=", P, " ", label, ": ",
                join(("$k=$(round(1e9 * results[k].median / np, sigdigits = 3))" for k in sort(collect(keys(results)))), " "), " ", checks)
        flush(stdout)
    end
    return rows
end

const PROTO_METHODS = ("recompute", "scratch", "patch_global", "patch_tuple", "patch_global_cm", "patch_tuple_cm", "recompute_nn",
                       "patch_tuple_nn", "fast", "deposit_ref", "deposit_fast")
const PORTED_METHODS = ("flux_ref", "flux_fast", "deposit_ref", "deposit_fast", "flux1d_ref", "flux1d_fast")
const METHOD_LABEL = Dict("recompute" => "flux Ref, per-node recompute", "scratch" => "flux Ref, per-particle scratch",
                          "patch_global" => "flux Ref, per-cell patch (global)", "patch_tuple" => "flux Ref, per-cell patch (tuple)",
                          "patch_global_cm" => "flux Ref, patch (global, cell-major)", "patch_tuple_cm" => "flux Ref, patch (tuple, cell-major)",
                          "recompute_nn" => "flux Ref, recompute, nearest-node bins", "patch_tuple_nn" => "flux Ref, patch, nearest-node bins",
                          "fast" => "flux Fast (engine)", "deposit_ref" => "deposit! Ref (engine)", "deposit_fast" => "deposit! Fast (engine)",
                          "flux_ref" => "flux Ref (engine)", "flux_fast" => "flux Fast (engine)", "flux1d_ref" => "1D flux Ref (engine)",
                          "flux1d_fast" => "1D flux Fast (engine)")

"Grouped bars on a log axis: one panel per grid size, one group per (T, order, stagger), one bar per method present."
function bars_figure(rows, methods, title)
    fig = Figure(size = (1400, 800))
    cases = [(T, P, l) for T in ("Float32", "Float64") for P in (1, 2) for (l, _) in STAGGER_CASES]
    colors = vcat(CairoMakie.Makie.wong_colors(), [:gray40, :black, :olive, :purple])
    for (r, n) in enumerate(SIZES)
        ax = Axis(fig[r, 1]; ylabel = "ns per particle", yscale = log10, title = "$title, $(n[1]) × $(n[2]) cells, $PPC per cell",
                  xticks = (1:length(cases), ["$(T == "Float32" ? "F32" : "F64") P$P $l" for (T, P, l) in cases]), xticklabelrotation = π / 10)
        w = 0.85 / length(methods)
        for (m, method) in enumerate(methods)
            xs, ys, lo, hi = Float64[], Float64[], Float64[], Float64[]
            for (c, (T, P, l)) in enumerate(cases)
                sel = filter(x -> x["T"] == T && x["order"] == P && x["stagger"] == l && x["cells"] == collect(n) && x["method"] == method, rows)
                isempty(sel) && continue
                x = only(sel)
                push!(xs, c + (m - (length(methods) + 1) / 2) * w)
                push!(ys, x["ns_per_particle"]); push!(lo, x["ns_per_particle_min"]); push!(hi, x["ns_per_particle_max"])
            end
            isempty(xs) && continue
            barplot!(ax, xs, ys; width = w, color = colors[m], label = METHOD_LABEL[method], fillto = 0.5)
            rangebars!(ax, xs, lo, hi; color = :black)
        end
        r == 1 && axislegend(ax; position = :lt, nbanks = 3, labelsize = 10)
    end
    return fig
end

const TIMER = "median of 7 samples after one warm-up call, each sample the mean of enough calls for about 0.1 s (1 to 50)"

function prototypes()
    nt = Threads.nthreads()
    dir = run_dir("flux-prototypes-t$nt", @__FILE__; work_package = "WP-020",
                  params = Dict{String,Any}("sizes" => [collect(n) for n in SIZES], "ppc" => PPC, "maxstep" => MAXSTEP, "timer" => TIMER))
    rows = prototype_rows()
    fig = bars_figure(rows, PROTO_METHODS, "2D flux mechanisms, $nt thread(s), $(cpu_model())")
    save_plot(dir, "flux_prototypes_t$nt", fig, Dict("rows" => rows, "threads" => nt, "cpu" => cpu_model(), "ppc" => PPC))
    println("prototypes: outputs in ", relpath(dir, REPO))
    return dir
end

# ---------------------------------------------------------------------------------------------
# Part 2: the engine

"The 1D comparison: the same particle count on prod(n) cells of 1, moved and sorted, with the 1D flux extras."
function moved_sorted_1d(::Type{T}, ncell, ppc; seed = 1) where {T}
    grid = PeriodicGrid(ncell, one(T), 3)
    s = Species(CPU(), T, ncell * ppc; q = -1, m = 1, extras = flux_extras(T, 1))
    state = UInt64(seed)
    nextu() = (state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(state >> 11) / 2.0^53)
    for i in 1:capacity(s)
        c, δ = Int32((i - 1) ÷ ppc + 1), T(nextu())
        s.extras.old_cell[i], s.extras.old_offset[i] = c, δ
        s.w[i], s.u[2][i], s.u[3][i] = T(1 / 2 + nextu()), T(2nextu() - 1), T(2nextu() - 1)
        s.status[i] = ALIVE
        s.cell[1][i], s.offset[1][i] = PMC.advance(c, δ, T(MAXSTEP * (2nextu() - 1)), Int32(ncell))
    end
    fill!(s.count, Int32(capacity(s)))
    ws = Workspace(s, grid)
    sort_by_cell!(s, grid, ws)
    return s, grid, ws
end
struct QY1 end
@inline (::QY1)(s, i) = s.q * s.w[i] * s.u[2][i]
struct QZ1 end
@inline (::QZ1)(s, i) = s.q * s.w[i] * s.u[3][i]

function ported_rows()
    rows = Dict{String,Any}[]
    be = CPU()
    for T in (Float32, Float64), n in SIZES, P in (1, 2), (label, st) in STAGGER_CASES
        shape = Shape{P}()
        s, grid, ws = moved_sorted_2d(T, n, PPC; flux = (shape, st))
        np = capacity(s)
        dt = T(1 // 4)
        JR = ntuple(_ -> field(be, T, grid), 3)
        JF = ntuple(_ -> field(be, T, grid), 3)
        ρ = field(be, T, grid)
        results = Dict{String,Any}()
        results["flux_ref"] = timed_adaptive(() -> deposit_flux!(Reference(), JR..., s, shape, st, grid, ws; dt, quantity_z = QZ()))
        results["flux_fast"] = timed_adaptive(() -> deposit_flux!(Fast(), JF..., s, shape, st, grid, ws; dt, quantity_z = QZ()))
        results["deposit_ref"] = timed_adaptive(() -> deposit!(Reference(), ρ, s, shape, st, grid, ws))
        results["deposit_fast"] = timed_adaptive(() -> deposit!(Fast(), ρ, s, shape, st, grid, ws))
        s1, g1, ws1 = moved_sorted_1d(T, prod(n), PPC)
        J1 = ntuple(_ -> field(be, T, g1), 3)
        results["flux1d_ref"] = timed_adaptive(() -> deposit_flux!(Reference(), J1..., s1, shape, st[1], g1, ws1; dt, quantity_y = QY1(),
                                                                   quantity_z = QZ1()))
        results["flux1d_fast"] = timed_adaptive(() -> deposit_flux!(Fast(), J1..., s1, shape, st[1], g1, ws1; dt, quantity_y = QY1(),
                                                                    quantity_z = QZ1()))
        for (k, v) in results
            push!(rows, row(T, n, np, P, label, k, v))
        end
        println(T, " ", n, " P=", P, " ", label, ": ",
                join(("$k=$(round(1e9 * results[k].median / np, sigdigits = 3))" for k in PORTED_METHODS), " "))
        flush(stdout)
    end
    return rows
end

"""
Scratch bytes of every measured mechanism, per particle and per cell, per element type, at ρ
(Node, Node) and (Center, Center), orders 1 and 2, next to the engine `Workspace` it adds to.
"""
function scratch_rows()
    rows = Dict{String,Any}[]
    for T in (Float32, Float64, Dual{Nothing,Float32,3}), P in (1, 2), (label, st) in STAGGER_CASES
        shape = Shape{P}()
        M = map(x -> valof(PMC.flux_width(shape, x)), st)
        K = 3 * prod(M)
        b = sizeof(T)
        n, ppc = (64, 48), 16
        s = Species(CPU(), T, prod(n) * ppc; q = -1, m = 1, N = 2, extras = flux_extras(T, 2))
        grid = PeriodicGrid(n, (1.0f0, 1.0f0), 3)
        base = PMC.scratch_bytes(Workspace(s, grid))
        withflux = PMC.scratch_bytes(Workspace(s, grid; flux = (shape, st)))
        push!(rows, Dict{String,Any}("T" => T === Dual{Nothing,Float32,3} ? "Dual{Float32,3}" : string(T), "order" => P, "stagger" => label,
                                     "stencil" => collect(M), "values_per_particle_or_cell" => K, "bytes_per_value" => b,
                                     "recompute_bytes_per_particle" => 0, "scratch_bytes_per_particle" => K * b,
                                     "patch_bytes_per_cell" => K * b, "patch_nn_bytes_per_cell" => (P == 2 && label == "Node,Node") ? 75b : nothing,
                                     "engine_workspace_bytes" => base, "engine_workspace_with_flux_bytes" => withflux,
                                     "engine_flux_bytes_per_cell" => (withflux - base) / prod(n),
                                     "workspace_case" => "64 × 48 cells, 16 per cell, extras of flux_extras(T, 2)"))
    end
    return rows
end

function ported()
    nt = Threads.nthreads()
    dir = run_dir("flux-ported-t$nt", @__FILE__; work_package = "WP-020",
                  params = Dict{String,Any}("sizes" => [collect(n) for n in SIZES], "ppc" => PPC, "maxstep" => MAXSTEP, "timer" => TIMER))
    rows = ported_rows()
    fig = bars_figure(rows, PORTED_METHODS, "2D flux in the engine, $nt thread(s), $(cpu_model())")
    save_plot(dir, "flux_ported_t$nt", fig, Dict("rows" => rows, "threads" => nt, "cpu" => cpu_model(), "ppc" => PPC))
    open(io -> JSON.print(io, Dict("rows" => scratch_rows()), 2), joinpath(dir, "flux_scratch.json"), "w")
    println("ported: outputs in ", relpath(dir, REPO))
    return dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    part = isempty(ARGS) ? "prototypes" : ARGS[1]
    part == "prototypes" ? prototypes() : part == "ported" ? ported() : error("unknown part $part")
end
