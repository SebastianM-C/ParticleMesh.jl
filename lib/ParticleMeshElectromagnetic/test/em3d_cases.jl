# WP-042: the 3D problems of em3d_step.jl and the bitwise digests that its tests and the
# subprocess worker (em3d_worker.jl, T-952 at 1 and 4 threads) share. After em2d_cases.jl
# (`zero3`).

"""
    plasma_step_3d(T; amplitude = 1/10, cells = (8, 7, 6), ppc = (2, 2, 2), nsteps = 20, seed = 7, vth = 1/10) -> EMProblem{3}

Two mobile species that cross cells and the periodic seams of every axis within a few steps,
`plasma_step_2d`'s plasma in 3D: thermal electrons (v_th = `vth` in every component, a
Maxwell–Jüttner load) with the drift (0.3, −0.2, 0.25) and an oblique perturbation of mode
(2, 1, −1), and positive particles of mass 4 (v_th = 0.05) drifting along (−0.2, 0.25, −0.15) with
the mode (1, −1, 1), on a box of 2π × 3π/2 × 5π/4 (three cell sizes), Δt at 0.9 of the Courant
limit. The electrons' `amplitude` may be a dual (T-955).
"""
function plasma_step_3d(::Type{T}; amplitude = one(T) / 10, cells = (8, 7, 6), ppc = (2, 2, 2), nsteps = 20, seed = 7,
                        vth = one(T) / 10) where {T}
    L = (2 * T(π), 3 * T(π) / 2, 5 * T(π) / 4)
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:3))) / 10
    tenth = one(T) / 10
    e = SpeciesDescription{T}(:electron, -one(T), one(T), one(T), T(amplitude), (2, 1, -1), ppc, (3tenth, -2tenth, 5tenth / 2),
                              (T(vth), T(vth), T(vth)), 1)
    p = SpeciesDescription{T}(:positive, one(T), 4 * one(T), one(T) / 2, tenth / 2, (1, -1, 1), ppc, (-2tenth, 5tenth / 2, -3tenth / 2),
                              (tenth / 2, tenth / 2, tenth / 2), 2)
    return EMProblem("plasma-step-3d", L, cells, T(dt), nsteps, (e, p), seed, (2, 1, -1))
end

"The 3D state of `prob` with the `current` (`Esirkepov()` by default), initialized."
function state_3d(prob; mode = Reference(), workgroupsize = nothing, shape = Quadratic, be = CPU(), current = Esirkepov(),
                  node_bins = true)
    st = EMState(prob, EMNumerics(; mode, workgroupsize, shape, current, node_bins); backend = be)
    initialize!(st, prob)
    return st
end

"A hash of the bits of every particle array (every axis, extras included), field, `K` and history of a 3D state."
function state_digest_3d(st::EMState{3})
    h = UInt64(0)
    for s in st.species, a in (s.cell..., s.offset..., s.u..., s.w, s.status, s.count, values(s.extras)...)
        h = hash(bits(a), h)
    end
    for a in (values(st.fields)..., st.K, values(st.recorder.hist)...)
        h = hash(bits(a), h)
    end
    return h
end

"""
    em3d_step_digests(; workgroupsize = nothing, be = CPU()) -> Vector{(label, T, digest)}

T-952: a Reference run of [`plasma_step_3d`](@ref) with the `Esirkepov()` current, 12 steps, per
element type and shape order, and for order 2 with cell bins too (`node_bins = false`), one
digest each.
"""
function em3d_step_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{String,DataType,UInt64}[]
    for T in TYPES, (label, shape, node_bins) in (("order2", Quadratic, true), ("order2-cells", Quadratic, false), ("order1", Linear, true))
        prob = plasma_step_3d(T; nsteps = 12)
        st = state_3d(prob; workgroupsize, be, shape, node_bins)
        run!(st, params(prob), prob.nsteps)
        push!(out, (label, T, state_digest_3d(st)))
    end
    return out
end

# The cold beams of T-953 (em3d_step.jl): a 2D problem and the same problem in 3D, invariant along
# one axis. Both vary along both axes of the plane (mode (1, 1)) and drift in all three components,
# so every in-plane Esirkepov component and the third one carry a current.
const INV3_DRIFT = (1 // 5, 3 // 20, -1 // 8)
const INV3_CELLS = (12, 10)
const INV3_LENGTH = (2, 3 // 2)                    # in π

function beams_plane_2d(::Type{T}; ppc = (3, 3), nsteps = 20, amplitude = one(T) / 20) where {T}
    L = T.(INV3_LENGTH) .* T(π)
    h = L ./ INV3_CELLS
    v = T.(INV3_DRIFT)
    sp = (SpeciesDescription{T}(:a, -one(T), one(T), one(T) / 2, T(amplitude), (1, 1), ppc, v, zero3(T), 1),
          SpeciesDescription{T}(:b, -one(T), one(T), one(T) / 2, zero(T), (1, 1), ppc, .-v, zero3(T), 2))
    return EMProblem("beams-plane-2d", L, INV3_CELLS, inv3_dt(h), nsteps, sp, 1, (1, 1))
end

# The step of both runs: 0.6 of the 3D Courant limit with the invariant axis' cells, which is
# below the 2D limit too.
inv3_dt(h) = 3 * inv(sqrt(sum(abs2, inv.(h)) + inv(2h[1])^2)) / 5

"""
    inv3_axes(axis) -> σ

The cyclic map of the 2D axes (and vector components) 1, 2, 3 onto the 3D ones for a 3D problem
invariant along `axis`: `σ[a]` is the 3D axis of the 2D axis `a`, `σ[3] = axis`. A cyclic
permutation is a rotation, so E, B and J map as vectors with no sign.
"""
inv3_axes(axis) = ntuple(a -> mod1(a + axis, 3), 3)

"""
    beams_3d(T, axis; ppc, nsteps, amplitude) -> EMProblem{3}

[`beams_plane_2d`](@ref) in 3D, invariant along `axis`: the plane's axes on the 3D axes
`inv3_axes(axis)`, the drift's components likewise, and the invariant axis with 6 cells (the
fewest the guards allow) of twice the plane's first cell size and 2 particles per cell.
"""
function beams_3d(::Type{T}, axis; ppc = (3, 3), nsteps = 20, amplitude = one(T) / 20) where {T}
    σ = inv3_axes(axis)
    L2 = T.(INV3_LENGTH) .* T(π)
    h = L2 ./ INV3_CELLS
    put(x2, x3) = (y = Vector{Any}(undef, 3); y[σ[1]] = x2[1]; y[σ[2]] = x2[2]; y[σ[3]] = x3; Tuple(y))
    box = put(L2, 12h[1])
    cells = put(INV3_CELLS, 6)
    p3 = put(ppc, 2)
    m = put((1, 1), 0)
    v2 = T.(INV3_DRIFT)
    v = put((v2[1], v2[2]), v2[3])
    sp = (SpeciesDescription{T}(:a, -one(T), one(T), one(T) / 2, T(amplitude), m, p3, v, zero3(T), 1),
          SpeciesDescription{T}(:b, -one(T), one(T), one(T) / 2, zero(T), m, p3, .-v, zero3(T), 2))
    return EMProblem("beams-3d", box, cells, inv3_dt(h), nsteps, sp, 1, m)
end
