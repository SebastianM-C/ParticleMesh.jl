# WP-093: the runs of nd_features.jl that its subprocess worker (nd_features_worker.jl, T-1253 at 1
# and 4 threads) and gpu.jl (T-1259) share. After em3d_cases.jl and boundaries_cases.jl (their
# problems and digests).

"The absorbers of the filtered open runs at N = 2: five cells, the guard width of the quadratic shape and two filter passes."
const FILTER_OPEN2 = (("layer", AbsorbingLayer(; cells = 5, strength = 1)), ("pml", PML(; cells = 5, kappa_max = 2, alpha_max = 0.1)))

"The run of T-1255 and T-1256: the dopant in the 3D window with its plasma (`ion_window_3d_small`), `nsteps` steps."
resort_window_3d(::Type{T}; nsteps = 110) where {T} = ion_window_3d_small(T; nsteps)

"""
    features3d_digests(; workgroupsize = nothing, be = CPU()) -> Vector{(label, T, digest)}

T-1253: Reference runs of the 3D step with what WP-093 gave it, per element type, one digest of
every particle array, field and history each: the periodic plasma of [`plasma_step_3d`](@ref) with
one filter pass (three axis passes, an odd count, so the result returns from the scratch) and the
sort from the last order, 12 steps; and [`open_plasma_3d`](@ref), open along y and z behind the
absorbing layer, with one filter pass, 12 steps (particles retired at the four open faces).
"""
function features3d_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{String,DataType,UInt64}[]
    for T in TYPES
        prob = plasma_step_3d(T; nsteps = 12)
        st = EMState(prob, EMNumerics(; workgroupsize, filter = BinomialFilter(1), resort = true); backend = be)
        initialize!(st, prob)
        run!(st, params(prob), prob.nsteps)
        push!(out, ("filter-resort", T, state_digest_3d(st)))
        op = open_plasma_3d(T; nsteps = 12)
        so = EMState(op, EMNumerics(; workgroupsize, absorber = OPEN3_ABSORBERS[1][2], filter = BinomialFilter(1)); backend = be)
        initialize!(so, op)
        run!(so, params(op), op.nsteps)
        push!(out, ("open-filter", T, state_digest_3d(so)))
    end
    return out
end
