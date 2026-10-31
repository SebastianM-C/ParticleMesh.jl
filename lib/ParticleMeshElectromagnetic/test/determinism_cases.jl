# The short runs of T-218, shared by the test and its subprocess worker: every case, every
# element type of the A8 matrix, Reference mode, 64 particles per cell, one time unit (93 to 131
# steps, enough for particles to cross cells and the seam).

const DETERMINISM_T_END = 1

function determinism_digests(; workgroupsize = nothing)
    out = Tuple{String,DataType,UInt64}[]
    for name in ("landau", "two-stream", "weibel"), T in TYPES
        st, p, prob, _ = case_state(name, T; mode = Reference(), workgroupsize, ppc = 64, t_end = DETERMINISM_T_END)
        run!(st, p, prob.nsteps)
        push!(out, (name, T, state_digest(st)))
    end
    return out
end
