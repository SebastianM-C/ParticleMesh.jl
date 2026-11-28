# WP-088 (#566): where the reverse pass of the ionization kernel's birth branch goes wrong. The
# state of trapped_pieces.jl at step kmid + 1 after the step's sharpen, then the reverse pass of a
# kernel variant with an indicator seed, against the adjoint the kernel's code implies, per atom:
#
#   seed pending = 1:  adj pending = 1 on an alive atom without a birth, 0 on one with a birth; adj a.w = 0
#   seed emitted = 1:  adj emitted = 1; adj pending = 1 on a birth, 0 elsewhere (emitted += pending)
#   seed e.w = 1:      adj a.w = pending + Σ out (the electron's w over a.w) and adj pending = a.w on a
#                      birth, 0 elsewhere; adj e.w = 0 on the birth slots, 1 on the others
#
#     PM_BACKEND=cuda PM_ENZYME_PATCH=scripts/enzyme/active_reduce.jl JULIA_LOAD_PATH=@:<repo>:@stdlib \
#       julia +1.13 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/spike/ionize_probe.jl [variants] [tmid] [ppl] [ny]
#
# variants: comma-separated pieces of trapped_pieces.jl (ionize_k, ionize_v1, ionize_v2), default all three
variants = Symbol.(split(get(ARGS, 1, "ionize_k,ionize_v1,ionize_v2"), ","))
let a = copy(ARGS)
    empty!(ARGS)
    append!(ARGS, ["ionize_k", get(a, 2, "180"), get(a, 3, "32"), get(a, 4, "2"), "lib"])
end
ENV["PM_CHECK_CTX"] = "1"
include(joinpath(@__DIR__, "trapped_pieces.jl"))

host(x) = Array(x)
st0 = deepcopy(st)
fr0 = PEM.ionization_frame(st0)
cap = PEM.capacity(fr0.electrons)
c0 = Int(host(fr0.electrons.count)[1])
alive = host(fr0.atoms.status) .== PEM.ALIVE
println("probe: electrons count $c0 of $cap; alive atoms $(count(alive)) of $(length(alive))")

# the array of state `s` that is `f(frame of s)`, and its shadow in `ds`, by identity
function pick(s, ds, f)
    L, Ld = leaves(s), leaves(ds)
    x = f(PEM.ionization_frame(s))
    k = findfirst(v -> v === x, L)
    k === nothing && error("no leaf for the array")
    return k, Ld[k]
end
const PICKS = (pending = fr -> fr.atoms.extras.pending, emitted = fr -> fr.atoms.extras.emitted, aw = fr -> fr.atoms.w, ew = fr -> fr.electrons.w)

function summarize(name, got, want, mask)
    idx = findall(mask)
    isempty(idx) && return @printf("    %-28s (no atoms)\n", name)
    d = abs.(got[idx] .- want[idx])
    tol = 1e-4 .* max.(abs.(want[idx]), 1)
    bad = idx[d .> tol]
    @printf("    %-28s %7d atoms, %7d off; max |got − want| %.3e", name, length(idx), length(bad), maximum(d))
    if !isempty(bad)
        print("; first off (atom: got, want)")
        for j in bad[1:min(end, 4)]
            @printf(" %d: %+.4e, %+.4e;", j, got[j], want[j])
        end
    end
    println()
end

for P in variants
    # the primal: which atoms give birth, pending + Σ out of each
    sp = deepcopy(st0)
    piece!(sp, p, kmid + 1, Val(P)); sync()
    frp = PEM.ionization_frame(sp)
    estat = host(frp.electrons.status)
    slots = c0 .+ (1:length(alive))
    born = [alive[i] && slots[i] <= cap && estat[slots[i]] == PEM.ALIVE && host(fr0.electrons.status)[slots[i]] != PEM.ALIVE for i in eachindex(alive)]
    nb = alive .& .!born
    aw = host(fr0.atoms.w)
    ewp = host(frp.electrons.w)
    ptot = [born[i] ? ewp[slots[i]] / aw[i] : 0.0f0 for i in eachindex(alive)]
    println("variant $P: $(count(born)) births, $(count(nb)) alive atoms without one")
    for seed in (:pending, :emitted, :ew)
        s = deepcopy(st0)
        ds = Enzyme.make_zero(s)
        _, sh = pick(s, ds, PICKS[seed])
        fill!(sh, 1)
        t = @elapsed Enzyme.autodiff(set_runtime_activity(Reverse), piece!, Const, Duplicated(s, ds), Active(p), Const(kmid + 1), Const(Val(P)))
        sync()
        adj(f) = host(pick(s, ds, f)[2])
        pend, emit, adjaw, adjew = adj(PICKS.pending), adj(PICKS.emitted), adj(PICKS.aw), adj(PICKS.ew)
        @printf("  seed %s = 1 (reverse %.1f s)\n", seed, t)
        if seed === :pending
            summarize("adj pending, no birth", pend, ones(F, length(pend)), nb)
            summarize("adj pending, birth", pend, zeros(F, length(pend)), born)
            summarize("adj a.w, alive", adjaw, zeros(F, length(adjaw)), alive)
        elseif seed === :emitted
            summarize("adj emitted, alive", emit, ones(F, length(emit)), alive)
            summarize("adj pending, no birth", pend, zeros(F, length(pend)), nb)
            summarize("adj pending, birth", pend, ones(F, length(pend)), born)
        else
            summarize("adj pending, no birth", pend, zeros(F, length(pend)), nb)
            summarize("adj pending, birth", pend, aw, born)
            summarize("adj a.w, no birth", adjaw, zeros(F, length(adjaw)), nb)
            summarize("adj a.w, birth", adjaw, ptot, born)
            bslot = falses(length(adjew)); bslot[slots[born]] .= true
            summarize("adj e.w, birth slots", adjew, zeros(F, length(adjew)), bslot)
            summarize("adj e.w, other slots", adjew, ones(F, length(adjew)), .!bslot)
        end
        flush(stdout)
    end
end
println("probe done")
