# WP-056: the closed state, T-740 to T-746. What a run evolves is one tuple of arrays (`evolved`,
# the harness's partition); with the step index, the parameters and a state built for the same
# problem it is a complete restart: `snapshot` after any step, `restore!` into a new or a used
# state, and the continuation has the bits of the uninterrupted run (Reference mode, per element
# type and backend, A4). Equality is on alive slots: a free slot and the particle extras keep
# stale data. After window_cases.jl, boundaries_cases.jl, em2d_cases.jl, harness.jl (its helpers)
# and params.jl (`params_case`, `cold_pair`, `particle_arrays`).

using ParticleMeshElectromagnetic: on_evolved, refresh_caches!, witness_frame, witness_species
using SciMLBase: SciMLBase, DiscreteProblem, ReturnCode, remake, solve
using OrdinaryDiffEqFunctionMap: FunctionMap

"The eight convolution fields of a state with a PML on the open axis and on the window's back edge."
psis(s) = (L = s.aux.layer; B = s.aux.window.back; (L.ψBx, L.ψBz, L.ψEx, L.ψEz, B.ψBy, B.ψBz, B.ψEy, B.ψEz))

"""
    evolved_host(st) -> Vector{Pair{String,Any}}

What a run evolves in a state, on the host and by name: of each species and of the witness bunch
the status, the count and the alive slots of the other arrays (`particle_arrays`), the fields,
and at N = 2 a PML's convolution fields and the window's counters.
"""
function evolved_host(st::EMState)
    out = Pair{String,Any}[]
    for (i, s) in enumerate((st.species..., witness_species(st)...))
        who = i <= length(st.species) ? "species $i" : "witness"
        names = ("status", "count", ("cell $d" for d in 1:ndims(st))..., ("offset $d" for d in 1:ndims(st))..., "u1", "u2", "u3", "w")
        append!(out, map((n, a) -> "$who $n" => a, names, particle_arrays(s)))
    end
    append!(out, map((n, a) -> string(n) => Array(a), keys(st.fields), values(st.fields)))
    ndims(st) == 2 || return out
    (; layer, window) = st.aux
    layer isa PMLFields && append!(out, map((n, a) -> "layer $n" => Array(a), ("ψBx", "ψBz", "ψEx", "ψEz"), (layer.ψBx, layer.ψBz, layer.ψEx, layer.ψEz)))
    window === nothing && return out
    append!(out, map(a -> "lost" => Array(a), window.lost))
    append!(out, map(a -> "fast" => Array(a), window.fast))
    b = window.back
    b === nothing || append!(out, map((n, a) -> "back $n" => Array(a), ("ψBy", "ψBz", "ψEy", "ψEz"), (b.ψBy, b.ψBz, b.ψEy, b.ψEz)))
    return out
end

"""
    differences(a, b, k; cache = true) -> Vector{String}

The names of what differs in bits between two states that should hold the same run from step
`k + 1` on: anything a run evolves (`evolved_host`), the cache `K`, and the recorder's history
entries after `k` (with `cache = false`, the evolved arrays alone). Empty when the runs agree.
"""
function differences(a::EMState, b::EMState, k::Integer; cache = true)
    x, y = evolved_host(a), evolved_host(b)
    if cache
        for (s, l) in ((a, x), (b, y))
            push!(l, "K" => Array(s.K))
            append!(l, map((n, h) -> "history $n" => Array(h)[(k + 1):end], keys(s.recorder.hist), values(s.recorder.hist)))
        end
    end
    length(x) == length(y) || return ["$(length(x)) arrays against $(length(y))"]
    return [first(p) for (p, q) in zip(x, y) if !(first(p) == first(q) && same_bits(last(p), last(q)))]
end

"""
    run_with_snapshots(prob, num, ks; be, init) -> (state, snapshots)

The uninterrupted run of `prob` to its end, with a snapshot taken after each step of `ks`
(0: after `initialize!`), in increasing order.
"""
function run_with_snapshots(prob, num, ks; be = CPU(), init = (;))
    st = EMState(prob, num; backend = be)
    initialize!(st, prob; init...)
    p = params(prob)
    snaps = EMSnapshot[]
    done = 0
    for k in sort(collect(ks))
        run!(st, p, k - done; first = done + 1)
        done = k
        push!(snaps, snapshot(st, k))
    end
    run!(st, p, prob.nsteps - done; first = done + 1)
    return st, snaps
end

"Restore `snap` into `st` and run to the end of `prob`."
function continue_run!(st, prob, snap)
    restore!(st, snap)
    return run!(st, params(prob), prob.nsteps - snap.k; first = snap.k + 1)
end

"""
    restart_full(T; a0, n0, mode, species) -> (problem, numerics)

T-734's run, everything a 2D state can evolve at once: a laser from an antenna into a plasma
behind a ramp, a moving window from t = 33 with refill, a PML on the open axis and one on the
window's back edge (the pulse polarized between y and z, so all eight convolution fields are
excited) and a witness bunch injected at t = 40; 72 × 16 cells, 100 steps. With `species = 2` a
second, heavier species of opposite charge on the same lattice.
"""
function restart_full(::Type{T}; a0 = 1, n0 = 2 // 100, mode = Reference(), species = 1) where {T}
    yc = 16 * 2π / 8 / 2
    wb = WitnessBunch(; inject = 40.0, region = ((20.0, 30.0), (yc - 1, yc + 1)), ppc = (2, 1), momentum = T(3), center = (T(25), T(yc)), length = T(2))
    prob = params_case((seeded(T, a0), T(3), seeded(T, n0), T(8)); nsteps = 100, cells = (72, 16), position = 10, witness = wb, open_y = true,
                       polarization = 1)
    if species == 2
        ion = SpeciesDescription{T}(:ion, one(T), T(100), seeded(T, n0), zero(T), (1, 0), (2, 2), ZERO3(T), ZERO3(T), 2)
        prob = EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, (prob.species..., ion), prob.seed, prob.kmode;
                         lasers = prob.lasers, boundaries = prob.boundaries, window = prob.window, profile = prob.profile, witness = prob.witness)
    end
    return prob, EMNumerics(; mode, back_absorber = PML(; cells = 6), absorber = PML(; cells = 4, alpha_max = 0.2))
end

"""
The steps after which `restart_full` is saved: after `initialize!`, after the first step, while
the antenna emits, the step before the window's first shift and the step of it, the step before
the witness bunch is injected and the step that injects it, and the last step but one.
"""
function full_steps(prob, num)
    st = EMState(prob, num)
    kshift = findfirst(k -> window_shifts(st, k) > 0, 1:prob.nsteps)
    kinj = witness_frame(st).kinj
    return (0, 1, 30, kshift - 1, kshift, kinj - 1, kinj, prob.nsteps - 1)
end

"A 1D case file's problem with `passes` filter passes."
function filtered_case(name, ::Type{T}, passes) where {T}
    case = load_case(name)
    case["numerics"]["filter_passes"] = passes
    return problem_from_case(case, T; HARNESS_KW...), numerics_from_case(case)
end

"`open_plasma_2d` with its absorbing layer and the `Esirkepov()` current."
open_case(::Type{T}) where {T} = ((prob, layer) = open_plasma_2d(T); (prob, EMNumerics(; current = Esirkepov(), absorber = layer)))

"""
The other runs of T-740, each `(label, T -> (problem, numerics), steps to save after, initialize! keywords)`:
1D and 2D, static and in a window, with every way a buffer gets free slots (retired at the open
edge, shifted out of the window) and both counters of the refill counting.
"""
const RESTART_CASES = (
    ("1D Landau", T -> case_problem("landau", T; HARNESS_KW...), (0, 1, 17, 50, 100), (;)),
    ("1D two-stream, 3 filter passes, momenta at −1/2, density at the cell centre", T -> filtered_case("two-stream", T, 3), (0, 5),
     (; momenta_at = :t_minus_half, density_at = :cell_centre)),
    ("2D two species, Esirkepov, 2 filter passes", T -> (plasma_step_2d(T), EMNumerics(; current = Esirkepov(), filter = BinomialFilter(2))), (0, 1, 13, 39), (;)),
    ("2D prescribed fields E0 and B0, no current",
     T -> (plasma_2d(T; E0 = (one(T) / 100, zero(T), zero(T)), B0 = (zero(T), zero(T), one(T) / 2)), EMNumerics(; current = NoCurrent())), (0, 3), (;)),
    ("2D open axis with an absorbing layer, particles retired at both edges",
     T -> open_case(T), (0, 10, 25), (;)),
    ("2D window whose refill overflows the buffer (lost)",
     T -> (window_plasma(T; drift = (one(T) / 2, zero(T), zero(T)), nsteps = 40, vacuum = -10, density = one(T) / 10000), EMNumerics(; headroom = 0)), (0, 15, 25), (;)),
    ("2D thermal window whose refill draws speeds above c (fast)",
     T -> (window_plasma(T; cells = (16, 6), ppc = (1, 1), vacuum = -100, ramp = 1, thermal = (3 * one(T) / 10, 3 * one(T) / 10, 3 * one(T) / 10), nsteps = 150, seed = 6),
           EMNumerics()), (0, 50, 100), (;)),
)

# --- T-740 ----------------------------------------------------------------------------------

# The restart, at several steps of every kind of run. One uninterrupted run takes a snapshot
# after each chosen step; each snapshot goes into a state that is new (allocated, never
# initialized) or used (it ran another problem to its end, so its buffers, convolution fields,
# counters and witness bunch hold that run), is continued to the end with the same parameters, and
# then holds the uninterrupted run bit for bit: every alive particle, the fields, the convolution
# fields, the counters, the witness bunch, the cache `K` and every history entry after the saved
# step. The full case is saved around each event that depends on the step index: while the
# antenna emits, across the window's first shift (and the refills after it), across the witness
# bunch's injection. The comparison is not empty: the saved states differ from each other and
# from the end, the counters count before and after the saved step, and a restart one step off
# is another run.
function body_T740(be)
    for T in TYPES
        prob, num = restart_full(T)
        ks = full_steps(prob, num)
        ref, snaps = run_with_snapshots(prob, num, ks; be)
        p = params(prob)
        @test window_shifts(ref, prob.nsteps) > 20 && witness_alive(ref) > 0 && all(ψ -> any(!iszero, Array(ψ)), psis(ref))
        @test window_shifts(ref, ks[4]) == 0 && window_shifts(ref, ks[5]) == 1                       # saved on both sides of the first shift
        @test all(i -> any(map(!same_bits, map(Array, snaps[i].arrays), map(Array, snaps[i + 1].arrays))), 1:(length(snaps) - 1))
        other, _ = restart_full(T; a0 = 3 // 2, n0 = 1 // 100)
        used = EMState(prob, num; backend = be)
        initialize!(used, other)
        run!(used, params(other), other.nsteps)
        @test !isempty(differences(ref, used, 0))
        for (i, snap) in enumerate(snaps)
            st = isodd(i) ? EMState(prob, num; backend = be) : used
            continue_run!(st, prob, snap)
            @test snap.k == ks[i] && differences(ref, st, snap.k) == String[]
        end
        # the witness bunch is saved empty before its injection and alive after it
        fresh = EMState(prob, num; backend = be)
        @test witness_alive(restore!(fresh, snaps[6])) == 0 && witness_alive(restore!(fresh, snaps[7])) == capacity(witness_frame(fresh).species) > 0
        # the step index is part of the restart: the same arrays continued over as many steps, each under the next
        # step's index, are another run (the antenna's time and the window's position follow the index)
        for snap in snaps[[3, 4]]
            restore!(fresh, snap)
            run!(fresh, p, prob.nsteps - snap.k; first = snap.k + 2, observer = nothing)
            @test !isempty(differences(ref, fresh, prob.nsteps; cache = false))
        end
    end
    for (label, build, ks, init) in RESTART_CASES, T in TYPES
        prob, num = build(T)
        ref, snaps = run_with_snapshots(prob, num, ks; be, init)
        used = EMState(prob, num; backend = be)
        initialize!(used, prob; init...)
        run!(used, params(prob), prob.nsteps ÷ 2)
        for (i, snap) in enumerate(snaps)
            st = isodd(i) ? EMState(prob, num; backend = be) : used
            continue_run!(st, prob, snap)
            d = differences(ref, st, snap.k)
            isempty(d) || @info "T-740 restart differs" label T k = snap.k d
            @test isempty(d)
        end
        if occursin("lost", label)          # the counter counts before the saved steps and after them
            lost = map(s -> Int(Array(s.arrays[end - 1])[1]), snaps)
            @test 0 == lost[1] < lost[2] < lost[3] < window_lost(ref)[1]
        elseif occursin("fast", label)
            fast = map(s -> Int(Array(s.arrays[end])[1]), snaps)
            @test 0 == fast[1] < fast[2] < fast[3] < window_superluminal(ref)[1]
        elseif occursin("retired", label)   # the saved buffers have free slots between alive ones
            status = Array(snaps[end].arrays[9])
            @test any(i -> status[i] != ParticleMeshCore.ALIVE && status[i + 1] == ParticleMeshCore.ALIVE, 1:(length(status) - 1))
        end
    end
    # Fast() on an element type without atomics deposits in sorted order, so its restart is bitwise too (at N = 1 the cache needs the sort)
    prob, num = case_problem("landau", D3; mode = Fast(), HARNESS_KW...)
    ref, snaps = run_with_snapshots(prob, num, (0, 40); be)
    for snap in snaps
        @test differences(ref, continue_run!(EMState(prob, num; backend = be), prob, snap), snap.k) == String[]
    end
end

@testset "T-740 a run saved after any step and restored into a new or a used state continues with the bits of the uninterrupted run [A4, A8, WP-056]" begin
    body_T740(CPU())
end

# --- T-741 ----------------------------------------------------------------------------------

# The same through a file: `save_snapshot` writes the step, the units and every array's memory,
# `load_snapshot` reads them back for a state of the saved run's configuration, every bit (a
# dual's partials included), and the file is exactly the header and the arrays' bytes.
function body_T741(be)
    mktempdir() do dir
        for T in TYPES, (name, (prob, num)) in (("full", restart_full(T)), ("landau", case_problem("landau", T; HARNESS_KW...)))
            k = name == "full" ? witness_frame(EMState(prob, num)).kinj + 2 : 40
            ref, snaps = run_with_snapshots(prob, num, (k,); be)
            snap = only(snaps)
            path = joinpath(dir, "$name-$(type_label(T)).snap")
            save_snapshot(path, snap)
            st = EMState(prob, num; backend = be)
            loaded = load_snapshot(path, st)
            @test loaded.k == k && loaded.units === snap.units
            @test all(a -> a isa Array, loaded.arrays) && all(map(same_bits, loaded.arrays, map(Array, snap.arrays)))
            @test filesize(path) == 8 + 3 * 8 + length(snap.units) * sizeof(T) + 8 + sum(a -> 16 + 8 * ndims(a) + sizeof(eltype(a)) * length(a), snap.arrays)
            @test differences(ref, continue_run!(st, prob, loaded), k) == String[]
            io = IOBuffer()
            save_snapshot(io, snap)
            @test take!(io) == read(path)
        end
    end
end

@testset "T-741 a snapshot written to a file and read back restarts the run bit for bit, duals included [A4, A8, WP-056]" begin
    body_T741(CPU())
end

# --- T-742 ----------------------------------------------------------------------------------

"Fill every array of `x` (arrays, tuples and named tuples of arrays) with a value that would show: NaN for floats, 2 for integers."
junk!(a::AbstractArray{<:Integer}) = fill!(a, eltype(a)(2))
junk!(a::AbstractArray) = fill!(a, eltype(a)(NaN))
junk!(x::Union{Tuple,NamedTuple}) = foreach(junk!, x)

"""
Overwrite what a state holds besides its evolved arrays and its configuration: the cache `K`, the
scratch field `ρ`, every workspace buffer (the sort's keys, permutation, temporaries and segment
bounds, the flux patches), and at N = 2 the potential, the spectral buffers, the refill's flags,
the PMLs' shift scratch and the recorder's partial sums.
"""
function scramble_scratch!(st::EMState)
    junk!((st.K, st.ρ))
    for ws in st.ws
        junk!((ws.key, ws.tmp_i32, ws.tmp_T, ws.tmp_u8, ws.tmp_extras, ws.flux_patch))
        fill!(ws.perm, 1)
        fill!(ws.seg_start, Int32(1))
        fill!(ws.seg_stop, Int32(2))
    end
    ndims(st) == 2 || return st
    (; φ, spectral, layer, window) = st.aux
    junk!((φ, spectral.re, spectral.im, st.recorder.aux.partials))
    layer isa PMLFields && junk!(layer.scratch)
    window === nothing && return st
    junk!(window.flags)
    window.back === nothing || junk!(window.back.scratch)
    return st
end

"Overwrite the particle data no result may depend on: every extra, and of the slots that are not alive everything but the status."
function scramble_free!(st::EMState)
    for s in (st.species..., witness_species(st)...)
        junk!(s.extras)
        free = Array(s.status) .!= ParticleMeshCore.ALIVE
        for a in (s.cell..., s.offset..., s.u..., s.w)
            h = Array(a)
            h[free] .= eltype(h) <: Integer ? eltype(h)(2) : eltype(h)(NaN)
            copyto!(a, h)
        end
    end
    return st
end

# The evolved set is the whole evolved state. `evolved` is the hand-written list of the state's
# arrays, in the documented order, at N = 1 and N = 2 with one and two species; the harness's
# partition is the same arrays; `on_evolved` puts a state on other arrays of that layout and shares
# everything else. And nothing else in a state carries a run: a restored state whose every
# scratch buffer, cache, extra and free slot was overwritten with NaN and 2 before the
# continuation gives the uninterrupted run's bits, at N = 2 with everything at once and at N = 1.
function body_T742(be)
    T = Float64
    prob, num = restart_full(T; species = 2)
    st = EMState(prob, num; backend = be)
    f, L, w, b = st.fields, st.aux.layer, st.aux.window, st.aux.window.back
    sp(s) = (s.cell[1], s.cell[2], s.offset[1], s.offset[2], s.u[1], s.u[2], s.u[3], s.w, s.status, s.count)
    expected = (sp(st.species[1])..., sp(st.species[2])..., f.Ex, f.Ey, f.Ez, f.Bx, f.By, f.Bz, f.Jx, f.Jy, f.Jz,
                L.ψBx, L.ψBz, L.ψEx, L.ψEz, w.lost[1], w.lost[2], w.fast[1], w.fast[2], b.ψBy, b.ψBz, b.ψEy, b.ψEz, sp(st.aux.witness.species)...)
    same_arrays(x, y) = length(x) == length(y) && all(map(===, x, y))
    @test length(expected) == 51 && same_arrays(@inferred(evolved(st)), expected)
    @test same_arrays(state_partition(st).x, expected)
    @test all(a -> count(x -> x === a, expected) == 1, expected)            # no array twice
    copies = map(copy, expected)
    v = @inferred on_evolved(st, copies)
    @test typeof(v) === typeof(st) && same_arrays(evolved(v), copies)
    @test v.ws === st.ws && v.K === st.K && v.ρ === st.ρ && v.recorder === st.recorder && v.numerics === st.numerics && v.grid === st.grid && v.dt === st.dt && v.setup === st.setup
    @test all(map((a, c) -> a.extras === c.extras && a.q === c.q && a.m === c.m, (v.species..., v.aux.witness.species), (st.species..., st.aux.witness.species)))
    @test v.aux.antennas === st.aux.antennas && v.aux.spectral === st.aux.spectral && v.aux.φ === st.aux.φ && v.aux.layer.scratch === L.scratch &&
          v.aux.layer.node === L.node && v.aux.layer.center === L.center && v.aux.window.flags === w.flags && v.aux.window.back.scratch === b.scratch &&
          v.aux.window.back.node === b.node && v.aux.window.back.center === b.center &&
          (v.aux.window.window, v.aux.window.dt, v.aux.window.dx1, v.aux.window.n1, v.aux.window.inj, v.aux.window.g) === (w.window, w.dt, w.dx1, w.n1, w.inj, w.g) &&
          (v.aux.witness.kinj, v.aux.witness.c0, v.aux.witness.nc, v.aux.witness.ppc) === (st.aux.witness.kinj, st.aux.witness.c0, st.aux.witness.nc, st.aux.witness.ppc)
    @test_throws ArgumentError on_evolved(st, copies[1:(end - 1)])
    @test_throws ArgumentError on_evolved(st, (copies..., copies[end]))
    # without absorbers, window and witness bunch: the species and the fields; with an absorbing layer nothing more; at N = 1 the same
    plain = EMState(plasma_step_2d(T), EMNumerics(; current = Esirkepov()); backend = be)
    @test same_arrays(evolved(plain), (sp(plain.species[1])..., sp(plain.species[2])..., values(plain.fields)...))
    oprob, layer = open_plasma_2d(T)
    @test length(evolved(EMState(oprob, EMNumerics(; current = Esirkepov(), absorber = layer); backend = be))) == 29
    lprob, lnum = case_problem("two-stream", T; HARNESS_KW...)
    one_d = EMState(lprob, lnum; backend = be)
    sp1(s) = (s.cell[1], s.offset[1], s.u[1], s.u[2], s.u[3], s.w, s.status, s.count)
    @test same_arrays(@inferred(evolved(one_d)), (mapreduce(sp1, (x, y) -> (x..., y...), one_d.species)..., values(one_d.fields)...))
    @test same_arrays(evolved(@inferred(on_evolved(one_d, evolved(one_d)))), evolved(one_d))
    # nothing else carries the run
    for S in TYPES, (name, (qprob, qnum)) in (("full", restart_full(S; species = 2)), ("landau", case_problem("landau", S; HARNESS_KW...)))
        k = name == "full" ? witness_frame(EMState(qprob, qnum)).kinj + 3 : 40
        ref, snaps = run_with_snapshots(qprob, qnum, (k,); be)
        s = EMState(qprob, qnum; backend = be)
        scramble_scratch!(s)
        restore!(s, only(snaps))
        scramble_free!(s)
        run!(s, params(qprob), qprob.nsteps - k; first = k + 1)
        @test differences(ref, s, k) == String[]
        name == "full" && @test any(x -> x != ParticleMeshCore.ALIVE, Array(s.species[1].status))     # there were free slots to overwrite
    end
end

@testset "T-742 the evolved arrays are the whole evolved state: the list at N = 1 and N = 2, and a restart that does not depend on scratch, caches, extras or free slots [A4, A8, WP-056]" begin
    body_T742(CPU())
end

# --- T-743 ----------------------------------------------------------------------------------

# The harness's partition is the evolved set, so a solve is the run of its `u0` with nothing kept
# on the side: a solve of the full case (window, both PMLs, witness bunch) is the loop bit for bit
# in every evolved array and history, a second solve is the first, each state saved by `saveat`
# restarts the run in a new state, and the read-outs of a solution take the solution's arrays. A
# window whose refill overflows counts its lost particles once per solve (#423: twice in the
# second, when the counters were the cache's), a `Dual` problem without species solves (#424: its
# empty `K` was in the partition), and a solve from a `u0` whose convolution fields are not zero
# uses them (no step and no map zeroes them): for a problem without sources the second half of a
# run is a solve from the first half's end.
@testset "T-743 the harness's partition is everything a run evolves: PML, counters and witness bunch through solve, restarts from saveat, #423 and #424 [A4, A9, WP-056, WP-008]" begin
    for T in TYPES
        prob, num = restart_full(T)
        p = params(prob)
        kinj = witness_frame(EMState(prob, num)).kinj
        steps = [0, 30, kinj, prob.nsteps]
        dprob = DiscreteProblem(prob, num; backend, saveat = steps)
        @test partition_bits(dprob.u0) == map(bits, evolved(harness_state(dprob))) && length(dprob.u0.x) == 39
        sol = solve(dprob, FunctionMap())
        ref = loop_state(prob, num)
        @test sol.retcode == ReturnCode.Success && sol.t == steps
        @test partition_bits(sol.u[end]) == state_bits(ref) && history_bits(harness_state(sol)) == history_bits(ref)
        again = solve(dprob, FunctionMap())
        @test partition_bits(again.u[end]) == state_bits(ref) && history_bits(harness_state(again)) == history_bits(ref)
        # the cache's own evolved arrays stay the problem's u0; a read-out takes the saved state
        @test witness_alive(harness_state(sol)) == 0 && all(ψ -> all(iszero, Array(ψ)), psis(harness_state(sol)))
        last_state = harness_state(sol, sol.u[end])
        @test all(map(===, evolved(last_state), sol.u[end].x)) && last_state.recorder === harness_state(sol).recorder
        @test witness_alive(last_state) == witness_alive(ref) > 0
        @test same_bits([witness_energy_gain(last_state, p; allow_lost = true)], [witness_energy_gain(ref, p; allow_lost = true)])
        @test all(ψ -> any(!iszero, Array(ψ)), psis(last_state))
        # every saved state is a restart of its step, through copyto! and through restore!
        for (i, k) in enumerate(steps[1:3])
            st = EMState(prob, num; backend)
            i == 2 ? restore!(st, sol.u[i].x) : copyto!(st, sol.u[i])
            run!(st, p, prob.nsteps - k; first = k + 1)
            @test differences(ref, st, k) == String[]
        end
    end
    # #423: the refill's counters are the solution's, one run's count in every solve
    fwd = window_plasma(Float64; drift = (0.5, 0.0, 0.0), nsteps = 40, vacuum = -10, density = 1e-4)
    tight = EMNumerics(; headroom = 0)
    lost = window_lost(loop_state(fwd, tight))
    dp = DiscreteProblem(fwd, tight; backend, save_everystep = false)
    for _ in 1:2
        sol = solve(dp, FunctionMap())
        @test window_lost(harness_state(sol, sol.u[end])) == lost && lost[1] > 0
        @test window_lost(harness_state(dp)) == (0,)
    end
    # #424: a problem without species, in every element type
    for T in TYPES
        vac = EMProblem("vacuum", (T(6), T(3)), (12, 6), T(3) / 10, 4, (), 1, (1, 0))
        vnum = EMNumerics(; current = NoCurrent())
        sol = solve(DiscreteProblem(vac, vnum; backend), FunctionMap())
        @test sol.retcode == ReturnCode.Success && length(sol.u[end].x) == 9 && all(a -> length(a) > 0, sol.u[end].x)
        @test partition_bits(sol.u[end]) == state_bits(loop_state(vac, vnum))
        half_dt = EMProblem("vacuum", (T(6), T(3)), (12, 6), T(3) / 20, 4, (), 1, (1, 0))     # another configuration is still refused at step 1
        @test_throws ArgumentError solve(remake(DiscreteProblem(vac, vnum; backend); p = half_dt), FunctionMap())
    end
    # a solve from a later state of a run without sources: nothing is zeroed at step 1
    for T in TYPES
        prob, _ = open_plasma_2d(T; nsteps = 20)
        num = EMNumerics(; current = Esirkepov(), absorber = PML(; cells = 4, sigma_max = 1.5, order = 3, kappa_max = 2, kappa_order = 2, alpha_max = 0.2))
        dp = DiscreteProblem(prob, num; backend, save_everystep = false)
        half = solve(dp, FunctionMap())
        @test any(!iszero, Array(harness_state(half, half.u[end]).aux.layer.ψEx))
        rest = solve(remake(dp; u0 = half.u[end]), FunctionMap())
        hrest = history_bits(harness_state(rest))
        ref = EMState(prob, num; backend)
        initialize!(ref, prob)
        rec = HistoryRecorder(prob, ref; nsteps = 2 * prob.nsteps)
        run!(ref, params(prob), 2 * prob.nsteps; observer = rec)
        @test partition_bits(rest.u[end]) == state_bits(ref)
        @test hrest == map(h -> bits(view(h, (prob.nsteps + 1):(2 * prob.nsteps))), values(rec.hist))     # its histories are the second half's, the first entry's K included
        @test partition_bits(solve(dp, FunctionMap()).u[end]) == partition_bits(half.u[end])       # and the problem's own u0 is untouched
    end
end

# --- T-744 ----------------------------------------------------------------------------------

# No step is special. Without a laser, a window and a witness bunch nothing in a step reads the
# step index, so the same ten steps taken as steps 11 to 20 and, from the saved state, as steps
# 1 to 10 are the same steps bit for bit, with a PML whose convolution fields are not zero at the
# saved state: a step that zeroed them at step 1, as `step!` did while they were outside the
# partition, would give another run.
function body_T744(be)
    for T in TYPES
        prob, _ = open_plasma_2d(T; nsteps = 20)
        num = EMNumerics(; current = Esirkepov(), absorber = PML(; cells = 4, sigma_max = 1.5, order = 3, kappa_max = 2, kappa_order = 2, alpha_max = 0.2))
        p = params(prob)
        a = EMState(prob, num; backend = be)
        initialize!(a, prob)
        run!(a, p, 10; observer = nothing)
        L = a.aux.layer
        @test all(ψ -> any(!iszero, Array(ψ)), (L.ψBx, L.ψBz, L.ψEx, L.ψEz))
        snap = snapshot(a, 10)
        run!(a, p, 10; first = 11, observer = nothing)
        b = restore!(EMState(prob, num; backend = be), snap)
        run!(b, p, 10; first = 1, observer = nothing)
        @test differences(a, b, 0; cache = false) == String[]
        @test same_bits(Array(a.K), Array(b.K))
    end
end

@testset "T-744 no step is special: without sources the same steps under other step indices are the same steps, convolution fields included [A4, WP-056]" begin
    body_T744(CPU())
end

# --- T-745 ----------------------------------------------------------------------------------

# What `restore!` and `load_snapshot` refuse, and what they must not refuse. A snapshot goes back
# into a state of its own problem when a species is described in another element type than the
# problem's with a mass and a charge that are not exact in it (the checks compare what two states
# hold, never a description's number): Float64 descriptions in a Float32 and a `Dual{Float32,3}`
# problem, at N = 2 and N = 1, in memory and through a file. Arrays of another layout (another
# grid, capacity, element type or set of absorbers) and a snapshot of another time step or cell
# size are refused before anything is written, and a file that is not a snapshot of the state is
# refused whatever is wrong with it.
@testset "T-745 restore! and load_snapshot refuse another layout, time step or cell size and a damaged file, and accept a mixed-precision problem's own snapshot [A8, WP-056]" begin
    for S in (Float32, D3)
        ion2 = SpeciesDescription{Float64}(:ion, 0.3, 1836.15267343, 0.7123, 0.0517, (2, 1), (3, 3), ZERO3(Float64), (0.05, 0.05, 0.05), 1)
        ion1 = SpeciesDescription{Float64}(:ion, 0.3, 0.1, 0.9, 0.05, 1, 16, (0.1, 0.0, 0.0), (0.05, 0.0, 0.0), 1)
        heavy2 = EMProblem("inexact-mass-2d", (2 * S(π), 2 * S(π)), (10, 8), plasma_2d(S).dt, 6, (ion2,), 1, (2, 1))
        heavy1 = EMProblem("inexact-mass-1d", 2 * S(π), 32, S(15) / 100, 6, (ion1,), 1, one(S))
        for (q, num) in ((heavy2, EMNumerics(; current = Esirkepov())), (heavy1, EMNumerics()))
            ref, snaps = run_with_snapshots(q, num, (2,))
            @test value_of(ref.species[1].m) != q.species[1].mass && value_of(ref.species[1].q) != 0.3      # rounded into the state
            @test differences(ref, continue_run!(EMState(q, num), q, only(snaps)), 2) == String[]
            io = IOBuffer()
            save_snapshot(io, only(snaps))
            st = EMState(q, num)
            @test differences(ref, continue_run!(st, q, load_snapshot(seekstart(io), st)), 2) == String[]
        end
    end
    T = Float64
    prob = plasma_step_2d(T)
    num = EMNumerics(; current = Esirkepov())
    ref, snaps = run_with_snapshots(prob, num, (7,))
    snap = only(snaps)
    @test_throws ArgumentError snapshot(ref, -1)
    remade(; length = prob.length, cells = prob.cells, dt = prob.dt, species = prob.species, boundaries = (:periodic, :periodic)) =
        EMProblem(prob.name, length, cells, dt, prob.nsteps, species, prob.seed, prob.kmode; boundaries)
    e, pos = prob.species
    with_ppc(ppc) = SpeciesDescription{T}(e.name, e.charge, e.mass, e.density, e.amplitude, e.mode, ppc, e.drift, e.thermal, e.stream; momentum = e.momentum)
    variants = (cells = (remade(; cells = (8, 8)), num),                                        # other cells
                transposed = (remade(; cells = (8, 10)), num),                                  # as many cells and particles, other sizes
                smaller = (remade(; species = (with_ppc((2, 3)), pos)), num),                   # another capacity
                larger = (remade(; species = (with_ppc((4, 3)), pos)), num),
                one = (remade(; species = (e,)), num),                                          # fewer arrays
                eltype = (plasma_step_2d(Float32), num),                                        # another element type
                pml = (remade(; boundaries = (:periodic, :open)), EMNumerics(; current = Esirkepov(), absorber = PML(; cells = 3))),   # more arrays
                dt = (remade(; dt = prob.dt / 2), num),                                         # the snapshot's layout, another time step
                size = (remade(; length = (prob.length[1], 2 * prob.length[2])), num))         # the snapshot's layout, another cell size
    states = map(((q, n),) -> initialize!(EMState(q, n), q), variants)
    for st in states
        before = map(copy, evolved(st))
        K = copy(st.K)
        @test_throws ArgumentError restore!(st, snap)
        @test all(map(same_bits, evolved(st), before)) && same_bits(st.K, K)               # refused before anything is written
    end
    a = snap.arrays
    # the arrays alone: the layout is checked, the units are the caller's
    for st in (states.cells, states.transposed, states.smaller, states.larger, states.one, states.eltype, states.pml)
        @test_throws ArgumentError restore!(st, a)
    end
    @test all(st -> map(size, evolved(st)) == map(size, a), (states.dt, states.size))
    @test restore!(states.dt, a) === states.dt && all(map(same_bits, evolved(states.dt), a))
    @test_throws ArgumentError restore!(EMState(prob, num), a[1:(end - 1)])
    @test_throws ArgumentError restore!(EMState(prob, num), (a..., a[end]))
    @test_throws ArgumentError restore!(EMState(prob, num), (a[1], a[3], a[2], a[4:end]...))    # an offset where a cell belongs
    # files
    io = IOBuffer()
    save_snapshot(io, snap)
    bytes = take!(io)
    fresh = EMState(prob, num)
    @test load_snapshot(IOBuffer(bytes), fresh).k == 7
    for cut in (0, 4, 8, 20, 40, 64, 100, length(bytes) ÷ 2, length(bytes) - 1)
        @test_throws ArgumentError load_snapshot(IOBuffer(bytes[1:cut]), fresh)
    end
    @test_throws ArgumentError load_snapshot(IOBuffer(vcat(bytes, 0x00)), fresh)                 # bytes left over
    @test_throws ArgumentError load_snapshot(IOBuffer(vcat(UInt8('X'), bytes[2:end])), fresh)     # not a snapshot
    for st in (states.cells, states.transposed, states.smaller, states.larger, states.one, states.eltype, states.pml)
        @test_throws ArgumentError load_snapshot(IOBuffer(bytes), st)
    end
    other_dt = EMState(variants.dt...)
    @test_throws ArgumentError restore!(other_dt, load_snapshot(IOBuffer(bytes), other_dt))      # another time step: read, then refused by its units
end

# --- T-746 ----------------------------------------------------------------------------------

include(joinpath(REPO, "scripts", "digest", "digest.jl"))

# The before/after digest script of the rewrite (`scripts/digest/digest.jl`) still runs against
# this tree: two cheap cases and one harness case give well-formed lines, the same lines twice
# (Reference mode); a harness line hashes the particle and field arrays of the partition, which
# every layout of it shares, not what follows them; the state group holds every evolved array of
# a state that has them all, once; and the digest sees one flipped bit and one more step.
@testset "T-746 the before/after digest script runs two cheap cases reproducibly and its hashes see a flipped bit [A4, WP-056, WP-055]" begin
    names = ["1d-float64-species-in-T", "2d-thermal-nocurrent-E0B0", "harness-2d-open-pml"]
    lines(; kw...) = (out = String[]; EMDigest.digest(l -> push!(out, l); modes = (("Reference", Reference()),), cases = names, kw...); out)
    out = lines()
    @test length(out) == 2 * 3 + 2 * 3                         # two runs and two solves, per element type
    @test lines() == out
    @test all(l -> occursin(r"^[0-9A-Za-z-]+ \| (Float32|Float64|Dual\{Float32,3\}) \| Reference \| [0-9a-f]{24} \d+/\d+ \| [0-9a-f]{24} \d+/\d+ \| [0-9a-f]{24} \d+/\d+$", l), out[1:6])
    @test all(l -> occursin(r"^harness-2d-open-pml \| \S+ \| Reference \| solve [12] \| [0-9a-f]{24} 29/\d+ \| [0-9a-f]{24} 21/\d+$", l), out[7:12])
    @test all(i -> split(out[i], " | ")[5:6] == split(out[i + 1], " | ")[5:6], 7:2:11)       # the second solve is the first
    @test length(unique(l -> split(l, " | ")[4], out[1:6])) == 6                              # every run has its own state hash
    @test_throws ErrorException EMDigest.main(["unused.txt", "Reference", "no-such-case"])
    full = EMState(restart_full(Float64; species = 2)...)
    g = EMDigest.groups(full)
    @test all(a -> count(x -> x === a, g.state) == 1, (evolved(full)..., full.K, full.ρ)) && all(a -> count(x -> x === a, g.state) <= 1, g.state)
    @test length(g.hist) == 21 && all(map(===, g.hist, values(full.recorder.hist))) && !any(a -> any(x -> x === a, g.state), g.ws)
    prob = plasma_2d(Float64)
    st = state_2d(prob)
    d0 = EMDigest.digest_line(st)
    @test EMDigest.digest_line(st) == d0
    Ex = Array(st.fields.Ex)
    Ex[2, 3] = nextfloat(Ex[2, 3])
    copyto!(st.fields.Ex, Ex)
    d1 = EMDigest.digest_line(st)
    @test split(d1, " | ")[1] != split(d0, " | ")[1] && split(d1, " | ")[2:3] == split(d0, " | ")[2:3]
    step!(st, params(prob), 1)
    @test all(map(!=, split(EMDigest.digest_line(st), " | ")[1:2], split(d1, " | ")[1:2]))
end
