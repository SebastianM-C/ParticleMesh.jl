# The module's step with the sort by nearest-node bins (`EMNumerics(; node_bins = true)`, the default
# of WP-041's third item) against the same step with the sort by cell, on a backend:
#
#   julia +1.13 -t 4 --project=<repo>/scripts module_bins_check.jl <repo> [cpu|cuda]
#
# The suite's problems (lib/ParticleMeshElectromagnetic/test/*_cases.jl), states built here, in
# Reference mode with the charge-conserving current, Float64 (and Float32 with a loose bound, to see
# that nothing of that type is off by more than its roundoff grown over the run):
#   periodic      two mobile species in 2D, 40 steps, order 2 and order 1
#   1D            counter-streaming beams, 30 steps, order 2 and order 1 (the kinetic energy at N = 1
#                 is a deposit by the bins too: the recorded histories are compared)
#   invariant     the beams in 2D along y, 6 cells on x (the fewest the guards allow)
#   open y        two species leaving through both y edges under an absorbing layer, order 2 and 1
#   window        a warm plasma entering a moving window from step 1, periodic y; the same with an
#                 open y under a PML; the same at order 1
#   lwfa          a laser pulse and a plasma, the window starting at t = 33 (before it, the front is
#                 a fixed edge); the same at a0 = 4 in a window that does not start (the case in which
#                 the PR's breaker found charge on the back columns before the x axis kept the cell);
#                 the same with a PML on the back edge and a pass of the current filter
#   ionization    a dopant in the window (births, refills); a dopant in a field with a plasma and an
#                 open y under an absorbing layer (births from the first step)
# Compared after the run, bins against cells: every field array (|difference| over the largest entry
# of its group: E and B, or J); per species (the atoms and the born electrons included) the alive
# count, and the sorted values of w, u and of sin and cos of the position's phase on each axis (the
# slots' order differs between the two sorts, and sorted values of two sets that agree to roundoff
# agree to roundoff); the recorded histories. Prints the largest of each and exits with 1 above the bound.
const REPO = ARGS[1]
include(joinpath(REPO, "scripts", "engine2d", "common.jl"))
using ParticleMeshElectromagnetic
const PME = ParticleMeshElectromagnetic
const PMC = ParticleMesh.ParticleMeshCore
const NAME = length(ARGS) >= 2 ? ARGS[2] : "cpu"
if NAME == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
    @eval const BE = CUDABackend()
    CUDACore.allowscalar(false)
else
    const BE = CPU()
end
bits(a::AbstractArray) = reinterpret(UInt8, vec(Array(a)))
const TEST = joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "test")
foreach(f -> include(joinpath(TEST, f)), ("em2d_cases.jl", "boundaries_cases.jl", "window_cases.jl", "pml_back_cases.jl", "ionization_cases.jl"))
const ABOVE = String[]
const ONLY = get(ENV, "ONLY", "")                    # a part of a case's label: that case alone
const VERBOSE = get(ENV, "VERBOSE", "") == "1"       # every history's gap

"What a comparison reads of a species: the alive count and sorted per-particle values."
function species_record(s::PMC.Species{N}, n) where {N}
    alive = Array(s.status) .== PMC.ALIVE
    pick(a) = Float64.(Array(a)[alive])
    phase = ntuple(d -> 2π .* (Float64.(Array(s.cell[d])[alive]) .- 1 .+ pick(s.offset[d])) ./ n[d], N)
    vals = (pick(s.w), map(pick, s.u)..., map(x -> sin.(x), phase)..., map(x -> cos.(x), phase)...)
    return (; alive = count(alive), vals = map(sort, vals))
end

function record(st)
    fr = PME.ionization_frame(st)
    species = fr === nothing ? st.species : (st.species..., fr.atoms, fr.electrons)
    f = st.fields
    arrays = [(k, Float64.(Array(getfield(f, k)))) for k in fieldnames(typeof(f)) if getfield(f, k) isa AbstractArray]
    h = histories(st)
    hist = [(k, Float64.(collect(getfield(h, k)))) for k in keys(h) if getfield(h, k) isa AbstractArray{<:Real}]
    return (; species = map(s -> species_record(s, st.grid.n), species), arrays, hist)
end

gap(a, b, scale) = isempty(a) ? 0.0 : maximum(abs.(a .- b)) / scale

"The largest differences of two records: fields, particles, histories (each relative), and whether the alive counts agree."
function gaps(on, off)
    group(k) = startswith(String(k), "J") ? :J : :EB
    scale = Dict(g => maximum((maximum(abs, a; init = 0.0) for (k, a) in off.arrays if group(k) == g); init = 0.0) for g in (:J, :EB))
    fields = maximum((scale[group(k)] > 0 ? gap(a, b, scale[group(k)]) : Float64(a != b) for ((k, a), (_, b)) in zip(on.arrays, off.arrays)); init = 0.0)
    counts = all(((x, y),) -> x.alive == y.alive, zip(on.species, off.species))
    particles = 0.0
    if counts
        for (x, y) in zip(on.species, off.species), (a, b) in zip(x.vals, y.vals)
            m = maximum(abs, b; init = 0.0)
            particles = max(particles, m > 0 ? gap(a, b, max(m, 1.0)) : Float64(a != b))
        end
    end
    # a history against its own largest entry, or, where it is itself roundoff (a mode the problem does not drive), against
    # 1e-8 of the largest entry of all histories
    top = maximum((maximum(abs, b; init = 0.0) for (_, b) in off.hist); init = 0.0)
    each = [(k, top > 0 ? gap(a, b, max(maximum(abs, b; init = 0.0), 1e-8 * top)) : Float64(a != b), maximum(abs, b; init = 0.0))
            for ((k, a), (_, b)) in zip(on.hist, off.hist)]
    VERBOSE && foreach(((k, g, m),) -> println("    history ", rpad(k, 10), " gap ", rpad(round(g; sigdigits = 2), 10), " largest entry ", round(m; sigdigits = 3)), each)
    hist = maximum((g for (_, g, _) in each); init = 0.0)
    return (; fields, particles, hist, counts, alive = map(x -> x.alive, on.species))
end

function run_case(label, ::Type{T}, prob, bound; nsteps = prob.nsteps, kw...) where {T}
    occursin(ONLY, label) || return nothing
    p = params(prob)
    recs = map((true, false)) do node_bins
        st = EMState(prob, EMNumerics(; mode = Reference(), node_bins, kw...); backend = BE)
        initialize!(st, prob)
        run!(st, p, nsteps)
        (record(st), st)
    end
    g = gaps(recs[1][1], recs[2][1])
    st = recs[1][2]
    shifts = PME.window_frame(st) === nothing ? 0 : window_shifts(st, nsteps)
    ok = g.counts && max(g.fields, g.particles, g.hist) <= bound
    ok || push!(ABOVE, "$label $T")
    println(rpad(label, 46), rpad(string(T), 9), ok ? "agree   " : "ABOVE   ", "fields ", rpad(round(g.fields; sigdigits = 2), 9), "particles ",
            rpad(round(g.particles; sigdigits = 2), 9), "histories ", rpad(round(g.hist; sigdigits = 2), 9), g.counts ? "alive " : "ALIVE COUNTS DIFFER ",
            join(g.alive, "/"), shifts > 0 ? ", $shifts shifts" : "", ", bins ", join(unique(map(w -> bins_label(w.bins), PME.deposited_workspaces(st))), " and "))
    flush(stdout)
    return nothing
end
bins_label(b) = b isa PMC.CellBins ? "cells" : "order $(PMC.order(b.shape)) " * join(map(x -> nameof(typeof(x)), b.stagger), "/")

function cases(::Type{T}, bound, all) where {T}
    tenth = T(1) / 10
    warm(; kw...) = window_plasma(T; cells = (24, 16), thermal = (tenth, tenth, tenth), nsteps = 60, kw...)
    run_case("periodic, order 2", T, plasma_step_2d(T), bound)
    run_case("lwfa, the window from t = 33", T, window_lwfa(T; nsteps = 200), bound)
    run_case("ionization in the window", T, ion_window(T; nsteps = 150), bound)
    all || return nothing
    run_case("periodic, order 1", T, plasma_step_2d(T), bound; shape = PME.Linear)
    run_case("1D, order 2", T, beams_1d(T), bound)
    run_case("1D, order 1", T, beams_1d(T), bound; shape = PME.Linear)
    run_case("invariant along x, 6 cells on x", T, beams_2d(T, 2), bound)
    prob, layer = open_plasma_2d(T)
    run_case("open y, absorbing layer, order 2", T, prob, bound; absorber = layer)
    run_case("open y, absorbing layer, order 1", T, prob, bound; absorber = layer, shape = PME.Linear)
    run_case("window, warm plasma", T, warm(), bound)
    run_case("window, warm plasma, order 1", T, warm(), bound; shape = PME.Linear)
    run_case("window, warm plasma, open y under a PML", T, warm(; open_y = true), bound; absorber = PML(; cells = 4))
    run_case("lwfa at a0 = 4, a window that does not start", T, window_lwfa(T; a0 = 4, start = 1000, nsteps = 200), bound)
    run_case("lwfa, back PML and a filter pass", T, back_lwfa(T; nsteps = 200), bound; back_absorber = PML(; cells = 6), filter = BinomialFilter(1),
             headroom = 1 / 4)
    run_case("ionization in a field, plasma, open y", T,
             ion_in_E(T; E0 = (zero(T), one(T) / 2, zero(T)), plasma = true, boundaries = (:periodic, :open), box = (8, 8), cells = (16, 16),
                      rows = (3.1, 4.9), every = 2, nsteps = 40, density = one(T) / 20), bound;
             absorber = AbsorbingLayer(; cells = 3, strength = 1, order = 2))
    return nothing
end

function main()
    NAME == "cuda" && println("device: ", CUDACore.name(CUDACore.device()))
    :node_bins in fieldnames(EMNumerics) || error("this tree's EMNumerics has no node_bins")
    cases(Float64, 1e-9, true)
    cases(Float32, 2e-2, false)
    if isempty(ABOVE)
        println("backend ", NAME, ": the step by nearest-node bins agrees with the step by cell in every case")
    else
        println("backend ", NAME, ": ABOVE THE BOUND in ", join(ABOVE, "; "))
        exit(1)
    end
end
Base.invokelatest(main)
