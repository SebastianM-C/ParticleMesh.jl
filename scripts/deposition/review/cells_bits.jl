# Digests of the engine's Reference deposits by cell and of the module's runs with the sort by cell,
# to compare two trees bit for bit (the head of #476 with `CellBins()` and `node_bins = false`
# against main before it):
#
#   julia +1.13 -t 2 --project=<repo>/scripts cells_bits.jl <repo> > <tree>.txt     # then diff the two files
#
# Engine: synthetic states (scripts/deposition/common.jl), moved by up to 0.99 cells with 5% dead,
# sorted with a plain workspace; the hash of the bits of Jx, Jy, Jz and ρ, of the particle arrays, the
# permutation and the segments. Module: the suite's problems in Reference mode, with
# `EMNumerics(; node_bins = false)` where the numerics have the keyword; the hash of the bits of
# everything the state evolves, of `K` and of the histories.
const REPO = ARGS[1]
include(joinpath(REPO, "scripts", "engine2d", "common.jl"))
include(joinpath(REPO, "scripts", "deposition", "common.jl"))
using ParticleMeshElectromagnetic
const PME = ParticleMeshElectromagnetic
bits(a::AbstractArray) = reinterpret(UInt8, vec(Array(a)))
# BINS=nodes: the same digests with the sort by nearest-node bins (the bins of each deposit in the engine part,
# `node_bins = true` in the module's runs), to compare two trees that both have the bins.
const NODES = get(ENV, "BINS", "cells") == "nodes"
const TEST = joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "test")
foreach(f -> include(joinpath(TEST, f)), ("em2d_cases.jl", "boundaries_cases.jl", "window_cases.jl", "ionization_cases.jl"))
digest(arrays) = string(foldl((h, a) -> hash(collect(bits(a)), h), arrays; init = UInt(0)); base = 16, pad = 16)
type_label(::Type{T}) where {T} = T <: Dual ? "Dual{$(base_type(T)),3}" : string(T)

function engine(::Type{T}, shape, st) where {T}
    V = base_type(T)
    grid = PMC.PeriodicGrid((37, 23), ntuple(d -> T(V(0.3) * d), 2), 3)
    s = synthetic_species(CPU(), T, grid; ppc = 5, fill = 0.8)
    ws = NODES ? PMC.Workspace(s, grid; flux = (shape, st), bins = PMC.NodeBins(shape, st)) : PMC.Workspace(s, grid; flux = (shape, st))
    PMC.sort_by_cell!(s, grid, ws)
    move!(s, grid; dmax = 0.99, kill = 0.05)
    PMC.sort_by_cell!(s, grid, ws)
    J = zero_fields(CPU(), T, grid)
    PMC.deposit_flux!(PMC.Reference(), J..., s, shape, st, grid, ws; dt = T(V(0.1)), quantity_z = MomentumZ())
    ρ = KernelAbstractions.zeros(CPU(), T, (grid.n .+ 2grid.g)...)
    PMC.deposit!(PMC.Reference(), ρ, s, shape, st, grid, ws)
    println(rpad("engine $(type_label(T)) order $(PMC.order(shape)) $(join(map(x -> nameof(typeof(x)), st), "/"))", 52),
            digest((J..., ρ, species_arrays(s)..., ws.perm, ws.seg_start, ws.seg_stop)))
    return nothing
end

function engine1d(::Type{T}, shape, st) where {T}
    V = base_type(T)
    n, ppc = 23, 6
    grid = PMC.PeriodicGrid(n, T(V(0.4)), 3)
    rng = Xoshiro(11)
    s = PMC.Species(CPU(), T, n * ppc; q = -1, m = 1, extras = (old_cell = Int32, old_offset = T))
    for i in 1:(n * ppc)
        c, δ = Int32(rand(rng, 1:n)), lift(T, rand(rng, V), rng)
        s.extras.old_cell[i], s.extras.old_offset[i] = c, δ
        s.cell[1][i], s.offset[1][i] = PMC.advance(c, δ, T(V(0.99) * (2 * rand(rng, V) - 1)), Int32(n))
        foreach(u -> u[i] = lift(T, randn(rng, V), rng), s.u)
        s.w[i] = lift(T, (1 + rand(rng, V)) / ppc, rng)
        s.status[i] = rand(rng) < 0.8 ? PMC.ALIVE : PMC.FREE
    end
    ws = NODES ? PMC.Workspace(s, grid; bins = PMC.NodeBins(shape, (st,))) : PMC.Workspace(s, grid)
    PMC.sort_by_cell!(s, grid, ws)
    J = ntuple(_ -> zeros(T, n + 2 * 3), 3)
    PMC.deposit_flux!(PMC.Reference(), J..., s, shape, st, grid, ws; dt = T(V(0.1)), quantity_y = PME.CurrentY(), quantity_z = PME.CurrentZ())
    ρ = zeros(T, n + 2 * 3)
    PMC.deposit!(PMC.Reference(), ρ, s, shape, st, grid, ws)
    println(rpad("engine 1D $(type_label(T)) order $(PMC.order(shape)) $(nameof(typeof(st)))", 52),
            digest((J..., ρ, s.cell[1], s.offset[1], s.w, s.status, ws.perm, ws.seg_start, ws.seg_stop)))
    return nothing
end

numerics(; kw...) = :node_bins in fieldnames(EMNumerics) ? EMNumerics(; mode = Reference(), node_bins = NODES, kw...) : EMNumerics(; mode = Reference(), kw...)
function module_run(label, ::Type{T}, prob; kw...) where {T}
    st = EMState(prob, numerics(; kw...); backend = CPU())
    initialize!(st, prob)
    run!(st, params(prob), prob.nsteps)
    h = histories(st)
    println(rpad("module $label $(type_label(T))", 52), digest((PME.evolved(st)..., st.K, (getfield(h, k) for k in keys(h) if getfield(h, k) isa AbstractArray)...)))
    flush(stdout)
    return nothing
end

function main()
    N, C = PMC.Node(), PMC.Center()
    D3 = Dual{Nothing,Float32,3}
    for T in (Float32, Float64, D3), shape in (PMC.Linear, PMC.Quadratic)
        foreach(st -> engine(T, shape, st), ((N, N), (C, C), (C, N), (N, C)))
        foreach(st -> engine1d(T, shape, st), (N, C))
    end
    for T in (Float32, Float64)
        module_run("periodic 2D", T, plasma_step_2d(T))
        module_run("periodic 2D order 1", T, plasma_step_2d(T); shape = PME.Linear)
        module_run("1D beams", T, beams_1d(T))
        prob, layer = open_plasma_2d(T)
        module_run("open y", T, prob; absorber = layer)
        module_run("lwfa window", T, window_lwfa(T; nsteps = 120))
        module_run("ionization window", T, ion_window(T; nsteps = 150))
    end
    module_run("ionization window", D3, ion_window(D3; nsteps = 150))
    module_run("1D beams", D3, beams_1d(D3))
end
Base.invokelatest(main)
