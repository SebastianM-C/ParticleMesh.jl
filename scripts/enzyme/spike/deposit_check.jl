# WP-088 (#566): the trapped-beam state's current deposit against its pullback, without Enzyme.
# At step `tmid`'s start (or after `current!`'s sort with `sort=1`): which alive slots the
# Reference deposit's segments miss and which slots they hold that are not alive, and the dot
# test ⟨r, J(v)⟩ = ⟨J̄ᵀ r, v⟩ of `deposit_flux!` (forward: Dual{1} weights and offsets on a copy)
# against `flux_pullback!`, per input array, for the plasma's electrons (1) and the born ones (2).
#     JULIA_LOAD_PATH=@:<repo>:@stdlib julia +1.13 --project=lib/ParticleMeshElectromagnetic/test \
#       scripts/enzyme/spike/deposit_check.jl [tmid] [ppl] [ny] [sort]
include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(REPO, "scripts", "optim", "trapped2d_case.jl"))
const PEM = ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: CompensatedGather
using ParticleMeshCore: ALIVE, capacity, flux_pullback!, species_data
using Random: Xoshiro
using Printf

tmid = parse(Float64, get(ARGS, 1, "180"))
ppl = parse(Int, get(ARGS, 2, "8"))
ny = parse(Int, get(ARGS, 3, "1"))
dosort = get(ARGS, 4, "0") == "1"
F = Float32
θ = [TCASE.x0[i] for i in TCASE.free]
st, p, prob = start_of(trapped_problem(trapped_point(F.(θ))...; energy = F(budget(1.9)), ppl, ny, float = F);
                       ppl, ny, backend = backend(), gather = CompensatedGather(; k = 1))
kmid = round(Int, tmid / Float64(st.dt))
run!(st, p, kmid; observer = ())
grid, numerics = st.grid, st.numerics
# `ctx=1` (5th argument): the check where the step deposits, after step kmid + 1's sharpening,
# ionization, push and retirement at the window's back edge (at the step's start the window's last
# shift has moved the cells but not the old cells, and the refill's new particles hold stale old
# positions, which no step deposits)
if get(ARGS, 5, "0") == "1"
    let k = kmid + 1, wgs = numerics.workgroupsize, dt = st.dt
        PEM.sharpen_gather!(st)
        PEM.ionize!(PEM.ionization_frame(st), st, p, k)
        pf = PEM.particle_fields(st)
        n32 = Int32.(grid.n); g32 = Int32(PEM.ParticleMeshCore.guards(grid)); dt_dx = map(h -> dt / h, grid.dx)
        for s in (PEM.deposited_species(st)..., PEM.witness_species(st)...)
            PEM.launcher(PEM.push_kernel!, get_backend(s), wgs)(s, pf, numerics.pusher, PEM.read_shape(numerics), g32, s.q / s.m * dt / 2, dt_dx, n32;
                                                                ndrange = PEM.capacity(s))
        end
        PEM.retire_x!(PEM.window_frame(st), st)
    end
end
stg = PEM.rho_stagger(Val(2))
tq = PEM.transverse_currents(Val(2))
struct DTag end
for (j, (s, ws)) in enumerate(zip(PEM.deposited_species(st), PEM.deposited_workspaces(st)))
    dosort && PEM.resort_by_cell!(s, grid, ws)
    status = Array(s.status); a, b = Array(ws.seg_start), Array(ws.seg_stop)
    inseg = falses(length(status))
    for L in eachindex(a)
        a[L] <= b[L] && (inseg[a[L]:b[L]] .= true)
    end
    alive = status .== ALIVE
    @printf("species %d: alive %d, count %d; alive outside the segments %d, in the segments but not alive %d\n",
            j, count(alive), Int(Array(s.count)[1]), count(alive .& .!inseg), count(.!alive .& inseg))
    rng = Xoshiro(5 + j)
    rJ = map(_ -> (r = similar(st.fields.Jx); copyto!(r, randn(rng, F, size(r))); r), 1:3)
    # the pullback
    sb = Enzyme.make_zero(s)
    flux_pullback!(sb, rJ..., s, numerics.shape, stg, grid; dt = st.dt, tq..., workgroupsize = nothing)
    # forward in the weights' direction: the deposit is linear in w, so J(v) is the deposit of a copy with w = v
    v = similar(s.w); copyto!(v, randn(rng, F, size(v)))
    a0 = species_data(s)
    c = PEM.ParticleMeshCore.SpeciesArrays(map(copy, a0.cell), map(copy, a0.offset), map(copy, a0.u), copy(v), copy(a0.status), map(copy, a0.extras))
    sv = PEM.ParticleMeshCore.Species(c.cell, c.offset, c.u, c.w, c.status, copy(s.count), s.q, s.m, c.extras)
    Js = Dict()
    for mode in (Reference(), Fast())
        J = map(_ -> fill!(similar(st.fields.Jx), 0), 1:3)
        deposit_flux!(mode, J..., sv, numerics.shape, stg, grid, ws; dt = st.dt, tq...)
        fwd = sum(Float64(sum(Float64.(J[d]) .* Float64.(rJ[d]))) for d in 1:3)
        rev = Float64(sum(Float64.(sb.w) .* Float64.(v)))
        Js[nameof(typeof(mode))] = map(Array, J)
        # the same restricted to the alive slots outside the segments and the dead ones inside
        @printf("  %-11s ⟨r, J(v_w)⟩ = %+.6e, ⟨J̄ᵀr, v_w⟩ = %+.6e, relative %.2e\n", nameof(typeof(mode)), fwd, rev, abs(fwd - rev) / abs(rev))
    end
    for (nm, x) in (("real weights", s), ("v as weights", sv))
        Jr = map(_ -> fill!(similar(st.fields.Jx), 0), 1:3); Jf = map(_ -> fill!(similar(st.fields.Jx), 0), 1:3)
        deposit_flux!(Reference(), Jr..., x, numerics.shape, stg, grid, ws; dt = st.dt, tq...)
        deposit_flux!(Fast(), Jf..., x, numerics.shape, stg, grid, ws; dt = st.dt, tq...)
        @printf("  %s: max |Reference − Fast| / max |Fast| = %.3e, %.3e\n", nm, (maximum(abs, Array(Jr[d]) .- Array(Jf[d])) / maximum(abs, Array(Jf[d])) for d in 1:2)...)
    end
    # the same with the weights v in the particles' own arrays (s itself, its w restored after)
    wsave = copy(s.w); copyto!(s.w, v)
    Jr = map(_ -> fill!(similar(st.fields.Jx), 0), 1:3); Jf = map(_ -> fill!(similar(st.fields.Jx), 0), 1:3)
    deposit_flux!(Reference(), Jr..., s, numerics.shape, stg, grid, ws; dt = st.dt, tq...)
    deposit_flux!(Fast(), Jf..., s, numerics.shape, stg, grid, ws; dt = st.dt, tq...)
    @printf("  v in s itself: max |Reference − Fast| / max |Fast| = %.3e, %.3e\n", (maximum(abs, Array(Jr[d]) .- Array(Jf[d])) / maximum(abs, Array(Jf[d])) for d in 1:2)...)
    copyto!(s.w, wsave)
    g = Int(PEM.ParticleMeshCore.guards(grid)); n = grid.n
    for d in 1:3
        D = Js[:Reference][d] .- Js[:Fast][d]
        big = findall(abs.(D) .> 1f-4 * maximum(abs, Js[:Fast][d]))
        rows1 = sort(unique(first.(Tuple.(big)))); rows2 = sort(unique(last.(Tuple.(big))))
        @printf("  J%s: |Reference − Fast| max %.3e (max |Fast| %.3e) at %d nodes; array %s, guards %d, cells %s; x indices %s; y indices %s\n",
                "xyz"[d], maximum(abs, D), maximum(abs, Js[:Fast][d]), length(big), size(D), g, n, rows1[1:min(end, 12)], rows2[1:min(end, 12)])
        # after folding the transverse guards and dropping the x guards, as the step does
        for (nm, A) in ((:Reference, Js[:Reference][d]), (:Fast, Js[:Fast][d]))
            Ad = similar(st.fields.Jx); copyto!(Ad, A)
            PEM.zero_x_edges!(PEM.window_frame(st), Ad); PEM.ParticleMeshCore.fold_guards!(Ad, grid)
            Js[Symbol(nm, :_folded, d)] = Array(Ad)
        end
        Df = Js[Symbol(:Reference_folded, d)] .- Js[Symbol(:Fast_folded, d)]
        bigf = findall(abs.(Df) .> 1f-4 * maximum(abs, Js[:Fast][d]))
        @printf("     after zero_x_edges! and fold_guards!: max %.3e at %d nodes; x indices %s\n", maximum(abs, Df), length(bigf),
                sort(unique(first.(Tuple.(bigf))))[1:min(end, 12)])
    end
    out = alive .& .!inseg
    any(out) && @printf("  ⟨J̄ᵀr, v_w⟩ over the alive slots outside the segments %+.6e\n", Float64(sum(Float64.(Array(sb.w))[out] .* Float64.(Array(v))[out])))
    if any(out)
        ii = findall(out)[1:min(end, 5)]
        println("  first such slots ", ii, " cells ", [Array(s.cell[d])[ii] for d in 1:2], " offsets ", [Array(s.offset[d])[ii] for d in 1:2],
                " old offsets ", Array(s.extras.old_offset_1)[ii])
    end
end
