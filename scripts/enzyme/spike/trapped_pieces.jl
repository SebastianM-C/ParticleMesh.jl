# WP-078: which piece of the trapped-beam case's step Enzyme's reverse mode handles. One piece per
# process (an Enzyme failure can be a segfault): the state after a primal run to `tmid`, then
# the reverse pass of that piece of `step!` (in its order) with Duplicated(state), Active(params).
#
#     PM_BACKEND=cuda PM_ENZYME_PATCH=scripts/enzyme/active_reduce.jl JULIA_LOAD_PATH=@:<repo>:@stdlib \
#       julia +1.13 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/spike/trapped_pieces.jl <piece> [tmid] [ppl] [ny]
#
# pieces: sharpen, ionize, push, retire_x, current, retire_open, solve, window, summand, step
include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(REPO, "scripts", "optim", "trapped2d_case.jl"))
const PEM = ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: CompensatedGather, ionization_electrons
using ParticleMeshCore: cellsize, guards

piece = ARGS[1]
tmid = parse(Float64, get(ARGS, 2, "180"))
ppl = parse(Int, get(ARGS, 3, "32"))
ny = parse(Int, get(ARGS, 4, "2"))
F = Float32
θ = [TCASE.x0[i] for i in TCASE.free]
st, p, prob = start_of(trapped_problem(trapped_point(F.(θ))...; energy = F(budget(1.9)), ppl, ny, float = F);
                       ppl, ny, backend = backend(), gather = CompensatedGather(; k = 1))
kmid = round(Int, tmid / Float64(st.dt))
run!(st, p, kmid; observer = ())
sync()
println("piece $piece at step $(kmid + 1) (t = $(kmid * Float64(st.dt))), ppl $ppl ny $ny; born alive ", alive_count(ionization_electrons(st)))
flush(stdout)

function piece!(st, p, k, ::Val{P}) where {P}
    (; numerics, grid, fields, dt) = st
    wgs = numerics.workgroupsize
    n = Int32.(grid.n)
    g = Int32(guards(grid))
    dt_dx = map(h -> dt / h, grid.dx)
    window = PEM.window_frame(st)
    if P === :sharpen
        PEM.sharpen_gather!(st)
    elseif P === :ionize
        PEM.ionize!(PEM.ionization_frame(st), st, p, k)
    elseif P === :push
        pf = PEM.particle_fields(st)
        for s in (PEM.deposited_species(st)..., PEM.witness_species(st)...)
            PEM.launcher(PEM.push_kernel!, get_backend(s), wgs)(s, pf, numerics.pusher, PEM.read_shape(numerics), g, s.q / s.m * dt / 2, dt_dx, n;
                                                                ndrange = PEM.capacity(s))
        end
    elseif P === :retire_x
        PEM.retire_x!(window, st)
    elseif P === :current
        PEM.current!(numerics.current, st)
    elseif P === :retire_open
        PEM.retire_open!(PEM.open_absorbers(st), st)
    elseif P === :solve
        PEM.solve_fields!(numerics.solver, fields, grid, dt, st.modifiers, p, k; workgroupsize = wgs)
    elseif P === :window
        PEM.advance_window!(window, st, p, k)
    elseif P === :step
        step!(st, p, k; observer = ())
    elseif P in (:resort_e, :resort_b, :dep_e, :dep_b)
        i = P in (:resort_e, :dep_e) ? 1 : 2
        s, ws = PEM.deposited_species(st)[i], PEM.deposited_workspaces(st)[i]
        if P in (:resort_e, :resort_b)
            PEM.resort_by_cell!(s, grid, ws; workgroupsize = wgs)
        else
            PEM.deposit_flux!(PEM.deposit_mode(st, s), fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, PEM.rho_stagger(Val(2)), grid, ws;
                              dt = st.dt, PEM.transverse_currents(Val(2))..., workgroupsize = wgs)
        end
    elseif P === :edges
        for J in (fields.Jx, fields.Jy, fields.Jz)
            PEM.zero_x_edges!(window, J; workgroupsize = wgs)
            PEM.fold_guards!(J, grid; workgroupsize = wgs)
        end
    elseif P === :w_fields
        for a in (fields.Ex, fields.Ey, fields.Ez, fields.Bx, fields.By, fields.Bz)
            PEM.launcher(PEM.shift_columns_kernel!, get_backend(a), wgs)(a, st.ρ, Int32(g + 1), Int32(g + n[1]), a === fields.Ex; ndrange = size(a))
        end
    elseif P === :w_absorbers
        PEM.shift_absorbers!(st.modifiers, wgs)
    elseif P === :w_species
        w = st.aux.window
        s, ws, setup, ps = st.species[1], st.ws[1], st.setup[1], p.species[1]
        PEM.launcher(PEM.shift_particles_kernel!, get_backend(s), wgs)(s; ndrange = PEM.capacity(s))
        PEM.sort_after_shift!(st, s, ws)
    elseif P === :w_load
        w = st.aux.window
        s, setup, ps = st.species[1], st.setup[1], p.species[1]
        PEM.load_window_columns!(s, grid, PEM.window_load(setup, ps, p), w.lost[1], PEM.injection_column(w), 1, 100;
                                 fast = w.fast[1], flags = w.flags[1], workgroupsize = wgs)
    elseif P === :w_ionization
        PEM.shift_ionization!(PEM.ionization_frame(st), st, p, 100)
    elseif P === :filter
        PEM.filter_current!(numerics.filter, fields, grid, st.ρ; workgroupsize = wgs, edges = window)
    end
    return nothing
end
@kernel function sq_kernel!(out, @Const(a))
    i = @index(Global, Linear)
    @inbounds out[i] = a[i] * a[i]
end
# the function differentiated: the piece, then Σ Ex² + Σ w² of the plasma (a summand per element, seeded with ones)
function f!(o1, o2, st, p, k, v)
    piece!(st, p, k, v)
    sq_kernel!(get_backend(o1))(o1, st.fields.Ey; ndrange = length(o1))
    sq_kernel!(get_backend(o2))(o2, st.species[1].u[1]; ndrange = length(o2))
    return nothing
end
o1 = similar(st.fields.Ey); fill!(o1, 0); d1 = similar(o1); fill!(d1, 1)
o2 = similar(st.species[1].u[1]); fill!(o2, 0); d2 = similar(o2); fill!(d2, 1)
dst = Enzyme.make_zero(st)
t = @elapsed res = Enzyme.autodiff(set_runtime_activity(Reverse), f!, Const, Duplicated(o1, d1), Duplicated(o2, d2), Duplicated(st, dst), Active(p),
                                   Const(kmid + 1), Const(Val(Symbol(piece))))
sync()
dp = res[1][4]
println("piece $piece: reverse OK in $(round(t; digits = 1)) s; |dst| over arrays ",
        sum(a -> Float64(sum(abs, a)), values(leaves(dst))), "; dp lasers ", dp.lasers)
println("kernel arguments with dropped adjoints: ", sort!(collect(ACTIVE_SEEN)))
