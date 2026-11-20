# WP-080: which piece of the 3D step Enzyme's reverse pass fails on, on a GPU. Each call
# differentiates one piece of `step!` (or the whole step) on WP-042's periodic 3D case, electrons
# only, followed by the field energy density, and prints whether the compile and the reverse pass
# went through. A crash of Enzyme's compiler takes the process with it, so run_bisect3d.sh runs
# each piece in its own process.
#
#     PM_BACKEND=cuda PM_ENZYME_PATCH=scripts/enzyme/spike/active_args.jl \
#       julia +1.13 -t 2 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/bisect3d.jl <piece> [T=Float32|Float64] [mode=fast|reference] [dim=3|2]
#
# Pieces: push (gather and push of every species), sort, deposit (the engine's deposit_flux!
# alone), current (sort, zero, deposit, fold), fold (the guard fold of Jx, Jy, Jz), solve (the
# Yee solve), kinetic, step (the whole step), step_nopush (the step's pieces without the push).

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "test", "em3d_cases.jl"))
const PME = ParticleMeshElectromagnetic
const PMC = ParticleMeshCore

const PIECE = get(ARGS, 1, "step")
const OPTS = Dict(String(first(split(a, "="))) => String(last(split(a, "="; limit = 2))) for a in ARGS[2:end] if occursin("=", a))
const T = Dict("Float64" => Float64, "Float32" => Float32)[get(OPTS, "T", "Float32")]
const MODE = Dict("fast" => Fast(), "reference" => Reference())[get(OPTS, "mode", "fast")]
const DIM = parse(Int, get(OPTS, "dim", "3"))

function case3(::Type{S}) where {S}
    prob = DIM == 3 ? plasma_step_3d(S; nsteps = 2) : plasma_step_2d(S; nsteps = 2)
    return EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, prob.species[1:1], prob.seed, prob.kmode)
end

function piece_push!(st)
    (; numerics, grid, dt) = st
    n = Int32.(grid.n); g = Int32(guards(grid)); dt_dx = map(h -> dt / h, grid.dx)
    pf = PME.particle_fields(st)
    for s in PME.deposited_species(st)
        PMC.launcher(PME.push_kernel!, get_backend(s), numerics.workgroupsize)(s, pf, numerics.pusher, PME.read_shape(numerics), g,
                                                                             s.q / s.m * dt / 2, dt_dx, n; ndrange = capacity(s))
    end
end
function piece_sort!(st)
    for (s, ws) in zip(PME.deposited_species(st), PME.deposited_workspaces(st))
        PME.resort_by_cell!(s, st.grid, ws; workgroupsize = st.numerics.workgroupsize)
    end
end
function piece_deposit!(st::EMState{N}) where {N}
    f = st.fields
    for (s, ws) in zip(PME.deposited_species(st), PME.deposited_workspaces(st))
        deposit_flux!(PME.deposit_mode(st, s), f.Jx, f.Jy, f.Jz, s, st.numerics.shape, PME.rho_stagger(Val(N)), st.grid, ws;
                      dt = st.dt, PME.transverse_currents(Val(N))..., workgroupsize = st.numerics.workgroupsize)
    end
end
piece_current!(st) = PME.current!(st.numerics.current, st)
piece_fold!(st) = foreach(J -> PME.fold_guards!(J, st.grid; workgroupsize = st.numerics.workgroupsize), (st.fields.Jx, st.fields.Jy, st.fields.Jz))
piece_solve!(st, p) = PME.solve_fields!(st.numerics.solver, st.fields, st.grid, st.dt, st.modifiers, p, 1; workgroupsize = st.numerics.workgroupsize)
piece_kinetic!(st) = PME.kinetic!(st)
function piece_step_nopush!(st, p)
    PME.current!(st.numerics.current, st)
    piece_solve!(st, p)
end

function run_piece!(E2, st, p)
    if PIECE == "push"
        piece_push!(st)
    elseif PIECE == "sort"
        piece_sort!(st)
    elseif PIECE == "deposit"
        piece_deposit!(st)
    elseif PIECE == "current"
        piece_current!(st)
    elseif PIECE == "fold"
        piece_fold!(st)
    elseif PIECE == "solve"
        piece_solve!(st, p)
    elseif PIECE == "kinetic"
        piece_kinetic!(st)
    elseif PIECE == "step_nopush"
        piece_step_nopush!(st, p)
    else
        step!(st, p, 1; observer = ())
    end
    f = st.fields
    # the currents enter the density too, so a piece that writes only J has an adjoint to carry
    energy_density_kernel!(get_backend(E2))(E2, f.Ex, f.Ey, f.Ez, f.Jx, f.Jy, f.Jz; ndrange = length(E2))
    return nothing
end

prob = case3(T)
st = initialized(prob, MODE)
p = params(prob)
println("bisect: $(DIM)D piece $PIECE $T $MODE, backend $(backend())"); flush(stdout)
E2 = similar(st.fields.Ex); fill!(E2, 0)
dE2 = similar(E2); fill!(dE2, 1)
dst = Enzyme.make_zero(st)
t = @elapsed begin
    Enzyme.autodiff(set_runtime_activity(Reverse), run_piece!, Const, Duplicated(E2, dE2), Duplicated(st, dst), Const(p))
    sync()
end
nz = count(a -> any(!iszero, Array(a)), values(leaves(dst)))
println("bisect: $(DIM)D piece $PIECE $T $MODE: ok in $(round(t; digits = 1)) s, $nz shadow arrays nonzero")
