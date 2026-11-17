# Whether the prototype's factored stencil in the engine's patch gives the engine's Reference bits:
#   julia +1.13 -t 4 --project=<repo>/scripts proto_bits.jl <repo> [cpu|cuda]
const REPO = ARGS[1]
include(joinpath(REPO, "scripts", "engine2d", "common.jl"))
include(joinpath(REPO, "scripts", "deposition", "common.jl"))
const NAME = length(ARGS) >= 2 ? ARGS[2] : "cpu"
if NAME == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
    @eval const BE = CUDABackend()
else
    const BE = CPU()
end
function main()
    for T in (Float32, Float64, Dual{Nothing,Float32,3}), shape in (PMC.Linear, PMC.Quadratic),
        st in ((PMC.Node(), PMC.Node()), (PMC.Center(), PMC.Center()), (PMC.Center(), PMC.Node()))

        V = base_type(T)
        grid = PMC.PeriodicGrid((37, 23), ntuple(d -> T(V(0.3) * d), 2), 3)
        dt = T(V(0.1))
        sx = map(h -> h / dt, grid.dx)
        s = synthetic_species(BE, T, grid; ppc = 5, fill = 0.8)
        ws = PMC.Workspace(s, grid; flux = (shape, st))
        PMC.sort_by_cell!(s, grid, ws)
        move!(s, grid; dmax = 0.99, kill = 0.05)
        PMC.sort_by_cell!(s, grid, ws)
        ref = zero_fields(BE, T, grid)
        PMC.deposit_flux!(PMC.Reference(), ref..., s, shape, st, grid, ws; dt, quantity_z = MomentumZ())
        a = deposit_patch!(zero_fields(BE, T, grid), s, factored_stencil, shape, st, grid, ws, sx, MomentumZ())
        b = deposit_patch!(zero_fields(BE, T, grid), s, engine_stencil, shape, st, grid, ws, sx, MomentumZ())
        println(rpad("$T order $(PMC.order(shape)) $(map(x -> nameof(typeof(x)), st))", 70), "prototype's factored stencil == engine: ",
                all(map(same, a, ref)), "   engine's stencil in the prototype's patch == engine: ", all(map(same, b, ref)),
                "   difference ", Float32(field_difference(a, ref)))
    end
end
Base.invokelatest(main)
