# The N = 1 engine bits before the 2D port (WP-018 deliverable 0), checked by T-260. Shared by
# engine2d.jl and threads_worker.jl, which repeats them with 1 and 4 threads. Needs setup.jl,
# helpers.jl, digest.jl, engine_cases.jl and flux_cases.jl.
#
# `n1_lines()` computes `"kind|type|case|digest"` (FNV-1a, fixed by its definition) for every N = 1
# engine entry point the port touches: `load_quiet!`, `sort_by_cell!` (permutation, segments,
# count), `deposit!` in both modes for orders 1 and 2 and both staggers, `gather!`,
# `fill_guards!`, `fold_guards!`, `advance` and `deposit_flux!` (Reference). Atomic `Fast()`
# lines have the kind `depositfast`: their bits are an A4 claim on one thread only.
# `N1_RECORDED` holds the lines computed on the commit before the port (89fbe78, the WP-018 base),
# on the CPU backend with Julia 1.13 at 1 and 4 threads (identical, the Fast lines included).

"The density callable of the N = 1 load digest: `p[1]·(1 + p[2]·cos(p[3]·x))`."
struct N1Density end
@inline (::N1Density)(x, p) = p[1] * (one(x) + p[2] * cos(p[3] * x))

function n1_lines()
    lines = String[]
    for T in TYPES
        L = type_label(T)
        # loading: 16 cells, 8 per cell, drift and thermal spread per component
        grid = PeriodicGrid(16, basetype(T)(1 // 8), 2)
        p = (seeded(T, 1), seeded(T, 1 // 20), T(2π / length(grid)))
        s = load_quiet!(Species(backend, T, 16 * 8 + 3; q = -1, m = 1), grid, N1Density(), p, 8;
                        vdrift = (T(1 // 10), zero(T), T(-1 // 10)), vth = (T(1 // 20), T(1 // 30), T(1 // 40)), seed = 4, stream = 2)
        push!(lines, "load|$L|ppc=8|$(fnv_digest(species_arrays(s)...))")
        # sort with holes and free slots
        grid = PeriodicGrid(9, one(basetype(T)), 2)
        s = test_species(T, grid, 500; seed = 13)
        foreach(i -> s.status[i] = ParticleMeshCore.FREE, DEAD_SLOTS)
        ws = Workspace(s, grid)
        sort_by_cell!(s, grid, ws)
        push!(lines, "sort|$L|dead|$(fnv_digest(species_arrays(s)..., ws.perm, ws.seg_start, ws.seg_stop))")
        for shape in SHAPES, stagger in STAGGERS
            case = "$(order(shape)),$(nameof(typeof(stagger)))"
            for mode in (Reference(), Fast())
                kind = effective_mode(mode, T) isa Fast ? "depositfast" : "deposit"
                push!(lines, "$kind|$L|$case,$(nameof(typeof(mode)))|$(fnv_digest(deposit_case(T, mode, shape, stagger)))")
            end
            grid = PeriodicGrid(16, basetype(T)(1 // 4), guard_width(shape))
            s = test_species(T, grid, 200; seed = 3)
            f = test_field(T, grid, j -> seeded(T, sin(0.7j) + 0.3))
            out = fill!(similar(s.w), zero(T))
            gather!(out, f, s, shape, stagger, grid)
            push!(lines, "gather|$L|$case|$(fnv_digest(out))")
            c = continuity_case(T, Reference(), shape, stagger)
            push!(lines, "flux|$L|$case|$(fnv_digest(c.Jx, c.Jy, c.Jz))")
        end
        for g in (1, 2), n in (4, 9)
            grid = PeriodicGrid(n, one(basetype(T)), g)
            x = field(backend, T, grid)
            for i in eachindex(x)
                x[i] = seeded(T, cos(1.3i))
            end
            push!(lines, "fill|$L|g=$g,n=$n|$(fnv_digest(fill_guards!(copy(x), grid)))")
            push!(lines, "fold|$L|g=$g,n=$n|$(fnv_digest(fold_guards!(copy(x), grid)))")
        end
        # advance: 200 steps of a dyadic and a general displacement through the seam
        cells, offs = Int32[], T[]
        for (i, v) in enumerate((1 // 8, -3 // 16, 0.3, -0.71))
            c, δ = Int32(mod1(3i, 7)), seeded(T, i / 5)
            for _ in 1:200
                c, δ = ParticleMeshCore.advance(c, δ, seeded(T, v), Int32(7))
            end
            push!(cells, c)
            push!(offs, δ)
        end
        push!(lines, "advance|$L|200 steps|$(fnv_digest(cells, offs))")
    end
    return lines
end

const N1_RECORDED = String[
    "load|Float32|ppc=8|16279186357063211989",
    "sort|Float32|dead|8036589102928655217",
    "deposit|Float32|1,Node,Reference|15998566008816839413",
    "depositfast|Float32|1,Node,Fast|5830856461633728742",
    "gather|Float32|1,Node|12732387332865134944",
    "flux|Float32|1,Node|10601622206799488075",
    "deposit|Float32|1,Center,Reference|11416266991958991875",
    "depositfast|Float32|1,Center,Fast|15875389354039682273",
    "gather|Float32|1,Center|2819465140318391814",
    "flux|Float32|1,Center|2720057627881916768",
    "deposit|Float32|2,Node,Reference|9522159278961380857",
    "depositfast|Float32|2,Node,Fast|779994930799167744",
    "gather|Float32|2,Node|17299690901488900378",
    "flux|Float32|2,Node|5405151325461383891",
    "deposit|Float32|2,Center,Reference|17201009702472206333",
    "depositfast|Float32|2,Center,Fast|9325181511736541967",
    "gather|Float32|2,Center|1664036443969556562",
    "flux|Float32|2,Center|11015925937314238840",
    "fill|Float32|g=1,n=4|18056138655254487525",
    "fold|Float32|g=1,n=4|457848096354124670",
    "fill|Float32|g=1,n=9|16126435890048811603",
    "fold|Float32|g=1,n=9|2225500603874627805",
    "fill|Float32|g=2,n=4|357920779931669125",
    "fold|Float32|g=2,n=4|11821963807219965366",
    "fill|Float32|g=2,n=9|12987018657866564995",
    "fold|Float32|g=2,n=9|9204949770327407421",
    "advance|Float32|200 steps|7381578017391173463",
    "load|Float64|ppc=8|473952038083710630",
    "sort|Float64|dead|14339648647056404198",
    "deposit|Float64|1,Node,Reference|2879889325475545550",
    "depositfast|Float64|1,Node,Fast|6051820227397420881",
    "gather|Float64|1,Node|10464646963922083169",
    "flux|Float64|1,Node|3235155552267559814",
    "deposit|Float64|1,Center,Reference|6608870792872779204",
    "depositfast|Float64|1,Center,Fast|1690420568892695511",
    "gather|Float64|1,Center|13688026596779528914",
    "flux|Float64|1,Center|4528112285381826166",
    "deposit|Float64|2,Node,Reference|13577750879516709081",
    "depositfast|Float64|2,Node,Fast|747180129670659292",
    "gather|Float64|2,Node|4737746082967733300",
    "flux|Float64|2,Node|12626697166405987830",
    "deposit|Float64|2,Center,Reference|1615599643379526662",
    "depositfast|Float64|2,Center,Fast|10524175623828819554",
    "gather|Float64|2,Center|8617961182756885415",
    "flux|Float64|2,Center|17975322438613757158",
    "fill|Float64|g=1,n=4|9228803808242321439",
    "fold|Float64|g=1,n=4|9640239322575545646",
    "fill|Float64|g=1,n=9|3322389386988433447",
    "fold|Float64|g=1,n=9|17355778824228461834",
    "fill|Float64|g=2,n=4|17476848833723165825",
    "fold|Float64|g=2,n=4|9317603540087655123",
    "fill|Float64|g=2,n=9|1294163868866022821",
    "fold|Float64|g=2,n=9|10583528227783589073",
    "advance|Float64|200 steps|2923944219449560319",
    "load|Dual{Float32,3}|ppc=8|16644863893499027467",
    "sort|Dual{Float32,3}|dead|1937150381090033889",
    "deposit|Dual{Float32,3}|1,Node,Reference|17597514089040169238",
    "deposit|Dual{Float32,3}|1,Node,Fast|17597514089040169238",
    "gather|Dual{Float32,3}|1,Node|17377094253272837115",
    "flux|Dual{Float32,3}|1,Node|3290151249276149631",
    "deposit|Dual{Float32,3}|1,Center,Reference|7207375279883545173",
    "deposit|Dual{Float32,3}|1,Center,Fast|7207375279883545173",
    "gather|Dual{Float32,3}|1,Center|15374273824648620436",
    "flux|Dual{Float32,3}|1,Center|11462438050508278638",
    "deposit|Dual{Float32,3}|2,Node,Reference|12788772580281692967",
    "deposit|Dual{Float32,3}|2,Node,Fast|12788772580281692967",
    "gather|Dual{Float32,3}|2,Node|7286428691004596061",
    "flux|Dual{Float32,3}|2,Node|2603812275282810812",
    "deposit|Dual{Float32,3}|2,Center,Reference|15968138676145903592",
    "deposit|Dual{Float32,3}|2,Center,Fast|15968138676145903592",
    "gather|Dual{Float32,3}|2,Center|15357051198112878360",
    "flux|Dual{Float32,3}|2,Center|16508870883881339410",
    "fill|Dual{Float32,3}|g=1,n=4|16066851732492384549",
    "fold|Dual{Float32,3}|g=1,n=4|11677108442430075658",
    "fill|Dual{Float32,3}|g=1,n=9|2622421283453467381",
    "fold|Dual{Float32,3}|g=1,n=9|2870011057065402287",
    "fill|Dual{Float32,3}|g=2,n=4|217850361817729925",
    "fold|Dual{Float32,3}|g=2,n=4|18362603712460864694",
    "fill|Dual{Float32,3}|g=2,n=9|12603197874256471553",
    "fold|Dual{Float32,3}|g=2,n=9|14088219884690498587",
    "advance|Dual{Float32,3}|200 steps|10912305070894470551",
]

