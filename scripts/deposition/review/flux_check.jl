# The engine's 2D flux deposition on a backend, held to what T-300 and T-302 hold it to on the CPU:
#
#   julia +1.13 -t 4 --project=<repo>/scripts flux_check.jl <repo> [cpu|cuda]
#
# Synthetic states (scripts/deposition/common.jl), both shapes, the four stagger pairs, Float32,
# Float64 and Dual{Float32,3}. Reference: discrete continuity, the same bits on a second call and
# at two explicit launch shapes, and agreement with the CPU's Reference on the same state. Fast
# (floats): continuity and agreement with Reference per node; with a Dual it must give Reference's
# bits (the fallback). Prints one line per check and throws on a failure.
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
const FAILED = String[]
const WORST = Dict{String,Float64}()

function report(label, name, value, bound)
    ok = value <= bound
    ok || push!(FAILED, "$label $name")
    value isa Bool || (WORST[name] = max(get(WORST, name, 0.0), Float64(value) / Float64(bound)))
    ok || println(rpad("$label $name", 100), "FAILED  ", value)
    return ok
end

function state(be, ::Type{T}, shape, st, n, ppc) where {T}
    V = base_type(T)
    grid = PMC.PeriodicGrid(n, ntuple(d -> T(V(0.3) * d), 2), 3)
    s = synthetic_species(be, T, grid; ppc, fill = 0.8)
    ws = PMC.Workspace(s, grid; flux = (shape, st))
    PMC.sort_by_cell!(s, grid, ws)
    move!(s, grid; dmax = 0.99, kill = 0.05)
    PMC.sort_by_cell!(s, grid, ws)
    return grid, s, ws
end

function flux(mode, be, ::Type{T}, grid, s, ws, shape, st, dt; workgroupsize = nothing) where {T}
    J = zero_fields(be, T, grid)
    PMC.deposit_flux!(mode, J..., s, shape, st, grid, ws; dt, quantity_z = MomentumZ(), workgroupsize)
    return map(Array, J)
end

function check(::Type{T}, shape, st, n, ppc, tol) where {T}
    label = "$(T <: Dual ? "Dual{$(base_type(T)),3}" : string(T)) order $(PMC.order(shape)) $(join(map(x -> nameof(typeof(x)), st), "/")) $(join(n, "x"))"
    dt = T(base_type(T)(0.1))
    grid, s, ws = state(BE, T, shape, st, n, ppc)
    ref = flux(PMC.Reference(), BE, T, grid, s, ws, shape, st, dt)
    report(label, "Reference: continuity", continuity_residual(ref, s, shape, st, grid, dt), tol)
    report(label, "Reference: a second call gives other bits", !all(map(same, ref, flux(PMC.Reference(), BE, T, grid, s, ws, shape, st, dt))), false)
    for wg in (64, 256)
        report(label, "Reference: launch shape $wg gives other bits",
               !all(map(same, ref, flux(PMC.Reference(), BE, T, grid, s, ws, shape, st, dt; workgroupsize = wg))), false)
    end
    fast = flux(PMC.Fast(), BE, T, grid, s, ws, shape, st, dt)
    if T <: AbstractFloat
        report(label, "Fast: continuity", continuity_residual(fast, s, shape, st, grid, dt), tol)
        report(label, "Fast against Reference", field_difference(fast, ref), tol)
    else
        report(label, "Fast with a Dual gives other bits than Reference", !all(map(same, ref, fast)), false)
    end
    if !(BE isa CPU)
        gridc, sc, wsc = state(CPU(), T, shape, st, n, ppc)
        report(label, "the state differs from the CPU's", !all(map(same, species_arrays(s), species_arrays(sc))), false)
        refc = flux(PMC.Reference(), CPU(), T, gridc, sc, wsc, shape, st, dt)
        report(label, "Reference against the CPU's Reference", field_difference(ref, refc), tol)
        if T <: AbstractFloat
            report(label, "Fast against the CPU's Reference", field_difference(fast, refc), tol)
        end
    end
    return nothing
end

function main()
    count = 0
    for (T, tol) in ((Float32, 2f-4), (Float64, 1e-12), (Dual{Nothing,Float32,3}, 2f-4)), shape in (PMC.Linear, PMC.Quadratic),
        st in ((PMC.Node(), PMC.Node()), (PMC.Center(), PMC.Center()), (PMC.Center(), PMC.Node()), (PMC.Node(), PMC.Center())),
        (n, ppc) in (((37, 23), 5), ((96, 64), 8))

        check(T, shape, st, n, ppc, tol)
        count += 1
    end
    println("backend ", NAME, ", ", count, " cases. Largest value of each check over its bound:")
    for k in sort(collect(keys(WORST)))
        println("  ", rpad(k, 44), round(WORST[k]; sigdigits = 3))
    end
    isempty(FAILED) || error("$(length(FAILED)) checks failed:\n" * join(FAILED, "\n"))
    println("all checks passed")
end
Base.invokelatest(main)
