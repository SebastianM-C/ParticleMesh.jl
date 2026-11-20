# WP-078: Enzyme's reverse mode through the full trapped-beam case of WP-065
# (`scripts/optim/trapped2d_case.jl`: antenna, moving window, PMLs, field ionization,
# CompensatedGather, Reference mode), on a GPU, checked against ForwardDiff by the chain rule.
#
#     PM_BACKEND=cuda PM_CUDA_HEAP=<bytes> PM_ENZYME_PATCH=scripts/enzyme/active_reduce.jl \
#       JULIA_LOAD_PATH=@:<repo>:@stdlib \
#       julia +1.13 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/trapped.jl <command> [key=value ...]
#
# The module's test environment carries Enzyme and CUDACore; the repository's root project is
# stacked behind it for `ParticleMesh` (both resolve through the root manifest). Keys: ppl=32
# ny=2 (Δx = λ₀/ppl, Δy = π/(4·ny)), a0E=1.9 (the energy budget's a₀ at the base pulse), the point
# `theta=duration,waist,focus,density` (default the base point), seed=7.
#
# `steps` (milestone 1): from the state after a primal run to `tmid` (default 150), the reverse
# pass of `ks` (default 1,4,16) steps of `step!(st, p, i; observer = ())` followed by a kernel
# that writes J's per-slot summand (`slot_trapped`) of each electron species into an array,
# whose adjoint is seeded with ones (no host reduction inside the differentiated function), with
# `Active(p)`. The check: a Dual state with five partials at the same step, the first a random
# tangent v on every floating-point array of the state, the other four the parameters'
# tangents dp/dθ of the four free parameters (the state's θ-tangent reset to zero), stepped
# forward the same steps; the chain rule gives ⟨∇_x J, v⟩ and ⟨∇_p J, dp/dθ_j⟩ (+ the state's
# scalars), which must equal the four forward derivatives of J of the k steps at a fixed state.
#
# `grad` (milestone 2): dJ/dθ over the whole run. A primal run stores the state every `C` steps
# (default 100); the segments are then reversed from the last, the state's adjoint carried from
# one to the next in the shadow (`Duplicated(st, dst)`) and the parameters' adjoints summed. The
# adjoint of the initial state is contracted with the tangent of `initialize!` (a Dual start with
# four partials), and the parameters' with dp/dθ of `params(trapped_problem(...))` with duals in
# the four free parameters (no simulation). The comparison: the same Dual{4} run to the end.
#
# Writes result.toml and manifest.toml into scripts/runs/enzyme-trapped-<command>-<stamp>/.
# Time and GPU memory per phase are printed with UTC stamps, to be read against an nvidia-smi
# log sampled beside the run.

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(REPO, "scripts", "optim", "trapped2d_case.jl"))
using ParticleMeshElectromagnetic: CompensatedGather, ionization_electrons
using ParticleMeshCore: cellsize
using Random: Xoshiro

const COMMAND = get(ARGS, 1, "steps")
const OPTS = Dict(String(first(split(a, "="))) => String(last(split(a, "="; limit = 2))) for a in ARGS[2:end] if occursin("=", a))
opt(k, d) = get(OPTS, k, d)
const F = Float32
const PPL = parse(Int, opt("ppl", "32"))
const NY = parse(Int, opt("ny", "2"))
const A0E = parse(Float64, opt("a0E", "1.9"))
const THETA = haskey(OPTS, "theta") ? parse.(Float64, split(OPTS["theta"], ",")) : [TCASE.x0[i] for i in TCASE.free]
const SEED = parse(Int, opt("seed", "7"))
struct TrapTag end
const FREED = Set{String}()
const DualF{N} = ForwardDiff.Dual{TrapTag,F,N}

stamp() = Dates.format(now(UTC), "HH:MM:SS")
function devmem()
    ON_GPU || return (; used = 0.0, reserved = 0.0)
    m = CUDACore.MemoryInfo()
    return (; used = m.pool_used_bytes / 2^30, reserved = m.pool_reserved_bytes / 2^30, free = m.free_bytes / 2^30)
end
function mark(what)
    m = devmem()
    @printf("[%s] %s (pool used %.2f GiB, reserved %.2f GiB)\n", stamp(), what, m.used, m.reserved)
    flush(stdout)
end
reclaim() = (GC.gc(true); ON_GPU && CUDACore.reclaim())

"The case's start at the four free parameters `θ` (their element type), on this script's backend."
make_start(θ; ppl = PPL, ny = NY, extend = 0, energy = budget(A0E)) =
    start_of(trapped_problem(trapped_point(θ)...; energy = F(energy), ppl, ny, float = F, extend);
             ppl, ny, backend = backend(), gather = CompensatedGather(; k = 1))
"Duals of the four free parameters with `N` partials, θ_j's tangent the unit vector `j + offset`."
dual_theta(θ, N, offset) = [ForwardDiff.Dual{TrapTag}(F(θ[j]), ntuple(i -> F(i == j + offset), N)...) for j in eachindex(θ)]

# J's summand per slot, into `out` (the slot's weight × kinetic energy × position and trapping weights)
@kernel function summand_kernel!(out, @Const(status), @Const(c1), @Const(o1), @Const(u1), @Const(u2), @Const(u3), @Const(w), h1, lo, hi, ulo, uhi)
    i = @index(Global, Linear)
    @inbounds out[i] = slot_trapped(status[i], c1[i], o1[i], u1[i], u2[i], u3[i], w[i], h1, lo, hi, ulo, uhi)
end
function summands!(out, s, h1, fade, rise)
    T = eltype(s.w)
    summand_kernel!(get_backend(out))(out, s.status, s.cell[1], s.offset[1], s.u[1], s.u[2], s.u[3], s.w, h1,
                                      T(fade[1]), T(fade[2]), T(rise[1]), T(rise[2]); ndrange = length(out))
    return nothing
end
function steps_only!(st, p, first, k)
    for i in first:(first + k - 1)
        step!(st, p, i; observer = ())
    end
    return nothing
end
"k steps from step `first`, then J's summands of the plasma's electrons into `Je` and of the born electrons into `Jb`."
function steps_J!(Je, Jb, st, p, first, k, fade, rise)
    steps_only!(st, p, first, k)
    h1 = cellsize(st.grid)[1]
    summands!(Je, st.species[1], h1, fade, rise)
    summands!(Jb, ionization_electrons(st), h1, fade, rise)
    return nothing
end
area(st) = prod(cellsize(st.grid))
summand_buffers(st) = (similar(st.species[1].w), similar(ionization_electrons(st).w))
"J of a state through the summand kernel (the host adds the summands in slot order)."
function J_of(st; fade = TCASE.fade, rise = TCASE.rise)
    Je, Jb = summand_buffers(st)
    h1 = cellsize(st.grid)[1]
    summands!(Je, st.species[1], h1, fade, rise)
    summands!(Jb, ionization_electrons(st), h1, fade, rise)
    return area(st) * (foldl(+, Array(Je)) + foldl(+, Array(Jb)))
end

"""
The reverse pass of `steps_J!` with the summands' adjoints at `seeds` (the plasma's, the born
electrons'; ones for J itself): the state's adjoint (`dst`), the parameters' (`dp`) and J of the
state after the steps (the summands from the augmented forward, unweighted by the seeds).
"""
function reverse_J(st, p, first, k; fade = TCASE.fade, rise = TCASE.rise, seeds = (1, 1))
    Je, Jb = summand_buffers(st)
    fill!(Je, 0); fill!(Jb, 0)
    dJe, dJb = similar(Je), similar(Jb)
    fill!(dJe, seeds[1]); fill!(dJb, seeds[2])
    dst = Enzyme.make_zero(st)
    res = Enzyme.autodiff(set_runtime_activity(Reverse), steps_J!, Const, Duplicated(Je, dJe), Duplicated(Jb, dJb), Duplicated(st, dst),
                          Active(p), Const(first), Const(k), Const(fade), Const(rise))
    sync()
    J = area(st) * (foldl(+, Array(Je)) + foldl(+, Array(Jb)))
    return (; dst, dp = res[1][4], J)
end
"The reverse pass of `k` steps from `first` with the adjoint of the state after them in `dst` (overwritten with the adjoint of the state before them); the parameters' adjoint."
function reverse_steps!(dst, st, p, first, k)
    res = Enzyme.autodiff(set_runtime_activity(Reverse), steps_only!, Const, Duplicated(st, dst), Active(p), Const(first), Const(k))
    sync()
    return res[1][2]
end

# every array of a state (any element type), by its path, each once
all_arrays!(out, path, x::AbstractArray) = (any(y -> y === x, values(out)) || (out[path] = x); out)
all_arrays!(out, path, x::Union{Tuple,NamedTuple}) = (foreach(k -> all_arrays!(out, "$path.$k", x[k]), keys(x)); out)
function all_arrays!(out, path, x)
    (isstructtype(typeof(x)) && !(x isa Module) && !(x isa Type)) || return out
    for f in fieldnames(typeof(x))
        isdefined(x, f) && all_arrays!(out, "$path.$f", getfield(x, f))
    end
    return out
end
all_arrays(x) = all_arrays!(Dict{String,Any}(), "st", x)

# every floating-point scalar of a struct (or a Dual), by its path; arrays are `leaves`'
scalars!(out, path, x::Union{AbstractFloat,ForwardDiff.Dual}) = (out[path] = x; out)
scalars!(out, path, x::AbstractArray) = out
scalars!(out, path, x::Union{Tuple,NamedTuple}) = (foreach(k -> scalars!(out, "$path.$k", x[k]), keys(x)); out)
function scalars!(out, path, x)
    (isstructtype(typeof(x)) && !(x isa Module) && !(x isa Type)) || return out
    for f in fieldnames(typeof(x))
        isdefined(x, f) && scalars!(out, "$path.$f", getfield(x, f))
    end
    return out
end
scalars(x) = scalars!(Dict{String,Any}(), "x", x)
partial(x::ForwardDiff.Dual, j) = ForwardDiff.partials(x, j)
partial(x::Real, j) = zero(x)

"⟨adjoint, tangent j⟩ over the scalars of two structs of the same shape (an adjoint and its Dual twin), by path."
function scalar_dot(adj, dual, j)
    A, D = scalars(adj), scalars(dual)
    return sum((Float64(A[k]) * Float64(partial(D[k], j)) for k in intersect(keys(A), keys(D))); init = 0.0)
end
"⟨adjoint, tangent j⟩ over the floating-point arrays of a state's adjoint and a Dual state of the same shape, by path."
function array_dot(adj, dual, j)
    A, D = leaves(adj), leaves(dual)
    return sum((Float64(sum(A[k] .* ForwardDiff.partials.(D[k], j))) for k in intersect(keys(A), keys(D)) if eltype(D[k]) <: ForwardDiff.Dual); init = 0.0)
end
"The same, term by term (for the cancellation)."
array_terms(adj, dual, j) = [Float64(sum(A .* ForwardDiff.partials.(D, j))) for (A, D) in
                             ((leaves(adj)[k], leaves(dual)[k]) for k in intersect(keys(leaves(adj)), keys(leaves(dual))) if eltype(leaves(dual)[k]) <: ForwardDiff.Dual)]

function write_run(name, result)
    out = joinpath(REPO, "scripts", "runs", "enzyme-trapped-$name-$(Dates.format(now(UTC), "yyyymmddTHHMMSS"))")
    mkpath(out)
    open(io -> TOML.print(io, result), joinpath(out, "result.toml"), "w")
    commit = try readchomp(`git -C $REPO rev-parse HEAD`) catch; "unknown" end
    dirty = try !isempty(readchomp(`git -C $REPO status --porcelain --untracked-files=no`)) catch; missing end
    hashfile(f) = bytes2hex(open(sha256, f))
    inputs = [@__FILE__, joinpath(@__DIR__, "common.jl"), joinpath(@__DIR__, "active_reduce.jl"), joinpath(REPO, "scripts", "optim", "trapped2d_case.jl"),
              joinpath(REPO, "scripts", "optim", "born2d_case.jl"), joinpath(REPO, "Manifest.toml")]
    open(io -> TOML.print(io, Dict(
        "commit" => commit, "dirty" => string(dirty), "julia" => string(VERSION), "threads" => Threads.nthreads(),
        "cpu" => Sys.cpu_info()[1].model, "args" => ARGS, "backend" => string(backend()),
        "gpu" => ON_GPU ? CUDACore.name(CUDACore.device()) : "none", "patches" => get(ENV, "PM_ENZYME_PATCH", ""),
        "cuda_heap" => get(ENV, "PM_CUDA_HEAP", "default"), "work_package" => "WP-078",
        "versions" => Dict(string(m) => string(pkgversion(m)) for m in (Enzyme, ForwardDiff, KernelAbstractions)),
        "inputs" => Dict(relpath(f, REPO) => hashfile(f) for f in inputs if isfile(f)))), joinpath(out, "manifest.toml"), "w")
    println("wrote ", relpath(out, REPO))
    return out
end

# ---------------------------------------------------------------- milestone 1
function main_steps()
    tmid = parse(Float64, opt("tmid", "150"))
    ks = parse.(Int, split(opt("ks", "1,4,16"), ","))
    # the check's functional: J's summand with the trapping weight at 1 for every electron by default
    # (u_x ≥ -2), so that it is not zero mid-run, where no electron is trapped yet
    rise = Tuple(parse.(Float64, split(opt("rise", "-3,-2"), ",")))
    fade = Tuple(parse.(Float64, split(opt("fade", join(TCASE.fade, ",")), ",")))
    mark("start: ppl $PPL ny $NY a0E $A0E θ $THETA, ks $ks")
    st, p, prob = make_start(F.(THETA))
    kmid = round(Int, tmid / Float64(st.dt))
    std, pd, _ = make_start(dual_theta(THETA, 5, 1))
    mark("states made; $(prob.nsteps) steps to t_end, mid-run step $kmid (t = $(kmid * Float64(st.dt)))")
    t_mid = @elapsed (steps_only!(st, p, 1, kmid); sync())
    t_midd = @elapsed (steps_only!(std, pd, 1, kmid); sync())
    mark(@sprintf("primal to step %d %.1f s, Dual{5} %.1f s", kmid, t_mid, t_midd))
    # The two runs differ by step kmid (the Float32 and the Dual kernels compile differently on the
    # device, and the particles' paths separate): the Dual state takes every array of the Float32
    # state, its values in the Duals' values
    A, Ad = all_arrays(st), all_arrays(std)
    gaps0 = Dict(k => Float64(maximum(abs.(Float64.(ForwardDiff.value.(Ad[k])) .- Float64.(A[k])); init = 0.0))
                 for k in intersect(keys(A), keys(Ad)) if size(A[k]) == size(Ad[k]) && (eltype(A[k]) <: Real))
    println("  before the copy, arrays that differ between the separate runs: ", sort([(k, g) for (k, g) in gaps0 if g > 0]; by = x -> -x[2])[1:min(end, 8)])
    for k in intersect(keys(A), keys(Ad))
        size(A[k]) == size(Ad[k]) || continue
        if eltype(Ad[k]) <: ForwardDiff.Dual
            Ad[k] .= ForwardDiff.Dual{TrapTag}.(A[k], ntuple(_ -> zero(F), 5)...)
        elseif eltype(Ad[k]) == eltype(A[k])
            copyto!(Ad[k], A[k])
        end
    end
    L, Ld = leaves(st), leaves(std)
    common = sort!(collect(intersect(keys(L), keys(Ld))))
    gaps = Dict(k => Float64(maximum(abs.(ForwardDiff.value.(Ld[k]) .- L[k]); init = zero(F))) for k in common)
    gap = maximum(values(gaps))
    println("  arrays whose values differ between the two states: ", [(k, g) for (k, g) in gaps if g > 0])
    println("  primal gap between the Float32 and the Dual state at step $kmid: $gap; ",
            "alive: ", alive_count(st.species[1]), " plasma, ", alive_count(ionization_electrons(st)), " born")
    # the Dual state's tangent: partial 1 a random v on every array, partials 2–5 zero (a fixed state)
    rng = Xoshiro(SEED)
    V = Dict{String,Any}()
    for k in common
        eltype(Ld[k]) <: ForwardDiff.Dual || continue
        v = similar(L[k]); copyto!(v, randn(rng, F, size(L[k]))); V[k] = v
        z = zero(F)
        Ld[k] .= ForwardDiff.Dual{TrapTag}.(ForwardDiff.value.(Ld[k]), v, z, z, z, z)
    end
    for k in setdiff(keys(Ld), keys(L))
        eltype(Ld[k]) <: ForwardDiff.Dual && (Ld[k] .= ForwardDiff.Dual{TrapTag}.(ForwardDiff.value.(Ld[k]), zero(F), zero(F), zero(F), zero(F), zero(F)))
    end
    sc = [k for (k, x) in scalars(std) if x isa ForwardDiff.Dual && any(!iszero, ForwardDiff.partials(x))]
    println("  scalars of the Dual state with a tangent: ", sc)
    rows = Dict{String,Any}[]
    first = kmid + 1
    for (n, k) in enumerate(ks)
        row = Dict{String,Any}("steps" => k)
        try
            if n == 1
                reclaim(); mark("reverse of $k steps, first call (compiles)")
                row["first_call_seconds"] = @elapsed reverse_J(deepcopy(st), p, first, k; fade, rise)
                mark(@sprintf("first call %.1f s", row["first_call_seconds"]))
            end
            reclaim(); before = devmem(); mark("reverse of $k steps")
            sc0 = deepcopy(st)
            row["reverse_seconds"] = @elapsed (r = reverse_J(sc0, p, first, k; fade, rise))
            after = devmem()
            mark(@sprintf("reverse of %d steps %.3f s (%.2f ms per step)", k, row["reverse_seconds"], 1e3 * row["reverse_seconds"] / k))
            row["pool_reserved_after_GiB"] = after.reserved
            row["pool_reserved_before_GiB"] = before.reserved
            # primal of the same steps
            s1 = deepcopy(st); sync()
            row["primal_seconds"] = @elapsed (steps_only!(s1, p, first, k); sync())
            J1 = J_of(s1; fade, rise)
            # forward: the Dual state
            d1 = deepcopy(std); sync()
            row["dual5_seconds"] = @elapsed (steps_only!(d1, pd, first, k); sync())
            Jd = J_of(d1; fade, rise)
            fwd = [Float64(ForwardDiff.partials(Jd, j)) for j in 1:5]
            a = Float64(area(st))
            terms = a .* array_terms(r.dst, std, 1)
            rev_v = sum(terms) + a * scalar_dot(r.dst, std, 1)
            rev_p = [a * (scalar_dot(r.dp, pd, j) + scalar_dot(r.dst, std, j) + array_dot(r.dst, std, j)) for j in 2:5]
            rev = [rev_v; rev_p]
            rel = abs.(rev .- fwd) ./ max.(abs.(fwd), floatmin())
            row["J"] = Float64(r.J); row["J_primal"] = Float64(J1); row["J_dual"] = Float64(ForwardDiff.value(Jd))
            row["forward"] = fwd; row["reverse"] = rev; row["relative_difference"] = rel
            row["cancellation_v"] = sum(abs, terms) / max(abs(sum(terms)), floatmin())
            @printf("  k = %d: J %.8g (reverse's forward) %.8g (primal) %.8g (dual); reverse %.3f s, primal %.4f s, Dual{5} %.4f s\n",
                    k, r.J, J1, ForwardDiff.value(Jd), row["reverse_seconds"], row["primal_seconds"], row["dual5_seconds"])
            for (j, name) in enumerate(("random v", TCASE.names[[TCASE.free...]]...))
                @printf("    %-10s forward %+.8e reverse %+.8e relative difference %.2e\n", name, fwd[j], rev[j], rel[j])
            end
            @printf("    cancellation of ⟨∇J, v⟩ over the arrays %.1f\n", row["cancellation_v"])
            r = nothing; sc0 = nothing
        catch e
            row["error"] = sprint(showerror, e)[1:min(end, 2000)]
            println("  k = $k FAILED: ", row["error"])
            push!(rows, row)
            break
        end
        push!(rows, row)
    end
    println("  kernel arguments with dropped adjoints: ", isdefined(Main, :ACTIVE_SEEN) ? sort!(collect(ACTIVE_SEEN)) : "n/a")
    write_run("steps", Dict("case" => Dict("ppl" => PPL, "ny" => NY, "a0E" => A0E, "theta" => THETA, "tmid" => tmid, "kmid" => kmid, "rise" => collect(rise), "fade" => collect(fade), "eltype" => "Float32",
                                           "gather" => "CompensatedGather(k = 1)", "mode" => "Reference", "seed" => SEED),
                            "primal_gap" => gap, "seconds" => Dict("primal_to_mid" => t_mid, "dual5_to_mid" => t_midd), "rows" => rows))
end

# ---------------------------------------------------------------- milestone 2
"""
    reverse_run(θ; C, extend, fade, rise, objective, kappa, ppl, ny, energy, verbose) -> NamedTuple

The objective at the free parameters `θ` and its gradient in them by Enzyme's reverse mode: a
primal run that stores the state before every `C`-th step, the segments reversed from the last
(the state's adjoint carried between them in one shadow, the parameters' adjoints summed), then
the adjoint of the start contracted with the tangent of `initialize!` and the parameters'
adjoints with dp/dθ (a Dual{4} start, no step). `objective` as `trapped2d_opt.jl`'s
`objective_value` (any, born, plasmacap): the summands' seeds are its derivatives in the two parts
at the end of the primal run. Returns J, its two parts, the gradient (and its parts from the
start and from the parameters), the final primal state's `trapped_summary`, the Dual{4} start
(`std0`, `pd`) for a forward comparison, and the seconds of each phase.
"""
function reverse_run(θ; C::Integer = 100, extend::Integer = 0, fade = TCASE.fade, rise = TCASE.rise, objective = "any", kappa = 0.05,
                     ppl = PPL, ny = NY, energy = budget(A0E), verbose = true, summary = true)
    st, p, prob = make_start(F.(θ); ppl, ny, extend, energy)
    nsteps = prob.nsteps
    s0area = Float64(area(st))
    # the checkpoints in host memory: every array of the state, by path (a state copy is about
    # 0.3 GiB at λ₀/32, ny = 2, the tape about 0.35 GiB per step: both would not fit the card)
    work = deepcopy(st)
    saved = Any[]
    t_fwd = @elapsed begin
        for k in 1:nsteps
            (k - 1) % C == 0 && push!(saved, Dict(path => Array(a) for (path, a) in all_arrays(st)))
            step!(st, p, k; observer = ())
        end
        sync()
    end
    lost = ionization_lost(st)
    lost == 0 || error("$lost births found no free slot: raise `capacity`")
    parts = trapped_parts(st; fade, rise)
    born, plasma = Float64(parts.born), Float64(parts.plasma)
    J = objective == "any" ? born + plasma : objective == "born" ? born :
        objective == "plasmacap" ? born + kappa * tanh(plasma / kappa) : error("objective any, born or plasmacap, got $objective")
    seeds = objective == "any" ? (1, 1) : objective == "born" ? (0, 1) : (sech(plasma / kappa)^2, 1)
    summ = summary ? trapped_summary(st; fade, rise) : nothing
    verbose && mark(@sprintf("primal with %d checkpoints %.1f s; J = %.8g (born %.6g, plasma %.6g)", length(saved), t_fwd, J, born, plasma))
    st = nothing; reclaim()
    dst = nothing
    dps = Any[]
    seg_seconds = Float64[]
    Jrev = NaN
    t_rev = @elapsed for i in length(saved):-1:1
        first = 1 + (i - 1) * C
        k = min(C, nsteps - first + 1)
        sc = work
        for (path, a) in all_arrays(work)
            haskey(saved[i], path) || continue
            try
                copyto!(a, saved[i][path])
            catch err
                # a temporary the step frees and allocates again (its state field then holds a freed array)
                (err isa ArgumentError && occursin("freed", sprint(showerror, err))) || rethrow()
                path in FREED || (push!(FREED, path); println("  checkpoint restore skips the freed array $path"))
            end
        end
        t = @elapsed if dst === nothing
            r = reverse_J(sc, p, first, k; fade, rise, seeds = F.(seeds))
            dst, Jrev = r.dst, r.J
            push!(dps, r.dp)
            r = nothing
        else
            push!(dps, reverse_steps!(dst, sc, p, first, k))
        end
        push!(seg_seconds, t)
        saved[i] = nothing
        verbose && i == length(saved) && mark(@sprintf("last segment (steps %d to %d) %.1f s; J from the reverse's forward %.8g", first, first + k - 1, t, Jrev))
        i % 4 == 0 && (reclaim(); verbose && mark(@sprintf("segment %d (steps %d to %d) %.1f s", i, first, first + k - 1, t)))
    end
    verbose && mark(@sprintf("reverse of %d segments %.1f s", length(seg_seconds), t_rev))
    t0 = time()
    std0, pd, _ = make_start(dual_theta(θ, 4, 0); ppl, ny, extend, energy)
    t_tan = time() - t0
    g_state = [s0area * (array_dot(dst, std0, j) + scalar_dot(dst, std0, j)) for j in 1:4]
    g_params = [s0area * sum(scalar_dot(dp, pd, j) for dp in dps) for j in 1:4]
    dst = nothing; reclaim()
    return (; J, born, plasma, J_reverse_forward = Float64(Jrev), gradient = g_state .+ g_params, g_state, g_params, summary = summ,
            std0, pd, nsteps, checkpoints = length(saved), seconds = (; primal = t_fwd, reverse = t_rev, tangent = t_tan, segments = reverse(seg_seconds)))
end

function main_grad()
    C = parse(Int, opt("C", "100"))
    objective = opt("objective", "any")
    mark("start: ppl $PPL ny $NY a0E $A0E θ $THETA, checkpoints every $C steps, objective $objective")
    r = reverse_run(THETA; C, objective)
    g = r.gradient
    # the forward reference: the same Dual{4} run to the end
    std0, pd = r.std0, r.pd
    t_dual = @elapsed (run!(std0, pd, r.nsteps; observer = ()); sync())
    parts = trapped_parts(std0)
    Jd = objective == "any" ? parts.born + parts.plasma : objective == "born" ? parts.born : parts.born + 0.05f0 * tanh(parts.plasma / 0.05f0)
    gd = [Float64(ForwardDiff.partials(Jd, j)) for j in 1:4]
    rel = abs.(g .- gd) ./ abs.(gd)
    @printf("  J = %.8g (primal), %.8g (Dual{4}); primal with checkpoints %.1f s, reverse %.1f s, Dual{4} %.1f s\n",
            r.J, ForwardDiff.value(Jd), r.seconds.primal, r.seconds.reverse, t_dual)
    for j in 1:4
        @printf("    dJ/d%-8s reverse %+.8e (start %+.3e, parameters %+.3e)  Dual{4} %+.8e  relative difference %.2e\n",
                TCASE.names[TCASE.free[j]], g[j], r.g_state[j], r.g_params[j], gd[j], rel[j])
    end
    println("  kernel arguments with dropped adjoints: ", isdefined(Main, :ACTIVE_SEEN) ? sort!(collect(ACTIVE_SEEN)) : "n/a")
    write_run("grad", Dict("case" => Dict("ppl" => PPL, "ny" => NY, "a0E" => A0E, "theta" => THETA, "eltype" => "Float32", "nsteps" => r.nsteps,
                                          "checkpoint_every" => C, "checkpoints" => r.checkpoints, "objective" => objective,
                                          "gather" => "CompensatedGather(k = 1)", "mode" => "Reference"),
                           "J" => r.J, "J_born" => r.born, "J_plasma" => r.plasma, "J_dual" => Float64(ForwardDiff.value(Jd)), "J_reverse_forward" => r.J_reverse_forward,
                           "gradient_reverse" => g, "gradient_reverse_start" => r.g_state, "gradient_reverse_params" => r.g_params,
                           "gradient_dual4" => gd, "relative_difference" => rel,
                           "seconds" => Dict("primal_with_checkpoints" => r.seconds.primal, "reverse" => r.seconds.reverse, "dual4" => t_dual,
                                             "start_tangent" => r.seconds.tangent, "segments" => r.seconds.segments)))
end

# ---------------------------------------------------------------- milestone 3
# WP-068's optimization (`scripts/optim/trapped2d_opt.jl`) with the reverse gradient: the same
# starts, coordinates, stopping rule and point store, each point one `reverse_run`. Runs in an
# environment with the scripts environment's packages and Enzyme (its manifest is recorded).
#     trapped.jl opt <dir> [trapped2d_opt.jl's opt keys] C=<checkpoint interval>
function reverse_point(free; ppl, ny, energy, read = (;), objective = "any", kappa = 0.05, extend = 0, C = 100, kw...)
    fade = get(read, :fade, TCASE.fade); rise = get(read, :rise, TCASE.rise)
    t0 = time()
    r = reverse_run(collect(Float64, free); C, extend, fade, rise, objective, kappa, ppl, ny, energy, verbose = false)
    t = time() - t0
    out = Dict{String,Any}(string(k) => v for (k, v) in pairs(r.summary))
    x = trapped_point(collect(Float64, free))
    out["objective"] = objective
    out["J"] = r.J
    out["free"] = collect(Float64, free)
    out["x"] = collect(Float64, x)
    out["z"] = collect(Float64, free) ./ TRAPPED_BOX.scale
    out["a0"] = a0_for_energy(x[1], x[2]; energy = Float64(energy))
    out["gradient"] = r.gradient
    out["gradient_start"] = r.g_state
    out["gradient_params"] = r.g_params
    out["seconds"] = t
    out["seconds_phases"] = Dict("primal" => r.seconds.primal, "reverse" => r.seconds.reverse, "tangent" => r.seconds.tangent)
    out["eltype"] = string(F)
    out["ppl"], out["ny"] = ppl, ny
    out["gather"] = "compensated"
    out["gradient_mode"] = "enzyme-reverse"
    out["checkpoint_every"] = C
    @printf("[%s] point %s: J %.8g, gradient %s, %.1f s (primal %.1f, reverse %.1f)\n", stamp(), string(round.(free; sigdigits = 6)), r.J,
            string(round.(r.gradient; sigdigits = 6)), t, r.seconds.primal, r.seconds.reverse); flush(stdout)
    return out
end

function main_opt()
    dir = ARGS[2]
    include(joinpath(REPO, "scripts", "optim", "trapped2d_opt.jl"))
    C = parse(Int, opt("C", "100"))
    o, _ = Base.invokelatest(parse_options, filter(a -> !startswith(a, "C="), ARGS[3:end]), OPT_DEFAULTS)
    o["gradient"] = "enzyme-reverse"; o["checkpoint_every"] = C
    Base.invokelatest() do
        cfg = point_config(delete!(copy(o), "gradient"))
        d = run_manifest("trapped2d-opt-reverse", dir, delete!(delete!(copy(o), "gradient"), "checkpoint_every");
                         extra_params = Dict{String,Any}("gradient" => "enzyme-reverse", "checkpoint_every" => C))
        cfgr = (; ppl = cfg.ppl, ny = cfg.ny, energy = cfg.energy, read = cfg.read, objective = cfg.objective, kappa = cfg.kappa, extend = cfg.extend, C)
        store = PointStore(joinpath(d, "points"); run = free -> reverse_point(free; cfgr...))
        coords = coords_of(o)
        starts = start_points(o)
        names = isempty(o["from"]) ? split(o["starts"], ",") : ["from"]
        zb = z_of(coords, starts[Symbol(first(names))])
        base = record!(store, free_of(coords, zb))
        gb = grad_z(coords, zb, base["gradient"])
        pgtol = o["pgtol"] * maximum(abs, gb)
        println("base point: J = $(base["J"]) at $(base["free"]), gradient in z $gb, pgtol $pgtol"); flush(stdout)
        for name in names
            sdir = joinpath(d, name)
            isfile(joinpath(sdir, "result.json")) && (println("start $name: done"); continue)
            println("start $name from $(starts[Symbol(name)])"); flush(stdout)
            r = optimize_start(store, starts[Symbol(name)], sdir; maxiters = o["maxiters"], pgtol, coords)
            println("start $name: J = $(r["J"]) at $(r["free"]) (a₀ $(r["a0"])), $(r["retcode"]) after $(r["iterations"]) iterations, ",
                    "$(r["points_run"]) points run in $(round(r["seconds"]; digits = 1)) s"); flush(stdout)
        end
        println("optimization done: $d")
    end
end

COMMAND == "steps" ? main_steps() : COMMAND == "grad" ? main_grad() : COMMAND == "opt" ? main_opt() : error("trapped.jl steps|grad|opt [key=value ...]")
