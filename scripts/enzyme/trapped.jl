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
make_start(θ) = start_of(trapped_problem(trapped_point(θ)...; energy = F(budget(A0E)), ppl = PPL, ny = NY, float = F);
                         ppl = PPL, ny = NY, backend = backend(), gather = CompensatedGather(; k = 1))
"Duals of the four free parameters with `N` partials, θ_j's tangent the unit vector `j + offset`."
dual_theta(θ, N, offset) = [ForwardDiff.Dual{TrapTag}(F(θ[j]), ntuple(i -> F(i == j + offset), N)...) for j in eachindex(θ)]

# J's summand per slot, into `out` (the slot's weight × kinetic energy × position and trapping weights)
@kernel function summand_kernel!(out, @Const(status), @Const(c1), @Const(o1), @Const(u1), @Const(u2), @Const(u3), @Const(w), h1, lo, hi, ulo, uhi)
    i = @index(Global, Linear)
    @inbounds out[i] = slot_trapped(status[i], c1[i], o1[i], u1[i], u2[i], u3[i], w[i], h1, lo, hi, ulo, uhi)
end
function summands!(out, s, h1)
    T = eltype(s.w)
    summand_kernel!(get_backend(out))(out, s.status, s.cell[1], s.offset[1], s.u[1], s.u[2], s.u[3], s.w, h1,
                                      T(TCASE.fade[1]), T(TCASE.fade[2]), T(TCASE.rise[1]), T(TCASE.rise[2]); ndrange = length(out))
    return nothing
end
function steps_only!(st, p, first, k)
    for i in first:(first + k - 1)
        step!(st, p, i; observer = ())
    end
    return nothing
end
"k steps from step `first`, then J's summands of the plasma's electrons into `Je` and of the born electrons into `Jb`."
function steps_J!(Je, Jb, st, p, first, k)
    steps_only!(st, p, first, k)
    h1 = cellsize(st.grid)[1]
    summands!(Je, st.species[1], h1)
    summands!(Jb, ionization_electrons(st), h1)
    return nothing
end
area(st) = prod(cellsize(st.grid))
summand_buffers(st) = (similar(st.species[1].w), similar(ionization_electrons(st).w))
"J of a state through the summand kernel (the host adds the summands in slot order)."
function J_of(st)
    Je, Jb = summand_buffers(st)
    h1 = cellsize(st.grid)[1]
    summands!(Je, st.species[1], h1)
    summands!(Jb, ionization_electrons(st), h1)
    return area(st) * (foldl(+, Array(Je)) + foldl(+, Array(Jb)))
end

"The reverse pass of `steps_J!` with the summands' adjoints at ones: the state's adjoint (`dst`) and the parameters' (`dp`)."
function reverse_J(st, p, first, k)
    Je, Jb = summand_buffers(st)
    fill!(Je, 0); fill!(Jb, 0)
    dJe, dJb = similar(Je), similar(Jb)
    fill!(dJe, 1); fill!(dJb, 1)
    dst = Enzyme.make_zero(st)
    res = Enzyme.autodiff(set_runtime_activity(Reverse), steps_J!, Const, Duplicated(Je, dJe), Duplicated(Jb, dJb), Duplicated(st, dst),
                          Active(p), Const(first), Const(k))
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
    mark("start: ppl $PPL ny $NY a0E $A0E θ $THETA, ks $ks")
    st, p, prob = make_start(F.(THETA))
    kmid = round(Int, tmid / Float64(st.dt))
    std, pd, _ = make_start(dual_theta(THETA, 5, 1))
    mark("states made; $(prob.nsteps) steps to t_end, mid-run step $kmid (t = $(kmid * Float64(st.dt)))")
    t_mid = @elapsed (steps_only!(st, p, 1, kmid); sync())
    t_midd = @elapsed (steps_only!(std, pd, 1, kmid); sync())
    mark(@sprintf("primal to step %d %.1f s, Dual{5} %.1f s", kmid, t_mid, t_midd))
    L, Ld = leaves(st), leaves(std)
    common = sort!(collect(intersect(keys(L), keys(Ld))))
    gap = maximum(k -> Float64(maximum(abs.(ForwardDiff.value.(Ld[k]) .- L[k]); init = zero(F))), common)
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
                row["first_call_seconds"] = @elapsed reverse_J(deepcopy(st), p, first, k)
                mark(@sprintf("first call %.1f s", row["first_call_seconds"]))
            end
            reclaim(); before = devmem(); mark("reverse of $k steps")
            sc0 = deepcopy(st)
            row["reverse_seconds"] = @elapsed (r = reverse_J(sc0, p, first, k))
            after = devmem()
            mark(@sprintf("reverse of %d steps %.3f s (%.2f ms per step)", k, row["reverse_seconds"], 1e3 * row["reverse_seconds"] / k))
            row["pool_reserved_after_GiB"] = after.reserved
            row["pool_reserved_before_GiB"] = before.reserved
            # primal of the same steps
            s1 = deepcopy(st); sync()
            row["primal_seconds"] = @elapsed (steps_only!(s1, p, first, k); sync())
            J1 = J_of(s1)
            # forward: the Dual state
            d1 = deepcopy(std); sync()
            row["dual5_seconds"] = @elapsed (steps_only!(d1, pd, first, k); sync())
            Jd = J_of(d1)
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
    write_run("steps", Dict("case" => Dict("ppl" => PPL, "ny" => NY, "a0E" => A0E, "theta" => THETA, "tmid" => tmid, "kmid" => kmid, "eltype" => "Float32",
                                           "gather" => "CompensatedGather(k = 1)", "mode" => "Reference", "seed" => SEED),
                            "primal_gap" => gap, "seconds" => Dict("primal_to_mid" => t_mid, "dual5_to_mid" => t_midd), "rows" => rows))
end

# ---------------------------------------------------------------- milestone 2
function main_grad()
    C = parse(Int, opt("C", "100"))
    mark("start: ppl $PPL ny $NY a0E $A0E θ $THETA, checkpoints every $C steps")
    st, p, prob = make_start(F.(THETA))
    nsteps = prob.nsteps
    s0 = deepcopy(st)
    # the primal run, saving the state before steps 1, C + 1, 2C + 1, ...
    saved = Any[]
    t_fwd = @elapsed begin
        for k in 1:nsteps
            (k - 1) % C == 0 && push!(saved, deepcopy(st))
            step!(st, p, k; observer = ())
        end
        sync()
    end
    J = trapped_energy(st)
    lost = ionization_lost(st)
    mark(@sprintf("primal with %d checkpoints %.1f s; J = %.8g; births without a slot %d", length(saved), t_fwd, J, lost))
    st = nothing; reclaim()
    # reverse, segment by segment from the last
    dst = nothing
    dps = Any[]
    seg_seconds = Float64[]
    Jrev = NaN
    t_rev = @elapsed for i in length(saved):-1:1
        first = 1 + (i - 1) * C
        k = min(C, nsteps - first + 1)
        sc = saved[i]
        t = @elapsed if dst === nothing
            r = reverse_J(sc, p, first, k)
            dst, Jrev = r.dst, r.J
            push!(dps, r.dp)
        else
            push!(dps, reverse_steps!(dst, sc, p, first, k))
        end
        push!(seg_seconds, t)
        saved[i] = nothing
        i == length(saved) && mark(@sprintf("last segment (steps %d to %d) %.1f s, J from the reverse's forward %.8g", first, first + k - 1, t, Jrev))
        i % 4 == 0 && (reclaim(); mark(@sprintf("segment %d (steps %d to %d) %.1f s", i, first, first + k - 1, t)))
    end
    mark(@sprintf("reverse of %d segments %.1f s", length(seg_seconds), t_rev))
    # the chain to the four free parameters
    t_tan = @elapsed (std0, pd, _ = make_start(dual_theta(THETA, 4, 0)))
    a = Float64(area(s0))
    g_state = [a * (array_dot(dst, std0, j) + scalar_dot(dst, std0, j)) for j in 1:4]
    g_params = [a * sum(scalar_dot(dp, pd, j) for dp in dps) for j in 1:4]
    g = g_state .+ g_params
    mark(@sprintf("tangent of the start %.1f s", t_tan))
    # the forward reference: the same Dual{4} run to the end
    t_dual = @elapsed (run!(std0, pd, nsteps; observer = ()); sync())
    Jd = trapped_energy(std0)
    gd = [Float64(ForwardDiff.partials(Jd, j)) for j in 1:4]
    rel = abs.(g .- gd) ./ abs.(gd)
    @printf("  J = %.8g (primal), %.8g (Dual{4}); primal %.1f s, Dual{4} %.1f s, reverse %.1f s\n", J, ForwardDiff.value(Jd), t_fwd, t_dual, t_rev)
    for j in 1:4
        @printf("    d J/d %-8s reverse %+.8e (start %+.3e, parameters %+.3e)  Dual{4} %+.8e  relative difference %.2e\n",
                TCASE.names[TCASE.free[j]], g[j], g_state[j], g_params[j], gd[j], rel[j])
    end
    println("  kernel arguments with dropped adjoints: ", isdefined(Main, :ACTIVE_SEEN) ? sort!(collect(ACTIVE_SEEN)) : "n/a")
    write_run("grad", Dict("case" => Dict("ppl" => PPL, "ny" => NY, "a0E" => A0E, "theta" => THETA, "eltype" => "Float32", "nsteps" => nsteps,
                                          "checkpoint_every" => C, "gather" => "CompensatedGather(k = 1)", "mode" => "Reference"),
                           "J" => Float64(J), "J_dual" => Float64(ForwardDiff.value(Jd)), "J_reverse_forward" => Float64(Jrev),
                           "gradient_reverse" => g, "gradient_reverse_start" => g_state, "gradient_reverse_params" => g_params,
                           "gradient_dual4" => gd, "relative_difference" => rel,
                           "seconds" => Dict("primal_with_checkpoints" => t_fwd, "reverse" => t_rev, "dual4" => t_dual, "start_tangent" => t_tan,
                                             "segments" => reverse(seg_seconds))))
end

COMMAND == "steps" ? main_steps() : COMMAND == "grad" ? main_grad() : error("trapped.jl steps|grad [key=value ...]")
