# WP-050 spike: Enzyme reverse mode through single kernels of the 2D step, one at a time
# (PM_BACKEND=cuda for the GPU, PM_MODE=reference, ONLY=<name>, PM_ENZYME_PATCH as in chain_rule.jl).
using ParticleMeshCore, ParticleMeshElectromagnetic, KernelAbstractions, Enzyme
using ParticleMeshCore: Fast, Reference
const MODE = get(ENV, "PM_MODE", "fast") == "reference" ? Reference() : Fast()
foreach(f -> isempty(f) || include(abspath(f)), split(get(ENV, "PM_ENZYME_PATCH", ""), ';'))
isdefined(Main, :ACTIVE_SEEN) || @eval const ACTIVE_SEEN = Set{String}()
const ON_GPU = get(ENV, "PM_BACKEND", "cpu") == "cuda"
if ON_GPU
    using CUDACore
    CUDACore.allowscalar(false)
end
const BE = ON_GPU ? CUDACore.CUDABackend() : CPU()
const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(REPO, "lib/ParticleMeshElectromagnetic/test/em2d_cases.jl"))
const PME = ParticleMeshElectromagnetic
const PMC = ParticleMeshCore
prob = plasma_step_2d(Float64; amplitude = 0.1, nsteps = 1, seed = 7)
prob = EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, prob.species[1:1], prob.seed, prob.kmode)
st = EMState(prob, EMNumerics(; mode = MODE, current = Esirkepov()); backend = BE)
initialize!(st, prob)
const p = params(prob)

@kernel function k_scale!(s)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == 0x01
        s.u[1][i] = 2 * s.u[1][i] + s.w[i]
    end
end
@kernel function k_arrays!(u1, w, status)
    i = @index(Global, Linear)
    @inbounds if status[i] == 0x01
        u1[i] = 2 * u1[i] + w[i]
    end
end
@kernel function k_gather!(out, s, f, shape, g::Int32)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == 0x01
        c = PMC.particle_cell(s, i, Val(2)); δ = PMC.particle_offset(s, i, Val(2))
        E, B = PME.fields_at(f, c, δ, shape, g)
        out[i] = E[1] + B[3]
    end
end
@kernel function k_stencil!(out, s, shape, st, n, sx, qz)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == 0x01
        f, _, (jx, jy, jz) = PMC.flux_stencil(s, i, shape, st, n, sx, qz, PMC.ParticleFrame())
        out[i] = sum(jx) + sum(jy) + sum(jz)
    end
end
@kernel function k_atomic!(J, s)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == 0x01
        w = s.w[i]
        v = ntuple(k -> w * k, Val(16))
        PMC.atomic_add_block!(J, v, 3, 3, Val(4))
    end
end
@kernel function k_shape!(out, s, shape, st, n)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == 0x01
        c = PMC.particle_cell(s, i, Val(2)); δ = PMC.particle_offset(s, i, Val(2))
        c0 = PMC.old_cell_2d(s, i); δ0 = PMC.old_offset_2d(s, i)
        lo1, Wo1, Wn1 = PMC.axis_frame(PMC.ParticleFrame(), shape, st[1], c[1], δ[1], c0[1], δ0[1], n[1], 1)
        out[i] = sum(Wo1) + sum(Wn1)
    end
end
# step! of step.jl written out, with its parts switchable
function step_body!(st, p, k, kinetic::Bool; check::Bool = true)
    check && PME.check_params(st, p)
    (; numerics, grid, fields, dt) = st
    wgs = numerics.workgroupsize
    n = Int32.(grid.n)
    g = Int32(guards(grid))
    dt_dx = map(h -> dt / h, grid.dx)
    window = PME.window_frame(st)
    st.fresh[] = false
    PME.inject_witness!(PME.witness_frame(st), p.witness, grid, k, wgs)
    PME.sharpen_gather!(st)
    PME.ionize!(PME.ionization_frame(st), st, p, k)
    pf = PME.particle_fields(st)
    for s in (PME.deposited_species(st)..., PME.witness_species(st)...)
        PME.launcher(PME.push_kernel!, get_backend(s), wgs)(s, pf, numerics.pusher, numerics.shape, g, s.q / s.m * dt / 2, dt_dx, n;
                                                            ndrange = capacity(s))
    end
    PME.retire_x!(window, st)
    PME.current!(numerics.current, st)
    PME.retire!(PME.open_absorber(st), st)
    PME.solve_fields!(numerics.solver, fields, grid, dt, st.modifiers, p, k; workgroupsize = wgs)
    PME.advance_window!(window, st, p, k)
    kinetic && PME.kinetic!(st)
    return st
end
# Each test mutates the state (and `out`) and returns nothing: a reduction to a scalar inside the
# differentiated function would synchronize the GPU stream there, which Enzyme cannot differentiate.
# The seed sets the adjoint of the outputs (ones), so the gradient is that of the outputs' sum.
seed_u1!(d, dout) = fill!(d.species[1].u[1], 1)
seed_out!(d, dout) = fill!(dout, 1)
seed_EB!(d, dout) = (fill!(d.fields.Ex, 1); fill!(d.fields.Bz, 1))
seed_J!(d, dout) = (fill!(d.fields.Jx, 1); fill!(d.fields.Jy, 1))
tests = [
  ("arrays only", (st, out) -> (s = st.species[1]; k_arrays!(BE)(s.u[1], s.w, s.status; ndrange = capacity(s)); nothing), seed_u1!),
  ("species struct", (st, out) -> (s = st.species[1]; k_scale!(BE)(s; ndrange = capacity(s)); nothing), seed_u1!),
  ("gather", (st, out) -> (s = st.species[1]; k_gather!(BE)(out, s, st.fields, st.numerics.shape, Int32(guards(st.grid)); ndrange = capacity(s)); nothing), seed_out!),
  ("solve_fields", (st, out) -> (PME.solve_fields!(st.numerics.solver, st.fields, st.grid, st.dt, st.modifiers, p, 1); nothing), seed_EB!),
  ("shape", (st, out) -> (s = st.species[1]; k_shape!(BE)(out, s, st.numerics.shape, PME.rho_stagger(Val(2)), Int32.(st.grid.n); ndrange = capacity(s)); nothing), seed_out!),
  ("stencil", (st, out) -> (s = st.species[1]; k_stencil!(BE)(out, s, st.numerics.shape, PME.rho_stagger(Val(2)), Int32.(st.grid.n), (1.0, 1.0), PME.transverse_currents(Val(2)).quantity_z; ndrange = capacity(s)); nothing), seed_out!),
  ("atomic", (st, out) -> (s = st.species[1]; k_atomic!(BE)(st.fields.Jx, s; ndrange = capacity(s)); nothing), (d, dout) -> fill!(d.fields.Jx, 1)),
  ("current", (st, out) -> (PME.current!(st.numerics.current, st); nothing), seed_J!),
  ("push", (st, out) -> (s = st.species[1]; PME.launcher(PME.push_kernel!, get_backend(s), nothing)(s, st.fields, st.numerics.pusher, st.numerics.shape, Int32(guards(st.grid)), 0.05, map(h -> st.dt / h, st.grid.dx), Int32.(st.grid.n); ndrange = capacity(s)); nothing), seed_u1!),
  ("push loop of step", (st, out) -> (pf = PME.particle_fields(st); n = Int32.(st.grid.n); g = Int32(guards(st.grid)); dt_dx = map(h -> st.dt / h, st.grid.dx);
      for s in (PME.deposited_species(st)..., PME.witness_species(st)...)
          PME.launcher(PME.push_kernel!, get_backend(s), nothing)(s, pf, st.numerics.pusher, st.numerics.shape, g, s.q / s.m * st.dt / 2, dt_dx, n; ndrange = capacity(s))
      end; nothing), seed_u1!),
  ("push and current", (st, out) -> (s = st.species[1]; PME.launcher(PME.push_kernel!, get_backend(s), nothing)(s, st.fields, st.numerics.pusher, st.numerics.shape, Int32(guards(st.grid)), 0.05, map(h -> st.dt / h, st.grid.dx), Int32.(st.grid.n); ndrange = capacity(s)); PME.current!(st.numerics.current, st); nothing), seed_J!),
  ("kinetic", (st, out) -> (PME.kinetic!(st); nothing), (d, dout) -> fill!(d.K, 1)),
  ("check_params", (st, out) -> (PME.check_params(st, p); st.fields.Ex .*= 2; nothing), seed_EB!),
  ("step without kinetic", (st, out) -> (step_body!(st, p, 1, false); nothing), seed_EB!),
  ("step without check_params", (st, out) -> (step_body!(st, p, 1, true; check = false); nothing), seed_EB!),
  ("step", (st, out) -> (step!(st, p, 1; observer = nothing); nothing), seed_EB!),
]
leaves!(out, path, x::AbstractArray{<:AbstractFloat}) = (any(y -> y === x, values(out)) || (out[path] = x); out)
leaves!(out, path, x::AbstractArray) = out
leaves!(out, path, x::Union{Tuple,NamedTuple}) = (foreach(k -> leaves!(out, "$path.$k", x[k]), keys(x)); out)
function leaves!(out, path, x)
    isstructtype(typeof(x)) || return out
    for f in fieldnames(typeof(x))
        isdefined(x, f) && leaves!(out, "$path.$f", getfield(x, f))
    end
    return out
end
# Each array once, by identity: workspace temporaries alias other arrays of the state, and counting
# an array under two paths would count its share of ⟨∇f, v⟩ twice (paths are visited in field
# order, so the same path wins in a state, its Dual twin and its shadow).
leaves(x) = leaves!(Dict{String,Any}(), "st", x)
println("bisect: backend ", BE, ", mode ", MODE, ", patches ", get(ENV, "PM_ENZYME_PATCH", ""))
only = get(ENV, "ONLY", "")
isempty(only) || println("only the test named ", repr(only))
for (name, f, seed!) in tests
    isempty(only) || only == name || continue
    s2 = deepcopy(st); d2 = Enzyme.make_zero(s2)
    out = similar(st.species[1].w); fill!(out, 0); dout = zero(out)
    seed!(d2, dout)
    t = @elapsed try
        Enzyme.autodiff(Reverse, f, Const, Duplicated(s2, d2), Duplicated(out, dout))
        KernelAbstractions.synchronize(BE)
        println("[$name] OK, sum |gradient| over the state = ", sum(a -> Float64(sum(abs, a)), values(leaves(d2))))
    catch e
        println("[$name] FAILED: ", replace(first(sprint(showerror, e), 6000), "\n" => " | "))
    end
    println("   ($(round(t, digits=1)) s)")
    flush(stdout)
end
println(sort!(collect(ACTIVE_SEEN)))
