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
p = params(prob)

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
energy(st) = sum(a -> sum(abs2, a), (st.fields.Ex, st.fields.Ey, st.fields.Ez, st.fields.Bx, st.fields.By, st.fields.Bz))
tests = [
  "arrays only" => st -> (s = st.species[1]; k_arrays!(BE)(s.u[1], s.w, s.status; ndrange = capacity(s)); sum(s.u[1])),
  "species struct" => st -> (s = st.species[1]; k_scale!(BE)(s; ndrange = capacity(s)); sum(s.u[1])),
  "gather" => st -> (s = st.species[1]; o = st.ρ; out = similar(s.w); fill!(out, 0); k_gather!(BE)(out, s, st.fields, st.numerics.shape, Int32(guards(st.grid)); ndrange = capacity(s)); sum(out)),
  "solve_fields" => st -> (PME.solve_fields!(st.numerics.solver, st.fields, st.grid, st.dt, st.modifiers, p, 1); energy(st)),
  "shape" => st -> (s = st.species[1]; out = similar(s.w); fill!(out, 0); k_shape!(BE)(out, s, st.numerics.shape, PME.rho_stagger(Val(2)), Int32.(st.grid.n); ndrange = capacity(s)); sum(out)),
  "stencil" => st -> (s = st.species[1]; out = similar(s.w); fill!(out, 0); k_stencil!(BE)(out, s, st.numerics.shape, PME.rho_stagger(Val(2)), Int32.(st.grid.n), (1.0, 1.0), PME.transverse_currents(Val(2)).quantity_z; ndrange = capacity(s)); sum(out)),
  "atomic" => st -> (s = st.species[1]; k_atomic!(BE)(st.fields.Jx, s; ndrange = capacity(s)); sum(abs2, st.fields.Jx)),
  "current" => st -> (PME.current!(st.numerics.current, st); sum(abs2, st.fields.Jx) + sum(abs2, st.fields.Jy)),
]
leaves!(out, path, x::AbstractArray{<:AbstractFloat}) = (out[path] = x; out)
leaves!(out, path, x::AbstractArray) = out
leaves!(out, path, x::Union{Tuple,NamedTuple}) = (foreach(k -> leaves!(out, "$path.$k", x[k]), keys(x)); out)
function leaves!(out, path, x)
    isstructtype(typeof(x)) || return out
    for f in fieldnames(typeof(x))
        isdefined(x, f) && leaves!(out, "$path.$f", getfield(x, f))
    end
    return out
end
leaves(x) = leaves!(Dict{String,Any}(), "st", x)
println("bisect: backend ", BE, ", mode ", MODE, ", patches ", get(ENV, "PM_ENZYME_PATCH", ""))
only = get(ENV, "ONLY", "")
for (name, f) in tests
    isempty(only) || occursin(only, name) || continue
    s2 = deepcopy(st); d2 = Enzyme.make_zero(s2)
    t = @elapsed try
        Enzyme.autodiff(Reverse, f, Active, Duplicated(s2, d2))
        KernelAbstractions.synchronize(BE)
        println("[$name] OK, sum |gradient| over the state = ", sum(a -> Float64(sum(abs, a)), values(leaves(d2))))
    catch e
        println("[$name] FAILED: ", replace(first(sprint(showerror, e), 6000), "\n" => " | "))
    end
    println("   ($(round(t, digits=1)) s)")
end
println(sort!(collect(ACTIVE_SEEN)))
