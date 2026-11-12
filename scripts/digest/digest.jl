# Before/after digest of the electromagnetic module's runs: the proof that a change of the state or
# step layer changes no run result (WP-055, WP-056; scripts/README.md).
#
#   julia +1.13 --project=<tree>/lib/ParticleMeshElectromagnetic/test scripts/digest/digest.jl <out.txt> [modes] [cases]
#
# The project picks the code: run this one file against the test environment of the old tree and
# of the new one, on one machine, and compare the two outputs. `modes` is "Reference", "Fast" or
# "Reference,Fast" (the default), `cases` a comma-separated list of case names (all by default);
# DIGEST_BACKEND=cuda runs on the GPU. One line per (case, element type, mode): the SHA-256 of the
# full state after the run (every particle array with its extras, the fields, ρ, K, the absorbers'
# and the window's arrays, the recorder's tables), of the diagnostic histories, and of the
# workspaces, each with the number of arrays and bytes hashed. Then, per harness case, two solves
# of one problem: the particle and field arrays of the last saved state (the leading arrays of the
# partition, which every layout of it shares; what follows them, `K` before WP-056 and the rest of
# the evolved arrays after it, is in the state group of the run lines) and the histories.
#
# The cases use the public constructors only (EMProblem, SpeciesDescription, GaussianLaser,
# EMNumerics, ...) and run with `initialize!` and `run!(state, params(problem), nsteps)`. The only
# layout-dependent part is `groups`, which reads the arrays of a state under fixed names.
#
# What not to compare: the workspace group of a moving-window case (its spectral scratch is never
# written, #422), and on CUDA anything in Fast mode (atomic deposition is not reproducible between
# two runs of one code there).

module EMDigest

using ParticleMeshCore, ParticleMeshElectromagnetic, KernelAbstractions, ForwardDiff, TOML
using ForwardDiff: Dual
import SHA
using SciMLBase: SciMLBase, DiscreteProblem, solve
using RecursiveArrayTools: ArrayPartition
using OrdinaryDiffEqFunctionMap: FunctionMap

const PME = ParticleMeshElectromagnetic
# The backend: the CPU, or with DIGEST_BACKEND=cuda the CUDA device (digests are per backend, A4).
if get(ENV, "DIGEST_BACKEND", "cpu") == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
    @eval const BE = CUDABackend()
else
    const BE = CPU()
end
const ROOT = normpath(joinpath(pkgdir(PME), "..", ".."))
const CASES = joinpath(ROOT, "scripts", "crosscode", "cases")
const D3 = Dual{Nothing,Float32,3}
const TYPES = (Float32, Float64, D3)
tlabel(T) = T === D3 ? "Dual{Float32,3}" : string(T)

"A value of `T`; for the dual type, with a unit partial in slot `i` (0: none)."
sd(::Type{T}, x, i = 0) where {T<:AbstractFloat} = T(x)
sd(::Type{D3}, x, i = 0) = Dual{Nothing}(Float32(x), ntuple(j -> j == i ? 1.0f0 : 0.0f0, 3)...)
val(x::Real) = x
val(x::Dual) = ForwardDiff.value(x)
z3(::Type{T}) where {T} = (zero(T), zero(T), zero(T))
courant(cells, L, c) = c * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:2)))

# --- the cases ----------------------------------------------------------------------------------
# Each returns (problem, numerics keywords, initialize! keywords); the mode is added by the driver.

function case_1d(name, ::Type{T}; passes = 0, init = (;), nkw = (;)) where {T}
    case = TOML.parsefile(joinpath(CASES, "$name.toml"))
    case["numerics"]["filter_passes"] = passes
    prob = problem_from_case(case, T; ppc = 64, t_end = 1, amplitude = sd(T, 1 // 100, 1))
    return prob, (; case, nkw...), init
end

function mixed_1d(::Type{T}) where {T}
    S = T === Float32 ? Float64 : T
    L = 2 * T(π)
    e = SpeciesDescription{S}(:electron, -one(S), one(S), sd(S, 0.9, 1), sd(S, 0.05, 2), 1, 16, (S(0.1), zero(S), zero(S)), (S(0.05), zero(S), zero(S)), 1)
    prob = EMProblem("mixed-1d", L, 32, T(15) / 100, 20, (e,), 1, 2 * T(π) / L)
    return prob, (;), (;)
end

function thermal_2d(::Type{T}) where {T}
    L = (2 * T(π), 2 * T(π))
    cells = (10, 8)
    dt = courant(cells, L, T(9) / 10)
    vth = one(T) / 20
    sp = (SpeciesDescription{T}(:electron, -one(T), one(T), sd(T, 1, 1), sd(T, 1 // 20, 2), (2, 1), (3, 3), z3(T), (vth, vth, vth), 1),)
    prob = EMProblem("thermal-2d", L, cells, T(dt), 8, sp, 1, (2, 1); E0 = (one(T) / 100, zero(T), zero(T)), B0 = (zero(T), zero(T), one(T) / 2))
    return prob, (; current = NoCurrent()), (;)
end

function two_species_2d(::Type{T}; momentum = :default, passes = 0, init = (;), nsteps = 20, nkw = (;)) where {T}
    L = (2 * T(π), 3 * T(π) / 2)
    cells = (10, 8)
    dt = courant(cells, L, T(9) / 10)
    tenth = one(T) / 10
    e = SpeciesDescription{T}(:electron, -one(T), one(T), sd(T, 1, 2), sd(T, 1 // 10, 1), (2, 1), (3, 3), (3tenth, -2tenth, tenth),
                              (tenth, tenth, tenth), 1; momentum)
    p = SpeciesDescription{T}(:positive, one(T), 4 * one(T), one(T) / 2, tenth / 2, (1, -1), (3, 3), (-2tenth, 5tenth / 2, -3tenth / 2),
                              (tenth / 2, tenth / 2, tenth / 2), 2; momentum)
    prob = EMProblem("two-species-2d", L, cells, T(dt), nsteps, (e, p), 7, (2, 1))
    return prob, (; current = Esirkepov(), filter = BinomialFilter(passes), nkw...), init
end

function open_2d(::Type{T}, absorber) where {T}
    cells = (10, 16)
    L = (2 * T(π), 3 * T(π))
    dt = courant(cells, L, T(9) / 10)
    tenth = one(T) / 10
    e = SpeciesDescription{T}(:electron, -one(T), one(T), sd(T, 1, 2), sd(T, 1 // 10, 1), (2, 1), (3, 3), (3tenth, -5tenth / 2, tenth),
                              (tenth, tenth, tenth), 1)
    p = SpeciesDescription{T}(:positive, one(T), 4 * one(T), one(T) / 2, tenth / 2, (1, -1), (3, 3), (-2tenth, 3tenth, -3tenth / 2),
                              (tenth / 2, tenth / 2, tenth / 2), 2)
    prob = EMProblem("open-2d", L, cells, T(dt), 20, (e, p), 7, (2, 1); boundaries = (:periodic, :open))
    return prob, (; current = Esirkepov(), absorber), (;)
end

function antenna_2d(::Type{T}, envelope, absorber; polarization = 0) where {T}
    cells = (48, 24)
    h = 2 * T(π) / 8
    L = (cells[1] * h, cells[2] * h)
    dt = T(95) / 100 * h / sqrt(T(2))
    laser = GaussianLaser(; a0 = sd(T, 1, 1), waist = T(4), duration = sd(T, 2, 2), delay = T(12), focus = sd(T, 12, 3), position = 6.0,
                          center = Float64(val(L[2])) / 2, chirp = T(1) / 50, polarization = T(polarization), phase = T(3) / 10, envelope)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(2) / 100, zero(T), (1, 0), (2, 2), z3(T), z3(T), 1)
    prob = EMProblem("antenna-2d", L, cells, dt, 60, (e,), 1, (1, 0); lasers = (laser,), boundaries = (:periodic, :open))
    return prob, (; current = Esirkepov(), absorber), (;)
end

"A species described in Float64 inside a Float32 problem (for the other element types: the plain run): static and thermal, Esirkepov."
function mixed_static(::Type{T}) where {T}
    S = T === Float32 ? Float64 : T
    L = (2 * T(π), 3 * T(π) / 2)
    cells = (10, 8)
    dt = courant(cells, L, T(9) / 10)
    e = SpeciesDescription{S}(:electron, -one(S), one(S), sd(S, 0.7123, 1), sd(S, 0.0517, 2), (2, 1), (3, 3), (S(0.3), S(-0.2), S(0.1)), (S(0.05), S(0.05), S(0.05)), 1)
    i = SpeciesDescription{S}(:ion, one(S), S(1836.15267343), S(0.3561), zero(S), (1, -1), (3, 3), z3(S), (S(0.02), S(0.01), S(0.03)), 2)   # a mass that is not exact in Float32
    prob = EMProblem("mixed-static", L, cells, T(dt), 12, (e, i), 3, (2, 1))
    return prob, (; current = Esirkepov()), (;)
end

"The same in a moving window from t = 0 with a thermal refill."
function mixed_window(::Type{T}) where {T}
    S = T === Float32 ? Float64 : T
    cells = (24, 8)
    L = (T(cells[1]) / 2, T(cells[2]) / 2)
    dt = courant(cells, L, T(9) / 10)
    e = SpeciesDescription{S}(:electron, -one(S), one(S), sd(S, 0.7123, 1), zero(S), (1, 0), (2, 2), (S(0.1), zero(S), S(-0.05)), (S(0.05), S(0.03), S(0.04)), 1;
                              momentum = :gaussian_velocity)
    prob = EMProblem("mixed-window", L, cells, T(dt), 40, (e,), 5, (1, 0); boundaries = (:window, :periodic),
                     window = MovingWindow(; start = zero(T)), profile = DensityRamp(; vacuum = 4.0, ramp = 4.0))
    return prob, (; headroom = 1 / 2), (;)
end

"Two lasers on two planes (different envelopes, polarizations and amplitudes), periodic box, cold plasma."
function two_lasers_2d(::Type{T}) where {T}
    cells = (56, 16)
    h = 2 * T(π) / 8
    L = (cells[1] * h, cells[2] * h)
    dt = T(95) / 100 * h / sqrt(T(2))
    l1 = GaussianLaser(; a0 = sd(T, 1, 1), waist = T(4), duration = sd(T, 2, 2), delay = T(12), focus = T(12), position = 6.0,
                       center = Float64(val(L[2])) / 2, envelope = AxisTime())
    l2 = GaussianLaser(; a0 = sd(T, 1 // 2, 3), waist = T(3), duration = T(3) / 2, delay = T(10), focus = T(20), position = 14.0,
                       center = Float64(val(L[2])) / 3, polarization = T(1), chirp = T(1) / 20, envelope = WavefrontTime())
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(2) / 100, zero(T), (1, 0), (2, 2), z3(T), z3(T), 1)
    prob = EMProblem("two-lasers-2d", L, cells, dt, 60, (e,), 1, (1, 0); lasers = (l1, l2))
    return prob, (; current = Esirkepov()), (;)
end

function window_lwfa(::Type{T}; nsteps = 120, cells = (64, 16), position = 6, witness = nothing, boundaries = (:window, :periodic)) where {T}
    h = 2 * T(π) / 8
    dt = T(95) / 100 * h / sqrt(T(2))
    pulse = GaussianLaser(; a0 = sd(T, 1, 1), waist = T(6), duration = T(3), delay = T(18), focus = T(position), position = T(position),
                          center = cells[2] * h / 2, envelope = WavefrontTime())
    e = SpeciesDescription{T}(:electron, -one(T), one(T), sd(T, 2 // 100, 3), zero(T), (1, 0), (2, 2), z3(T), z3(T), 1)
    return EMProblem("window-lwfa", (cells[1] * h, cells[2] * h), cells, dt, nsteps, (e,), 1, (1, 0);
                     lasers = (pulse,), boundaries, window = MovingWindow(; start = T(33)),
                     profile = DensityRamp(; vacuum = T(24), ramp = sd(T, 8, 2)), witness)
end

function window_thermal(::Type{T}) where {T}
    cells = (24, 8)
    L = (T(cells[1]) / 2, T(cells[2]) / 2)
    dt = courant(cells, L, T(9) / 10)
    v = one(T) / 20
    e = SpeciesDescription{T}(:electron, -one(T), one(T), sd(T, 1, 1), zero(T), (1, 0), (2, 2), (2v, zero(T), -v), (v, v, v), 1;
                              momentum = :gaussian_velocity)
    i = SpeciesDescription{T}(:ion, one(T), 100 * one(T), sd(T, 1, 1), zero(T), (1, 0), (1, 2), z3(T), (v / 10, v / 10, v / 10), 2;
                              momentum = :gaussian_velocity)
    prob = EMProblem("window-thermal", L, cells, T(dt), 40, (e, i), 5, (1, 0); boundaries = (:window, :periodic),
                     window = MovingWindow(; start = zero(T)), profile = DensityRamp(; vacuum = T(4), ramp = sd(T, 4, 2)))
    return prob, (; filter = BinomialFilter(1), back_absorber = PML(; cells = 4), headroom = 1 / 2), (;)
end

function window_witness(::Type{T}) where {T}
    cells = (72, 16)
    h = 2 * Float64(π) / 8
    yc = cells[2] * h / 2
    wb = WitnessBunch(; inject = 40.0, region = ((20.0, 30.0), (yc - 1.0, yc + 1.0)), ppc = (2, 1), momentum = sd(T, 3, 2),
                      center = (sd(T, 25, 3), T(yc)), length = T(2))
    prob = window_lwfa(T; nsteps = 140, cells, position = 10, witness = wb, boundaries = (:window, :open))
    return prob, (; back_absorber = PML(; cells = 6), absorber = PML(; cells = 4, alpha_max = 0.2)), (;)
end

const CASE_LIST = [
    ("1d-landau", T -> case_1d("landau", T)),
    ("1d-two-stream", T -> case_1d("two-stream", T)),
    ("1d-weibel", T -> case_1d("weibel", T)),
    ("1d-landau-filter3", T -> case_1d("landau", T; passes = 3)),
    ("1d-two-stream-t_minus_half-cell_centre", T -> case_1d("two-stream", T; init = (; momenta_at = :t_minus_half, density_at = :cell_centre))),
    ("2d-thermal-nocurrent-E0B0", thermal_2d),
    ("2d-two-species-esirkepov", T -> two_species_2d(T)),
    ("2d-two-species-filter2-gaussian_velocity", T -> two_species_2d(T; momentum = :gaussian_velocity, passes = 2)),
    ("2d-two-species-t_minus_half-cell_centre", T -> two_species_2d(T; momentum = :gaussian_momentum, init = (; momenta_at = :t_minus_half, density_at = :cell_centre))),
    ("2d-open-layer", T -> open_2d(T, AbsorbingLayer(; cells = 4, strength = 1, order = 2))),
    ("2d-open-pml", T -> open_2d(T, PML(; cells = 4, sigma_max = 1.5, order = 3, kappa_max = 2, kappa_order = 2, alpha_max = 0.2))),
    ("2d-antenna-axis-layer", T -> antenna_2d(T, AxisTime(), AbsorbingLayer(; cells = 4, strength = 1))),
    ("2d-antenna-wavefront-pml", T -> antenna_2d(T, WavefrontTime(), PML(; cells = 4); polarization = 1)),
    ("2d-window-lwfa-refill", T -> (window_lwfa(T), (;), (;))),
    ("2d-window-thermal-backpml-filter1", window_thermal),
    ("2d-window-open-pml-backpml-witness", window_witness),
    ("1d-weibel-higuera-cary-wgs32", T -> case_1d("weibel", T; nkw = (; pusher = HigueraCary(), workgroupsize = 32))),
    ("2d-two-species-linear-vay-wgs32", T -> two_species_2d(T; nkw = (; shape = Linear, pusher = Vay(), workgroupsize = 32))),
    ("2d-two-lasers", two_lasers_2d),
    ("2d-window-lwfa-linear-filter2", T -> (window_lwfa(T), (; shape = Linear, filter = BinomialFilter(2)), (;))),
    ("2d-float64-species-in-T-static", mixed_static),
    ("2d-float64-species-in-T-window", mixed_window),
    ("1d-float64-species-in-T", T -> mixed_1d(T)),
]

# --- arrays of a state under canonical names, from either layout ----------------------------------

walk!(out, x::AbstractArray) = (isbitstype(eltype(x)) && push!(out, x); out)
walk!(out, x::Union{Tuple,NamedTuple}) = (foreach(y -> walk!(out, y), x); out)
function walk!(out, x)
    parentmodule(typeof(x)) in (ParticleMeshCore, PME) || return out
    for f in fieldnames(typeof(x))
        walk!(out, getfield(x, f))
    end
    return out
end
walk(x) = walk!(AbstractArray[], x)

species_arrays(s) = AbstractArray[s.cell..., s.offset..., s.u..., s.w, s.status, s.count, values(s.extras)...]
hist_of(st) = st.recorder.hist
diag_arrays(st) = haskey(st.recorder.aux, :tables) ? AbstractArray[st.recorder.aux.tables..., st.recorder.aux.partials] : AbstractArray[]
auxget(st, k) = haskey(st.aux, k) ? getfield(st.aux, k) : nothing

function groups(st)
    state = AbstractArray[]
    foreach(s -> append!(state, species_arrays(s)), st.species)
    w = auxget(st, :witness)
    w === nothing || append!(state, species_arrays(w.species))
    append!(state, values(st.fields))
    push!(state, st.ρ, st.K)
    append!(state, walk(auxget(st, :layer)))
    win = auxget(st, :window)
    if win !== nothing
        append!(state, win.lost)
        append!(state, win.fast)
        append!(state, win.flags)
        append!(state, walk(win.back))
    end
    append!(state, diag_arrays(st))
    φ = auxget(st, :φ)
    φ === nothing || push!(state, φ)
    ws = walk(st.ws)
    append!(ws, walk(auxget(st, :spectral)))
    return (; state, hist = AbstractArray[values(hist_of(st))...], ws)
end

function sha(arrays)
    ctx = SHA.SHA256_CTX()
    nbytes = 0
    for a in arrays
        h = Array(a)
        SHA.update!(ctx, Vector{UInt8}(codeunits(string(eltype(h), size(h)))))
        b = collect(reinterpret(UInt8, vec(h)))
        nbytes += length(b)
        SHA.update!(ctx, b)
    end
    return string(bytes2hex(SHA.digest!(ctx))[1:24], " ", length(arrays), "/", nbytes)
end

digest_line(st) = (g = groups(st); join((sha(g.state), sha(g.hist), sha(g.ws)), " | "))

# --- drivers ------------------------------------------------------------------------------------

function numerics_for(kw, mode)
    haskey(kw, :case) && return numerics_from_case(kw.case; mode, Base.structdiff(kw, (; case = nothing))...)
    return EMNumerics(; mode, kw...)
end

function run_case(build, ::Type{T}, mode) where {T}
    prob, nkw, ikw = build(T)
    st = EMState(prob, numerics_for(nkw, mode); backend = BE)
    initialize!(st, prob; ikw...)
    run!(st, params(prob), prob.nsteps)
    return st
end

"The particle and field arrays of a saved partition of the harness: its leading arrays, per species the cells, offsets, momenta, weights, status and count, then the fields."
function common_arrays(st, u)
    n = sum(s -> length(s.cell) + length(s.offset) + 6, st.species; init = 0) + length(st.fields)
    return AbstractArray[u.x[1:n]...]
end

"The harness: two solves of the same problem; the digest of the particle and field arrays of the last saved state and of the cache's histories after each."
function run_harness(build, ::Type{T}, mode) where {T}
    prob, nkw, _ = build(T)
    dprob = DiscreteProblem(prob, numerics_for(nkw, mode); backend = BE)
    out = String[]
    for _ in 1:2
        sol = solve(dprob, FunctionMap(); save_everystep = false)
        st = harness_state(sol)
        push!(out, sha(common_arrays(st, sol.u[end])) * " | " * sha(AbstractArray[values(hist_of(st))...]))
    end
    return out
end

const HARNESS_LIST = [
    ("harness-1d-landau", T -> case_1d("landau", T)),
    ("harness-2d-open-pml", T -> open_2d(T, PML(; cells = 4, sigma_max = 1.5, order = 3, kappa_max = 2, kappa_order = 2, alpha_max = 0.2))),
]

const ALL_MODES = (("Reference", Reference()), ("Fast", Fast()))

"""
    digest(emit; modes = ALL_MODES, cases = nothing, types = TYPES)

Run the cases (all, or those named in `cases`) per element type and mode and hand each line to
`emit`: `name | type | mode | state | histories | workspaces` for a run, and
`name | type | mode | solve i | particles and fields | histories` for the two solves of a harness case.
"""
function digest(emit; modes = ALL_MODES, cases = nothing, types = TYPES)
    wanted(name) = cases === nothing || name in cases
    for (name, build) in CASE_LIST, T in types, (mlabel, mode) in modes
        wanted(name) || continue
        emit(join((name, tlabel(T), mlabel, digest_line(run_case(build, T, mode))), " | "))
    end
    for (name, build) in HARNESS_LIST, T in types, (mlabel, mode) in modes
        wanted(name) || continue
        for (i, l) in enumerate(run_harness(build, T, mode))
            emit(join((name, tlabel(T), mlabel, "solve $i", l), " | "))
        end
    end
    return nothing
end

function main(args)
    modes = length(args) >= 2 ? filter(m -> m[1] in split(args[2], ","), collect(ALL_MODES)) : ALL_MODES
    cases = length(args) >= 3 ? String.(split(args[3], ",")) : nothing
    known = (first.(CASE_LIST)..., first.(HARNESS_LIST)...)
    cases === nothing || all(in(known), cases) || error("unknown case among $cases; the cases are $(join(known, ", "))")
    println("threads = ", Threads.nthreads(), ", code = ", ROOT)
    open(args[1], "w") do io
        digest(s -> (println(io, s); flush(io); println(s); flush(stdout)); modes, cases)
    end
end

end # module

abspath(PROGRAM_FILE) == @__FILE__() && EMDigest.main(ARGS)
