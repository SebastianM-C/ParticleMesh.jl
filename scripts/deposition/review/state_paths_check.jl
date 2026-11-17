# The module's step with the sort from the last order (`EMNumerics(; resort = true)`) against the
# step with the full sort, on a backend, along the ways a state reaches a step:
#
#   julia +1.13 -t 4 --project=<repo>/scripts state_paths_check.jl <repo> [cpu|cuda]
#
# On the suite's ionization cases (lib/ParticleMeshElectromagnetic/test/ionization_cases.jl), per
# element type, in Reference mode:
#   run      `ion_window`, 150 steps: the evolved arrays' bits with resort = true and false, every 30 steps
#   restore  the snapshot of step 100 restored into the state at 150, continued to 150
#   again    `ion_in_E` with a plasma (births from the first step), 20 steps, `initialize!` again, 20 steps:
#            the second run against the first, in the fields and the alive slots
#   view     `ion_window`: a state at step 100, a `state_view` over a copy of its snapshot of step 50
#            takes steps 51 to 60, then the state goes on to 150
# Each of the last three is compared with the uninterrupted run with resort = true. Prints one line
# per case and ends with the list of the cases that differ (exit code 1 then).
const REPO = ARGS[1]
include(joinpath(REPO, "scripts", "engine2d", "common.jl"))
using ParticleMeshElectromagnetic
using ForwardDiff: Dual
const PME = ParticleMeshElectromagnetic
const PMC = ParticleMesh.ParticleMeshCore
const NAME = length(ARGS) >= 2 ? ARGS[2] : "cpu"
if NAME == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
    @eval const BE = CUDABackend()
    CUDACore.allowscalar(false)
else
    const BE = CPU()
end
zero3(::Type{T}) where {T} = (zero(T), zero(T), zero(T))
bits(a::AbstractArray) = reinterpret(UInt8, vec(Array(a)))
include(joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "test", "ionization_cases.jl"))
const DIFFERENT = String[]
const HAS = :resort in fieldnames(EMNumerics)

numerics(resort) = HAS ? EMNumerics(; mode = Reference(), resort) : EMNumerics(; mode = Reference())
new_state(prob, resort) = (st = EMState(prob, numerics(resort); backend = BE); initialize!(st, prob); st)
evolved_bits(st) = map(a -> copy(bits(a)), PME.evolved(st))
steps!(st, p, ks) = (foreach(k -> step!(st, p, k), ks); st)

"What a comparison reads of a species: status, count, and each array in the alive slots."
function species_record(s)
    alive = Array(s.status) .== PMC.ALIVE
    pick(a) = copy(reinterpret(UInt8, Array(a)[alive]))
    return (Array(s.status), Array(s.count), map(pick, s.cell), map(pick, s.offset), map(pick, s.u), pick(s.w))
end
function alive_record(st)
    fr = PME.ionization_frame(st)
    return (map(species_record, (st.species..., fr.atoms, fr.electrons)), map(a -> copy(bits(a)), values(st.fields)))
end

function report(label, case, ok, note = "")
    println(rpad(label, 18), rpad(case, 9), ok ? "same bits" : "DIFFERENT", isempty(note) ? "" : "  ($note)")
    flush(stdout)
    ok || push!(DIFFERENT, "$label $case")
    return nothing
end

function check(::Type{T}, label) where {T}
    prob = ion_window(T; nsteps = 150)
    p = params(prob)
    # run: resort against the full sort, every 30 steps
    a, b = new_state(prob, true), new_state(prob, false)
    ok = true
    local snap50, snap100
    for k in 1:150
        step!(a, p, k)
        step!(b, p, k)
        k == 50 && (snap50 = PME.snapshot(a, k))
        k == 100 && (snap100 = PME.snapshot(a, k))
        k % 30 == 0 && (ok &= evolved_bits(a) == evolved_bits(b))
    end
    want = evolved_bits(a)
    fr = PME.ionization_frame(a)
    HAS && report(label, "run", ok, "plasma count $(Int(Array(a.species[1].count)[1])), electrons $(Int(Array(fr.electrons.count)[1]))")
    # restore: into the state that ran to the end
    PME.restore!(a, snap100)
    steps!(a, p, 101:150)
    report(label, "restore", evolved_bits(a) == want)
    # view: the state at 100 goes on after a view of it took ten steps
    st = steps!(new_state(prob, true), p, 1:100)
    view = PME.state_view(st, map(copy, snap50.arrays))
    steps!(view, p, 51:60)
    steps!(st, p, 101:150)
    report(label, "view", evolved_bits(st) == want)
    # again: births from the first step, the state initialized a second time
    prob2 = ion_in_E(T; plasma = true, nsteps = 20)
    p2 = params(prob2)
    s2 = steps!(new_state(prob2, true), p2, 1:20)
    first_run = alive_record(s2)
    initialize!(s2, prob2)
    steps!(s2, p2, 1:20)
    full = alive_record(steps!(new_state(prob2, false), p2, 1:20))
    report(label, "again", alive_record(s2) == first_run && first_run == full,
           "electrons alive: $(count(==(PMC.ALIVE), first_run[1][end][1])) in the first run")
    return nothing
end

function main()
    NAME == "cuda" && println("device: ", CUDACore.name(CUDACore.device()))
    for (T, label) in ((Float32, "Float32"), (Float64, "Float64"), (Dual{Nothing,Float32,3}, "Dual{Float32,3}"))
        check(T, label)
    end
    if isempty(DIFFERENT)
        println("backend ", NAME, ": every path gives the full sort's bits")
    else
        println("backend ", NAME, ": DIFFERENT in ", join(DIFFERENT, "; "))
        exit(1)
    end
end
Base.invokelatest(main)
