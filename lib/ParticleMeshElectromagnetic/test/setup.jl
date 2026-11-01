# Shared by runtests.jl and the subprocess worker: the engine's test setup (backend, the A8
# element-type matrix, comparison helpers, `record!`), the engine's test helpers (`fd_scan`),
# this package's tolerances and the case files.

const CORE_TEST = normpath(joinpath(@__DIR__, "..", "..", "ParticleMeshCore", "test"))
const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
const CASES = joinpath(REPO, "scripts", "crosscode", "cases")

include(joinpath(CORE_TEST, "setup.jl"))     # ParticleMeshCore, backend, TYPES, D3, absdiff, record!, ...
include(joinpath(CORE_TEST, "helpers.jl"))   # fd_scan, test_species, ...
using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: kick_kernel!, push_kernel!
using LinearAlgebra: det
using Random: Xoshiro

const EMTOL = TOML.parsefile(joinpath(@__DIR__, "tolerances.toml"))
"Proposed constant `C` of a test of this package, from its `tolerances.toml`."
emC(id, key = "C") = EMTOL[id][key]

"A case file of `scripts/crosscode/cases/`, parsed."
load_case(name) = TOML.parsefile(joinpath(CASES, "$name.toml"))
