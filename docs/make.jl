# Documenter reads `git rev-parse HEAD` for source links, so this needs a commit; pass `remotes = nothing` to build without one.
using Documenter
using ParticleMesh

makedocs(;
    sitename = "ParticleMesh.jl",
    modules = [ParticleMesh],
    pages = ["Home" => "index.md"],
    checkdocs = :none,
)
