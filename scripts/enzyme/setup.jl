# WP-050: the side environment of the Enzyme spike.
#
# No released Enzyme (0.13.219, nor its main branch on 2026-11-09) accepts LLVM.jl 10, which
# KernelAbstractions 0.10 requires, so Enzyme cannot join the repository's environments. This
# builds an environment beside them: copies of lib/ParticleMeshCore and
# lib/ParticleMeshElectromagnetic with KernelAbstractions relaxed to 0.9 (the engine uses only
# @kernel, @index, @Const, allocate, zeros, ones and get_backend, all present in 0.9), and
# Enzyme, ForwardDiff and the stdlibs of the run manifest at pinned versions.
#
#     julia +1.13 --startup=no scripts/enzyme/setup.jl <dir> [--foreach]
#
# `--foreach` rewrites the 2D step's loops over the species tuple (the push, `kinetic!` and the
# `zip(st.species, st.ws)` loops of `initialize!` and `current!`) as `foreach`, which two
# species need: Enzyme fails on the `for` loops (#384; `reports/wp-050-enzyme-spike.md`). Then run
#
#     julia +1.13 --project=<dir>/env scripts/enzyme/chain_rule.jl ...

using Pkg

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const PINS = (Enzyme = v"0.13.219", ForwardDiff = v"1.4.7", KernelAbstractions = v"0.9.45")

function main(args)
    isempty(args) && error("usage: setup.jl <dir> [--foreach]")
    dir = abspath(args[1])
    foreach_push = "--foreach" in args
    ispath(dir) && error("$dir exists; give a new directory")
    lib = joinpath(dir, "lib")
    for p in ("ParticleMeshCore", "ParticleMeshElectromagnetic")
        src = joinpath(REPO, "lib", p)
        dst = joinpath(lib, p)
        mkpath(dst)
        for f in ("src", "ext", "Project.toml")
            ispath(joinpath(src, f)) && cp(joinpath(src, f), joinpath(dst, f))
        end
        toml = joinpath(dst, "Project.toml")
        s = read(toml, String)
        s = replace(s, r"KernelAbstractions = \"0[.]10[^\"]*\"" => "KernelAbstractions = \"0.9.45\"",
                    r"\[workspace\]\n[^\[]*" => "",
                    "path = \"../ParticleMeshCore\"" => "path = \"$(joinpath(lib, "ParticleMeshCore"))\"")
        chmod(toml, 0o644)
        write(toml, s)
    end
    if foreach_push
        f = joinpath(lib, "ParticleMeshElectromagnetic", "src", "step2d.jl")
        s = read(f, String)
        old = "    for s in st.species\n        launcher(push_2d_kernel!"
        occursin(old, s) || error("the push loop of step2d.jl has changed; update setup.jl")
        s = replace(s, old => "    foreach(st.species) do s\n        launcher(push_2d_kernel!")
        kin_old = join(["    first = 1",
                        "    for s in st.species",
                        "        nc = cld(capacity(s), KINETIC_CHUNK)",
                        "        launcher(kinetic_chunks_kernel!, get_backend(s), wgs)(st.K, s, Int32(first), Int32(capacity(s)), Int32(KINETIC_CHUNK);",
                        "                                                             ndrange = nc)",
                        "        first += nc",
                        "    end"], "\n")
        kin_new = join(["    firsts = cumsum((1, map(s -> cld(capacity(s), KINETIC_CHUNK), Base.front(st.species))...))",
                        "    foreach(st.species, firsts) do s, first",
                        "        nc = cld(capacity(s), KINETIC_CHUNK)",
                        "        launcher(kinetic_chunks_kernel!, get_backend(s), wgs)(st.K, s, Int32(first), Int32(capacity(s)), Int32(KINETIC_CHUNK);",
                        "                                                             ndrange = nc)",
                        "    end"], "\n")
        occursin(kin_old, s) || error("kinetic! of step2d.jl has changed; update setup.jl")
        s = replace(s, kin_old => kin_new)
        zip_old = "    for (s, ws) in zip(st.species, st.ws)\n"
        s = replace(s, zip_old => "    foreach(st.species, st.ws) do s, ws\n")
        chmod(f, 0o644)
        write(f, s)
    end
    env = joinpath(dir, "env")
    mkpath(env)
    Pkg.activate(env)
    Pkg.develop([PackageSpec(path = joinpath(lib, "ParticleMeshCore")), PackageSpec(path = joinpath(lib, "ParticleMeshElectromagnetic"))])
    Pkg.add([PackageSpec(name = string(k), version = v) for (k, v) in pairs(PINS)])
    Pkg.add(["SHA", "TOML", "Dates", "Printf"])
    Pkg.instantiate()
    Pkg.status()
    return env
end

main(ARGS)
