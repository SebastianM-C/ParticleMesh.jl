# Spike-only: run another script with spike patches applied first.
#   PM_PATCH="scripts/enzyme/spike/unroll/gather.jl;..." \
#     julia +1.13 --project=scripts scripts/enzyme/spike/with_patches.jl scripts/optim/step_timing.jl cuda fast
# PM_PATCH is a ';'-separated list of patch files, empty for none; the remaining arguments go to the script.
using ParticleMesh, ParticleMeshElectromagnetic
isdefined(Main, :ParticleMeshCore) || @eval const ParticleMeshCore = ParticleMesh.ParticleMeshCore
for f in filter(!isempty, split(get(ENV, "PM_PATCH", ""), ';'))
    include(abspath(f))
    println("with_patches: applied ", f)
end
include(abspath(popfirst!(ARGS)))
