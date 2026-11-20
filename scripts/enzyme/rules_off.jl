# WP-077: switch off the engine's Enzyme reverse rules in this process, so that Enzyme
# differentiates the transfer and sort kernels itself. For comparisons of the reverse pass with
# and without the hand-written pullbacks; a patch for the scripts that take one
# (`PM_ENZYME_PATCH` of chain_rule.jl):
#
#     PM_ENZYME_PATCH=scripts/enzyme/rules_off.jl julia +1.13 --project=<env> scripts/enzyme/chain_rule.jl fast 4
#
# It deletes every method that ParticleMeshCoreEnzymeCoreExt adds to EnzymeRules
# (`augmented_primal`, `reverse`, `inactive_kwarg`) and prints how many it deleted.
using ParticleMeshCore, Enzyme
let ext = Base.get_extension(ParticleMeshCore, :ParticleMeshCoreEnzymeCoreExt)
    ext === nothing && error("rules_off.jl: the extension ParticleMeshCoreEnzymeCoreExt is not loaded")
    R = Enzyme.EnzymeRules
    gone = 0
    for f in (R.augmented_primal, R.reverse, R.inactive_kwarg), m in collect(methods(f))
        if m.module === ext
            Base.delete_method(m)
            gone += 1
        end
    end
    println("rules_off.jl: deleted ", gone, " methods of ParticleMeshCoreEnzymeCoreExt; Enzyme differentiates the transfer and sort kernels itself")
end
