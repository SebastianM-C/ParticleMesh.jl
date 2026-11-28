# The WP-064 case, which lives in Wakefields (lib/Wakefields/src/cases2d.jl): a small 2D LWFA run
# whose pulse ionizes the K shell of a nitrogen dopant, with the energy the born electrons carry in
# the first bucket as the objective. The scripts that include this file get the case's names and
# the packages they use with it.
using ParticleMesh, KernelAbstractions
using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: window_back_band
using ParticleMeshElectromagnetic.ParticleMeshCore: ALIVE
using ForwardDiff: ForwardDiff
using Wakefields: adk_level, BCASE, born_problem, born_numerics, born_start, start_of, run_to_end, born_state,
                  smooth_rise, check_fade, slot_energy, born_energy, born_objective, back_band_end, check_from,
                  plasma_peak_energy, plasma_fast_weight, plasma_fast_energy, born_summary

value_of(x::Real) = x
value_of(d::ForwardDiff.Dual) = ForwardDiff.value(d)
