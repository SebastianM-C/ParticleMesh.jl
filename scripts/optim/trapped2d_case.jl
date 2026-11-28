# The WP-065 case, which lives in Wakefields (lib/Wakefields/src/cases2d.jl): WP-064's case at a
# fixed laser energy, with every trapped electron of the first bucket as the objective. The
# scripts that include this file get the case's names.
isdefined(@__MODULE__, :BCASE) || include(joinpath(@__DIR__, "born2d_case.jl"))
using Wakefields: TCASE, pulse_energy, TRAPPED_BOX, trapped_point, budget, a0_for_energy, trapped_capacity,
                  trapped_problem, trapped_state, slot_trapped, species_trapped, check_rise, trapped_parts,
                  trapped_energy, trapped_objective, trapped_summary
