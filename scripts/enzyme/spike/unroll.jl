# Spike-only: every unrolling change at once (gather stencil, atomic block add, prefix sums, pick, γ);
# each is its own file in unroll/, so that they can be applied and timed one at a time.
for f in ("gather", "atomic", "prefix", "pick", "gamma")
    include(joinpath(@__DIR__, "unroll", f * ".jl"))
end
