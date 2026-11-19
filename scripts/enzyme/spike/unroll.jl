# Spike-only: the unrolling changes the engine does not carry (prefix sums, pick, γ). Each is its
# own file in unroll/. They matter for Enzyme on the CPU's POCL backend only (OpenCL builtins
# without derivatives); on CUDA the engine's own gather and atomic block add suffice (WP-050).
for f in ("prefix", "pick", "gamma")
    include(joinpath(@__DIR__, "unroll", f * ".jl"))
end
