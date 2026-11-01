# The GPU switch of WP-013, read once by setup.jl (and on its own by T-199's subprocess).

using CUDACore: CUDACore

"""
    gpu_switch(env = get(ENV, "PM_TEST_GPU", ""), functional = CUDACore.functional()) -> (; run, reason)

Whether the GPU testsets run. `PM_TEST_GPU` unset (or empty): they run if and only if CUDA is
functional, so a plain `Pkg.test()` on a machine with a GPU runs them and CPU CI skips them.
`PM_TEST_GPU=1` requires them: without a functional GPU this throws, so the suite fails and a
GPU run can never skip silently. `PM_TEST_GPU=0` turns them off. Any other value throws.
"""
function gpu_switch(env::AbstractString = get(ENV, "PM_TEST_GPU", ""), functional::Bool = CUDACore.functional())
    if env == "0"
        return (; run = false, reason = "PM_TEST_GPU=0")
    elseif env == "1"
        functional || error("PM_TEST_GPU=1 requires the GPU tests, but CUDA is not functional on this machine")
        return (; run = true, reason = "PM_TEST_GPU=1")
    elseif isempty(env)
        return functional ? (; run = true, reason = "a functional CUDA GPU was found") :
               (; run = false, reason = "no functional CUDA GPU (CUDACore.functional() is false)")
    end
    error("PM_TEST_GPU must be unset, 0 or 1; it is $(repr(env))")
end

"The one line the suite prints when the GPU testsets do not run."
gpu_skip_line(sw) = "GPU tests (WP-013) not run: $(sw.reason)"
