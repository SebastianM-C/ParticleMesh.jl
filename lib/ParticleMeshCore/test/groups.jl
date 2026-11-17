# The test groups that CI runs in parallel jobs (.github/workflows/CI.yml), shared by this package's
# runtests.jl and the electromagnetic module's. A suite's groups are in its `test/groups.toml`;
# `PM_TEST_GROUP=<n>` runs the n-th group, and unset or empty, the whole suite runs.

using TOML: TOML

"""
    selected_test_files(files, spec; env = get(ENV, "PM_TEST_GROUP", "")) -> Vector{String}

The files a suite includes. `files` is the whole suite in its order; `spec` is its parsed
`groups.toml`: `groups`, each a list of files in suite order that also carries the earlier files whose
helpers or runs its own files reuse, so that a group runs on its own in a fresh process; and
`full_suite_only`, the files that run only in the whole suite. Throws when a file is in no group and
not in `full_suite_only`, so that a new test file cannot drop out of CI, and when `env` names no group.
"""
function selected_test_files(files, spec; env::AbstractString = get(ENV, "PM_TEST_GROUP", ""))
    groups = Vector{String}.(spec["groups"])
    full_only = Vector{String}(spec["full_suite_only"])
    for (i, g) in enumerate(groups)
        unknown = setdiff(g, files)
        isempty(unknown) || error("test group $i names files the suite does not have: $(join(unknown, ", "))")
        allunique(g) && issorted(indexin(g, files)) || error("test group $i is not in the suite's order")
    end
    isempty(setdiff(full_only, files)) || error("full_suite_only names files the suite does not have")
    orphans = setdiff(files, full_only, reduce(vcat, groups; init = String[]))
    isempty(orphans) || error("test files in no group of groups.toml, so CI would not run them: $(join(orphans, ", "))")
    isempty(env) && return collect(files)
    i = tryparse(Int, env)
    (i !== nothing && 1 <= i <= length(groups)) ||
        error("PM_TEST_GROUP must be unset or a group number from 1 to $(length(groups)); it is $(repr(env))")
    return groups[i]
end

"""
    include_test_files(mod, dir, files)

Include the selected files of the suite in `dir` (its `groups.toml`) in `mod`, printing which group
runs and the time of each file, the numbers to rebalance the groups by. A group with `PM_TEST_GPU=1`
throws: the GPU testsets run only in the whole suite, and a required GPU run must not skip them.
"""
function include_test_files(mod::Module, dir, files)
    spec = TOML.parsefile(joinpath(dir, "groups.toml"))
    selected = selected_test_files(files, spec)
    env = get(ENV, "PM_TEST_GROUP", "")
    if !isempty(env)
        get(ENV, "PM_TEST_GPU", "") == "1" &&
            error("PM_TEST_GPU=1 requires the GPU testsets, which run only in the whole suite; unset PM_TEST_GROUP")
        println("Test group $env of $(length(spec["groups"])): ", join(selected, ", "))
        println("Run only in the whole suite (PM_TEST_GROUP unset): ", join(spec["full_suite_only"], ", "))
    end
    for f in selected
        t = time()
        Base.include(mod, joinpath(dir, f))
        println("Time of $f: $(round(time() - t; digits = 1)) s")
    end
end
