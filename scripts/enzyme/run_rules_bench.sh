#!/usr/bin/env bash
# WP-077: Enzyme's reverse pass through the 2D step on CUDA with and without the engine's
# hand-written reverse rules (ParticleMeshCoreEnzymeCoreExt; scripts/enzyme/rules_off.jl switches
# them off), on the chain-rule case of chain_rule.jl (WP-023's plasma_step_2d, 10 × 8 cells,
# Float64, seed 7), for 1, 4 and 16 steps in Fast() and Reference() mode, then the engine's test
# suite with the GPU testsets required (T-1066).
#
# It runs in a fresh worktree of this branch and takes the WP-050 spike's harness from its branch
# (chain_rule.jl and scripts/enzyme/spike: the patch for Active scalar kernel arguments, KA draft
# #979, and the unrolled gather and atomic adds), and adds Enzyme and KernelAbstractions 0.10.8 (its
# Enzyme extension) to the electromagnetic module's test environment, as that branch does. The worktree is left dirty on purpose; each run's
# manifest says so.
#
#     scripts/enzyme/run_rules_bench.sh [spike branch, default claude/project-thread-4mzyoi]
set -u
cd "$(dirname "$0")/../.."
spike=${1:-claude/project-thread-4mzyoi}
out="scripts/runs/wp077-rules-bench-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$out"
git rev-parse HEAD > "$out/commit.txt"
git fetch -q origin "$spike" && git rev-parse FETCH_HEAD > "$out/spike-commit.txt"
git checkout FETCH_HEAD -- scripts/enzyme/spike scripts/enzyme/chain_rule.jl scripts/enzyme/common.jl
# the unrolled gather and atomic adds as patches: the spike's tip moved them into its engine
# (21d1cc5), which this branch does not have; 8443716 has the same code as patches
git checkout 8443716d -- scripts/enzyme/spike/unroll/gather.jl scripts/enzyme/spike/unroll/atomic.jl
julia +1.13 --startup=no --project=lib/ParticleMeshElectromagnetic/test -e 'using Pkg; Pkg.add([PackageSpec(name = "Enzyme", version = "0.13.221"), PackageSpec(name = "KernelAbstractions", version = "0.10.8")])' > "$out/setup.log" 2>&1
julia +1.13 --startup=no --project -e 'using Pkg; Pkg.instantiate(workspace = true)' >> "$out/setup.log" 2>&1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$out/gpu.txt" 2>&1
J=(julia +1.13 --startup=no --project=lib/ParticleMeshElectromagnetic/test)
export PM_BACKEND=cuda PM_CUDA_HEAP=1073741824
base="scripts/enzyme/spike/active_args.jl;scripts/enzyme/spike/unroll/gather.jl;scripts/enzyme/spike/unroll/atomic.jl"
for mode in fast reference; do
    for steps in 1 4 16; do
        for rules in on off; do
            patch=$base
            [ "$rules" = off ] && patch="$base;scripts/enzyme/rules_off.jl"
            log="$out/$mode-$steps-rules-$rules.log"
            echo "== $(date -u +%H:%M:%S) $mode $steps steps, rules $rules" | tee -a "$out/summary.txt"
            PM_ENZYME_PATCH=$patch timeout 3600 "${J[@]}" scripts/enzyme/chain_rule.jl "$mode" "$steps" 1 Float64 7 > "$log" 2>&1
            echo "   exit $?" >> "$out/summary.txt"
            grep -E 'rules_off|relative difference|verdict|times:|wrote|^ERROR|Segmentation|signal [0-9]' "$log" | cut -c1-300 >> "$out/summary.txt"
        done
    done
done
echo "== $(date -u +%H:%M:%S) engine suite with PM_TEST_GPU=1" | tee -a "$out/summary.txt"
PM_TEST_GPU=1 timeout 7200 julia +1.13 --startup=no --project=lib/ParticleMeshCore -e 'using Pkg; Pkg.test()' > "$out/core-suite-gpu.log" 2>&1
echo "   exit $?" >> "$out/summary.txt"
grep -E 'T-1066|Test Summary|^ParticleMeshCore|Fail|Error' "$out/core-suite-gpu.log" | cut -c1-300 | tail -40 >> "$out/summary.txt"
echo "== done $(date -u +%H:%M:%S), logs in $out" | tee -a "$out/summary.txt"
