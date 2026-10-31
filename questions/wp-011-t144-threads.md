# WP-011: T-144 at 4 threads, and the signature of `fd_scan`

Work package: WP-011. Files: `lib/ParticleMeshCore/test/electrostatic.jl` (T-144), `lib/ParticleMeshCore/test/helpers.jl` (`fd_scan`), `trace/wp/WP-011-engine-module-split.md`.

## 1. T-144 does not hold at 4 threads, before or after the move

WP-011's first acceptance item asks T-130 to T-145 to pass unchanged on 1 and 4 threads. All of them do on 1 thread. On 4 threads every one passes except T-144 (allocation per step does not grow with the particle count): in `Reference()` mode one step allocates about 8 B per particle with more than one thread (4 threads: 17712 B at 512 particles, 79840 B at 8192, 4.2 MB at 524288), and a constant 1856 B with one thread. The spike's code at 4ed3667 gives the same bytes, so the move did not cause it; the spike measured T-144 on one thread only. The `Fast()` step allocates nothing at any thread count, so the growth is in the sort: AcceleratedKernels' threaded CPU sample sort appears to call `Base.sort!` on each bucket without the scratch the workspace provides (suspected, reported as an `upstream` issue).

Options: (a) keep WP-011 `active` until the upstream fix or a workaround lands; (b) count the item as met for the code WP-011 changed, and track T-144 at more than one thread in the upstream issue; (c) work around it in the engine (a sort algorithm or task count that does not allocate), which changes the Reference mode's performance and is outside WP-011's scope.

Assumption: (a). WP-011 stays `active` with this as its one open acceptance item; T-144's assertion is unchanged and the CPU CI, which runs one thread, passes.

## 2. `fd_scan` takes the AD value as a keyword

The package names the signature `fd_scan(f, x0, hs; seed)` and asks it to return the floor relative to the AD value, which the caller computes. A function that does not receive the AD value cannot return a floor relative to it, so `fd_scan` takes it as a required keyword: `fd_scan(f, x0, hs; seed, ad)`. Computing `ad` with the same `f`, mode and seed stays the caller's obligation, as the docstring says.

Assumption: the keyword is what the package meant; the alternative, returning the absolute spread and leaving the division to the caller, would move part of A7's definition out of the helper.
