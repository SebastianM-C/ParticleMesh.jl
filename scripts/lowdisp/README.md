# Low-dispersion field solvers in Smilei on the ionization-injection case

A study, not part of the package: the WP-064/065 case (`scripts/optim/born2d_case.jl`, a₀ = 1.9 at
the base pulse) in Smilei v5.1 with its Yee, M4 and Lehe Maxwell solvers at λ₀/16 to λ₀/64, to see
what a low-dispersion solver would change in J, the wake's phase velocity and the injection.

| File | Role |
|---|---|
| `gen.py` | The Smilei namelist of the case for a solver, a time step, a resolution (`--ppl`, `--ny` as `gridppc.jl`), vacuum or plasma |
| `run.sh`, `campaign.sh`, `campaign2.sh` | One run on 4 OpenMP threads; the plasma campaign (after the vacuum runs); the λ₀/96 reference |
| `calib.py`, `energy.py` | The launched E_y energy of a vacuum run at t = 100; the plasma runs scale a₀ so that every configuration launches the energy of the λ₀/64 M4 run |
| `analyze.py` | `summary.json` of a run: J as `trapped_energy` (window x from ParticleMesh's box back), born electrons, axis series of the laser and of the wake's E_x zeros |
| `vacfit.py` | The laser's velocity in vacuum per configuration, against `dispersion.py` |
| `collect.py`, `regimes.py` | All summaries, ParticleMesh's grid campaign (the grid report's data) and the dispersion theory into the page's `data.json`; Yee's error against the physical wake deficit over resolution and geometry |
| `pack.py` | A run directory for its run branch: HDF5 outputs repacked as float32 with gzip, plasma runs without their field dump |
| `page/` | The report page (`page.html`) and `build.py`, which embeds `data.json` |
| `dispersion.py` | The numerical dispersion and stability of Smilei's 2D Yee, Lehe and M4 stencils (Faraday stencil of the solver, Ampère stencil Yee) |

Smilei is built from tag v5.1 at commit 438d43d (the image recipe's pin) with MPI, OpenMP and
HDF5; `analyze.py` needs h5py and numpy.

Run data: branch `runs/lowdisp-smilei-1` (`scripts/rundata/run-data.sh fetch lowdisp-smilei-1`).
