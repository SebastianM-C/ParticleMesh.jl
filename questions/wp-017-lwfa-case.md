# WP-017: what the 2D LWFA case still needs before it can be written in both codes

Work package: `trace/wp/WP-017-crosscode-picongpu.md`. A5's first 2D LWFA case is to be run in ParticleMesh, Smilei and PIConGPU. After WP-019 (schema 2, the Smilei 2D generator) and WP-017 (the PIConGPU generator, the 2D periodic path proven in both codes), this is what is missing. The case's values are A5's (`scripts/lwfa/cases/lwfa-2d.toml`); what follows is the schema and matching work it still needs.

## 2. Schema fields that neither WP-019 nor WP-017 added

Schema 2 describes a periodic box with one perturbed Fourier mode. The LWFA case needs, at least:

- **Laser antenna**: the plane, the time envelope and the transverse profile (Gaussian beam at a focus), a₀, polarization, the phase convention (A3: focus and chirp enter through the antenna phase).
- **Moving window**: start time, speed (c), direction.
- **Density profile**: a plateau with a ramp along the propagation axis, uniform or matched transversally, vacuum regions; the refill rule at the window front (A3: through weights).
- **Injected bunch** (if the case has one): charge, energy, spread, position, and its space-charge field at t = 0.
- **Open transverse boundaries**: the absorbing layer (A2: simple absorbing layer first, PML later) for fields and the particle boundary.
- **Current filter**: the binomial pass count per step (schema 1 and 2 have `filter_passes` but both generators refuse anything but 0).
- **Diagnostics**: the electron energy spectrum, a field on a moving grid, the probe positions.

## 3. Matching issues found in WP-017 that will carry over

- **Moving-window axis.** PIConGPU's moving window slides only along y (`simulation/control/MovingWindow.hpp`: "moving window can only slide in y direction"), and its incident-field laser is set per box side. Smilei's moving window moves along x (from Smilei's documentation; not checked in its source here). `scripts/lwfa/picongpu.jl` maps the case's propagation axis x to PIConGPU's y and swaps the field components back in its reader; its window is the visible part of a box with one hidden row of local domains, so the run puts five ranks on the one GPU.
- **Laser injection.** Smilei injects a laser through a boundary condition (`Laser` on a box side, or `LaserGaussian2D`; from Smilei's documentation, not checked in its source here); PIConGPU 0.8.0 injects through its total-field/scattered-field `incidentField` (`param/incidentField.param`) at a plane inside the box, which A2's "antenna plane at the window front" resembles. The two are not the same source term; the injected field should be compared in vacuum before any plasma run.
- **Initial field.** PIConGPU solves no Poisson equation (WP-017 supplies the electrostatic field of a periodic perturbation through `FieldBackgroundE`). For a neutral plasma with a laser this does not matter. For an injected bunch it does: its relativistic space-charge field has to be supplied or the bunch started outside the plasma, while Smilei has `relativistic_field_initialization` (documentation, not checked in the source). The static-background approach of WP-017 does not apply to a moving bunch.
- **Current filter.** PIConGPU's only filter is `--currentInterpolation binomial`, one fixed 3-point pass per axis per step (`fields/currentInterpolation/Binomial.hpp`); Smilei's binomial filter takes a pass count (documentation, not checked in the source). A pass count other than one cannot be matched in PIConGPU without code.
- **Absorbers.** PIConGPU has an exponential damping layer and a PML (`fields/absorber/`), Smilei Silver-Müller and PML conditions (documentation); "simple absorbing layer" (A2) has no common implementation in the two codes, so the comparison must sit far enough from the transverse boundaries that the absorber does not enter.
- **Grid constraints.** PIConGPU needs every axis to be a multiple of the supercell (8 per axis in 2D here) with at least three supercells, and refuses to compile a time step beyond the Courant limit; Smilei needs powers of 2 of patches dividing the cells. A box size that satisfies both should be chosen once.
- **Precision.** PIConGPU defaults to FP32, Smilei runs FP64 (A5). WP-017 measures the FP32–FP64 difference on the 1D set; for LWFA the comparison should name its precision per column.
- **Thermal loading.** PIConGPU's `Temperature` manipulator is the non-relativistic Maxwellian in momentum, Smilei's `maxwell-juettner` below T = 0.1 m_e c² samples the classical energy distribution and sets |p| relativistically; for a cold LWFA plasma this is irrelevant, for a warm one it is an O(T) difference.
- **Kinetic energy.** PIConGPU's particle-energy plugin reports p²/2m below γ = 1.005 and (γ − 1)mc² above (`algorithms/KinEnergy.hpp`); an energy balance of a warm or partly relativistic plasma must be formed from its total-energy column instead, as WP-017's reader does.
- **Diagnostics.** PIConGPU prints the per-component field energies with 6 significant digits (`plugins/EnergyFields.x.cpp`), and every openPMD dump costs about 0.6 s on the RTX 4080 SUPER whatever its size; a moving-window LWFA run with frequent dumps will be dominated by output.
