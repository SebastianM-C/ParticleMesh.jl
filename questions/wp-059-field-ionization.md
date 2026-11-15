# Choices WP-059 took that its text did not settle

The package asks for field ionization without random numbers, released by weight from atoms on a fixed lattice. These are the choices made on the way; each can be changed without changing the others.

## 1. The rate

**Assumption:** the static-field ADK rate in the form PIC codes that resolve the laser cycle use, `W = C²·B·I·(2(2I)^{3/2}/E)^{2n*−|m|−1}·exp(−2(2I)^{3/2}/(3E))` in atomic units, with l = m = 0 by default, written as the module's own generic function of three numbers per level (`ADKLevel`). The form and the constants were checked against FBPIC 0.27.1's source, read from its PyPI source distribution (T-783). The documentation of Smilei and WarpX and the NIST database could not be reached from the session (the hosts are not on its allow list), so nothing here was compared with them.

- **l\*.** FBPIC's choice, l* = n*₀ − 1 with n*₀ of the neutral atom, is the default of the user layer. The convention l* = n* − 1 per level is available (`lstar = :level`); that it is Smilei's is the author's memory, not checked.
- **The unit of energy.** The formula is in atomic units, so the user layer uses half a Hartree. FBPIC uses hydrogen's measured ionization energy instead, 5.3·10⁻⁴ smaller, which moves n* by 2.7·10⁻⁴ and the rates by what T-783 measures. Not followed, because it is not the unit the formula is written in.
- **No barrier-suppression correction.** Above the barrier-suppression field the ADK form overestimates the rate. For a level that the pulse's foot ionizes completely this changes when, by a few steps, not how much. For a level ionized near the peak of a strong pulse it matters; a follow-up issue holds it.
- **The cutoff.** The rate is taken to exactly zero, with a continuous derivative, where `field/(3E)` exceeds `cutoff` (64 by default, a factor below e⁻⁵⁶ of the prefactor). An atom in a weaker field releases nothing and occupies no electron slot.

## 2. The populations

**Assumption:** per atom and level a bound population, advanced each step by `f_j ← f_j·exp(−W_j·Δt) + f_{j−1}·(1 − exp(−W_{j−1}·Δt))` from the populations at the start of the step. It is exact for one level in a constant field, conserves the number of electrons to roundoff, keeps every population in [0, 1] for any `W·Δt`, and is first order in Δt in the coupling between levels (T-772): what a level gains starts to decay one step late. An atom therefore passes at most one level per step, so the outer shell of a dopant in a strong field takes one step per level. FBPIC's Monte Carlo has the same limit: one draw and one increment of the level per ion and step (`ionize_ions_numba`, read in its source). An exact solution of the chain within a step would remove it; it was not needed for the levels the injection depends on.

## 3. The births

**Assumption:** one electron per atom and birth step, of the weight the atom has released since its last birth, at the atom's place, at rest at the start of the step (the pusher's half step back, as `initialize!` centres a load). Electrons from several levels released at one birth are one particle.

- **`every`.** Births every `every` steps bound the number of particles. What is released between births is born at the later step's phase of the laser, so `every·Δt` has to stay well below the laser period; `every = 1` is exact in this respect.
- **Slots.** Atom slot i is offered electron slot `count + i` at each birth step, so a birth step needs as many free slots as there are atom slots whether or not the atoms release. An atom that finds none keeps its released fraction and is counted; no charge is lost (T-776).
- **A species of their own.** The born electrons are not put into one of the problem's species: those buffers, their loads and their window refill stay as they are, and the run without ionization keeps its bits.

## 4. The ions and the charge

**Assumption:** the dopant's ions are immobile (A2 has immobile ions as the default). An electron is born on its ion, so a birth changes no charge density; Gauss's law holds to roundoff with the ions' charge counted (`ionization_charge!`, T-773, T-774). The energy the ionization takes from the field (the ionization current of the standard codes) is not modelled: a dopant of 5·10⁻⁴ n_c whose K shell takes 552 eV per electron holds 5·10⁻⁷ n_c·mₑc² of ionization energy per electron released, against the laser's a₀²/2 n_c·mₑc², of order one. A follow-up issue holds it.

## 5. The lattice

**Assumption:** atoms on every column the plasma is loaded on and on the rows the description selects, each of the weight of the dopant's profile at its place, including zero. A doped section therefore enters through weights only (A3), the atoms' buffer has a static shape (§0.4), and the atoms ahead of or behind the section cost a rate evaluation each per step and release nothing.

## 6. Dimensions and backends

**Assumption:** N = 2 only, as the moving window and the laser antenna are. At N = 1 a problem with ionization is refused.

## 7. Atomic data

**Assumption:** a table of ionization energies in `scripts/ionization/ionization_energies.toml` for H, He, Li, N, O, Ne and Ar, copied by program from Mendeleev.jl 1.0.3 in a temporary environment and compared with the table FBPIC ships (NIST ASD 5.2). Carbon and krypton are left out because the two sources differ in a late digit. Mendeleev.jl is not a dependency: the table is 7 lines, and `scripts/Manifest.toml` is being changed by another open pull request.

## 8. Cross-code runs

Not done: a run of the same doped target in Smilei or PIConGPU. Their ionization is a Monte Carlo draw per macro-particle, so a comparison is statistical (ensemble means of the trapped charge), which is a campaign of its own; a follow-up issue holds it.
