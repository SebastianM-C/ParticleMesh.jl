# Choices WP-063 took that its text did not settle

The package asks for a rate law that plugs in by the level's type and for a barrier-suppression level. These are the choices made on the way.

## 1. Which barrier-suppression rate

**Assumption:** the tunnelling rate times the empirical factor of Tong and Lin, `exp(−α·(Z²/I)·E/(2I)^{3/2})` in atomic units (`TongLinLevel`, in the engine's units `exp(−suppression·E/field)` with `suppression = 4α·n*²`). It is smooth in the field and in its numbers at every field (A2), costs one number per level, and goes over to the tunnelling rate at weak fields.

- **Not the piecewise rate.** The other form PIC codes use joins the tunnelling rate to a quadratic and then a linear law at the fields where they cross (Bauer and Mulser; Kostyukov, Artemenko and Golovanov, as Remme et al., arXiv:2502.02697, describe them). The joins are continuous but have a kink, which a derivative does not pass through; it would need smoothed joins. It is not built; with the rate dispatched on the level's type it is a level type of its own, with no change to the step.
- **Where the form was read.** Tong and Lin's paper (J. Phys. B 38 (2005) 2593) could not be reached from the session. The factor is taken as two papers that use it state it, both read in their arXiv sources: Lötstedt, Ciappina and Yamanouchi (Phys. Rev. A 102, 013112, arXiv:2201.03840) and Remme, Voitkiv, Pretzler and Müller (arXiv:2502.02697); the two agree, and the second gives the factor for hydrogen as `exp(−12E)`, which T-820 and T-821 hold the level to. That Smilei has this rate (its `tunnel_TL` model, with a default α of 6) is the author's memory and Remme et al.'s remark, not checked in Smilei's source or documentation, which the session could not reach.
- **Its range.** The factor is a fit to rates computed up to about twice the field `E_bs = I²/(4Z)` at which the barrier is gone (Lötstedt et al.: accurate below about 2·E_bs; Remme et al.: hydrogen's α = 6 works to about 2.5·E_bs). Far above that the rate falls to zero with rising field (T-820 has the maximum), which is wrong. For nitrogen's K shell at 0.8 µm, E_bs is a₀ = 2.2 (N⁵⁺) and 2.7 (N⁶⁺), so the form covers ionization injection up to a₀ of about 5; a level in a field several times its E_bs (an outer shell under the pulse's peak) keeps the ADK level or gets another law. The constructor does not refuse such a use: the engine does not know the fields of a run in advance.

## 2. The numbers α

**Assumption:** `scripts/ionization/tong_lin.toml` holds α = 6 for hydrogen-like ions and the table of Lötstedt et al. for helium-like ions of Z = 2 to 36; `tong_lin_alpha` gives these and refuses every other level, whose α the caller passes.

- **Hydrogen-like ions.** α = 6 is Tong and Lin's value for hydrogen as both papers quote it. A hydrogen-like ion of nuclear charge Z has hydrogen's rate at the field F/Z³ times Z² (non-relativistic), and the factor depends on F/Z³ alone, so the same α holds.
- **Other levels.** Tong and Lin tabulate α for more species (rare gases among them); those values were not read and are not in the table. Nothing is guessed for them.
- **The conventions of the fit.** Lötstedt et al. fitted α with a tunnelling rate that counts both 1s electrons (a factor 2) and has l* = 0 in its coefficient. `tong_lin_level` therefore defaults to `lstar = 0` and, for a helium-like ion, `electrons = 2`, so that its rate is the one T-821 holds against their reference rates. These are not the defaults of `adk_level`, which follow FBPIC (l* from the neutral atom, no factor for equivalent electrons): at weak fields the two levels of N⁵⁺ differ by a factor of about 2 for that reason, not because of the suppression. `adk_level(…; lstar = 0, electrons = 2)` is the tunnelling level with the fit's conventions. Whether WP-059's default should count both K-shell electrons is the owner's call; it is left as it is because T-783 holds it to FBPIC's coefficients.

## 3. The reference rates

**Assumption:** the supplementary table of Lötstedt et al. holds the rate in atomic units at the scaled field F' = 8F/Z³. The file says "ionization rates" and does not say whether they are scaled; with the paper's scaled rate W' = 4W/Z² the formula of the same paper would miss the rows with Z > 2 by Z²/4, and unscaled it reproduces them to about 10 %, which is what the paper states. Rows with five entries leave columns empty in a file whose tabs do not line up; the entries are taken as F' = 0.2, 0.3, 0.4, 0.5 and 0.6, where they continue the neighbouring full rows. Only the six ions whose energies `ionization_energies.toml` has are copied.

- **What agrees and what does not.** From 0.7 to 1.75 E_bs the level is within 12 % of the reference for all six ions (T-821, the report). Below 0.7 E_bs it is up to 31 % low: there the tunnelling rate without the factor is close to the reference and the factor, linear in the field in the exponent, takes too much. Above 1.75 E_bs (He and Li⁺ only) it is up to 48 % low at 2.9 E_bs, the fit's end. The energies here are measured ones and the paper's a non-relativistic expansion, which is a tenth of the difference at Z = 18 and less below.

## 4. The interface

**Assumption:** a level type subtypes `IonizationLevel{T}` and has four methods: `ionization_rate_sq` (the rate from |E|², a device function), `level_as` (the element type's conversion), `level_numbers` and `with_level_numbers` (its parameters, for the tunables of WP-060). A description may list levels of different types; the kernel maps over the tuple.

- **The contract is not checked.** A rate that is not zero at zero field with zero partials, not C¹, or negative breaks what WP-059's tests establish (no slot taken in no field, smooth weights); the module documents the contract and does not test a foreign type against it.
- **`adk_rate` stays** as the ADK level's `ionization_rate`, since reports and scripts use it.
- **Bits.** A problem with ADK levels runs the same arithmetic as before (T-824 holds a Tong–Lin level of zero suppression to the ADK level's digest; `scripts/digest/digest.jl` and T-775's digests were compared between `main` and this branch on one machine, the pull request says with what result).
