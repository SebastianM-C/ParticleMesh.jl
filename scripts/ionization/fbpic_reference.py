"""ADK rates as FBPIC computes them (WP-059), the reference of the ionization tests.

    python3 -I fbpic_reference.py <path to fbpic's atomic_data.txt> > ../../lib/Wakefields/test/fbpic_reference.toml

The three coefficient lines are those of FBPIC 0.27.1, `Ionizer.initialize_ADK_parameters`
(fbpic/particles/elementary_process/ionization/ionizer.py, 3-Clause-BSD-LBNL), without its factor
dt, and the rate is that of its `get_ionization_probability` for an ion at rest:
`prefactor * E**power * exp(exp_prefactor / E)`. The ionization energies are read from the table
FBPIC ships. Two variants: "fbpic", with FBPIC's own unit of energy, U_H = hydrogen's measured
ionization energy; and "rydberg", the same lines with U_H = R∞·h·c, half a Hartree, which is the
atomic unit the formula is written in. Needs numpy and scipy.
"""
import re
import sys

import numpy as np
import scipy
from scipy.constants import c, e, m_e, physical_constants
from scipy.special import gamma


def energies(path):
    rows = {}
    for line in open(path):
        m = re.match(r"\s*(\d+)\s*\|\s*(\S+)\s+\S+\s*\|\s*\+?(\d+)\s*\|\s*[\[\(]?([0-9.]+)", line)
        if m:
            rows.setdefault(m.group(2), []).append(float(m.group(4)) * e)
    return {k: np.array(v) for k, v in rows.items()}


def coefficients(Uion, UH):
    alpha = physical_constants["fine-structure constant"][0]
    r_e = physical_constants["classical electron radius"][0]
    wa = alpha**3 * c / r_e
    Ea = m_e * c**2 / e * alpha**4 / r_e
    Z = np.arange(len(Uion)) + 1
    n_eff = Z * np.sqrt(UH / Uion)
    l_eff = n_eff[0] - 1
    C2 = 2 ** (2 * n_eff) / (n_eff * gamma(n_eff + l_eff + 1) * gamma(n_eff - l_eff))
    adk_power = -(2 * n_eff - 1)
    adk_prefactor = wa * C2 * (Uion / (2 * UH)) * (2 * (Uion / UH) ** (3.0 / 2) * Ea) ** (2 * n_eff - 1)
    adk_exp_prefactor = -2.0 / 3 * (Uion / UH) ** (3.0 / 2) * Ea
    return n_eff, C2, adk_prefactor, adk_power, adk_exp_prefactor


def main(path):
    table = energies(path)
    wavelength = 0.8e-6
    omega = 2 * np.pi * c / wavelength
    a0s = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    UH_fbpic = table["H"][0]
    UH_rydberg = physical_constants["Rydberg constant times hc in J"][0]
    num = lambda x: repr(float(x))
    print("# Written by fbpic_reference.py, whose header says what the two variants are. Rates are per ω₀.")
    print('source = "FBPIC 0.27.1 ionizer.py coefficient lines and atomic_data.txt; scipy %s constants"' % scipy.__version__)
    print("wavelength_m = %s" % num(wavelength))
    print("a0 = [%s]" % ", ".join(num(a) for a in a0s))
    for element in ("H", "He", "N", "Ar"):
        Uion = table[element]
        cf = coefficients(Uion, UH_fbpic)
        cr = coefficients(Uion, UH_rydberg)
        for i in range(len(Uion)):
            print("\n[[levels]]")
            print('element = "%s"' % element)
            print("charge = %d" % (i + 1))
            print("energy_eV = %s" % num(Uion[i] / e))
            for name, co in (("fbpic", cf), ("rydberg", cr)):
                n_eff, C2, pre, power, expo = (x[i] for x in co)
                E = np.array(a0s) * m_e * c * omega / e
                rate = pre * E**power * np.exp(expo / E)
                print("%s = { n_eff = %s, C2 = %s, rate = [%s] }" % (name, num(n_eff), num(C2), ", ".join(num(r) for r in rate / omega)))


if __name__ == "__main__":
    main(sys.argv[1])
