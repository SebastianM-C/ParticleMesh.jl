"""Laser energy in the box from the t = 100 field dump: 1/2 sum (Ex^2 + Ey^2 + Bz_m^2) dx dy."""
import h5py, json, math, sys, numpy as np

def target():
    a0, w, tau = 1.9, 10.0, 6.0
    return math.pi * a0**2 * w * tau / 4 * (1 + 1 / (2 * w**2))

def box_energy(run):
    m = json.load(open(f"{run}/meta.json"))
    g = h5py.File(f"{run}/Fields0.h5")
    k = list(g["data"].keys())[0]
    U = 0.0
    for f in ("Ex", "Ey", "Bz_m"):
        a = np.array(g["data"][k][f])
        U += 0.5 * np.sum(a**2) * m["dx"] * m["dy"]
    return U

if __name__ == "__main__":
    for r in sys.argv[1:]:
        U = box_energy(r)
        print(r, "U", U, "target", target(), "ratio", U / target(), "a0scale_fix", math.sqrt(target() / U))
