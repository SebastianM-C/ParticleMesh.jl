"""Numerical dispersion of Smilei's 2D Yee, Lehe and M4 solvers (Faraday stencil of the solver,
Ampere stencil Yee, as MF_Solver2D_*.cpp and MA_Solver2D_norm.cpp in Smilei v5.1).

w(kx, ky): sin^2(w dt/2)/dt^2 = Fx(k) Ax(k) + Fy(k) Ay(k), F the solver's discrete d/dx (B update), A Yee's."""
import numpy as np, math

def coeffs(solver, dx, dy, dt):
    rx, ry = dt / dx, dt / dy
    if solver == "Yee":
        return dict(ax=1.0, bxy=0.0, dx_=0.0, ay=1.0, byx=0.0, dy_=0.0)
    if solver == "Lehe":
        byx = 1 / 8; bxy = (dx / dy) ** 2 / 8
        dlx = 0.25 * (1 - (math.sin(math.pi * rx / 2) / rx) ** 2)
        return dict(ax=1 - 2 * bxy - 3 * dlx, bxy=bxy, dx_=dlx, ay=1 - 2 * byx, byx=byx, dy_=0.0)
    if solver == "M4":
        byx = rx * rx / 12; bxy = ry * ry / 12
        dlx = byx - 1 / 12; dly = bxy - 1 / 12
        return dict(ax=1 - 2 * bxy - 3 * dlx, bxy=bxy, dx_=dlx, ay=1 - 2 * byx - 3 * dly, byx=byx, dy_=dly)
    raise ValueError(solver)

def rhs(solver, dx, dy, dt, kx, ky):
    c = coeffs(solver, dx, dy, dt)
    sx, sy = np.sin(kx * dx / 2), np.sin(ky * dy / 2)
    Fx = (2 / dx) * ((c["ax"] + 2 * c["bxy"] * np.cos(ky * dy)) * sx + c["dx_"] * np.sin(3 * kx * dx / 2))
    Fy = (2 / dy) * ((c["ay"] + 2 * c["byx"] * np.cos(kx * dx)) * sy + c["dy_"] * np.sin(3 * ky * dy / 2))
    Ax, Ay = (2 / dx) * sx, (2 / dy) * sy
    return Fx * Ax + Fy * Ay

def omega(solver, dx, dy, dt, kx, ky=0.0):
    r = rhs(solver, dx, dy, dt, kx, ky) * dt * dt / 4
    return 2 / dt * np.arcsin(np.sqrt(np.clip(r, 0, None)))

def stability(solver, dx, dy, dt, n=400):
    kx = np.linspace(0, math.pi / dx, n)[:, None]; ky = np.linspace(0, math.pi / dy, n)[None, :]
    return float(np.max(rhs(solver, dx, dy, dt, kx, ky) * dt * dt / 4))  # stable if <= 1 (and >= 0)

def velocities(solver, dx, dy, dt, k=1.0, h=1e-5):
    """Phase and group velocity along x of the laser's k (in c) at ky = 0."""
    w = omega(solver, dx, dy, dt, k)
    vg = (omega(solver, dx, dy, dt, k + h) - omega(solver, dx, dy, dt, k - h)) / (2 * h)
    return float(w / k), float(vg)

if __name__ == "__main__":
    for ppl in (16, 32, 48, 64):
        ny = ppl // 16
        dx, dy = 2 * math.pi / ppl, math.pi / 4 / ny
        dty = 0.95 / math.sqrt(1 / dx**2 + 1 / dy**2)
        for s in ("Yee", "Lehe", "M4"):
            for f in (dty / dx, 0.9, 0.95, 1.0):
                dt = f * dx
                st = stability(s, dx, dy, dt)
                vp, vg = velocities(s, dx, dy, dt)
                print(f"r{ppl}x{ny} {s:4s} dt/dx={f:.3f} max={st:.4f} {'OK ' if st <= 1 + 1e-12 else 'BAD'} 1-vphi={1-vp:+.2e} 1-vg={1-vg:+.2e}")
