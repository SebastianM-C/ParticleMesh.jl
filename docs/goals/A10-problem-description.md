# A10. Problem description separated from numerical implementation

Principle, the reason to keep the symbolic connection: it enforces a separation of concerns between describing the problem and implementing the numerics. Not blocking; it shapes the kernel interfaces (architecture §0.5, R13). The mesh-solve slot admits Yee kernels and MOL's staggered update now, the MOL Ferrite backend (expected, not public) and Trixi DG later; none of the later ones serves the LWFA module.
