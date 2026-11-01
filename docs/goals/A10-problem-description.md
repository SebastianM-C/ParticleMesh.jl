# A10. Problem description separated from numerical implementation

Principle, the reason to keep the symbolic connection: it enforces a separation of concerns between describing the problem and implementing the numerics. Not blocking; it shapes the kernel interfaces, and with them the numerical backends the mesh-solve slot admits (`architecture-draft.md` §0.5).
