# A1. Units and normalization

The core works in normalized units (c = 1, lengths in c/ω₀ or c/ω_p, fields in m_e c ω/e); the user layer offers SI and possibly other unit systems, as EDM does. ω₀ is the reference frequency (time in 1/ω₀, length in c/ω₀, fields in m_e c ω₀/e, density in units of n_c), matching Smilei; wake quantities in ω_p units are derived views. Nondimensionalize the optimization parameters to O(1), so FP32 tangents are well scaled.
