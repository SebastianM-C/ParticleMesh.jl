| method | Julia threads | one run, compiled (s) | of which reverse phase (s) | × primal | × Dual{4} | pool peak (GiB) | state arrays (GiB) | card peak above idle (GiB) | dJ/dτ | dJ/dw | dJ/dfocus | dJ/dn | largest relative difference from Dual{4} |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primal | 16 | 2.08 |  | 1.00 | 0.18 | 0.21 | 0.21 | 0.77 |  |  |  |  |  |
| Dual{1} | 16 | 3.01 |  | 1.44 | 0.26 | 0.37 | 0.36 | 0.97 | -0.00616895 |  |  |  | 4.2e-05 |
| Dual{4} | 16 | 11.50 |  | 5.52 | 1.00 | 0.83 | 0.81 | 2.31 | -0.00616921 | 0.000767023 | 0.00203784 | 1.6519 |  |
| reverse | 16 | 26.13 | 23.5 | 12.54 | 2.27 | 28.72 | 19.11 (1 + 90 checkpoints) | 30.52 (card full) | -0.00616891 | 0.000767262 | 0.00203786 | 1.65184 | 3.1e-04 |
| reverse | 2 | 32.90 | 29.9 | 15.79 | 2.86 | 28.72 | 19.11 (1 + 90 checkpoints) | 30.52 (card full) | -0.0061689 | 0.000767263 | 0.00203786 | 1.65184 | 3.1e-04 |
| reverse | 1 | 43.89 | 40.2 | 21.06 | 3.82 | 28.72 | 19.11 (1 + 90 checkpoints) | 30.52 (card full) | -0.00616891 | 0.000767258 | 0.00203786 | 1.65184 | 3.1e-04 |
