# Validation performed before packaging

The package was syntax-compiled and its unit tests were run with `PYTHONPATH=src pytest -q` in the build environment.

Result: **9 tests passed**.

The tests cover:

- paper matrix dimensions and unit-norm pilots;
- optimized Cholesky E-step vs the full matrix-inverse reference implementation;
- numerically stable CA-SBL root vs the direct quadratic root;
- `phi <= 0` fallback;
- CA-SBL with `alpha=0` vs conventional SBL;
- `Omega` and the fast `phi` expression;
- event probability cutoff at `D`;
- exact-support recovery on an identity-system sanity test for MMV-OMP and MMV-CoSaMP;
- deterministic communication realization generation.

A separate tiny end-to-end smoke configuration was also run successfully through activity generation, communication preview, correlation construction, alpha/beta tuning, threshold selection, all four estimators, convergence output, evaluation CSVs, gamma-distribution output, and figure generation.
