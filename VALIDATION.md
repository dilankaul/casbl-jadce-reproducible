# Validation performed before packaging

The cleaned package was validated in the build environment with:

```bash
PYTHONPATH=src pytest -q
```

Result: **23 tests passed**.

The test suite covers:

- paper matrix dimensions and unit-norm pilots;
- optimized Cholesky posterior calculations against the full reference form;
- numerically stable CA-SBL update and the `phi <= 0` SBL fallback;
- CA-SBL with `alpha=0` against conventional SBL;
- `Omega` construction and corrected fast `phi = (Omega @ gamma) / M` expression;
- event activation probability cutoff at `D`;
- unconditioned/variable realized activity and non-forcing of the target `S`;
- deterministic kappa calibration behavior;
- MMV-OMP and MMV-CoSaMP oracle-support sanity tests, including zero sparsity and the `K_r=min(S_r,L,N)` cap;
- deterministic communication-condition generation;
- threshold-curve selection behavior;
- activity support-size plotting from saved Task 02 data;
- probability-grid agreement with the event model;
- vector `contourf` probability-field output;
- contour levels spanning exactly `[0, 1]` and a colormap whose value at `P=0` is pure white.

The probability field is not epsilon-shifted, thresholded, or masked for visual effect. The only mask retained by the plotting grid is outside the physical circular BS cell.

Manual repository checks additionally confirmed that scientific task scripts are uniquely numbered 01-09, utility scripts are unnumbered, and numbered tasks default to saving their stage-specific figures automatically. All script entry points also passed `--help` import/syntax checks.
