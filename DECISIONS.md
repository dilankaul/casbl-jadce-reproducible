# Locked implementation decisions

1. Use paper variables: `a`, `H`, `Z`, `Theta`, `Y`, `gamma`, `mu`, `Sigma_diag`, `C`, `Omega`, `eta`, `phi`.
2. Use proper complex-MMV SBL scaling with factor `M`, not the old real-Gaussian `1/2`.
3. Keep `Omega = alpha * (beta * 1 - C)` unchanged.
4. Do not insert an artificial `M` into the gamma correlation term.
5. Therefore use `phi = (Omega @ gamma) / M`.
6. Use the numerically stable positive root `2*eta/(sqrt(1+4*phi*eta)+1)` for `phi>0`.
7. Retain the legacy SBL fallback `gamma=eta` for `phi<=0`.
8. Do not clip gamma to an upper bound of 1. Only roundoff-level negative variances are projected to zero.
9. Retune CA-SBL alpha, beta and threshold under the corrected update. Tune the SBL threshold independently.
10. Use Cholesky solves and compute only `diag(Sigma)` during normal SBL/CA-SBL runs.
11. Use unit-norm QPSK pilots.
12. Implement the paper's event-distance cutoff exactly at `d <= D`.
13. The default event model is unconditioned: `system.S` is a target mean active count, not a hard per-realization constraint. MMV-OMP and MMV-CoSaMP use the feasible oracle sparsity budget `K_r = min(S_r, L, N)`, where `S_r` is the realized support size. They are described as oracle sparsity-budget baselines; capping is recorded whenever `S_r > L`.
14. CoSaMP uses SVD least squares with no hidden ridge regularizer.
15. All algorithms share identical Monte Carlo realizations per condition.
16. Tuning and final evaluation use separate deterministic seeds.
17. Large communication tensors are regenerated from saved seeds instead of stored, keeping outputs Git-friendly.

18. Use `D=15 m` for the revised activity model and calibrate `kappa` separately before Task 02. The initial configured value is `kappa=3.785`, subject to Task 01 calibration.
19. Spatial activity figures may display the exact combined activation field `P(x)=1-prod_v(1-P_v(x))`; this is a visualization of the model probability, not a decorative gradient.
20. Replotting a saved Task 02 realization uses the `D` and `kappa` recorded by Task 02, so later config edits cannot silently change the probability field shown for old data.
21. Probability-field data are never thresholded, offset, epsilon-shifted, or masked for visual effect. Only the region outside the physical BS cell may be masked.
22. Publication probability-field PDFs use vector `contourf` with levels spanning exactly `[0, 1]` and a custom colormap mapping `P=0` to pure white and `P=1` to dark red.

23. Scientific tasks are numbered 01-09 only. Preview, plotting, status, and orchestration scripts are unnumbered utilities.
24. Each numbered task saves its standard PNG and vector-PDF figures automatically by default; plotting utilities exist only for regeneration from saved results.
