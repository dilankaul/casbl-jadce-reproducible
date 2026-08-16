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
13. MMV-OMP and MMV-CoSaMP receive the known sparsity `S`; they are oracle-sparsity baselines.
14. CoSaMP uses SVD least squares with no hidden ridge regularizer.
15. All algorithms share identical Monte Carlo realizations per condition.
16. Tuning and final evaluation use separate deterministic seeds.
17. Large communication tensors are regenerated from saved seeds instead of stored, keeping outputs Git-friendly.
