# AGENTS.md — CASBL JADCE Reproducibility Project

## Project purpose

This repository implements reproducible experiments for the paper
**Correlation-Aware SBL for Device Detection and Channel Estimation in MTC**.
The implementation must remain mathematically consistent with the paper and
suitable for reproducible IEEE journal experiments.

Read `DECISIONS.md` before making algorithmic or experimental changes.

## Working rules

1. Work incrementally; do not redesign the entire pipeline unless explicitly requested.
2. Inspect the current implementation before changing mathematical code.
3. Do not silently change experiment definitions, seeds, dimensions, probability models, or hyperparameter meanings.
4. Preserve existing uncommitted work. Never run `git reset --hard`, `git clean`, or broad `git restore` commands unless explicitly requested.
5. Do not commit changes unless explicitly asked.
6. Run relevant tests after edits; before considering a change complete run `PYTHONPATH=src pytest -q`.
7. Prefer numerically stable linear algebra and avoid explicit matrix inverses.
8. If code behavior and paper equations disagree, identify the discrepancy instead of silently choosing one.

## Numbered task workflow

The scientific workflow has nine uniquely numbered tasks. Utilities are intentionally
unnumbered because they do not represent extra experimental stages.

- Task 01: calibrate the activity model
- Task 02: generate activity realizations
- Task 03: generate the communication-model preview
- Task 04: construct spatial correlation
- Task 05: tune CA-SBL alpha and beta
- Task 06: tune activity thresholds
- Task 07: run estimators
- Task 08: convergence analysis
- Task 09: evaluation / aggregation

Standard figures must be saved automatically by the numbered task that produces
their source data, in both PNG and vector PDF by default. Separate plotting scripts
are regeneration utilities only and must not be required for the normal workflow.
The unnumbered utilities are `preview_activity.py`, `plot_selected_realization.py`,
`plot_activity_s_distribution.py`, `make_figures.py`, `task_status.py`, and
`run_all.py`.

After each task inspect saved outputs before continuing. Long-running tasks must
show progress. Figures must remain regenerable from saved data without rerunning
expensive tasks.

## Reproducibility

Use `np.random.default_rng` and `np.random.SeedSequence`. Keep activity and
communication random streams separate. Do not silently change configured seeds.
All algorithms compared in one condition must receive the same underlying
realization. Tuning and final evaluation data must remain separate.

## Paper notation

Use paper notation in new code where practical:

- `N`: number of MTDs
- `M`: number of BS antennas
- `L`: pilot length
- `S`: target/realized active MTD count as defined by the experiment
- `a`: binary activity vector `(N,)`
- `H`: physical channel matrix `(M, N)`
- `Z`: sparse effective channel `(N, M)`
- `Theta`: pilot/sensing matrix `(L, N)`
- `Y`: received pilots `(L, M)`
- `gamma`: SBL row-variance vector `(N,)`
- `Gamma = diag(gamma)`
- `C`: spatial correlation matrix
- `Omega`: correlation interaction matrix

The system model is `Z = diag(a) H^T` and `Y = Theta Z + W`.
Do **not** replace `H^T` with `H^H`.

## Complex-MMV SBL theory

Rows satisfy `z_i | gamma_i ~ CN(0, gamma_i I_M)`.

Use

`eta_i = Sigma_ii + ||mu_i||_2^2 / M`.

Conventional SBL uses `gamma_i_new = eta_i`.

Use the observation covariance

`Pi = Theta diag(gamma) Theta^H + sigma^2 I`

and Cholesky solves rather than `inv(Pi)`. Do not construct full `N x N`
posterior covariance during normal runs. Compute `mu` and `diag(Sigma)` only.

## Corrected CA-SBL theory

Keep

`Omega = alpha * (beta * 1 - C)`

unchanged. Do not insert an artificial `M` into the gamma interaction term.
The corrected complex-MMV interaction is

`phi = (Omega @ gamma) / M`.

For `phi_i > 0`, use the stable root

`gamma_i_new = 2 * eta_i / (sqrt(1 + 4 * phi_i * eta_i) + 1)`.

Do not use the cancellation-prone quadratic form. For `phi_i <= 0`, retain the
current SBL fallback unless the theory is explicitly revisited. Do not upper-clip
gamma to 1; only numerical non-negativity safeguards are allowed.

Because the interaction scaling is corrected, alpha, beta, and activity
thresholds must be retuned. Do not reuse legacy optimal values as if unchanged.

## Spatial activity model

For device/event distance `d_iv`,

`P_iv = (exp(-d_iv/kappa) - exp(-D/kappa)) / (1 - exp(-D/kappa))`

for `d_iv <= D`, and exactly zero for `d_iv > D`.

For multiple events use

`P_i = 1 - product_v(1 - P_iv)`.

Current intended event influence radius is `D = 15 m`. The main experiment uses
variable realized activity. Do not force every realization to have exactly
`S=10`; instead calibrate `kappa` so that `E[S] ~= 10`. The configured
`kappa=3.785` is provisional until Task 01 calibration is inspected.

Task 02 must report realized-S mean, standard deviation, minimum, maximum, and
distribution/histogram.

## Baselines

SBL uses the same optimized posterior engine as CA-SBL where possible and an
independently tuned threshold.

MMV-OMP/SOMP is a joint-row-support oracle sparsity-budget baseline. MMV-CoSaMP is also
joint-support, uses stable unregularized least squares, and has no hidden ridge
penalty. For realization `r`, OMP and CoSaMP use the feasible oracle sparsity budget
`K_r = min(S_r, L, N)`. Record whether capping occurs. CA-SBL and SBL never
receive `S_r` or `K_r`.

## Activity figures

Selected-realization figures must be reproducible from saved Task-02 data and
may show the BS, cell boundary, inactive/active MTDs, events, optional cutoff
circles, activation-probability field, and a colorbar. Save PNG previews and
vector PDF publication output.

The displayed field must be the actual model probability

`P(x) = 1 - product_v(1 - P_v(x))`.

Never alter the probability values for visual purposes. Specifically do not:

- threshold low probabilities;
- replace zeros;
- add `1e-6` or machine epsilon;
- mask values merely because `P=0`;
- artificially whiten regions outside event influence;
- modify `probability_field` before plotting.

Masking only the region outside the physical circular BS cell is allowed.
The colormap itself maps `P=0` to pure white and increasing probability toward
dark red. Publication PDFs use vector `contourf`, with contour levels spanning
`np.linspace(0.0, 1.0, probability_levels)`.

## Results and tests

Prefer compressed NPZ for numerical arrays, CSV for tabular results, JSON/YAML
for summaries/manifests, PDF for publication figures, and PNG for previews. Do
not save enormous per-iteration histories for every realization.

Important tests include matrix dimensions, system-model identity, complex
Gaussian generation, optimized posterior vs a full reference, conventional SBL,
stable CA-SBL root, corrected `phi`, ANC/correlation construction, activity
cutoff, multi-event probability, deterministic seeds, simple noiseless baseline
support recovery, and plotting that does not modify probability values.

Do not weaken tests merely to make an implementation pass.
