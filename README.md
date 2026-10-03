# CA-SBL for JADCE in MTC — Reproducible Implementation

Codespaces-ready implementation for the paper **Correlation-Aware SBL for Device Detection and Channel Estimation in MTC**.

The scientific workflow contains **nine uniquely numbered tasks (01–09)**. Utility scripts are deliberately unnumbered because they do not represent additional experimental stages. Standard figures are saved automatically by the numbered task that produces the corresponding data. Separate plotting utilities are retained only for regenerating figures from saved outputs without rerunning expensive computations.

`AGENTS.md` contains persistent instructions for Codex, while `DECISIONS.md` records locked scientific and implementation decisions.

## Implemented CA-SBL theory

For the proper complex MMV model,

\[
\mathbb E[\log p(\mathbf Z\mid\boldsymbol\gamma)]
\propto
-M\sum_i\left(\log\gamma_i+\eta_i/\gamma_i\right),
\qquad
\eta_i=\Sigma_{ii}+\|\boldsymbol\mu_i\|_2^2/M.
\]

The ANC construction remains

\[
\mathbf\Omega=\alpha(\beta\mathbf1-\mathbf C),
\]

and the corrected interaction is

\[
\phi_i=(\mathbf\Omega\boldsymbol\gamma)_i/M.
\]

For \(\phi_i>0\), the implementation uses

\[
\gamma_i^{\mathrm{new}}
=
\frac{2\eta_i}{\sqrt{1+4\phi_i\eta_i}+1}.
\]

For \(\phi_i\le0\), the current implementation uses the conventional SBL fallback \(\gamma_i=\eta_i\). There is no upper clipping of \(\gamma_i\).

## Setup

Verify the environment with:

```bash
PYTHONPATH=src pytest -q
```

Use `configs/quick.yaml` for smoke tests and `configs/paper.yaml` for paper-scale runs. The current activity model uses `D=15 m`; `kappa=3.785` is an initial calibrated candidate and should be confirmed with Task 01. `system.S` is the **target mean activity level**, not a hard per-realization sparsity constraint.

## Numbered scientific workflow

By default, each numbered task saves its standard figures as both PNG and vector PDF. Use `--figures none` only when figures are deliberately not wanted.

### Task 01 — Calibrate the activity model

```bash
python scripts/01_calibrate_activity.py \
  --config configs/paper.yaml \
  --D 15 \
  --samples 5000
```

This saves the calibration CSV/JSON and the \(\kappa\)-calibration figure. It does not overwrite Task-02 activity datasets.

### Task 02 — Generate activity realizations

```bash
python scripts/02_generate_activity.py \
  --config configs/quick.yaml \
  --sample 0
```

The event process is unconditioned, so the realized activity count \(S_r\) varies. This task saves tuning/evaluation activity datasets, `realized_s.csv`, summary statistics, a selected activity realization with the activation-probability field, and the realized-\(S\) distribution.

### Task 03 — Generate the communication-model preview

```bash
python scripts/03_generate_communication.py --config configs/quick.yaml
```

The model uses

\[
\mathbf Z=\operatorname{diag}(\mathbf a)\mathbf H^T,
\qquad
\mathbf Y=\mathbf\Theta\mathbf Z+\mathbf W,
\]

with unit-norm QPSK pilots and deterministic condition-specific random streams. The pilot cross-correlation figure is saved automatically.

### Task 04 — Build the spatial correlation matrix

```bash
python scripts/04_build_correlation.py --config configs/quick.yaml
```

The correlation summary and \(\mathbf C\) visualization are saved automatically.

### Task 05 — Tune \(\alpha,\beta\)

```bash
python scripts/05_tune_alpha_beta.py \
  --config configs/quick.yaml \
  --workers 2
```

Task 05 uses an internally optimized provisional threshold only for ranking the \((\alpha,\beta)\) grid. The selected grid result and tuning figure are saved automatically. Final CA-SBL and SBL activity thresholds are independently selected in Task 06.

### Task 06 — Tune activity thresholds

```bash
python scripts/06_tune_thresholds.py --config configs/quick.yaml
```

This saves `casbl_thresholds.csv`, `sbl_thresholds.csv`, `selected.json`, and the CA-SBL/SBL threshold-tuning figures.

### Task 07 — Run all estimators

```bash
python scripts/07_run_estimators.py \
  --config configs/quick.yaml \
  --workers 2
```

This runs CA-SBL-ANC, conventional SBL, MMV-OMP/SOMP, and MMV-CoSaMP on identical communication realizations and automatically saves the gamma-distribution diagnostics.

For each realization, the greedy baselines use

\[
K_r=\min(S_r,L,N),
\]

where \(S_r=\sum_i a_i\). Thus `K_r == S_r` whenever feasible. If `S_r > L`, the greedy support budget is capped at `L`. Raw baseline rows record `realized_S`, `oracle_K`, and `oracle_capped`; `run_summary.json` reports how often capping occurs. CA-SBL and SBL are **not** given \(S_r\) or \(K_r\).

### Task 08 — Run convergence analysis

```bash
python scripts/08_run_convergence.py --config configs/quick.yaml
```

The convergence data and convergence figure are saved automatically.

### Task 09 — Aggregate, evaluate, and generate performance figures

```bash
python scripts/09_evaluate.py --config configs/quick.yaml
```

This task does not rerun estimators. It aggregates the raw Task-07 results and automatically saves the standard F1, NMSE, and runtime performance figures.

## Unnumbered utilities

These scripts are utilities, not scientific tasks:

```text
preview_activity.py
plot_selected_realization.py
plot_activity_s_distribution.py
make_figures.py
task_status.py
run_all.py
```

Preview an activity realization before Task 02:

```bash
python scripts/preview_activity.py \
  --config configs/paper.yaml \
  --D 15 \
  --kappa 3.785 \
  --sample 10 \
  --probability-field
```

Regenerate a selected saved Task-02 realization without rerunning Task 02:

```bash
python scripts/plot_selected_realization.py \
  --config configs/quick.yaml \
  --split evaluation \
  --sample 5 \
  --probability-field
```

Regenerate the realized-\(S\) distribution:

```bash
python scripts/plot_activity_s_distribution.py --config configs/quick.yaml
```

Regenerate all figures that can be reconstructed from saved outputs:

```bash
python scripts/make_figures.py --config configs/quick.yaml
```

Inspect a completed scientific task:

```bash
python scripts/task_status.py --config configs/quick.yaml --task 7
```

Run the core experiment pipeline (Tasks 02–09) and then regenerate all available figures:

```bash
python scripts/run_all.py \
  --config configs/paper.yaml \
  --workers 4
```

Task 01 remains a calibration/verification step because it does not silently rewrite the configured \(\kappa\).

## Activity-probability figure

The displayed field is the actual event-model probability

\[
P(\mathbf x)=1-\prod_v\left[1-P_v(\mathbf x)\right].
\]

The probability data are never thresholded, epsilon-shifted, or modified to make the figure white. The only mask is outside the physical circular BS cell. A custom colormap maps `P=0` to pure white and increasing probability toward dark red. Publication PDFs use vector `contourf` with levels spanning exactly `[0,1]`.

## Baselines

- **SBL:** proper complex-MMV SBL using the optimized posterior engine.
- **MMV-OMP / SOMP:** joint row-support greedy baseline using oracle budget \(K_r\).
- **MMV-CoSaMP:** joint row-support CoSaMP using unregularized SVD least squares and the same oracle budget \(K_r\).

The greedy methods should be described in the paper as **oracle sparsity-budget baselines**, not as methods always receiving the uncapped realized sparsity.

## Reproducibility

- Tuning and evaluation activity streams are deterministic and separate.
- Activity and communication random streams are separate.
- All algorithms receive identical \(\mathbf a,\mathbf H,\mathbf Z,\mathbf\Theta,\mathbf W,\mathbf Y\) for a given condition.
- \(\mathbf H\) is fixed across pilot lengths and SNRs for a realization.
- \(\mathbf\Theta\) is fixed across SNR values for a given realization and pilot length.
- Thresholds are selected only from tuning data.
- CA-SBL \(\alpha,\beta\) must be retuned under the corrected \(1/M\) interaction scaling.
- Large communication tensors are deterministically regenerated rather than stored for every Monte-Carlo condition.

## Validation

Run:

```bash
PYTHONPATH=src pytest -q
```

See `VALIDATION.md` for the validation scope.

### Native PDF fonts

Figures default to PNG + PDF using Matplotlib's native renderer; no LaTeX
installation is required. Serif fonts are selected in order: Computer Modern
Roman, CMU Serif, STIXGeneral, DejaVu Serif. Math uses Computer Modern.
PDF output embeds TrueType fonts.

```bash
PYTHONPATH=src python scripts/make_figures.py --config configs/paper.yaml --format both
```

### Per-figure font sizes

Edit `figures.styles` in the YAML config. Each key is the output filename without
its extension. Unspecified settings keep the shared defaults; `figures.style`
can optionally supply overrides shared by every figure.

```yaml
figures:
  styles:
    02_activity_realization_tuning_0000_probability_field:
      axes.titlesize: 12
      axes.labelsize: 11
      xtick.labelsize: 9
      ytick.labelsize: 9
      legend.fontsize: 9
      colorbar.labelsize: 11
      colorbar.ticksize: 9
    08_convergence_nmse:
      axes.labelsize: 14
      legend.fontsize: 11
```

`font.size` changes all text on that figure before the more specific overrides.
Regenerate figures from saved data after changing sizes. PNG and PDF use the
same overrides, and other figures retain their own settings.
