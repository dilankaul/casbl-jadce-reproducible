# CA-SBL for JADCE in MTC — Reproducible Implementation

Codespaces-ready implementation for the paper **Correlation-Aware SBL for Device Detection and Channel Estimation in MTC**.

The scientific workflow contains **nine uniquely numbered tasks (01–09)**. Standard figures are saved automatically by the numbered task that produces the corresponding data. Numbered tasks save PNG previews for the repository; the paper activity script saves vector PDF.

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

Use `configs/experiment.yaml` for the experiment workflow. The activity model uses `D=15 m`; Task 01 automatically selects kappa before Task 02. `system.S` is the **target mean activity level**, not a hard per-realization sparsity constraint.

## Numbered scientific workflow

Each numbered task automatically saves its standard figures as PNG. Paper activity figures are exported separately as vector PDF.

### Task 01 — Calibrate kappa

```bash
python scripts/01_calibrate_kappa.py
```

This saves the calibration CSV/JSON and the \(\kappa\)-calibration figure. It does not overwrite Task-02 activity datasets.

### Task 02 — Generate activity realizations

```bash
python scripts/02_generate_activity.py \
  --config configs/experiment.yaml \
  --sample 0
```

The event process is unconditioned, so the realized activity count \(S_r\) varies. This task saves tuning/evaluation activity datasets, `realized_s.csv`, summary statistics, every tuning/evaluation activity realization with the activation-probability field, and the realized-\(S\) distribution.

### Task 03 — Generate the communication-model preview

```bash
python scripts/03_generate_communication.py --config configs/experiment.yaml
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
python scripts/04_build_correlation.py --config configs/experiment.yaml
```

The correlation summary and \(\mathbf C\) visualization are saved automatically.

### Task 05 — Tune \(\alpha,\beta\)

```bash
python scripts/05_tune_alpha_beta.py
```

Task 05 uses an internally optimized provisional threshold only for ranking the \((\alpha,\beta)\) grid. The selected grid result and tuning figure are saved automatically. Final CA-SBL and SBL activity thresholds are independently selected in Task 06.

### Task 06 — Tune activity thresholds

```bash
python scripts/06_tune_thresholds.py --config configs/experiment.yaml
```

This saves `casbl_thresholds.csv`, `sbl_thresholds.csv`, `selected.json`, and the CA-SBL/SBL threshold-tuning figures.

### Task 07 — Run all estimators

```bash
python scripts/07_run_estimators.py
```

This runs CA-SBL-ANC, conventional SBL, MMV-OMP/SOMP, and MMV-CoSaMP on identical communication realizations and automatically saves the gamma-distribution diagnostics.

For each realization, the greedy baselines use

\[
K_r=\min(S_r,L,N),
\]

where \(S_r=\sum_i a_i\). Thus `K_r == S_r` whenever feasible. If `S_r > L`, the greedy support budget is capped at `L`. Raw baseline rows record `realized_S`, `oracle_K`, and `oracle_capped`; `run_summary.json` reports how often capping occurs. CA-SBL and SBL are **not** given \(S_r\) or \(K_r\).

### Task 08 — Run convergence analysis

```bash
python scripts/08_run_convergence.py --config configs/experiment.yaml
```

The convergence data and convergence figure are saved automatically.

### Task 09 — Aggregate, evaluate, and generate performance figures

```bash
python scripts/09_evaluate.py --config configs/experiment.yaml
```

This task does not rerun estimators. It aggregates the raw Task-07 results and automatically saves the standard F1, NMSE, and runtime performance figures.

## Automatic task figures

Run the numbered tasks normally; each saves its PNG figures automatically.
All task scripts default to `configs/experiment.yaml`. The only CLI options
are `--config` and `--no-progress` (plus standard `--help`); the paper export
additionally requires `--split` and `--index`. All scientific
settings come from the experiment YAML. Run tasks in order for a
new experiment; Task 02 reads Task 01's calibrated kappa. For title-free paper
activity PDFs, use `scripts/paper_figures/activity_realization.py` after Task 02.

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

Paper figures use PDF with Matplotlib's native renderer; no LaTeX
installation is required. Serif fonts are selected in order: Computer Modern
Roman, CMU Serif, STIXGeneral, DejaVu Serif. Math uses Computer Modern.
PDF output embeds TrueType fonts.

```bash
python scripts/paper_figures/activity_realization.py --split tuning --index 0
```

### Figure measurements

Each plotting function in `src/casbl_jadce/figures/` defines its own `figsize`
and `figure_style` dictionary. Edit those values to change title, axis, tick,
legend, or colorbar font sizes for that figure. Activity realization labels remain
15 pt and ticks/legends 13 pt. Experiment YAML does not contain font overrides.

### Experiment outputs and all activity realizations

`configs/experiment.yaml` is the full experiment configuration and writes to
`results/experiment/`. Saved numerical results, summaries, and PNG/PDF figures
are eligible for version control. Previously saved config/manifest files retain
their original paths as historical run provenance.

Task 02 automatically exports every tuning and evaluation activity realization
in PNG, with the split, zero-based realization number, and realized active
MTD count in its title. The full configuration produces 100 tuning and 1,000
evaluation spatial PNG figures. Batch export shows progress.

```bash
PYTHONPATH=src python scripts/02_generate_activity.py --config configs/experiment.yaml
```

`plot_activity_sample()` controls dimensions and font sizes for all spatial realizations.
Figures are saved as part of the normal task run.

### Task output folders

Each task owns its data and figures under `results/experiment/`:

```text
01_kappa_calibration/       # calibration CSV and summary; figures/
02_activity_realizations/      # summary and realized_s.csv
  tuning/                     # tuning_activity.npz
  evaluation/                 # evaluation_activity.npz
  figures/
    tuning/                   # one PDF per tuning realization
    evaluation/               # one PDF per evaluation realization
                              # distribution PDF is directly in figures/
03_communication_preview/      # preview and summaries; figures/
04_spatial_correlation/        # correlation data and summary; figures/
05_alpha_beta_tuning/          # grid and selection; figures/tuning/
06_threshold_tuning/           # thresholds and selection; figures/tuning/
07_estimators/                 # per-sample results and gamma; figures/evaluation/
08_convergence/                # traces and summary; figures/evaluation/
09_evaluation/                 # aggregated results and summary; figures/evaluation/
```

Run-level `config.yaml` and `manifest.json` remain at the results root. Task status
and regeneration utilities read this layout. Existing saved outputs have been
moved without changing their numerical data or historical run metadata.

### Automatic calibrated kappa

The experiment configuration sets `activity.kappa_source: calibration`. Run Task 01,
then Task 02; Task 02 reads `closest_grid_kappa` from `01_kappa_calibration/summary.json`.
No manual YAML update is needed. Missing calibration or mismatched N, R, V, D,
target activity, or calibration seed stops generation and asks you to rerun Task 01.
Task 02 prints its selected kappa and records its source in the activity summary;
the saved run config contains the effective value. Input YAML is unchanged.
The experiment config uses automatic calibration and omits a fixed `activity.kappa`.
Run Task 01 before Task 02. Explicit manual mode remains available
by setting `kappa_source: configured` and adding `activity.kappa`.

### Coarse-to-fine kappa calibration

Task 01 scans ascending `activity.kappa_scan` values, finds a bracket around the
requested mean activity, then refines using `kappa_refinement_steps` (defaults:
0.1, 0.02, 0.005). The selected value comes from the final aligned grid. Every
stage replays the same seed, sample count, and batch size to compare identical
geometries and Bernoulli uniforms. Set the coarse candidates in the experiment YAML.

`kappa_sweep.csv` records all unique candidates, their first evaluation stage,
and expected/realized activity statistics. The summary records brackets,
selection, final spacing, and validation on an independent deterministic seed
(`seeds.kappa_validation`).
Validation reports expected-mean standard error, an approximate 95% interval,
and whether the mean is within `calibration_tolerance` (default 0.1 device).
Validation does not retune the selected value. A tolerance miss is reported;
it does not silently redefine the target. Calibration and validation remain
separate from Task 02 datasets. A missing bracket saves a failed summary and
stops; Task 02 rejects that summary instead of consuming an old selection.

The experiment requires its own Task-01 results. Preview utilities also use the saved calibrated value unless `--kappa` is supplied.

### Configuration provenance

`configs/experiment.yaml` is the input. Results contain only one run-level
`config.yaml`: Task 01 creates it when absent, and Task 02 updates it with the
activity-generation configuration. Later tasks leave this root snapshot unchanged.
Each task's `manifest.json` stores its exact effective configuration under
`configuration`, alongside runtime options and environment versions. This preserves
calibration overrides, selected kappa, thresholds, and seeds without extra YAML
files. Existing task YAML snapshots were migrated into JSON manifests before removal.
Source configuration is never rewritten.

### Code organization

The nine numbered scripts are thin, import-safe wrappers around
`task_cli.run_task(task_number)`. Shared CLI handling loads the experiment
configuration, executes one stage, and saves PNG figures. The paper activity
script reads saved Task 02 data and exports separate PDFs.

```text
src/casbl_jadce/
├── models/          # Activity, communication, realizations, correlation
├── algorithms/      # SBL, CA-SBL, OMP, CoSaMP, posterior engine
├── experiments/     # Calibration, threshold tuning, nine-stage pipeline
├── figures/         # Descriptive task-numbered plots and shared rendering
├── task_cli.py      # Registry and common --config / --no-progress arguments
├── config.py
├── dataset.py
├── io.py
├── metrics.py
├── paths.py
├── provenance.py
└── reporting.py
```

Figure filenames identify the producing task and the plotted quantity:

```text
figures/
├── common.py
├── task01_kappa_calibration.py
├── task02_activity_realizations.py
├── task02_active_mtd_count_distribution.py
├── task03_communication_preview.py
├── task04_spatial_correlation.py
├── task05_alpha_beta_tuning.py
├── task06_activity_threshold_tuning.py
├── task07_gamma_distribution.py
├── task08_convergence.py
├── task09_performance.py
└── paper_activity_realizations.py
```

Shared typography and saving live in `figures/common.py`. Figure dimensions and
font sizes stay local to their plotting functions. Paper styling lives in
`paper_activity_realizations.py` and reuses Task 02's drawing function.
`experiments/pipeline.py` coordinates all nine scientific stages.

### Activity figure splits

Task 02 generates both activity datasets and exports both sets of PNG figures by
default. No separate figure selection switches are required.
Known fontTools timestamp messages are filtered during figure saving; other
warnings remain visible.

### Explicit random seeds

The `seeds` section names six independent streams: `kappa_calibration` selects
the decay parameter that gives the target mean active MTD count;
`kappa_validation` checks that selection on independent samples.
`activity_tuning` and `activity_evaluation` generate the two activity datasets.
`communication_tuning` and `communication_evaluation` generate their channels,
pilots, and noise. All six seeds are explicit and distinct; no seed offsets
are applied between these streams. Algorithms still share realizations within
each condition.

The six streams use distinct four-digit seeds listed in `configs/experiment.yaml`. These replace
the previous seed values, so regenerate results starting with Task 01.
Existing saved results retain their original recorded seeds.

### Paper activity figures

Task 02 continues saving all standard figures with titles and its default sizes.
Export selected saved realizations as separate, title-free publication PDFs:

```bash
python scripts/paper_figures/activity_realization.py --split tuning --index 0
```

This uses the existing experiment configuration and saved Task 02 data. No
separate paper YAML is needed. Edit `figure_style` and `figsize` inside
`plot_paper_activity_figure` in
`src/casbl_jadce/figures/paper_activity_realizations.py` to adjust font
sizes in points and dimensions in inches, just like the task plotting functions.
Each run exports exactly one saved realization. Choose `--split tuning` or
`--split evaluation` and a zero-based `--index`. PDFs are written to
`results/experiment/paper_figures/activity/`, with tuning/evaluation subfolders.
The script does not run scientific tasks or write configuration snapshots.
Both export paths reuse the same saved-model and probability-field settings.
Their font sizes remain separate local definitions. The experiment config omits
`max_attempts` because variable-activity sampling does not use rejection retries;
exact-sparsity experiments can still set it explicitly.


Paper activity PDFs use `event_driven_activation_model_[split]_[index].pdf` with four-digit index padding,
for example `event_driven_activation_model_evaluation_0005.pdf`.
