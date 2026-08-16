# CA-SBL for JADCE in MTC — Reproducible Implementation

Reproducible, Codespaces-ready implementation for the paper **Correlation-Aware SBL for Device Detection and Channel Estimation in MTC**.

The repository replaces the notebook-heavy legacy workflow with tested Python modules and deterministic experiment scripts while preserving the same research stages: activity modelling, communication simulation, correlation construction, hyperparameter tuning, thresholding, estimation, convergence analysis, evaluation, and figure generation.

## Mathematical change implemented

The code uses the proper **complex MMV** SBL objective. For each device row \(\mathbf z_i\in\mathbb C^M\),

\[
\mathbf z_i\mid\gamma_i\sim\mathcal{CN}(\mathbf0,\gamma_i\mathbf I_M),
\]

which gives an SBL term proportional to

\[
-M\sum_i\left(\log\gamma_i+\eta_i/\gamma_i\right),
\qquad
\eta_i=\Sigma_{ii}+\|\boldsymbol\mu_i\|_2^2/M.
\]

The ANC definition is unchanged:

\[
\mathbf\Omega=\alpha(\beta\mathbf1-\mathbf C).
\]

The corrected interaction is

\[
\phi_i=(\mathbf\Omega\boldsymbol\gamma)_i/M.
\]

For \(\phi_i>0\), CA-SBL uses the numerically stable update

\[
\gamma_i^{new}=
\frac{2\eta_i}{\sqrt{1+4\phi_i\eta_i}+1}.
\]

There is **no upper clipping** of \(\gamma_i\). See `docs/theory.md` for the derivation implemented by the code.

## Baselines

- **SBL**: complex MMV SBL using the same optimized posterior engine as CA-SBL.
- **MMV-OMP / SOMP**: joint row-support OMP using known sparsity \(S\).
- **MMV-CoSaMP**: joint-support CoSaMP using SVD least squares; no hidden ridge penalty.

OMP and CoSaMP are therefore oracle-sparsity baselines because they receive the true \(S\).

## Codespaces

Open the repository in GitHub Codespaces. The devcontainer installs the package automatically.

Run the tests first:

```bash
pytest
```

Run a small end-to-end verification:

```bash
python scripts/run_all.py --config configs/quick.yaml --workers 2
```

Run the full paper experiment:

```bash
python scripts/run_all.py --config configs/paper.yaml --workers 4
```

The best worker count depends on the Codespaces machine. BLAS threads are fixed to one per worker to avoid CPU oversubscription.

## Pipeline

```text
01_generate_activity.py
02_generate_communication.py
03_build_correlation.py
04_tune_alpha_beta.py
05_tune_thresholds.py
06_run_estimators.py
07_run_convergence.py
08_evaluate.py
09_make_figures.py
```

`run_all.py` runs these stages in dependency order.

### Why the communication stage is compact

Large \(Y\), \(W\), and pilot arrays are not saved for every Monte Carlo condition. They are generated deterministically from the recorded seeds. `02_generate_communication.py` saves a full numerical preview plus a manifest describing the seed mapping. This keeps paper results small enough to commit while preserving exact reproducibility.

## Result layout

A run such as `configs/paper.yaml` produces:

```text
results/paper/
├── config.yaml
├── manifest.json
├── activity/
│   ├── tuning_activity.npz
│   ├── evaluation_activity.npz
│   └── summary.json
├── communication/
│   ├── manifest.json
│   └── preview.npz
├── correlation/
│   ├── preview_C.npz
│   └── summary.json
├── tuning/
│   ├── alpha_beta.csv
│   └── selected.json
├── evaluation/
│   ├── per_sample.csv
│   ├── aggregate.csv
│   └── gamma_distribution.npz
├── convergence/
│   └── convergence.npz
└── figures/
    ├── f1_vs_snr.png
    ├── nmse_vs_snr.png
    ├── f1_vs_pilot_length.png
    ├── nmse_vs_pilot_length.png
    ├── runtime_vs_snr.png
    └── convergence_nmse.png
```

## Reproducibility decisions

- Tuning and final evaluation use separate activity seeds.
- For each evaluation sample, all algorithms receive the exact same \(\mathbf a,\mathbf H,\mathbf Z,\mathbf\Theta,\mathbf W,\mathbf Y\).
- \(\mathbf H\) is fixed across pilot lengths and SNR values for a sample.
- \(\mathbf\Theta\) is fixed across SNR values for a given sample and pilot length.
- Noise is deterministic for each sample/pilot-length/SNR condition.
- CA-SBL and SBL use `complex128`/`float64` in paper runs.
- Thresholds are selected only from the tuning set.
- CA-SBL \(\alpha,\beta\) are retuned because the corrected theory changes the interaction from the legacy factor \(2\) to \(1/M\).

## Configuration

`configs/paper.yaml` is the source of truth for experiment parameters. If a manuscript parameter changes, change it there before rerunning.
