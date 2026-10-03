# Figures

Standard figures are generated automatically by the numbered scientific tasks and
saved under the configured results directory. PNG previews and vector PDFs are
created by default.

Separate plotting scripts are regeneration utilities only:

```bash
python scripts/make_figures.py --config configs/paper.yaml --format both
python scripts/plot_selected_realization.py --config configs/paper.yaml --split evaluation --sample 5 --format both
python scripts/plot_activity_s_distribution.py --config configs/paper.yaml --format both
```

These utilities use already-saved numerical outputs and do not rerun the expensive
experiment stages.
