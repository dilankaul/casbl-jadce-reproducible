#!/usr/bin/env python
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.pipeline import tune_stage
from casbl_jadce.plotting import plot_alpha_beta_tuning
p = argparse.ArgumentParser(description="Task 05: tune CA-SBL alpha and beta.")
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--workers", type=int, default=1)
p.add_argument("--figures", choices=["none", "png", "pdf", "both"], default="both")
p.add_argument("--no-progress", action="store_true")
a = p.parse_args(); cfg = load_config(a.config); tune_stage(cfg, workers=a.workers, show_progress=not a.no_progress)
if a.figures != "none":
    paths = plot_alpha_beta_tuning(cfg, a.figures); print("Tuning figure(s):"); [print(f"  - {x}") for x in paths]
