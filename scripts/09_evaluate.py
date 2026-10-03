#!/usr/bin/env python
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.pipeline import aggregate_stage
from casbl_jadce.plotting import make_performance_figures
p = argparse.ArgumentParser(description="Task 09: aggregate, analyze, and plot estimator results.")
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--figures", choices=["none", "png", "pdf", "both"], default="both")
p.add_argument("--no-progress", action="store_true")
a = p.parse_args(); cfg = load_config(a.config); aggregate_stage(cfg)
if a.figures != "none":
    paths = make_performance_figures(cfg, a.figures, show_progress=not a.no_progress); print("Performance figure(s):"); [print(f"  - {x}") for x in paths]
