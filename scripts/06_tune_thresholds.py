#!/usr/bin/env python
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.pipeline import threshold_stage
from casbl_jadce.plotting import plot_threshold_tuning
p = argparse.ArgumentParser(description="Task 06: tune CA-SBL and SBL activity thresholds.")
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--figures", choices=["none", "png", "pdf", "both"], default="both")
p.add_argument("--no-progress", action="store_true")
a = p.parse_args(); cfg = load_config(a.config); threshold_stage(cfg, show_progress=not a.no_progress)
if a.figures != "none":
    paths = plot_threshold_tuning(cfg, a.figures); print("Threshold figure(s):"); [print(f"  - {x}") for x in paths]
