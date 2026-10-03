#!/usr/bin/env python
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.pipeline import correlation_stage
from casbl_jadce.plotting import plot_correlation_preview
p = argparse.ArgumentParser(description="Task 04: build and inspect spatial correlation C.")
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--figures", choices=["none", "png", "pdf", "both"], default="both")
a = p.parse_args(); cfg = load_config(a.config); correlation_stage(cfg)
if a.figures != "none":
    paths = plot_correlation_preview(cfg, a.figures); print("Correlation figure(s):"); [print(f"  - {x}") for x in paths]
