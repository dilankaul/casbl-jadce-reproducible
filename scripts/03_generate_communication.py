#!/usr/bin/env python
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.pipeline import communication_stage
from casbl_jadce.plotting import plot_communication_preview
p = argparse.ArgumentParser(description="Task 03: generate deterministic communication preview.")
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--figures", choices=["none", "png", "pdf", "both"], default="both")
a = p.parse_args(); cfg = load_config(a.config); communication_stage(cfg)
if a.figures != "none":
    paths = plot_communication_preview(cfg, a.figures); print("Communication figure(s):"); [print(f"  - {x}") for x in paths]
