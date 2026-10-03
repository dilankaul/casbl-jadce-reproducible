#!/usr/bin/env python
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.pipeline import convergence_stage
from casbl_jadce.plotting import plot_convergence
p = argparse.ArgumentParser(description="Task 08: run convergence trace at the reference condition.")
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--figures", choices=["none", "png", "pdf", "both"], default="both")
a = p.parse_args(); cfg = load_config(a.config); convergence_stage(cfg)
if a.figures != "none":
    paths = plot_convergence(cfg, a.figures); print("Convergence figure(s):"); [print(f"  - {x}") for x in paths]
