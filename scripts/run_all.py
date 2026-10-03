#!/usr/bin/env python
import argparse
from casbl_jadce.pipeline import run_all
from casbl_jadce.config import load_config
from casbl_jadce.plotting import make_figures
p = argparse.ArgumentParser(description="Run the core experiment pipeline (Tasks 02-09); figures are saved automatically. Task 01 remains a separate calibration check.")
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--workers", type=int, default=1)
p.add_argument("--figures", choices=["none", "png", "pdf", "both"], default="both")
p.add_argument("--no-progress", action="store_true")
a = p.parse_args(); run_all(a.config, workers=a.workers, show_progress=not a.no_progress)
if a.figures != "none":
    cfg = load_config(a.config); make_figures(cfg, formats=a.figures, show_progress=not a.no_progress)
