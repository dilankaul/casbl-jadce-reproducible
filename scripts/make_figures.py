#!/usr/bin/env python
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.plotting import make_figures
p = argparse.ArgumentParser(description="Regenerate all available diagnostic and paper figures from saved task outputs.")
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--format", choices=["png", "pdf", "both"], default="both")
p.add_argument("--sample", type=int, default=None, help="Selected tuning realization for the spatial figure.")
p.add_argument("--no-progress", action="store_true")
a = p.parse_args(); cfg = load_config(a.config); make_figures(cfg, formats=a.format, show_progress=not a.no_progress, selected_sample=a.sample)
