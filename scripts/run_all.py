#!/usr/bin/env python
import argparse
from casbl_jadce.pipeline import run_all
from casbl_jadce.config import load_config
from casbl_jadce.plotting import make_figures
p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/quick.yaml"); p.add_argument("--workers", type=int, default=1)
a = p.parse_args(); run_all(a.config, workers=a.workers); cfg = load_config(a.config); make_figures(cfg["run"]["output_dir"])
