#!/usr/bin/env python
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.plotting import make_figures
p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/quick.yaml")
a = p.parse_args(); cfg = load_config(a.config); make_figures(cfg["run"]["output_dir"])
