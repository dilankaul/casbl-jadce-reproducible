#!/usr/bin/env python
"""Threshold tuning is integrated into 04_tune_alpha_beta.py.

This compatibility stage prints the selected thresholds written by stage 04.
"""
import argparse, json
from pathlib import Path
from casbl_jadce.config import load_config
p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/quick.yaml")
a = p.parse_args(); cfg = load_config(a.config)
path = Path(cfg["run"]["output_dir"]) / "tuning" / "selected.json"
print(path.read_text(encoding="utf-8"))
