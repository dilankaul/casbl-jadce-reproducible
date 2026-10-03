#!/usr/bin/env python
"""Plot the realized active-MTD-count distribution from saved Task-02 data."""
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.plotting import plot_activity_s_distribution

p = argparse.ArgumentParser(description="Plot Task-02 realized S distribution without rerunning Task 02.")
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--format", choices=["png", "pdf", "both"], default="both")
a = p.parse_args(); cfg = load_config(a.config)
paths = plot_activity_s_distribution(cfg, formats=a.format)
print("Activity-count distribution figure(s):")
for path in paths:
    print(f"  - {path}")
