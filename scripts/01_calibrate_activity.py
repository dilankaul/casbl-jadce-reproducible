#!/usr/bin/env python
"""Calibrate kappa for a target *average* number of active MTDs.

This calibration is intentionally separate from Task 02. It does not create or overwrite
Task-02 tuning/evaluation datasets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from casbl_jadce.activity_model import best_kappa_for_target, calibrate_kappa_grid
from casbl_jadce.config import load_config
from casbl_jadce.io import ensure_dir, save_csv, save_json
from casbl_jadce.plotting import plot_activity_calibration
from casbl_jadce.reporting import print_task_report


def parse_kappas(text: str | None, fallback: list[float]) -> list[float]:
    if text is None:
        return [float(x) for x in fallback]
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("--kappas did not contain any values.")
    return values


p = argparse.ArgumentParser(description="Task 01: estimate mean active MTD count versus kappa without running Task 02.")
p.add_argument("--config", default="configs/paper.yaml")
p.add_argument("--D", type=float, default=None, help="Override event influence radius D for this calibration only.")
p.add_argument("--kappas", default=None, help='Comma-separated kappa grid, e.g. "3.6,3.7,3.8,3.9,4.0".')
p.add_argument("--samples", type=int, default=None, help="Number of Monte-Carlo geometries. Defaults to activity.calibration_samples.")
p.add_argument("--target", type=float, default=None, help="Target mean active count. Defaults to system.S.")
p.add_argument("--figures", choices=["none", "png", "pdf", "both"], default="both")
p.add_argument("--no-progress", action="store_true")
a = p.parse_args()

cfg = load_config(a.config)
sys = cfg["system"]; act = cfg["activity"]; run = cfg["run"]
D = float(act["D"] if a.D is None else a.D)
kappas = parse_kappas(a.kappas, act.get("kappa_scan", [act["kappa"]]))
num_samples = int(act.get("calibration_samples", 2000) if a.samples is None else a.samples)
target = float(sys["S"] if a.target is None else a.target)
seed = int(run["activity_seed"]) + int(act.get("calibration_seed_offset", 90000))

rows = calibrate_kappa_grid(
    seed=seed,
    num_samples=num_samples,
    N=int(sys["N"]),
    V=int(act["V"]),
    R=float(sys["R"]),
    D=D,
    kappa_values=kappas,
    batch_size=int(act.get("batch_size", 128)),
    show_progress=not a.no_progress,
)
best = best_kappa_for_target(rows, target)
out = Path(run["output_dir"])
caldir = ensure_dir(out / "activity_calibration")
csv_path = caldir / "kappa_sweep.csv"
summary_path = caldir / "summary.json"
save_csv(csv_path, [row.__dict__ for row in rows])
save_json(summary_path, {
    "N": int(sys["N"]),
    "V": int(act["V"]),
    "R": float(sys["R"]),
    "D": D,
    "target_mean_active": target,
    "calibration_samples": num_samples,
    "seed": seed,
    "closest_grid_kappa": best.kappa,
    "closest_grid_mean_expected_active": best.mean_expected_active,
    "closest_grid_mean_realized_active": best.mean_realized_active,
    "absolute_expected_gap": abs(best.mean_expected_active - target),
    "configured_kappa": float(act["kappa"]),
})

# Use the override D for the plot title without changing the loaded file on disk.
cfg_for_plot = dict(cfg)
cfg_for_plot["activity"] = dict(cfg["activity"])
cfg_for_plot["activity"]["D"] = D
fig_paths = [] if a.figures == "none" else plot_activity_calibration(cfg_for_plot, formats=a.figures)

print_task_report("TASK 01 — ACTIVITY KAPPA CALIBRATION", {
    "D (m)": D,
    "target mean active S": target,
    "calibration realizations": num_samples,
    "closest grid kappa": best.kappa,
    "mean expected active": best.mean_expected_active,
    "mean realized active": best.mean_realized_active,
    "expected gap from target": best.mean_expected_active - target,
    "configured kappa": float(act["kappa"]),
}, [csv_path, summary_path, *fig_paths])
print("This command did NOT rerun or overwrite Task 02.")
