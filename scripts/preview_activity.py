#!/usr/bin/env python
"""Preview one deterministic activity realization without running Task 02."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from casbl_jadce.activity_model import generate_activity_sample
from casbl_jadce.config import load_config
from casbl_jadce.plotting import plot_activity_sample

p = argparse.ArgumentParser(description="Preview one event-driven activity realization without generating a full dataset.")
p.add_argument("--config", default="configs/paper.yaml")
p.add_argument("--sample", type=int, default=0, help="Deterministic tuning-realization index.")
p.add_argument("--D", type=float, default=None, help="Override D for this preview only.")
p.add_argument("--kappa", type=float, default=None, help="Override kappa for this preview only.")
p.add_argument("--format", choices=["png", "pdf", "both"], default="both")
p.add_argument("--hide-event-radius", action="store_true")
field = p.add_mutually_exclusive_group()
field.add_argument("--probability-field", dest="probability_field", action="store_true", help="Draw the combined activation-probability field.")
field.add_argument("--no-probability-field", dest="probability_field", action="store_false", help="Draw devices/events only, without the probability field.")
p.set_defaults(probability_field=None)
a = p.parse_args()

cfg = load_config(a.config)
sys = cfg["system"]; act = cfg["activity"]; run = cfg["run"]; fig_cfg = cfg.get("figures", {})
D = float(act["D"] if a.D is None else a.D)
kappa = float(act["kappa"] if a.kappa is None else a.kappa)
show_probability_field = bool(fig_cfg.get("show_probability_field", False)) if a.probability_field is None else bool(a.probability_field)
if a.sample < 0:
    raise ValueError("--sample must be non-negative.")

# Use the exact same child-seed convention as the tuning portion of Task 02.
child = np.random.SeedSequence(int(run["activity_seed"])).spawn(a.sample + 1)[a.sample]
sample = generate_activity_sample(
    np.random.default_rng(child),
    N=int(sys["N"]), V=int(act["V"]), R=float(sys["R"]),
    kappa=kappa, D=D, S=int(sys["S"]), exact_sparsity=False,
    batch_size=int(act.get("batch_size", 128)),
)

out = Path(run["output_dir"])
suffix = "_probability_field" if show_probability_field else "_plain"
stem = out / "figures" / "activity_previews" / f"preview_D{D:g}_kappa{kappa:g}_sample{a.sample:04d}{suffix}"
paths = plot_activity_sample(
    sample,
    R=float(sys["R"]), D=D, kappa=kappa, stem=stem, formats=a.format,
    show_event_radius=not a.hide_event_radius,
    show_probability_field=show_probability_field,
    probability_resolution=int(fig_cfg.get("probability_resolution", 301)),
    probability_alpha=float(fig_cfg.get("probability_alpha", 0.55)),
    probability_levels=int(fig_cfg.get("probability_levels", 128)),
    title=f"Activity preview #{a.sample}: {int(np.sum(sample.a))} active MTDs",
)
print(f"Preview parameters: D={D:g} m, kappa={kappa:g}, active={int(np.sum(sample.a))}, expected-active-sum={float(np.sum(sample.activation_probabilities)):.4f}")
print(f"Probability field: {'on' if show_probability_field else 'off'}")
print("Generated without running Task 02:")
for path in paths:
    print(f"  - {path}")
