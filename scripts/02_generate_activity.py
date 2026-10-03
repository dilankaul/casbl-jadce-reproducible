#!/usr/bin/env python
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.pipeline import manifest_stage, generate_activity_stage
from casbl_jadce.plotting import plot_activity_realization, plot_activity_s_distribution

p = argparse.ArgumentParser(description="Task 02: generate spatial activity realizations.")
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--figures", choices=["none", "png", "pdf", "both"], default="both")
p.add_argument("--sample", type=int, default=0, help="Selected realization index for the automatically saved activity figure.")
p.add_argument("--split", choices=["tuning", "evaluation"], default="tuning")
p.add_argument("--no-progress", action="store_true")
field = p.add_mutually_exclusive_group()
field.add_argument("--probability-field", dest="probability_field", action="store_true")
field.add_argument("--no-probability-field", dest="probability_field", action="store_false")
p.set_defaults(probability_field=None)
a = p.parse_args(); cfg = load_config(a.config)
manifest_stage(cfg); generate_activity_stage(cfg, show_progress=not a.no_progress)
if a.figures != "none":
    paths = plot_activity_realization(
        cfg, split=a.split, sample_index=a.sample, formats=a.figures,
        show_event_radius=cfg.get("figures", {}).get("show_event_radius", True),
        show_probability_field=a.probability_field,
    )
    paths += plot_activity_s_distribution(cfg, formats=a.figures)
    print("Activity figure(s):"); [print(f"  - {x}") for x in paths]
