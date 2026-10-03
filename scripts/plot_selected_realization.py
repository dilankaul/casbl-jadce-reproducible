#!/usr/bin/env python
"""Replot a saved Task 02 realization without regenerating the dataset."""
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.plotting import plot_activity_realization

p = argparse.ArgumentParser(description="Plot any selected saved spatial activity realization.")
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--sample", type=int, required=True)
p.add_argument("--split", choices=["tuning", "evaluation"], default="evaluation")
p.add_argument("--format", choices=["png", "pdf", "both"], default="both")
p.add_argument("--hide-event-radius", action="store_true", help="Do not draw the D cutoff radius around event locations.")
field = p.add_mutually_exclusive_group()
field.add_argument("--probability-field", dest="probability_field", action="store_true", help="Draw P(x)=1-prod_v(1-P_v(x)) as a red probability field.")
field.add_argument("--no-probability-field", dest="probability_field", action="store_false", help="Do not draw the probability field.")
p.set_defaults(probability_field=None)
a = p.parse_args(); cfg = load_config(a.config)
paths = plot_activity_realization(
    cfg, split=a.split, sample_index=a.sample, formats=a.format,
    show_event_radius=not a.hide_event_radius,
    show_probability_field=a.probability_field,
)
print("Selected-realization figure(s); Task 02 was not rerun:")
for path in paths:
    print(f"  - {path}")
