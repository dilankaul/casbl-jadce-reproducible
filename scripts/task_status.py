#!/usr/bin/env python
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.reporting import report_task

p = argparse.ArgumentParser(description="Inspect one completed scientific task (01-09) without rerunning it.")
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--task", type=int, required=True, help="Completed task number to inspect (01-09).")
a = p.parse_args(); cfg = load_config(a.config)
report_task(cfg["run"]["output_dir"], a.task)
