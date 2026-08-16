#!/usr/bin/env python
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.pipeline import manifest_stage, generate_activity_stage

p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/quick.yaml")
a = p.parse_args(); cfg = load_config(a.config); manifest_stage(cfg); generate_activity_stage(cfg)
