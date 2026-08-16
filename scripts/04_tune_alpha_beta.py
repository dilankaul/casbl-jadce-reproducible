#!/usr/bin/env python
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.pipeline import tune_stage
p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/quick.yaml"); p.add_argument("--workers", type=int, default=1)
a = p.parse_args(); tune_stage(load_config(a.config), workers=a.workers)
