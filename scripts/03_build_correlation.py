#!/usr/bin/env python
import argparse
from casbl_jadce.config import load_config
from casbl_jadce.pipeline import correlation_stage
p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/quick.yaml")
a = p.parse_args(); correlation_stage(load_config(a.config))
