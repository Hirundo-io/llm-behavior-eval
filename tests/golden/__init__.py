"""Frozen outputs from bae7a508649f48f2755ce831822b2fcb91d2ba71; no engine code."""

import json
from pathlib import Path

BASELINE = json.loads(Path(__file__).with_name("removal_baseline.json").read_text())
