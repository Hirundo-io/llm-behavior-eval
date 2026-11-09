"""Expose repository examples without shadowing installed package metadata."""

import sys
from pathlib import Path

# Console pytest with importlib mode does not add the repository to sys.path.
# Append it so examples/scripts remain importable, but the installed dist-info
# takes precedence over setuptools' source-tree egg-info (which has no PEP 610
# editable-install metadata). Inspect uses that metadata to namespace tasks.
sys.path.append(str(Path(__file__).resolve().parents[1]))
