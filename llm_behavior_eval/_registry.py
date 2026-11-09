"""Explicit task imports for Inspect's package entry point."""

from .bbq.task import bbq
from .inspect_template.task import inspect_template

__all__ = ["bbq", "inspect_template"]
