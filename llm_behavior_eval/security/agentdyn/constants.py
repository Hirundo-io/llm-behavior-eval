"""AgentDyn's pinned revision and the task's choices.

Kept apart from the modules that import ``agentdojo`` (slow to import), so
loading the task's entry point stays cheap.
"""

from typing import Final

AGENTDYN_REPOSITORY: Final = "SaFo-Lab/AgentDyn"
AGENTDYN_COMMIT: Final = "5353cf7615b135cace8d07c8f12dac53a16b6db3"
"""The commit of the ``agentdojo`` fork this package is written against."""
BENCHMARK_VERSION: Final = "v1.2.2"
"""Upstream's default ``--benchmark-version``; AgentDyn's suites are the same
objects in every version."""
SUITES: Final = ("shopping", "github", "dailylife")
"""AgentDyn's suites: 20 user tasks each; 9, 9 and 10 injection tasks."""
MODES: Final = ("clean", "attack", "both")
DEFAULT_ATTACK: Final = "important_instructions"
"""The attack of AgentDyn's README and paper."""
MODEL_NAME_ATTACKS: Final = (
    "important_instructions",
    "important_instructions_no_user_name",
    "important_instructions_wrong_user_name",
)
"""Attacks whose template addresses the model under test by name (``{model}``)."""
FIXED_ATTACKS: Final = (
    "important_instructions_no_model_name",
    "important_instructions_no_names",
    "direct",
    "ignore_previous",
    "system_message",
    "injecagent",
)
"""Attacks whose text does not depend on the model under test."""
SUPPORTED_ATTACKS: Final = (*MODEL_NAME_ATTACKS, *FIXED_ATTACKS)
"""Upstream's registered attacks this task runs. Left out: ``manual`` (asks a
person for each injection), ``important_instructions_wrong_model_name`` (a
wrong name drawn from upstream's closed model registry), ``tool_knowledge``
(AgentDyn's injection tasks have no placeholder arguments, so upstream's
attack raises) and the denial-of-service attacks (one injection task each,
scored differently)."""
DEFAULT_MODEL_NAME: Final = "the AI language model"
"""Upstream's ``agentdojo.attacks.base_attacks.DEFAULT_MODEL_NAME``."""
