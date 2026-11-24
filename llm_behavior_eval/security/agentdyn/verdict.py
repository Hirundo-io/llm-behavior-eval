"""An episode's verdicts, kept in the sample's store between solver and scorer."""

from dataclasses import asdict, dataclass

from inspect_ai.solver import TaskState

STORE_KEY = "agentdyn_verdict"
CONTEXT_EXCEEDED = "context_length_exceeded"
"""The model's context ran out (Inspect's ``model_length`` stop reason)."""
TOOL_CALL_PARSE_ERROR = "tool_call_parse_error"
"""A tool call's arguments were not valid JSON."""


@dataclass
class Verdict:
    """Upstream's verdicts for one episode, and how the episode ended."""

    utility: bool
    """The user task's check (upstream's ``utility``)."""
    attack_success: bool | None
    """The injection task's check (upstream's ``security``: ``True`` when the
    injection task was carried out); ``None`` for a clean episode."""
    error: str | None
    """Why upstream's benchmark would have aborted the episode, if it would."""
    iterations: int = 0
    """Tool-execution rounds run."""
    hit_iteration_limit: bool = False
    """The last answer still asked for tools when the rounds ran out."""


def save_verdict(state: TaskState, verdict: Verdict) -> None:
    state.store.set(STORE_KEY, asdict(verdict))


def load_verdict(state: TaskState) -> Verdict | None:
    """The episode's verdict, or ``None`` when no AgentDyn agent ran."""
    saved = state.store.get(STORE_KEY)
    return None if saved is None else Verdict(**saved)
