"""Generate both SEP conversations for one item.

One Inspect sample is one SEP item, and the solver generates twice: probe in
data, then probe in instruction (the authors' order). Keeping the pair in one
sample keeps it together under ``--limit``, shuffling and ``--epochs``: each
(sample, epoch) is one complete observation of both outputs, so the unreduced
separation metric never has to join samples.

The probe-in-instruction conversation is left as the sample's final
``messages`` and ``output``; both outputs are also kept in the sample store.
"""

from typing import Any

from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver

from .prompts import PROBE_POSITIONS, ProbePosition, messages

STORE_PREFIX = "sep_probe_in_"


def store_key(position: ProbePosition) -> str:
    """The sample-store key holding the output with the probe at ``position``."""
    return f"{STORE_PREFIX}{position}"


def output_record(output: ModelOutput) -> dict[str, Any]:
    """What the store keeps of one generation: text and stop reason."""
    return {
        "completion": output.completion,
        "stop_reason": output.stop_reason if output.choices else None,
    }


@solver
def sep_solver(*, system_role: bool = True) -> Solver:
    """Generate with the probe in the data argument, then in the instruction.

    Args:
        system_role: Send the instruction argument as a system message; with
            ``False`` both arguments go into one user message (see
            ``prompts.messages``).

    Returns:
        A solver that leaves both outputs in the store under ``store_key``.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = state.metadata
        for position in PROBE_POSITIONS:
            state.messages = messages(metadata, position, system_role=system_role)
            state = await generate(state)
            record = output_record(state.output)
            if position == "data":
                # The final messages are the probe-in-instruction conversation;
                # keep the other one for the log.
                record["messages"] = [m.model_dump(mode="json") for m in state.messages]
            state.store.set(store_key(position), record)
        return state

    return solve
