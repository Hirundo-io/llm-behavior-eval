"""The AgentDyn agent loop and upstream's utility and security checks.

As upstream's default pipeline (``AgentPipeline.from_config`` with no defense,
``agentdojo/agent_pipeline/{agent_pipeline,tool_execution}.py`` and
``TaskSuite.run_task_with_pipeline`` at SaFo-Lab/AgentDyn commit
``5353cf7615b135cace8d07c8f12dac53a16b6db3``): the system message and the user
task's prompt, a first model call, then up to ``max_iterations`` rounds that
run every tool call of the last answer in order and call the model again. The
tools act on the episode's own copy of the suite environment, with the
injections filled in; each result goes back as upstream's YAML (or its error
message). At the end, the user task's ``utility`` check and, for an attacked
episode, the injection task's ``security`` check run on the last answer's
text, the environment before and after, and every tool call the model made.
Inspect owns inference and the transcript; there is no judge.
"""

import random
from ast import literal_eval
from dataclasses import dataclass, field
from typing import Any

from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str
from agentdojo.functions_runtime import (
    Function,
    FunctionCall,
    FunctionsRuntime,
    TaskEnvironment,
)
from agentdojo.task_suite.task_suite import TaskSuite
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolCall, ToolDef, ToolParams

from .dataset import get_suite
from .verdict import CONTEXT_EXCEEDED, TOOL_CALL_PARSE_ERROR, Verdict, save_verdict

TOOL_SEED = 42
"""Seeds the simulated tools' ``random`` draws (one-time codes, order IDs)."""


def upstream_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Arguments as upstream's ``ToolsExecutor`` passes them on.

    A string argument that parses as a Python list literal becomes that list
    (``"['a', 'b']"`` -> ``['a', 'b']``), as upstream does for every call to
    a known tool.
    """
    return {
        key: literal_eval(value)
        if isinstance(value, str) and is_string_list(value)
        else value
        for key, value in arguments.items()
    }


@dataclass
class Episode:
    """One episode's environment and tools; never shared between samples."""

    environment: TaskEnvironment
    runtime: FunctionsRuntime
    generator: random.Random = field(
        default_factory=lambda: random.Random(TOOL_SEED)  # noqa: S311 -- simulated codes
    )

    def run(self, function: str, arguments: dict[str, Any]) -> str:
        """Run one tool call and return what the model sees, as upstream does.

        Returns:
            Upstream's error text for an empty or unknown tool name or a
            failing call, otherwise the result as ``tool_result_to_str`` (YAML).
        """
        if function == EMPTY_FUNCTION_NAME:
            return "Empty function name provided. Provide a valid function name."
        if function not in self.runtime.functions:
            return f"Invalid tool {function} provided."
        # The tools draw from the global ``random``; give each episode its own
        # seeded stream (tool calls run synchronously, so nothing interleaves).
        outer_state = random.getstate()
        random.setstate(self.generator.getstate())
        try:
            result, error = self.runtime.run_function(
                self.environment, function, upstream_arguments(arguments)
            )
            self.generator.setstate(random.getstate())
        finally:
            random.setstate(outer_state)
        return error if error is not None else tool_result_to_str(result)

    def tool(self, function: Function) -> Tool:
        """``function`` as an Inspect tool with upstream's JSON schema."""

        async def execute(**arguments: Any) -> str:
            return self.run(function.name, arguments)

        return ToolDef(
            execute,
            name=function.name,
            description=function.description,
            parameters=ToolParams.model_validate(
                function.parameters.model_json_schema()
            ),
            parallel=False,
        ).as_tool()

    def results(self, calls: list[ToolCall]) -> list[ChatMessageTool]:
        """The tool messages for one answer's calls, run in order.

        The content is exactly what upstream's OpenAI client sends: the
        error text in place of the result when there is one.
        """
        return [
            ChatMessageTool(
                content=self.run(call.function, call.arguments),
                tool_call_id=call.id,
                function=call.function,
            )
            for call in calls
        ]


def trace(state: TaskState, known: set[str]) -> list[FunctionCall]:
    """Every tool call of every answer, as upstream's ``functions_stack_trace``.

    Calls to known tools that ran carry the arguments upstream's executor
    passed on (it converts them in place); calls that never ran (the last
    answer's, when the rounds ran out) keep the model's arguments.
    """
    messages = state.messages
    calls = []
    for index, message in enumerate(messages):
        if not isinstance(message, ChatMessageAssistant):
            continue
        ran = index + 1 < len(messages) and messages[index + 1].role == "tool"
        calls.extend(
            FunctionCall(
                function=call.function,
                args=upstream_arguments(call.arguments)
                if ran and call.function in known
                else dict(call.arguments),
                id=call.id,
            )
            for call in message.tool_calls or []
        )
    return calls


def last_answer(state: TaskState) -> str:
    """The text of the last answer (upstream's ``model_output_from_messages``)."""
    for message in reversed(state.messages):
        if isinstance(message, ChatMessageAssistant):
            return message.text
    return ""


def check(
    suite: TaskSuite,
    state: TaskState,
    pre_environment: TaskEnvironment,
    episode: Episode,
    error: str | None,
) -> Verdict:
    """Upstream's verdicts for a finished (or stopped) episode.

    An episode that upstream's benchmark would have aborted (``error``: the
    context was exceeded, or a tool call's arguments were not JSON) gets
    upstream's fixed verdict instead of the checks: no utility and, when
    attacked, a successful attack.
    """
    metadata = state.metadata
    user_task = suite.get_user_task_by_id(metadata["user_task"])
    injection_id = metadata["injection_task"]
    if error is not None:
        return Verdict(
            utility=False,
            attack_success=None if injection_id is None else True,
            error=error,
        )
    output = last_answer(state)
    calls = trace(state, set(episode.runtime.functions))
    environment = episode.environment
    utility = user_task.utility_from_traces(output, pre_environment, environment, calls)
    if utility is None:
        utility = user_task.utility(output, pre_environment, environment)
    attack_success = None
    if injection_id is not None:
        injection_task = suite.get_injection_task_by_id(injection_id)
        attack_success = injection_task.security_from_traces(
            output, pre_environment, environment, calls
        )
        if attack_success is None:
            attack_success = injection_task.security(
                output, pre_environment, environment
            )
    return Verdict(utility=utility, attack_success=attack_success, error=None)


def stop_error(state: TaskState) -> str | None:
    """Why upstream's benchmark would abort the episode after this answer."""
    if state.output.stop_reason == "model_length":
        return CONTEXT_EXCEEDED
    calls = (state.output.message.tool_calls or []) if state.output.choices else []
    if any(call.parse_error is not None for call in calls):
        return TOOL_CALL_PARSE_ERROR
    return None


@solver
def agentdyn_agent(max_iterations: int = 15) -> Solver:
    """Run one AgentDyn episode with upstream's agent loop, then check it.

    Args:
        max_iterations: Tool-execution rounds after the first answer
            (upstream's ``max_iters``); at most ``max_iterations + 1`` model calls.

    Raises:
        ValueError: ``max_iterations`` is negative.
    """
    if max_iterations < 0:
        raise ValueError("max_iterations must be zero or more")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        suite = get_suite(state.metadata["suite"])
        user_task = suite.get_user_task_by_id(state.metadata["user_task"])
        environment = user_task.init_environment(
            suite.load_and_inject_default_environment(state.metadata["injections"])
        )
        pre_environment = environment.model_copy(deep=True)
        episode = Episode(environment, FunctionsRuntime(suite.tools))
        state.tools = [episode.tool(function) for function in suite.tools]
        error = None
        rounds = 0
        pending = False
        try:
            state = await generate(state, tool_calls="none")
            while (error := stop_error(state)) is None:
                calls = state.output.message.tool_calls
                if not calls:
                    break
                if rounds == max_iterations:
                    pending = True
                    break
                rounds += 1
                state.messages.extend(episode.results(calls))
                state = await generate(state, tool_calls="none")
        finally:
            # Also when an Inspect limit stops the episode: it is scored as it stands.
            verdict = check(suite, state, pre_environment, episode, error)
            verdict.iterations = rounds
            verdict.hit_iteration_limit = pending
            save_verdict(state, verdict)
        return state

    return solve
