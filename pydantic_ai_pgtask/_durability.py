from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import ValidationError
from pydantic_ai.agent import EventStreamHandler, ParallelExecutionMode
from pydantic_ai.capabilities.abstract import WrapRunHandler
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BaseDurabilityCapability,
    DurabilityCodec,
    DurabilityEngineSpec,
    DurableOperationId,
    JournalCallableOperationBackend,
    RoleBasedOperationConfig,
)
from pydantic_ai.models import Model
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext

from ._context import current_context

PGTaskParallelExecutionMode = Literal['sequential', 'parallel_ordered_events']
"""Tool-call execution modes usable with pgtask. A subset of `ParallelExecutionMode`: `'parallel'`
is excluded because the durable context disambiguates repeated step names with an encounter-order
occurrence counter, so checkpoints must be reached in a deterministic order for a replay to line
up with them."""


class CheckpointDecodeError(RuntimeError):
    """A step's checkpoint doesn't decode as what the step returns now."""


class _PGTaskCodec(DurabilityCodec):
    """JSON, failing the run when a checkpoint doesn't decode.

    Pydantic AI decodes a tool's checkpoint inside the tool call, where a `ValidationError` is read as
    bad model arguments: the model would be asked to retry, and the retry would run the tool again.
    Raising something else keeps a side effect from repeating. This includes the raw function and MCP
    tool results that 0.0.2 checkpointed, so tasks in flight have to finish before upgrading from it.
    """

    def dump(self, tp: Any, value: Any) -> Any:
        return JSON_CODEC.dump(tp, value)

    def load(self, tp: Any, payload: Any) -> Any:
        try:
            return JSON_CODEC.load(tp, payload)
        except ValidationError as exc:
            raise CheckpointDecodeError(f'A pgtask checkpoint could not be decoded: {exc}') from exc


class PGTaskOperationBackend(JournalCallableOperationBackend[None]):
    """Runs each durable operation Pydantic AI hands over as one `task.step(...)` checkpoint."""

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: None,
    ) -> object:
        task_ctx = current_context()
        assert task_ctx is not None  # pragma: no cover - operations only run inside a durable context
        return await task_ctx.step(name, body)


@dataclass(init=False)
class PGTaskDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by checkpointing I/O into pgtask steps.

    Attach it to an agent via `capabilities=[PGTaskDurability()]` and call `agent.run()`
    inside any pgtask handler:
    every model request, MCP call, and function tool call is wrapped in `task.step(...)`,
    so a worker crash mid-run resumes from the last completed step instead of restarting -
    no tokens are re-spent, and side effects run once. Outside a durable task the
    capability is transparent and the run is a normal, non-durable agent run.

    The capability discovers the agent's model, name, and toolsets automatically when it
    is bound to the agent. Step results are stored in Postgres as JSON, so a checkpointed
    tool's return value must be JSON-serializable.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_pgtask import PGTaskDurability

        agent = Agent('openai:gpt-5.2', name='analyst', capabilities=[PGTaskDurability()])

        @tasks.task('analyse')
        async def analyse(task, payload):
            result = await agent.run(payload['prompt'])
            return {'output': result.output}
        ```
    """

    engine_spec: ClassVar = DurabilityEngineSpec(
        engine_name='pgtask',
        durable_unit_noun='step',
        durable_container_noun='task',
        codec=_PGTaskCodec(),
        wrapped_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
        # The first MCP step connects, so a replay whose MCP steps are all checkpointed never does.
        toolset_lifecycles={'function': 'enter-always', 'mcp': 'enter-in-durable-unit', 'dynamic': 'enter-never'},
        # `wrap_run` applies `parallel_execution_mode`, which already excludes `'parallel'`.
        sequential_tools_in_durable_context=False,
        unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
    )

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        parallel_execution_mode: PGTaskParallelExecutionMode = 'sequential',
    ) -> None:
        """Create a PGTaskDurability capability.

        The agent's model, name, and toolsets are discovered automatically.

        Args:
            models: Optional additional models keyed by ID for runtime model switching via
                `agent.run(model='<id>')`. The agent's primary model is always registered as
                `'default'`; the ID is folded into the checkpoint step name so a replay
                resolves to the same model.
            event_stream_handler: Optional event stream handler. Model events are handled
                live inside the model-request step; each tool event is handled in its own
                checkpointed step.
            name: Unique agent name used as the prefix for every checkpoint step. Defaults
                to the agent's `name` when the capability is bound.
            parallel_execution_mode: Tool-call execution mode applied for the duration of
                every run. Defaults to `'sequential'`. `'parallel'` is excluded by type:
                repeated step names are disambiguated with an encounter-order counter, so
                steps must be reached in a deterministic order for a replay to line up
                with its checkpoints.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        self._parallel_execution_mode: ParallelExecutionMode = parallel_execution_mode

    @property
    def in_durable_context(self) -> bool:
        return current_context() is not None

    def get_durable_operation_backend(self) -> PGTaskOperationBackend:
        return PGTaskOperationBackend(
            agent_name=self.name,
            default_model_id=self.default_model_id,
            config=RoleBasedOperationConfig(model=None, event=None, capability=None, tool=None),
        )

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Apply the configured parallel-execution mode for every entry point."""
        agent = self.agent
        if agent is None:  # pragma: no cover
            return await handler()
        with agent.parallel_tool_call_execution_mode(self._parallel_execution_mode):
            return await handler()
