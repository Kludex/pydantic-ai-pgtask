from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

import pytest
from pgtask import Task
from pydantic_ai import Agent, ModelMessage, ModelResponse
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import ExternalToolset, FunctionToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_pgtask import CheckpointDecodeError, PGTaskDurability

from .conftest import CheckpointStore, make_model, make_task, running_task

pytestmark = pytest.mark.anyio


async def test_requires_name() -> None:
    with pytest.raises(UserError, match='unique `name`'):
        Agent(make_model(), capabilities=[PGTaskDurability()])


async def test_name_from_capability() -> None:
    agent = Agent(make_model(), capabilities=[PGTaskDurability(name='custom')])
    bound = PGTaskDurability.from_agent(agent)
    assert bound is not None
    assert bound.name == 'custom'


async def test_requires_model() -> None:
    with pytest.raises(UserError, match='needs to have a `model`'):
        Agent(name='a', capabilities=[PGTaskDurability()])


async def test_reserved_default_model_id_raises() -> None:
    with pytest.raises(UserError, match="'default' is reserved"):
        Agent(make_model(), name='a', capabilities=[PGTaskDurability(models={'default': make_model()})])


async def test_duplicate_toolset_id_raises() -> None:
    first = FunctionToolset[None](id='tools')

    @first.tool_plain
    def echo(value: str) -> str:  # pragma: no cover - never invoked, only wrap check
        return value

    second = FunctionToolset[None](id='tools')

    @second.tool_plain
    def shout(value: str) -> str:  # pragma: no cover - never invoked, only wrap check
        return value.upper()

    with pytest.raises(UserError, match='same `id`'):
        Agent(make_model(), name='a', toolsets=[first, second], capabilities=[PGTaskDurability()])


async def test_from_agent_without_capability_returns_none() -> None:
    agent = Agent(make_model(), name='a')
    assert PGTaskDurability.from_agent(agent) is None


async def test_from_agent_multiple_raises() -> None:
    agent = Agent(make_model(), name='a', capabilities=[PGTaskDurability(), PGTaskDurability()])
    with pytest.raises(UserError, match='at most one'):
        PGTaskDurability.from_agent(agent)


async def test_run_outside_task_is_transparent() -> None:
    counter = {'calls': 0}
    agent = Agent(make_model(counter), name='a', capabilities=[PGTaskDurability()])
    result = await agent.run('hi')
    assert result.output == 'ok'
    assert counter['calls'] == 1


async def test_run_inside_task_completes(task: Task) -> None:
    agent = Agent(make_model(), name='a', capabilities=[PGTaskDurability()])
    async with running_task(task):
        result = await agent.run('hi')
    assert result.output == 'ok'


async def test_replay_serves_cached_model_response(store: CheckpointStore) -> None:
    counter = {'calls': 0}
    agent = Agent(make_model(counter), name='crash', capabilities=[PGTaskDurability()])

    async with running_task(make_task(store)):
        first = await agent.run('hi')

    # Retry after a simulated crash: a fresh Task backed by the same store.
    async with running_task(make_task(store)):
        replayed = await agent.run('hi')

    assert counter['calls'] == 1
    assert replayed.output == first.output == 'ok'


async def test_replay_does_not_rerun_function_tool(store: CheckpointStore) -> None:
    tool_calls = {'calls': 0}
    toolset = FunctionToolset[None](id='tools')

    @toolset.tool_plain
    def charge_card(amount: int) -> str:
        tool_calls['calls'] += 1
        return f'charged {amount}'

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='charge_card', args={'amount': 42})])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(
        FunctionModel(fn, model_name='fn'),
        name='billing',
        toolsets=[toolset],
        capabilities=[PGTaskDurability()],
    )

    async with running_task(make_task(store)):
        first = await agent.run('charge it')

    async with running_task(make_task(store)):
        replayed = await agent.run('charge it')

    assert tool_calls['calls'] == 1
    assert replayed.output == first.output == 'done'


async def test_raw_tool_result_from_earlier_release_raises(store: CheckpointStore) -> None:
    """0.0.2 checkpointed a tool's raw return value, which can't be told apart from a corrupt
    checkpoint: the run fails instead of the model being asked to retry, which would run the tool again."""
    store.checkpoints[('billing__function_toolset__tools.call_tool:charge_card', 0)] = 'charged 42'
    toolset = FunctionToolset[None](id='tools')

    @toolset.tool_plain
    def charge_card(amount: int) -> str:  # pragma: no cover - never re-run
        return 'charged again'

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name='charge_card', args={'amount': 42})])

    agent = Agent(
        FunctionModel(fn, model_name='fn'), name='billing', toolsets=[toolset], capabilities=[PGTaskDurability()]
    )
    async with running_task(make_task(store)):
        with pytest.raises(CheckpointDecodeError):
            await agent.run('charge it')


async def test_leaf_toolset_without_id_raises() -> None:
    """Step names are built from the toolset `id`, so a leaf toolset needs one."""
    with pytest.raises(UserError, match='need to have a unique `id`'):
        Agent(make_model(), name='a', toolsets=[FunctionToolset[None]()], capabilities=[PGTaskDurability()])


async def test_registered_model_selected_per_run(task: Task, store: CheckpointStore) -> None:
    primary = {'calls': 0}
    cheap = {'calls': 0}

    def primary_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        primary['calls'] += 1
        return ModelResponse(parts=[TextPart(content='primary')])

    def cheap_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        cheap['calls'] += 1
        return ModelResponse(parts=[TextPart(content='cheap')])

    agent = Agent(
        FunctionModel(primary_fn, model_name='primary'),
        name='a',
        capabilities=[PGTaskDurability(models={'cheap': FunctionModel(cheap_fn, model_name='cheap')})],
    )

    async with running_task(task):
        default_result = await agent.run('hi')
        cheap_result = await agent.run('hi', model='cheap')

    assert default_result.output == 'primary'
    assert cheap_result.output == 'cheap'
    assert primary['calls'] == 1
    assert cheap['calls'] == 1
    assert ('a__model.request', 0) in store.executions
    assert ('a__model.request.cheap', 0) in store.executions


async def test_string_default_model_checkpoints_without_suffix(task: Task, store: CheckpointStore) -> None:
    agent = Agent('test', name='strdef', capabilities=[PGTaskDurability()])
    async with running_task(task):
        result = await agent.run('hi')
    assert result.output
    assert ('strdef__model.request', 0) in store.executions


async def test_runtime_function_toolset_rejected(task: Task) -> None:
    agent: Agent[None, str] = Agent(make_model(), name='a', capabilities=[PGTaskDurability()])
    toolset = FunctionToolset[None](id='late')

    @toolset.tool_plain
    def echo(value: str) -> str:  # pragma: no cover - rejected before it can run
        return value

    async with running_task(task):
        with pytest.raises(UserError, match='cannot be added at runtime with pgtask'):
            await agent.run('hi', toolsets=[toolset])


def _tool_calling_model(tool_name: str) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args={})])
        return ModelResponse(parts=[TextPart(content='done')])

    return FunctionModel(fn, model_name='fn')


def _late_toolset(calls: dict[str, int]) -> FunctionToolset[None]:
    toolset = FunctionToolset[None](id='late')

    @toolset.tool_plain
    def late() -> str:
        calls['calls'] += 1
        return 'late result'

    return toolset


async def test_agent_level_tools_are_durable(store: CheckpointStore) -> None:
    """Tools registered with `@agent.tool_plain` live on Pydantic AI's own toolset, whose `id`
    is the literal `<agent>` - a step name pgtask rejects unless it is normalized."""
    tool_calls = {'calls': 0}
    agent: Agent[None, str] = Agent(
        _tool_calling_model('charge_card'), name='billing', capabilities=[PGTaskDurability()]
    )

    @agent.tool_plain
    def charge_card() -> str:
        tool_calls['calls'] += 1
        return 'charged'

    async with running_task(make_task(store)):
        first = await agent.run('charge it')

    async with running_task(make_task(store)):
        replayed = await agent.run('charge it')

    assert tool_calls['calls'] == 1
    assert replayed.output == first.output == 'done'
    assert ('billing__function_toolset___agent_.call_tool:charge_card', 0) in store.executions


async def test_override_toolsets_rejected_inside_task(task: Task) -> None:
    calls = {'calls': 0}
    agent: Agent[None, str] = Agent(_tool_calling_model('late'), name='a', capabilities=[PGTaskDurability()])
    async with running_task(task):
        with agent.override(toolsets=[_late_toolset(calls)]):
            with pytest.raises(UserError, match='cannot be added at runtime with pgtask'):
                await agent.run('hi')
    assert calls['calls'] == 0


async def test_override_toolsets_respected_outside_task() -> None:
    calls = {'calls': 0}
    agent: Agent[None, str] = Agent(_tool_calling_model('late'), name='a', capabilities=[PGTaskDurability()])
    with agent.override(toolsets=[_late_toolset(calls)]):
        result = await agent.run('hi')
    assert result.output == 'done'
    assert calls['calls'] == 1


async def test_override_tools_rejected_inside_task(task: Task) -> None:
    calls = {'calls': 0}

    def late() -> str:  # pragma: no cover - rejected before it can run
        calls['calls'] += 1
        return 'late result'

    agent: Agent[None, str] = Agent(_tool_calling_model('late'), name='a', capabilities=[PGTaskDurability()])
    async with running_task(task):
        with agent.override(tools=[late]):
            with pytest.raises(UserError, match='cannot be added at runtime with pgtask'):
                await agent.run('hi')
    assert calls['calls'] == 0


async def test_override_tools_respected_outside_task() -> None:
    """The overriding toolset shares the `<agent>` id; instance-keyed wrapping must not swap it away."""
    calls = {'calls': 0}

    def late() -> str:
        calls['calls'] += 1
        return 'late result'

    agent: Agent[None, str] = Agent(_tool_calling_model('late'), name='a', capabilities=[PGTaskDurability()])
    with agent.override(tools=[late]):
        result = await agent.run('hi')
    assert result.output == 'done'
    assert calls['calls'] == 1


async def test_capability_owned_toolset_is_durable(store: CheckpointStore) -> None:
    """A toolset contributed by a capability is registered at construction, so it must be
    wrapped and checkpointed, not rejected as a runtime toolset because of the
    `CapabilityOwnedToolset` wrapper Pydantic AI puts around it."""
    tool_calls = {'calls': 0}
    toolset = FunctionToolset[None](id='owned')

    @toolset.tool_plain
    def charge_card(amount: int) -> str:
        tool_calls['calls'] += 1
        return f'charged {amount}'

    class DemoCapability(AbstractCapability[None]):
        def get_toolset(self) -> FunctionToolset[None]:
            return toolset

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='charge_card', args={'amount': 5})])
        return ModelResponse(parts=[TextPart(content='done')])

    agent: Agent[None, str] = Agent(
        FunctionModel(fn, model_name='fn'),
        name='owner',
        capabilities=[DemoCapability(), PGTaskDurability()],
    )

    async with running_task(make_task(store)):
        first = await agent.run('charge it')

    async with running_task(make_task(store)):
        replayed = await agent.run('charge it')

    assert tool_calls['calls'] == 1
    assert replayed.output == first.output == 'done'


async def test_runtime_toolset_still_rejected_alongside_capability_toolset(task: Task) -> None:
    """Ignoring wrapper nodes must not make genuine runtime toolsets slip through."""
    owned = FunctionToolset[None](id='owned')

    @owned.tool_plain
    def greet() -> str:  # pragma: no cover - never invoked
        return 'hello'

    class DemoCapability(AbstractCapability[None]):
        def get_toolset(self) -> FunctionToolset[None]:
            return owned

    agent: Agent[None, str] = Agent(make_model(), name='a', capabilities=[DemoCapability(), PGTaskDurability()])
    async with running_task(task):
        with pytest.raises(UserError, match='cannot be added at runtime with pgtask'):
            await agent.run('hi', toolsets=[_late_toolset({'calls': 0})])


async def test_runtime_external_toolset_allowed(task: Task) -> None:
    agent: Agent[None, str] = Agent(make_model(), name='a', capabilities=[PGTaskDurability()])
    async with running_task(task):
        result = await agent.run('hi', toolsets=[ExternalToolset[None](tool_defs=[])])
    assert result.output == 'ok'


async def test_construction_external_toolset_passes_through_unwrapped() -> None:
    external = ExternalToolset[None](tool_defs=[])
    agent = Agent(make_model(), name='a', toolsets=[external], capabilities=[PGTaskDurability()])
    assert any(t is external for t in agent.toolsets)


async def test_mcp_tool_call_inside_task(task: Task, store: CheckpointStore) -> None:
    from fastmcp import FastMCP
    from pydantic_ai.mcp import MCPToolset

    server: FastMCP[None] = FastMCP(name='calc')

    @server.tool
    def add(a: int, b: int) -> int:
        return a + b

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='add', args={'a': 2, 'b': 3})])
        return ModelResponse(parts=[TextPart(content='summed')])

    def make_agent(toolset: MCPToolset[None]) -> Agent[None, str]:
        return Agent(
            FunctionModel(fn, model_name='fn'), name='calc', toolsets=[toolset], capabilities=[PGTaskDurability()]
        )

    async with running_task(task):
        result = await make_agent(MCPToolset[None](server, id='calc')).run('add 2 and 3')
    assert result.output == 'summed'
    # Step names are what a replay looks checkpoints up by, so they must not drift across releases.
    assert store.executions == [
        ('calc__mcp_server__calc.get_tools', 0),
        ('calc__model.request', 0),
        ('calc__mcp_server__calc.call_tool', 0),
        ('calc__model.request', 1),
    ]

    # A replay served entirely from checkpoints doesn't connect, so an unreachable server is fine.
    async with running_task(make_task(store)):
        replayed = await make_agent(MCPToolset[None]('http://127.0.0.1:9/mcp', id='calc')).run('add 2 and 3')
    assert replayed.output == 'summed'
    assert len(store.executions) == 4


async def test_event_stream_handler_receives_events(task: Task) -> None:
    events: list[AgentStreamEvent] = []

    async def handler(run_ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='greet', json_args='{}')}
        else:
            yield 'done'

    toolset = FunctionToolset[None](id='tools')

    @toolset.tool_plain
    def greet() -> str:
        return 'hello'

    agent = Agent(
        FunctionModel(stream_function=stream_fn, model_name='fn'),
        name='a',
        toolsets=[toolset],
        capabilities=[PGTaskDurability(event_stream_handler=handler)],
    )

    async with running_task(task):
        result = await agent.run('hi')

    assert result.output == 'done'
    assert any(isinstance(e, PartStartEvent | PartDeltaEvent) for e in events)
    assert any(isinstance(e, FunctionToolCallEvent) for e in events)


async def test_run_stream_inside_task_replays_buffered_stream(task: Task) -> None:
    counter = {'calls': 0}
    agent = Agent(make_model(counter), name='a', capabilities=[PGTaskDurability()])
    async with running_task(task):
        async with agent.run_stream('hi') as result:
            assert await result.get_output() == 'ok'
    assert counter['calls'] == 1


async def test_run_stream_events_inside_task(task: Task) -> None:
    agent = Agent(make_model(), name='a', capabilities=[PGTaskDurability()])
    async with running_task(task):
        async with agent.run_stream_events('hi') as stream:
            events = [event async for event in stream]
    assert any(isinstance(e, PartStartEvent) for e in events)


async def test_stream_replay_serves_cached_events(store: CheckpointStore) -> None:
    counter = {'calls': 0}
    agent = Agent(make_model(counter), name='a', capabilities=[PGTaskDurability()])

    async with running_task(make_task(store)):
        async with agent.run_stream('hi') as result:
            first = await result.get_output()

    async with running_task(make_task(store)):
        async with agent.run_stream('hi') as result:
            replayed = await result.get_output()

    assert counter['calls'] == 1
    assert replayed == first == 'ok'


async def test_iter_inside_task(task: Task) -> None:
    agent = Agent(make_model(), name='a', capabilities=[PGTaskDurability()])
    async with running_task(task):
        async with agent.iter('hi') as run:
            async for _ in run:
                pass
    assert run.result is not None
    assert run.result.output == 'ok'


async def test_cancel_suspended_response_is_checkpointed(task: Task) -> None:
    cancelled: list[ModelResponse] = []

    class CancellableModel(FunctionModel):
        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            cancelled.append(response)

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
        return ModelResponse(parts=[TextPart(content='ok')])

    model = CancellableModel(fn, model_name='fn')
    agent = Agent(model, name='a', capabilities=[PGTaskDurability()])
    bound = PGTaskDurability.from_agent(agent)
    assert bound is not None

    ctx = RunContext[None](deps=None, model=model, usage=RunUsage())
    request_context = ModelRequestContext(
        model=model, messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
    )
    response = ModelResponse(parts=[TextPart(content='suspended')])

    async def handler(request: ModelRequestContext) -> ModelResponse:
        await request.model.cancel_suspended_response(response)
        return response

    async with running_task(task):
        result = await bound.wrap_model_request(ctx, request_context=request_context, handler=handler)

    assert result is response
    assert cancelled == [response]
