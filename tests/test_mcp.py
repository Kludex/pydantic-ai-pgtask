from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import FastMCP
from pgtask import Task
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_pgtask import PGTaskMCPToolset

from .conftest import CheckpointStore, running_task

MCP_SCRIPT = str(Path(__file__).resolve().parent / 'fixtures' / 'mcp_server.py')

pytestmark = pytest.mark.anyio


async def _return_hello() -> str:
    return 'hello'


def _make_server(id: str | None = None) -> MCPToolset[None]:
    server: FastMCP[None] = FastMCP(name='calc')

    @server.tool
    def add(a: int, b: int) -> int:
        return a + b

    return MCPToolset(server, id=id)


def _run_context() -> RunContext[None]:
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt='x', messages=[])


async def test_server_access() -> None:
    toolset = PGTaskMCPToolset(_make_server(), step_name_prefix='a')
    assert toolset._server is toolset.wrapped


async def test_visit_and_replace_returns_self() -> None:
    toolset = PGTaskMCPToolset(_make_server(), step_name_prefix='a')

    def visitor(t: MCPToolset[None]) -> MCPToolset[None]:  # pragma: no cover
        return t

    assert toolset.visit_and_replace(visitor) is toolset


async def test_id_passthrough() -> None:
    inner = _make_server()
    toolset = PGTaskMCPToolset(inner, step_name_prefix='a')
    assert toolset.id == inner.id


async def test_aenter_aexit() -> None:
    toolset = PGTaskMCPToolset(_make_server(), step_name_prefix='a')
    async with toolset as entered:
        assert entered is toolset


async def test_get_tools_and_call_tool_are_checkpointed(task: Task) -> None:
    toolset = PGTaskMCPToolset(_make_server(), step_name_prefix='a')
    run_context = _run_context()
    async with running_task(task):
        tools = await toolset.get_tools(run_context)
        assert 'add' in tools
        result = await toolset.call_tool('add', {'a': 2, 'b': 3}, run_context, tools['add'])
    assert result


async def test_server_id_with_unsupported_characters_is_normalized(task: Task, store: CheckpointStore) -> None:
    toolset = PGTaskMCPToolset(_make_server(id='calc<1>'), step_name_prefix='a')
    run_context = _run_context()
    async with running_task(task):
        tools = await toolset.get_tools(run_context)
        await toolset.call_tool('add', {'a': 2, 'b': 3}, run_context, tools['add'])
    assert ('a__mcp_server__calc_1_.get_tools', 0) in store.executions
    assert ('a__mcp_server__calc_1_.call_tool', 0) in store.executions


async def test_get_tools_without_context_passes_through() -> None:
    toolset = PGTaskMCPToolset(_make_server(), step_name_prefix='a')
    tools = await toolset.get_tools(_run_context())
    assert 'add' in tools


async def test_get_instructions_without_context_passes_through() -> None:
    toolset = PGTaskMCPToolset(_make_server(), step_name_prefix='a')
    result = await toolset.get_instructions(_run_context())
    # Servers without include_instructions return None.
    assert result is None


async def test_get_instructions_inside_context_returns_none_when_disabled(task: Task) -> None:
    toolset = PGTaskMCPToolset(_make_server(), step_name_prefix='a')
    async with running_task(task):
        result = await toolset.get_instructions(_run_context())
    assert result is None


async def test_get_instructions_inside_context_with_include(task: Task) -> None:
    server: FastMCP[None] = FastMCP(name='hello', instructions='Be brief.')
    inner = MCPToolset[None](server, include_instructions=True)
    toolset = PGTaskMCPToolset(inner, step_name_prefix='a')
    async with running_task(task):
        result = await toolset.get_instructions(_run_context())
    assert result is not None


async def test_run_step_without_context_passes_through() -> None:
    toolset = PGTaskMCPToolset(_make_server(), step_name_prefix='a')
    assert await toolset._run_step('x', _return_hello) == 'hello'


async def test_stdio_get_tools_and_call_tool_inside_context(task: Task) -> None:
    server = MCPToolset(MCP_SCRIPT)
    toolset = PGTaskMCPToolset(server, step_name_prefix='a')
    run_context = _run_context()
    async with running_task(task):
        async with server:
            tools = await toolset.get_tools(run_context)
            assert 'add' in tools
            # Second call hits the cache path (cache_tools defaults to True).
            cached = await toolset.get_tools(run_context)
            assert cached.keys() == tools.keys()
            result = await toolset.call_tool('add', {'a': 4, 'b': 5}, run_context, tools['add'])
    assert result is not None


async def test_stdio_get_tools_without_cache(task: Task) -> None:
    server = MCPToolset(MCP_SCRIPT, cache_tools=False)
    toolset = PGTaskMCPToolset(server, step_name_prefix='a')
    run_context = _run_context()
    async with running_task(task):
        async with server:
            first = await toolset.get_tools(run_context)
            second = await toolset.get_tools(run_context)
    assert first.keys() == second.keys() == {'add'}
