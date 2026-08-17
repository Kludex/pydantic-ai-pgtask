from __future__ import annotations

import pytest
from pgtask import Task

from pydantic_ai_pgtask import current_context
from pydantic_ai_pgtask._context import _CONTEXT_CACHE_MAX, _contexts

from .conftest import CheckpointStore, make_task, running_task

pytestmark = pytest.mark.anyio


async def test_no_context_outside_handler() -> None:
    assert current_context() is None


async def test_context_follows_ambient_task(task: Task) -> None:
    async with running_task(task):
        ctx = current_context()
        assert ctx is not None
        assert ctx.task is task
        # Same attempt resolves to the same context (and thus shared counters).
        assert current_context() is ctx
    assert current_context() is None


async def test_step_occurrences_count_per_name(task: Task, store: CheckpointStore) -> None:
    async with running_task(task):
        ctx = current_context()
        assert ctx is not None

        async def one() -> int:
            return 1

        assert await ctx.step('a', one) == 1
        assert await ctx.step('a', one) == 1
        assert await ctx.step('b', one) == 1

    assert store.executions == [('a', 0), ('a', 1), ('b', 0)]


async def test_checkpoint_store_rejects_what_pgtask_rejects(store: CheckpointStore) -> None:
    """The guard that makes every other test fail on a name a real worker would refuse."""

    async def one() -> int:  # pragma: no cover - never reached, the name is rejected first
        return 1

    with pytest.raises(ValueError, match='must not be empty'):
        await store.step('', 0, one)
    with pytest.raises(ValueError, match="unsupported character '<'"):
        await store.step('a__function_toolset__<agent>', 0, one)


async def test_step_name_is_normalized_to_pgtask_charset(task: Task, store: CheckpointStore) -> None:
    async with running_task(task):
        ctx = current_context()
        assert ctx is not None

        async def one() -> int:
            return 1

        assert await ctx.step('a__function_toolset__<agent>.call_tool:x', one) == 1

    assert store.executions == [('a__function_toolset___agent_.call_tool:x', 0)]


async def test_replay_serves_cached_step(store: CheckpointStore) -> None:
    calls = {'n': 0}

    async def op() -> int:
        calls['n'] += 1
        return calls['n']

    async with running_task(make_task(store)):
        ctx = current_context()
        assert ctx is not None
        assert await ctx.step('s', op) == 1

    # A retry is a fresh Task backed by the same checkpoints: occurrence counters
    # restart at 0, so the step resolves to the same checkpoint and does not re-run.
    async with running_task(make_task(store)):
        ctx = current_context()
        assert ctx is not None
        assert await ctx.step('s', op) == 1

    assert calls['n'] == 1


async def test_retry_gets_fresh_occurrence_counters(store: CheckpointStore) -> None:
    task = make_task(store)
    async with running_task(task):
        first = current_context()
    retry = make_task(store, attempt=2)
    object.__setattr__(retry, 'id', task.id)
    async with running_task(retry):
        second = current_context()
    assert first is not None and second is not None
    assert first is not second


async def test_context_cache_is_bounded(store: CheckpointStore) -> None:
    for _ in range(_CONTEXT_CACHE_MAX + 5):
        async with running_task(make_task(store)):
            assert current_context() is not None
    assert len(_contexts) <= _CONTEXT_CACHE_MAX
