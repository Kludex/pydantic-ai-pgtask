from __future__ import annotations

import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4

import pytest
from pgtask import Task
from pgtask.client import _current_task
from pydantic import TypeAdapter
from pydantic_ai import ModelMessage, ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

JSON_ADAPTER: TypeAdapter[Any] = TypeAdapter(Any)

UNSUPPORTED_STEP_NAME_CHAR = re.compile(r'[^A-Za-z0-9._:-]')
"""The character set pgtask accepts in a step name."""


@dataclass
class CheckpointStore:
    """In-memory stand-in for pgtask's step checkpoint table.

    Mirrors the durability contract that matters to this package: a step keyed by
    `(name, occurrence)` runs its operation once, stores the JSON result, and serves
    the stored result on every later call - exactly what a pgtask replay does after
    a crash.
    """

    checkpoints: dict[tuple[str, int], Any] = field(default_factory=dict)
    executions: list[tuple[str, int]] = field(default_factory=list)

    async def step(self, name: str, occurrence: int, operation: Callable[[], Awaitable[Any]]) -> Any:
        # pgtask validates step names natively, so mirror that validation here: a name a real
        # worker would reject must fail the unit tests too, not only the integration ones.
        if not name:
            raise ValueError('step name must not be empty')
        if unsupported := UNSUPPORTED_STEP_NAME_CHAR.search(name):
            raise ValueError(f'step name contains unsupported character {unsupported.group()!r}')
        key = (name, occurrence)
        if key not in self.checkpoints:
            self.executions.append(key)
            result = await operation()
            self.checkpoints[key] = JSON_ADAPTER.dump_python(result, mode='json')
        return self.checkpoints[key]


def make_task(store: CheckpointStore, attempt: int = 1) -> Task:
    """Build a `Task` whose `step` is backed by an in-memory checkpoint store."""

    class FakeNativeContext:
        async def step(self, name: str, occurrence: int, operation: Callable[[], Awaitable[Any]]) -> Any:
            return await store.step(name, occurrence, operation)

    now = datetime.now(timezone.utc)
    return Task(
        id=str(uuid4()),
        parent_task_id=None,
        queue_name='default',
        task_name='test',
        handler_version=1,
        payload=None,
        headers={},
        state='running',
        attempt=attempt,
        max_attempts=5,
        run_at=now,
        created_at=now,
        _context=cast(Any, FakeNativeContext()),
    )


def make_model(counter: dict[str, int] | None = None, content: str = 'ok') -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if counter is not None:
            counter['calls'] += 1
        return ModelResponse(parts=[TextPart(content=content)])

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> Any:
        if counter is not None:
            counter['calls'] += 1
        yield content

    return FunctionModel(fn, stream_function=stream_fn, model_name='fn')


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@asynccontextmanager
async def running_task(task: Task) -> AsyncIterator[Task]:
    """Make `task` ambient, exactly as pgtask's worker adapter does around a handler."""
    token = _current_task.set(task)
    try:
        yield task
    finally:
        _current_task.reset(token)


@pytest.fixture
def store() -> CheckpointStore:
    return CheckpointStore()


@pytest.fixture
def task(store: CheckpointStore) -> Task:
    return make_task(store)
