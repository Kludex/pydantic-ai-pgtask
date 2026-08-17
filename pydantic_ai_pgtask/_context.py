from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from pgtask import Task, get_current_task

from ._step_names import normalize_step_name

StepT = TypeVar('StepT')

_CONTEXT_CACHE_MAX = 1024


@dataclass
class DurableTaskContext:
    """The current pgtask task plus per-name occurrence counters for its steps.

    pgtask disambiguates repeated step names with an explicit `occurrence` argument
    instead of counting encounters itself. This context assigns occurrences in
    encounter order, so as long as steps are reached in a deterministic order a
    replayed attempt maps each step call back to the same checkpoint.

    Names are normalized to pgtask's supported character set here, so callers can compose
    them from agent, toolset, and tool names without knowing that set.
    """

    task: Task
    _occurrences: Counter[str] = field(default_factory=Counter)

    async def step(self, name: str, operation: Callable[[], Awaitable[StepT]]) -> StepT:
        name = normalize_step_name(name)
        occurrence = self._occurrences[name]
        self._occurrences[name] += 1
        return await self.task.step(name, operation, occurrence=occurrence)


# Occurrence counters must be shared across every step call of one attempt, wherever in
# the call graph it happens, and must reset on a retry - hence keyed by (task id, attempt)
# rather than stored in a ContextVar (a set() inside a spawned subtask would not be seen
# by its siblings). Bounded so a long-lived worker cannot grow it without limit.
_contexts: dict[tuple[str, int], DurableTaskContext] = {}


def current_context() -> DurableTaskContext | None:
    """Return the durable context for the ambient pgtask task, or None outside a handler."""
    task = get_current_task()
    if task is None:
        return None
    key = (task.id, task.attempt)
    ctx = _contexts.get(key)
    if ctx is None:
        while len(_contexts) >= _CONTEXT_CACHE_MAX:
            _contexts.pop(next(iter(_contexts)))
        ctx = _contexts[key] = DurableTaskContext(task)
    return ctx
