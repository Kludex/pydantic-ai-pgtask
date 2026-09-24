# Pydantic AI pgtask

<p align="center"><em>Durable execution for Pydantic AI agents, on Postgres alone.</em></p>

---

Agents run for a while, a model call, a tool call, another model call. When the worker dies in the middle of that, the run is usually lost: you restart from zero and pay for every token again.

**Pydantic AI pgtask** fixes that. Call `agent.run()` inside a durable [pgtask](https://github.com/Kludex/pgtask) handler and every model and MCP call is checkpointed into Postgres. If the worker crashes, a new one resumes from the last completed step, no restart, no re-spent tokens. Same idea as Pydantic AI's Temporal integration, but with no Temporal, no Redis, no broker: just the Postgres you already have.

## Installation

```bash
pip install pydantic-ai-pgtask
```

## Example

```python
import asyncio
import os
from typing import Any

from pgtask import Client, Task, TaskRegistry, Worker
from pydantic_ai import Agent
from pydantic_ai_pgtask import PGTaskDurability

tasks = TaskRegistry(queue_name="agents")
agent = Agent("openai:gpt-5.2", name="analyst", capabilities=[PGTaskDurability()])


@tasks.task("analyse")
async def analyse(task: Task, payload: dict[str, Any]) -> dict[str, Any]:
    result = await agent.run(payload["prompt"])
    return {"output": result.output}


async def main() -> None:
    database_url = os.environ["PGTASK_DATABASE_URL"]
    client = await Client.connect(database_url)
    await client.migrate()
    await client.enqueue(analyse.request({"prompt": "Analyse Q3 revenue"}))
    await Worker(database_url, tasks).run()


if __name__ == "__main__":
    asyncio.run(main())
```

You author a task, call the agent inside it, and run it durably. That's the whole idea.

## How it works

`PGTaskDurability` is a Pydantic AI capability. When the agent runs inside any pgtask handler (discovered via pgtask's `get_current_task()`), it wraps:

- every **model request** in a `task.step(...)` checkpoint,
- every **function tool call** in its own checkpoint, so side effects run exactly once,
- every **MCP** `get_tools`, `get_instructions`, and `call_tool` in its own checkpoint.

Step names are built from the agent's `name` and each toolset's `id`, so every toolset that runs its own tools needs an `id` (including `Capability(tools=...)`); without one, building the agent raises, even if it never runs in a task. Step results are stored in Postgres as JSON. On a retry after a crash, pgtask replays completed steps from their cached results instead of re-executing them, so the run resumes exactly where it stopped.

Outside a durable task the capability is transparent: the agent behaves like a regular, non-durable agent.

!!! warning "Deterministic step order"
    Repeated step names are disambiguated with an encounter-order occurrence counter, so tool calls run in `'sequential'` mode by default. `'parallel'` execution is excluded by type - a replay must reach checkpoints in the same order they were recorded.
