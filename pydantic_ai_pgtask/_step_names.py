from __future__ import annotations

import re

_UNSUPPORTED = re.compile(r'[^A-Za-z0-9._:-]')


def normalize_step_name(name: str) -> str:
    """Replace every character pgtask rejects in a step name with `_`.

    pgtask accepts ASCII alphanumerics plus `-`, `.`, `:` and `_`. Step names are composed from
    values this package doesn't choose - the agent's name, a toolset `id`, a tool name - and
    Pydantic AI names an agent's own toolset `<agent>`, so normalize rather than fail the run.
    """
    return _UNSUPPORTED.sub('_', name)
