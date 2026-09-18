"""What an agent step reads out of the agent's OWN memory — declared,
never implied. A step's `context.memory:` names the parts, one entry
per part, and nothing else of that memory travels to the model:

  - ``<slug>: true``    ONLY the `## <slug>` section of memory.md — a
                        slug matches a heading as a whole whitespace-
                        separated token (`shopping` never bleeds into
                        `## shopping_blacklist`; a heading may carry a
                        translation after its slug), and no match means
                        no text (fail closed)
  - ``all: true``       the whole memory.md
  - ``log: <n>``        the n most recent daily-log entries — the same
                        window the engine preloads into a wake

A part is included because a line says so: there is no default and no
switch, and an absent `memory:` sends nothing. A slug that matches no
heading loads "" rather than vanishing: the author declared it, so the
model reads an empty slot instead of wondering whether it was told. The conductor may never
import the engine (architecture rule), so memory.md is read off the
shared path constant here.

Nothing here labels what it returns — `load` keys each body by the part
that asked for it, and the step's own `context:` block is where the
label is written.
"""

from collections.abc import Mapping
from typing import Any

from physiclaw.common import daylog, paths
from physiclaw.common.text import read_text
from physiclaw.conductor.spec.specfile import INPUT_NAME_RE

ALL = "all"  # the whole memory.md
LOG = "log"  # the recent daily-log window, `log: <n>`
# The parts the ENGINE names; any other key is a slug — one `## <slug>`
# section of memory.md. `memory_gap` branches on this tuple and
# `lints` reads its exemption off it (a part is the engine's own word, so a
# prompt has no reason to spell it), which is why a new part is one row.
PARTS = (ALL, LOG)
MAX_LOG = 50  # daily-log entries one step may pull


def memory_gap(spec: object) -> str | None:
    """None when `spec` is a legal `memory:` block, else the rule it
    breaks. The one legality reader: the parser calls it, and nothing
    re-derives the vocabulary."""
    if not isinstance(spec, dict):
        return "must be a mapping of `<part>: <how much>`"
    for part, value in spec.items():
        if not isinstance(part, str):
            return f"part {part!r} must be a name"
        if part == LOG:
            if isinstance(value, bool) or not isinstance(value, int):
                return f"`{LOG}: {value!r}` must be how many entries, a number"
            if not 1 <= value <= MAX_LOG:
                return f"`{LOG}: {value}` must be from 1 to {MAX_LOG}"
        elif value is not True:
            return (
                f"`{part}: {value!r}` must be `true` — a part is named to "
                "include it, and left out to leave it behind"
            )
        elif part not in PARTS and not INPUT_NAME_RE.match(part):
            return f"part {part!r} must be a `## <slug>` heading of memory.md"
    return None


def load(spec: Mapping[str, Any]) -> dict[str, str]:
    """Each named part, read NOW and keyed by its own name — the label
    the model reads, which is why no body carries a heading of its own.
    memory.md is read and carved once however many sections are named,
    and not touched at all when none is."""
    slugs = [p for p in spec if p not in PARTS]
    text = _memory_text() if slugs or ALL in spec else ""
    sections = split_sections(text) if slugs else []
    out = {slug: match_sections(sections, slug) for slug in slugs}
    if ALL in spec:
        out[ALL] = text
    if LOG in spec:
        out[LOG] = daylog.load_recent_entries(spec[LOG])
    return out


def _memory_text() -> str:
    f = paths.memory_file()
    return read_text(f).strip() if f.exists() else ""


# One parsed section: (heading tokens, whole section text).
Sections = list[tuple[frozenset[str], str]]


def split_sections(text: str) -> Sections:
    """The file carved at its `## ` headings — parsed once, matched many
    times."""
    sections: Sections = []
    tokens: frozenset[str] | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if tokens is not None:
                sections.append((tokens, "\n".join(body).strip()))
            tokens = frozenset(line[3:].casefold().split())
            body = [line]
        elif tokens is not None:
            body.append(line)
    if tokens is not None:
        sections.append((tokens, "\n".join(body).strip()))
    return sections


def match_sections(sections: Sections, slug: str) -> str:
    wanted = slug.casefold()
    return "\n".join(body for tokens, body in sections if wanted in tokens)
