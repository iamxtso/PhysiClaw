"""What one route's compile threads through every entry parser — the
`Ctx` (the pack, the inputs, the payloads recorded so far, the macro
and playbook resolvers) and the resolver's type.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from physiclaw.common.paths import (
    entry_of,
)
from physiclaw.conductor.spec.model import (
    Pack,
    Playbook,
    Scanned,
)
from physiclaw.macros.model import (
    Macro,
)

# The shape `route_resolver` returns.
MacroResolve = Callable[..., Macro]


@dataclass
class Ctx:
    """What every entry parser reads — one object instead of the same
    five arguments threaded through each signature. `payloads` grows as
    the compile pass advances (an agent's return fields, in route
    order), so a `{move.field}` ref is defined-before-use by
    construction."""

    playbook: str
    pack: Pack
    input_names: set[str]
    resolve: "MacroResolve"
    # Seeded with the ONE slot every text may quote, `{ask.replies}`;
    # a payment ask's messages add `{ask.total}` (`payloads_with_total`).
    payloads: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"ask": ("replies",)}
    )
    # The prompt files an agent step may name — this route's own
    # (`prompts.<name>`) and the pack's (`app.prompts.<name>`) — and
    # the files it did read, pack-relative (`buy/prompts/pick.md`).
    prompts_local: Scanned[str] = field(default_factory=Scanned)
    prompts_pack: Scanned[str] = field(default_factory=Scanned)
    prompts_used: set[str] = field(default_factory=set)
    # The playbooks this entry runs, parsed on demand for a `run` line
    # (by id, `<entry>.<name>`) — None where the caller has no pack of
    # playbooks to offer (a playbook parsed from bare text).
    resolve_playbook: Callable[[str], Playbook] | None = None

    @property
    def entry(self) -> str:
        """The folder this file sits in — the playbook itself for an
        entry, its entry for a playbook that entry runs. The leaf files
        beside this one are the entry's (`macros/`, `prompts/`)."""
        return entry_of(self.playbook)

    def payloads_with_total(self) -> dict[str, tuple[str, ...]]:
        """The refs a payment step may quote: every recorded return
        field plus the ONE gate slot, `{ask.total}` — the consented
        amount its ask binds (`lints.check_money` keeps the two adjacent)."""
        return {**self.payloads, "ask": ("replies", "total")}
