"""The route as the cursor walks it — one `Slot` per node, and the
`Round`s a `run` expands into: the playbook's nodes walked once per
item, each with its own inputs and its own record in the ledger. The
walk (`program.py`) owns the cursor; this is the shape it moves over.
"""

from dataclasses import dataclass

from physiclaw.conductor.spec.model import Node, RunNode
from physiclaw.conductor.walk.ledger import round_prefix


@dataclass(frozen=True)
class Round:
    """One round of a `run`: the playbook's nodes walked once, with its
    own inputs and its own record in the ledger (`ledger.rounds`, under
    `ledger.round_prefix`), so a re-plan can tell a finished round from
    one still to do, and a round's own refs read only its own record."""

    run: RunNode
    key: str
    inputs: dict[str, str]  # `inputs.<name>` → value, this round's

    @property
    def prefix(self) -> str:
        return round_prefix(self.run.id, self.key)


@dataclass(frozen=True)
class Slot:
    """One place on the route the cursor walks: the node, the spec index
    it came from (what a suspension stores), and the round it belongs
    to — None for the route's own nodes."""

    node: Node
    origin: int
    round: Round | None = None


def run_keys(run: RunNode, vals: dict[str, str]) -> list[str]:
    """The rounds a run has now: one, or one per distinct line of its
    list (the list's CURRENT value — a re-plan changes it)."""
    if run.each is None:
        return [""]
    lines = vals.get(run.each[1], "").splitlines()
    return list(dict.fromkeys(k for line in lines if (k := line.strip())))
