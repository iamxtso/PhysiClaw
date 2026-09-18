"""The `do` / `start` move — a recorded macro with its `with:` values."""

from physiclaw.conductor.route.fields import (
    check_with,
    irreversible_class,
    on_fail_mode,
)
from physiclaw.conductor.route.scope import Ctx
from physiclaw.conductor.spec.model import (
    DoNode,
    PlaybookError,
)


def parse_do(
    ctx: Ctx,
    where: str,
    nid: str,
    entry: dict,
    args: dict,
    enter: str,
    verify: str,
) -> DoNode:
    """A `do` move: the `macro:` it runs (a pack macro by name, or an
    inline body), with `with:` as that macro's inputs. `enter`/`verify`
    arrive derived from the route's waypoints."""
    if "." in enter or "." in verify:
        raise PlaybookError(
            f"{where}: a `do` runs on this pack's own pages — a reserved "
            "built-in cannot frame it"
        )
    if "macro" not in entry:
        raise PlaybookError(
            f"{where}: a `do` names its `macro:` — a pack macro by name, or an "
            "inline body with `steps:`"
        )
    spec = ctx.resolve(entry["macro"], where, nid)
    macro = spec.name
    check_with(where, args, spec.inputs, f"macro {macro!r}")
    return DoNode(
        id=nid,
        macro=macro,
        args=dict(args),
        enter=enter,
        verify=verify,
        irreversible=irreversible_class(entry, where),
        on_fail=on_fail_mode(entry, where),
    )
