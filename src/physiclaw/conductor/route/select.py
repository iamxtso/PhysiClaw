"""The boot's `select` move — parse_task over the enabled playbooks."""

from physiclaw.conductor.route.fields import limit_int, limit_mapping, think_level
from physiclaw.conductor.route.scope import Ctx
from physiclaw.conductor.spec.conventions import (
    BOOT_PLAYBOOK,
    CHANNEL_APP,
    THREAD_PAGE,
)
from physiclaw.conductor.spec.limits import (
    DEFAULT_BOOT_SCROLLS,
    MAX_AGENT_CALLS,
)
from physiclaw.conductor.spec.model import (
    PlaybookError,
    SelectNode,
)


def parse_select(
    ctx: Ctx, where: str, nid: str, entry: dict, current_page: str | None
) -> SelectNode:
    """The `select` step — the channel boot's own: read the thread and
    select the playbook it asks for. It reads the user's thread, so the
    thread page must sit immediately before it (its place at the route's
    end is `lints.check_boot`'s rule). `limit: {scrolls}` bounds
    parse_task's scroll-for-history escape."""
    if not is_boot(ctx):
        raise PlaybookError(
            f"{where}: `select` is the channel boot's own step — it belongs "
            f"in {CHANNEL_APP}/{BOOT_PLAYBOOK}/PLAYBOOK.yml only"
        )
    if current_page != THREAD_PAGE:
        raise PlaybookError(
            f"{where}: `select` reads the user's thread — put the "
            f"`{THREAD_PAGE}` page waypoint immediately before it"
        )
    limit = limit_mapping(entry, where, {"scrolls"})
    max_scrolls = limit_int(
        limit.get("scrolls", DEFAULT_BOOT_SCROLLS),
        f"{where}: `limit.scrolls`",
        0,
        MAX_AGENT_CALLS,
    )
    return SelectNode(
        id=nid,
        enter=current_page,
        max_scrolls=max_scrolls,
        think=think_level(entry, where),
    )


def is_boot(ctx: Ctx) -> bool:
    """Whether this route is the channel pack's boot playbook — the one
    file the `select` step is admitted in."""
    return ctx.pack.app == CHANNEL_APP and ctx.playbook == BOOT_PLAYBOOK
