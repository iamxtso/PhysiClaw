"""The route compiler — `route:` → the compiled moves, the start page,
and each page's declared recovery hand; the whole-route lints are
`lints.py`.

Waypoints do not become nodes — they become the adjacent moves' checks:
a `do`'s (and an acting `agent`'s) enter is the nearest preceding page,
its verify the page that must follow it. The route opens with an
optional prefix of pure-text `agent` steps and an optional `start` (the
unconditional cold launch); the first page is the start contract. Moves
fall through in route order — an `ask` once approved, a `tell` once its
message landed — and past the last entry the walk is done.

An inline macro is single-use by construction: its name is synthesized
`<playbook>.<move>` or `<playbook>.<name>.<role>`, dot-joined, a
spelling no directory macro can take, so the pack's dispatch namespace
never collides. Under an inline `macro:` the macro grammar applies
(single-name `{x}` templates); outside it, refs stay dotted.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar

from physiclaw.common import gesture_vocab
from physiclaw.common.bbox import parse_within
from physiclaw.common.paths import (
    PACK_FILENAME,
    PACK_MACROS_DIRNAME,
    PACK_PROMPTS_DIRNAME,
    PROMPT_SUFFIX,
    entry_of,
    playbook_file,
    run_by,
)
from physiclaw.conductor.spec import context, lints, match, reply
from physiclaw.conductor.spec.calls import AGENT_TOOLS, CONTRACT_FIELDS, RESERVED_KEYS
from physiclaw.conductor.spec.conventions import (
    BOOT_PLAYBOOK,
    CHANNEL_APP,
    RESERVED_APPS,
    THREAD_PAGE,
)
from physiclaw.conductor.spec.limits import (
    DEFAULT_AGENT_CALLS,
    DEFAULT_AGENT_SCROLLS,
    DEFAULT_ASK_ROUNDS,
    DEFAULT_ASK_WAIT_SECONDS,
    DEFAULT_BOOT_SCROLLS,
    DEFAULT_RECOVER_LIMIT,
    DEFAULT_REVISIONS,
    DEFAULT_RUN_ROUNDS,
    MAX_AGENT_CALLS,
    MAX_ASK_ROUNDS,
    MAX_ASK_WAIT_SECONDS,
    MAX_MESSAGE_LINES,
    MAX_NEVER_TAP,
    MAX_NODES,
    MAX_PROMPT_LEN,
    MAX_RECOVER_ACTIONS,
    MAX_RETURNS,
    MAX_REVISIONS,
    MAX_RUN_ROUNDS,
    MIN_ASK_WAIT_SECONDS,
)
from physiclaw.conductor.spec.match import normalize, page_resolver
from physiclaw.conductor.spec.model import (
    INPUTS_ROOT,
    IRREVERSIBLE_CLASSES,
    MISS_MODES,
    ON_FAIL_MODES,
    READING_COVERED,
    READING_ELSEWHERE,
    READING_LOCKED,
    RECOVER_READINGS,
    ActivateNode,
    AgentNode,
    AskNode,
    DoNode,
    NeverTap,
    Node,
    Pack,
    Playbook,
    PlaybookError,
    RecoverHand,
    Recovery,
    RunNode,
    Scanned,
    TellNode,
    check_name,
    macro_resolver,
    prose,
    require_str,
)
from physiclaw.conductor.spec.pages import (
    PAGE_DECL_FIELDS,
    PAGE_RECOVERY_FIELDS,
    PagesError,
    decl_fields,
    page_menu,
    parse_pages_data,
    recovery_fields,
    route_decl,
)
from physiclaw.conductor.spec.refs import (
    check_arg_refs,
    check_refs,
    field_name,
    refs_in,
)
from physiclaw.contract.dto import THINKING_LEVELS, Thinking
from physiclaw.macros.model import (
    LANDMARKS_KIND,
    MACROS_KIND,
    PAGES_KIND,
    PROMPTS_KIND,
    Macro,
    MacroError,
    MacroInput,
    app_ref,
    checked_readings,
    parse_ref,
)
from physiclaw.macros.parse import parse_inline_macro

# Agent-step grammar. `tools` is the closed per-episode gesture allowlist
# (`calls.AGENT_TOOLS`, the keys of the tool → verbs map the runner
# reads); the bounds on calls, scrolls, prompt, and returns are
# `limits.py`'s.
_AGENT_LIMIT_KEYS = {"calls", "scrolls"}

# The declared-recovery hand, in a macro step's own shape: a bare
# gesture, `{tap: app.landmarks.<name>}`, or `{macro: app.macros.<name>}`. Closed
# vocabulary — a recover hand resets state, it does not navigate (the
# route does that). A page declares one hand for any deviation, or one
# per reading (`covered:` / `elsewhere:` / `locked:`), plus its own
# `tries:`.
BARE_HANDS = tuple(sorted(gesture_vocab.NAV_TOOLS | {gesture_vocab.UNLOCK_PHONE}))
_HAND_KEYS = {"tap", "macro"}
_RECOVER_KEYS = _HAND_KEYS | set(RECOVER_READINGS)


# Route entry vocabularies. An entry's KIND is its leading key and the
# value is the entry's name — the map-key-is-the-name doctrine, applied
# to the route. Page-declaration fields come from `pages.py`'s ONE
# spelling (PAGE_DECL_FIELDS) — their content is validated there; they
# appear here only so the unknown-key check names them as legal.
LINE_KINDS = ("page", "start", "do", "agent", "ask", "tell", "run", "select")
_ENTRY_KEYS = {
    "page": {"page", *PAGE_RECOVERY_FIELDS, *PAGE_DECL_FIELDS},
    "start": {"start", "macro", "on_fail"},
    "do": {"do", "with", "macro", "irreversible", "on_fail"},
    "agent": {
        "agent",
        "prompt",
        "tools",
        "give",
        "never_tap",
        "returns",
        "limit",
        "context",
        "irreversible",
        "think",
        "on_fail",
    },
    "ask": {
        "ask",
        "approve",
        "message",
        "yes",
        "no",
        "denied",
        "total_label",
        "wait",
        "rounds",
        "resume",
        "think",
        "on_fail",
    },
    "tell": {"tell", "message", "on_fail"},
    "run": {"run", "with", "each", "miss", "revise", "limit", "on_fail"},
    "select": {"select", "limit", "think"},
}
_RUN_LIMIT_KEYS = {"rounds", "revisions"}

# The shape `_macro_resolver` returns.
_MacroResolve = Callable[..., Macro]

_T = TypeVar("_T")


@dataclass
class _Ctx:
    """What every entry parser reads — one object instead of the same
    five arguments threaded through each signature. `payloads` grows as
    the compile pass advances (an agent's return fields, in route
    order), so a `{move.field}` ref is defined-before-use by
    construction."""

    playbook: str
    pack: Pack
    input_names: set[str]
    resolve: "_MacroResolve"
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


@dataclass(frozen=True)
class CompiledRoute:
    """What `compile_route` hands back: the moves, the start page, each
    page's declared recovery hand, and the inline macro bodies under
    their synthesized names."""

    nodes: list[Node]
    start: str
    recovers: dict[str, Recovery]
    inline: dict[str, Macro]
    prompts_used: frozenset[str] = frozenset()
    # The last waypoint ("" when the route ends on a move) and every
    # move's declared outputs — what the playbook's own `returns:` and a
    # `run` of it read.
    end: str = ""
    payloads: dict[str, tuple[str, ...]] = field(default_factory=dict)


def compile_route(
    raw: Any,
    *,
    playbook: str,
    input_names: set[str],
    pack: Pack,
    resolve_playbook: Callable[[str], Playbook] | None = None,
    pages: Any = None,
) -> CompiledRoute:
    """`route:` → the compiled route (see `CompiledRoute`): the shape
    prepass first (every rule about WHERE an entry may sit), then one
    forward pass compiling the moves against the waypoints around them,
    then the lints that need the whole route."""
    own = _own_pages(pages, pack)
    entries, wp_ids, start, page_names = _shape(raw, pack, set(own))
    local = pack.local_for(playbook)
    # The entry's recorded hands enter the dispatch table here, once,
    # under their `<entry>.<name>` spelling (the playbooks it runs read
    # the same folder) — the inline bodies the route embeds join them as
    # the compile pass meets them.
    inline = _local_registry(entry_of(playbook), pack, local.macros)
    ctx = _Ctx(
        playbook,
        pack,
        input_names,
        _macro_resolver(playbook, pack, inline, local.macros),
        prompts_local=local.prompts,
        prompts_pack=pack.prompts,
        resolve_playbook=resolve_playbook,
    )
    # A run's returns are known before the route is walked, so a text
    # ABOVE the run may quote them (empty until its rounds end — the
    # one forward ref, for a plan that re-reads what a run already
    # did). Resolved here, once, and the bodies kept for the parse.
    subs: dict[int, Playbook] = {}
    for i, (kind, name, _entry) in enumerate(entries):
        if kind == "run":
            subs[i] = _sub_playbook(ctx, f"route line {i + 1}", name)
            ctx.payloads[name] = tuple(subs[i].returns)
    moves: list[Node] = []
    seen: dict[str, int] = {}
    recovers: dict[str, Recovery] = {}  # this route's own hands, by page
    current_page: str | None = None
    for i, (kind, name, entry) in enumerate(entries):
        pos = i + 1
        if kind == "page":
            current_page = wp_ids[i]
            fields = recovery_fields(entry)
            if fields:
                rpage = current_page
                assert rpage is not None
                if "." in rpage:
                    raise PlaybookError(
                        f"route line {pos}: {rpage!r} is a reserved built-in "
                        "— packs declare recovery for their own pages only"
                    )
                declared = _parse_recover(ctx, fields, f"route line {pos}", rpage)
                recovers[rpage] = _declared_once(
                    recovers.get(rpage), declared, f"route line {pos}", rpage
                )
            continue
        where = f"route line {pos}"
        check_name(name, f"{where}: `{kind}`")
        if name == INPUTS_ROOT:
            raise PlaybookError(
                f"{where}: name {name!r} is a reserved ref root — "
                "{inputs.*} always reads the declared inputs"
            )
        if name in page_names:
            raise PlaybookError(
                f"{where}: {name!r} is also a page on this route — moves "
                "and pages share one namespace, so the names must not collide"
            )
        if name in seen:
            raise PlaybookError(
                f"{where}: duplicate move name {name!r} (line {seen[name]} "
                "already uses it) — refs address moves by name, so they "
                "must be unique"
            )
        seen[name] = pos
        where = f"move {name!r}"
        args = entry.get("with", {})
        if not isinstance(args, dict):
            raise PlaybookError(f"{where}: `with` must be a mapping of arguments")
        check_arg_refs(args, input_names, ctx.payloads, where)
        nxt = wp_ids[i + 1] if i + 1 < len(entries) else None
        if kind == "do":
            if nxt is None:
                raise PlaybookError(
                    f"{where}: a `do` must be followed by the page it lands "
                    "on — the landing check is what proves the move ran"
                )
            assert current_page is not None  # checked by the prefix rule
            moves.append(_parse_do(ctx, where, name, entry, args, current_page, nxt))
        elif kind == "start":
            assert nxt is not None  # `start` sits immediately before a page
            spec = ctx.resolve(entry.get("macro"), where, name)
            moves.append(
                DoNode(
                    id=name,
                    macro=spec.name,
                    args={},
                    enter="",  # unconditional: the start runs from anywhere
                    verify=nxt,
                    on_fail=_on_fail(entry, where),
                )
            )
        elif kind == "agent":
            moves.append(_parse_agent(ctx, where, name, entry, current_page, nxt))
        elif kind == "ask":
            moves.append(_parse_ask(ctx, where, name, entry, current_page))
        elif kind == "select":
            moves.append(_parse_select(ctx, where, name, entry, current_page))
        elif kind == "run":
            moves.append(
                _parse_run(
                    ctx, where, name, entry, args, current_page, nxt, subs[i], moves
                )
            )
        else:  # tell
            message, _ = _entry_message(ctx, where, entry, ctx.payloads)
            moves.append(
                TellNode(id=name, message=message, on_fail=_on_fail(entry, where))
            )
    if len(moves) > MAX_NODES:
        raise PlaybookError(f"too many moves ({len(moves)} > {MAX_NODES})")
    flat = lints.flatten(moves)
    lints.check_money(flat)
    lints.check_resume(flat)
    if _is_boot(ctx):
        lints.check_boot(moves)
    # The route's defaults beneath its waypoints' hands: a waypoint that
    # declares a page's hand replaces the default whole.
    return CompiledRoute(
        nodes=moves,
        start=start,
        recovers=_overlay(_route_defaults(ctx, own), recovers),
        inline=inline,
        prompts_used=frozenset(ctx.prompts_used),
        end=wp_ids[-1] or "",  # "" when the route ends on a move
        payloads=dict(ctx.payloads),
    )


def _sub_playbook(ctx: _Ctx, where: str, name: str) -> Playbook:
    """The playbook a `run` names — `<name>.yml` beside this entry's
    PLAYBOOK.yml, parsed against the same pack by the resolver the pack
    loader wired (a playbook parsed from bare text has none to offer).
    Only an ENTRY runs: what it runs is a file, with no folder for
    playbooks of its own, so a run goes one level deep by shape."""
    if ctx.resolve_playbook is None:
        raise PlaybookError(f"{where}: `run` names a playbook, but none are loaded")
    if run_by(ctx.playbook) is not None:
        raise PlaybookError(
            f"{where}: only an entry runs a playbook — this file is one that "
            f"{playbook_file(ctx.entry)} runs"
        )
    check_name(name, f"{where}: `run`")
    return ctx.resolve_playbook(f"{ctx.playbook}.{name}")


def _parse_run(
    ctx: _Ctx,
    where: str,
    nid: str,
    entry: dict,
    args: dict,
    current_page: str | None,
    nxt: str | None,
    sub: Playbook,
    earlier: list[Node],
) -> RunNode:
    """A `run` move: a playbook beside this entry walked as one move.
    Its frame is derived like a `do`'s — it starts where that playbook
    starts (cold, or on the page before it) and lands on its last page,
    which the route must name next."""
    if not sub.end:
        raise PlaybookError(
            f"{where}: playbook {sub.name!r} ends on a move — a playbook run "
            "as a move must end on a page, the landing the run checks"
        )
    if not sub.self_starting:
        if current_page is None:
            raise PlaybookError(
                f"{where}: playbook {sub.name!r} starts on page {sub.start!r} — "
                "put that page before the run, or give the playbook its own "
                "`start`"
            )
        if current_page != sub.start:
            raise PlaybookError(
                f"{where}: the page before it is {current_page!r}, but playbook "
                f"{sub.name!r} starts on {sub.start!r}"
            )
    if nxt is None:
        raise PlaybookError(
            f"{where}: a `run` must be followed by the page it lands on — "
            f"playbook {sub.name!r} ends on {sub.end!r}"
        )
    if nxt != sub.end:
        raise PlaybookError(
            f"{where}: playbook {sub.name!r} lands on {sub.end!r}, but the "
            f"route continues with page {nxt!r}"
        )
    declared = {inp.name for inp in sub.inputs}
    agents = {n.id for n in earlier if isinstance(n, AgentNode)}
    each: tuple[str, str] | None = None
    raw_each = entry.get("each")
    if raw_each is not None:
        if not (isinstance(raw_each, dict) and len(raw_each) == 1):
            raise PlaybookError(
                f"{where}: `each` is one mapping, `{{<input>: <move>.<field>}}` — "
                "the input each round fills, from a list an earlier agent returned"
            )
        ((inp, ref),) = raw_each.items()
        inp, ref = str(inp), require_str(ref, f"{where}: `each` value")
        if inp not in declared:
            raise PlaybookError(
                f"{where}: `each` fills {inp!r}, which is not an input of "
                f"playbook {sub.name!r}"
            )
        if inp in args:
            raise PlaybookError(f"{where}: {inp!r} is filled by both `with` and `each`")
        check_refs({ref}, ctx.input_names, ctx.payloads, f"{where}: `each`")
        if ref.split(".")[0] not in agents:
            raise PlaybookError(
                f"{where}: `each` iterates a list an EARLIER agent returned "
                f"({{{ref}}} is not one)"
            )
        each = (inp, ref)
        if not sub.self_starting and sub.end != sub.start:
            # Round two enters where round one LANDED. Without a `start`
            # of its own the sub has no way back, so every round after
            # the first fails its enter check — recorded "missed" with
            # nothing ever attempted.
            raise PlaybookError(
                f"{where}: `each` walks {sub.name!r} once per item, and each "
                f"round starts where the last one landed — but it ends on "
                f"{sub.end!r} and starts on {sub.start!r}. Give it a `start`."
            )
    _check_with(
        where,
        args,
        sub.inputs,
        f"playbook {sub.name!r}",
        filled=frozenset({each[0]}) if each else frozenset(),
    )
    miss = _closed_word(entry, "miss", MISS_MODES, where)
    if miss is not None:
        if each is None:
            raise PlaybookError(f"{where}: `miss: skip` goes with `each`")
        if any(
            isinstance(n, (AskNode, TellNode)) or getattr(n, "irreversible", None)
            for n in sub.nodes
        ):
            raise PlaybookError(
                f"{where}: `miss: skip` needs a playbook that never asks, tells or pays "
                f"— {sub.name!r} does; a skipped round must leave nothing owed"
            )
    revise = entry.get("revise")
    if revise is not None:
        revise = require_str(revise, f"{where}: `revise`")
        if revise not in agents:
            raise PlaybookError(
                f"{where}: `revise` names {revise!r}, which is not an EARLIER "
                "agent of this route — a revision re-runs the walk from there"
            )
        if not any(isinstance(n, AskNode) for n in sub.nodes):
            raise PlaybookError(
                f"{where}: `revise` needs an ask inside playbook {sub.name!r} "
                "— it is that ask's uncovered reply that revises"
            )
    raw_limit = _limit_mapping(entry, where, _RUN_LIMIT_KEYS)
    if "rounds" in raw_limit and each is None:
        raise PlaybookError(f"{where}: `limit.rounds` goes with `each`")
    if "revisions" in raw_limit and revise is None:
        raise PlaybookError(f"{where}: `limit.revisions` goes with `revise`")
    max_rounds = _limit_int(
        raw_limit.get("rounds", DEFAULT_RUN_ROUNDS),
        f"{where}: `limit.rounds`",
        1,
        MAX_RUN_ROUNDS,
    )
    revise_limit = _limit_int(
        raw_limit.get("revisions", DEFAULT_REVISIONS),
        f"{where}: `limit.revisions`",
        1,
        MAX_REVISIONS,
    )
    return RunNode(
        id=nid,
        args=args,
        sub=sub,
        enter="" if sub.self_starting else (current_page or ""),
        verify=sub.end,
        each=each,
        miss=miss,
        revise=revise,
        revise_limit=revise_limit,
        max_rounds=max_rounds,
        on_fail=_on_fail(entry, where),
    )


def _shape(
    raw: Any, pack: Pack, own_names: set[str]
) -> tuple[list[tuple[str, str, dict]], list[str | None], str, set[str]]:
    """The route's shape, proved before any move is compiled: a
    non-empty list whose first page is the start contract, at most one
    `start` sitting right before it, only pure-text agents above it,
    and every waypoint's id resolved (`_waypoint_id`, one grammar at
    every door) so a move can read the page after it in one look.
    Returns (classified entries, the waypoint id per entry or None,
    the start page id, the set of page ids on the route)."""
    if not isinstance(raw, list) or not raw:
        raise PlaybookError("`route` must be a non-empty list")
    entries = [_classify_line(i, e) for i, e in enumerate(raw, start=1)]
    first_page = next((i for i, (k, _, _) in enumerate(entries) if k == "page"), None)
    if first_page is None:
        raise PlaybookError(
            "the route needs a page waypoint — the walk's start contract"
        )
    starts = [i for i, (k, _, _) in enumerate(entries) if k == "start"]
    if len(starts) > 1:
        raise PlaybookError("at most one `start` — a route cold-launches once")
    if starts and starts[0] != first_page - 1:
        raise PlaybookError(
            "`start` must sit immediately before the first page — the page "
            "that follows it is the landing it must reach"
        )
    for i in range(first_page):
        kind, _, _ = entries[i]
        # A tell speaks over the channel from any screen; a run up here
        # must open with its playbook's own start (`_parse_run`); an
        # acting agent fails in `_parse_agent`, having no page to start on.
        if kind not in ("agent", "start", "tell", "run"):
            raise PlaybookError(
                f"route line {i + 1}: only pure-text `agent` steps (no "
                "tools), `start`, a `tell` and a self-starting `run` may "
                f"precede the first page — a `{kind}` needs a screen the "
                "route has not reached yet"
            )
    if all(kind == "page" for kind, _, _ in entries):
        raise PlaybookError(
            "the route needs at least one move (start/do/agent/ask/tell)"
        )

    # Waypoint prepass: every page id resolved and its in-place
    # declaration validated up front (`_waypoint_id` — one grammar at
    # every door), so a `do` can read the page that follows it in one
    # forward look. `pages.route_decl` is the one declaration predicate,
    # shared with `collect_page_decls` so the two doors cannot disagree.
    declared_here = own_names | {
        name
        for kind, name, entry in entries
        if kind == "page" and route_decl(entry) is not None
    }
    wp_ids: list[str | None] = []
    page_names: set[str] = set()
    for i, (kind, name, entry) in enumerate(entries):
        if kind != "page":
            wp_ids.append(None)
            continue
        pid = _waypoint_id(i + 1, name, entry, pack, declared_here)
        wp_ids.append(pid)
        page_names.add(pid)
    start = wp_ids[first_page]
    assert start is not None  # first_page indexes a page entry
    return entries, wp_ids, start, page_names


def _classify_line(i: int, entry: Any) -> tuple[str, str, dict]:
    """(kind, name, entry) for one route line — the kind is its leading
    key, the value the name; exactly one kind key, and only that kind's
    field vocabulary beside it."""
    where = f"route line {i}"
    if not isinstance(entry, dict):
        raise PlaybookError(f"{where} must be a mapping")
    kinds = [k for k in LINE_KINDS if k in entry]
    if len(kinds) != 1:
        raise PlaybookError(
            f"{where} must carry exactly one of {', '.join(LINE_KINDS)} "
            f"(got: {', '.join(map(str, sorted(entry))) or '(empty)'})"
        )
    kind = kinds[0]
    unknown = sorted(set(map(str, entry.keys())) - _ENTRY_KEYS[kind])
    if unknown:
        raise PlaybookError(
            f"{where}: unknown key(s) for `{kind}`: {', '.join(unknown)}"
        )
    return kind, require_str(entry.get(kind), f"{where}: `{kind}`"), entry


def _own_pages(raw: Any, pack: Pack) -> dict[str, dict]:
    """A route's `pages:` block — the manifest's page shape, this
    route's ownership — as page name → its recovery fields ({} when it
    declares none): the names seed `_shape`'s declared set, the fields
    are parsed once the route's context exists (`_route_defaults`)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise PlaybookError("`pages` must be a mapping of page name → spec")
    own: dict[str, dict] = {}
    for name, spec in raw.items():
        name = str(name)
        where = f"route page {name!r}"
        if not isinstance(spec, dict):
            raise PlaybookError(f"{where}: spec must be a mapping")
        _check_decl(where, name, decl_fields(spec), pack)
        own[name] = recovery_fields(spec)
    return own


def _check_decl(where: str, page: str, decl: Any, pack: Pack) -> None:
    """A page declaration's CONTENT, checked at the text door
    (`parse_playbook` — tests, tooling) with the page grammar the pack
    door runs via `collect_page_decls`, so a playbook green at one door
    is never red at the other. The pack door already parsed it when the
    pack knows the page."""
    if page in pack.pages:
        return
    try:
        parse_pages_data({page: decl}, pack.app)
    except PagesError as e:
        raise PlaybookError(f"{where}: {e}") from e


def _waypoint_id(pos: int, name: str, entry: dict, pack: Pack, declared: set) -> str:
    """One page waypoint's id. Three spellings: a bare name is a page
    THIS route declares (in its `pages:` block, or beside a waypoint —
    there, and at every later waypoint); `app.pages.<name>` is the manifest's; the
    reserved built-ins stay `ios.<page>` / `channel.<page>` and can only
    be referenced, never declared here. So a bare `page:` with no
    declaration in the file is an error, never a lookup elsewhere."""
    where = f"route line {pos}"
    r = parse_ref(name)
    shared = r is not None and r.shared
    reserved = not shared and "." in name
    page = (
        r.name
        if r is not None and shared
        else name.partition(".")[2]
        if reserved
        else name
    )
    declares = route_decl(entry) is not None
    owner = pack.route_pages.get(page)
    if r is not None and shared and r.kind != PAGES_KIND:
        raise PlaybookError(
            f"{where}: page {name!r} — the manifest's pages are `{app_ref(PAGES_KIND, '<name>')}`"
        )
    if reserved and name.partition(".")[0] not in RESERVED_APPS:
        raise PlaybookError(
            f"{where}: page {name!r} — a waypoint names a page declared in this "
            f"route bare, the manifest's as `{app_ref(PAGES_KIND, '<page>')}`, or a "
            f"reserved namespace ({', '.join(sorted(RESERVED_APPS))}).<page>"
        )
    check_name(page, f"{where}: `page`")
    if "description" in entry and not declares:
        raise PlaybookError(
            f"{where}: a `description:` says what a page IS — it belongs where "
            f"the page is declared, with its anchors"
        )
    if declares and shared:
        raise PlaybookError(
            f"{where}: {name!r} carries a declaration — a page declared beside "
            f"its waypoint is this route's own, written bare: `page: {page}`"
        )
    if declares and reserved:
        raise PlaybookError(
            f"{where}: {name!r} is a reserved built-in — it cannot be declared from a pack"
        )
    if shared and owner is not None:
        hint = (
            f"write `page: {page}`"
            if page in declared
            else f"a route's page is its own — move it to {PACK_FILENAME} to share it"
        )
        raise PlaybookError(
            f"{where}: page {page!r} is declared in {playbook_file(owner)}, "
            f"not in {PACK_FILENAME} — {hint}"
        )
    if shared and page not in pack.pages:
        raise PlaybookError(
            f"{where}: page {page!r} is not declared in {PACK_FILENAME}. "
            f"Declared:{page_menu(pack.shared_pages())}"
        )
    if declares:
        _check_decl(where, page, route_decl(entry), pack)
    if not shared and not reserved and not declares and page not in declared:
        raise PlaybookError(
            f"{where}: page {page!r} is not declared in this route — declare it "
            f"in the route's `pages:` block or beside a waypoint (anchors under `page:`)"
            + _pack_hint(PAGES_KIND, page, page in pack.pages and owner is None)
        )
    return name if reserved else page


# ---------- shared field rules ----------


def _unique_list(raw: Any, where: str, check: Callable[[Any], _T]) -> list[_T]:
    """A list of distinct entries, each validated (and normalized) by
    `check` — the one shape `tools`, `give`, and the reply words share."""
    if not isinstance(raw, list):
        raise PlaybookError(f"{where} must be a list")
    out: list[_T] = []
    for item in raw:
        value = check(item)
        if value in out:
            raise PlaybookError(f"{where}: duplicate entry {value!r}")
        out.append(value)
    return out


def _closed_word(entry: dict, key: str, allowed: tuple[str, ...], where: str) -> Any:
    """An optional key whose value is one word of a closed vocabulary —
    absent stays None."""
    word = entry.get(key)
    if word is not None and word not in allowed:
        raise PlaybookError(
            f"{where}: `{key}` must be one of {', '.join(allowed)} (got {word!r})"
        )
    return word


def _on_fail(entry: dict, where: str) -> str | None:
    """An entry's optional `on_fail:` — stop or handover once its own
    means are spent; absent = handover."""
    return _closed_word(entry, "on_fail", ON_FAIL_MODES, where)


def _declared_once(
    prior: Recovery | None, declared: Recovery, where: str, page: str
) -> Recovery:
    """A page's recovery, declared at every waypoint of that page: a
    later waypoint may add what an earlier left unsaid (a hand, an
    `on_fail`), never contradict it."""
    if prior is None:
        return declared
    if declared.hands and prior.hands and _hands_of(declared) != _hands_of(prior):
        raise PlaybookError(
            f"{where}: page {page!r} declares `recover` twice with different "
            "hands — declare it once"
        )
    if (
        declared.on_fail is not None
        and prior.on_fail is not None
        and declared.on_fail != prior.on_fail
    ):
        raise PlaybookError(
            f"{where}: page {page!r} declares `on_fail` twice with different "
            "words — declare it once"
        )
    return _overlay_one(prior, declared)


def _hands_of(r: Recovery) -> tuple:
    return (r.covered, r.elsewhere, r.locked, r.tries)


def _overlay_one(base: Recovery, over: Recovery) -> Recovery:
    """`over`'s hands where it declares any (else `base`'s), and its
    `on_fail` where said."""
    hands = over if over.hands else base
    return replace(
        hands, on_fail=over.on_fail if over.on_fail is not None else base.on_fail
    )


def _overlay(
    base: dict[str, Recovery], over: dict[str, Recovery]
) -> dict[str, Recovery]:
    """The manifest's page recovery under the route's own: a route
    inherits a shared page's hands and `on_fail` unless it declares its
    own."""
    out = dict(base)
    for page, r in over.items():
        out[page] = _overlay_one(base[page], r) if page in base else r
    return out


def _guard_grants(
    ctx: _Ctx,
    where: str,
    never_tap: tuple[NeverTap, ...],
    give: tuple[str, ...],
    granted: tuple[Macro, ...],
) -> None:
    """Refuse a grant that walks around this episode's `never_tap:`.

    Both contradictions are declared by name, so both belong here rather
    than at run time. A granted LANDMARK is a box the model may press
    blind, and a landmark with no text row leaves the runtime check
    nothing to find. A granted MACRO presses its own recorded targets
    without ever proposing a tap — its labels are refused here, and its
    boxes at run time against the live screen (`step_agent.macro_refusal`,
    off the same `Macro.taps`), since a label is the author's word and
    the same coordinates under another word tap the same button.
    Node-scoped on purpose: the payment move's macro presses these same
    buttons by design and stays legal."""
    if not never_tap:
        return

    def _named(labels: tuple[str, ...]) -> str | None:
        """Which target these recorded labels name, by the rule the
        RUNTIME reads a screen row with (`match.label_matches`) — not
        exact equality. The natural thing to write is the label as the
        listing shows it, price and all ("Pay now ($3.60)"), which is
        not equal to the target and is the same button."""
        written = [normalize(w) for w in labels]
        for target in never_tap:
            if any(
                match.label_matches(normalize(reading), w, ())
                for reading in target.label
                for w in written
            ):
                return " / ".join(target.label)
        return None

    for name in give:
        hit = _named(ctx.pack.landmarks[name].label)
        if hit is not None:
            raise PlaybookError(
                f"{where}: `give` grants landmark {name!r}, which this step "
                f"declares never_tap ({hit}) — the grant would hand the model "
                "the box the guard exists to refuse"
            )
    for macro in granted:
        for tap in macro.taps():
            hit = _named(tap.label)
            if hit is not None:
                raise PlaybookError(
                    f"{where}: `give` grants macro {macro.name!r}, which presses "
                    f"{hit} — this step declares that never_tap, and a macro "
                    "runs its recorded steps without proposing a tap"
                )
            if any(isinstance(v, str) for v in tap.bbox):
                # A box the run fills from an input default is one the
                # runtime guard cannot judge against the screen (a
                # placeholder has no centre); under a never_tap it is
                # refused here rather than let through unjudged.
                raise PlaybookError(
                    f"{where}: `give` grants macro {macro.name!r}, whose "
                    f"{tap.label[0]!r} box carries a placeholder — this step "
                    "declares never_tap, and a box filled at run time cannot "
                    "be judged against it; record the box"
                )


def _never_tap(entry: dict, where: str) -> tuple[NeverTap, ...]:
    """`never_tap:` — the targets an episode's taps may never land on.
    Each item is a reading, alternate readings of ONE target, or a
    mapping with `label:` and an optional `within:` band; the readings
    grammar is the one every other target list uses."""
    raw = entry.get("never_tap")
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise PlaybookError(f"{where}: `never_tap` takes a non-empty LIST")
    if len(raw) > MAX_NEVER_TAP:
        raise PlaybookError(f"{where}: at most {MAX_NEVER_TAP} `never_tap` targets")
    out: list[NeverTap] = []
    for i, item in enumerate(raw):
        at = f"{where}: `never_tap[{i}]`"
        spec = item if isinstance(item, dict) else {"label": item}
        extra = set(spec) - {"label", "within"}
        if extra:
            raise PlaybookError(f"{at}: unknown key(s): {', '.join(sorted(extra))}")
        label = checked_readings(spec, at, require_str, PlaybookError, key="label")
        try:
            within = parse_within(spec["within"]) if "within" in spec else None
        except (ValueError, TypeError) as e:
            # `parse_within` raises a bare ValueError; every spec parser
            # wraps it, or `playbooks check` prints a traceback instead
            # of a located message.
            raise PlaybookError(f"{at}: `within` {e}") from e
        out.append(NeverTap(label=label, within=within))
    return tuple(out)


def _think_level(entry: dict, where: str) -> Thinking | None:
    """A model step's optional `think:` — how much hidden thinking its
    calls ask for; absent leaves the vendor's default (`playbooks
    check` says so, since a thinking model's default is minutes per
    call)."""
    think: Thinking | None = _closed_word(entry, "think", THINKING_LEVELS, where)
    return think


def _irreversible_class(entry: dict, where: str) -> str | None:
    """A move's optional `irreversible:` class — the same closed
    vocabulary on a `do` and an `agent`."""
    irreversible: str | None = _closed_word(
        entry, "irreversible", IRREVERSIBLE_CLASSES, where
    )
    return irreversible


def _limit_int(value: Any, where: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise PlaybookError(f"{where} must be {lo}–{hi} (got {value!r})")
    return value


def _entry_message(
    ctx: _Ctx,
    where: str,
    entry: dict,
    payloads: dict[str, tuple[str, ...]],
    key: str = "message",
) -> tuple[str, set[str]]:
    """A REQUIRED authored `message:` (or an ask's `denied:`, the same
    shape under `key`) — the exact text sent to the user; only the
    author knows the user's language, so the conductor composes no prose
    around it. Refs held to the same defined-before-use rules as `with:`
    values; returned with them so the ask lints can inspect."""
    text = prose(entry.get(key), f"{where}: `{key}`", lines=MAX_MESSAGE_LINES)
    refs = refs_in(text, f"{where}: `{key}`")
    check_refs(refs, ctx.input_names, payloads, f"{where}: `{key}`")
    return text, refs


def _reply_words(entry: dict, key: str, where: str) -> list[str]:
    """An ask's `yes:` / `no:` — the whole-message replies it reads, a
    non-empty list of distinct strings, stored in `reply.normalize`
    space so every reader compares without re-normalizing."""
    out = _unique_list(
        entry.get(key),
        f"{where}: `{key}`",
        lambda w: reply.normalize(require_str(w, f"{where}: `{key}` entry")),
    )
    if not out:
        raise PlaybookError(f"{where}: `{key}` must list at least one reply word")
    return out


def _context_entries(entry: dict, where: str) -> list[str]:
    """An agent's `context:` — the sources it loads beside its prompt."""

    def _one(item: Any) -> str:
        bad = context.check_entry(item)
        if bad is not None:
            raise PlaybookError(f"{where}: `context` entry {item!r} {bad}")
        return item

    return _unique_list(entry.get("context", []), f"{where}: `context`", _one)


def _landmark_name(ctx: _Ctx, value: Any, where: str) -> str:
    """An `app.landmarks.<name>` reference resolved to its bare name —
    ONE spelling for `give:` entries and a recover hand's `tap:`. The name
    half rides the shared name grammar (`check_name`), so a landmark
    reference can never drift from the section's own naming rule."""
    r = parse_ref(value) if isinstance(value, str) else None
    if r is None or not r.is_app(LANDMARKS_KIND):
        raise PlaybookError(
            f"{where}: {value!r} must look like `{app_ref(LANDMARKS_KIND, '<name>')}`"
        )
    name = r.name
    check_name(name, where)
    if name not in ctx.pack.landmarks:
        known = ", ".join(sorted(ctx.pack.landmarks)) or "(none)"
        raise PlaybookError(
            f"{where}: names landmark {name!r} — not declared under "
            f"`landmarks`. Declared: {known}"
        )
    return name


@dataclass(frozen=True)
class _Grant:
    """One resolved `give:` entry. Two grants are the same grant by root
    and name (`_unique_list`); a macro's resolved body rides beside its
    name for the grant guard and is neither compared nor printed."""

    root: str
    name: str
    macro: Macro | None = field(default=None, compare=False, repr=False)


def _grant(ctx: _Ctx, value: Any, where: str, nid: str) -> _Grant:
    """One `give:` entry: an `app.landmarks.<name>` the episode may tap
    blind, or a macro it may run — `macros.<name>` for this route's own
    hand, `app.macros.<name>` for the pack's — argument-less, like every
    helper hand, and never spelled like a fixed answer (`done`,
    `escalate`, a verb) that the episode legend already owns. A macro's
    name is its dispatch name (`_argless_macro`)."""
    r = parse_ref(value) if isinstance(value, str) else None
    own = r is not None and not r.shared and r.kind == MACROS_KIND
    shared = r is not None and (r.is_app(LANDMARKS_KIND) or r.is_app(MACROS_KIND))
    if r is None or not (own or shared):
        raise PlaybookError(
            f"{where}: {value!r} must look like `macros.<name>` (this route's own "
            f"hand), `{app_ref(MACROS_KIND, '<name>')}` or "
            f"`{app_ref(LANDMARKS_KIND, '<name>')}` (the pack's)"
        )
    if r.name in RESERVED_KEYS:
        raise PlaybookError(
            f"{where}: {r.name!r} is a fixed episode answer — a granted "
            "landmark or macro cannot be spelled like one"
        )
    if r.kind == LANDMARKS_KIND:
        return _Grant(LANDMARKS_KIND, _landmark_name(ctx, value, where))
    macro = _argless_macro(value if shared else r.name, "give", where, nid, ctx.resolve)
    return _Grant(MACROS_KIND, macro.name, macro)


# ---------- macros ----------


def _dispatch(*parts: str) -> str:
    """One dot-joined dispatch name, the macro namespace's one speller:
    `<entry>.<file>` for a recorded hand, `<playbook>.<move>[.<role>]`
    for an inline body (a run-only playbook's id carries its entry, so
    its bodies can never claim the entry's name). `check_name` rejects
    dots, so neither can collide with a pack macro."""
    return ".".join(parts)


def _local_registry(entry: str, pack: Pack, local: Scanned[Macro]) -> dict[str, Macro]:
    """The route's inline registry, opened with the entry's recorded
    hands — the playbooks it runs share them: each
    `<entry>/macros/<name>.yml` dispatches as `<entry>.<name>` — an
    inline body written down — referenced or not (a stepping tool, an
    agent's `give:` may name it). A pack hand of the same name is no
    clash: a bare reference is always the file beside this one, the
    pack's is `app.macros.<name>`."""
    return {
        _dispatch(entry, name): replace(spec, name=_dispatch(entry, name))
        for name, spec in local.ok.items()
    }


def _macro_resolver(
    playbook: str, pack: Pack, inline: dict[str, Macro], local: Scanned[Macro]
) -> _MacroResolve:
    """The name-or-inline resolution every macro-carrying slot shares —
    a do's `macro:`, an ask's `resume:`, a page's `recover:`. ONE home
    for the whole idiom: the synthesized-name rule
    (`<playbook>.<name>[.<role>]`, dot-joined so it can never collide
    with a pack macro — `check_name` rejects dots), the MacroError
    framing, the inline registry, and the file validation (a broken
    macro reports its cause, an unknown one lists what exists) — so
    the slots can never drift. Returns the resolved Macro; its `.name`
    is the dispatch name either way. A bare name is the file beside
    this one — the ENTRY's `macros/`, shared by the playbooks it runs
    (already in `inline`, dispatch `app/<entry>.<name>`) — and
    `app.macros.<name>` one of the pack's (dispatch `app/<name>`). An
    inline body is named for the file that writes it
    (`app/<entry>.<playbook>.<move>`), so one can never claim the
    entry's name."""

    entry = entry_of(playbook)
    folder = f"{entry}/{PACK_MACROS_DIRNAME}/"
    page_of = page_resolver(pack.app, pack.pages, pack.prints)
    pack_macro = macro_resolver(pack.macros, pack.macro_errors)

    def macro_of(ref: str) -> Macro:
        """The one lookup a `do:`, a hand, a grant and an inline body's
        `run:` share."""
        r = parse_ref(ref)
        if r is not None and r.shared:
            return pack_macro(ref)  # `app.macros.<name>`, or its own refusal
        if ref in local.errors:
            raise MacroError(
                f"macro {ref!r} ({folder}{ref}.yml) is invalid: {local.errors[ref]}"
            )
        if ref in local.ok:
            return inline[_dispatch(entry, ref)]
        raise MacroError(
            _not_here("macro", ref, folder, local.ok)
            + _pack_hint(
                MACROS_KIND, ref, ref in pack.macros or ref in pack.macro_errors
            )
        )

    def resolve(raw: Any, where: str, nid: str, role: str | None = None) -> Macro:
        slot = role or "macro"
        if isinstance(raw, dict):
            mname = _dispatch(playbook, nid, *([role] if role else []))
            if mname in inline:
                raise PlaybookError(
                    f"{where}: inline `{slot}` would be named {mname!r}, which "
                    f"the recorded {folder}{nid}.yml already holds — rename one"
                )
            try:
                spec = parse_inline_macro(raw, mname, page_of, macro_of)
            except MacroError as e:
                raise PlaybookError(f"{where}: inline `{slot}`: {e}") from e
            inline[mname] = spec
            return spec
        if raw is not None and not isinstance(raw, str):
            raise PlaybookError(
                f"{where}: `{slot}` must be a macro name or an inline "
                "mapping with `steps:`"
            )
        mname = require_str(raw, f"{where}: `{slot}`")
        try:
            return macro_of(mname)
        except MacroError as e:
            raise PlaybookError(f"{where}: {slot} {e}") from e

    return resolve


def _not_here(kind: str, name: str, folder: str, have: "Mapping[str, Any]") -> str:
    """The one wording of "no such file beside this one", listing what is."""
    listed = ", ".join(sorted(have)) or "(none)"
    return f"no {kind} {name!r} in {folder} ({listed})"


def _pack_hint(kind: str, name: str, exists: bool) -> str:
    """The clause that names the pack's spelling when a bare name only
    the pack holds was written — "" otherwise."""
    return f" — the pack's is `{app_ref(kind, name)}`" if exists else ""


def _prompt_text(ctx: _Ctx, raw: str, where: str) -> str:
    """An agent's `prompt:` — the prose itself, `prompts.<name>` for this
    route's `prompts/<name>.md`, or `app.prompts.<name>` for the
    pack's. Resolved here, at parse, so the node carries text either
    way and nothing downstream knows which form the author chose."""
    ref = raw.strip()
    r = parse_ref(ref) if not any(c.isspace() for c in ref) else None
    if r is None or r.kind != PROMPTS_KIND:
        return raw
    local = not r.shared
    files = ctx.prompts_local if local else ctx.prompts_pack
    folder = (
        f"{ctx.entry}/{PACK_PROMPTS_DIRNAME}/" if local else f"{PACK_PROMPTS_DIRNAME}/"
    )
    name = r.name
    check_name(name, f"{where}: `prompt` ({ref})")
    if name in files.errors:
        raise PlaybookError(
            f"{where}: `prompt` names {ref}, and {folder}{name}{PROMPT_SUFFIX} is "
            f"invalid: {files.errors[name]}"
        )
    if name not in files.ok:
        raise PlaybookError(
            f"{where}: `prompt` names {ref}, but "
            + _not_here("file", f"{name}{PROMPT_SUFFIX}", folder, files.ok)
            + _pack_hint(PROMPTS_KIND, name, local and name in ctx.prompts_pack.ok)
        )
    ctx.prompts_used.add(f"{folder}{name}{PROMPT_SUFFIX}")
    return files.ok[name]


def _argless_macro(
    raw: Any, key: str, where: str, nid: str, resolve: _MacroResolve
) -> Macro:
    """Resolve one argument-less pack macro for a helper-hand slot
    (`resume:`, `recover:`, an agent's `give:`). The slot may wrap its
    body one level (`{macro: ...}`) — unwrapped HERE, the rule's one
    home, so the resolver sees the same shapes a `do` does. All roles
    dispatch with no arguments, so a required input could only abort at
    run time — right after a confirmed ask, at the worst moment — hence
    the lint. Returns the resolved Macro: its `.name` is the dispatch
    name every slot stores (a playbook-local one reads
    `<playbook>.<name>`), and a grant's guard reads its taps."""
    if isinstance(raw, dict) and set(raw) == {"macro"}:
        raw = raw["macro"]
    spec = resolve(raw, where, nid, key)
    required = sorted(i.name for i in spec.inputs if i.required)
    if required:
        raise PlaybookError(
            f"{where}: `{key}` macro {spec.name!r} requires input(s) "
            f"{', '.join(required)} — the walk dispatches {key} with no "
            "arguments"
        )
    return spec


# ---------- the moves ----------


def _check_with(
    where: str,
    args: dict,
    inputs: "tuple[MacroInput, ...]",
    what: str,
    filled: frozenset[str] = frozenset(),
) -> None:
    """A move's `with:` against the inputs of what it runs (a macro, a
    playbook): every key an input, every required input filled — by
    `with`, or by the one `each` fills (`filled`)."""
    declared = {inp.name: inp for inp in inputs}
    unknown = sorted(set(args.keys()) - set(declared))
    if unknown:
        raise PlaybookError(
            f"{where}: `with` key(s) {', '.join(unknown)} are not inputs of "
            f"{what} (declares: {', '.join(sorted(declared)) or '(none)'})"
        )
    missing = sorted(
        n
        for n, i in declared.items()
        if i.required and n not in args and n not in filled
    )
    if missing:
        raise PlaybookError(
            f"{where}: {what} requires input(s) {', '.join(missing)} — "
            "supply them under `with`" + (" (or one with `each:`)" if filled else "")
        )


def _limit_mapping(entry: dict, where: str, keys: set[str]) -> dict:
    """An entry's optional `limit:` mapping, its keys held to `keys` —
    empty when absent; each caller reads its own bounds off it."""
    raw = entry.get("limit")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise PlaybookError(f"{where}: `limit` must be a mapping")
    unknown = sorted(set(map(str, raw)) - keys)
    if unknown:
        raise PlaybookError(f"{where}: `limit`: unknown key(s): {', '.join(unknown)}")
    return raw


def _parse_do(
    ctx: _Ctx,
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
    _check_with(where, args, spec.inputs, f"macro {macro!r}")
    return DoNode(
        id=nid,
        macro=macro,
        args=dict(args),
        enter=enter,
        verify=verify,
        irreversible=_irreversible_class(entry, where),
        on_fail=_on_fail(entry, where),
    )


def _parse_agent(
    ctx: _Ctx,
    where: str,
    nid: str,
    entry: dict,
    current_page: str | None,
    next_wp: str | None,
) -> AgentNode:
    """An `agent` move. No `tools` = a pure-text call (needs `returns`,
    no pages); tools = an acting episode framed by the adjacent
    waypoints exactly like a `do`. The prompt is the author's whole
    brief — refs validated here, filled once when the step opens; it
    may quote the step's own returns (its last answer, empty the first
    time — what a revision re-reads), so they are declared before it."""

    def _tool(t: Any) -> str:
        if not isinstance(t, str) or t not in AGENT_TOOLS:
            raise PlaybookError(
                f"{where}: tool {t!r} — the episode vocabulary is "
                f"{', '.join(AGENT_TOOLS)}"
            )
        return t

    tools = _unique_list(entry.get("tools", []), f"{where}: `tools`", _tool)
    if not tools:
        # Everything below `tools` is about the screen an episode acts
        # on; on a pure-text call it is dead config.
        for key in ("give", "irreversible", "limit"):
            if key in entry:
                raise PlaybookError(
                    f"{where}: `{key}` is for acting episodes — a pure-text "
                    "call has no screen"
                )
    grants = _unique_list(
        entry.get("give", []),
        f"{where}: `give`",
        lambda g: _grant(ctx, g, f"{where}: `give` entry", nid),
    )
    give = tuple(g.name for g in grants if g.root == LANDMARKS_KIND)
    granted = tuple(g.macro for g in grants if g.macro is not None)
    macros = tuple(m.name for m in granted)
    shared = sorted(set(give) & set(macros))
    if shared:
        raise PlaybookError(
            f"{where}: `give` names {', '.join(shared)} as both a landmark and "
            "a macro — the model answers by name, so the two must differ"
        )
    never_tap = _never_tap(entry, where)
    # A granted macro presses its own recorded boxes without ever
    # proposing a tap, so `never_tap` has something to guard on a
    # macro-only episode too (`_guard_grants` at parse, and
    # `step_agent.macro_refusal` on the live screen).
    if never_tap and "tap" not in tools and not granted:
        raise PlaybookError(
            f"{where}: `never_tap` guards what this episode presses, but it has "
            "neither a `tap` tool nor a granted macro — grant one or drop the "
            "targets"
        )
    _guard_grants(ctx, where, never_tap, give, granted)
    if give and "tap" not in tools:
        raise PlaybookError(
            f"{where}: `give` grants landmarks, but without `tap` the episode "
            "cannot press one — grant `tap` or drop the landmarks"
        )

    raw_returns = entry.get("returns")
    returns: list[tuple[str, str]] = []
    if raw_returns is not None:
        if not isinstance(raw_returns, dict) or not raw_returns:
            raise PlaybookError(
                f"{where}: `returns` must be a mapping of field → description"
            )
        if len(raw_returns) > MAX_RETURNS:
            raise PlaybookError(
                f"{where}: {len(raw_returns)} return fields > max {MAX_RETURNS}"
            )
        for fname, desc in raw_returns.items():
            field_name(fname, f"{where}: return field")
            if fname in CONTRACT_FIELDS:
                raise PlaybookError(
                    f"{where}: return field {fname!r} is one of the reply "
                    f"contract's own fields ({', '.join(sorted(CONTRACT_FIELDS))})"
                    " — rename it"
                )
            returns.append((fname, prose(desc, f"{where}: `returns.{fname}`")))
    ctx.payloads[nid] = tuple(f for f, _ in returns)
    prompt = _prompt_text(
        ctx, require_str(entry.get("prompt"), f"{where}: `prompt`"), where
    )
    if len(prompt) > MAX_PROMPT_LEN:
        raise PlaybookError(
            f"{where}: `prompt` is {len(prompt)} characters (max {MAX_PROMPT_LEN})"
        )

    if not tools and not returns:
        raise PlaybookError(
            f"{where}: an agent with neither `tools` nor `returns` can do "
            "nothing — give it hands, fields to fill, or both"
        )
    irreversible = _irreversible_class(entry, where)

    enter = verify = ""
    if tools:
        if current_page is None:
            raise PlaybookError(
                f"{where}: an acting agent needs the page it starts on — "
                "put a page waypoint before it"
            )
        if next_wp is None:
            raise PlaybookError(
                f"{where}: an acting agent must be followed by the page it "
                "finishes on — the landing check is its exit contract"
            )
        if "." in current_page or "." in next_wp:
            raise PlaybookError(
                f"{where}: an agent episode runs on this pack's own pages — "
                "reserved built-ins cannot frame it"
            )
        enter, verify = current_page, next_wp

    raw_limit = _limit_mapping(entry, where, _AGENT_LIMIT_KEYS)
    max_calls = _limit_int(
        raw_limit.get("calls", DEFAULT_AGENT_CALLS),
        f"{where}: `limit.calls`",
        1,
        MAX_AGENT_CALLS,
    )
    max_scrolls = _limit_int(
        raw_limit.get("scrolls", min(DEFAULT_AGENT_SCROLLS, max_calls)),
        f"{where}: `limit.scrolls`",
        0,
        MAX_AGENT_CALLS,
    )
    if "scroll" not in tools:
        max_scrolls = 0
    elif max_scrolls == 0:
        raise PlaybookError(
            f"{where}: `scroll` is granted but `limit.scrolls` is 0 — the "
            "first scroll would hand over; raise it or drop the tool"
        )

    g_payloads = (
        ctx.payloads_with_total() if irreversible == "payment" else ctx.payloads
    )
    check_refs(
        refs_in(prompt, f"{where}: `prompt`"),
        ctx.input_names,
        g_payloads,
        f"{where}: `prompt`",
    )

    return AgentNode(
        id=nid,
        prompt=prompt,
        tools=tuple(tools),
        give=give,
        never_tap=never_tap,
        returns=tuple(returns),
        enter=enter,
        verify=verify,
        max_calls=max_calls,
        max_scrolls=max_scrolls,
        irreversible=irreversible,
        context=tuple(_context_entries(entry, where)),
        macros=macros,
        think=_think_level(entry, where),
        on_fail=_on_fail(entry, where),
    )


def _parse_ask(
    ctx: _Ctx, where: str, nid: str, entry: dict, current_page: str | None
) -> AskNode:
    approve = require_str(entry.get("approve"), f"{where}: `approve`")
    check_name(approve, f"{where}: `approve`")
    if approve == "payment":
        if current_page is None or "." in current_page:
            raise PlaybookError(
                f"{where}: a payment ask reads its total off the page before "
                "it — put the sheet's page waypoint immediately before the ask"
            )
    # A payment ask may quote the consent slot (a move literally named
    # `ask` is shadowed in this message — the money slot wins, both at
    # parse and at fill).
    g_payloads = ctx.payloads_with_total() if approve == "payment" else ctx.payloads
    message, msg_refs = _entry_message(ctx, where, entry, g_payloads)
    total: tuple[str, ...] = ()
    if approve == "payment":
        if "ask.total" not in msg_refs:
            raise PlaybookError(
                f"{where}: a payment ask's `message` must quote the sheet "
                "total — reference {ask.total} (the ask IS the consent record)"
            )
        if "total_label" not in entry:
            raise PlaybookError(
                f"{where}: a payment ask declares `total_label:` — the label "
                "the sheet total sits beside, read off that row only"
            )
        total = checked_readings(
            entry, where, require_str, PlaybookError, key="total_label"
        )
    elif "total_label" in entry:
        raise PlaybookError(f"{where}: `total_label` goes with `approve: payment`")
    # The answer to a no quotes what the ask could (the total included).
    denied = None
    if entry.get("denied") is not None:
        denied, _ = _entry_message(ctx, where, entry, g_payloads, key="denied")
    wait_seconds, rounds = _ask_wait(entry, where)
    resume = None
    if entry.get("resume") is not None:
        resume = _argless_macro(entry["resume"], "resume", where, nid, ctx.resolve).name
    return AskNode(
        id=nid,
        approve=approve,
        message=message,
        yes=tuple(_reply_words(entry, "yes", where)),
        no=tuple(_reply_words(entry, "no", where)),
        denied=denied,
        resume=resume,
        enter=current_page or "",
        total_label=total,
        wait_seconds=wait_seconds,
        silence_rounds=rounds,
        think=_think_level(entry, where),
        on_fail=_on_fail(entry, where),
    )


def _parse_select(
    ctx: _Ctx, where: str, nid: str, entry: dict, current_page: str | None
) -> ActivateNode:
    """The `select` step — the channel boot's own: read the thread and
    select the playbook it asks for. It reads the user's thread, so the
    thread page must sit immediately before it (its place at the route's
    end is `lints.check_boot`'s rule). `limit: {scrolls}` bounds
    parse_task's scroll-for-history escape."""
    if not _is_boot(ctx):
        raise PlaybookError(
            f"{where}: `select` is the channel boot's own step — it belongs "
            f"in {CHANNEL_APP}/{BOOT_PLAYBOOK}/PLAYBOOK.yml only"
        )
    if current_page != THREAD_PAGE:
        raise PlaybookError(
            f"{where}: `select` reads the user's thread — put the "
            f"`{THREAD_PAGE}` page waypoint immediately before it"
        )
    raw_limit = entry.get("limit")
    max_scrolls = DEFAULT_BOOT_SCROLLS
    if raw_limit is not None:
        if not isinstance(raw_limit, dict) or set(map(str, raw_limit)) - {"scrolls"}:
            raise PlaybookError(f"{where}: `limit` takes only `scrolls`")
        max_scrolls = _limit_int(
            raw_limit.get("scrolls", DEFAULT_BOOT_SCROLLS),
            f"{where}: `limit.scrolls`",
            0,
            MAX_AGENT_CALLS,
        )
    return ActivateNode(
        id=nid,
        enter=current_page,
        max_scrolls=max_scrolls,
        think=_think_level(entry, where),
    )


def _is_boot(ctx: _Ctx) -> bool:
    """Whether this route is the channel pack's boot playbook — the one
    file the `select` step is admitted in."""
    return ctx.pack.app == CHANNEL_APP and ctx.playbook == BOOT_PLAYBOOK


def _ask_wait(entry: dict, where: str) -> tuple[int, int]:
    """An ask's patience — `wait:` seconds between reply polls and
    `rounds:` silent polls before the session suspends; both optional,
    defaults visible in the scaffold; bounded by the engine's single-sleep
    cap and a sane number of rounds."""
    seconds = _limit_int(
        entry.get("wait", DEFAULT_ASK_WAIT_SECONDS),
        f"{where}: `wait`",
        MIN_ASK_WAIT_SECONDS,
        MAX_ASK_WAIT_SECONDS,
    )
    rounds = _limit_int(
        entry.get("rounds", DEFAULT_ASK_ROUNDS), f"{where}: `rounds`", 1, MAX_ASK_ROUNDS
    )
    return seconds, rounds


def _route_defaults(ctx: _Ctx, own: dict[str, dict]) -> dict[str, Recovery]:
    """What the route's waypoints start from: the manifest's hands every
    route inherits (resolved once, at pack load — `manifest_recovers`)
    under the hands its own `pages:` block declares, parsed with the
    route's own resolver."""
    return _overlay(dict(ctx.pack.recovers), _page_hands(ctx, own, "route"))


def _page_hands(ctx: _Ctx, raw: dict[str, dict], site: str) -> dict[str, Recovery]:
    """The hands a `pages:` mapping declares, by page — the
    `recovery_fields` slice of each, the pages declaring none skipped."""
    return {
        name: _parse_recover(ctx, fields, f"{site} page {name!r}", name)
        for name, fields in raw.items()
        if fields
    }


def manifest_recovers(pack: Pack, raw: dict[str, dict]) -> dict[str, Recovery]:
    """The manifest's `pages: <name>: recover:` hands, resolved with the
    pack's own resolver — `app.macros.<name>` and `app.landmarks.<name>`,
    the one spelling everywhere. The one hand grammar
    (`_parse_recover`), the pack's one resolver. A manifest
    carries settings, never bodies: an inline `macro: {steps: ...}` is
    refused. Raises PlaybookError naming the page."""
    pack_macro = macro_resolver(pack.macros, pack.macro_errors)

    def resolve(value: Any, where: str, nid: str, role: str | None = None) -> Macro:
        try:
            return pack_macro(require_str(value, f"{where}: `{role or 'macro'}`"))
        except MacroError as e:
            raise PlaybookError(f"{where}: {role or 'macro'} {e}") from e

    for name, spec in raw.items():
        _refuse_bodies(spec.get("recover"), f"manifest page {name!r}")
    return _page_hands(_Ctx("", pack, set(), resolve), raw, "manifest")


def _refuse_bodies(raw: Any, where: str) -> None:
    """A manifest hand names a pack macro; it never embeds one."""
    if not isinstance(raw, dict):
        return
    if isinstance(raw.get("macro"), dict):
        raise PlaybookError(
            f"{where}: the manifest names a pack macro for a recover hand — "
            "record the body as macros/<name>.yml and name it here"
        )
    for reading in RECOVER_READINGS:
        _refuse_bodies(raw.get(reading), where)


def _parse_recover(ctx: _Ctx, fields: dict, where: str, page: str) -> Recovery:
    """A page's `recover:`, `tries:` and `on_fail:` (the
    `PAGE_RECOVERY_FIELDS` slice of its mapping, in a route waypoint or
    the manifest alike). `recover:` is one hand for any deviation (a
    bare gesture, `{tap: app.landmarks.<name>}`, or `{macro: app.macros.<name>}`), or
    one per reading (`covered:` the page itself under a sheet or popup,
    `locked:` the phone's lock screen, `elsewhere:` any other screen);
    `tries:` beside it is the page's own bound under the walk-wide
    ceiling, and means nothing without a hand to count; `on_fail:` is
    the page's word once they are spent — legal on its own."""
    on_fail = _on_fail(fields, where)
    if "recover" not in fields:
        if "tries" in fields:
            raise PlaybookError(
                f"{where}: `tries` bounds a `recover:` — declare the hand it counts"
            )
        return Recovery(on_fail=on_fail)
    raw = fields["recover"]
    tries = _limit_int(
        fields.get("tries", DEFAULT_RECOVER_LIMIT),
        f"{where}: `tries`",
        1,
        MAX_RECOVER_ACTIONS,
    )
    if isinstance(raw, dict):
        unknown = sorted(set(map(str, raw)) - _RECOVER_KEYS)
        if unknown:
            hint = (
                " — `tries` sits beside `recover`, not inside"
                if "tries" in unknown
                else ""
            )
            raise PlaybookError(
                f"{where}: `recover`: unknown key(s): {', '.join(unknown)}{hint}"
            )
        keyed = [k for k in RECOVER_READINGS if k in raw]
        if keyed:
            if set(raw) - set(keyed):
                raise PlaybookError(
                    f"{where}: `recover` declares one hand OR one per reading "
                    f"({', '.join(RECOVER_READINGS)}), not both"
                )
            hands = {
                k: _parse_hand(ctx, raw[k], f"{where}: `recover.{k}`", page)
                for k in keyed
            }
            return Recovery(
                covered=hands.get(READING_COVERED),
                elsewhere=hands.get(READING_ELSEWHERE),
                locked=hands.get(READING_LOCKED),
                tries=tries,
                on_fail=on_fail,
            )
    # A bare gesture, `{tap: ...}` / `{macro: ...}`, or a non-hand — the
    # one hand parser judges the shape and names the alternatives.
    hand = _parse_hand(ctx, raw, f"{where}: `recover`", page)
    return Recovery(
        covered=hand, elsewhere=hand, locked=hand, tries=tries, on_fail=on_fail
    )


def _parse_hand(ctx: _Ctx, raw: Any, where: str, page: str) -> RecoverHand:
    """One recovery hand, in a step's shape: a bare gesture that takes no
    object (`go_back`), `{tap: app.landmarks.<name>}` (the declared spot,
    as declared), or `{macro: app.macros.<name>}` (argument-less)."""
    if isinstance(raw, str):
        if raw == "tap":
            raise PlaybookError(
                f"{where}: a tap names its spot — `{{tap: app.landmarks.<name>}}`"
            )
        if raw not in BARE_HANDS:
            raise PlaybookError(
                f"{where}: {raw!r} is not a hand — one of {', '.join(BARE_HANDS)}, "
                "{tap: app.landmarks.<name>}, or {macro: app.macros.<name>}"
            )
        return RecoverHand(tool=raw)
    if not isinstance(raw, dict) or len(raw) != 1 or not set(raw) <= _HAND_KEYS:
        raise PlaybookError(
            f"{where} is one hand — a bare gesture ({', '.join(BARE_HANDS)}), "
            "{tap: app.landmarks.<name>}, or {macro: app.macros.<name>}"
        )
    if "macro" in raw:
        return RecoverHand(
            macro=_argless_macro(raw["macro"], "recover", where, page, ctx.resolve).name
        )
    return RecoverHand(
        tool="tap", landmark=_landmark_name(ctx, raw["tap"], f"{where}: `tap`")
    )
