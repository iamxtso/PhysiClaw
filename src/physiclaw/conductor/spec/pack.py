"""The pack door — `playbooks/<app>/` on disk → validated `Playbook`s.

A pack is a folder: `APP.yml`, the MANIFEST (what the app is and what
its routes share — meta, placeholders, landmarks, pages; every section
optional, the file may be empty), one `<name>/PLAYBOOK.yml` folder per
playbook beside it (the folder is the name, referenced as
`<app>/<name>`, and `name:` must agree with it), `macros/<name>.yml`
for the recorded hands routes share, and inside a playbook folder its
own `macros/` and `prompts/`. The manifest never carries a route.

This module loads and scans packs, spells the qualified `app/name`
dispatch key every macro site shares, and holds the live rule a wake
requires of a playbook. The model is `model.py`; the compiler is
`route.py`.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

from physiclaw.common import paths
from physiclaw.common.paths import (
    PACK_FILENAME,
    PACK_MACROS_DIRNAME,
    PACK_PROMPTS_DIRNAME,
    PROMPT_SUFFIX,
)
from physiclaw.common.placeholders import placeholder_values, resolve_placeholders
from physiclaw.common.text import read_text
from physiclaw.conductor.spec import scaffold, specfile
from physiclaw.conductor.spec.conventions import CHANNEL_APP, ROUND_MARKS
from physiclaw.conductor.spec.match import page_resolver
from physiclaw.conductor.spec.model import (
    AgentNode,
    AskNode,
    DoNode,
    Files,
    Pack,
    Playbook,
    PlaybookEntry,
    PlaybookError,
    PlaybookInput,
    Scanned,
    check_name,
    macro_resolver,
    prose,
    require_str,
)
from physiclaw.conductor.spec.pages import (
    PagesError,
    collect_page_recovers,
    pack_landmarks,
    page_sites,
    parse_pages_data,
    parse_thread,
    prints_for_app,
)
from physiclaw.conductor.spec.refs import check_refs, field_name, refs_in
from physiclaw.conductor.spec.route import compile_route, manifest_recovers
from physiclaw.macros import inputs as macro_inputs
from physiclaw.macros import parse as macro_parse
from physiclaw.macros import store as macro_store
from physiclaw.macros.model import APP_ROOT, REF_KINDS, Macro, MacroError


def qualified_macro(app: str, name: str) -> str:
    """The qualified `app/name` dispatch key — the ONE spelling of the
    convention the run_macro handler resolves (user macro names can
    never contain "/", so no collision). Lives beside `Pack`, the owner
    of macro dicts — every pack site (channel included) consumes it."""
    return f"{app}/{name}"


def split_ref(ref: str) -> tuple[str, str]:
    """`<app>/<playbook>` → (app, playbook) — the one parse of the ref
    every skin takes (the CLI exits on the error, the studio answers
    400); a run-only playbook's is `<app>/<entry>.<name>`. Raises
    PlaybookError."""
    app, sep, name = ref.partition("/")
    if not sep or not app or not name or "/" in name:
        raise PlaybookError(
            f"{ref!r} is not <app>/<playbook> (one an entry runs: <app>/<entry>.<name>)"
        )
    return app, name


def macro_app(name: str) -> str:
    """The app half of a qualified dispatch key — `qualified_macro`'s
    inverse, kept beside it so the "/" convention has one spelling.
    "" for an unqualified name (user macros never carry an app)."""
    app, sep, _ = name.partition("/")
    return app if sep else ""


def qualified_pack(app: str, pack: Pack) -> dict[str, Macro]:
    """A pack's macros under their qualified dispatch keys."""
    return {qualified_macro(app, n): m for n, m in pack.macros.items()}


def qualified_inline(app: str, spec: Playbook) -> dict[str, Macro]:
    """A playbook's inline macros under their qualified dispatch keys —
    `qualified_pack`'s sibling for the hands that live in the playbook
    itself, the playbooks it runs included. Every registry a walk can
    dispatch through takes both."""
    return {
        qualified_macro(app, n): m
        for pb in _with_subs(spec)
        for n, m in pb.inline_macros.items()
    }


def qualified_all(app: str, pack: Pack) -> dict[str, Macro]:
    """Everything a walk of this pack can dispatch: the directory macros
    plus every playbook's inline bodies — disabled playbooks included
    (gating is the caller's filter, never the dispatch table)."""
    macros = qualified_pack(app, pack)
    for entry in scan_playbooks(app, pack):
        if entry.spec is not None:
            macros.update(qualified_inline(app, entry.spec))
    return macros


def load_pack(app: str) -> Pack:
    """The app pack, whole: the manifest (`APP.yml` — what the app
    is and what its routes share: meta, placeholders, landmarks,
    pages), the playbook files beside it (raw, parsed per entry by
    `scan_playbooks`), and the recorded macros. A broken pack macro is
    carried as its error string so the playbook referencing it fails
    with the cause; a broken playbook file rides the same way."""
    doc = specfile.load_pack_doc(app, PlaybookError)
    if doc is None:
        raise PlaybookError(paths.pack_gap(app))
    root = paths.pack_root(app)
    _check_pack_meta(doc, app, root.name)
    if app == CHANNEL_APP:
        # The boot file is a template the user owns, materialized beside
        # an existing channel pack on first look (the ios pack's
        # pattern): a channel recorded before the boot was a file keeps
        # its wake, and every door — wake, step, run, check — sees the
        # same pack.
        scaffold.ensure_channel_boot(root)
    try:
        values = placeholder_values()  # read once for every file of the pack
    except ValueError as e:
        raise PlaybookError(str(e)) from e
    docs, pb_errors = specfile.load_playbook_docs(app, PlaybookError, root, values)
    try:
        # Appendix + every route's own declarations across every
        # playbook file, ONE walk — the matcher and every playbook
        # validate against the same set, and the same walk says whose
        # a route-declared page is.
        sites = page_sites(doc, docs)
        pages = parse_pages_data({n: spec for n, (spec, _) in sites.items()}, app)
    except PagesError as e:
        raise PlaybookError(f"{app}/{PACK_FILENAME} pages: {e}") from e
    try:
        landmarks = pack_landmarks(doc)
    except PagesError as e:
        raise PlaybookError(f"{app}/{PACK_FILENAME} landmarks: {e}") from e
    try:
        thread_incoming = parse_thread(doc.get("thread"))
    except PagesError as e:
        raise PlaybookError(f"{app}/{PACK_FILENAME}: {e}") from e
    # One scanner per leaf kind, run on the pack's folders and on each
    # playbook's: traversal guard, skip convention, and the broad-except
    # lesson live in `store.scan` and `paths.leaf_files`. A macro's jump
    # reads a page of THIS pack, through the one resolver over the one
    # candidate set the walk will match against.
    prints = tuple(prints_for_app(app, decls=pages))
    page_of = page_resolver(app, pages, prints)
    macros = _scan_macros(root / PACK_MACROS_DIRNAME, page_of)
    shared = macro_resolver(macros.ok, macros.errors)
    pack = Pack(
        app=app,
        pages=pages,
        macros=macros.ok,
        prints=prints,
        macro_errors=macros.errors,
        playbook_docs=docs,
        playbook_errors=pb_errors,
        prompts=_scan_prompts(root / PACK_PROMPTS_DIRNAME, values),
        local={
            name: Files(
                macros=_scan_macros(root / name / PACK_MACROS_DIRNAME, page_of, shared),
                prompts=_scan_prompts(root / name / PACK_PROMPTS_DIRNAME, values),
            )
            for name in {paths.entry_of(n) for n in docs}  # an entry's serve its own
        },
        landmarks=landmarks,
        thread_incoming=thread_incoming,
        route_pages={n: pb for n, (_, pb) in sites.items() if pb is not None},
    )
    # The manifest's hands, resolved once here with the pack's own
    # resolver: a broken one is the pack's load error, not every
    # route's.
    return replace(pack, recovers=manifest_recovers(pack, collect_page_recovers(doc)))


def _scan_macros(
    root: Path,
    pages: macro_parse.PageResolver,
    fallback: macro_parse.MacroResolver | None = None,
) -> Scanned[Macro]:
    """The macro files under one `macros/` root, folded from `store.scan`."""
    out: Scanned[Macro] = Scanned()
    for entry in macro_store.scan(root, pages, fallback):
        if entry.spec is not None:
            out.ok[entry.name] = entry.spec
        else:
            out.errors[entry.name] = entry.error or "invalid"
    return out


def _scan_prompts(root: Path, values: dict[str, str]) -> Scanned[str]:
    """The prompt files under one `prompts/` root — `<name>.md`, the
    whole file verbatim as the model's prose: placeholders filled,
    trailing whitespace trimmed. An empty file, an unfillable token, or
    a name the grammar refuses rides as its error, so the route naming
    it fails with the cause. Nothing else in the folder is read."""
    out: Scanned[str] = Scanned()
    for path in paths.leaf_files(root, PROMPT_SUFFIX):
        try:
            check_name(path.stem, "prompt file name")
            text = resolve_placeholders(read_text(path), PlaybookError, values).rstrip()
            if not text:
                raise PlaybookError("the prompt file is empty")
            out.ok[path.stem] = text
        except Exception as e:  # broad: exclude the file, never the pack
            out.errors[path.stem] = str(e) or type(e).__name__
    return out


def macros_root(app: str, playbook: str | None = None) -> Path:
    """Where a recorded hand is written: the pack's `macros/`, or a
    playbook's own — the layout rule spelled once for every door that
    scaffolds into a pack (`macros init --app`). Raises PlaybookError
    when the pack or the playbook is not on disk."""
    root = paths.pack_root(app)
    if not (root / PACK_FILENAME).exists():
        raise PlaybookError(paths.pack_gap(app))
    if playbook is not None:
        playbook = paths.entry_of(playbook)  # the entry's hands are its own too
        if not (root / playbook / paths.PLAYBOOK_FILENAME).is_file():
            raise PlaybookError(
                f"no playbook {app}/{playbook} on disk ({root / playbook})"
            )
        root = root / playbook
    return root / PACK_MACROS_DIRNAME


def _check_pack_meta(doc: dict, app: str, folder: str) -> None:
    """The manifest's meta, every field optional: `app` (which app this
    pack automates) must equal the directory it loads from when present
    — the folder IS the app, the field catches a pack copied under the
    wrong name;
    `description` is real prose when present (`install` prints it);
    `placeholders` (install-time constants, validated here so `check`
    catches a malformed map before install prompts read it) is
    name → {description, [example]}."""
    if "app" in doc:
        declared = require_str(doc.get("app"), "`app`")
        if declared != folder:
            raise PlaybookError(
                f"app {declared!r} must equal the pack directory {folder!r}"
            )
    if app == APP_ROOT or app in REF_KINDS:
        raise PlaybookError(f"a pack cannot be named {app!r} — it is a reference word")
    if "description" in doc:
        prose(doc.get("description"), "`description`")
    ph = doc.get("placeholders")
    if ph is None:
        return
    if not isinstance(ph, dict):
        raise PlaybookError("`placeholders` must be a mapping of TOKEN → spec")
    for key, spec in ph.items():
        where = f"placeholder {key!r}"
        if not isinstance(spec, dict) or set(spec) - {"description", "example"}:
            raise PlaybookError(f"{where} must be a {{description, example}} mapping")
        prose(spec.get("description"), f"{where}: `description`")


def scan_playbooks(app: str, pack: Pack | None = None) -> list[PlaybookEntry]:
    """Every playbook file of the pack, parsed against it — plus the
    files that would not load, as invalid entries. Callers that already
    hold the Pack thread it through so the spec file and macros are
    not re-read."""
    if pack is None:
        if not (paths.pack_root(app) / PACK_FILENAME).exists():
            return []
        pack = load_pack(app)
    out: list[PlaybookEntry] = [
        PlaybookEntry(app=app, name=n, error=e)
        for n, e in sorted(pack.playbook_errors.items())
    ]
    parsed: dict[str, Playbook] = {}
    # An inline body is named `<playbook id>.<move>[.<role>]`, and a
    # playbook an entry runs carries a dot in its id — so two FILES of
    # one pack can reach the same dispatch name (an entry's page `leg`
    # with a `recover:` body, and `leg.yml`'s own move `recover`). The
    # walk's table is one namespace: refuse the second file by name
    # rather than let it shadow the first silently.
    claimed: dict[str, str] = {}
    for name in pack.playbook_docs:
        name = str(name)
        try:
            spec = _sub_playbook(name, pack, parsed)
            _claim_inline(spec, pack, claimed)
            out.append(PlaybookEntry(app=app, name=name, spec=spec))
        except Exception as e:  # broad: exclude whole, never take a session down
            out.append(
                PlaybookEntry(app=app, name=name, error=str(e) or type(e).__name__)
            )
    return out


def _claim_inline(spec: Playbook, pack: Pack, claimed: dict[str, str]) -> None:
    """Record the dispatch names this FILE writes, raising on one another
    file of the pack already holds — the route compiler's own no-shadow
    rule (`route._macro_resolver`), at pack scope. The entry's recorded
    hands are in every one of its playbooks' registries by design, so
    they are not this file's to claim: only its inline bodies are."""
    entry = paths.entry_of(spec.name)
    recorded = {f"{entry}.{n}" for n in pack.local_for(spec.name).macros.ok}
    for mname in spec.inline_macros:
        if mname in recorded:
            continue
        held = claimed.setdefault(mname, spec.name)
        if held != spec.name:
            raise PlaybookError(
                f"inline macro {mname!r} is already {held!r}'s — one dispatch "
                f"name, one body; rename the move or the page it hangs on"
            )


def stray_dirs() -> list[str]:
    """Folders under a playbooks root holding YAML beside no manifest —
    an author who wrote a pack and forgot `APP.yml`. Skips the `_` and
    `.` prefixes every lister does, and a name a home pack already
    claims (the home layer shadows the tree's)."""
    packs = set(list_apps())
    out: list[str] = []
    for root in paths.playbooks_dirs():
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if (
                d.is_dir()
                and not paths.is_skipped(d.name)
                and d.name not in packs
                and any(d.rglob("*.yml"))
            ):
                out.append(f"{root.name}/{d.name}")
    return out


def list_apps() -> list[str]:
    """Packs across the search path (the `paths.playbooks_dirs` layering),
    sorted — an APP.yml marks a pack. The channel is listed whenever a
    `channel/` dir exists at all, so `check` can report why it does
    not resolve."""
    names = paths.marked_subdirs(paths.playbooks_dirs(), PACK_FILENAME)
    if any((d / CHANNEL_APP).is_dir() for d in paths.playbooks_dirs()):
        names.add(CHANNEL_APP)
    return sorted(names)


def parse_playbook(text: str, name: str, pack: Pack) -> Playbook:
    """One playbook given as YAML text — the text-shaped door tests and
    tooling use; the live path is `scan_playbooks` over the pack's
    playbook files. Raises PlaybookError naming the offending field;
    never a partial spec."""
    data = specfile.load_yaml(text, PlaybookError)
    return _parse_playbook_data(data, name, pack)


_PLAY_KEYS = {
    "kind",
    "name",
    "description",
    "enabled",
    "inputs",
    "pages",
    "route",
    "returns",
}


def _parse_playbook_data(
    data: Any, name: str, pack: Pack, parsed: dict[str, Playbook] | None = None
) -> Playbook:
    """One playbook's document → a validated Playbook. `name` is the id
    — the folder's name for an entry, `<entry>.<name>` for a playbook
    it runs — and the `name:` inside must be its own name (the folder's,
    or the file's stem), under the `kind:` its position requires. An
    entry's `run` line names one of the playbooks beside it, parsed on
    demand against the same pack (`_sub_playbook`); `parsed` is the
    scan's memo, so every document compiles once and the entry and the
    scan hold one object."""
    if not isinstance(data, dict):
        raise PlaybookError("a playbook must be a YAML mapping (key: value pairs)")
    # A playbook names itself, like a macro and a skill do — and the
    # name must be the file's, so a copied folder cannot lie about
    # what it is (the same rule `app` keeps with the pack folder).
    runs_under = paths.run_by(name)
    own, file = paths.own_name(name), paths.playbook_file(name)
    # `kind:` FIRST: a file in the wrong folder is named for what it is,
    # never refused for a key the other grammar happens not to know.
    gap = paths.kind_gap(data.get("kind"), paths.kind_of(name))
    if gap is not None:
        raise PlaybookError(f"{file}: {gap}")
    unknown = sorted(set(map(str, data.keys())) - _PLAY_KEYS)
    if unknown:
        raise PlaybookError(f"unknown key(s): {', '.join(unknown)}")
    if "name" not in data:
        raise PlaybookError(f"a playbook has no `name:` — {file} starts `name: {own}`")
    check_name(paths.entry_of(name), "playbook name")
    if runs_under is not None:
        check_name(own, "playbook name")
    declared = require_str(data.get("name"), "`name`")
    if declared != own:
        raise PlaybookError(
            f"name {declared!r} must equal the "
            f"{'file' if runs_under else 'folder'} name {own!r} ({file})"
        )
    description = prose(data.get("description"), "`description`")
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise PlaybookError("`enabled` must be true or false")
    inputs = _parse_inputs(data.get("inputs", {}))
    input_names = {i.name for i in inputs}
    memo = {} if parsed is None else parsed  # ONE memo for every `run` of this route
    route = compile_route(
        data.get("route"),
        pages=data.get("pages"),
        playbook=name,
        input_names=input_names,
        pack=pack,
        resolve_playbook=lambda other: _sub_playbook(other, pack, memo, run_by=name),
    )
    returns = _parse_returns(data.get("returns"), input_names, route.payloads)
    return Playbook(
        app=pack.app,
        name=name,
        description=description,
        enabled=enabled,
        inputs=inputs,
        nodes=tuple(route.nodes),
        start=route.start,
        inline_macros=route.inline,
        recovers=route.recovers,
        prompts_used=route.prompts_used,
        returns=returns,
        end=route.end,
    )


def _sub_playbook(
    name: str,
    pack: Pack,
    parsed: dict[str, Playbook],
    *,
    run_by: str | None = None,
) -> Playbook:
    """A playbook of the pack by id — for an entry's `run` line (`run_by`
    the entry, so the error names the file at fault; the id is then
    `<entry>.<name>`) and for the scan alike, so a run and a walk of it
    read the same file, once (`parsed` memoises). Only an entry runs, so
    a parse can never reach itself."""
    if name in parsed:
        return parsed[name]
    if name in pack.playbook_errors:
        raise PlaybookError(
            f"playbook {name!r} is invalid: {pack.playbook_errors[name]}"
        )
    if name not in pack.playbook_docs:
        # Always run by an entry: the scan only asks for ids it read off
        # disk, and a route's `run` asks for `<this entry>.<name>`.
        entry = paths.entry_of(name)
        siblings = sorted(
            paths.own_name(n) for n in pack.playbook_docs if paths.run_by(n) == entry
        )
        raise PlaybookError(
            f"no playbook {paths.own_name(name)!r} of {entry!r} — it would be "
            f"{paths.playbook_file(name)} (has: {', '.join(siblings) or '(none)'})"
        )
    try:
        spec = _parse_playbook_data(pack.playbook_docs[name], name, pack, parsed)
    except PlaybookError as e:
        if run_by is not None:
            raise PlaybookError(f"playbook {name!r} (run by {run_by!r}): {e}") from e
        raise
    parsed[name] = spec
    return spec


def _parse_returns(
    raw: Any, input_names: set[str], payloads: dict[str, tuple[str, ...]]
) -> dict[str, str]:
    """`returns:` — field → template over the playbook's own refs, what
    a `run` of it yields (`{<run>.<field>}`) once its round ends."""
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not raw:
        raise PlaybookError("`returns` must be a mapping of field → template")
    out: dict[str, str] = {}
    for fname, template in raw.items():
        field_name(str(fname), "`returns` field")
        if str(fname) in ROUND_MARKS:
            raise PlaybookError(
                f"`returns` field {fname!r} is how a round records whether it "
                f"ran ({', '.join(sorted(ROUND_MARKS))}) — rename it"
            )
        text = prose(template, f"`returns.{fname}`")
        check_refs(
            refs_in(text, f"`returns.{fname}`"),
            input_names,
            payloads,
            f"`returns.{fname}`",
        )
        out[str(fname)] = text
    return out


def resolve_inputs(spec: Playbook, provided: dict[str, str]) -> dict[str, str]:
    """Provided values against the declared inputs — the macro layer's
    resolution contract verbatim (unknown keys, missing required, defaults,
    strings only), translated to this spec's error class at the one seam."""
    try:
        return macro_inputs.resolve_inputs(spec, provided)
    except MacroError as e:
        raise PlaybookError(str(e)) from e


def _parse_inputs(raw: Any) -> tuple[PlaybookInput, ...]:
    """`inputs:` through the macro grammar's parser, the error class
    translated at this one seam."""
    try:
        return macro_parse.parse_inputs(raw)
    except MacroError as e:
        raise PlaybookError(str(e)) from e


def require_live(spec: Playbook, pack: Pack) -> None:
    """The live rule, spelled once: what a real wake needs of a playbook
    — enabled, with every referenced pack macro enabled. A resuming
    suspension and the boot must satisfy it; a rehearsal deliberately
    need not (you rehearse BEFORE you enable). Raises PlaybookError
    naming the gap."""
    if not spec.offered:
        raise PlaybookError(
            f"{spec.app}/{spec.name}: walked by {spec.run_by}'s `run:`, "
            "never launched on its own"
        )
    gap = live_gap(spec, pack)
    if gap is not None:
        raise PlaybookError(f"{spec.app}/{spec.name}: {gap} — rehearse, then enable")


def live_gap(spec: Playbook, pack: Pack) -> str | None:
    """The one thing that keeps a valid playbook from a wake, in a word
    or two — None when it is live. `require_live` raises off it; the
    wake roster prints it; both read one rule."""
    if not spec.enabled:
        return "disabled"
    if not spec.offered:
        # Walked by its entry only — never launched alone, so the roster
        # owes this reason for it (`offered` is the rule itself).
        return f"run by {spec.run_by}"
    for r in spec.runs:
        if not r.sub.enabled:
            return f"runs disabled playbook {r.sub.name!r}"
    disabled = disabled_macros(spec, pack)
    if disabled:
        return (
            f"disabled macro{'s' if len(disabled) > 1 else ''}: {', '.join(disabled)}"
        )
    return None


def disabled_macros(spec: Playbook, pack: Pack) -> list[str]:
    """Referenced pack macros still disabled — the live-readiness rule:
    `playbooks check` warns about it and the boot will not offer such
    a playbook at all. Covers every dispatching role: do moves, an ask's
    `resume:`, a page's `recover:` hands, and an agent's granted macros.
    Safe unguarded access: parse
    validated every directory name against `pack.macros`."""
    named: set[str] = set()
    inline: dict[str, Macro] = {}
    for pb in _with_subs(spec):
        inline.update(pb.inline_macros)
        for recovery in pb.recovers.values():
            named.update(h.macro for h in recovery.hands if h.macro is not None)
        for n in pb.nodes:
            if isinstance(n, DoNode):
                named.add(n.macro)
            elif isinstance(n, AskNode) and n.resume is not None:
                named.add(n.resume)
            elif isinstance(n, AgentNode):
                named.update(n.macros)
    # One rule, no special case: each name resolves through the merged
    # view. An inline body is enabled by construction (its gate is the
    # playbook's own `enabled:`); a pack macro or a playbook's recorded
    # file carries its own flag, and both are read here — as `live`, so
    # a hand that runs a disabled macro counts as disabled itself.
    return sorted(m for m in named if not (inline.get(m) or pack.macros[m]).live)


def _with_subs(spec: Playbook) -> list[Playbook]:
    """This playbook and every playbook it runs — one level, by the
    compiler's rule."""
    return [spec, *(r.sub for r in spec.runs)]
