"""Shared scalar layer for the conductor's YAML spec files.

A pack's manifest (APP.yml) follows the macro-file conventions — same
naming rule, same prose caps, same strict-string coercion hints, same
YAML 1.2 pure loader. The rules' one true home is the macro layer
(`macros/model.py` constants, `check_name`'s wording); this module
re-exports them for the conductor and binds the scalar validators to
each spec's own error class, so `pages.py` and `model.py` keep their
distinct catch surfaces without re-spelling the validation:

    _require_str, _prose, _opt_prose, _check_name = specfile.bind(PagesError)
"""

from pathlib import Path
from typing import Any, Callable

from ruamel.yaml import YAML

from physiclaw.common import paths
from physiclaw.common.paths import (
    PACK_FILENAME,
    PLAYBOOK_FILENAME,
    RESERVED_PACK_DIRS,
)
from physiclaw.common.placeholders import placeholder_values, resolve_placeholders
from physiclaw.common.text import read_text
from physiclaw.macros.model import (
    INPUT_NAME_RE as INPUT_NAME_RE,
)
from physiclaw.macros.model import (
    MAX_NAME_LEN as MAX_NAME_LEN,
)
from physiclaw.macros.model import (
    MAX_PROSE_LEN as MAX_PROSE_LEN,
)
from physiclaw.macros.model import (
    NAME_RE as NAME_RE,
)
from physiclaw.macros.model import MacroError
from physiclaw.macros.model import check_name as _model_check_name
from physiclaw.macros.parse import reject_aliases


class SpecError(ValueError):
    """A pack's spec file is invalid — the one base every spec error
    (`PlaybookError`, `PagesError`) derives from, so a skin catches one
    class. Messages are user-facing and name their place in the file."""


# YAML 1.2 safe loader, pure-python, load-only — the same construction as
# `macros/parse.py`. `load_yaml` below is the resolved-spec door; the
# install flow's raw template read (`scaffold.read_template_manifest`)
# shares the instance.
yaml_loader = YAML(typ="safe", pure=True)


# The manifest's sections — what the app IS and what its routes share.
# Never a route: each playbook is its own `<name>/PLAYBOOK.yml` beside the
# manifest (`load_playbook_docs`), so `playbooks:` is refused with the
# reason.
_PACK_TOP_KEYS = frozenset(
    {"kind", "app", "description", "placeholders", "pages", "landmarks"}
)


def load_yaml(
    text: str,
    error_cls: type[Exception],
    where: str = "",
    values: dict[str, str] | None = None,
) -> Any:
    """The one load ritual every spec door shares: fill `<<TOKEN>>`s
    from the local values file (rejecting survivors), YAML 1.2 load,
    reject anchors/aliases document-wide, wrap errors in the door's own
    error class. The alias guard is the macro parser's: inline macros
    put clause parsing — which materializes a Clause per path, the
    alias-bomb ride — behind these doors, and running it HERE keeps the
    grammar door-independent (a text that parses at the test/tooling
    door must not fail only at the live pack file)."""
    text = resolve_placeholders(text, error_cls, values)
    prefix = f"{where}: " if where else ""
    try:
        data = yaml_loader.load(text)
    except Exception as e:  # loader errors are not confined to YAMLError
        raise error_cls(f"{prefix}invalid YAML: {e or type(e).__name__}") from e
    try:
        reject_aliases(data)
    except MacroError as e:
        raise error_cls(f"{prefix}{e}") from e
    return data


def load_pack_doc(app: str, error_cls: type[Exception]) -> dict | None:
    """`playbooks/<app>/APP.yml`, the manifest, loaded and
    top-checked: a mapping with only the known sections, no unpopulated
    template placeholders — or empty, which is a valid manifest (the
    file is the pack marker; every section is optional). None when the
    file doesn't exist (not a pack). Section contents are validated by
    their owners (`pages.py`, `model.py`) — this is just the shared
    front door."""
    path = paths.pack_root(app) / PACK_FILENAME
    if not path.exists():
        return None
    data = load_yaml(read_text(path), error_cls, where=f"{app}/{PACK_FILENAME}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise error_cls(f"{app}/{PACK_FILENAME} must be a YAML mapping")
    if "playbooks" in data:
        raise error_cls(
            f"{app}/{PACK_FILENAME}: `playbooks` does not live in the manifest "
            "— write each playbook as its own <name>/PLAYBOOK.yml folder beside "
            "it (the file body is the playbook: name, description, enabled, "
            "inputs, route)"
        )
    gap = paths.kind_gap(data.get("kind"), paths.KIND_MANIFEST)
    if gap is not None:
        raise error_cls(f"{app}/{PACK_FILENAME}: {gap}")
    # `thread:` (how the user's thread is read) is the channel's section.
    keys = _PACK_TOP_KEYS | ({"thread"} if app == paths.CHANNEL_DIRNAME else set())
    unknown = sorted(set(map(str, data.keys())) - keys)
    if unknown:
        raise error_cls(
            f"{app}/{PACK_FILENAME}: unknown key(s): {', '.join(unknown)} "
            f"(sections: {', '.join(sorted(keys))})"
        )
    return data


def load_playbook_docs(
    app: str,
    error_cls: type[ValueError],
    root: Path | None = None,
    values: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Every playbook of a pack as raw documents, plus what would not
    load, by id with the reason. An ENTRY is `<app>/<name>/PLAYBOOK.yml`,
    one folder per entry beside the manifest, keyed by the folder name
    (referenced as `<app>/<name>`); any other `<name>.yml` in that
    folder is a playbook the entry runs, keyed `<entry>.<name>`
    (referenced as `<app>/<entry>.<name>`) and listed right after it. A
    bad file excludes itself, never the pack: `scan_playbooks` reports
    it as an invalid entry. Folder and file names follow the move-name
    grammar, so one the grammar refuses is an error entry too; a `_` or
    `.` prefix is the one skip convention every artifact lister shares
    (a draft the author parks beside the pack). Strays are error entries
    so they never vanish silently: a `*.yml` at pack level that is not
    the manifest (a route that belongs in a folder, keyed by its file
    name so it can never shadow the folder it should be), a folder with
    no PLAYBOOK.yml that is not `macros/` or `prompts/`, and a folder
    inside an entry that is not one of those two (what an entry runs is
    a file). `root` is the pack's directory when the caller already
    resolved it; `values` the placeholder values when the caller read
    them (one read per pack load)."""
    root = paths.pack_root(app) if root is None else root
    name_check = bind(error_cls)[3]
    if values is None:
        try:
            values = placeholder_values()
        except ValueError as e:
            raise error_cls(str(e)) from e
    docs: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def read(playbook: str, path: Path, role: str) -> None:
        """One file into `docs` under its id, or into `errors` with the
        reason — the id's own name checked against the move-name
        grammar (the folder's for an entry, the file stem for a playbook
        it runs)."""
        where = f"{app}/{paths.playbook_file(playbook)}"
        try:
            name_check(paths.own_name(playbook), f"{where}: {role}")
            data = load_yaml(read_text(path), error_cls, where=where, values=values)
            if not isinstance(data, dict):
                raise error_cls(
                    f"{where} must be a YAML mapping (name, description, enabled, "
                    "inputs, route)"
                )
        except Exception as e:  # broad: exclude the file, never the pack
            errors[playbook] = str(e) or type(e).__name__
            return
        docs[playbook] = data

    for stray in paths.leaf_files(root, paths.PLAYBOOK_SUFFIX):
        if stray.name != PACK_FILENAME:
            errors[stray.name] = (
                f"{app}/{stray.name}: a playbook is a folder — move it to "
                f"{stray.stem}/{PLAYBOOK_FILENAME}"
            )
    for folder in paths.leaf_dirs(root):
        name = folder.name
        if name in RESERVED_PACK_DIRS:
            continue
        path = folder / PLAYBOOK_FILENAME
        if not path.is_file():
            errors[name] = (
                f"{app}/{name}/: a folder with no {PLAYBOOK_FILENAME} is not a playbook"
            )
            continue
        read(name, path, "entry folder name")
        for sibling in paths.leaf_files(folder, paths.PLAYBOOK_SUFFIX):
            if sibling.name != PLAYBOOK_FILENAME:
                read(f"{name}.{sibling.stem}", sibling, "playbook file name")
        for sub in paths.leaf_dirs(folder):
            if sub.name not in RESERVED_PACK_DIRS:
                # Keyed by the offending PATH, as the pack-level stray is:
                # a playbook of the same name may be valid beside it, and
                # must not be reported invalid in its place.
                errors[f"{name}/{sub.name}/"] = (
                    f"{app}/{name}/{sub.name}/: what an entry runs is a file "
                    f"beside its {PLAYBOOK_FILENAME} — "
                    f"{paths.playbook_file(f'{name}.{sub.name}')}"
                )
    return docs, errors


def bind(
    err: type[ValueError],
) -> tuple[
    Callable[[Any, str], str],
    Callable[..., str],
    Callable[[Any, str], str | None],
    Callable[[Any, str], None],
]:
    """The four scalar terminals, raising `err`. Wording matches the macro
    parser's — one user-facing voice for one rule set."""

    def require_str(value: Any, where: str) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is None or isinstance(value, str):
            raise err(f"{where} is required and must be a non-empty string")
        raise err(
            f"{where} must be a string but YAML parsed {value!r} "
            f'({type(value).__name__}) — quote it: "{value}"'
        )

    def prose(value: Any, where: str, *, lines: int = 1) -> str:
        """Bounded prose: one line by default, each line under
        MAX_PROSE_LEN; `lines` admits that many (a message to the user)."""
        text = require_str(value, where)
        held = text.splitlines()
        if lines == 1 and len(held) > 1:
            raise err(f"{where} must be a single line")
        if len(held) > lines:
            raise err(f"{where} has {len(held)} lines (max {lines})")
        for line in held:
            if len(line) > MAX_PROSE_LEN:
                raise err(
                    f"{where}{' has a line that' if lines > 1 else ''} is "
                    f"{len(line)} characters (max {MAX_PROSE_LEN})"
                )
        return text

    def opt_prose(value: Any, where: str) -> str | None:
        return None if value is None else prose(value, where)

    def name_check(name: Any, where: str) -> None:
        if not isinstance(name, str):
            raise err(f"{where} must be a string (got {name!r})")
        try:
            # The macro rule verbatim (wording included), translated to
            # this spec's error class at the one seam.
            _model_check_name(name, where)
        except MacroError as e:
            raise err(str(e)) from e

    return require_str, prose, opt_prose, name_check
