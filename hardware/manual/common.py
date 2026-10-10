"""Helpers shared by both document builders — localization, content
loading, table row-span math, step timing, and the document chrome
conventions. Keeps ``build_sourcing_guide`` from reaching into
``build_manual``'s internals.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

_MANUAL_DIR = Path(__file__).resolve().parent
CONTENT_DIR = _MANUAL_DIR / "content"
VENDOR_FILES = {
    "zh": _MANUAL_DIR / "sourcing_vendors.cn.json",
    "en": _MANUAL_DIR / "sourcing_vendors.global.json",
}
MANUAL_VERSION_FILE = _MANUAL_DIR / "MANUAL_VERSION"

# The <html lang> attribute value per language.
HTML_LANG = {"en": "en", "zh": "zh-Hans"}

URL_MARK = "PhysiClaw.ai"  # masthead mark, never localized.


def loc(value: Any, lang: str) -> str:
    """Resolve a localized value.

    Localized text is ``{"en": ..., "zh": ...}``; this returns the requested
    language, falling back to English when the translation is empty. Plain
    strings (specs, ids, URLs) pass straight through. Localized strings are
    trusted HTML and are emitted raw — the source embeds inline ``<span>``/``<a>``.
    """
    if isinstance(value, dict):
        return value.get(lang) or value.get("en", "")
    return value


# Typography the content files cannot express, applied to text nodes only so
# attributes and inline CSS are never touched.
_TEXT_NODE = re.compile(r">([^<]*)<")
# A quantity with a unit symbol is wrapped so the stylesheet can exempt it from
# capitals ("18 mm", not "18 MM"); listed are the symbols with a lower-case letter.
_UNIT_QUANTITY = re.compile(r"\d+(?:\.\d+)? (?:mm|cm|m|µm|ms|Hz|kHz|kg|g|mA|mAh)\b")
# A number stays on one line with the word after it ("2 mm", "4 screws").
_QUANTITY_GAP = re.compile(r"(?<=\d) (?=[^\W\d_]|°)")


def _typeset(text: str) -> str:
    text = _UNIT_QUANTITY.sub(r'<span class="unit">\g<0></span>', text)
    return _QUANTITY_GAP.sub("\u00a0", text)


def typeset_quantities(markup: str) -> str:
    """``markup`` with every quantity typeset. Pass the body markup, not a
    whole document: a ``<style>`` or ``<script>`` block would be rewritten."""
    return _TEXT_NODE.sub(lambda m: f">{_typeset(m.group(1))}<", markup)


def manual_version() -> str:
    """The manual's cover version stamp; MANUAL_VERSION is the single source of truth."""
    return MANUAL_VERSION_FILE.read_text(encoding="utf-8").strip()


def load_pages() -> list[dict]:
    """Load every content/*.json (sorted by filename = page order) into pages."""
    pages: list[dict] = []
    for path in sorted(CONTENT_DIR.glob("*.json")):
        pages.extend(json.loads(path.read_text(encoding="utf-8")))
    return pages


def _rowspans(rows: list[dict], key: Callable[[dict], Any]) -> list[int]:
    """For consecutive rows sharing ``key``, return the group size at each
    group's first row and 0 at the rest — i.e. the rowspan to emit (or skip)."""
    spans = [0] * len(rows)
    i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and key(rows[j]) == key(rows[i]):
            j += 1
        spans[i] = j - i
        i = j
    return spans


_STEP_LABEL_W = 22  # pad labels to a column so every duration lines up


@contextlib.contextmanager
def _step(label: str) -> Iterator[None]:
    """Print a build step (padded with dot leaders to a fixed column) and, once
    it finishes, how long it took — so a slow phase shows what's running and the
    times read down a clean right-aligned column."""
    print(f"  {label:.<{_STEP_LABEL_W}} ", end="", flush=True)
    t0 = time.monotonic()
    try:
        yield
    except BaseException:
        print("FAILED", flush=True)
        raise
    print(f"{time.monotonic() - t0:>5.1f}s")
