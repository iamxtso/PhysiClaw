"""Money runs in code — the payment doctrine's predicates, one home.

Two pure functions, deliberately free of walk state so the money
rules can be read (and audited) without the state machine around them.
The walk supplies the numbers and acts on the answers:

  - `declared_total` — the amount beside the label the ask's `total_label:`
    names: the number the user is quoted and consents to, read the
    same way again at fire time.
  - `fire_block` — the fire-time predicate, run against the CURRENT
    screen after the human's consent: the declared total must still
    read as the consented amount — same label, same rule, same number.
    Where the pack says the total is read is all the walk knows about
    money on the page; every other price on it (a promo card, an add-on
    carousel, a struck-through original) is not the order, and a page
    re-renders those between two frames. None = pay; else the bare
    reason — the walk prefixes the move it was guarding and hands over.

Consent itself — quoting, binding, consuming — stays with the gate
(`step_ask.py`, `speak.py`, `gate.Gate`): consent is a conversation, these are
arithmetic.
"""

from physiclaw.common.bbox import Bbox, center_of, same_line
from physiclaw.common.listing import Screen, label_hit
from physiclaw.conductor.spec.conventions import PRICE_RE

# The one sentence a fired-but-unverified payment leaves behind — the
# daily log and the handover brief say it alike.
VERIFY_AFTER_PAY = "verify what it paid for before paying again"


def plain(value: float) -> str:
    """An amount as a person reads it — `40.8`, `40`, `12345.67`. The
    ONE spelling, because every place an amount is shown is a consent
    record or its audit trail. The currency sign belongs to whoever
    writes the sentence (a pack's `message:`, our own log lines).

    Never `%g`, which is six significant figures and an exponent past a
    million: it quotes a ¥12,345.67 sheet as 12345.7 and a ¥99,999.99
    one as 100000. A total the user is asked to approve must be the
    total on the screen, to the fen."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def amount(text: str) -> float:
    """One `PRICE_RE` group as a number — thousands separators stripped."""
    return float(text.replace(",", ""))


def _is_label_row(label: str, readings: tuple[str, ...]) -> tuple[bool, bool]:
    """(hits, exact): whether a row reads as the total label at all, and
    whether it is that label and nothing else — "Total" or "Total: $59.9",
    never "Items total $79" (a subtotal wearing the same word)."""
    if not any(label_hit(r, label) for r in readings):
        return False, False
    bare = PRICE_RE.sub("", label)
    bare = "".join(ch for ch in bare if ch.isalnum())
    return True, any("".join(ch for ch in r if ch.isalnum()) == bare for r in readings)


def declared_total(screen: Screen, readings: tuple[str, ...]) -> float | None:
    """The amount beside the declared total label: on the label's own
    row, else the nearest amount on a row sharing its line (OCR splits
    "Total" from its "$59.9") — never a row above or below, whatever the
    distance. A row that is the label and nothing else beats one that
    merely contains it (the payable Total over an Items total subtotal); among
    equals the lowest on screen wins (the footer); a label row with no
    amount on its line yields to the next. None when none reads."""
    priced = [
        (r.bbox, amount(m[0])) for r in screen.rows if (m := PRICE_RE.findall(r.label))
    ]
    ranked = []
    for row in screen.rows:
        if row.kind != "text":
            continue
        hits, exact = _is_label_row(row.label, readings)
        if not hits:
            continue
        c = center_of(row.bbox)
        assert c is not None  # Element bboxes are valid by construction
        ranked.append((not exact, -c[1], row))
    for _, _, row in sorted(ranked, key=lambda t: (t[0], t[1])):
        own = PRICE_RE.findall(row.label)
        if own:
            return amount(own[0])
        beside = min(
            (
                (_gap(row.bbox, bbox), amt)
                for bbox, amt in priced
                if same_line(row.bbox, bbox)
            ),
            default=None,
        )
        if beside is not None:
            return beside[1]
    return None


def _gap(a: Bbox, b: Bbox) -> float:
    """Horizontal distance between two boxes (0 when they overlap)."""
    return max(b[0] - a[2], a[0] - b[2], 0.0)


def fire_block(
    *, consented: float | None, total_label: tuple[str, ...], screen: Screen
) -> str | None:
    """The fire-time predicate. `total_label` is the ask's own readings —
    the total is read at fire time exactly as it was quoted. None = pay;
    else the bare reason to block — the walk prefixes the move it was
    guarding."""
    if consented is None:
        return "reached without a confirmed total"
    total = declared_total(screen, total_label)
    if total is not None and abs(total - consented) < 0.01:
        return None
    now = plain(total) if total is not None else "no amount"
    return (
        f"sheet changed after consent: confirmed {plain(consented)}, "
        f"now {now} beside {' / '.join(total_label)}"
    )
