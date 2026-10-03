"""Tests for hardware/manual/common.py — shared builder helpers."""

from __future__ import annotations

import pytest

from hardware.manual import common

# ── loc ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "lang", "expected"),
    [
        ({"en": "Frame", "zh": "框架"}, "zh", "框架"),
        ({"en": "Frame", "zh": "框架"}, "en", "Frame"),
        ({"en": "Frame", "zh": ""}, "zh", "Frame"),
        ({"en": "Frame"}, "zh", "Frame"),
        ("M6×16", "zh", "M6×16"),
    ],
    ids=["zh", "en", "empty-zh-falls-back", "missing-zh-falls-back", "plain-string"],
)
def test_loc_resolves_language_with_english_fallback(value, lang, expected):
    assert common.loc(value, lang) == expected


# ── _rowspans ─────────────────────────────────────────────────────────────────


def test_rowspans_marks_group_starts_with_size_and_rest_with_zero():
    rows = [{"k": "a"}, {"k": "a"}, {"k": "b"}, {"k": "a"}]

    assert common._rowspans(rows, lambda r: r["k"]) == [2, 0, 1, 1]


def test_rowspans_of_empty_rows_is_empty():
    assert common._rowspans([], lambda r: r) == []


# ── load_pages ────────────────────────────────────────────────────────────────


def test_load_pages_concatenates_content_files_in_filename_order(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "CONTENT_DIR", tmp_path)
    (tmp_path / "01_a.json").write_text('[{"page": "a"}]')
    (tmp_path / "00_front.json").write_text('[{"page": "front"}]')

    assert common.load_pages() == [{"page": "front"}, {"page": "a"}]


NBSP = "\u00a0"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sits 2 mm clear", f"sits 2{NBSP}mm clear"),
        ("4 screws", f"4{NBSP}screws"),
        ("12 V 2 A", f"12{NBSP}V 2{NBSP}A"),
        ("100 µm", f"100{NBSP}µm"),
        ("90 °C", f"90{NBSP}°C"),
        ("留 2 mm。", f"留 2{NBSP}mm。"),
        ("2 到 3 小时", f"2{NBSP}到 3{NBSP}小时"),
        ("M3×8", "M3×8"),
        ("0 0 1.5mm", "0 0 1.5mm"),
    ],
)
def test_keep_quantities_together_joins_a_number_to_the_word_after_it(text, expected):
    assert common.keep_quantities_together(f"<p>{text}</p>") == f"<p>{expected}</p>"


def test_keep_quantities_together_leaves_attributes_and_inline_css_alone():
    markup = '<div style="margin: 0 auto; left:110mm"><img alt="4 parts">2 mm</div>'
    assert common.keep_quantities_together(markup) == (
        f'<div style="margin: 0 auto; left:110mm"><img alt="4 parts">2{NBSP}mm</div>'
    )
