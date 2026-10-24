"""Unit checks for admin KPI helpers (no DB)."""

from __future__ import annotations

from app.admin import kpis


def test_reply_words_and_rates() -> None:
    assert kpis._reply_words({"parts": [{"text": "one two three"}]}) == 3
    assert kpis._reply_words({"parts": []}) is None
    assert kpis._rate(1, 4) == 0.25
    assert kpis._rate(1, 0) is None
    arm = kpis._arm(words=[10, 20, 30], closed=4, corrected=1)
    assert arm.median_words == 20
    assert arm.correction_rate == 0.25
