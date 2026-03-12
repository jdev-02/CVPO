from __future__ import annotations

from cvpo.education import get_socratic_block, get_tradeoff_cards, resolve_goal


def test_resolve_goal_geese_tracking() -> None:
    decision = resolve_goal("geese_tracking")
    assert decision["pipeline_level"] == "level3_detect_segment_classify_track"
    assert "tracking" in decision["cv_tasks"]


def test_socratic_block_varies_by_experience() -> None:
    beginner = get_socratic_block("image_labeling", "beginner", skip_socratic=False)
    advanced = get_socratic_block("image_labeling", "advanced", skip_socratic=False)
    assert beginner["question"] != advanced["question"]


def test_tradeoff_cards_exist_for_goal() -> None:
    cards = get_tradeoff_cards("geese_tracking")
    assert len(cards) >= 1
    assert "stage" in cards[0]
