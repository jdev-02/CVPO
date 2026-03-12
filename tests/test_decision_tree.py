from __future__ import annotations

from cvpo.education import onboarding_questions


def test_onboarding_has_frontend_question() -> None:
    questions = onboarding_questions()
    frontend = next(q for q in questions if q.id == "frontend_preference")
    option_ids = {option.id for option in frontend.options}
    assert {"cli", "gradio", "notebook"}.issubset(option_ids)
