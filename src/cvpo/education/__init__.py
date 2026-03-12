"""Educational system components (decision tree, questions, tradeoff guidance, citations)."""

from cvpo.education.citations import CITATIONS, cite, format_citations
from cvpo.education.decision_tree import Question, QuestionOption, discovery_questions, onboarding_questions, resolve_goal
from cvpo.education.questions import get_socratic_block
from cvpo.education.tradeoffs import get_tradeoff_cards

__all__ = [
    "CITATIONS",
    "Question",
    "QuestionOption",
    "cite",
    "format_citations",
    "get_socratic_block",
    "get_tradeoff_cards",
    "onboarding_questions",
    "resolve_goal",
]
