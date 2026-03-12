from __future__ import annotations

from pathlib import Path

from tools.hygiene_check import run


def test_hygiene_check_passes_repo_state() -> None:
    root = Path(__file__).resolve().parents[1]
    assert run(root) == 0
