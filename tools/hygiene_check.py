"""Repo hygiene checks for public-readiness."""

from __future__ import annotations

from pathlib import Path

BLOCKED_TERMS = [
    "CS4330",
    "student",
    "professor",
    "university",
]

# Process logs are intentionally educational trace artifacts.
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "libs",
    "__pycache__",
}
EXCLUDE_PATH_PREFIXES = {
    "docs/process/",
    "tools/hygiene_check.py",
    "tests/test_hygiene_check.py",
}

TEXT_EXTS = {
    ".py",
    ".md",
    ".toml",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
}


def should_skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    for prefix in EXCLUDE_PATH_PREFIXES:
        if rel.startswith(prefix):
            return True
    return False


def run(root: Path) -> int:
    hits: list[tuple[str, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if should_skip(path, root):
            continue
        if path.suffix.lower() not in TEXT_EXTS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for term in BLOCKED_TERMS:
            if term.lower() in text.lower():
                hits.append((path.relative_to(root).as_posix(), term))

    if not hits:
        print("Hygiene check passed: no blocked terms found.")
        return 0

    print("Hygiene check failed. Blocked terms found:")
    for rel_path, term in hits:
        print(f"- {rel_path}: {term}")
    return 1


if __name__ == "__main__":
    raise SystemExit(run(Path(__file__).resolve().parents[1]))
