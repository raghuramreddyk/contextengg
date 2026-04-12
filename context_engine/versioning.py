"""context_engine/versioning.py — Semantic version management for context entries."""
from __future__ import annotations

from typing import Literal


def parse_version(version: str) -> tuple[int, int]:
    """Parse 'major.minor' string into (major, minor) integers."""
    parts = version.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid version format: {version!r}. Expected 'major.minor'.")
    return int(parts[0]), int(parts[1])


def format_version(major: int, minor: int) -> str:
    return f"{major}.{minor}"


def bump_version(
    current: str,
    bump: Literal["minor", "major"] = "minor",
) -> str:
    """Return the next version string.

    bump="minor" → 1.3 → 1.4  (new fact added from approved response)
    bump="major" → 1.3 → 2.0  (structural schema or domain change)
    """
    major, minor = parse_version(current)
    if bump == "major":
        return format_version(major + 1, 0)
    return format_version(major, minor + 1)


def initial_version() -> str:
    """The first version assigned to any new context seed."""
    return "1.0"


def is_version_after(version_a: str, version_b: str) -> bool:
    """Return True if version_a is strictly after version_b."""
    return parse_version(version_a) > parse_version(version_b)
