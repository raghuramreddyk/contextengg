"""tests/test_versioning.py — Unit tests for version bump logic."""
import pytest

from context_engine.versioning import (
    bump_version,
    initial_version,
    is_version_after,
    parse_version,
)


def test_initial_version():
    assert initial_version() == "1.0"


def test_parse_valid():
    assert parse_version("1.3") == (1, 3)
    assert parse_version("2.0") == (2, 0)


def test_parse_invalid():
    with pytest.raises(ValueError):
        parse_version("1.2.3")
    with pytest.raises(ValueError):
        parse_version("abc")


def test_bump_minor():
    assert bump_version("1.0") == "1.1"
    assert bump_version("1.9") == "1.10"


def test_bump_major():
    assert bump_version("1.3", bump="major") == "2.0"
    assert bump_version("3.7", bump="major") == "4.0"


def test_is_version_after():
    assert is_version_after("1.4", "1.3") is True
    assert is_version_after("2.0", "1.9") is True
    assert is_version_after("1.3", "1.4") is False
    assert is_version_after("1.3", "1.3") is False
