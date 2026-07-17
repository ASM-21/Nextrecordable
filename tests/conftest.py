"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def tmp_db_path(tmp_path):
    """A temporary DuckDB path that won't conflict with the real project DB."""
    return tmp_path / "test_osha.duckdb"
