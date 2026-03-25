"""Smoke tests for src.core package import."""

import src.core



def test_core_package_import_smoke() -> None:
    """Importing src.core should succeed and expose package metadata."""
    assert src.core.__doc__ is not None
