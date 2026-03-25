"""Smoke tests for src.backend package import."""

import src.backend



def test_backend_package_import_smoke() -> None:
    """Importing src.backend should succeed and expose package metadata."""
    assert src.backend.__doc__ is not None
