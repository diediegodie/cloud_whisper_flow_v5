"""Pytest configuration for repository-wide test imports.

Ensures the project root is available on sys.path so tests can import modules
from the src package when running plain `pytest`.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
