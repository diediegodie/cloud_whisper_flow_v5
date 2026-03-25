"""Pre-frontend validation tests for entry/config/docs artifacts."""

import importlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]



def test_main_module_import_smoke() -> None:
    """The entry module should be importable without crashing."""
    importlib.import_module("main")



def test_config_json_has_required_keys_and_value_types() -> None:
    """config.json should expose the required schema with expected types."""
    config_path = PROJECT_ROOT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    required_keys = {
        "source_language",
        "target_language",
        "translation_enabled",
        "auto_stop_seconds",
        "vosk_model_path",
    }
    assert required_keys.issubset(config.keys())

    assert isinstance(config["source_language"], str)
    assert isinstance(config["target_language"], str)
    assert isinstance(config["translation_enabled"], bool)
    assert isinstance(config["auto_stop_seconds"], int)
    assert isinstance(config["vosk_model_path"], str)



def test_config_json_vosk_model_path_points_to_existing_directory() -> None:
    """Configured Vosk model path should resolve to an existing directory."""
    config_path = PROJECT_ROOT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    model_dir = PROJECT_ROOT / config["vosk_model_path"]
    assert model_dir.exists() and model_dir.is_dir()
    assert (model_dir / "final.mdl").exists()
    assert (model_dir / "mfcc.conf").exists()



def test_requirements_contains_runtime_and_test_dependencies() -> None:
    """requirements.txt should include all direct dependencies used by repo code."""
    requirements_path = PROJECT_ROOT / "requirements.txt"
    content = requirements_path.read_text(encoding="utf-8")

    expected_snippets = [
        "PySide6",
        "vosk",
        "sounddevice",
        "numpy",
        "deep-translator",
        "pytest",
        "pytest-cov",
    ]

    for snippet in expected_snippets:
        assert snippet in content



def test_docs_guideline_contains_acceptance_sections() -> None:
    """Guideline doc should include key acceptance and UX sections."""
    guideline_path = PROJECT_ROOT / ".docs" / "guideline.md"
    content = guideline_path.read_text(encoding="utf-8")

    assert "## 2. Required UX" in content
    assert "## 11. Testing Checklist" in content
    assert "Two mandatory text boxes" in content



def test_docs_standards_mentions_threading_constraints() -> None:
    """Standards doc should preserve threading and architecture constraints."""
    standards_path = PROJECT_ROOT / ".docs" / "standards.md"
    content = standards_path.read_text(encoding="utf-8")

    assert "QThread" in content or "QtConcurrent" in content
    assert "Clear separation" in content or "layer separation" in content.lower()



def test_development_log_contains_testing_status_section() -> None:
    """Development log should track testing status as implementations evolve."""
    log_path = PROJECT_ROOT / ".docs" / "development_log.md"
    content = log_path.read_text(encoding="utf-8")

    assert "## Testing Checklist" in content
    assert "core and backend" in content.lower()
