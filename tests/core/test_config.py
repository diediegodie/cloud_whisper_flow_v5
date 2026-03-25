"""Tests for core configuration management."""

import json
from pathlib import Path

from src.core import config as config_module



def test_loads_valid_config_json(tmp_path: Path) -> None:
    """ConfigManager should load a valid JSON file."""
    config_file = tmp_path / "config.json"
    payload = {
        "source_language": "pt",
        "target_language": "en",
        "auto_stop_seconds": 8,
    }
    config_file.write_text(json.dumps(payload), encoding="utf-8")

    manager = config_module.ConfigManager(config_path=config_file)

    assert manager.get("source_language") == "pt"
    assert manager.get("target_language") == "en"
    assert manager.get("auto_stop_seconds") == 8



def test_handles_missing_file_gracefully(tmp_path: Path) -> None:
    """Missing config file should not crash and should yield empty config."""
    missing_file = tmp_path / "missing-config.json"

    manager = config_module.ConfigManager(config_path=missing_file)

    assert manager.get_all() == {}



def test_handles_invalid_json_gracefully(tmp_path: Path) -> None:
    """Malformed JSON should not crash and should yield empty config."""
    invalid_file = tmp_path / "config.json"
    invalid_file.write_text("{ invalid json ", encoding="utf-8")

    manager = config_module.ConfigManager(config_path=invalid_file)

    assert manager.get_all() == {}



def test_save_persists_changes(tmp_path: Path) -> None:
    """save() should write current config values to disk."""
    config_file = tmp_path / "saved-config.json"

    manager = config_module.ConfigManager(config_path=config_file)
    manager.set("source_language", "pt")
    manager.set("target_language", "en")
    manager.set("auto_stop_seconds", 8)
    manager.save()

    raw = json.loads(config_file.read_text(encoding="utf-8"))
    assert raw["source_language"] == "pt"
    assert raw["target_language"] == "en"
    assert raw["auto_stop_seconds"] == 8



def test_get_set_update_get_all_behave_correctly(tmp_path: Path) -> None:
    """Core dictionary-like operations should work as expected."""
    config_file = tmp_path / "config.json"
    manager = config_module.ConfigManager(config_path=config_file)

    manager.set("source_language", "pt")
    assert manager.get("source_language") == "pt"
    assert manager.get("missing", "fallback") == "fallback"

    manager.update({"target_language": "en", "auto_stop_seconds": 8})
    full_config = manager.get_all()
    assert full_config["target_language"] == "en"
    assert full_config["auto_stop_seconds"] == 8

    # get_all() returns a copy, not the internal dictionary.
    full_config["source_language"] = "changed"
    assert manager.get("source_language") == "pt"



def test_singleton_get_config_returns_stable_instance() -> None:
    """get_config() should return the same ConfigManager instance."""
    config_module._instance = None
    try:
        first = config_module.get_config()
        second = config_module.get_config()
        assert first is second
    finally:
        config_module._instance = None
