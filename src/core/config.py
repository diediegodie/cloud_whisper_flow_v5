"""Configuration management for Cloud Whisper Flow.

This module handles loading and saving application configuration from config.json.
It provides a singleton instance to access settings throughout the application.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages loading and persistence of application configuration.

    The ConfigManager reads configuration from config.json in the project root
    and provides methods to access and update settings. Changes are persisted
    back to the file.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize ConfigManager and load configuration.

        Args:
            config_path: Path to config.json. If None, searches up from src/core/.
        """
        if config_path is None:
            # Find config.json by navigating up from src/core/
            current = Path(__file__).parent.parent.parent
            config_path = current / "config.json"

        self._config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load configuration from config.json.

        If the file doesn't exist, initializes with empty dict and logs a warning.
        If JSON is invalid, logs an error and initializes with empty dict.
        """
        try:
            if self._config_path.exists():
                with open(self._config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
                logger.info(f"Configuration loaded from {self._config_path}")
            else:
                logger.warning(f"Config file not found at {self._config_path}")
                self._config = {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config.json: {e}")
            self._config = {}
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._config = {}

    def save(self) -> None:
        """Save current configuration to config.json.

        Raises:
            IOError: If unable to write to config file.
        """
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
            logger.info(f"Configuration saved to {self._config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key.
            default: Default value if key not found.

        Returns:
            Configuration value or default if not found.
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.

        Args:
            key: Configuration key.
            value: Value to set.
        """
        self._config[key] = value

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as dictionary.

        Returns:
            Copy of current configuration.
        """
        return self._config.copy()

    def update(self, data: Dict[str, Any]) -> None:
        """Update configuration with multiple values.

        Args:
            data: Dictionary of key-value pairs to update.
        """
        self._config.update(data)


# Global singleton instance
_instance: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get or create the global ConfigManager instance.

    Returns:
        Singleton ConfigManager instance.
    """
    global _instance
    if _instance is None:
        _instance = ConfigManager()
    return _instance
