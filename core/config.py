"""
Core configuration management for the thesis assistant.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the thesis assistant application."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_file: Path to configuration file. If None, uses default location.
        """
        self.config_file = config_file or self._get_default_config_path()
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        home_dir = Path.home()
        config_dir = home_dir / ".thesis_assistant"
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / "config.json")
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
                self._config = {}
        else:
            self._config = {}
        
        # Set defaults
        self._set_defaults()
    
    def _set_defaults(self) -> None:
        """Set default configuration values."""
        defaults = {
            "default_model": "all-MiniLM-L6-v2",
            "openrouter_api_key": "",
            "data_dir": str(Path.home() / ".thesis_assistant" / "data"),
            "index_dir": str(Path.home() / ".thesis_assistant" / "indexes"),
            "projects_dir": str(Path.home() / ".thesis_assistant" / "projects"),
            "max_chunk_size": 512,
            "chunk_overlap": 50,
            "search_results_limit": 10,
            "similarity_threshold": 0.7,
            "gui_theme": "dark",
            "font_size": 12,
            "auto_save": True,
            "auto_save_interval": 300  # seconds
        }
        
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value
    
    def save(self) -> None:
        """
        Save current configuration to the JSON file.

        Creates the configuration directory if it doesn't exist and writes
        the current configuration state to the JSON file with proper formatting.
        Handles I/O errors gracefully with warning messages.

        Raises:
            IOError: If the configuration file cannot be written (handled gracefully)
        """
        try:
            # Ensure the configuration directory exists
            config_dir = os.path.dirname(self.config_file)
            os.makedirs(config_dir, exist_ok=True)

            # Write configuration to file with proper JSON formatting
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)

        except IOError as e:
            # Handle file write errors gracefully - don't crash the application
            print(f"⚠️  Warning: Could not save config file {self.config_file}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value by key.

        Provides safe access to configuration values with optional default
        fallback. This is the preferred method for accessing configuration
        values throughout the application.

        Args:
            key: Configuration key to retrieve
            default: Default value to return if key is not found

        Returns:
            Configuration value if key exists, otherwise the default value

        Example:
            >>> config = Config()
            >>> chunk_size = config.get('max_chunk_size', 512)
            >>> api_key = config.get('openrouter_api_key', '')
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Updates the in-memory configuration with the new value. Note that
        this does not automatically save to disk - call save() explicitly
        if persistence is required.

        Args:
            key: Configuration key to set
            value: Value to assign to the key

        Example:
            >>> config = Config()
            >>> config.set('max_chunk_size', 1024)
            >>> config.save()  # Persist changes to disk
        """
        self._config[key] = value

    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update multiple configuration values from a dictionary.

        Merges the provided dictionary with the current configuration,
        overwriting existing keys and adding new ones. This is useful
        for bulk configuration updates.

        Args:
            config_dict: Dictionary of configuration key-value pairs

        Example:
            >>> config = Config()
            >>> updates = {'max_chunk_size': 1024, 'similarity_threshold': 0.8}
            >>> config.update(updates)
            >>> config.save()  # Persist changes to disk
        """
        self._config.update(config_dict)
    
    @property
    def default_model(self) -> str:
        """
        Get the default sentence transformer model name.

        Returns the model identifier used for generating text embeddings.
        This model is used by the vector indexing system for semantic search.

        Returns:
            Model name string (e.g., 'all-MiniLM-L6-v2')
        """
        return self._config["default_model"]

    @property
    def openrouter_api_key(self) -> str:
        """
        Get the OpenRouter API key for language model access.

        Returns the API key used to authenticate with OpenRouter services
        for advanced language model features like text generation and analysis.

        Returns:
            API key string (empty string if not configured)
        """
        return self._config["openrouter_api_key"]

    @property
    def data_dir(self) -> str:
        """
        Get the data directory path.

        Returns the directory path where application data files are stored,
        including processed documents, metadata, and temporary files.

        Returns:
            Absolute path to the data directory
        """
        return self._config["data_dir"]

    @property
    def index_dir(self) -> str:
        """
        Get the index directory path.

        Returns the directory path where search indexes are stored,
        including vector embeddings, keyword indexes, and FAISS files.

        Returns:
            Absolute path to the index directory
        """
        return self._config["index_dir"]

    @property
    def projects_dir(self) -> str:
        """
        Get the projects directory path.

        Returns the directory path where thesis projects are stored,
        including project files, chapters, and associated metadata.

        Returns:
            Absolute path to the projects directory
        """
        return self._config["projects_dir"]

    def ensure_directories(self) -> None:
        """
        Ensure all configured directories exist.

        Creates any missing directories specified in the configuration.
        This method should be called during application initialization
        to ensure the file system is properly set up.

        The following directories are created if they don't exist:
        - data_dir: For application data and processed documents
        - index_dir: For search indexes and embeddings
        - projects_dir: For thesis projects and chapters

        Raises:
            OSError: If directory creation fails due to permissions or disk space
        """
        # List of directory configuration keys that need to exist
        directory_keys = ["data_dir", "index_dir", "projects_dir"]

        for dir_key in directory_keys:
            directory = self._config[dir_key]
            try:
                # Create directory with parents if it doesn't exist
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                # Re-raise with more context about which directory failed
                raise OSError(f"Failed to create {dir_key} directory '{directory}': {e}") from e

    def validate_config(self) -> bool:
        """
        Validate the current configuration for completeness and correctness.

        Checks that all required configuration values are present and valid.
        This method can be used to verify configuration integrity before
        starting the application.

        Returns:
            True if configuration is valid, False otherwise

        Example:
            >>> config = Config()
            >>> if not config.validate_config():
            ...     print("Configuration validation failed")
        """
        required_keys = [
            "default_model", "data_dir", "index_dir", "projects_dir",
            "max_chunk_size", "chunk_overlap", "search_results_limit"
        ]

        # Check that all required keys are present
        for key in required_keys:
            if key not in self._config:
                print(f"❌ Missing required configuration key: {key}")
                return False

        # Validate specific configuration values
        if self._config["max_chunk_size"] <= 0:
            print("❌ max_chunk_size must be positive")
            return False

        if self._config["chunk_overlap"] < 0:
            print("❌ chunk_overlap cannot be negative")
            return False

        if self._config["search_results_limit"] <= 0:
            print("❌ search_results_limit must be positive")
            return False

        return True

