"""
GUI module for the AI-Powered Thesis Assistant.

This module provides a comprehensive graphical user interface built with tkinter,
featuring modern design, dark mode support, and full application functionality.

Components:
    - Main application window
    - Document processing interface
    - AI chat interface
    - Model selection interface
    - Settings management
    - Progress tracking
    - Statistics dashboard

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

# Import main GUI components
try:
    from .main_window import ThesisAssistantGUI
    from .components.document_processor import DocumentProcessorWidget
    from .components.ai_chat import AIChatWidget
    from .components.model_selector import ModelSelectorWidget
    from .components.settings import SettingsWidget
    from .components.progress_tracker import ProgressTrackerWidget
    from .components.statistics import StatisticsWidget
except ImportError:
    # Handle missing components gracefully
    ThesisAssistantGUI = None
    DocumentProcessorWidget = None
    AIChatWidget = None
    ModelSelectorWidget = None
    SettingsWidget = None
    ProgressTrackerWidget = None
    StatisticsWidget = None

__all__ = [
    'ThesisAssistantGUI',
    'DocumentProcessorWidget',
    'AIChatWidget',
    'ModelSelectorWidget',
    'SettingsWidget',
    'ProgressTrackerWidget',
    'StatisticsWidget'
]
