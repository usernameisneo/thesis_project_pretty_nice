"""
CLI module for the AI-Powered Thesis Assistant.

This module provides a comprehensive command-line interface for power users
and automation scenarios with full application functionality.

Components:
    - Main CLI application
    - Command handlers
    - Interactive mode
    - Batch processing
    - Configuration management

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

from .main import ThesisAssistantCLI
from .commands import CommandHandler
from .interactive import InteractiveMode

__all__ = ['ThesisAssistantCLI', 'CommandHandler', 'InteractiveMode']
