"""
Thesis Management System for the AI-Powered Thesis Assistant.

This module provides comprehensive thesis project management with
document organization, writing session tracking, and collaboration features.

Components:
    - ThesisProject: Main project management class
    - ThesisChapter: Chapter organization and tracking
    - WritingSession: Writing session management
    - WritingPrompt: AI-powered writing assistance
    - ProjectManager: Multi-project management

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

from .project import ThesisProject
from .chapter import ThesisChapter
from .writing_session import WritingSession
from .writing_prompt import WritingPrompt
from .project_manager import ProjectManager

__all__ = [
    'ThesisProject',
    'ThesisChapter', 
    'WritingSession',
    'WritingPrompt',
    'ProjectManager'
]
