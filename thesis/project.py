"""
Thesis Project Management for the AI-Powered Thesis Assistant.

This module implements comprehensive thesis project management with
document organization, version control, and collaboration features.

Features:
    - Project creation and configuration
    - Document organization and tracking
    - Version control integration
    - Collaboration and sharing
    - Progress tracking and analytics
    - Backup and recovery

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Local imports
from core.config import Config
from core.exceptions import ProjectError

logger = logging.getLogger(__name__)


class ProjectStatus(Enum):
    """Project status enumeration."""
    PLANNING = "planning"
    ACTIVE = "active"
    REVIEW = "review"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    SUSPENDED = "suspended"


@dataclass
class ProjectMetadata:
    """Project metadata structure."""
    name: str
    description: str
    author: str
    institution: str
    department: str
    supervisor: str
    degree_type: str  # PhD, Masters, etc.
    field_of_study: str
    start_date: datetime
    target_completion: datetime
    status: ProjectStatus
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    version: str = "1.0"


@dataclass
class ProjectStatistics:
    """Project statistics tracking."""
    total_words: int = 0
    total_pages: int = 0
    chapters_completed: int = 0
    total_chapters: int = 0
    citations_count: int = 0
    references_count: int = 0
    writing_sessions: int = 0
    total_writing_time: float = 0.0  # hours
    last_activity: Optional[datetime] = None
    completion_percentage: float = 0.0


class ThesisProject:
    """
    Comprehensive thesis project management system.
    
    This class provides complete project lifecycle management including
    document organization, version control, and collaboration features.
    """
    
    def __init__(self, project_path: str, config: Optional[Config] = None):
        """
        Initialize a thesis project.
        
        Args:
            project_path: Path to the project directory
            config: Application configuration
        """
        self.project_path = Path(project_path)
        self.config = config or Config()
        self.metadata: Optional[ProjectMetadata] = None
        self.statistics = ProjectStatistics()
        
        # Project structure
        self.documents_dir = self.project_path / "documents"
        self.sources_dir = self.project_path / "sources"
        self.analysis_dir = self.project_path / "analysis"
        self.output_dir = self.project_path / "output"
        self.backup_dir = self.project_path / "backups"
        self.config_file = self.project_path / "project.json"
        self.stats_file = self.project_path / "statistics.json"
        
        # Load existing project or initialize new one
        if self.project_path.exists():
            self._load_project()
        
        logger.info(f"Thesis project initialized: {self.project_path}")
    
    def create_project(self, metadata: ProjectMetadata) -> None:
        """
        Create a new thesis project.
        
        Args:
            metadata: Project metadata
        """
        try:
            # Create project directory structure
            self.project_path.mkdir(parents=True, exist_ok=True)
            self._create_directory_structure()
            
            # Set metadata
            self.metadata = metadata
            
            # Save project configuration
            self._save_project()
            
            # Initialize statistics
            self.statistics = ProjectStatistics()
            self._save_statistics()
            
            # Create initial files
            self._create_initial_files()
            
            logger.info(f"New thesis project created: {metadata.name}")
            
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            raise ProjectError(f"Project creation failed: {e}")
    
    def _create_directory_structure(self) -> None:
        """Create the standard project directory structure."""
        directories = [
            self.documents_dir,
            self.sources_dir,
            self.analysis_dir,
            self.output_dir,
            self.backup_dir,
            self.documents_dir / "chapters",
            self.documents_dir / "drafts",
            self.documents_dir / "templates",
            self.sources_dir / "primary",
            self.sources_dir / "secondary",
            self.sources_dir / "data",
            self.analysis_dir / "claims",
            self.analysis_dir / "citations",
            self.analysis_dir / "reports",
            self.output_dir / "exports",
            self.output_dir / "presentations"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _create_initial_files(self) -> None:
        """Create initial project files."""
        # Create README
        readme_content = f"""# {self.metadata.name}

## Project Information
- **Author**: {self.metadata.author}
- **Institution**: {self.metadata.institution}
- **Department**: {self.metadata.department}
- **Supervisor**: {self.metadata.supervisor}
- **Degree**: {self.metadata.degree_type}
- **Field**: {self.metadata.field_of_study}

## Description
{self.metadata.description}

## Project Structure
- `documents/` - Thesis documents and chapters
- `sources/` - Reference materials and data
- `analysis/` - Analysis results and reports
- `output/` - Generated outputs and exports
- `backups/` - Project backups

## Getting Started
1. Add your source materials to the `sources/` directory
2. Create chapters in the `documents/chapters/` directory
3. Use the AI-Powered Thesis Assistant for analysis and citation management

---
Created: {self.metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        readme_file = self.project_path / "README.md"
        readme_file.write_text(readme_content, encoding='utf-8')
        
        # Create .gitignore
        gitignore_content = """# AI-Powered Thesis Assistant
*.log
*.tmp
__pycache__/
.env
.venv/
node_modules/
.DS_Store
Thumbs.db

# Backup files
*.bak
*.backup
*~

# Output files
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.synctex.gz
*.toc

# Personal files
personal_notes.md
private/
"""
        
        gitignore_file = self.project_path / ".gitignore"
        gitignore_file.write_text(gitignore_content, encoding='utf-8')
    
    def _load_project(self) -> None:
        """Load existing project configuration."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert datetime strings back to datetime objects
                for date_field in ['start_date', 'target_completion', 'created_at', 'last_modified']:
                    if date_field in data and data[date_field]:
                        data[date_field] = datetime.fromisoformat(data[date_field])
                
                # Convert status string to enum
                if 'status' in data:
                    data['status'] = ProjectStatus(data['status'])
                
                self.metadata = ProjectMetadata(**data)
            
            # Load statistics
            if self.stats_file.exists():
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats_data = json.load(f)
                
                # Convert datetime strings
                if 'last_activity' in stats_data and stats_data['last_activity']:
                    stats_data['last_activity'] = datetime.fromisoformat(stats_data['last_activity'])
                
                self.statistics = ProjectStatistics(**stats_data)
            
            logger.info("Project configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            raise ProjectError(f"Project loading failed: {e}")
    
    def _save_project(self) -> None:
        """Save project configuration."""
        try:
            if not self.metadata:
                return
            
            # Convert to dictionary
            data = {
                'name': self.metadata.name,
                'description': self.metadata.description,
                'author': self.metadata.author,
                'institution': self.metadata.institution,
                'department': self.metadata.department,
                'supervisor': self.metadata.supervisor,
                'degree_type': self.metadata.degree_type,
                'field_of_study': self.metadata.field_of_study,
                'start_date': self.metadata.start_date.isoformat(),
                'target_completion': self.metadata.target_completion.isoformat(),
                'status': self.metadata.status.value,
                'tags': self.metadata.tags,
                'keywords': self.metadata.keywords,
                'created_at': self.metadata.created_at.isoformat(),
                'last_modified': datetime.now().isoformat(),
                'version': self.metadata.version
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug("Project configuration saved")
            
        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            raise ProjectError(f"Project saving failed: {e}")
    
    def _save_statistics(self) -> None:
        """Save project statistics."""
        try:
            data = {
                'total_words': self.statistics.total_words,
                'total_pages': self.statistics.total_pages,
                'chapters_completed': self.statistics.chapters_completed,
                'total_chapters': self.statistics.total_chapters,
                'citations_count': self.statistics.citations_count,
                'references_count': self.statistics.references_count,
                'writing_sessions': self.statistics.writing_sessions,
                'total_writing_time': self.statistics.total_writing_time,
                'last_activity': self.statistics.last_activity.isoformat() if self.statistics.last_activity else None,
                'completion_percentage': self.statistics.completion_percentage
            }
            
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Project statistics saved")
            
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
    
    def update_metadata(self, **kwargs) -> None:
        """
        Update project metadata.
        
        Args:
            **kwargs: Metadata fields to update
        """
        if not self.metadata:
            raise ProjectError("Project not initialized")
        
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
        
        self.metadata.last_modified = datetime.now()
        self._save_project()
        
        logger.info(f"Project metadata updated: {list(kwargs.keys())}")
    
    def update_statistics(self, **kwargs) -> None:
        """
        Update project statistics.
        
        Args:
            **kwargs: Statistics fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self.statistics, key):
                setattr(self.statistics, key, value)
        
        self.statistics.last_activity = datetime.now()
        self._save_statistics()
        
        logger.debug(f"Project statistics updated: {list(kwargs.keys())}")
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """
        Create a project backup.
        
        Args:
            backup_name: Optional backup name
            
        Returns:
            Path to the created backup
        """
        try:
            if not backup_name:
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_path = self.backup_dir / backup_name
            
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy important directories
            for source_dir in [self.documents_dir, self.sources_dir, self.analysis_dir]:
                if source_dir.exists():
                    dest_dir = backup_path / source_dir.name
                    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
            
            # Copy configuration files
            for config_file in [self.config_file, self.stats_file]:
                if config_file.exists():
                    shutil.copy2(config_file, backup_path)
            
            logger.info(f"Project backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise ProjectError(f"Backup creation failed: {e}")
    
    def get_project_info(self) -> Dict[str, Any]:
        """
        Get comprehensive project information.
        
        Returns:
            Project information dictionary
        """
        if not self.metadata:
            return {}
        
        return {
            'metadata': {
                'name': self.metadata.name,
                'description': self.metadata.description,
                'author': self.metadata.author,
                'institution': self.metadata.institution,
                'department': self.metadata.department,
                'supervisor': self.metadata.supervisor,
                'degree_type': self.metadata.degree_type,
                'field_of_study': self.metadata.field_of_study,
                'status': self.metadata.status.value,
                'tags': self.metadata.tags,
                'keywords': self.metadata.keywords,
                'created_at': self.metadata.created_at.isoformat(),
                'last_modified': self.metadata.last_modified.isoformat()
            },
            'statistics': {
                'total_words': self.statistics.total_words,
                'total_pages': self.statistics.total_pages,
                'chapters_completed': self.statistics.chapters_completed,
                'total_chapters': self.statistics.total_chapters,
                'citations_count': self.statistics.citations_count,
                'references_count': self.statistics.references_count,
                'writing_sessions': self.statistics.writing_sessions,
                'total_writing_time': self.statistics.total_writing_time,
                'completion_percentage': self.statistics.completion_percentage
            },
            'paths': {
                'project_path': str(self.project_path),
                'documents_dir': str(self.documents_dir),
                'sources_dir': str(self.sources_dir),
                'analysis_dir': str(self.analysis_dir),
                'output_dir': str(self.output_dir)
            }
        }
    
    def archive_project(self) -> None:
        """Archive the project."""
        if self.metadata:
            self.metadata.status = ProjectStatus.ARCHIVED
            self._save_project()
            logger.info(f"Project archived: {self.metadata.name}")
    
    def delete_project(self, confirm: bool = False) -> None:
        """
        Delete the project (with confirmation).
        
        Args:
            confirm: Confirmation flag
        """
        if not confirm:
            raise ProjectError("Project deletion requires explicit confirmation")
        
        try:
            shutil.rmtree(self.project_path)
            logger.info(f"Project deleted: {self.project_path}")
            
        except Exception as e:
            logger.error(f"Failed to delete project: {e}")
            raise ProjectError(f"Project deletion failed: {e}")
