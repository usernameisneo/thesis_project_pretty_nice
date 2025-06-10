"""
Project Manager for the AI-Powered Thesis Assistant.

This module provides comprehensive multi-project management with
workspace organization, project discovery, and collaboration features.

Features:
    - Multi-project workspace management
    - Project discovery and indexing
    - Template management
    - Collaboration and sharing
    - Project analytics and reporting
    - Backup and synchronization

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
from dataclasses import dataclass

# Local imports
from core.config import Config
from core.exceptions import ProjectError
from thesis.project import ThesisProject, ProjectMetadata, ProjectStatus

logger = logging.getLogger(__name__)


@dataclass
class ProjectSummary:
    """Project summary for listing and overview."""
    name: str
    path: str
    status: ProjectStatus
    author: str
    last_modified: datetime
    word_count: int
    completion_percentage: float
    description: str


class ProjectManager:
    """
    Comprehensive multi-project management system.
    
    This class provides workspace-level management of multiple thesis
    projects with discovery, organization, and collaboration features.
    """
    
    def __init__(self, workspace_path: str, config: Optional[Config] = None):
        """
        Initialize the project manager.
        
        Args:
            workspace_path: Path to the workspace directory
            config: Application configuration
        """
        self.workspace_path = Path(workspace_path)
        self.config = config or Config()
        
        # Workspace structure
        self.projects_dir = self.workspace_path / "projects"
        self.templates_dir = self.workspace_path / "templates"
        self.shared_dir = self.workspace_path / "shared"
        self.backups_dir = self.workspace_path / "backups"
        self.workspace_config_file = self.workspace_path / "workspace.json"
        
        # Initialize workspace
        self._initialize_workspace()
        
        logger.info(f"Project manager initialized: {self.workspace_path}")
    
    def _initialize_workspace(self) -> None:
        """Initialize the workspace structure."""
        try:
            # Create workspace directories
            directories = [
                self.workspace_path,
                self.projects_dir,
                self.templates_dir,
                self.shared_dir,
                self.backups_dir,
                self.shared_dir / "references",
                self.shared_dir / "resources",
                self.shared_dir / "templates"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Create workspace configuration if it doesn't exist
            if not self.workspace_config_file.exists():
                self._create_workspace_config()
            
            # Create default templates if they don't exist
            self._create_default_templates()
            
            logger.info("Workspace initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize workspace: {e}")
            raise ProjectError(f"Workspace initialization failed: {e}")
    
    def _create_workspace_config(self) -> None:
        """Create default workspace configuration."""
        config = {
            "workspace_name": "Thesis Workspace",
            "created_at": datetime.now().isoformat(),
            "version": "2.0",
            "settings": {
                "auto_backup": True,
                "backup_interval_hours": 24,
                "max_backups": 10,
                "sync_enabled": False,
                "collaboration_enabled": False
            },
            "project_count": 0,
            "last_accessed": datetime.now().isoformat()
        }
        
        with open(self.workspace_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _create_default_templates(self) -> None:
        """Create default project templates."""
        templates = {
            "phd_thesis": {
                "name": "PhD Thesis Template",
                "description": "Comprehensive PhD thesis template with standard structure",
                "chapters": [
                    {"number": 1, "title": "Introduction", "target_words": 3000},
                    {"number": 2, "title": "Literature Review", "target_words": 8000},
                    {"number": 3, "title": "Methodology", "target_words": 4000},
                    {"number": 4, "title": "Results", "target_words": 6000},
                    {"number": 5, "title": "Discussion", "target_words": 5000},
                    {"number": 6, "title": "Conclusion", "target_words": 2000}
                ]
            },
            "masters_thesis": {
                "name": "Master's Thesis Template",
                "description": "Standard master's thesis template",
                "chapters": [
                    {"number": 1, "title": "Introduction", "target_words": 2000},
                    {"number": 2, "title": "Literature Review", "target_words": 5000},
                    {"number": 3, "title": "Methodology", "target_words": 3000},
                    {"number": 4, "title": "Results and Discussion", "target_words": 6000},
                    {"number": 5, "title": "Conclusion", "target_words": 1500}
                ]
            },
            "research_proposal": {
                "name": "Research Proposal Template",
                "description": "Template for research proposals",
                "chapters": [
                    {"number": 1, "title": "Introduction and Background", "target_words": 1500},
                    {"number": 2, "title": "Literature Review", "target_words": 3000},
                    {"number": 3, "title": "Research Methodology", "target_words": 2000},
                    {"number": 4, "title": "Timeline and Resources", "target_words": 1000}
                ]
            }
        }
        
        for template_id, template_data in templates.items():
            template_file = self.templates_dir / f"{template_id}.json"
            if not template_file.exists():
                with open(template_file, 'w', encoding='utf-8') as f:
                    json.dump(template_data, f, indent=2, ensure_ascii=False)
    
    def create_project(self, project_name: str, metadata: ProjectMetadata, 
                      template_id: Optional[str] = None) -> ThesisProject:
        """
        Create a new thesis project.
        
        Args:
            project_name: Name of the project
            metadata: Project metadata
            template_id: Optional template to use
            
        Returns:
            Created thesis project
        """
        try:
            # Sanitize project name for filesystem
            safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            
            project_path = self.projects_dir / safe_name
            
            # Check if project already exists
            if project_path.exists():
                raise ProjectError(f"Project '{project_name}' already exists")
            
            # Create project
            project = ThesisProject(str(project_path), self.config)
            project.create_project(metadata)
            
            # Apply template if specified
            if template_id:
                self._apply_template(project, template_id)
            
            # Update workspace configuration
            self._update_workspace_stats()
            
            logger.info(f"Project created: {project_name}")
            return project
            
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            raise ProjectError(f"Project creation failed: {e}")
    
    def _apply_template(self, project: ThesisProject, template_id: str) -> None:
        """Apply a template to a project."""
        template_file = self.templates_dir / f"{template_id}.json"
        
        if not template_file.exists():
            logger.warning(f"Template not found: {template_id}")
            return
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            
            # Create chapter files based on template
            chapters = template_data.get('chapters', [])
            for chapter_info in chapters:
                chapter_file = project.documents_dir / "chapters" / f"chapter_{chapter_info['number']:02d}_{chapter_info['title'].lower().replace(' ', '_')}.md"
                
                if not chapter_file.exists():
                    chapter_content = f"""# Chapter {chapter_info['number']}: {chapter_info['title']}

## Overview
[Provide an overview of this chapter]

## Introduction
[Write your introduction here]

## Main Content
[Write your main content here]

## Conclusion
[Write your conclusion here]

---
Target Word Count: {chapter_info.get('target_words', 1000)}
"""
                    chapter_file.write_text(chapter_content, encoding='utf-8')
            
            logger.info(f"Template applied: {template_id}")
            
        except Exception as e:
            logger.error(f"Failed to apply template: {e}")
    
    def list_projects(self) -> List[ProjectSummary]:
        """
        List all projects in the workspace.
        
        Returns:
            List of project summaries
        """
        try:
            projects = []
            
            # Scan projects directory
            for project_dir in self.projects_dir.iterdir():
                if project_dir.is_dir():
                    try:
                        project = ThesisProject(str(project_dir), self.config)
                        
                        if project.metadata:
                            summary = ProjectSummary(
                                name=project.metadata.name,
                                path=str(project_dir),
                                status=project.metadata.status,
                                author=project.metadata.author,
                                last_modified=project.metadata.last_modified,
                                word_count=project.statistics.total_words,
                                completion_percentage=project.statistics.completion_percentage,
                                description=project.metadata.description
                            )
                            projects.append(summary)
                            
                    except Exception as e:
                        logger.warning(f"Failed to load project {project_dir}: {e}")
            
            # Sort by last modified (newest first)
            projects.sort(key=lambda p: p.last_modified, reverse=True)
            
            return projects
            
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            return []
    
    def open_project(self, project_name: str) -> ThesisProject:
        """
        Open an existing project.
        
        Args:
            project_name: Name of the project to open
            
        Returns:
            Opened thesis project
        """
        try:
            # Find project directory
            project_path = None
            
            for project_dir in self.projects_dir.iterdir():
                if project_dir.is_dir():
                    try:
                        project = ThesisProject(str(project_dir), self.config)
                        if project.metadata and project.metadata.name == project_name:
                            project_path = project_dir
                            break
                    except Exception:
                        continue
            
            if not project_path:
                raise ProjectError(f"Project '{project_name}' not found")
            
            # Open project
            project = ThesisProject(str(project_path), self.config)
            
            # Update last accessed time
            self._update_workspace_stats()
            
            logger.info(f"Project opened: {project_name}")
            return project
            
        except Exception as e:
            logger.error(f"Failed to open project: {e}")
            raise ProjectError(f"Project opening failed: {e}")
    
    def delete_project(self, project_name: str, confirm: bool = False) -> None:
        """
        Delete a project.
        
        Args:
            project_name: Name of the project to delete
            confirm: Confirmation flag
        """
        if not confirm:
            raise ProjectError("Project deletion requires explicit confirmation")
        
        try:
            # Find and delete project
            project_deleted = False
            
            for project_dir in self.projects_dir.iterdir():
                if project_dir.is_dir():
                    try:
                        project = ThesisProject(str(project_dir), self.config)
                        if project.metadata and project.metadata.name == project_name:
                            shutil.rmtree(project_dir)
                            project_deleted = True
                            break
                    except Exception:
                        continue
            
            if not project_deleted:
                raise ProjectError(f"Project '{project_name}' not found")
            
            # Update workspace stats
            self._update_workspace_stats()
            
            logger.info(f"Project deleted: {project_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete project: {e}")
            raise ProjectError(f"Project deletion failed: {e}")
    
    def backup_workspace(self, backup_name: Optional[str] = None) -> str:
        """
        Create a complete workspace backup.
        
        Args:
            backup_name: Optional backup name
            
        Returns:
            Path to the created backup
        """
        try:
            if not backup_name:
                backup_name = f"workspace_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_path = self.backups_dir / backup_name
            
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy projects
            projects_backup = backup_path / "projects"
            if self.projects_dir.exists():
                shutil.copytree(self.projects_dir, projects_backup, dirs_exist_ok=True)
            
            # Copy shared resources
            shared_backup = backup_path / "shared"
            if self.shared_dir.exists():
                shutil.copytree(self.shared_dir, shared_backup, dirs_exist_ok=True)
            
            # Copy workspace configuration
            if self.workspace_config_file.exists():
                shutil.copy2(self.workspace_config_file, backup_path)
            
            # Create backup manifest
            manifest = {
                "backup_name": backup_name,
                "created_at": datetime.now().isoformat(),
                "workspace_path": str(self.workspace_path),
                "projects_count": len(list(self.projects_dir.iterdir())) if self.projects_dir.exists() else 0,
                "backup_size_mb": self._calculate_directory_size(backup_path) / (1024 * 1024)
            }
            
            manifest_file = backup_path / "backup_manifest.json"
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Workspace backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create workspace backup: {e}")
            raise ProjectError(f"Workspace backup failed: {e}")
    
    def get_workspace_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive workspace analytics.
        
        Returns:
            Analytics data
        """
        try:
            projects = self.list_projects()
            
            # Calculate statistics
            total_projects = len(projects)
            active_projects = len([p for p in projects if p.status == ProjectStatus.ACTIVE])
            completed_projects = len([p for p in projects if p.status == ProjectStatus.COMPLETED])
            total_words = sum(p.word_count for p in projects)
            avg_completion = sum(p.completion_percentage for p in projects) / total_projects if total_projects > 0 else 0
            
            # Status breakdown
            status_breakdown = {}
            for status in ProjectStatus:
                count = len([p for p in projects if p.status == status])
                status_breakdown[status.value] = count
            
            # Author breakdown
            authors = {}
            for project in projects:
                authors[project.author] = authors.get(project.author, 0) + 1
            
            # Recent activity
            recent_projects = [p for p in projects if (datetime.now() - p.last_modified).days <= 7]
            
            return {
                "workspace_path": str(self.workspace_path),
                "total_projects": total_projects,
                "active_projects": active_projects,
                "completed_projects": completed_projects,
                "total_words": total_words,
                "average_completion": avg_completion,
                "status_breakdown": status_breakdown,
                "authors": authors,
                "recent_activity": len(recent_projects),
                "workspace_size_mb": self._calculate_directory_size(self.workspace_path) / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Failed to get workspace analytics: {e}")
            return {}
    
    def _update_workspace_stats(self) -> None:
        """Update workspace statistics."""
        try:
            if self.workspace_config_file.exists():
                with open(self.workspace_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Update stats
            config["project_count"] = len(list(self.projects_dir.iterdir())) if self.projects_dir.exists() else 0
            config["last_accessed"] = datetime.now().isoformat()
            
            with open(self.workspace_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to update workspace stats: {e}")
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Failed to calculate size for {directory}: {e}")
        
        return total_size
