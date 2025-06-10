"""
Main GUI window for the AI-Powered Thesis Assistant.

This module implements the primary application window with a modern, professional
interface supporting dark mode, responsive design, and comprehensive functionality.

Features:
    - Modern tabbed interface
    - Dark mode support
    - Responsive layout
    - Real-time progress tracking
    - Non-blocking operations
    - Professional styling

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Local imports
from core.config import Config
from core.exceptions import ApplicationError
from gui.components.document_processor import DocumentProcessorWidget
from gui.components.ai_chat import AIChatWidget
from gui.components.model_selector import ModelSelectorWidget
from gui.components.settings import SettingsWidget
from gui.components.progress_tracker import ProgressTrackerWidget
from gui.components.statistics import StatisticsWidget
from gui.themes.dark_theme import DarkTheme
from gui.themes.light_theme import LightTheme

logger = logging.getLogger(__name__)


class ThesisAssistantGUI:
    """
    Main GUI application for the AI-Powered Thesis Assistant.
    
    This class manages the primary application window and coordinates
    all GUI components and user interactions.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the main GUI application.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.root = tk.Tk()
        self.current_theme = None
        self.widgets = {}
        
        # Initialize GUI
        self._setup_window()
        self._setup_themes()
        self._create_menu()
        self._create_main_interface()
        self._apply_theme()
        
        logger.info("GUI application initialized")
    
    def _setup_window(self) -> None:
        """Setup the main application window."""
        self.root.title("AI-Powered Thesis Assistant v2.0")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1400 // 2)
        y = (self.root.winfo_screenheight() // 2) - (900 // 2)
        self.root.geometry(f"1400x900+{x}+{y}")
        
        # Configure window icon (if available)
        try:
            icon_path = Path(__file__).parent / "assets" / "icon.ico"
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
        except Exception:
            pass  # Icon not critical
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_themes(self) -> None:
        """Setup application themes."""
        self.themes = {
            'dark': DarkTheme(),
            'light': LightTheme()
        }
        
        # Load theme preference from config
        theme_name = self.config.get('gui_theme', 'dark')
        self.current_theme = self.themes.get(theme_name, self.themes['dark'])
    
    def _create_menu(self) -> None:
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project", command=self._new_project)
        file_menu.add_command(label="Open Project", command=self._open_project)
        file_menu.add_command(label="Save Project", command=self._save_project)
        file_menu.add_separator()
        file_menu.add_command(label="Import Documents", command=self._import_documents)
        file_menu.add_command(label="Export Results", command=self._export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Preferences", command=self._show_preferences)
        edit_menu.add_command(label="API Keys", command=self._manage_api_keys)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Dark Theme", command=lambda: self._switch_theme('dark'))
        view_menu.add_command(label="Light Theme", command=lambda: self._switch_theme('light'))
        view_menu.add_separator()
        view_menu.add_command(label="Show Statistics", command=self._show_statistics)
        view_menu.add_command(label="Show Progress", command=self._show_progress)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Run Analysis", command=self._run_analysis)
        tools_menu.add_command(label="Validate Citations", command=self._validate_citations)
        tools_menu.add_command(label="Generate Bibliography", command=self._generate_bibliography)
        tools_menu.add_separator()
        tools_menu.add_command(label="Test API Connections", command=self._test_api_connections)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self._show_user_guide)
        help_menu.add_command(label="API Documentation", command=self._show_api_docs)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _create_main_interface(self) -> None:
        """Create the main application interface."""
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self._create_document_tab()
        self._create_ai_chat_tab()
        self._create_analysis_tab()
        self._create_settings_tab()
        
        # Create status bar
        self._create_status_bar(main_frame)
    
    def _create_document_tab(self) -> None:
        """Create the document processing tab."""
        doc_frame = ttk.Frame(self.notebook)
        self.notebook.add(doc_frame, text="📄 Documents")
        
        # Create document processor widget
        self.widgets['document_processor'] = DocumentProcessorWidget(
            doc_frame, self.config
        )
        self.widgets['document_processor'].pack(fill=tk.BOTH, expand=True)
    
    def _create_ai_chat_tab(self) -> None:
        """Create the AI chat interface tab."""
        chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(chat_frame, text="🤖 AI Chat")
        
        # Create AI chat widget
        self.widgets['ai_chat'] = AIChatWidget(
            chat_frame, self.config
        )
        self.widgets['ai_chat'].pack(fill=tk.BOTH, expand=True)
    
    def _create_analysis_tab(self) -> None:
        """Create the analysis and results tab."""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="📊 Analysis")
        
        # Create paned window for analysis components
        paned = ttk.PanedWindow(analysis_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel: Model selector and progress
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # Model selector
        self.widgets['model_selector'] = ModelSelectorWidget(
            left_frame, self.config
        )
        self.widgets['model_selector'].pack(fill=tk.X, pady=(0, 10))
        
        # Progress tracker
        self.widgets['progress_tracker'] = ProgressTrackerWidget(
            left_frame, self.config
        )
        self.widgets['progress_tracker'].pack(fill=tk.BOTH, expand=True)
        
        # Right panel: Statistics
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        self.widgets['statistics'] = StatisticsWidget(
            right_frame, self.config
        )
        self.widgets['statistics'].pack(fill=tk.BOTH, expand=True)
    
    def _create_settings_tab(self) -> None:
        """Create the settings and configuration tab."""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Settings")
        
        # Create settings widget
        self.widgets['settings'] = SettingsWidget(
            settings_frame, self.config
        )
        self.widgets['settings'].pack(fill=tk.BOTH, expand=True)
    
    def _create_status_bar(self, parent: tk.Widget) -> None:
        """Create the application status bar."""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            status_frame, 
            variable=self.progress_var,
            length=200
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=(10, 0))
    
    def _apply_theme(self) -> None:
        """Apply the current theme to all widgets."""
        if self.current_theme:
            self.current_theme.apply_to_root(self.root)
            
            # Apply theme to all custom widgets
            for widget in self.widgets.values():
                if hasattr(widget, 'apply_theme'):
                    widget.apply_theme(self.current_theme)
    
    def _switch_theme(self, theme_name: str) -> None:
        """Switch to a different theme."""
        if theme_name in self.themes:
            self.current_theme = self.themes[theme_name]
            self.config.set('gui_theme', theme_name)
            self.config.save()
            self._apply_theme()
            self.update_status(f"Switched to {theme_name} theme")
    
    def update_status(self, message: str) -> None:
        """Update the status bar message."""
        self.status_var.set(message)
        logger.info(f"Status: {message}")
    
    def update_progress(self, value: float) -> None:
        """Update the progress bar."""
        self.progress_var.set(value)
        self.root.update_idletasks()
    
    # Menu command implementations
    def _new_project(self) -> None:
        """Create a new thesis project."""
        self.update_status("Creating new project...")
        # Implementation will be added in thesis management
    
    def _open_project(self) -> None:
        """Open an existing thesis project."""
        self.update_status("Opening project...")
        # Implementation will be added in thesis management
    
    def _save_project(self) -> None:
        """Save the current thesis project."""
        self.update_status("Saving project...")
        # Implementation will be added in thesis management
    
    def _import_documents(self) -> None:
        """Import documents for processing."""
        files = filedialog.askopenfilenames(
            title="Select Documents to Import",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("Text files", "*.txt"),
                ("Markdown files", "*.md"),
                ("All files", "*.*")
            ]
        )
        
        if files:
            self.widgets['document_processor'].import_documents(files)
    
    def _export_results(self) -> None:
        """Export analysis results."""
        self.update_status("Exporting results...")
        # Implementation will be added
    
    def _show_preferences(self) -> None:
        """Show application preferences."""
        self.notebook.select(3)  # Switch to settings tab
    
    def _manage_api_keys(self) -> None:
        """Manage API keys."""
        self.widgets['settings'].show_api_keys_section()
    
    def _show_statistics(self) -> None:
        """Show statistics dashboard."""
        self.notebook.select(2)  # Switch to analysis tab
    
    def _show_progress(self) -> None:
        """Show progress tracker."""
        self.notebook.select(2)  # Switch to analysis tab
    
    def _run_analysis(self) -> None:
        """Run thesis analysis."""
        self.update_status("Starting analysis...")
        # Implementation will be added
    
    def _validate_citations(self) -> None:
        """Validate citations."""
        self.update_status("Validating citations...")
        # Implementation will be added
    
    def _generate_bibliography(self) -> None:
        """Generate bibliography."""
        self.update_status("Generating bibliography...")
        # Implementation will be added
    
    def _test_api_connections(self) -> None:
        """Test API connections."""
        self.update_status("Testing API connections...")
        # Implementation will be added
    
    def _show_user_guide(self) -> None:
        """Show user guide."""
        messagebox.showinfo("User Guide", "User guide will be available soon.")
    
    def _show_api_docs(self) -> None:
        """Show API documentation."""
        messagebox.showinfo("API Documentation", "API documentation will be available soon.")
    
    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """AI-Powered Thesis Assistant v2.0
        
Production-Grade Academic Research Tool

Features:
• Advanced document processing
• AI-powered citation analysis
• Multi-API integration
• APA7 compliance validation
• Real-time collaboration

© 2024 AI-Powered Thesis Assistant Team
Licensed under MIT License"""
        
        messagebox.showinfo("About", about_text)
    
    def _on_closing(self) -> None:
        """Handle application closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            logger.info("Application closing")
            self.root.destroy()
    
    def run(self) -> None:
        """Run the GUI application."""
        try:
            logger.info("Starting GUI application")
            self.update_status("Application ready")
            self.root.mainloop()
        except Exception as e:
            logger.error(f"GUI application error: {e}")
            messagebox.showerror("Application Error", f"An error occurred: {e}")
        finally:
            logger.info("GUI application terminated")
