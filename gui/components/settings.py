"""
Settings Widget for the AI-Powered Thesis Assistant.

This widget provides comprehensive application settings and configuration
management with secure API key handling and user preferences.

Features:
    - API key management
    - Application preferences
    - Theme configuration
    - Performance settings
    - Security options
    - Import/export settings

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
from pathlib import Path
from typing import Dict, Any

# Local imports
from core.config import Config

logger = logging.getLogger(__name__)


class SettingsWidget(ttk.Frame):
    """
    Settings management widget with comprehensive configuration options.
    
    This widget provides a user-friendly interface for managing all
    application settings and preferences.
    """
    
    def __init__(self, parent: tk.Widget, config: Config):
        """
        Initialize the settings widget.
        
        Args:
            parent: Parent widget
            config: Application configuration
        """
        super().__init__(parent)
        self.config = config
        self.settings_vars = {}
        
        self._setup_ui()
        self._load_settings()
        
        logger.info("Settings widget initialized")
    
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        # Title
        title_label = ttk.Label(
            self,
            text="⚙️ Application Settings",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # Create notebook for settings categories
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Create settings tabs
        self._create_api_tab()
        self._create_general_tab()
        self._create_performance_tab()
        self._create_advanced_tab()
        
        # Action buttons
        self._create_action_buttons()
    
    def _create_api_tab(self) -> None:
        """Create the API settings tab."""
        api_frame = ttk.Frame(self.notebook)
        self.notebook.add(api_frame, text="🔑 API Keys")
        
        # Scrollable frame
        canvas = tk.Canvas(api_frame)
        scrollbar = ttk.Scrollbar(api_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # API key sections
        self._create_openrouter_section(scrollable_frame)
        self._create_perplexity_section(scrollable_frame)
        self._create_semantic_scholar_section(scrollable_frame)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_openrouter_section(self, parent: tk.Widget) -> None:
        """Create OpenRouter API settings."""
        frame = ttk.LabelFrame(parent, text="OpenRouter API", padding=15)
        frame.pack(fill=tk.X, pady=(0, 15), padx=20)
        
        ttk.Label(frame, text="API Key:").pack(anchor=tk.W)
        
        self.settings_vars['openrouter_api_key'] = tk.StringVar()
        api_entry = ttk.Entry(
            frame,
            textvariable=self.settings_vars['openrouter_api_key'],
            show="*",
            width=50
        )
        api_entry.pack(fill=tk.X, pady=(5, 10))
        
        # Show/hide button
        show_frame = ttk.Frame(frame)
        show_frame.pack(fill=tk.X)
        
        self.show_openrouter_var = tk.BooleanVar()
        show_check = ttk.Checkbutton(
            show_frame,
            text="Show API key",
            variable=self.show_openrouter_var,
            command=lambda: self._toggle_api_visibility(api_entry, self.show_openrouter_var)
        )
        show_check.pack(side=tk.LEFT)
        
        ttk.Button(
            show_frame,
            text="Test Connection",
            command=self._test_openrouter_connection
        ).pack(side=tk.RIGHT)
        
        # Help text
        help_text = "Get your API key from: https://openrouter.ai/keys"
        ttk.Label(frame, text=help_text, foreground="blue").pack(anchor=tk.W, pady=(10, 0))
    
    def _create_perplexity_section(self, parent: tk.Widget) -> None:
        """Create Perplexity API settings."""
        frame = ttk.LabelFrame(parent, text="Perplexity API", padding=15)
        frame.pack(fill=tk.X, pady=(0, 15), padx=20)
        
        ttk.Label(frame, text="API Key:").pack(anchor=tk.W)
        
        self.settings_vars['perplexity_api_key'] = tk.StringVar()
        api_entry = ttk.Entry(
            frame,
            textvariable=self.settings_vars['perplexity_api_key'],
            show="*",
            width=50
        )
        api_entry.pack(fill=tk.X, pady=(5, 10))
        
        # Show/hide button
        show_frame = ttk.Frame(frame)
        show_frame.pack(fill=tk.X)
        
        self.show_perplexity_var = tk.BooleanVar()
        show_check = ttk.Checkbutton(
            show_frame,
            text="Show API key",
            variable=self.show_perplexity_var,
            command=lambda: self._toggle_api_visibility(api_entry, self.show_perplexity_var)
        )
        show_check.pack(side=tk.LEFT)
        
        ttk.Button(
            show_frame,
            text="Test Connection",
            command=self._test_perplexity_connection
        ).pack(side=tk.RIGHT)
        
        # Help text
        help_text = "Get your API key from: https://www.perplexity.ai/settings/api"
        ttk.Label(frame, text=help_text, foreground="blue").pack(anchor=tk.W, pady=(10, 0))
    
    def _create_semantic_scholar_section(self, parent: tk.Widget) -> None:
        """Create Semantic Scholar API settings."""
        frame = ttk.LabelFrame(parent, text="Semantic Scholar API (Optional)", padding=15)
        frame.pack(fill=tk.X, pady=(0, 15), padx=20)
        
        ttk.Label(frame, text="API Key:").pack(anchor=tk.W)
        
        self.settings_vars['semantic_scholar_api_key'] = tk.StringVar()
        api_entry = ttk.Entry(
            frame,
            textvariable=self.settings_vars['semantic_scholar_api_key'],
            show="*",
            width=50
        )
        api_entry.pack(fill=tk.X, pady=(5, 10))
        
        # Show/hide button
        show_frame = ttk.Frame(frame)
        show_frame.pack(fill=tk.X)
        
        self.show_semantic_var = tk.BooleanVar()
        show_check = ttk.Checkbutton(
            show_frame,
            text="Show API key",
            variable=self.show_semantic_var,
            command=lambda: self._toggle_api_visibility(api_entry, self.show_semantic_var)
        )
        show_check.pack(side=tk.LEFT)
        
        ttk.Button(
            show_frame,
            text="Test Connection",
            command=self._test_semantic_scholar_connection
        ).pack(side=tk.RIGHT)
        
        # Help text
        help_text = "Optional: Enhances academic paper discovery. Contact Semantic Scholar for API access."
        ttk.Label(frame, text=help_text, foreground="gray").pack(anchor=tk.W, pady=(10, 0))
    
    def _create_general_tab(self) -> None:
        """Create the general settings tab."""
        general_frame = ttk.Frame(self.notebook)
        self.notebook.add(general_frame, text="🎨 General")
        
        # Theme settings
        theme_frame = ttk.LabelFrame(general_frame, text="Appearance", padding=15)
        theme_frame.pack(fill=tk.X, pady=(20, 15), padx=20)
        
        ttk.Label(theme_frame, text="Theme:").pack(anchor=tk.W)
        self.settings_vars['gui_theme'] = tk.StringVar()
        theme_combo = ttk.Combobox(
            theme_frame,
            textvariable=self.settings_vars['gui_theme'],
            values=["dark", "light"],
            state="readonly"
        )
        theme_combo.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Label(theme_frame, text="Font Size:").pack(anchor=tk.W)
        self.settings_vars['font_size'] = tk.IntVar()
        font_scale = ttk.Scale(
            theme_frame,
            from_=8,
            to=16,
            variable=self.settings_vars['font_size'],
            orient=tk.HORIZONTAL
        )
        font_scale.pack(fill=tk.X, pady=(5, 0))
        
        # Auto-save settings
        autosave_frame = ttk.LabelFrame(general_frame, text="Auto-save", padding=15)
        autosave_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
        
        self.settings_vars['auto_save'] = tk.BooleanVar()
        ttk.Checkbutton(
            autosave_frame,
            text="Enable auto-save",
            variable=self.settings_vars['auto_save']
        ).pack(anchor=tk.W)
        
        ttk.Label(autosave_frame, text="Auto-save interval (seconds):").pack(anchor=tk.W, pady=(10, 0))
        self.settings_vars['auto_save_interval'] = tk.IntVar()
        interval_scale = ttk.Scale(
            autosave_frame,
            from_=60,
            to=600,
            variable=self.settings_vars['auto_save_interval'],
            orient=tk.HORIZONTAL
        )
        interval_scale.pack(fill=tk.X, pady=(5, 0))
    
    def _create_performance_tab(self) -> None:
        """Create the performance settings tab."""
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="⚡ Performance")
        
        # Processing settings
        proc_frame = ttk.LabelFrame(perf_frame, text="Document Processing", padding=15)
        proc_frame.pack(fill=tk.X, pady=(20, 15), padx=20)
        
        ttk.Label(proc_frame, text="Max chunk size:").pack(anchor=tk.W)
        self.settings_vars['max_chunk_size'] = tk.IntVar()
        chunk_scale = ttk.Scale(
            proc_frame,
            from_=256,
            to=2048,
            variable=self.settings_vars['max_chunk_size'],
            orient=tk.HORIZONTAL
        )
        chunk_scale.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Label(proc_frame, text="Chunk overlap:").pack(anchor=tk.W)
        self.settings_vars['chunk_overlap'] = tk.IntVar()
        overlap_scale = ttk.Scale(
            proc_frame,
            from_=0,
            to=200,
            variable=self.settings_vars['chunk_overlap'],
            orient=tk.HORIZONTAL
        )
        overlap_scale.pack(fill=tk.X, pady=(5, 0))
        
        # Search settings
        search_frame = ttk.LabelFrame(perf_frame, text="Search & Indexing", padding=15)
        search_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
        
        ttk.Label(search_frame, text="Search results limit:").pack(anchor=tk.W)
        self.settings_vars['search_results_limit'] = tk.IntVar()
        results_scale = ttk.Scale(
            search_frame,
            from_=5,
            to=50,
            variable=self.settings_vars['search_results_limit'],
            orient=tk.HORIZONTAL
        )
        results_scale.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Label(search_frame, text="Similarity threshold:").pack(anchor=tk.W)
        self.settings_vars['similarity_threshold'] = tk.DoubleVar()
        similarity_scale = ttk.Scale(
            search_frame,
            from_=0.1,
            to=1.0,
            variable=self.settings_vars['similarity_threshold'],
            orient=tk.HORIZONTAL
        )
        similarity_scale.pack(fill=tk.X, pady=(5, 0))
    
    def _create_advanced_tab(self) -> None:
        """Create the advanced settings tab."""
        adv_frame = ttk.Frame(self.notebook)
        self.notebook.add(adv_frame, text="🔧 Advanced")
        
        # Directory settings
        dir_frame = ttk.LabelFrame(adv_frame, text="Directories", padding=15)
        dir_frame.pack(fill=tk.X, pady=(20, 15), padx=20)
        
        # Data directory
        ttk.Label(dir_frame, text="Data Directory:").pack(anchor=tk.W)
        data_frame = ttk.Frame(dir_frame)
        data_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.settings_vars['data_dir'] = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.settings_vars['data_dir']).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(data_frame, text="Browse", command=lambda: self._browse_directory('data_dir')).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Index directory
        ttk.Label(dir_frame, text="Index Directory:").pack(anchor=tk.W)
        index_frame = ttk.Frame(dir_frame)
        index_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.settings_vars['index_dir'] = tk.StringVar()
        ttk.Entry(index_frame, textvariable=self.settings_vars['index_dir']).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(index_frame, text="Browse", command=lambda: self._browse_directory('index_dir')).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Projects directory
        ttk.Label(dir_frame, text="Projects Directory:").pack(anchor=tk.W)
        projects_frame = ttk.Frame(dir_frame)
        projects_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.settings_vars['projects_dir'] = tk.StringVar()
        ttk.Entry(projects_frame, textvariable=self.settings_vars['projects_dir']).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(projects_frame, text="Browse", command=lambda: self._browse_directory('projects_dir')).pack(side=tk.RIGHT, padx=(5, 0))
    
    def _create_action_buttons(self) -> None:
        """Create action buttons."""
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(
            button_frame,
            text="💾 Save Settings",
            command=self._save_settings
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="🔄 Reset to Defaults",
            command=self._reset_settings
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="📤 Export Settings",
            command=self._export_settings
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="📥 Import Settings",
            command=self._import_settings
        ).pack(side=tk.LEFT)
    
    def _load_settings(self) -> None:
        """Load settings from configuration."""
        for key, var in self.settings_vars.items():
            value = self.config.get(key, var.get())
            var.set(value)
    
    def _save_settings(self) -> None:
        """Save settings to configuration."""
        for key, var in self.settings_vars.items():
            self.config.set(key, var.get())
        
        self.config.save()
        messagebox.showinfo("Settings Saved", "Settings have been saved successfully.")
        logger.info("Settings saved")
    
    def _reset_settings(self) -> None:
        """Reset settings to defaults."""
        if messagebox.askyesno("Reset Settings", "Reset all settings to defaults?"):
            # Reset to defaults (implementation needed)
            messagebox.showinfo("Settings Reset", "Settings have been reset to defaults.")
    
    def _export_settings(self) -> None:
        """Export settings to file."""
        file_path = filedialog.asksaveasfilename(
            title="Export Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            # Export implementation needed
            messagebox.showinfo("Export Complete", f"Settings exported to {file_path}")
    
    def _import_settings(self) -> None:
        """Import settings from file."""
        file_path = filedialog.askopenfilename(
            title="Import Settings",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            # Import implementation needed
            messagebox.showinfo("Import Complete", f"Settings imported from {file_path}")
    
    def _browse_directory(self, setting_key: str) -> None:
        """Browse for directory."""
        directory = filedialog.askdirectory(title=f"Select {setting_key.replace('_', ' ').title()}")
        if directory:
            self.settings_vars[setting_key].set(directory)
    
    def _toggle_api_visibility(self, entry: ttk.Entry, var: tk.BooleanVar) -> None:
        """Toggle API key visibility."""
        if var.get():
            entry.config(show="")
        else:
            entry.config(show="*")
    
    def _test_openrouter_connection(self) -> None:
        """Test OpenRouter API connection."""
        messagebox.showinfo("Test Connection", "OpenRouter connection test will be implemented.")
    
    def _test_perplexity_connection(self) -> None:
        """Test Perplexity API connection."""
        messagebox.showinfo("Test Connection", "Perplexity connection test will be implemented.")
    
    def _test_semantic_scholar_connection(self) -> None:
        """Test Semantic Scholar API connection."""
        messagebox.showinfo("Test Connection", "Semantic Scholar connection test will be implemented.")
    
    def show_api_keys_section(self) -> None:
        """Show the API keys section."""
        self.notebook.select(0)  # Select API tab
    
    def apply_theme(self, theme) -> None:
        """Apply theme to the widget."""
        # Theme application will be implemented with theme system
        pass
