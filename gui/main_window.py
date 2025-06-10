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

        # Set up component connections
        self._setup_component_connections()
    
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

    def _setup_component_connections(self) -> None:
        """Set up connections between components for data sharing."""
        # Connect document processor to AI chat for context sharing
        if 'document_processor' in self.widgets and 'ai_chat' in self.widgets:
            # Set up a callback to update AI chat context when documents are processed
            doc_processor = self.widgets['document_processor']
            ai_chat = self.widgets['ai_chat']

            # Store original process method
            original_process = doc_processor._process_documents

            def enhanced_process():
                # Call original processing
                original_process()

                # Update AI chat context with processed documents
                try:
                    processed_docs = doc_processor.get_processed_documents()
                    if processed_docs:
                        # Create context from first few documents
                        context_parts = []
                        for doc_path, doc_text, _ in processed_docs[:3]:  # Limit to first 3 docs
                            doc_name = Path(doc_path).name
                            context_parts.append(f"Document: {doc_name}\n{doc_text[:1000]}...")

                        combined_context = "\n\n".join(context_parts)
                        ai_chat.set_document_context(combined_context)

                        logger.info(f"Updated AI chat context with {len(processed_docs)} documents")
                except Exception as e:
                    logger.warning(f"Failed to update AI chat context: {e}")

            # Replace the process method
            doc_processor._process_documents = enhanced_process
    
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
        """Run comprehensive thesis analysis."""
        try:
            self.update_status("Starting comprehensive analysis...")

            # Check if we have documents to analyze
            doc_processor = self.widgets.get('document_processor')
            if not doc_processor or not doc_processor.get_processed_documents():
                messagebox.showwarning("No Documents", "Please process some documents first.")
                return

            # Get thesis file from user
            thesis_file = filedialog.askopenfilename(
                title="Select Thesis File",
                filetypes=[
                    ("PDF files", "*.pdf"),
                    ("Text files", "*.txt"),
                    ("Markdown files", "*.md"),
                    ("All files", "*.*")
                ]
            )

            if not thesis_file:
                return

            # Run analysis in background thread
            import threading
            threading.Thread(
                target=self._run_analysis_thread,
                args=(thesis_file,),
                daemon=True
            ).start()

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            messagebox.showerror("Analysis Error", f"Failed to start analysis: {e}")

    def _run_analysis_thread(self, thesis_file: str) -> None:
        """Run analysis in background thread."""
        try:
            from scripts.complete_thesis_analysis import CompleteThesisAnalysisSystem
            import asyncio

            async def run_analysis():
                system = CompleteThesisAnalysisSystem()
                await system.initialize_system()

                # Get sources directory from document processor
                sources_dir = self.config.get('data_dir', 'data')
                output_dir = Path(thesis_file).parent / "analysis_output"

                await system.run_complete_analysis(
                    thesis_file=thesis_file,
                    sources_directory=sources_dir,
                    output_directory=str(output_dir)
                )

            # Update status
            self.after(0, lambda: self.update_status("Running comprehensive analysis..."))

            # Run analysis
            asyncio.run(run_analysis())

            # Update status on completion
            self.after(0, lambda: self.update_status("Analysis completed successfully!"))
            self.after(0, lambda: messagebox.showinfo("Analysis Complete", "Thesis analysis completed successfully!"))

        except Exception as e:
            logger.error(f"Analysis thread failed: {e}")
            self.after(0, lambda: self.update_status(f"Analysis failed: {e}"))
            self.after(0, lambda: messagebox.showerror("Analysis Error", f"Analysis failed: {e}"))

    def _validate_citations(self) -> None:
        """Validate citations in processed documents."""
        try:
            self.update_status("Validating citations...")

            # Check if we have documents to validate
            doc_processor = self.widgets.get('document_processor')
            if not doc_processor or not doc_processor.get_processed_documents():
                messagebox.showwarning("No Documents", "Please process some documents first.")
                return

            # Run validation in background thread
            import threading
            threading.Thread(
                target=self._validate_citations_thread,
                daemon=True
            ).start()

        except Exception as e:
            logger.error(f"Citation validation failed: {e}")
            messagebox.showerror("Validation Error", f"Failed to validate citations: {e}")

    def _validate_citations_thread(self) -> None:
        """Validate citations in background thread."""
        try:
            from reasoning.enhanced_citation_engine import EnhancedCitationEngine
            from reasoning.apa7_compliance_engine import APA7ComplianceEngine

            # Initialize engines
            citation_engine = EnhancedCitationEngine()
            apa7_engine = APA7ComplianceEngine()

            # Get processed documents
            doc_processor = self.widgets.get('document_processor')
            documents = doc_processor.get_processed_documents()

            validation_results = []
            total_docs = len(documents)

            for i, (doc_path, doc_text, doc_metadata) in enumerate(documents):
                # Update progress
                progress = (i / total_docs) * 100
                self.after(0, lambda p=progress: self.update_status(f"Validating citations... {p:.1f}%"))

                # Validate citations in document
                citations = citation_engine.extract_citations(doc_text)
                apa7_compliance = apa7_engine.validate_citations(citations)

                validation_results.append({
                    'document': Path(doc_path).name,
                    'citations_found': len(citations),
                    'apa7_compliant': sum(1 for c in apa7_compliance if c.is_compliant),
                    'issues': [c.issues for c in apa7_compliance if not c.is_compliant]
                })

            # Show results
            self._show_validation_results(validation_results)

        except Exception as e:
            logger.error(f"Citation validation thread failed: {e}")
            self.after(0, lambda: self.update_status(f"Validation failed: {e}"))
            self.after(0, lambda: messagebox.showerror("Validation Error", f"Citation validation failed: {e}"))

    def _show_validation_results(self, results: list) -> None:
        """Show citation validation results."""
        # Create results window
        results_window = tk.Toplevel(self)
        results_window.title("Citation Validation Results")
        results_window.geometry("600x400")

        # Create text widget with scrollbar
        frame = ttk.Frame(results_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text_widget = tk.Text(frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Format results
        results_text = "CITATION VALIDATION RESULTS\n" + "="*50 + "\n\n"

        for result in results:
            results_text += f"Document: {result['document']}\n"
            results_text += f"Citations Found: {result['citations_found']}\n"
            results_text += f"APA7 Compliant: {result['apa7_compliant']}\n"

            if result['issues']:
                results_text += "Issues Found:\n"
                for issues in result['issues']:
                    for issue in issues:
                        results_text += f"  - {issue}\n"

            results_text += "\n" + "-"*30 + "\n\n"

        text_widget.insert(tk.END, results_text)
        text_widget.config(state=tk.DISABLED)

        self.after(0, lambda: self.update_status("Citation validation completed"))

    def _generate_bibliography(self) -> None:
        """Generate bibliography from processed documents."""
        try:
            self.update_status("Generating bibliography...")

            # Check if we have documents to process
            doc_processor = self.widgets.get('document_processor')
            if not doc_processor or not doc_processor.get_processed_documents():
                messagebox.showwarning("No Documents", "Please process some documents first.")
                return

            # Get output file from user
            output_file = filedialog.asksaveasfilename(
                title="Save Bibliography",
                defaultextension=".txt",
                filetypes=[
                    ("Text files", "*.txt"),
                    ("Markdown files", "*.md"),
                    ("All files", "*.*")
                ]
            )

            if not output_file:
                return

            # Run generation in background thread
            import threading
            threading.Thread(
                target=self._generate_bibliography_thread,
                args=(output_file,),
                daemon=True
            ).start()

        except Exception as e:
            logger.error(f"Bibliography generation failed: {e}")
            messagebox.showerror("Generation Error", f"Failed to generate bibliography: {e}")

    def _generate_bibliography_thread(self, output_file: str) -> None:
        """Generate bibliography in background thread."""
        try:
            from reasoning.enhanced_citation_engine import EnhancedCitationEngine
            from reasoning.apa7_compliance_engine import APA7ComplianceEngine

            # Initialize engines
            citation_engine = EnhancedCitationEngine()
            apa7_engine = APA7ComplianceEngine()

            # Get processed documents
            doc_processor = self.widgets.get('document_processor')
            documents = doc_processor.get_processed_documents()

            all_citations = []
            total_docs = len(documents)

            for i, (doc_path, doc_text, doc_metadata) in enumerate(documents):
                # Update progress
                progress = (i / total_docs) * 100
                self.after(0, lambda p=progress: self.update_status(f"Extracting citations... {p:.1f}%"))

                # Extract citations from document
                citations = citation_engine.extract_citations(doc_text)
                all_citations.extend(citations)

            # Remove duplicates and format
            unique_citations = list({c.id: c for c in all_citations}.values())
            formatted_bibliography = apa7_engine.format_bibliography(unique_citations)

            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("BIBLIOGRAPHY\n")
                f.write("="*50 + "\n\n")
                f.write(formatted_bibliography)

            # Show completion message
            self.after(0, lambda: self.update_status("Bibliography generated successfully"))
            self.after(0, lambda: messagebox.showinfo(
                "Bibliography Generated",
                f"Bibliography saved to:\n{output_file}\n\nFound {len(unique_citations)} unique citations."
            ))

        except Exception as e:
            logger.error(f"Bibliography generation thread failed: {e}")
            self.after(0, lambda: self.update_status(f"Generation failed: {e}"))
            self.after(0, lambda: messagebox.showerror("Generation Error", f"Bibliography generation failed: {e}"))
    
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
