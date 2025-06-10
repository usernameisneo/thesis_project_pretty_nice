"""
Model Selector Widget for the AI-Powered Thesis Assistant.

This widget provides comprehensive AI model selection and configuration
with filtering, sorting, and detailed model information display.

Features:
    - Complete model catalog browsing
    - Advanced filtering and sorting
    - Model comparison and selection
    - Cost and performance metrics
    - Real-time availability status
    - Model configuration options

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Local imports
from core.config import Config
from core.exceptions import APIError
from api.openrouter_client import OpenRouterClient, ModelInfo

logger = logging.getLogger(__name__)


class ModelSelectorWidget(ttk.Frame):
    """
    Model selection widget with comprehensive model management.
    
    This widget provides a full-featured interface for browsing,
    filtering, and selecting AI models from OpenRouter.
    """
    
    def __init__(self, parent: tk.Widget, config: Config):
        """
        Initialize the model selector widget.
        
        Args:
            parent: Parent widget
            config: Application configuration
        """
        super().__init__(parent)
        self.config = config
        self.openrouter_client = None
        self.all_models = []
        self.filtered_models = []
        self.selected_model = None
        
        self._setup_ui()
        self._initialize_client()
        
        logger.info("Model selector widget initialized")
    
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        # Title
        title_label = ttk.Label(
            self,
            text="🤖 AI Model Selection",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=(0, 15))
        
        # Create main sections
        self._create_filter_section()
        self._create_model_list_section()
        self._create_model_details_section()
        self._create_action_section()
    
    def _create_filter_section(self) -> None:
        """Create the model filtering section."""
        filter_frame = ttk.LabelFrame(self, text="Filters & Search", padding=10)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Search box
        search_frame = ttk.Frame(filter_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=(0, 10))
        search_entry.bind('<KeyRelease>', self._on_search_change)
        
        ttk.Button(
            search_frame,
            text="🔍 Search",
            command=self._apply_filters
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            search_frame,
            text="🔄 Refresh",
            command=self._refresh_models
        ).pack(side=tk.RIGHT)
        
        # Filter options
        options_frame = ttk.Frame(filter_frame)
        options_frame.pack(fill=tk.X)
        
        # Provider filter
        ttk.Label(options_frame, text="Provider:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.provider_var = tk.StringVar(value="All")
        provider_combo = ttk.Combobox(
            options_frame,
            textvariable=self.provider_var,
            values=["All", "OpenAI", "Anthropic", "Google", "Meta", "Mistral", "Others"],
            state="readonly",
            width=15
        )
        provider_combo.grid(row=0, column=1, padx=(0, 20))
        provider_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_filters())
        
        # Model type filter
        ttk.Label(options_frame, text="Type:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.type_var = tk.StringVar(value="All")
        type_combo = ttk.Combobox(
            options_frame,
            textvariable=self.type_var,
            values=["All", "Chat", "Completion", "Code", "Image"],
            state="readonly",
            width=15
        )
        type_combo.grid(row=0, column=3, padx=(0, 20))
        type_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_filters())
        
        # Sort options
        ttk.Label(options_frame, text="Sort by:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(10, 0))
        self.sort_var = tk.StringVar(value="Name")
        sort_combo = ttk.Combobox(
            options_frame,
            textvariable=self.sort_var,
            values=["Name", "Provider", "Cost (Low to High)", "Cost (High to Low)", "Context Length"],
            state="readonly",
            width=20
        )
        sort_combo.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=(10, 0))
        sort_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_filters())
    
    def _create_model_list_section(self) -> None:
        """Create the model list section."""
        list_frame = ttk.LabelFrame(self, text="Available Models", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Treeview for model list
        columns = ('Name', 'Provider', 'Cost', 'Context', 'Status')
        self.model_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)
        
        # Configure columns
        self.model_tree.heading('Name', text='Model Name')
        self.model_tree.heading('Provider', text='Provider')
        self.model_tree.heading('Cost', text='Cost/1K tokens')
        self.model_tree.heading('Context', text='Context Length')
        self.model_tree.heading('Status', text='Status')
        
        self.model_tree.column('Name', width=200)
        self.model_tree.column('Provider', width=100)
        self.model_tree.column('Cost', width=100)
        self.model_tree.column('Context', width=100)
        self.model_tree.column('Status', width=80)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.model_tree.yview)
        h_scrollbar = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.model_tree.xview)
        self.model_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.model_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.model_tree.bind('<<TreeviewSelect>>', self._on_model_select)
        self.model_tree.bind('<Double-1>', self._on_model_double_click)
    
    def _create_model_details_section(self) -> None:
        """Create the model details section."""
        details_frame = ttk.LabelFrame(self, text="Model Details", padding=10)
        details_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Details text widget
        self.details_text = tk.Text(
            details_frame,
            height=6,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=('Arial', 10),
            bg='#f8f9fa'
        )
        
        details_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initial message
        self._update_model_details("Select a model to view details...")
    
    def _create_action_section(self) -> None:
        """Create the action buttons section."""
        action_frame = ttk.Frame(self)
        action_frame.pack(fill=tk.X)
        
        # Selection info
        self.selection_var = tk.StringVar(value="No model selected")
        selection_label = ttk.Label(action_frame, textvariable=self.selection_var)
        selection_label.pack(side=tk.LEFT)
        
        # Action buttons
        button_frame = ttk.Frame(action_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.select_button = ttk.Button(
            button_frame,
            text="✅ Select Model",
            command=self._select_model,
            state=tk.DISABLED
        )
        self.select_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="🧪 Test Model",
            command=self._test_model
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="📊 Compare",
            command=self._compare_models
        ).pack(side=tk.LEFT)
    
    def _initialize_client(self) -> None:
        """Initialize the OpenRouter client."""
        try:
            api_key = self.config.get('openrouter_api_key', '')
            if api_key:
                self.openrouter_client = OpenRouterClient(api_key)
                self._refresh_models()
            else:
                self._update_model_details("Please configure your OpenRouter API key in settings.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self._update_model_details(f"Failed to connect to OpenRouter: {e}")
    
    def _refresh_models(self) -> None:
        """Refresh the models list from OpenRouter."""
        if not self.openrouter_client:
            return
        
        def refresh_thread():
            try:
                self.selection_var.set("Loading models...")
                models = self.openrouter_client.get_models()
                
                # Update on main thread
                self.after(0, self._update_models_list, models)
                
            except Exception as e:
                logger.error(f"Failed to refresh models: {e}")
                self.after(0, lambda: self.selection_var.set("Failed to load models"))
        
        threading.Thread(target=refresh_thread, daemon=True).start()
    
    def _update_models_list(self, models: List[ModelInfo]) -> None:
        """Update the models list display."""
        self.all_models = models
        self._apply_filters()
        self.selection_var.set(f"Loaded {len(models)} models")
    
    def _apply_filters(self) -> None:
        """Apply current filters to the models list."""
        if not self.all_models:
            return
        
        # Start with all models
        filtered = self.all_models.copy()
        
        # Apply search filter
        search_term = self.search_var.get().lower()
        if search_term:
            filtered = [m for m in filtered if search_term in m.name.lower() or 
                       search_term in m.description.lower()]
        
        # Apply provider filter
        provider = self.provider_var.get()
        if provider != "All":
            if provider == "Others":
                known_providers = ["OpenAI", "Anthropic", "Google", "Meta", "Mistral"]
                filtered = [m for m in filtered if not any(p.lower() in m.name.lower() for p in known_providers)]
            else:
                filtered = [m for m in filtered if provider.lower() in m.name.lower()]
        
        # Apply sorting
        sort_by = self.sort_var.get()
        if sort_by == "Name":
            filtered.sort(key=lambda m: m.name)
        elif sort_by == "Provider":
            filtered.sort(key=lambda m: m.name.split('/')[0] if '/' in m.name else m.name)
        elif sort_by == "Cost (Low to High)":
            filtered.sort(key=lambda m: m.pricing.prompt if m.pricing else float('inf'))
        elif sort_by == "Cost (High to Low)":
            filtered.sort(key=lambda m: m.pricing.prompt if m.pricing else 0, reverse=True)
        elif sort_by == "Context Length":
            filtered.sort(key=lambda m: m.context_length, reverse=True)
        
        self.filtered_models = filtered
        self._populate_model_tree()
    
    def _populate_model_tree(self) -> None:
        """Populate the model tree with filtered models."""
        # Clear existing items
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        
        # Add filtered models
        for model in self.filtered_models:
            # Format cost
            cost_str = f"${model.pricing.prompt:.4f}" if model.pricing else "N/A"
            
            # Format context length
            context_str = f"{model.context_length:,}" if model.context_length else "N/A"
            
            # Determine status
            status = "Available" if model.context_length > 0 else "Limited"
            
            # Extract provider name
            provider = model.name.split('/')[0] if '/' in model.name else "Unknown"
            
            self.model_tree.insert('', tk.END, values=(
                model.name,
                provider,
                cost_str,
                context_str,
                status
            ), tags=(model.id,))
    
    def _on_search_change(self, event) -> None:
        """Handle search text change."""
        # Apply filters after a short delay to avoid excessive filtering
        self.after(500, self._apply_filters)
    
    def _on_model_select(self, event) -> None:
        """Handle model selection in the tree."""
        selection = self.model_tree.selection()
        if selection:
            item = selection[0]
            model_id = self.model_tree.item(item, 'tags')[0]
            
            # Find the selected model
            selected_model = next((m for m in self.filtered_models if m.id == model_id), None)
            if selected_model:
                self.selected_model = selected_model
                self._update_model_details_from_model(selected_model)
                self.select_button.config(state=tk.NORMAL)
                self.selection_var.set(f"Selected: {selected_model.name}")
    
    def _on_model_double_click(self, event) -> None:
        """Handle double-click on model (auto-select)."""
        self._select_model()
    
    def _update_model_details(self, text: str) -> None:
        """Update the model details display."""
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert("1.0", text)
        self.details_text.config(state=tk.DISABLED)
    
    def _update_model_details_from_model(self, model: ModelInfo) -> None:
        """Update model details from a ModelInfo object."""
        details = f"Model: {model.name}\n"
        details += f"ID: {model.id}\n"
        details += f"Description: {model.description}\n\n"
        
        if model.pricing:
            details += f"Pricing:\n"
            details += f"  Prompt: ${model.pricing.prompt:.6f} per token\n"
            details += f"  Completion: ${model.pricing.completion:.6f} per token\n\n"
        
        details += f"Context Length: {model.context_length:,} tokens\n"
        details += f"Top Provider: {model.top_provider}\n\n"
        
        if hasattr(model, 'architecture') and model.architecture:
            details += f"Architecture: {model.architecture}\n"
        
        if hasattr(model, 'modality') and model.modality:
            details += f"Modality: {', '.join(model.modality)}\n"
        
        self._update_model_details(details)
    
    def _select_model(self) -> None:
        """Select the current model for use."""
        if not self.selected_model:
            messagebox.showwarning("No Selection", "Please select a model first.")
            return
        
        # Save selection to config
        self.config.set('selected_model_id', self.selected_model.id)
        self.config.set('selected_model_name', self.selected_model.name)
        self.config.save()
        
        messagebox.showinfo(
            "Model Selected",
            f"Selected model: {self.selected_model.name}\n\nThis model will be used for AI operations."
        )
        
        logger.info(f"Selected model: {self.selected_model.name}")
    
    def _test_model(self) -> None:
        """Test the selected model with a simple query."""
        if not self.selected_model:
            messagebox.showwarning("No Selection", "Please select a model first.")
            return
        
        messagebox.showinfo("Test Model", "Model testing will be implemented.")
    
    def _compare_models(self) -> None:
        """Compare multiple selected models."""
        messagebox.showinfo("Compare Models", "Model comparison will be implemented.")
    
    def get_selected_model(self) -> Optional[ModelInfo]:
        """Get the currently selected model."""
        return self.selected_model
    
    def apply_theme(self, theme) -> None:
        """Apply theme to the widget."""
        # Theme application will be implemented with theme system
        pass
