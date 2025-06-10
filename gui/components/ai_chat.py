"""
AI Chat Widget for the AI-Powered Thesis Assistant.

This widget provides a comprehensive AI chat interface with model selection,
conversation history, and integration with the document knowledge base.

Features:
    - Real-time AI chat interface
    - Model selection and configuration
    - Conversation history management
    - Document context integration
    - Streaming responses
    - Export conversations

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio
import json

# Local imports
from core.config import Config
from core.exceptions import APIError
from api.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


class AIChatWidget(ttk.Frame):
    """
    AI chat interface widget with comprehensive chat functionality.
    
    This widget provides a full-featured chat interface for interacting
    with AI models through the OpenRouter API.
    """
    
    def __init__(self, parent: tk.Widget, config: Config):
        """
        Initialize the AI chat widget.
        
        Args:
            parent: Parent widget
            config: Application configuration
        """
        super().__init__(parent)
        self.config = config
        self.openrouter_client = None
        self.current_model = None
        self.conversation_history = []
        self.is_streaming = False
        self.document_context = ""
        
        self._setup_ui()
        self._initialize_client()
        
        logger.info("AI chat widget initialized")
    
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="🤖 AI Chat Assistant",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # Create main layout
        self._create_chat_area(main_frame)
        self._create_input_area(main_frame)
        self._create_control_panel(main_frame)
    
    def _create_chat_area(self, parent: tk.Widget) -> None:
        """Create the chat display area."""
        # Chat frame
        chat_frame = ttk.LabelFrame(parent, text="Conversation", padding=10)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Chat display with scrollbar
        chat_container = ttk.Frame(chat_frame)
        chat_container.pack(fill=tk.BOTH, expand=True)
        
        # Text widget for chat display
        self.chat_display = tk.Text(
            chat_container,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=('Arial', 11),
            bg='#f8f9fa',
            fg='#212529',
            padx=15,
            pady=15
        )
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(chat_container, orient=tk.VERTICAL, command=self.chat_display.yview)
        self.chat_display.configure(yscrollcommand=scrollbar.set)
        
        # Pack chat display and scrollbar
        self.chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure text tags for styling
        self.chat_display.tag_configure("user", foreground="#0066cc", font=('Arial', 11, 'bold'))
        self.chat_display.tag_configure("assistant", foreground="#28a745", font=('Arial', 11, 'bold'))
        self.chat_display.tag_configure("system", foreground="#6c757d", font=('Arial', 10, 'italic'))
        self.chat_display.tag_configure("timestamp", foreground="#6c757d", font=('Arial', 9))
        
        # Welcome message
        self._add_system_message("Welcome to AI Chat Assistant! Select a model and start chatting.")
    
    def _create_input_area(self, parent: tk.Widget) -> None:
        """Create the message input area."""
        input_frame = ttk.LabelFrame(parent, text="Message", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Input text area
        input_container = ttk.Frame(input_frame)
        input_container.pack(fill=tk.BOTH, expand=True)
        
        # Text input with scrollbar
        self.message_input = tk.Text(
            input_container,
            height=4,
            wrap=tk.WORD,
            font=('Arial', 11),
            bg='white',
            fg='#212529',
            padx=10,
            pady=10
        )
        
        input_scrollbar = ttk.Scrollbar(input_container, orient=tk.VERTICAL, command=self.message_input.yview)
        self.message_input.configure(yscrollcommand=input_scrollbar.set)
        
        # Pack input and scrollbar
        self.message_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        input_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind Enter key for sending
        self.message_input.bind('<Control-Return>', self._send_message)
        
        # Button frame
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Send button
        self.send_button = ttk.Button(
            button_frame,
            text="📤 Send (Ctrl+Enter)",
            command=self._send_message
        )
        self.send_button.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Clear button
        ttk.Button(
            button_frame,
            text="🗑️ Clear Input",
            command=self._clear_input
        ).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Context toggle
        self.use_context_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            button_frame,
            text="Use document context",
            variable=self.use_context_var
        ).pack(side=tk.LEFT)
    
    def _create_control_panel(self, parent: tk.Widget) -> None:
        """Create the control panel."""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        control_frame.pack(fill=tk.X)
        
        # Left side: Model selection
        left_frame = ttk.Frame(control_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(left_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            left_frame,
            textvariable=self.model_var,
            state="readonly",
            width=30
        )
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_change)
        
        ttk.Button(
            left_frame,
            text="🔄 Refresh Models",
            command=self._refresh_models
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        # Right side: Conversation controls
        right_frame = ttk.Frame(control_frame)
        right_frame.pack(side=tk.RIGHT)
        
        ttk.Button(
            right_frame,
            text="💾 Save Chat",
            command=self._save_conversation
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            right_frame,
            text="📂 Load Chat",
            command=self._load_conversation
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            right_frame,
            text="🗑️ Clear Chat",
            command=self._clear_conversation
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Status indicator
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(pady=(10, 0))
    
    def _initialize_client(self) -> None:
        """Initialize the OpenRouter client."""
        try:
            api_key = self.config.get('openrouter_api_key', '')
            if api_key:
                self.openrouter_client = OpenRouterClient(api_key)
                self._refresh_models()
                self.status_var.set("Connected to OpenRouter")
            else:
                self.status_var.set("No API key configured")
                self._add_system_message("Please configure your OpenRouter API key in settings.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self.status_var.set("Connection failed")
            self._add_system_message(f"Failed to connect to OpenRouter: {e}")
    
    def _refresh_models(self) -> None:
        """Refresh the available models list."""
        if not self.openrouter_client:
            return
        
        def refresh_thread():
            try:
                self.status_var.set("Loading models...")
                models = self.openrouter_client.get_models()
                
                # Update model list on main thread
                self.after(0, self._update_model_list, models)
                
            except Exception as e:
                logger.error(f"Failed to refresh models: {e}")
                self.after(0, lambda: self.status_var.set("Failed to load models"))
        
        threading.Thread(target=refresh_thread, daemon=True).start()
    
    def _update_model_list(self, models: List[Dict[str, Any]]) -> None:
        """Update the model selection list."""
        model_names = [f"{model['name']} - ${model.get('pricing', {}).get('prompt', 'N/A')}/1K tokens" 
                      for model in models[:50]]  # Limit to first 50 models
        
        self.model_combo['values'] = model_names
        
        if model_names:
            self.model_combo.set(model_names[0])
            self.current_model = models[0]['id']
        
        self.status_var.set(f"Loaded {len(models)} models")
    
    def _on_model_change(self, event) -> None:
        """Handle model selection change."""
        selection = self.model_combo.get()
        if selection:
            # Extract model ID from selection
            model_name = selection.split(' - ')[0]
            self.current_model = model_name
            self.status_var.set(f"Selected model: {model_name}")
    
    def _send_message(self, event=None) -> None:
        """Send a message to the AI."""
        if self.is_streaming:
            messagebox.showwarning("Busy", "Please wait for the current response to complete.")
            return
        
        message = self.message_input.get("1.0", tk.END).strip()
        if not message:
            return
        
        if not self.current_model:
            messagebox.showwarning("No Model", "Please select an AI model first.")
            return
        
        # Add user message to chat
        self._add_user_message(message)
        
        # Clear input
        self.message_input.delete("1.0", tk.END)
        
        # Send to AI in background thread
        threading.Thread(
            target=self._send_to_ai_thread,
            args=(message,),
            daemon=True
        ).start()
    
    def _send_to_ai_thread(self, message: str) -> None:
        """Send message to AI in background thread."""
        try:
            self.is_streaming = True
            self.after(0, lambda: self.status_var.set("Generating response..."))
            self.after(0, lambda: self.send_button.config(state='disabled'))

            # Check if API key is configured
            api_key = self.config.get('openrouter_api_key', '')
            if not api_key:
                self.after(0, self._add_system_message, "❌ OpenRouter API key not configured. Please set it in Settings.")
                return

            # Ensure client has API key
            if not self.openrouter_client.api_key:
                self.openrouter_client.set_api_key(api_key)

            # Prepare conversation context
            messages = []

            # Add system message if using context
            if self.use_context_var.get():
                system_content = "You are an AI assistant helping with academic thesis research. Use the provided document context to give accurate, well-sourced responses."

                # Add document context if available
                if hasattr(self, 'document_context') and self.document_context:
                    system_content += f"\n\nDocument Context:\n{self.document_context[:2000]}..."

                messages.append({
                    "role": "system",
                    "content": system_content
                })

            # Add conversation history (last 10 messages)
            for msg in self.conversation_history[-10:]:
                messages.append(msg)

            # Add current message
            messages.append({"role": "user", "content": message})

            # Send to OpenRouter with error handling
            try:
                response = self.openrouter_client.chat_completion(
                    model=self.current_model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7
                )

                # Extract response text
                if response and 'choices' in response and response['choices']:
                    ai_response = response['choices'][0]['message']['content']

                    # Add to conversation history
                    self.conversation_history.append({"role": "user", "content": message})
                    self.conversation_history.append({"role": "assistant", "content": ai_response})

                    # Display response
                    self.after(0, self._add_assistant_message, ai_response)
                else:
                    self.after(0, self._add_system_message, "❌ No response received from AI. Please check your API key and model selection.")

            except Exception as api_error:
                error_msg = str(api_error)
                if "401" in error_msg or "unauthorized" in error_msg.lower():
                    self.after(0, self._add_system_message, "❌ API authentication failed. Please check your OpenRouter API key in Settings.")
                elif "429" in error_msg or "rate limit" in error_msg.lower():
                    self.after(0, self._add_system_message, "❌ Rate limit exceeded. Please wait a moment before trying again.")
                elif "400" in error_msg or "bad request" in error_msg.lower():
                    self.after(0, self._add_system_message, "❌ Invalid request. Please check your model selection and try again.")
                else:
                    self.after(0, self._add_system_message, f"❌ API Error: {error_msg}")

                logger.error(f"OpenRouter API error: {api_error}")
            
        except Exception as e:
            logger.error(f"AI chat error: {e}")
            self.after(0, self._add_system_message, f"Error: {e}")
        
        finally:
            self.is_streaming = False
            self.after(0, lambda: self.status_var.set("Ready"))
            self.after(0, lambda: self.send_button.config(state='normal'))
    
    def _add_user_message(self, message: str) -> None:
        """Add a user message to the chat display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.chat_display.insert(tk.END, "You: ", "user")
        self.chat_display.insert(tk.END, f"{message}\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _add_assistant_message(self, message: str) -> None:
        """Add an assistant message to the chat display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.chat_display.insert(tk.END, "AI: ", "assistant")
        self.chat_display.insert(tk.END, f"{message}\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _add_system_message(self, message: str) -> None:
        """Add a system message to the chat display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.chat_display.insert(tk.END, "System: ", "system")
        self.chat_display.insert(tk.END, f"{message}\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _clear_input(self) -> None:
        """Clear the message input."""
        self.message_input.delete("1.0", tk.END)
    
    def _clear_conversation(self) -> None:
        """Clear the conversation history."""
        if messagebox.askyesno("Clear Chat", "Clear the entire conversation?"):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.conversation_history.clear()
            self._add_system_message("Conversation cleared.")
    
    def _save_conversation(self) -> None:
        """Save the conversation to a file."""
        if not self.conversation_history:
            messagebox.showinfo("No Conversation", "No conversation to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Conversation",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    if file_path.endswith('.json'):
                        json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
                    else:
                        for msg in self.conversation_history:
                            f.write(f"{msg['role'].title()}: {msg['content']}\n\n")
                
                self._add_system_message(f"Conversation saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save conversation: {e}")
    
    def _load_conversation(self) -> None:
        """Load a conversation from a file."""
        file_path = filedialog.askopenfilename(
            title="Load Conversation",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_history = json.load(f)
                
                # Validate format
                if isinstance(loaded_history, list) and all(
                    isinstance(msg, dict) and 'role' in msg and 'content' in msg 
                    for msg in loaded_history
                ):
                    self.conversation_history = loaded_history
                    self._rebuild_chat_display()
                    self._add_system_message(f"Conversation loaded from {file_path}")
                else:
                    messagebox.showerror("Load Error", "Invalid conversation file format.")
                    
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load conversation: {e}")
    
    def _rebuild_chat_display(self) -> None:
        """Rebuild the chat display from conversation history."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        
        for msg in self.conversation_history:
            if msg['role'] == 'user':
                self._add_user_message(msg['content'])
            elif msg['role'] == 'assistant':
                self._add_assistant_message(msg['content'])
        
        self.chat_display.config(state=tk.DISABLED)
    
    def apply_theme(self, theme) -> None:
        """Apply theme to the widget."""
        # Theme application will be implemented with theme system
        pass

    def set_document_context(self, context: str) -> None:
        """
        Set document context for AI conversations.

        Args:
            context: Document context to provide to the AI
        """
        self.document_context = context
        logger.info(f"Document context set ({len(context)} characters)")

    def clear_document_context(self) -> None:
        """Clear the document context."""
        self.document_context = ""
        logger.info("Document context cleared")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.

        Returns:
            List of conversation messages
        """
        return self.conversation_history.copy()

    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self._add_system_message("Conversation history cleared.")
        logger.info("Conversation history cleared")
