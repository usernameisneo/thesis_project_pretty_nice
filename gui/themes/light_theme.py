"""
Light Theme for the AI-Powered Thesis Assistant.

This module implements a clean light theme with professional aesthetics
and high contrast for optimal readability.

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any


class LightTheme:
    """
    Professional light theme implementation.
    
    This theme provides a clean, bright interface optimized for
    high contrast and professional appearance.
    """
    
    def __init__(self):
        """Initialize the light theme."""
        self.colors = {
            # Primary colors
            'bg_primary': '#ffffff',      # Main background
            'bg_secondary': '#f8f9fa',    # Secondary background
            'bg_tertiary': '#e9ecef',     # Tertiary background
            
            # Text colors
            'text_primary': '#212529',     # Primary text
            'text_secondary': '#6c757d',   # Secondary text
            'text_disabled': '#adb5bd',    # Disabled text
            
            # Accent colors
            'accent_primary': '#007bff',   # Primary accent (blue)
            'accent_secondary': '#28a745', # Secondary accent (green)
            'accent_warning': '#ffc107',   # Warning (yellow)
            'accent_danger': '#dc3545',    # Danger (red)
            
            # UI element colors
            'border': '#dee2e6',           # Borders
            'hover': '#e9ecef',            # Hover states
            'selected': '#007bff',         # Selected items
            'focus': '#80bdff',            # Focus indicators
            
            # Special colors
            'success': '#28a745',          # Success indicators
            'warning': '#ffc107',          # Warning indicators
            'error': '#dc3545',            # Error indicators
            'info': '#17a2b8',             # Info indicators
        }
        
        self.fonts = {
            'default': ('Segoe UI', 9),
            'heading': ('Segoe UI', 12, 'bold'),
            'title': ('Segoe UI', 14, 'bold'),
            'code': ('Consolas', 9),
        }
    
    def apply_to_root(self, root: tk.Tk) -> None:
        """
        Apply the light theme to the root window.
        
        Args:
            root: Root tkinter window
        """
        # Configure root window
        root.configure(bg=self.colors['bg_primary'])
        
        # Create and configure ttk style
        style = ttk.Style()
        
        # Configure ttk theme
        style.theme_use('clam')  # Use clam as base theme
        
        # Configure general styles
        style.configure('.',
            background=self.colors['bg_primary'],
            foreground=self.colors['text_primary'],
            bordercolor=self.colors['border'],
            focuscolor=self.colors['focus'],
            selectbackground=self.colors['selected'],
            selectforeground='white',
            font=self.fonts['default']
        )
        
        # Configure Frame styles
        style.configure('TFrame',
            background=self.colors['bg_primary'],
            borderwidth=0
        )
        
        # Configure Label styles
        style.configure('TLabel',
            background=self.colors['bg_primary'],
            foreground=self.colors['text_primary'],
            font=self.fonts['default']
        )
        
        style.configure('Heading.TLabel',
            background=self.colors['bg_primary'],
            foreground=self.colors['text_primary'],
            font=self.fonts['heading']
        )
        
        style.configure('Title.TLabel',
            background=self.colors['bg_primary'],
            foreground=self.colors['text_primary'],
            font=self.fonts['title']
        )
        
        # Configure Button styles
        style.configure('TButton',
            background=self.colors['bg_secondary'],
            foreground=self.colors['text_primary'],
            borderwidth=1,
            focuscolor=self.colors['focus'],
            font=self.fonts['default']
        )
        
        style.map('TButton',
            background=[
                ('active', self.colors['hover']),
                ('pressed', self.colors['selected'])
            ],
            foreground=[
                ('active', self.colors['text_primary']),
                ('pressed', 'white')
            ]
        )
        
        # Configure Entry styles
        style.configure('TEntry',
            fieldbackground='white',
            background='white',
            foreground=self.colors['text_primary'],
            bordercolor=self.colors['border'],
            insertcolor=self.colors['text_primary'],
            font=self.fonts['default']
        )
        
        style.map('TEntry',
            focuscolor=[('focus', self.colors['focus'])],
            bordercolor=[('focus', self.colors['focus'])]
        )
        
        # Configure Combobox styles
        style.configure('TCombobox',
            fieldbackground='white',
            background='white',
            foreground=self.colors['text_primary'],
            bordercolor=self.colors['border'],
            arrowcolor=self.colors['text_primary'],
            font=self.fonts['default']
        )
        
        style.map('TCombobox',
            focuscolor=[('focus', self.colors['focus'])],
            bordercolor=[('focus', self.colors['focus'])]
        )
        
        # Configure Notebook styles
        style.configure('TNotebook',
            background=self.colors['bg_primary'],
            borderwidth=1,
            tabmargins=[2, 5, 2, 0]
        )
        
        style.configure('TNotebook.Tab',
            background=self.colors['bg_secondary'],
            foreground=self.colors['text_secondary'],
            padding=[12, 8],
            font=self.fonts['default']
        )
        
        style.map('TNotebook.Tab',
            background=[
                ('selected', self.colors['bg_primary']),
                ('active', self.colors['hover'])
            ],
            foreground=[
                ('selected', self.colors['text_primary']),
                ('active', self.colors['text_primary'])
            ]
        )
        
        # Configure LabelFrame styles
        style.configure('TLabelframe',
            background=self.colors['bg_primary'],
            bordercolor=self.colors['border'],
            borderwidth=1,
            relief='solid'
        )
        
        style.configure('TLabelframe.Label',
            background=self.colors['bg_primary'],
            foreground=self.colors['text_primary'],
            font=self.fonts['default']
        )
        
        # Configure Treeview styles
        style.configure('Treeview',
            background='white',
            foreground=self.colors['text_primary'],
            fieldbackground='white',
            bordercolor=self.colors['border'],
            font=self.fonts['default']
        )
        
        style.configure('Treeview.Heading',
            background=self.colors['bg_secondary'],
            foreground=self.colors['text_primary'],
            font=self.fonts['default']
        )
        
        style.map('Treeview',
            background=[('selected', self.colors['selected'])],
            foreground=[('selected', 'white')]
        )
        
        # Configure Progressbar styles
        style.configure('TProgressbar',
            background=self.colors['accent_primary'],
            troughcolor=self.colors['bg_secondary'],
            bordercolor=self.colors['border'],
            lightcolor=self.colors['accent_primary'],
            darkcolor=self.colors['accent_primary']
        )
        
        # Configure Scrollbar styles
        style.configure('TScrollbar',
            background=self.colors['bg_secondary'],
            troughcolor=self.colors['bg_primary'],
            bordercolor=self.colors['border'],
            arrowcolor=self.colors['text_secondary'],
            darkcolor=self.colors['bg_tertiary'],
            lightcolor=self.colors['bg_secondary']
        )
        
        style.map('TScrollbar',
            background=[
                ('active', self.colors['hover']),
                ('pressed', self.colors['selected'])
            ]
        )
        
        # Configure Checkbutton styles
        style.configure('TCheckbutton',
            background=self.colors['bg_primary'],
            foreground=self.colors['text_primary'],
            focuscolor=self.colors['focus'],
            font=self.fonts['default']
        )
        
        style.map('TCheckbutton',
            background=[('active', self.colors['bg_primary'])],
            foreground=[('active', self.colors['text_primary'])]
        )
        
        # Configure Radiobutton styles
        style.configure('TRadiobutton',
            background=self.colors['bg_primary'],
            foreground=self.colors['text_primary'],
            focuscolor=self.colors['focus'],
            font=self.fonts['default']
        )
        
        style.map('TRadiobutton',
            background=[('active', self.colors['bg_primary'])],
            foreground=[('active', self.colors['text_primary'])]
        )
        
        # Configure Scale styles
        style.configure('TScale',
            background=self.colors['bg_primary'],
            troughcolor=self.colors['bg_secondary'],
            bordercolor=self.colors['border'],
            lightcolor=self.colors['accent_primary'],
            darkcolor=self.colors['accent_primary']
        )
        
        # Configure PanedWindow styles
        style.configure('TPanedwindow',
            background=self.colors['bg_primary']
        )
        
        style.configure('Sash',
            background=self.colors['border'],
            bordercolor=self.colors['border'],
            sashthickness=3
        )
    
    def get_color(self, color_name: str) -> str:
        """
        Get a color value by name.
        
        Args:
            color_name: Name of the color
            
        Returns:
            Color hex value
        """
        return self.colors.get(color_name, '#000000')
    
    def get_font(self, font_name: str) -> tuple:
        """
        Get a font configuration by name.
        
        Args:
            font_name: Name of the font
            
        Returns:
            Font tuple (family, size, style)
        """
        return self.fonts.get(font_name, self.fonts['default'])
    
    def apply_to_text_widget(self, text_widget: tk.Text) -> None:
        """
        Apply light theme to a Text widget.
        
        Args:
            text_widget: Text widget to theme
        """
        text_widget.configure(
            bg='white',
            fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary'],
            selectbackground=self.colors['selected'],
            selectforeground='white',
            font=self.fonts['default']
        )
    
    def apply_to_canvas(self, canvas: tk.Canvas) -> None:
        """
        Apply light theme to a Canvas widget.
        
        Args:
            canvas: Canvas widget to theme
        """
        canvas.configure(
            bg='white',
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['focus']
        )
