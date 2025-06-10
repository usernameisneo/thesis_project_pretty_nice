"""
Progress Tracker Widget for the AI-Powered Thesis Assistant.

This widget provides real-time progress tracking for all application operations
with detailed status information and visual progress indicators.

Features:
    - Real-time progress tracking
    - Multi-stage operation monitoring
    - Visual progress indicators
    - Detailed status information
    - Operation history
    - Performance metrics

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import tkinter as tk
from tkinter import ttk
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Local imports
from core.config import Config

logger = logging.getLogger(__name__)


class ProgressTrackerWidget(ttk.Frame):
    """
    Progress tracking widget with comprehensive operation monitoring.
    
    This widget provides real-time feedback on all application operations
    with detailed progress information and visual indicators.
    """
    
    def __init__(self, parent: tk.Widget, config: Config):
        """
        Initialize the progress tracker widget.
        
        Args:
            parent: Parent widget
            config: Application configuration
        """
        super().__init__(parent)
        self.config = config
        self.operations = []
        self.current_operation = None
        
        self._setup_ui()
        
        logger.info("Progress tracker widget initialized")
    
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        # Title
        title_label = ttk.Label(
            self,
            text="📊 Progress Tracker",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=(0, 15))
        
        # Current operation section
        self._create_current_operation_section()
        
        # Progress history section
        self._create_history_section()
        
        # Statistics section
        self._create_statistics_section()
    
    def _create_current_operation_section(self) -> None:
        """Create the current operation tracking section."""
        current_frame = ttk.LabelFrame(self, text="Current Operation", padding=10)
        current_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Operation name
        self.operation_var = tk.StringVar(value="No operation in progress")
        operation_label = ttk.Label(
            current_frame,
            textvariable=self.operation_var,
            font=('Arial', 11, 'bold')
        )
        operation_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            current_frame,
            variable=self.progress_var,
            mode='determinate',
            length=300
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Status and details
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(current_frame, textvariable=self.status_var)
        status_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.details_var = tk.StringVar(value="")
        details_label = ttk.Label(
            current_frame,
            textvariable=self.details_var,
            foreground="gray"
        )
        details_label.pack(anchor=tk.W)
        
        # Time information
        time_frame = ttk.Frame(current_frame)
        time_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_time_var = tk.StringVar(value="")
        ttk.Label(time_frame, textvariable=self.start_time_var).pack(side=tk.LEFT)
        
        self.elapsed_time_var = tk.StringVar(value="")
        ttk.Label(time_frame, textvariable=self.elapsed_time_var).pack(side=tk.RIGHT)
    
    def _create_history_section(self) -> None:
        """Create the operation history section."""
        history_frame = ttk.LabelFrame(self, text="Operation History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # History list
        columns = ('Time', 'Operation', 'Status', 'Duration')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=8)
        
        # Configure columns
        self.history_tree.heading('Time', text='Time')
        self.history_tree.heading('Operation', text='Operation')
        self.history_tree.heading('Status', text='Status')
        self.history_tree.heading('Duration', text='Duration')
        
        self.history_tree.column('Time', width=80)
        self.history_tree.column('Operation', width=200)
        self.history_tree.column('Status', width=80)
        self.history_tree.column('Duration', width=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Clear history button
        ttk.Button(
            history_frame,
            text="🗑️ Clear History",
            command=self._clear_history
        ).pack(pady=(10, 0))
    
    def _create_statistics_section(self) -> None:
        """Create the statistics section."""
        stats_frame = ttk.LabelFrame(self, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X)
        
        # Statistics grid
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # Total operations
        ttk.Label(stats_grid, text="Total Operations:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.total_ops_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.total_ops_var).grid(row=0, column=1, sticky=tk.W)
        
        # Successful operations
        ttk.Label(stats_grid, text="Successful:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.success_ops_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.success_ops_var).grid(row=1, column=1, sticky=tk.W)
        
        # Failed operations
        ttk.Label(stats_grid, text="Failed:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        self.failed_ops_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.failed_ops_var).grid(row=2, column=1, sticky=tk.W)
        
        # Average duration
        ttk.Label(stats_grid, text="Avg Duration:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.avg_duration_var = tk.StringVar(value="0s")
        ttk.Label(stats_grid, textvariable=self.avg_duration_var).grid(row=0, column=3, sticky=tk.W)
        
        # Total time
        ttk.Label(stats_grid, text="Total Time:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10))
        self.total_time_var = tk.StringVar(value="0s")
        ttk.Label(stats_grid, textvariable=self.total_time_var).grid(row=1, column=3, sticky=tk.W)
    
    def start_operation(self, name: str, total_steps: int = 100) -> str:
        """
        Start tracking a new operation.
        
        Args:
            name: Operation name
            total_steps: Total number of steps (for progress calculation)
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"op_{datetime.now().timestamp()}"
        
        operation = {
            'id': operation_id,
            'name': name,
            'start_time': datetime.now(),
            'total_steps': total_steps,
            'current_step': 0,
            'status': 'Running',
            'details': ''
        }
        
        self.current_operation = operation
        self.operations.append(operation)
        
        # Update UI
        self.operation_var.set(name)
        self.status_var.set("Starting...")
        self.details_var.set("")
        self.progress_var.set(0)
        self.start_time_var.set(f"Started: {operation['start_time'].strftime('%H:%M:%S')}")
        
        # Start elapsed time updates
        self._update_elapsed_time()
        
        logger.info(f"Started operation: {name}")
        return operation_id
    
    def update_progress(self, operation_id: str, step: int, status: str = "", details: str = "") -> None:
        """
        Update operation progress.
        
        Args:
            operation_id: Operation ID
            step: Current step number
            status: Status message
            details: Detailed information
        """
        if not self.current_operation or self.current_operation['id'] != operation_id:
            return
        
        self.current_operation['current_step'] = step
        if status:
            self.current_operation['status'] = status
        if details:
            self.current_operation['details'] = details
        
        # Calculate progress percentage
        progress = (step / self.current_operation['total_steps']) * 100
        progress = min(100, max(0, progress))
        
        # Update UI
        self.progress_var.set(progress)
        if status:
            self.status_var.set(status)
        if details:
            self.details_var.set(details)
    
    def complete_operation(self, operation_id: str, success: bool = True, message: str = "") -> None:
        """
        Complete an operation.
        
        Args:
            operation_id: Operation ID
            success: Whether operation was successful
            message: Completion message
        """
        if not self.current_operation or self.current_operation['id'] != operation_id:
            return
        
        # Update operation
        self.current_operation['end_time'] = datetime.now()
        self.current_operation['duration'] = (
            self.current_operation['end_time'] - self.current_operation['start_time']
        ).total_seconds()
        self.current_operation['success'] = success
        self.current_operation['message'] = message
        
        # Update UI
        self.progress_var.set(100)
        status = "Completed" if success else "Failed"
        self.status_var.set(f"{status}: {message}" if message else status)
        
        # Add to history
        self._add_to_history(self.current_operation)
        
        # Clear current operation after delay
        self.after(3000, self._clear_current_operation)
        
        logger.info(f"Completed operation: {self.current_operation['name']} ({'success' if success else 'failed'})")
    
    def _add_to_history(self, operation: Dict[str, Any]) -> None:
        """Add operation to history."""
        time_str = operation['start_time'].strftime('%H:%M:%S')
        status = "✅ Success" if operation.get('success', False) else "❌ Failed"
        duration = f"{operation.get('duration', 0):.1f}s"
        
        self.history_tree.insert('', 0, values=(
            time_str,
            operation['name'],
            status,
            duration
        ))
        
        # Update statistics
        self._update_statistics()
    
    def _update_statistics(self) -> None:
        """Update operation statistics."""
        total_ops = len(self.operations)
        successful_ops = sum(1 for op in self.operations if op.get('success', False))
        failed_ops = total_ops - successful_ops
        
        # Calculate average duration
        durations = [op.get('duration', 0) for op in self.operations if 'duration' in op]
        avg_duration = sum(durations) / len(durations) if durations else 0
        total_time = sum(durations)
        
        # Update UI
        self.total_ops_var.set(str(total_ops))
        self.success_ops_var.set(str(successful_ops))
        self.failed_ops_var.set(str(failed_ops))
        self.avg_duration_var.set(f"{avg_duration:.1f}s")
        self.total_time_var.set(f"{total_time:.1f}s")
    
    def _update_elapsed_time(self) -> None:
        """Update elapsed time display."""
        if self.current_operation and 'end_time' not in self.current_operation:
            elapsed = (datetime.now() - self.current_operation['start_time']).total_seconds()
            self.elapsed_time_var.set(f"Elapsed: {elapsed:.1f}s")
            
            # Schedule next update
            self.after(1000, self._update_elapsed_time)
    
    def _clear_current_operation(self) -> None:
        """Clear the current operation display."""
        self.current_operation = None
        self.operation_var.set("No operation in progress")
        self.status_var.set("Ready")
        self.details_var.set("")
        self.progress_var.set(0)
        self.start_time_var.set("")
        self.elapsed_time_var.set("")
    
    def _clear_history(self) -> None:
        """Clear operation history."""
        if tk.messagebox.askyesno("Clear History", "Clear all operation history?"):
            self.history_tree.delete(*self.history_tree.get_children())
            self.operations.clear()
            self._update_statistics()
    
    def get_current_operation_id(self) -> Optional[str]:
        """Get the current operation ID."""
        return self.current_operation['id'] if self.current_operation else None
    
    def apply_theme(self, theme) -> None:
        """Apply theme to the widget."""
        # Theme application will be implemented with theme system
        pass
