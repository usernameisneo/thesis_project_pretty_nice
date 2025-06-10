"""
Statistics Widget for the AI-Powered Thesis Assistant.

This widget provides comprehensive analytics and statistics display
for document processing, AI operations, and system performance.

Features:
    - Document processing statistics
    - AI operation metrics
    - System performance data
    - Visual charts and graphs
    - Export capabilities
    - Real-time updates

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import tkinter as tk
from tkinter import ttk
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Local imports
from core.config import Config

logger = logging.getLogger(__name__)


class StatisticsWidget(ttk.Frame):
    """
    Statistics display widget with comprehensive analytics.
    
    This widget provides detailed statistics and analytics for all
    application operations and system performance.
    """
    
    def __init__(self, parent: tk.Widget, config: Config):
        """
        Initialize the statistics widget.
        
        Args:
            parent: Parent widget
            config: Application configuration
        """
        super().__init__(parent)
        self.config = config
        self.stats_data = {
            'documents': {'processed': 0, 'failed': 0, 'total_size': 0},
            'ai_operations': {'total': 0, 'successful': 0, 'failed': 0},
            'citations': {'generated': 0, 'validated': 0, 'apa7_compliant': 0},
            'performance': {'avg_processing_time': 0, 'total_processing_time': 0}
        }
        
        self._setup_ui()
        self._load_statistics()
        
        logger.info("Statistics widget initialized")
    
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        # Title
        title_label = ttk.Label(
            self,
            text="📊 System Statistics",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=(0, 15))
        
        # Create statistics sections
        self._create_overview_section()
        self._create_documents_section()
        self._create_ai_operations_section()
        self._create_citations_section()
        self._create_performance_section()
        
        # Refresh button
        ttk.Button(
            self,
            text="🔄 Refresh Statistics",
            command=self._refresh_statistics
        ).pack(pady=(15, 0))
    
    def _create_overview_section(self) -> None:
        """Create the overview statistics section."""
        overview_frame = ttk.LabelFrame(self, text="Overview", padding=10)
        overview_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create grid for overview stats
        stats_grid = ttk.Frame(overview_frame)
        stats_grid.pack(fill=tk.X)
        
        # Total documents
        ttk.Label(stats_grid, text="📄 Total Documents:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10)
        )
        self.total_docs_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.total_docs_var).grid(row=0, column=1, sticky=tk.W)
        
        # Total AI operations
        ttk.Label(stats_grid, text="🤖 AI Operations:", font=('Arial', 10, 'bold')).grid(
            row=0, column=2, sticky=tk.W, padx=(20, 10)
        )
        self.total_ai_ops_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.total_ai_ops_var).grid(row=0, column=3, sticky=tk.W)
        
        # Total citations
        ttk.Label(stats_grid, text="📚 Citations Generated:", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0)
        )
        self.total_citations_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.total_citations_var).grid(
            row=1, column=1, sticky=tk.W, pady=(10, 0)
        )
        
        # System uptime
        ttk.Label(stats_grid, text="⏱️ Session Time:", font=('Arial', 10, 'bold')).grid(
            row=1, column=2, sticky=tk.W, padx=(20, 10), pady=(10, 0)
        )
        self.uptime_var = tk.StringVar(value="0m")
        ttk.Label(stats_grid, textvariable=self.uptime_var).grid(
            row=1, column=3, sticky=tk.W, pady=(10, 0)
        )
        
        # Start uptime counter
        self.start_time = datetime.now()
        self._update_uptime()
    
    def _create_documents_section(self) -> None:
        """Create the document statistics section."""
        docs_frame = ttk.LabelFrame(self, text="Document Processing", padding=10)
        docs_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Document stats grid
        docs_grid = ttk.Frame(docs_frame)
        docs_grid.pack(fill=tk.X)
        
        # Processed successfully
        ttk.Label(docs_grid, text="✅ Processed:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.docs_processed_var = tk.StringVar(value="0")
        ttk.Label(docs_grid, textvariable=self.docs_processed_var).grid(row=0, column=1, sticky=tk.W)
        
        # Processing failed
        ttk.Label(docs_grid, text="❌ Failed:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.docs_failed_var = tk.StringVar(value="0")
        ttk.Label(docs_grid, textvariable=self.docs_failed_var).grid(row=0, column=3, sticky=tk.W)
        
        # Total size processed
        ttk.Label(docs_grid, text="📊 Total Size:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.total_size_var = tk.StringVar(value="0 MB")
        ttk.Label(docs_grid, textvariable=self.total_size_var).grid(
            row=1, column=1, sticky=tk.W, pady=(10, 0)
        )
        
        # Success rate
        ttk.Label(docs_grid, text="📈 Success Rate:").grid(
            row=1, column=2, sticky=tk.W, padx=(20, 10), pady=(10, 0)
        )
        self.docs_success_rate_var = tk.StringVar(value="0%")
        ttk.Label(docs_grid, textvariable=self.docs_success_rate_var).grid(
            row=1, column=3, sticky=tk.W, pady=(10, 0)
        )
        
        # Progress bar for success rate
        self.docs_progress_var = tk.DoubleVar()
        docs_progress = ttk.Progressbar(
            docs_frame,
            variable=self.docs_progress_var,
            mode='determinate',
            length=200
        )
        docs_progress.pack(pady=(10, 0))
    
    def _create_ai_operations_section(self) -> None:
        """Create the AI operations statistics section."""
        ai_frame = ttk.LabelFrame(self, text="AI Operations", padding=10)
        ai_frame.pack(fill=tk.X, pady=(0, 10))
        
        # AI operations grid
        ai_grid = ttk.Frame(ai_frame)
        ai_grid.pack(fill=tk.X)
        
        # Total operations
        ttk.Label(ai_grid, text="🔄 Total Operations:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.ai_total_var = tk.StringVar(value="0")
        ttk.Label(ai_grid, textvariable=self.ai_total_var).grid(row=0, column=1, sticky=tk.W)
        
        # Successful operations
        ttk.Label(ai_grid, text="✅ Successful:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.ai_success_var = tk.StringVar(value="0")
        ttk.Label(ai_grid, textvariable=self.ai_success_var).grid(row=0, column=3, sticky=tk.W)
        
        # Failed operations
        ttk.Label(ai_grid, text="❌ Failed:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.ai_failed_var = tk.StringVar(value="0")
        ttk.Label(ai_grid, textvariable=self.ai_failed_var).grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # Average response time
        ttk.Label(ai_grid, text="⏱️ Avg Response:").grid(
            row=1, column=2, sticky=tk.W, padx=(20, 10), pady=(10, 0)
        )
        self.ai_avg_time_var = tk.StringVar(value="0.0s")
        ttk.Label(ai_grid, textvariable=self.ai_avg_time_var).grid(
            row=1, column=3, sticky=tk.W, pady=(10, 0)
        )
        
        # API usage breakdown
        api_frame = ttk.Frame(ai_frame)
        api_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(api_frame, text="API Usage:", font=('Arial', 9, 'bold')).pack(anchor=tk.W)
        
        api_grid = ttk.Frame(api_frame)
        api_grid.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(api_grid, text="OpenRouter:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.openrouter_calls_var = tk.StringVar(value="0")
        ttk.Label(api_grid, textvariable=self.openrouter_calls_var).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(api_grid, text="Perplexity:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.perplexity_calls_var = tk.StringVar(value="0")
        ttk.Label(api_grid, textvariable=self.perplexity_calls_var).grid(row=0, column=3, sticky=tk.W)
        
        ttk.Label(api_grid, text="Semantic Scholar:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.semantic_calls_var = tk.StringVar(value="0")
        ttk.Label(api_grid, textvariable=self.semantic_calls_var).grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
    
    def _create_citations_section(self) -> None:
        """Create the citations statistics section."""
        citations_frame = ttk.LabelFrame(self, text="Citations & Bibliography", padding=10)
        citations_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Citations grid
        cit_grid = ttk.Frame(citations_frame)
        cit_grid.pack(fill=tk.X)
        
        # Generated citations
        ttk.Label(cit_grid, text="📚 Generated:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.cit_generated_var = tk.StringVar(value="0")
        ttk.Label(cit_grid, textvariable=self.cit_generated_var).grid(row=0, column=1, sticky=tk.W)
        
        # Validated citations
        ttk.Label(cit_grid, text="✅ Validated:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.cit_validated_var = tk.StringVar(value="0")
        ttk.Label(cit_grid, textvariable=self.cit_validated_var).grid(row=0, column=3, sticky=tk.W)
        
        # APA7 compliant
        ttk.Label(cit_grid, text="📖 APA7 Compliant:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.cit_apa7_var = tk.StringVar(value="0")
        ttk.Label(cit_grid, textvariable=self.cit_apa7_var).grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # Compliance rate
        ttk.Label(cit_grid, text="📊 Compliance Rate:").grid(
            row=1, column=2, sticky=tk.W, padx=(20, 10), pady=(10, 0)
        )
        self.cit_compliance_var = tk.StringVar(value="0%")
        ttk.Label(cit_grid, textvariable=self.cit_compliance_var).grid(
            row=1, column=3, sticky=tk.W, pady=(10, 0)
        )
    
    def _create_performance_section(self) -> None:
        """Create the performance statistics section."""
        perf_frame = ttk.LabelFrame(self, text="Performance Metrics", padding=10)
        perf_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Performance grid
        perf_grid = ttk.Frame(perf_frame)
        perf_grid.pack(fill=tk.X)
        
        # Average processing time
        ttk.Label(perf_grid, text="⏱️ Avg Processing:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.perf_avg_var = tk.StringVar(value="0.0s")
        ttk.Label(perf_grid, textvariable=self.perf_avg_var).grid(row=0, column=1, sticky=tk.W)
        
        # Total processing time
        ttk.Label(perf_grid, text="🕐 Total Time:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.perf_total_var = tk.StringVar(value="0.0s")
        ttk.Label(perf_grid, textvariable=self.perf_total_var).grid(row=0, column=3, sticky=tk.W)
        
        # Memory usage (real-time monitoring)
        ttk.Label(perf_grid, text="💾 Memory Usage:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.memory_usage_var = tk.StringVar(value="0 MB")
        ttk.Label(perf_grid, textvariable=self.memory_usage_var).grid(row=1, column=1, sticky=tk.W, pady=(10, 0))

        # Start memory monitoring
        self._start_memory_monitoring()
        
        # Cache hit rate
        ttk.Label(perf_grid, text="🎯 Cache Hit Rate:").grid(
            row=1, column=2, sticky=tk.W, padx=(20, 10), pady=(10, 0)
        )
        self.cache_hit_var = tk.StringVar(value="N/A")
        ttk.Label(perf_grid, textvariable=self.cache_hit_var).grid(row=1, column=3, sticky=tk.W, pady=(10, 0))
    
    def _load_statistics(self) -> None:
        """Load statistics from storage."""
        # This would load from a persistent storage
        # For now, we'll start with empty statistics
        self._update_display()
    
    def _update_display(self) -> None:
        """Update the statistics display."""
        # Overview
        total_docs = self.stats_data['documents']['processed'] + self.stats_data['documents']['failed']
        self.total_docs_var.set(str(total_docs))
        self.total_ai_ops_var.set(str(self.stats_data['ai_operations']['total']))
        self.total_citations_var.set(str(self.stats_data['citations']['generated']))
        
        # Documents
        self.docs_processed_var.set(str(self.stats_data['documents']['processed']))
        self.docs_failed_var.set(str(self.stats_data['documents']['failed']))
        
        # Calculate success rate
        if total_docs > 0:
            success_rate = (self.stats_data['documents']['processed'] / total_docs) * 100
            self.docs_success_rate_var.set(f"{success_rate:.1f}%")
            self.docs_progress_var.set(success_rate)
        else:
            self.docs_success_rate_var.set("0%")
            self.docs_progress_var.set(0)
        
        # Format file size
        size_mb = self.stats_data['documents']['total_size'] / (1024 * 1024)
        self.total_size_var.set(f"{size_mb:.1f} MB")
        
        # AI Operations
        self.ai_total_var.set(str(self.stats_data['ai_operations']['total']))
        self.ai_success_var.set(str(self.stats_data['ai_operations']['successful']))
        self.ai_failed_var.set(str(self.stats_data['ai_operations']['failed']))
        
        # Citations
        self.cit_generated_var.set(str(self.stats_data['citations']['generated']))
        self.cit_validated_var.set(str(self.stats_data['citations']['validated']))
        self.cit_apa7_var.set(str(self.stats_data['citations']['apa7_compliant']))
        
        # Calculate compliance rate
        if self.stats_data['citations']['generated'] > 0:
            compliance_rate = (self.stats_data['citations']['apa7_compliant'] / 
                             self.stats_data['citations']['generated']) * 100
            self.cit_compliance_var.set(f"{compliance_rate:.1f}%")
        else:
            self.cit_compliance_var.set("0%")
        
        # Performance
        self.perf_avg_var.set(f"{self.stats_data['performance']['avg_processing_time']:.1f}s")
        self.perf_total_var.set(f"{self.stats_data['performance']['total_processing_time']:.1f}s")
    
    def _update_uptime(self) -> None:
        """Update the uptime display."""
        if hasattr(self, 'start_time'):
            uptime = datetime.now() - self.start_time
            total_seconds = int(uptime.total_seconds())
            
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            
            if hours > 0:
                uptime_str = f"{hours}h {minutes}m"
            else:
                uptime_str = f"{minutes}m"
            
            self.uptime_var.set(uptime_str)
            
            # Schedule next update
            self.after(60000, self._update_uptime)  # Update every minute
    
    def _refresh_statistics(self) -> None:
        """Refresh all statistics."""
        # This would reload from storage and update counters
        self._update_display()
        logger.info("Statistics refreshed")
    
    def update_document_stats(self, processed: int = 0, failed: int = 0, size_bytes: int = 0) -> None:
        """Update document processing statistics."""
        self.stats_data['documents']['processed'] += processed
        self.stats_data['documents']['failed'] += failed
        self.stats_data['documents']['total_size'] += size_bytes
        self._update_display()
    
    def update_ai_stats(self, total: int = 0, successful: int = 0, failed: int = 0) -> None:
        """Update AI operation statistics."""
        self.stats_data['ai_operations']['total'] += total
        self.stats_data['ai_operations']['successful'] += successful
        self.stats_data['ai_operations']['failed'] += failed
        self._update_display()
    
    def update_citation_stats(self, generated: int = 0, validated: int = 0, apa7_compliant: int = 0) -> None:
        """Update citation statistics."""
        self.stats_data['citations']['generated'] += generated
        self.stats_data['citations']['validated'] += validated
        self.stats_data['citations']['apa7_compliant'] += apa7_compliant
        self._update_display()
    
    def _start_memory_monitoring(self) -> None:
        """Start real-time memory usage monitoring."""
        self._update_memory_usage()

    def _update_memory_usage(self) -> None:
        """Update memory usage display."""
        try:
            import psutil

            # Get current process memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB

            # Update display
            self.memory_usage_var.set(f"{memory_mb:.1f} MB")

            # Schedule next update (every 5 seconds)
            self.after(5000, self._update_memory_usage)

        except ImportError:
            # psutil not available, show static message
            self.memory_usage_var.set("Monitor N/A")
        except Exception as e:
            logger.error(f"Memory monitoring error: {e}")
            self.memory_usage_var.set("Error")

    def apply_theme(self, theme) -> None:
        """Apply theme to the widget."""
        # Theme application will be implemented with theme system
        pass
