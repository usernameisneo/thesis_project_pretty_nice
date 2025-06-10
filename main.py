#!/usr/bin/env python3
"""
AI-Powered Thesis Assistant - Main Application Entry Point.

This is the primary entry point for the AI-Powered Thesis Assistant application.
It provides a unified interface to launch the GUI, CLI, or run specific analysis tasks.

Features:
    - GUI application launcher
    - CLI interface selector
    - Direct analysis script execution
    - Configuration management
    - Error handling and logging
    - Multi-interface support

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional, List

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Core imports
from core.config import Config
from core.exceptions import ApplicationError, ConfigurationError
from core.lazy_imports import LazyImporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('thesis_assistant.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Lazy imports for optional components
lazy_imports = LazyImporter()


class ThesisAssistantApp:
    """
    Main application controller for the AI-Powered Thesis Assistant.
    
    This class manages the application lifecycle, interface selection,
    and component initialization.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the thesis assistant application.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = Config(config_path)
        self.config.ensure_directories()
        
        logger.info("AI-Powered Thesis Assistant initialized")
        logger.info(f"Version: 2.0 - Production Grade")
        logger.info(f"Project root: {PROJECT_ROOT}")
    
    def launch_gui(self) -> None:
        """Launch the graphical user interface."""
        try:
            logger.info("Launching GUI interface...")
            
            # Import GUI components (lazy loading)
            gui_main = lazy_imports.import_module('gui.main_window')
            
            # Create and run GUI application
            app = gui_main.ThesisAssistantGUI(self.config)
            app.run()
            
        except ImportError as e:
            logger.error(f"GUI components not available: {e}")
            print("❌ GUI interface is not available.")
            print("   Please ensure all GUI dependencies are installed.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"GUI launch failed: {e}")
            print(f"❌ Failed to launch GUI: {e}")
            sys.exit(1)
    
    def launch_cli(self, args: List[str]) -> None:
        """Launch the command-line interface."""
        try:
            logger.info("Launching CLI interface...")
            
            # Import CLI components (lazy loading)
            cli_main = lazy_imports.import_module('cli.main')
            
            # Create and run CLI application
            cli_app = cli_main.ThesisAssistantCLI(self.config)
            cli_app.run(args)
            
        except ImportError as e:
            logger.error(f"CLI components not available: {e}")
            print("❌ CLI interface is not available.")
            print("   Please ensure all CLI dependencies are installed.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"CLI launch failed: {e}")
            print(f"❌ Failed to launch CLI: {e}")
            sys.exit(1)
    
    def run_analysis(self, thesis_file: str, sources_dir: str, output_dir: str = "output") -> None:
        """Run direct thesis analysis."""
        try:
            logger.info("Running direct thesis analysis...")
            
            # Import analysis script
            from scripts.complete_thesis_analysis import CompleteThesisAnalysisSystem
            import asyncio
            
            async def run_analysis():
                system = CompleteThesisAnalysisSystem()
                await system.initialize_system()
                await system.run_complete_analysis(
                    thesis_file=thesis_file,
                    sources_directory=sources_dir,
                    output_directory=output_dir
                )
            
            # Run analysis
            asyncio.run(run_analysis())
            print("✅ Analysis completed successfully!")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            print(f"❌ Analysis failed: {e}")
            sys.exit(1)
    
    def show_status(self) -> None:
        """Show application status and component availability."""
        print("🎯 AI-Powered Thesis Assistant - System Status")
        print("=" * 50)
        
        # Check GUI availability
        try:
            lazy_imports.import_module('gui.main_window')
            gui_status = "✅ Available"
        except ImportError:
            gui_status = "❌ Not Available"
        
        # Check CLI availability
        try:
            lazy_imports.import_module('cli.main')
            cli_status = "✅ Available"
        except ImportError:
            cli_status = "❌ Not Available"
        
        # Check core components
        core_components = [
            ('Document Processing', 'processing.document_parser'),
            ('Text Processing', 'processing.text_processor'),
            ('Hybrid Search', 'indexing.hybrid_search'),
            ('Claim Detection', 'analysis.master_thesis_claim_detector'),
            ('Citation Engine', 'reasoning.enhanced_citation_engine'),
            ('APA7 Compliance', 'reasoning.apa7_compliance_engine'),
            ('OpenRouter API', 'api.openrouter_client'),
            ('Semantic Scholar API', 'api.semantic_scholar_client'),
            ('Perplexity API', 'api.perplexity_client')
        ]
        
        print(f"GUI Interface:        {gui_status}")
        print(f"CLI Interface:        {cli_status}")
        print()
        print("Core Components:")
        
        for name, module in core_components:
            try:
                lazy_imports.import_module(module)
                status = "✅ Available"
            except ImportError:
                status = "❌ Not Available"
            print(f"  {name:<20} {status}")
        
        print()
        print(f"Configuration:        ✅ Loaded from {self.config.config_file}")
        print(f"Data Directory:       {self.config.get('data_dir')}")
        print(f"Index Directory:      {self.config.get('index_dir')}")
        print(f"Projects Directory:   {self.config.get('projects_dir')}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Thesis Assistant - Production-Grade Academic Research Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Launch GUI (default)
    python main.py
    python main.py --gui
    
    # Launch CLI
    python main.py --cli
    
    # Run direct analysis
    python main.py --analyze --thesis thesis.pdf --sources ./sources
    
    # Show system status
    python main.py --status
    
    # Use custom configuration
    python main.py --config config.json --gui
        """
    )
    
    # Interface selection
    interface_group = parser.add_mutually_exclusive_group()
    interface_group.add_argument(
        '--gui',
        action='store_true',
        help='Launch graphical user interface (default)'
    )
    interface_group.add_argument(
        '--cli',
        action='store_true',
        help='Launch command-line interface'
    )
    interface_group.add_argument(
        '--analyze',
        action='store_true',
        help='Run direct thesis analysis'
    )
    interface_group.add_argument(
        '--status',
        action='store_true',
        help='Show system status and component availability'
    )
    
    # Analysis options
    parser.add_argument(
        '--thesis',
        type=str,
        help='Path to thesis file for analysis'
    )
    parser.add_argument(
        '--sources',
        type=str,
        help='Path to sources directory for analysis'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for analysis results (default: output)'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main():
    """Main entry point for the AI-Powered Thesis Assistant."""
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize application
        app = ThesisAssistantApp(args.config)
        
        # Handle different modes
        if args.status:
            app.show_status()
        elif args.analyze:
            if not args.thesis or not args.sources:
                print("❌ Analysis mode requires --thesis and --sources arguments")
                parser.print_help()
                sys.exit(1)
            app.run_analysis(args.thesis, args.sources, args.output)
        elif args.cli:
            app.launch_cli(sys.argv[1:])
        else:
            # Default to GUI
            app.launch_gui()
            
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application failed: {e}")
        print(f"❌ Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
