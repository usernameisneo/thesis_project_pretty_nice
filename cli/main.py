"""
Main CLI Application for the AI-Powered Thesis Assistant.

This module implements the primary command-line interface with comprehensive
functionality for document processing, AI operations, and thesis management.

Features:
    - Interactive and batch modes
    - Full document processing pipeline
    - AI chat and analysis
    - Citation management
    - Project management
    - Configuration handling

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Local imports
from core.config import Config
from core.exceptions import ApplicationError
from cli.commands import CommandHandler
from cli.interactive import InteractiveMode

logger = logging.getLogger(__name__)


class ThesisAssistantCLI:
    """
    Main CLI application for the AI-Powered Thesis Assistant.
    
    This class provides a comprehensive command-line interface with
    support for both interactive and batch processing modes.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the CLI application.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.command_handler = CommandHandler(config)
        self.interactive_mode = InteractiveMode(config, self.command_handler)
        
        logger.info("CLI application initialized")
    
    def run(self, args: List[str]) -> None:
        """
        Run the CLI application with the given arguments.
        
        Args:
            args: Command line arguments
        """
        try:
            # Parse arguments
            parser = self._create_parser()
            parsed_args = parser.parse_args(args[1:] if args and args[0].endswith('.py') else args)
            
            # Set logging level
            if parsed_args.verbose:
                logging.getLogger().setLevel(logging.DEBUG)
            elif parsed_args.quiet:
                logging.getLogger().setLevel(logging.WARNING)
            
            # Handle different commands
            if parsed_args.command == 'interactive':
                self._run_interactive_mode()
            elif parsed_args.command == 'process':
                asyncio.run(self._process_documents(parsed_args))
            elif parsed_args.command == 'analyze':
                asyncio.run(self._analyze_thesis(parsed_args))
            elif parsed_args.command == 'chat':
                asyncio.run(self._ai_chat(parsed_args))
            elif parsed_args.command == 'citations':
                asyncio.run(self._manage_citations(parsed_args))
            elif parsed_args.command == 'project':
                self._manage_project(parsed_args)
            elif parsed_args.command == 'config':
                self._manage_config(parsed_args)
            elif parsed_args.command == 'status':
                self._show_status()
            else:
                # Default to interactive mode
                self._run_interactive_mode()
                
        except KeyboardInterrupt:
            print("\n👋 Operation cancelled by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"CLI error: {e}")
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="AI-Powered Thesis Assistant - Command Line Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
    # Interactive mode
    python main.py --cli
    python main.py --cli interactive
    
    # Process documents
    python main.py --cli process --input docs/ --output processed/
    
    # Analyze thesis
    python main.py --cli analyze --thesis thesis.pdf --sources sources/
    
    # AI chat
    python main.py --cli chat --model gpt-4 --message "Explain APA citation format"
    
    # Manage citations
    python main.py --cli citations --generate --input thesis.pdf
    
    # Project management
    python main.py --cli project --create "My Thesis Project"
    
    # Configuration
    python main.py --cli config --set openrouter_api_key YOUR_KEY
            """
        )
        
        # Global options
        parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
        parser.add_argument('--quiet', '-q', action='store_true', help='Suppress non-error output')
        parser.add_argument('--config-file', type=str, help='Path to configuration file')
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Interactive mode
        interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
        
        # Document processing
        process_parser = subparsers.add_parser('process', help='Process documents')
        process_parser.add_argument('--input', '-i', required=True, help='Input file or directory')
        process_parser.add_argument('--output', '-o', default='output', help='Output directory')
        process_parser.add_argument('--format', choices=['pdf', 'txt', 'md', 'all'], default='all', help='File formats to process')
        process_parser.add_argument('--ocr', action='store_true', help='Enable OCR for scanned documents')
        process_parser.add_argument('--index', action='store_true', help='Index processed documents')
        
        # Thesis analysis
        analyze_parser = subparsers.add_parser('analyze', help='Analyze thesis')
        analyze_parser.add_argument('--thesis', required=True, help='Path to thesis file')
        analyze_parser.add_argument('--sources', help='Path to sources directory')
        analyze_parser.add_argument('--output', '-o', default='analysis', help='Output directory')
        analyze_parser.add_argument('--claims', action='store_true', help='Detect claims requiring citations')
        analyze_parser.add_argument('--citations', action='store_true', help='Validate existing citations')
        analyze_parser.add_argument('--apa7', action='store_true', help='Check APA7 compliance')
        
        # AI chat
        chat_parser = subparsers.add_parser('chat', help='AI chat interface')
        chat_parser.add_argument('--model', help='AI model to use')
        chat_parser.add_argument('--message', '-m', help='Single message to send')
        chat_parser.add_argument('--context', help='Context file to include')
        chat_parser.add_argument('--save', help='Save conversation to file')
        
        # Citations management
        citations_parser = subparsers.add_parser('citations', help='Manage citations')
        citations_parser.add_argument('--generate', action='store_true', help='Generate citations')
        citations_parser.add_argument('--validate', action='store_true', help='Validate citations')
        citations_parser.add_argument('--format', choices=['apa7', 'mla', 'chicago'], default='apa7', help='Citation format')
        citations_parser.add_argument('--input', help='Input file')
        citations_parser.add_argument('--output', help='Output file')
        
        # Project management
        project_parser = subparsers.add_parser('project', help='Manage thesis projects')
        project_parser.add_argument('--create', help='Create new project')
        project_parser.add_argument('--open', help='Open existing project')
        project_parser.add_argument('--list', action='store_true', help='List all projects')
        project_parser.add_argument('--delete', help='Delete project')
        
        # Configuration
        config_parser = subparsers.add_parser('config', help='Manage configuration')
        config_parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), help='Set configuration value')
        config_parser.add_argument('--get', help='Get configuration value')
        config_parser.add_argument('--list', action='store_true', help='List all configuration')
        config_parser.add_argument('--reset', action='store_true', help='Reset to defaults')
        
        # Status
        status_parser = subparsers.add_parser('status', help='Show system status')
        
        return parser
    
    def _run_interactive_mode(self) -> None:
        """Run the interactive CLI mode."""
        print("🎯 AI-Powered Thesis Assistant - Interactive Mode")
        print("=" * 50)
        print("Type 'help' for available commands or 'exit' to quit.")
        print()
        
        self.interactive_mode.run()
    
    async def _process_documents(self, args) -> None:
        """Process documents based on CLI arguments."""
        print(f"📄 Processing documents from: {args.input}")
        
        try:
            result = await self.command_handler.process_documents(
                input_path=args.input,
                output_path=args.output,
                file_format=args.format,
                enable_ocr=args.ocr,
                enable_indexing=args.index
            )
            
            print(f"✅ Processing completed successfully!")
            print(f"   Processed: {result.get('processed', 0)} files")
            print(f"   Failed: {result.get('failed', 0)} files")
            print(f"   Output: {args.output}")
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            sys.exit(1)
    
    async def _analyze_thesis(self, args) -> None:
        """Analyze thesis based on CLI arguments."""
        print(f"📊 Analyzing thesis: {args.thesis}")
        
        try:
            result = await self.command_handler.analyze_thesis(
                thesis_path=args.thesis,
                sources_path=args.sources,
                output_path=args.output,
                detect_claims=args.claims,
                validate_citations=args.citations,
                check_apa7=args.apa7
            )
            
            print(f"✅ Analysis completed successfully!")
            print(f"   Claims detected: {result.get('claims_count', 0)}")
            print(f"   Citations validated: {result.get('citations_count', 0)}")
            print(f"   APA7 compliance: {result.get('apa7_compliance', 'N/A')}")
            print(f"   Report: {result.get('report_path', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            sys.exit(1)
    
    async def _ai_chat(self, args) -> None:
        """Handle AI chat based on CLI arguments."""
        if args.message:
            # Single message mode
            print(f"🤖 Sending message to AI...")
            
            try:
                response = await self.command_handler.ai_chat(
                    message=args.message,
                    model=args.model,
                    context_file=args.context
                )
                
                print(f"\n💬 AI Response:")
                print(f"{response}")

                if args.save:
                    # Save conversation to file
                    conversation_data = {
                        'timestamp': datetime.now().isoformat(),
                        'model': args.model or 'default',
                        'conversation': [
                            {'role': 'user', 'content': args.message},
                            {'role': 'assistant', 'content': response}
                        ]
                    }

                    save_path = Path(args.save)
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    import json
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(conversation_data, f, indent=2, ensure_ascii=False)

                    print(f"💾 Conversation saved to: {save_path}")
                    
            except Exception as e:
                print(f"❌ AI chat failed: {e}")
                sys.exit(1)
        else:
            # Interactive chat mode
            print("🤖 AI Chat Mode - Type 'exit' to quit")
            print("-" * 40)
            
            await self.command_handler.interactive_chat(model=args.model)
    
    async def _manage_citations(self, args) -> None:
        """Manage citations based on CLI arguments."""
        if args.generate:
            print(f"📚 Generating citations from: {args.input}")
            
            try:
                result = await self.command_handler.generate_citations(
                    input_path=args.input,
                    output_path=args.output,
                    citation_format=args.format
                )
                
                print(f"✅ Citations generated successfully!")
                print(f"   Generated: {result.get('count', 0)} citations")
                print(f"   Format: {args.format.upper()}")
                print(f"   Output: {result.get('output_path', 'N/A')}")
                
            except Exception as e:
                print(f"❌ Citation generation failed: {e}")
                sys.exit(1)
        
        elif args.validate:
            print(f"🔍 Validating citations in: {args.input}")
            
            try:
                result = await self.command_handler.validate_citations(
                    input_path=args.input,
                    citation_format=args.format
                )
                
                print(f"✅ Citation validation completed!")
                print(f"   Valid: {result.get('valid', 0)}")
                print(f"   Invalid: {result.get('invalid', 0)}")
                print(f"   Compliance: {result.get('compliance_rate', 0):.1f}%")
                
            except Exception as e:
                print(f"❌ Citation validation failed: {e}")
                sys.exit(1)
    
    def _manage_project(self, args) -> None:
        """Manage thesis projects based on CLI arguments."""
        if args.create:
            print(f"📁 Creating new project: {args.create}")
            
            try:
                project_path = self.command_handler.create_project(args.create)
                print(f"✅ Project created successfully!")
                print(f"   Path: {project_path}")
                
            except Exception as e:
                print(f"❌ Project creation failed: {e}")
                sys.exit(1)
        
        elif args.open:
            print(f"📂 Opening project: {args.open}")
            
            try:
                self.command_handler.open_project(args.open)
                print(f"✅ Project opened successfully!")
                
            except Exception as e:
                print(f"❌ Failed to open project: {e}")
                sys.exit(1)
        
        elif args.list:
            print("📋 Available projects:")
            
            try:
                projects = self.command_handler.list_projects()
                if projects:
                    for i, project in enumerate(projects, 1):
                        print(f"   {i}. {project['name']} ({project['path']})")
                else:
                    print("   No projects found.")
                    
            except Exception as e:
                print(f"❌ Failed to list projects: {e}")
                sys.exit(1)
        
        elif args.delete:
            print(f"🗑️ Deleting project: {args.delete}")
            
            try:
                self.command_handler.delete_project(args.delete)
                print(f"✅ Project deleted successfully!")
                
            except Exception as e:
                print(f"❌ Failed to delete project: {e}")
                sys.exit(1)
    
    def _manage_config(self, args) -> None:
        """Manage configuration based on CLI arguments."""
        if args.set:
            key, value = args.set
            print(f"⚙️ Setting {key} = {value}")
            
            try:
                self.config.set(key, value)
                self.config.save()
                print(f"✅ Configuration updated successfully!")
                
            except Exception as e:
                print(f"❌ Failed to update configuration: {e}")
                sys.exit(1)
        
        elif args.get:
            print(f"⚙️ Getting configuration value: {args.get}")
            
            try:
                value = self.config.get(args.get)
                print(f"   {args.get} = {value}")
                
            except Exception as e:
                print(f"❌ Failed to get configuration: {e}")
                sys.exit(1)
        
        elif args.list:
            print("⚙️ Current configuration:")
            
            try:
                config_dict = self.config.to_dict()
                for key, value in config_dict.items():
                    # Hide sensitive values
                    if 'key' in key.lower() or 'password' in key.lower():
                        value = '*' * len(str(value)) if value else 'Not set'
                    print(f"   {key} = {value}")
                    
            except Exception as e:
                print(f"❌ Failed to list configuration: {e}")
                sys.exit(1)
        
        elif args.reset:
            print("⚙️ Resetting configuration to defaults...")
            
            try:
                self.config.reset_to_defaults()
                self.config.save()
                print(f"✅ Configuration reset successfully!")
                
            except Exception as e:
                print(f"❌ Failed to reset configuration: {e}")
                sys.exit(1)
    
    def _show_status(self) -> None:
        """Show system status."""
        print("🎯 AI-Powered Thesis Assistant - System Status")
        print("=" * 50)
        
        try:
            status = self.command_handler.get_system_status()
            
            print(f"Version: {status.get('version', 'Unknown')}")
            print(f"Configuration: {status.get('config_status', 'Unknown')}")
            print(f"API Connections: {status.get('api_status', 'Unknown')}")
            print(f"Data Directory: {status.get('data_dir', 'Unknown')}")
            print(f"Index Directory: {status.get('index_dir', 'Unknown')}")
            print(f"Projects Directory: {status.get('projects_dir', 'Unknown')}")
            print()
            
            # Component status
            print("Component Status:")
            components = status.get('components', {})
            for name, component_status in components.items():
                status_icon = "✅" if component_status else "❌"
                print(f"  {status_icon} {name}")
            
        except Exception as e:
            print(f"❌ Failed to get system status: {e}")
            sys.exit(1)
