"""
Interactive Mode for the AI-Powered Thesis Assistant CLI.

This module implements an interactive command-line interface with
tab completion, command history, and user-friendly prompts.

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import asyncio
import logging
import shlex
from typing import Dict, List, Any, Optional, Callable

# Local imports
from core.config import Config
from cli.commands import CommandHandler

logger = logging.getLogger(__name__)


class InteractiveMode:
    """
    Interactive CLI mode with comprehensive command support.
    
    This class provides a user-friendly interactive interface with
    command completion, history, and contextual help.
    """
    
    def __init__(self, config: Config, command_handler: CommandHandler):
        """
        Initialize interactive mode.
        
        Args:
            config: Application configuration
            command_handler: Command handler instance
        """
        self.config = config
        self.command_handler = command_handler
        self.running = True
        self.command_history = []
        
        # Command registry
        self.commands = {
            'help': self._cmd_help,
            'exit': self._cmd_exit,
            'quit': self._cmd_exit,
            'status': self._cmd_status,
            'config': self._cmd_config,
            'process': self._cmd_process,
            'analyze': self._cmd_analyze,
            'chat': self._cmd_chat,
            'citations': self._cmd_citations,
            'project': self._cmd_project,
            'search': self._cmd_search,
            'clear': self._cmd_clear,
            'history': self._cmd_history
        }
        
        logger.info("Interactive mode initialized")
    
    def run(self) -> None:
        """Run the interactive mode."""
        self._show_welcome()
        
        while self.running:
            try:
                # Get user input
                prompt = self._get_prompt()
                user_input = input(prompt).strip()
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Add to history
                self.command_history.append(user_input)
                
                # Parse and execute command
                self._execute_command(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Use 'exit' to quit or Ctrl+C again to force quit.")
                try:
                    input()  # Wait for another Ctrl+C
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Interactive mode error: {e}")
    
    def _show_welcome(self) -> None:
        """Show welcome message."""
        print("🎯 Welcome to AI-Powered Thesis Assistant Interactive Mode!")
        print("=" * 60)
        print("Available commands:")
        print("  help          - Show available commands")
        print("  status        - Show system status")
        print("  process       - Process documents")
        print("  analyze       - Analyze thesis")
        print("  chat          - AI chat interface")
        print("  citations     - Manage citations")
        print("  project       - Manage projects")
        print("  config        - Manage configuration")
        print("  exit/quit     - Exit interactive mode")
        print()
        print("Type 'help <command>' for detailed help on a specific command.")
        print()
    
    def _get_prompt(self) -> str:
        """Get the command prompt."""
        current_project = self.config.get('current_project', '')
        if current_project:
            project_name = current_project.split('/')[-1]
            return f"thesis-assistant ({project_name})> "
        else:
            return "thesis-assistant> "
    
    def _execute_command(self, user_input: str) -> None:
        """
        Execute a user command.
        
        Args:
            user_input: User input string
        """
        try:
            # Parse command and arguments
            parts = shlex.split(user_input)
            if not parts:
                return
            
            command = parts[0].lower()
            args = parts[1:]
            
            # Execute command
            if command in self.commands:
                # Run async commands in event loop
                if asyncio.iscoroutinefunction(self.commands[command]):
                    asyncio.run(self.commands[command](args))
                else:
                    self.commands[command](args)
            else:
                print(f"❌ Unknown command: {command}")
                print("Type 'help' for available commands.")
                
        except Exception as e:
            print(f"❌ Command error: {e}")
            logger.error(f"Command execution error: {e}")
    
    def _cmd_help(self, args: List[str]) -> None:
        """Show help information."""
        if not args:
            # General help
            print("📚 Available Commands:")
            print()
            for cmd_name in sorted(self.commands.keys()):
                if cmd_name in ['quit']:  # Skip aliases
                    continue
                print(f"  {cmd_name:<12} - {self._get_command_description(cmd_name)}")
            print()
            print("Type 'help <command>' for detailed help on a specific command.")
        else:
            # Specific command help
            command = args[0].lower()
            if command in self.commands:
                self._show_command_help(command)
            else:
                print(f"❌ Unknown command: {command}")
    
    def _cmd_exit(self, args: List[str]) -> None:
        """Exit interactive mode."""
        print("👋 Goodbye!")
        self.running = False
    
    def _cmd_status(self, args: List[str]) -> None:
        """Show system status."""
        print("🎯 System Status")
        print("-" * 20)
        
        status = self.command_handler.get_system_status()
        
        print(f"Version: {status.get('version', 'Unknown')}")
        print(f"Configuration: {status.get('config_status', 'Unknown')}")
        print(f"API Status: {status.get('api_status', 'Unknown')}")
        print()
        
        print("Directories:")
        print(f"  Data: {status.get('data_dir', 'Unknown')}")
        print(f"  Index: {status.get('index_dir', 'Unknown')}")
        print(f"  Projects: {status.get('projects_dir', 'Unknown')}")
        print()
        
        print("Components:")
        components = status.get('components', {})
        for name, component_status in components.items():
            status_icon = "✅" if component_status else "❌"
            print(f"  {status_icon} {name}")
    
    def _cmd_config(self, args: List[str]) -> None:
        """Manage configuration."""
        if not args:
            print("📋 Configuration Commands:")
            print("  config list           - List all configuration")
            print("  config get <key>      - Get configuration value")
            print("  config set <key> <value> - Set configuration value")
            print("  config reset          - Reset to defaults")
            return
        
        subcommand = args[0].lower()
        
        if subcommand == 'list':
            print("⚙️ Current Configuration:")
            config_dict = self.config.to_dict()
            for key, value in config_dict.items():
                # Hide sensitive values
                if 'key' in key.lower() or 'password' in key.lower():
                    value = '*' * len(str(value)) if value else 'Not set'
                print(f"  {key} = {value}")
        
        elif subcommand == 'get':
            if len(args) < 2:
                print("❌ Usage: config get <key>")
                return
            
            key = args[1]
            value = self.config.get(key)
            print(f"  {key} = {value}")
        
        elif subcommand == 'set':
            if len(args) < 3:
                print("❌ Usage: config set <key> <value>")
                return
            
            key = args[1]
            value = ' '.join(args[2:])  # Join remaining args as value
            
            self.config.set(key, value)
            self.config.save()
            print(f"✅ Set {key} = {value}")
        
        elif subcommand == 'reset':
            confirm = input("⚠️ Reset all configuration to defaults? (y/N): ")
            if confirm.lower() == 'y':
                self.config.reset_to_defaults()
                self.config.save()
                print("✅ Configuration reset to defaults")
            else:
                print("❌ Reset cancelled")
        
        else:
            print(f"❌ Unknown config subcommand: {subcommand}")
    
    async def _cmd_process(self, args: List[str]) -> None:
        """Process documents."""
        if not args:
            print("📄 Process Documents:")
            print("Usage: process <input_path> [options]")
            print("Options:")
            print("  --output <path>    - Output directory (default: output)")
            print("  --format <format>  - File format filter (pdf, txt, md, all)")
            print("  --ocr             - Enable OCR processing")
            print("  --index           - Enable document indexing")
            return
        
        # Parse arguments
        input_path = args[0]
        output_path = 'output'
        file_format = 'all'
        enable_ocr = False
        enable_indexing = False
        
        i = 1
        while i < len(args):
            if args[i] == '--output' and i + 1 < len(args):
                output_path = args[i + 1]
                i += 2
            elif args[i] == '--format' and i + 1 < len(args):
                file_format = args[i + 1]
                i += 2
            elif args[i] == '--ocr':
                enable_ocr = True
                i += 1
            elif args[i] == '--index':
                enable_indexing = True
                i += 1
            else:
                i += 1
        
        # Execute processing
        try:
            result = await self.command_handler.process_documents(
                input_path=input_path,
                output_path=output_path,
                file_format=file_format,
                enable_ocr=enable_ocr,
                enable_indexing=enable_indexing
            )
            
            print(f"✅ Processing completed!")
            print(f"   Processed: {result.get('processed', 0)} files")
            print(f"   Failed: {result.get('failed', 0)} files")
            print(f"   Output: {result.get('output_path', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
    
    async def _cmd_analyze(self, args: List[str]) -> None:
        """Analyze thesis."""
        if not args:
            print("📊 Analyze Thesis:")
            print("Usage: analyze <thesis_path> [options]")
            print("Options:")
            print("  --sources <path>   - Sources directory")
            print("  --output <path>    - Output directory (default: analysis)")
            print("  --claims          - Detect claims requiring citations")
            print("  --citations       - Validate existing citations")
            print("  --apa7            - Check APA7 compliance")
            return
        
        # Parse arguments
        thesis_path = args[0]
        sources_path = None
        output_path = 'analysis'
        detect_claims = False
        validate_citations = False
        check_apa7 = False
        
        i = 1
        while i < len(args):
            if args[i] == '--sources' and i + 1 < len(args):
                sources_path = args[i + 1]
                i += 2
            elif args[i] == '--output' and i + 1 < len(args):
                output_path = args[i + 1]
                i += 2
            elif args[i] == '--claims':
                detect_claims = True
                i += 1
            elif args[i] == '--citations':
                validate_citations = True
                i += 1
            elif args[i] == '--apa7':
                check_apa7 = True
                i += 1
            else:
                i += 1
        
        # If no specific analysis requested, do all
        if not any([detect_claims, validate_citations, check_apa7]):
            detect_claims = validate_citations = check_apa7 = True
        
        # Execute analysis
        try:
            result = await self.command_handler.analyze_thesis(
                thesis_path=thesis_path,
                sources_path=sources_path,
                output_path=output_path,
                detect_claims=detect_claims,
                validate_citations=validate_citations,
                check_apa7=check_apa7
            )
            
            print(f"✅ Analysis completed!")
            print(f"   Claims detected: {result.get('claims_count', 'N/A')}")
            print(f"   Citations validated: {result.get('citations_count', 'N/A')}")
            print(f"   APA7 compliance: {result.get('apa7_compliance', 'N/A')}")
            print(f"   Report: {result.get('report_path', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    async def _cmd_chat(self, args: List[str]) -> None:
        """AI chat interface."""
        if args and args[0] == '--help':
            print("🤖 AI Chat:")
            print("Usage: chat [message]")
            print("  chat                    - Start interactive chat")
            print("  chat \"your message\"     - Send single message")
            print("  chat --model <model>    - Use specific model")
            return
        
        # Parse arguments
        model = None
        message = None
        
        i = 0
        while i < len(args):
            if args[i] == '--model' and i + 1 < len(args):
                model = args[i + 1]
                i += 2
            else:
                # Treat as message
                message = ' '.join(args[i:])
                break
        
        if message:
            # Single message mode
            try:
                response = await self.command_handler.ai_chat(message=message, model=model)
                print(f"\n🤖 AI: {response}")
            except Exception as e:
                print(f"❌ AI chat failed: {e}")
        else:
            # Interactive chat mode
            print("🤖 Starting AI chat session...")
            print("Type 'exit' to return to main prompt.")
            await self.command_handler.interactive_chat(model=model)
    
    async def _cmd_citations(self, args: List[str]) -> None:
        """Manage citations."""
        if not args:
            print("📚 Citations Management:")
            print("  citations generate <file>  - Generate citations")
            print("  citations validate <file>  - Validate citations")
            print("  citations format <format>  - Set citation format (apa7, mla, chicago)")
            return
        
        subcommand = args[0].lower()
        
        if subcommand == 'generate':
            if len(args) < 2:
                print("❌ Usage: citations generate <file>")
                return
            
            try:
                result = await self.command_handler.generate_citations(
                    input_path=args[1],
                    citation_format='apa7'
                )
                print(f"✅ Generated {result.get('count', 0)} citations")
                print(f"   Output: {result.get('output_path', 'N/A')}")
            except Exception as e:
                print(f"❌ Citation generation failed: {e}")
        
        elif subcommand == 'validate':
            if len(args) < 2:
                print("❌ Usage: citations validate <file>")
                return
            
            try:
                result = await self.command_handler.validate_citations(
                    input_path=args[1],
                    citation_format='apa7'
                )
                print(f"✅ Citation validation completed")
                print(f"   Valid: {result.get('valid', 0)}")
                print(f"   Invalid: {result.get('invalid', 0)}")
                print(f"   Compliance: {result.get('compliance_rate', 0):.1f}%")
            except Exception as e:
                print(f"❌ Citation validation failed: {e}")
        
        else:
            print(f"❌ Unknown citations subcommand: {subcommand}")
    
    def _cmd_project(self, args: List[str]) -> None:
        """Manage projects."""
        if not args:
            print("📁 Project Management:")
            print("  project create <name>  - Create new project")
            print("  project open <name>    - Open existing project")
            print("  project list          - List all projects")
            print("  project delete <name>  - Delete project")
            return
        
        subcommand = args[0].lower()
        
        if subcommand == 'create':
            if len(args) < 2:
                print("❌ Usage: project create <name>")
                return
            
            try:
                project_path = self.command_handler.create_project(args[1])
                print(f"✅ Project created: {project_path}")
            except Exception as e:
                print(f"❌ Project creation failed: {e}")
        
        elif subcommand == 'open':
            if len(args) < 2:
                print("❌ Usage: project open <name>")
                return
            
            try:
                self.command_handler.open_project(args[1])
                print(f"✅ Opened project: {args[1]}")
            except Exception as e:
                print(f"❌ Failed to open project: {e}")
        
        elif subcommand == 'list':
            try:
                projects = self.command_handler.list_projects()
                if projects:
                    print("📋 Available projects:")
                    for i, project in enumerate(projects, 1):
                        print(f"   {i}. {project['name']}")
                else:
                    print("📋 No projects found")
            except Exception as e:
                print(f"❌ Failed to list projects: {e}")
        
        elif subcommand == 'delete':
            if len(args) < 2:
                print("❌ Usage: project delete <name>")
                return
            
            confirm = input(f"⚠️ Delete project '{args[1]}'? (y/N): ")
            if confirm.lower() == 'y':
                try:
                    self.command_handler.delete_project(args[1])
                    print(f"✅ Project deleted: {args[1]}")
                except Exception as e:
                    print(f"❌ Failed to delete project: {e}")
            else:
                print("❌ Deletion cancelled")
        
        else:
            print(f"❌ Unknown project subcommand: {subcommand}")
    
    def _cmd_search(self, args: List[str]) -> None:
        """Search documents."""
        if not args:
            print("❌ Usage: search <query>")
            return

        query = ' '.join(args)
        print(f"🔍 Searching for: {query}")

        try:
            # Use the search engine from command handler
            if not self.command_handler.search_engine:
                print("❌ Search engine not initialized. Please process documents first.")
                return

            # Perform search
            results = self.command_handler.search_engine.search(query, limit=10)

            if results:
                print(f"✅ Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    score = result.get('score', 0.0)
                    doc_path = result.get('document_path', 'Unknown')
                    snippet = result.get('text', '')[:100] + '...' if len(result.get('text', '')) > 100 else result.get('text', '')

                    print(f"\n{i}. {doc_path} (Score: {score:.3f})")
                    print(f"   {snippet}")
            else:
                print("❌ No results found for your query.")

        except Exception as e:
            print(f"❌ Search failed: {e}")
            logger.error(f"Search error: {e}")
    
    def _cmd_clear(self, args: List[str]) -> None:
        """Clear the screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _cmd_history(self, args: List[str]) -> None:
        """Show command history."""
        print("📜 Command History:")
        for i, cmd in enumerate(self.command_history[-20:], 1):  # Show last 20 commands
            print(f"   {i:2d}. {cmd}")
    
    def _get_command_description(self, command: str) -> str:
        """Get command description."""
        descriptions = {
            'help': 'Show help information',
            'exit': 'Exit interactive mode',
            'status': 'Show system status',
            'config': 'Manage configuration',
            'process': 'Process documents',
            'analyze': 'Analyze thesis',
            'chat': 'AI chat interface',
            'citations': 'Manage citations',
            'project': 'Manage projects',
            'search': 'Search documents',
            'clear': 'Clear the screen',
            'history': 'Show command history'
        }
        return descriptions.get(command, 'No description available')
    
    def _show_command_help(self, command: str) -> None:
        """Show detailed help for a specific command."""
        help_text = {
            'process': """📄 Process Documents
Usage: process <input_path> [options]

Process documents from the specified input path.

Arguments:
  input_path           Path to file or directory to process

Options:
  --output <path>      Output directory (default: output)
  --format <format>    File format filter: pdf, txt, md, all (default: all)
  --ocr               Enable OCR for scanned documents
  --index             Enable document indexing

Examples:
  process documents/
  process thesis.pdf --output processed/
  process sources/ --format pdf --ocr --index""",
            
            'analyze': """📊 Analyze Thesis
Usage: analyze <thesis_path> [options]

Perform comprehensive thesis analysis.

Arguments:
  thesis_path          Path to thesis document

Options:
  --sources <path>     Sources directory for reference
  --output <path>      Output directory (default: analysis)
  --claims            Detect claims requiring citations
  --citations         Validate existing citations
  --apa7              Check APA7 compliance

Examples:
  analyze thesis.pdf
  analyze thesis.pdf --sources references/ --claims
  analyze thesis.pdf --output results/ --apa7""",
            
            'chat': """🤖 AI Chat Interface
Usage: chat [message] [options]

Interact with AI models for assistance.

Options:
  --model <model>      Specify AI model to use

Examples:
  chat                           # Start interactive chat
  chat "Explain APA citations"   # Send single message
  chat --model gpt-4 "Help me"   # Use specific model"""
        }
        
        if command in help_text:
            print(help_text[command])
        else:
            print(f"📚 {command.title()}")
            print(f"Description: {self._get_command_description(command)}")
            print("No detailed help available for this command.")
