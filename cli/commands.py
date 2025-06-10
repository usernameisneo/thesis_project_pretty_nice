"""
Command Handler for the AI-Powered Thesis Assistant CLI.

This module implements all CLI command handlers with comprehensive
functionality for document processing, AI operations, and system management.

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Local imports
from core.config import Config
from core.exceptions import ApplicationError
from processing.document_parser import parse_document
from processing.text_processor import TextProcessor
from indexing.hybrid_search import HybridSearchEngine
from analysis.master_thesis_claim_detector import MasterThesisClaimDetector
from reasoning.enhanced_citation_engine import EnhancedCitationEngine
from reasoning.apa7_compliance_engine import APA7ComplianceEngine
from api.openrouter_client import OpenRouterClient
from api.perplexity_client import PerplexityClient
from api.semantic_scholar_client import SemanticScholarClient

logger = logging.getLogger(__name__)


class CommandHandler:
    """
    Command handler for CLI operations.
    
    This class implements all CLI command functionality with proper
    error handling and progress reporting.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the command handler.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.text_processor = TextProcessor()
        self.search_engine = None
        self.claim_detector = None
        self.citation_engine = None
        self.apa7_engine = None
        self.openrouter_client = None
        self.perplexity_client = None
        self.semantic_client = None
        
        self._initialize_components()
        
        logger.info("Command handler initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all components."""
        try:
            # Initialize search engine
            index_dir = self.config.get('index_dir', 'indexes')
            self.search_engine = HybridSearchEngine(index_dir)
            
            # Initialize claim detector
            self.claim_detector = MasterThesisClaimDetector()
            
            # Initialize citation engines
            self.citation_engine = EnhancedCitationEngine()
            self.apa7_engine = APA7ComplianceEngine()
            
            # Initialize API clients
            openrouter_key = self.config.get('openrouter_api_key')
            if openrouter_key:
                self.openrouter_client = OpenRouterClient(openrouter_key)
            
            perplexity_key = self.config.get('perplexity_api_key')
            if perplexity_key:
                self.perplexity_client = PerplexityClient(perplexity_key)
            
            semantic_key = self.config.get('semantic_scholar_api_key')
            if semantic_key:
                self.semantic_client = SemanticScholarClient(semantic_key)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
    
    async def process_documents(self, input_path: str, output_path: str, 
                              file_format: str = 'all', enable_ocr: bool = False,
                              enable_indexing: bool = False) -> Dict[str, Any]:
        """
        Process documents from input path.
        
        Args:
            input_path: Input file or directory path
            output_path: Output directory path
            file_format: File format filter
            enable_ocr: Enable OCR processing
            enable_indexing: Enable document indexing
            
        Returns:
            Processing results
        """
        input_path_obj = Path(input_path)
        output_path_obj = Path(output_path)
        output_path_obj.mkdir(parents=True, exist_ok=True)
        
        # Collect files to process
        files_to_process = []
        
        if input_path_obj.is_file():
            files_to_process.append(input_path_obj)
        elif input_path_obj.is_dir():
            # Find files based on format filter
            extensions = []
            if file_format == 'all':
                extensions = ['.pdf', '.txt', '.md', '.doc', '.docx']
            elif file_format == 'pdf':
                extensions = ['.pdf']
            elif file_format == 'txt':
                extensions = ['.txt']
            elif file_format == 'md':
                extensions = ['.md']
            
            for ext in extensions:
                files_to_process.extend(input_path_obj.rglob(f"*{ext}"))
        
        # Process files
        processed_count = 0
        failed_count = 0
        
        for file_path in files_to_process:
            try:
                print(f"Processing: {file_path.name}")
                
                # Parse document
                text, metadata = parse_document(str(file_path))
                
                # Process text
                processed_chunks = self.text_processor.process_text(text, metadata)
                
                # Save processed text
                output_file = output_path_obj / f"{file_path.stem}_processed.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                # Index if enabled
                if enable_indexing and self.search_engine:
                    self.search_engine.add_document(str(file_path), processed_chunks, metadata)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                failed_count += 1
        
        return {
            'processed': processed_count,
            'failed': failed_count,
            'output_path': str(output_path_obj)
        }
    
    async def analyze_thesis(self, thesis_path: str, sources_path: Optional[str] = None,
                           output_path: str = 'analysis', detect_claims: bool = True,
                           validate_citations: bool = True, check_apa7: bool = True) -> Dict[str, Any]:
        """
        Analyze thesis document.
        
        Args:
            thesis_path: Path to thesis file
            sources_path: Path to sources directory
            output_path: Output directory path
            detect_claims: Enable claim detection
            validate_citations: Enable citation validation
            check_apa7: Enable APA7 compliance check
            
        Returns:
            Analysis results
        """
        thesis_path_obj = Path(thesis_path)
        output_path_obj = Path(output_path)
        output_path_obj.mkdir(parents=True, exist_ok=True)
        
        # Parse thesis
        print("Parsing thesis document...")
        thesis_text, thesis_metadata = parse_document(str(thesis_path_obj))
        
        results = {
            'thesis_path': str(thesis_path_obj),
            'output_path': str(output_path_obj)
        }
        
        # Detect claims
        if detect_claims and self.claim_detector:
            print("Detecting claims requiring citations...")
            claims = await self.claim_detector.detect_claims(thesis_text)
            results['claims_count'] = len(claims)
            
            # Save claims report
            claims_file = output_path_obj / 'detected_claims.json'
            import json
            with open(claims_file, 'w', encoding='utf-8') as f:
                json.dump([claim.__dict__ for claim in claims], f, indent=2, default=str)
        
        # Validate citations
        if validate_citations and self.citation_engine:
            print("Validating citations...")
            citation_results = await self.citation_engine.validate_citations(thesis_text)
            results['citations_count'] = len(citation_results)
            
            # Save citation report
            citations_file = output_path_obj / 'citation_validation.json'
            import json
            with open(citations_file, 'w', encoding='utf-8') as f:
                json.dump(citation_results, f, indent=2, default=str)
        
        # Check APA7 compliance
        if check_apa7 and self.apa7_engine:
            print("Checking APA7 compliance...")
            compliance_results = await self.apa7_engine.check_compliance(thesis_text)
            results['apa7_compliance'] = f"{compliance_results.get('compliance_score', 0):.1f}%"
            
            # Save compliance report
            compliance_file = output_path_obj / 'apa7_compliance.json'
            import json
            with open(compliance_file, 'w', encoding='utf-8') as f:
                json.dump(compliance_results, f, indent=2, default=str)
        
        # Generate summary report
        report_file = output_path_obj / 'analysis_report.md'
        self._generate_analysis_report(results, report_file)
        results['report_path'] = str(report_file)
        
        return results
    
    async def ai_chat(self, message: str, model: Optional[str] = None,
                     context_file: Optional[str] = None) -> str:
        """
        Send message to AI and get response.
        
        Args:
            message: Message to send
            model: AI model to use
            context_file: Context file to include
            
        Returns:
            AI response
        """
        if not self.openrouter_client:
            raise ApplicationError("OpenRouter API key not configured")
        
        # Prepare messages
        messages = []
        
        # Add context if provided
        if context_file:
            context_path = Path(context_file)
            if context_path.exists():
                context_text = context_path.read_text(encoding='utf-8')
                messages.append({
                    "role": "system",
                    "content": f"Context: {context_text}"
                })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Get response
        response = self.openrouter_client.chat_completion(
            model=model or self.config.get('selected_model_id', 'gpt-3.5-turbo'),
            messages=messages,
            max_tokens=1000
        )
        
        if response and 'choices' in response and response['choices']:
            return response['choices'][0]['message']['content']
        else:
            raise ApplicationError("No response received from AI")
    
    async def interactive_chat(self, model: Optional[str] = None) -> None:
        """
        Start interactive chat session.
        
        Args:
            model: AI model to use
        """
        if not self.openrouter_client:
            print("❌ OpenRouter API key not configured")
            return
        
        conversation = []
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Add to conversation
                conversation.append({"role": "user", "content": user_input})
                
                # Get AI response
                print("🤖 AI: ", end="", flush=True)
                
                response = self.openrouter_client.chat_completion(
                    model=model or self.config.get('selected_model_id', 'gpt-3.5-turbo'),
                    messages=conversation[-10:],  # Keep last 10 messages
                    max_tokens=1000
                )
                
                if response and 'choices' in response and response['choices']:
                    ai_response = response['choices'][0]['message']['content']
                    print(ai_response)
                    conversation.append({"role": "assistant", "content": ai_response})
                else:
                    print("❌ No response received")
                
            except KeyboardInterrupt:
                print("\n👋 Chat session ended")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    async def generate_citations(self, input_path: str, output_path: Optional[str] = None,
                               citation_format: str = 'apa7') -> Dict[str, Any]:
        """
        Generate citations from document.
        
        Args:
            input_path: Input document path
            output_path: Output file path
            citation_format: Citation format
            
        Returns:
            Generation results
        """
        if not self.citation_engine:
            raise ApplicationError("Citation engine not available")
        
        # Parse document
        text, metadata = parse_document(input_path)
        
        # Generate citations
        citations = await self.citation_engine.generate_citations(text, citation_format)
        
        # Save citations
        if output_path:
            output_file = Path(output_path)
        else:
            input_file = Path(input_path)
            output_file = input_file.parent / f"{input_file.stem}_citations.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for citation in citations:
                f.write(f"{citation}\n")
        
        return {
            'count': len(citations),
            'output_path': str(output_file)
        }
    
    async def validate_citations(self, input_path: str, citation_format: str = 'apa7') -> Dict[str, Any]:
        """
        Validate citations in document.
        
        Args:
            input_path: Input document path
            citation_format: Citation format
            
        Returns:
            Validation results
        """
        if not self.citation_engine:
            raise ApplicationError("Citation engine not available")
        
        # Parse document
        text, metadata = parse_document(input_path)
        
        # Validate citations
        validation_results = await self.citation_engine.validate_citations(text)
        
        valid_count = sum(1 for result in validation_results if result.get('valid', False))
        invalid_count = len(validation_results) - valid_count
        compliance_rate = (valid_count / len(validation_results)) * 100 if validation_results else 0
        
        return {
            'valid': valid_count,
            'invalid': invalid_count,
            'compliance_rate': compliance_rate,
            'details': validation_results
        }
    
    def create_project(self, project_name: str) -> str:
        """Create a new thesis project."""
        projects_dir = Path(self.config.get('projects_dir', 'projects'))
        projects_dir.mkdir(parents=True, exist_ok=True)
        
        project_path = projects_dir / project_name
        project_path.mkdir(exist_ok=True)
        
        # Create project structure
        (project_path / 'documents').mkdir(exist_ok=True)
        (project_path / 'sources').mkdir(exist_ok=True)
        (project_path / 'analysis').mkdir(exist_ok=True)
        (project_path / 'output').mkdir(exist_ok=True)
        
        # Create project config
        project_config = {
            'name': project_name,
            'created': str(Path().cwd()),
            'version': '1.0'
        }
        
        import json
        with open(project_path / 'project.json', 'w') as f:
            json.dump(project_config, f, indent=2)
        
        return str(project_path)
    
    def open_project(self, project_name: str) -> None:
        """Open an existing project."""
        projects_dir = Path(self.config.get('projects_dir', 'projects'))
        project_path = projects_dir / project_name
        
        if not project_path.exists():
            raise ApplicationError(f"Project '{project_name}' not found")
        
        # Set current project in config
        self.config.set('current_project', str(project_path))
        self.config.save()
    
    def list_projects(self) -> List[Dict[str, str]]:
        """List all available projects."""
        projects_dir = Path(self.config.get('projects_dir', 'projects'))
        
        if not projects_dir.exists():
            return []
        
        projects = []
        for project_path in projects_dir.iterdir():
            if project_path.is_dir():
                projects.append({
                    'name': project_path.name,
                    'path': str(project_path)
                })
        
        return projects
    
    def delete_project(self, project_name: str) -> None:
        """Delete a project."""
        projects_dir = Path(self.config.get('projects_dir', 'projects'))
        project_path = projects_dir / project_name
        
        if not project_path.exists():
            raise ApplicationError(f"Project '{project_name}' not found")
        
        import shutil
        shutil.rmtree(project_path)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information."""
        return {
            'version': '2.0',
            'config_status': 'Loaded',
            'api_status': 'Connected' if self.openrouter_client else 'Not configured',
            'data_dir': self.config.get('data_dir', 'data'),
            'index_dir': self.config.get('index_dir', 'indexes'),
            'projects_dir': self.config.get('projects_dir', 'projects'),
            'components': {
                'Text Processor': bool(self.text_processor),
                'Search Engine': bool(self.search_engine),
                'Claim Detector': bool(self.claim_detector),
                'Citation Engine': bool(self.citation_engine),
                'APA7 Engine': bool(self.apa7_engine),
                'OpenRouter Client': bool(self.openrouter_client),
                'Perplexity Client': bool(self.perplexity_client),
                'Semantic Scholar Client': bool(self.semantic_client)
            }
        }
    
    def _generate_analysis_report(self, results: Dict[str, Any], output_file: Path) -> None:
        """Generate analysis report in Markdown format."""
        report = f"""# Thesis Analysis Report

## Summary
- **Thesis**: {results.get('thesis_path', 'N/A')}
- **Analysis Date**: {Path().cwd()}
- **Output Directory**: {results.get('output_path', 'N/A')}

## Results
- **Claims Detected**: {results.get('claims_count', 'N/A')}
- **Citations Validated**: {results.get('citations_count', 'N/A')}
- **APA7 Compliance**: {results.get('apa7_compliance', 'N/A')}

## Files Generated
- Detected Claims: `detected_claims.json`
- Citation Validation: `citation_validation.json`
- APA7 Compliance: `apa7_compliance.json`

---
Generated by AI-Powered Thesis Assistant v2.0
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
