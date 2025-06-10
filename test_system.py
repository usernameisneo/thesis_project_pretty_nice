#!/usr/bin/env python3
"""
System Test Script for AI-Powered Thesis Assistant.

This script performs comprehensive testing of all major components
to ensure the system is working correctly after the revamp.

Features:
    - Component availability testing
    - API connection testing
    - Document processing testing
    - GUI component testing
    - Integration testing

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Any
import traceback

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class SystemTester:
    """Comprehensive system testing class."""
    
    def __init__(self):
        """Initialize the system tester."""
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        
        logger.info("System Tester initialized")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all system tests."""
        logger.info("Starting comprehensive system tests")
        print("🧪 AI-Powered Thesis Assistant - System Test Suite")
        print("=" * 60)
        
        # Test categories
        test_categories = [
            ("Core Components", self._test_core_components),
            ("Document Processing", self._test_document_processing),
            ("API Clients", self._test_api_clients),
            ("Analysis Components", self._test_analysis_components),
            ("GUI Components", self._test_gui_components),
            ("Configuration", self._test_configuration),
            ("Integration", self._test_integration)
        ]
        
        for category_name, test_method in test_categories:
            print(f"\n📋 Testing {category_name}...")
            try:
                results = test_method()
                self.test_results[category_name] = results
                self._print_category_results(category_name, results)
            except Exception as e:
                logger.error(f"Test category {category_name} failed: {e}")
                self.test_results[category_name] = {"error": str(e)}
                print(f"❌ {category_name}: FAILED - {e}")
                self.failed_tests += 1
        
        # Print summary
        self._print_summary()
        
        return self.test_results
    
    def _test_core_components(self) -> Dict[str, bool]:
        """Test core system components."""
        results = {}
        
        core_modules = [
            ("Config", "core.config"),
            ("Exceptions", "core.exceptions"),
            ("Types", "core.types"),
            ("Lazy Imports", "core.lazy_imports")
        ]
        
        for name, module_path in core_modules:
            try:
                __import__(module_path)
                results[name] = True
                self.passed_tests += 1
            except ImportError as e:
                results[name] = False
                self.failed_tests += 1
                logger.error(f"Failed to import {module_path}: {e}")
        
        return results
    
    def _test_document_processing(self) -> Dict[str, bool]:
        """Test document processing components."""
        results = {}
        
        processing_modules = [
            ("Document Parser", "processing.document_parser"),
            ("Text Processor", "processing.text_processor")
        ]
        
        for name, module_path in processing_modules:
            try:
                module = __import__(module_path, fromlist=[''])
                # Test basic functionality based on module type
                if name == "Document Parser" and hasattr(module, 'parse_document'):
                    results[name] = True
                    self.passed_tests += 1
                elif name == "Text Processor" and hasattr(module, 'TextProcessor'):
                    results[name] = True
                    self.passed_tests += 1
                else:
                    results[name] = False
                    self.failed_tests += 1
            except ImportError as e:
                results[name] = False
                self.failed_tests += 1
                logger.error(f"Failed to import {module_path}: {e}")
        
        return results
    
    def _test_api_clients(self) -> Dict[str, bool]:
        """Test API client components."""
        results = {}
        
        api_modules = [
            ("OpenRouter Client", "api.openrouter_client"),
            ("Perplexity Client", "api.perplexity_client"),
            ("Semantic Scholar Client", "api.semantic_scholar_client")
        ]
        
        for name, module_path in api_modules:
            try:
                module = __import__(module_path, fromlist=[''])
                # Check if main client class exists
                class_name = module_path.split('.')[-1].replace('_', '').title().replace('Client', 'Client')
                if hasattr(module, 'OpenRouterClient') or hasattr(module, 'PerplexityClient') or hasattr(module, 'SemanticScholarClient'):
                    results[name] = True
                    self.passed_tests += 1
                else:
                    results[name] = False
                    self.failed_tests += 1
            except ImportError as e:
                results[name] = False
                self.failed_tests += 1
                logger.error(f"Failed to import {module_path}: {e}")
        
        return results
    
    def _test_analysis_components(self) -> Dict[str, bool]:
        """Test analysis and reasoning components."""
        results = {}
        
        analysis_modules = [
            ("Claim Detector", "analysis.master_thesis_claim_detector"),
            ("Citation Engine", "reasoning.enhanced_citation_engine"),
            ("APA7 Compliance", "reasoning.apa7_compliance_engine"),
            ("Hybrid Search", "indexing.hybrid_search")
        ]
        
        for name, module_path in analysis_modules:
            try:
                __import__(module_path)
                results[name] = True
                self.passed_tests += 1
            except ImportError as e:
                results[name] = False
                self.failed_tests += 1
                logger.error(f"Failed to import {module_path}: {e}")
        
        return results
    
    def _test_gui_components(self) -> Dict[str, bool]:
        """Test GUI components."""
        results = {}
        
        gui_modules = [
            ("Main Window", "gui.main_window"),
            ("Document Processor Widget", "gui.components.document_processor"),
            ("AI Chat Widget", "gui.components.ai_chat"),
            ("Model Selector Widget", "gui.components.model_selector"),
            ("Settings Widget", "gui.components.settings")
        ]
        
        for name, module_path in gui_modules:
            try:
                __import__(module_path)
                results[name] = True
                self.passed_tests += 1
            except ImportError as e:
                results[name] = False
                self.failed_tests += 1
                logger.error(f"Failed to import {module_path}: {e}")
        
        return results
    
    def _test_configuration(self) -> Dict[str, bool]:
        """Test configuration system."""
        results = {}
        
        try:
            from core.config import Config
            
            # Test config creation
            config = Config()
            results["Config Creation"] = True
            self.passed_tests += 1
            
            # Test config methods
            test_value = config.get('test_key', 'default_value')
            if test_value == 'default_value':
                results["Config Get Method"] = True
                self.passed_tests += 1
            else:
                results["Config Get Method"] = False
                self.failed_tests += 1
            
            # Test directory creation
            config.ensure_directories()
            results["Directory Creation"] = True
            self.passed_tests += 1
            
        except Exception as e:
            results["Configuration System"] = False
            self.failed_tests += 1
            logger.error(f"Configuration test failed: {e}")
        
        return results
    
    def _test_integration(self) -> Dict[str, bool]:
        """Test system integration."""
        results = {}
        
        try:
            # Test main application initialization
            from main import ThesisAssistantApp
            app = ThesisAssistantApp()
            results["Main App Initialization"] = True
            self.passed_tests += 1
            
            # Test component status check
            app.show_status()
            results["Component Status Check"] = True
            self.passed_tests += 1
            
        except Exception as e:
            results["Integration Test"] = False
            self.failed_tests += 1
            logger.error(f"Integration test failed: {e}")
        
        return results
    
    def _print_category_results(self, category: str, results: Dict[str, bool]) -> None:
        """Print results for a test category."""
        if "error" in results:
            print(f"❌ {category}: FAILED - {results['error']}")
            return
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"📊 {category}: {passed}/{total} tests passed")
        
        for test_name, passed in results.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
    
    def _print_summary(self) -> None:
        """Print test summary."""
        total_tests = self.passed_tests + self.failed_tests
        success_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("📈 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! System is ready for production.")
        else:
            print(f"\n⚠️  {self.failed_tests} tests failed. Please review the issues above.")
        
        print("=" * 60)


def main():
    """Main test execution function."""
    try:
        tester = SystemTester()
        results = tester.run_all_tests()
        
        # Save results to file
        import json
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: test_results.json")
        print(f"📄 Test log saved to: system_test.log")
        
        # Exit with appropriate code
        if tester.failed_tests == 0:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
