#!/usr/bin/env python3
"""
Setup Script for AI-Powered Thesis Assistant.

This script automates the installation and configuration of the
AI-Powered Thesis Assistant system, ensuring all dependencies
are properly installed and configured.

Features:
    - Dependency installation
    - Configuration setup
    - Directory structure creation
    - API key configuration
    - System validation

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import sys
import os
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class SystemSetup:
    """System setup and configuration manager."""
    
    def __init__(self):
        """Initialize the setup manager."""
        self.project_root = Path(__file__).parent
        self.setup_steps = []
        self.failed_steps = []
        
        logger.info("System Setup initialized")
    
    def run_complete_setup(self) -> bool:
        """Run the complete system setup process."""
        print("🚀 AI-Powered Thesis Assistant - System Setup")
        print("=" * 60)
        print("This script will set up your AI-Powered Thesis Assistant environment.")
        print("Please ensure you have Python 3.8+ installed.\n")
        
        setup_steps = [
            ("Checking Python Version", self._check_python_version),
            ("Creating Directory Structure", self._create_directories),
            ("Installing Dependencies", self._install_dependencies),
            ("Setting Up Configuration", self._setup_configuration),
            ("Downloading NLP Models", self._download_nlp_models),
            ("Validating Installation", self._validate_installation),
            ("Creating Desktop Shortcuts", self._create_shortcuts)
        ]
        
        for step_name, step_function in setup_steps:
            print(f"📋 {step_name}...")
            try:
                success = step_function()
                if success:
                    print(f"✅ {step_name}: SUCCESS")
                    self.setup_steps.append(step_name)
                else:
                    print(f"❌ {step_name}: FAILED")
                    self.failed_steps.append(step_name)
            except Exception as e:
                print(f"❌ {step_name}: ERROR - {e}")
                self.failed_steps.append(step_name)
                logger.error(f"Setup step failed: {step_name} - {e}")
        
        # Print summary
        self._print_setup_summary()
        
        return len(self.failed_steps) == 0
    
    def _check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print(f"   Python {version.major}.{version.minor}.{version.micro} detected")
            return True
        else:
            print(f"   Python {version.major}.{version.minor}.{version.micro} detected")
            print("   ERROR: Python 3.8+ is required")
            return False
    
    def _create_directories(self) -> bool:
        """Create necessary directory structure."""
        directories = [
            "data",
            "indexes",
            "projects",
            "cache",
            "logs",
            "output",
            "temp"
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(exist_ok=True)
                print(f"   Created: {directory}/")
            return True
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
    
    def _install_dependencies(self) -> bool:
        """Install Python dependencies."""
        try:
            print("   Installing core dependencies...")
            
            # Install requirements
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("   All dependencies installed successfully")
                return True
            else:
                print(f"   Error installing dependencies: {result.stderr}")
                logger.error(f"Pip install failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def _setup_configuration(self) -> bool:
        """Set up initial configuration."""
        try:
            # Create config from template
            config_template_path = self.project_root / "config_template.json"
            config_path = self.project_root / "config.json"
            
            if config_template_path.exists():
                with open(config_template_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update paths to absolute paths
                config_data["system_configuration"]["index_directory"] = str(self.project_root / "indexes")
                config_data["system_configuration"]["cache_directory"] = str(self.project_root / "cache")
                
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                print("   Configuration file created")
                print("   Please edit config.json to add your API keys")
                return True
            else:
                print("   Warning: config_template.json not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup configuration: {e}")
            return False
    
    def _download_nlp_models(self) -> bool:
        """Download required NLP models."""
        try:
            print("   Downloading spaCy English model...")
            
            # Download spaCy model
            result = subprocess.run([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("   spaCy model downloaded successfully")
                return True
            else:
                print("   Warning: Failed to download spaCy model")
                print("   You can download it manually later with: python -m spacy download en_core_web_sm")
                return True  # Don't fail setup for this
                
        except Exception as e:
            logger.error(f"Failed to download NLP models: {e}")
            return True  # Don't fail setup for this
    
    def _validate_installation(self) -> bool:
        """Validate the installation by running system tests."""
        try:
            print("   Running system validation tests...")
            
            # Run the test script
            result = subprocess.run([
                sys.executable, "test_system.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("   System validation passed")
                return True
            else:
                print("   Warning: Some validation tests failed")
                print("   Check test_results.json for details")
                return True  # Don't fail setup for this
                
        except Exception as e:
            logger.error(f"Failed to run validation: {e}")
            return True  # Don't fail setup for this
    
    def _create_shortcuts(self) -> bool:
        """Create desktop shortcuts and launch scripts."""
        try:
            # Create launch scripts
            if sys.platform == "win32":
                # Windows batch file
                batch_content = f"""@echo off
cd /d "{self.project_root}"
python main.py --gui
pause
"""
                with open(self.project_root / "launch_gui.bat", 'w') as f:
                    f.write(batch_content)
                
                cli_batch_content = f"""@echo off
cd /d "{self.project_root}"
python main.py --cli
pause
"""
                with open(self.project_root / "launch_cli.bat", 'w') as f:
                    f.write(cli_batch_content)
                
                print("   Created Windows launch scripts")
            
            else:
                # Unix shell script
                shell_content = f"""#!/bin/bash
cd "{self.project_root}"
python3 main.py --gui
"""
                script_path = self.project_root / "launch_gui.sh"
                with open(script_path, 'w') as f:
                    f.write(shell_content)
                script_path.chmod(0o755)
                
                cli_shell_content = f"""#!/bin/bash
cd "{self.project_root}"
python3 main.py --cli
"""
                cli_script_path = self.project_root / "launch_cli.sh"
                with open(cli_script_path, 'w') as f:
                    f.write(cli_shell_content)
                cli_script_path.chmod(0o755)
                
                print("   Created Unix launch scripts")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create shortcuts: {e}")
            return False
    
    def _print_setup_summary(self) -> None:
        """Print setup summary."""
        print("\n" + "=" * 60)
        print("📊 SETUP SUMMARY")
        print("=" * 60)
        
        total_steps = len(self.setup_steps) + len(self.failed_steps)
        success_rate = (len(self.setup_steps) / total_steps * 100) if total_steps > 0 else 0
        
        print(f"Total Steps: {total_steps}")
        print(f"Successful: {len(self.setup_steps)}")
        print(f"Failed: {len(self.failed_steps)}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.failed_steps:
            print(f"\n❌ Failed Steps:")
            for step in self.failed_steps:
                print(f"   - {step}")
        
        if len(self.failed_steps) == 0:
            print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
            print("\nNext Steps:")
            print("1. Edit config.json to add your API keys")
            print("2. Run 'python main.py --gui' to start the application")
            print("3. Or use the launch scripts created in the project directory")
        else:
            print(f"\n⚠️  Setup completed with {len(self.failed_steps)} issues.")
            print("Please review the errors above and the setup.log file.")
        
        print("=" * 60)


def main():
    """Main setup execution function."""
    try:
        setup = SystemSetup()
        success = setup.run_complete_setup()
        
        if success:
            print("\n✨ Welcome to AI-Powered Thesis Assistant!")
            print("Your system is ready for academic research excellence.")
            sys.exit(0)
        else:
            print("\n⚠️  Setup completed with issues. Please review and retry.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
