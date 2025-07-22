#!/usr/bin/env python3
"""
YAML Syntax Checker for GitHub Actions Workflows

This script validates YAML files for syntax errors and common issues.
"""

import yaml
import sys
import os
from pathlib import Path


def check_yaml_file(file_path):
    """Check a single YAML file for syntax errors"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for trailing spaces
        lines = content.split('\n')
        trailing_spaces = []
        for i, line in enumerate(lines, 1):
            if line.endswith(' '):
                trailing_spaces.append(i)
        
        if trailing_spaces:
            print(f"⚠️  {file_path}: Trailing spaces found on lines: {trailing_spaces}")
        
        # Parse YAML
        yaml.safe_load(content)
        print(f"✅ {file_path}: Valid YAML")
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ {file_path}: YAML syntax error")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path}: Error reading file")
        print(f"   Error: {e}")
        return False


def check_workflow_files():
    """Check all GitHub Actions workflow files"""
    workflow_dir = Path('.github/workflows')
    
    if not workflow_dir.exists():
        print("❌ .github/workflows directory not found")
        return False
    
    yaml_files = list(workflow_dir.glob('*.yml')) + list(workflow_dir.glob('*.yaml'))
    
    if not yaml_files:
        print("❌ No YAML workflow files found")
        return False
    
    print(f"🔍 Checking {len(yaml_files)} workflow files...")
    print()
    
    all_valid = True
    for yaml_file in yaml_files:
        if not check_yaml_file(yaml_file):
            all_valid = False
        print()
    
    return all_valid


def check_specific_file(file_path):
    """Check a specific YAML file"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    return check_yaml_file(file_path)


def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Check specific file
        file_path = sys.argv[1]
        success = check_specific_file(file_path)
    else:
        # Check all workflow files
        success = check_workflow_files()
    
    if success:
        print("🎉 All YAML files are valid!")
        sys.exit(0)
    else:
        print("❌ Some YAML files have errors!")
        sys.exit(1)


if __name__ == "__main__":
    main() 