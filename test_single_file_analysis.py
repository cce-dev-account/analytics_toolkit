#!/usr/bin/env python3
"""Test single file analysis to debug the enhanced analyzer."""

from pathlib import Path
from enhanced_hierarchical_analysis import EnhancedCodeAnalyzer

def test_single_file():
    """Test analysis of a single Python file."""
    file_path = Path("src/analytics_toolkit/pytorch_regression/linear.py")

    print(f"Testing analysis of: {file_path}")
    print(f"File exists: {file_path.exists()}")

    if file_path.exists():
        analyzer = EnhancedCodeAnalyzer()
        result = analyzer.analyze_file_detailed(file_path)

        print("\nAnalysis result:")
        for key, value in result.items():
            if isinstance(value, list) and len(value) > 0:
                print(f"  {key}: {len(value)} items")
                if key == 'classes':
                    for cls in value:
                        print(f"    Class: {cls['name']} ({len(cls['methods'])} methods)")
                elif key == 'functions':
                    for func in value:
                        print(f"    Function: {func['name']} ({func['complexity']} complexity)")
            else:
                print(f"  {key}: {value}")
    else:
        print("File not found!")

if __name__ == "__main__":
    test_single_file()