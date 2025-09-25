#!/usr/bin/env python3
"""
Hierarchical Codebase Analysis Tool
===================================

Analyzes codebase structure by hierarchy levels (0/1/2/3) and provides:
- Directory structure analysis
- Code metrics (LOC, complexity, files)
- Semantic descriptions for each level
- Tabular output for easy analysis

Usage:
    python hierarchical_analysis.py [path]
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import re


@dataclass
class HierarchyMetrics:
    """Metrics for a specific hierarchy level."""
    level: int
    path: str
    relative_path: str
    files_count: int = 0
    python_files: int = 0
    total_loc: int = 0
    code_loc: int = 0
    comment_loc: int = 0
    blank_loc: int = 0
    avg_complexity: float = 0.0
    max_complexity: int = 0
    functions_count: int = 0
    classes_count: int = 0
    imports_count: int = 0
    subdirectories: List[str] = field(default_factory=list)
    file_types: Dict[str, int] = field(default_factory=dict)
    semantic_description: str = ""
    domain_classification: str = ""
    key_files: List[str] = field(default_factory=list)


class CodeAnalyzer:
    """Analyzes Python code for metrics."""

    @staticmethod
    def analyze_file(file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Count lines
            lines = content.split('\n')
            total_lines = len(lines)
            blank_lines = sum(1 for line in lines if not line.strip())
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            code_lines = total_lines - blank_lines - comment_lines

            # Parse AST for detailed analysis
            try:
                tree = ast.parse(content)

                # Count functions and classes
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                imports = [node for node in ast.walk(tree)
                          if isinstance(node, (ast.Import, ast.ImportFrom))]

                # Calculate complexity (simplified cyclomatic complexity)
                complexity = CodeAnalyzer._calculate_complexity(tree)

                return {
                    'total_lines': total_lines,
                    'code_lines': code_lines,
                    'comment_lines': comment_lines,
                    'blank_lines': blank_lines,
                    'functions': len(functions),
                    'classes': len(classes),
                    'imports': len(imports),
                    'complexity': complexity,
                    'function_names': [f.name for f in functions],
                    'class_names': [c.name for c in classes]
                }
            except SyntaxError:
                return {
                    'total_lines': total_lines,
                    'code_lines': code_lines,
                    'comment_lines': comment_lines,
                    'blank_lines': blank_lines,
                    'functions': 0,
                    'classes': 0,
                    'imports': 0,
                    'complexity': 0,
                    'function_names': [],
                    'class_names': []
                }

        except Exception as e:
            return {
                'total_lines': 0, 'code_lines': 0, 'comment_lines': 0,
                'blank_lines': 0, 'functions': 0, 'classes': 0,
                'imports': 0, 'complexity': 0, 'function_names': [],
                'class_names': []
            }

    @staticmethod
    def _calculate_complexity(tree: ast.AST) -> int:
        """Calculate simplified cyclomatic complexity."""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1

        return complexity


class SemanticAnalyzer:
    """Analyzes semantic meaning of code directories and files."""

    DOMAIN_KEYWORDS = {
        'data': ['data', 'dataset', 'csv', 'json', 'database', 'db', 'sql'],
        'ml': ['model', 'train', 'predict', 'ml', 'machine', 'learning', 'algorithm'],
        'stats': ['stats', 'statistics', 'regression', 'analysis', 'statistical'],
        'feature': ['feature', 'transform', 'encoder', 'preprocessing', 'engineering'],
        'automl': ['automl', 'auto', 'pipeline', 'optimization', 'hyperparameter'],
        'visualization': ['viz', 'plot', 'chart', 'graph', 'visual', 'dashboard'],
        'utils': ['util', 'helper', 'common', 'base', 'core'],
        'test': ['test', 'spec', 'mock', 'fixture'],
        'ui': ['ui', 'web', 'streamlit', 'interface', 'frontend'],
        'config': ['config', 'setting', 'env', 'constant']
    }

    @staticmethod
    def classify_domain(path_parts: List[str], files: List[str]) -> str:
        """Classify the domain/purpose of a directory."""
        path_text = ' '.join(path_parts).lower()
        files_text = ' '.join([f.lower() for f in files]).lower()
        combined_text = f"{path_text} {files_text}"

        scores = {}
        for domain, keywords in SemanticAnalyzer.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'general'

    @staticmethod
    def generate_description(metrics: HierarchyMetrics, path_parts: List[str]) -> str:
        """Generate semantic description for a hierarchy level."""
        if metrics.level == 0:
            return f"Root project directory containing {metrics.files_count} files across {len(metrics.subdirectories)} main modules"

        # Analyze path components for meaning
        current_dir = path_parts[-1] if path_parts else ""

        descriptions = {
            'src': "Source code directory containing core implementation",
            'tests': f"Test suite with {metrics.python_files} test files covering {metrics.functions_count} test functions",
            'analytics_toolkit': f"Main package module with {metrics.python_files} Python modules implementing core functionality",
            'pytorch_regression': f"PyTorch-based statistical regression module with {metrics.classes_count} model classes and {metrics.functions_count} statistical functions",
            'feature_engineering': f"Advanced feature engineering toolkit with {metrics.classes_count} transformer classes",
            'automl': f"Automated machine learning pipeline with {metrics.classes_count} components for optimization and model selection",
            'ui': f"User interface components with {metrics.python_files} modules for web interface",
            'pages': f"Streamlit page components with {metrics.python_files} page modules",
            'utils': f"Utility functions and helpers with {metrics.functions_count} utility functions",
            'docs': f"Documentation directory with {metrics.files_count} documentation files",
            'examples': f"Example usage with {metrics.python_files} example scripts"
        }

        if current_dir in descriptions:
            return descriptions[current_dir]

        # Generate dynamic description based on metrics
        if metrics.python_files > 0:
            if metrics.classes_count > metrics.functions_count:
                return f"Object-oriented module with {metrics.classes_count} classes implementing {metrics.domain_classification} functionality"
            else:
                return f"Functional module with {metrics.functions_count} functions for {metrics.domain_classification} operations"
        else:
            return f"Directory containing {metrics.files_count} files for {metrics.domain_classification} purposes"


class HierarchicalAnalyzer:
    """Main analyzer for hierarchical codebase analysis."""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path).resolve()
        self.code_analyzer = CodeAnalyzer()
        self.semantic_analyzer = SemanticAnalyzer()

    def analyze(self) -> List[HierarchyMetrics]:
        """Perform complete hierarchical analysis."""
        hierarchy_data = []

        # Walk through directory structure
        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)

            # Skip hidden directories and common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'node_modules'}]

            # Calculate hierarchy level
            relative_path = root_path.relative_to(self.root_path)
            level = len(relative_path.parts) if str(relative_path) != '.' else 0

            # Only analyze up to level 3
            if level > 3:
                continue

            metrics = HierarchyMetrics(
                level=level,
                path=str(root_path),
                relative_path=str(relative_path) if str(relative_path) != '.' else '',
                subdirectories=dirs.copy()
            )

            # Analyze files
            self._analyze_files(metrics, files, root_path)

            # Generate semantic information
            path_parts = relative_path.parts if str(relative_path) != '.' else []
            metrics.domain_classification = self.semantic_analyzer.classify_domain(
                list(path_parts), files
            )
            metrics.semantic_description = self.semantic_analyzer.generate_description(
                metrics, list(path_parts)
            )

            hierarchy_data.append(metrics)

        return hierarchy_data

    def _analyze_files(self, metrics: HierarchyMetrics, files: List[str], root_path: Path):
        """Analyze all files in a directory."""
        metrics.files_count = len(files)

        # Count file types
        for file in files:
            ext = Path(file).suffix.lower()
            metrics.file_types[ext] = metrics.file_types.get(ext, 0) + 1

        # Analyze Python files specifically
        python_files = [f for f in files if f.endswith('.py')]
        metrics.python_files = len(python_files)

        total_complexity = []
        important_files = []

        for py_file in python_files:
            file_path = root_path / py_file
            analysis = self.code_analyzer.analyze_file(file_path)

            metrics.total_loc += analysis['total_lines']
            metrics.code_loc += analysis['code_lines']
            metrics.comment_loc += analysis['comment_lines']
            metrics.blank_loc += analysis['blank_lines']
            metrics.functions_count += analysis['functions']
            metrics.classes_count += analysis['classes']
            metrics.imports_count += analysis['imports']

            if analysis['complexity'] > 0:
                total_complexity.append(analysis['complexity'])

            # Identify important files
            if (analysis['classes'] > 3 or analysis['functions'] > 10 or
                analysis['code_lines'] > 100 or py_file == '__init__.py'):
                important_files.append(py_file)

        if total_complexity:
            metrics.avg_complexity = sum(total_complexity) / len(total_complexity)
            metrics.max_complexity = max(total_complexity)

        metrics.key_files = important_files[:5]  # Top 5 important files

    def generate_report(self, hierarchy_data: List[HierarchyMetrics]) -> pd.DataFrame:
        """Generate tabular report from hierarchy data."""
        rows = []

        for metrics in sorted(hierarchy_data, key=lambda x: (x.level, x.relative_path)):
            row = {
                'Level': metrics.level,
                'Path': metrics.relative_path if metrics.relative_path else 'ROOT',
                'Domain': metrics.domain_classification.title(),
                'Files': metrics.files_count,
                'Python_Files': metrics.python_files,
                'Total_LOC': metrics.total_loc,
                'Code_LOC': metrics.code_loc,
                'Classes': metrics.classes_count,
                'Functions': metrics.functions_count,
                'Avg_Complexity': round(metrics.avg_complexity, 2),
                'Max_Complexity': metrics.max_complexity,
                'Subdirectories': len(metrics.subdirectories),
                'Key_Files': ', '.join(metrics.key_files[:3]),  # Top 3
                'Description': metrics.semantic_description
            }
            rows.append(row)

        return pd.DataFrame(rows)


def main():
    """Main entry point."""
    # Determine project path
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
    else:
        project_path = Path(__file__).parent

    print(f"[ANALYSIS] Hierarchical Analysis of: {Path(project_path).absolute()}")
    print("=" * 80)

    # Perform analysis
    analyzer = HierarchicalAnalyzer(project_path)
    hierarchy_data = analyzer.analyze()

    # Generate report
    df = analyzer.generate_report(hierarchy_data)

    # Display results
    print("\n[REPORT] HIERARCHICAL CODEBASE ANALYSIS")
    print("=" * 80)

    # Configure pandas display options for better output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    print(df.to_string(index=False))

    # Summary statistics
    print("\n[SUMMARY] HIERARCHY SUMMARY")
    print("=" * 40)

    level_summary = df.groupby('Level').agg({
        'Files': 'sum',
        'Python_Files': 'sum',
        'Code_LOC': 'sum',
        'Classes': 'sum',
        'Functions': 'sum',
        'Avg_Complexity': 'mean'
    }).round(2)

    print(level_summary.to_string())

    # Save to files
    output_path = Path(project_path) / "hierarchical_analysis_report.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[SAVED] Report saved to: {output_path}")

    # Save detailed Excel report
    excel_path = Path(project_path) / "hierarchical_analysis_detailed.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Hierarchy_Analysis', index=False)
        level_summary.to_excel(writer, sheet_name='Level_Summary')

        # Create metrics by level sheet
        metrics_by_level = df.pivot_table(
            index='Level',
            values=['Files', 'Python_Files', 'Code_LOC', 'Classes', 'Functions'],
            aggfunc='sum'
        )
        metrics_by_level.to_excel(writer, sheet_name='Metrics_by_Level')

    print(f"[EXCEL] Detailed Excel report saved to: {excel_path}")


if __name__ == "__main__":
    main()