#!/usr/bin/env python3
"""
Enhanced Hierarchical Codebase Analysis Tool
=============================================

Analyzes codebase structure with configurable depth levels:
- Level 0: Root directory
- Level 1-3: Directory hierarchy
- Level 4: Individual Python files
- Level 5: Classes within files
- Level 6: Functions/methods within classes
- Level 7: Function parameters and complexity details

Usage:
    python enhanced_hierarchical_analysis.py [path] [--max-depth=5] [--include-functions] [--include-methods]
"""

import ast
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import re


@dataclass
class EnhancedHierarchyMetrics:
    """Enhanced metrics for any hierarchy level."""
    level: int
    path: str
    relative_path: str
    name: str  # Just the name (file, class, function)
    type: str  # 'directory', 'file', 'class', 'function', 'method'
    parent_path: str = ""

    # File/Directory metrics
    files_count: int = 0
    python_files: int = 0
    subdirectories: List[str] = field(default_factory=list)

    # Code metrics
    total_loc: int = 0
    code_loc: int = 0
    comment_loc: int = 0
    blank_loc: int = 0

    # Structural metrics
    classes_count: int = 0
    functions_count: int = 0
    methods_count: int = 0
    parameters_count: int = 0

    # Complexity metrics
    complexity: int = 0
    avg_complexity: float = 0.0
    max_complexity: int = 0

    # Import and dependency metrics
    imports_count: int = 0
    dependencies: List[str] = field(default_factory=list)

    # Classification and description
    domain_classification: str = ""
    semantic_description: str = ""
    docstring: str = ""

    # File type information
    file_types: Dict[str, int] = field(default_factory=dict)
    key_files: List[str] = field(default_factory=list)

    # Function/method specific
    return_type: str = ""
    parameters: List[Dict] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_property: bool = False
    visibility: str = "public"  # public, private, protected


class EnhancedCodeAnalyzer:
    """Enhanced analyzer for Python code with deep inspection."""

    @staticmethod
    def analyze_file_detailed(file_path: Path) -> Dict[str, Any]:
        """Perform detailed analysis of a Python file."""
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

                # Extract detailed information
                classes = EnhancedCodeAnalyzer._extract_classes(tree)
                functions = EnhancedCodeAnalyzer._extract_functions(tree)
                imports = EnhancedCodeAnalyzer._extract_imports(tree)

                # Calculate complexity
                complexity = EnhancedCodeAnalyzer._calculate_detailed_complexity(tree)

                # Extract module docstring
                docstring = ast.get_docstring(tree) or ""

                return {
                    'total_lines': total_lines,
                    'code_lines': code_lines,
                    'comment_lines': comment_lines,
                    'blank_lines': blank_lines,
                    'classes': classes,
                    'functions': functions,
                    'imports': imports,
                    'complexity': complexity,
                    'docstring': docstring[:200] + "..." if len(docstring) > 200 else docstring
                }
            except SyntaxError as e:
                return EnhancedCodeAnalyzer._create_basic_metrics(total_lines, code_lines, comment_lines, blank_lines, str(e))

        except Exception as e:
            return EnhancedCodeAnalyzer._create_basic_metrics(0, 0, 0, 0, str(e))

    @staticmethod
    def _create_basic_metrics(total_lines: int, code_lines: int, comment_lines: int, blank_lines: int, error: str = "") -> Dict:
        """Create basic metrics when detailed analysis fails."""
        return {
            'total_lines': total_lines, 'code_lines': code_lines,
            'comment_lines': comment_lines, 'blank_lines': blank_lines,
            'classes': [], 'functions': [], 'imports': [],
            'complexity': 0, 'docstring': f"Analysis error: {error}" if error else ""
        }

    @staticmethod
    def _extract_classes(tree: ast.AST) -> List[Dict]:
        """Extract detailed class information."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                properties = [n for n in node.body if isinstance(n, ast.FunctionDef)
                             and any(isinstance(d, ast.Name) and d.id == 'property' for d in n.decorator_list)]

                class_info = {
                    'name': node.name,
                    'lineno': node.lineno,
                    'docstring': ast.get_docstring(node) or "",
                    'methods': [EnhancedCodeAnalyzer._extract_function_details(m, is_method=True) for m in methods],
                    'properties': len(properties),
                    'bases': [EnhancedCodeAnalyzer._extract_name(base) for base in node.bases],
                    'decorators': [EnhancedCodeAnalyzer._extract_name(d) for d in node.decorator_list],
                    'complexity': EnhancedCodeAnalyzer._calculate_node_complexity(node)
                }
                classes.append(class_info)
        return classes

    @staticmethod
    def _extract_functions(tree: ast.AST) -> List[Dict]:
        """Extract detailed function information."""
        functions = []

        # Get top-level functions (not methods)
        if hasattr(tree, 'body'):
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    functions.append(EnhancedCodeAnalyzer._extract_function_details(node))

        return functions

    @staticmethod
    def _extract_function_details(node: ast.FunctionDef, is_method: bool = False) -> Dict:
        """Extract detailed function/method information."""
        # Extract parameters
        args = []
        if node.args:
            # Regular arguments
            for arg in node.args.args:
                arg_info = {
                    'name': arg.arg,
                    'type': EnhancedCodeAnalyzer._extract_annotation(arg.annotation) if arg.annotation else None,
                    'kind': 'regular'
                }
                args.append(arg_info)

            # Default arguments
            defaults_offset = len(node.args.args) - len(node.args.defaults)
            for i, default in enumerate(node.args.defaults):
                if defaults_offset + i < len(args):
                    args[defaults_offset + i]['default'] = EnhancedCodeAnalyzer._extract_default_value(default)

            # *args
            if node.args.vararg:
                args.append({
                    'name': node.args.vararg.arg,
                    'type': EnhancedCodeAnalyzer._extract_annotation(node.args.vararg.annotation) if node.args.vararg.annotation else None,
                    'kind': 'vararg'
                })

            # **kwargs
            if node.args.kwarg:
                args.append({
                    'name': node.args.kwarg.arg,
                    'type': EnhancedCodeAnalyzer._extract_annotation(node.args.kwarg.annotation) if node.args.kwarg.annotation else None,
                    'kind': 'kwarg'
                })

        # Determine visibility
        visibility = "private" if node.name.startswith('_') else "public"
        if node.name.startswith('__') and not node.name.endswith('__'):
            visibility = "protected"

        return {
            'name': node.name,
            'lineno': node.lineno,
            'is_method': is_method,
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'visibility': visibility,
            'docstring': ast.get_docstring(node) or "",
            'parameters': args,
            'return_type': EnhancedCodeAnalyzer._extract_annotation(node.returns) if node.returns else None,
            'decorators': [EnhancedCodeAnalyzer._extract_name(d) for d in node.decorator_list],
            'complexity': EnhancedCodeAnalyzer._calculate_node_complexity(node),
            'loc': EnhancedCodeAnalyzer._count_node_lines(node)
        }

    @staticmethod
    def _extract_imports(tree: ast.AST) -> List[Dict]:
        """Extract import information."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'alias': alias.asname,
                        'type': 'import',
                        'lineno': node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        'module': f"{module}.{alias.name}" if module else alias.name,
                        'alias': alias.asname,
                        'type': 'from_import',
                        'lineno': node.lineno
                    })
        return imports

    @staticmethod
    def _extract_name(node) -> str:
        """Extract name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{EnhancedCodeAnalyzer._extract_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(node)

    @staticmethod
    def _extract_annotation(annotation) -> str:
        """Extract type annotation as string."""
        if annotation is None:
            return ""
        return EnhancedCodeAnalyzer._extract_name(annotation)

    @staticmethod
    def _extract_default_value(default) -> str:
        """Extract default value as string."""
        if isinstance(default, ast.Constant):
            return repr(default.value)
        else:
            return "..."

    @staticmethod
    def _calculate_detailed_complexity(tree: ast.AST) -> int:
        """Calculate detailed cyclomatic complexity."""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                               ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.ListComp):
                complexity += 1

        return complexity

    @staticmethod
    def _calculate_node_complexity(node: ast.AST) -> int:
        """Calculate complexity for a specific node."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                ast.With, ast.AsyncWith, ast.ExceptHandler,
                                ast.And, ast.Or)):
                complexity += 1
        return complexity

    @staticmethod
    def _count_node_lines(node: ast.AST) -> int:
        """Count lines of code in a node."""
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
            return (node.end_lineno or node.lineno) - node.lineno + 1
        return 1


class EnhancedSemanticAnalyzer:
    """Enhanced semantic analysis with deeper understanding."""

    DOMAIN_KEYWORDS = {
        'data': ['data', 'dataset', 'csv', 'json', 'database', 'db', 'sql', 'etl', 'warehouse', 'load', 'save', 'read', 'write', 'import', 'export'],
        'ml': ['model', 'train', 'predict', 'ml', 'machine', 'learning', 'algorithm', 'neural', 'deep', 'fit', 'score', 'accuracy', 'loss', 'optimizer'],
        'stats': ['stats', 'statistics', 'regression', 'analysis', 'statistical', 'hypothesis', 'inference', 'pvalue', 'confidence', 'correlation', 'covariance', 'std', 'mean', 'variance'],
        'feature': ['feature', 'transform', 'encoder', 'preprocessing', 'engineering', 'scaling', 'normalization', 'scaler', 'encode', 'decode', 'categorical', 'numerical'],
        'automl': ['automl', 'auto', 'pipeline', 'optimization', 'hyperparameter', 'tuning', 'optuna', 'bayesian', 'grid', 'random', 'search', 'optimize'],
        'visualization': ['viz', 'plot', 'chart', 'graph', 'visual', 'dashboard', 'streamlit', 'plotly', 'matplotlib', 'seaborn', 'figure', 'axis', 'color', 'legend'],
        'utils': ['util', 'helper', 'common', 'base', 'core', 'tools', 'validate', 'check', 'ensure', 'convert', 'format'],
        'test': ['test', 'spec', 'mock', 'fixture', 'pytest', 'unittest', 'assert', 'should', 'expect', 'verify'],
        'ui': ['ui', 'web', 'streamlit', 'interface', 'frontend', 'app', 'page', 'button', 'input', 'form', 'widget', 'component'],
        'config': ['config', 'setting', 'env', 'constant', 'parameter', 'option', 'default', 'init', 'setup'],
        'api': ['api', 'endpoint', 'route', 'service', 'client', 'server', 'request', 'response', 'http', 'rest'],
        'security': ['auth', 'security', 'token', 'credential', 'permission', 'access', 'login', 'password', 'encrypt'],
        'math': ['calculate', 'compute', 'math', 'matrix', 'tensor', 'array', 'vector', 'algebra', 'equation', 'solve'],
        'performance': ['optimize', 'performance', 'speed', 'memory', 'cache', 'parallel', 'async', 'efficient']
    }

    FUNCTION_PATTERNS = {
        'initialization': ['__init__', 'init', 'setup', 'configure', 'create'],
        'validation': ['validate', 'check', 'verify', 'ensure', 'assert', 'test'],
        'transformation': ['transform', 'convert', 'encode', 'decode', 'scale', 'normalize', 'process'],
        'computation': ['compute', 'calculate', 'evaluate', 'solve', 'optimize'],
        'data_access': ['get', 'set', 'load', 'save', 'read', 'write', 'fetch', 'store'],
        'analysis': ['analyze', 'summarize', 'report', 'profile', 'describe', 'explain'],
        'prediction': ['predict', 'forecast', 'estimate', 'infer', 'classify'],
        'training': ['fit', 'train', 'learn', 'update', 'adjust'],
        'visualization': ['plot', 'draw', 'render', 'display', 'show', 'visualize'],
        'utility': ['helper', 'util', 'format', 'parse', 'clean', 'prepare']
    }

    @staticmethod
    def classify_element(name: str, content: str, element_type: str) -> str:
        """Classify any code element (file, class, function)."""
        combined_text = f"{name} {content}".lower()

        scores = {}
        for domain, keywords in EnhancedSemanticAnalyzer.DOMAIN_KEYWORDS.items():
            score = sum(2 if keyword in name.lower() else 1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'general'

    @staticmethod
    def classify_function_purpose(func_name: str, docstring: str) -> str:
        """Classify the purpose of a function based on name and docstring."""
        combined = f"{func_name} {docstring}".lower()

        for purpose, keywords in EnhancedSemanticAnalyzer.FUNCTION_PATTERNS.items():
            if any(keyword in combined for keyword in keywords):
                return purpose
        return 'general'

    @staticmethod
    def generate_description(metrics: EnhancedHierarchyMetrics) -> str:
        """Generate semantic description based on element type and metrics."""
        if metrics.type == 'directory':
            return EnhancedSemanticAnalyzer._describe_directory(metrics)
        elif metrics.type == 'file':
            return EnhancedSemanticAnalyzer._describe_file(metrics)
        elif metrics.type == 'class':
            return EnhancedSemanticAnalyzer._describe_class(metrics)
        elif metrics.type in ['function', 'method']:
            return EnhancedSemanticAnalyzer._describe_function(metrics)
        else:
            return f"{metrics.type.title()} element with {metrics.complexity} complexity"

    @staticmethod
    def _describe_directory(metrics: EnhancedHierarchyMetrics) -> str:
        """Generate directory description."""
        if metrics.level == 0:
            complexity_level = "high" if metrics.avg_complexity > 20 else "moderate" if metrics.avg_complexity > 10 else "low"
            return f"Root {metrics.domain_classification} project: {metrics.files_count} files, {metrics.code_loc} LOC, {complexity_level} complexity ({metrics.avg_complexity:.1f})"

        # Specific descriptions based on domain and content
        specific_descriptions = {
            ('pytorch_regression', 'stats'): f"Statistical regression module: {metrics.classes_count} model classes, {metrics.methods_count} methods, advanced inference capabilities",
            ('feature_engineering', 'feature'): f"Feature engineering toolkit: {metrics.classes_count} transformers, {metrics.methods_count} methods, ML preprocessing",
            ('automl', 'automl'): f"AutoML pipeline: {metrics.classes_count} components, {metrics.methods_count} methods, hyperparameter optimization",
            ('visualization', 'visualization'): f"Data visualization suite: {metrics.classes_count} plotters, {metrics.methods_count} methods, interactive charts",
            ('tests', 'test'): f"Test suite: {metrics.python_files} test modules, {metrics.functions_count} test cases, {metrics.code_loc} LOC coverage",
            ('examples', 'feature'): f"Usage examples: {metrics.python_files} demo scripts, {metrics.functions_count} examples, {metrics.code_loc} LOC demos",
            ('ui', 'ui'): f"Web interface: {metrics.python_files} components, {metrics.functions_count} handlers, Streamlit-based dashboard"
        }

        key = (metrics.name.lower(), metrics.domain_classification.lower())
        if key in specific_descriptions:
            return specific_descriptions[key]

        # Generic but informative fallback
        if metrics.python_files > 0:
            complexity_desc = f"high-complexity" if metrics.avg_complexity > 15 else f"moderate-complexity" if metrics.avg_complexity > 5 else f"simple"
            return f"{metrics.domain_classification.title()} module: {metrics.python_files} files, {metrics.classes_count}+{metrics.functions_count} components, {complexity_desc} ({metrics.avg_complexity:.1f})"
        else:
            return f"{metrics.domain_classification.title()} directory: {metrics.files_count} files, non-Python content"

    @staticmethod
    def _describe_file(metrics: EnhancedHierarchyMetrics) -> str:
        """Generate file description."""
        if not metrics.name.endswith('.py'):
            return f"{metrics.name} file ({metrics.total_loc} lines)"

        # Specific file descriptions based on name and content
        specific_files = {
            'linear.py': f"Linear regression implementation: OLS with full statistical inference ({metrics.code_loc} LOC, {metrics.methods_count} methods)",
            'advanced.py': f"Advanced regression models: Polynomial, Robust, Regularization Path ({metrics.code_loc} LOC, {metrics.classes_count} classes)",
            'base.py': f"Base regression class: Core functionality and statistical computations ({metrics.code_loc} LOC, {metrics.methods_count} methods)",
            'logistic.py': f"Logistic regression: Binary/multiclass classification with inference ({metrics.code_loc} LOC, {metrics.methods_count} methods)",
            '__init__.py': f"Module initialization: Exports and package structure ({metrics.code_loc} LOC)",
            'transforms.py': f"Feature transformers: B-splines, Fourier, Radial Basis Functions ({metrics.code_loc} LOC, {metrics.classes_count} classes)",
            'stats.py': f"Statistical functions: Inference calculations and model diagnostics ({metrics.code_loc} LOC, {metrics.functions_count} functions)",
            'utils.py': f"Utility functions: Helper methods and data validation ({metrics.code_loc} LOC, {metrics.functions_count} functions)"
        }

        if metrics.name in specific_files:
            return specific_files[metrics.name]

        # Generic but informative descriptions
        if metrics.classes_count > 0 and metrics.methods_count > 0:
            architecture = "object-oriented" if metrics.classes_count > metrics.functions_count else "mixed"
            complexity_desc = "complex" if metrics.avg_complexity > 10 else "moderate" if metrics.avg_complexity > 5 else "simple"
            return f"{metrics.domain_classification.title()} module ({architecture}): {metrics.classes_count} classes, {metrics.methods_count} methods, {complexity_desc} logic ({metrics.code_loc} LOC)"
        elif metrics.functions_count > 0:
            complexity_desc = "complex" if metrics.avg_complexity > 8 else "moderate" if metrics.avg_complexity > 4 else "simple"
            return f"{metrics.domain_classification.title()} module (functional): {metrics.functions_count} functions, {complexity_desc} logic ({metrics.code_loc} LOC)"
        else:
            return f"{metrics.domain_classification.title()} module: Configuration/constants ({metrics.code_loc} LOC)"

    @staticmethod
    def _describe_class(metrics: EnhancedHierarchyMetrics) -> str:
        """Generate class description."""
        visibility = "Private" if metrics.name.startswith('_') else "Public"

        # Specific class descriptions
        specific_classes = {
            'LinearRegression': f"OLS linear regression: Statistical inference, confidence intervals, R-squared ({metrics.methods_count} methods)",
            'BaseRegression': f"Abstract regression base: Common functionality for all regression models ({metrics.methods_count} methods)",
            'LogisticRegression': f"Logistic regression: Binary/multiclass classification with statistical inference ({metrics.methods_count} methods)",
            'PolynomialRegression': f"Polynomial regression: Non-linear relationships with feature interactions ({metrics.methods_count} methods)",
            'RobustRegression': f"Robust regression: Outlier-resistant modeling with Huber loss ({metrics.methods_count} methods)",
            'RegularizationPath': f"Regularization path: L1/L2/ElasticNet with cross-validation ({metrics.methods_count} methods)",
            'TargetEncoder': f"Target encoding: Mean target encoding with cross-validation regularization ({metrics.methods_count} methods)",
            'FeatureSelector': f"Feature selection: Multi-algorithm wrapper for optimal feature subsets ({metrics.methods_count} methods)",
            'AutoMLPipeline': f"AutoML pipeline: Automated model selection and hyperparameter optimization ({metrics.methods_count} methods)"
        }

        if metrics.name in specific_classes:
            return specific_classes[metrics.name]

        complexity_desc = "high-complexity" if metrics.complexity > 15 else "moderate-complexity" if metrics.complexity > 8 else "simple"
        return f"{visibility} {metrics.domain_classification} class: {metrics.methods_count} methods, {complexity_desc} implementation (complexity: {metrics.complexity})"

    @staticmethod
    def _describe_function(metrics: EnhancedHierarchyMetrics) -> str:
        """Generate function description."""
        visibility = "Private" if metrics.name.startswith('_') else "Public"
        func_type = "Async function" if metrics.is_async else ("Method" if metrics.type == 'method' else "Function")

        # Classify function purpose
        purpose = EnhancedSemanticAnalyzer.classify_function_purpose(metrics.name, metrics.docstring)

        # Specific function descriptions based on common patterns
        specific_functions = {
            '__init__': f"Constructor: Initialize {metrics.domain_classification} instance with {metrics.parameters_count} parameters",
            'fit': f"Model training: Fit {metrics.domain_classification} model to training data (complexity: {metrics.complexity})",
            'predict': f"Prediction: Generate predictions from fitted model (complexity: {metrics.complexity})",
            'transform': f"Data transformation: Apply {metrics.domain_classification} transformation (complexity: {metrics.complexity})",
            'score': f"Model evaluation: Compute performance score on test data (complexity: {metrics.complexity})",
            '_compute_statistics': f"Statistical computation: Calculate model statistics and inference metrics (complexity: {metrics.complexity})",
            '_validate_input': f"Input validation: Ensure data format and constraints are met (complexity: {metrics.complexity})",
            'summary': f"Model summary: Generate comprehensive statistical summary report (complexity: {metrics.complexity})"
        }

        if metrics.name in specific_functions:
            return specific_functions[metrics.name]

        # Purpose-based descriptions
        purpose_descriptions = {
            'initialization': f"{visibility} {func_type.lower()}: Initialize/configure {metrics.domain_classification} component",
            'validation': f"{visibility} {func_type.lower()}: Validate/verify {metrics.domain_classification} inputs/state",
            'transformation': f"{visibility} {func_type.lower()}: Transform/process {metrics.domain_classification} data",
            'computation': f"{visibility} {func_type.lower()}: Compute/calculate {metrics.domain_classification} results",
            'data_access': f"{visibility} {func_type.lower()}: Access/manipulate {metrics.domain_classification} data",
            'analysis': f"{visibility} {func_type.lower()}: Analyze/summarize {metrics.domain_classification} information",
            'prediction': f"{visibility} {func_type.lower()}: Generate {metrics.domain_classification} predictions",
            'training': f"{visibility} {func_type.lower()}: Train/fit {metrics.domain_classification} model",
            'visualization': f"{visibility} {func_type.lower()}: Create {metrics.domain_classification} visualizations",
            'utility': f"{visibility} {func_type.lower()}: Utility/helper for {metrics.domain_classification} operations"
        }

        if purpose in purpose_descriptions:
            base_desc = purpose_descriptions[purpose]
        else:
            base_desc = f"{visibility} {func_type.lower()}: {metrics.domain_classification} operation"

        complexity_note = f" (high complexity: {metrics.complexity})" if metrics.complexity > 8 else f" (complexity: {metrics.complexity})"
        params_note = f", {metrics.parameters_count} params" if metrics.parameters_count > 3 else ""

        return f"{base_desc}{complexity_note}{params_note}"


class EnhancedHierarchicalAnalyzer:
    """Enhanced analyzer with configurable depth."""

    def __init__(self, root_path: str, max_depth: int = 3, include_functions: bool = False, include_methods: bool = False):
        self.root_path = Path(root_path).resolve()
        self.max_depth = max_depth
        self.include_functions = include_functions
        self.include_methods = include_methods
        self.code_analyzer = EnhancedCodeAnalyzer()
        self.semantic_analyzer = EnhancedSemanticAnalyzer()

    def analyze(self) -> List[EnhancedHierarchyMetrics]:
        """Perform enhanced hierarchical analysis."""
        hierarchy_data = []

        # Directory analysis (levels 0-3)
        hierarchy_data.extend(self._analyze_directories())

        # File analysis (level 4)
        if self.max_depth >= 4:
            hierarchy_data.extend(self._analyze_files())

        # Class analysis (level 5)
        if self.max_depth >= 5:
            hierarchy_data.extend(self._analyze_classes())

        # Function/Method analysis (level 6+)
        if self.max_depth >= 6 and (self.include_functions or self.include_methods):
            hierarchy_data.extend(self._analyze_functions())

        return hierarchy_data

    def _analyze_directories(self) -> List[EnhancedHierarchyMetrics]:
        """Analyze directory structure (levels 0-3)."""
        directory_data = []

        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'node_modules'}]

            relative_path = root_path.relative_to(self.root_path)
            level = len(relative_path.parts) if str(relative_path) != '.' else 0

            if level > min(3, self.max_depth):
                continue

            metrics = EnhancedHierarchyMetrics(
                level=level,
                path=str(root_path),
                relative_path=str(relative_path) if str(relative_path) != '.' else '',
                name=relative_path.name if level > 0 else 'ROOT',
                type='directory',
                subdirectories=dirs.copy()
            )

            self._analyze_directory_files_enhanced(metrics, files, root_path)

            # Semantic analysis
            path_parts = list(relative_path.parts) if str(relative_path) != '.' else []
            content = ' '.join(path_parts + files)
            metrics.domain_classification = self.semantic_analyzer.classify_element(
                metrics.name, content, 'directory'
            )
            metrics.semantic_description = self.semantic_analyzer.generate_description(metrics)

            directory_data.append(metrics)

        return directory_data

    def _analyze_files(self) -> List[EnhancedHierarchyMetrics]:
        """Analyze individual Python files (level 4)."""
        file_data = []

        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'node_modules'}]

            relative_path = root_path.relative_to(self.root_path)
            dir_level = len(relative_path.parts) if str(relative_path) != '.' else 0

            if dir_level > 3:  # Only process files in directories up to level 3
                continue

            python_files = [f for f in files if f.endswith('.py')]

            for py_file in python_files:
                file_path = root_path / py_file
                relative_file_path = file_path.relative_to(self.root_path)

                metrics = EnhancedHierarchyMetrics(
                    level=4,
                    path=str(file_path),
                    relative_path=str(relative_file_path),
                    name=py_file,
                    type='file',
                    parent_path=str(relative_path) if str(relative_path) != '.' else ''
                )

                # Analyze file content
                analysis = self.code_analyzer.analyze_file_detailed(file_path)

                metrics.total_loc = analysis['total_lines']
                metrics.code_loc = analysis['code_lines']
                metrics.comment_loc = analysis['comment_lines']
                metrics.blank_loc = analysis['blank_lines']
                metrics.classes_count = len(analysis['classes'])
                metrics.functions_count = len(analysis['functions'])
                metrics.methods_count = sum(len(cls['methods']) for cls in analysis['classes'])
                metrics.imports_count = len(analysis['imports'])
                metrics.complexity = analysis['complexity']
                metrics.docstring = analysis['docstring']

                # Calculate averages
                all_complexities = []
                all_complexities.extend([cls['complexity'] for cls in analysis['classes']])
                all_complexities.extend([func['complexity'] for func in analysis['functions']])
                for cls in analysis['classes']:
                    all_complexities.extend([method['complexity'] for method in cls['methods']])

                if all_complexities:
                    metrics.avg_complexity = sum(all_complexities) / len(all_complexities)
                    metrics.max_complexity = max(all_complexities)

                # Semantic analysis
                metrics.domain_classification = self.semantic_analyzer.classify_element(
                    py_file, analysis['docstring'], 'file'
                )
                metrics.semantic_description = self.semantic_analyzer.generate_description(metrics)

                file_data.append(metrics)

        return file_data

    def _analyze_classes(self) -> List[EnhancedHierarchyMetrics]:
        """Analyze classes within files (level 5)."""
        class_data = []

        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'node_modules'}]

            relative_path = root_path.relative_to(self.root_path)
            dir_level = len(relative_path.parts) if str(relative_path) != '.' else 0

            if dir_level > 3:
                continue

            python_files = [f for f in files if f.endswith('.py')]

            for py_file in python_files:
                file_path = root_path / py_file
                relative_file_path = file_path.relative_to(self.root_path)

                analysis = self.code_analyzer.analyze_file_detailed(file_path)

                for class_info in analysis['classes']:
                    metrics = EnhancedHierarchyMetrics(
                        level=5,
                        path=f"{file_path}:{class_info['lineno']}",
                        relative_path=f"{relative_file_path}:{class_info['name']}",
                        name=class_info['name'],
                        type='class',
                        parent_path=str(relative_file_path)
                    )

                    metrics.methods_count = len(class_info['methods'])
                    metrics.complexity = class_info['complexity']
                    metrics.docstring = class_info['docstring'][:200] + "..." if len(class_info['docstring']) > 200 else class_info['docstring']

                    # Calculate method complexities
                    method_complexities = [method['complexity'] for method in class_info['methods']]
                    if method_complexities:
                        metrics.avg_complexity = sum(method_complexities) / len(method_complexities)
                        metrics.max_complexity = max(method_complexities)

                    # Count parameters across all methods
                    metrics.parameters_count = sum(len(method['parameters']) for method in class_info['methods'])

                    # Semantic analysis
                    metrics.domain_classification = self.semantic_analyzer.classify_element(
                        class_info['name'], class_info['docstring'], 'class'
                    )
                    metrics.semantic_description = self.semantic_analyzer.generate_description(metrics)

                    class_data.append(metrics)

        return class_data

    def _analyze_functions(self) -> List[EnhancedHierarchyMetrics]:
        """Analyze functions and methods (level 6+)."""
        function_data = []

        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'node_modules'}]

            relative_path = root_path.relative_to(self.root_path)
            dir_level = len(relative_path.parts) if str(relative_path) != '.' else 0

            if dir_level > 3:
                continue

            python_files = [f for f in files if f.endswith('.py')]

            for py_file in python_files:
                file_path = root_path / py_file
                relative_file_path = file_path.relative_to(self.root_path)

                analysis = self.code_analyzer.analyze_file_detailed(file_path)

                # Top-level functions
                if self.include_functions:
                    for func_info in analysis['functions']:
                        function_data.append(self._create_function_metrics(
                            func_info, file_path, relative_file_path, 6, 'function'
                        ))

                # Methods within classes
                if self.include_methods:
                    for class_info in analysis['classes']:
                        for method_info in class_info['methods']:
                            function_data.append(self._create_function_metrics(
                                method_info, file_path, relative_file_path, 6, 'method',
                                parent_class=class_info['name']
                            ))

        return function_data

    def _create_function_metrics(self, func_info: Dict, file_path: Path, relative_file_path: Path,
                               level: int, func_type: str, parent_class: str = "") -> EnhancedHierarchyMetrics:
        """Create metrics for a function or method."""
        parent_path = f"{relative_file_path}:{parent_class}" if parent_class else str(relative_file_path)

        metrics = EnhancedHierarchyMetrics(
            level=level,
            path=f"{file_path}:{func_info['lineno']}",
            relative_path=f"{relative_file_path}:{func_info['name']}",
            name=func_info['name'],
            type=func_type,
            parent_path=parent_path
        )

        metrics.parameters_count = len(func_info['parameters'])
        metrics.complexity = func_info['complexity']
        metrics.code_loc = func_info.get('loc', 0)
        metrics.docstring = func_info['docstring'][:200] + "..." if len(func_info['docstring']) > 200 else func_info['docstring']
        metrics.is_async = func_info['is_async']
        metrics.visibility = func_info['visibility']
        metrics.parameters = func_info['parameters']
        metrics.return_type = func_info['return_type'] or ""
        metrics.decorators = func_info['decorators']

        # Semantic analysis
        metrics.domain_classification = self.semantic_analyzer.classify_element(
            func_info['name'], func_info['docstring'], func_type
        )
        metrics.semantic_description = self.semantic_analyzer.generate_description(metrics)

        return metrics

    def _analyze_directory_files_enhanced(self, metrics: EnhancedHierarchyMetrics, files: List[str], root_path: Path):
        """Analyze files within a directory for directory-level metrics."""
        metrics.files_count = len(files)

        # Count file types
        for file in files:
            ext = Path(file).suffix.lower()
            metrics.file_types[ext] = metrics.file_types.get(ext, 0) + 1

        # Analyze Python files
        python_files = [f for f in files if f.endswith('.py')]
        metrics.python_files = len(python_files)

        total_complexity = []
        important_files = []

        for py_file in python_files:
            file_path = root_path / py_file
            analysis = self.code_analyzer.analyze_file_detailed(file_path)

            metrics.total_loc += analysis['total_lines']
            metrics.code_loc += analysis['code_lines']
            metrics.comment_loc += analysis['comment_lines']
            metrics.blank_loc += analysis['blank_lines']
            metrics.functions_count += len(analysis['functions'])
            metrics.classes_count += len(analysis['classes'])
            metrics.imports_count += len(analysis['imports'])

            # Count methods in classes
            for cls in analysis['classes']:
                metrics.methods_count += len(cls['methods'])

            if analysis['complexity'] > 0:
                total_complexity.append(analysis['complexity'])

            # Identify important files
            if (len(analysis['classes']) > 3 or len(analysis['functions']) > 10 or
                analysis['code_lines'] > 100 or py_file == '__init__.py'):
                important_files.append(py_file)

        if total_complexity:
            metrics.avg_complexity = sum(total_complexity) / len(total_complexity)
            metrics.max_complexity = max(total_complexity)

        metrics.key_files = important_files[:5]

    def generate_report(self, hierarchy_data: List[EnhancedHierarchyMetrics]) -> pd.DataFrame:
        """Generate enhanced tabular report."""
        rows = []

        for metrics in sorted(hierarchy_data, key=lambda x: (x.level, x.relative_path)):
            # Base columns for all levels
            row = {
                'Level': metrics.level,
                'Type': metrics.type.title(),
                'Name': metrics.name,
                'Path': metrics.relative_path if metrics.relative_path else 'ROOT',
                'Parent': metrics.parent_path,
                'Domain': metrics.domain_classification.title(),
                'Description': metrics.semantic_description[:100] + "..." if len(metrics.semantic_description) > 100 else metrics.semantic_description
            }

            # Add metrics based on type
            if metrics.type == 'directory':
                row.update({
                    'Files': metrics.files_count,
                    'Python_Files': metrics.python_files,
                    'Subdirs': len(metrics.subdirectories),
                    'Total_LOC': metrics.total_loc,
                    'Code_LOC': metrics.code_loc,
                    'Classes': metrics.classes_count,
                    'Functions': metrics.functions_count,
                    'Methods': metrics.methods_count,
                    'Avg_Complexity': round(metrics.avg_complexity, 2),
                    'Max_Complexity': metrics.max_complexity,
                    'Key_Files': ', '.join(metrics.key_files[:2])
                })
            elif metrics.type == 'file':
                row.update({
                    'Files': 1,
                    'Python_Files': 1,
                    'Subdirs': 0,
                    'Total_LOC': metrics.total_loc,
                    'Code_LOC': metrics.code_loc,
                    'Classes': metrics.classes_count,
                    'Functions': metrics.functions_count,
                    'Methods': metrics.methods_count,
                    'Avg_Complexity': round(metrics.avg_complexity, 2),
                    'Max_Complexity': metrics.max_complexity,
                    'Key_Files': metrics.name
                })
            elif metrics.type == 'class':
                row.update({
                    'Files': 0,
                    'Python_Files': 0,
                    'Subdirs': 0,
                    'Total_LOC': 0,
                    'Code_LOC': metrics.code_loc,
                    'Classes': 1,
                    'Functions': 0,
                    'Methods': metrics.methods_count,
                    'Avg_Complexity': round(metrics.avg_complexity, 2),
                    'Max_Complexity': metrics.max_complexity,
                    'Key_Files': f"{metrics.methods_count} methods"
                })
            elif metrics.type in ['function', 'method']:
                row.update({
                    'Files': 0,
                    'Python_Files': 0,
                    'Subdirs': 0,
                    'Total_LOC': 0,
                    'Code_LOC': metrics.code_loc,
                    'Classes': 0,
                    'Functions': 1 if metrics.type == 'function' else 0,
                    'Methods': 1 if metrics.type == 'method' else 0,
                    'Avg_Complexity': metrics.complexity,
                    'Max_Complexity': metrics.complexity,
                    'Key_Files': f"{metrics.parameters_count} params, {metrics.return_type}"
                })

            rows.append(row)

        return pd.DataFrame(rows)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Enhanced Hierarchical Codebase Analysis')
    parser.add_argument('path', nargs='?', default='.', help='Path to analyze (default: current directory)')
    parser.add_argument('--max-depth', type=int, default=4, help='Maximum depth level (default: 4)')
    parser.add_argument('--include-functions', action='store_true', help='Include function-level analysis')
    parser.add_argument('--include-methods', action='store_true', help='Include method-level analysis')
    parser.add_argument('--output', default='enhanced_analysis', help='Output file prefix (default: enhanced_analysis)')

    args = parser.parse_args()

    print(f"[ANALYSIS] Enhanced Hierarchical Analysis of: {Path(args.path).absolute()}")
    print(f"[CONFIG] Max Depth: {args.max_depth}, Functions: {args.include_functions}, Methods: {args.include_methods}")
    print("=" * 100)

    # Perform analysis
    analyzer = EnhancedHierarchicalAnalyzer(
        args.path,
        max_depth=args.max_depth,
        include_functions=args.include_functions,
        include_methods=args.include_methods
    )
    hierarchy_data = analyzer.analyze()

    # Generate report
    df = analyzer.generate_report(hierarchy_data)

    # Display results
    print(f"\n[REPORT] ENHANCED HIERARCHICAL ANALYSIS ({len(hierarchy_data)} elements)")
    print("=" * 100)

    # Configure pandas display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)

    print(df.to_string(index=False))

    # Summary by level
    print(f"\n[SUMMARY] ANALYSIS BY LEVEL")
    print("=" * 50)

    level_summary = df.groupby(['Level', 'Type']).size().unstack(fill_value=0)
    print(level_summary.to_string())

    # Complexity analysis
    complexity_summary = df.groupby('Level').agg({
        'Avg_Complexity': 'mean',
        'Max_Complexity': 'max',
        'Code_LOC': 'sum'
    }).round(2)

    print(f"\n[COMPLEXITY] BY LEVEL")
    print("=" * 30)
    print(complexity_summary.to_string())

    # Save outputs
    csv_path = Path(args.path) / f"{args.output}_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] CSV report: {csv_path}")

    # Save detailed Excel report
    try:
        excel_path = Path(args.path) / f"{args.output}_detailed.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Full_Analysis', index=False)
            level_summary.to_excel(writer, sheet_name='Level_Summary')
            complexity_summary.to_excel(writer, sheet_name='Complexity_Summary')

            # Create separate sheets by level
            for level in sorted(df['Level'].unique()):
                level_df = df[df['Level'] == level]
                sheet_name = f"Level_{level}"
                level_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"[EXCEL] Detailed report: {excel_path}")
    except ImportError:
        print("[WARNING] openpyxl not available, skipping Excel export")


if __name__ == "__main__":
    main()