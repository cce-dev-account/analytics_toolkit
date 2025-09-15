"""
Automated Machine Learning (AutoML) Pipeline Builder.

This module provides intelligent automation for machine learning pipelines including:
- Automated data preprocessing and feature engineering
- Hyperparameter optimization with Optuna integration
- Automated model selection and ensemble building
- Experiment tracking and model registry
- Cross-validation and performance evaluation
- Pipeline serialization and deployment utilities

Key Components:
- pipeline_builder: Automated ML pipeline construction
- hyperparameter_tuning: Intelligent hyperparameter optimization
- model_selection: Automated algorithm selection and comparison
- experiment_tracking: MLflow-style experiment management
- ensemble: Automated ensemble model creation
"""

from .experiment_tracking import (
    ExperimentTracker,
    ModelRegistry,
    RunMetrics,
)
from .hyperparameter_tuning import (
    HyperparameterOptimizer,
    OptimizationConfig,
    OptunaOptimizer,
)
from .model_selection import (
    AutoModelSelector,
    EnsembleBuilder,
    ModelComparison,
)
from .pipeline_builder import (
    AutoMLPipeline,
    DataTypeInference,
    PipelineConfig,
)

__all__ = [
    # Pipeline Builder
    "AutoMLPipeline",
    "PipelineConfig",
    "DataTypeInference",
    # Hyperparameter Tuning
    "HyperparameterOptimizer",
    "OptimizationConfig",
    "OptunaOptimizer",
    # Model Selection
    "AutoModelSelector",
    "ModelComparison",
    "EnsembleBuilder",
    # Experiment Tracking
    "ExperimentTracker",
    "ModelRegistry",
    "RunMetrics",
]
