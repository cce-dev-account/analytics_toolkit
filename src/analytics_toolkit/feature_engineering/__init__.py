"""
Advanced Feature Engineering Toolkit.

This module provides sophisticated feature engineering capabilities including:
- Advanced categorical encoding (target encoding, Bayesian encoding)
- Automated feature selection with multiple algorithms
- Interaction detection and generation
- Custom sklearn-compatible transformers
- Time-based feature extraction
- Statistical feature engineering

Key Components:
- transformers: Custom sklearn-compatible transformers
- selection: Advanced feature selection methods
- interactions: Automated interaction detection and generation
- encoding: Advanced categorical encoding techniques
- temporal: Time-based feature engineering
"""

from .encoding import (
    BayesianTargetEncoder,
    FrequencyEncoder,
    OrdinalEncoderAdvanced,
    RareClassEncoder,
    TargetEncoder,
)
from .interactions import (
    InteractionDetector,
    InteractionGenerator,
    PolynomialInteractions,
)
from .selection import (
    CorrelationFilter,
    FeatureSelector,
    MutualInfoSelector,
    RecursiveFeatureElimination,
    VarianceThresholdAdvanced,
)
from .temporal import (
    DateTimeFeatures,
    LagFeatures,
    RollingFeatures,
    SeasonalDecomposition,
)
from .transformers import (
    BinningTransformer,
    LogTransformer,
    OutlierCapTransformer,
    PolynomialFeaturesAdvanced,
    RobustScaler,
)

__all__ = [
    # Transformers
    "LogTransformer",
    "OutlierCapTransformer",
    "BinningTransformer",
    "PolynomialFeaturesAdvanced",
    "RobustScaler",
    # Feature Selection
    "FeatureSelector",
    "MutualInfoSelector",
    "VarianceThresholdAdvanced",
    "CorrelationFilter",
    "RecursiveFeatureElimination",
    # Interactions
    "InteractionDetector",
    "InteractionGenerator",
    "PolynomialInteractions",
    # Encoding
    "TargetEncoder",
    "BayesianTargetEncoder",
    "FrequencyEncoder",
    "RareClassEncoder",
    "OrdinalEncoderAdvanced",
    # Temporal
    "DateTimeFeatures",
    "LagFeatures",
    "RollingFeatures",
    "SeasonalDecomposition",
]
