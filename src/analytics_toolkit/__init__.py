"""Analytics Toolkit - A comprehensive Python toolkit for data analytics and machine learning."""

from typing import Any

__version__ = "0.1.0"
__author__ = "Analytics Team"
__description__ = "Python Analytics Toolkit with PyTorch"

# Safe imports to avoid CI failures
pytorch_regression: Any = None
try:
    from . import pytorch_regression
except ImportError:
    pytorch_regression = None

feature_engineering: Any = None
try:
    from . import feature_engineering
except ImportError:
    feature_engineering = None

automl: Any = None
try:
    from . import automl
except ImportError:
    automl = None

visualization: Any = None
try:
    from . import visualization
except ImportError:
    visualization = None

try:
    from .utils import *
except ImportError:
    pass

try:
    from .models import *
except ImportError:
    pass

try:
    from .preprocessing import *
except ImportError:
    pass

try:
    from .scorecard_integration import ScorecardIntegrator
except ImportError:
    ScorecardIntegrator = None

__all__ = [
    "utils",
    "models",
    "preprocessing",
    "pytorch_regression",
    "feature_engineering",
    "automl",
    "visualization",
    "ScorecardIntegrator",
]
