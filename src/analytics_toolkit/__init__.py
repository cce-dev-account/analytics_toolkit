"""Analytics Toolkit - A comprehensive Python toolkit for data analytics and machine learning."""

__version__ = "0.1.0"
__author__ = "Analytics Team"
__description__ = "Python Analytics Toolkit with PyTorch"

from . import pytorch_regression
from .models import *
from .preprocessing import *
from .utils import *

__all__ = [
    "utils",
    "models",
    "preprocessing",
    "pytorch_regression",
]
