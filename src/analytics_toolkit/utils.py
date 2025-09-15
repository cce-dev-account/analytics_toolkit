"""Utility functions for analytics toolkit."""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load data from various file formats."""
    filepath = Path(filepath)

    if filepath.suffix == '.csv':
        return pd.read_csv(filepath)
    elif filepath.suffix == '.parquet':
        return pd.read_parquet(filepath)
    elif filepath.suffix in ['.json', '.jsonl']:
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def save_data(data: pd.DataFrame, filepath: Union[str, Path], **kwargs: Any) -> None:
    """Save data to various file formats."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.suffix == '.csv':
        data.to_csv(filepath, index=False, **kwargs)
    elif filepath.suffix == '.parquet':
        data.to_parquet(filepath, **kwargs)
    elif filepath.suffix == '.json':
        data.to_json(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def describe_data(data: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive data description."""
    return {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum(),
        'numeric_summary': data.describe().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 0 else None
    }