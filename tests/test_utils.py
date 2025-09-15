"""Tests for utils module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from analytics_toolkit.utils import describe_data, load_data, save_data


def test_describe_data():
    """Test describe_data function."""
    # Create sample data
    data = pd.DataFrame(
        {
            "numeric_col": [1, 2, 3, 4, 5],
            "string_col": ["a", "b", "c", "d", "e"],
            "missing_col": [1, 2, None, 4, 5],
        }
    )

    result = describe_data(data)

    assert result["shape"] == (5, 3)
    assert "numeric_col" in result["columns"]
    assert "string_col" in result["columns"]
    assert "missing_col" in result["columns"]
    assert result["missing_values"]["missing_col"] == 1


def test_save_load_csv():
    """Test save and load CSV functionality."""
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "test.csv"
        save_data(data, filepath)
        loaded_data = load_data(filepath)

        pd.testing.assert_frame_equal(data, loaded_data)


def test_unsupported_format():
    """Test handling of unsupported file formats."""
    data = pd.DataFrame({"col1": [1, 2, 3]})

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "test.txt"

        with pytest.raises(ValueError, match="Unsupported file format"):
            save_data(data, filepath)

        # Create a dummy file to test load
        filepath.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_data(filepath)
