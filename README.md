# Analytics Toolkit

A comprehensive Python toolkit for data analytics and machine learning with PyTorch support.

## Features

- **Data Processing**: Comprehensive data loading, saving, and preprocessing utilities
- **Machine Learning**: PyTorch-based neural networks and training utilities
- **Analytics**: Statistical analysis and data description tools
- **Quality**: Pre-commit hooks, testing, and code formatting

## Installation

1. Clone the repository
2. Install Poetry if not already installed
3. Install dependencies:

```bash
poetry install --with dev
```

## Project Structure

```
analytics_toolkit/
├── src/analytics_toolkit/    # Main package
│   ├── __init__.py
│   ├── utils.py             # Data utilities
│   ├── preprocessing.py     # Data preprocessing
│   └── models.py           # ML models and PyTorch utilities
├── tests/                  # Test suite
├── notebooks/              # Jupyter notebooks
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── data/                   # Data files (gitignored)
└── pyproject.toml         # Project configuration
```

## Usage

```python
from analytics_toolkit import utils, preprocessing, models

# Load and describe data
data = utils.load_data('data.csv')
description = utils.describe_data(data)

# Preprocess data
preprocessor = preprocessing.DataPreprocessor()
X_processed, y = preprocessor.fit_transform(data, target_column='target')

# Create and train a model
model = models.SimpleNN(input_size=X_processed.shape[1],
                       hidden_sizes=[64, 32],
                       output_size=1)
trainer = models.ModelTrainer(model)
```

## Development

### Quick Start

Set up your development environment:

```bash
# Linux/macOS
./scripts/dev.sh dev-setup

# Windows PowerShell
.\scripts\dev.ps1 dev-setup

# Windows Command Prompt
scripts\dev.bat dev-setup

# Or using Make (if available)
make dev-setup
```

### Development Commands

We provide cross-platform development scripts for common tasks:

| Command | Description |
|---------|-------------|
| `dev-setup` | Complete development environment setup |
| `install` | Install dependencies |
| `test` | Run tests with coverage |
| `lint` | Run all linting tools (black, ruff, mypy) |
| `format` | Format code with black and ruff |
| `clean` | Clean up cache and build files |
| `jupyter` | Start Jupyter Lab |
| `build` | Build package |
| `security` | Run security scans |

### Platform-Specific Usage

**Linux/macOS:**
```bash
./scripts/dev.sh [command]
make [command]  # if make is available
```

**Windows:**
```powershell
# PowerShell (recommended)
.\scripts\dev.ps1 [command]

# Command Prompt
scripts\dev.bat [command]
```

### Manual Commands

If you prefer using Poetry directly:

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest --cov=src/analytics_toolkit

# Format code
poetry run black .

# Lint code
poetry run ruff check .

# Type checking
poetry run mypy src/
```

## Dependencies

- PyTorch >= 2.0.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- And more (see pyproject.toml)

## 🚀 Releases

This project uses automated releases triggered by version tags. See [Release Documentation](docs/RELEASE.md) for complete setup instructions.

### Quick Release

```bash
# Using release script (recommended)
./scripts/release.sh release 1.0.0

# Using make commands
make release VERSION=1.0.0

# Manual tag creation
git tag v1.0.0
git push origin v1.0.0
```

### Release Types

- **Stable**: `v1.0.0` → Publishes to PyPI
- **Pre-release**: `v1.0.0-alpha.1` → Publishes to Test PyPI
- **Hotfix**: `v1.0.1` → Critical bug fixes

For detailed release instructions and troubleshooting, see:
- [Release Documentation](docs/RELEASE.md)
- [Release Checklist](RELEASE_CHECKLIST.md)