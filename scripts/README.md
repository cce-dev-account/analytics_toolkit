# Development Scripts

This directory contains cross-platform development scripts for the Analytics Toolkit project.

## Available Scripts

### 🐧 Linux/macOS: `dev.sh`
```bash
# Make executable first
chmod +x scripts/dev.sh

# Usage
./scripts/dev.sh [command]
```

### 🪟 Windows: `dev.bat` and `dev.ps1`
```cmd
# Command Prompt
scripts\dev.bat [command]

# PowerShell (recommended)
.\scripts\dev.ps1 [command]
```

### 🔧 Make (if available)
```bash
make [command]
```

## Common Commands

### Setup Commands
- `install` - Install dependencies
- `install-dev` - Install with dev dependencies
- `pre-commit-install` - Install pre-commit hooks
- `dev-setup` - Complete development setup

### Development Commands
- `test` - Run tests with coverage
- `test-fast` - Run tests without coverage
- `lint` - Run all linting tools (black, ruff, mypy)
- `format` - Format code with black and ruff
- `typecheck` - Run mypy type checking
- `pre-commit-run` - Run pre-commit on all files

### Build Commands
- `build` - Build package
- `clean` - Clean up cache and build files
- `docs` - Build documentation

### Development Tools
- `jupyter` - Start Jupyter Lab
- `security` - Run security scans
- `env-info` - Show environment information
- `check-all` - Run all quality checks

### Docker Commands
- `docker-build` - Build Docker image
- `docker-run` - Run Docker container

## Quick Start

1. **Set up development environment:**
   ```bash
   # Linux/macOS
   ./scripts/dev.sh dev-setup

   # Windows PowerShell
   .\scripts\dev.ps1 dev-setup

   # Windows CMD
   scripts\dev.bat dev-setup
   ```

2. **Run tests:**
   ```bash
   ./scripts/dev.sh test
   .\scripts\dev.ps1 test
   scripts\dev.bat test
   ```

3. **Format and lint code:**
   ```bash
   ./scripts/dev.sh format
   ./scripts/dev.sh lint
   ```

4. **Start Jupyter for development:**
   ```bash
   ./scripts/dev.sh jupyter
   ```

## Features

- ✅ **Cross-platform**: Works on Linux, macOS, and Windows
- ✅ **Auto-detection**: Automatically finds Poetry installation
- ✅ **Error handling**: Proper error reporting and exit codes
- ✅ **Colored output**: Enhanced readability with colored messages
- ✅ **Help system**: Built-in help for all commands
- ✅ **Comprehensive**: Covers entire development workflow

## Requirements

- Python 3.11+
- Poetry (installed via `py -m pip install poetry` or official installer)
- Docker (optional, for containerized development)

## Troubleshooting

### Poetry Not Found
If you get "Poetry not found" errors:

1. **Install Poetry:**
   ```bash
   # Via pip
   py -m pip install poetry

   # Via official installer (Linux/macOS)
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Add to PATH** (if needed):
   ```bash
   # Windows
   $env:PATH += ";C:\Users\$env:USERNAME\AppData\Roaming\Python\Scripts"

   # Linux/macOS
   export PATH="$HOME/.local/bin:$PATH"
   ```

### Permission Denied (Linux/macOS)
```bash
chmod +x scripts/dev.sh
```

### PowerShell Execution Policy (Windows)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```