# Analytics Toolkit Development Script for PowerShell
# Cross-platform development commands with better Windows support

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Check for Poetry
try {
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        $PoetryCmd = "poetry"
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        py -m poetry --version | Out-Null
        $PoetryCmd = "py -m poetry"
    } else {
        throw "Poetry not found"
    }
} catch {
    Write-Host "❌ Error: Poetry not found. Please install Poetry first." -ForegroundColor Red
    exit 1
}

# Helper functions
function Write-Header {
    param([string]$Message)
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Show-Help {
    Write-Host "Analytics Toolkit Development Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Green
    Write-Host "  .\scripts\dev.ps1 [command]"
    Write-Host ""
    Write-Host "Setup Commands:" -ForegroundColor Green
    Write-Host "  install             Install dependencies"
    Write-Host "  install-dev         Install with dev dependencies"
    Write-Host "  pre-commit-install  Install pre-commit hooks"
    Write-Host "  dev-setup          Complete development setup"
    Write-Host ""
    Write-Host "Development Commands:" -ForegroundColor Green
    Write-Host "  test               Run tests with coverage"
    Write-Host "  test-fast          Run tests without coverage"
    Write-Host "  lint               Run all linting tools"
    Write-Host "  format             Format code with black and ruff"
    Write-Host "  typecheck          Run mypy type checking"
    Write-Host "  pre-commit-run     Run pre-commit on all files"
    Write-Host ""
    Write-Host "Build Commands:" -ForegroundColor Green
    Write-Host "  build              Build package"
    Write-Host "  clean              Clean up cache and build files"
    Write-Host "  docs               Build documentation"
    Write-Host ""
    Write-Host "Development Tools:" -ForegroundColor Green
    Write-Host "  jupyter            Start Jupyter Lab"
    Write-Host "  security           Run security scans"
    Write-Host "  env-info           Show environment information"
    Write-Host "  check-all          Run all quality checks"
    Write-Host ""
    Write-Host "Docker Commands:" -ForegroundColor Green
    Write-Host "  docker-build       Build Docker image"
    Write-Host "  docker-run         Run Docker container"
    Write-Host ""
}

function Install-Dependencies {
    Write-Header "Installing dependencies"
    Invoke-Expression "$PoetryCmd install"
    Write-Success "Dependencies installed"
}

function Install-DevDependencies {
    Write-Header "Installing dependencies with dev extras"
    Invoke-Expression "$PoetryCmd install --with dev"
    Write-Success "Dev dependencies installed"
}

function Run-Tests {
    Write-Header "Running tests with coverage"
    Invoke-Expression "$PoetryCmd run pytest --cov=src/analytics_toolkit --cov-report=html --cov-report=term-missing --cov-report=xml -v"
    Write-Success "Tests completed"
}

function Run-FastTests {
    Write-Header "Running tests (fast mode)"
    Invoke-Expression "$PoetryCmd run pytest -v"
    Write-Success "Fast tests completed"
}

function Run-Lint {
    Write-Header "Running all linting tools"

    Write-Host "Running Black..." -ForegroundColor Yellow
    try {
        Invoke-Expression "$PoetryCmd run black --check --diff ."
        Write-Success "Black formatting check passed"
    } catch {
        Write-Error "Black formatting check failed"
        throw
    }

    Write-Host "Running Ruff..." -ForegroundColor Yellow
    try {
        Invoke-Expression "$PoetryCmd run ruff check ."
        Write-Success "Ruff linting passed"
    } catch {
        Write-Error "Ruff linting failed"
        throw
    }

    Write-Host "Running MyPy..." -ForegroundColor Yellow
    try {
        Invoke-Expression "$PoetryCmd run mypy src/ --ignore-missing-imports"
        Write-Success "MyPy type checking passed"
    } catch {
        Write-Error "MyPy type checking failed"
        throw
    }

    Write-Success "All linting checks passed"
}

function Format-Code {
    Write-Header "Formatting code"

    Write-Host "Running Black..." -ForegroundColor Yellow
    Invoke-Expression "$PoetryCmd run black ."

    Write-Host "Running Ruff fixes..." -ForegroundColor Yellow
    Invoke-Expression "$PoetryCmd run ruff check --fix ."

    Write-Success "Code formatting completed"
}

function Run-TypeCheck {
    Write-Header "Running type checking"
    Invoke-Expression "$PoetryCmd run mypy src/ --ignore-missing-imports"
    Write-Success "Type checking completed"
}

function Install-PreCommitHooks {
    Write-Header "Installing pre-commit hooks"
    Invoke-Expression "$PoetryCmd run pre-commit install"
    Write-Success "Pre-commit hooks installed"
}

function Run-PreCommit {
    Write-Header "Running pre-commit on all files"
    Invoke-Expression "$PoetryCmd run pre-commit run --all-files"
    Write-Success "Pre-commit checks completed"
}

function Build-Package {
    Write-Header "Building package"
    Invoke-Expression "$PoetryCmd build"
    Write-Success "Package built successfully"
}

function Clean-Project {
    Write-Header "Cleaning up cache and build files"

    Write-Host "Removing Python cache files..." -ForegroundColor Yellow
    Get-ChildItem -Path . -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -Directory -Name "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -Directory -Filter "*.egg-info" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    Write-Host "Removing build artifacts..." -ForegroundColor Yellow
    @("build", "dist", ".pytest_cache", "htmlcov", ".mypy_cache", ".ruff_cache") | ForEach-Object {
        if (Test-Path $_) {
            Remove-Item $_ -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    if (Test-Path ".coverage") {
        Remove-Item ".coverage" -Force -ErrorAction SilentlyContinue
    }

    Write-Success "Cleanup completed"
}

function Build-Docs {
    Write-Header "Building documentation"

    if (-not (Test-Path "docs")) {
        Write-Warning "Creating docs directory..."
        New-Item -ItemType Directory -Path "docs" | Out-Null
        Invoke-Expression "$PoetryCmd add --group dev sphinx sphinx-rtd-theme sphinx-autodoc-typehints"
        Invoke-Expression "$PoetryCmd run sphinx-quickstart docs --quiet --project=`"Analytics Toolkit`" --author=`"Analytics Team`" --release=`"0.1.0`" --language=`"en`" --extensions=`"sphinx.ext.autodoc,sphinx.ext.viewcode,sphinx.ext.napoleon`""
    }

    Invoke-Expression "$PoetryCmd run sphinx-build -b html docs docs/_build/html"
    Write-Success "Documentation built successfully"
}

function Start-Jupyter {
    Write-Header "Starting Jupyter Lab"
    Write-Warning "Jupyter will be available at http://localhost:8888"
    Invoke-Expression "$PoetryCmd run jupyter lab --ip=0.0.0.0 --no-browser"
}

function Run-Security {
    Write-Header "Running security scans"

    Write-Host "Installing security tools..." -ForegroundColor Yellow
    try {
        Invoke-Expression "$PoetryCmd add --group dev safety bandit"
    } catch {
        # Continue if tools are already installed
    }

    Write-Host "Running Safety check..." -ForegroundColor Yellow
    try {
        Invoke-Expression "$PoetryCmd run safety check"
    } catch {
        Write-Warning "Safety check completed with warnings"
    }

    Write-Host "Running Bandit security linter..." -ForegroundColor Yellow
    try {
        Invoke-Expression "$PoetryCmd run bandit -r src/"
    } catch {
        Write-Warning "Bandit scan completed with warnings"
    }

    Write-Success "Security scans completed"
}

function Show-EnvInfo {
    Write-Header "Environment Information"

    Write-Host "Python version:" -ForegroundColor Yellow
    python --version

    Write-Host "Poetry version:" -ForegroundColor Yellow
    Invoke-Expression "$PoetryCmd --version"

    Write-Host "Project info:" -ForegroundColor Yellow
    Invoke-Expression "$PoetryCmd show --tree" | Select-Object -First 20

    Write-Success "Environment info displayed"
}

function Run-AllChecks {
    Write-Header "Running all quality checks"

    try {
        Run-Lint
        Run-Tests
        Write-Success "All quality checks passed!"
    } catch {
        Write-Error "Some quality checks failed"
        throw
    }
}

function Setup-DevEnvironment {
    Write-Header "Setting up development environment"

    Install-DevDependencies
    Install-PreCommitHooks

    Write-Success "Development environment setup complete!"
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  - Run '.\scripts\dev.ps1 test' to verify everything works"
    Write-Host "  - Run '.\scripts\dev.ps1 jupyter' to start development"
    Write-Host "  - Run '.\scripts\dev.ps1 help' to see all available commands"
}

function Build-DockerImage {
    Write-Header "Building Docker image"
    docker build -t analytics-toolkit:latest .
    Write-Success "Docker image built successfully"
}

function Run-DockerContainer {
    Write-Header "Running Docker container"
    Write-Warning "Container will be available at http://localhost:8888"
    docker run -it --rm -p 8888:8888 analytics-toolkit:latest
}

# Main script logic
try {
    switch ($Command.ToLower()) {
        "help" { Show-Help }
        "install" { Install-Dependencies }
        "install-dev" { Install-DevDependencies }
        "test" { Run-Tests }
        "test-fast" { Run-FastTests }
        "lint" { Run-Lint }
        "format" { Format-Code }
        "typecheck" { Run-TypeCheck }
        "pre-commit-install" { Install-PreCommitHooks }
        "pre-commit-run" { Run-PreCommit }
        "build" { Build-Package }
        "clean" { Clean-Project }
        "docs" { Build-Docs }
        "jupyter" { Start-Jupyter }
        "security" { Run-Security }
        "env-info" { Show-EnvInfo }
        "check-all" { Run-AllChecks }
        "dev-setup" { Setup-DevEnvironment }
        "docker-build" { Build-DockerImage }
        "docker-run" { Run-DockerContainer }
        default {
            Write-Error "Unknown command: $Command"
            Write-Host ""
            Show-Help
            exit 1
        }
    }
} catch {
    Write-Error "Command failed: $_"
    exit 1
}