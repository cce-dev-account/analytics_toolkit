@echo off
REM Analytics Toolkit Development Script for Windows
REM Provides cross-platform development commands

setlocal enabledelayedexpansion

REM Check for Poetry
where poetry >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set POETRY_CMD=poetry
) else (
    py -m poetry --version >nul 2>nul
    if !ERRORLEVEL! EQU 0 (
        set POETRY_CMD=py -m poetry
    ) else (
        echo Error: Poetry not found. Please install Poetry first.
        exit /b 1
    )
)

REM Colors (limited in Windows CMD)
set "BLUE=[36m"
set "GREEN=[32m"
set "YELLOW=[33m"
set "RED=[31m"
set "RESET=[0m"

if "%1"=="" goto :help
if "%1"=="help" goto :help
if "%1"=="--help" goto :help
if "%1"=="-h" goto :help
if "%1"=="install" goto :install
if "%1"=="install-dev" goto :install_dev
if "%1"=="test" goto :test
if "%1"=="test-fast" goto :test_fast
if "%1"=="lint" goto :lint
if "%1"=="format" goto :format
if "%1"=="typecheck" goto :typecheck
if "%1"=="pre-commit-install" goto :pre_commit_install
if "%1"=="pre-commit-run" goto :pre_commit_run
if "%1"=="build" goto :build
if "%1"=="clean" goto :clean
if "%1"=="docs" goto :docs
if "%1"=="jupyter" goto :jupyter
if "%1"=="security" goto :security
if "%1"=="env-info" goto :env_info
if "%1"=="check-all" goto :check_all
if "%1"=="dev-setup" goto :dev_setup
if "%1"=="docker-build" goto :docker_build
if "%1"=="docker-run" goto :docker_run

echo Unknown command: %1
echo.
goto :help

:help
echo Analytics Toolkit Development Script
echo.
echo Usage: scripts\dev.bat [command]
echo.
echo Setup Commands:
echo   install             Install dependencies
echo   install-dev         Install with dev dependencies
echo   pre-commit-install  Install pre-commit hooks
echo   dev-setup          Complete development setup
echo.
echo Development Commands:
echo   test               Run tests with coverage
echo   test-fast          Run tests without coverage
echo   lint               Run all linting tools
echo   format             Format code with black and ruff
echo   typecheck          Run mypy type checking
echo   pre-commit-run     Run pre-commit on all files
echo.
echo Build Commands:
echo   build              Build package
echo   clean              Clean up cache and build files
echo   docs               Build documentation
echo.
echo Development Tools:
echo   jupyter            Start Jupyter Lab
echo   security           Run security scans
echo   env-info           Show environment information
echo   check-all          Run all quality checks
echo.
echo Docker Commands:
echo   docker-build       Build Docker image
echo   docker-run         Run Docker container
echo.
goto :end

:install
echo === Installing dependencies ===
%POETRY_CMD% install
if %ERRORLEVEL% EQU 0 (
    echo Dependencies installed successfully
) else (
    echo Failed to install dependencies
    exit /b 1
)
goto :end

:install_dev
echo === Installing dependencies with dev extras ===
%POETRY_CMD% install --with dev
if %ERRORLEVEL% EQU 0 (
    echo Dev dependencies installed successfully
) else (
    echo Failed to install dev dependencies
    exit /b 1
)
goto :end

:test
echo === Running tests with coverage ===
%POETRY_CMD% run pytest --cov=src/analytics_toolkit --cov-report=html --cov-report=term-missing --cov-report=xml -v
if %ERRORLEVEL% EQU 0 (
    echo Tests completed successfully
) else (
    echo Tests failed
    exit /b 1
)
goto :end

:test_fast
echo === Running tests (fast mode) ===
%POETRY_CMD% run pytest -v
if %ERRORLEVEL% EQU 0 (
    echo Fast tests completed successfully
) else (
    echo Fast tests failed
    exit /b 1
)
goto :end

:lint
echo === Running all linting tools ===
echo Running Black...
%POETRY_CMD% run black --check --diff .
if !ERRORLEVEL! NEQ 0 (
    echo Black formatting check failed
    exit /b 1
)

echo Running Ruff...
%POETRY_CMD% run ruff check .
if !ERRORLEVEL! NEQ 0 (
    echo Ruff linting failed
    exit /b 1
)

echo Running MyPy...
%POETRY_CMD% run mypy src/ --ignore-missing-imports
if !ERRORLEVEL! NEQ 0 (
    echo MyPy type checking failed
    exit /b 1
)

echo All linting checks passed
goto :end

:format
echo === Formatting code ===
echo Running Black...
%POETRY_CMD% run black .
echo Running Ruff fixes...
%POETRY_CMD% run ruff check --fix .
echo Code formatting completed
goto :end

:typecheck
echo === Running type checking ===
%POETRY_CMD% run mypy src/ --ignore-missing-imports
if %ERRORLEVEL% EQU 0 (
    echo Type checking completed successfully
) else (
    echo Type checking failed
    exit /b 1
)
goto :end

:pre_commit_install
echo === Installing pre-commit hooks ===
%POETRY_CMD% run pre-commit install
if %ERRORLEVEL% EQU 0 (
    echo Pre-commit hooks installed successfully
) else (
    echo Failed to install pre-commit hooks
    exit /b 1
)
goto :end

:pre_commit_run
echo === Running pre-commit on all files ===
%POETRY_CMD% run pre-commit run --all-files
if %ERRORLEVEL% EQU 0 (
    echo Pre-commit checks completed successfully
) else (
    echo Pre-commit checks failed
    exit /b 1
)
goto :end

:build
echo === Building package ===
%POETRY_CMD% build
if %ERRORLEVEL% EQU 0 (
    echo Package built successfully
) else (
    echo Package build failed
    exit /b 1
)
goto :end

:clean
echo === Cleaning up cache and build files ===
echo Removing Python cache files...
for /r %%i in (*.pyc) do del "%%i" 2>nul
for /d /r %%i in (__pycache__) do rmdir /s /q "%%i" 2>nul
for /d /r %%i in (*.egg-info) do rmdir /s /q "%%i" 2>nul

echo Removing build artifacts...
if exist build rmdir /s /q build 2>nul
if exist dist rmdir /s /q dist 2>nul
if exist .pytest_cache rmdir /s /q .pytest_cache 2>nul
if exist .coverage del .coverage 2>nul
if exist htmlcov rmdir /s /q htmlcov 2>nul
if exist .mypy_cache rmdir /s /q .mypy_cache 2>nul
if exist .ruff_cache rmdir /s /q .ruff_cache 2>nul

echo Cleanup completed
goto :end

:docs
echo === Building documentation ===
if not exist docs (
    echo Creating docs directory...
    mkdir docs
    %POETRY_CMD% add --group dev sphinx sphinx-rtd-theme sphinx-autodoc-typehints
    %POETRY_CMD% run sphinx-quickstart docs --quiet --project="Analytics Toolkit" --author="Analytics Team" --release="0.1.0" --language="en" --extensions="sphinx.ext.autodoc,sphinx.ext.viewcode,sphinx.ext.napoleon"
)

%POETRY_CMD% run sphinx-build -b html docs docs/_build/html
if %ERRORLEVEL% EQU 0 (
    echo Documentation built successfully
) else (
    echo Documentation build failed
    exit /b 1
)
goto :end

:jupyter
echo === Starting Jupyter Lab ===
echo Jupyter will be available at http://localhost:8888
%POETRY_CMD% run jupyter lab --ip=0.0.0.0 --no-browser
goto :end

:security
echo === Running security scans ===
echo Installing security tools...
%POETRY_CMD% add --group dev safety bandit

echo Running Safety check...
%POETRY_CMD% run safety check

echo Running Bandit security linter...
%POETRY_CMD% run bandit -r src/

echo Security scans completed
goto :end

:env_info
echo === Environment Information ===
echo Python version:
python --version
echo Poetry version:
%POETRY_CMD% --version
echo Project info (top 20 dependencies):
%POETRY_CMD% show --tree | head -20 2>nul || %POETRY_CMD% show --tree
echo Environment info displayed
goto :end

:check_all
echo === Running all quality checks ===
call :lint
if !ERRORLEVEL! NEQ 0 exit /b 1
call :test
if !ERRORLEVEL! NEQ 0 exit /b 1
echo All quality checks passed!
goto :end

:dev_setup
echo === Setting up development environment ===
call :install_dev
if !ERRORLEVEL! NEQ 0 exit /b 1
call :pre_commit_install
if !ERRORLEVEL! NEQ 0 exit /b 1

echo Development environment setup complete!
echo Next steps:
echo   - Run 'scripts\dev.bat test' to verify everything works
echo   - Run 'scripts\dev.bat jupyter' to start development
echo   - Run 'scripts\dev.bat help' to see all available commands
goto :end

:docker_build
echo === Building Docker image ===
docker build -t analytics-toolkit:latest .
if %ERRORLEVEL% EQU 0 (
    echo Docker image built successfully
) else (
    echo Docker image build failed
    exit /b 1
)
goto :end

:docker_run
echo === Running Docker container ===
echo Container will be available at http://localhost:8888
docker run -it --rm -p 8888:8888 analytics-toolkit:latest
goto :end

:end
endlocal