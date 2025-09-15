#!/bin/bash
# Analytics Toolkit Development Script
# Cross-platform development commands

set -e  # Exit on any error

# Colors for output
BLUE='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
RESET='\033[0m'

# Detect Poetry command
if command -v poetry &> /dev/null; then
    POETRY_CMD="poetry"
elif command -v py &> /dev/null && py -m poetry --version &> /dev/null; then
    POETRY_CMD="py -m poetry"
else
    echo -e "${RED}Error: Poetry not found. Please install Poetry first.${RESET}"
    exit 1
fi

# Helper functions
print_header() {
    echo -e "${BLUE}=== $1 ===${RESET}"
}

print_success() {
    echo -e "${GREEN}✅ $1${RESET}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${RESET}"
}

print_error() {
    echo -e "${RED}❌ $1${RESET}"
}

# Function to show help
show_help() {
    echo -e "${BLUE}Analytics Toolkit Development Script${RESET}"
    echo ""
    echo -e "${GREEN}Usage:${RESET} ./scripts/dev.sh [command]"
    echo ""
    echo -e "${GREEN}Setup Commands:${RESET}"
    echo "  install             Install dependencies"
    echo "  install-dev         Install with dev dependencies"
    echo "  pre-commit-install  Install pre-commit hooks"
    echo "  dev-setup          Complete development setup"
    echo ""
    echo -e "${GREEN}Development Commands:${RESET}"
    echo "  test               Run tests with coverage"
    echo "  test-fast          Run tests without coverage"
    echo "  lint               Run all linting tools"
    echo "  format             Format code with black and ruff"
    echo "  typecheck          Run mypy type checking"
    echo "  pre-commit-run     Run pre-commit on all files"
    echo ""
    echo -e "${GREEN}Build Commands:${RESET}"
    echo "  build              Build package"
    echo "  clean              Clean up cache and build files"
    echo "  docs               Build documentation"
    echo ""
    echo -e "${GREEN}Development Tools:${RESET}"
    echo "  jupyter            Start Jupyter Lab"
    echo "  security           Run security scans"
    echo "  env-info           Show environment information"
    echo "  check-all          Run all quality checks"
    echo ""
    echo -e "${GREEN}Quality Assurance Commands:${RESET}"
    echo "  qa-all             Run comprehensive QA analysis"
    echo "  qa-security        Run security analysis"
    echo "  qa-complexity      Run code complexity analysis"
    echo "  qa-dependencies    Check dependencies"
    echo "  qa-licenses        Check license compatibility"
    echo ""
    echo -e "${GREEN}Docker Commands:${RESET}"
    echo "  docker-build       Build Docker image"
    echo "  docker-run         Run Docker container"
    echo ""
}

# Function to install dependencies
install() {
    print_header "Installing dependencies"
    $POETRY_CMD install
    print_success "Dependencies installed"
}

# Function to install dev dependencies
install_dev() {
    print_header "Installing dependencies with dev extras"
    $POETRY_CMD install --with dev
    print_success "Dev dependencies installed"
}

# Function to run tests
test() {
    print_header "Running tests with coverage"
    $POETRY_CMD run pytest --cov=src/analytics_toolkit --cov-report=html --cov-report=term-missing --cov-report=xml -v
    print_success "Tests completed"
}

# Function to run fast tests
test_fast() {
    print_header "Running tests (fast mode)"
    $POETRY_CMD run pytest -v
    print_success "Fast tests completed"
}

# Function to run linting
lint() {
    print_header "Running all linting tools"

    echo -e "${YELLOW}Running Black...${RESET}"
    if $POETRY_CMD run black --check --diff .; then
        print_success "Black formatting check passed"
    else
        print_error "Black formatting check failed"
        return 1
    fi

    echo -e "${YELLOW}Running Ruff...${RESET}"
    if $POETRY_CMD run ruff check .; then
        print_success "Ruff linting passed"
    else
        print_error "Ruff linting failed"
        return 1
    fi

    echo -e "${YELLOW}Running MyPy...${RESET}"
    if $POETRY_CMD run mypy src/ --ignore-missing-imports; then
        print_success "MyPy type checking passed"
    else
        print_error "MyPy type checking failed"
        return 1
    fi

    print_success "All linting checks passed"
}

# Function to format code
format_code() {
    print_header "Formatting code"

    echo -e "${YELLOW}Running Black...${RESET}"
    $POETRY_CMD run black .

    echo -e "${YELLOW}Running Ruff fixes...${RESET}"
    $POETRY_CMD run ruff check --fix .

    print_success "Code formatting completed"
}

# Function to run type checking
typecheck() {
    print_header "Running type checking"
    $POETRY_CMD run mypy src/ --ignore-missing-imports
    print_success "Type checking completed"
}

# Function to install pre-commit hooks
pre_commit_install() {
    print_header "Installing pre-commit hooks"
    $POETRY_CMD run pre-commit install
    print_success "Pre-commit hooks installed"
}

# Function to run pre-commit
pre_commit_run() {
    print_header "Running pre-commit on all files"
    $POETRY_CMD run pre-commit run --all-files
    print_success "Pre-commit checks completed"
}

# Function to build package
build() {
    print_header "Building package"
    $POETRY_CMD build
    print_success "Package built successfully"
}

# Function to clean up
clean() {
    print_header "Cleaning up cache and build files"

    echo -e "${YELLOW}Removing Python cache files...${RESET}"
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

    echo -e "${YELLOW}Removing build artifacts...${RESET}"
    rm -rf build/ 2>/dev/null || true
    rm -rf dist/ 2>/dev/null || true
    rm -rf .pytest_cache/ 2>/dev/null || true
    rm -rf .coverage 2>/dev/null || true
    rm -rf htmlcov/ 2>/dev/null || true
    rm -rf .mypy_cache/ 2>/dev/null || true
    rm -rf .ruff_cache/ 2>/dev/null || true

    print_success "Cleanup completed"
}

# Function to build docs
docs() {
    print_header "Building documentation"

    # Install docs dependencies
    $POETRY_CMD install --with docs

    # Create docs directories if they don't exist
    mkdir -p docs/_static docs/_templates

    # Generate API documentation
    $POETRY_CMD run sphinx-apidoc -o docs/api src/analytics_toolkit --force --module-first

    # Build documentation
    cd docs
    $POETRY_CMD run sphinx-build -b html . _build/html -W
    cd ..

    print_success "Documentation built successfully"
    echo "Open docs/_build/html/index.html to view the documentation"
}

# Function to start Jupyter
jupyter() {
    print_header "Starting Jupyter Lab"
    print_warning "Jupyter will be available at http://localhost:8888"
    $POETRY_CMD run jupyter lab --ip=0.0.0.0 --no-browser
}

# Function to run security scans
security() {
    print_header "Running security scans"

    echo -e "${YELLOW}Installing QA tools...${RESET}"
    $POETRY_CMD install --with qa

    echo -e "${YELLOW}Running comprehensive security analysis...${RESET}"
    if [ -f "scripts/qa.sh" ]; then
        chmod +x scripts/qa.sh
        ./scripts/qa.sh security
    else
        # Fallback to individual tools
        echo -e "${YELLOW}Running Safety check...${RESET}"
        $POETRY_CMD run safety check || print_warning "Safety check completed with warnings"

        echo -e "${YELLOW}Running Bandit security linter...${RESET}"
        $POETRY_CMD run bandit -r src/ || print_warning "Bandit scan completed with warnings"

        echo -e "${YELLOW}Running pip-audit...${RESET}"
        $POETRY_CMD export -f requirements.txt --output requirements.txt --without-hashes
        $POETRY_CMD run pip-audit -r requirements.txt || print_warning "pip-audit completed with warnings"
    fi

    print_success "Security scans completed"
}

# Function to show environment info
env_info() {
    print_header "Environment Information"

    echo -e "${YELLOW}Python version:${RESET}"
    python --version

    echo -e "${YELLOW}Poetry version:${RESET}"
    $POETRY_CMD --version

    echo -e "${YELLOW}Project info:${RESET}"
    $POETRY_CMD show --tree | head -20

    print_success "Environment info displayed"
}

# Function to run all checks
check_all() {
    print_header "Running all quality checks"

    if lint && test; then
        print_success "All quality checks passed!"
    else
        print_error "Some quality checks failed"
        return 1
    fi
}

# Function for complete dev setup
dev_setup() {
    print_header "Setting up development environment"

    install_dev
    pre_commit_install

    print_success "Development environment setup complete!"
    echo -e "${BLUE}Next steps:${RESET}"
    echo "  - Run './scripts/dev.sh test' to verify everything works"
    echo "  - Run './scripts/dev.sh jupyter' to start development"
    echo "  - Run './scripts/dev.sh --help' to see all available commands"
}

# Function to build Docker image
docker_build() {
    print_header "Building Docker image"
    docker build -t analytics-toolkit:latest .
    print_success "Docker image built successfully"
}

# Function to run Docker container
docker_run() {
    print_header "Running Docker container"
    print_warning "Container will be available at http://localhost:8888"
    docker run -it --rm -p 8888:8888 analytics-toolkit:latest
}

# Main script logic
case "${1:-help}" in
    "help"|"--help"|"-h")
        show_help
        ;;
    "install")
        install
        ;;
    "install-dev")
        install_dev
        ;;
    "test")
        test
        ;;
    "test-fast")
        test_fast
        ;;
    "lint")
        lint
        ;;
    "format")
        format_code
        ;;
    "typecheck")
        typecheck
        ;;
    "pre-commit-install")
        pre_commit_install
        ;;
    "pre-commit-run")
        pre_commit_run
        ;;
    "build")
        build
        ;;
    "clean")
        clean
        ;;
    "docs")
        docs
        ;;
    "jupyter")
        jupyter
        ;;
    "security")
        security
        ;;
    "env-info")
        env_info
        ;;
    "check-all")
        check_all
        ;;
    "dev-setup")
        dev_setup
        ;;
    "docker-build")
        docker_build
        ;;
    "docker-run")
        docker_run
        ;;
    "qa-all")
        if [ -f "scripts/qa.sh" ]; then
            chmod +x scripts/qa.sh
            ./scripts/qa.sh all
        else
            print_error "QA script not found"
            exit 1
        fi
        ;;
    "qa-security")
        if [ -f "scripts/qa.sh" ]; then
            chmod +x scripts/qa.sh
            ./scripts/qa.sh security
        else
            security
        fi
        ;;
    "qa-complexity")
        if [ -f "scripts/qa.sh" ]; then
            chmod +x scripts/qa.sh
            ./scripts/qa.sh complexity
        else
            print_error "QA script not found"
            exit 1
        fi
        ;;
    "qa-dependencies")
        if [ -f "scripts/qa.sh" ]; then
            chmod +x scripts/qa.sh
            ./scripts/qa.sh dependencies
        else
            print_error "QA script not found"
            exit 1
        fi
        ;;
    "qa-licenses")
        if [ -f "scripts/qa.sh" ]; then
            chmod +x scripts/qa.sh
            ./scripts/qa.sh licenses
        else
            print_error "QA script not found"
            exit 1
        fi
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac