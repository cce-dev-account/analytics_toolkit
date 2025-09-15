# Analytics Toolkit Development Makefile
# Requires Poetry to be installed

.PHONY: help install test lint clean format pre-commit-install pre-commit-run build docs serve-docs jupyter security release-check release-prepare release-tag release

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Poetry command
POETRY := py -m poetry

help: ## Show this help message
	@echo "$(BLUE)Analytics Toolkit Development Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(RESET)"
	@echo "  make install           Install dependencies"
	@echo "  make install-dev       Install with dev dependencies"
	@echo "  make pre-commit-install Install pre-commit hooks"
	@echo ""
	@echo "$(GREEN)Development Commands:$(RESET)"
	@echo "  make test              Run tests with coverage"
	@echo "  make test-fast         Run tests without coverage"
	@echo "  make lint              Run all linting tools"
	@echo "  make format            Format code with black and ruff"
	@echo "  make typecheck         Run mypy type checking"
	@echo "  make pre-commit-run    Run pre-commit on all files"
	@echo ""
	@echo "$(GREEN)Build Commands:$(RESET)"
	@echo "  make build             Build package"
	@echo "  make clean             Clean up cache and build files"
	@echo "  make docs              Build documentation"
	@echo "  make serve-docs        Serve docs locally"
	@echo ""
	@echo "$(GREEN)Development Tools:$(RESET)"
	@echo "  make jupyter           Start Jupyter Lab"
	@echo "  make security          Run security scans"
	@echo "  make benchmark         Run performance benchmarks"
	@echo "  make docker-build      Build Docker image"
	@echo "  make docker-run        Run Docker container"
	@echo ""
	@echo "$(GREEN)Quality Assurance Commands:$(RESET)"
	@echo "  make qa-all            Run comprehensive QA analysis"
	@echo "  make qa-security       Run security analysis"
	@echo "  make qa-complexity     Run code complexity analysis"
	@echo "  make qa-dependencies   Check dependencies"
	@echo "  make qa-licenses       Check license compatibility"
	@echo ""
	@echo "$(GREEN)Release Commands:$(RESET)"
	@echo "  make release-check     Run pre-release validation checks"
	@echo "  make release-prepare   Prepare release (specify VERSION=x.y.z)"
	@echo "  make release-tag       Create and push release tag (specify VERSION=x.y.z)"
	@echo "  make release           Full release process (specify VERSION=x.y.z)"

# Installation commands
install: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(RESET)"
	$(POETRY) install

install-dev: ## Install with development dependencies
	@echo "$(BLUE)Installing dependencies with dev extras...$(RESET)"
	$(POETRY) install --with dev

# Testing commands
test: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(POETRY) run pytest --cov=src/analytics_toolkit --cov-report=html --cov-report=term-missing --cov-report=xml -v

test-fast: ## Run tests without coverage
	@echo "$(BLUE)Running tests (fast mode)...$(RESET)"
	$(POETRY) run pytest -v

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(RESET)"
	$(POETRY) run pytest-watch -- --cov=src/analytics_toolkit -v

# Code quality commands
lint: ## Run all linting tools
	@echo "$(BLUE)Running all linting tools...$(RESET)"
	@echo "$(YELLOW)Running Black...$(RESET)"
	$(POETRY) run black --check --diff .
	@echo "$(YELLOW)Running Ruff...$(RESET)"
	$(POETRY) run ruff check .
	@echo "$(YELLOW)Running MyPy...$(RESET)"
	$(POETRY) run mypy src/ --ignore-missing-imports

format: ## Format code with black and ruff
	@echo "$(BLUE)Formatting code...$(RESET)"
	@echo "$(YELLOW)Running Black...$(RESET)"
	$(POETRY) run black .
	@echo "$(YELLOW)Running Ruff fixes...$(RESET)"
	$(POETRY) run ruff check --fix .

typecheck: ## Run mypy type checking
	@echo "$(BLUE)Running type checking...$(RESET)"
	$(POETRY) run mypy src/ --ignore-missing-imports

# Pre-commit commands
pre-commit-install: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(RESET)"
	$(POETRY) run pre-commit install

pre-commit-run: ## Run pre-commit on all files
	@echo "$(BLUE)Running pre-commit on all files...$(RESET)"
	$(POETRY) run pre-commit run --all-files

# Build commands
build: ## Build package
	@echo "$(BLUE)Building package...$(RESET)"
	$(POETRY) build

clean: ## Clean up cache and build files
	@echo "$(BLUE)Cleaning up...$(RESET)"
	@echo "$(YELLOW)Removing Python cache files...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "$(YELLOW)Removing build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	@echo "$(GREEN)Cleanup complete!$(RESET)"

# Documentation commands
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	@echo "$(YELLOW)Installing documentation dependencies...$(RESET)"
	$(POETRY) install --with docs
	@echo "$(YELLOW)Creating docs directories...$(RESET)"
	mkdir -p docs/_static docs/_templates
	@echo "$(YELLOW)Generating API documentation...$(RESET)"
	$(POETRY) run sphinx-apidoc -o docs/api src/analytics_toolkit --force --module-first
	@echo "$(YELLOW)Building HTML documentation...$(RESET)"
	cd docs && $(POETRY) run sphinx-build -b html . _build/html -W
	@echo "$(GREEN)Documentation built successfully!$(RESET)"
	@echo "$(BLUE)Open docs/_build/html/index.html to view$(RESET)"

serve-docs: docs ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(RESET)"
	@cd docs/_build/html && python -m http.server 8000

# Development tools
jupyter: ## Start Jupyter Lab
	@echo "$(BLUE)Starting Jupyter Lab...$(RESET)"
	$(POETRY) run jupyter lab --ip=0.0.0.0 --no-browser

jupyter-notebook: ## Start Jupyter Notebook
	@echo "$(BLUE)Starting Jupyter Notebook...$(RESET)"
	$(POETRY) run jupyter notebook --ip=0.0.0.0 --no-browser

# Security commands
security: ## Run security scans
	@echo "$(BLUE)Running security scans...$(RESET)"
	@echo "$(YELLOW)Running Safety check...$(RESET)"
	$(POETRY) add --group dev safety || true
	$(POETRY) run safety check || true
	@echo "$(YELLOW)Running Bandit security linter...$(RESET)"
	$(POETRY) add --group dev bandit || true
	$(POETRY) run bandit -r src/ || true

# Performance commands
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	@if [ ! -d "tests/performance" ]; then \
		echo "$(YELLOW)Creating performance tests...$(RESET)"; \
		mkdir -p tests/performance; \
	fi
	$(POETRY) add --group dev pytest-benchmark || true
	$(POETRY) run pytest tests/performance/ --benchmark-only -v || echo "$(YELLOW)No performance tests found$(RESET)"

# Docker commands
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t analytics-toolkit:latest .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(RESET)"
	docker run -it --rm -p 8888:8888 analytics-toolkit:latest

docker-dev: ## Run Docker container for development
	@echo "$(BLUE)Running Docker container in development mode...$(RESET)"
	docker-compose up -d

# Environment commands
env-info: ## Show environment information
	@echo "$(BLUE)Environment Information:$(RESET)"
	@echo "$(YELLOW)Python version:$(RESET)"
	python --version
	@echo "$(YELLOW)Poetry version:$(RESET)"
	$(POETRY) --version
	@echo "$(YELLOW)Project dependencies:$(RESET)"
	$(POETRY) show --tree

update-deps: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	$(POETRY) update

# Quality gates
check-all: lint test ## Run all quality checks
	@echo "$(GREEN)All quality checks passed!$(RESET)"

# Development workflow
dev-setup: install-dev pre-commit-install ## Complete development setup
	@echo "$(GREEN)Development environment setup complete!$(RESET)"
	@echo "$(BLUE)Next steps:$(RESET)"
	@echo "  - Run 'make test' to verify everything works"
	@echo "  - Run 'make jupyter' to start development"
	@echo "  - Run 'make help' to see all available commands"

# Release commands
version-patch: ## Bump patch version
	$(POETRY) version patch

version-minor: ## Bump minor version
	$(POETRY) version minor

version-major: ## Bump major version
	$(POETRY) version major

# Release commands
release-check: ## Run pre-release validation checks
	@echo "$(BLUE)Running pre-release checks...$(RESET)"
	@if [ ! -f "scripts/release.sh" ]; then \
		echo "$(RED)Release script not found. Please ensure scripts/release.sh exists.$(RESET)"; \
		exit 1; \
	fi
	@chmod +x scripts/release.sh
	@./scripts/release.sh check

release-prepare: ## Prepare release (specify VERSION=x.y.z)
	@if [ -z "$(VERSION)" ]; then \
		echo "$(RED)VERSION is required. Usage: make release-prepare VERSION=1.0.0$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Preparing release $(VERSION)...$(RESET)"
	@chmod +x scripts/release.sh
	@./scripts/release.sh prepare $(VERSION)

release-tag: ## Create and push release tag (specify VERSION=x.y.z)
	@if [ -z "$(VERSION)" ]; then \
		echo "$(RED)VERSION is required. Usage: make release-tag VERSION=1.0.0$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Creating release tag for $(VERSION)...$(RESET)"
	@chmod +x scripts/release.sh
	@./scripts/release.sh tag $(VERSION)

release: ## Full release process (specify VERSION=x.y.z)
	@if [ -z "$(VERSION)" ]; then \
		echo "$(RED)VERSION is required. Usage: make release VERSION=1.0.0$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Starting full release process for $(VERSION)...$(RESET)"
	@chmod +x scripts/release.sh
	@./scripts/release.sh release $(VERSION)