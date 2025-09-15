#!/bin/bash
# Release Helper Script for Analytics Toolkit
# Automates the release process and validation

set -e  # Exit on any error

# Colors for output
BLUE='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
RESET='\033[0m'

# Poetry command detection
if command -v poetry &> /dev/null; then
    POETRY_CMD="poetry"
elif command -v py &> /dev/null && py -m poetry --version &> /dev/null; then
    POETRY_CMD="py -m poetry"
else
    echo -e "${RED}❌ Error: Poetry not found. Please install Poetry first.${RESET}"
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

# Show help
show_help() {
    echo -e "${BLUE}Analytics Toolkit Release Helper${RESET}"
    echo ""
    echo -e "${GREEN}Usage:${RESET} ./scripts/release.sh [command] [version]"
    echo ""
    echo -e "${GREEN}Commands:${RESET}"
    echo "  check               Pre-release validation checks"
    echo "  prepare VERSION     Prepare release (bump version, run tests)"
    echo "  tag VERSION         Create and push release tag"
    echo "  release VERSION     Full release process (prepare + tag)"
    echo "  hotfix VERSION      Create hotfix release"
    echo "  status              Show current release status"
    echo ""
    echo -e "${GREEN}Examples:${RESET}"
    echo "  ./scripts/release.sh check"
    echo "  ./scripts/release.sh prepare 1.0.0"
    echo "  ./scripts/release.sh release 1.0.0"
    echo "  ./scripts/release.sh hotfix 1.0.1"
    echo ""
}

# Validate version format
validate_version() {
    local version=$1
    if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+(\.[0-9]+)?)?$ ]]; then
        print_error "Invalid version format: $version"
        echo "Expected format: X.Y.Z or X.Y.Z-suffix (e.g., 1.0.0, 2.1.0-alpha.1)"
        exit 1
    fi
    print_success "Version format is valid: $version"
}

# Check if git working directory is clean
check_git_clean() {
    if ! git diff-index --quiet HEAD --; then
        print_error "Git working directory is not clean"
        echo "Please commit or stash your changes before releasing"
        git status --porcelain
        exit 1
    fi
    print_success "Git working directory is clean"
}

# Check if on main branch
check_main_branch() {
    local current_branch=$(git branch --show-current)
    if [ "$current_branch" != "main" ] && [ "$current_branch" != "master" ]; then
        print_warning "Not on main/master branch (currently on: $current_branch)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "On main branch: $current_branch"
    fi
}

# Run pre-release checks
run_pre_release_checks() {
    print_header "Running Pre-Release Checks"

    # Check git status
    check_git_clean
    check_main_branch

    # Check if dependencies are up to date
    echo -e "${YELLOW}Checking dependencies...${RESET}"
    $POETRY_CMD install --with dev
    print_success "Dependencies installed"

    # Run tests
    echo -e "${YELLOW}Running tests...${RESET}"
    $POETRY_CMD run pytest --cov=src/analytics_toolkit --cov-fail-under=80 -q
    print_success "Tests passed"

    # Run linting
    echo -e "${YELLOW}Running code quality checks...${RESET}"
    $POETRY_CMD run black --check . --quiet
    $POETRY_CMD run ruff check . --quiet
    $POETRY_CMD run mypy src/ --ignore-missing-imports --quiet
    print_success "Code quality checks passed"

    # Check for security issues
    echo -e "${YELLOW}Running security checks...${RESET}"
    $POETRY_CMD run safety check --short-report || print_warning "Security check completed with warnings"

    print_success "All pre-release checks passed!"
}

# Prepare release
prepare_release() {
    local version=$1

    if [ -z "$version" ]; then
        print_error "Version is required for prepare command"
        echo "Usage: ./scripts/release.sh prepare 1.0.0"
        exit 1
    fi

    validate_version "$version"

    print_header "Preparing Release $version"

    # Run pre-release checks
    run_pre_release_checks

    # Update version
    echo -e "${YELLOW}Updating version to $version...${RESET}"
    $POETRY_CMD version "$version"

    # Verify version was set correctly
    local actual_version=$($POETRY_CMD version --short)
    if [ "$actual_version" != "$version" ]; then
        print_error "Version mismatch: expected $version, got $actual_version"
        exit 1
    fi
    print_success "Version updated to: $actual_version"

    # Build package
    echo -e "${YELLOW}Building package...${RESET}"
    $POETRY_CMD build
    print_success "Package built successfully"

    # Test package installation
    echo -e "${YELLOW}Testing package installation...${RESET}"
    $POETRY_CMD run pip install dist/analytics_toolkit-${version}-py3-none-any.whl --force-reinstall --quiet
    $POETRY_CMD run python -c "import analytics_toolkit; print(f'Package version: {analytics_toolkit.__version__}')"
    print_success "Package installation test passed"

    # Show what will be released
    echo ""
    print_header "Release Summary"
    echo "Version: $version"
    echo "Built files:"
    ls -la dist/analytics_toolkit-${version}*

    print_success "Release preparation completed!"
    echo ""
    echo -e "${BLUE}Next steps:${RESET}"
    echo "1. Review the changes above"
    echo "2. Run: ./scripts/release.sh tag $version"
    echo "3. Or run: git add pyproject.toml && git commit -m 'bump: version $version'"
}

# Create and push tag
create_tag() {
    local version=$1

    if [ -z "$version" ]; then
        print_error "Version is required for tag command"
        echo "Usage: ./scripts/release.sh tag 1.0.0"
        exit 1
    fi

    validate_version "$version"

    local tag="v$version"

    print_header "Creating Release Tag $tag"

    # Check if tag already exists
    if git tag -l | grep -q "^$tag$"; then
        print_error "Tag $tag already exists"
        echo "Use a different version or delete the existing tag with: git tag -d $tag"
        exit 1
    fi

    # Check git status
    check_git_clean

    # Create tag
    echo -e "${YELLOW}Creating tag $tag...${RESET}"
    git tag -a "$tag" -m "Release $version"
    print_success "Tag $tag created"

    # Push tag
    echo -e "${YELLOW}Pushing tag to origin...${RESET}"
    git push origin "$tag"
    print_success "Tag pushed to origin"

    print_success "Release tag created and pushed!"
    echo ""
    echo -e "${BLUE}Release initiated! Monitor progress at:${RESET}"
    echo "https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\([^.]*\).*/\1/')/actions"
}

# Full release process
full_release() {
    local version=$1

    if [ -z "$version" ]; then
        print_error "Version is required for release command"
        echo "Usage: ./scripts/release.sh release 1.0.0"
        exit 1
    fi

    prepare_release "$version"

    echo ""
    read -p "Ready to create and push release tag v$version? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Commit version bump
        git add pyproject.toml
        git commit -m "bump: version $version"

        create_tag "$version"
    else
        echo "Release cancelled. You can manually create the tag later with:"
        echo "  git add pyproject.toml"
        echo "  git commit -m 'bump: version $version'"
        echo "  ./scripts/release.sh tag $version"
    fi
}

# Create hotfix release
hotfix_release() {
    local version=$1

    if [ -z "$version" ]; then
        print_error "Version is required for hotfix command"
        echo "Usage: ./scripts/release.sh hotfix 1.0.1"
        exit 1
    fi

    print_header "Creating Hotfix Release $version"
    print_warning "Hotfix releases should only contain critical bug fixes"

    read -p "Continue with hotfix release? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        full_release "$version"
    else
        echo "Hotfix cancelled"
    fi
}

# Show release status
show_status() {
    print_header "Release Status"

    # Current version
    local current_version=$($POETRY_CMD version --short)
    echo "Current version: $current_version"

    # Last tag
    local last_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "No tags found")
    echo "Last tag: $last_tag"

    # Commits since last tag
    if [ "$last_tag" != "No tags found" ]; then
        local commits_since=$(git rev-list ${last_tag}..HEAD --count)
        echo "Commits since last tag: $commits_since"

        if [ "$commits_since" -gt 0 ]; then
            echo ""
            echo "Recent commits:"
            git log --oneline ${last_tag}..HEAD | head -10
        fi
    fi

    # Git status
    echo ""
    echo "Git status:"
    if git diff-index --quiet HEAD --; then
        print_success "Working directory is clean"
    else
        print_warning "Working directory has uncommitted changes"
        git status --porcelain
    fi
}

# Main script logic
case "${1:-help}" in
    "help"|"--help"|"-h")
        show_help
        ;;
    "check")
        run_pre_release_checks
        ;;
    "prepare")
        prepare_release "$2"
        ;;
    "tag")
        create_tag "$2"
        ;;
    "release")
        full_release "$2"
        ;;
    "hotfix")
        hotfix_release "$2"
        ;;
    "status")
        show_status
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac