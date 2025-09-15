#!/bin/bash
# Quality Assurance Script for Analytics Toolkit
# Comprehensive security, dependency, and code quality checks

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
    echo -e "${BLUE}Analytics Toolkit Quality Assurance Tools${RESET}"
    echo ""
    echo -e "${GREEN}Usage:${RESET} ./scripts/qa.sh [command]"
    echo ""
    echo -e "${GREEN}Commands:${RESET}"
    echo "  security           Run security scans (bandit, safety, pip-audit)"
    echo "  dependencies       Check dependency vulnerabilities and updates"
    echo "  complexity         Analyze code complexity with radon"
    echo "  licenses           Check license compatibility"
    echo "  all                Run all QA checks"
    echo "  report             Generate comprehensive QA report"
    echo ""
    echo -e "${GREEN}Individual Tools:${RESET}"
    echo "  bandit             Run Bandit security scanner"
    echo "  safety             Run Safety vulnerability checker"
    echo "  pip-audit          Run pip-audit vulnerability scanner"
    echo "  radon-cc           Run radon cyclomatic complexity analysis"
    echo "  radon-mi           Run radon maintainability index analysis"
    echo "  radon-raw          Run radon raw metrics analysis"
    echo ""
}

# Install QA dependencies
install_qa_deps() {
    print_header "Installing QA Dependencies"
    $POETRY_CMD install --with qa
    pip install pip-licenses
    print_success "QA dependencies installed"
}

# Run Bandit security scanner
run_bandit() {
    print_header "Running Bandit Security Scanner"

    echo -e "${YELLOW}Scanning source code for security issues...${RESET}"
    if $POETRY_CMD run bandit -r src/ --severity-level medium; then
        print_success "Bandit scan completed - no critical issues found"
    else
        print_warning "Bandit found potential security issues"
    fi

    # Generate JSON report
    $POETRY_CMD run bandit -r src/ -f json -o bandit-report.json
    echo "Detailed report saved to: bandit-report.json"
}

# Run Safety vulnerability checker
run_safety() {
    print_header "Running Safety Vulnerability Checker"

    echo -e "${YELLOW}Checking dependencies for known vulnerabilities...${RESET}"
    $POETRY_CMD export -f requirements.txt --output requirements.txt --without-hashes

    if $POETRY_CMD run safety check; then
        print_success "Safety check completed - no vulnerabilities found"
    else
        print_warning "Safety found potential vulnerabilities"
    fi

    # Generate JSON report
    $POETRY_CMD run safety check --json --output safety-report.json || true
    echo "Detailed report saved to: safety-report.json"
}

# Run pip-audit vulnerability scanner
run_pip_audit() {
    print_header "Running pip-audit Vulnerability Scanner"

    echo -e "${YELLOW}Auditing dependencies with pip-audit...${RESET}"
    if $POETRY_CMD run pip-audit -r requirements.txt; then
        print_success "pip-audit completed - no vulnerabilities found"
    else
        print_warning "pip-audit found potential vulnerabilities"
    fi

    # Generate JSON report
    $POETRY_CMD run pip-audit -r requirements.txt --format=json --output=pip-audit-report.json || true
    echo "Detailed report saved to: pip-audit-report.json"
}

# Run comprehensive security scans
run_security() {
    print_header "Comprehensive Security Analysis"

    install_qa_deps

    run_bandit
    echo ""
    run_safety
    echo ""
    run_pip_audit

    print_success "Security analysis completed"
}

# Check dependencies
check_dependencies() {
    print_header "Dependency Analysis"

    install_qa_deps

    echo -e "${YELLOW}Checking for outdated dependencies...${RESET}"
    $POETRY_CMD show --outdated > outdated-deps.txt || true

    if [ -s outdated-deps.txt ]; then
        print_warning "Found outdated dependencies:"
        cat outdated-deps.txt
    else
        print_success "All dependencies are up to date"
    fi

    echo ""
    echo -e "${YELLOW}Generating dependency tree...${RESET}"
    $POETRY_CMD show --tree > dependency-tree.txt
    print_success "Dependency tree saved to: dependency-tree.txt"

    # Check for duplicate dependencies
    echo ""
    echo -e "${YELLOW}Checking for duplicate dependencies...${RESET}"
    if $POETRY_CMD show | sort | uniq -d | head -5; then
        print_warning "Found potential duplicate dependencies"
    else
        print_success "No duplicate dependencies found"
    fi
}

# Run code complexity analysis
run_complexity() {
    print_header "Code Complexity Analysis"

    install_qa_deps

    echo -e "${YELLOW}Analyzing cyclomatic complexity...${RESET}"
    $POETRY_CMD run radon cc src/ --average --show-complexity

    echo ""
    echo -e "${YELLOW}Analyzing maintainability index...${RESET}"
    $POETRY_CMD run radon mi src/ --show

    echo ""
    echo -e "${YELLOW}Generating raw metrics...${RESET}"
    $POETRY_CMD run radon raw src/ --summary

    # Generate JSON reports
    $POETRY_CMD run radon cc src/ --json > complexity-report.json
    $POETRY_CMD run radon mi src/ --json > maintainability-report.json
    $POETRY_CMD run radon raw src/ --json > raw-metrics-report.json

    print_success "Complexity analysis completed"
    echo "Reports saved to: complexity-report.json, maintainability-report.json, raw-metrics-report.json"

    # Check for high complexity functions
    echo ""
    echo -e "${YELLOW}Checking for high complexity functions...${RESET}"
    python3 -c "
import json
import sys

try:
    with open('complexity-report.json', 'r') as f:
        complexity = json.load(f)
except FileNotFoundError:
    print('No complexity report found')
    sys.exit(0)

high_complexity = []
for file_path, functions in complexity.items():
    for func in functions:
        if func['complexity'] > 10:
            high_complexity.append({
                'file': file_path,
                'function': func['name'],
                'complexity': func['complexity'],
                'line': func['lineno']
            })

if high_complexity:
    print(f'⚠️  Found {len(high_complexity)} functions with high complexity (>10):')
    for item in high_complexity[:10]:
        print(f'  {item[\"file\"]}:{item[\"line\"]} {item[\"function\"]} (complexity: {item[\"complexity\"]})')
    if len(high_complexity) > 10:
        print(f'  ... and {len(high_complexity) - 10} more')
else:
    print('✅ All functions have acceptable complexity')
"
}

# Check license compatibility
check_licenses() {
    print_header "License Compatibility Check"

    echo -e "${YELLOW}Installing pip-licenses...${RESET}"
    pip install pip-licenses

    echo -e "${YELLOW}Generating license report...${RESET}"
    pip-licenses --format=json --output-file=licenses.json
    pip-licenses --format=markdown --output-file=licenses.md

    echo -e "${YELLOW}License summary:${RESET}"
    pip-licenses --format=plain --summary

    # Check for problematic licenses
    echo ""
    echo -e "${YELLOW}Checking for problematic licenses...${RESET}"
    PROBLEMATIC_LICENSES="GPL-3.0,AGPL-3.0"
    if pip-licenses --format=json | jq -r '.[].License' | grep -E "$PROBLEMATIC_LICENSES" 2>/dev/null; then
        print_warning "Found potentially problematic licenses"
    else
        print_success "No problematic licenses found"
    fi

    print_success "License compatibility check completed"
    echo "Reports saved to: licenses.json, licenses.md"
}

# Run all QA checks
run_all() {
    print_header "Comprehensive Quality Assurance Analysis"

    echo "Starting comprehensive QA analysis..."
    echo ""

    run_security
    echo ""
    check_dependencies
    echo ""
    run_complexity
    echo ""
    check_licenses

    print_success "All QA checks completed!"
    echo ""
    echo -e "${BLUE}Reports generated:${RESET}"
    echo "  - bandit-report.json"
    echo "  - safety-report.json"
    echo "  - pip-audit-report.json"
    echo "  - complexity-report.json"
    echo "  - maintainability-report.json"
    echo "  - raw-metrics-report.json"
    echo "  - licenses.json"
    echo "  - outdated-deps.txt"
    echo "  - dependency-tree.txt"
}

# Generate comprehensive QA report
generate_report() {
    print_header "Generating QA Report"

    cat > qa-report.md << 'EOF'
# Quality Assurance Report

Generated on: $(date)

## Security Analysis

### Bandit Security Scanner
EOF

    if [ -f "bandit-report.json" ]; then
        echo "```json" >> qa-report.md
        jq '.metrics."_totals"' bandit-report.json >> qa-report.md 2>/dev/null || echo "No data available" >> qa-report.md
        echo "```" >> qa-report.md
    else
        echo "Report not available" >> qa-report.md
    fi

    cat >> qa-report.md << 'EOF'

### Safety Vulnerability Check
EOF

    if [ -f "safety-report.json" ]; then
        SAFETY_COUNT=$(jq '. | length' safety-report.json 2>/dev/null || echo "0")
        echo "Vulnerabilities found: $SAFETY_COUNT" >> qa-report.md
    else
        echo "Report not available" >> qa-report.md
    fi

    cat >> qa-report.md << 'EOF'

## Code Quality

### Complexity Analysis
EOF

    if [ -f "complexity-report.json" ]; then
        python3 -c "
import json
try:
    with open('complexity-report.json', 'r') as f:
        complexity = json.load(f)
    high_complexity = sum(1 for file_funcs in complexity.values() for func in file_funcs if func['complexity'] > 10)
    total_functions = sum(len(file_funcs) for file_funcs in complexity.values())
    print(f'High complexity functions: {high_complexity}/{total_functions}')
except:
    print('Data not available')
" >> qa-report.md
    else
        echo "Report not available" >> qa-report.md
    fi

    cat >> qa-report.md << 'EOF'

## Dependencies

### Outdated Dependencies
EOF

    if [ -f "outdated-deps.txt" ]; then
        if [ -s "outdated-deps.txt" ]; then
            echo "```" >> qa-report.md
            head -10 outdated-deps.txt >> qa-report.md
            echo "```" >> qa-report.md
        else
            echo "All dependencies are up to date ✅" >> qa-report.md
        fi
    else
        echo "Report not available" >> qa-report.md
    fi

    print_success "QA report generated: qa-report.md"
}

# Individual tool functions
run_radon_cc() {
    install_qa_deps
    print_header "Radon Cyclomatic Complexity Analysis"
    $POETRY_CMD run radon cc src/ --average --show-complexity
}

run_radon_mi() {
    install_qa_deps
    print_header "Radon Maintainability Index Analysis"
    $POETRY_CMD run radon mi src/ --show
}

run_radon_raw() {
    install_qa_deps
    print_header "Radon Raw Metrics Analysis"
    $POETRY_CMD run radon raw src/ --summary
}

# Main script logic
case "${1:-help}" in
    "help"|"--help"|"-h")
        show_help
        ;;
    "security")
        run_security
        ;;
    "dependencies")
        check_dependencies
        ;;
    "complexity")
        run_complexity
        ;;
    "licenses")
        check_licenses
        ;;
    "all")
        run_all
        ;;
    "report")
        generate_report
        ;;
    "bandit")
        install_qa_deps
        run_bandit
        ;;
    "safety")
        install_qa_deps
        run_safety
        ;;
    "pip-audit")
        install_qa_deps
        run_pip_audit
        ;;
    "radon-cc")
        run_radon_cc
        ;;
    "radon-mi")
        run_radon_mi
        ;;
    "radon-raw")
        run_radon_raw
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac