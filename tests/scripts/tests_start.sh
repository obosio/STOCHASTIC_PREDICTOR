#!/bin/bash
################################################################################
# TESTS_START - Universal Test Suite Orchestrator
################################################################################
#
# Sequentially runs all validation and test scripts for the Universal
# Stochastic Predictor project. Execution scope: /Python/ and subdirectories.
#
# Exit immediately on first failure (fail-fast strategy).
#
# USAGE:
#   ./tests_start.sh [OPTION]
#
# OPTIONS:
#   --help                Show this help message
#   --all                 Run all tests (default)
#   --compliance          Run only policy compliance check
#   --coverage            Run only structural coverage validation
#   --execute             Run only code structure execution tests
#   --scope PYTHON/path   Set custom scope for coverage/execution tests
#
# EXAMPLES:
#   ./tests_start.sh                          # Run all in order: compliance → coverage → execute
#   ./tests_start.sh --compliance             # Run only policy compliance check
#   ./tests_start.sh --coverage --execute     # Run coverage then execution (skip compliance)
#   ./tests_start.sh --scope Python/api       # Run coverage/execute with custom Python/api scope
#
# EXIT CODES:
#   0  All tests passed
#   1  Test failure (stopped at first failure)
#   2  Invalid argument
#   127 Command not found
#
################################################################################

set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TESTS_DIR="${SCRIPT_DIR}"
RESULTS_DIR="${PROJECT_ROOT}/tests/results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Default options
RUN_COMPLIANCE=true
RUN_COVERAGE=true
RUN_EXECUTE=true
CUSTOM_SCOPE=""

################################################################################
# Functions
################################################################################

show_help() {
    grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# //'
}

print_header() {
    local title="$1"
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}${title}${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_failure() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

run_test() {
    local test_name="$1"
    local script_path="$2"
    
    print_header "RUNNING: ${test_name}"
    
    if [ ! -f "$script_path" ]; then
        print_failure "Script not found: $script_path"
        return 127
    fi
    
    # Run the test
    cd "$PROJECT_ROOT" || return 1
    python3 "$script_path"
    local exit_code=$?
    
    # Check result
    if [ $exit_code -eq 0 ]; then
        print_success "${test_name} - PASSED"
        return 0
    else
        print_failure "${test_name} - FAILED (exit code: $exit_code)"
        return $exit_code
    fi
}

print_summary() {
    local total=$1
    local passed=$2
    local failed=$3
    
    print_header "TEST SUMMARY"
    
    echo -e "Total:  ${total}"
    echo -e "${GREEN}Passed: ${passed}${NC}"
    if [ $failed -gt 0 ]; then
        echo -e "${RED}Failed: ${failed}${NC}"
    fi
    
    # List latest artifacts
    echo ""
    echo "📂 Latest artifacts in ${RESULTS_DIR}:"
    if ls "${RESULTS_DIR}"/*.json 1>/dev/null 2>&1; then
        ls -1t "${RESULTS_DIR}"/*.json | head -3 | sed 's|.*/||' | sed 's/^/   - /'
    else
        echo "   (no artifacts yet)"
    fi
    
    if [ $failed -eq 0 ]; then
        echo ""
        print_success "ALL TESTS PASSED!"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
        return 0
    else
        echo ""
        print_failure "${failed} test suite(s) failed"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
        return 1
    fi
}

################################################################################
# Main
################################################################################

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --all)
            RUN_COMPLIANCE=true
            RUN_COVERAGE=true
            RUN_EXECUTE=true
            shift
            ;;
        --compliance)
            RUN_COMPLIANCE=true
            RUN_COVERAGE=false
            RUN_EXECUTE=false
            shift
            ;;
        --coverage)
            RUN_COMPLIANCE=false
            RUN_COVERAGE=true
            RUN_EXECUTE=false
            shift
            ;;
        --execute)
            RUN_COMPLIANCE=false
            RUN_COVERAGE=false
            RUN_EXECUTE=true
            shift
            ;;
        --scope)
            RUN_COMPLIANCE=false
            CUSTOM_SCOPE="$2"
            shift 2
            ;;
        *)
            print_failure "Unknown option: $1"
            show_help
            exit 2
            ;;
    esac
done

# If --compliance and --coverage both specified, allow both
# (parse_args above sets defaults, so we need to handle combination)
# For now, this simple logic works for the documented use cases

# Initialize counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

print_header "UNIVERSAL STOCHASTIC PREDICTOR - TEST ORCHESTRATOR"
echo "Execution Scope: /Python/ and subdirectories"
echo "Test Execution Strategy: Sequential with fail-fast"
echo ""

# Test 1: Policy Compliance (repository-wide)
if [ "$RUN_COMPLIANCE" = true ]; then
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if run_test "Policy Compliance Check (code_alignement.py)" "${TESTS_DIR}/code_alignement.py"; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        print_warning "Stopping execution: compliance check failed"
        print_summary $TOTAL_TESTS $PASSED_TESTS $FAILED_TESTS
        exit 1
    fi
fi

# Test 2: Structural Coverage (Python/ scope)
if [ "$RUN_COVERAGE" = true ]; then
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if run_test "Structural Coverage Validation (tests_coverage.py)" "${TESTS_DIR}/tests_coverage.py"; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        print_warning "Stopping execution: coverage validation failed"
        print_summary $TOTAL_TESTS $PASSED_TESTS $FAILED_TESTS
        exit 1
    fi
fi

# Test 3: Code Execution (Python/ scope, pytest)
if [ "$RUN_EXECUTE" = true ]; then
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if run_test "Code Structure Execution Tests (code_structure.py)" "${TESTS_DIR}/code_structure.py"; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        print_warning "Stopping execution: code structure tests failed"
        print_summary $TOTAL_TESTS $PASSED_TESTS $FAILED_TESTS
        exit 1
    fi
fi

# Print final summary
print_summary $TOTAL_TESTS $PASSED_TESTS $FAILED_TESTS
exit $?
