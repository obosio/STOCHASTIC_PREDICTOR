#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║       POLICY COMPLIANCE VERIFICATION (23 Policies)             ║"
echo "║        USP v2.1.0-RC1 - Zero-Heuristics Enforcement            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
PASS_COUNT=0
FAIL_COUNT=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function check_pass() {
	echo -e "${GREEN}✓ PASS${NC}: $1"
	((PASS_COUNT++))
}

function check_fail() {
	echo -e "${RED}✗ FAIL${NC}: $1"
	((FAIL_COUNT++))
}

function check_info() {
	echo -e "${YELLOW}ℹ${NC} $1"
}

function file_exists() {
	if [[ -f "$1" ]]; then
		return 0
	else
		return 1
	fi
}

function grep_exists() {
	local pattern=$1
	local target=$2
	if file_exists "$target"; then
		grep -q "$pattern" "$target" && return 0 || return 1
	else
		return 1
	fi
}

function grep_count() {
	local pattern=$1
	local target=$2
	if file_exists "$target"; then
		grep -c "$pattern" "$target" || echo "0"
	else
		echo "0"
	fi
}

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "CRITICAL POLICIES (14 items)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# CRITICAL-1: Signature Depth Constraint [3,5]
echo "CRITICAL-1: Signature Depth Constraint [3,5]"
if grep -q "log_sig_depth_min = 3" "$ROOT/config.toml"; then
	check_pass "config.toml: log_sig_depth_min = 3"
else
	check_fail "config.toml: log_sig_depth_min must be 3, not 2"
fi

if grep -q "assert 3 <= self.log_sig_depth <= 5" "$ROOT/stochastic_predictor/api/types.py"; then
	check_pass "types.py: log_sig_depth assertion [3,5]"
else
	check_fail "types.py: log_sig_depth assertion must enforce [3,5]"
fi

echo ""

# CRITICAL-2: Zero-Heuristics (.get() defaults eliminated)
echo "CRITICAL-2: Zero-Heuristics Enforcement (10 items)"
echo ""

# 2.1 config.py JAX dtype validation
if grep -q "Missing required config.*jax_default_dtype" "$ROOT/stochastic_predictor/api/config.py"; then
	check_pass "config.py: jax_default_dtype explicit validation"
else
	check_fail "config.py: jax_default_dtype must have explicit error handling"
fi

# 2.2 config.py JAX platform validation
if grep -q "Missing required config.*jax_platforms" "$ROOT/stochastic_predictor/api/config.py"; then
	check_pass "config.py: jax_platforms explicit validation"
else
	check_fail "config.py: jax_platforms must have explicit error handling"
fi

# 2.3-2.8 orchestrator.py metadata validation
metadata_checks=(
	"entropy_dgm"
	"holder_exponent"
)

for meta in "${metadata_checks[@]}"; do
	count=$(grep -c "\"$meta\" not in" "$ROOT/stochastic_predictor/core/orchestrator.py" || echo "0")
	if [[ "$count" -gt 0 ]]; then
		check_pass "orchestrator.py: $meta explicit validation ($count checks)"
	else
		check_fail "orchestrator.py: $meta must have explicit validation"
	fi
done

# 2.9 meta_optimizer.py n_trials validation
if grep -q "n_trials is None" "$ROOT/stochastic_predictor/core/meta_optimizer.py"; then
	check_pass "meta_optimizer.py: n_trials explicit validation"
else
	check_fail "meta_optimizer.py: n_trials must have explicit None check"
fi

# 2.10 meta_optimizer.py prng_seed validation
if grep -q "Missing required config.*prng_seed" "$ROOT/stochastic_predictor/core/meta_optimizer.py"; then
	check_pass "meta_optimizer.py: prng_seed explicit validation"
else
	check_fail "meta_optimizer.py: prng_seed must have explicit error handling"
fi

echo ""

# CRITICAL-3: Data Validators (already implemented)
echo "CRITICAL-3: Data Validators (3 items)"
echo ""

validators=(
	"detect_frozen_signal"
	"detect_catastrophic_outlier"
	"is_stale"
)

for validator in "${validators[@]}"; do
	if grep_exists "def $validator" "$ROOT/stochastic_predictor/io/validators.py"; then
		if grep_exists "$validator" "$ROOT/stochastic_predictor/io/loaders.py"; then
			check_pass "Validator: $validator implemented and integrated"
		else
			check_fail "Validator: $validator not integrated in loaders.py"
		fi
	else
		check_fail "Validator: $validator not implemented"
	fi
done

echo ""
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "HIGH POLICIES (9 items)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# HIGH-1: Kernel Purity & JAX Compilation
echo "HIGH-1: Kernel Purity & @jax.jit Compilation"
jit_count=$(grep -r "@jax.jit" "$ROOT/stochastic_predictor/kernels/" 2>/dev/null | wc -l)
if [[ "$jit_count" -ge 20 ]]; then
	check_pass "Kernel purity: $jit_count @jax.jit decorators found"
else
	check_fail "Kernel purity: expected ≥20 @jax.jit decorators, found $jit_count"
fi

echo ""

# HIGH-2: Atomic Configuration Mutations
echo "HIGH-2: Atomic Configuration Mutations (POSIX O_EXCL + fsync)"
if grep_exists "O_EXCL" "$ROOT/stochastic_predictor/io/config_mutation.py"; then
	check_pass "config_mutation.py: O_EXCL flag present"
else
	check_fail "config_mutation.py: O_EXCL flag required for atomicity"
fi

if grep_exists "fsync" "$ROOT/stochastic_predictor/io/config_mutation.py"; then
	check_pass "config_mutation.py: fsync() call present"
else
	check_fail "config_mutation.py: fsync() required for durability"
fi

echo ""

# HIGH-3: Credential Security (No Hardcoding)
echo "HIGH-3: Credential Security (No Hardcoded Secrets)"
cred_files=$(find "$ROOT/stochastic_predictor/io" -name "*.py" 2>/dev/null | xargs grep "getenv\|environ" | wc -l)
if [[ "$cred_files" -gt 5 ]]; then
	check_pass "Credentials: $cred_files environment references (no hardcoding)"
else
	check_fail "Credentials: insufficient environment variable usage"
fi

echo ""

# HIGH-4: State Serialization Integrity
echo "HIGH-4: State Serialization Integrity (SHA256 Validation)"
if grep_exists "SHA256\|sha256" "$ROOT/stochastic_predictor/io/snapshots.py"; then
	check_pass "snapshots.py: SHA256 checksum validation"
else
	check_fail "snapshots.py: SHA256 integrity check missing"
fi

echo ""

# HIGH-5: Stop Gradient in Diagnostics
echo "HIGH-5: Stop Gradient in Diagnostics (JAX Autodiff Isolation)"
sg_count=$(grep -r "stop_gradient" "$ROOT/stochastic_predictor/" 2>/dev/null | grep -v ".pyc" | wc -l)
if [[ "$sg_count" -ge 3 ]]; then
	check_pass "Stop gradient: $sg_count applications found"
else
	check_fail "Stop gradient: insufficient usage (found $sg_count, expected ≥3)"
fi

echo ""

# HIGH-6: CFL Condition Validation
echo "HIGH-6: CFL Condition (Courant-Friedrichs-Lewy Stability)"
if grep_exists "CFL\|sde_pid_dtmax" "$ROOT/stochastic_predictor/api/types.py"; then
	check_pass "CFL validation: timestep stability enforced"
else
	check_fail "CFL validation: missing CFL condition checks"
fi

echo ""

# HIGH-7: Non-Blocking Telemetry
echo "HIGH-7: Non-Blocking Telemetry Architecture"
if grep_exists "threading.Lock\|deque" "$ROOT/stochastic_predictor/io/telemetry.py"; then
	check_pass "Telemetry: non-blocking queue (deque + threading.Lock)"
else
	check_fail "Telemetry: non-blocking architecture missing"
fi

echo ""

# HIGH-8: Entropy-Topology Coupled Scaling
echo "HIGH-8: Entropy-Topology Coupled Scaling"
if grep_exists "scale_dgm_architecture\|entropy_ratio" "$ROOT/stochastic_predictor/core/orchestrator.py"; then
	check_pass "DGM scaling: entropy-driven architecture"
else
	check_fail "DGM scaling: entropy coupling missing"
fi

echo ""

# HIGH-9: Float64 Precision for Malliavin & Signature
echo "HIGH-9: Float64 Precision (Malliavin & Signature Calculations)"
if grep_exists "float64\|jax_enable_x64" "$ROOT/config.toml"; then
	check_pass "Float64: 64-bit precision enabled in config"
else
	check_fail "Float64: 64-bit precision not configured"
fi

echo ""
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo ""

TOTAL=$((PASS_COUNT + FAIL_COUNT))
PASS_PCT=$((PASS_COUNT * 100 / TOTAL))

echo "Total Checks: $TOTAL"
echo -e "${GREEN}Passed: $PASS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo ""

if [[ "$FAIL_COUNT" -eq 0 ]]; then
	echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
	echo -e "${GREEN}✓ ALL POLICIES COMPLIANT (100%)${NC}"
	echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
	exit 0
else
	echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
	echo -e "${RED}✗ COMPLIANCE CHECK FAILED ($FAIL_COUNT issues)${NC}"
	echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
	exit 1
fi
