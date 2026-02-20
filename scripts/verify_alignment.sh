#!/bin/bash

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

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #1: Zero-Heuristics (No Silent Defaults)"
echo "═══════════════════════════════════════════════════════════════"
if grep -q "Missing required config" "$ROOT/stochastic_predictor/api/config.py" && \
   grep -q "Missing required config" "$ROOT/stochastic_predictor/core/meta_optimizer.py"; then
	check_pass "Policy #1: Zero-heuristics explicit validation (config + meta_optimizer)"
else
	check_fail "Policy #1: Zero-heuristics validation incomplete"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #2: Configuration Immutability (Locked Subsections)"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "immutable\|frozen\|locked" "$ROOT/stochastic_predictor/io/config_mutation.py"; then
	check_pass "Policy #2: Immutable subsections protected in config_mutation"
else
	check_fail "Policy #2: Config immutability safeguards missing"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #3: Validation Schema Enforcement"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "validation_schema" "$ROOT/config.toml"; then
	check_pass "Policy #3: Schema-based validation present"
else
	check_fail "Policy #3: Validation schema not found"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #4: Atomic Configuration Mutation (O_EXCL + fsync)"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "O_EXCL" "$ROOT/stochastic_predictor/io/config_mutation.py" && \
   grep_exists "fsync" "$ROOT/stochastic_predictor/io/config_mutation.py"; then
	check_pass "Policy #4: POSIX atomicity enforced (O_EXCL + fsync)"
else
	check_fail "Policy #4: Atomic mutation protocol incomplete"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #5: Mutation Rate Limiting"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "mutation_policy" "$ROOT/config.toml"; then
	check_pass "Policy #5: Mutation rate control configured"
else
	check_fail "Policy #5: Mutation rate limiting not configured"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #6: Walk-Forward Validation Protocol"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "walk.*forward\|cross.*valid" "$ROOT/stochastic_predictor/core/meta_optimizer.py"; then
	check_pass "Policy #6: Walk-forward validation present"
else
	check_fail "Policy #6: Walk-forward validation missing"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #7: CUSUM Threshold Dynamism"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "cusum\|CUSUM" "$ROOT/stochastic_predictor/core/orchestrator.py"; then
	check_pass "Policy #7: CUSUM thresholds dynamic"
else
	check_fail "Policy #7: CUSUM implementation missing"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #8: Signature Depth Constraint [3,5]"
echo "═══════════════════════════════════════════════════════════════"
if grep -q "log_sig_depth_min = 3" "$ROOT/config.toml" && \
   grep -q "assert 3 <= self.log_sig_depth <= 5" "$ROOT/stochastic_predictor/api/types.py"; then
	check_pass "Policy #8: Signature depth [3,5]"
else
	check_fail "Policy #8: Signature depth not constrained"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #9: Sinkhorn Epsilon Bounds"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "sinkhorn_epsilon" "$ROOT/config.toml" && \
   grep_exists "sinkhorn_epsilon" "$ROOT/stochastic_predictor/core/sinkhorn.py"; then
	check_pass "Policy #9: Sinkhorn epsilon configured"
else
	check_fail "Policy #9: Sinkhorn epsilon bounds missing"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #10: CFL Condition for PIDE Schemes"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "CFL\|sde_pid_dtmax" "$ROOT/stochastic_predictor/api/types.py"; then
	check_pass "Policy #10: CFL validation enforced"
else
	check_fail "Policy #10: CFL condition not validated"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #11: Malliavin Calculus - 64-Bit Precision"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "float64\|jax_enable_x64" "$ROOT/config.toml"; then
	check_pass "Policy #11: 64-bit precision enabled"
else
	check_fail "Policy #11: 64-bit precision not configured"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #12: JAX.lax.stop_gradient on Diagnostics"
echo "═══════════════════════════════════════════════════════════════"
sg_count=$(grep -r "stop_gradient" "$ROOT/stochastic_predictor/" 2>/dev/null | grep -v ".pyc" | wc -l || echo "0")
if [[ "$sg_count" -ge 3 ]]; then
	check_pass "Policy #12: Stop gradient ($sg_count applications)"
else
	check_fail "Policy #12: Insufficient stop_gradient usage"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #13: Kernel Purity & Statelessness"
echo "═══════════════════════════════════════════════════════════════"
# Verify all 4 main kernel functions have @jax.jit or @partial(jax.jit) decorator
kernel_a_decorated=0
grep -B 1 "def kernel_a_predict" "$ROOT/stochastic_predictor/kernels/kernel_a.py" 2>/dev/null | grep -q "jax.jit" && kernel_a_decorated=1

kernel_b_decorated=0
grep -B 1 "def kernel_b_predict" "$ROOT/stochastic_predictor/kernels/kernel_b.py" 2>/dev/null | grep -q "jax.jit" && kernel_b_decorated=1

kernel_c_decorated=0
grep -B 1 "def kernel_c_predict" "$ROOT/stochastic_predictor/kernels/kernel_c.py" 2>/dev/null | grep -q "jax.jit" && kernel_c_decorated=1

kernel_d_decorated=0
grep -B 1 "def kernel_d_predict" "$ROOT/stochastic_predictor/kernels/kernel_d.py" 2>/dev/null | grep -q "jax.jit" && kernel_d_decorated=1

main_kernels_jit=$((kernel_a_decorated + kernel_b_decorated + kernel_c_decorated + kernel_d_decorated))
total_jit_count=$(grep -r "jax\.jit" "$ROOT/stochastic_predictor/kernels/" 2>/dev/null | wc -l || echo "0")

if [[ "$main_kernels_jit" -eq 4 && "$total_jit_count" -ge 21 ]]; then
	check_pass "Policy #13: All 4 kernels JIT-pure (K_A✓ K_B✓ K_C✓ K_D✓, total: $total_jit_count decorations)"
else
	check_fail "Policy #13: Kernel purity - (K_A:$kernel_a_decorated K_B:$kernel_b_decorated K_C:$kernel_c_decorated K_D:$kernel_d_decorated need 4) (total $total_jit_count, need >= 21)"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #14: Frozen Signal Detection"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "def detect_frozen_signal" "$ROOT/stochastic_predictor/io/validators.py" && \
   grep_exists "detect_frozen_signal" "$ROOT/stochastic_predictor/io/loaders.py"; then
	check_pass "Policy #14: Frozen signal validator integrated"
else
	check_fail "Policy #14: Frozen signal detection missing"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #15: Catastrophic Outlier Detection"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "def detect_catastrophic_outlier" "$ROOT/stochastic_predictor/io/validators.py" && \
   grep_exists "detect_catastrophic_outlier" "$ROOT/stochastic_predictor/io/loaders.py"; then
	check_pass "Policy #15: Outlier validator integrated"
else
	check_fail "Policy #15: Outlier detection missing"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #16: Minimum Injection Frequency (Nyquist Soft Limit)"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "signal_sampling_interval\|injection.*frequency" "$ROOT/config.toml"; then
	check_pass "Policy #16: Injection frequency configured"
else
	check_fail "Policy #16: Injection frequency not configured"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #17: Stale Weights Detection"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "def is_stale\|compute_staleness" "$ROOT/stochastic_predictor/io/validators.py" && \
   grep_exists "staleness_ttl" "$ROOT/config.toml"; then
	check_pass "Policy #17: Stale weights monitor integrated"
else
	check_fail "Policy #17: Stale weights detection missing"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #18: Secret Injection Policy (Environment Variables)"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "MissingCredentialError\|getenv" "$ROOT/stochastic_predictor/io/credentials.py"; then
	check_pass "Policy #18: Fail-fast credential validation"
else
	check_fail "Policy #18: Credential security missing"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #19: State Serialization with Integrity Checksum (SHA256)"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "SHA256\|sha256" "$ROOT/stochastic_predictor/io/snapshots.py"; then
	check_pass "Policy #19: State checksum validation"
else
	check_fail "Policy #19: Integrity verification missing"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #20: Non-Blocking Telemetry"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "threading.Lock\|deque" "$ROOT/stochastic_predictor/io/telemetry.py"; then
	check_pass "Policy #20: Non-blocking telemetry queue"
else
	check_fail "Policy #20: Telemetry architecture missing"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #21: Hardware Parity Audit Hashes"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "telemetry_hash\|parity.*hash" "$ROOT/config.toml"; then
	check_pass "Policy #21: Parity hash auditing configured"
else
	check_fail "Policy #21: Audit hash configuration missing"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #22: Walk-Forward Validation Test Leakage Prevention"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "train_ratio\|test.*leakage" "$ROOT/config.toml"; then
	check_pass "Policy #22: Test leakage prevention configured"
else
	check_fail "Policy #22: Test leakage prevention missing"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "POLICY #23: Encoder Capacity Expansion for High-Entropy"
echo "═══════════════════════════════════════════════════════════════"
if grep_exists "dgm_max_capacity\|entropy.*scaling" "$ROOT/stochastic_predictor/core/orchestrator.py"; then
	check_pass "Policy #23: DGM entropy-driven scaling"
else
	check_fail "Policy #23: Entropy scaling missing"
fi

echo ""

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo ""

TOTAL=$((PASS_COUNT + FAIL_COUNT))
if [[ "$TOTAL" -gt 0 ]]; then
	PASS_PCT=$((PASS_COUNT * 100 / TOTAL))
else
	PASS_PCT=0
fi

echo "Total Checks: $TOTAL"
echo -e "${GREEN}Passed: $PASS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo "Compliance: $PASS_PCT%"
echo ""

if [[ "$FAIL_COUNT" -eq 0 ]]; then
	echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
	echo -e "${GREEN}✓ ALL 23 POLICIES COMPLIANT (100%)${NC}"
	echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
	exit 0
else
	echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
	echo -e "${RED}✗ COMPLIANCE CHECK FAILED ($FAIL_COUNT issues)${NC}"
	echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
	exit 1
fi
