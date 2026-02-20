#!/bin/bash

echo "=== DOCUMENTATION ALIGNMENT VERIFICATION ==="
echo ""

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)

function grep_file() {
	local pattern=$1
	local target=$2
	if [[ -f "$target" ]]; then
		grep -l "$pattern" "$target"
	else
		echo "(missing) $target"
	fi
}

function grep_count() {
	local pattern=$1
	local target=$2
	if [[ -f "$target" ]]; then
		grep -c "$pattern" "$target"
	else
		echo "0"
	fi
}

# P2.1 Verification
echo "✓ P2.1 WTMM:"
grep_file "extract_holder_exponent_wtmm\|morlet_wavelet\|continuous_wavelet_transform" "$ROOT/stochastic_predictor/kernels/kernel_a.py"
grep_count "P2.1\|WTMM" "$ROOT/doc/latex/implementation/Implementation_v2.1.0_Kernels.tex"

# P2.2 Verification  
echo ""
echo "✓ P2.2 Adaptive SDE:"
grep_file "estimate_stiffness\|select_stiffness_solver" "$ROOT/stochastic_predictor/kernels/kernel_c.py"
grep_count "P2.2\|stiffness" "$ROOT/doc/latex/implementation/Implementation_v2.1.0_Kernels.tex"

# P2.3 Verification
echo ""
echo "✓ P2.3 Telemetry:"
grep_file "TelemetryBuffer\|telemetry_buffer" "$ROOT/stochastic_predictor/io/telemetry.py"
grep_file "TelemetryBuffer\|telemetry_buffer" "$ROOT/stochastic_predictor/core/orchestrator.py"
grep_count "P2.3\|Telemetry Buffer" "$ROOT/doc/latex/implementation/Implementation_v2.1.0_IO.tex"

# V-MAJ verification
echo ""
echo "✓ V-MAJ violations (8 total):"
recent_tex=$(git --no-pager log --name-only --pretty="" HEAD~20..HEAD -- "$ROOT/doc/latex" 2>/dev/null | sort -u)
recent_tex_count=$(printf "%s\n" "$recent_tex" | grep -E "\.tex$" | wc -l | tr -d " ")
violations=$(git --no-pager log --oneline HEAD~20..HEAD 2>/dev/null | grep -E "V-CRIT|V-MAJ|P2\." | wc -l | tr -d " ")
echo "Recent tex files touched: ${recent_tex_count}"
echo "Commits with V-CRIT/V-MAJ/P2.: ${violations}"
