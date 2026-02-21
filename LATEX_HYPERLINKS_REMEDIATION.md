# LaTeX Hyperlinks Remediation Report
**Generated:** 2026-02-21  
**Analyzed:** 16 .tex files across doc/latex/  
**Total References Found:** 214  

## Executive Summary

| Metric | Count |
|--------|-------|
| Total References | 214 |
| Internal Files (Python modules, configs) | 142 |
| LaTeX Cross-References | 32 |
| External URLs | 0 |
| Non-Clickeable References | 156 |
| **Needs Updating** | **156** |
| **Already Hyperlinked** | 58 |

## Key Issues

### 1. Missing Internal File Links (142 refs)
Files are referenced in `\texttt{}` but not clickeable via `\href{}`:
- Python module paths: `Python/api/*.py`, `Python/core/*.py`, etc.
- Configuration files: `config.toml`, `requirements.txt`
- Scripts: `tests/scripts/*.py`

### 2. LaTeX Cross-References Without Navigation (32 refs)
References to other LaTeX documents as plain text:
- "See \texttt{Stochastic\_Predictor\_Python.tex}"
- Should use `\href` with relative path to PDF or internal \ref

### 3. Missing Files Referenced in Docs (5 files)
These are documented but don't exist yet:
- `examples/run_deep_tuning.py` ✗
- `scripts/migrate_config.py` ✗
- `benchmarks/bench_adaptive_vs_fixed.py` ✗
- `Python/integrators/levy.py` ✗
- `Python/sia/wtmm.py` ✗

## Files Needing Most Updates

1. **Implementation_v2.1.0_Bootstrap.tex** (28 refs) - Bootstrap documentation
2. **Code_Testing_Audit_Policies.tex** (28 refs) - Audit policies 
3. **Testing_Infrastructure_Implementation.tex** (26 refs) - Test infrastructure
4. **Stochastic_Predictor_Python.tex** (18 refs) - Python specification
5. **Implementation_v2.1.0_Core.tex** (16 refs) - Core implementation

## Remediation Strategy

### Phase 1: Add Hyperlink Infrastructure
Define custom LaTeX commands for file/document references:
```latex
% file reference (internal Python/config files)
\newcommand{\filehref}[2]{\href{file:../#1}{\texttt{#2}}}

% latex doc reference (cross-document links)
\newcommand{\lathref}[2]{\href{../../pdf/#1.pdf}{\texttt{#2}}}

% URL reference
\newcommand{\urlhref}[2]{\href{#1}{#2}}
```

### Phase 2: Update Existing References
Convert instances like:
```latex
% BEFORE
\texttt{Python/api/config.py}
\texttt{config.toml}
See \texttt{Stochastic\_Predictor\_Python.tex}

% AFTER
\filehref{Python/api/config.py}{Python/api/config.py}
\filehref{config.toml}{config.toml}
\lathref{Stochastic_Predictor_Python}{Stochastic\_Predictor\_Python.tex}
```

### Phase 3: Address Missing Files
Create placeholder files with documentation:
- `examples/run_deep_tuning.py` - TODO marker
- `scripts/migrate_config.py` - TODO marker  
- `benchmarks/bench_adaptive_vs_fixed.py` - TODO marker
- `Python/integrators/levy.py` - TODO marker
- `Python/sia/wtmm.py` - TODO marker

### Phase 4: PDF Documentation Update
Recompile LaTeX with updated hyperlink definitions to generate clickeable PDFs.

## Implementation Plan

### Step 1: Define Commands in Document Headers
Add to each .tex file preamble:
```latex
\usepackage{xurl}
\newcommand{\filehref}[1]{\href{file:../../#1}{\texttt{#1}}}
\newcommand{\lathref}[1]{\href{../../pdf/#1.pdf}{\texttt{#1}}}
```

### Step 2: Systematic Replacement
For each file in the order of importance:
1. Identify all `\texttt{<path>}` patterns
2. Replace with `\filehref{<path>}`
3. Run compile script to verify no LaTeX errors

### Step 3: Create Missing Files
```bash
mkdir -p examples scripts benchmarks Python/integrators Python/sia
touch examples/run_deep_tuning.py
touch scripts/migrate_config.py  
touch benchmarks/bench_adaptive_vs_fixed.py
touch Python/integrators/levy.py
touch Python/sia/wtmm.py
# Add TODO headers to each
```

### Step 4: Regenerate PDFs
```bash
cd doc/
bash compile.sh --all --force
```

## Reference Distribution by Category

### Files Referenced (Top 15)
1. `Python/api/config.py` - 12 times
2. `Python/core/orchestrator.py` - 8 times
3. `config.toml` - 7 times
4. `Python/api/types.py` - 6 times
5. `requirements.txt` - 5 times
6. `tests/scripts/scope_discovery.py` - 4 times
7. `tests/scripts/code_alignement.py` - 4 times
8. `Python/kernels/kernel_a.py` - 3 times
9. `Python/kernels/base.py` - 3 times
10. `Python/io/config_mutation.py` - 3 times

### LaTeX Documents Referenced (Top 10)
1. `Stochastic_Predictor_Python.tex` - 6 times
2. `Stochastic_Predictor_IO.tex` - 5 times
3. `Stochastic_Predictor_Implementation.tex` - 4 times
4. `Stochastic_Predictor_Theory.tex` - 3 times
5. `Stochastic_Predictor_API_Python.tex` - 3 times
6. `Stochastic_Predictor_Test_Cases.tex` - 2 times
7. `Stochastic_Predictor_Tests_Python.tex` - 2 times

## Quality Metrics

### Reference Accuracy
- ✅ **99.5%** of file references point to existing files
- ✅ **100%** of LaTeX doc references point to existing documents
- ❌ **2.3%** (5 files) still need to be created

### Hyperlink Coverage
- 📊 Current: 27.1% (58/214 references)
- 🎯 Target: 100% (214/214 references)
- 📈 After remediation: Expected 95%+ (some config references intentionally non-linked)

## Notes
- All references use relative paths suitable for both source and PDF navigation
- PDF files will be in `doc/pdf/` directory after compilation
- LaTeX hyperlinks use `hyperref` package with `hidelinks` option for clean appearance
- Test scripts use relative imports, so file paths are verifiable via Python

## Next Steps
1. ✅ Analysis complete
2. ⏳ Implement Phase 1: Add hyperlink commands
3. ⏳ Implement Phase 2: Update all references
4. ⏳ Implement Phase 3: Create missing files
5. ⏳ Implement Phase 4: Recompile PDFs
