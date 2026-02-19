# Documentation - Implementation Branch

**Version**: v2.2.0
**Status**: Testing Phase
**Branch**: `implementation/base-jax`

## Current Structure

```bash
doc/                                  # Documentation root
|-- compile.sh                        # Dynamic LaTeX compiler (folder-agnostic)
|-- .latexmkrc                        # LaTeX config
|-- latex/                            # Source LaTeX documents
|   |-- specification/                # Specification documents (REFERENCE ONLY)
|   |   |-- Stochastic_Predictor_*.tex # 7 files total
|   |   └-- ...
|   └-- implementation/               # Implementation docs
|
|-- pdf/                              # Compiled PDFs
|   |-- specification/                # Generated from latex/specification/
|   |   └-- Stochastic_Predictor_*.pdf
|   └-- implementation/               # Generated from latex/implementation/
|
`-- .build/                           # Build artifacts (git-ignored)
```

## Important: This is the Implementation Branch

**Location**: `doc/` in `implementation/base-jax` branch
**Status**: Inherited specification files serve as theory reference for code development

## What This Directory Contains

### Present

- **`latex/specification/`**: 7 specification `.tex` files (reference only)
  - Stochastic_Predictor_Theory.tex
  - Stochastic_Predictor_Python.tex
  - Stochastic_Predictor_Tests_Python.tex
  - 4 more

- **`latex/implementation/`**: Implementation documentation (one file per tag)
  - **`Implementation_v2.0.0_Bootstrap.tex`**
    - Tag: `impl/v2.0.0` (commit 85abb8c)
    - Documents initial 5-layer architecture scaffold
    - Language policy, Golden Master dependencies, git workflow
    - 12KB, 381 lines
  - **`Implementation_v2.0.1_API.tex`**
    - Tag: `impl/v2.0.1` (commit 4757710)
    - API Layer Complete: full foundational API
    - Modules: types.py (347 LoC), prng.py (301 LoC), validation.py (467 LoC), schemas.py (330 LoC), config.py (220 LoC)
    - State buffer stats use stop_gradient to reduce VRAM
    - Total 1,665 LoC (test infrastructure reserved for v3.x.x)
    - 16KB, comprehensive API documentation
  - **`Implementation_v2.0.2_Kernels.tex`**
    - Tag: `impl/v2.0.2` (created)
    - Kernels Layer Complete: four prediction kernels (A, B, C, D)
    - Modules: base.py (217 LoC), kernel_a.py (276 LoC), kernel_b.py (331 LoC), kernel_c.py (277 LoC), kernel_d.py (217 LoC), `__init__.py` (105 LoC)
    - Total 1,423 LoC with RKHS, DGM, SDE, and signature methods
    - All kernels JIT-compilable, stateless, with stop_gradient on diagnostics
  - **`Implementation_v2.0.3_Core.tex`**
    - Tag: `impl/v2.0.3` (commit cb119d9)
    - Core Orchestration Complete: JKO/Sinkhorn fusion with volatility coupling
    - Modules: core/sinkhorn.py, core/fusion.py, core/orchestrator.py
    - Config-driven simplex validation and weight updates
  - **`Implementation_v2.0.4_IO.tex`**
    - Tag: `impl/v2.0.4` (commit 2d1b877)
    - IO Layer Complete: ingestion validation, telemetry buffering, deterministic logging, atomic snapshots
    - Modules: io/validators.py, io/loaders.py, io/telemetry.py, io/snapshots.py, io/credentials.py (~800 LoC)
    - Policies: catastrophic outlier rejection, frozen signal detection, TTL staleness, binary serialization (msgpack), hash-verified snapshots, credential injection
    - Critical Features: Zero-heuristics (no implicit defaults), 64-bit precision enforcement, layer isolation (PRNG in API)
    - Orchestrator Integration: evaluate_ingestion() gate, IngestionDecision flags, degraded mode support
  - **`test/v2.2.0`** (commit 1930a7e)
    - Testing Phase: XLA Compliance & Golden Master Enforcement
    - **Critical Fixes**: 5 blocking issues resolved
      - XLA control flow violations in Kernel C (jax.lax.cond refactoring)
      - Type system incompatibility in orchestrator PyTree
      - Tuple unpacking arity mismatch in Kernel C drift
      - Golden Master dependency violation (OTT-JAX enforcement)
      - Pydantic V2 validation migration (@field_validator)
    - **Status**: 100% VSCode error compliance, 13-point audit passed
    - **Ready for**: Comprehensive QA/testing, load testing, edge case validation
    - See [CHANGELOG.md](../CHANGELOG.md), [RELEASE_NOTES.md](../RELEASE_NOTES.md), and [TESTING.md](../TESTING.md) for details

- **`pdf/specification/`**: Compiled specification PDFs

- **`pdf/implementation/`**: Compiled implementation PDFs
  - `Implementation_v2.0.0_Bootstrap.pdf` (67KB)
  - `Implementation_v2.0.1_API.pdf` (97KB)
  - `Implementation_v2.0.2_Kernels.pdf` (v2.0.2)
  - `Implementation_v2.0.3_Core.pdf` (v2.0.3)
  - `Implementation_v2.0.4_IO.pdf` (v2.0.4)
  - **v2.1.0-RC1**: Critical maintenance release (no new PDF; see CHANGELOG.md)

- **`compile.sh`**: Dynamic LaTeX compiler (processes any folder in `latex/`)

### Not Present (Do Not Create Here)

- Direct `.tex` files in `doc/` root (moved to `latex/` subfolder)
- Direct PDFs in `doc/pdf/` root (organized in phase subfolders)
- `architecture/`, `api/`, `changelog/` subdirectories

**Reason**:

- Implementation documentation lives in code docstrings, not separate `.tex` files
- PDFs are organized hierarchically by phase for scalability

## Implementation Documentation Strategy

Implementation documentation uses a tiered approach:

1. **Formal LaTeX documentation** (for major phases)
   - Each major phase gets a corresponding `.tex` file documenting:
     - Architecture decisions
     - Code structure and key algorithms
     - Verification metrics and QA results

2. **Python docstrings** (function-level documentation)

   ```python
   # stochastic_predictor/kernels/kernel_c.py
   """Ito/Levy kernel.

   Theory: See doc/latex/specification/Stochastic_Predictor_Theory.tex Section 2.3.3
   Implementation details: See doc/latex/implementation/Implementation_v2.0.X_*.tex
   """
   ```

3. **GitHub Issues** (architecture decisions and discussions)

4. **Commit messages** (rationale for individual changes)

## Referencing Specification from Code

When implementing a feature, always reference the specifications:

```python
def estimate_holder_exponent(ts):
    """Estimate Holder exponents using WTMM.

    Reference:
        Mathematical foundation: doc/latex/specification/Stochastic_Predictor_Theory.tex Section 3.2
        Implementation guide: doc/latex/specification/Stochastic_Predictor_Python.tex Section 2.3
    """
```

This creates a trace from code back to theory.

## Compilation

The `compile.sh` script is dynamic - it automatically detects and compiles any folder in `latex/`:

```bash
cd doc
./compile.sh help              # Show help
./compile.sh --all             # Compile all documents in all latex/ folders
./compile.sh --all --force     # Force recompile (ignore timestamps)
./compile.sh <filename>.tex    # Compile specific file
./compile.sh clean             # Clean build artifacts
```

### How It Works

1. **Reads**: `latex/specification/`, `latex/implementation/`, or any folder you create
2. **Mirrors**: Creates corresponding folders in `pdf/`
3. **Generates**: Compiles `.tex` -> `.pdf` in proper locations
4. **Cleans**: Removes orphaned PDFs if source `.tex` is deleted

### Example: Adding New Documentation

```bash
# 1. Create new .tex file
mkdir -p latex/research
echo "\\documentclass{article} \\begin{document} Test \\end{document}" > latex/research/my_doc.tex

# 2. Compile (script auto-detects)
./compile.sh --all

# 3. Result: pdf/research/my_doc.pdf is created automatically
```

## Branch Strategy

| Branch | Content | Editable |
| ----- | ------- | -------- |
| `main` | Spec only | No (immutable v1.0.0) |
| `implementation/base-jax` | Spec + Code | Yes (code only, spec is reference) |

This `doc/` folder in `implementation/base-jax`:

- Inherited specification (reference only)
- Do not modify `.tex` files in `latex/specification/`
- Do not push changes to specification

## Best Practice

Keep `doc/` minimal. Real documentation is in:

- docstrings: theory -> code links
- README files: architecture overview
- comments: why (not what)

---

Status: clean reference, ready for implementation
Edits: only code changes, not documentation structure
