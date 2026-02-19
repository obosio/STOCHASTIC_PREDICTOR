# Documentation - Implementation Branch

**Version**: v2.0.3-Phase3-Complete  
**Status**: Core Orchestration Complete ✅  
**Branch**: `implementation/base-jax`

## 📂 Current Structure

```bash
doc/                                  # Documentation root
|-- compile.sh                        # Dynamic LaTeX compiler (agnóstico de carpetas)
|-- .latexmkrc                        # LaTeX config
|-- latex/                            # Source LaTeX documents
|   |-- specification/                # Specification documents (REFERENCE ONLY)
|   |   |-- Predictor_Estocastico_*.tex   # 7 files total
|   |   └-- ...
|   └-- implementation/               # Ready for implementation docs
|
|-- pdf/                              # Compiled PDFs
|   |-- specification/                # Generated from latex/specification/
|   |   └-- Predictor_Estocastico_*.pdf
|   └-- implementation/               # Generated from latex/implementation/
|
`-- .build/                           # Build artifacts (git-ignored)
```

## 📌 Important: This is the Implementation Branch

**Location**: `doc/` in `implementation/base-jax` branch  
**Status**: Inherited specification files serve as theory reference for code development

## ⚠️ What This Directory Contains

### ✅ Present

- **`latex/specification/`**: 7 Specification `.tex` files (reference only)
  - Predictor_Estocastico_Teoria.tex
  - Predictor_Estocastico_Python.tex
  - Predictor_Estocastico_Tests_Python.tex
  - 4 more

- **`latex/implementation/`**: Implementation documentation (one file per tag)
  - **`Implementacion_v2.0.0_Bootstrap.tex`** ✨
    - Tag: `impl/v2.0.0` (commit 85abb8c)
    - Documents initial 5-layer architecture scaffold
    - Language policy, Golden Master dependencies, git workflow
    - 12KB, 381 lines
  - **`Implementacion_v2.0.1_API.tex`** ✨
    - Tag: `impl/v2.0.1` (commit 4757710)
    - API Layer Complete: Full foundational API
    - Modules: types.py (347 LoC), prng.py (301 LoC), validation.py (467 LoC), schemas.py (330 LoC), config.py (220 LoC)
    - Total 1,665 LoC (test infrastructure reserved for v3.x.x)
    - 16KB, comprehensive API documentation
  - **`Implementacion_v2.0.2_Kernels.tex`** ✨
    - Tag: `impl/v2.0.2` (created)
    - Kernels Layer Complete: Four prediction kernels (A, B, C, D)
    - Modules: base.py (217 LoC), kernel_a.py (276 LoC), kernel_b.py (331 LoC), kernel_c.py (277 LoC), kernel_d.py (217 LoC), \_\_init\_\_.py (105 LoC)
    - Total 1,423 LoC with RKHS, DGM, SDE, and Signature methods
    - All kernels JIT-compilable, stateless, with stop_gradient on diagnostics
  - **`Implementation_v2.0.3_Core.tex`** ✨
    - Tag: `impl/v2.0.3` (commit b566b2f)
    - Core Orchestration Complete: JKO/Sinkhorn fusion with volatility coupling
    - Modules: core/sinkhorn.py, core/fusion.py, core/orchestrator.py
    - Config-driven simplex validation and weight updates

- **`pdf/specification/`**: Compiled specification PDFs

- **`pdf/implementation/`**: Compiled implementation PDFs
  - `Implementacion_v2.0.0_Bootstrap.pdf` (67KB) ✅
  - `Implementacion_v2.0.1_API.pdf` (97KB) ✅
  - `Implementacion_v2.0.2_Kernels.pdf` (v2.0.2) ✅
  - `Implementation_v2.0.3_Core.pdf` (v2.0.3) ✅

- **`compile.sh`**: Dynamic LaTeX compiler (processes ANY folder in `latex/`)

### ❌ NOT Present (Don't Create Here)

- Direct `.tex` files in `doc/` root (moved to `latex/` subfolder)
- Direct PDFs in `doc/pdf/` root (organized in phase subfolders)
- `architecture/`, `api/`, `changelog/` subdirectories

**Reason**:

- Implementation documentation lives in **code docstrings**, not separate `.tex` files
- PDFs are organized hierarchically by phase for scalability

## 📖 Implementation Documentation Strategy

Implementation documentation uses a **tiered approach**:

1. **Formal LaTeX documentation** (for major phases):
   - `Implementacion_Phase1_Bootstrap.tex` - Phase 1 & Bootstrap complete write-up
   - Each major phase gets a corresponding `.tex` file documenting:
     - Architecture decisions
     - Code structure and key algorithms
     - Verification metrics and QA results

2. **Python docstrings** (function-level documentation):

   ```python
   # stochastic_predictor/kernels/kernel_c.py
   """Itô/Lévy kernel.
   
   Theory: See doc/Predictor_Estocastico_Teoria.tex §2.3.3
   Implementation details: See doc/latex/implementation/Implementacion_Phase*.tex
   """
   ```

3. **GitHub Issues** (architecture decisions & discussions)

4. **Commit messages** (rationale for individual changes)

## 🔗 Referencing Specification from Code

When implementing a feature, always reference the specifications:

```python
def estimate_holder_exponent(ts):
    """Estimate Hölder exponents using WTMM.
    
    Reference:
        Mathematical foundation: doc/Predictor_Estocastico_Teoria.tex §3.2
        Implementation guide: doc/Predictor_Estocastico_Python.tex §2.3
    """
```

This creates a trace from code back to theory.

## 🧹 Compilation

The `compile.sh` script is **dynamic** - it automatically detects and compiles ANY folder in `latex/`:

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
3. **Generates**: Compiles `.tex` → `.pdf` in proper locations
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

## 📋 Branch Strategy

| Branch                    | Content     | Editable                   |
| ------------------------- | ----------- | -------------------------- |
| `main`                    | Spec only   | ❌ No (immutable v1.0.0)   |
| `implementation/base-jax` | Spec + Code | ✅ Code yes, Spec ref only |

This `doc/` folder in `implementation/base-jax`:

- Inherited specification (read reference)
- Do NOT modify `.tex` files
- Do NOT push changes to specification

## ✨ Best Practice

Keep `doc/` minimal. Real documentation is in:

- **docstrings**: Theory ↔ code links
- **README files**: Architecture overview
- **Comments**: Why (not what - code shows that)

---

**Status**: ✅ Clean reference, ready for implementation  
**Edits**: Only code changes, not documentation structure
