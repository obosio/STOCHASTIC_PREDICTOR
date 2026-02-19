# Documentation - Implementation Branch

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
|-- pdf/                              # Compiled PDFs (auto-generated, git-ignored)
|   |-- specification/                # Generated from latex/specification/
|   |   └-- Predictor_Estocastico_*.pdf
|   └-- implementation/               # Will generate from latex/implementation/
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

- **`latex/implementation/`**: Ready for new implementation documentation (currently empty)

- **`pdf/specification/`**: Auto-generated compiled PDFs (git-ignored)

- **`pdf/implementation/`**: Will be auto-generated when `.tex` files are added

- **`compile.sh`**: Dynamic LaTeX compiler (processes ANY folder in `latex/`)

### ❌ NOT Present (Don't Create Here)

- Direct `.tex` files in `doc/` root (moved to `latex/` subfolder)
- Direct PDFs in `doc/pdf/` root (organized in phase subfolders)
- `architecture/`, `api/`, `changelog/` subdirectories

**Reason**:

- Implementation documentation lives in **code docstrings**, not separate `.tex` files
- PDFs are organized hierarchically by phase for scalability

## 📖 Implementation Documentation Strategy

Documentation for implementation is NOT in `doc/` but in:

1. **Python docstrings** (primary):

   ```python
   # stochastic_predictor/kernels/kernel_c.py
   """Itô/Lévy kernel.
   
   Theory: See doc/Predictor_Estocastico_Teoria.tex §2.3.3
   Implementation: See doc/Predictor_Estocastico_Python.tex §3
   """
   ```

2. **GitHub Issues** (architecture decisions)

3. **Commit messages** (rationale for choices)

4. **Top-level README.md** (this repo's main README)

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

**Never commit** PDFs - they're auto-generated and git-ignored.

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
