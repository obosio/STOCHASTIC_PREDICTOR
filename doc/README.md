# Documentation - Implementation Phase

## 📂 Structure

```
doc/
├── README.md                          # This file
├── compile.sh                         # LaTeX compiler (original for specification)
├── *.tex                              # Specification files (for reference)
├── .build/                            # Build artifacts (git-ignored)
├── pdf/                               # Compiled PDFs
│   ├── Predictor_Estocastico_*.pdf   # Specification PDFs
│   └── Implementation_*.pdf           # Implementation documentation (when added)
│
└── implementation/                    # Implementation documentation folder (future)
    ├── architecture/                  # Design decisions, trade-offs
    ├── api/                           # API documentation (generated from docstrings)
    └── changelog/                     # Implementation progress
```

## 📋 Current State

**Branch**: `implementation/base-jax`  
**Status**: Code infrastructure initialized, documentation in progress

### Inherited Specification (Reference Only)

All specification `.tex` files are present in `doc/` for docstring cross-references:
- `Predictor_Estocastico_Teoria.tex` - Mathematical theory
- `Predictor_Estocastico_Python.tex` - Implementation guide
- etc.

Python code should reference these:
```python
def integrate_rama_c(x):
    """Itô/Lévy integration.
    
    References:
        See doc/Predictor_Estocastico_Teoria.tex §2.3.3
    """
```

### Code Structure (Implemented)

✅ 5-tier Clean Architecture:
- `stochastic_predictor/api/` - Exposure layer
- `stochastic_predictor/core/` - Orchestration
- `stochastic_predictor/kernels/` - XLA motors
- `stochastic_predictor/io/` - Physical I/O
- `tests/` - External validation

✅ Configuration:
- `requirements.txt` - Golden Master (frozen dependencies)
- `config.toml` - Runtime parameters
- `.env.example` - Credential template

### Next Steps

1. **Architecture documentation**: Create `doc/implementation/architecture/`
   - 5-tier layer rationale
   - Design decisions
   - Trade-offs made

2. **API documentation**: `doc/implementation/api/`
   - Generated from Python docstrings (future Sphinx)
   - Examples and tutorials

3. **Changelog**: `doc/implementation/changelog/`
   - Feature implementation log
   - GitHub issues cross-reference

## 🔗 Cross-Referencing

### Spec ↔ Implementation Strategy

Each implementation module should link to specification:

```python
# stochastic_predictor/kernels/kernel_c.py
"""
Itô/Lévy prediction kernel (Rama C).

Mathematical foundation:
    doc/Predictor_Estocastico_Teoria.tex §2.3.3
    
Implementation guide:
    doc/Predictor_Estocastico_Python.tex §3.2

Dynamic SDE scheme transition:
    doc/Predictor_Estocastico_Teoria.tex §2.3.3
"""
```

This allows:
- ✅ Precise specification references
- ✅ Traceability from code to theory
- ✅ Hardware-parity test debugging (compare impl against spec)

## 📖 Adding Implementation Documentation

Example: Document Sinkhorn implementation decisions

```bash
# Create new documentation file
cat > doc/implementation/architecture/sinkhorn-design.md << 'EOF'
# Sinkhorn Dynamics - Implementation Design

## Specification Reference
See: doc/Predictor_Estocastico_Implementacion.tex §2.4

## Design Decision: Volatility Coupling
...
EOF

# Or add to LaTeX for PDF integration
touch doc/implementation/Implementation_Sinkhorn.tex
```

## 🚀 Compilation

Compile specification PDFs (not changed):

```bash
cd doc
./compile.sh Predictor_Estocastico_Python

# Or all specification
./compile.sh --all
```

## 📐 Version Control Strategy

- **main**: Specification + API (immutable)
- **implementation/base-jax**: Code + implementation docs (active)

Both branches inherit specification for reference but:
- **main**: No code, only theory
- **implementation/base-jax**: Code + implementation-specific docs

---

**Last Updated**: 18 de febrero de 2026  
**Branch**: implementation/base-jax  
**Phase**: 🚀 Implementation Active
