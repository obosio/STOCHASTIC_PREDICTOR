# Copilot Instructions: Universal Stochastic Predictor (USP)

## 1. Role & Context
You are an expert Implementation Assistant for the Universal Stochastic Predictor (USP). Your goal is to help implement a mission-critical system for unknown probability laws using JAX and Optimal Transport.

## 2. Language Policy (Mandatory - 100% English Code)
**CRITICAL RULE**: All code files MUST be written 100% in English.

**English (MANDATORY for ALL code)**:
- File names, class names, variable names, method names
- Docstrings (triple quotes)
- Inline comments (#)
- Log messages and error messages
- Configuration files (TOML, YAML, JSON)
- Requirements files and dependencies metadata
- README files and inline documentation

**Spanish (ONLY for external communication)**:
- Chat responses and conversations
- Technical specification documents (.tex, .pdf)

**Example (CORRECT)**:
```python
def compute_jko_flow(rho: Array) -> Array:
    """
    Computes the Wasserstein gradient flow using JKO scheme.
    
    This function implements the Jordan-Kinderlehrer-Otto variational
    formulation for optimal transport dynamics.
    """
    # Apply Sinkhorn regularization
    return sinkhorn_step(rho, epsilon=1e-3)
```

**Rationale**: Bit-exact reproducibility across global development environments requires linguistic homogeneity in all executable and configuration artifacts.

## 3. Architectural Constraints (Clean Architecture)
All code must reside in the specified 5-layer structure:

- **api/**: External contracts, Pydantic schemas, and global configuration.
- **core/**: Orchestration, JKO-Flow logic, and Sinkhorn dynamics.
- **kernels/**: Pure XLA/JAX stateless functions for Kernels A, B, C, and D.
- **io/**: Asynchronous atomic snapshots and stream sanitization.
- **tests/**: Hardware parity, causality, and mathematical validation.

## 4. Technical Stack & "Golden Master"
Adhere strictly to the pinned versions to ensure bit-exact parity:

- **Core**: jax==0.4.20, equinox==0.11.2, diffrax==0.4.1.
- **Specialized**: signax==0.1.4 (Signatures), ott-jax==0.4.5 (Sinkhorn).
- **Logic**: Use functional programming, JIT-compilable code, and Pytrees.

## 5. Coding Standards
- **Statelessness**: Kernels must be pure functions to maximize JIT/XLA efficiency.
- **VRAM Optimization**: Apply `jax.lax.stop_gradient()` to diagnostic modules like SIA and CUSUM.
- **Precision**: Use `jax.config.update('jax_enable_x64', True)` for Malliavin and Signature calculations.
- **Security**: No hardcoded secrets; use environment variable injection pattern.

## 6. Tone & Style
Maintain a technical, industrial-grade, and concise tone. Avoid conversational fillers like "Based on the provided files" or "I think". Provide direct, rigorous implementation proposals that match the Teoria.tex and Implementacion.tex specifications.

## 7. Pre-Commit/Push Workflow (MANDATORY)

**CRITICAL RULE**: Follow this exact sequence before EVERY `git commit` and `git push`. Non-compliance causes corrupted releases and project delays.

### **Step-by-Step Checklist**

1. **✅ Fix All VSCode Errors**
   ```bash
   get_errors()  # Must return "No errors found"
   ```
   - Fix Markdown errors (MD040, MD050, MD032, MD036, MD060)
   - Fix LaTeX errors (unescaped `_`, missing language specifiers, unicode issues)
   - Fix Python errors (type hints, imports, syntax)
   - Fix YAML/TOML errors (indentation, escaping)
   - **Never proceed if errors exist** (Reference: v1.1.0 release corruption incident)

2. **📝 Update LaTeX Documentation** (if code changes affect public API/architecture)
   ```bash
   # Edit corresponding file in doc/latex/implementation/
   # - Implementacion_v2.0.1_API.tex (API layer)
   # - Implementacion_v2.0.2_Kernels.tex (Kernels layer)
   # - Future: Implementacion_v2.0.X_*.tex for each phase
   
   # Update sections:
   # - Tag Information (add commit hashes for fixes)
   # - Code Examples (reflect CURRENT implementation)
   # - Critical Fixes Applied table (document corrections)
   # - Metrics (LoC, module counts)
   # - Compliance checklist (add ✓ for fixed issues)
   
   # Compile PDFs (only modified files)
   cd doc && ./compile.sh --all
   
   # Verify no LaTeX errors
   get_errors()
   ```
   
   **When to Update LaTeX**:
   - ✅ Critical bug fixes (e.g., config injection, type consistency)
   - ✅ Major refactors (e.g., automated field mapping)
   - ✅ Phase milestone completion
   - ✅ Public API contract changes
   - ❌ Minor internal refactors (no public impact)

3. **📖 Update README Files** (before creating tags only)
   ```bash
   # Only required when creating impl/vX.Y.Z tags
   # Update:
   # - README.md (root): version table, phase status
   # - doc/README.md: documentation structure
   # - Layer READMEs (if exist): API descriptions, examples
   
   # Checklist:
   # - [ ] Current version/tag referenced
   # - [ ] Phase status accurate (Implementation vs Specification)
   # - [ ] Code examples up-to-date
   # - [ ] No broken links
   # - [ ] 100% English (no Spanish)
   
   get_errors()  # Verify Markdown compliance
   ```

4. **💾 Stage, Commit, Push**
   ```bash
   # Stage code changes
   git add <files>
   
   # If LaTeX updated: stage docs too
   git add doc/latex/implementation/*.tex doc/pdf/implementation/*.pdf
   
   # If READMEs updated: stage them
   git add README.md doc/README.md
   
   # Commit with meaningful message
   git commit -m "type(scope): description
   
   - Detail 1
   - Detail 2
   
   Fixes: #issue (if applicable)
   Refs: commit_hash (if docs updated for specific fix)"
   
   # Push
   git push origin <branch>
   
   # Verify push successful
   ```

5. **🏷️ Create Tag** (only for phase milestones)
   ```bash
   # Only after steps 1-4 complete AND READMEs updated
   git tag impl/vX.Y.Z -m "Implementation vX.Y.Z: <Description>"
   git push origin impl/vX.Y.Z
   ```

### **Quick Reference: What to Update When**

| Change Type | VSCode Errors | LaTeX Docs | READMEs | Example |
|-------------|---------------|------------|---------|---------|
| Bug fix (critical) | ✅ Required | ✅ Required | ❌ No | Config injection fix (dc16b1a) |
| Bug fix (minor) | ✅ Required | ❌ No | ❌ No | Typo in log message |
| Refactor (public API) | ✅ Required | ✅ Required | ❌ No | Automated config introspection (65e4bcf) |
| Refactor (internal) | ✅ Required | ❌ No | ❌ No | Private function rename |
| New feature (phase) | ✅ Required | ✅ Required | ✅ Required | Phase 2 Kernels (a0dc577) |
| Phase milestone tag | ✅ Required | ✅ Required | ✅ Required | impl/v2.0.1, impl/v2.0.2 |

### **Common Mistakes to Avoid**

- ❌ Committing with VSCode errors present
- ❌ Updating code without updating LaTeX for architectural changes
- ❌ Creating tags before README updates
- ❌ Stale code examples in LaTeX docs
- ❌ Missing commit hashes in "Critical Fixes Applied" tables
- ❌ PDFs not regenerated after LaTeX changes
- ❌ Generic commit messages ("fix bug", "update docs")

### **LaTeX Documentation Patterns**

**Example: Documenting Critical Fix**
```latex
\section{Tag Information}
\begin{itemize}
    \item \textbf{Initial Commits}: 4757710 through 76f87c2
    \item \textbf{Critical Fixes}: dc16b1a (config injection) + 65e4bcf (automated introspection)
    \item \textbf{Status}: Complete, audited, and verified
\end{itemize}

\section{Critical Fixes Applied}
\begin{table}[h!]
\begin{tabular}{|l|l|l|}
\hline
\textbf{Issue} & \textbf{Commit} & \textbf{Resolution} \\
\hline
Config injection incomplete & dc16b1a & All 15 fields now mapped \\
Manual field mapping & 65e4bcf & Automated dataclass introspection \\
\hline
\end{tabular}
\end{table}
```

**LaTeX Errors to Avoid**:
- Unescaped underscores (use `\_` in text, OK in `lstlisting`)
- Missing `[language=Python]` in `\begin{lstlisting}`
- Unicode characters in verbatim blocks
- Table column overflow (use `p{width}` for long text)

### **Reference Standards**
- ✅ Commit 65e4bcf: Code refactor + LaTeX update + README update
- ✅ Commit 94c5296: Phase 2 docs + compiled PDFs
- ✅ Commit dc16b1a: Critical fixes documented with commit hashes
- ❌ v1.1.0 incident: Release corrupted due to uncaught Markdown errors

---