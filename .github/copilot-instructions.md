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

## 7. Pre-Commit Quality Assurance (MANDATORY)
**CRITICAL RULE**: ALWAYS verify there are NO VSCode errors before executing `git commit` or `git push`.

**Workflow**:
1. After making changes, run VSCode error check using `get_errors()` tool
2. If errors found: Fix all errors BEFORE staging
3. Only after `get_errors()` returns "No errors found" → proceed to commit/push
4. Never bypass this check - errors cause corrupted releases and broken CI/CD

**Common Error Types to Watch**:
- Markdown (MD040: Code blocks missing language specifier, MD060: Table formatting, MD032: list spacing, MD036: heading punctuation)
- LaTeX (Unicode incompatibility in verbatim blocks, escape character issues)
- Python (Type hints, import statements, syntax errors)
- YAML/TOML (Indentation, key format, string escaping)

**Reference Incident**: Previous release v1.1.0 was corrupted due to Markdown formatting errors not caught pre-commit. Required deletion and recreation of entire release artifact.

## 8. Git Workflow (Strict Discipline)
1. Make code changes
2. **ALWAYS**: `get_errors()` → Fix if needed
3. `git add <files>`
4. `git commit -m "..."` (meaningful, technical commit message)
5. `git push origin <branch>`
6. Verify push successful and CI/CD passing

Non-compliance with this sequence has caused project delays.

## 9. README Updates Before Tagging (MANDATORY)
**CRITICAL RULE**: ALL README.md files in the project structure MUST be updated BEFORE creating and pushing a new version tag.

**When to Update READMEs:**
- After completing a new Phase (Phase 1, Phase 2, etc.)
- When adding new layers, modules, or public APIs
- When updating version numbers or architecture diagrams
- Before running `git tag` command

**Files to Check/Update:**
- `README.md` (root) - Overall project status, quick start, version table
- `doc/README.md` - Documentation guide and structure
- `stochastic_predictor/api/README.md` - API layer documentation (if created)
- `stochastic_predictor/core/README.md` - Core layer documentation (if created)
- `stochastic_predictor/kernels/README.md` - Kernels layer documentation (if created)
- `stochastic_predictor/io/README.md` - I/O layer documentation (if created)
- `tests/README.md` - Testing strategy and fixture guide (if created)

**README Content Checklist:**
- [ ] Current version/tag referenced
- [ ] Phase status (Implementation vs Specification)
- [ ] Layer description matches current implementation
- [ ] Code examples are up-to-date
- [ ] References to spec documents are correct
- [ ] No broken links or typos
- [ ] 100% English (no Spanish comments or descriptions)

**Workflow Before Tagging:**
```bash
# 1. Update all relevant README files
# 2. Review with get_errors() to ensure Markdown compliance
get_errors()

# 3. Stage README changes
git add README.md doc/README.md stochastic_predictor/*/README.md

# 4. Commit README updates
git commit -m "docs: Update READMEs for vX.Y.Z"

# 5. Only then create the tag
git tag impl/vX.Y.Z -m "Implementation vX.Y.Z: <Description>"

# 6. Push both commits and tags
git push origin <branch>
git push origin impl/vX.Y.Z
```

**Reference Standard**: Previous commits like cf46ff4 and 3976210 show proper README updates before tagging.
