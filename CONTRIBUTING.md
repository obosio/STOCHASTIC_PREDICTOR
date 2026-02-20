# Contribution Guide

Thank you for your interest in contributing to the Universal Stochastic Predictor (USP) project.

## Scope of Contributions

This repository contains the technical specification documents (LaTeX). Contributions must focus on improving, clarifying, and extending the specification, not implementing code.

## How to Contribute

### Report Specification Issues

- Use GitHub issues to report:
  - Mathematical errors
  - Inconsistencies between sections (for example, undefined variables)
  - Ambiguities or missing clarity
  - Algorithms that require clarification

- Always include the exact file and section (for example, `Python.tex` section 3.2).

### Suggest Specification Improvements

- Algorithmic extensions with mathematical justification
- Rejected alternatives with comparative analysis
- Additional use cases
- Improved computational complexity analysis

### Pull Request Process

1. Fork the repository.
2. Create a branch with a descriptive name (for example, `fix/typo-sde` or `enhance/sinkhorn-analysis`).
3. Edit `.tex` files in the `doc/` directory.
4. Compile locally with `./doc/compile.sh` to verify LaTeX validity.
5. Commit with a descriptive message:

   ```bash
   docs: correct matrix notation in Python.tex section 2.1
   docs: expand WTMM analysis in Teoria.tex section 3.3
   docs: clarify CUSUM grace period in API_Python.tex
   ```

6. Push and open a Pull Request with a clear change summary.

## Specification Standards

### LaTeX and Documentation

- Use LaTeX commands consistent with existing documents.
- Maintain coherent section structure.
- Include cross-references (`\ref{}`, `\cite{}`).
- Define mathematical notation before use.
- Include examples or pseudocode when appropriate.
- Use English in English documents and Spanish only in Spanish documents.
- Keep line length at or below 100 characters for readable diffs.

### Mathematical Notation

- Use `\textbf{}` for emphasis.
- Define spaces (for example, $\mathbb{R}$, $L^2(\Omega)$, $\mathcal{H}$) when introduced.
- Use consistent subscripts (for example, always $X_t$, never $X(t)$).
- Include dimensions when critical.

## Areas of Contribution

### Base Specification (High Priority)

- Errors in mathematical derivations
- Notation inconsistencies
- Broken cross-references
- Pseudocode needing clarification

### Proposed Extensions (Medium Priority)

- New prediction kernels (with justification)
- Adaptive orchestration alternatives
- Comparative analysis with existing methods
- Specialized use cases

### Documentation Improvements (Low Priority)

- Conceptual diagrams or visualizations
- Improved index
- Additional pseudocode examples
- Appendices with detailed derivations

## Code of Conduct

### Our Commitment

- Maintain a welcoming and inclusive environment grounded in intellectual rigor.
- Respect different mathematical and engineering perspectives.
- Accept constructive technical critique.
- Focus on quality and integrity of the specification.

### Expected Behavior

- Use precise technical language.
- Respect alternative viewpoints with justification.
- Accept specification critique without ego.
- Show empathy toward other reviewers.

### Unacceptable Behavior

- Ad hominem attacks against authors or contributors
- Rejecting valid changes without technical justification
- Discriminatory or harassing language
- Publishing private information without permission

## Review Process

1. LaTeX syntax: CI verifies the specification compiles.
2. Technical review: maintainers verify mathematical consistency.
3. Completeness: check clarity and cross-reference updates.
4. Merge: once approved, changes are merged into `main`.

## Contact

- Issues: for specification-specific reports
- Discussions: for broader architecture or algorithm topics
- Email: contact maintainers for questions

## Acknowledgements

All specification contributors will be recognized in CHANGELOG.md and in relevant commits.
