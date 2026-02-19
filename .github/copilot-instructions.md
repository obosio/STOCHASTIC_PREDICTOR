# Copilot Instructions: Universal Stochastic Predictor (USP)

## 1. Role & Scope
Act as an expert implementation assistant. Deliver precise, minimal, and correct changes that align with the system architecture and the pinned technical stack.

## 2. Language Policy (Mandatory)
All project artifacts must be written 100% in English.

## 3. Architecture (Clean Layers)
Use a strict 5-layer structure: API, Core, Kernel, IO, Tests.

## 4. Technical Stack (Pinned)
Respect the pinned versions for core and specialized libraries. Do not upgrade or swap dependencies without explicit instruction.

## 5. Coding Standards
- Kernels are pure and stateless.
- Apply `jax.lax.stop_gradient()` to diagnostic modules.
- Enable 64-bit precision for Malliavin and Signature calculations.
- No hardcoded secrets; use environment variables.

## 6. Tone & Output
Be concise, technical, and direct. Avoid filler.

## 7. Pre-Commit/Push Workflow (Mandatory)
1) Fix all VSCode errors. Do not proceed if any exist.
2) Update LaTeX docs when public API or architecture changes.
3) Update overview docs before creating release tags.
4) Stage, commit, and push with a meaningful message.
5) Create tags only for phase milestones.