# Testing Flow Analysis - Universal Stochastic Predictor (v2.1.0-RC1)

**Fecha**: 20 de febrero de 2026  
**Versión**: 2.1.0-RC1  
**Estado**: Complete - Ready for Execution

---

## 1. Executive Overview

El sistema de testing ha sido reorganizado en una arquitectura modular de **3 capas de validación** orquestadas por un **entrypoint central** (`TESTS_START.py`). Cada capa valida un aspecto diferente del código:

| Capa | Script | Propósito | Artefacto |
| --- | --- | --- | --- |
| **1. Compliance** | `code_alignement.py` | Valida cumplimiento de políticas de audit | `reports/policies/` |
| **2. Coverage** | `tests_coverage.py` | Valida cobertura estructural 100% | `tests/results/coverage_validation.json` |
| **3. Execution** | `code_structure.py` | Valida ejecución real con pytest | pytest stdout/stderr |

---

## 2. Arquitectura Lógica

```text
┌─────────────────────────────────────────────────────────────┐
│                    TESTS_START.py                           │
│              (Entrypoint Orchestrator)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
                    ▼         ▼         ▼
         ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
         │ code_        │ │ tests_       │ │ code_        │
         │ alignement   │ │ coverage     │ │ structure    │
         └──────────────┘ └──────────────┘ └──────────────┘
                    │         │         │
                    ▼         ▼         ▼
         ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
         │ reports/     │ │ tests/       │ │ pytest       │
         │ policies/    │ │ results/     │ │ output       │
         └──────────────┘ └──────────────┘ └──────────────┘
```

---

## 3. Script por Script

### 3.1 Stage 1: `code_alignement.py` (Policy Compliance Checker)

**Ubicación**: `tests/scripts/code_alignement.py` (471 líneas)

**Responsabilidades**:

- Valida el repositorio contra especificaciones de audit en `tests/doc/AUDIT_POLICIES_SPECIFICATION.md`
- Verifica estructuras obligatorias (archivos, paths, contenido)
- Genera reportes JSON timestamped

**Funciones Clave**:

```python
policy_checks()      # List[PolicyResult] - Define todas las políticas
run_checks()         # Ejecuta validaciones contra criterios
write_report()       # Genera JSON con timestamp
main()              # Orchestrador que imprime resumen y exit code
```

**Artefactos Generados**:

- Ubicación: `reports/policies/`
- Formato: `policy_audit_<YYYYMMDD_HHMMSS>.json`
- Contenido: Lista de resultados (policy_id, name, passed, details)

**Salida de Consola**:

```text
PASS: Policy #1 - Language Policy (All code in English)
FAIL: Policy #10 - Deprecated Components Removal
...
SUMMARY
Total: 15 | Passed: 14 | Failed: 1
Report: reports/policies/policy_audit_20260220_184532.json
```

**Exit Code**:

- `0` = Todas las políticas PASS
- `1` = Alguna política FAIL
- `2` = Archivo de especificación no encontrado

**Dependencias**:

- `tests/doc/AUDIT_POLICIES_SPECIFICATION.md` (required)
- No depende de otros scripts

---

### 3.2 Stage 2: `tests_coverage.py` (Structural Coverage Validator)

**Ubicación**: `tests/scripts/tests_coverage.py` (333 líneas)

**Responsabilidades**:

- Analiza cobertura estructural del código a nivel AST
- Extrae funciones públicas del archivo `__all__` de cada módulo
- Identifica gaps (funciones que deberían ser testeadas pero no lo son)
- Detecta tests huérfanos (tests que referencia funciones inexistentes)

**Clases/Funciones Clave**:

```python
StructuralCoverageValidator  # Main validator class
  .extract_public_api()       # Parsea AST para obtener funciones públicas
  .extract_test_functions()   # Extrae funciones de test
  .generate_report()          # Reporte textual detallado
  .generate_json_report()     # JSON con resumen y gaps
validate_coverage()           # API pública
main()                        # Orchestrador
```

**Módulos Verificados**:

- `Python.api.*` (config, prng, types, validation, schemas, state_buffer, warmup)
- `Python.core.*` (fusion, meta_optimizer, orchestrator, sinkhorn)
- `Python.io.*` (config_mutation, credentials, dashboard, loaders, snapshots, telemetry, validators)
- `Python.kernels.*` (kernel_a, kernel_b, kernel_c, kernel_d)

**Artefactos Generados**:

- Ubicación: `tests/results/`
- Formato: `coverage_validation.json`
- Estructura:

  ```json
  {
    "summary": {
      "total_functions": 156,
      "tested_functions": 155,
      "gaps_count": 1,
      "orphans_count": 0
    },
    "gaps": [
      {
        "module": "Python.api.config",
        "function_name": "SomeFunction",
        "is_class": false
      }
    ]
  }
  ```

**Reporte Textual**:

```text
✅ STRUCTURAL COVERAGE VALIDATOR

📦 Module: Python.api.config
  ├─ Public Functions: 8
  ├─ Tested: 8 (100%)
  └─ Status: ✓ PASS

📦 Module: Python.core.fusion
  ├─ Public Functions: 12
  ├─ Tested: 11 (92%)
  └─ Status: ✗ FAIL
     - Missing tests: compute_divergence

SUMMARY
─────────────────────────────────
Total Checked: 12 modules
Total Functions: 156
Tested: 155 (99.4%)
Gaps: 1 function(s)
Orphans: 0 test(s)
```

**Exit Code**:

- `0` = 100% cobertura (gaps_count == 0)
- `1` = Gaps detectados (gaps_count > 0)

**Dependencias**:

- Requiere acceso al AST de `Python/` y `tests/scripts/code_structure.py`
- No depende de `code_alignement.py`

---

### 3.3 Stage 3: `code_structure.py` (Structural Execution Tests)

**Ubicación**: `tests/scripts/code_structure.py` (678+ líneas)

**Responsabilidades**:

- Valida 100% de cobertura ejecutable con inputs reales
- Usa pytest fixtures para configuración y PRNG
- Tests ejecutan código real contra valores válidos
- Verifica que no haya excepciones en paths críticos

**Frameworks**:

- pytest para orquestación
- JAX con x64 enabled para computación
- Real configuration injection via `PredictorConfigInjector`

**Fixtures Disponibles**:

```python
@pytest.fixture
def config_obj() -> PredictorConfig
    # Provee instancia válida de PredictorConfig

@pytest.fixture  
def prng_key() -> jax.random.PRNGKeyArray
    # Provee PRNG key válido inicializado
```

**Test Classes** (examples):

- `TestAPIConfig` - Validación de API config
- `TestPRNG` - PRNG initialization y operations
- `TestValidation` - Batch validation functions
- `TestStateBuffer` - State history management
- `TestFusion` - Core fusion algorithms
- `TestKernels` - Kernel execution A/B/C/D
- `TestMetaOptimizer` - Meta-optimizer behavior
- `etc...`

**Salida Requerida**:

```text
tests/scripts/code_structure.py::TestAPIConfig::test_config_injection PASSED
tests/scripts/code_structure.py::TestPRNG::test_prng_init PASSED
tests/scripts/code_structure.py::TestValidation::test_validate_shape PASSED
...
===== 127 passed in 42.15s =====
```

**Exit Code**:

- `0` = Todos los tests PASSED
- `1` = Algún test FAILED
- `5` = Ningún test recolectado

**Dependencias**:

- `Python/` debe estar importable (package renamed from `stochastic_predictor`)
- `tests/scripts/tests_coverage.py` (referencias mutuas vía imports)
- JAX, pytest, numpy en environment

---

## 4. Orden de Ejecución

El entrypoint `TESTS_START.py` ejecuta en este orden (secuencial, no paralelo):

```text
1️⃣  code_alignement.py      (Policy compliance)
      └─ Tiempo típico: 2-5 segundos
      └─ Output: reports/policies/*.json

2️⃣  tests_coverage.py       (Coverage validation)
      └─ Tiempo típico: 5-10 segundos
      └─ Output: tests/results/coverage_validation.json

3️⃣  code_structure.py       (Full test execution)
      └─ Tiempo típico: 30-60 segundos
      └─ Output: pytest summary + exit code
      └─ Requiere: JAX, completo X64 setup
```

**Estrategia Sequential**:

- ✅ Early fail: Detiene si compliance falla
- ✅ Ahorra recursos: No ejecuta tests si coverage tiene gaps
- ✅ Debugging claro: Exit code indica qué falló

---

## 5. Uso de `TESTS_START.py`

### 5.1 Ejecución Completa

```bash
# Ejecutar todos los stages en orden
python tests/scripts/TESTS_START.py

# Output esperado:
# Stage 1: Policy checks + report generation
# Stage 2: Coverage analysis + JSON report
# Stage 3: pytest session with 127+ tests
# Final summary with exit code
```

### 5.2 Ejecución Selectiva

```bash
# Solo validación de cobertura
python tests/scripts/TESTS_START.py tests_coverage

# Solo pytest structural tests
python tests/scripts/TESTS_START.py code_structure

# Solo audit de políticas
python tests/scripts/TESTS_START.py code_alignement
```

### 5.3 Exit Codes

| Exit Code | Significado | Acción |
| --- | --- | --- |
| `0` | ✅ TODO PASS | Merge ready |
| `1` | ❌ Algún stage FAIL | Revisar logs |
| `2` | ⚠️ Error crítico | Problema configuración |

---

## 6. Artefactos y Reportes

### 6.1 Generación de Artifacts

```text
STOCHASTIC_PREDICTOR/
└── tests/
    └── results/
        ├── code_alignement_2026-02-20_18-00-00.123456.json    ← code_alignement.py
        ├── tests_coverage_2026-02-20_18-00-05.234567.json     ← tests_coverage.py
        ├── code_structure_2026-02-20_18-00-40.345678.json     ← code_structure.py
        └── ...[más timestamped artifacts]
```

### 6.2 Persistencia de Reportes

- **Policy Reports**: 1 archivo JSON por ejecución (timestamped, preservados históricos)
- **Coverage Reports**: 1 archivo JSON por ejecución (timestamped, preservados históricos)
- **Structure Reports**: 1 archivo JSON por ejecución (timestamped, preservados históricos)
- **Convención**: Todos usan formato `[script_name]_YYYY-MM-DD_HH-MM-SS.ffffff.json`

### 6.3 Consumo de Artefactos

```text
CI/CD Pipeline (future):
├── Parse policy_audit_*.json → Validate compliance
├── Parse coverage_validation.json → Check gaps
└── Parse pytest stdout → Verify test suite
```

---

## 7. Dependencias y Relaciones

### 7.1 Mapa de Imports

```text
tests_start.py (Entrypoint)
├── imports: code_alignement.main()
├── imports: tests_coverage.main()
└── imports: code_structure (via pytest)

code_alignement.py
├── reads: tests/doc/AUDIT_POLICIES_SPECIFICATION.md
├── writes: reports/policies/policy_audit_*.json
└── [INDEPENDENT - no Python imports]

tests_coverage.py
├── imports: ast, json, pathlib
├── reads: Python/ (AST parsing all modules)
├── reads: tests/scripts/code_structure.py (test extraction)
└── writes: tests/results/coverage_validation.json

code_structure.py
├── imports: pytest, jax, numpy
├── imports: Python.api.* (real code execution)
├── imports: Python.core.*
├── imports: Python.io.*
├── imports: Python.kernels.*
└── no writes (pytest handles output)
```

### 7.2 Ciclo Crítico

**Critical Path para CI/CD**:

```text
Pass all stages → Deploy
     ↓
code_alignement FAIL → Don't merge (policy violation)
     ↓
tests_coverage FAIL → Don't merge (incomplete coverage)
     ↓
code_structure FAIL → Don't merge (broken functionality)
```

---

## 8. Estados y Transiciones

### 8.1 State Machine

```text
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│ Compliance Check Phase  │ ← code_alignement.py
│ (tests_start.py invoca) │
└──────┬──────────────────┘
       │
       ├─ PASS ──────────┐
       │                 │
       │                 ▼
       │            ┌─────────────────────────┐
       │            │ Coverage Analysis Phase │ ← tests_coverage.py
       │            │ (tests_start.py invoca) │
       │            └──────┬──────────────────┘
       │                   │
       │                   ├─ PASS (0 gaps) ──┐
       │                   │                   │
       │                   │                   ▼
       │                   │          ┌─────────────────────────┐
       │                   │          │ Execution Test Phase    │ ← code_structure.py
       │                   │          │ (tests_start.py invoca) │
       │                   │          └──────┬──────────────────┘
       │                   │                 │
       │                   │                 ├─ ALL PASS ──┐
       │                   │                 │             │
       │                   │                 │             ▼
       │                   │                 │        ┌──────────┐
       │                   │                 │        │   EXIT   │
       │                   │                 │        │ CODE: 0  │
       │                   │                 │        └──────────┘
       │                   │                 │
       │                   │                 ├─ ANY FAIL ─┐
       │                   │                 │            │
       │                   │                 │            ▼
       │                   │                 │       ┌──────────┐
       │                   │                 │       │   EXIT   │
       │                   │                 │       │ CODE: 1  │
       │                   │                 │       └──────────┘
       │                   │
       │                   └─ FAIL (gaps) ───┐
       │                                      │
       │ FAIL ──────────────────────────────┐ │
       │                                    │ │
       └────────────────────────────────────┼─┴───────┐
                                            │         │
                                            ▼         ▼
                                       ┌──────────┐
                                       │   EXIT   │
                                       │ CODE: 1  │
                                       └──────────┘
```

---

## 9. Configuración Actual (v2.1.0-RC1)

### 9.1 Estructura de Directorios

```text
STOCHASTIC_PREDICTOR/
├── Python/                           # ← Package (renamed from stochastic_predictor)
│   ├── __init__.py
│   ├── api/
│   ├── core/
│   ├── io/
│   └── kernels/
│
├── tests/
│   ├── audit/                        # ← Policy specifications (moved from doc/)
│   │   └── AUDIT_POLICIES_SPECIFICATION.md
│   ├── scripts/                      # ← Test orchestration
│   │   ├── TESTS_START.py           # ◄ ENTRYPOINT
│   │   ├── code_alignement.py       # Stage 1: Compliance
│   │   ├── tests_coverage.py        # Stage 2: Coverage
│   │   ├── code_structure.py        # Stage 3: Execution
│   │   └── __init__.py
│   ├── results/                      # ← Artifacts dir (stage outputs)
│   │   └── coverage_validation.json
│   └── reports/                      # ← Reports dir (reserved for future)
│       └── (empty - for future use)
│
└── reports/
    └── policies/                     # ← Policy audit outputs
        └── policy_audit_*.json
```

### 9.2 Environment Requirements

```text
# Python: 3.11+
# JAX: Latest (with x64 flags required)
# pytest: Latest
# No additional dependencies beyond requirements.txt
```

---

## 10. Integration Points

### 10.1 CI/CD Integration (future)

```yaml
# .github/workflows/test.yml (proposed)
on: [push, pull_request]
jobs:
  full-test-suite:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python tests/scripts/TESTS_START.py
        # Exit code determines pass/fail
```

### 10.2 Local Development Workflow

```bash
# Before committing:
python tests/scripts/TESTS_START.py

# If all stages pass → Ready to commit
# If any stage fails:
#   1. Review stage output
#   2. Run individual stage for debugging
#   3. Fix code
#   4. Re-run full suite
```

### 10.3 Pre-commit Hook (future)

```bash
#!/bin/bash
# .git/hooks/pre-commit
python tests/scripts/TESTS_START.py || exit 1
```

---

## 11. Performance Characteristics

### 11.1 Timing Breakdown

| Stage | Typical Time | Bottleneck |
| --- | --- | --- |
| code_alignement | 2-5s | File I/O + policy checks |
| tests_coverage | 5-10s | AST parsing all modules |
| code_structure (pytest) | 30-60s | JAX initialization + test execution |
| **Total** | **40-75s** | JAX startup & X64 precision |

### 11.2 Optimization Opportunities

1. **Parallelization**: code_alignement y tests_coverage podrían correr en paralelo (no comparten state)
2. **Caching**: Cache AST parse results de tests_coverage
3. **JAX Warmup**: Pre-compile kernels en fixture setup

---

## 12. Known Limitations & Caveats

### 12.1 Current Limitations

1. **Sequential Execution**: No hay paralelización (todos corren lineales)
2. **No Caching**: Cada ejecución rescandea AST completo
3. **Policy Doc Required**: code_alignement MUST find `tests/doc/AUDIT_POLICIES_SPECIFICATION.md`
4. **JAX X64 Global**: X64 precision es global en `code_structure.py`

### 12.2 Future Enhancements

- [ ] Parallel stage execution (with dependency ordering)
- [ ] AST caching layer
- [ ] HTML report generation (policy audit)
- [ ] Coverage trend tracking
- [ ] Performance regression detection
- [ ] Integration with code coverage tools (`coverage.py`)

---

## 13. Troubleshooting Guide

### 13.1 Issue: code_alignement Always FAIL

```bash
# Verificar especificación de políticas existe:
ls tests/doc/AUDIT_POLICIES_SPECIFICATION.md

# Si no existe: Restaurar desde git
git checkout tests/doc/AUDIT_POLICIES_SPECIFICATION.md
```

### 13.2 Issue: tests_coverage Detects Gaps

```bash
# Ver qué funciones tienen gaps:
python tests/scripts/tests_coverage.py

# Luego agregar tests a code_structure.py para cubrir gaps
```

### 13.3 Issue: code_structure pytest FAIL

```bash
# Ver qué test específico falla:
python -m pytest tests/scripts/code_structure.py -v

# Correr test específico:
python -m pytest tests/scripts/code_structure.py::TestAPIConfig::test_config_injection -vv
```

### 13.4 Issue: JAX X64 not Enabled

```bash
# Verificar en code_structure.py:
# os.environ["JAX_ENABLE_X64"] = "1"
# jax.config.update("jax_enable_x64", True)

# Si falta: Agregar setup code
```

---

## 14. Version & Metadata

| Propiedad | Valor |
| --- | --- |
| **Project** | Universal Stochastic Predictor (USP) |
| **Version** | 2.1.0-RC1 |
| **Test Architecture** | v2 (Reorganized) |
| **Date** | 20 de febrero de 2026 |
| **Status** | Complete - Ready for Use |
| **Entrypoint** | `tests/scripts/TESTS_START.py` |
| **Total Test Scripts** | 4 (1 orchestrator + 3 validators) |
| **Total Test Cases** | 127+ (in code_structure.py) |
| **Expected Pass Rate** | 100% when all policies enforced |

---

## 15. Summary & Recommendations

### 15.1 Key Achievements

✅ **Modular Design**: 3 independent validation layers  
✅ **Single Entrypoint**: `TESTS_START.py` coordinates all  
✅ **Comprehensive Coverage**: Policy + Structural + Execution  
✅ **Clear Exit Codes**: Simple pass/fail semantics  
✅ **Artifact Separation**: Policies (reports/), Coverage (tests/results/)  
✅ **Selective Execution**: Can run individual stages  

### 15.2 Next Steps

1. **Execute Full Suite**: `python tests/scripts/TESTS_START.py`
2. **Verify All Artifacts**: Check reports/ and tests/results/
3. **Integrate with CI/CD**: Add to GitHub Actions workflow
4. **Monitor Metrics**: Track test pass rate, coverage gaps, execution time

### 15.3 Best Practices

- Always run `TESTS_START.py` before `git commit`
- Review policy audit reports on every merge
- Keep `AUDIT_POLICIES_SPECIFICATION.md` up-to-date
- Add new tests to `code_structure.py` when gaps detected
- Don't manually edit JSON reports (generated programmatically)

---

### End of Report
