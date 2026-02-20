# Verificador de Alineación de Políticas - USP v2.1.0-RC1

## Descripción

`verify_alignment.sh` es un script de auditoría automatizado que valida la conformidad de la arquitectura del Universal Stochastic Predictor (USP) contra 23 políticas de ingeniería clave. Ejecuta verificaciones exhaustivas del codebase para garantizar que se cumplen los requisitos de calidad, seguridad y estabilidad numérica.

**Status Actual**: ✅ 23/23 políticas compliant (100%)

---

## Requisitos

- **Bash 4.0+** (macOS, Linux, WSL)
- **Git** (para detectar la raíz del repositorio)
- **grep** (búsqueda de patrones en archivos)

### Verificación de Requisitos

```bash
# Verificar versión de Bash
bash --version

# Verificar disponibilidad de git
which git
```

---

## Instalación

No requiere instalación. El script es independiente:

```bash
# Hacer ejecutable (opcional)
chmod +x scripts/verify_alignment.sh

# O ejecutar directamente con bash
bash scripts/verify_alignment.sh
```

---

## Uso

### Ejecución Básica

```bash
# Desde la raíz del proyecto
bash scripts/verify_alignment.sh

# O si está en la raíz del proyecto
./scripts/verify_alignment.sh
```

### Capturar Salida en Archivo

```bash
bash scripts/verify_alignment.sh > compliance_report.txt 2>&1
```

### Filtrar Resultados

```bash
# Ver solo fallos
bash scripts/verify_alignment.sh 2>&1 | grep "FAIL"

# Ver resumen final
bash scripts/verify_alignment.sh 2>&1 | tail -20

# Ver status de una política específica
bash scripts/verify_alignment.sh 2>&1 | grep "Policy #5"
```

---

## Políticas Validadas

El script verifica 23 políticas críticas organizadas en dos categorías:

### Políticas de Estabilidad y Control (1-8)

| # | Política | Descripción |
| --- | ---------- | ------------- |
| **1** | Zero-Heuristics | Sin valores por defecto silenciosos; validación explícita en `config.py` y `meta_optimizer.py` |
| **2** | Configuration Immutability | Subsecciones protegidas contra mutación en `config_mutation.py` |
| **3** | Validation Schema Enforcement | Schema definido en `[mutation_policy.validation_schema]` de `config.toml` |
| **4** | Atomic Configuration Mutation | POSIX `O_EXCL` + `fsync()` para durabilidad garantizada |
| **5** | Mutation Rate Limiting | Control de tasa de mutación en `[mutation_policy]` |
| **6** | Walk-Forward Validation Protocol | Validación walk-forward en `core/meta_optimizer.py` |
| **7** | CUSUM Threshold Dynamism | Umbrales CUSUM dinámicos en `core/orchestrator.py` |
| **8** | Signature Depth Constraint | Depth log ∈ [3,5] en `config.toml` y verificación en `types.py` |

### Políticas de Precisión Numérica y Características (9-23)

| # | Política | Descripción |
| --- | ---------- | ------------- |
| **9** | Sinkhorn Epsilon Bounds | Epsilon configurado en `config.toml` y `core/sinkhorn.py` |
| **10** | CFL Condition | Validación de estabilidad de timesteps PIDE |
| **11** | Malliavin 64-Bit Precision | `float64` o `jax_enable_x64` activo en config |
| **12** | JAX.lax.stop_gradient | ≥3 aplicaciones de stop_gradient en diagnostics |
| **13** | Kernel Purity & Statelessness | ≥20 decoradores `@jax.jit` en kernels |
| **14** | Frozen Signal Detection | Validator `detect_frozen_signal` implementado e integrado |
| **15** | Catastrophic Outlier Detection | Validator `detect_catastrophic_outlier` implementado e integrado |
| **16** | Nyquist Soft Limit | Frecuencia de inyección configurada |
| **17** | Stale Weights Detection | Validator `is_stale` y `staleness_ttl` en config |
| **18** | Secret Injection Policy | Validación fail-fast en `io/credentials.py` |
| **19** | State Serialization Checksum | SHA256 en `io/snapshots.py` |
| **20** | Non-Blocking Telemetry | Queue architecture con `deque` + `threading.Lock` |
| **21** | Hardware Parity Audit Hashes | Telemetry hash auditing configurado |
| **22** | Walk-Forward Test Leakage Prevention | `train_ratio` configurado para prevenir leakage |
| **23** | Encoder Capacity Expansion | DGM entropy-driven scaling en `core/orchestrator.py` |

---

## Interpretación de Resultados

### Salida Exitosa (100% Compliance)

```text
═══════════════════════════════════════════════════════════════
SUMMARY
═══════════════════════════════════════════════════════════════

Total Checks: 23
Passed: 23
Failed: 0
Compliance: 100%

═══════════════════════════════════════════════════════════════
✓ ALL 23 POLICIES COMPLIANT (100%)
═══════════════════════════════════════════════════════════════
```

**Exit Code**: `0` (éxito)

### Salida con Fallos

```text
═══════════════════════════════════════════════════════════════
SUMMARY
═══════════════════════════════════════════════════════════════

Total Checks: 23
Passed: 21
Failed: 2
Compliance: 91%

═══════════════════════════════════════════════════════════════
✗ COMPLIANCE CHECK FAILED (2 issues)
═══════════════════════════════════════════════════════════════
```

**Exit Code**: `1` (fallo)

### Códigos de Salida

| Exit Code | Significado |
| ----------- | ------------ |
| `0` | ✅ Todas las 23 políticas pasan |
| `1` | ❌ Una o más políticas fallan |

---

## Ejemplos de Uso

### Verificación Manual Previa a Commit

```bash
# Verificar alineación antes de hacer commit
bash scripts/verify_alignment.sh

# Si resultado es 0 (éxito), proceder con commit
if [ $? -eq 0 ]; then
    git add .
    git commit -m "feature: implementation with full policy compliance"
else
    echo "⚠️ Compliance issues detected. Fix before committing."
    exit 1
fi
```

### CI/CD Pipeline

```yaml
# Ejemplo: GitHub Actions workflow
- name: Verify Policy Alignment
  run: bash scripts/verify_alignment.sh
  continue-on-error: false  # Fallar si hay incumplimiento
```

### Debugging de Fallos

```bash
# Ver política específica que falla
bash scripts/verify_alignment.sh 2>&1 | grep -A 5 "FAIL"

# Generador de reporte de auditoría
bash scripts/verify_alignment.sh > /tmp/audit_$(date +%Y%m%d_%H%M%S).txt

# Comparar cambios recientes con baseline
git diff HEAD^ -- stochastic_predictor/ | grep -E "^\+" > /tmp/recent_changes.diff
bash scripts/verify_alignment.sh  # Re-validar tras cambios
```

---

## Troubleshooting

### Problema: "Permission denied"

```bash
# Solución: Ejecutar con bash explícitamente
bash scripts/verify_alignment.sh
```

### Problema: Script no encuentra archivos

```bash
# Verificar que se ejecuta desde la raíz del proyecto
pwd
# Debe mostrar: .../STOCHASTIC_PREDICTOR

# Si está en otro directorio:
cd /path/to/STOCHASTIC_PREDICTOR
bash scripts/verify_alignment.sh
```

### Problema: Falsos Negativos (Policy #X falla pero el código existe)

```bash
# Verificar manualmente el patrón buscado:
grep -r "pattern_name" stochastic_predictor/

# Revisar el mensaje de error específico para ajustar el patrón
bash scripts/verify_alignment.sh 2>&1 | grep "Policy #"
```

### Problema: Git no está disponible

```bash
# El script intenta detectar raíz con git, fallará si no está instalado
# Solución: Instalar git o ejecutar desde la raíz del proyecto explícitamente

# Verificar git:
which git
```

---

## Referencias de Implementación

### Documentación Técnica

- **Especificación de Políticas**: [reports/policies/AUDIT_POLICIES_SPECIFICATION.md](../reports/policies/AUDIT_POLICIES_SPECIFICATION.md)
- **Reporte de Alineación**: [reports/policies/PRE_TEST_COMPLIANCE_REPORT.md](../reports/policies/PRE_TEST_COMPLIANCE_REPORT.md)

### Archivos Validados

- **Configuración**: `config.toml`
- **API Core**: `stochastic_predictor/api/` (config.py, types.py, validation.py, etc.)
- **Kernels**: `stochastic_predictor/kernels/`
- **Orquestación**: `stochastic_predictor/core/` (orchestrator.py, meta_optimizer.py, sinkhorn.py)
- **I/O**: `stochastic_predictor/io/` (validators.py, credentials.py, snapshots.py, telemetry.py)

---

## Mantenimiento

### Actualizar el Script

Cuando se agreguen nuevas políticas:

1. Adicionar nueva política en `AUDIT_POLICIES_SPECIFICATION.md`
2. Agregar check correspondiente en `verify_alignment.sh`
3. Incrementar contador de políticas en comentario de header
4. Actualizar esta tabla README

### Validación de Cambios

```bash
# Después de modificar el script:
bash scripts/verify_alignment.sh

# Verificar conteo de checks:
bash scripts/verify_alignment.sh 2>&1 | grep "Total Checks:"
# Debe mostrarse: Total Checks: 23 (o número actualizado)
```

---

## Licencia

Este script es parte del proyecto **Universal Stochastic Predictor (USP)** y está disponible bajo los términos descritos en el archivo LICENSE del proyecto.

---

## Soporte

Para reportar problemas o sugerir mejoras:

1. Verificar que el script se ejecuta desde la raíz del proyecto
2. Incluir la salida completa del script y el exit code
3. Consultar la sección "Troubleshooting" arriba

---

**Última Actualización**: 20 de febrero de 2026  
**Versión del Script**: USP v2.1.0-RC1  
**Status**: ✅ Production Ready
