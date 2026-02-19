# Configuración del Entorno de Desarrollo

## Entorno Virtual Python

✅ **Entorno configurado exitosamente**

### Especificaciones (Golden Master)

- **Python Version**: 3.10.12 ⚠️ **CRÍTICO**: Python 3.11+ cambia comportamiento RNG de JAX
- **Location**: `/Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/.venv`
- **Interpreter**: `.venv/bin/python`
- **Dependency Pinning**: Todas las versiones DEBEN usar `==` (no `>=`)

### Activación

```bash
# Activar el entorno virtual
source .venv/bin/activate

# Desactivar
deactivate
```

## Dependencias Instaladas

### ⚠️ GOLDEN MASTER - Versiones Congeladas (Mandatory ==)

| Paquete | Versión Requerida | Estado | Notas |
| --------- | --------- | -------- | -------- |
| jax | 0.4.20 | ✅ | Motor XLA - CRÍTICO |
| jaxlib | 0.4.20 | ✅ | Compilador XLA - CRÍTICO |
| equinox | 0.11.2 | ✅ | Framework neuronal (Ramas B/C) |
| diffrax | 0.4.1 | ✅ | Solvers SDE/ODE diferenciables |
| signax | 0.1.4 | ✅ | Cálculo de signatures (Rama D) |
| ott-jax | 0.4.5 | ✅ | Transporte óptimo (Orquestador JKO) |
| jaxtyping | 0.3.9 | ✅ | Type hints para JAX |

### Cálculo Científico

| Paquete | Versión | Nota |
| --------- | --------- | -------- |
| numpy | 1.24.0 | Mínimo para compatibilidad JAX |
| scipy | 1.10.0 | Funciones científicas |
| pandas | 2.0.0 | Manipulación de datos |
| pywavelets | 1.4.1 | WTMM para SIA |

### Herramientas de Desarrollo

| Paquete | Versión | Uso |
| --------- | --------- | -------- |
| pytest | 7.3.0+ | Testing |
| pytest-cov | 4.1.0+ | Coverage reporting |
| black | 23.0.0+ | Code formatting |
| flake8 | 6.0.0+ | Linting |
| mypy | 1.0.0+ | Type checking |
| isort | 5.12.0+ | Import sorting |

> ⚠️ **Restricción Crítica**: Si alguna versión de la tabla anterior no coincide, ejecutar script de validación de entorno ANTES de pytest. Ver [Tests_Python.tex §1.1](../doc/Predictor_Estocastico_Tests_Python.tex).

## Verificación Rápida

```bash
# Verificar instalación de JAX
.venv/bin/python -c "import jax; print('JAX version:', jax.__version__)"

# Verificar que el entorno está activo
which python  # Debe mostrar la ruta al .venv
```

## Instalación de Nuevas Dependencias

```bash
# Con el entorno activado
pip install nombre-paquete

# Actualizar requirements.txt después de instalar
pip freeze > requirements-frozen.txt
```

## Problemas Conocidos

- ⚠️ scipy puede tener problemas de importación en algunos casos (estamos investigando)
- ✅ JAX funciona correctamente en CPU

## Estado Actual de la Estructura

✅ **Entorno Python**: Completamente configurado

🔄 **Código**: Estructura siendo recreada desde cero

- Especificaciones completas disponibles en `doc/pdf/` (7 documentos)
- Módulos vacíos listos para implementación:
  - `stochastic_predictor/` (config.py + **init**.py)
  - `tests/` (**init**.py)

## Siguiente Paso

Implementar módulos siguiendo especificaciones en `doc/Predictor_Estocastico_Python.pdf` y otros documentos.

Ver [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) para roadmap detallado.
