# Universal Stochastic Predictor (USP)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-Implementation%20Scaffold-green.svg)
![Version](https://img.shields.io/badge/version-v1.1.0-blue.svg)

## 📋 Descripción

**Sistema de predicción estocástica universal** capaz de operar sobre procesos dinámicos cuya ley de probabilidad subyacente es desconocida *a priori*.

Este repositorio contiene:

- ✅ **Especificación técnica completa**: 7 documentos LaTeX (3000+ líneas, 1.73 MB PDFs)
- ✅ **Scaffold de implementación**: Estructura de 5 capas validada (Nivel Diamante)
- ✅ **Golden Master**: Dependency pinning estricto (`==`)
- 🚧 **Código de implementación**: En desarrollo activo (branch `implementation/base-jax`)

## 🎯 Características Principales del Sistema Especificado

### Arquitectura Multinúcleo

1. **Motor de Identificación (SIA)**: Caracterización topológica del proceso mediante WTMM, detección de estacionariedad, estimación de exponentes de Hölder, cálculo de entropía.

2. **Núcleos de Predicción Especializados**:
   - **Rama A (Hilbert)**: RKHS
   - **Rama B (Fokker-Planck)**: DGM/Neural ODEs
   - **Rama C (Itô/Lévy)**: Ecuaciones diferenciales estocásticas diferenciables
   - **Rama D (Signatures)**: Análisis topológico de rough paths

3. **Orquestador Adaptativo**: Transporte de Wasserstein con esquema JKO, detección de cambios CUSUM.

## 🛠️ Stack Tecnológico Especificado

### Golden Master (Dependency Pinning Obligatorio)

```bash
JAX          == 0.4.20
Equinox      == 0.11.2
Diffrax      == 0.4.1
Signax       == 0.1.4
OTT-JAX      == 0.4.5
PyWavelets   == 1.4.1
Python       == 3.10.12
```

**Restricción crítica**: Versiones congeladas con `==`. Prohibido `>=` o `-U`. Ver [Python.tex §2.1](doc/latex/specification/Predictor_Estocastico_Python.tex).

### Arquitectura de 5 Capas Obligatoria

Para futuras implementaciones:

```bash
stochastic_predictor/
|-- api/          # Façade, config, load shedding
|-- core/         # JKO, Sinkhorn, monitoring
|-- kernels/      # Motores XLA (A,B,C,D)
|-- io/           # I/O física, snapshots atómicos
`-- tests/        # Validación externa
```

Ver [Python.tex §2](doc/latex/specification/Predictor_Estocastico_Python.tex).

### Políticas de Seguridad

- **Prohibido**: Credenciales hardcoded
- **Obligatorio**: Inyección de variables de entorno (`.env`)
- **Regla `.gitignore`**: `.env`, `secrets/`, `*.log`

Ver [IO.tex §2.2](doc/latex/specification/Predictor_Estocastico_IO.tex).

### Validación de Entorno CI/CD

Antes de pytest, validar Golden Master:

```bash
EXPECTED_JAX=$(grep "^jax==" requirements.txt | cut -d'=' -f3)
ACTUAL_JAX=$(python -c "import jax; print(jax.__version__)")
[[ "$EXPECTED_JAX" == "$ACTUAL_JAX" ]] || exit 1
```

Ver [Tests_Python.tex §1.1](doc/latex/specification/Predictor_Estocastico_Tests_Python.tex).

## 📚 Documentación

7 documentos LaTeX compilados a PDFs en `doc/pdf/specification/`:

| Documento | Líneas | Contenido |
| --------- | -------- | ---------- |
| Teoria.tex | 500+ | Fundamentación matemática, procesos estocásticos, transporte óptimo |
| Implementacion.tex | 800+ | Algoritmos, dinámica de Sinkhorn acoplada a volatilidad |
| Python.tex | 1700+ | Stack JAX/Python, arquitectura 5 capas, especificaciones técnicas |
| API_Python.tex | 685+ | API de alto nivel, período de gracia CUSUM |
| IO.tex | 292+ | Interfaz I/O, políticas de seguridad |
| Tests_Python.tex | 1623+ | Suite de tests, validación CI/CD, entorno |
| Pruebas.tex | 400+ | Casos de prueba adicionales |

### Compilación (Automática)

The `compile.sh` script automatically detects and compiles all LaTeX source files:

```bash
cd doc

# Ver opciones
./compile.sh help

# Compilar documentos con cambios
./compile.sh --all

# Forzar recompilación total (ignora timestamps)
./compile.sh --all --force

# Compilar documento específico
./compile.sh Predictor_Estocastico_Python.tex

# Limpiar artefactos de compilación
./compile.sh clean
```

**Estructura automática:**

- Fuente: `latex/specification/*.tex` → Compilado: `pdf/specification/*.pdf`
- El script es agnóstico - funciona con cualquier carpeta en `latex/`

Para detalles, ver [doc/README.md](doc/README.md).

## 🚀 Estado Actual

### FASE: Implementation Scaffold (v1.1.0) - Diamond Level Validated ✅

**Branch activo**: `implementation/base-jax`  
**Tag actual**: `v1.1.0-Implementation-Scaffold`  
**Fecha**: 19 Feb 2026

✅ **Completado (100% Auditoría Nivel Diamante)**:

- 7 documentos LaTeX especificación exhaustiva (1.73 MB PDFs)
- Estructura de 5 capas implementada (`api/`, `core/`, `kernels/`, `io/`, `utils/`)
- Golden Master con dependency pinning estricto (`==`)
- Documentación reorganizada en estructura jerárquica
- Políticas de seguridad (.env, .gitignore)
- Configuración centralizada (config.toml)
- Tests base configurados (pytest, coverage)
- LaTeX Workshop configurado
- Stack tecnológico completo (JAX 0.4.20 + Equinox 0.11.2 + Diffrax 0.4.1)

🚧 **En desarrollo**:

- Implementación de kernels (A, B, C, D)
- Motor SIA (WTMM, entropía, estacionariedad)
- Orquestador JKO/Sinkhorn
- Suite de tests completa
- Validación CPU/GPU parity

**Este repositorio está listo para desarrollo activo** con scaffold validado y especificación rigurosa como referencia.

## 🔬 Conceptos Clave Especificados

- **Análisis Multifractal (WTMM)**: Detección de singularidades locales
- **Transporte Óptimo Adaptativo**: Regularización dinámica acoplada a volatilidad
- **Esquemas SDE Dinámicos**: Transición automática Euler → implícito según rigidez
- **Truncamiento de Gradientes**: Optimización XLA para SIA/CUSUM (30-50% VRAM)
- **Período de Gracia CUSUM**: Refractario post-cambio de régimen (10-60 pasos)
- **Rough Paths Theory**: Signatures para procesos con H ≤ 1/2
- **Circuit Breaker**: Protección cuando H < H_min, activa Rama D

Ver documentos LaTeX para derivaciones completas y pseudocódigo.

## 🤝 Contribuciones

Este repositorio es especificación. Contribuciones enfocadas en:

- **Mejoras a especificación**: Correcciones, aclaraciones, extensiones matemáticas
- **Revisión técnica**: Validación de algoritmos, detección de inconsistencias
- **Uso futuro**: Base para implementaciones en JAX, otros lenguajes, etc.

Consulta [CONTRIBUTING.md](CONTRIBUTING.md) antes de contribuir.

## 👥 Autores

Consorcio de Desarrollo de Meta-Predicción Adaptativa

## 📄 Licencia

[MIT License](LICENSE)

## 🙏 Agradecimientos

Especificación integra JAX, Equinox, Diffrax, Signax, PyWavelets, OTT-JAX.

---

📐 **v1.1.0-Implementation-Scaffold**: Scaffold validado con especificación Nivel Diamante  
⚡ **Stack garantizado**: JAX==0.4.20 | Equinox==0.11.2 | Diffrax==0.4.1 | Signax==0.1.4 | OTT-JAX==0.4.5  
🏗️ **Branch activo**: `implementation/base-jax` - Desarrollo en progreso
