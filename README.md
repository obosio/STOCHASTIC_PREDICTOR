# Universal Stochastic Predictor (USP)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-Specification%20Only-blue.svg)

## 📋 Descripción

**Especificación matemática y algorítmica completa** de un sistema de predicción estocástica universal capaz de operar sobre procesos dinámicos cuya ley de probabilidad subyacente es desconocida *a priori*.

Este repositorio contiene **únicamente la especificación técnica** (7 documentos LaTeX, 3000+ líneas, 1.73 MB PDFs de especificación rigurosa), **sin código de implementación**.

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

```
JAX          == 0.4.20
Equinox      == 0.11.2
Diffrax      == 0.4.1
Signax       == 0.1.4
OTT-JAX      == 0.4.5
PyWavelets   == 1.4.1
Python       == 3.10.12
```

**Restricción crítica**: Versiones congeladas con `==`. Prohibido `>=` o `-U`. Ver [Python.tex §2.1](doc/Predictor_Estocastico_Python.tex).

### Arquitectura de 5 Capas Obligatoria

Para futuras implementaciones:

```
stochastic_predictor/
├── api/          # Façade, config, load shedding
├── core/         # JKO, Sinkhorn, monitoring
├── kernels/      # Motores XLA (A,B,C,D)
├── io/           # I/O física, snapshots atómicos
└── tests/        # Validación externa
```

Ver [Python.tex §2](doc/Predictor_Estocastico_Python.tex).

### Políticas de Seguridad

- **Prohibido**: Credenciales hardcoded
- **Obligatorio**: Inyección de variables de entorno (`.env`)
- **Regla `.gitignore`**: `.env`, `secrets/`, `*.log`

Ver [IO.tex §2.2](doc/Predictor_Estocastico_IO.tex).

### Validación de Entorno CI/CD

Antes de pytest, validar Golden Master:

```bash
EXPECTED_JAX=$(grep "^jax==" requirements.txt | cut -d'=' -f3)
ACTUAL_JAX=$(python -c "import jax; print(jax.__version__)")
[[ "$EXPECTED_JAX" == "$ACTUAL_JAX" ]] || exit 1
```

Ver [Tests_Python.tex §1.1](doc/Predictor_Estocastico_Tests_Python.tex).

## 📚 Documentación

7 documentos LaTeX compilados a PDFs en `doc/pdf/`:

| Documento | Líneas | Contenido |
|-----------|--------|----------|
| Teoria.tex | 500+ | Fundamentación matemática, procesos estocásticos, transporte óptimo |
| Implementacion.tex | 800+ | Algoritmos, dinámica de Sinkhorn acoplada a volatilidad |
| Python.tex | 1700+ | Stack JAX/Python, arquitectura 5 capas, especificaciones técnicas |
| API_Python.tex | 685+ | API de alto nivel, período de gracia CUSUM |
| IO.tex | 292+ | Interfaz I/O, políticas de seguridad |
| Tests_Python.tex | 1623+ | Suite de tests, validación CI/CD, entorno |
| Pruebas.tex | 400+ | Casos de prueba adicionales |

### Compilación

```bash
cd doc

# Mostrar opciones
./compile.sh

# Compilar documentos modificados
./compile.sh --all

# Forzar recompilación total
./compile.sh --all --force

# Compilar documento específico
./compile.sh Predictor_Estocastico_Python

# Limpiar artefactos
./compile.sh clean
```

## 🚀 Estado Actual

**FASE: Especificación Técnica Completa (Diamond Level)**

✅ Disponible:
- 7 documentos LaTeX especificación exhaustiva
- 1.73 MB PDFs compilados con índices y referencias
- Stack tecnológico justificado y especificado
- Arquitectura Clean Archit (5 capas) definida
- Políticas de seguridad integradas
- Procedimientos CI/CD pre-test especificados

❌ No incluido:
- Código de implementación
- Tests ejecutables
- Entorno virtual pre-configurado

Este repositorio es el **punto de partida** para que equipos de desarrollo implementen el sistema basándose en especificación rigurosa.

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

📐 **Nivel Diamante**: Especificación matemática rigurosa lista para implementación  
⚡ Stack especificado: JAX 0.4.20 + Equinox 0.11.2 + Diffrax 0.4.1 + Signax 0.1.4 + OTT-JAX 0.4.5
