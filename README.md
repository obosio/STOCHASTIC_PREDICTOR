# Universal Stochastic Predictor (USP)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-Specification-blue.svg)

## 📋 Descripción

**Especificación matemática y algorítmica completa** de un sistema de predicción estocástica universal capaz de operar sobre procesos dinámicos cuya ley de probabilidad subyacente es desconocida *a priori*. El proyecto integra teoría de procesos estocásticos, análisis multifractal, ecuaciones diferenciales estocásticas y transporte óptimo en un framework unificado.

> ⚠️ **Estado del Proyecto**: Este repositorio contiene **únicamente especificaciones técnicas completas** (7 documentos LaTeX, 3000+ líneas, 1.73 MB PDFs). **No incluye código de implementación**.

## 🎯 Características Principales

### Arquitectura Multinúcleo

El sistema se estructura en tres fases operativas:

1. **Motor de Identificación (SIA)**: Caracterización topológica del proceso mediante:
   - Análisis multifractal (WTMM - Wavelet Transform Modulus Maxima)
   - Detección de estacionariedad y ergodicidad
   - Estimación de exponentes de Hölder
   - Cálculo de entropía de transferencia

2. **Núcleos de Predicción Especializados**:
   - **Rama A (Hilbert)**: Proyecciones en espacios de Hilbert reproducibles (RKHS)
   - **Rama B (Markov/Fokker-Planck)**: Procesos markovianos y ecuaciones de Fokker-Planck
   - **Rama C (Itô/Lévy)**: Integración de procesos con saltos y componentes de Lévy
   - **Rama D (Rough Paths/Signature)**: Análisis topológico mediante teoría de signatures

3. **Orquestador Adaptativo**:
   - Fusión óptima mediante transporte de Wasserstein
   - Esquema JKO (Jordan-Kinderlehrer-Otto)
   - Detección de cambio de régimen (CUSUM)

### Fundamento Matemático

El sistema opera sobre un espacio de probabilidad completo $(\Omega, \mathcal{F}, P)$ con filtración $\{\mathcal{F}_t\}_{t \geq 0}$. El problema central es encontrar el operador de predicción óptimo:

$$\hat{X}_{t+h} = \underset{Z \in L^2(\mathcal{F}_t)}{\text{argmin}} \, \mathbb{E}\left[ \| X_{t+h} - Z \|^2 \right] = \mathbb{E}[X_{t+h} \mid \mathcal{F}_t]$$

## 🛠️ Stack Tecnológico Especificado

### Herramientas de Documentación

- **LuaLaTeX**: Motor de compilación LaTeX con soporte Unicode nativo
- **Bash Script**: `doc/compile.sh` con detección inteligente de cambios
  - Compila solo documentos modificados (ahorro de tiempo)
  - Modo `--force` para fuerza recompilación completa
  - Reporting de errores LaTeX integrado
  - Dos pasadas automáticas para actualizar índices

### Stack Python Especificado (Grabado en Piedra)

La especificación define y justifica rigurosamente el siguiente stack para implementación futura:

- **JAX 0.4.20**: Motor XLA con diferenciación automática y vectorización (capa fundamental)
- **Equinox 0.11.3**: Framework neuronal pythonico para Ramas B y C (DGM, Neural ODEs)
- **Diffrax 0.4.1**: Solver diferenciable de SDEs/ODEs para Rama C
- **Signax 0.1.4**: Cálculo de log-signatures en GPU para Rama D
- **PyWavelets 1.4.1**: Transformada wavelet continua para SIA (WTMM)
- **OTT-JAX 0.4.5**: Transporte óptimo diferenciable para Orquestador JKO

> 📘 **Justificación completa**: Ver [Python.tex §1](doc/Predictor_Estocastico_Python.tex) (~250 líneas) con análisis técnico y alternativas descartadas.

## 📚 Documentación

El proyecto incluye documentación técnica completa en LaTeX con especificaciones e implementaciones:

- **`Predictor_Estocastico_Teoria.tex`**: Fundamentación matemática, arquitectura y teoría (500+ líneas, transición dinámica SDE)
- **`Predictor_Estocastico_Implementacion.tex`**: Guía algorítmica con volatilidad acoplada en Sinkhorn (800+ líneas)
- **`Predictor_Estocastico_Python.tex`**: Implementación Python/JAX con truncamiento de gradientes (1700+ líneas)
- **`Predictor_Estocastico_API_Python.tex`**: Especificación de la API con período de gracia CUSUM (685+ líneas)
- **`Predictor_Estocastico_IO.tex`**: Interfaz de entrada/salida del sistema
- **`Predictor_Estocastico_Tests_Python.tex`**: Suite de tests y validaciones
- **`Predictor_Estocastico_Pruebas.tex`**: Pruebas adicionales y casos especiales

**PDFs compilados**: 7 documentos (1.73 MB total) disponibles en `doc/pdf/` con índices y referencias sincronizadas.

### Compilación de Documentos

```bash
# Mostrar ayuda (opción por defecto sin argumentos)
cd doc && ./compile.sh

# Compilar solo documentos con cambios
./compile.sh --all

# Forzar recompilación de todos
./compile.sh --all --force

# Compilar un documento específico
./compile.sh Predictor_Estocastico_Python

# Limpiar artefactos de compilación
./compile.sh clean
```

El script utiliza **detección inteligente de cambios** basada en timestamps para evitar compilaciones innecesarias.

## 🚀 Estado del Proyecto

### 📂 Estructura Actual (Febrero 18, 2026)

**Fase de Re-construcción**: La estructura de código Python está siendo recreada desde cero basándose en especificaciones completas.

✅ **Disponible**:

- Especificaciones detalladas (7 documentos LaTeX, 1.73 MB PDFs)
- Build system optimizado (compile.sh con inteligencia de cambios)
- Entorno Python configurado (Python 3.10+, todas las dependencias JAX)
- Stack tecnológico validado

🔄 **En construcción**:

- Módulos `stochastic_predictor/` (vacíos, listos para implementación)
- Suite de tests `tests/` (vacía, lista para agregarse)

### � Avances Recientes (Febrero 2026)

**Arquitectura mejorada con algoritmos robustos**:

- ✨ Transición dinámica de esquemas SDE (explícito ↔ implícito según rigidez)
- ✨ Dinámica de Sinkhorn acoplada a volatilidad (regularización adaptativa)
- ✨ Período de gracia CUSUM para evitar cascadas de falsas alarmas
- ✨ Optimización del grafo XLA con `stop_gradient` (ahorro: 30-50% VRAM)
- ✨ Script de compilación con detección inteligente de cambios

**Documentación completa**: 7 PDFs (1.73 MB) con especificaciones matemáticas e implementación.

### �📖 Fase Actual: Especificación y Arquitectura Avanzada

El proyecto está en fase de **especificación detallada de arquitectura** con implementaciones de algoritmos clave ya documentadas.

#### ✅ Completado en Documentación

- [x] Arquitectura multinúcleo especificada (4 ramas de predicción)
- [x] Fundamentación matemática completa (teoría de procesos estocásticos, óptimo transporte, rough paths)
- [x] Algoritmo SIA (System Identification Archive) especificado
- [x] Núcleo B (Fokker-Planck, DGM) documentado
- [x] Núcleo C (Itô/Lévy) con **transición dinámica de esquemas SDE** (Euler explícito ↔ implícito)
- [x] Núcleo D (Signatures) especificado
- [x] Orquestador JKO con **dinámica de Sinkhorn acoplada a volatilidad**
- [x] Sistema CUSUM con **período de gracia (refractario)** post-cambio de régimen
- [x] Optimización del grafo computacional con **JAX stop_gradient**
- [x] Suite de tests para validación de módulos

#### 🔄 En Progreso: Implementación

- [ ] Motor de identificación (SIA/WTMM) - inicio prioritario
- [ ] Kernels A, B, C, D - según roadmap
- [ ] Orquestador adaptativo (JKO/Sinkhorn) con volatilidad acoplada
- [ ] Sistema de detección de régimen (CUSUM) con período de gracia
- [ ] API de alto nivel para inferencia
- [ ] Benchmarks y ejemplos con datos sintéticos/reales

#### 📋 Características Algorítmicas Documentadas

| Componente | Estado | Documento |
| --- | --- | --- |
| Stop Gradient Optimization | ✅ Documentado | Python.tex §3.1 |
| Dinámica Sinkhorn Volátil | ✅ Documentado | Implementacion.tex §2.4 |
| Período de Gracia CUSUM | ✅ Documentado | API_Python.tex §3.2 |
| Esquemas SDE Dinámicos | ✅ Documentado | Teoria.tex §2.3.3 |
| Detección Adaptativa CUSUM | ✅ Documentado | Teoria.tex §6.2 |
| Stack Equinox/Diffrax | ✅ Grabado en piedra | Python.tex §1 |

## 📖 Conceptos Clave

### Análisis Multifractal

Caracterización de singularidades locales mediante wavelets y estimación del espectro de singularidades $D(h)$ usando técnicas de WTMM (Wavelet Transform Modulus Maxima).

### Transporte Óptimo Adaptativo

Actualización de distribuciones de probabilidad mediante el esquema JKO con **regulación dinámica de entropía acoplada a volatilidad**:

$$\varepsilon_t = \max(\varepsilon_{\min}, \varepsilon_0 \cdot (1 + \alpha \cdot \sigma_t))$$

Donde $\sigma_t$ es volatilidad estimada mediante EMA. Esta formulación garantiza suavidad del paisaje de optimización durante crisis de mercado.

### Esquemas de Discretización Dinámica para SDEs

En la Rama C (Itô/Lévy), **transición automática** entre esquemas numéricos según rigidez (stiffness):

- **Bajo stiffness** ($S_t < 100$): Euler-Maruyama explícito (rápido)
- **Medio stiffness**: Esquema híbrido interpolado
- **Alto stiffness** ($S_t > 1000$): Método implícito trapezial (robusto)

Métrica: $S_t = \max(\text{ratio de valores propios}, |d\log\sigma/dt| \cdot \Delta t)$

### Truncamiento de Gradientes en Diagnósticos

Optimización del grafo computacional JAX mediante `stop_gradient` para outputs no-entrenable (SIA, CUSUM):

$$\frac{\partial H}{\partial \rho} = 0, \quad \frac{\partial \text{alarm}}{\partial \rho} = 0$$

Ahorro esperado: **30-50% VRAM, 20-40% tiempo JIT, 50%+ backward pass**.

### Período de Gracia (Refractario) en CUSUM

Mecanismo de silenciamiento temporal post-cambio de régimen para evitar cascadas de falsas alarmas:

$$\text{alarm}_t = \left\{ \begin{array}{ll} \text{False} & \text{si } t - t_{\text{change}} < \tau_g \\ G^+ > h_t & \text{si no} \end{array} \right.$$

Parámetro: $\tau_g \in [10, 60]$ pasos según volatilidad del mercado.

### Rough Paths Theory

Integração robusta mediante cálculo de signatures para procesos con baja regularidad de Hölder ($H \leq 1/2$).

### Circuit Breaker

Mecanismo de protección que suspende operaciones cuando $H < H_{\min}$, fuerza Rama D (signatures) y activa pérdida de Huber robusta.

## 🔬 Aplicaciones Especificadas

La especificación está diseñada para:

- Predicción de series temporales financieras de alta frecuencia
- Análisis de procesos físicos con componentes estocásticos
- Sistemas con cambios de régimen no anticipados
- Procesos con memoria larga y dependencias complejas

> 📐 **Nivel de detalle**: Las especificaciones incluyen pseudocódigo Python completo, análisis de complejidad computacional, y estrategias de optimización GPU/XLA listas para traducción directa a código.

## 👥 Autores

Consorcio de Desarrollo de Meta-Predicción Adaptativa

## 📄 Licencia

[MIT License](LICENSE) - Pendiente de añadir

## 🤝 Contribuciones

Este repositorio contiene **especificaciones técnicas completas** sin implementación. Posibles contribuciones:

- 📝 **Mejoras a la especificación**: Correcciones, aclaraciones, extensiones matemáticas
- 🔍 **Revisión técnica**: Validación de algoritmos, detección de inconsistencias
- 🚀 **Implementación futura**: Uso de estas especificaciones como base para proyectos derivados

Por favor, consulta [CONTRIBUTING.md](CONTRIBUTING.md) antes de contribuir.

## 📧 Contacto

Para preguntas o colaboraciones, por favor abre un issue en este repositorio.

## 🙏 Agradecimientos

Esta especificación integra metodologías de múltiples áreas de las matemáticas aplicadas y la computación científica. Agradecemos a la comunidad de desarrolladores de JAX, Equinox, Diffrax, Signax, PyWavelets y OTT-JAX, cuyas herramientas fueron seleccionadas como base del stack tecnológico especificado.

---

📐 **Nivel Diamante**: Especificación matemática rigurosa lista para implementación  
⚡ Stack especificado: JAX + Equinox + Diffrax + Signax + OTT-JAX
