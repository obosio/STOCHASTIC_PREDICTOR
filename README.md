# Universal Stochastic Predictor (USP)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-In%20Development-yellow.svg)

## 📋 Descripción

Sistema de predicción estocástica universal capaz de operar sobre procesos dinámicos cuya ley de probabilidad subyacente es desconocida *a priori*. El proyecto integra teoría de procesos estocásticos, análisis multifractal, ecuaciones diferenciales estocásticas y transporte óptimo en un framework unificado.

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

## 🛠️ Stack Tecnológico

### Implementación Python

- **JAX**: Computación numérica acelerada con XLA, vectorización automática y diferenciación
- **Equinox/Diffrax**: Frameworks para redes neuronales y solvers de SDEs sobre JAX
- **Signax**: Cálculo de signatures y log-signatures en GPU
- **PyWavelets**: Transformada wavelet continua
- **OTT-JAX**: Optimal Transport Tools (Sinkhorn-Knopp diferenciable)

### Requisitos

```text
python >= 3.10
jax >= 0.4.0
equinox >= 0.11.0
diffrax >= 0.4.0
signax >= 0.1.0
pywavelets >= 1.4.0
ott-jax >= 0.4.0
```

## 📚 Documentación

El proyecto incluye documentación técnica completa en LaTeX:

- **`Predictor_Estocastico_Teoria.tex`**: Fundamentación matemática y teoría
- **`Predictor_Estocastico_Implementacion.tex`**: Guía de implementación numérica y algorítmica
- **`Predictor_Estocastico_Python.tex`**: Implementación específica en Python/JAX
- **`Predictor_Estocastico_API_Python.tex`**: Especificación de la API
- **`Predictor_Estocastico_IO.tex`**: Interfaz de entrada/salida del sistema

Los PDFs compilados están disponibles en el directorio `doc/`.

## 🚀 Estado del Proyecto

### ⚠️ En Desarrollo Activo

Actualmente el proyecto está en fase de especificación y documentación. La implementación de código está planificada para incluir:

- [ ] Motor de identificación (SIA/WTMM)
- [ ] Núcleos de predicción (A, B, C, D)
- [ ] Orquestador adaptativo (JKO/Sinkhorn)
- [ ] Sistema de detección de cambio de régimen (CUSUM)
- [ ] API de alto nivel para inferencia en tiempo real
- [ ] Suite de tests y benchmarks
- [ ] Ejemplos de uso con datos sintéticos y reales

**Plan detallado**: Consulta [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) para el roadmap completo (6 fases, 26-38 semanas estimadas).

## 📖 Conceptos Clave

### Análisis Multifractal

Caracterización de singularidades locales mediante wavelets y estimación del espectro de singularidades $D(h)$.

### Transporte Óptimo

Actualización de pesos mediante el esquema de minimización JKO en el espacio de Wasserstein:

$$\rho_{n+1} = \underset{\rho \in \mathcal{P}_2(\Omega)}{\text{argmin}} \left\{ E(\rho) + \frac{1}{2\tau} W_2^2(\rho, \rho_n) \right\}$$

### Rough Paths Theory

Integración robusta mediante el cálculo de signatures para procesos con baja regularidad de Hölder.

### Circuit Breaker

Mecanismo de protección que suspende operaciones cuando $H < H_{min}$, evitando divergencias numéricas.

## 🔬 Aplicaciones

- Predicción de series temporales financieras de alta frecuencia
- Análisis de procesos físicos con componentes estocásticos
- Sistemas con cambios de régimen no anticipados
- Procesos con memoria larga y dependencias complejas

## 👥 Autores

Consorcio de Desarrollo de Meta-Predicción Adaptativa

## 📄 Licencia

[MIT License](LICENSE) - Pendiente de añadir

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, consulta la guía de contribución (pendiente) antes de hacer un pull request.

## 📧 Contacto

Para preguntas o colaboraciones, por favor abre un issue en este repositorio.

## 🙏 Agradecimientos

Este proyecto integra metodologías de múltiples áreas de las matemáticas aplicadas y la computación científica. Agradecemos a la comunidad de desarrolladores de JAX, PyWavelets y OTT-JAX por sus excelentes herramientas de código abierto.

---

⚡ Powered by JAX & Differential Geometry
