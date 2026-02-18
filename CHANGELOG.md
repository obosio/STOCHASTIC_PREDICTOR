# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [Unreleased]

### En Desarrollo

- Motor de Identificación de Sistemas (SIA)
- Núcleos de predicción (A, B, C, D)
- Orquestador adaptativo
- Suite de tests y benchmarks

### Agregado

- Plan de implementación detallado (IMPLEMENTATION_PLAN.md)
  - 6 fases de desarrollo con cronograma de 26-38 semanas
  - Especificación completa de módulos y componentes
  - Milestones, métricas de éxito y análisis de riesgos
- Entorno virtual de Python 3.13.12 configurado
- Documentación de configuración del entorno (ENVIRONMENT_SETUP.md)
- Script de verificación del entorno (verify_environment.py)
- Todas las dependencias principales instaladas:
  - JAX 0.4.38 + JAXlib para computación acelerada
  - Equinox 0.13.4, Diffrax 0.7.2 para EDEs
  - OTT-JAX para transporte óptimo
  - Signax para cálculo de signatures
  - Herramientas de desarrollo (pytest, black, flake8, mypy)

## [0.1.0] - 2026-02-18

### Inicialización del Proyecto

- README completo con descripción del proyecto
- Documentación técnica en LaTeX (5 documentos)
- Estructura básica del paquete Python
- Configuración de CI/CD con GitHub Actions
- LICENSE (MIT)
- CONTRIBUTING.md con guías de contribución
- requirements.txt y pyproject.toml
- Estructura de directorios para el código fuente
- .gitignore configurado para Python y LaTeX

### Documentación

- Fundamentos teóricos completos
- Guía de implementación numérica
- Especificación de API Python
- Especificación de I/O
- Guía de implementación en Python/JAX

[Unreleased]: https://github.com/obosio/STOCHASTIC_PREDICTOR/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/obosio/STOCHASTIC_PREDICTOR/releases/tag/v0.1.0
