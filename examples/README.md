# Ejemplos de Uso

Esta carpeta contendrá ejemplos de uso del Universal Stochastic Predictor.

## Ejemplos Planificados

### 1. Datos Sintéticos

- `example_brownian.py`: Predicción de movimiento Browniano
- `example_levy.py`: Procesos con saltos de Lévy
- `example_multifractal.py`: Series con propiedades multifractales

### 2. Datos Financieros

- `example_stock_prices.py`: Predicción de precios de acciones
- `example_forex.py`: Predicción de tasas de cambio
- `example_crypto.py`: Predicción de criptomonedas

### 3. Datos Físicos

- `example_turbulence.py`: Análisis de turbulencia
- `example_climate.py`: Datos climáticos

### 4. Notebooks

- `tutorial_01_basic_usage.ipynb`: Introducción básica
- `tutorial_02_sia_module.ipynb`: Uso del módulo SIA
- `tutorial_03_kernels.ipynb`: Núcleos de predicción
- `tutorial_04_orchestrator.ipynb`: Orquestador adaptativo

## Estructura de un Ejemplo

Cada ejemplo incluirá:

1. Carga de datos
2. Configuración del predictor
3. Calibración/entrenamiento
4. Predicción
5. Evaluación de resultados
6. Visualización

## Instalación de Dependencias para Ejemplos

```bash
pip install -e ".[dev,viz]"
```
