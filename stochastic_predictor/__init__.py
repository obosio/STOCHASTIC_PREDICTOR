# Stochastic Predictor - Python Package

Este es el package principal del Universal Stochastic Predictor.

## Estructura

El package se organizará en los siguientes módulos:

- `sia/`: Motor de Identificación de Sistemas (SIA)
- `kernels/`: Núcleos de predicción (A, B, C, D)
- `orchestrator/`: Orquestador adaptativo y transporte óptimo
- `utils/`: Utilidades y funciones auxiliares
- `config/`: Configuración y constantes

## Instalación en modo desarrollo

```bash
pip install -e ".[dev]"
```

## Uso

```python
from stochastic_predictor import UniversalPredictor, PredictorConfig

# Configurar el predictor
config = PredictorConfig(
    epsilon=1e-3,
    learning_rate=0.01,
    log_sig_depth=3
)

# Crear instancia
predictor = UniversalPredictor(config)

# Entrenar con datos históricos
predictor.calibrate(historical_data)

# Hacer predicción
prediction = predictor.predict(current_observation)
```
