"""
Stochastic Predictor - Universal Stochastic Predictor Package

Este es el package principal del Universal Stochastic Predictor.

Estructura del package:
- sia/: Motor de Identificación de Sistemas (SIA)
- kernels/: Núcleos de predicción (A, B, C, D)
- orchestrator/: Orquestador adaptativo y transporte óptimo
- utils/: Utilidades y funciones auxiliares
- config/: Configuración y constantes

Ejemplo de uso:
    from stochastic_predictor import UniversalPredictor, PredictorConfig
    
    config = PredictorConfig(epsilon=1e-3, learning_rate=0.01)
    predictor = UniversalPredictor(config)
    predictor.calibrate(historical_data)
    prediction = predictor.predict(current_observation)
"""

__version__ = "0.1.0"
__author__ = "Consorcio de Desarrollo de Meta-Predicción Adaptativa"

# Imports principales que se expondrán en el namespace del paquete
# TODO: Descomentar cuando se implementen las clases

# from .config import PredictorConfig
# from .predictor import UniversalPredictor
# from .sia import SIA, WTMM_Estimator
# from .orchestrator import AdaptiveOrchestrator

__all__ = [
    "__version__",
    "__author__",
    # "PredictorConfig",
    # "UniversalPredictor",
    # "SIA",
    # "WTMM_Estimator",
    # "AdaptiveOrchestrator",
]
