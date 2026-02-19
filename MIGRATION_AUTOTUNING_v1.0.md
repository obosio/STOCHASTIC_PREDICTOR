# Plan de Auditoría y Migración: Arquitectura de Auto-Parametrización (USP)

**Documento de Referencia:** MIGRATION_AUTOTUNING_v1.0.md  
**Rol:** Lead Implementation Auditor (USP)

## Objetivo

Establecer la hoja de ruta técnica para migrar el Universal Stochastic Predictor (USP) desde un paradigma de hiperparámetros estáticos hacia un sistema integralmente auto-parametrizable ("Learning to Learn"), en estricto cumplimiento con la Taxonomía de Control definida en las especificaciones Diamond Level.

---

## 1. Análisis de Brechas (Gap Analysis)

Actualmente, el vector de configuración $\Lambda$ (PredictorConfig en api/types.py) contiene heurísticas fijadas en tiempo de inicialización (ej. cusum_h, sinkhorn_epsilon_0). Esta rigidez contraviene el principio de universalidad del sistema ante procesos estocásticos de topología desconocida.

Para lograr la auto-parametrización, la auditoría clasifica la migración en tres capas de control dinámico:

- **Capa 1:** Ponderación de Enjambre (Tiempo Real) - Flujo JKO de los pesos $\rho$ de los kernels.
- **Capa 2:** Acoplamiento Topológico (Tiempo Real) - Modificación de parámetros internos basados en diagnósticos del flujo de datos (Volatilidad y Curtosis).
- **Capa 3:** Meta-Optimización Libre de Derivadas (Background/Batch) - Búsqueda de hiperparámetros estructurales mediante Optimización Bayesiana sobre validación Walk-Forward.

---

## 2. Recomendaciones de Migración por Capa

### Capa 1: Orquestación Adaptativa JKO (Completitud)

El sistema ya implementa la estructura básica, pero debe asegurarse que el reseteo de entropía tras una alarma CUSUM sea puramente automático y no dependa de intervenciones manuales.

**Mandato:** Validar que `RegimeChangedEvent` desencadene una inyección de máxima entropía en el simplex de pesos: $\rho \to \text{Softmax}(\mathbf{0})$. (Ref: Theory.tex §3.4).

### Capa 2: Variables de Control Topológico (Auto-ajuste Continuo)

Se deben eliminar las constantes "duras" en los algoritmos de detección y transporte.

**Acoplamiento Volatilidad-Entropía:** El parámetro $\varepsilon$ del algoritmo de Sinkhorn no puede ser estático. Debe escalar linealmente con la varianza empírica local para mantener la contracción del transporte de masa en regímenes turbulentos. (Ref: Implementation.tex §5.4).

**Acoplamiento Curtosis-CUSUM:** El umbral de acumulación de deriva $h_t$ debe ajustarse logarítmicamente frente a la curtosis de la señal para reducir falsos positivos en regímenes leptocúrticos. (Ref: Theory.tex §4.5).

### Capa 3: Meta-Optimización Bayesiana (Learning to Learn)

Para los parámetros estructurales (profundidad de firmas $L$, ventana de memoria WTMM $N_{buf}$, cono de Besov $C_{besov}$), se debe implementar un bucle externo de optimización.

**Mandato:** Implementar un validador estricto Walk-Forward sin "look-ahead bias".

**Herramienta:** Utilizar Procesos Gaussianos (TPE vía Optuna) para minimizar el error de generalización. (Ref: Implementation.tex §8.3).

---

## 3. Guía de Implementación y Snippets de Código

A continuación, se presentan las estructuras en Python/JAX exigidas para la refactorización de los módulos afectados.

### 3.1. Auto-ajuste de Capa 2 (Tiempo Real)

**Módulo:** core/sinkhorn.py y api/state_buffer.py

```python
import jax
import jax.numpy as jnp
from stochastic_predictor.api.types import InternalState, PredictorConfig

@jax.jit
def compute_adaptive_sinkhorn_epsilon(
    ema_variance: jax.Array, 
    config: PredictorConfig
) -> jax.Array:
    """
    Dynamic Sinkhorn Regularization: Coupling to Local Volatility.
    Implementation of Implementation.tex §5.4
    """
    sigma_t = jnp.sqrt(ema_variance)
    # eps_t = max(eps_min, eps_0 * (1 + alpha * sigma_t))
    dynamic_eps = config.sinkhorn_epsilon_0 * (1.0 + config.sinkhorn_alpha * sigma_t)
    return jnp.maximum(config.sinkhorn_epsilon_min, dynamic_eps)

@jax.jit
def compute_adaptive_cusum_threshold(
    state: InternalState, 
    config: PredictorConfig
) -> jax.Array:
    """
    Adaptive Threshold with Kurtosis adjustment for heavy tails.
    Implementation of Theory.tex Lemma 4.5.1
    """
    sigma_t = jnp.sqrt(state.ema_variance)
    kappa_t = state.kurtosis
    
    # h_t = k * sigma * (1 + ln(max(1, kappa / 3)))
    tail_adjustment = 1.0 + jnp.log(jnp.maximum(1.0, kappa_t / 3.0))
    return config.cusum_k * sigma_t * tail_adjustment
```

### 3.2. Auto-ajuste de Capa 3 (Meta-Optimización Bayesiana)

**Módulo:** core/meta_optimizer.py (Nuevo Módulo Sugerido)

```python
import optuna
from typing import Callable, Tuple
import jax.numpy as jnp

class BayesianMetaOptimizer:
    """
    Derivative-Free Meta-Optimization (Gaussian Processes/TPE).
    Ensures structural hyperparameters evolve to fit process topology.
    Ref: Implementation.tex §8.3
    """
    def __init__(self, walk_forward_evaluator: Callable):
        self.evaluator = walk_forward_evaluator
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(multivariate=True)
        )

    def _objective(self, trial: optuna.Trial) -> float:
        """Defines the search space mapping to PredictorConfig."""
        # Structural parameters
        log_sig_depth = trial.suggest_int("log_sig_depth", 2, 5)
        wtmm_buffer_size = trial.suggest_int("wtmm_buffer_size", 64, 512, step=64)
        
        # Sensitivity parameters
        cusum_k = trial.suggest_float("cusum_k", 2.0, 6.0)
        sinkhorn_alpha = trial.suggest_float("sinkhorn_alpha", 0.1, 1.0)
        
        # Assemble candidate configuration (mockup)
        # candidate_config = PredictorConfig(..., log_sig_depth=log_sig_depth)
        
        # Evaluate strictly via Walk-Forward to prevent look-ahead bias
        generalization_error = self.evaluator(trial.params)
        return generalization_error

    def optimize_personality(self, n_trials: int = 50) -> dict:
        """Runs the GP optimization to find the optimal system personality."""
        self.study.optimize(self._objective, n_trials=n_trials)
        return self.study.best_params
```

---

## 4. Protocolo de Auditoría y Cumplimiento (Checklist)

Para dar por aprobada la migración a un modelo auto-parametrizable, el equipo de ingeniería debe garantizar que se cumplan las siguientes aserciones en la suite de pruebas (tests/):

- [ ] **Test de Paridad de Curtosis:** Al simular una señal con saltos de varianza extremos (distribución de Cauchy o similar), el sistema no debe disparar alarmas CUSUM falsas, validando que `compute_adaptive_cusum_threshold` detiene la acumulación de deriva.

- [ ] **Test de Resiliencia Sinkhorn:** Durante periodos de alta volatilidad artificial, el costo OT (Wasserstein distance) no debe divergir a NaN, asegurando que `compute_adaptive_sinkhorn_epsilon` relaja el transporte geométrico.

- [ ] **Aislamiento Walk-Forward:** El `BayesianMetaOptimizer` no puede tener acceso a métricas futuras del set de datos en ninguna iteración. Toda evaluación debe ser puramente causal.

- [ ] **VRAM Constraint:** Los cálculos dinámicos de $\varepsilon_t$ y $h_t$ deben operar encapsulados bajo `jax.lax.stop_gradient()` para no perturbar el árbol de backpropagation de las redes neuronales DGM o Neural SDEs.
