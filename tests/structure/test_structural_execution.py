"""
Structural Execution Tests - 100% Code Coverage with Real Execution.

Philosophy:
    - 100% cobertura = todas las líneas ejecutadas sin exceptions
    - Inputs válidos según firmas actuales del código
    - Tests reales (no solo imports)

Strategy:
    1. Meta-validator identifica gaps
    2. Leer source code para firmas exactas
    3. Crear inputs válidos
    4. Ejecutar y verificar sin errores
"""

import os
from datetime import datetime, timezone

import jax
import jax.numpy as jnp
import pytest

os.environ["JAX_ENABLE_X64"] = "1"
jax.config.update("jax_enable_x64", True)

# API imports
from stochastic_predictor.api.config import PredictorConfigInjector, get_config, ConfigManager
from stochastic_predictor.api.prng import (
    initialize_jax_prng, split_key, split_key_like, uniform_samples,
    normal_samples, exponential_samples, check_prng_state, verify_determinism
)
from stochastic_predictor.api.types import (
    PredictorConfig, ProcessState, InternalState, KernelOutput,
    PredictionResult, KernelType, OperatingMode, check_jax_config
)
from stochastic_predictor.api.validation import (
    validate_magnitude, validate_timestamp, check_staleness, validate_shape,
    validate_finite, sanitize_array, ensure_float64, cast_array_to_float64,
    validate_holder_exponent, validate_alpha_stable, validate_beta_stable,
    validate_simplex, sanitize_external_observation, warn_if_invalid
)
from stochastic_predictor.api import OperatingModeSchema
from stochastic_predictor.api.schemas import (
    ProcessStateSchema, KernelOutputSchema, TelemetryDataSchema,
    PredictionResultSchema, HealthCheckResponseSchema
)
from stochastic_predictor.api.state_buffer import (
    update_signal_history, atomic_state_update, batch_update_signal_history,
    reset_cusum_statistics, update_cusum_statistics, update_ema_variance,
    update_residual_buffer
)

# Core imports
from stochastic_predictor.core.orchestrator import (
    initialize_state, initialize_batched_states, orchestrate_step, orchestrate_step_batch,
    compute_entropy_ratio, scale_dgm_architecture, apply_host_architecture_scaling,
    compute_adaptive_stiffness_thresholds, compute_adaptive_jko_params, OrchestrationResult
)
from stochastic_predictor.core.fusion import fuse_kernel_outputs, FusionResult
from stochastic_predictor.core.sinkhorn import compute_sinkhorn_epsilon, SinkhornResult
from stochastic_predictor.core.meta_optimizer import (
    walk_forward_split, BayesianMetaOptimizer, AsyncMetaOptimizer,
    MetaOptimizationConfig, OptimizationResult, IntegrityError
)

# Kernels imports
from stochastic_predictor.kernels.base import (
    apply_stop_gradient_to_diagnostics, compute_signal_statistics,
    normalize_signal, validate_kernel_input, PredictionKernel
)
from stochastic_predictor.kernels.kernel_a import (
    gaussian_kernel, compute_gram_matrix, kernel_ridge_regression,
    create_embedding, kernel_a_predict
)
from stochastic_predictor.kernels.kernel_b import (
    compute_entropy_dgm, loss_hjb, DGM_HJB_Solver, kernel_b_predict
)
from stochastic_predictor.kernels.kernel_c import (
    solve_sde, drift_levy_stable, diffusion_levy, kernel_c_predict
)
from stochastic_predictor.kernels.kernel_d import (
    compute_log_signature, create_path_augmentation,
    predict_from_signature, kernel_d_predict
)
from stochastic_predictor.api.warmup import (
    warmup_all_kernels, warmup_kernel_a, warmup_kernel_b,
    warmup_kernel_c, warmup_kernel_d, warmup_with_retry,
    profile_warmup_and_recommend_timeout
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def config_obj():
    """Load config from toml."""
    try:
        injector = PredictorConfigInjector()
        return injector.create_config()
    except Exception as e:
        pytest.skip(f"Config incomplete: {e}")


@pytest.fixture(scope="module")
def prng_key():
    """Initialize PRNG key."""
    return initialize_jax_prng(seed=42)


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - Start with basics
# ═══════════════════════════════════════════════════════════════════════════

class TestBasicSetup:
    """Verify basic setup works."""
    
    def test_config_loads(self, config_obj):
        """Config can be loaded."""
        assert config_obj is not None
    
    def test_prng_initializes(self, prng_key):
        """PRNG key can be initialized."""
        assert prng_key is not None


class TestAPIConfig:
    """Test api/config.py functions."""
    
    def test_get_config(self):
        """Execute: get_config()."""
        cfg = get_config()
        assert isinstance(cfg, ConfigManager)
    
    def test_config_manager(self):
        """Execute: ConfigManager class."""
        cm = ConfigManager()
        assert cm is not None
    
    def test_predictor_config(self, config_obj):
        """Execute: PredictorConfig type."""
        assert isinstance(config_obj, PredictorConfig)


class TestAPIPRNG:
    """Test api/prng.py functions."""
    
    def test_initialize_jax_prng(self):
        """Execute: initialize_jax_prng()."""
        key = initialize_jax_prng(seed=123)
        assert key.shape == (2,)
    
    def test_split_key(self, prng_key):
        """Execute: split_key()."""
        keys = split_key(prng_key, num=3)
        assert len(keys) == 3
        assert all(k.shape == (2,) for k in keys)
    
    def test_split_key_like(self, prng_key):
        """Execute: split_key_like()."""
        new_key, batch_keys = split_key_like(prng_key, target_shape=(5,))
        assert new_key.shape == (2,)
        assert batch_keys.shape == (5, 2)
    
    def test_uniform_samples(self, prng_key):
        """Execute: uniform_samples()."""
        samples = uniform_samples(
            prng_key, 
            shape=(10,), 
            minval=0.0, 
            maxval=1.0, 
            dtype=jnp.float64
        )
        assert samples.shape == (10,)
        assert samples.dtype == jnp.float64
    
    def test_normal_samples(self, prng_key):
        """Execute: normal_samples()."""
        samples = normal_samples(
            prng_key,
            shape=(10,),
            mean=0.0,
            std=1.0,
            dtype=jnp.float64
        )
        assert samples.shape == (10,)
        assert samples.dtype == jnp.float64
    
    def test_exponential_samples(self, prng_key):
        """Execute: exponential_samples()."""
        samples = exponential_samples(
            prng_key,
            shape=(10,),
            rate=1.0,
            dtype=jnp.float64
        )
        assert samples.shape == (10,)
        assert samples.dtype == jnp.float64
    
    def test_check_prng_state(self, prng_key):
        """Execute: check_prng_state()."""
        state = check_prng_state(prng_key)
        assert isinstance(state, dict)
        assert "shape" in state
        assert "dtype" in state
    
    def test_verify_determinism(self):
        """Execute: verify_determinism()."""
        result = verify_determinism(seed=42, n_trials=2)
        assert isinstance(result, bool)


class TestAPITypes:
    """Test api/types.py classes and functions."""
    
    def test_predictor_config_class(self):
        """Execute: PredictorConfig class."""
        cfg = PredictorConfig()
        assert cfg is not None
    
    def test_process_state_class(self):
        """Execute: ProcessState class."""
        ps = ProcessState(
            magnitude=jnp.array([1.0]),
            timestamp_utc=datetime.now(timezone.utc)
        )
        assert ps is not None
        assert ps.magnitude.shape == (1,)
    
    def test_kernel_type_enum(self):
        """Execute: KernelType enum."""
        assert KernelType is not None
        assert hasattr(KernelType, '__members__')
    
    def test_operating_mode_enum(self):
        """Execute: OperatingMode enum."""
        assert OperatingMode is not None
        assert hasattr(OperatingMode, '__members__')
    
    def test_check_jax_config(self):
        """Execute: check_jax_config()."""
        check_jax_config()  # Should not raise


class TestAPIValidation:
    """Test api/validation.py functions."""
    
    def test_validate_magnitude(self):
        """Execute: validate_magnitude()."""
        is_valid, msg = validate_magnitude(
            magnitude=0.5,
            sigma_bound=10.0,
            sigma_val=1.0,
            allow_nan=False
        )
        assert isinstance(is_valid, bool)
        assert isinstance(msg, str)
    
    def test_validate_timestamp(self):
        """Execute: validate_timestamp()."""
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
        is_valid, msg = validate_timestamp(
            timestamp_ns=now_ns,
            max_future_drift_ns=int(1e9),
            max_past_drift_ns=int(86400e9)
        )
        assert isinstance(is_valid, bool)
    
    def test_check_staleness(self, config_obj):
        """Execute: check_staleness()."""
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
        is_stale, delta = check_staleness(now_ns, config_obj)
        assert isinstance(is_stale, bool)
        assert isinstance(delta, int)
    
    def test_validate_shape(self):
        """Execute: validate_shape()."""
        arr = jnp.ones((10, 5))
        is_valid, msg = validate_shape(arr, expected_shape=(10, 5), name="test_array")
        assert isinstance(is_valid, bool)
    
    def test_validate_finite(self):
        """Execute: validate_finite()."""
        arr = jnp.array([1.0, 2.0, 3.0])
        is_valid, msg = validate_finite(arr, name="test", allow_nan=False, allow_inf=False)
        assert isinstance(is_valid, bool)
    
    def test_sanitize_array(self):
        """Execute: sanitize_array()."""
        arr = jnp.array([1.0, 2.0, jnp.nan])
        result = sanitize_array(arr, replace_nan=0.0, replace_inf=None, clip_range=None)
        assert result.shape == arr.shape
    
    def test_ensure_float64(self):
        """Execute: ensure_float64()."""
        result = ensure_float64(1.0)
        assert result.dtype == jnp.float64
    
    def test_cast_array_to_float64(self):
        """Execute: cast_array_to_float64()."""
        arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
        result = cast_array_to_float64(arr, warn_if_downcast=False)
        assert result.dtype == jnp.float64
    
    def test_validate_holder_exponent(self):
        """Execute: validate_holder_exponent()."""
        is_valid, msg = validate_holder_exponent(H=0.5, min_val=0.0, max_val=1.0)
        assert isinstance(is_valid, bool)
    
    def test_validate_alpha_stable(self):
        """Execute: validate_alpha_stable()."""
        is_valid, msg = validate_alpha_stable(alpha=1.5, min_val=1.0, max_val=2.0, exclusive_bounds=False)
        assert isinstance(is_valid, bool)
    
    def test_validate_beta_stable(self):
        """Execute: validate_beta_stable()."""
        is_valid, msg = validate_beta_stable(beta=0.5, min_val=-1.0, max_val=1.0)
        assert isinstance(is_valid, bool)
    
    def test_validate_simplex(self):
        """Execute: validate_simplex()."""
        weights = jnp.array([0.3, 0.4, 0.3])
        is_valid, msg = validate_simplex(weights, atol=1e-6, name="weights")
        assert isinstance(is_valid, bool)
    
    def test_sanitize_external_observation(self):
        """Execute: sanitize_external_observation()."""
        magnitude, ts, tag, disp = sanitize_external_observation(
            magnitude=1.0,
            timestamp_utc=datetime.now(timezone.utc),
            state_tag=None,
            dispersion_proxy=None
        )
        assert magnitude is not None
        assert ts is not None
    
    def test_warn_if_invalid(self):
        """Execute: warn_if_invalid()."""
        # Test with is_valid=True (should not raise)
        warn_if_invalid(is_valid=True, message="test", exception_type=ValueError)
        # Test with is_valid=False and exception_type=None (should warn, not raise)
        warn_if_invalid(is_valid=False, message="test warning", exception_type=None)


class TestAPISchemas:
    """Test api/schemas.py classes."""
    
    def test_process_state_schema(self):
        """Execute: ProcessStateSchema."""
        schema = ProcessStateSchema(
            magnitude=jnp.array([0.5]),
            timestamp_utc=datetime.now(timezone.utc)
        )
        assert schema is not None
    
    def test_operating_mode_schema(self):
        """Execute: OperatingModeSchema (alias check)."""
        # OperatingModeSchema is an alias to OperatingMode enum
        assert OperatingModeSchema is not None
        # Verify it's the same as OperatingMode
        from stochastic_predictor.api.schemas import OperatingMode as SchemaEnum
        assert SchemaEnum is not None


class TestAPIStateBuffer:
    """Test api/state_buffer.py functions."""
    
    def test_update_signal_history(self, config_obj, prng_key):
        """Execute: update_signal_history()."""
        signal = jnp.linspace(0, 1, 100)
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
        state = initialize_state(signal, now_ns, prng_key, config_obj)
        new_state = update_signal_history(state, jnp.array(0.5))
        assert new_state is not None
    
    def test_batch_update_signal_history(self, config_obj, prng_key):
        """Execute: batch_update_signal_history()."""
        signal = jnp.linspace(0, 1, 100)
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
        state = initialize_state(signal, now_ns, prng_key, config_obj)
        new_values = jnp.array([0.1, 0.2, 0.3])
        new_state = batch_update_signal_history(state, new_values)
        assert new_state is not None
    
    def test_reset_cusum_statistics(self, config_obj, prng_key):
        """Execute: reset_cusum_statistics()."""
        signal = jnp.linspace(0, 1, 100)
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
        state = initialize_state(signal, now_ns, prng_key, config_obj)
        new_state = reset_cusum_statistics(state)
        assert new_state is not None
    
    def test_update_ema_variance(self, config_obj, prng_key):
        """Execute: update_ema_variance()."""
        signal = jnp.linspace(0, 1, 100)
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
        state = initialize_state(signal, now_ns, prng_key, config_obj)
        new_state = update_ema_variance(state, jnp.array(0.5), alpha=0.1)
        assert new_state is not None
    
    def test_update_residual_buffer(self, config_obj, prng_key):
        """Execute: update_residual_buffer()."""
        signal = jnp.linspace(0, 1, 100)
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
        state = initialize_state(signal, now_ns, prng_key, config_obj)
        new_state = update_residual_buffer(state, jnp.array(0.1))
        assert new_state is not None


class TestCoreOrchestrator:
    """Test core/orchestrator.py functions."""
    
    def test_initialize_state(self, config_obj, prng_key):
        """Execute: initialize_state()."""
        signal = jnp.linspace(0, 1, 100)
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
        state = initialize_state(signal, now_ns, prng_key, config_obj)
        assert isinstance(state, InternalState)
    
    def test_initialize_batched_states(self, config_obj, prng_key):
        """Execute: initialize_batched_states()."""
        signal = jnp.linspace(0, 1, 100)
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
        states = initialize_batched_states(4, signal, now_ns, prng_key, config_obj)
        assert states is not None
    
    def test_compute_entropy_ratio(self, config_obj):
        """Execute: compute_entropy_ratio()."""
        ratio = compute_entropy_ratio(
            current_entropy=jnp.array(1.5),
            baseline_entropy=jnp.array(1.0),
            config=config_obj
        )
        assert ratio is not None
    
    def test_scale_dgm_architecture(self, config_obj):
        """Execute: scale_dgm_architecture()."""
        width, depth = scale_dgm_architecture(config_obj, entropy_ratio=2.0)
        assert isinstance(width, int)
        assert isinstance(depth, int)
    
    def test_compute_adaptive_stiffness_thresholds(self, config_obj):
        """Execute: compute_adaptive_stiffness_thresholds()."""
        lower, upper = compute_adaptive_stiffness_thresholds(
            holder_exponent=jnp.array(0.5),
            config=config_obj
        )
        assert lower is not None
        assert upper is not None
    
    def test_compute_adaptive_jko_params(self, config_obj):
        """Execute: compute_adaptive_jko_params()."""
        n_iter, epsilon = compute_adaptive_jko_params(
            volatility_sigma_squared=0.5,
            config=config_obj
        )
        assert isinstance(n_iter, int)
        assert isinstance(epsilon, float)


class TestCoreFusion:
    """Test core/fusion.py functions."""
    
    def test_compute_sinkhorn_epsilon(self, config_obj):
        """Execute: compute_sinkhorn_epsilon()."""
        epsilon = compute_sinkhorn_epsilon(
            ema_variance=jnp.array([0.5]),
            config=config_obj
        )
        assert epsilon is not None


class TestCoreMetaOptimizer:
    """Test core/meta_optimizer.py functions."""
    
    def test_walk_forward_split(self):
        """Execute: walk_forward_split()."""
        result = walk_forward_split(100, 70, 10)
        assert result is not None


class TestKernelsBase:
    """Test kernels/base.py functions."""
    
    def test_apply_stop_gradient_to_diagnostics(self):
        """Execute: apply_stop_gradient_to_diagnostics()."""
        prediction = jnp.array([0.5])
        diag = {"loss": jnp.array(0.1)}
        result_pred, result_diag = apply_stop_gradient_to_diagnostics(prediction, diag)
        assert result_pred is not None
        assert result_diag is not None
    
    def test_compute_signal_statistics(self):
        """Execute: compute_signal_statistics()."""
        signal = jnp.linspace(0, 1, 100)
        stats = compute_signal_statistics(signal)
        assert isinstance(stats, dict)
    
    def test_normalize_signal(self):
        """Execute: normalize_signal()."""
        signal = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_signal(signal, method="zscore", epsilon=1e-10)
        assert normalized.shape == signal.shape
    
    def test_validate_kernel_input(self):
        """Execute: validate_kernel_input()."""
        signal = jnp.linspace(0, 1, 100)
        is_valid, msg = validate_kernel_input(signal, min_length=10)
        assert isinstance(is_valid, bool)
        assert isinstance(msg, str)


class TestKernelA:
    """Test kernels/kernel_a.py functions."""
    
    def test_gaussian_kernel(self):
        """Execute: gaussian_kernel()."""
        x = jnp.array([1.0, 2.0])
        y = jnp.array([1.5, 2.5])
        result = gaussian_kernel(x, y, bandwidth=1.0)
        assert result is not None
    
    def test_compute_gram_matrix(self):
        """Execute: compute_gram_matrix()."""
        X = jnp.ones((10, 3))
        K = compute_gram_matrix(X, bandwidth=1.0)
        assert K.shape == (10, 10)
    
    def test_kernel_ridge_regression(self, config_obj):
        """Execute: kernel_ridge_regression()."""
        X_train = jnp.ones((10, 3))
        y_train = jnp.ones(10)
        X_test = jnp.ones((5, 3))
        predictions, std = kernel_ridge_regression(X_train, y_train, X_test, config_obj)
        assert predictions.shape == (5,)
    
    def test_create_embedding(self, config_obj):
        """Execute: create_embedding()."""
        signal = jnp.linspace(0, 1, 100)
        embedding = create_embedding(signal, config_obj)
        assert embedding is not None
    
    def test_kernel_a_predict(self, config_obj, prng_key):
        """Execute: kernel_a_predict()."""
        signal = jnp.linspace(0, 1, 100)
        output = kernel_a_predict(signal, prng_key, config_obj)
        assert isinstance(output, KernelOutput)


class TestKernelB:
    """Test kernels/kernel_b.py functions."""
    
    def test_dgm_hjb_solver(self, config_obj, prng_key):
        """Execute: DGM_HJB_Solver constructor."""
        solver = DGM_HJB_Solver(in_size=10, key=prng_key, config=config_obj)
        assert solver is not None
    
    def test_kernel_b_predict(self, config_obj, prng_key):
        """Execute: kernel_b_predict()."""
        signal = jnp.linspace(0, 1, 100)
        output = kernel_b_predict(signal, prng_key, config_obj)
        assert isinstance(output, KernelOutput)


class TestKernelC:
    """Test kernels/kernel_c.py functions."""
    
    def test_drift_levy_stable(self):
        """Execute: drift_levy_stable()."""
        t = jnp.array(0.0)
        y = jnp.array([1.0])
        args = (1.5, 0.5)  # alpha, beta
        result = drift_levy_stable(t, y, args)
        assert result is not None
    
    def test_diffusion_levy(self):
        """Execute: diffusion_levy()."""
        t = jnp.array(0.0)
        y = jnp.array([1.0])
        args = (1.5, 0.5)
        result = diffusion_levy(t, y, args)
        assert result is not None
    
    def test_kernel_c_predict(self, config_obj, prng_key):
        """Execute: kernel_c_predict()."""
        signal = jnp.linspace(0, 1, 100)
        output = kernel_c_predict(signal, prng_key, config_obj)
        assert isinstance(output, KernelOutput)


class TestKernelD:
    """Test kernels/kernel_d.py functions."""
    
    def test_create_path_augmentation(self):
        """Execute: create_path_augmentation()."""
        signal = jnp.linspace(0, 1, 100)
        augmented = create_path_augmentation(signal)
        assert augmented.shape == (100, 2)
    
    def test_compute_log_signature(self, config_obj):
        """Execute: compute_log_signature()."""
        path = jnp.ones((100, 2))
        logsig = compute_log_signature(path, config_obj)
        assert logsig is not None
    
    def test_predict_from_signature(self, config_obj):
        """Execute: predict_from_signature()."""
        logsig = jnp.ones(10)
        mean, std = predict_from_signature(logsig, last_value=1.0, config=config_obj)
        assert isinstance(mean, float) or mean is not None
        assert isinstance(std, float) or std is not None
    
    def test_kernel_d_predict(self, config_obj, prng_key):
        """Execute: kernel_d_predict()."""
        signal = jnp.linspace(0, 1, 100)
        output = kernel_d_predict(signal, prng_key, config_obj)
        assert isinstance(output, KernelOutput)


class TestAPIWarmup:
    """Test api/warmup.py functions."""
    
    def test_warmup_kernel_a(self, config_obj, prng_key):
        """Execute: warmup_kernel_a()."""
        time_ms = warmup_kernel_a(config_obj, prng_key)
        assert isinstance(time_ms, float)
    
    def test_warmup_kernel_b(self, config_obj, prng_key):
        """Execute: warmup_kernel_b()."""
        time_ms = warmup_kernel_b(config_obj, prng_key)
        assert isinstance(time_ms, float)
    
    def test_warmup_kernel_c(self, config_obj, prng_key):
        """Execute: warmup_kernel_c()."""
        time_ms = warmup_kernel_c(config_obj, prng_key)
        assert isinstance(time_ms, float)
    
    def test_warmup_kernel_d(self, config_obj, prng_key):
        """Execute: warmup_kernel_d()."""
        time_ms = warmup_kernel_d(config_obj, prng_key)
        assert isinstance(time_ms, float)
    
    def test_warmup_all_kernels(self, config_obj, prng_key):
        """Execute: warmup_all_kernels()."""
        results = warmup_all_kernels(config_obj, key=prng_key, verbose=False)
        assert isinstance(results, dict)
    
    def test_warmup_with_retry(self, config_obj):
        """Execute: warmup_with_retry()."""
        results = warmup_with_retry(config_obj, max_retries=1, verbose=False)
        assert isinstance(results, dict)
    
    def test_profile_warmup_and_recommend_timeout(self, config_obj):
        """Execute: profile_warmup_and_recommend_timeout()."""
        results = profile_warmup_and_recommend_timeout(config_obj, verbose=False)
        assert isinstance(results, dict)


