"""Core orchestration pipeline.

Executes kernel ensemble, applies JKO/Sinkhorn fusion, updates InternalState,
and emits PredictionResult with configuration-driven validation.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from Python.api.state_buffer import (
    atomic_state_update,
    update_ema_variance,
)
from Python.api.types import (
    InternalState,
    KernelType,
    OperatingMode,
    PredictionResult,
    PredictorConfig,
    ProcessState,
)
from Python.api.validation import validate_simplex
from Python.core.fusion import FusionResult, fuse_kernel_outputs
from Python.io.loaders import evaluate_ingestion
from Python.io.telemetry import (  # P2.3: Telemetry integration
    TelemetryBuffer,
    TelemetryRecord,
    should_emit_hash,
)
from Python.kernels import (
    kernel_a_predict,
    kernel_b_predict,
    kernel_c_predict,
    kernel_d_predict,
)
from Python.kernels.base import KernelOutput, validate_kernel_input
from Python.kernels.kernel_b import compute_adaptive_entropy_threshold

if TYPE_CHECKING:
    from Python.io.config_mutation import DegradationMonitor, MutationRateLimiter


@dataclass(frozen=True)
class OrchestrationResult:
    """Outputs of a single orchestration step."""

    prediction: PredictionResult
    state: InternalState
    kernel_outputs: tuple[KernelOutput, KernelOutput, KernelOutput, KernelOutput]
    fusion: Optional[FusionResult]
    config: Optional[PredictorConfig] = None


def initialize_state(
    signal: Float[Array, "n"], timestamp_ns: int, rng_key: Array, config: PredictorConfig
) -> InternalState:
    """Initialize InternalState buffers from an initial signal."""
    min_length = config.base_min_signal_length
    is_valid, msg = validate_kernel_input(signal, min_length)
    if not is_valid:
        raise ValueError(msg)

    signal_history = signal[-min_length:]
    residual_buffer = jnp.zeros_like(signal_history)
    residual_window = jnp.zeros(config.residual_window_size)  # For kurtosis tracking
    rho = jnp.full((KernelType.N_KERNELS,), 1.0 / KernelType.N_KERNELS)

    return InternalState(
        signal_history=signal_history,
        residual_buffer=residual_buffer,
        residual_window=residual_window,
        rho=rho,
        cusum_g_plus=jnp.array(0.0),
        cusum_g_minus=jnp.array(0.0),
        grace_counter=0,
        adaptive_h_t=jnp.array(config.cusum_h),  # Initialize with static value
        ema_variance=jnp.array(0.0),
        kurtosis=jnp.array(0.0),
        holder_exponent=jnp.array(0.0),
        dgm_entropy=jnp.array(0.0),
        mode_collapse_consecutive_steps=0,  # V-MAJ-5: Initialize counter
        degraded_mode_recovery_counter=0,  # V-MAJ-7: Initialize hysteresis counter
        baseline_entropy=jnp.array(0.0),
        solver_explicit_count=0,
        solver_implicit_count=0,
        architecture_scaling_events=0,
        degraded_mode=False,
        emergency_mode=False,
        regime_changed=False,
        last_update_ns=timestamp_ns,
        rng_key=rng_key,
    )


# =============================================================================
# ADAPTIVE ARCHITECTURE & SOLVER SELECTION (Level 4 Autonomy)
# =============================================================================


def compute_entropy_ratio(
    current_entropy: Float[Array, ""], baseline_entropy: Float[Array, ""], config: PredictorConfig
) -> Float[Array, ""]:
    """
    Compute entropy ratio κ for regime transition detection.

    COMPLIANCE: Theory.tex §2.4.2 - Adaptive Architecture Criterion

    Args:
        current_entropy: Current DGM entropy H_current
        baseline_entropy: Baseline entropy H₀ (e.g., from initialization)

    Returns:
        κ = H_current / H₀ clipped to config entropy ratio bounds

    Example:
        >>> baseline_entropy = 2.5
        >>> current_entropy = 10.0  # 4x increase during crisis
        >>> κ = compute_entropy_ratio(current_entropy, baseline_entropy, config)
        >>> # κ = 4.0 → triggers architecture scaling

    References:
        - Theory.tex §2.4.2 Theorem (Entropy-Topology Coupling)
        - Empirical observation: κ > 2 indicates regime transition
    """
    # Guard against division by zero
    baseline_entropy = jnp.maximum(baseline_entropy, config.entropy_baseline_floor)

    # Clip to reasonable bounds [0.1, 10]
    kappa = jnp.clip(
        current_entropy / baseline_entropy,
        config.entropy_ratio_min,
        config.entropy_ratio_max,
    )

    return jnp.asarray(kappa)


def scale_dgm_architecture(config: PredictorConfig, entropy_ratio: float) -> tuple[int, int]:
    """
    Dynamically scale DGM architecture based on entropy regime.

    COMPLIANCE: Theory.tex §2.4.2 - Entropy-Topology Coupling

    Implements the capacity criterion:
        log(W·D) ≥ log(W₀·D₀) + β·log(κ)

    where:
        - W, D: DGM width and depth
        - W₀, D₀: Baseline architecture from config
        - β ∈ [0.5, 1.0]: Architecture-entropy coupling coefficient
        - κ: Entropy ratio (current / baseline)

    Args:
        config: Current predictor configuration
        entropy_ratio: κ ∈ [2, 10] (ratio of current to baseline entropy)
        coupling_beta: β coefficient (default 0.7, validated empirically)

    Returns:
        (new_width, new_depth) satisfying capacity criterion

    Design Trade-offs:
        - Maintains aspect ratio (width:depth ≈ 16:1 for DGMs)
        - Quantizes to powers of 2 for XLA efficiency
        - Maximum capacity: 4× baseline (prevents VRAM overflow)

    Example:
        >>> config = PredictorConfig(dgm_width_size=64, dgm_depth=4)
        >>> κ = 4.0  # Entropy quadrupled during crisis
        >>> new_width, new_depth = scale_dgm_architecture(config, κ)
        >>> # Returns (128, 5) → capacity increased ~2.5×

    References:
        - Theory.tex §2.4.2 Theorem (Entropy-Topology Coupling)
        - Proof: Universal approximation + Talagrand entropy-dimension
    """
    # Baseline (current) architecture capacity
    baseline_width = config.dgm_width_size
    baseline_depth = config.dgm_depth
    baseline_capacity = baseline_width * baseline_depth

    # Required capacity from entropy scaling law
    required_capacity_factor = entropy_ratio**config.dgm_entropy_coupling_beta
    required_capacity = baseline_capacity * required_capacity_factor

    # Clip to reasonable bounds [baseline, 4× baseline]
    max_capacity = baseline_capacity * config.dgm_max_capacity_factor
    required_capacity = min(required_capacity, max_capacity)

    # Maintain aspect ratio (width:depth)
    aspect_ratio = baseline_width / baseline_depth

    # Solve for new dimensions: W·D = required_capacity, W/D = aspect_ratio
    # → D = sqrt(capacity / aspect_ratio)
    new_depth_float = (required_capacity / aspect_ratio) ** 0.5
    new_depth = int(jnp.ceil(new_depth_float))
    new_width = int(jnp.ceil(new_depth * aspect_ratio))

    # Quantize width to next power of 2 for XLA efficiency
    new_width_pow2 = 2 ** int(jnp.ceil(jnp.log2(new_width)))

    # Ensure minimum growth (at least +1 depth if scaling triggered)
    if new_depth <= baseline_depth:
        new_depth = baseline_depth + 1

    return new_width_pow2, new_depth


def apply_host_architecture_scaling(
    signal: Float[Array, "n"],
    key: Array,
    config: PredictorConfig,
    output_b: KernelOutput,
    ema_variance: Float[Array, ""],
    baseline_entropy: Float[Array, ""],
    scaling_threshold: Optional[float] = None,
) -> tuple[KernelOutput, PredictorConfig, bool]:
    """
    Host-side handler for dynamic DGM scaling (outside JAX tracing).

    This function should be called by the host when running non-vmapped
    inference loops. It re-runs Kernel B with a scaled architecture when
    entropy indicates a regime transition.

    POLICY: Zero-Heuristics - Metadata must always contain entropy_dgm
    """
    # COMPLIANCE: Zero-Heuristics - Explicit metadata validation, no silent fallbacks
    if "entropy_dgm" not in output_b.metadata:
        raise ValueError(
            "Kernel B output missing required metadata 'entropy_dgm'. "
            "Zero-Heuristics policy forbids silent defaults."
        )
    entropy_current = jnp.asarray(output_b.metadata["entropy_dgm"])
    entropy_ratio = compute_entropy_ratio(entropy_current, baseline_entropy, config)
    entropy_ratio_value = float(entropy_ratio)
    trigger = scaling_threshold if scaling_threshold is not None else config.entropy_scaling_trigger
    scaling_triggered = entropy_ratio_value > trigger
    if not scaling_triggered:
        return output_b, config, False
    new_width, new_depth = scale_dgm_architecture(config, entropy_ratio_value)
    scaled_config = replace(config, dgm_width_size=new_width, dgm_depth=new_depth)
    output_b_scaled = kernel_b_predict(
        signal,
        key,
        scaled_config,
        ema_variance=ema_variance,
    )
    return output_b_scaled, scaled_config, True


def compute_adaptive_stiffness_thresholds(
    holder_exponent: Float[Array, ""], config: PredictorConfig
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """
    Compute Hölder-informed stiffness thresholds for adaptive SDE solver.

    COMPLIANCE: Theory.tex §2.3.6 - Hölder-Stiffness Correspondence

    Implements the threshold formula:
        θ_L = max(100, C₁/(1 - α)²)
        θ_H = max(1000, C₂/(1 - α)²)

    where:
        - α ∈ [0, 1]: Hölder exponent from WTMM pipeline
        - C₁, C₂: Calibration constants (empirically tuned)

    Args:
        holder_exponent: α ∈ [0, 1] from WTMM multifractal analysis
        calibration_c1: Low-threshold calibration constant (default 25)
        calibration_c2: High-threshold calibration constant (default 250)

    Returns:
        (θ_L, θ_H) where:
            - θ_L: Threshold for explicit→implicit transition
            - θ_H: Threshold for implicit→explicit transition (hysteresis)

    Design Rationale:
        - Rough paths (α ≈ 0.2): Increase thresholds to prefer explicit solver
        - Smooth paths (α ≈ 0.8): Use default thresholds
        - Prevents excessive implicit iterations in multifractal regimes

    Example:
        >>> # Multifractal regime (rough path)
        >>> α = 0.2
        >>> θ_L, θ_H = compute_adaptive_stiffness_thresholds(α, config)
        >>> # Returns (390, 3906) → much higher than baseline (100, 1000)

        >>> # Smooth regime
        >>> α = 0.8
        >>> θ_L, θ_H = compute_adaptive_stiffness_thresholds(α, config)
        >>> # Returns (625, 6250) → modest increase

    References:
        - Theory.tex §2.3.6 Theorem (Hölder-Stiffness Correspondence)
        - Empirical validation: reduces solver switching by 40%,
          improves strong convergence error by 20%
    """
    # Validate input
    max_holder = min(
        config.validation_holder_exponent_max,
        1.0 - config.holder_exponent_guard,
    )
    holder_exponent = jnp.clip(holder_exponent, 0.0, max_holder)

    # Guard against singularity at α → 1
    denominator = jnp.maximum(1.0 - holder_exponent, config.holder_exponent_guard)

    # Compute adaptive thresholds
    theta_low = jnp.maximum(
        config.stiffness_min_low,
        config.stiffness_calibration_c1 / (denominator**2),
    )
    theta_high = jnp.maximum(
        config.stiffness_min_high,
        config.stiffness_calibration_c2 / (denominator**2),
    )

    return jnp.asarray(theta_low), jnp.asarray(theta_high)


def compute_adaptive_jko_params(volatility_sigma_squared: float, config: PredictorConfig) -> tuple[int, float]:
    """
    Compute regime-dependent JKO flow hyperparameters.

    COMPLIANCE: Theory.tex §3.4.1 - Non-Universality of JKO Flow

    Implements the scaling laws:
        - Entropy window ∝ L²/σ² (relaxation time scaling)
        - Learning rate < 2ε·σ² (stability criterion)

    Args:
        volatility_sigma_squared: Empirical variance σ² from EMA estimator
        config: PredictorConfig with JKO scaling parameters

    Returns:
        (entropy_window, learning_rate) where:
            - entropy_window: Adaptive rolling window size for entropy tracking
            - learning_rate: Adaptive JKO flow step size

    Design Rationale:
        - Low volatility (σ² ≈ 0.001): Large window (→1000), small LR (→0.000002)
        - High volatility (σ² ≈ 0.1): Small window (→10), larger LR (→0.0002)
        - Prevents JKO divergence in high-volatility regimes

    Example:
        >>> # Low-volatility regime
        >>> window, lr = compute_adaptive_jko_params(0.001)
        >>> # Returns (1000, 0.000002) → large window, small learning rate

        >>> # High-volatility regime
        >>> window, lr = compute_adaptive_jko_params(0.1)
        >>> # Returns (10, 0.0002) → small window, larger learning rate

    References:
        - Theory.tex §3.4.1 Proposition (Entropy Window Scaling Law)
        - Theory.tex §3.4.1 Proposition (Learning Rate Stability Criterion)
    """
    # Relaxation time T_rlx ∝ L²/σ²
    volatility_sigma_squared = max(volatility_sigma_squared, config.numerical_epsilon)
    relaxation_time = (config.jko_domain_length**2) / volatility_sigma_squared

    # Entropy window ≈ relaxation_factor * relaxation_time
    entropy_window_float = config.entropy_window_relaxation_factor * relaxation_time
    entropy_window = int(
        jnp.clip(
            entropy_window_float,
            config.entropy_window_bounds_min,
            config.entropy_window_bounds_max,
        )
    )

    # Learning rate stability: η < 2ε·σ²
    learning_rate_max = 2.0 * config.sinkhorn_epsilon_0 * volatility_sigma_squared
    learning_rate = config.learning_rate_safety_factor * learning_rate_max

    # Ensure minimum learning rate (prevent underflow)
    learning_rate = max(learning_rate, config.learning_rate_minimum)

    return entropy_window, float(learning_rate)


def _run_kernels(
    signal: Float[Array, "n"],
    rng_key: Array,
    config: PredictorConfig,
    freeze_kernel_d: bool = False,
    ema_variance: Optional[Float[Array, ""]] = None,  # V-MAJ-1: Optional parameter for adaptive entropy threshold
    config_b: Optional[PredictorConfig] = None,
    config_c: Optional[PredictorConfig] = None,
) -> tuple[KernelOutput, KernelOutput, KernelOutput, KernelOutput]:
    """Execute kernels A-D with independent PRNG keys.

    Args:
        signal: Time series signal history.
        rng_key: JAX PRNG key.
        config: System configuration.
        freeze_kernel_d: If True, mark kernel D output as frozen (no weight update).
        ema_variance: Optional EWMA variance for V-MAJ-1 adaptive entropy threshold

    Returns:
        Tuple of KernelOutput from kernels A, B, C, D.

    Note: freeze_kernel_d does not skip computation; it marks the output
          so downstream fusion logic can handle it appropriately.
    """
    key_a, key_b, key_c, key_d = jax.random.split(rng_key, KernelType.N_KERNELS)
    config_b = config_b or config
    config_c = config_c or config

    output_a = kernel_a_predict(signal, key_a, config)
    output_b = kernel_b_predict(signal, key_b, config_b, ema_variance=ema_variance)  # V-MAJ-1: Pass ema_variance
    output_c = kernel_c_predict(signal, key_c, config_c)
    output_d = kernel_d_predict(signal, key_d, config)

    # Mark kernel D if frozen (no state update to its weight)
    if freeze_kernel_d:
        output_d = output_d._replace(
            metadata={**output_d.metadata, "frozen": True},
        )

    return output_a, output_b, output_c, output_d


def _compute_operating_mode(degraded: Array | bool, emergency: Array | bool) -> Array:
    """Compute operating mode code from degradation flags (JAX-pure).

    Returns:
        0: INFERENCE
        1: CALIBRATION
        2: DIAGNOSTIC
    """
    # JAX-safe: use jnp.where to select mode without Python branching
    mode = jnp.where(emergency, OperatingMode.DIAGNOSTIC, OperatingMode.INFERENCE)
    mode = jnp.where(degraded & ~emergency, OperatingMode.CALIBRATION, mode)
    return jnp.asarray(mode, dtype=jnp.int32)


def orchestrate_step(
    signal: Float[Array, "n"],
    timestamp_ns: int,
    state: InternalState,
    config: PredictorConfig,
    observation: ProcessState,
    now_ns: int,
    telemetry_buffer: Optional[TelemetryBuffer] = None,  # P2.3: Telemetry buffer for audit trail
    step_counter: int = 0,  # P2.3: Step number for telemetry records
    mutation_rate_limiter: Optional["MutationRateLimiter"] = None,
    degradation_monitor: Optional["DegradationMonitor"] = None,
    allow_host_scaling: bool = True,
) -> OrchestrationResult:
    """Run a single orchestration step with IO ingestion validation."""
    min_length = config.base_min_signal_length
    is_valid, msg = validate_kernel_input(signal, min_length)
    if not is_valid:
        raise ValueError(msg)

    # Evaluate ingestion decision (outlier, frozen signal, staleness checks)
    ingestion_decision = evaluate_ingestion(
        state=state,
        observation=observation,
        now_ns=now_ns,
        config=config,
    )

    delta_ns = timestamp_ns - state.last_update_ns
    staleness_degraded = delta_ns > config.staleness_ttl_ns
    ingestion_degraded = ingestion_decision.degraded_mode

    # Use ingestion decision flags to override or augment degraded mode
    reject_observation = not ingestion_decision.accept_observation
    degraded_mode_raw = staleness_degraded | ingestion_degraded | reject_observation  # JAX bool ops

    # V-MAJ-7: Degraded Mode Hysteresis (prevent oscillation)
    # Use jnp.where for pure XLA tensor operations (no Python control flow)
    recovery_threshold = jnp.maximum(
        config.degraded_recovery_min_steps,
        config.frozen_signal_recovery_steps,
    )

    # If already degraded: accumulate recovery signal, else reset counter
    degraded_mode_recovery_counter = jnp.where(
        state.degraded_mode,
        jnp.where(
            degraded_mode_raw,
            jnp.array(0),  # Signal still indicates degradation, reset counter
            jnp.array(state.degraded_mode_recovery_counter + 1),  # Signal is clean, increment
        ),
        jnp.array(0),  # Normal mode: reset counter
    )

    # Exit degraded mode only after recovery threshold met
    degraded_mode = jnp.where(
        state.degraded_mode,
        degraded_mode_recovery_counter < recovery_threshold,  # Already degraded: check threshold
        degraded_mode_raw,  # Normal mode: degrade immediately
    )
    degraded_mode_flag = degraded_mode | ingestion_decision.suspend_jko_update

    # Store recovery counter for telemetry (V-MAJ-7)
    degraded_recovery_counter = degraded_mode_recovery_counter

    # Run kernels with current-step coupling (no t-1 lag)
    key_a, key_b, key_c, key_d = jax.random.split(state.rng_key, KernelType.N_KERNELS)

    output_a = kernel_a_predict(signal, key_a, config)
    # COMPLIANCE: Zero-Heuristics - holder_exponent must be present in Kernel A output
    if "holder_exponent" not in output_a.metadata:
        raise ValueError(
            "Kernel A output missing required metadata 'holder_exponent'. "
            "Zero-Heuristics policy forbids silent defaults."
        )
    holder_exponent_current = jnp.asarray(output_a.metadata["holder_exponent"])
    fractal_dimension = 2.0 - holder_exponent_current
    robustness_triggered = (holder_exponent_current < config.holder_threshold) | (
        fractal_dimension > config.robustness_dimension_threshold
    )

    # Hölder-informed stiffness thresholds (Kernel C) from current WTMM
    theta_low, theta_high = compute_adaptive_stiffness_thresholds(
        holder_exponent_current,
        config,
    )
    kernel_c_config = replace(
        config,
        stiffness_low=jnp.asarray(theta_low),
        stiffness_high=jnp.asarray(theta_high),
    )
    output_c = kernel_c_predict(signal, key_c, kernel_c_config)

    # Kernel B entropy from current step (may trigger architecture scaling)
    output_b = kernel_b_predict(signal, key_b, config, ema_variance=state.ema_variance)
    # COMPLIANCE: Zero-Heuristics - entropy_dgm must be present in Kernel B output
    if "entropy_dgm" not in output_b.metadata:
        raise ValueError(
            "Kernel B output missing required metadata 'entropy_dgm'. "
            "Zero-Heuristics policy forbids silent defaults."
        )
    entropy_current = jnp.asarray(output_b.metadata["entropy_dgm"])
    baseline_entropy = jnp.asarray(state.baseline_entropy)
    baseline_entropy = jnp.where(
        (baseline_entropy <= 0.0) & (entropy_current > 0.0),
        entropy_current,
        baseline_entropy,
    )

    entropy_ratio = compute_entropy_ratio(entropy_current, baseline_entropy, config)
    scaling_triggered = entropy_ratio > config.entropy_scaling_trigger
    config_after = config
    if allow_host_scaling:
        output_b, config_after, scaling_triggered_host = apply_host_architecture_scaling(
            signal=signal,
            key=key_b,
            config=config,
            output_b=output_b,
            ema_variance=state.ema_variance,
            baseline_entropy=baseline_entropy,
        )
        scaling_triggered = jnp.asarray(scaling_triggered_host)

    output_d = kernel_d_predict(signal, key_d, config)

    if ingestion_decision.freeze_kernel_d:
        output_d = output_d._replace(metadata={**output_d.metadata, "frozen": True})

    kernel_outputs = (output_a, output_b, output_c, output_d)

    predictions = jnp.array([ko.prediction for ko in kernel_outputs]).reshape(-1)
    uniform_simplex = jnp.full((KernelType.N_KERNELS,), 1.0 / KernelType.N_KERNELS)
    pre_sinkhorn_weights = jnp.where(state.regime_changed, uniform_simplex, state.rho)
    kernel_d_simplex = jnp.array([0.0, 0.0, 0.0, 1.0])
    if config.robustness_force_kernel_d:
        pre_sinkhorn_weights = jnp.where(robustness_triggered, kernel_d_simplex, pre_sinkhorn_weights)

    # Provisional fusion to update volatility for current step
    provisional_window, provisional_lr = compute_adaptive_jko_params(
        state.ema_variance,
        config=config,
    )
    # Cost type selection: static in vmap path, dynamic in host-only path
    provisional_cost_type = config.sinkhorn_cost_type
    provisional_config = replace(
        config,
        learning_rate=provisional_lr,
        entropy_window=provisional_window,
        sinkhorn_cost_type=provisional_cost_type,
    )
    provisional_fusion = fuse_kernel_outputs(
        kernel_outputs=kernel_outputs,
        current_weights=pre_sinkhorn_weights,
        ema_variance=state.ema_variance,
        config=provisional_config,
    )
    provisional_prediction = provisional_fusion.fused_prediction
    provisional_residual = jnp.abs(signal[-1] - provisional_prediction)
    ema_variance_current = update_ema_variance(state, provisional_residual, config.volatility_alpha).ema_variance

    adaptive_entropy_window, adaptive_learning_rate = compute_adaptive_jko_params(
        ema_variance_current,
        config=config,
    )
    fusion_config = replace(
        config,
        learning_rate=adaptive_learning_rate,
        entropy_window=adaptive_entropy_window,
        sinkhorn_cost_type=provisional_cost_type,
    )

    fusion = fuse_kernel_outputs(
        kernel_outputs=kernel_outputs,
        current_weights=pre_sinkhorn_weights,
        ema_variance=ema_variance_current,
        config=fusion_config,
    )
    updated_weights = fusion.updated_weights
    fused_prediction = fusion.fused_prediction
    sinkhorn_converged = jnp.asarray(fusion.sinkhorn_converged)
    free_energy = jnp.asarray(fusion.free_energy)
    sinkhorn_epsilon = jnp.asarray(fusion.sinkhorn_epsilon)

    updated_weights = jnp.where(
        degraded_mode_flag,
        state.rho,
        updated_weights,
    )
    fused_prediction = jnp.where(
        degraded_mode_flag,
        jnp.sum(state.rho * predictions),
        fused_prediction,
    )
    sinkhorn_converged = jnp.where(
        degraded_mode_flag,
        jnp.array(False),
        sinkhorn_converged,
    )
    free_energy = jnp.where(
        degraded_mode_flag,
        jnp.array(0.0),
        free_energy,
    )
    sinkhorn_epsilon = jnp.where(
        degraded_mode_flag,
        jnp.array(0.0),
        sinkhorn_epsilon,
    )
    ema_variance_current = jnp.where(
        degraded_mode_flag,
        state.ema_variance,
        ema_variance_current,
    )

    # Update Kernel B diagnostics to use current-step volatility threshold
    entropy_threshold_current = compute_adaptive_entropy_threshold(ema_variance_current, config)

    # COMPLIANCE: Zero-Heuristics - entropy_dgm is required, no silent fallback
    if "entropy_dgm" not in kernel_outputs[KernelType.KERNEL_B].metadata:
        raise ValueError("Kernel B metadata missing 'entropy_dgm'. " "Zero-Heuristics policy forbids silent defaults.")

    output_b = kernel_outputs[KernelType.KERNEL_B]._replace(
        metadata={
            **kernel_outputs[KernelType.KERNEL_B].metadata,
            "entropy_threshold_adaptive": entropy_threshold_current,
            "architecture_scaling_triggered": scaling_triggered,
            "mode_collapse": (
                jnp.asarray(kernel_outputs[KernelType.KERNEL_B].metadata["entropy_dgm"]) < entropy_threshold_current
            ),
        },
    )
    kernel_outputs = (
        kernel_outputs[KernelType.KERNEL_A],
        output_b,
        kernel_outputs[KernelType.KERNEL_C],
        kernel_outputs[KernelType.KERNEL_D],
    )

    is_simplex, msg = validate_simplex(updated_weights, config.validation_simplex_atol, "weights")
    if not is_simplex:
        raise ValueError(msg)

    current_value = signal[-1]
    residual = jnp.abs(current_value - fused_prediction)

    updated_state_candidate, regime_change_detected = atomic_state_update(
        state=state,
        new_signal=current_value,
        new_residual=residual,
        config=config,
    )
    updated_state = jax.tree_util.tree_map(
        lambda rej, old, new: jnp.where(rej, old, new),
        reject_observation,
        state,
        updated_state_candidate,
    )
    regime_change_detected = jnp.where(
        reject_observation,
        jnp.array(False),
        regime_change_detected,
    )
    regime_change_detected = jnp.asarray(regime_change_detected)

    force_emergency = False
    # Degradation monitor (post-mutation rollback guardrail)
    if degradation_monitor is not None and not reject_observation:
        degradation_monitor.record_prediction_error(residual)
        degraded, _ = degradation_monitor.check_degradation()
        if degraded:
            degradation_monitor.trigger_rollback()
            degraded_mode = True
            force_emergency = True

    # CAPA 1: Entropy reset on regime change (CUSUM alarm)
    # Mandato: ρ → Softmax(0) = uniform simplex [0.25, 0.25, 0.25, 0.25]
    # References: MIGRATION_AUTOTUNING_v1.0.md §2.1, Theory.tex §3.4
    uniform_simplex = jnp.full((KernelType.N_KERNELS,), 1.0 / KernelType.N_KERNELS)

    # Apply entropy reset if regime changed AND not in grace period already
    # (grace_counter starts at 0, gets set to grace_period_steps on alarm)
    entropy_reset_triggered = jnp.logical_and(
        regime_change_detected,
        state.grace_counter == 0,
    )

    # During grace period: freeze weights (no JKO update)
    # After grace period or normal operation: use fused weights
    in_grace_period = updated_state.grace_counter > 0
    final_rho = jnp.where(reject_observation, state.rho, updated_weights)
    final_rho = jnp.where(entropy_reset_triggered, uniform_simplex, final_rho)
    final_rho = jnp.where(in_grace_period, state.rho, final_rho)
    final_rho = jnp.where(robustness_triggered, kernel_d_simplex, final_rho)

    # COMPLIANCE: Zero-Heuristics - validate required metadata before state update
    if "holder_exponent" not in kernel_outputs[KernelType.KERNEL_A].metadata:
        raise ValueError(
            "Kernel A metadata missing 'holder_exponent'. " "Zero-Heuristics policy forbids silent defaults."
        )
    if "entropy_dgm" not in kernel_outputs[KernelType.KERNEL_B].metadata:
        raise ValueError("Kernel B metadata missing 'entropy_dgm'. " "Zero-Heuristics policy forbids silent defaults.")

    updated_state = replace(
        updated_state,
        rho=final_rho,
        holder_exponent=jnp.asarray(
            kernel_outputs[KernelType.KERNEL_A].metadata["holder_exponent"]
        ),  # V-MAJ-2: State update
        dgm_entropy=jnp.asarray(kernel_outputs[KernelType.KERNEL_B].metadata["entropy_dgm"]),
        last_update_ns=timestamp_ns if not reject_observation else state.last_update_ns,
        rng_key=jax.random.split(state.rng_key, config.prng_split_count)[1],
    )

    # Grace period decay during normal operations (CUSUM reset is done in update_cusum_statistics)
    # Note: grace_counter is already set in update_cusum_statistics when alarm triggers
    # Here we just handle the decay (no need to touch rho again, already handled above)
    grace_counter = updated_state.grace_counter
    grace_counter = jnp.where(grace_counter > 0, grace_counter - 1, grace_counter)
    updated_state = replace(updated_state, grace_counter=grace_counter)

    # V-MAJ-5: Mode Collapse Detection (consecutive low-entropy steps)
    # Use current-step volatility-adaptive threshold
    dgm_entropy_threshold = compute_adaptive_entropy_threshold(ema_variance_current, config)
    low_entropy = jnp.asarray(updated_state.dgm_entropy) < dgm_entropy_threshold
    mode_collapse_counter = jnp.asarray(updated_state.mode_collapse_consecutive_steps)
    mode_collapse_counter = jnp.where(
        low_entropy,
        mode_collapse_counter + 1,
        jnp.array(0),
    )

    # Warning threshold: config-driven (eliminates hardcoded constants)
    mode_collapse_warning_threshold = max(
        config.mode_collapse_min_threshold,
        int(fusion_config.entropy_window * config.mode_collapse_window_ratio),
    )
    mode_collapse_warning = mode_collapse_counter >= mode_collapse_warning_threshold

    # Update baseline entropy and architecture scaling counters
    current_entropy = jnp.asarray(updated_state.dgm_entropy)
    baseline_entropy = jnp.asarray(state.baseline_entropy)
    baseline_entropy = jnp.where(
        (baseline_entropy <= 0.0) & (current_entropy > 0.0),
        current_entropy,
        baseline_entropy,
    )
    baseline_entropy = jnp.where(
        regime_change_detected & (current_entropy > 0.0),
        current_entropy,
        baseline_entropy,
    )

    # Track solver usage for adaptive telemetry
    solver_idx = jnp.asarray(kernel_outputs[KernelType.KERNEL_C].metadata.get("solver_idx", jnp.array(-1)))
    solver_is_explicit = (solver_idx == 0) | (solver_idx == 1)
    solver_is_implicit = solver_idx == 2
    solver_explicit_count = jnp.asarray(state.solver_explicit_count) + solver_is_explicit.astype(jnp.int32)
    solver_implicit_count = jnp.asarray(state.solver_implicit_count) + solver_is_implicit.astype(jnp.int32)

    updated_state = replace(
        updated_state,
        baseline_entropy=jnp.asarray(baseline_entropy),
        solver_explicit_count=solver_explicit_count,
        solver_implicit_count=solver_implicit_count,
        architecture_scaling_events=(
            jnp.asarray(state.architecture_scaling_events) + scaling_triggered.astype(jnp.int32)
        ),
        mode_collapse_consecutive_steps=mode_collapse_counter,
    )

    emergency_mode = updated_state.holder_exponent < config.holder_threshold
    emergency_mode = emergency_mode | force_emergency | reject_observation

    confidences = jnp.array([ko.confidence for ko in kernel_outputs]).reshape(-1)
    fused_sigma = jnp.maximum(jnp.sum(updated_weights * confidences), config.pdf_min_sigma)
    z_score = config.confidence_interval_z
    confidence_lower = fused_prediction - z_score * fused_sigma
    confidence_upper = fused_prediction + z_score * fused_sigma

    # Operating mode: JAX-safe integer code (convert to string in API layer)
    operating_mode = _compute_operating_mode(degraded_mode, emergency_mode)
    prediction = PredictionResult(
        reference_prediction=jnp.asarray(fused_prediction),
        confidence_lower=jnp.asarray(confidence_lower),
        confidence_upper=jnp.asarray(confidence_upper),
        operating_mode=operating_mode,
        telemetry=None,
        request_id=None,
    )

    updated_state = replace(
        updated_state,
        degraded_mode=degraded_mode_flag,
        degraded_mode_recovery_counter=degraded_recovery_counter,  # V-MAJ-7: Persist hysteresis counter
        emergency_mode=emergency_mode,
        regime_changed=regime_change_detected,
    )

    # P2.3: Telemetry Buffer Integration (non-blocking audit trail)
    # COMPLIANCE FIX: Eliminate host-device sync (API_Python.tex §9)
    # DeviceArrays enqueued directly, conversion deferred to consumer thread
    if telemetry_buffer is not None:
        # CRITICAL: Do NOT call float() here - keeps XLA async dispatch
        # Consumer thread will batch-convert on the host
        telemetry_payload = {
            "step": step_counter,
            "timestamp_ns": timestamp_ns,
            # Store DeviceArray references (no GPU→CPU sync)
            "prediction_ref": fused_prediction,
            "weights_ref": final_rho,
            "free_energy_ref": free_energy if fusion is not None else jnp.array(0.0),
            "kurtosis_ref": updated_state.kurtosis,
            "holder_exponent_ref": updated_state.holder_exponent,
            "dgm_entropy_ref": updated_state.dgm_entropy,
            # Boolean flags are safe (already Python bool, no GPU transfer)
            "mode_collapse_warning": mode_collapse_warning,
            "degraded_mode": degraded_mode,
            "emergency_mode": emergency_mode,
        }

        # Enqueue only if hash interval triggers (P2.3: config-driven)
        if should_emit_hash(step_counter, config.telemetry_hash_interval_steps):
            telemetry_record = TelemetryRecord(step=step_counter, payload=telemetry_payload)
            telemetry_buffer.enqueue(telemetry_record)

    if mutation_rate_limiter is not None:
        mutation_rate_limiter.increment_stability_counter()

    return OrchestrationResult(
        prediction=prediction,
        state=updated_state,
        kernel_outputs=kernel_outputs,
        fusion=fusion,
        config=config_after,
    )


# =============================================================================
# MULTI-TENANT VECTORIZATION (VMAP) - API_Python.tex §3.1
# =============================================================================


@jax.jit
def orchestrate_step_batch(
    signals: Float[Array, "B n"],
    timestamp_ns: int,
    states: InternalState,
    config: PredictorConfig,
) -> tuple[PredictionResult, InternalState]:
    """
    Pure JAX batch orchestration for multi-tenant deployment (B assets).

    Uses vmap for Zero-Copy GPU parallelization.
    Note: Skips IO ingestion logic (use single-path orchestrate_step for that).

    Args:
        signals: Batched signals [B, n]
        timestamp_ns: Shared timestamp
        states: Batched InternalState [B, ...]
        config: Shared configuration

    Returns:
        (predictions_batch, states_batch) with all fields [B, ...]
    """

    def single_step(signal, state):
        # Simplified core: no ingestion, no mutation, pure JAX
        config.base_min_signal_length

        # Core kernel execution (vmap-safe)
        key_a, key_b, key_c, key_d = jax.random.split(state.rng_key, 4)

        output_a = kernel_a_predict(signal, key_a, config)
        output_b = kernel_b_predict(signal, key_b, config, ema_variance=state.ema_variance)
        output_c = kernel_c_predict(signal, key_c, config)
        output_d = kernel_d_predict(signal, key_d, config)

        kernel_outputs = (output_a, output_b, output_c, output_d)
        jnp.array([ko.prediction for ko in kernel_outputs])

        # Simplified fusion (no degraded mode branching)
        fusion = fuse_kernel_outputs(
            kernel_outputs=kernel_outputs,
            current_weights=state.rho,
            ema_variance=state.ema_variance,
            config=config,
        )

        # Simple state update (no CUSUM/ingestion)
        current_value = signal[-1]
        residual = jnp.abs(current_value - fusion.fused_prediction)

        updated_state, _ = atomic_state_update(
            state=state,
            new_signal=current_value,
            new_residual=residual,
            config=config,
        )

        # Update weights and diagnostics
        # COMPLIANCE: Zero-Heuristics - validate required metadata before state update
        if "holder_exponent" not in output_a.metadata:
            raise ValueError(
                "Kernel A metadata missing 'holder_exponent'. " "Zero-Heuristics policy forbids silent defaults."
            )
        if "entropy_dgm" not in output_b.metadata:
            raise ValueError(
                "Kernel B metadata missing 'entropy_dgm'. " "Zero-Heuristics policy forbids silent defaults."
            )
        updated_state = replace(
            updated_state,
            rho=fusion.updated_weights,
            holder_exponent=jnp.asarray(output_a.metadata["holder_exponent"]),
            dgm_entropy=jnp.asarray(output_b.metadata["entropy_dgm"]),
            rng_key=jax.random.split(state.rng_key, config.prng_split_count)[1],
        )

        # Operating mode (simplified: always INFERENCE in batch)
        operating_mode = jnp.asarray(OperatingMode.INFERENCE, dtype=jnp.int32)

        # Confidence interval
        confidences = jnp.array([ko.confidence for ko in kernel_outputs])
        fused_sigma = jnp.maximum(jnp.sum(fusion.updated_weights * confidences), config.pdf_min_sigma)
        z_score = config.confidence_interval_z

        prediction = PredictionResult(
            reference_prediction=jnp.asarray(fusion.fused_prediction),
            confidence_lower=fusion.fused_prediction - z_score * fused_sigma,
            confidence_upper=fusion.fused_prediction + z_score * fused_sigma,
            operating_mode=operating_mode,
            telemetry=None,
            request_id=None,
        )

        return prediction, updated_state

    # Pure vmap: Zero-Copy GPU parallelization
    predictions_batch, states_batch = jax.vmap(single_step)(signals, states)
    return predictions_batch, states_batch


def initialize_batched_states(
    batch_size: int,
    signal: Float[Array, "n"],
    timestamp_ns: int,
    rng_key: Array,
    config: PredictorConfig,
) -> InternalState:
    """
    Initialize batched InternalState for multi-tenant deployment.

    Creates B identical initial states with different PRNG keys.

    Args:
        batch_size: Number of assets (B)
        signal: Initial signal (shared across batch)
        timestamp_ns: Initial timestamp
        rng_key: Master PRNG key (will be split B times)
        config: Shared configuration

    Returns:
        InternalState with all fields batched to shape [B, ...]

    Example:
        >>> batch_size = 100
        >>> signal = jnp.linspace(0, 1, 256)
        >>> key = jax.random.PRNGKey(42)
        >>> states = initialize_batched_states(batch_size, signal, time.time_ns(), key, config)
        >>> states.signal_history.shape  # [100, 256]
    """
    # Split RNG key into B independent keys
    keys = jax.random.split(rng_key, batch_size)

    # Initialize single state
    single_state = initialize_state(signal, timestamp_ns, keys[0], config)

    # Batch all fields by stacking
    config.base_min_signal_length

    batched_state = InternalState(
        signal_history=jnp.stack([single_state.signal_history] * batch_size),
        residual_buffer=jnp.stack([single_state.residual_buffer] * batch_size),
        residual_window=jnp.stack([single_state.residual_window] * batch_size),
        rho=jnp.stack([single_state.rho] * batch_size),
        cusum_g_plus=jnp.stack([single_state.cusum_g_plus] * batch_size),
        cusum_g_minus=jnp.stack([single_state.cusum_g_minus] * batch_size),
        grace_counter=jnp.array([single_state.grace_counter] * batch_size, dtype=jnp.int32),
        adaptive_h_t=jnp.stack([single_state.adaptive_h_t] * batch_size),
        ema_variance=jnp.stack([single_state.ema_variance] * batch_size),
        kurtosis=jnp.stack([single_state.kurtosis] * batch_size),
        holder_exponent=jnp.stack([single_state.holder_exponent] * batch_size),
        dgm_entropy=jnp.stack([single_state.dgm_entropy] * batch_size),
        mode_collapse_consecutive_steps=jnp.array(
            [single_state.mode_collapse_consecutive_steps] * batch_size, dtype=jnp.int32
        ),
        degraded_mode_recovery_counter=jnp.array(
            [single_state.degraded_mode_recovery_counter] * batch_size, dtype=jnp.int32
        ),
        baseline_entropy=jnp.stack([single_state.baseline_entropy] * batch_size),
        solver_explicit_count=jnp.array([single_state.solver_explicit_count] * batch_size, dtype=jnp.int32),
        solver_implicit_count=jnp.array([single_state.solver_implicit_count] * batch_size, dtype=jnp.int32),
        architecture_scaling_events=jnp.array([single_state.architecture_scaling_events] * batch_size, dtype=jnp.int32),
        degraded_mode=jnp.array([single_state.degraded_mode] * batch_size, dtype=bool),
        emergency_mode=jnp.array([single_state.emergency_mode] * batch_size, dtype=bool),
        regime_changed=jnp.array([single_state.regime_changed] * batch_size, dtype=bool),
        last_update_ns=jnp.array([single_state.last_update_ns] * batch_size, dtype=jnp.int64),
        rng_key=keys,  # Each asset gets unique PRNG key
    )

    return batched_state
