"""Atomic snapshot persistence with binary serialization and integrity hashes."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict
import hashlib
import os
import uuid
import numpy as np

import jax

from stochastic_predictor.api.types import InternalState, PredictorConfig


def _require_msgpack() -> Any:
    try:
        import msgpack  # type: ignore
    except ImportError as exc:
        raise RuntimeError("msgpack is required for snapshot serialization") from exc
    return msgpack


def _require_crc32c() -> Any:
    try:
        import google_crc32c  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-crc32c is required for crc32c hashing") from exc
    return google_crc32c


def _encode_array(value: Any) -> Dict[str, Any]:
    array = np.asarray(jax.device_get(value), dtype=np.float64)
    return {
        "dtype": str(array.dtype),
        "shape": array.shape,
        "data": array.tobytes(order="C"),
    }


def _decode_array(payload: Dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(payload["dtype"])
    shape = tuple(payload["shape"])
    data = payload["data"]
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _hash_bytes(data: bytes, algorithm: str) -> bytes:
    if algorithm == "sha256":
        return hashlib.sha256(data).digest()
    if algorithm == "crc32c":
        crc32c = _require_crc32c()
        return crc32c.value(data).to_bytes(4, byteorder="big", signed=False)
    raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def _hash_size(algorithm: str) -> int:
    if algorithm == "sha256":
        return 32
    if algorithm == "crc32c":
        return 4
    raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def serialize_state(state: InternalState) -> Dict[str, Any]:
    return {
        "signal_history": _encode_array(state.signal_history),
        "residual_buffer": _encode_array(state.residual_buffer),
        "rho": _encode_array(state.rho),
        "cusum_g_plus": _encode_array(state.cusum_g_plus),
        "cusum_g_minus": _encode_array(state.cusum_g_minus),
        "grace_counter": int(state.grace_counter),
        "ema_variance": _encode_array(state.ema_variance),
        "kurtosis": _encode_array(state.kurtosis),
        "holder_exponent": _encode_array(state.holder_exponent),
        "dgm_entropy": _encode_array(state.dgm_entropy),
        "degraded_mode": bool(state.degraded_mode),
        "emergency_mode": bool(state.emergency_mode),
        "regime_changed": bool(state.regime_changed),
        "last_update_ns": int(state.last_update_ns),
        "rng_key": _encode_array(state.rng_key),
    }


def deserialize_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "signal_history": _decode_array(payload["signal_history"]),
        "residual_buffer": _decode_array(payload["residual_buffer"]),
        "rho": _decode_array(payload["rho"]),
        "cusum_g_plus": _decode_array(payload["cusum_g_plus"]),
        "cusum_g_minus": _decode_array(payload["cusum_g_minus"]),
        "grace_counter": int(payload["grace_counter"]),
        "ema_variance": _decode_array(payload["ema_variance"]),
        "kurtosis": _decode_array(payload["kurtosis"]),
        "holder_exponent": _decode_array(payload["holder_exponent"]),
        "dgm_entropy": _decode_array(payload["dgm_entropy"]),
        "degraded_mode": bool(payload["degraded_mode"]),
        "emergency_mode": bool(payload["emergency_mode"]),
        "regime_changed": bool(payload["regime_changed"]),
        "last_update_ns": int(payload["last_update_ns"]),
        "rng_key": _decode_array(payload["rng_key"]),
    }


def serialize_snapshot(state: InternalState, config: PredictorConfig) -> bytes:
    if config.snapshot_format != "msgpack":
        raise NotImplementedError("Only msgpack snapshot format is supported")
    msgpack = _require_msgpack()
    payload = {
        "schema_version": config.schema_version,
        "state": serialize_state(state),
    }
    packed = msgpack.packb(payload, use_bin_type=True)
    digest = _hash_bytes(packed, config.snapshot_hash_algorithm)
    return packed + digest


def load_snapshot_bytes(data: bytes, config: PredictorConfig) -> Dict[str, Any]:
    hash_len = _hash_size(config.snapshot_hash_algorithm)
    if len(data) <= hash_len:
        raise ValueError("Snapshot payload is too small")
    payload = data[:-hash_len]
    expected_hash = data[-hash_len:]
    actual_hash = _hash_bytes(payload, config.snapshot_hash_algorithm)
    if actual_hash != expected_hash:
        raise ValueError("Snapshot hash verification failed")
    msgpack = _require_msgpack()
    unpacked = msgpack.unpackb(payload, raw=False)
    return deserialize_state(unpacked["state"])


def write_then_rename(path: str, data: bytes, fsync: bool) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + f".{uuid.uuid4().hex}.tmp")
    with open(temp_path, "wb") as handle:
        handle.write(data)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    os.replace(temp_path, target)


def save_snapshot(path: str, state: InternalState, config: PredictorConfig) -> None:
    payload = serialize_snapshot(state, config)
    if config.snapshot_compression == "gzip":
        import gzip
        payload = gzip.compress(payload)
    elif config.snapshot_compression == "brotli":
        try:
            import brotli  # type: ignore
        except ImportError as exc:
            raise RuntimeError("brotli compression requires brotli package") from exc
        payload = brotli.compress(payload)
    write_then_rename(path, payload, config.snapshot_atomic_fsync)


def load_snapshot(path: str, config: PredictorConfig) -> Dict[str, Any]:
    data = Path(path).read_bytes()
    if config.snapshot_compression == "gzip":
        import gzip
        data = gzip.decompress(data)
    elif config.snapshot_compression == "brotli":
        try:
            import brotli  # type: ignore
        except ImportError as exc:
            raise RuntimeError("brotli compression requires brotli package") from exc
        data = brotli.decompress(data)
    return load_snapshot_bytes(data, config)
