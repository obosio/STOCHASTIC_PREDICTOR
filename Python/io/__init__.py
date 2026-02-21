"""IO Layer - Physical I/O and Data Management

Manages data ingestion, storage, and serialization for the predictor.

Responsibilities:
  - Time series data loading (OHLCV, real-time feeds)
  - Atomic snapshot persistence
  - Channel management for market feeds
  - Credential injection via environment variables
  - Data validation and preprocessing

See: Doc/latex/specification/Stochastic_Predictor_Python.tex §2 - Physical I/O Layer
See: Doc/latex/specification/Stochastic_Predictor_IO.tex - Complete I/O interface specification

CRITICAL SECURITY POLICIES (Doc/latex/specification/Stochastic_Predictor_IO.tex §2.2):

  ❌ PROHIBITED:
    - Hardcoding API keys, database passwords, tokens in code
    - Storing credentials in version control
    - Logging sensitive data

  ✅ REQUIRED:
    - Environment variable injection (.env files, never committed)
    - Configuration via os.getenv() or similar
    - Explicit .gitignore rules:
      .env
      .env.local
      secrets/
      credentials/
      *.log

Expected module structure:
  - loaders.py: Data loading from various sources
  - snapshots.py: Atomic snapshot persistence
  - credentials.py: Secure credential management
  - validators.py: Data validation
"""

from Python.io.config_mutation import (
    ConfigMutationError,
    append_audit_log,
    atomic_write_config,
    create_config_backup,
    validate_config_mutation,
)
from Python.io.credentials import (
    MissingCredentialError,
    get_required_env,
    load_env_file,
)
from Python.io.dashboard import (
    DashboardSeries,
    build_dashboard_html,
    export_dashboard_snapshot,
)
from Python.io.loaders import IngestionDecision, evaluate_ingestion
from Python.io.snapshots import (
    load_snapshot,
    load_snapshot_bytes,
    save_snapshot,
    serialize_snapshot,
    write_then_rename,
)
from Python.io.telemetry import (
    TelemetryBuffer,
    TelemetryRecord,
    parity_hashes,
    should_emit_hash,
)
from Python.io.validators import (
    FrozenSignalAlarmEvent,
    OutlierRejectedEvent,
    StaleSignalEvent,
    compute_staleness_ns,
    detect_catastrophic_outlier,
    detect_frozen_recovery,
    detect_frozen_signal,
    is_stale,
)

__all__ = [
    "OutlierRejectedEvent",
    "FrozenSignalAlarmEvent",
    "StaleSignalEvent",
    "detect_catastrophic_outlier",
    "detect_frozen_signal",
    "detect_frozen_recovery",
    "compute_staleness_ns",
    "is_stale",
    "IngestionDecision",
    "evaluate_ingestion",
    "serialize_snapshot",
    "load_snapshot_bytes",
    "save_snapshot",
    "load_snapshot",
    "write_then_rename",
    "TelemetryRecord",
    "TelemetryBuffer",
    "parity_hashes",
    "should_emit_hash",
    "DashboardSeries",
    "build_dashboard_html",
    "export_dashboard_snapshot",
    "MissingCredentialError",
    "load_env_file",
    "get_required_env",
    "ConfigMutationError",
    "atomic_write_config",
    "validate_config_mutation",
    "append_audit_log",
    "create_config_backup",
]
