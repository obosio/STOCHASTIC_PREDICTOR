"""Credential injection helpers for IO integrations."""

import os
from pathlib import Path
from typing import Dict, Optional


class MissingCredentialError(RuntimeError):
    pass


def load_env_file(path: str) -> Dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: Dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"")
    return values


def get_required_env(name: str, env: Optional[Dict[str, str]] = None) -> str:
    if env is not None and name in env:
        return env[name]
    value = os.getenv(name)
    if value is None or value == "":
        raise MissingCredentialError(f"Missing required credential: {name}")
    return value
