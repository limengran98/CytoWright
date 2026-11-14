"""Load local model-service settings and environment configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlparse

from .paths import workspace_root


class LocalConfigError(RuntimeError):
    """Raised when a local endpoint file is malformed."""


_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LocalConfigError(f"Invalid JSON in local endpoint configuration: {path}") from exc
    if not isinstance(payload, Mapping):
        raise LocalConfigError("Local endpoint configuration must be a JSON object")
    return payload


def _source_env_file(path: Path) -> None:
    """Load simple local KEY=value entries without shell evaluation.

    The parser intentionally does not support command substitution, exports, or
    arbitrary shell syntax.  Existing process values win so cluster launchers
    can override this private convenience file.
    """
    if not path.exists():
        raise LocalConfigError(f"Configured credential file does not exist: {path}")
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise LocalConfigError(f"Malformed credential entry at {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY.fullmatch(key):
            raise LocalConfigError(f"Invalid credential variable name at {path}:{line_number}")
        os.environ.setdefault(key, value.strip())


def _local_path(path: str | Path | None = None) -> Path | None:
    configured = path or os.environ.get("CYTOWRIGHT_LOCAL_ENDPOINTS")
    candidate = Path(configured).expanduser() if configured else workspace_root() / "configs" / "local_endpoints.json"
    if configured and not candidate.is_file():
        raise LocalConfigError("Configured local endpoint file does not exist")
    return candidate if candidate.exists() else None


def configure_local_endpoint(entry: Mapping[str, Any], *, path: str | Path | None = None) -> None:
    """Fill the environment expected by a model-panel entry from local metadata.

    The local configuration is optional.  When it is absent, the normal
    environment-only behavior is retained.  It can name a credential file and
    map an endpoint's *non-secret* base URL plus a credential variable into the
    generic variables used by ``model_panel.json``.
    """
    local_file = _local_path(path)
    if local_file is None:
        return
    payload = _read_json(local_file)
    credential_files = payload.get("credential_env_files", [])
    if not isinstance(credential_files, list) or not all(isinstance(item, str) for item in credential_files):
        raise LocalConfigError("credential_env_files must be a list of paths")
    for raw_path in credential_files:
        credential_path = Path(raw_path).expanduser()
        if not credential_path.is_absolute():
            credential_path = local_file.parent / credential_path
        _source_env_file(credential_path)

    endpoint_id = str(entry.get("local_endpoint_id", ""))
    endpoints = payload.get("endpoints", {})
    if not endpoint_id or not isinstance(endpoints, Mapping) or endpoint_id not in endpoints:
        return
    endpoint = endpoints[endpoint_id]
    if not isinstance(endpoint, Mapping):
        raise LocalConfigError(f"Endpoint {endpoint_id!r} must be an object")

    base_var = str(entry.get("base_url_env", ""))
    if base_var and isinstance(endpoint.get("base_url"), str):
        os.environ.setdefault(base_var, str(endpoint["base_url"]))

    key_var = str(entry.get("api_key_env", ""))
    credential_var = endpoint.get("api_key_env")
    default_key = endpoint.get("default_api_key")
    if default_key is not None and (
        not isinstance(default_key, str) or not default_key.strip()
    ):
        raise LocalConfigError(
            f"Endpoint {endpoint_id!r} default_api_key must be a non-empty string"
        )
    if default_key is not None:
        endpoint_base_url = endpoint.get("base_url")
        hostname = (
            urlparse(endpoint_base_url).hostname
            if isinstance(endpoint_base_url, str)
            else None
        )
        if hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise LocalConfigError(
                f"Endpoint {endpoint_id!r} default_api_key is allowed only for localhost"
            )
    if key_var and isinstance(credential_var, str) and credential_var:
        value = os.environ.get(credential_var)
        if value:
            os.environ.setdefault(key_var, value)
    if key_var and not os.environ.get(key_var) and isinstance(default_key, str):
        # A localhost OpenAI-compatible server commonly accepts the conventional
        # non-secret ``EMPTY`` bearer value.  This fallback is opt-in per local
        # endpoint and never applies to remote providers.
        os.environ.setdefault(key_var, default_key)

    model_var = str(entry.get("model_env", ""))
    if model_var and isinstance(endpoint.get("model"), str):
        os.environ.setdefault(model_var, str(endpoint["model"]))


def local_endpoint_status(entry: Mapping[str, Any], *, path: str | Path | None = None) -> Mapping[str, Any]:
    """Return non-secret diagnostics for a configured provider."""
    local_file = _local_path(path)
    return {
        "local_config_present": local_file is not None,
        "local_config_path": str(local_file) if local_file else None,
        "endpoint_id": entry.get("local_endpoint_id"),
        "base_url_env": entry.get("base_url_env"),
        "api_key_env": entry.get("api_key_env"),
    }
