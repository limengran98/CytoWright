"""Configure the model service used to generate candidate programs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from .local_config import LocalConfigError, configure_local_endpoint
from .schemas import digest


PROVIDER_RESOLVER_SCHEMA_VERSION = "cytowright_provider_resolver"
CREDENTIAL_POLICY = "machine_local_credentials_only"
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_FALLBACK_ENTRY_KEYS = frozenset({
    "fallback", "fallback_model", "fallback_models", "fallback_panel_key",
    "fallback_model_panel_key", "fallback_provider",
})


class ProviderResolverError(RuntimeError):
    """A configuration error that contains no credential or endpoint value."""


def _plain_json(value: Any, *, name: str) -> Any:
    def detach(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): detach(member) for key, member in item.items()}
        if isinstance(item, (tuple, list)):
            return [detach(member) for member in item]
        return item

    try:
        return json.loads(json.dumps(detach(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError):
        raise ProviderResolverError(f"{name} must be finite JSON-compatible metadata") from None


def _env_name(value: Any, *, name: str, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not _ENV_KEY_RE.fullmatch(value):
        raise ProviderResolverError(f"{name} is not a valid environment-variable name")
    return value


def _model_panel(path: str | Path = "configs/model_panel.json") -> Mapping[str, Any]:
    from .paths import project_path

    try:
        value = json.loads(project_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ProviderResolverError("cannot read model-panel configuration") from None
    if not isinstance(value, Mapping):
        raise ProviderResolverError("model panel must be an object")
    return value


def validate_provider_config(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"model_panel", "panel_key", "allow_fallback"}:
        raise ProviderResolverError("provider configuration requires model_panel, panel_key, and allow_fallback")
    if value["allow_fallback"] is not False:
        raise ProviderResolverError("a discovery campaign must disable provider fallback")
    if any(not isinstance(value[key], str) or not value[key].strip() for key in ("model_panel", "panel_key")):
        raise ProviderResolverError("provider model_panel and panel_key must be non-empty strings")


@dataclass(frozen=True)
class ProviderProfile:
    panel_key: str
    provider_family: str
    model_id: str
    local_endpoint_id: str
    supports_reasoning_effort: bool = False

    def __post_init__(self) -> None:
        for name in ("panel_key", "provider_family", "local_endpoint_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _IDENTITY_RE.fullmatch(value):
                raise ProviderResolverError(f"provider {name} must be a short public identifier")
        if not isinstance(self.model_id, str) or not self.model_id.strip() or any(ord(char) < 32 for char in self.model_id):
            raise ProviderResolverError("provider model must be a non-empty model identifier")
        if not isinstance(self.supports_reasoning_effort, bool):
            raise ProviderResolverError("supports_reasoning_effort must be boolean")


def profile_for_panel_key(value: str, model_panel: Mapping[str, Any] | None = None) -> ProviderProfile:
    panel = _model_panel() if model_panel is None else model_panel
    entry = panel.get(value) if isinstance(value, str) else None
    if not isinstance(entry, Mapping):
        raise ProviderResolverError("selected provider model-panel entry is unavailable")
    return ProviderProfile(
        panel_key=value,
        provider_family=entry.get("provider_family", "openai_compatible"),
        model_id=entry.get("model", ""),
        local_endpoint_id=entry.get("local_endpoint_id", value),
        supports_reasoning_effort=entry.get("supports_reasoning_effort", entry.get("reasoning_effort") is not None),
    )


def profile_for_provider_identity(*, provider_family: str, model_id: str) -> ProviderProfile:
    panel = _model_panel()
    for key in panel:
        profile = profile_for_panel_key(key, panel)
        if profile.provider_family == provider_family and profile.model_id == model_id:
            return profile
    raise ProviderResolverError("provider family/model identity is absent from the model panel")


def _provider_lock(profile: ProviderProfile) -> dict[str, Any]:
    return {
        "model_panel_key": profile.panel_key,
        "provider_family": profile.provider_family,
        "model_id": profile.model_id,
        "local_endpoint_id": profile.local_endpoint_id,
        "supports_reasoning_effort": profile.supports_reasoning_effort,
        "credential_policy": CREDENTIAL_POLICY,
        "allow_fallback": False,
        "all_llm_conditions_use_lock": True,
    }


def provider_lock_for_panel_key(panel_key: str, model_panel: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return _provider_lock(profile_for_panel_key(panel_key, model_panel))


def validate_provider_lock(value: Mapping[str, Any]) -> ProviderProfile:
    if not isinstance(value, Mapping):
        raise ProviderResolverError("provider lock must be an object")
    profile = ProviderProfile(
        panel_key=value.get("model_panel_key"),
        provider_family=value.get("provider_family"),
        model_id=value.get("model_id"),
        local_endpoint_id=value.get("local_endpoint_id"),
        supports_reasoning_effort=value.get("supports_reasoning_effort"),
    )
    if dict(value) != _provider_lock(profile) or value.get("allow_fallback") is not False or value.get("all_llm_conditions_use_lock") is not True:
        raise ProviderResolverError("provider lock must retain one backend without fallback")
    return profile


@dataclass(frozen=True)
class ProviderClientParameters:
    """Resolved private connection values and a public, immutable model identity."""

    _base_url: str = field(repr=False)
    _api_key: str = field(repr=False)
    profile: ProviderProfile
    reasoning_effort: str | None
    stream: bool
    base_url_env: str
    api_key_env: str
    model_env: str | None
    model_panel_entry_hash: str
    provider_lock_hash: str

    def __post_init__(self) -> None:
        try:
            parsed = urlparse(self._base_url)
            valid_endpoint = (
                parsed.scheme in {"http", "https"} and parsed.hostname
                and not parsed.username and not parsed.password
                and not parsed.query and not parsed.fragment
            )
        except ValueError:
            valid_endpoint = False
        if not valid_endpoint:
            raise ProviderResolverError("provider base URL must use HTTP(S) without embedded credentials, query, or fragment")
        if not isinstance(self._api_key, str) or not self._api_key.strip():
            raise ProviderResolverError("resolved provider credential is unavailable; set the configured API-key environment variable")
        if self.reasoning_effort is not None and (
            not isinstance(self.reasoning_effort, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", self.reasoning_effort)
        ):
            raise ProviderResolverError("reasoning effort must be a short provider-supported identifier")
        if not self.profile.supports_reasoning_effort and self.reasoning_effort is not None:
            raise ProviderResolverError("selected provider does not declare reasoning-effort support")
        if not isinstance(self.stream, bool):
            raise ProviderResolverError("provider stream setting must be boolean")
        _env_name(self.base_url_env, name="base_url_env")
        _env_name(self.api_key_env, name="api_key_env")
        _env_name(self.model_env, name="model_env", required=False)
        for name in ("model_panel_entry_hash", "provider_lock_hash"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ProviderResolverError(f"{name} must be a SHA-256 digest")
        if self.provider_lock_hash != digest(self.provider_lock):
            raise ProviderResolverError("resolved provider identity differs from its campaign lock")

    @property
    def model(self) -> str:
        return self.profile.model_id

    @property
    def provider_id(self) -> str:
        return self.profile.local_endpoint_id

    @property
    def provider_family(self) -> str:
        return self.profile.provider_family

    @property
    def provider_lock(self) -> dict[str, Any]:
        return _provider_lock(self.profile)

    @property
    def resolved_base_url_hash(self) -> str:
        return digest(self._base_url.rstrip("/"))

    @property
    def candidate_reasoning_effort(self) -> str | None:
        return self.reasoning_effort

    def client_kwargs(self) -> dict[str, Any]:
        return {
            "base_url": self._base_url.rstrip("/"),
            "api_key": self._api_key,
            "model": self.model,
            "provider_id": self.provider_id,
            "reasoning_effort": self.reasoning_effort,
            "stream": self.stream,
        }

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": PROVIDER_RESOLVER_SCHEMA_VERSION,
            "model_panel_key": self.profile.panel_key,
            "model": self.model,
            "provider_family": self.provider_family,
            "provider_id": self.provider_id,
            "reasoning_effort": self.reasoning_effort,
            "candidate_reasoning_effort": self.candidate_reasoning_effort,
            "stream": self.stream,
            "allow_fallback": False,
            "base_url_env": self.base_url_env,
            "api_key_env": self.api_key_env,
            "model_env": self.model_env,
            "credential_present": True,
            "resolved_base_url_hash": self.resolved_base_url_hash,
            "model_panel_entry_hash": self.model_panel_entry_hash,
            "provider_lock_hash": self.provider_lock_hash,
            "provider_call_performed": False,
        }


def resolve_provider_client_parameters(
    *, model_panel: Mapping[str, Any], local_endpoints: str | Path | None,
    provider_lock: Mapping[str, Any], base_url: str | None = None,
) -> ProviderClientParameters:
    profile = validate_provider_lock(provider_lock)
    if not isinstance(model_panel, Mapping):
        raise ProviderResolverError("model panel must be an object")
    entry = _plain_json(model_panel.get(profile.panel_key), name="selected provider model-panel entry")
    if not isinstance(entry, Mapping) or entry.get("kind") not in {"openai_compatible", "local_or_openai_compatible"}:
        raise ProviderResolverError("selected provider entry is not OpenAI-compatible")
    if profile_for_panel_key(profile.panel_key, model_panel) != profile:
        raise ProviderResolverError("selected model-panel entry differs from the campaign lock")
    if any(key in entry and entry[key] not in (None, False, "", [], {}) for key in _FALLBACK_ENTRY_KEYS):
        raise ProviderResolverError("selected provider entry must not declare a fallback")
    try:
        configure_local_endpoint(entry, path=local_endpoints)
    except (LocalConfigError, OSError, ValueError):
        raise ProviderResolverError("local selected-provider configuration could not be resolved") from None
    base_url_env = _env_name(entry.get("base_url_env"), name="base_url_env")
    api_key_env = _env_name(entry.get("api_key_env"), name="api_key_env")
    model_env = _env_name(entry.get("model_env"), name="model_env", required=False)
    resolved_model = os.environ.get(model_env, entry.get("model", "")) if model_env else entry.get("model", "")
    if resolved_model != profile.model_id:
        raise ProviderResolverError("resolved model differs from the frozen campaign identity")
    return ProviderClientParameters(
        _base_url=base_url if base_url is not None else os.environ.get(base_url_env, ""),
        _api_key=os.environ.get(api_key_env, ""),
        profile=profile,
        reasoning_effort=entry.get("reasoning_effort"),
        stream=entry.get("stream", False),
        base_url_env=base_url_env,
        api_key_env=api_key_env,
        model_env=model_env,
        model_panel_entry_hash=digest(entry),
        provider_lock_hash=digest(_plain_json(provider_lock, name="provider_lock")),
    )


def resolve_campaign_provider(
    provider: Mapping[str, Any], *, model: str | None = None,
    base_url: str | None = None, api_key_env: str | None = None,
    stream: bool | None = None, reasoning_effort: str | None = None,
    model_panel_path: str | Path | None = None, panel_key: str | None = None,
    local_endpoints: str | Path | None = None,
) -> ProviderClientParameters:
    """Resolve CLI/environment overrides once, before executing any trajectory."""
    validate_provider_config(provider)
    panel = dict(_model_panel(model_panel_path or provider["model_panel"]))
    selected_model = model if model is not None else (
        None if panel_key is not None else os.environ.get("CYTOWRIGHT_LLM_MODEL")
    )
    selected_base = base_url if base_url is not None else os.environ.get("CYTOWRIGHT_LLM_BASE_URL")
    selected_key_env = api_key_env if api_key_env is not None else os.environ.get("CYTOWRIGHT_LLM_API_KEY_ENV")
    selected_key = panel_key or provider["panel_key"]
    if selected_model is not None:
        selected_key = "custom"
        entry: dict[str, Any] = {
            "kind": "openai_compatible", "provider_family": "openai_compatible",
            "model": selected_model, "local_endpoint_id": "custom",
            "base_url_env": "CYTOWRIGHT_LLM_BASE_URL", "api_key_env": "CYTOWRIGHT_LLM_API_KEY",
            "stream": False,
        }
    else:
        raw_entry = panel.get(selected_key)
        if not isinstance(raw_entry, Mapping):
            raise ProviderResolverError("selected provider model-panel entry is unavailable")
        entry = dict(raw_entry)
    if selected_base is not None:
        entry["base_url_env"] = "CYTOWRIGHT_LLM_BASE_URL"
    if selected_key_env is not None:
        entry["api_key_env"] = _env_name(selected_key_env, name="api_key_env")
    elif os.environ.get("CYTOWRIGHT_LLM_API_KEY"):
        entry["api_key_env"] = "CYTOWRIGHT_LLM_API_KEY"
    if stream is not None:
        entry["stream"] = stream
    if reasoning_effort is not None:
        entry["reasoning_effort"] = reasoning_effort
        entry["supports_reasoning_effort"] = True
    try:
        configure_local_endpoint(entry, path=local_endpoints)
    except (LocalConfigError, OSError, ValueError):
        raise ProviderResolverError("local selected-provider configuration could not be resolved") from None
    model_env = _env_name(entry.get("model_env"), name="model_env", required=False)
    if model_env and model_env in os.environ:
        entry["model"] = os.environ[model_env]
    panel[selected_key] = entry
    lock = provider_lock_for_panel_key(selected_key, panel)
    return resolve_provider_client_parameters(
        model_panel=panel, local_endpoints=local_endpoints, provider_lock=lock,
        base_url=selected_base,
    )
