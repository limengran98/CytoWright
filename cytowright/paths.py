"""Resolve workspace data and installed program resources."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent


def workspace_root() -> Path:
    configured = os.environ.get("CYTOWRIGHT_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()
    if (SOURCE_ROOT / "pyproject.toml").is_file():
        return SOURCE_ROOT
    return Path.cwd().resolve()


def project_path(value: str | Path) -> Path:
    """Resolve workspace data and outputs, with bundled config/source fallback.

    Editable checkouts use the repository root by default. Installed wheels use
    the current directory for data, cache and runs. CYTOWRIGHT_WORKSPACE selects
    a different workspace in either installation mode.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    local = workspace_root() / path
    if path.parts and path.parts[0] in {"configs", "assets", "programs", "examples"}:
        if local.is_file():
            return local.resolve()
        source = SOURCE_ROOT / path
        if (SOURCE_ROOT / "pyproject.toml").is_file() and source.is_file():
            return source.resolve()
        return (PACKAGE_ROOT / "resources" / path).resolve()
    if path.parts and path.parts[0] == "cytowright":
        return (PACKAGE_ROOT / Path(*path.parts[1:])).resolve()
    return local.resolve()


def receipt_path(value: str | Path) -> str:
    """Keep workspace receipts relative and installed resource receipts absolute."""
    path = project_path(value)
    try:
        return str(path.relative_to(workspace_root()))
    except ValueError:
        return str(path)
