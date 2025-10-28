"""Backward compatible CLI for Quake 4 -> Quake 3 conversion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:  # pragma: no cover - runtime import fallback
    from .idtech4_to_idtech23_converter import main as unified_main
except ImportError:  # Direct execution (no package context)
    _spec = importlib.util.spec_from_file_location(
        "idtech4_to_idtech23_converter",
        Path(__file__).with_name("idtech4_to_idtech23_converter.py"),
    )
    if _spec is None or _spec.loader is None:  # pragma: no cover - safety net
        raise
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    unified_main = _module.main

DEFAULT_PROFILE = "quake4"


def main(argv: list[str] | None = None) -> int:
    """Invoke the unified converter, defaulting to the Quake 4 profile."""

    args = list(argv if argv is not None else sys.argv[1:])
    if "--profile" not in args:
        args = ["--profile", DEFAULT_PROFILE, *args]
    return unified_main(args)


if __name__ == "__main__":  # pragma: no cover - CLI shim
    sys.exit(main())
