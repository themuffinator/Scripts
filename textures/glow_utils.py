"""Utilities for generating glow map variants."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PIL import Image

LogFn = Callable[[str], None]


def _ensure_png(path: Path) -> Path:
    """Return ``path`` with a ``.png`` suffix, preserving the base name."""

    if path.suffix.lower() == ".png":
        return path
    if path.suffix:
        return path.with_suffix(".png")
    return path.with_name(path.name + ".png")


def _ensure_glow_suffix(path: Path) -> Path:
    """Ensure the filename ends with ``_glow`` (case-insensitive)."""

    stem = path.stem
    if stem.lower().endswith("_glow"):
        return path
    return path.with_name(f"{stem}_glow{path.suffix}")


def generate_glow_png(
    source: Path,
    desired_output: Path,
    *,
    logger: Optional[LogFn] = None,
) -> Optional[Path]:
    """Emit a grayscale glow copy with luminance-driven alpha.

    The output file will always be suffixed with ``_glow`` and saved as ``.png``.
    ``desired_output`` may use any extension – it is normalised automatically.
    Returns the final path on success or ``None`` on failure.
    """

    def _log(message: str) -> None:
        if logger:
            logger(message)

    try:
        final_path = _ensure_png(desired_output)
        final_path = _ensure_glow_suffix(final_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(source) as img:
            luma = img.convert("L")
            rgba = Image.merge("RGBA", (luma, luma, luma, luma))
            rgba.save(final_path)

        _log(f"[GLOW] {source} -> {final_path}")
        return final_path
    except Exception as exc:  # pragma: no cover - defensive catch for Pillow I/O
        _log(f"[WARN] Glow conversion failed for {source}: {exc}")
        return None
