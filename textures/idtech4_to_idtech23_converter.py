"""Unified idTech4 -> idTech23 texture/shader converter.

This module centralises the shared Doom 3 / Quake 4 conversion logic so that
new titles can plug in a lightweight *profile* describing their naming rules
and a couple of guide-macro tweaks.  The heavy lifting (asset indexing,
material parsing, Blender baking, shader emission) now lives here and the
legacy entry points simply forward into :func:`main` with the desired profile
identifier.

The design goals for the refactor are:

* Keep the mature Quake 4 pipeline intact while letting Doom 3 reuse it.
* Remove copy/paste differences by consolidating helpers (stage parsing,
  guide-macro execution, additive/glow baking) that were duplicated across the
  standalone scripts.
* Allow future games to opt into slightly different behaviours (directory
  prefixes, suffix normalisation, macro overrides, etc.) without editing the
  core logic.

Profiles are described in ``textures/convert_config.json`` – see
``TitleProfile`` below for the fields that can be tuned.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import traceback
import hashlib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration dataclasses and JSON loader
# ---------------------------------------------------------------------------


@dataclass
class BakeSettings:
    sun_az: float = 45.0
    sun_el: float = 55.0
    sun_power: float = 3.5
    sun_color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    world_ambient: float = 0.05
    normal_strength: float = 1.33
    samples: int = 11
    margin: int = 4
    clearcoat_roughness: float = 0.1


@dataclass
class VibranceSettings:
    spec_gain: float = 1.33
    roughness_bias: float = -0.0
    sat_gain: float = 1.0
    contrast: float = 0.0
    brightness: float = 0.0
    diffuse_gain: float = 1.0
    clearcoat_gain: float = 0.0


@dataclass
class Quake2Options:
    enabled: bool = False
    palette_path: Optional[Path] = None
    scale_percent: float = 50.0
    alpha_threshold: int = 128
    skip_auxiliary_maps: bool = True
    light_value: int = 1000

    def resolved_palette(self, base_root: Path) -> Optional[Path]:
        if not self.palette_path:
            return None
        if self.palette_path.is_absolute():
            return self.palette_path
        return base_root / self.palette_path


@dataclass
class PrefixRule:
    """Map a first-level source directory to a destination prefix.

    ``replacement`` may contain ``{value}`` (the original directory) and
    ``{suffix}`` (``value`` with the matched prefix stripped when the rule is a
    ``startswith`` type).  A ``default`` rule acts as the fallback when no
    previous rule matches.
    """

    type: str
    replacement: str
    match: Optional[str] = None

    def apply(self, value: str) -> Optional[str]:
        value_l = value.lower()
        if self.type == "equals":
            if self.match is None:
                return None
            return self.replacement.format(value=value, suffix=value) if value_l == self.match.lower() else None
        if self.type == "startswith":
            if self.match is None:
                return None
            match_l = self.match.lower()
            if value_l.startswith(match_l):
                suffix = value[len(self.match) :]
                return self.replacement.format(value=value, suffix=suffix)
            return None
        if self.type == "default":
            return self.replacement.format(value=value, suffix=value)
        return None


@dataclass
class TitleProfile:
    """Settings that differentiate individual idTech4 titles."""

    key: str
    display_name: str
    directory_rules: List[PrefixRule] = field(default_factory=list)
    filename_replacements: List[Tuple[str, str]] = field(default_factory=list)
    guide_macro_overrides: Dict[str, str] = field(default_factory=dict)
    features: Dict[str, object] = field(default_factory=dict)
    shader_overrides: Dict[str, str] = field(default_factory=dict)

    def directory_prefix(self, value: str) -> str:
        fallback_prefix = str(self.features.get("default_prefix", (self.key[:3] or "id4"))).strip("_")
        if not fallback_prefix:
            fallback_prefix = "id4"
        for rule in self.directory_rules:
            mapped = rule.apply(value)
            if mapped:
                return mapped
        return f"{fallback_prefix}_{value}"


@dataclass
class Config:
    base_root: Path
    dst_base: Path
    blender_exe: Path
    blender_bake_script: Path
    output_format: str = "png"
    shader_output_dir: Path = Path("scripts")
    force_rebuild: bool = False
    rebuild_if_newer: bool = True
    verbose: bool = False
    cache_index: bool = False
    parallel_jobs: int = 1
    dry_run: bool = False
    profile_key: str = ""
    bake: BakeSettings = field(default_factory=BakeSettings)
    vibrance: VibranceSettings = field(default_factory=VibranceSettings)
    quake2: Quake2Options = field(default_factory=Quake2Options)
    profiles: Dict[str, TitleProfile] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        def pick(d: dict, dc_type):
            if not isinstance(d, dict):
                return {}
            allowed = {f.name for f in fields(dc_type)}
            return {k: v for k, v in d.items() if k in allowed}

        if "output_ext" in raw:
            raw["output_format"] = raw["output_ext"].strip(".")

        bake = BakeSettings(**pick(raw.get("bake", {}), BakeSettings))
        vibr = VibranceSettings(**pick(raw.get("vibrance", {}), VibranceSettings))
        quake2 = Quake2Options(**pick(raw.get("quake2", {}), Quake2Options))

        top = pick(raw, Config)
        for key in ("base_root", "dst_base", "blender_exe", "blender_bake_script", "shader_output_dir"):
            if key in raw:
                top[key] = Path(raw[key])

        profile_key = raw.get("profile") or raw.get("profile_key") or ""

        profiles_raw = raw.get("profiles", {})
        profiles: Dict[str, TitleProfile] = {}
        for key, pdata in profiles_raw.items():
            rules = [PrefixRule(**rule) for rule in pdata.get("directory_rules", [])]
            replacements = [tuple(pair) for pair in pdata.get("filename_replacements", [])]
            overrides = {k.lower(): v for k, v in pdata.get("guide_macro_overrides", {}).items()}
            profile = TitleProfile(
                key=key,
                display_name=pdata.get("display_name", key),
                directory_rules=rules,
                filename_replacements=replacements,
                guide_macro_overrides=overrides,
                features=pdata.get("features", {}),
                shader_overrides=pdata.get("shader_overrides", {}),
            )
            profiles[key] = profile

        if quake2.palette_path is not None:
            quake2.palette_path = Path(quake2.palette_path)

        top["bake"] = bake
        top["vibrance"] = vibr
        top["quake2"] = quake2
        top["profiles"] = profiles
        top["profile_key"] = profile_key
        return Config(**top)


# ---------------------------------------------------------------------------
# Asset indexing utilities
# ---------------------------------------------------------------------------


IMAGE_EXTS = [".tga", ".dds", ".png", ".jpg", ".jpeg", ".tif", ".bmp"]


SURF_LIGHT = 0x00000001
SURF_WARP = 0x00000008
SURF_TRANS33 = 0x00000010

CONTENTS_SOLID = 0x00000001
CONTENTS_WINDOW = 0x00000002
CONTENTS_WATER = 0x00000020

DEFAULT_Q2_PALETTE_DATA = [0] * (256 * 3)
Q2_AUX_SUFFIXES = ("_add", "_glow")


class AssetIndex:
    def __init__(self, base_root: Path):
        self.base_root = base_root
        self._map: Dict[str, Path] = {}

    def build(self):
        root = self.base_root
        for ext in IMAGE_EXTS:
            for p in root.rglob(f"*{ext}"):
                rel = p.relative_to(root).as_posix().lower()
                self._map[rel] = p

    def resolve(self, token: Optional[str]) -> Optional[Path]:
        if not token:
            return None
        key = token.replace("\\", "/").lstrip("/").lower()
        if key in self._map:
            return self._map[key]
        key_noext, _ = os.path.splitext(key)
        for ext in IMAGE_EXTS:
            candidate = f"{key_noext}{ext}"
            if candidate in self._map:
                return self._map[candidate]
        return None


# ---------------------------------------------------------------------------
# Material representation & parser helpers
# ---------------------------------------------------------------------------


def _norm_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return token
    token = token.replace("\\", "/").lstrip("/")
    if not token.lower().startswith(("textures/", "models/")):
        token = "textures/" + token
    return token


@dataclass
class Material:
    name: str
    source_file: Path
    diffuse: Optional[str] = None
    normal: Optional[str] = None
    specular: Optional[str] = None
    height: Optional[str] = None
    height_scale: float = 1.0
    additive_maps: List[str] = field(default_factory=list)
    additive_maps_src: List[str] = field(default_factory=list)
    qer_editorimage: Optional[str] = None
    materialType: Optional[str] = None
    translucent: bool = False
    twoSided: bool = False
    noShadows: bool = False
    nonsolid: bool = False
    noimpact: bool = False
    is_guide: bool = False
    guide_macro: Optional[str] = None
    force_copy: bool = False
    original_text: str = ""


def _extract_stages(block: str) -> Iterable[str]:
    i, n = 0, len(block)
    while i < n:
        start = block.find("{", i)
        if start == -1:
            break
        depth, j = 1, start + 1
        while j < n and depth > 0:
            if block[j] == "{":
                depth += 1
            elif block[j] == "}":
                depth -= 1
            j += 1
        yield block[start + 1 : j - 1]
        i = j


def _detect_additive_maps(block: str) -> List[str]:
    out: List[str] = []
    for stage in _extract_stages(block):
        blend_m = re.search(r"\bblend\s+([^\s,;}]+)", stage, re.I)
        blend_mode = blend_m.group(1).lower() if blend_m else None
        if blend_mode != "add":
            continue
        mm = re.search(r"\bmap\s+([^\s;}]+)", stage, re.I)
        if not mm:
            continue
        out.append(_norm_token(mm.group(1)))
    return [o for o in out if o]


def _detect_height_stage(block: str) -> Tuple[Optional[str], float]:
    for stage in _extract_stages(block):
        blend_m = re.search(r"\bblend\s+([^\s,;}]+)", stage, re.I)
        blend_mode = blend_m.group(1).lower() if blend_m else None
        if blend_mode != "bumpmap":
            continue
        height_m = re.search(
            r"\bmap\s+heightmap\s*\(\s*([^\s,)]+)\s*,\s*([0-9.+\-]+)\s*\)",
            stage,
            re.I,
        )
        if height_m:
            return _norm_token(height_m.group(1)), float(height_m.group(2))
    return None, 1.0


def _detect_diffuse_stage(block: str) -> Optional[str]:
    for stage in _extract_stages(block):
        blend_m = re.search(r"\bblend\s+([^\s,;}]+)", stage, re.I)
        blend_mode = blend_m.group(1).lower() if blend_m else None
        if blend_mode not in ("diffusemap", "diffuse"):
            continue
        mm = re.search(r"\bmap\s+([^\s;}]+)", stage, re.I)
        if mm:
            return _norm_token(mm.group(1))
    return None


class GuideContext:
    def __init__(self, mat: Material, args: List[str], profile: TitleProfile):
        self.mat = mat
        self.args = args
        self.profile = profile

    def _token(self, idx: int) -> Optional[str]:
        if idx < len(self.args) and self.args[idx]:
            return _norm_token(self.args[idx])
        return None

    @property
    def base(self) -> Optional[str]:
        return self._token(0)

    @property
    def variant(self) -> Optional[str]:
        return self._token(1)

    def maps_from(self, prefix: Optional[str], include_height: bool = True):
        if not prefix:
            return
        p = prefix
        self.mat.diffuse = f"{p}_d"
        self.mat.normal = f"{p}_local"
        self.mat.specular = f"{p}_s"
        self.mat.height = f"{p}_h" if include_height else None
        self.mat.qer_editorimage = f"{p}_d"

    def variant_diffuse_only(self, diff: Optional[str], base: Optional[str]):
        if not diff:
            return
        self.mat.diffuse = f"{diff}_d"
        self.mat.qer_editorimage = f"{diff}_d"
        if base:
            self.mat.normal = f"{base}_local"
            self.mat.specular = f"{base}_s"
            self.mat.height = f"{base}_h"

    def alpha_flags(self, two_sided: bool = False, no_shadows: bool = False):
        self.mat.translucent = True
        self.mat.twoSided = two_sided
        self.mat.noShadows = no_shadows

    def set_type_from_arg(self, idx: int = 1):
        if idx < len(self.args) and self.args[idx]:
            self.mat.materialType = self.args[idx].strip()

    def set_height_scale(self, idx: int):
        if idx < len(self.args) and self.args[idx]:
            try:
                self.mat.height_scale = float(self.args[idx])
            except ValueError:
                pass

    def add_additive(self, token: Optional[str]):
        tok = _norm_token(token)
        if tok and tok not in self.mat.additive_maps:
            self.mat.additive_maps.append(tok)
            self.mat.additive_maps_src.append(tok)


GuideAction = Callable[[GuideContext], None]


def _action_maps_from_base(ctx: GuideContext):
    ctx.maps_from(ctx.base)


def _action_maps_from_base_noheight(ctx: GuideContext):
    ctx.maps_from(ctx.base, include_height=False)


def _action_maps_from_variant_or_base(ctx: GuideContext):
    ctx.maps_from(ctx.variant or ctx.base)


def _action_variant_diffuse(ctx: GuideContext):
    if ctx.variant and ctx.base:
        ctx.variant_diffuse_only(ctx.variant, ctx.base)
    elif ctx.variant:
        ctx.variant_diffuse_only(ctx.variant, ctx.variant)
    else:
        ctx.maps_from(ctx.base)


def _action_variant_diffuse_no_height(ctx: GuideContext):
    _action_variant_diffuse(ctx)
    ctx.mat.height = None


def _action_two_sided(ctx: GuideContext):
    ctx.maps_from(ctx.base)
    ctx.mat.twoSided = True


def _action_alpha(ctx: GuideContext):
    ctx.maps_from(ctx.base)
    ctx.alpha_flags()


def _action_alpha_noshadows(ctx: GuideContext):
    ctx.maps_from(ctx.base)
    ctx.alpha_flags(no_shadows=True)


def _action_alpha_two_sided(ctx: GuideContext):
    ctx.maps_from(ctx.base)
    ctx.alpha_flags(two_sided=True)


def _action_alpha_two_sided_variant(ctx: GuideContext):
    if ctx.variant and ctx.base:
        ctx.variant_diffuse_only(ctx.variant, ctx.base)
    else:
        ctx.maps_from(ctx.variant or ctx.base)
    ctx.alpha_flags(two_sided=True)


def _action_set_type(ctx: GuideContext):
    ctx.maps_from(ctx.base)
    ctx.set_type_from_arg(1)


def _action_set_type_two_sided(ctx: GuideContext):
    ctx.maps_from(ctx.base)
    ctx.alpha_flags(two_sided=True)
    ctx.set_type_from_arg(1)


def _action_color_variant(ctx: GuideContext):
    if ctx.base and ctx.variant:
        ctx.variant_diffuse_only(ctx.base, ctx.variant)
    else:
        ctx.maps_from(ctx.base or ctx.variant)


def _action_nonormal(ctx: GuideContext):
    ctx.maps_from(ctx.base)
    ctx.mat.normal = None


def _action_nonormal_height(ctx: GuideContext):
    ctx.maps_from(ctx.base)
    ctx.mat.normal = None
    ctx.set_height_scale(1)


def _action_shader_heightmap(ctx: GuideContext):
    base = ctx.base
    if base:
        ctx.mat.diffuse = base
        ctx.mat.qer_editorimage = base
        ctx.mat.height = base
        ctx.mat.normal = None
        ctx.mat.specular = None
    ctx.set_height_scale(1)


def _action_nonormal_height_type(ctx: GuideContext):
    ctx.maps_from(ctx.base)
    ctx.mat.normal = None
    ctx.set_height_scale(1)
    ctx.set_type_from_arg(2)


def _action_glow(ctx: GuideContext):
    ctx.maps_from(ctx.base)
    suffix = ctx.profile.features.get("guide_glow_suffix", "_g")
    token = ctx.variant or (f"{ctx.base}{suffix}" if ctx.base else None)
    ctx.add_additive(token)


def _action_terminal_glow(ctx: GuideContext):
    ctx.maps_from(ctx.base)
    suffix = ctx.profile.features.get("guide_terminal_suffix", "_add")
    token = ctx.variant or (f"{ctx.base}{suffix}" if ctx.base else None)
    ctx.add_additive(token)


def _action_alpha_glow(ctx: GuideContext):
    _action_alpha(ctx)
    _action_glow(ctx)


DEFAULT_GUIDE_ACTIONS: Dict[str, GuideAction] = {
    "maps_from_base": _action_maps_from_base,
    "maps_from_base_noheight": _action_maps_from_base_noheight,
    "maps_from_variant_or_base": _action_maps_from_variant_or_base,
    "variant_diffuse": _action_variant_diffuse,
    "variant_diffuse_no_height": _action_variant_diffuse_no_height,
    "two_sided": _action_two_sided,
    "alpha": _action_alpha,
    "alpha_noshadows": _action_alpha_noshadows,
    "alpha_two_sided": _action_alpha_two_sided,
    "alpha_two_sided_variant": _action_alpha_two_sided_variant,
    "set_type": _action_set_type,
    "set_type_two_sided": _action_set_type_two_sided,
    "color_variant": _action_color_variant,
    "nonormal": _action_nonormal,
    "nonormal_height": _action_nonormal_height,
    "shader_heightmap": _action_shader_heightmap,
    "nonormal_height_type": _action_nonormal_height_type,
    "glow": _action_glow,
    "terminal_glow": _action_terminal_glow,
    "alpha_glow": _action_alpha_glow,
}


DEFAULT_GUIDE_MACRO_MAP: Dict[str, str] = {
    "generic_materialimageshader": "maps_from_base",
    "generic_shader": "maps_from_base",
    "generic_shader_ed": "maps_from_base",
    "generic_shader_mi": "variant_diffuse",
    "generic_full_noheight_mi": "variant_diffuse",
    "generic_localvariant": "variant_diffuse",
    "generic_localvariant_mi": "variant_diffuse",
    "generic_shader2sided": "two_sided",
    "generic_full_noheight": "maps_from_base_noheight",
    "generic_full_noheight_type": "set_type",
    "generic_variant_noheight": "variant_diffuse_no_height",
    "generic_variant_noheight_mi": "variant_diffuse_no_height",
    "generic_nonormal": "nonormal",
    "generic_nonormal_height": "nonormal_height",
    "generic_shader_heightmap": "shader_heightmap",
    "generic_nonormal_height_type": "nonormal_height_type",
    "generic_alpha": "alpha",
    "generic_alpha_ed": "alpha",
    "generic_alpha_lv": "alpha",
    "generic_alpha_mi": "alpha",
    "generic_alphaglow": "alpha_glow",
    "generic_alpha_noshadows": "alpha_noshadows",
    "generic_alphanoshadow2s": "alpha_two_sided",
    "generic_shader2sidedalpha": "alpha_two_sided",
    "generic_localvalpha2": "alpha_two_sided",
    "generic_shader2sidedalpha_lv": "alpha_two_sided_variant",
    "generic_shader2sidedalpha_mi": "alpha_two_sided_variant",
    "generic_shader2sidedalpha_miv": "alpha_two_sided_variant",
    "generic_shader2sidedalpha_type": "set_type_two_sided",
    "generic_glow": "glow",
    "generic_glow_mi": "glow",
    "generic_terminal_replaceglow": "terminal_glow",
    "generic_terminal_replaceglow2": "terminal_glow",
    "generic_typeshader": "set_type",
    "generic_localvariant_typeshader": "set_type",
    "generic_alpha_type": "alpha",
    "generic_colorvariant": "color_variant",
}


class GuideMacroRegistry:
    def __init__(self, profile: TitleProfile):
        self.profile = profile

    def resolve_action(self, macro: str) -> Optional[GuideAction]:
        macro_l = macro.lower()
        action_key = self.profile.guide_macro_overrides.get(macro_l)
        if not action_key:
            action_key = DEFAULT_GUIDE_MACRO_MAP.get(macro_l)
        if not action_key:
            return None
        action = DEFAULT_GUIDE_ACTIONS.get(action_key)
        return action

    def apply(self, mat: Material, macro: str, args: List[str]):
        action = self.resolve_action(macro)
        if not action:
            return
        ctx = GuideContext(mat, args, self.profile)
        action(ctx)


class IdTech4MaterialParser:
    def __init__(self, profile: TitleProfile, verbose: bool = False):
        self.profile = profile
        self.verbose = verbose
        self.re_comment_line = re.compile(r"//.*?$", re.MULTILINE)
        self.re_comment_block = re.compile(r"/\*.*?\*/", re.DOTALL)
        self.re_guide = re.compile(
            r"^\s*guide\s+(?P<mat>\S+)\s+(?P<macro>[A-Za-z0-9_]+)\s*\((?P<args>[^)]*)\)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        self.re_block_header = re.compile(
            r"^\s*((?:textures|models)/[^\s{]+)\s*\{\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        self.registry = GuideMacroRegistry(profile)

    def strip_comments(self, text: str) -> str:
        text = self.re_comment_block.sub("", text)
        text = self.re_comment_line.sub("", text)
        return text

    def _iter_top_level_blocks(self, text: str) -> List[Tuple[str, str]]:
        out = []
        for m in self.re_block_header.finditer(text):
            name = m.group(1)
            i, depth, j = m.end(), 1, m.end()
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            out.append((name, text[i : j - 1]))
        return out

    def _parse_material_block(self, name: str, block: str, src: Path) -> Material:
        mat = Material(name=name, source_file=src)
        mat.original_text = f"{name}{{\n{block}\n}}"

        if re.search(r"\btwosided\b", block, re.I):
            mat.twoSided = True
        if re.search(r"\btranslucent\b", block, re.I):
            mat.translucent = True
        if re.search(r"\bnoshadows\b", block, re.I):
            mat.noShadows = True
        if re.search(r"\bnonsolid\b", block, re.I):
            mat.nonsolid = True
        if re.search(r"\bnoimpact\b", block, re.I):
            mat.noimpact = True

        if m := re.search(r"qer_?editorimage\s+([^\s}]+)", block, re.I):
            mat.qer_editorimage = _norm_token(m.group(1))

        if m := re.search(r"^\s*materialType\s+([^\s;}]+)", block, re.MULTILINE | re.I):
            mat.materialType = m.group(1).lower()

        if m := re.search(r"^\s*diffusemap\s+([^\s;}]+)", block, re.MULTILINE | re.I):
            mat.diffuse = _norm_token(m.group(1))

        if m := re.search(r"^\s*specularmap\s+([^\s;}]+)", block, re.MULTILINE | re.I):
            mat.specular = _norm_token(m.group(1))

        if m := re.search(r"^\s*bumpmap\s+([^\s;}]+)", block, re.MULTILINE | re.I):
            token = m.group(1)
            if "addnormals" not in token.lower():
                mat.normal = _norm_token(token)

        if not mat.diffuse:
            mat.diffuse = _detect_diffuse_stage(block)

        mat.additive_maps.extend(_detect_additive_maps(block))
        mat.additive_maps_src.extend(mat.additive_maps)

        if not mat.height:
            height_token, height_scale = _detect_height_stage(block)
            if height_token:
                mat.height = height_token
                mat.height_scale = height_scale

        addnormals = re.search(
            r"^\s*bumpmap\s+addnormals\s*\(\s*([^\s,()]+)\s*,\s*heightmap\s*\(\s*([^\s,()]+)\s*,\s*([0-9.+\-]+)\s*\)\s*\)",
            block,
            re.MULTILINE | re.I,
        )
        if addnormals and not mat.normal and not mat.height:
            mat.normal = _norm_token(addnormals.group(1))
            mat.height = _norm_token(addnormals.group(2))
            mat.height_scale = float(addnormals.group(3))

        if not mat.diffuse and mat.qer_editorimage:
            mat.diffuse = mat.qer_editorimage

        if not mat.diffuse:
            m = re.search(r"^\s*map\s+([^\s}]+)", block, re.MULTILINE | re.I)
            if m:
                mv = m.group(1)
                if "heightmap" not in mv.lower() and "addnormals" not in mv.lower():
                    mat.diffuse = _norm_token(mv)

        return mat

    def parse_file(self, path: Path) -> Dict[str, Material]:
        if not path.exists():
            return {}
        text = self.strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
        materials: Dict[str, Material] = {}

        for name, block in self._iter_top_level_blocks(text):
            if name.lower().startswith(("textures/", "models/")):
                materials[name] = self._parse_material_block(name, block, path)

        for m in self.re_guide.finditer(text):
            mat_name = m.group("mat")
            macro = m.group("macro").lower().strip()
            args = [a.strip().strip('"') for a in m.group("args").split(",") if a.strip()]
            if not mat_name.lower().startswith(("textures/", "models/")):
                continue
            mat = materials.setdefault(mat_name, Material(name=mat_name, source_file=path))
            mat.is_guide = True
            mat.guide_macro = macro
            if not mat.original_text:
                mat.original_text = m.group(0).strip()
            # Attempt to collect likely additive args before macro handlers adjust them
            for tok in args[2:4]:
                norm = _norm_token(tok)
                if norm and norm not in mat.additive_maps:
                    mat.additive_maps.append(norm)
                    mat.additive_maps_src.append(norm)
            self.registry.apply(mat, macro, args)

        return materials


# ---------------------------------------------------------------------------
# Naming utilities
# ---------------------------------------------------------------------------


class NameManager:
    def __init__(self, original_paths: Iterable[str], profile: TitleProfile):
        self.profile = profile
        self.dir_map = self._build_directory_map(original_paths)
        self.name_map: Dict[str, str] = {}
        self.used_names_by_category: Dict[str, set] = defaultdict(set)

    def _build_directory_map(self, original_paths: Iterable[str]) -> Dict[str, str]:
        first_level_dirs = set()
        for path in original_paths:
            low = path.lower()
            if low.startswith("textures/"):
                relative = path[9:]
            elif low.startswith("models/"):
                relative = path[7:]
            else:
                continue
            if "/" in relative:
                first_level_dirs.add(relative.split("/")[0])
        dir_map = {}
        for original_dir in sorted(first_level_dirs):
            dir_map[original_dir] = self.profile.directory_prefix(original_dir)
        return dir_map

    def _apply_replacements(self, filename: str) -> str:
        result = filename
        for src, dst in self.profile.filename_replacements:
            result = result.replace(src, dst)
        return result

    def get_new_name(self, original_full_path: str) -> str:
        if not original_full_path:
            return ""
        if original_full_path in self.name_map:
            return self.name_map[original_full_path]

        low = original_full_path.lower()
        if low.startswith("textures/"):
            relative = original_full_path[9:]
        elif low.startswith("models/"):
            relative = original_full_path[7:]
        else:
            relative = original_full_path

        first_dir = relative.split("/")[0] if "/" in relative else "uncategorized"
        new_dir = self.dir_map.get(first_dir, self.profile.directory_prefix(first_dir))
        original_filename, _ = os.path.splitext(Path(relative).name)

        processed = self._apply_replacements(original_filename)
        max_path_len = int(self.profile.features.get("max_path_length", 31))
        max_filename_len = max_path_len - len(new_dir) - 1

        if len(processed) <= max_filename_len:
            new_filename = processed
        else:
            hasher = hashlib.md5(original_full_path.encode("utf-8"))
            hash_str = hasher.hexdigest()[:4]
            trunc_len = max(0, max_filename_len - 5)
            base = processed[:trunc_len]
            new_filename = f"{base}_{hash_str}"

        final_filename = new_filename
        suffix = 1
        base_stem, base_ext = Path(new_filename).stem, Path(new_filename).suffix
        while f"{new_dir}/{final_filename}" in self.used_names_by_category[first_dir]:
            suffix_str = f"_{suffix}"
            new_stem_len = max_filename_len - len(suffix_str) - len(base_ext)
            new_stem = base_stem[:max(0, new_stem_len)]
            final_filename = f"{new_stem}{suffix_str}{base_ext}"
            suffix += 1

        if len(final_filename) > max_filename_len:
            final_filename = final_filename[:max_filename_len]

        final_path = f"{new_dir}/{final_filename}"
        self.name_map[original_full_path] = final_path
        self.used_names_by_category[first_dir].add(final_path)
        return final_path


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def write_log_lines(log_path: Path, lines: Iterable[str]):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def should_rebuild(out_img: Path, deps: List[Path], cfg: Config) -> bool:
    if cfg.force_rebuild or not out_img.exists():
        return True
    if not cfg.rebuild_if_newer:
        return False
    out_mtime = out_img.stat().st_mtime
    for dep in deps:
        if dep and dep.exists() and dep.stat().st_mtime > out_mtime:
            return True
    return False


_SUFFIX_RE = re.compile(r"_(?:add|glow|g)(\d+)?$", re.I)


def apply_suffix(q3_noext: str, desired: str, index: int = 0) -> str:
    parts = q3_noext.replace("\\", "/").split("/")
    base = parts[-1]
    base = _SUFFIX_RE.sub("", base)
    suffix = desired if index == 0 else f"{desired}{index+1}"
    parts[-1] = f"{base}_{suffix}"
    return "/".join(parts)


def blender_convert(cfg: Config, src: Path, dst: Path, log_path: Optional[Path] = None) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(cfg.blender_exe),
        "-b",
        "-P",
        str(cfg.blender_bake_script),
        "--",
        "--convert",
        str(src),
        "--out",
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log_path:
        write_log_lines(log_path, ["[CONVERT] " + " ".join(cmd), proc.stdout.strip(), f"RC={proc.returncode}", ""])
    return proc.returncode == 0 and dst.exists()


def black_to_transparency_glow(in_path: Path, out_path: Path, log_path: Optional[Path] = None) -> bool:
    try:
        from PIL import Image

        img = Image.open(in_path).convert("RGBA")
        luma = Image.open(in_path).convert("L")
        r, g, b, _ = img.split()
        rgba = Image.merge("RGBA", (r, g, b, luma))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rgba.save(out_path)
        if log_path:
            write_log_lines(log_path, [f"[GLOW_LUMA] {in_path} -> {out_path}"])
        return True
    except Exception as exc:
        if log_path:
            write_log_lines(log_path, [f"[WARN] glow conversion failed: {exc}"])
        return False


class Quake2Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.options = cfg.quake2
        self.enabled = bool(self.options.enabled)
        self.Image = None
        self.palette_img = None
        self.transparency_color = (0, 0, 0)
        self.can_write_wal = False

        if not self.enabled:
            return

        try:
            from PIL import Image
        except ImportError:
            print("[WARN] Pillow is required for Quake II WAL conversion but is not installed. WAL files will not be generated.")
            return

        self.Image = Image
        palette_path = self.options.resolved_palette(cfg.base_root)
        self.palette_img, self.transparency_color = self._load_palette(palette_path)
        if self.palette_img is not None:
            self.can_write_wal = True

    def _load_palette(self, palette_path: Optional[Path]):
        raw_palette = None
        if palette_path:
            try:
                with self.Image.open(palette_path) as pal_img:
                    if pal_img.mode != "P":
                        raise ValueError("Palette image is not paletted")
                    raw_palette = pal_img.getpalette()
                    if self.cfg.verbose:
                        print(f"[INFO] Loaded Quake II palette from {palette_path}")
            except (FileNotFoundError, ValueError) as exc:
                print(f"[WARN] {exc}. Using built-in Quake II palette.")

        if not raw_palette or len(raw_palette) != 768:
            if raw_palette and len(raw_palette) != 768:
                print(f"[WARN] Palette length {len(raw_palette)} invalid; expected 768. Using built-in Quake II palette.")
            raw_palette = list(DEFAULT_Q2_PALETTE_DATA)

        palette_img = self.Image.new("P", (1, 1))
        palette_img.putpalette(raw_palette)
        last_color_index = 255 * 3
        transparency_color = tuple(raw_palette[last_color_index:last_color_index + 3])
        return palette_img, transparency_color

    def _is_auxiliary(self, image_path: Path) -> bool:
        stem = image_path.stem.lower()
        return any(stem.endswith(suffix) for suffix in Q2_AUX_SUFFIXES)

    def _resample_filter(self, name: str):
        resampling = getattr(self.Image, "Resampling", self.Image)
        return getattr(resampling, name, getattr(self.Image, name, self.Image.NEAREST))

    def _scale_image(self, image):
        scale_percent = max(self.options.scale_percent, 1.0)
        if abs(scale_percent - 100.0) < 1e-6:
            return image
        scale_factor = scale_percent / 100.0
        new_width = max(1, int(round(image.width * scale_factor)))
        new_height = max(1, int(round(image.height * scale_factor)))
        if new_width == image.width and new_height == image.height:
            return image
        lanczos = self._resample_filter("LANCZOS")
        return image.resize((new_width, new_height), lanczos)

    def _derive_flags(self, mat: Material, original_name: str) -> Tuple[int, int, int]:
        surfaceflags = 0
        contentflags = CONTENTS_SOLID
        lightvalue = 0

        material_type = (mat.materialType or "").lower()
        original_lower = (original_name or mat.name or "").lower()

        if "light" in material_type or "/lights/" in original_lower:
            surfaceflags |= SURF_LIGHT
        if "glass" in material_type or "window" in material_type or "/glass" in original_lower:
            surfaceflags |= SURF_TRANS33
            contentflags = CONTENTS_WINDOW
        if any(tok in material_type for tok in ("water", "slime", "liquid")) or any(
            f"/{tok}/" in original_lower for tok in ("water", "fluids", "liquids")
        ):
            surfaceflags |= SURF_WARP
            contentflags = CONTENTS_WATER

        if mat.translucent and not (surfaceflags & SURF_WARP):
            surfaceflags |= SURF_TRANS33
            if contentflags == CONTENTS_SOLID:
                contentflags = CONTENTS_WINDOW

        if surfaceflags & SURF_LIGHT or "light" in original_lower:
            lightvalue = int(self.options.light_value)

        return surfaceflags, contentflags, lightvalue

    def _create_wal(self, image_path: Path, wal_path: Path, surfaceflags: int, contentflags: int, lightvalue: int):
        with self.Image.open(image_path) as src:
            img_rgba = src.convert("RGBA")

        img_rgba = self._scale_image(img_rgba)
        width, height = img_rgba.size

        img_rgb = self.Image.new("RGB", (width, height))
        pixels_rgba = img_rgba.load()
        pixels_rgb = img_rgb.load()

        threshold = max(0, min(255, int(self.options.alpha_threshold)))
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels_rgba[x, y]
                if a < threshold:
                    pixels_rgb[x, y] = self.transparency_color
                else:
                    pixels_rgb[x, y] = (r, g, b)

        dither_none = getattr(getattr(self.Image, "Dither", None), "NONE", 0)
        img_paletted = img_rgb.quantize(palette=self.palette_img, dither=dither_none)

        mipmaps = [img_paletted]
        box_filter = self._resample_filter("BOX")
        for _ in range(3):
            mip_w, mip_h = mipmaps[-1].size
            if mip_w <= 1 and mip_h <= 1:
                break
            next_size = (max(1, mip_w // 2), max(1, mip_h // 2))
            if next_size == mipmaps[-1].size:
                break
            mipmaps.append(mipmaps[-1].resize(next_size, box_filter))

        wal_path.parent.mkdir(parents=True, exist_ok=True)
        offsets = [0, 0, 0, 0]
        header_data_size = 32 + 8 + 16 + 32 + 12
        current_offset = header_data_size
        for idx, mip in enumerate(mipmaps):
            offsets[idx] = current_offset
            mip_w, mip_h = mip.size
            current_offset += mip_w * mip_h
        if mipmaps:
            last_offset = offsets[len(mipmaps) - 1]
        else:
            last_offset = header_data_size
        for idx in range(len(mipmaps), 4):
            offsets[idx] = last_offset

        with wal_path.open("wb") as f:
            name_bytes = wal_path.stem.encode("ascii", errors="ignore")[:32]
            name_bytes = name_bytes.ljust(32, b"\0")
            f.write(struct.pack("<32s", name_bytes))
            f.write(struct.pack("<II", width, height))
            f.write(struct.pack("<4I", *offsets))
            f.write(struct.pack("<32s", b""))
            f.write(struct.pack("<III", surfaceflags, contentflags, lightvalue))
            for mip in mipmaps:
                f.write(mip.tobytes())

        if self.cfg.verbose:
            try:
                rel = wal_path.relative_to(self.cfg.dst_base)
            except ValueError:
                rel = wal_path
            print(f"[WAL] Wrote {rel} (flags=0x{surfaceflags:08X}, contents=0x{contentflags:08X}, light={lightvalue})")

    def _build_material_name(self, original_name: str, mat: Material, image_path: Path) -> str:
        tokens: List[str] = []

        original_norm = (original_name or "").replace("\\", "/")
        if original_norm.startswith("textures/") or original_norm.startswith("models/"):
            stripped = original_norm.split("/", 1)[1]
        else:
            stripped = original_norm
        parts = [p for p in stripped.split("/") if p]
        if parts:
            tokens.append(parts[0])
            tokens.append(parts[-1])

        new_name = (mat.name or image_path.stem).replace("\\", "/")
        new_tail = new_name.split("/")[-1]
        if new_tail:
            tokens.append(new_tail)

        if mat.materialType:
            tokens.append(mat.materialType.lower())
        if mat.translucent:
            tokens.append("translucent")
        if mat.twoSided:
            tokens.append("twosided")
        if mat.noShadows:
            tokens.append("noshadows")
        if mat.nonsolid:
            tokens.append("nonsolid")
        if mat.noimpact:
            tokens.append("noimpact")

        cleaned: List[str] = []
        for tok in tokens:
            tok_clean = re.sub(r"[^a-z0-9_]+", "_", tok.lower()).strip("_")
            if tok_clean and tok_clean not in cleaned:
                cleaned.append(tok_clean)
        return "_".join(cleaned) if cleaned else "default"

    def _write_mat(self, image_path: Path, mat: Material, original_name: str):
        mat_path = image_path.with_suffix(".mat")
        material_name = self._build_material_name(original_name, mat, image_path)
        existing = None
        if mat_path.exists():
            try:
                existing = mat_path.read_text(encoding="utf-8").strip()
            except OSError:
                existing = None
        if existing == material_name:
            return
        mat_path.parent.mkdir(parents=True, exist_ok=True)
        mat_path.write_text(material_name + "\n", encoding="utf-8")
        if self.cfg.verbose:
            try:
                rel = mat_path.relative_to(self.cfg.dst_base)
            except ValueError:
                rel = mat_path
            print(f"[MAT] Wrote {rel} -> {material_name}")

    def write_outputs(self, image_path: Path, mat: Material, original_name: str):
        if not self.enabled:
            return
        if self.options.skip_auxiliary_maps and self._is_auxiliary(image_path):
            return
        if not image_path.exists():
            return
        if image_path.suffix.lower() != ".tga":
            return

        surfaceflags, contentflags, lightvalue = self._derive_flags(mat, original_name)
        if self.can_write_wal:
            try:
                self._create_wal(image_path, image_path.with_suffix(".wal"), surfaceflags, contentflags, lightvalue)
            except Exception as exc:
                print(f"[WARN] Failed to write WAL for {image_path}: {exc}")

        try:
            self._write_mat(image_path, mat, original_name)
        except Exception as exc:
            print(f"[WARN] Failed to write MAT for {image_path}: {exc}")


# ---------------------------------------------------------------------------
# Blender invocation and shader emission
# ---------------------------------------------------------------------------


def call_blender_bake(
    cfg: Config,
    mat: Material,
    diffuse: Path,
    out_path: Path,
    normal: Optional[Path],
    specular: Optional[Path],
    height: Optional[Path],
    log_path: Path,
    glow_path: Optional[Path] = None,
) -> int:
    args = [
        str(cfg.blender_exe),
        "-b",
        "-P",
        str(cfg.blender_bake_script),
        "--",
        "--diffuse",
        str(diffuse),
        "--out",
        str(out_path),
        "--samples",
        str(cfg.bake.samples),
        "--margin",
        str(cfg.bake.margin),
        "--sun_az",
        str(cfg.bake.sun_az),
        "--sun_el",
        str(cfg.bake.sun_el),
        "--power",
        str(cfg.bake.sun_power),
        "--ambient",
        str(cfg.bake.world_ambient),
        "--normal_strength",
        str(cfg.bake.normal_strength),
        "--height_scale",
        str(mat.height_scale),
        "--brightness",
        str(cfg.vibrance.brightness),
        "--contrast",
        str(cfg.vibrance.contrast),
        "--sat_gain",
        str(cfg.vibrance.sat_gain),
        "--diffuse_gain",
        str(cfg.vibrance.diffuse_gain),
        "--spec_gain",
        str(cfg.vibrance.spec_gain),
        "--roughness_bias",
        str(cfg.vibrance.roughness_bias),
        "--clearcoat_gain",
        str(cfg.vibrance.clearcoat_gain),
        "--clearcoat_roughness",
        str(cfg.bake.clearcoat_roughness),
    ]
    if normal:
        args.extend(["--normal", str(normal)])
    if specular:
        args.extend(["--spec", str(specular)])
    if height:
        args.extend(["--height", str(height)])
    if glow_path:
        args.extend(["--glow", str(glow_path)])

    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    write_log_lines(
        log_path,
        [
            "---- BLENDER INVOCATION ----",
            "CMD: " + " ".join(args),
            "---- BLENDER STDOUT ----",
            proc.stdout.strip(),
            "---- BLENDER STDERR ----",
            proc.stderr.strip(),
            f"---- END BLENDER (RC={proc.returncode}) ----",
        ],
    )
    return proc.returncode


def make_q3_shader(mat: Material, out_format: str, profile: TitleProfile) -> str:
    relative = mat.name.replace("\\", "/")
    q3_path = f"textures/{relative}"
    lines = [f"// Source: {mat.source_file.name}", q3_path, "{"]

    is_decal = "decals" in mat.name.lower()
    is_translucent = mat.translucent or is_decal or mat.materialType == "glass"
    if mat.twoSided:
        lines.append("\tcull disable")
    if mat.noShadows:
        lines.append("\tsurfaceparm nomarks")
    if is_translucent:
        lines.append("\tsurfaceparm trans")
    if mat.nonsolid:
        lines.append("\tsurfaceparm nonsolid")
    if mat.noimpact:
        lines.append("\tsurfaceparm noimpact")
    if mat.materialType:
        lines.append(f"\tsurfaceparm {mat.materialType}")
    if is_decal:
        lines.append("\tpolygonOffset")
        lines.append("\tsort decal")

    lines.append(f"\tqer_editorimage {q3_path}.{out_format}")

    if is_translucent:
        lines.extend(
            [
                "\t{",
                f"\t\tmap {q3_path}.{out_format}",
                "\t\tblendFunc GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA",
                "\t}",
            ]
        )
    else:
        lines.extend(
            [
                "\t{",
                "\t\tmap $lightmap",
                "\t}",
                "\t{",
                f"\t\tmap {q3_path}.{out_format}",
                "\t\tblendFunc GL_DST_COLOR GL_ZERO",
                "\t}",
            ]
        )

    for i, _ in enumerate(mat.additive_maps):
        add_noext = apply_suffix(q3_path, "add", i)
        lines.extend(
            [
                "\t{",
                f"\t\tmap {add_noext}.{out_format}",
                "\t\tblendFunc GL_ONE GL_ONE",
                "\t\trgbGen identity",
                "\t}",
            ]
        )

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def _resolve_profile(cfg: Config, profile_key: Optional[str]) -> TitleProfile:
    key = profile_key or cfg.profile_key
    if not key:
        if len(cfg.profiles) == 1:
            return next(iter(cfg.profiles.values()))
        raise SystemExit("No profile specified and multiple profiles present.")
    if key not in cfg.profiles:
        raise SystemExit(f"Unknown profile '{key}'. Available: {', '.join(sorted(cfg.profiles))}")
    return cfg.profiles[key]


def run_conversion(cfg: Config, profile: TitleProfile):
    output_format = cfg.output_format.lstrip('.')

    print(f"[INFO] Using profile: {profile.display_name} ({profile.key})")

    index = AssetIndex(cfg.base_root)
    index.build()

    parser = IdTech4MaterialParser(profile, verbose=cfg.verbose)
    all_materials: Dict[str, Material] = {}
    materials_dir = cfg.base_root / "materials"
    for mtr in materials_dir.rglob("*.mtr"):
        all_materials.update(parser.parse_file(mtr))

    print(f"[INFO] Parsed {len(all_materials)} total materials.")

    interesting: Dict[str, Material] = {}
    for name, mat in all_materials.items():
        low = name.lower()
        if low.startswith("textures/") or low.startswith("models/mapobjects/"):
            interesting[name] = mat
    print(f"[INFO] Filtered to {len(interesting)} materials for processing (textures & mapobjects).")

    name_manager = NameManager(interesting.keys(), profile)
    print("[INFO] New directory map created:")
    for old, new in name_manager.dir_map.items():
        print(f"    '{old}' -> '{new}'")

    quake2_pipeline = Quake2Pipeline(cfg)

    parser_failures = [
        name
        for name, mat in interesting.items()
        if mat.diffuse is None and "collision" not in name.lower() and not name.lower().startswith("textures/skies")
    ]
    if parser_failures:
        failure_log = cfg.dst_base / "parser_failures.txt"
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[WARN] {len(parser_failures)} materials failed parsing (diffuse token not found). See {failure_log}")
        with open(failure_log, "w", encoding="utf-8") as f:
            f.write(f"# {len(parser_failures)} materials failed parsing (no diffuse token found):\n\n")
            for name in sorted(parser_failures):
                mat = interesting[name]
                f.write(f"--- From file: {mat.source_file.name} ---\n")
                f.write(f"{mat.original_text}\n\n")

    processed_materials: Dict[str, Tuple[str, Material]] = {}
    for original_name, mat in interesting.items():
        new_name = name_manager.get_new_name(original_name)
        mat.name = new_name
        if mat.qer_editorimage:
            mat.qer_editorimage = name_manager.get_new_name(mat.qer_editorimage)
        if mat.additive_maps:
            mat.additive_maps_src = list(mat.additive_maps)
            mat.additive_maps = [name_manager.get_new_name(tok) for tok in mat.additive_maps]
        processed_materials[new_name] = (original_name, mat)

    shaders_by_file: Dict[str, List[str]] = {}
    for new_name, (original_name, mat) in sorted(processed_materials.items()):
        out_img = cfg.dst_base / "textures" / f"{new_name}.{output_format}"
        log_path = out_img.with_suffix(out_img.suffix + ".log")
        try:
            if not mat.diffuse:
                if cfg.verbose:
                    print(f"[SKIP] {original_name}: No diffuse stage.")
                continue

            diffuse_path = index.resolve(mat.diffuse)
            if not diffuse_path:
                if cfg.verbose and original_name not in parser_failures:
                    print(f"[SKIP] {original_name}: Cannot resolve file for token '{mat.diffuse}'.")
                continue

            normal_path = index.resolve(mat.normal)
            spec_path = index.resolve(mat.specular)
            height_path = index.resolve(mat.height)

            deps = [p for p in [diffuse_path, normal_path, spec_path, height_path] if p]
            if not should_rebuild(out_img, deps if deps else [diffuse_path], cfg):
                if cfg.verbose:
                    print(f"[SKIP] {original_name} -> {new_name}: Up-to-date.")
            else:
                print(f"[PROCESS] {original_name} -> {new_name}")
                if cfg.dry_run:
                    continue
                out_img.parent.mkdir(parents=True, exist_ok=True)
                glow_path = None
                if mat.additive_maps_src:
                    glow_token = mat.additive_maps_src[0]
                    glow_path = index.resolve(glow_token)
                if not any([normal_path, spec_path, height_path]) or mat.force_copy:
                    shutil.copy2(diffuse_path, out_img)
                else:
                    rc = call_blender_bake(cfg, mat, diffuse_path, out_img, normal_path, spec_path, height_path, log_path, glow_path)
                    if rc != 0:
                        print(f"[ERROR] Blender failed for {original_name} (rc={rc}). Falling back to copy.")
                        shutil.copy2(diffuse_path, out_img)

            if cfg.dry_run:
                continue

            if quake2_pipeline.enabled:
                quake2_pipeline.write_outputs(out_img, mat, original_name)

            q3_noext = f"textures/{new_name}"
            for idx, original_token in enumerate(mat.additive_maps_src):
                add_noext = apply_suffix(q3_noext, "add", idx)
                glow_noext = apply_suffix(q3_noext, "glow", idx)
                add_out = cfg.dst_base / (add_noext + f".{output_format}")
                glow_out = cfg.dst_base / (glow_noext + f".{output_format}")

                add_src = index.resolve(original_token)
                if not add_src:
                    write_log_lines(log_path, [f"[WARN] additive_map present but source not resolved: {original_token}"])
                    continue

                if add_out.suffix.lower() == add_src.suffix.lower():
                    add_out.parent.mkdir(parents=True, exist_ok=True)
                    if not add_out.exists() or add_src.stat().st_mtime > add_out.stat().st_mtime:
                        shutil.copy2(add_src, add_out)
                else:
                    if not blender_convert(cfg, add_src, add_out, log_path):
                        try:
                            shutil.copy2(add_src, add_out)
                        except Exception as exc:
                            write_log_lines(log_path, [f"[WARN] _add copy failed: {exc}"])

                if not black_to_transparency_glow(add_src, glow_out, log_path):
                    write_log_lines(log_path, [f"[WARN] _glow write failed for {add_src}"])

            source_filename = mat.source_file.stem
            shader_block = make_q3_shader(mat, output_format, profile)
            shaders_by_file.setdefault(source_filename, []).append(shader_block)
        except Exception as exc:
            print(f"[FATAL] Unhandled error processing {original_name}: {exc}")
            if not cfg.dry_run:
                write_log_lines(log_path, [f"FATAL ERROR: {exc}", traceback.format_exc()])

    if cfg.dry_run:
        print("[INFO] Dry run complete. No files were written.")
        return

    scripts_dir = cfg.dst_base / cfg.shader_output_dir
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shader_list = []
    for source_name, blocks in sorted(shaders_by_file.items()):
        shader_filename = profile.shader_overrides.get(source_name, f"{source_name}.shader")
        shader_list.append(shader_filename)
        shader_path = scripts_dir / shader_filename
        shader_path.write_text("\n\n".join(blocks), encoding="utf-8")

    (scripts_dir / "shaderlist.txt").write_text("\n".join(shader_list), encoding="utf-8")

    name_map_path = cfg.dst_base / "restructure_name_map.json"
    with open(name_map_path, "w", encoding="utf-8") as f:
        json.dump(name_manager.name_map, f, indent=2, sort_keys=True)

    print(f"[INFO] Wrote name mapping log to {name_map_path}")
    print(f"[DONE] Wrote {len(shaders_by_file)} shader files to {scripts_dir}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="idTech4 -> idTech23 converter")
    ap.add_argument("--config", required=True, help="Path to convert_config.json")
    ap.add_argument("--profile", help="Profile key to use (overrides config)")
    args = ap.parse_args(argv)

    cfg = Config.from_json(args.config)
    profile = _resolve_profile(cfg, args.profile)
    run_conversion(cfg, profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())

