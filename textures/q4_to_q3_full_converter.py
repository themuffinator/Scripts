# q4_to_q3_full_converter.py
# ---------------------------------------------------------------------------
# Quake 4 -> Quake 3 texture & shader converter (Blender bake driver)
# This version contains the final, definitive parser and the dynamic
# directory restructuring and name compliance logic.
#
# CHANGES (minimal):
# - Pass resolved additive map to Blender (--glow) so it's baked additively into the base.
# - Emit two extra files when glow exists:
#     <base>_add.<ext>  (color as-is)
#     <base>_glow.<ext> (grayscale for Q2Re - **NOW USING PILLOW FOR DESATURATION**)
# - Update shader to reference <base>_add as blend add stage (suffix-normalized).
# - Light materials are given q3map_surfacelight.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import hashlib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from collections import defaultdict

# ============================== CONFIG =======================================

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
    bake: BakeSettings = field(default_factory=BakeSettings)
    vibrance: VibranceSettings = field(default_factory=VibranceSettings)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        def pick(d: dict, dc_type):
            if not isinstance(d, dict): return {}
            allowed = {f.name for f in fields(dc_type)}
            return {k: v for k, v in d.items() if k in allowed}
        
        if "output_ext" in raw:
            raw["output_format"] = raw["output_ext"].strip(".")

        bake = BakeSettings(**pick(raw.get("bake", {}), BakeSettings))
        vibr = VibranceSettings(**pick(raw.get("vibrance", {}), VibranceSettings))

        top = pick(raw, Config)
        for key in ("base_root", "dst_base", "blender_exe", "blender_bake_script", "shader_output_dir"):
            if key in raw: top[key] = Path(raw[key])

        top["bake"] = bake
        top["vibrance"] = vibr
        return Config(**top)

# =============================== INDEX =======================================

IMAGE_EXTS = [".tga", ".dds", ".png", ".jpg", ".jpeg", ".tif", ".bmp"]

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

    def resolve(self, token: str) -> Optional[Path]:
        if not token: return None
        key = token.replace("\\", "/").lstrip("/").lower()
        if key in self._map: return self._map[key]
        key_noext, _ = os.path.splitext(key)
        for ext in IMAGE_EXTS:
            if f"{key_noext}{ext}" in self._map: return self._map[f"{key_noext}{ext}"]
        return None

# =============================== MATERIAL PARSER (DEFINITIVE VERSION) ====================================

@dataclass
class Q4Material:
    name: str
    source_file: Path
    diffuse: Optional[str] = None
    normal: Optional[str] = None
    specular: Optional[str] = None
    height: Optional[str] = None
    additive_map: Optional[str] = None
    additive_map_src: Optional[str] = None  # <--- preserve original token before remap
    height_scale: float = 1.0
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

class Q4MaterialParser:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.re_comment_line = re.compile(r"//.*?$", re.MULTILINE)
        self.re_comment_block = re.compile(r"/\*.*?\*/", re.DOTALL)
        self.re_guide = re.compile(r'^\s*guide\s+(?P<mat>\S+)\s+(?P<macro>[A-Za-z0-9_]+)\s*\((?P<args>[^)]*)\)\s*$', re.IGNORECASE | re.MULTILINE)
        self.re_block_header = re.compile(r"^\s*((?:textures|models)/[^\s{]+)\s*\{\s*$", re.IGNORECASE | re.MULTILINE)

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
                if text[j] == "{": depth += 1
                elif text[j] == "}": depth -= 1
                j += 1
            out.append((name, text[i:j-1]))
        return out

    def _parse_material_block(self, name: str, block: str, src: Path) -> Q4Material:
        mat = Q4Material(name=name, source_file=src)
        mat.original_text = f"{name} {{\n{block}\n}}"
        if re.search(r"\btwosided\b", block, re.I): mat.twoSided = True
        if re.search(r"\btranslucent\b", block, re.I): mat.translucent = True
        if re.search(r"\bnoshadows\b", block, re.I): mat.noShadows = True
        if re.search(r"\bnonsolid\b", block, re.I): mat.nonsolid = True
        if re.search(r"\bnoimpact\b", block, re.I): mat.noimpact = True
        if m := re.search(r"qer_?editorimage\s+([^\s}]+)", block, re.I): mat.qer_editorimage = m.group(1)
        if m := re.search(r"^\s*materialType\s+([^\s;}]+)", block, re.MULTILINE | re.I): mat.materialType = m.group(1).lower()
        if m := re.search(r"^\s*diffusemap\s+([^\s;}]+)", block, re.MULTILINE | re.I): mat.diffuse = m.group(1)
        if m := re.search(r"^\s*specularmap\s+([^\s;}]+)", block, re.MULTILINE | re.I): mat.specular = m.group(1)
        if m := re.search(r"^\s*bumpmap\s+([^\s;}]+)", block, re.MULTILINE | re.I):
            if 'addnormals' not in m.group(1).lower(): mat.normal = m.group(1)
        i, n = 0, len(block)
        while i < n:
            start_brace = block.find('{', i)
            if start_brace == -1: break
            depth, j = 1, start_brace + 1
            while j < n and depth > 0:
                if block[j] == '{': depth += 1
                elif block[j] == '}': depth -= 1
                j += 1
            if depth == 0:
                stage_content = block[start_brace + 1 : j - 1]
                blend_match = re.search(r'\bblend\s+([^\s,;}]+)', stage_content, re.I)
                blend_mode = blend_match.group(1).lower() if blend_match else None
                if not mat.diffuse and blend_mode in ('diffusemap', 'diffuse'):
                    if m := re.search(r'\bmap\s+([^\s;}]+)', stage_content, re.I): mat.diffuse = m.group(1)
                if not mat.additive_map and blend_mode == 'add':
                    if m := re.search(r'\bmap\s+([^\s;}]+)', stage_content, re.I): 
                        mat.additive_map = m.group(1)
                        mat.additive_map_src = mat.additive_map
                if not mat.height and blend_mode == 'bumpmap':
                    if m := re.search(r'\bmap\s+heightmap\s*\(\s*([^\s,)]+)\s*,\s*([0-9.-]+)\s*\)', stage_content, re.I):
                         mat.height, mat.height_scale = m.group(1).replace("\\", "/"), float(m.group(2))
                i = j
            else: i = start_brace + 1
        if not mat.normal and not mat.height:
             bn = re.search(r"^\s*bumpmap\s+addnormals\s*\(\s*([^\s,()]+)\s*,\s*heightmap\s*\(\s*([^\s,()]+)\s*,\s*([0-9.+\-]+)\s*\)\s*\)", block, re.MULTILINE | re.I)
             if bn: mat.normal, mat.height, mat.height_scale = bn.group(1), bn.group(2), float(bn.group(3))
        if not mat.diffuse and mat.qer_editorimage: mat.diffuse = mat.qer_editorimage
        if not mat.diffuse:
            if m := re.search(r"^\s*map\s+([^\s}]+)", block, re.MULTILINE | re.I):
                map_val = m.group(1)
                if 'heightmap' not in map_val.lower() and 'addnormals' not in map_val.lower():
                    mat.diffuse = map_val
        return mat

    def parse_file(self, path: Path) -> Dict[str, Q4Material]:
        if not path.exists(): return {}
        text = self.strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
        materials: Dict[str, Q4Material] = {}
        for name, block in self._iter_top_level_blocks(text):
            if name.lower().startswith(("textures/", "models/")):
                materials[name] = self._parse_material_block(name, block, path)
        
        for m in self.re_guide.finditer(text):
            mat_name, macro = m.group("mat"), m.group("macro").lower().strip()
            args = [a.strip().strip('"') for a in m.group("args").split(",") if a.strip()]
            if not mat_name.lower().startswith(("textures/", "models/")): continue
            
            mat = materials.setdefault(mat_name, Q4Material(name=mat_name, source_file=path))
            mat.is_guide, mat.guide_macro = True, macro
            if not mat.original_text: mat.original_text = m.group(0).strip()

            def norm_token(t):
                if not t: return t
                t = t.replace("\\", "/").lstrip("/")
                if not t.lower().startswith(("textures/", "models/")): t = "textures/" + t
                return t
            def maps_from(p): mat.diffuse,mat.normal,mat.specular,mat.height,mat.qer_editorimage=f"{p}_d",f"{p}_local",f"{p}_s",f"{p}_h",f"{p}_d"
            def maps_from_noheight(p): maps_from(p); mat.height=None
            def variant_diffuse_only(v,b): mat.diffuse,mat.qer_editorimage=f"{v}_d",f"{v}_d"; mat.normal,mat.specular,mat.height=f"{b}_local",f"{b}_s",f"{b}_h"
            def alpha_flags(two_sided=False, no_shadows=False): mat.translucent=True; mat.twoSided=two_sided; mat.noShadows=no_shadows
            def set_type(i=1):
                if len(args) > i and args[i]: mat.materialType = args[i].strip()

            base = norm_token(args[0] if len(args) >= 1 else "")
            variant = norm_token(args[1] if len(args) >= 2 else "")

            if macro in ("generic_materialimageshader", "generic_shader", "generic_shader_ed"): maps_from(base)
            elif macro in ("generic_shader_mi", "generic_full_noheight_mi"):
                if variant and base: variant_diffuse_only(variant, base)
                elif variant: mat.diffuse, mat.qer_editorimage = f"{variant}_d", f"{variant}_d"
                else: maps_from(base)
            elif macro in ("generic_localvariant", "generic_localvariant_mi"):
                if variant and base: variant_diffuse_only(variant, base)
                else: maps_from(base)
            elif macro in ("generic_shader2sided",): maps_from(base); mat.twoSided = True
            elif macro == "generic_full_noheight": maps_from_noheight(base)
            elif macro == "generic_full_noheight_type": maps_from_noheight(base); set_type()
            elif macro in ("generic_variant_noheight", "generic_variant_noheight_mi"):
                if variant and base: variant_diffuse_only(variant, base)
                else: maps_from(variant or base)
                mat.height = None
            elif macro == "generic_nonormal": maps_from(base); mat.normal = None
            elif macro == "generic_nonormal_height":
                maps_from(base); mat.normal = None; mat.height_scale = float(args[1]) if len(args)>1 else 1.0
            elif macro == "generic_shader_heightmap":
                mat.diffuse = base; mat.qer_editorimage = base; mat.height = base
                mat.normal = None; mat.specular = None
                mat.height_scale = float(args[1]) if len(args) > 1 else 1.0
            elif macro == "generic_nonormal_height_type":
                maps_from(base); mat.normal = None; mat.height_scale = float(args[1]) if len(args)>1 else 1.0; set_type(2)
            elif macro in ("generic_alpha", "generic_alpha_ed", "generic_alpha_lv", "generic_alpha_mi"): alpha_flags(); maps_from(base)
            elif macro == "generic_alphaglow": alpha_flags(); maps_from(base); mat.additive_map = variant or f"{base}_g"; mat.additive_map_src = mat.additive_map
            elif macro == "generic_alpha_noshadows": alpha_flags(no_shadows=True); maps_from(base)
            elif macro == "generic_alphanoshadow2s": alpha_flags(two_sided=True, no_shadows=True); maps_from(base)
            elif macro in ("generic_shader2sidedalpha", "generic_localvalpha2"): maps_from(base); alpha_flags(two_sided=True)
            elif macro in ("generic_shader2sidedalpha_lv", "generic_shader2sidedalpha_mi", "generic_shader2sidedalpha_miv"):
                if variant and base: variant_diffuse_only(variant, base); alpha_flags(two_sided=True)
                else: maps_from(variant or base); alpha_flags(two_sided=True)
            elif macro == "generic_shader2sidedalpha_type": maps_from(base); alpha_flags(two_sided=True); set_type()
            elif macro in ("generic_glow", "generic_glow_mi"): maps_from(base); mat.additive_map = variant or f"{base}_g"; mat.additive_map_src = mat.additive_map
            elif macro in ("generic_terminal_replaceglow", "generic_terminal_replaceglow2"): maps_from(base); mat.additive_map = variant or f"{base}_add"; mat.additive_map_src = mat.additive_map
            elif macro == "generic_typeshader": maps_from(base); set_type()
            elif macro == "generic_localvariant_typeshader": maps_from(variant or base); set_type(2)
            elif macro == "generic_alpha_type": maps_from(base); alpha_flags(); set_type()
            elif macro == "generic_colorvariant":
                # A supplies _d; B supplies _local/_s/_h
                if base and variant:
                    variant_diffuse_only(base, variant)
                else:
                    maps_from(base or variant)
            
        return materials

# =============================== NAME RESTRUCTURING =====================================

class NameManager:
    def __init__(self, original_paths: list):
        self.dir_map = self._build_directory_map(original_paths)
        self.name_map: Dict[str, str] = {}
        self.used_names_by_category: Dict[str, set] = defaultdict(set)

    def _build_directory_map(self, original_paths: list) -> dict:
        first_level_dirs = set()
        for path in original_paths:
            path_lower = path.lower()
            if path_lower.startswith("textures/"):
                relative_path = path[9:]
            elif path_lower.startswith("models/"):
                relative_path = path[7:]
            else:
                continue

            if "/" in relative_path:
                first_level_dirs.add(relative_path.split('/')[0])

        dir_map = {}
        for original_dir in sorted(list(first_level_dirs)):
            if original_dir == "command_ship": dir_map[original_dir] = "q4_ship"
            elif original_dir == "consoles": dir_map[original_dir] = "q4_con"
            elif original_dir == "decals": dir_map[original_dir] = "q4_dec"
            elif original_dir == "medlabs": dir_map[original_dir] = "q4_labs"
            elif original_dir == "mptextures": dir_map[original_dir] = "q4_mp"
            elif original_dir == "stroyent": dir_map[original_dir] = "q4_stroy"
            elif original_dir == "terminal": dir_map[original_dir] = "q4_term"
            elif original_dir == "q4x_mptextures": dir_map[original_dir] = "q4x_mp"
            elif original_dir == "q4x_q4xdm2": dir_map[original_dir] = "q4x_dm2"
            elif original_dir == "q4x_powercore": dir_map[original_dir] = "q4x_power"
            elif original_dir == "q4x_speeder": dir_map[original_dir] = "q4x_speed"
            elif original_dir.lower().startswith("q4x_"): dir_map[original_dir] = original_dir
            elif original_dir.lower().startswith("common_"): dir_map[original_dir] = f"q4_{original_dir[7:]}"
            else: dir_map[original_dir] = f"q4_{original_dir}"
        return dir_map

    def get_new_name(self, original_full_path: str) -> str:
        if not original_full_path: return ""
        if original_full_path in self.name_map:
            return self.name_map[original_full_path]

        path_lower = original_full_path.lower()
        if path_lower.startswith("textures/"):
            relative_path = original_full_path[9:]
        elif path_lower.startswith("models/"):
            relative_path = original_full_path[7:]
        else:
            relative_path = original_full_path

        original_first_dir = relative_path.split('/')[0] if '/' in relative_path else "uncategorized"
        new_dir = self.dir_map.get(original_first_dir, f"q4_{original_first_dir}")
        original_filename, _ = os.path.splitext(Path(relative_path).name)

        p = original_filename
        p = p.replace("noshadows", "nsw").replace("shadows", "sw").replace("shadow", "shw")
        p = p.replace("light", "lt").replace("vertical", "vert").replace("visportal", "vis")
        p = p.replace("support", "supp").replace("nonsolid", "ns").replace("texture", "tex")
        p = p.replace("collision", "coll")
        processed_filename = p

        max_path_len = 31
        max_filename_len = max_path_len - len(new_dir) - 1

        if len(processed_filename) <= max_filename_len:
            new_filename = processed_filename
        else:
            hasher = hashlib.md5(original_full_path.encode('utf-8'))
            hash_str = hasher.hexdigest()[:4]
            trunc_len = max_filename_len - 5
            base_name = processed_filename[:trunc_len]
            new_filename = f"{base_name}_{hash_str}"

        final_filename = new_filename
        suffix = 1
        base_stem, base_ext = Path(new_filename).stem, Path(new_filename).suffix
        while f"{new_dir}/{final_filename}" in self.used_names_by_category[original_first_dir]:
            suffix_str = f"_{suffix}"
            new_stem_len = max_filename_len - len(suffix_str) - len(base_ext)
            new_stem = base_stem[:new_stem_len]
            final_filename = f"{new_stem}{suffix_str}{base_ext}"
            suffix += 1
        
        if len(final_filename) > max_filename_len:
            final_filename = final_filename[:max_filename_len]

        final_path = f"{new_dir}/{final_filename}"
        self.name_map[original_full_path] = final_path
        self.used_names_by_category[original_first_dir].add(final_path)
        return final_path

# =============================== UTILITIES ========================

def write_log_lines(log_path: Path, lines: List[str]):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        for ln in lines: f.write(ln + "\n")

def should_rebuild(out_img: Path, deps: List[Path], cfg: Config) -> bool:
    if cfg.force_rebuild or not out_img.exists(): return True
    if not cfg.rebuild_if_newer: return False
    out_mtime = out_img.stat().st_mtime
    for d in deps:
        if d and d.exists() and d.stat().st_mtime > out_mtime: return True
    return False

# --- helpers for additive/glow outputs ---
import re as _re
_SUFFIX_RE = _re.compile(r"_(?:add|glow|g)$", _re.I)

def apply_suffix(q3_noext: str, desired: str) -> str:
    """Return path with terminal _add/_glow/_g normalized to _<desired>."""
    parts = q3_noext.replace("\\", "/").split("/")
    base = parts[-1]
    base = _SUFFIX_RE.sub("", base)
    parts[-1] = f"{base}_{desired}"
    return "/".join(parts)

def blender_convert(cfg: Config, src: Path, dst: Path, log_path: Optional[Path] = None) -> bool:
    """Ask Blender helper to load & save to desired container (no content changes)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(cfg.blender_exe), "-b", "-P", str(cfg.blender_bake_script), "--", "--convert", str(src), "--out", str(dst)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log_path:
        write_log_lines(log_path, ["[CONVERT] " + " ".join(cmd), p.stdout.strip(), f"RC={p.returncode}", ""])
    return p.returncode == 0 and dst.exists()

def grayscale_to_alpha_with_pillow(in_path: Path, out_path: Path, log_path: Optional[Path] = None) -> bool:
    """
    Keep original RGB; set alpha to luminance (white=opaque, black=transparent).
    """
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
    except Exception as e:
        if log_path:
            write_log_lines(log_path, [f"[ERROR] grayscale_to_alpha failed: {e}"])
        return False

# =============================== BLENDER & SHADER ==========================

def call_blender_bake(cfg: Config, mat: Q4Material, d_path: Path, out_path: Path, n_path, s_path, h_path, log_path, glow_path=None) -> int:
    args = [
        str(cfg.blender_exe), "-b", "-P", str(cfg.blender_bake_script), "--",
        "--diffuse", str(d_path), "--out", str(out_path),
        "--samples", str(cfg.bake.samples), "--margin", str(cfg.bake.margin),
        "--sun_az", str(cfg.bake.sun_az), "--sun_el", str(cfg.bake.sun_el),
        "--power", str(cfg.bake.sun_power), "--ambient", str(cfg.bake.world_ambient),
        "--normal_strength", str(cfg.bake.normal_strength), "--height_scale", str(mat.height_scale),
        "--brightness", str(cfg.vibrance.brightness), "--contrast", str(cfg.vibrance.contrast),
        "--sat_gain", str(cfg.vibrance.sat_gain), "--diffuse_gain", str(cfg.vibrance.diffuse_gain),
        "--spec_gain", str(cfg.vibrance.spec_gain), "--roughness_bias", str(cfg.vibrance.roughness_bias)
    ]
    if n_path: args.extend(["--normal", str(n_path)])
    if s_path: args.extend(["--spec", str(s_path)])
    if h_path: args.extend(["--height", str(h_path)])
    if glow_path: args.extend(["--glow", str(glow_path)])  # optional
    proc = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if log_path:
        write_log_lines(log_path, [
            "---- BLENDER INVOCATION ----", "CMD: " + " ".join(args),
            "---- BLENDER STDOUT ----", proc.stdout.strip(),
            "---- BLENDER STDERR ----", proc.stderr.strip(),
            f"---- END BLENDER (RC={proc.returncode}) ----"
        ])
    return proc.returncode

def make_q3_shader(mat: Q4Material, out_format: str) -> str:
    relative_path = mat.name.replace("\\", "/")
    q3_path = f"textures/{relative_path}"

    lines = [f"// Source: {mat.source_file.name}", q3_path, "{"]
    is_decal = mat.name.lower().startswith("q4_dec/")
    is_translucent = mat.translucent or is_decal or mat.materialType == "glass"
    if mat.twoSided: lines.append("\tcull disable")
    if mat.noShadows: lines.append("\tsurfaceparm nomarks")
    if is_translucent: lines.append("\tsurfaceparm trans")
    if mat.nonsolid: lines.append("\tsurfaceparm nonsolid")
    if mat.noimpact: lines.append("\tsurfaceparm noimpact")
    if mat.materialType: lines.append(f"\tsurfaceparm {mat.materialType}")
    if is_decal: lines.append("\tpolygonOffset"); lines.append("\tsort decal")
    lines.append(f"\tqer_editorimage {q3_path}.{out_format}")
    if is_translucent:
        lines.extend(["\t{", f"\t\tmap {q3_path}.{out_format}", "\t\tblendFunc GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA", "\t}"])
    else:
        lines.extend(["\t{", f"\t\tmap $lightmap", "\t}", "\t{", f"\t\tmap {q3_path}.{out_format}", "\t\tblendFunc GL_DST_COLOR GL_ZERO", "\t}"])
    if mat.additive_map:
        add_noext = apply_suffix(q3_path, "add")
        lines.extend(["\t{", f"\t\tmap {add_noext}.{out_format}", "\t\tblendFunc add", "\t}"])
    lines.append("}")
    return "\n".join(lines)

# =============================== MAIN ========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = Config.from_json(args.config)
    output_format = cfg.output_format.lstrip('.')
    index = AssetIndex(cfg.base_root); index.build()
    parser = Q4MaterialParser(verbose=cfg.verbose)
    all_mats: Dict[str, Q4Material] = {}
    for mtr in (cfg.base_root / "materials").rglob("*.mtr"):
        all_mats.update(parser.parse_file(mtr))
    print(f"[INFO] Parsed {len(all_mats)} total materials.")

    # Filter materials to only include textures/ and models/mapobjects/
    materials_to_process = {}
    for name, mat in all_mats.items():
        name_lower = name.lower()
        if name_lower.startswith("textures/") or name_lower.startswith("models/mapobjects/"):
            materials_to_process[name] = mat
    print(f"[INFO] Filtered to {len(materials_to_process)} materials for processing (textures & mapobjects).")
    
    print("[INFO] Building new name and directory structure...")
    name_manager = NameManager(list(materials_to_process.keys()))
    print("[INFO] New directory map created:")
    for old, new in name_manager.dir_map.items():
        print(f"    '{old}' -> '{new}'")
    
    parser_failures = [name for name, mat in materials_to_process.items() if mat.diffuse is None and "collision" not in name.lower() and not name.lower().startswith("textures/skies")]
    if parser_failures:
        failure_log_path = cfg.dst_base / "parser_failures.txt"
        failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[WARN] {len(parser_failures)} materials failed parsing (diffuse token not found). See {failure_log_path}")
        with open(failure_log_path, "w", encoding="utf-8") as f:
            f.write(f"# {len(parser_failures)} materials failed parsing (no diffuse token found):\n\n")
            for name in sorted(parser_failures):
                mat = materials_to_process[name]
                f.write(f"--- From file: {mat.source_file.name} ---\n")
                f.write(f"{mat.original_text}\n\n")
    
    shaders_by_file: Dict[str, List[str]] = {}
    processed_materials = {}
    for original_name, mat in materials_to_process.items():
        new_name = name_manager.get_new_name(original_name)
        mat.name = new_name
        if mat.qer_editorimage:
            mat.qer_editorimage = name_manager.get_new_name(mat.qer_editorimage)
        if mat.additive_map:
            mat.additive_map_src = mat.additive_map
            mat.additive_map = name_manager.get_new_name(mat.additive_map)
        processed_materials[new_name] = (original_name, mat)

    for new_name, (original_name, mat) in sorted(processed_materials.items()):
        # --- NEW: Place output files inside a 'textures' root directory ---
        out_img = cfg.dst_base / "textures" / f"{new_name}.{output_format}"
        log_path = out_img.with_suffix(out_img.suffix + ".log")
        try:
            if not mat.diffuse:
                if cfg.verbose: print(f"[SKIP] {original_name}: No diffuse stage.")
                continue
            d_path = index.resolve(mat.diffuse)
            if not d_path:
                if cfg.verbose and original_name not in parser_failures: print(f"[SKIP] {original_name}: Cannot resolve file for token '{mat.diffuse}'.")
                continue
            n_path, s_path, h_path = index.resolve(mat.normal), index.resolve(mat.specular), index.resolve(mat.height)

            deps = [p for p in [d_path, n_path, s_path, h_path] if p]
            if not should_rebuild(out_img, deps if deps else [d_path], cfg):
                if cfg.verbose: print(f"[SKIP] {original_name} -> {new_name}: Up-to-date.")
            else:
                print(f"[PROCESS] {original_name} -> {new_name}")
                out_img.parent.mkdir(parents=True, exist_ok=True)
                if not any([n_path, s_path, h_path]) or mat.force_copy:
                    shutil.copy2(d_path, out_img)
                else:
                    rc = call_blender_bake(cfg, mat, d_path, out_img, n_path, s_path, h_path, log_path)
                    if rc != 0:
                        print(f"[ERROR] Blender failed for {original_name} (rc={rc}). Falling back to copy.")
                        shutil.copy2(d_path, out_img)

            # --- Ensure _add and _glow files exist when additive map is declared ---
            if mat.additive_map:
                q3_noext = f"textures/{new_name}"
                add_noext  = apply_suffix(q3_noext, "add")
                glow_noext = apply_suffix(q3_noext, "glow")
                add_out  = cfg.dst_base / (add_noext  + f".{output_format}")
                glow_out = cfg.dst_base / (glow_noext + f".{output_format}")
                # Resolve source of additive map from original token (pre-remap) if available
                add_src = None
                if getattr(mat, "additive_map_src", None):
                    add_src = index.resolve(mat.additive_map_src)
                if add_src:
                    # Write _add as unchanged visual (convert container if needed)
                    if add_out.suffix.lower() == add_src.suffix.lower():
                        add_out.parent.mkdir(parents=True, exist_ok=True)
                        if not add_out.exists() or add_src.stat().st_mtime > add_out.stat().st_mtime:
                            shutil.copy2(add_src, add_out)
                    else:
                        if not blender_convert(cfg, add_src, add_out, log_path):
                            # last resort: try raw copy if same type; else skip
                            try: shutil.copy2(add_src, add_out)
                            except Exception as e: write_log_lines(log_path, [f"[WARN] _add copy failed: {e}"])
                    # Write _glow with alpha=luminance
                    if not grayscale_to_alpha_with_pillow(add_src, glow_out, log_path):
                        write_log_lines(log_path, [f"[WARN] _glow write failed for {add_src}"])
                else:
                    write_log_lines(log_path, [f"[WARN] additive_map present but source not resolved: {mat.additive_map}"])

            # Shader
            source_filename = mat.source_file.stem
            shader_block = make_q3_shader(mat, output_format)
            shaders_by_file.setdefault(source_filename, []).append(shader_block)
        except Exception as e:
            print(f"[FATAL] Unhandled error processing {original_name}: {e}")
            if log_path: write_log_lines(log_path, [f"FATAL ERROR: {e}", traceback.format_exc()])
    
    scripts_dir = cfg.dst_base / cfg.shader_output_dir
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shader_list = []
    for source_name, blocks in sorted(shaders_by_file.items()):
        shader_filename = f"{source_name}"
        shader_list.append(shader_filename)
        shader_file_path = scripts_dir / shader_filename
        shader_file_path.write_text("\n\n".join(blocks), encoding="utf-8")
    (scripts_dir / "shaderlist.txt").write_text("\n".join(shader_list), encoding="utf-8")
    
    name_map_path = cfg.dst_base / "restructure_name_map.json"
    with open(name_map_path, "w", encoding="utf-8") as f:
        json.dump(name_manager.name_map, f, indent=2, sort_keys=True)
    print(f"[INFO] Wrote name mapping log to {name_map_path}")
    print(f"[DONE] Wrote {len(shaders_by_file)} shader files to {scripts_dir}")

if __name__ == "__main__":
    sys.exit(main())
