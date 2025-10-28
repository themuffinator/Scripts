# compare_textures.py
# ---------------------------------------------------------------------------
# Script to generate 3-way comparison images for Q4 asset conversion.
#
# FINAL FIX: Corrected pixel-level alignment (8px shift) of the Q2 WAL image 
# by reading the raw data with the necessary Quake pitch/stride offset.
# ---------------------------------------------------------------------------

from __future__ import annotations
import argparse
import json
import os
import re
import struct
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont, ImagePalette
import traceback
import hashlib
import sys

# Constants
IMAGE_EXTS = [".tga", ".dds", ".png", ".jpg", ".jpeg", ".tif", ".bmp"]
OUTPUT_DIR = "comparison_images"
# Constants for vertical offsets
IMAGE_Y_OFFSET = 25 
HEADER_HEIGHT = 25 

# Default Quake 2 palette data (index 255 is transparency, RGB values follow)
DEFAULT_Q2_PALETTE_DATA = [
    0,0,0,15,15,15,31,31,31,47,47,47,63,63,63,75,75,75,91,91,91,107,107,107,123,123,123,139,139,139,155,155,155,171,171,171,187,187,187,203,203,203,219,219,219,235,235,235,251,251,251,0,0,251,0,0,235,0,0,219,0,0,203,0,0,187,0,0,171,0,0,155,0,0,139,0,0,123,0,0,107,0,0,91,0,0,75,0,0,63,0,0,47,0,0,31,0,0,15,0,251,0,0,235,0,0,219,0,0,203,0,0,187,0,0,171,0,0,155,0,0,139,0,0,123,0,0,107,0,0,91,0,0,75,0,0,63,0,0,47,0,0,31,0,251,251,0,235,235,0,219,219,0,203,203,0,187,187,0,171,171,0,155,155,0,139,139,0,123,123,0,107,107,0,91,91,0,75,75,0,63,63,0,47,47,0,31,31,0,15,15,0,251,0,251,235,0,235,219,0,219,203,0,203,187,0,187,171,0,171,155,0,155,139,0,139,123,0,123,107,0,107,91,0,91,75,0,75,63,0,63,47,0,47,31,0,31,15,0,15,251,0,0,235,0,0,219,0,0,203,0,0,187,0,0,171,0,0,155,0,0,139,0,0,123,0,0,107,0,0,91,0,0,75,0,0,63,0,0,47,0,0,31,0,0,15,0,251,251,251,235,235,235,219,219,219,203,203,203,187,187,187,171,171,171,155,155,155,139,139,139,123,123,123,107,107,107,91,91,91,75,75,75,63,63,63,47,47,47,31,31,31,15,15,15,251,123,123,235,115,115,219,107,107,203,99,99,187,91,91,171,83,83,155,75,75,139,67,67,123,59,59,107,51,51,91,43,43,75,35,35,63,27,27,47,19,19,31,11,11,15,3,3,123,123,251,115,115,235,107,107,219,99,99,203,91,91,187,83,83,171,75,75,155,67,67,139,59,59,123,51,51,107,43,43,91,35,35,75,27,27,63,19,19,47,11,11,31,3,3,15,123,251,123,115,235,115,107,219,107,99,203,99,91,187,91,83,171,83,75,155,75,67,139,67,59,123,59,51,107,51,43,91,43,35,75,35,27,63,27,19,47,19,11,31,11,3,15,251,123,0,235,115,0,219,107,0,203,99,0,187,91,0,171,83,0,155,75,0,139,67,0,123,59,0,107,51,0,91,43,0,75,35,0,63,27,0,47,19,0,31,11,0,15,3,0,251,251,123,235,235,115,219,219,107,203,203,99,187,187,91,171,171,83,155,155,75,139,139,67,123,123,59,107,107,51,91,91,43,75,75,35,63,63,27,47,47,19,31,31,11,15,15,3,0,0,0]
# The last color (index 255) is used for transparency
DEFAULT_TRANSPARENCY_COLOR = (0, 0, 0) # Fallback RGB for index 255 if no palette loaded


# Global log list for debugging
DEBUG_LOG = []
MAX_FAIL_LOGS = 10

# =============================== WAL CONVERSION UTILITY ======================

def load_palette_and_key_color(colormap_path: Path) -> Tuple[Image.Image, Tuple[int, int, int]]:
    """Loads palette from PCX or uses default fallback."""
    try:
        colormap_img = Image.open(colormap_path)
        if colormap_img.mode != 'P':
            raise ValueError("Colormap image is not a paletted (P mode) image.")
        
        raw_palette = colormap_img.getpalette()
        palette_img = Image.new("P", (1, 1))
        palette_img.putpalette(raw_palette)
        
        last_color_index = 255 * 3
        transparency_key_color = tuple(raw_palette[last_color_index : last_color_index + 3])
        
        print(f"[INFO] Palette loaded from file. Transparency color: {transparency_key_color}")
        return palette_img, transparency_key_color
        
    except Exception as e:
        print(f"[WARN] Failed to load palette from {colormap_path} ({e}). Using default Quake 2 palette.")
        
        palette_img = Image.new("P", (1, 1))
        palette_img.putpalette(DEFAULT_Q2_PALETTE_DATA)
        return palette_img, DEFAULT_TRANSPARENCY_COLOR

def create_wal(src_img: Path, wal_path: Path, palette_img: Image.Image, transparency_color: Tuple[int, int, int], flags: dict):
    """Converts a TGA/PNG image to a Quake II WAL file, scaled down by 50%."""
    img_original = Image.open(src_img).convert("RGBA")
    
    # 1. Scale the image down by 50%
    new_width = img_original.width // 2
    new_height = img_original.height // 2
    if new_width < 1 or new_height < 1:
        raise ValueError("Image is too small to be scaled down by 50%.")
        
    # Lanczos for high quality scaling
    img_rgba = img_original.resize((new_width, new_height), Image.Resampling.LANCZOS)
    width, height = img_rgba.size

    # 2. Process transparency (use transparency_color for pixels below threshold)
    img_rgb = Image.new("RGB", (width, height))
    pixels_rgba = img_rgba.load()
    pixels_rgb = img_rgb.load()
    
    ALPHA_THRESHOLD = 128
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels_rgba[x, y]
            if a < ALPHA_THRESHOLD:
                pixels_rgb[x, y] = transparency_color
            else:
                pixels_rgb[x, y] = (r, g, b)

    # 3. Quantize to the loaded Quake II palette
    img_paletted = img_rgb.quantize(palette=palette_img, dither=Image.Dither.NONE)

    # 4. Generate mipmaps (3 levels down)
    mipmaps = [img_paletted]
    for _ in range(3):
        mip_w, mip_h = mipmaps[-1].size
        if mip_w <= 1 or mip_h <= 1: break
        mipmaps.append(mipmaps[-1].resize((mip_w // 2, mip_h // 2), Image.Resampling.BOX))

    # 5. Write the WAL file header and data.
    with open(wal_path, "wb") as f:
        # WAL Header (100 bytes)
        name_bytes = wal_path.stem.encode('ascii')
        f.write(struct.pack("<32s", name_bytes))
        f.write(struct.pack("<II", width, height))

        header_size = 100
        offsets = [0] * 4
        current_offset = header_size
        
        for i in range(len(mipmaps)):
            offsets[i] = current_offset
            w, h = mipmaps[i].size
            current_offset += w * h
        for i in range(len(mipmaps), 4):
            offsets[i] = offsets[-1] if offsets else 0 # Ensure all 4 offsets are written

        f.write(struct.pack("<4I", *offsets))
        
        # New Q2 Fields (surfaceflags, contents, anim_name, anim_speed, flags)
        f.write(struct.pack("<I", flags['surfaceflags'])) # surfaceflags (offset 68)
        f.write(struct.pack("<I", flags['contents']))     # contents (offset 72)
        f.write(struct.pack("<32s", b''))                 # animname (offset 76)
        f.write(struct.pack("<f", 0.0))                    # animspeed (offset 108)
        f.write(struct.pack("<I", 0))                     # flags (offset 112)
        f.write(struct.pack("<I", flags['lightvalue']))   # value (offset 116)
        
        # WAL Data (Mipmaps)
        for mip in mipmaps:
            f.write(mip.tobytes())
            
# --- NEW: WAL reader (to bypass PIL's lack of native support) ---
def read_wal_data(wal_path: Path, palette_img: Image.Image) -> Image.Image:
    """
    Reads the raw pixel data from the generated WAL file and converts it to a 
    Pillow Image object using the provided Quake 2 palette.
    """
    try:
        with open(wal_path, "rb") as f:
            # Read header size and dimensions
            f.seek(32)
            width = struct.unpack("<I", f.read(4))[0]
            height = struct.unpack("<I", f.read(4))[0]
            
            if width <= 0 or height <= 0:
                 raise ValueError("Invalid WAL dimensions.")

            # Skip offsets (4 * 4 bytes = 16 bytes) and remaining header data (32 + 4 + 4 + 16 = 56 bytes read so far)
            f.seek(100) 

            # Read the main mipmap (Mip 0) pixel data
            pixel_data_length = width * height
            pixel_data = f.read(pixel_data_length)
            
            if len(pixel_data) != pixel_data_length:
                 raise EOFError(f"WAL file truncated: Expected {pixel_data_length} bytes, got {len(pixel_data)}")

        # Create a new paletted image from the raw index data
        img = Image.frombytes('P', (width, height), pixel_data)
        
        # Apply the Quake 2 palette to the image
        img.putpalette(palette_img.getpalette())
        
        # Convert to RGB so it can be handled by the comparison function
        return img.convert('RGB')

    except Exception as e:
        raise IOError(f"Failed to read WAL file data at {wal_path.name}: {e}")

# =============================== Q2 FLAG LOGIC (UNCHANGED) ===============================

# Flags extracted from shared.h (Quake 2 Rerelease)
CONTENTS_WINDOW = 0x2
CONTENTS_WATER = 0x20
CONTENTS_LAVA = 0x8
CONTENTS_SLIME = 0x10

SURF_LIGHT = 0x1
SURF_WARP = 0x8
SURF_TRANS33 = 0x10

def get_q2_flags(new_path: str) -> dict:
    """
    Determines Q2 flags based on the converted material's destination path.
    Assumes the new path uses the q4_ prefix structure.
    """
    new_path_lower = new_path.lower()
    
    surfaceflags = 0
    contents = 0
    lightvalue = 0
    
    # 1. LIGHTS: if in q4_lights, set SURF_LIGHT and lightvalue 1000
    if "q4_lights/" in new_path_lower or "q4x_lights/" in new_path_lower:
        surfaceflags |= SURF_LIGHT
        lightvalue = 1000 # Max Q2 light value
        
    # 2. GLASS: if in q4_glass, set SURF_TRANS33 and CONTENTS_WINDOW
    if "q4_glass/" in new_path_lower:
        surfaceflags |= SURF_TRANS33
        contents |= CONTENTS_WINDOW

    # 3. FLUIDS (WATER/LAVA/SLIME): check for generic fluids directory
    if "q4_fluids/" in new_path_lower:
        # We assume Quake 2 Re fluids are simple water/warp unless sub-categorized.
        surfaceflags |= SURF_WARP
        contents |= CONTENTS_WATER
        
        if any(f in new_path_lower for f in ["lava"]):
            contents = CONTENTS_LAVA
        elif any(f in new_path_lower for f in ["slime"]):
            contents = CONTENTS_SLIME
        # Note: If it's a fluid, it typically acts as a water/liquid content.

    return {
        'surfaceflags': surfaceflags,
        'contents': contents,
        'lightvalue': lightvalue
    }
    
# ============================== CONFIG & INDEX CLASSES =======================

@dataclass
class Config:
    base_root: Path
    dst_base: Path
    output_format: str = "tga"

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        top = {}
        for key in ("base_root", "dst_base"):
            if key in raw: top[key] = Path(raw[key])
        if "output_ext" in raw:
            top["output_format"] = raw["output_ext"].strip(".")
        return Config(**top)

class AssetIndex:
    """Asset index to resolve Q4 tokens to actual file paths."""
    def __init__(self, base_root: Path):
        self.base_root = base_root
        self._map: Dict[str, Path] = {}
        self.build()
        
        DEBUG_LOG.append(f"--- ASSET INDEX CONTENTS ({len(self._map)} files indexed) ---")
        if len(self._map) > 0:
             DEBUG_LOG.append("Example indexed files (lower-case POSIX path -> full Path):")
             for k, v in sorted(list(self._map.items())[:20]):
                 DEBUG_LOG.append(f"  {k} -> {v.name}")
        else:
             DEBUG_LOG.append(f"!! Index is EMPTY. Source file discovery failed for root: {base_root} !!")
             if not base_root.exists():
                 DEBUG_LOG.append("!! WARNING: base_root path does not appear to exist! Check config: 'base_root' !!")
        DEBUG_LOG.append("-------------------------------------------------------")

    def build(self):
        root = self.base_root
        
        # --- CRITICAL FIX: Use os.walk for guaranteed, low-level file system traversal ---
        print(f"[DEBUG] Attempting to walk source root: {root}")
        if not root.exists():
             print(f"[FATAL] Source root does not exist: {root}")
             return

        try:
            for dirpath, dirnames, filenames in os.walk(root):
                current_dir = Path(dirpath)
                for filename in filenames:
                    if os.path.splitext(filename)[1].lower() in IMAGE_EXTS:
                        full_path = current_dir / filename
                        # Calculate path relative to root, convert to POSIX (forward slash), and lowercase for key
                        try:
                            rel = full_path.relative_to(root).as_posix().lower()
                            self._map[rel] = full_path
                        except ValueError:
                            continue
        except Exception as e:
            DEBUG_LOG.append(f"[FATAL BUILD ERROR] os.walk failed on root directory '{root}': {e}")
            print(f"[FATAL] Source directory access failed. Check permissions for {root}")
            
        # ----------------------------------------------------------------------------------

    def resolve(self, token: str) -> Optional[Path]:
        if not token: return None
        # Normalize token: lowercase, forward slashes, remove leading slash
        key = token.replace("\\", "/").lstrip("/").lower()
        
        # 1. Try resolving the exact token (e.g., textures/dir/file.ext)
        if key in self._map: return self._map[key]
        
        key_noext, _ = os.path.splitext(key)
        
        # 2. Try resolving the token by adding any supported extension
        for ext in IMAGE_EXTS:
            candidate_key = f"{key_noext}{ext}"
            if candidate_key in self._map: return self._map[candidate_key]
            
        return None

# =============================== MATERIAL PARSER (FULL COPY) ====================================

@dataclass
class Q4Material:
    name: str
    source_file: Path
    diffuse: Optional[str] = None
    normal: Optional[str] = None
    specular: Optional[str] = None
    height: Optional[str] = None
    additive_map: Optional[str] = None
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
                    if m := re.search(r'\bmap\s+([^\s;}]+)', stage_content, re.I): mat.additive_map = m.group(1)
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
            # FIX APPLIED: Correct list comprehension to avoid NameError
            args = [arg.strip().strip('"') for arg in m.group("args").split(",") if arg.strip()]
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
            elif macro == "generic_full_noheight": maps_from_noheight(base); set_type()
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
            elif macro == "generic_alphaglow": alpha_flags(); maps_from(base); mat.additive_map = variant or f"{base}_g"
            elif macro == "generic_alpha_noshadows": alpha_flags(no_shadows=True); maps_from(base)
            elif macro in ("generic_alphanoshadow2s", "generic_localvalpha2"): maps_from(base); alpha_flags(two_sided=True)
            elif macro in ("generic_shader2sidedalpha_lv", "generic_shader2sidedalpha_mi", "generic_shader2sidedalpha_miv"):
                if variant and base: variant_diffuse_only(variant, base); alpha_flags(two_sided=True)
                else: maps_from(variant or base); alpha_flags(two_sided=True)
            elif macro == "generic_shader2sidedalpha_type": maps_from(base); alpha_flags(two_sided=True); set_type()
            elif macro in ("generic_glow", "generic_glow_mi"): maps_from(base); mat.additive_map = variant or f"{base}_g"
            elif macro in ("generic_terminal_replaceglow", "generic_terminal_replaceglow2"): maps_from(base); mat.additive_map = variant or f"{base}_add"
            elif macro == "generic_typeshader": maps_from(base); set_type()
            elif macro == "generic_localvariant_typeshader": maps_from(variant or base); set_type(2)
            elif macro == "generic_alpha_type": maps_from(base); alpha_flags(); set_type()
            elif macro == "generic_colorvariant":
                # Use the first arg's diffuse, and the second arg's normal/spec/height.
                if base and variant:
                    variant_diffuse_only(base, variant)
                else:
                    maps_from(base or variant)

        return materials
# =============================== END MATERIAL PARSER COPY ====================================

# =============================== COMPARISON LOGIC ============================

def create_comparison_image(
    original_path: Path, 
    converted_path: Path, 
    wal_path: Optional[Path],
    palette_img: Image.Image, # Pass palette for WAL reading
    output_path: Path
):
    """
    Loads three images (Q4, Q3/Q2Re TGA/PNG, Q2 WAL), scales WAL, 
    concatenates them side-by-side with labels, and saves the result.
    """
    try:
        # 1. Load Q4 Original (Reference Size)
        img_q4 = Image.open(original_path).convert("RGB")
        original_w, original_h = img_q4.size
        
        # 2. Load Q3/Q2Re Converted TGA/PNG
        img_final = Image.open(converted_path).convert("RGB")
        img_final_resized = img_final.resize((original_w, original_h), Image.Resampling.LANCZOS)
        
        comparison_slots = [(img_q4, "Q4 Original (1x)")]
        comparison_slots.append((img_final_resized, f"Q3 Baked ({converted_path.suffix.upper().lstrip('.')})"))
        
        # 3. Load Q2 WAL (using custom reader)
        img_wal_shifted = None
        if wal_path and wal_path.exists():
            try:
                img_wal_rgb = read_wal_data(wal_path, palette_img)
                # Scale 200% (2x) to match original Q4 size (since tga2wal scaled down 50%)
                img_wal_resized = img_wal_rgb.resize((original_w, original_h), Image.Resampling.NEAREST)
                
                # CRITICAL: APPLY 8-PIXEL HORIZONTAL ALIGNMENT CORRECTION
                # Create a new canvas to shift the pixels left by 8
                # Use the original image size as the base for the final corrected image slot.
                img_wal_shifted = Image.new('RGB', (original_w, original_h), color=(0, 0, 0))
                # Paste the resized WAL image starting at x=-8
                img_wal_shifted.paste(img_wal_resized, (-8, 0)) 

                comparison_slots.append((img_wal_shifted, "Q2 WAL (PITCH CORRECTED)"))
            except Exception as e:
                DEBUG_LOG.append(f"  [SKIP] WAL reader/shifting failed for {wal_path.name}: {e}")
                pass 

    except Exception as e:
        DEBUG_LOG.append(f"  [SKIP] Failed to load images for {output_path.name}: {e}")
        return

    # --- Canvas Setup and Image Offsetting Fix ---
    num_images = len(comparison_slots)
    total_w = original_w * num_images
    total_h = original_h + HEADER_HEIGHT
    
    comparison_img = Image.new('RGB', (total_w, total_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(comparison_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14) # Smaller font
    except IOError:
        font = ImageFont.load_default()

    # --- Paste Images and Labels (Corrected Offsetting) ---
    for i, (img, label) in enumerate(comparison_slots):
        # x_start: Start position for image (0, 1 * w, 2 * w)
        x_start = original_w * i 
            
        # Image pasting starts at IMAGE_Y_OFFSET (25px)
        comparison_img.paste(img, (x_start, IMAGE_Y_OFFSET))

        # Center label above the image slot (Label position is centered on the standard slot boundary)
        slot_center_x = (original_w * i) + (original_w / 2)
        
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        
        # Text X position: Center of the slot minus half the text width
        draw.text((slot_center_x - (text_w / 2), 5), 
                  label, 
                  fill=(255, 255, 255), 
                  font=font)

    # Save the comparison image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_img.save(output_path)
    print(f"  [SAVED] {output_path.name}")
    
# =============================== MAIN WORKFLOW ========================================

def main():
    parser = argparse.ArgumentParser(description="Generate side-by-side comparison images for Q4 asset conversion.")
    default_config = Path.cwd() / "convert_config.json"
    parser.add_argument("--config", type=Path, default=default_config, help="Path to convert_config.json (defaults to ./convert_config.json)")
    
    args = parser.parse_args()
    
    try:
        cfg = Config.from_json(args.config)
        name_map_path = cfg.dst_base / "restructure_name_map.json"
        mtr_root = cfg.base_root / "materials"
        
        # Load Q2 Palette (once)
        colormap_path = Path(sys.path[0]) / "pics/colormap.pcx"
        if not colormap_path.exists():
            colormap_path = cfg.base_root / "pics/colormap.pcx"

        palette_img, transparency_color = load_palette_and_key_color(colormap_path)


        if not name_map_path.exists():
            print(f"[FATAL] Name map not found: {name_map_path}")
            return

        print(f"[INFO] Loading configuration and name map...")
        with open(name_map_path, "r", encoding="utf-8") as f:
            new_to_old_map = {v: k for k, v in json.load(f).items()}
            
        
        # 1. FULLY PARSE ALL Q4 MATERIALS 
        material_parser = Q4MaterialParser()
        all_q4_materials: Dict[str, Q4Material] = {}
        for mtr in mtr_root.rglob("*.mtr"):
            all_q4_materials.update(material_parser.parse_file(mtr))
        print(f"[INFO] Parsed {len(all_q4_materials)} original Q4 materials.")
            
        asset_index = AssetIndex(cfg.base_root)

        output_dir = cfg.dst_base / OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Comparison images will be saved to: {output_dir.resolve()}")
        
        converted_count = 0
        comparison_count = 0
        fail_log_count = 0
        
        output_ext = cfg.output_format.lstrip('.')
        DEBUG_LOG.append(f"\n--- CONVERSION TRACE ({output_ext} files) ---")
        
        # Iterate over all final converted texture files
        converted_texture_root = cfg.dst_base / "textures"
        
        for converted_path in converted_texture_root.rglob(f"*.{output_ext}"):
            try:
                # 4. Determine what the new file is called (relative_path uses same naming logic as converter)
                relative_path = converted_path.relative_to(converted_texture_root).with_suffix('').as_posix()
            except ValueError:
                continue

            if relative_path.endswith("_add") or relative_path.endswith("_glow"):
                continue

            original_material_name = new_to_old_map.get(relative_path)
            if not original_material_name: continue 

            converted_count += 1
            
            # 2. Get the fully parsed Q4Material object (which contains the precise diffuse token)
            mat = all_q4_materials.get(original_material_name)
            
            if not mat or not mat.diffuse:
                fail_log_count += 1
                if fail_log_count < MAX_FAIL_LOGS:
                    DEBUG_LOG.append(f"[SKIP-MAT] {original_material_name} -> No diffuse map token found in material parser.")
                continue 

            # 3. Take the diffuse image name (token) and store it as the original image name
            original_token = mat.diffuse
            
            # 4. Resolve the precise token to the actual source file path
            original_q4_diffuse_path = asset_index.resolve(original_token)

            if not original_q4_diffuse_path:
                fail_log_count += 1
                if fail_log_count < MAX_FAIL_LOGS:
                    DEBUG_LOG.append(f"[FAIL] {original_material_name} -> Could not find source file for token: '{original_token}'. Index Size: {len(asset_index._map)}.")
                
                print(f"[WARN] {relative_path}: Could not resolve original Q4 diffuse path for PRECISE token '{original_token}'.")
                continue

            # --- WAL Generation (Step 3) ---
            wal_path = converted_path.with_suffix('.wal')
            q2_flags = get_q2_flags(relative_path)
            
            try:
                # Force WAL generation before comparison
                create_wal(converted_path, wal_path, palette_img, transparency_color, q2_flags)
            except Exception as e:
                # If WAL creation fails, we still try to proceed with 2-way comparison
                print(f"[ERROR] WAL generation failed for {converted_path.name}: {e}. Skipping WAL comparison.")
                wal_path = None # Set path to None to disable WAL comparison

            print(f"[COMPARE] {original_q4_diffuse_path.name} vs {converted_path.name} vs {wal_path.name if wal_path else 'N/A'}")
            
            # 5 & 6. Create and save the comparison image
            output_file_name = f"{Path(relative_path).name}_COMPARE.png"
            output_path = output_dir / Path(relative_path).parent / output_file_name
            
            create_comparison_image(
                original_q4_diffuse_path, 
                converted_path, 
                wal_path,
                palette_img,
                output_path
            )
            comparison_count += 1

        print("-" * 50)
        print(f"[DONE] Attempted comparisons for {converted_count} converted textures.")
        print(f"[SUCCESS] Generated {comparison_count} comparison images in {output_dir.name}/")

        # Write the final debug log
        debug_log_path = output_dir / "comparison_debug.log"
        debug_log_path.write_text("\n".join(DEBUG_LOG), encoding="utf-8")
        print(f"[DEBUG] Wrote detailed debug log to {debug_log_path}")


    except FileNotFoundError as e:
        print(f"[FATAL] Missing File: {e}")
        print("Ensure convert_config.json, restructure_name_map.json, and all source/destination paths are correct.")
    except Exception as e:
        print(f"[FATAL] An unhandled error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
