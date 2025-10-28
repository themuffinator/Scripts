import os
import re
import argparse
import struct
from collections import defaultdict, OrderedDict

# ---------------- WAL headers (100 / 132) ----------------
_WAL_FMT_100 = "<32sII4I32sIII"
_WAL_SIZE_100 = struct.calcsize(_WAL_FMT_100)
_WAL_FMT_132 = "<32sII4I32s32sIII"
_WAL_SIZE_132 = struct.calcsize(_WAL_FMT_132)

def _clean_cstr(raw):
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore")

def _validate_offsets(width, height, offs, file_size, header_len):
    if width <= 0 or height <= 0 or width > 16384 or height > 16384:
        return False
    o0 = offs[0]
    if o0 == 0:
        return True
    return header_len <= o0 < file_size

def try_read_wal_header(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size >= _WAL_SIZE_132:
            data = f.read(_WAL_SIZE_132)
            t = struct.unpack(_WAL_FMT_132, data)
            if _validate_offsets(t[1], t[2], t[3:7], size, _WAL_SIZE_132):
                return {
                    "name": _clean_cstr(t[0]), "width": t[1], "height": t[2],
                    "offsets": t[3:7], "next_name": _clean_cstr(t[7]),
                    "anim_name": _clean_cstr(t[8]), "flags": t[9],
                    "contents": t[10], "value": t[11], "header_len": 132
                }
        if size >= _WAL_SIZE_100:
            f.seek(0)
            data = f.read(_WAL_SIZE_100)
            t = struct.unpack(_WAL_FMT_100, data)
            if _validate_offsets(t[1], t[2], t[3:7], size, _WAL_SIZE_100):
                return {
                    "name": _clean_cstr(t[0]), "width": t[1], "height": t[2],
                    "offsets": t[3:7], "next_name": "", "anim_name": _clean_cstr(t[7]),
                    "flags": t[8], "contents": t[9], "value": t[10], "header_len": 100
                }
    return None

# ---------------- .mat parser (optional) ----------------
def parse_mat_file(mat_path):
    mat_type = None
    numeric = {}
    try:
        with open(mat_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("//"):
                    continue
                m_type = re.match(r"['\"]?([A-Za-z0-9_]+)['\"]?\s*:\s*$", line)
                if m_type:
                    mat_type = m_type.group(1).lower()
                    continue
                m_num = re.match(r"([A-Za-z_]+)\s*=\s*(-?\d+)\s*$", line)
                if m_num:
                    numeric[m_num.group(1).lower()] = int(m_num.group(2))
    except FileNotFoundError:
        pass
    return mat_type, numeric

# ---------------- .mat -> surfaceparms ----------------
def surfaceparms_from_mat_type(surface_type):
    if not surface_type:
        return []
    mapping = {
        "glass":  ["surfaceparm trans", "surfaceparm nonsolid", "surfaceparm nolightmap"],
        "carpet": ["surfaceparm nomarks"],
        "snow":   ["surfaceparm slick"],
        "splash": ["surfaceparm slick"],
        "wood":   ["surfaceparm woodsteps"],
        "grass":  ["surfaceparm grasssteps"],
        "tile":   ["surfaceparm metalsteps"],
        "flesh":  ["surfaceparm flesh"],
        "clank":  ["surfaceparm metalsteps"],
        "mech":   ["surfaceparm metalsteps"],
        "energy": ["surfaceparm nodamage"],
        "junk":   ["surfaceparm nomarks"],
        "step":   ["surfaceparm nodamage"],
        "sky":    ["surfaceparm sky"],
        "water":  ["surfaceparm water", "surfaceparm trans"],
        "slime":  ["surfaceparm slime", "surfaceparm trans"],
        "lava":   ["surfaceparm lava", "surfaceparm trans"],
    }
    return mapping.get(surface_type, [])

# ---------------- common.shader helpers ----------------
COMMON_NAME_KEYS = [
    "clip","weapclip","botclip","hint","skip","caulk","nodraw","origin",
    "trigger","areaportal","fog","water","slime","lava","sky","slick","monclip"
]

COMMON_NAME_TO_SHADER = {
    "clip":"textures/common/clip",
    "weapclip":"textures/common/weapclip",
    "botclip":"textures/common/botclip",
    "hint":"textures/common/hint",
    "skip":"textures/common/skip",
    "caulk":"textures/common/caulk",
    "nodraw":"textures/common/nodraw",
    "origin":"textures/common/origin",
    "trigger":"textures/common/trigger",
    "areaportal":"textures/common/areaportal",
    "fog":"textures/common/fog",
    "water":"textures/common/water",
    "slime":"textures/common/slime",
    "lava":"textures/common/lava",
    "sky":"textures/common/sky",
    "slick":"textures/common/slick",
    "monclip":"textures/common/monclip",
}

def match_common_key(rel_noext):
    bn = os.path.basename(rel_noext).lower()
    for key in COMMON_NAME_KEYS:
        if bn == key or bn.startswith(key) or bn.endswith("_" + key):
            return key
    return None

def load_common_shader_blocks(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    blocks = {}
    i = 0
    n = len(txt)
    while i < n:
        while i < n and txt[i].isspace():
            i += 1
        if i >= n: break
        j = i
        while j < n and txt[j] not in "{\n\r":
            j += 1
        name = txt[i:j].strip()
        if not name or j >= n or txt[j] != "{":
            i = j + 1
            continue
        brace = 0
        k = j
        while k < n:
            c = txt[k]
            if c == "{": brace += 1
            elif c == "}":
                brace -= 1
                if brace == 0:
                    k += 1
                    break
            k += 1
        blocks[name] = txt[j:k].strip()
        i = k
    return blocks

# ---------------- +X name anim grouping ----------------
_ANIM_INDEX_MAP = {str(i): i for i in range(10)}
_ANIM_INDEX_MAP.update({chr(ord('A')+i): 10+i for i in range(10)})
_ANIM_INDEX_MAP.update({chr(ord('a')+i): 10+i for i in range(10)})
_anim_name_re = re.compile(r"^[\+\-]([0-9A-Ja-j])(.+)$")

def split_anim_name(base_name):
    m = _anim_name_re.match(base_name)
    if not m: return None
    idx_ch = m.group(1)
    tail = m.group(2)
    if idx_ch in _ANIM_INDEX_MAP:
        return (_ANIM_INDEX_MAP[idx_ch], tail)
    return None

# ---------------- glow helpers ----------------
def find_glow_any(base_fs_noext, base_rel_noext):
    for ext in (".png", ".tga", ".jpg"):
        cand = base_fs_noext + "_glow" + ext
        if os.path.isfile(cand):
            return base_rel_noext + "_glow" + ext
    return None

# ---------------- build anim sets ----------------
def build_anim_sets(entries_by_dir):
    consumed = set()
    sets = []

    # WAL-linked chains
    for dir_rel, entries in entries_by_dir.items():
        by_name = {e["base_name"]: e for e in entries if e["wal_ok"]}
        starts = set()
        for e in entries:
            a = e.get("wal", {}).get("anim_name", "")
            if a and a in by_name:
                starts.add(a)
        for s in sorted(starts):
            if (dir_rel, s) in consumed: continue
            chain, seen, cur = [], set(), s
            while True:
                if cur not in by_name: break
                e = by_name[cur]
                k = (dir_rel, e["base_name"])
                if k in consumed: break
                chain.append(e); seen.add(cur)
                nxt = e.get("wal", {}).get("next_name", "")
                if not nxt or nxt == s or nxt in seen: break
                cur = nxt
            if len(chain) > 1:
                for e in chain: consumed.add((dir_rel, e["base_name"]))
                sets.append(chain)

    # +X sets
    for dir_rel, entries in entries_by_dir.items():
        buckets = defaultdict(list)
        for e in entries:
            if (dir_rel, e["base_name"]) in consumed: continue
            sp = split_anim_name(e["base_name"])
            if sp:
                idx, tail = sp
                buckets[tail].append((idx, e))
        for tail, items in buckets.items():
            if len(items) < 2: continue
            items.sort(key=lambda x: x[0])
            chain = [e for _, e in items]
            for e in chain: consumed.add((dir_rel, e["base_name"]))
            sets.append(chain)

    singles = []
    for dir_rel, entries in entries_by_dir.items():
        for e in entries:
            if (dir_rel, e["base_name"]) not in consumed:
                singles.append(e)

    return sets, singles

def top_level_group(rel_noext):
    parts = rel_noext.split("/")
    return parts[0] if len(parts) > 1 else "baseq2"

# ---------------- special hard overrides ----------------
def try_emit_hard_override(lines, shader_noext, base_name, dir_rel):
    bn = base_name.lower()
    if bn == "clip":
        lines.append("    qer_trans 0.40")
        lines.append("    surfaceparm nodraw")
        lines.append("    surfaceparm nolightmap")
        lines.append("    surfaceparm nonsolid")
        lines.append("    surfaceparm trans")
        lines.append("    surfaceparm nomarks")
        lines.append("    surfaceparm noimpact")
        lines.append("    surfaceparm playerclip")
        return True
    if bn == "hint":
        lines.append("    qer_nocarve")
        lines.append("    qer_trans 0.30")
        lines.append("    surfaceparm nodraw")
        lines.append("    surfaceparm nonsolid")
        lines.append("    surfaceparm structural")
        lines.append("    surfaceparm trans")
        lines.append("    surfaceparm noimpact")
        lines.append("    surfaceparm hint")
        return True
    if bn == "trigger":
        lines.append("    qer_trans 0.50")
        lines.append("    qer_nocarve")
        lines.append("    surfaceparm nodraw")
        return True
    if bn == "monclip":
        lines.append("    qer_trans 0.40")
        lines.append("    surfaceparm nodraw")
        lines.append("    surfaceparm nolightmap")
        lines.append("    surfaceparm nonsolid")
        lines.append("    surfaceparm trans")
        lines.append("    surfaceparm nomarks")
        lines.append("    surfaceparm noimpact")
        lines.append("    surfaceparm monclip")
        return True
    if bn in ("sky1", "sky2"):
        # Build image paths from this dir
        prefix = ("textures/" + dir_rel + "/") if dir_rel else "textures/"
        # Exact sky block as requested
        lines.append("    surfaceParm noImpact")
        lines.append("    surfaceParm noLightmap")
        lines.append("    surfaceParm sky")
        lines.append("    q3map_globalTexture")
        lines.append("    q3map_sunExt .9216 .5608 .0745 80 300 60 2 16")
        lines.append("    q3map_surfaceLight 80")
        lines.append("    qer_editorImage " + prefix + "sky1.tga")
        lines.append("    q3map_lightImage " + prefix + "sky2.tga")
        lines.append("    skyParms env/unit1 768 -")
        return True
    return False

# ---------------- main generation ----------------
def generate_grouped_shaders(input_dir, anim_fps, common_shader_path):
    root_abs = os.path.abspath(input_dir)
    common_blocks = load_common_shader_blocks(common_shader_path)

    entries = []
    for r, _, files in os.walk(root_abs):
        for fn in files:
            if not fn.lower().endswith(".wal"):
                continue
            wal_fs = os.path.join(r, fn)
            wal = try_read_wal_header(wal_fs)
            rel_fs = os.path.relpath(wal_fs, root_abs).replace("\\", "/")
            rel_noext = os.path.splitext(rel_fs)[0]
            dir_rel = os.path.dirname(rel_noext)
            base_name = os.path.basename(rel_noext)
            mat_fs = os.path.splitext(wal_fs)[0] + ".mat"
            mat_type, mat_nums = parse_mat_file(mat_fs)
            entries.append({
                "rel_noext": rel_noext,
                "dir_rel": dir_rel,
                "base_name": base_name,
                "wal": wal if wal else {},
                "wal_ok": wal is not None,
                "mat_type": mat_type,
                "mat_nums": mat_nums,
                "wal_fs_noext": os.path.splitext(wal_fs)[0],
            })

    if not entries:
        raise SystemExit("No .wal files found.")

    by_dir = defaultdict(list)
    for e in entries:
        by_dir[e["dir_rel"]].append(e)
    anim_sets, singles = build_anim_sets(by_dir)

    shader_entries = []
    for chain in anim_sets:
        shader_entries.append({"kind": "anim", "frames": chain, "owner": chain[0]})
    for e in singles:
        shader_entries.append({"kind": "single", "frames": [e], "owner": e})

    grouped = defaultdict(list)
    for sh in shader_entries:
        grouped[top_level_group(sh["owner"]["rel_noext"])].append(sh)

    out_dir = os.path.join(os.getcwd(), "scripts")
    os.makedirs(out_dir, exist_ok=True)

    for top, shaders in grouped.items():
        out_path = os.path.join(out_dir, top + ".shader")
        lines = []
        lines.append("/*")
        lines.append("============================================================")
        lines.append(" Generated from: " + input_dir)
        lines.append(" Group: " + top)
        lines.append(" Notes:")
        lines.append(" - Hard overrides applied for clip/hint/trigger/monclip/sky1/sky2.")
        lines.append(" - Anim textures grouped by WAL links and +X name scheme.")
        lines.append(" - Glow animMap emitted only if every frame has a glow image.")
        lines.append(" - Default animMap rate set to 10.0 fps (override with --anim-fps).")
        lines.append(" - q3map_surfacelight emitted when wal.value > 0 (non-hard overrides).")
        lines.append(" - Common textures may be cloned from common.shader when not hard-overridden.")
        lines.append("============================================================")
        lines.append("*/")
        lines.append("")

        for sh in shaders:
            owner = sh["owner"]
            shader_noext = owner["rel_noext"]
            base_name = owner["base_name"]
            dir_rel = owner["dir_rel"]
            wal = owner["wal"]
            wal_ok = owner["wal_ok"]
            mat_type = owner["mat_type"]

            # Header
            lines.append("textures/" + shader_noext)
            lines.append("{")

            # 1) Hard overrides (exact output only)
            if try_emit_hard_override(lines, shader_noext, base_name, dir_rel):
                lines.append("}")
                lines.append("")
                continue

            # 2) common.shader clone (if available and matched)
            common_key = match_common_key(shader_noext)
            if common_key:
                ref_name = COMMON_NAME_TO_SHADER.get(common_key)
                body = common_blocks.get(ref_name)
                if body:
                    # Clone body exactly
                    lines.append(body)
                    lines.append("")
                    continue

            # 3) Normal shader build
            lines.append("    qer_editorimage textures/" + shader_noext + ".tga")

            if wal_ok and wal.get("value", 0) > 0:
                lines.append("    q3map_surfacelight " + str(wal["value"]))

            # surfaceparms from .mat type only (no contents guessing)
            sp_order = OrderedDict()
            for p in surfaceparms_from_mat_type(mat_type):
                sp_order[p] = True
            for p in list(sp_order.keys()):
                lines.append("    " + p)

            if wal_ok:
                lines.append("    // wal.flags    = " + str(wal.get("flags", 0)))
                lines.append("    // wal.contents = " + str(wal.get("contents", 0)))
                lines.append("    // wal.value    = " + str(wal.get("value", 0)))
            else:
                lines.append("    // wal: header not parsed")

            if owner["mat_nums"]:
                for k, v in owner["mat_nums"].items():
                    lines.append("    // mat." + k + " = " + str(v))

            # Stages
            if sh["kind"] == "anim":
                frames = ["textures/" + e["rel_noext"] + ".tga" for e in sh["frames"]]
                lines.append("    {")
                lines.append("        animMap " + str(float(anim_fps)) + " " + " ".join(frames))
                lines.append("    }")
                glow_frames = []
                all_have_glow = True
                for e in sh["frames"]:
                    g = find_glow_any(e["wal_fs_noext"], e["rel_noext"])
                    if g: glow_frames.append("textures/" + g)
                    else:
                        all_have_glow = False
                        break
                if all_have_glow and glow_frames:
                    lines.append("    {")
                    lines.append("        animMap " + str(float(anim_fps)) + " " + " ".join(glow_frames))
                    lines.append("        blendFunc add")
                    lines.append("        rgbGen identity")
                    lines.append("    }")
            else:
                lines.append("    {")
                lines.append("        map textures/" + shader_noext + ".tga")
                lines.append("    }")
                g = find_glow_any(owner["wal_fs_noext"], owner["rel_noext"])
                if g:
                    lines.append("    {")
                    lines.append("        map textures/" + g)
                    lines.append("        blendFunc add")
                    lines.append("        rgbGen identity")
                    lines.append("    }")

            lines.append("}")
            lines.append("")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("Wrote: " + out_path)

def main():
    ap = argparse.ArgumentParser(description="Generate Quake 3 shaders from Quake 2 WAL/.mat files with hard overrides and common.shader cloning.")
    ap.add_argument("input_dir", help="Root textures directory to scan recursively")
    ap.add_argument("--anim-fps", type=float, default=10.0, help="animMap frames per second (default 10.0)")
    ap.add_argument("--common", type=str, default="common.shader", help="Path to common.shader for cloning common blocks")
    args = ap.parse_args()
    generate_grouped_shaders(args.input_dir, args.anim_fps, args.common)

if __name__ == "__main__":
    main()
