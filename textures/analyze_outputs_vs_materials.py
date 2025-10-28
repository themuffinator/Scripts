#!/usr/bin/env python3
# analyze_outputs_vs_materials.py
from __future__ import annotations

import argparse, os, re, sys, csv, json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict, Counter

IMAGE_EXTS = [".tga", ".dds", ".png", ".jpg", ".jpeg", ".tif", ".bmp"]

# ──────────────────────────────────────────────────────────────────────────────
# Simple asset index (case-insensitive) to resolve tokens to files
# ──────────────────────────────────────────────────────────────────────────────
class AssetIndex:
    def __init__(self, base_root: Path):
        self.base = base_root
        self.map: Dict[str, Path] = {}

    def build(self):
        for ext in IMAGE_EXTS:
            for p in self.base.rglob(f"*{ext}"):
                key = p.relative_to(self.base).as_posix().lower()
                self.map[key] = p

    def resolve(self, token: Optional[str]) -> Optional[Path]:
        if not token:
            return None
        key = token.replace("\\", "/").lstrip("/").lower()
        # direct
        p = self.map.get(key)
        if p:
            return p
        # try common extensions
        if not any(key.endswith(e) for e in IMAGE_EXTS):
            for ext in IMAGE_EXTS:
                p = self.map.get((key + ext))
                if p: return p
        # last resort: path under base
        guess = self.base / key
        if guess.exists(): return guess
        for ext in IMAGE_EXTS:
            g = self.base / (key + ext)
            if g.exists(): return g
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Tiny image sniffers to detect size and alpha (best-effort; robust for PNG/TGA)
# ──────────────────────────────────────────────────────────────────────────────
def sniff_png_alpha(fp):
    import struct
    sig = fp.read(8)
    if sig != b"\x89PNG\r\n\x1a\n": return None
    ln = fp.read(4); tp = fp.read(4)
    if tp != b'IHDR': return None
    w, h = struct.unpack(">II", fp.read(8))
    bit_depth = fp.read(1)
    color_type = fp.read(1)
    # color_type: 6=RGBA, 4=GA, 2=RGB, 0=G
    has_alpha = color_type in (b"\x06", b"\x04")
    return ("PNG", int(w), int(h), bool(has_alpha))

def sniff_tga_alpha(fp):
    import struct
    # TGA: width@12, height@14 (LE); bpp@16; image descriptor @17 (low 4 bits = alpha bits)
    fp.seek(12); w, h = struct.unpack("<HH", fp.read(4))
    fp.seek(16); bpp = ord(fp.read(1))
    img_desc = ord(fp.read(1))
    alpha_bits = img_desc & 0x0F
    has_alpha = alpha_bits > 0 or bpp in (32,)
    return ("TGA", int(w), int(h), bool(has_alpha))

def sniff_dds(fp):
    import struct
    if fp.read(4) != b"DDS ": return None
    fp.seek(12); h, w = struct.unpack("<II", fp.read(8))
    # Alpha detect is complex; leave unknown
    return ("DDS", int(w), int(h), None)

def sniff_jpg(fp):
    if fp.read(2) != b"\xFF\xD8": return None
    return ("JPG", None, None, False)

def sniff_image(path: Path) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[bool]]:
    try:
        with open(path, "rb") as f:
            head = f.read(12); f.seek(0)
            if head.startswith(b"\x89PNG"): return sniff_png_alpha(f)
            if head.startswith(b"DDS "):    return sniff_dds(f)
            if head[:2] == b"\xFF\xD8":     return sniff_jpg(f)
            return sniff_tga_alpha(f)       # fallback assume TGA-like
    except Exception:
        return (None, None, None, None)

# ──────────────────────────────────────────────────────────────────────────────
# Q4 material parser (blocks + guide macros). Lightweight (analysis only).
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class MatRec:
    name: str
    source_file: Path
    # tokens
    diffuse: Optional[str] = None
    normal: Optional[str] = None
    specular: Optional[str] = None
    height: Optional[str] = None
    height_scale: float = 1.0
    # flags
    twoSided: bool = False
    translucent: bool = False
    noShadows: bool = False
    # macro
    is_guide: bool = False
    guide_macro: Optional[str] = None
    # resolution (to be filled)
    d_path: Optional[Path] = None
    n_path: Optional[Path] = None
    s_path: Optional[Path] = None
    h_path: Optional[Path] = None
    d_info: Tuple[Optional[str], Optional[int], Optional[int], Optional[bool]] = (None, None, None, None)

class Parser:
    def __init__(self):
        self.re_comment_line = re.compile(r"//.*?$", re.MULTILINE)
        self.re_comment_block = re.compile(r"/\*.*?\*/", re.DOTALL)
        self.re_block_header = re.compile(r"^\s*(textures/[^\s{]+)\s*\{\s*$", re.I|re.M)
        self.re_guide = re.compile(r'^\s*guide\s+(\S+)\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)\s*$', re.I|re.M)

    def strip_comments(self, text: str) -> str:
        return self.re_comment_line.sub("", self.re_comment_block.sub("", text))

    def _iter_blocks(self, text: str) -> List[Tuple[str,str]]:
        out = []
        for m in self.re_block_header.finditer(text):
            name = m.group(1)
            i = m.end(); depth = 1; j = i
            while j < len(text) and depth > 0:
                if text[j] == "{": depth += 1
                elif text[j] == "}": depth -= 1
                j += 1
            out.append((name, text[i:j-1]))
        return out

    def parse_file(self, path: Path) -> Dict[str, MatRec]:
        if not path.exists(): return {}
        text = self.strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
        recs: Dict[str, MatRec] = {}

        # blocks
        for name, block in self._iter_blocks(text):
            rec = MatRec(name=name, source_file=path)
            if re.search(r"\btwosided\b", block, re.I): rec.twoSided = True
            if re.search(r"\btranslucent\b", block, re.I): rec.translucent = True
            if re.search(r"\bnoshadows\b", block, re.I): rec.noShadows = True

            m = re.search(r"\bdiffusemap\s+([^\s}]+)", block, re.I)
            if m: rec.diffuse = m.group(1)
            m = re.search(r"\bspecularmap\s+([^\s}]+)", block, re.I)
            if m: rec.specular = m.group(1)
            # bumpMap addnormals ( n , heightmap ( h , s ) )
            bn = re.search(r"\bbumpmap\s+addnormals\s*\(\s*([^\s,()]+)\s*,\s*heightmap\s*\(\s*([^\s,()]+)\s*,\s*([0-9.+\-]+)\s*\)\s*\)", block, re.I)
            if bn:
                rec.normal = bn.group(1); rec.height = bn.group(2)
                try: rec.height_scale = float(bn.group(3))
                except: pass
            else:
                m = re.search(r"\bbumpmap\s+([^\s}]+)", block, re.I)
                if m: rec.normal = m.group(1)
                hm = re.search(r"\bheightmap\s*\(\s*([^\s,()]+)\s*,\s*([0-9.+\-]+)\s*\)", block, re.I)
                if hm:
                    rec.height = hm.group(1)
                    try: rec.height_scale = float(hm.group(2))
                    except: pass

            if not rec.diffuse:
                m = re.search(r"(?m)^\s*map\s+([^\s}]+)", block, re.I)
                if m: rec.diffuse = m.group(1)

            recs[name] = rec

        # guide macros (fill tokens/flags as in converter logic—minimal here)
        for mat, macro, args_raw in self.re_guide.findall(text):
            if not mat.lower().startswith("textures/"): continue
            args = [a.strip().strip('"') for a in args_raw.split(",") if a.strip()]
            rec = recs.setdefault(mat, MatRec(name=mat, source_file=path))
            rec.is_guide = True
            rec.guide_macro = macro.lower().strip()

            def norm(t: str) -> str:
                if not t: return t
                t = t.replace("\\", "/").lstrip("/")
                if not t.lower().startswith(("textures/","models/","gfx/")):
                    t = "textures/" + t
                return t
            base    = norm(args[0]) if args else None
            variant = norm(args[1]) if len(args) >= 2 else None

            def maps_from(prefix: str):
                rec.diffuse  = f"{prefix}_d"
                rec.normal   = f"{prefix}_local"
                rec.specular = f"{prefix}_s"
                rec.height   = f"{prefix}_h"

            m = rec.guide_macro
            if m in ("generic_materialimageshader","generic_shader"):
                if base: maps_from(base)
            elif m == "generic_full_noheight":
                if base: maps_from(base); rec.height = None
            elif m in ("generic_localvariant", "generic_colorvariant"):
                maps_from(variant or base or "")
            elif m in ("generic_localvariant_mi",):
                if variant and base:
                    rec.diffuse = f"{variant}_d"
                    rec.normal   = f"{base}_local"
                    rec.specular = f"{base}_s"
                    rec.height   = f"{base}_h"
                elif variant:
                    rec.diffuse = f"{variant}_d"
            elif m in ("depthspriteadditive","depthspritealphablend"):
                if base: rec.diffuse = base
                rec.translucent = True; rec.twoSided = True
            elif m.startswith("icon_"):
                if base: rec.diffuse = base
                rec.translucent = True; rec.twoSided = True; rec.noShadows = True
            else:
                # plenty more macros exist; for analysis we at least set diffuse
                if base and not rec.diffuse:
                    rec.diffuse = f"{base}_d"

        return recs

# ──────────────────────────────────────────────────────────────────────────────
# Log parsing (reads the per-material .log emitted by your converter)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class LogInfo:
    decision: Optional[str] = None    # COPY-FORCE / COPY / BAKE / SKIP ...
    error: Optional[str] = None       # last [ERROR] line
    blender_rc: Optional[int] = None

def parse_log(log_path: Path) -> LogInfo:
    info = LogInfo()
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return info
    for ln in lines:
        if ln.startswith("[DECISION]"):
            info.decision = ln.replace("[DECISION]","").strip()
        if ln.startswith("[ERROR]"):
            info.error = ln
        if ln.startswith("RETURN CODE:"):
            try:
                info.blender_rc = int(ln.split(":")[1].strip())
            except:
                pass
    return info

# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Analyze patterns: materials vs produced images")
    ap.add_argument("--base-root", required=True, help="Q4 base root (contains materials/, textures/)")
    ap.add_argument("--dst-base", required=True, help="Output root used by converter")
    ap.add_argument("--ext", default="png", choices=["png","tga"], help="Output image extension")
    ap.add_argument("--report", default="analysis_report.md", help="Markdown report output path")
    ap.add_argument("--csv", default="analysis_rows.csv", help="CSV rows output path")
    args = ap.parse_args()

    base_root = Path(args.base_root)
    dst_base  = Path(args.dst_base)
    ext = args.ext.lower()

    # Build index
    index = AssetIndex(base_root)
    index.build()

    # Parse all .mtr
    parser = Parser()
    mat_dir = base_root / "materials"
    mats: Dict[str, MatRec] = {}
    for mtr in mat_dir.rglob("*.mtr"):
        mats.update(parser.parse_file(mtr))

    # Associate to outputs + logs
    rows = []
    for name, rec in sorted(mats.items()):
        q3_noext = name.replace("\\","/")
        out_img  = dst_base / (q3_noext + f".{ext}")
        log_path = out_img.with_suffix(out_img.suffix + ".log")

        # resolve tokens to actual files
        rec.d_path = index.resolve(rec.diffuse) if rec.diffuse else None
        rec.n_path = index.resolve(rec.normal)  if rec.normal  else None
        rec.s_path = index.resolve(rec.specular)if rec.specular else None
        rec.h_path = index.resolve(rec.height)  if rec.height  else None
        if rec.d_path and rec.d_path.exists():
            rec.d_info = sniff_image(rec.d_path)

        out_exists = out_img.exists()
        info = parse_log(log_path) if log_path.exists() else LogInfo()

        # Feature engineering
        have_n = bool(rec.n_path and rec.n_path.exists())
        have_s = bool(rec.s_path and rec.s_path.exists())
        have_h = bool(rec.h_path and rec.h_path.exists())
        maps_count = int(bool(rec.d_path))+int(have_n)+int(have_s)+int(have_h)

        rows.append({
            "material": name,
            "macro": rec.guide_macro or "",
            "is_guide": int(rec.is_guide),
            "twoSided": int(rec.twoSided),
            "translucent": int(rec.translucent),
            "noShadows": int(rec.noShadows),
            "diffuse_token": rec.diffuse or "",
            "normal_token": rec.normal or "",
            "specular_token": rec.specular or "",
            "height_token": rec.height or "",
            "diffuse_resolved": str(rec.d_path) if rec.d_path else "",
            "normal_resolved":  str(rec.n_path) if rec.n_path else "",
            "specular_resolved":str(rec.s_path) if rec.s_path else "",
            "height_resolved":  str(rec.h_path) if rec.h_path else "",
            "diffuse_fmt": rec.d_info[0] if rec.d_info else "",
            "diffuse_w": rec.d_info[1] if rec.d_info else "",
            "diffuse_h": rec.d_info[2] if rec.d_info else "",
            "diffuse_has_alpha": int(bool(rec.d_info[3])) if rec.d_info[3] is not None else "",
            "maps_count": maps_count,
            "out_image": str(out_img),
            "out_exists": int(out_exists),
            "log_exists": int(log_path.exists()),
            "decision": info.decision or "",
            "error": info.error or "",
            "blender_rc": info.blender_rc if info.blender_rc is not None else "",
        })

    # Write CSV
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate patterns
    def rate_by(key):
        grp = defaultdict(lambda: [0,0])  # [total, success]
        for r in rows:
            k = r[key]
            grp[k][0] += 1
            grp[k][1] += int(r["out_exists"])
        stats = []
        for k,(tot,succ) in grp.items():
            rate = (succ/tot) if tot else 0.0
            stats.append((k, tot, succ, rate))
        stats.sort(key=lambda x: (-x[3], -x[1]))  # by rate desc, then volume
        return stats

    by_macro = rate_by("macro")
    by_trans = rate_by("translucent")
    by_two   = rate_by("twoSided")
    by_maps  = rate_by("maps_count")
    by_alpha = rate_by("diffuse_has_alpha")

    # Common failure reasons from logs
    fail_reason = Counter()
    for r in rows:
        if not r["out_exists"]:
            reason = "no_log"
            if r["log_exists"]:
                if r["error"]:
                    reason = r["error"]
                elif r["decision"]:
                    reason = f"decision:{r['decision']}"
                else:
                    reason = "log_without_reason"
            fail_reason[reason] += 1
    top_fail = fail_reason.most_common(12)

    # Markdown report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        W = f.write
        total = len(rows); produced = sum(r["out_exists"] for r in rows)
        W(f"# Material → Output Pattern Report\n\n")
        W(f"- Base root: `{base_root}`\n")
        W(f"- Output root: `{dst_base}`\n")
        W(f"- Materials analyzed: **{total}**\n")
        W(f"- Outputs present: **{produced}** ({produced/total*100:.1f}%)\n")
        W(f"- Rows CSV: `{csv_path}`\n\n")

        def dump_table(title, stats, keyname):
            W(f"## {title}\n\n")
            W(f"| {keyname} | count | outputs | success rate |\n|---|---:|---:|---:|\n")
            for k,tot,succ,rate in stats:
                label = "(none)" if (k=="" or k is None) else str(k)
                W(f"| {label} | {tot} | {succ} | {rate*100:.1f}% |\n")
            W("\n")

        dump_table("Success by guide macro", by_macro, "macro")
        dump_table("Success by translucency", by_trans, "translucent(0/1)")
        dump_table("Success by two-sided", by_two, "twoSided(0/1)")
        dump_table("Success by number of maps resolved (d+n+s+h)", by_maps, "maps_count")
        dump_table("Success by diffuse alpha presence", by_alpha, "diffuse_has_alpha(0/1)")

        W("## Top failure reasons (from logs)\n\n")
        if top_fail:
            W("| reason | count |\n|---|---:|\n")
            for reason, cnt in top_fail:
                trunc = reason.strip()
                if len(trunc) > 100: trunc = trunc[:97] + "..."
                W(f"| {trunc} | {cnt} |\n")
            W("\n")
        else:
            W("_No explicit reasons found in logs._\n\n")

        # Quick “links”
        W("## Quick correlations\n\n")
        W("- **Low success by macro**: look for macros near the bottom of the 'Success by guide macro' table.\n")
        W("- **Translucent/two-sided** materials often fail to bake; ensure they are copied (not baked) or made OPAQUE during bake.\n")
        W("- **Diffuse with alpha** correlates with missing/black outputs when used in COMBINED bakes without forcing OPAQUE.\n")
        W("- **maps_count=1** (diffuse only) should copy, not bake. If these are missing, check converter copy branch and path mapping.\n")

    print(f"[OK] Wrote report: {report_path}")
    print(f"[OK] Wrote rows CSV: {csv_path}")

if __name__ == "__main__":
    sys.exit(main())
