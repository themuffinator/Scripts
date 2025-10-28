#!/usr/bin/env python3

"""
compare_dll_structure.py

Compare the PE/COFF structure of two Windows DLLs (or EXEs). Focuses on headers,
sections, imports, exports, resources, and several other structural features.

Usage:
  python compare_dll_structure.py A.dll B.dll [--json out.json] [--no-color]

Exit codes:
  0 = No structural differences
  1 = Differences found
  2 = Usage or runtime error

Requires:
  - Python 3.8+
  - pefile (pip install pefile)  -- recommended for full parsing

Notes:
  - If pefile is not installed, a limited built-in parser will run and check
    only coarse headers and sections. For deep comparison, install pefile.

Author: ChatGPT
License: MIT
"""
import argparse
import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Try to import pefile; if not available, degrade gracefully
try:
    import pefile  # type: ignore
    HAS_PEFILE = True
except Exception:
    pefile = None  # type: ignore
    HAS_PEFILE = False


def md5_bytes(data: bytes) -> str:
    m = hashlib.md5()
    m.update(data)
    return m.hexdigest()


def calc_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    from math import log2
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    entropy = 0.0
    length = len(data)
    for c in counts:
        if c:
            p = c / length
            entropy -= p * log2(p)
    return entropy


@dataclass
class SectionInfo:
    name: str
    virtual_size: int
    virtual_address: int
    raw_size: int
    raw_ptr: int
    characteristics: int
    md5: Optional[str] = None
    entropy: Optional[float] = None


@dataclass
class PEInfo:
    path: str
    machine: Optional[int] = None
    characteristics: Optional[int] = None
    subsystem: Optional[int] = None
    dll_characteristics: Optional[int] = None
    timestamp: Optional[int] = None
    entry_point: Optional[int] = None
    image_base: Optional[int] = None
    section_alignment: Optional[int] = None
    file_alignment: Optional[int] = None
    number_of_sections: Optional[int] = None

    sections: List[SectionInfo] = None  # type: ignore
    imports: Dict[str, List[str]] = None  # type: ignore
    exports: List[str] = None  # type: ignore
    resources: List[str] = None  # type: ignore
    has_tls: bool = False
    tls_callbacks: List[int] = None  # type: ignore
    reloc_count: Optional[int] = None
    debug_types: List[str] = None  # type: ignore


def _read_c_string(b: bytes) -> str:
    try:
        end = b.index(0)
        return b[:end].decode("ascii", errors="ignore")
    except ValueError:
        return b.decode("ascii", errors="ignore")
    except Exception:
        return ""


def parse_with_pefile(path: str) -> PEInfo:
    pe = pefile.PE(path, fast_load=False)
    info = PEInfo(path=path)
    # COFF / File headers
    info.machine = pe.FILE_HEADER.Machine
    info.characteristics = pe.FILE_HEADER.Characteristics
    info.timestamp = pe.FILE_HEADER.TimeDateStamp
    # Optional header
    opt = pe.OPTIONAL_HEADER
    info.subsystem = getattr(opt, "Subsystem", None)
    info.dll_characteristics = getattr(opt, "DllCharacteristics", None)
    info.entry_point = getattr(opt, "AddressOfEntryPoint", None)
    info.image_base = getattr(opt, "ImageBase", None)
    info.section_alignment = getattr(opt, "SectionAlignment", None)
    info.file_alignment = getattr(opt, "FileAlignment", None)
    info.number_of_sections = pe.FILE_HEADER.NumberOfSections

    # Sections
    info.sections = []
    with open(path, "rb") as f:
        raw = f.read()
    for s in pe.sections:
        name = _read_c_string(s.Name)
        raw_slice = raw[s.PointerToRawData:s.PointerToRawData + s.SizeOfRawData] if s.PointerToRawData and s.SizeOfRawData else b""
        info.sections.append(SectionInfo(
            name=name,
            virtual_size=s.Misc_VirtualSize,
            virtual_address=s.VirtualAddress,
            raw_size=s.SizeOfRawData,
            raw_ptr=s.PointerToRawData,
            characteristics=s.Characteristics,
            md5=md5_bytes(raw_slice) if raw_slice else None,
            entropy=calc_entropy(raw_slice) if raw_slice else None
        ))

    # Imports
    info.imports = {}
    try:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll = entry.dll.decode("ascii", errors="ignore").lower()
                names = []
                for imp in entry.imports:
                    if imp.name:
                        names.append(imp.name.decode("ascii", errors="ignore"))
                    else:
                        names.append(f"ord:{imp.ordinal}")
                info.imports[dll] = sorted(set(names))
    except Exception:
        info.imports = info.imports or {}

    # Exports
    info.exports = []
    try:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"]])
        if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
            for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                if exp.name:
                    info.exports.append(exp.name.decode("ascii", errors="ignore"))
                else:
                    info.exports.append(f"ord:{exp.ordinal}")
            info.exports = sorted(set(info.exports))
    except Exception:
        pass

    # Resources (names/types only)
    info.resources = []
    try:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]])
        if hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
            for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                if entry.name is not None:
                    info.resources.append(f"name:{entry.name.string.decode('ascii', errors='ignore')}")
                else:
                    info.resources.append(f"type:{entry.id}")
    except Exception:
        pass

    # TLS
    info.has_tls = False
    info.tls_callbacks = []
    try:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_TLS"]])
        if hasattr(pe, "DIRECTORY_ENTRY_TLS"):
            info.has_tls = True
            tls = pe.DIRECTORY_ENTRY_TLS
            if hasattr(tls, "struct") and hasattr(tls.struct, "AddressOfCallBacks"):
                # pefile may already resolve callbacks in DIRECTORY_ENTRY_TLS.callback_functions
                cbs = getattr(tls, "callback_functions", None)
                if cbs:
                    info.tls_callbacks = list(cbs)
                else:
                    info.tls_callbacks = []
    except Exception:
        pass

    # Relocations (count only)
    info.reloc_count = None
    try:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_BASERELOC"]])
        if hasattr(pe, "DIRECTORY_ENTRY_BASERELOC"):
            cnt = 0
            for base in pe.DIRECTORY_ENTRY_BASERELOC:
                cnt += len(base.entries)
            info.reloc_count = cnt
    except Exception:
        pass

    # Debug info types (coarse)
    info.debug_types = []
    try:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DEBUG"]])
        if hasattr(pe, "DIRECTORY_ENTRY_DEBUG"):
            for dbg in pe.DIRECTORY_ENTRY_DEBUG:
                t = getattr(dbg.struct, "Type", None)
                if t is not None:
                    info.debug_types.append(str(t))
    except Exception:
        pass

    return info


# Minimal fallback parser: DOS header + PE signature + FileHeader + OptionalHeader (very basic) + Section table
# This is not a full PE parser. It is a best-effort fallback to allow basic comparisons.
DOS_HDR = struct.Struct("<2s58xI")  # e_magic, e_lfanew at offset 0x3C
PE_SIG = b"PE\x00\x00"
FILE_HDR = struct.Struct("<HHIIIHH")  # Machine, NumberOfSections, TimeDateStamp, PtrSymTable, NumSymbols, SizeOptHdr, Characteristics

def parse_minimal(path: str) -> PEInfo:
    info = PEInfo(path=path)
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 0x40:
        raise ValueError("File too small to be a PE")

    e_magic, e_lfanew = DOS_HDR.unpack_from(data, 0)
    if e_magic != b"MZ":
        raise ValueError("Missing MZ header")

    if e_lfanew + 4 + FILE_HDR.size > len(data):
        raise ValueError("Invalid e_lfanew")

    if data[e_lfanew:e_lfanew+4] != PE_SIG:
        raise ValueError("Missing PE signature")

    coff_off = e_lfanew + 4
    fh = FILE_HDR.unpack_from(data, coff_off)
    machine, nsects, tstamp, _, _, size_opt, chars = fh
    info.machine = machine
    info.number_of_sections = nsects
    info.timestamp = tstamp
    info.characteristics = chars

    opt_off = coff_off + FILE_HDR.size
    # Optional header magic tells 32-bit vs 64-bit
    if opt_off + 2 <= len(data):
        magic = struct.unpack_from("<H", data, opt_off)[0]
        if magic == 0x10B:  # PE32
            fmt = "<HBBIIIIII"  # partial only: Magic, MajorLinker, MinorLinker, SizeOfCode, SizeOfInitializedData, SizeOfUninitializedData, AddressOfEntryPoint, BaseOfCode, BaseOfData
            if opt_off + 28 <= len(data):
                vals = struct.unpack_from(fmt, data, opt_off)
                info.entry_point = vals[6]
        elif magic == 0x20B:  # PE32+
            fmt = "<HBBIIIII"  # partial: Magic, MajorLinker, MinorLinker, SizeOfCode, SizeOfInitializedData, SizeOfUninitializedData, AddressOfEntryPoint
            if opt_off + 24 <= len(data):
                vals = struct.unpack_from(fmt, data, opt_off)
                info.entry_point = vals[6]

    # Sections
    sect_off = opt_off + size_opt
    info.sections = []
    SEC_HDR = struct.Struct("<8sIIIIIIHHI")
    for i in range(nsects or 0):
        off = sect_off + i * SEC_HDR.size
        if off + SEC_HDR.size > len(data):
            break
        (name, vsize, vaddr, rsize, rptr, _, _, _, _, ch) = SEC_HDR.unpack_from(data, off)
        name_str = _read_c_string(name)
        raw_slice = data[rptr:rptr+rsize] if rptr and rsize and (rptr + rsize) <= len(data) else b""
        info.sections.append(SectionInfo(
            name=name_str,
            virtual_size=vsize,
            virtual_address=vaddr,
            raw_size=rsize,
            raw_ptr=rptr,
            characteristics=ch,
            md5=md5_bytes(raw_slice) if raw_slice else None,
            entropy=calc_entropy(raw_slice) if raw_slice else None
        ))

    # No imports/exports/resources in minimal mode
    info.imports = {}
    info.exports = []
    info.resources = []
    info.has_tls = False
    info.tls_callbacks = []
    info.reloc_count = None
    info.debug_types = []
    return info


def parse_pe(path: str) -> PEInfo:
    if HAS_PEFILE:
        try:
            return parse_with_pefile(path)
        except Exception as e:
            print(f"[warn] pefile parse failed: {e}. Falling back to minimal parser.", file=sys.stderr)
    return parse_minimal(path)


def _sorted_sections(sections: List[SectionInfo]) -> List[SectionInfo]:
    return sorted(sections, key=lambda s: (s.name.lower(), s.virtual_address))


def compare_lists(a: List[Any], b: List[Any]) -> Tuple[List[Any], List[Any]]:
    sa, sb = set(a), set(b)
    return sorted(sa - sb), sorted(sb - sa)


def compare_dict_of_lists(a: Dict[str, List[str]], b: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    all_keys = sorted(set(a.keys()) | set(b.keys()))
    diff: Dict[str, Dict[str, List[str]]] = {}
    for k in all_keys:
        va = set(a.get(k, []))
        vb = set(b.get(k, []))
        only_a = sorted(va - vb)
        only_b = sorted(vb - va)
        if only_a or only_b:
            diff[k] = {"only_left": only_a, "only_right": only_b}
    return diff


def colorize(s: str, color: str, enable: bool) -> str:
    if not enable:
        return s
    colors = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "cyan": "\033[36m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color,'')}{s}{colors['reset']}"


def summarize_pe(info: PEInfo) -> Dict[str, Any]:
    return {
        "path": info.path,
        "machine": info.machine,
        "characteristics": info.characteristics,
        "subsystem": info.subsystem,
        "dll_characteristics": info.dll_characteristics,
        "timestamp": info.timestamp,
        "entry_point": info.entry_point,
        "image_base": info.image_base,
        "section_alignment": info.section_alignment,
        "file_alignment": info.file_alignment,
        "number_of_sections": info.number_of_sections,
        "sections": [asdict(s) for s in _sorted_sections(info.sections or [])],
        "imports": info.imports or {},
        "exports": sorted(info.exports or []),
        "resources": sorted(info.resources or []),
        "has_tls": info.has_tls,
        "tls_callbacks": info.tls_callbacks or [],
        "reloc_count": info.reloc_count,
        "debug_types": sorted(info.debug_types or []),
    }


def diff_pes(left: PEInfo, right: PEInfo) -> Dict[str, Any]:
    L = summarize_pe(left)
    R = summarize_pe(right)

    diffs: Dict[str, Any] = {"files": (left.path, right.path)}

    # Simple scalar fields
    scalars = ["machine", "characteristics", "subsystem", "dll_characteristics",
               "timestamp", "entry_point", "image_base", "section_alignment",
               "file_alignment", "number_of_sections"]
    scalar_diffs = {}
    for k in scalars:
        if L.get(k) != R.get(k):
            scalar_diffs[k] = {"left": L.get(k), "right": R.get(k)}
    if scalar_diffs:
        diffs["header_differences"] = scalar_diffs

    # Sections by name
    left_secs = {s["name"]: s for s in L["sections"]}
    right_secs = {s["name"]: s for s in R["sections"]}
    sec_names_only_left = sorted(set(left_secs.keys()) - set(right_secs.keys()))
    sec_names_only_right = sorted(set(right_secs.keys()) - set(left_secs.keys()))
    if sec_names_only_left or sec_names_only_right:
        diffs["section_name_differences"] = {
            "only_left": sec_names_only_left,
            "only_right": sec_names_only_right,
        }

    common_secs = sorted(set(left_secs.keys()) & set(right_secs.keys()))
    per_section = {}
    for name in common_secs:
        a = left_secs[name]
        b = right_secs[name]
        fields = ["virtual_size", "virtual_address", "raw_size", "characteristics", "md5"]
        sub = {}
        for f in fields:
            if a.get(f) != b.get(f):
                sub[f] = {"left": a.get(f), "right": b.get(f)}
        if sub:
            per_section[name] = sub
    if per_section:
        diffs["section_differences"] = per_section

    # Imports
    imp_diff = compare_dict_of_lists(L["imports"], R["imports"])
    if imp_diff:
        diffs["import_differences"] = imp_diff

    # Exports
    only_left, only_right = compare_lists(L["exports"], R["exports"])
    if only_left or only_right:
        diffs["export_differences"] = {"only_left": only_left, "only_right": only_right}

    # Resources
    res_left, res_right = compare_lists(L["resources"], R["resources"])
    if res_left or res_right:
        diffs["resource_differences"] = {"only_left": res_left, "only_right": res_right}

    # TLS callbacks presence and addresses (coarse)
    if (L["has_tls"] != R["has_tls"]) or (set(L["tls_callbacks"]) != set(R["tls_callbacks"])):
        diffs["tls_differences"] = {
            "left_has_tls": L["has_tls"],
            "right_has_tls": R["has_tls"],
            "left_callbacks": sorted(L["tls_callbacks"]),
            "right_callbacks": sorted(R["tls_callbacks"]),
        }

    # Relocation count
    if L.get("reloc_count") != R.get("reloc_count"):
        diffs["relocation_count_difference"] = {"left": L.get("reloc_count"), "right": R.get("reloc_count")}

    # Debug types
    dbg_left, dbg_right = compare_lists(L["debug_types"], R["debug_types"])
    if dbg_left or dbg_right:
        diffs["debug_type_differences"] = {"only_left": dbg_left, "only_right": dbg_right}

    return diffs


def print_human(diff: Dict[str, Any], color: bool = True) -> None:
    left, right = diff.get("files", ("left", "right"))
    print(colorize(f"Comparing:", "bold", color), left, "<->", right)
    any_diff = False

    def section(title: str) -> None:
        print(colorize(f"\n{title}", "cyan", color))

    if "header_differences" in diff:
        any_diff = True
        section("Header differences")
        for k, v in diff["header_differences"].items():
            print(f"  {k}: {v['left']}  !=  {v['right']}")

    if "section_name_differences" in diff:
        any_diff = True
        section("Section name differences")
        only_left = diff["section_name_differences"].get("only_left", [])
        only_right = diff["section_name_differences"].get("only_right", [])
        if only_left:
            print(colorize("  Only in left:", "yellow", color), ", ".join(only_left))
        if only_right:
            print(colorize("  Only in right:", "yellow", color), ", ".join(only_right))

    if "section_differences" in diff:
        any_diff = True
        section("Per-section differences")
        for name, changes in diff["section_differences"].items():
            print(colorize(f"  [{name}]", "bold", color))
            for f, vals in changes.items():
                print(f"    {f}: {vals['left']}  !=  {vals['right']}")

    if "import_differences" in diff:
        any_diff = True
        section("Import differences by DLL")
        for dll, changes in diff["import_differences"].items():
            left_only = changes.get("only_left", [])
            right_only = changes.get("only_right", [])
            print(colorize(f"  {dll}", "bold", color))
            if left_only:
                print("    only in left:", ", ".join(left_only))
            if right_only:
                print("    only in right:", ", ".join(right_only))

    if "export_differences" in diff:
        any_diff = True
        section("Export differences")
        left_only = diff["export_differences"].get("only_left", [])
        right_only = diff["export_differences"].get("only_right", [])
        if left_only:
            print("  only in left:", ", ".join(left_only))
        if right_only:
            print("  only in right:", ", ".join(right_only))

    if "resource_differences" in diff:
        any_diff = True
        section("Resource differences (types/names)")
        left_only = diff["resource_differences"].get("only_left", [])
        right_only = diff["resource_differences"].get("only_right", [])
        if left_only:
            print("  only in left:", ", ".join(left_only))
        if right_only:
            print("  only in right:", ", ".join(right_only))

    if "tls_differences" in diff:
        any_diff = True
        section("TLS differences")
        d = diff["tls_differences"]
        print(f"  left has TLS: {d['left_has_tls']}  right has TLS: {d['right_has_tls']}")
        if d.get("left_callbacks") or d.get("right_callbacks"):
            print("  left callbacks:", d.get("left_callbacks"))
            print("  right callbacks:", d.get("right_callbacks"))

    if "relocation_count_difference" in diff:
        any_diff = True
        section("Relocation entry count differs")
        d = diff["relocation_count_difference"]
        print(f"  left: {d['left']}  right: {d['right']}")

    if "debug_type_differences" in diff:
        any_diff = True
        section("Debug directory type differences")
        d = diff["debug_type_differences"]
        if d.get("only_left"):
            print("  only in left:", ", ".join(d["only_left"]))
        if d.get("only_right"):
            print("  only in right:", ", ".join(d["only_right"]))

    if not any_diff:
        print(colorize("\nNo structural differences detected.", "green", color))


def main() -> int:
    p = argparse.ArgumentParser(description="Compare the structure of two PE DLL/EXE files.")
    p.add_argument("left", help="Path to the first DLL/EXE")
    p.add_argument("right", help="Path to the second DLL/EXE")
    p.add_argument("--json", dest="json_out", default=None, help="Optional path to write a JSON diff report")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors in console output")
    args = p.parse_args()

    try:
        left = parse_pe(args.left)
        right = parse_pe(args.right)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    diff = diff_pes(left, right)

    # Human report
    print_human(diff, color=(not args.no_color))

    # JSON report
    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(diff, f, indent=2)
            print(f"\nWrote JSON report to: {args.json_out}")
        except Exception as e:
            print(f"Failed to write JSON: {e}", file=sys.stderr)

    # Exit status
    # If any top-level key other than 'files' exists, we found differences
    diff_found = any(k for k in diff.keys() if k != "files")
    return 1 if diff_found else 0


if __name__ == "__main__":
    sys.exit(main())
