#!/usr/bin/env python3
# png2tga_rgba.py — recursively convert PNG -> 32-bit TGA (RGBA), always saving an alpha channel

from pathlib import Path
from PIL import Image

def ensure_rgba(img: Image.Image) -> Image.Image:
    # Preserve existing alpha; if none, add an opaque channel
    if img.mode == "RGBA":
        return img
    if img.mode in ("LA", "P"):
        return img.convert("RGBA")
    if img.mode == "RGB":
        rgba = Image.new("RGBA", img.size)
        rgba.paste(img)
        return rgba
    return img.convert("RGBA")

def main():
    root = Path(".")  # change if needed
    for png in root.rglob("*.png"):
        tga = png.with_suffix(".tga")
        try:
            img = Image.open(png)
            rgba = ensure_rgba(img)
            tga.parent.mkdir(parents=True, exist_ok=True)
            # Saving RGBA ensures 32-bit TGA with 8-bit alpha
            rgba.save(tga, format="TGA")
            print(f"OK  -> {tga}")
        except Exception as e:
            print(f"FAIL: {png}: {e}")

if __name__ == "__main__":
    main()
