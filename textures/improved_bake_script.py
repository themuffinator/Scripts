#!/usr/bin/env python3
"""
Improved Blender Bake Script for Q4 to Q3 Converter
Definitive version with restored legacy material logic for correct visual output.
Fully compatible with Blender 4.x.

CHANGES (minimal):
- Added optional --glow to additively blend a glow map over diffuse before bake.
- Added two tiny utility modes:
    --convert <in> --out <path>     (just loads & saves with format from extension)
    --save_glow <in> --out <path>   (writes grayscale luma copy)
"""

import bpy
import sys
import os
import argparse
import traceback
from pathlib import Path
from math import radians
from PIL import Image

def save_glow_greyscale(in_path: str, out_path: str):
    """
    Make a greyscale '_glow' copy with black->transparent, white->opaque.
    Matches the 'black to transparency' behaviour you described.
    """
    img = Image.open(in_path).convert("RGBA")
    # Luma drives alpha (0=transparent, 255=opaque)
    luma = Image.open(in_path).convert("L")
    r, g, b, _ = img.split()
    a = luma

    # Un-premultiply RGB to avoid dark fringes on edges
    rp, gp, bp, ap = r.load(), g.load(), b.load(), a.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            al = ap[x, y]
            if al <= 0:
                rp[x, y] = gp[x, y] = bp[x, y] = 0
            else:
                rp[x, y] = min(255, (rp[x, y] * 255) // al)
                gp[x, y] = min(255, (gp[x, y] * 255) // al)
                bp[x, y] = min(255, (bp[x, y] * 255) // al)

    out = Image.merge("RGBA", (r, g, b, a))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)

# ======================== ARGUMENT PARSING ========================

def parse_args():
    """Parse command line arguments passed after '--'"""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    
    # Utility modes (minimal)
    parser.add_argument("--convert", type=str, default=None, help="Convert single image to --out (format by ext)")
    parser.add_argument("--save_glow", type=str, default=None, help="Save grayscale luma version of image to --out")
    parser.add_argument("--out", type=str, default=None, help="Output path for --convert/--save_glow or bake result")

    # Core paths
    parser.add_argument("--diffuse", help="Diffuse map path")
    
    # Optional map paths
    parser.add_argument("--normal", help="Normal map path")
    parser.add_argument("--height", help="Height map path")
    parser.add_argument("--spec", help="Specular map path")
    parser.add_argument("--glow", help="Optional glow/additive map blended over diffuse")  # <--- NEW
    
    # Bake settings
    parser.add_argument("--height_scale", type=float, default=1.0)
    parser.add_argument("--normal_strength", type=float, default=1.0)
    parser.add_argument("--sun_az", type=float, default=45.0)
    parser.add_argument("--sun_el", type=float, default=55.0)
    parser.add_argument("--power", type=float, default=3.5)
    parser.add_argument("--ambient", type=float, default=0.05)
    
    # Vibrance settings from legacy script
    parser.add_argument("--spec_gain", type=float, default=1.2)
    parser.add_argument("--roughness_bias", type=float, default=-0.05)
    parser.add_argument("--sat_gain", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=0.0)
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--diffuse_gain", type=float, default=1.0)
    
    # Quality settings
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--margin", type=int, default=4)
    
    return parser.parse_args(argv)

# ======================== UTILITIES (minimal) ========================

def _fmt_from_ext(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "PNG",
        ".tga": "TARGA",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".bmp": "BMP",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
    }.get(ext, "PNG")

def save_image_to(img: bpy.types.Image, out_path: Path):
    img.colorspace_settings.name = "sRGB"
    img.alpha_mode = 'STRAIGHT'
    img.file_format = _fmt_from_ext(out_path)
    img.filepath_raw = str(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save()

def convert_image(in_path: str, out_path: str):
    img = bpy.data.images.load(in_path, check_existing=True)
    save_image_to(img, Path(out_path))
    print(f"[OK] convert: {in_path} -> {out_path}")

def desaturate_image_with_pillow(in_path: Path, out_path: Path):
    """
    Loads an image using Pillow, converts it to grayscale (desaturates it),
    and saves it to the specified output path.
    """
    try:
        img = Image.open(in_path)
        # Convert to 'L' mode for grayscale (luminance), which is a full desaturation.
        # This uses the ITU-R 601-2 luma transform, similar to our earlier
        # weighted average approach, but handled internally by Pillow.
        greyscale_img = img.convert('L')
        
        # To ensure it's still treated as an RGB-like image (e.g., for Q2Re),
        # convert it back to RGB, effectively making it R=G=B=Luma.
        rgb_greyscale_img = greyscale_img.convert('RGB')
        
        rgb_greyscale_img.save(out_path)
        print(f"[OK] Desaturated glow map (Pillow): {in_path} -> {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to desaturate {in_path} with Pillow: {e}")
        
# ======================== SCENE SETUP ========================

def clear_scene():
    """Wipes the scene completely."""
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.worlds, bpy.data.lights]:
        while collection:
            collection.remove(collection[0])

def setup_scene(args):
    """Configures the Blender scene for baking."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.samples
    scene.cycles.device = 'GPU'
    
    world = bpy.data.worlds.new("BakeWorld")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = args.ambient
    
    bpy.ops.object.light_add(type='SUN')
    sun = bpy.context.active_object
    sun.data.energy = args.power
    sun.rotation_euler = (radians(90 - args.sun_el), 0, radians(args.sun_az))

def create_bake_plane():
    """Creates a 2x2 plane with default UVs."""
    bpy.ops.mesh.primitive_plane_add(size=2.0)
    plane = bpy.context.active_object
    plane.name = "BakePlane"
    return plane

# ======================== MATERIAL & BAKING (RESTORED LOGIC + glow) ========================

def create_bake_material(args, diffuse_img):
    """Builds the material using logic from the legacy bake_diffuse.py for visual correctness.
       Minimal change: if --glow is provided, additively blend it over the graded diffuse."""
    mat = bpy.data.materials.new(name="BakeMaterial")
    mat.use_nodes = True
    nt = mat.node_tree
    
    for node in nt.nodes: nt.nodes.remove(node)

    out_node = nt.nodes.new("ShaderNodeOutputMaterial"); out_node.location = (1200, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (1000, 0)
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    # --- Diffuse Chain ---
    tex_d = nt.nodes.new("ShaderNodeTexImage"); tex_d.location = (-200, 300)
    tex_d.image = diffuse_img
    
    hsv = nt.nodes.new("ShaderNodeHueSaturation"); hsv.location = (0, 300)
    hsv.inputs["Saturation"].default_value = args.sat_gain
    hsv.inputs["Value"].default_value = args.diffuse_gain
    
    bc = nt.nodes.new("ShaderNodeBrightContrast"); bc.location = (200, 300)
    bc.inputs["Bright"].default_value = args.brightness
    bc.inputs["Contrast"].default_value = args.contrast
    
    nt.links.new(tex_d.outputs["Color"], hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], bc.inputs["Color"])

    # --- Minimal glow ADD stage over diffuse ---
    base_color_socket = bc.outputs["Color"]
    if args.glow and os.path.exists(args.glow):
        tex_g = nt.nodes.new("ShaderNodeTexImage"); tex_g.location = (200, 120)
        try:
            tex_g.image = bpy.data.images.load(args.glow, check_existing=True)
            tex_g.image.colorspace_settings.name = 'sRGB'
        except Exception:
            tex_g.image = None
        add = nt.nodes.new("ShaderNodeMixRGB"); add.location = (420, 240)
        add.blend_type = 'ADD'
        add.inputs["Fac"].default_value = 1.0
        nt.links.new(base_color_socket, add.inputs["Color1"])
        if tex_g.image:
            nt.links.new(tex_g.outputs["Color"], add.inputs["Color2"])
        base_color_socket = add.outputs["Color"]

    nt.links.new(base_color_socket, bsdf.inputs["Base Color"])

    # Connect alpha channel to prevent transparency issues (kept from legacy)
    nt.links.new(tex_d.outputs["Alpha"], bsdf.inputs["Alpha"])
    
    # --- Normal + Height Chain ---
    last_normal_socket = None
    if args.normal and os.path.exists(args.normal):
        tex_n = nt.nodes.new("ShaderNodeTexImage"); tex_n.location = (-200, 0)
        tex_n.image = bpy.data.images.load(args.normal); tex_n.image.colorspace_settings.name = 'Non-Color'
        nrm_map = nt.nodes.new("ShaderNodeNormalMap"); nrm_map.location = (0, 0)
        nrm_map.inputs["Strength"].default_value = args.normal_strength
        nt.links.new(tex_n.outputs["Color"], nrm_map.inputs["Color"])
        last_normal_socket = nrm_map.outputs["Normal"]

    if args.height and os.path.exists(args.height):
        tex_h = nt.nodes.new("ShaderNodeTexImage"); tex_h.location = (-200, -300)
        tex_h.image = bpy.data.images.load(args.height); tex_h.image.colorspace_settings.name = 'Non-Color'
        bump = nt.nodes.new("ShaderNodeBump"); bump.location = (0, -200)
        bump.inputs["Strength"].default_value = args.height_scale
        nt.links.new(tex_h.outputs["Color"], bump.inputs["Height"])
        if last_normal_socket: nt.links.new(last_normal_socket, bump.inputs["Normal"])
        last_normal_socket = bump.outputs["Normal"]
    
    if last_normal_socket: nt.links.new(last_normal_socket, bsdf.inputs["Normal"])
        
    # --- Specular -> Roughness Chain (RESTORED LOGIC FROM bake_diffuse.py) ---
    bsdf.inputs["IOR"].default_value = 1.45
    if args.spec and os.path.exists(args.spec):
        tex_s = nt.nodes.new("ShaderNodeTexImage"); tex_s.location = (0, -600)
        tex_s.image = bpy.data.images.load(args.spec); tex_s.image.colorspace_settings.name = 'Non-Color'
        
        luma = nt.nodes.new("ShaderNodeRGBToBW"); luma.location = (200, -600)
        nt.links.new(tex_s.outputs["Color"], luma.inputs["Color"])
        
        spec_gain_math = nt.nodes.new("ShaderNodeMath"); spec_gain_math.location = (400, -600)
        spec_gain_math.operation = 'MULTIPLY'
        spec_gain_math.inputs[1].default_value = args.spec_gain
        nt.links.new(luma.outputs["Val"], spec_gain_math.inputs[0])
        
        invert = nt.nodes.new("ShaderNodeInvert"); invert.location = (600, -600)
        nt.links.new(spec_gain_math.outputs["Value"], invert.inputs["Color"])
        
        bias_math = nt.nodes.new("ShaderNodeMath"); bias_math.location = (800, -600)
        bias_math.operation = 'ADD'
        bias_math.use_clamp = True
        bias_math.inputs[1].default_value = args.roughness_bias
        nt.links.new(invert.outputs["Color"], bias_math.inputs[0])
        nt.links.new(bias_math.outputs["Value"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = 0.7

    mat.blend_method = 'OPAQUE'
    return mat

def bake_and_save(plane, args, diffuse_img):
    """Performs the bake operation and saves the file."""
    width, height = diffuse_img.size
    bake_target_img = bpy.data.images.new("BakeTarget", width, height, alpha=True)
    
    mat = plane.active_material
    target_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
    target_node.image = bake_target_img
    mat.node_tree.nodes.active = target_node

    print("[INFO] Starting bake...")
    bpy.ops.object.bake(type='COMBINED', use_clear=True, margin=args.margin)
    print("[INFO] Bake completed.")
    
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_path.suffix.lower().lstrip('.')
    if fmt == "tga": fmt = "TARGA"
    
    bake_target_img.filepath_raw = str(output_path)
    bake_target_img.file_format = fmt.upper()
    bake_target_img.save()
    
    if not output_path.exists():
        raise RuntimeError(f"Bake failed to write file: {output_path}")
    print(f"[SUCCESS] Saved baked texture to {output_path}")

# ======================== MAIN WORKFLOW ========================

def main():
    try:
        args = parse_args()

        # Utility modes (minimal; no scene)
        if args.convert and args.out:
            convert_image(args.convert, args.out)
            sys.exit(0)
        if args.save_glow and args.out:
            save_glow_greyscale(args.save_glow, args.out)
            sys.exit(0)

        # Bake mode requires diffuse + out
        if not args.diffuse or not args.out:
            raise RuntimeError("Bake mode requires --diffuse and --out")

        out_path = Path(args.out)
        
        clear_scene()
        
        diffuse_img = bpy.data.images.load(args.diffuse)
        if not diffuse_img:
            raise RuntimeError(f"Could not load diffuse map: {args.diffuse}")
            
        setup_scene(args)
        plane = create_bake_plane()
        
        material = create_bake_material(args, diffuse_img)
        plane.data.materials.append(material)
        
        # CRITICAL: Ensure the plane is the active object before baking.
        bpy.ops.object.select_all(action='DESELECT')
        plane.select_set(True)
        bpy.context.view_layer.objects.active = plane
        
        bake_and_save(plane, args, diffuse_img)
        
        sys.exit(0)
    
    except Exception as e:
        print(f"[FATAL] A critical error occurred in the bake script: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
