from pathlib import Path

# --- Configuration ---
# The directory to start searching from. '.' means the current directory.
SEARCH_DIRECTORY = Path('.')
# The fallback material name if no keywords match.
DEFAULT_MATERIAL = 'step'
# --- End Configuration ---

# This dictionary maps the final material name to a list of keywords.
# The script will check for these keywords in filenames and folder names.
# The order is important: more specific materials should come before general ones.
MATERIAL_ASSOCIATIONS = {
    'glass': ['glass', 'window', 'pane', 'crystal', 'gem'],
    'carpet': ['carpet', 'rug', 'cloth', 'fabric'],
    'snow': ['snow', 'ice', 'frost'],
    'splash': ['water', 'liquid', 'slime', 'sludge', 'ooze', 'puddle', 'sewer'],
    'wood': ['wood', 'plank', 'board', 'crate', 'door', 'frame'],
    'grass': ['grass', 'dirt', 'soil', 'earth', 'mud', 'sand', 'gravel', 'ground', 'terra', 'grnd'],
    'tile': ['tile', 'ceramic', 'brick', 'marble'],
    'flesh': ['flesh', 'skin', 'organ', 'gore', 'blood', 'gut', 'monster', 'meat', 'gib'],
    'clank': ['metal', 'steel', 'iron', 'grate', 'girder', 'plate', 'vent', 'rust', 'rivet', 'met', 'support', 'pipe', 'cable'],
    'mech': ['mech', 'machine', 'tech', 'panel', 'button', 'switch', 'mach', 'button'],
    'energy': ['energy', 'forcefield', 'shield', 'laser', 'plasma', 'beam', 'monitor', 'screen', 'computer', 'console', 'light'],
    'junk': ['junk', 'trash', 'debris', 'pile', 'scrap', 'rubble'],
    # 'step' is a generic material, so its keywords are checked last.
    'step': ['concrete', 'cement', 'stone', 'floor', 'pavement', 'path', 'asphalt', 'rock', 'crete'],
}

def guess_material(name_to_check: str) -> str | None:
    """
    Tries to guess a material from a string (e.g., a filename).
    Returns the material name on success, otherwise None.
    """
    name_lower = name_to_check.lower()
    for material, keywords in MATERIAL_ASSOCIATIONS.items():
        if any(keyword in name_lower for keyword in keywords):
            return material
    return None

def create_mat_files():
    """
    Recursively finds all .tga files and generates a .mat file for each
    if one does not already exist.
    """
    print(f"🔍 Starting search for .tga files in: {SEARCH_DIRECTORY.resolve()}")
    tga_files_found = 0
    mats_created = 0

    # Path.rglob('*.tga') recursively finds all files ending with .tga
    for tga_file in SEARCH_DIRECTORY.rglob('*.tga'):
        tga_files_found += 1
        mat_file = tga_file.with_suffix('.mat')

        # Skip if a .mat file already exists for this texture
        if mat_file.exists():
            continue

        # --- Guessing Logic ---
        material_name = None
        reason = ""

        # 1. Guess from the texture filename
        material_name = guess_material(tga_file.stem) # .stem is filename without extension
        if material_name:
            reason = f"(from filename: '{tga_file.name}')"
        else:
            # 2. If that fails, guess from the parent folder name
            material_name = guess_material(tga_file.parent.name)
            if material_name:
                reason = f"(from folder: '{tga_file.parent.name}')"
            else:
                # 3. If all else fails, use the default
                material_name = DEFAULT_MATERIAL
                reason = "(default fallback)"

        # --- File Creation ---
        try:
            mat_file.write_text(material_name)
            print(f"✅ Created '{mat_file.name}' with material '{material_name}' {reason}")
            mats_created += 1
        except Exception as e:
            print(f"❌ Failed to create file '{mat_file}': {e}")
            
    print("\n--- Generation Complete ---")
    print(f"Processed {tga_files_found} total .tga files.")
    print(f"Generated {mats_created} new .mat files.")


if __name__ == "__main__":
    create_mat_files()