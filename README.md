# Scripts

## idTech4 texture conversion

The Doom 3 and Quake 4 texture pipelines now share a unified implementation in
`textures/idtech4_to_idtech23_converter.py`.  The legacy entry points
(`d3_to_q3_full_converter.py` and `q4_to_q3_full_converter.py`) remain as thin
wrappers so existing automation can keep calling them, but all configuration is
now profile-driven via `textures/convert_config.json`.

