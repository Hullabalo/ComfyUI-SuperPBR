from .nodes import SuperPBRDecompose, SuperPBRDetilt, SuperPBRModelLoader, SuperPBRORMSplit
from .tiles import SuperPBRTileSplit, SuperPBRTileStitch

NODE_CLASS_MAPPINGS = {
    "SuperPBRModelLoader": SuperPBRModelLoader,
    "SuperPBRDecompose": SuperPBRDecompose,
    "SuperPBRDetilt": SuperPBRDetilt,
    "SuperPBRORMSplit": SuperPBRORMSplit,
    "SuperPBRTileSplit": SuperPBRTileSplit,
    "SuperPBRTileStitch": SuperPBRTileStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperPBRModelLoader": "🆒 Load SuperPBR Model",
    "SuperPBRDecompose": "🆒 SuperPBR Material",
    "SuperPBRDetilt": "🆒 SuperPBR Detilt",
    "SuperPBRORMSplit": "🆒 SuperPBR ORM Split",
    "SuperPBRTileSplit": "🆒 SuperPBR Tile Split",
    "SuperPBRTileStitch": "🆒 SuperPBR Tile Stitch",
}
