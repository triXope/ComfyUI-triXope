import os
import server
from aiohttp import web

from .ColorFX import *
from .DirectoryFileSelector import NODE_CLASS_MAPPINGS as DFS_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DFS_DISPLAY_NAME_MAPPINGS
from .filmauteur_ltx import *
from .ltxv_eta_node import *

@server.PromptServer.instance.routes.get("/gemini/get_files")
async def get_files_in_directory(request):
    """
    API endpoint to scan a directory and return a list of its files.
    """
    directory_path = request.rel_url.query.get("directory", "")
    if not directory_path or not os.path.isdir(directory_path):
        return web.json_response({"files": []})

    try:
        files = sorted([
            f for f in os.listdir(directory_path) 
            if os.path.isfile(os.path.join(directory_path, f))
        ])
        return web.json_response({"files": files})
    except Exception as e:
        print(f"Error reading directory '{directory_path}': {e}")
        return web.json_response({"files": []})

NODE_CLASS_MAPPINGS = {
    "ColorFX": ColorFX,
    "FilmAuteur_LTX": FilmAuteur_LTX,
    "LTXVPostSliceAV": LTXVPostSliceAV,
    "LTXV_ETA_Display": LTXV_ETA_Display,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorFX": "triXope ColorFX 🎬",
    "FilmAuteur_LTX": "triXope Film Auteur (LTX)",
    "LTXVPostSliceAV": "triXope LTXV Trim A/V",
    "LTXV_ETA_Display": "triXope LTXV Real-Time ETA",
}

NODE_CLASS_MAPPINGS.update(DFS_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(DFS_DISPLAY_NAME_MAPPINGS)
WEB_DIRECTORY = "."

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
