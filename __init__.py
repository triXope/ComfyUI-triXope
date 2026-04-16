import os
import server
from aiohttp import web

from .Trixope_VAE_Encoder import Trixope_WanVAEEncodeBatched
from .Trixope_VAE_Decoder import Trixope_WanVAEDecodeBatched
from .Trixope_VAE_Decoder_Native import Trixope_VAEDecodeBatched_Native
from .nodes import *
from .ColorFX import *
from .image_batch_upscaler import *
from .video_upscale_dimensions import *
from .text_switch import text_switch
from .prompt_split import *
from .AudioToTablature import *
from .GroupMonitor import *
from .DirectoryFileSelector import NODE_CLASS_MAPPINGS as DFS_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DFS_DISPLAY_NAME_MAPPINGS
from .story_select import *
from .character_craft import *
from .image_batch_selector import *
from .video_latents import *
from .toggle_folder_save import *
from .path_combiner_node import *
from .ensure_directory import *
from .filmauteur_ltxv import *
from .eta_node import *

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
    "triXope File Copier": FileCopier,
    "AudioFileSaver": AudioFileSaver,
    "triXope Image Sequence Preview": PreviewAnimation,
    "triXope Load (Character) Image": ImageCharacterLoader,
    "triXope Load (Outfit) Image": ImageOutfitLoader,
    "triXope Load (Scene) Image": ImageSceneLoader,
    "LoadText": LoadText,
    "StringCleaner": StringCleaner,
    "FilenameFromDirectory": FilenameFromDirectory,
    "triXope_VideoPreview": triXope_VideoPreview,
    "ColorFX": ColorFX,
    "Trixope_WanVAEEncodeBatched": Trixope_WanVAEEncodeBatched,
    "Trixope_WanVAEDecodeBatched": Trixope_WanVAEDecodeBatched,
    "Trixope_VAEDecodeBatched_Native": Trixope_VAEDecodeBatched_Native,
    "ImageUpscaleToSize": ImageUpscaleToSize,
    "Video_Upscale_To_Dimensions": Video_Upscale_To_Dimensions,
    "text_switch": text_switch,
    "TriXopePromptSplitSelect": TriXopePromptSplitSelect,
    "AudioToTablature": AudioToTablature,
    "GroupMonitor": GroupMonitor,
    "TriXopeStorySelect": TriXopeStorySelect,
    "TriXopeCharacterCraft": TriXopeCharacterCraft,
    "ImageBatchSelector": ImageBatchSelector,
    "SaveWanVideoLatent": SaveWanVideoLatent,
    "SaveLTXVideoLatent": SaveLTXVideoLatent,
    "ToggleFolderSaver": ToggleFolderSaver,
    "PathCombinerNode": PathCombinerNode,
    "EnsureDirectoryNode": EnsureDirectoryNode,
    "LoadAndConcatVideoLatentsFromDir": LoadAndConcatVideoLatentsFromDir,
    "WanDirectoryVideoDecoder": WanDirectoryVideoDecoder,
    "LTXVDirectoryVideoDecoder": LTXVDirectoryVideoDecoder,
    "FilmAuteur_LTXV": FilmAuteur_LTXV,
    "LTXVPostSliceAV": LTXVPostSliceAV,
    "RealtimeLoopTracker": RealtimeLoopTracker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "triXope File Copier": "triXope File Copier",
    "AudioFileSaver": "triXope Audio File Saver",
    "triXope Image Sequence Preview": "triXope Image Sequence Preview",
    "triXope Load (Character) Image": "triXope Load (Character) Image",
    "triXope Load (Outfit) Image": "triXope Load (Outfit) Image",
    "triXope Load (Scene) Image": "triXope Load (Scene) Image",
    "LoadText": "triXope Load Text",
    "StringCleaner": "triXope String Cleaner",
    "FilenameFromDirectory": "triXope Filename Extractor",
    "triXope_VideoPreview": "triXope Video Preview",
    "ColorFX": "triXope ColorFX 🎬",
    "Trixope_WanVAEEncodeBatched": "triXope VAE Encode Batched (WanVideo)",
    "Trixope_WanVAEDecodeBatched": "triXope VAE Decode Batched (WanVideo)",
    "Trixope_VAEDecodeBatched_Native": "triXope VAE Decode Batched (Native)",
    "ImageUpscaleToSize": "triXope Upscale Image to Size (Model)",
    "Video_Upscale_To_Dimensions": "triXope Video Upscale To Dimensions",
    "text_switch": "triXope Text Switch",
    "TriXopePromptSplitSelect": "triXope Prompt Split & Select",
    "AudioToTablature": "triXope Audio To Tablature 🎵",
    "GroupMonitor": "triXope Group Monitor",
    "TriXopeStorySelect": "triXope Story Craft",
    "TriXopeCharacterCraft": "triXope Character Craft",
    "ImageBatchSelector": "triXope Image Batch Filter",
    "SaveWanVideoLatent": "triXope Save WAN Latent",
    "SaveLTXVideoLatent": "triXope Save LTXV Latent",
    "ToggleFolderSaver": "triXope Save Image",
    "PathCombinerNode": "triXope Path Combiner",
    "EnsureDirectoryNode": "triXope Create Directory",
    "LoadAndConcatVideoLatentsFromDir": "triXope Load Video Latent",
    "WanDirectoryVideoDecoder": "triXope Wan Directory Video Decoder",
    "LTXVDirectoryVideoDecoder": "triXope LTXV Directory Video Decoder",
    "FilmAuteur_LTXV": "triXope Film Auteur (LTXV)",
    "LTXVPostSliceAV": "triXope LTXV Trim A/V",
    "RealtimeLoopTracker": "triXope Realtime Loop Tracker",
}

NODE_CLASS_MAPPINGS.update(DFS_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(DFS_DISPLAY_NAME_MAPPINGS)
WEB_DIRECTORY = "."

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]