import logging
import sys
import os

logger = logging.getLogger(__name__)

# Check for original USDU script
current_dir = os.path.dirname(os.path.realpath(__file__))
repos_dir = os.path.join(current_dir, "repositories")
usdu_dir = os.path.join(repos_dir, "ultimate_sd_upscale")
if not len(os.listdir(usdu_dir)):
    logger.info("Original USDU script not found, downloading it from https://github.com/Coyote-A/ultimate-upscale-for-automatic1111")
    import urllib.request
    import zipfile
    import shutil

    url = "https://github.com/Coyote-A/ultimate-upscale-for-automatic1111/archive/master.zip"
    zip_path = os.path.join(current_dir, "usdu_temp.zip")

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        top_folder = zip_ref.namelist()[0].split('/')[0] + '/'
        for member in zip_ref.namelist():
            if member.startswith(top_folder) and not member.endswith('/'):
                target_path = os.path.join(usdu_dir, member[len(top_folder):])
                os.makedirs(os.path.dirname(target := os.path.join(usdu_dir, member[len(top_folder):])), exist_ok=True)
                with zip_ref.open(member) as source, open(target, 'wb') as target_file:
                    shutil.copyfileobj(fsrc=zip_ref.open(member), fdst=target_file)

    os.remove(zip_path)
    logger.info("Original USDU script downloaded successfully")

# Remove other custom_node paths from sys.path to avoid conflicts
custom_node_paths = [path for path in sys.path if "custom_node" in path]
original_sys_path = sys.path.copy()
for path in custom_node_paths:
    sys.path.remove(path)

# Add this repository's path to sys.path for third-party imports
repo_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, repo_dir)
original_modules = sys.modules.copy()

# Place aside potentially conflicting modules
modules_used = [
    "modules",
    "modules.devices",
    "modules.images",
    "modules.processing",
    "modules.scripts",
    "modules.shared",
    "modules.upscaler",
    "utils",
]
original_imported_modules = {}
for module in modules_used:
    if module in sys.modules:
        original_imported_modules[module] = sys.modules.pop(module)

import server
from aiohttp import web
from comfy_api.latest import ComfyExtension

from .nodes_audio import *
from .nodes_film import *
from .nodes_film import FilmAuteurH3Extension
from .nodes_image import *
from .nodes_io import *
from .nodes_latent import *
from .nodes_loaders import *
from .nodes_utilities import *
from .nodes_video import *
from .nodes_writing import *

async def comfy_entrypoint() -> FilmAuteurH3Extension:
    return FilmAuteurH3Extension()

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
    "triXope AnyCast": AnyCast,
    "triXope AnySwitch": triXopeAnySwitch,
    "triXope AnySwitchInversed": triXopeAnySwitchInversed,
    "triXope AudioFileSaver": AudioFileSaver,
    "triXope AudioToTablature": AudioToTablature,
    "triXope BatchOllamaWriter": TriXopeBatchOllamaWriter,
    "triXope CharacterCraft": TriXopeCharacterCraft,
    "triXope ColorFX": ColorFX,
    "triXope CurrentPromptDisplay": TriXopeCurrentPromptDisplay,
    "triXope DirectoryFileSelectorGemini": DirectoryFileSelector,
    "triXope EfficientImageBatchLoader": EfficientImageBatchLoader,
    "triXope EnsureDirectoryNode": EnsureDirectoryNode,
    "triXope File Copier": FileCopier,
    "triXope FilenameFromDirectory": FilenameFromDirectory,
    "triXope FilmAuteurH3": FilmAuteurH3,
    "triXope FilmAuteur_LTX": FilmAuteur_LTX,
    "triXope FinalFrameSelector": FinalFrameSelector,
    "triXope FirstFrameSelector": FirstFrameSelector,
    "triXope GroupMonitor": GroupMonitor,
#    "triXope H3FL2VAandRef2VA": H3FL2VAandRef2VA,
#    "triXope H3SlidingContextSampler": H3SlidingContextSampler,
    "triXope HistogramMatchHSV": HistogramMatchNode,
    "triXope Image Sequence Preview": PreviewAnimation,
    "triXope ImageBatchSelector": ImageBatchSelector,
    "triXope ImageUpscaleToSize": ImageUpscaleToSize,
    "triXope Load (Character) Image": ImageCharacterLoader,
    "triXope Load (Outfit) Image": ImageOutfitLoader,
    "triXope Load (Scene) Image": ImageSceneLoader,
    "triXope LoadAndConcatVideoLatentsFromDir": LoadAndConcatVideoLatentsFromDir,
    "triXope LoadText": LoadText,
    "triXope LTXV_ETA_Display": LTXV_ETA_Display,
    "triXope LTXVDirectoryVideoDecoder": LTXVDirectoryVideoDecoder,
    "triXope LTXVPostSliceAV": LTXVPostSliceAV,
    "triXope MiniMaxH3AVLatentConcat": MiniMaxH3AVLatentConcat,
    "triXope MultiClipTrimmer": MultiClipTrimmer,
    "triXope Pano360Rotate": Pano360Rotate,
    "triXope PathCombinerNode": PathCombinerNode,
    "triXope PromptSplitSelect": TriXopePromptSplitSelect,
    "triXope RealtimeLoopTracker": RealtimeLoopTracker,
    "triXope SafeLatentListDecoder": SafeLatentListDecoder,
    "triXope SaveLTXVideoLatent": SaveLTXVideoLatent,
    "triXope SaveWanVideoLatent": SaveWanVideoLatent,
    "triXope ScreenplayFormatterNode": ScreenplayFormatterNode,
    "triXope ShotLengthBufferNode": ShotLengthBufferNode,
    "triXope ShotLengthSelectorNode": ShotLengthSelectorNode,
    "triXope SmartConditioningSelector": SmartConditioningSelector,
    "triXope SmartReferenceFilter": SmartReferenceFilter,
    "triXope StorySelect": TriXopeStorySelect,
    "triXope StringCleaner": StringCleaner,
    "triXope text_switch": text_switch,
    "triXope TextSaver": triXopeTextSaver,
    "triXope ToggleFolderSaver": ToggleFolderSaver,
    "triXope UltimateSDUpscaleMiniMaxH3": UltimateSDUpscaleMiniMaxH3,
    "triXope UniversalLoaderMultiGPU": TriXopeUniversalLoaderMultiGPU,
    "triXope VAEDecodeBatched_Native": Trixope_VAEDecodeBatched_Native,
    "triXope Video_Upscale_To_Dimensions": Video_Upscale_To_Dimensions,
    "triXope VideoPreview": triXope_VideoPreview,
    "triXope VideoTimelineEditor": VideoTimelineEditor,
    "triXope WanDirectoryVideoDecoder": WanDirectoryVideoDecoder,
    "triXope WanVAEDecodeBatched": Trixope_WanVAEDecodeBatched,
    "triXope WanVAEEncodeBatched": Trixope_WanVAEEncodeBatched,
    "triXope WorkflowGroupHighlighter": WorkflowGroupHighlighter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "triXope AnyCast": "Δ Any Cast",
    "triXope AnySwitch": "Δ Any Switch",
    "triXope AnySwitchInversed": "Δ Any Switch Inversed",
    "triXope AudioFileSaver": "Δ Audio File Saver",
    "triXope AudioToTablature": "Δ Audio To Tablature 🎵",
    "triXope BatchOllamaWriter": "Δ Batch Ollama Writer",
    "triXope CharacterCraft": "Δ Character Craft",
    "triXope ColorFX": "Δ ColorFX 🎬",
    "triXope CurrentPromptDisplay": "Δ Prompt Counter",
    "triXope DirectoryFileSelectorGemini": "Δ Directory File Selector 📂 (Live)",
    "triXope EfficientImageBatchLoader": "Δ Efficient Image Batch Loader",
    "triXope EnsureDirectoryNode": "Δ Create Directory",
    "triXope File Copier": "Δ File Copier",
    "triXope FilenameFromDirectory": "Δ Filename Extractor",
    "triXope FilmAuteurH3": "Δ Film Auteur (H3)",
    "triXope FilmAuteur_LTX": "Δ Film Auteur (LTX)",
    "triXope FinalFrameSelector": "Δ Final Frame Selector",
    "triXope FirstFrameSelector": "Δ First Frame Selector",
    "triXope GroupMonitor": "Δ Group Monitor",
#    "triXope H3FL2VAandRef2VA": "Δ H3 Conditioner",
#    "triXope H3SlidingContextSampler": "Δ H3 Sampler",
    "triXope HistogramMatchHSV": "Δ HSV 🎨",
    "triXope Image Sequence Preview": "Δ Image Sequence Preview",
    "triXope ImageBatchSelector": "Δ Image Batch Filter",
    "triXope ImageUpscaleToSize": "Δ Upscale Image to Size (Model)",
    "triXope Load (Character) Image": "Δ Load (Character) Image",
    "triXope Load (Outfit) Image": "Δ Load (Outfit) Image",
    "triXope Load (Scene) Image": "Δ Load (Scene) Image",
    "triXope LoadAndConcatVideoLatentsFromDir": "Δ Load Video Latent",
    "triXope LoadText": "Δ Load Text",
    "triXope LTXV_ETA_Display": "Δ LTXV Real-Time ETA",
    "triXope LTXVDirectoryVideoDecoder": "Δ LTXV Directory Video Decoder",
    "triXope LTXVPostSliceAV": "Δ LTXV Trim A/V",
    "triXope MiniMaxH3AVLatentConcat": "Δ MiniMax H3 AV Latent Concat",
    "triXope MultiClipTrimmer": "Δ Multi-Clip Duration Trimmer",
    "triXope Pano360Rotate": "Δ Panorama 360 Rotate",
    "triXope PathCombinerNode": "Δ Path Combiner",
    "triXope PromptSplitSelect": "Δ Prompt Split & Select",
    "triXope RealtimeLoopTracker": "Δ Realtime Loop Tracker",
    "triXope SafeLatentListDecoder": "Δ Latent List Decoder",
    "triXope SaveLTXVideoLatent": "Δ Save LTXV Latent",
    "triXope SaveWanVideoLatent": "Δ Save WAN Latent",
    "triXope ScreenplayFormatterNode": "🎬 Δ H3 Screenwriter",
    "triXope ShotLengthBufferNode": "💾 Δ Shot Length Buffer / Holder",
    "triXope ShotLengthSelectorNode": "⏱️ Δ Shot Length Selector",
    "triXope SmartConditioningSelector": "Δ Smart Conditioning Selector",
    "triXope SmartReferenceFilter": "Δ Smart Reference Filter",
    "triXope StorySelect": "Δ Story Craft",
    "triXope StringCleaner": "Δ String Cleaner",
    "triXope text_switch": "Δ Text Switch",
    "triXope TextSaver": "Δ Text Saver",
    "triXope ToggleFolderSaver": "Δ Save Image",
    "triXope UltimateSDUpscaleMiniMaxH3": "Δ H3 Upscaler",
    "triXope UniversalLoaderMultiGPU": "Δ Universal Loader (MultiGPU)",
    "triXope VAEDecodeBatched_Native": "Δ VAE Decode Batched (Native)",
    "triXope Video_Upscale_To_Dimensions": "Δ Video Upscale To Dimensions",
    "triXope VideoPreview": "Δ Video Preview",
    "triXope VideoTimelineEditor": "Δ Video Timeline Editor",
    "triXope WanDirectoryVideoDecoder": "Δ Wan Directory Video Decoder",
    "triXope WanVAEDecodeBatched": "Δ VAE Decode Batched (WanVideo)",
    "triXope WanVAEEncodeBatched": "Δ VAE Encode Batched (WanVideo)",
    "triXope WorkflowGroupHighlighter": "Δ Group Highlighter",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Clean up imports
# Remove any new modules
modules_to_remove = []
for module in sys.modules:
    if module not in original_modules:
        modules_to_remove.append(module)
for module in modules_to_remove:
    del sys.modules[module]

# Restore original modules
sys.modules.update(original_imported_modules)

# Restore original sys.path
sys.path = original_sys_path