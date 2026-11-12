"""triXope ComfyUI nodes — Model and image loaders.

Every node class in this module is registered in the pack's top-level
``__init__.py`` and appears in the ComfyUI node menu under ``triXope/loaders``.
The module also holds the private helpers those nodes need.
"""

# Imports the moved code relied on from its original module headers.
import hashlib

from PIL import ImageOps

# --- moved from universal_loader.py -----------------------------------------
import torch
import logging
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management as mm
from .device_manager import get_available_devices, select_optimal_devices
from .autogpu_pytorch import resolve_clip_type, FULL_MODEL_TYPES
from .gguf_support import (
    is_gguf_file,
    get_combined_unet_list,
    get_combined_clip_list,
    resolve_unet_path,
    resolve_clip_path,
    load_unet_gguf_on_device,
    load_clip_gguf_on_device,
)

logger = logging.getLogger("AutoMultiGPU.triXope")


class TriXopeUniversalLoaderMultiGPU:
    """
    Universal All-in-One Multi-GPU Loader for triXope:
    Loads UNet/DiT, CLIP(s), Image/Video VAE, and an optional Audio VAE in a single node
    with automatic Multi-GPU balancing across model architectures.
    """
    @classmethod
    def INPUT_TYPES(s):
        from nodes import VAELoader
        unet_files = get_combined_unet_list()
        clip_files = ["none"] + get_combined_clip_list()
        vae_files = ["none"] + VAELoader.vae_list(VAELoader)
        devices = get_available_devices()

        return {
            "required": {
                "unet_name": (unet_files, {"tooltip": "Select UNet / DiT model (.safetensors or .gguf)"}),
                "clip_name1": (clip_files, {"default": clip_files[1] if len(clip_files) > 1 else "none", "tooltip": "First Text Encoder (.safetensors or .gguf)"}),
                "clip_name2": (clip_files, {"default": "none", "tooltip": "Optional Second Text Encoder (e.g. T5 / CLIP-G / Llama)"}),
                "vae_name": (vae_files, {"default": vae_files[1] if len(vae_files) > 1 else "none", "tooltip": "Primary VAE model (for Image/Video)"}),
                "audio_vae_name": (vae_files, {"default": "none", "tooltip": "Optional Audio VAE model"}),
                "model_type": (FULL_MODEL_TYPES, {"default": "auto", "tooltip": "Comprehensive Architecture Model Type"}),
                "distribution_mode": ([
                    "Auto: UNet GPU0 / Encoders GPU1",
                    "Auto: Balanced Free VRAM",
                    "Manual Override"
                ], {"default": "Auto: UNet GPU0 / Encoders GPU1"}),
            },
            "optional": {
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "float16", "bfloat16", "float32"], {"default": "default"}),
                "manual_unet_device": (devices, {"default": "cuda:0" if "cuda:0" in devices else "cpu"}),
                "manual_clip_device": (devices, {"default": "cuda:1" if "cuda:1" in devices else "cpu"}),
                "manual_vae_device": (devices, {"default": "cuda:1" if "cuda:1" in devices else "cpu"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "AUDIO_VAE")
    FUNCTION = "load_all"
    CATEGORY = "triXope/loaders"
    TITLE = "triXope Universal Loader (MultiGPU + Audio VAE)"
    DESCRIPTION = "Loads UNet, Dual/Single CLIP, Image/Video VAE, and optional Audio VAE with multi-GPU distribution."

    def load_all(self, unet_name, clip_name1, clip_name2, vae_name, audio_vae_name="none",
                 model_type="auto", distribution_mode="Auto: UNet GPU0 / Encoders GPU1",
                 weight_dtype="default", manual_unet_device="cuda:0",
                 manual_clip_device="cuda:1", manual_vae_device="cuda:1"):

        unet_dev, clip_dev, vae_dev = select_optimal_devices(
            mode=distribution_mode,
            manual_unet=manual_unet_device,
            manual_clip=manual_clip_device,
            manual_vae=manual_vae_device
        )

        logger.info(f"triXope Loader: Starting loading for {model_type} (UNet->{unet_dev}, CLIP->{clip_dev}, VAEs->{vae_dev})...")

        # 1. Load UNet
        unet_path = resolve_unet_path(unet_name)
        if is_gguf_file(unet_path):
            dequant = weight_dtype if weight_dtype in ["float16", "bfloat16", "float32"] else "default"
            model = load_unet_gguf_on_device(unet_path, target_device=unet_dev, dequant_dtype=dequant)
        else:
            model_options = {
                "load_device": torch.device(unet_dev),
                "offload_device": mm.unet_offload_device()
            }
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2
            elif weight_dtype == "float16":
                model_options["dtype"] = torch.float16
            elif weight_dtype == "bfloat16":
                model_options["dtype"] = torch.bfloat16
            elif weight_dtype == "float32":
                model_options["dtype"] = torch.float32
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

        # 2. Load CLIP
        clip = None
        c_type = resolve_clip_type(model_type, unet_name)

        if clip_name1 != "none" and clip_name2 != "none":
            p1 = resolve_clip_path(clip_name1)
            p2 = resolve_clip_path(clip_name2)
            if is_gguf_file(p1) or is_gguf_file(p2):
                clip = load_clip_gguf_on_device([p1, p2], clip_type=c_type, target_device=clip_dev)
            else:
                clip = comfy.sd.load_clip(
                    ckpt_paths=[p1, p2],
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=c_type,
                    model_options={"load_device": torch.device(clip_dev), "offload_device": torch.device("cpu")}
                )
        elif clip_name1 != "none":
            p1 = resolve_clip_path(clip_name1)
            if is_gguf_file(p1):
                clip = load_clip_gguf_on_device([p1], clip_type=c_type, target_device=clip_dev)
            else:
                clip = comfy.sd.load_clip(
                    ckpt_paths=[p1],
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=c_type,
                    model_options={"load_device": torch.device(clip_dev), "offload_device": torch.device("cpu")}
                )

        # 3. Load Image/Video VAE
        vae = None
        if vae_name != "none":
            vae_path = folder_paths.get_full_path("vae", vae_name)
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        # 4. Load Audio VAE
        audio_vae = None
        if audio_vae_name != "none":
            audio_vae_path = folder_paths.get_full_path("vae", audio_vae_name)
            audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(audio_vae_path))

        logger.info(f"✅ triXope Loader: Successfully loaded components (Image/Video VAE: {vae_name != 'none'}, Audio VAE: {audio_vae_name != 'none'})")
        return (model, clip, vae, audio_vae)

# --- moved from EfficientImageBatchLoader.py --------------------------------
import os
import torch
import numpy as np
from PIL import Image
import tempfile
import folder_paths
from tqdm import tqdm

class EfficientImageBatchLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "C:/path/to/your/frames"}),
                "start_index": ("INT", {"default": 1, "min": 0}),
                "frame_count": ("INT", {"default": 100, "min": 1}),
                "file_pattern": ("STRING", {"default": "frame_%04d.png"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_batch"
    CATEGORY = "triXope/loaders"

    def load_batch(self, directory_path, start_index, frame_count, file_pattern):
        image_paths = []
        for i in range(start_index, start_index + frame_count):
            filename = file_pattern % i
            path = os.path.join(directory_path, filename)
            if os.path.exists(path):
                image_paths.append(path)

        if not image_paths:
            raise FileNotFoundError("No images found with the specified pattern in the directory.")

        # Get dimensions from the first image
        with Image.open(image_paths[0]) as img:
            first_img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            h, w, c = first_img_np.shape
            dtype = first_img_np.dtype

        num_images = len(image_paths)
        
        # Create a temporary file to back the memory-mapped array
        # This file will be automatically deleted when it's closed
        temp_file = tempfile.NamedTemporaryFile(suffix='.mmap', delete=True)
        print(f"INFO: Creating memory-mapped file at: {temp_file.name}")
        
        # Create the memory-mapped array with the correct shape
        # This is an empty "virtual" array; no data is loaded yet.
        memmapped_array = np.memmap(temp_file.name, dtype=dtype, mode='w+', shape=(num_images, h, w, c))

        print("INFO: Populating memory-mapped array (this is fast disk I/O)...")
        # Efficiently write image data into the memory-mapped array
        for i, path in enumerate(tqdm(image_paths, desc="Mapping Images")):
            with Image.open(path) as img:
                img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
                memmapped_array[i] = img_np
        
        # 'Flush' ensures all writes are sent to the OS buffer, but not necessarily to disk
        memmapped_array.flush()

        print("INFO: Converting memory-map to tensor (no data copy)...")
        # Convert the numpy memmap to a torch tensor.
        # This is extremely fast as it shares the underlying memory without copying.
        tensor = torch.from_numpy(memmapped_array)

        return (tensor,)

# --- moved from nodes.py:image loaders --------------------------------------
class ImageCharacterLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        exclude_folders = ["clipspace", "folder_to_exclude2"]
        file_list = []

        for root, dirs, files in os.walk(input_dir + "/_film/input/characters"):
            dirs[:] = [d for d in dirs if d not in exclude_folders]
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    file_path = os.path.relpath(os.path.join(root, file), start=input_dir).replace("\\", "/")
                    file_list.append(file_path)

        return {"required": {"image": (sorted(file_list), {"image_upload": True})}}

    CATEGORY = "triXope/loaders"
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        filename_with_ext = os.path.basename(image_path)
        filename, ext = os.path.splitext(filename_with_ext)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, filename)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

class ImageOutfitLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        exclude_folders = ["clipspace", "folder_to_exclude2"]
        file_list = []

        for root, dirs, files in os.walk(input_dir + "/_film/input/outfits"):
            dirs[:] = [d for d in dirs if d not in exclude_folders]
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    file_path = os.path.relpath(os.path.join(root, file), start=input_dir).replace("\\", "/")
                    file_list.append(file_path)

        return {"required": {"image": (sorted(file_list), {"image_upload": True})}}

    CATEGORY = "triXope/loaders"
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        filename_with_ext = os.path.basename(image_path)
        filename, ext = os.path.splitext(filename_with_ext)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, filename)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

class ImageSceneLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        exclude_folders = ["clipspace", "folder_to_exclude2"]
        file_list = []

        for root, dirs, files in os.walk(input_dir + "/_film/input/scenes"):
            dirs[:] = [d for d in dirs if d not in exclude_folders]
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    file_path = os.path.relpath(os.path.join(root, file), start=input_dir).replace("\\", "/")
                    file_list.append(file_path)

        return {"required": {"image": (sorted(file_list), {"image_upload": True})}}

    CATEGORY = "triXope/loaders"
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        filename_with_ext = os.path.basename(image_path)
        filename, ext = os.path.splitext(filename_with_ext)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, filename)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

__all__ = [
    "TriXopeUniversalLoaderMultiGPU",
    "EfficientImageBatchLoader",
    "ImageCharacterLoader",
    "ImageOutfitLoader",
    "ImageSceneLoader",
]
