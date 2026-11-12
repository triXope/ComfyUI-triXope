import os
import sys
import torch
import logging
import inspect
import importlib
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management as mm

logger = logging.getLogger("AutoMultiGPU")

# Check if ComfyUI-GGUF is available
_GGUF_AVAILABLE = False
_gguf_module = None
_gguf_sd_loader = None
_gguf_clip_loader = None
_GGMLOps = None
_GGUFModelPatcher = None

try:
    for mod_name in ["custom_nodes.ComfyUI-GGUF_KREA-2", "ComfyUI-GGUF_KREA-2", "custom_nodes.ComfyUI-GGUF", "ComfyUI-GGUF"]:
        try:
            _gguf_module = importlib.import_module(mod_name)
            break
        except ImportError:
            pass

    if _gguf_module is not None:
        loader_mod = importlib.import_module(f"{_gguf_module.__name__}.loader")
        ops_mod = importlib.import_module(f"{_gguf_module.__name__}.ops")
        nodes_mod = importlib.import_module(f"{_gguf_module.__name__}.nodes")

        _gguf_sd_loader = loader_mod.gguf_sd_loader
        _gguf_clip_loader = loader_mod.gguf_clip_loader
        _GGMLOps = ops_mod.GGMLOps
        _GGUFModelPatcher = nodes_mod.GGUFModelPatcher
        _GGUF_AVAILABLE = True
        logger.info("ComfyUI-GGUF integration activated for AutoMultiGPU!")
except Exception as e:
    logger.warning(f"GGUF support note: {e}")

def is_gguf_available():
    return _GGUF_AVAILABLE

def is_gguf_file(filename_or_path):
    if not filename_or_path:
        return False
    return filename_or_path.lower().endswith(".gguf")

def get_combined_unet_list():
    """Returns sorted list of all UNet / DiT models (safetensors + gguf)."""
    files = set()
    for cat in ["diffusion_models", "unet", "unet_gguf"]:
        try:
            names = folder_paths.get_filename_list(cat)
            files.update(names)
        except Exception:
            pass
    return sorted(list(files))

def get_combined_clip_list():
    """Returns sorted list of all CLIP / Text Encoders (safetensors + gguf)."""
    files = set()
    for cat in ["text_encoders", "clip", "clip_gguf"]:
        try:
            names = folder_paths.get_filename_list(cat)
            files.update(names)
        except Exception:
            pass
    return sorted(list(files))

def resolve_unet_path(unet_name):
    """Finds the absolute path of a unet file across diffusion_models, unet, unet_gguf."""
    for cat in ["diffusion_models", "unet", "unet_gguf"]:
        try:
            p = folder_paths.get_full_path(cat, unet_name)
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
    return folder_paths.get_full_path_or_raise("diffusion_models", unet_name)

def resolve_clip_path(clip_name):
    """Finds the absolute path of a clip file across text_encoders, clip, clip_gguf."""
    for cat in ["text_encoders", "clip", "clip_gguf"]:
        try:
            p = folder_paths.get_full_path(cat, clip_name)
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
    return folder_paths.get_full_path_or_raise("text_encoders", clip_name)

def load_unet_gguf_on_device(unet_path, target_device, dequant_dtype="default", patch_dtype="default", patch_on_device=False, cpu_offload=True):
    """Loads a GGUF UNet / DiT model with custom ops placed on target_device and dynamic CPU offloading."""
    if not _GGUF_AVAILABLE:
        raise RuntimeError("ComfyUI-GGUF is required to load .gguf models. Please ensure ComfyUI-GGUF is installed.")

    ops = _GGMLOps()
    if dequant_dtype in ("default", None):
        ops.Linear.dequant_dtype = None
    elif dequant_dtype == "target":
        ops.Linear.dequant_dtype = "target"
    else:
        ops.Linear.dequant_dtype = getattr(torch, dequant_dtype, None)

    if patch_dtype in ("default", None):
        ops.Linear.patch_dtype = None
    elif patch_dtype == "target":
        ops.Linear.patch_dtype = "target"
    else:
        ops.Linear.patch_dtype = getattr(torch, patch_dtype, None)

    sd, extra = _gguf_sd_loader(unet_path)

    kwargs = {}
    valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
    if "metadata" in valid_params:
        kwargs["metadata"] = extra.get("metadata", {})

    dev_obj = torch.device(target_device)
    offload_dev = torch.device("cpu") if cpu_offload else dev_obj
    
    model_options = {
        "custom_operations": ops,
        "load_device": dev_obj,
        "offload_device": offload_dev
    }

    model = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options=model_options, **kwargs
    )
    if model is None:
        raise RuntimeError(f"Could not detect model architecture for GGUF: {unet_path}")

    model = _GGUFModelPatcher.clone(model)
    model.load_device = dev_obj
    model.offload_device = offload_dev
    model.patch_on_device = patch_on_device and not cpu_offload
    return model

def load_clip_gguf_on_device(clip_paths, clip_type, target_device, cpu_offload=True):
    """Loads GGUF / Safetensors CLIP data and creates ModelPatcher on target_device with dynamic CPU offload."""
    if not _GGUF_AVAILABLE:
        raise RuntimeError("ComfyUI-GGUF is required to load .gguf CLIP files.")

    clip_data = []
    has_gguf = False
    for p in clip_paths:
        if is_gguf_file(p):
            sd = _gguf_clip_loader(p)
            has_gguf = True
        else:
            sd = comfy.utils.load_torch_file(p, safe_load=True)
            if "scaled_fp8" in sd:
                raise NotImplementedError(f"Mixing scaled FP8 with GGUF is not supported: {p}")
        clip_data.append(sd)

    dev_obj = torch.device(target_device)
    offload_dev = torch.device("cpu") if cpu_offload else dev_obj
    
    model_options = {
        "load_device": dev_obj,
        "offload_device": offload_dev
    }
    if has_gguf:
        model_options["custom_operations"] = _GGMLOps

    clip = comfy.sd.load_text_encoder_state_dicts(
        clip_type=clip_type,
        state_dicts=clip_data,
        model_options=model_options,
        embedding_directory=folder_paths.get_folder_paths("embeddings")
    )
    if has_gguf and hasattr(clip, "patcher") and clip.patcher is not None:
        clip.patcher = _GGUFModelPatcher.clone(clip.patcher)
        clip.patcher.load_device = dev_obj
        clip.patcher.offload_device = offload_dev

    return clip
