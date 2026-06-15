import torch
import math
import comfy.utils
import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import comfy.sample
import comfy.patcher_extension
import re
import latent_preview
import urllib.request
import json
import base64
import requests
import os
import cv2
import folder_paths
from io import BytesIO
from PIL import Image
import numpy as np
import torchaudio
import types
import logging
import av
import random

# --- COLORFX INJECTIONS ---
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageChops, ImageDraw
import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from scipy.interpolate import RegularGridInterpolator
    SCIPY_AVAILABLE = True
    print("INFO: Scipy found. LUT application will be optimized.")
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: Scipy not found. LUT application will use a slower pixel-by-pixel method if enabled.")

try:
    dir_luts = os.path.join(folder_paths.models_dir, "luts")
    os.makedirs(dir_luts, exist_ok=True)
    if "luts" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["luts"] = ([dir_luts], {".cube"})
    else:
        folder_paths.folder_names_and_paths["luts"][1].add(".cube")
        if dir_luts not in folder_paths.folder_names_and_paths["luts"][0]:
             folder_paths.folder_names_and_paths["luts"][0].append(dir_luts)
except Exception as e:
    pass

def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
# --------------------------

from server import PromptServer
from comfy.ldm.modules.attention import wrap_attn, optimized_attention, attention_pytorch

from fractions import Fraction
try:
    from comfy_api.latest import InputImpl, Types
except ImportError:
    InputImpl = None
    Types = None

# ==========================================
# CUSTOM TRIXOPE DIRECTORY SETUP
# ==========================================
current_dir = os.path.dirname(os.path.realpath(__file__))
trixope_facerestore_dir = os.path.join(current_dir, "facerestore_models")
os.makedirs(trixope_facerestore_dir, exist_ok=True)
folder_paths.folder_names_and_paths["trixope_facerestore"] = ([trixope_facerestore_dir], folder_paths.supported_pt_extensions)

def get_trixope_facerestore_models():
    models = folder_paths.get_filename_list("trixope_facerestore")
    return models if models else ["None"]

# ==========================================
# OLLAMA API FETCH (Builds the Dropdown at Boot)
# ==========================================
def get_ollama_models():
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as response:
            data = json.loads(response.read().decode('utf-8'))
            models = [model['name'] for model in data.get('models', [])]
            return models if models else ["llama3.2-vision:latest", "llava:latest"]
    except Exception:
        return ["llama3.2-vision:latest", "(Start Ollama & Restart ComfyUI)"]

OLLAMA_MODELS = get_ollama_models()

# ==========================================
# SAGE ATTENTION CORE
# ==========================================
sageattn_modes = ["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3", "sageattn3_per_block_mean"]

def get_sage_func(sage_attention, allow_compile=False):
    import logging
    logging.info(f"Using sage attention mode: {sage_attention}")
    from sageattention import sageattn
    if sage_attention == "auto":
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda++":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout)
    elif "sageattn3" in sage_attention:
        from sageattn3 import sageattn3_blackwell
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD", **kwargs):
            q, k, v = [x.transpose(1, 2) if tensor_layout == "NHD" else x for x in (q, k, v)]
            out = sageattn3_blackwell(q, k, v, is_causal=is_causal, attn_mask=attn_mask, per_block_mean=(sage_attention == "sageattn3_per_block_mean"))
            return out.transpose(1, 2) if tensor_layout == "NHD" else out

    if not allow_compile:
        sage_func = torch.compiler.disable()(sage_func)

    @wrap_attn
    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        if kwargs.get("low_precision_attention", True) is False:
            return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        if skip_reshape:
            b, _, _, dim_head = q.shape
            tensor_layout="HND"
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(
                lambda t: t.view(b, -1, heads, dim_head),
                (q, k, v),
            )
            tensor_layout="NHD"
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        else:
            if skip_output_reshape:
                out = out.transpose(1, 2)
            else:
                out = out.reshape(b, -1, heads * dim_head)
        return out
    return attention_sage

# ==========================================
# CHUNK FFN CORE
# ==========================================
def ffn_chunked_forward(self_module, x):
    if x.shape[1] > self_module.dim_threshold:
        chunk_size = x.shape[1] // self_module.num_chunks
        for i in range(self_module.num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self_module.num_chunks - 1 else x.shape[1]
            x[:, start_idx:end_idx] = self_module.net(x[:, start_idx:end_idx])
        return x
    else:
        return self_module.net(x)

class LTXVffnChunkPatch:
    def __init__(self, num_chunks, dim_threshold=4096):
        self.num_chunks = num_chunks
        self.dim_threshold = dim_threshold
    def __get__(self, obj, objtype=None):
        def wrapped_forward(self_module, *args, **kwargs):
            self_module.num_chunks = self.num_chunks
            self_module.dim_threshold = self.dim_threshold
            return ffn_chunked_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_forward, obj)

# ==========================================
# NORMALIZED ATTENTION GUIDANCE (NAG) CORE
# ==========================================
def _compute_attention(self, query, context, attn_precision=None, transformer_options={}):
    k = self.k_norm(self.to_k(context)).to(query.dtype)
    v = self.to_v(context).to(query.dtype)
    x = comfy.ldm.modules.attention.optimized_attention(query, k, v, heads=self.heads, attn_precision=attn_precision, transformer_options=transformer_options).flatten(2)
    del k, v
    return x

def nag_attention(self, query, context_positive, nag_context, attn_precision=None, transformer_options={}):
    x_positive = _compute_attention(self, query, context_positive, attn_precision, transformer_options)
    x_negative = _compute_attention(self, query, nag_context, attn_precision, transformer_options)
    return x_positive, x_negative

def normalized_attention_guidance(self, x_positive, x_negative):
    if self.inplace:
        nag_guidance = x_negative.mul_(self.nag_scale - 1).neg_().add_(x_positive, alpha=self.nag_scale)
    else:
        nag_guidance = x_positive * self.nag_scale - x_negative * (self.nag_scale - 1)

    del x_negative

    norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
    norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True)

    scale = norm_guidance / norm_positive
    torch.nan_to_num_(scale, nan=10.0)
    mask = scale > self.nag_tau
    del scale

    adjustment = (norm_positive * self.nag_tau) / (norm_guidance + 1e-7)
    del norm_positive, norm_guidance

    nag_guidance.mul_(torch.where(mask, adjustment, 1.0))
    del mask, adjustment

    if self.inplace:
        nag_guidance.sub_(x_positive).mul_(self.nag_alpha).add_(x_positive)
    else:
        nag_guidance = nag_guidance * self.nag_alpha + x_positive * (1 - self.nag_alpha)
    del x_positive

    return nag_guidance

def ltxv_crossattn_forward_nag(self, x, context, mask=None, transformer_options={}, **kwargs):
    if context.shape[0] == 1:
        x_pos, context_pos = x, context
        x_neg, context_neg = None, None
    else:
        x_pos, x_neg = torch.chunk(x, 2, dim=0)
        context_pos, context_neg = torch.chunk(context, 2, dim=0)

    q_pos = self.q_norm(self.to_q(x_pos))
    del x_pos

    x_positive, x_negative = nag_attention(self, q_pos, context_pos, self.nag_context, attn_precision=self.attn_precision, transformer_options=transformer_options)
    del context_pos, q_pos

    x_pos_out = normalized_attention_guidance(self, x_positive, x_negative)
    del x_positive, x_negative

    if x_neg is not None and context_neg is not None:
        q_neg = self.q_norm(self.to_q(x_neg))
        k_neg = self.k_norm(self.to_k(context_neg))
        v_neg = self.to_v(context_neg)

        x_neg_out = comfy.ldm.modules.attention.optimized_attention(q_neg, k_neg, v_neg, heads=self.heads, attn_precision=self.attn_precision, transformer_options=transformer_options)
        out = torch.cat([x_pos_out, x_neg_out], dim=0)
    else:
        out = x_pos_out

    if self.to_gate_logits is not None:
        gate_logits = self.to_gate_logits(x)  
        b, t, _ = out.shape
        out = out.view(b, t, self.heads, self.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits) 
        out = out * gates.unsqueeze(-1)
        out = out.view(b, t, self.heads * self.dim_head)

    return self.to_out(out)

class LTXVCrossAttentionPatch:
    def __init__(self, context, nag_scale, nag_alpha, nag_tau, inplace=True):
        self.nag_context = context
        self.nag_scale = nag_scale
        self.nag_alpha = nag_alpha
        self.nag_tau = nag_tau
        self.inplace = inplace

    def __get__(self, obj, objtype=None):
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.nag_context = self.nag_context
            self_module.nag_scale = self.nag_scale
            self_module.nag_alpha = self.nag_alpha
            self_module.nag_tau = self.nag_tau
            self_module.inplace = self.inplace
            return ltxv_crossattn_forward_nag(self_module, *args, **kwargs)
        return types.MethodType(wrapped_attention, obj)


# ==========================================
# HELPER DECODE FUNCTIONS
# ==========================================
def compute_chunk_boundaries(chunk_start: int, temporal_tile_length: int, temporal_overlap: int, total_latent_frames: int):
    if chunk_start == 0:
        chunk_end = min(chunk_start + temporal_tile_length, total_latent_frames)
        overlap_start = chunk_start
    else:
        overlap_start = max(1, chunk_start - temporal_overlap - 1)
        extra_frames = chunk_start - overlap_start
        chunk_end = min(chunk_start + temporal_tile_length - extra_frames, total_latent_frames)
    return overlap_start, chunk_end

def calculate_temporal_output_boundaries(overlap_start: int, time_scale_factor: int, tile_out_frames: int):
    out_t_start = 1 + overlap_start * time_scale_factor
    out_t_end = out_t_start + tile_out_frames
    return out_t_start, out_t_end

class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent.get("batch_index", None)
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)

def encode_single_frame(output_file, image_array: np.ndarray, crf):
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height = image_array.shape[0]
        stream.width = image_array.shape[1]
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()

def decode_single_frame(video_file):
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")

def preprocess_compression(image: torch.Tensor, crf=29):
    if crf == 0:
        return image
    image_array = (image[:(image.shape[0] // 2) * 2, :(image.shape[1] // 2) * 2] * 255.0).byte().cpu().numpy()
    with BytesIO() as output_file:
        encode_single_frame(output_file, image_array, crf)
        video_bytes = output_file.getvalue()
    with BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    tensor = torch.tensor(image_array, dtype=image.dtype, device=image.device) / 255.0
    return tensor

RESOLUTION_OPTIONS = [
    "--- 𝟭:𝟭 𝗔𝘀𝗽𝗲𝗰𝘁 𝗥𝗮𝘁𝗶𝗼 (𝗦𝗾𝘂𝗮𝗿𝗲) ---",
    "512\u00A0 × 512\u00A0 — Fast low-VRAM generation",
    "768\u00A0 × 768\u00A0 — Balanced resolution sweet spot",
    "1024 × 1024 — Standard high-definition square layout",
    "1536 × 1536 — Pro fidelity upscale target",
    "--- 𝟰:𝟯 𝗔𝘀𝗽𝗲𝗰𝘁 𝗥𝗮𝘁𝗶𝗼 (𝗦𝘁𝗮𝗻𝗱𝗮𝗿𝗱 𝗟𝗮𝗻𝗱𝘀𝗰𝗮𝗽𝗲) ---",
    "512\u00A0 × 384\u00A0 — Low-resolution draft option",
    "1024 × 768\u00A0 — Exact native classic frame",
    "1536 × 1152 — Sharp high-resolution landscape grid",
    "2048 × 1536 — Ultra-high density presentation resolution",
    "--- 𝟯:𝟰 𝗔𝘀𝗽𝗲𝗰𝘁 𝗥𝗮𝘁𝗶𝗼 (𝗦𝘁𝗮𝗻𝗱𝗮𝗿𝗱 𝗣𝗼𝗿𝘁𝗿𝗮𝗶𝘁) ---",
    "384\u00A0 × 512\u00A0 — Low-resolution vertical layout",
    "768\u00A0 × 1024 — Exact vertical presentation shape",
    "1152 × 1536 — High-definition mobile portrait preview",
    "1536 × 2048 — Maximum density vertical layout",
    "--- 𝟭𝟲:𝟵 𝗔𝘀𝗽𝗲𝗰𝘁 𝗥𝗮𝘁𝗶𝗼 (𝗪𝗶𝗱𝗲𝘀𝗰𝗿𝗲𝗲𝗻) ---",
    "1152 × 640\u00A0 — High-speed widescreen draft approximation",
    "1792 × 1024 — Close standard HD variation",
    "2048 × 1152 — Exact 16:9 math configuration",
    "3840 × 2176 — Immersive 4K UHD approximation",
    "--- 𝟵:𝟭𝟲 𝗔𝘀𝗽𝗲𝗰𝘁 𝗥𝗮𝘁𝗶𝗼 (𝗩𝗲𝗿𝘁𝗶𝗰𝗮𝗹 𝗪𝗶𝗱𝗲𝘀𝗰𝗿𝗲𝗲𝗻) ---",
    "640\u00A0 × 1152 — Fast vertical mobile approximation",
    "1024 × 1792 — Optimized social reel layout",
    "1152 × 2048 — Exact native 9:16 configuration",
    "2176 × 3840 — Ultra-high definition mobile layout",
    "--- 𝟮𝟭:𝟵 𝗔𝘀𝗽𝗲𝗰𝘁 𝗥𝗮𝘁𝗶𝗼 (𝗖𝗶𝗻𝗲𝗺𝗮𝘁𝗶𝗰 𝗨𝗹𝘁𝗿𝗮𝘄𝗶𝗱𝗲) ---",
    "896\u00A0 × 384\u00A0 — Low-VRAM horizontal cinematic draft",
    "1792 × 768\u00A0 — Exact cinematic widescreen resolution",
    "2688 × 1152 — Immersive high-definition panoramic target",
    "3584 × 1536 — Maximum resolution multi-window layout",
    "--- 𝟵:𝟮𝟭 𝗔𝘀𝗽𝗲𝗰𝘁 𝗥𝗮𝘁𝗶𝗼 (𝗩𝗲𝗿𝘁𝗶𝗰𝗮𝗹 𝗨𝗹𝘁𝗿𝗮𝘄𝗶𝗱𝗲) ---",
    "384\u00A0 × 896\u00A0 — Tall smartphone display layout",
    "768\u00A0 × 1792 — Exact vertical cinematic layout",
    "1152 × 2688 — Ultra-tall narrow video layout",
    "1536 × 3584 — Maximum resolution panoramic portrait"
]

class FilmAuteur_LTX:

    @classmethod
    def INPUT_TYPES(cls):
        lut_files = ["None"]
        try:
            raw_files = folder_paths.get_filename_list("luts")
            if raw_files:
                 lut_files.extend([f for f in raw_files if f.lower().endswith('.cube')])
        except Exception as e: pass
        
        sampler_names = comfy.samplers.SAMPLER_NAMES
        primary_default = "euler_ancestral" if "euler_ancestral" in sampler_names else ("euler" if "euler" in sampler_names else sampler_names[0])
        upsample_default = "euler_cfg_pp" if "euler_cfg_pp" in sampler_names else ("euler" if "euler" in sampler_names else sampler_names[0])

        return {
            "required": {
                # --- Input/Setup ---
                "clip": ("CLIP",),
                "video_vae": ("VAE", {"tooltip": "The LTXV Video VAE model."}),
                "audio_vae": ("VAE", {"tooltip": "The LTXV Audio VAE model."}),
                "model1_primary": ("MODEL", {"tooltip": "The primary LTXV Model (will be patched if ID-LoRA is active)."}),

                # --- GROUP: Mode Activation ---
                "grp_mode": (["▼ Mode Activation"], {}),
                "primary_sampling": ("BOOLEAN", {"default": True}),
                "spatial_upscale": ("BOOLEAN", {"default": True}),
                "temporal_upscale": ("BOOLEAN", {"default": True}),
                "restore_faces": ("BOOLEAN", {"default": True}),
                "enable_colorfx": ("BOOLEAN", {"default": True}),
                "video_mode": (["text-to-video", "image-to-video", "reference-to-video"], {"default": "text-to-video"}),
                "image_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "img_compression": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "audio_select": (["internal", "source", "reference"], {"default": "internal"}),
                "identity_guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),

                # --- GROUP: Prompting ---
                "grp_prompting": (["▼ Prompting"], {}),
                "character_descriptions": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "location_description": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "scene_descriptions": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "use_ollama": (["disabled", "enhance prompt", "lyrics-to-script"], {"default": "disabled"}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ollama_model": (OLLAMA_MODELS,),

                # --- GROUP: Specs ---
                "grp_specs": (["▼ Video Specs"], {}),
                "seed_number": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "control_before_generate": (["randomize", "fixed", "increment", "decrement"], {"default": "randomize"}),
                "target_resolution": (RESOLUTION_OPTIONS, {"default": "1792 × 768\u00A0 — Exact cinematic widescreen resolution"}),
                "length_in_seconds": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),

                # --- GROUP: Primary Sampling ---
                "grp_sampling": (["▼ Primary Sampling"], {}),
                "primary_sampler_name": (sampler_names, {"default": primary_default}),
                "primary_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "primary_steps": ("STRING", {"multiline": False, "default": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.881203, 0.863321, 0.841251, 0.820089, 0.655, 0.381875, 0.0"}),
                "eta": ("FLOAT", {"default": 0.95, "min": -100.0, "max": 100.0, "step": 0.01, "round": False}),
                "bongmath": ("BOOLEAN", {"default": True}),
                "enable_nag": ("BOOLEAN", {"default": True}),

                # --- GROUP: Upscale & Refine ---
                "grp_refinement": (["▼ Upscale & Refine"], {}),
                "spatial_passes": ("INT", {"default": 1, "min": 1, "max": 2}),
                "spatial_sampler": (sampler_names, {"default": upsample_default}),
                "spatial_cfg": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "spatial_sigmas": ("STRING", {"multiline": False, "default": "0.55, 0.35, 0.15, 0.0"}),
                "temporal_denoise": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 1.0, "step": 0.01}),
                "facerestore_model": (get_trixope_facerestore_models(), {}),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"], {}),
                "codeformer_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "face_restore_color_match": ("BOOLEAN", {"default": True}),
                "face_restore_edge_blur": ("BOOLEAN", {"default": True}),
                "face_restore_blend": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),

                # --- GROUP: Color FX (Master Parent) ---
                "grp_cfx_main": (["▼ Color FX"], {}),
                "enable_color_correction": ("BOOLEAN", {"default": True}),
                "enable_lut_processing": ("BOOLEAN", {"default": True}),
                "enable_enhancements": ("BOOLEAN", {"default": True}),
                "enable_blur_effects": ("BOOLEAN", {"default": True}),
                "enable_stylistic_effects": ("BOOLEAN", {"default": True}),
                
                # --- SUBGROUP: Color FX (Correction) ---
                "grp_cfx_color": (["▼ Color Correction"], {}),
                "hdr_intensity": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "shadow_intensity": ("FLOAT", {"default": 0.10, "min": 0.00, "max": 2.00, "step": 0.01}),
                "highlight_intensity": ("FLOAT", {"default": 0.20, "min": 0.00, "max": 2.00, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.00, "min": 0.10, "max": 5.00, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 3.00, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 0.75, "min": 0.00, "max": 3.00, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 0.90, "min": 0.00, "max": 3.00, "step": 0.01}),
                "enhance_color": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 3.00, "step": 0.01}),

                # --- SUBGROUP: Color FX (LUTs) ---
                "grp_cfx_lut": (["▼ LUT Processing"], {}),
                "lut_name": (lut_files, {"default": "None"}),
                "lut_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lut_log_process": ("BOOLEAN", {"default": True}),

                # --- SUBGROUP: Color FX (Enhancements) ---
                "grp_cfx_enhancements": (["▼ Enhancements"], {}),
                "sharpness": ("FLOAT", {"default": 3.0, "min": -2.0, "max": 5.0, "step": 0.1}),
                "edge_enhance_strength": ("FLOAT", {"default": 0.10, "min": 0.00, "max": 1.00, "step": 0.01}),
                "detail_enhance_strength": ("FLOAT", {"default": 0.20, "min": 0.00, "max": 1.00, "step": 0.01}),

                # --- SUBGROUP: Color FX (Blur) ---
                "grp_cfx_blur": (["▼ Blur Effects"], {}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "gaussian_blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.1}),
                "radial_blur_strength": ("FLOAT", {"default": 32.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "radial_blur_center_x": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "radial_blur_center_y": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "radial_blur_focus_spread": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "radial_blur_steps": ("INT", {"default": 10, "min": 1, "max": 32, "step": 1}),

                # --- SUBGROUP: Color FX (Stylistic) ---
                "grp_cfx_stylistic": (["▼ Stylistic Effects"], {}),
                "chromatic_aberration_r_x": ("FLOAT", {"default": 1.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_aberration_r_y": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_aberration_b_x": ("FLOAT", {"default": -1.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_aberration_b_y": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_blur_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "simple_film_grain_intensity": ("FLOAT", {"default": 0.04, "min": 0.00, "max": 1.00, "step": 0.01}),
                "simple_film_grain_monochrome": ("BOOLEAN", {"default": True}),
                "scanline_intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0}),
                "vignette_intensity": ("FLOAT", {"default": 0.40, "min": 0.00, "max": 1.00, "step": 0.01}),
                "vignette_center_x": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "vignette_center_y": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "soft_light_opacity": ("FLOAT", {"default": 0.30, "min": 0.00, "max": 1.00, "step": 0.01}),
                "soft_light_blur_radius": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1}),

                # --- GROUP: Performance ---
                "grp_performance": (["▼ Performance"], {}),
                "enable_fp16_accumulation": ("BOOLEAN", {"default": True}),
                "sage_attention": (sageattn_modes, {"default": "auto"}),
                "autoregressive_chunking": ("BOOLEAN", {"default": True}),
                "chunk_size_seconds": ("FLOAT", {"default": 20.0, "min": 5.0, "max": 300.0, "step": 1.0}),
                "context_window_seconds": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 300.0, "step": 1.0}),
                "chunks_feedforward": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "clear_models_and_cache": ("BOOLEAN", {"default": True}),

                # --- GROUP: Preview ---
                "grp_preview": (["▼ Preview"], {}),
                "enable_preview": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model2_spatial": ("MODEL", {"tooltip": "Optional model for upsample stages 2 and 3. If disconnected, model1_primary is used.", "forceInput": True}),
                "spatial_upscaler": ("LATENT_UPSCALE_MODEL", {"tooltip": "Connect the LTXV Spatial Upscale model here to upsample the video latent by 2x.", "forceInput": True}),
                "model3_temporal": ("MODEL", {"tooltip": "Optional model for the temporal upscaler. If disconnected, model1_primary is used.", "forceInput": True}),
                "temporal_upscaler": ("LATENT_UPSCALE_MODEL", {"tooltip": "Connect the LTXV Temporal Upscale model here to double the framerate.", "forceInput": True}),
                "positive": ("CONDITIONING", {"tooltip": "Optional override for the positive text conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Optional override for the negative text conditioning."}),
                "image(s)": ("IMAGE", {"tooltip": "Input image batch for image-to-video or reference-to-video."}),
                "audio": ("AUDIO", {"tooltip": "Connect audio here to encode it directly or use it as a voice reference for ID-LoRA."}),
                "latent_override": ("LATENT", {"tooltip": "Optional latent override. Bypasses generation if primary_sampling is off, or acts as the base for Video-to-Video if on."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("STRING", "CONDITIONING", "CONDITIONING", "LATENT", "LATENT", "LATENT", "VIDEO", "IMAGE", "AUDIO", "FLOAT", "INT", "ETA_TRACKER")
    RETURN_NAMES = ("text_prompt(s)", "positive", "negative", "av_latent", "video_latent", "audio_latent", "video", "images", "audio", "fps", "ref_frame_count", "real_time_eta")
    FUNCTION = "process"
    CATEGORY = "triXope"

    def _read_cube_file(self, filepath):
        lines = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'): lines.append(line)
        lut_size = 0
        domain_min, domain_max = np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)
        table_data_lines = []
        for line in lines:
            if line.startswith('LUT_3D_SIZE'): lut_size = int(line.split()[-1])
            elif line.startswith('DOMAIN_MIN'): domain_min = np.array([float(x) for x in line.split()[1:]], dtype=np.float32)
            elif line.startswith('DOMAIN_MAX'): domain_max = np.array([float(x) for x in line.split()[1:]], dtype=np.float32)
            elif len(line.split()) == 3 and not line.startswith('TITLE'): table_data_lines.append(list(map(float, line.split())))
        return np.array(table_data_lines, dtype=np.float32).reshape(lut_size, lut_size, lut_size, 3), lut_size, domain_min, domain_max

    def _apply_lut_to_image_scipy(self, image_np_rgb_0_1, lut_data):
        lut_table, lut_size, _, _ = lut_data
        r_axis = g_axis = b_axis = np.linspace(0, 1, lut_size)
        interp_R = RegularGridInterpolator((b_axis, g_axis, r_axis), lut_table[..., 0], bounds_error=False)
        interp_G = RegularGridInterpolator((b_axis, g_axis, r_axis), lut_table[..., 1], bounds_error=False)
        interp_B = RegularGridInterpolator((b_axis, g_axis, r_axis), lut_table[..., 2], bounds_error=False)
        pts = image_np_rgb_0_1[..., [2, 1, 0]].reshape(-1, 3)
        return np.clip(np.stack([interp_R(pts), interp_G(pts), interp_B(pts)], axis=-1).reshape(image_np_rgb_0_1.shape), 0.0, 1.0)

    def _apply_lut_effect(self, pil_image, lut_name, strength, log_process):
        if strength == 0.0 or lut_name == "None": return pil_image
        lut_path = folder_paths.get_full_path("luts", lut_name)
        if not lut_path or not SCIPY_AVAILABLE: return pil_image
        lut_data = self._read_cube_file(lut_path)
        img_np = np.array(pil_image, dtype=np.float32) / 255.0
        orig = img_np.copy()
        if log_process: img_np = np.power(np.clip(img_np, 1e-5, 1.0), 1.0 / 2.2)
        out_np = self._apply_lut_to_image_scipy(img_np, lut_data)
        if log_process: out_np = np.power(np.clip(out_np, 0.0, 1.0), 2.2)
        blended = np.clip((1.0 - strength) * orig + strength * out_np, 0.0, 1.0)
        return Image.fromarray((blended * 255).astype(np.uint8))

    def _apply_shadows_highlights(self, pil_image, shadow_adj, highlight_adj, hdr_intensity):
        if shadow_adj == 0.0 and highlight_adj == 0.0 and hdr_intensity == 1.0: return pil_image
        img_array = np.array(pil_image, dtype=np.float32) / 255.0
        if hdr_intensity != 1.0:
            mean = np.mean(img_array)
            img_array = mean + (img_array - mean) * (1.0 + (hdr_intensity - 1.0) * 0.5) 
        lum = np.clip(0.299 * img_array[..., 0] + 0.587 * img_array[..., 1] + 0.114 * img_array[..., 2], 0.0, 1.0)[..., np.newaxis]
        if shadow_adj > 0.0: img_array += (1.0 - lum) * shadow_adj
        if highlight_adj > 0.0: img_array -= lum * highlight_adj
        return Image.fromarray((np.clip(img_array, 0.0, 1.0) * 255).astype(np.uint8))

    def _apply_color_enhancements(self, pil_image, b, c, s, ec):
        if b != 1.: pil_image = ImageEnhance.Brightness(pil_image).enhance(b)
        if c != 1.: pil_image = ImageEnhance.Contrast(pil_image).enhance(c)
        if s * ec != 1.: pil_image = ImageEnhance.Color(pil_image).enhance(s * ec)
        return pil_image

    def _apply_sharpness_detail(self, pil_image, sh, ee, de):
        if sh != 0.: pil_image = ImageEnhance.Sharpness(pil_image).enhance(1. + sh)
        if ee > 0.: pil_image = Image.blend(pil_image, pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE), ee)
        if de > 0.: pil_image = Image.blend(pil_image, pil_image.filter(ImageFilter.DETAIL), de)
        return pil_image

    def _apply_blurs(self, pil_image, br, gbr):
        if br > 0: pil_image = pil_image.filter(ImageFilter.BoxBlur(br))
        if gbr > 0.: pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=gbr))
        return pil_image

    def _apply_chromatic_aberration(self, pil_image, rx, ry, bx, by, blur):
        img = np.array(pil_image).astype(np.float32)
        rc, gc, bc = img[...,0], img[...,1], img[...,2]
        if blur > 0.: 
            bk = int(blur * 2) * 2 + 1
            rc, bc = cv2.GaussianBlur(rc, (bk, bk), blur), cv2.GaussianBlur(bc, (bk, bk), blur)
        rs = cv2.warpAffine(rc, np.float32([[1, 0, rx], [0, 1, ry]]), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
        bs = cv2.warpAffine(bc, np.float32([[1, 0, bx], [0, 1, by]]), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
        return Image.fromarray(np.clip(cv2.merge([rs, gc, bs]), 0, 255).astype(np.uint8))

    def _apply_radial_blur(self, pil_image, strength, cx_r, cy_r, spread, steps):
        img = np.array(pil_image).astype(np.float32) / 255.
        h, w = img.shape[:2]; cx, cy = int(w * cx_r), int(h * cy_r)
        max_d = np.sqrt(max(cx**2+cy**2, (w-cx)**2+cy**2, cx**2+(h-cy)**2, (w-cx)**2+(h-cy)**2))
        X, Y = np.meshgrid(np.arange(w) - cx, np.arange(h) - cy)
        rad_mask = np.sqrt(X**2 + Y**2) / max(1., max_d)
        blurred, cur_blur = [], strength
        for _ in range(steps):
            k = int(cur_blur)
            blurred.append(cv2.GaussianBlur(img, (k | 1, k | 1), 0) if k > 0 else img.copy())
            cur_blur /= spread
        final = np.zeros_like(img)
        for i in range(steps):
            m_i = np.dstack([np.clip((rad_mask - i / steps) * steps, 0, 1)] * 3)
            final = blurred[steps - 1 - i] * m_i + (final if i > 0 else img) * (1. - m_i)
        return Image.fromarray(np.clip(final * 255, 0, 255).astype(np.uint8))

    def _apply_scanlines(self, pil_image, intensity):
        img = np.array(pil_image)
        scan = np.ones(img.shape[:2], dtype=img.dtype)
        scan[::2, :] = 1. - intensity
        return Image.fromarray((img * np.dstack([scan]*3)).astype(np.uint8))

    def _apply_soft_light(self, pil_image, opacity, blur_radius):
        blur = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return Image.blend(pil_image, ImageChops.soft_light(pil_image, blur), opacity)

    def _apply_gamma_torch(self, tensor_images, gamma_value):
        if gamma_value == 1.0: return tensor_images
        return torch.pow(tensor_images.clamp(1e-5, 1.0), 1.0 / max(gamma_value, 0.01))

    def _apply_simple_film_grain_torch(self, tensor_images, intensity, monochrome):
        if intensity == 0.: return tensor_images
        B, H, W, C = tensor_images.shape
        noise = torch.randn((B, H, W, 1 if monochrome else C), device=tensor_images.device) * intensity
        if monochrome and C > 1: noise = noise.repeat(1, 1, 1, C)
        return (tensor_images + noise).clamp(0.0, 1.0)

    def _apply_vignette_torch(self, tensor_images, intensity, cx_r, cy_r):
        if intensity == 0.: return tensor_images
        B, H, W, C = tensor_images.shape
        cx, cy = W * cx_r, H * cy_r
        Y, X = torch.meshgrid(torch.arange(H, device=tensor_images.device) - cy, torch.arange(W, device=tensor_images.device) - cx, indexing='ij')
        dist = torch.sqrt(X**2 + Y**2)
        max_dist = torch.sqrt(torch.tensor(max(cx**2+cy**2, (W-cx)**2+cy**2, cx**2+(H-cy)**2, (W-cx)**2+(H-cy)**2), device=tensor_images.device))
        v_mask = (1.0 - (dist / max(1.0, max_dist))**2 * intensity).clamp(0.0, 1.0).unsqueeze(0).unsqueeze(-1)
        return (tensor_images * v_mask).clamp(0.0, 1.0)

    def process(self, clip, video_vae, audio_vae, model1_primary, character_descriptions, location_description, scene_descriptions,
                primary_sampling, spatial_upscale, temporal_upscale, restore_faces, enable_colorfx, video_mode, image_strength, img_compression, audio_select, identity_guidance_scale,
                use_ollama, ollama_url, ollama_model,
                seed_number, control_before_generate, target_resolution, length_in_seconds, frame_rate,
                primary_sampler_name, primary_cfg, primary_steps,
                eta, bongmath, enable_nag,
                spatial_passes, spatial_sampler, spatial_cfg, spatial_sigmas, temporal_denoise,
                facerestore_model, facedetection, codeformer_fidelity,
                face_restore_color_match, face_restore_edge_blur, face_restore_blend,
                enable_color_correction, hdr_intensity, shadow_intensity, highlight_intensity, gamma, brightness, contrast, saturation, enhance_color,
                enable_lut_processing, lut_name, lut_strength, lut_log_process,
                enable_enhancements, sharpness, edge_enhance_strength, detail_enhance_strength,
                enable_blur_effects, blur_radius, gaussian_blur_radius, radial_blur_strength, radial_blur_center_x, radial_blur_center_y, radial_blur_focus_spread, radial_blur_steps,
                enable_stylistic_effects, chromatic_aberration_r_x, chromatic_aberration_r_y, chromatic_aberration_b_x, chromatic_aberration_b_y, chromatic_blur_amount, simple_film_grain_intensity, simple_film_grain_monochrome, scanline_intensity, vignette_intensity, vignette_center_x, vignette_center_y, soft_light_opacity, soft_light_blur_radius,
                enable_fp16_accumulation, sage_attention, autoregressive_chunking, chunk_size_seconds, context_window_seconds, chunks_feedforward, clear_models_and_cache, enable_preview,
                model2_spatial=None, spatial_upscaler=None, model3_temporal=None, temporal_upscaler=None,
                positive=None, negative=None, audio=None, latent_override=None, unique_id=None, **kwargs):

        # --- EXTRACT THE ILLEGAL VARIABLE NAME FROM KWARGS ---
        images = kwargs.get("image(s)", None)

        unique_id_val = unique_id[0] if isinstance(unique_id, list) else unique_id

        # --- MAP TO INTERNAL VARIABLES TO PRESERVE ALL MATH ---
        bypass_img_ref = (video_mode != "reference-to-video")
        bypass_first_frame = (video_mode != "image-to-video")
        
        image_ref = images if not bypass_img_ref else None
        first_frame = images if not bypass_first_frame else None
        
        image_ref_str = image_strength
        first_frame_str = image_strength

        # --- EXPLICIT LEGACY AUDIO VARIABLE MAPPING ---
        if audio_select == "internal":
            load_audio_from_file = False
            bypass_audio_ref = True
        elif audio_select == "source":
            load_audio_from_file = True
            bypass_audio_ref = True
        elif audio_select == "reference":
            load_audio_from_file = False
            bypass_audio_ref = False
        else:
            load_audio_from_file = False
            bypass_audio_ref = True

        current_fps = frame_rate
        decode = True 
        
        # ==========================================
        # ETA TRACKER MATH & CALLBACK FACTORY
        # ==========================================
        node_id = unique_id_val
        
        # 1. Simulate Base Chunks & Steps
        base_chunks = 0
        if not autoregressive_chunking or length_in_seconds <= chunk_size_seconds:
            base_chunks = 1
        else:
            shot_dur = length_in_seconds / max(1, len([p for p in scene_descriptions.split("|") if p.strip()]))
            for s in range(max(1, len([p for p in scene_descriptions.split("|") if p.strip()]))):
                curr = 0.0
                end = (s + 1) * shot_dur
                while curr < end:
                    curr = min(curr + chunk_size_seconds, end)
                    base_chunks += 1

        if ',' in str(primary_steps):
            base_steps = max(1, len(re.findall(r"[-+]?(?:\d*\.*\d+)", str(primary_steps))) - 1)
        else:
            try:
                base_steps = max(1, int(float(str(primary_steps).strip())))
            except ValueError:
                base_steps = 8
                    
        # 2. Simulate Upscaler Chunks & Steps
        temp_tile_len = 48
        temp_overlap = 8

        v_frames_base = int((length_in_seconds * current_fps) + 1)
        latent_frames_base = ((v_frames_base - 1) // 8) + 1
        
        def count_sliding_chunks(total_latent_frames):
            return len(list(range(0, total_latent_frames, temp_tile_len - temp_overlap)))

        sp_chunks = count_sliding_chunks(latent_frames_base) * spatial_passes if spatial_upscale else 0
        sp_steps = max(1, len(re.findall(r"[-+]?(?:\d*\.*\d+)", str(spatial_sigmas))) - 1) if spatial_upscale else 0

        tm_chunks = 0
        tm_steps = 0

        total_chunks = base_chunks + sp_chunks + tm_chunks
        total_global_steps = (base_chunks * (base_steps + 1)) + (sp_chunks * sp_steps) + (tm_chunks * tm_steps)
        
        current_chunk_idx = [0]
        current_global_step = [0]

        def wrap_callback(base_cb, pass_name):
            current_chunk_idx[0] += 1
            chunk_num = current_chunk_idx[0]
            def custom_cb(step, x0, x, total_steps_in_pass):
                current_global_step[0] += 1
                PromptServer.instance.send_sync("ltxv_eta_update", {
                    "node_id": node_id,
                    "step": int(step) + 1,
                    "total_steps": int(total_steps_in_pass),
                    "chunk": int(chunk_num),
                    "total_chunks": int(total_chunks),
                    "global_step": int(current_global_step[0]),
                    "total_global_steps": int(total_global_steps),
                    "pass_name": pass_name,
                    "is_face_restore": bool(restore_faces) 
                })
                if base_cb is not None:
                    base_cb(step, x0, x, total_steps_in_pass)
            return custom_cb
        
        # Audio Safeguard Variables
        has_audio_ref = not bypass_audio_ref and audio is not None
        has_audio_input = load_audio_from_file and audio is not None

        negative_prompt = "music, background music, soundtrack, worst quality, deformed, glitch, static, bad teeth, deformed teeth, blurry, soft focus, out of focus, smooth, plastic, washed out, hazy, illustration, painting, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, unreadable texts, text, watermarks, 3d render, cgi"

        def build_custom_sampler(name, eta_val, bongmath_val):
            if name.startswith("res_") or name.startswith("rk_"):
                try:
                    return comfy.samplers.ksampler("rk_beta", {
                        "rk_type": name,
                        "eta": eta_val,
                        "eta_substep": eta_val,
                        "BONGMATH": bongmath_val,
                        "sampler_mode": "standard"
                    })
                except Exception:
                    pass 
            return comfy.samplers.sampler_object(name)

        # Parse Resolution Dropdown
        parsed_width = 1792
        parsed_height = 768
        
        if "×" in target_resolution and not target_resolution.startswith("---"):
            try:
                dims = target_resolution.split("—")[0]
                parsed_width = int(dims.split("×")[0].strip())
                parsed_height = int(dims.split("×")[1].strip())
            except Exception as e:
                print(f"LTXV Custom Warning: Failed to parse resolution '{resolution}'. Defaulting to 1792x768.")
        else:
            print(f"LTXV Custom Warning: Category Spacer selected instead of resolution. Defaulting to 1792x768.")

        target_width = parsed_width
        target_height = parsed_height

        divisor = (2 ** spatial_passes) if spatial_upscale else 1
        initial_width = target_width // divisor
        initial_height = target_height // divisor

        expected_width = (initial_width // 32) * 32
        expected_height = (initial_height // 32) * 32

        def process_ref_image(img):
            if img.shape[1] != expected_height or img.shape[2] != expected_width:
                return comfy.utils.common_upscale(
                    img.movedim(-1, 1), expected_width, expected_height, "bilinear", "center"
                ).movedim(1, -1)
            return img

        # --- LYRICS-TO-SCRIPT: WHISPER TRANSCRIPTION ENGINE ---
        if use_ollama == "lyrics-to-script":
            if audio is None:
                print("\nLTXV Custom Warning: 'lyrics-to-script' selected but no audio provided. Falling back to manual scene_descriptions.")
            else:
                print("\n--- Transcribing Audio for Lyrics-to-Script ---")
                try:
                    from transformers import WhisperProcessor, WhisperForConditionalGeneration

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    waveform = audio["waveform"][0]
                    sample_rate = audio["sample_rate"]
                    target_sr = 16000

                    if sample_rate != target_sr:
                        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)

                    if waveform.dim() == 2 and waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0)
                    waveform = waveform.squeeze()

                    model_name = "openai/whisper-large-v3"
                    print(f"-> Loading {model_name} (This may take a moment on first run)...")
                    processor = WhisperProcessor.from_pretrained(model_name)
                    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()

                    max_length = target_sr * 30  # Process in 30 second chunks to prevent OOM
                    total_len = waveform.size(0)
                    chunks = [waveform[i:i + max_length] for i in range(0, total_len, max_length)]

                    transcriptions = []
                    for chunk in chunks:
                        # Pad short chunks
                        if chunk.size(0) < max_length:
                            pad_len = max_length - chunk.size(0)
                            chunk = torch.nn.functional.pad(chunk, (0, pad_len))

                        inputs = processor(chunk, sampling_rate=target_sr, return_tensors="pt", padding="longest", truncation=False)
                        input_features = inputs["input_features"].to(model.device)

                        generated_ids = model.generate(input_features)
                        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        transcriptions.append(text.strip())

                    # Combine chunks and split by punctuation to simulate distinct "Lines" of lyrics
                    full_text = " ".join(transcriptions).strip()
                    phrases = [p.strip() for p in re.split(r'[.,!?\n]+', full_text) if p.strip()]
                    
                    if not phrases:
                        phrases = ["(Instrumental audio)"]
                        
                    # OVERWRITE the manual text box entirely!
                    scene_descriptions = " | ".join(phrases)
                    print(f"-> Transcription Complete! Generated {len(phrases)} lyrical scenes.")
                    print(f"-> Extracted Lyrics: {scene_descriptions}")
                    
                    # Clean up VRAM immediately so we have room for generation
                    del model
                    del processor
                    comfy.model_management.soft_empty_cache()
                except ImportError:
                    print("\nLTXV Custom Error: 'transformers' or 'torchaudio' library missing. Falling back to manual scenes.")
                except Exception as e:
                    print(f"\nLTXV Custom Error during transcription: {e}. Falling back to manual scenes.")

        raw_prompts = [p.strip() for p in scene_descriptions.split("|") if p.strip()]
        num_prompts = len(raw_prompts)
        if num_prompts == 0:
            raw_prompts = [""]
            num_prompts = 1

        override_char_desc = (not bypass_img_ref) and (image_ref is not None) and (use_ollama in ["enhance prompt", "lyrics-to-script"])
        c_desc = character_descriptions.strip()
        l_desc = location_description.strip()

        new_prompts = []
        for p in raw_prompts:
            prefix = ""
            if not override_char_desc and c_desc:
                prefix += f"Character(s): {c_desc}. "
            if l_desc:
                prefix += f"Location: {l_desc}. "
            new_prompts.append(prefix + p)
        raw_prompts = new_prompts

        if num_prompts > 1:
            shot_duration_int = max(1, round(length_in_seconds / num_prompts))
            ideal_length = float(shot_duration_int * num_prompts)
            
            if ideal_length != length_in_seconds:
                print(f"\n--- Auto-Rounding Timeline: Adjusted total duration from {length_in_seconds}s to {ideal_length}s for perfect {shot_duration_int}s shots. ---")
                length_in_seconds = ideal_length

            autoregressive_chunking = True 
            chunk_size_seconds = length_in_seconds / num_prompts
            print(f"\n--- Multi-Shot Director Mode Active: Timeline synced to {num_prompts} shots ({chunk_size_seconds:.2f}s per shot). ---")

        if use_ollama in ["enhance prompt", "lyrics-to-script"]:
            print(f"\n--- Querying Ollama ({ollama_model}) for Multi-Shot Enhancement ---")

            def create_grid_b64(tensor_list):
                pil_images = []
                for t in tensor_list:
                    img_arr = t[0].cpu().numpy()
                    img_arr = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_arr)
                    pil_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                    pil_images.append(pil_img)

                if not pil_images:
                    return None

                num_imgs = len(pil_images)
                if num_imgs == 1:
                    grid = pil_images[0]
                else:
                    cols = 2
                    rows = math.ceil(num_imgs / 2)
                    w = max(img.width for img in pil_images)
                    h = max(img.height for img in pil_images)
                    grid = Image.new('RGB', (cols * w, rows * h))
                    for i, img in enumerate(pil_images):
                        x = (i % cols) * w
                        y = (i // cols) * h
                        grid.paste(img, (x, y))

                buffered = BytesIO()
                grid.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode("utf-8")

            system_prompt = """You write prompts for LTX Video. Output one single flowing paragraph only — no preamble, no label, no explanation, no markdown, no variations. Begin writing immediately.
I will provide you with a base prompt and a single reference image collage containing the Subject, Object, and/or Location. Your job is to seamlessly combine them into a single, highly detailed, flowing paragraph.
Do not describe the initial pose or orientation of the subject from the reference image - only if included in the base prompt.
Do not leave out a single detail that is included in the base prompt.

CORE FORMAT:
- Single flowing paragraph, present tense, no line breaks
- 8–14 descriptive sentences scaled to clip length
- Specificity wins — LTX handles complexity, do not oversimplify
- Block the scene like a director: name positions (left/right), distances (foreground/background), facing directions
- Every sentence should contain at least one verb driving action or motion

REQUIRED ELEMENTS — write in this order, woven into natural sentences:

1. SHOT + CINEMATOGRAPHY
Open with shot scale and camera position. Examples: close-up, medium shot, wide establishing shot, low angle, Dutch tilt, over-the-shoulder, overhead, POV. Match detail level to shot scale — close-ups need more texture detail than wide shots.

2. SCENE + ATMOSPHERE
Location, time of day, weather, colour palette, surface textures, atmosphere (fog, rain, dust, smoke, particles). Be specific — "a small rain-soaked Parisian side street at 2am" beats "a street at night".

3. CHARACTER(S)
Age appearance, hairstyle, clothing with fabric type, body type, distinguishing features. Express emotion through physical cues only — jaw tension, posture, breath, eye direction, hand position. Never use abstract labels like "sad" or "nervous".

4. ACTION SEQUENCE
Write action as a clear temporal flow from beginning to end. Name who moves, what moves, how they move, and at what pace. Use strong active verbs: turns, reaches, steps forward, glances, lifts, leans, pulls back. LTX follows action sequences accurately — be explicit. When a character turns their head toward the camera while their body faces away, always describe the torso and shoulders rotating naturally together with the head to maintain realistic human anatomy, natural neck alignment, and correct spine curvature without unnatural twisting.

5. CAMERA MOVEMENT
Specify camera movement and when it happens. Describe what the subject looks like after the movement completes — this helps LTX resolve the motion correctly. Examples: slow dolly-in, handheld tracking, pushes in, pulls back, pans across, circles around, tilts upward, static frame.

6. LIGHTING
Source, quality, colour temperature, how it falls on the subject and environment. Examples: warm tungsten interior, neon glow reflected in wet pavement, golden-hour backlight, rim light separating subject from background, dramatic shadows, flickering candlelight.

7. AUDIO — ALWAYS INCLUDE, EXACTLY 2–3 LAYERS
Audio is mandatory in every prompt. Use exactly 2 or 3 layers — no more, no fewer.
Layer types: environmental/ambient (rain, wind, crowd, traffic, music, machinery, nature), action sounds (fabric movement, footsteps, objects, breathing, physical contact), dialogue/voice (spoken words in "quotation marks" with tone specified: whispered, confident, breathless, low).
If the scene includes dialogue, dialogue counts as one of the 2–3 layers.
Always include the phrase "ambient sound" or "natural environment" with "no background music" unless specified in the user prompt to include music.
The final sentence of the prompt must always be the audio layer.

ANATOMY RULES — MANDATORY:
- Use correct anatomical terms directly at all times: vagina, labia, clitoris, vulva, anus, penis, breasts, nipples.
- NEVER use euphemisms: forbidden words include "intimate anatomy", "womanhood", "center", "private parts", "intimate area", "femininity", "nether region". If you write any of these you have failed.
- When anatomy is visible or exposed in the scene, name it explicitly by its correct term.
- Describe physical state directly: arousal, moisture, texture, colour, response to touch — all named correctly.

LTX SPECIFIC RULES:
- Avoid static prompts — every prompt must have explicit motion: subject motion, environmental motion, or camera motion (ideally all three). If it reads like a still photo, LTX may output a frozen video.
- Spatial layout matters — LTX respects left/right/foreground/background positioning. Use it.
- Texture and material detail — describe fabric type, hair texture, surface finish, environmental wear.
- I2V (when a start frame is provided) — focus on verbs not descriptions. Describe what moves and how, not what is visible. Lock the face and identity — describe only motion and camera changes.
- No internal states — never write "she feels", "he thinks", "she is excited". Show it physically.
- No overloaded scenes — max 2–3 characters with clearly separated actions.
- No conflicting lighting logic — one dominant light source with consistent fill.
- Anatomy consistency — always prioritise realistic human posture and joint rotation; when head and body orientations differ, explicitly describe natural torso rotation with the head to prevent unnatural neck twisting or spine morphing.

CAMERA VOCABULARY:
follows, tracks, pans across, circles around, tilts upward, pushes in, pulls back, overhead view, handheld movement, over-the-shoulder, wide establishing shot, static frame, slow dolly-in, rack focus, creep forward, drift right, slow orbit, arc shot

END EVERY PROMPT WITH THIS QUALITY TAIL (woven into the final sentence, not as a separate line):
cinematic, ultra-detailed, sharp focus, photorealistic, masterpiece, maintains realistic human anatomy and natural joint rotation throughout

Output only the prompt. Nothing before it, nothing after it."""

            enhanced_prompts = []

            for i, p in enumerate(raw_prompts):
                tensors_to_grid = []
                if not bypass_img_ref and image_ref is not None:
                    for j in range(image_ref.shape[0]):
                        tensors_to_grid.append(image_ref[j:j+1])
                if not bypass_first_frame and first_frame is not None:
                    img_idx = min(i, first_frame.shape[0] - 1)
                    tensors_to_grid.append(first_frame[img_idx:img_idx+1])

                grid_b64 = create_grid_b64(tensors_to_grid) if tensors_to_grid else None

                # --- CINEMATIC SHOT VARIETY ENGINE ---
                # A carefully curated rotation of focal distances to guarantee dynamic editing pacing
                shot_scales = [
                    "wide establishing shot",
                    "intimate close-up",
                    "medium tracking shot",
                    "low angle extreme wide shot",
                    "medium close-up",
                    "extreme close-up macro shot",
                    "wide sweeping crane shot"
                ]
                # Rotate through the list based on the current shot index
                assigned_scale = shot_scales[i % len(shot_scales)]

                if use_ollama == "lyrics-to-script":
                    user_message = {
                        "role": "user",
                        "content": f"This is Shot {i+1} of a music video sequence. MANDATORY CAMERA FRAMING: You must frame this specific shot as a **{assigned_scale}**.\n\nThe prompt below contains the characters, the location, and the specific sung lyric for this exact shot:\n\n{p}\n\nAct as a music video director. Generate a cohesive, highly detailed visual prompt that blends the characters and location with the emotion of the lyric. CRITICAL INSTRUCTION: You MUST explicitly describe the main character actively singing, performing, or lip-syncing the words in the scene. Do not just describe their mood; describe them physically singing. You must also explicitly describe the camera perspective using the {assigned_scale}."
                    }
                else:
                    user_message = {
                        "role": "user",
                        "content": f"This is Shot {i+1} of a multi-shot sequence. MANDATORY CAMERA FRAMING: You must frame this specific shot as a **{assigned_scale}**.\n\nBase prompt: {p}\n\nAnalyze the provided reference image grid and generate the final LTX-Video prompt. You must explicitly describe the camera perspective using the {assigned_scale}."
                    }

                if grid_b64:
                    user_message["images"] = [grid_b64]

                payload = {
                    "model": ollama_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        user_message
                    ],
                    "stream": False
                }

                try:
                    req = urllib.request.Request(f"{ollama_url}/api/chat", data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=120) as response:
                        result = json.loads(response.read().decode('utf-8'))
                        ollama_prompt = result.get('message', {}).get('content', '').strip()
                        if ollama_prompt:
                            enhanced_prompts.append(ollama_prompt)
                            print(f"-> Shot {i+1} Enhanced:\n{ollama_prompt}\n")
                        else:
                            enhanced_prompts.append(p)
                except urllib.error.HTTPError as e:
                    try:
                        error_body = e.read().decode('utf-8')
                        print(f"-> Ollama HTTP Error {e.code}: {error_body}")
                    except:
                        print(f"-> Ollama HTTP Error {e.code}: {e.reason}")
                    print(f"-> CRASH FIX: Falling back to original prompt for Shot {i+1}.")
                    enhanced_prompts.append(p)
                except Exception as e:
                    print(f"-> Ollama API Error: {e}. Falling back to original prompt for Shot {i+1}.")
                    enhanced_prompts.append(p)

            prompt_list = enhanced_prompts
            print("---------------------------------------------------------")
        else:
            prompt_list = raw_prompts

        # ==========================================
        # 0. DEFINE HARDCODED VARIABLES & HIDDEN TEXT
        # ==========================================
        duplicate_frames = 8  
        hidden_prefix = ""
        hidden_suffix = " Shot on 85mm lens, f/8 aperture, raw DSLR footage, ultra-sharp focus, 8k resolution, hyperrealistic, intricate details, cinematic lighting."

        time_scale_factor, height_scale_factor, width_scale_factor = video_vae.downscale_index_formula

        # ==========================================
        # 1. TEXT CONDITIONING (ISOLATED FOR HARD CUTS)
        # ==========================================
        final_positive = []
        final_prompt_strings = []
        
        # --- POSITIVE OVERRIDE ---
        if positive is not None:
            print("\n--- Positive Conditioning Override Detected ---")
            final_prompt_strings.append("--- Custom Positive Conditioning Override Active ---")
            for i in range(num_prompts):
                # Safely map external conditioning elements to our internal multi-shot arrays
                idx = min(i, len(positive) - 1)
                cond_t, cond_d = positive[idx][0], positive[idx][1].copy()
                cond_d["frame_rate"] = float(current_fps)
                final_positive.append([cond_t, cond_d])
        else:
            for i, p in enumerate(prompt_list):
                modified_prompt = f"{hidden_prefix}{p}{hidden_suffix}"
                final_prompt_strings.append(modified_prompt)
                tokens = clip.tokenize(modified_prompt)
                out = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
                cond = out.pop("cond")
                c_dict = out.copy() if isinstance(out, dict) else {}
                c_dict["start_percent"] = 0.0 
                c_dict["end_percent"] = 1.0
                c_dict["frame_rate"] = float(current_fps)
                final_positive.append([cond, c_dict])

        final_prompt_string_out = " | ".join(final_prompt_strings)

        # --- NEGATIVE OVERRIDE ---
        if negative is not None:
            print("--- Negative Conditioning Override Detected ---")
            final_negative = []
            for i in range(num_prompts):
                idx = min(i, len(negative) - 1)
                cond_t, cond_d = negative[idx][0], negative[idx][1].copy()
                cond_d["frame_rate"] = float(current_fps)
                final_negative.append([cond_t, cond_d])
        else:
            tokens_neg = clip.tokenize(negative_prompt)
            out_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True, return_dict=True)
            cond_neg = out_neg.pop("cond")
            dict_neg = out_neg.copy() if isinstance(out_neg, dict) else {}
            dict_neg["start_percent"] = 0.000
            dict_neg["end_percent"] = 1.000
            dict_neg["frame_rate"] = float(current_fps)
            final_negative = [[cond_neg, dict_neg]]

        # ==========================================
        # 2. CALCULATE 'A' AND ASSEMBLE SETUP REFERENCES
        # ==========================================
        pixel_frames = []
        strengths = []
        a = 0

        if not bypass_img_ref and image_ref is not None:
            if img_compression > 0:
                compressed_images = []
                for i in range(image_ref.shape[0]):
                    compressed_images.append(preprocess_compression(image_ref[i], img_compression))
                image_ref = torch.stack(compressed_images)
                
            image_ref_processed = process_ref_image(image_ref)
            a += image_ref_processed.shape[0]
            for idx in range(image_ref_processed.shape[0]):
                img = image_ref_processed[idx:idx+1]
                pixel_frames.append(img.repeat(duplicate_frames, 1, 1, 1))
                strengths.extend([image_ref_str] * duplicate_frames)

        if not bypass_first_frame and first_frame is not None:
            if img_compression > 0:
                compressed_images = []
                for i in range(first_frame.shape[0]):
                    compressed_images.append(preprocess_compression(first_frame[i], img_compression))
                first_frame = torch.stack(compressed_images)
                
            first_frame_processed = process_ref_image(first_frame)
            pixel_frames.append(first_frame_processed[0:1])
            strengths.append(first_frame_str)

        ref_frame_count = len(strengths)

        if len(strengths) > 0:
            final_pixels = torch.cat(pixel_frames, dim=0)
        else:
            final_pixels = None

        # ==========================================
        # 3. EXACT FRAME MATH & BASE LATENT GENERATION
        # ==========================================
        shot_seconds = length_in_seconds / num_prompts
        
        if not bypass_img_ref:
            raw_frames_per_shot = (a * duplicate_frames) + (shot_seconds * current_fps) + 9
        else:
            raw_frames_per_shot = (shot_seconds * current_fps) + 1
            
        latents_per_shot = int(((raw_frames_per_shot - 1) // 8) + 1)
        if latents_per_shot < 1: latents_per_shot = 1
        
        frames_per_shot = int(((latents_per_shot - 1) * 8) + 1)
        
        # Multiply by num_prompts to allocate perfectly segregated blocks
        latent_count = latents_per_shot * num_prompts
        frame_length = frames_per_shot * num_prompts
        
        device = comfy.model_management.intermediate_device()
        batch_size = 1 

        # --- LATENT OVERRIDE EXTRACTION ---
        v_override = None
        a_override = None
        if latent_override is not None:
            print("\n--- Latent Override Detected ---")
            samples = latent_override.get("samples", latent_override) 
            if getattr(samples, "is_nested", False) or isinstance(samples, comfy.nested_tensor.NestedTensor):
                v_override, a_override = samples.unbind()
            elif isinstance(samples, (list, tuple)) and len(samples) == 2:
                v_override, a_override = samples[0], samples[1]
            else:
                v_override = samples
                
        # --- APPLY OVERRIDE TO TIMELINE MATH ---
        if v_override is not None:
            print("-> Overriding timeline and resolution math to match injected tensor.")
            video_samples = v_override.to(device).clone()
            batch_size, _, latent_count, v_h, v_w = video_samples.shape
            initial_height = v_h * 32
            initial_width = v_w * 32
            frame_length = int(((latent_count - 1) * 8) + 1)
            length_in_seconds = frame_length / current_fps
            pure_latents_per_shot = latent_count // num_prompts
            setup_latents = 0
            shot_seconds = length_in_seconds / num_prompts
        else:
            video_samples = torch.zeros([batch_size, 128, latent_count, initial_height // 32, initial_width // 32], device=device)
            video_samples = comfy.sample.fix_empty_latent_channels(model1_primary, video_samples, None)
            
        video_noise_mask = torch.ones((batch_size, video_samples.shape[2], video_samples.shape[3], video_samples.shape[4]), dtype=torch.float32, device=device)

        z_channels = getattr(audio_vae, "latent_channels", audio_vae.first_stage_model.latent_channels)
        audio_freq = getattr(audio_vae, "latent_frequency_bins", audio_vae.first_stage_model.latent_frequency_bins)
        sampling_rate = int(getattr(audio_vae, "sample_rate", audio_vae.first_stage_model.sample_rate))

        get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
        if get_audio_latents_func is None:
            raise AttributeError("Audio VAE is missing 'num_of_latents_from_frames' method.")
            
        if a_override is not None:
            num_audio_latents = a_override.shape[2]
        else:
            num_audio_latents = get_audio_latents_func(frame_length, int(current_fps))

        total_silence_samples = int(((frame_length / current_fps) + 5.0) * sampling_rate)
        silent_wf = torch.zeros((batch_size, 2, total_silence_samples), dtype=torch.float32, device=device)
        
        # --- SAFE AUDIO VAE ENCODE (DEVICE MISMATCH FIX) ---
        vae_device = audio_vae.first_stage_model.device if hasattr(audio_vae.first_stage_model, "device") else comfy.model_management.get_torch_device()
        silent_wf_mapped = silent_wf.movedim(1, -1).to(vae_device)
        
        try:
            true_silence_latent = audio_vae.encode(silent_wf_mapped).to(device)
        except AttributeError:
            true_silence_latent = audio_vae.first_stage_model.encode(silent_wf_mapped).to(device)
        
        audio_samples = torch.zeros((batch_size, z_channels, num_audio_latents, audio_freq), device=device)
        
        if a_override is not None:
            use_len = min(num_audio_latents, a_override.shape[2])
            audio_samples[:, :, :use_len, :] = a_override[:, :, :use_len, :]
        else:
            use_silence_len = min(num_audio_latents, true_silence_latent.shape[2])
            if use_silence_len > 0:
                audio_samples[:, :, :use_silence_len, :] = true_silence_latent[:, :, :use_silence_len, :]
                
        audio_noise_mask = torch.ones_like(audio_samples)

        # ==========================================
        # 3.4 HELPER: LATENT COUNTER
        # ==========================================
        def get_latent_counts(sec):
            if sec <= 0.0:
                return 0, 0
                
            shot_idx = int(sec // shot_seconds)
            if shot_idx >= num_prompts:
                shot_idx = num_prompts - 1
                
            remainder_sec = sec - (shot_idx * shot_seconds)
            
            # ACCORDION FIX: Calculate exact isolated audio length for one full shot to prevent continuous drift!
            get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
            base_a_lat_per_shot = get_audio_latents_func(frames_per_shot, int(current_fps))
            
            if remainder_sec <= 0.001:
                v_lat = shot_idx * latents_per_shot
                a_lat = shot_idx * base_a_lat_per_shot
            else:
                if bypass_img_ref:
                    raw_frames = int((remainder_sec * current_fps) + 1)
                else:
                    raw_frames = int((a * duplicate_frames) + (remainder_sec * current_fps) + 9)
                    
                local_v_lat = int(((raw_frames - 1) // 8) + 1)
                if local_v_lat < 1: local_v_lat = 1
                
                v_lat = (shot_idx * latents_per_shot) + local_v_lat
                
                local_synced_frames = int(((local_v_lat - 1) * 8) + 1) if local_v_lat > 0 else 1
                local_a_lat = get_audio_latents_func(local_synced_frames, int(current_fps))
                
                a_lat = (shot_idx * base_a_lat_per_shot) + local_a_lat
                
            return v_lat, a_lat

        if bypass_img_ref:
            out_ref_frame_count = 1
        else:
            n = ref_frame_count // duplicate_frames
            out_ref_frame_count = (8 * n) + 8 + 1

        # ==========================================
        # 3.5 WAVEFORM MIXING (NATIVE COMFYUI ASSEMBLY)
        # ==========================================
        # PERFECT SYNC FIX: Video injects exactly out_ref_frame_count (17) frames of setup. 
        setup_frames = 0 if bypass_img_ref else out_ref_frame_count

        if bypass_img_ref:
            region_a_frames = 1
            region_b_frames = 0
        else:
            region_a_frames = ref_frame_count 
            region_b_frames = duplicate_frames 

        get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
        setup_total_latents = get_audio_latents_func(setup_frames, int(current_fps))
        
        vae_sample_rate = getattr(audio_vae, "audio_sample_rate", getattr(audio_vae, "sample_rate", 24000))
        sampling_rate = vae_sample_rate

        total_samples = int((frame_length / current_fps) * sampling_rate)
        region_a_samples = int((region_a_frames / current_fps) * sampling_rate)
        
        # PERFECT SYNC FIX: Use the exact setup_frames to calculate absolute silence padding!
        setup_total_samples = int((setup_frames / current_fps) * sampling_rate)

        master_wf = torch.zeros((batch_size, 2, total_samples), dtype=torch.float32, device=device)
        master_latents = torch.zeros((batch_size, z_channels, num_audio_latents, audio_freq), device=device)

        def encode_native(audio_dict):
            orig_wf = audio_dict["waveform"].to(device)
            orig_sr = audio_dict.get("sample_rate", sampling_rate)
            if orig_sr != sampling_rate:
                wf = torchaudio.functional.resample(orig_wf, orig_sr, sampling_rate)
            else:
                wf = orig_wf

            if wf.dim() == 2:
                wf = wf.unsqueeze(1).repeat(1, 2, 1)
            elif wf.dim() == 3 and wf.shape[1] == 1:
                wf = wf.repeat(1, 2, 1)
            elif wf.dim() == 3 and wf.shape[1] > 2:
                wf = wf[:, :2, :]

            if wf.shape[0] != batch_size:
                wf = wf.repeat(batch_size, 1, 1)[:batch_size]

            # THE FIX: Removed the 20-second chunking limit. 
            # Encoding as one continuous waveform prevents the 8-frame VAE boundary seam!
            wf_mapped = wf.movedim(1, -1).to(vae_device)
            try:
                latents = audio_vae.encode(wf_mapped).to(device)
            except AttributeError:
                latents = audio_vae.first_stage_model.encode(wf_mapped).to(device)

            return wf, latents

        silent_wf = torch.zeros((batch_size, 2, total_samples), dtype=torch.float32, device=device)
        silent_dict = {"waveform": silent_wf, "sample_rate": sampling_rate}
        _, true_silence_latent = encode_native(silent_dict)

        use_silence_len = min(num_audio_latents, true_silence_latent.shape[2])
        if use_silence_len > 0:
            master_latents[:, :, :use_silence_len, :] = true_silence_latent[:, :, :use_silence_len, :]

        if has_audio_input:
            inp_wf, inp_latents = encode_native(audio)

            rem_latents = num_audio_latents - setup_total_latents
            if rem_latents > 0:
                use_lats = min(inp_latents.shape[2], rem_latents)
                if use_lats > 0:
                    master_latents[:, :, setup_total_latents:setup_total_latents+use_lats, :] = inp_latents[:, :, :use_lats]

            rem_samps = total_samples - setup_total_samples
            if rem_samps > 0:
                use_s = min(inp_wf.shape[2], rem_samps)
                if use_s > 0:
                    master_wf[:, :, setup_total_samples:setup_total_samples+use_s] = inp_wf[:, :, :use_s]

        if has_audio_ref:
            ref_wf, ref_latents = encode_native(audio)

            use_lats = min(ref_latents.shape[2], setup_total_latents, num_audio_latents)
            if use_lats > 0:
                master_latents[:, :, :use_lats, :] = ref_latents[:, :, :use_lats]

            use_s = min(ref_wf.shape[2], setup_total_samples, total_samples)
            if use_s > 0:
                master_wf[:, :, :use_s] = ref_wf[:, :, :use_s]

        use_len = min(num_audio_latents, master_latents.shape[2])
        if use_len > 0:
            audio_samples[:, :, :use_len, :] = master_latents[:, :, :use_len, :]

        if has_audio_ref:
            lock_a = min(setup_total_latents, use_len)
            if lock_a > 0:
                audio_noise_mask[:, :, :lock_a, :] = 0.0

        if has_audio_input:
            audio_noise_mask[:, :, :, :] = 0.0

        # ==========================================
        # 4. VIDEO INJECTION & MASKING (PRIMARY PASS)
        # ==========================================
        if final_pixels is not None:
            t_width = video_samples.shape[4] * width_scale_factor
            t_height = video_samples.shape[3] * height_scale_factor

            pass1_pixels = final_pixels.clone()
            if pass1_pixels.shape[1] != t_height or pass1_pixels.shape[2] != t_width:
                pass1_pixels = comfy.utils.common_upscale(pass1_pixels.movedim(-1, 1), t_width, t_height, "bilinear", "center").movedim(1, -1)

            encoded_t = video_vae.encode(pass1_pixels[:, :, :, :3])
            
            # Inject into Shot 1 (Default behavior)
            frames_to_inject = min(encoded_t.shape[2], video_samples.shape[2])
            video_samples[:, :, :frames_to_inject] = encoded_t[:, :, :frames_to_inject]

            for i in range(frames_to_inject):
                pixel_idx = min(i * time_scale_factor, max(0, len(strengths) - 1))
                video_noise_mask[:, i, :, :] = 1.0 - strengths[pixel_idx]
                
            # IDENTITY FIX: If Reference-to-Video is active, we MUST inject this exact setup block into Shots 2+ as well!
            if not bypass_img_ref and num_prompts > 1:
                shot_duration = length_in_seconds / num_prompts
                for s in range(1, num_prompts):
                    sec = s * shot_duration
                    v_lat, _ = get_latent_counts(sec)
                    
                    inject_len = min(encoded_t.shape[2], video_samples.shape[2] - v_lat)
                    if inject_len > 0:
                        video_samples[:, :, v_lat : v_lat + inject_len] = encoded_t[:, :, :inject_len]
                        
                        for i in range(inject_len):
                            pixel_idx = min(i * time_scale_factor, max(0, len(strengths) - 1))
                            video_noise_mask[:, v_lat + i, :, :] = 1.0 - strengths[pixel_idx]
                            
                        print(f"-> Identity Setup Block injected into Shot {s+1} at {sec:.2f}s (Latent Frame {v_lat})")

        # 4.5 MULTI-SHOT DIRECTOR CUT INJECTIONS
        if not bypass_first_frame and first_frame is not None and num_prompts > 1 and autoregressive_chunking:
            shot_duration = length_in_seconds / num_prompts
            for i in range(1, num_prompts):
                img_idx = min(i, first_frame.shape[0] - 1)
                img = first_frame[img_idx:img_idx+1]

                img_scaled = comfy.utils.common_upscale(img.movedim(-1, 1), t_width, t_height, "bilinear", "center").movedim(1, -1)
                encoded_img = video_vae.encode(img_scaled[:, :, :, :3])
                if encoded_img.ndim == 4:
                    encoded_img = encoded_img.unsqueeze(0)
                encoded_img = encoded_img.to(device)

                sec = i * shot_duration
                v_lat, _ = get_latent_counts(sec)

                inject_len = min(encoded_img.shape[2], video_samples.shape[2] - v_lat)
                if inject_len > 0:
                    video_samples[:, :, v_lat : v_lat + inject_len] = encoded_img[:, :, :inject_len]
                    for j in range(inject_len):
                        video_noise_mask[:, v_lat + j, :, :] = 1.0 - first_frame_str
                    print(f"-> Shot {i+1} Reference Frame Injected at {sec:.2f}s (Latent Frame {v_lat})")

        # ==========================================
        # 5. ALL UNET VRAM PATCHING (FP16, Sage, Chunks, NAG, LoRA)
        # ==========================================
        model_to_use = model1_primary.clone()
        model2_to_use = model2_spatial.clone() if model2_spatial is not None else model_to_use.clone()
        model3_to_use = model3_temporal.clone() if model3_temporal is not None else model_to_use.clone()

        # Gather all unique models that require patching
        models_to_patch = [model_to_use]
        if model2_spatial is not None: models_to_patch.append(model2_to_use)
        if model3_temporal is not None: models_to_patch.append(model3_to_use)

        for target_model in models_to_patch:
            diffusion_model = target_model.get_model_object("diffusion_model")
            
            def patch_enable_fp16_accum(model):
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            def patch_disable_fp16_accum(model):
                torch.backends.cuda.matmul.allow_fp16_accumulation = False

            if enable_fp16_accumulation:
                if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                    target_model.add_callback(comfy.patcher_extension.CallbacksMP.ON_PRE_RUN, patch_enable_fp16_accum)
                    target_model.add_callback(comfy.patcher_extension.CallbacksMP.ON_CLEANUP, patch_disable_fp16_accum)
            else:
                if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                    target_model.add_callback(comfy.patcher_extension.CallbacksMP.ON_PRE_RUN, patch_disable_fp16_accum)

            if sage_attention != "disabled":
                target_model.model_options["transformer_options"] = target_model.model_options.get("transformer_options", {}).copy()
                
                new_attention = get_sage_func(sage_attention)
                def attention_override_sage(func, *args, **kwargs):
                    return new_attention.__wrapped__(*args, **kwargs)
                target_model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage

            dim_threshold = 4096
            if chunks_feedforward > 1:
                for idx, block in enumerate(diffusion_model.transformer_blocks):
                    patched_ffn = LTXVffnChunkPatch(chunks_feedforward, dim_threshold).__get__(block.ff, block.__class__)
                    target_model.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.ff.forward", patched_ffn)
            
            # --- HARDCODED NORMALIZED ATTENTION GUIDANCE (NAG) ---
            if enable_nag:
                # To prevent printing multiple times, only print on the first model
                if target_model is model_to_use:
                    print("\n--- Injecting Normalized Attention Guidance (NAG) ---")
                nag_scale = 11.0
                nag_alpha = 0.25
                nag_tau = 2.5
                inplace = True
                
                dtype = target_model.model.manual_cast_dtype
                if dtype is None:
                    dtype = diffusion_model.dtype
                    
                compute_device = comfy.model_management.get_torch_device()
                offload_device = comfy.model_management.unet_offload_device()
                
                context_video = final_negative[0][0].to(compute_device, dtype)
                vid_split = getattr(diffusion_model, "cross_attention_dim", None)
                if vid_split is not None and context_video.shape[-1] == vid_split + getattr(diffusion_model, "audio_cross_attention_dim", 0):
                    context_video = context_video[:, :, :vid_split]
                    
                if getattr(diffusion_model, "caption_proj_before_connector", False) and getattr(diffusion_model, "caption_projection_first_linear", False):
                    diffusion_model.caption_projection.to(compute_device)
                    context_video = diffusion_model.caption_projection(context_video)
                    diffusion_model.caption_projection.to(offload_device)
                    
                if hasattr(diffusion_model, "video_embeddings_connector"):
                    diffusion_model.video_embeddings_connector.to(compute_device)
                    context_video = diffusion_model.video_embeddings_connector(context_video)[0]
                    diffusion_model.video_embeddings_connector.to(offload_device)
                    
                context_video = context_video.view(1, -1, diffusion_model.inner_dim)
                
                for idx, block in enumerate(diffusion_model.transformer_blocks):
                    patched_attn2 = LTXVCrossAttentionPatch(context_video, nag_scale, nag_alpha, nag_tau, inplace=inplace).__get__(block.attn2, block.__class__)
                    target_model.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.attn2.forward", patched_attn2)

        model2_to_use = model2_spatial.clone() if model2_spatial is not None else model_to_use.clone()
        model3_to_use = model3_temporal.clone() if model3_temporal is not None else model_to_use.clone()

        # ==========================================
        # ID-LORA VOICE CLONING (CFG OVERRIDE)
        # ==========================================
        actual_cond_audio = None
        if has_audio_ref:
            actual_cond_audio = audio
        elif has_audio_input:
            actual_cond_audio = audio 
            
        if actual_cond_audio is not None:
            sample_rate_ref = actual_cond_audio["sample_rate"]
            vae_sample_rate = getattr(audio_vae, "audio_sample_rate", 44100)
            if vae_sample_rate != sample_rate_ref:
                wf_lora = torchaudio.functional.resample(actual_cond_audio["waveform"], sample_rate_ref, vae_sample_rate)
            else:
                wf_lora = actual_cond_audio["waveform"]

            max_samps = int(20.0 * vae_sample_rate)
            if wf_lora.shape[2] > max_samps:
                lats = []
                for i in range(0, wf_lora.shape[2], max_samps):
                    chunk_wf = wf_lora[:, :, i:i+max_samps]
                    if chunk_wf.shape[2] < 4000:
                        chunk_wf = torch.nn.functional.pad(chunk_wf, (0, 4000 - chunk_wf.shape[2]))
                    
                    wf_lora_mapped = chunk_wf.movedim(1, -1).to(vae_device)
                    try: lats.append(audio_vae.encode(wf_lora_mapped).to(device))
                    except AttributeError: lats.append(audio_vae.first_stage_model.encode(wf_lora_mapped).to(device))
                audio_latents_lora = torch.cat(lats, dim=2)
            else:
                max_samps = int(20.0 * vae_sample_rate)
                if wf_lora.shape[2] > max_samps:
                    lats = []
                    for i in range(0, wf_lora.shape[2], max_samps):
                        chunk_wf = wf_lora[:, :, i:i+max_samps]
                        if chunk_wf.shape[2] < 2400:
                            continue
                        wf_lora_mapped = chunk_wf.movedim(1, -1).to(vae_device)
                        try: lats.append(audio_vae.encode(wf_lora_mapped).to(device))
                        except AttributeError: lats.append(audio_vae.first_stage_model.encode(wf_lora_mapped).to(device))
                    audio_latents_lora = torch.cat(lats, dim=2)
                else:
                    wf_lora_mapped = wf_lora.movedim(1, -1).to(vae_device)
                    try: audio_latents_lora = audio_vae.encode(wf_lora_mapped).to(device)
                    except AttributeError: audio_latents_lora = audio_vae.first_stage_model.encode(wf_lora_mapped).to(device)
            
            b_, c_, t_, f_ = audio_latents_lora.shape
            ref_tokens = audio_latents_lora.permute(0, 2, 1, 3).reshape(b_, t_, c_ * f_)
            ref_audio_dict = {"tokens": ref_tokens}

            for i in range(len(final_positive)):
                final_positive[i][1]["ref_audio"] = ref_audio_dict
            for i in range(len(final_negative)):
                final_negative[i][1].pop("ref_audio", None)

            scale = identity_guidance_scale

            def get_post_cfg_function(target_model):
                model_sampling = target_model.get_model_object("model_sampling")
                sigma_start = model_sampling.percent_to_sigma(0.0)
                sigma_end = model_sampling.percent_to_sigma(1.0)
                audio_channels = getattr(audio_vae, "latent_channels", getattr(audio_vae.first_stage_model, "latent_channels", 128))
                
                def post_cfg_function(args):
                    if scale == 0:
                        return args["denoised"]

                    sigma = args["sigma"]
                    sigma_ = sigma[0].item()
                    if sigma_ > sigma_start or sigma_ < sigma_end:
                        return args["denoised"]

                    cond_pred = args["cond_denoised"]
                    cond = args["cond"]
                    cfg_result = args["denoised"]
                    model_options = args["model_options"].copy()
                    x = args["input"]

                    is_nested = isinstance(cfg_result, comfy.nested_tensor.NestedTensor)
                    is_video_only = (not is_nested) and (len(cfg_result.shape) > 1 and cfg_result.shape[1] == 128)
                    if is_video_only:
                        return cfg_result

                    noref_cond = []
                    for entry in cond:
                        new_entry = entry.copy()
                        mc = new_entry.get("model_conds", {}).copy()
                        mc.pop("ref_audio", None)
                        new_entry["model_conds"] = mc
                        noref_cond.append(new_entry)

                    (pred_noref,) = comfy.samplers.calc_cond_batch(args["model"], [noref_cond], x, sigma, model_options)

                    if is_nested:
                        cfg_v, cfg_a = cfg_result.unbind()
                        cond_v, cond_a = cond_pred.unbind()
                        noref_v, noref_a = pred_noref.unbind()
                        new_a = cfg_a + (cond_a - noref_a) * scale
                        return comfy.nested_tensor.NestedTensor((cfg_v, new_a))
                    else:
                        is_audio_only = (len(cfg_result.shape) > 1 and cfg_result.shape[1] == audio_channels)
                        if is_audio_only:
                            return cfg_result + (cond_pred - pred_noref) * scale
                    return cfg_result + (cond_pred - pred_noref) * scale
                return post_cfg_function

            if has_audio_ref:
                model_to_use.set_model_sampler_post_cfg_function(get_post_cfg_function(model_to_use))
                model2_to_use.set_model_sampler_post_cfg_function(get_post_cfg_function(model2_to_use))
                model3_to_use.set_model_sampler_post_cfg_function(get_post_cfg_function(model3_to_use))

        # ==========================================
        # 6. SAMPLING & UPSCALING LOOP
        # ==========================================
        noise_obj = Noise_RandomNoise(seed_number)
        
        primary_guider = comfy.samplers.CFGGuider(model_to_use)
        primary_guider.set_cfg(primary_cfg)

        upsample_guider = comfy.samplers.CFGGuider(model2_to_use)
        upsample_guider.set_cfg(spatial_cfg)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        primary_sampler = build_custom_sampler(primary_sampler_name, eta, bongmath)
        
        if ',' in str(primary_steps):
            sigmas_list = re.findall(r"[-+]?(?:\d*\.*\d+)", str(primary_steps))
            primary_sigmas = torch.FloatTensor([float(i) for i in sigmas_list])
        else:
            try:
                p_steps = int(float(str(primary_steps).strip()))
                p_steps = max(1, p_steps)
            except ValueError:
                p_steps = 8

            if autoregressive_chunking:
                active_sec = min(length_in_seconds, chunk_size_seconds + context_window_seconds)
            else:
                active_sec = length_in_seconds
                
            pass_frames = int((active_sec * current_fps) + 1)
            pass_latents = ((pass_frames - 1) // 8) + 1
            active_tokens = pass_latents * (initial_height // 32) * (initial_width // 32)
            
            sigmas = torch.linspace(1.0, 0.0, p_steps + 1)
            max_shift, base_shift, terminal = 2.05, 0.95, 0.1
            x1, x2 = 1024, 4096
            mm_shift = (max_shift - base_shift) / (x2 - x1)
            b_shift = base_shift - mm_shift * x1
            
            sigma_shift = (active_tokens) * mm_shift + b_shift
            sigma_shift = min(sigma_shift, 13.5)

            power = 1
            sigmas = torch.where(sigmas != 0, math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power), 0)
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor_math = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor_math)
            sigmas[non_zero_mask] = stretched
            primary_sigmas = sigmas

        # ==========================================
        # 6.5 UNIFIED AUTOREGRESSIVE & ISOLATED SAMPLING LOOP
        # ==========================================
        pure_image_passthrough = False
        
        if primary_sampling:
            print(f"\n--- Base Generation Active: Building {num_prompts} Isolated Shot(s) ---")
            
            # --- BASE TEXTURE & TIMING LOCK ---
            # Generate pristine global noise for the entire video to prevent sliding hallucinations!
            global_base_v_noise = comfy.sample.prepare_noise(video_samples, seed_number, None).to(device)
            global_base_a_noise = comfy.sample.prepare_noise(audio_samples, seed_number, None).to(device)
            
            class BaseSlicedNoise:
                def __init__(self, v_noise, a_noise, v_start, a_start):
                    self.v_noise = v_noise
                    self.a_noise = a_noise
                    self.v_start = v_start
                    self.a_start = a_start
                    
                def generate_noise(self, input_latent):
                    v_len = input_latent["samples"].unbind()[0].shape[2]
                    a_len = input_latent["samples"].unbind()[1].shape[2]
                    
                    v_slice = self.v_noise[:, :, self.v_start : self.v_start + v_len].clone()
                    a_slice = self.a_noise[:, :, self.a_start : self.a_start + a_len].clone()
                    
                    # ROUNDING FAILSAFE: Pad the noise tensors if multi-shot math exceeds the master timeline!
                    if v_slice.shape[2] < v_len:
                        pad_len = v_len - v_slice.shape[2]
                        v_slice = torch.nn.functional.pad(v_slice, (0, 0, 0, 0, 0, pad_len))
                        
                    if a_slice.shape[2] < a_len:
                        pad_len = a_len - a_slice.shape[2]
                        a_slice = torch.nn.functional.pad(a_slice, (0, 0, 0, pad_len))
                        
                    return comfy.nested_tensor.NestedTensor((v_slice, a_slice))

            isolated_v_shots = []
            isolated_a_shots = []
            
            shot_duration = length_in_seconds / num_prompts
            
            # 0-Indexed local timeline mappers
            def sec_to_v_idx(sec):
                if sec <= 0.0: return 0
                frames = int(sec * current_fps) + 1 if bypass_img_ref else int(a * duplicate_frames) + int(sec * current_fps) + 9
                return ((frames - 1) // 8) + 1
                
            def sec_to_a_idx(sec):
                if sec <= 0.0: return 0
                frames = int(sec * current_fps) + 1 if bypass_img_ref else int(a * duplicate_frames) + int(sec * current_fps) + 9
                synced_frames = int(((((frames - 1) // 8) + 1) - 1) * 8) + 1
                get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
                return get_audio_latents_func(synced_frames, int(current_fps))

            v_len = sec_to_v_idx(shot_duration)
            a_len = sec_to_a_idx(shot_duration)
            
            x0_output = {}
            for s in range(num_prompts):
                shot_start_sec = s * shot_duration
                shot_end_sec = (s + 1) * shot_duration
                
                # Simulate the chunking math to get the exact total chunks for this shot
                total_chunks_in_shot = 0
                temp_sec = shot_start_sec
                while temp_sec < shot_end_sec:
                    total_chunks_in_shot += 1
                    if not autoregressive_chunking:
                        temp_sec = shot_end_sec
                    else:
                        temp_sec = min(temp_sec + chunk_size_seconds, shot_end_sec)
                
                print(f"\n-> Generating Isolated Shot {s+1}/{num_prompts} (Total Chunks: {total_chunks_in_shot})...")
                
                # Map absolute global latents to determine exact local length
                base_v_lat, base_a_lat = get_latent_counts(shot_start_sec)
                end_v_lat, end_a_lat = get_latent_counts(shot_end_sec)
                
                v_len = end_v_lat - base_v_lat
                a_len = end_a_lat - base_a_lat
                
                # 1. Pull the initial latents and masks from the master setup tensors (Enables V2V & Image Injections!)
                pass_v_samples = torch.zeros([batch_size, 128, v_len, initial_height // 32, initial_width // 32], device=device)
                pass_v_masks = torch.ones_like(pass_v_samples)
                
                actual_v_len = min(end_v_lat, video_samples.shape[2]) - base_v_lat
                if actual_v_len > 0:
                    pass_v_samples[:, :, :actual_v_len] = video_samples[:, :, base_v_lat:base_v_lat+actual_v_len]
                    # FIX: video_noise_mask is 4D [B, F, H, W]. We slice it and unsqueeze the channel dimension to broadcast to 5D!
                    pass_v_masks[:, :, :actual_v_len, :, :] = video_noise_mask[:, base_v_lat:base_v_lat+actual_v_len, :, :].unsqueeze(1)
                
                pass_a_samples = torch.zeros([batch_size, z_channels, a_len, audio_freq], device=device)
                pass_a_masks = torch.ones_like(pass_a_samples)
                
                # 2. Slice the correct audio chunk for this shot
                a_start_master = base_a_lat
                a_end_master = min(end_a_lat, audio_samples.shape[2])
                actual_a_len = a_end_master - a_start_master
                if actual_a_len > 0:
                    pass_a_samples[:, :, :actual_a_len] = audio_samples[:, :, a_start_master:a_end_master]
                    pass_a_masks[:, :, :actual_a_len] = audio_noise_mask[:, :, a_start_master:a_end_master]
                    
                # 4. Autoregressive Inner Chunking Loop!
                curr_global_sec = shot_start_sec
                chunk_idx = 1
                while curr_global_sec < shot_end_sec:
                    comfy.model_management.soft_empty_cache()
                    chunk_start_sec = curr_global_sec
                    
                    # Respect UI Settings for chunking boundaries
                    if not autoregressive_chunking:
                        curr_global_sec = shot_end_sec
                    else:
                        curr_global_sec = min(chunk_start_sec + chunk_size_seconds, shot_end_sec)
                    
                    curr_v_global, curr_a_global = get_latent_counts(curr_global_sec)
                    prev_v_global, prev_a_global = get_latent_counts(chunk_start_sec)
                    
                    # Zero-indexed local array matching
                    curr_v_lat = curr_v_global - base_v_lat
                    curr_a_lat = curr_a_global - base_a_lat
                    prev_v_lat = prev_v_global - base_v_lat
                    prev_a_lat = prev_a_global - base_a_lat
                    
                    if chunk_start_sec == shot_start_sec:
                        ctx_v_lat = 0
                        ctx_a_lat = 0
                        # INITIALIZATION FIX: Define the start time for Chunk 1 so the audio math doesn't crash!
                        context_start_sec = shot_start_sec
                        print(f"   -> [Shot {s+1}/{num_prompts} - Chunk {chunk_idx}/{total_chunks_in_shot}] 0.0s to {curr_global_sec - shot_start_sec:.2f}s (Base Block)")
                    else:
                        context_start_sec = max(shot_start_sec, chunk_start_sec - context_window_seconds)
                        ctx_v_global, ctx_a_global = get_latent_counts(context_start_sec)
                        ctx_v_lat = ctx_v_global - base_v_lat
                        ctx_a_lat = ctx_a_global - base_a_lat
                        print(f"   -> [Shot {s+1}/{num_prompts} - Chunk {chunk_idx}/{total_chunks_in_shot}] {chunk_start_sec - shot_start_sec:.2f}s to {curr_global_sec - shot_start_sec:.2f}s (Context Lookback: {chunk_start_sec - context_start_sec:.2f}s)")
                        
                    # Slice the temporary chunk from the isolated shot tensor
                    chunk_v_len = curr_v_lat - ctx_v_lat
                    chunk_v_samples = pass_v_samples[:, :, ctx_v_lat:curr_v_lat]
                    chunk_v_masks = pass_v_masks[:, :, ctx_v_lat:curr_v_lat, :, :].clone()
                    
                    # --- DYNAMIC MASTER AUDIO SLICING (Absolute Sync Fix) ---
                    _, true_a_start = get_latent_counts(context_start_sec)
                    _, true_a_end = get_latent_counts(curr_global_sec)
                    
                    req_a_len = true_a_end - true_a_start
                    true_a_start = max(0, true_a_start)
                    
                    # THE USER'S FIX: Pause audio conditioning during Ref Mode setup frames!
                    if s > 0 and not bypass_img_ref:
                        a_setup_latents = get_audio_latents_func(out_ref_frame_count, int(current_fps))
                        base_a_lat_per_shot = get_audio_latents_func(frames_per_shot, int(current_fps))
                        action_a_latents = base_a_lat_per_shot - a_setup_latents
                        
                        # Reset the clock: use local shot boundaries to prevent accumulating drift!
                        local_a_start = true_a_start - base_a_lat
                        setup_chunk_len = max(0, min(req_a_len, a_setup_latents - local_a_start))
                        action_chunk_len = req_a_len - setup_chunk_len
                        
                        chunk_a_samples = torch.zeros((batch_size, z_channels, req_a_len, audio_freq), device=device)
                        chunk_a_masks = torch.ones_like(chunk_a_samples)
                        
                        if setup_chunk_len > 0:
                            # We are inside the setup frames pause period. Feed it silence!
                            fill_len = min(setup_chunk_len, true_silence_latent.shape[2])
                            chunk_a_samples[:, :, :fill_len] = true_silence_latent[:, :, :fill_len]
                            
                        if action_chunk_len > 0:
                            # Post-setup action frames. Feed it the delayed audio!
                            action_offset = max(0, local_a_start - a_setup_latents)
                            audio_read_start = a_setup_latents + (s * action_a_latents) + action_offset
                            
                            src_end = min(audio_read_start + action_chunk_len, audio_samples.shape[2])
                            copy_len = src_end - audio_read_start
                            
                            if copy_len > 0:
                                chunk_a_samples[:, :, setup_chunk_len : setup_chunk_len + copy_len] = audio_samples[:, :, audio_read_start : src_end]
                                chunk_a_masks[:, :, setup_chunk_len : setup_chunk_len + copy_len] = audio_noise_mask[:, :, audio_read_start : src_end]
                    else:
                        # THE TRUE INVERSE 4-FRAME SHIFT
                        # Since shifting left (-) worsened the drift from 4 to 8, it mathematically proves 
                        # the audio was arriving too early. We must shift the read index RIGHT (+) 
                        # by exactly 4 frames to neutralize the VAE temporal offset!
                        if has_audio_input and chunk_start_sec > shot_start_sec:
                            shift_frames = 4
                            base_lats = get_audio_latents_func(1, int(current_fps))
                            shift_lats = get_audio_latents_func(1 + shift_frames, int(current_fps)) - base_lats
                            
                            local_a_start = true_a_start + shift_lats
                        else:
                            local_a_start = true_a_start
                            
                        # Safe slicing to prevent out-of-bounds errors when shifting right
                        available_lats = max(0, audio_samples.shape[2] - local_a_start)
                        slice_len = min(req_a_len, available_lats)
                        
                        chunk_a_samples = audio_samples[:, :, local_a_start : local_a_start + slice_len].clone()
                        chunk_a_masks = audio_noise_mask[:, :, local_a_start : local_a_start + slice_len].clone()
                        
                        # Pad the tail with zeroes if the right-shift pushed us past the end of the master tensor
                        if chunk_a_samples.shape[2] < req_a_len:
                            pad_len = req_a_len - chunk_a_samples.shape[2]
                            chunk_a_samples = torch.nn.functional.pad(chunk_a_samples, (0,0,0,pad_len), value=0)
                            chunk_a_masks = torch.nn.functional.pad(chunk_a_masks, (0,0,0,pad_len), value=0)
                    
                    # --- VISUAL & AUDIO HARD-LOCK ---
                    if chunk_start_sec > shot_start_sec:
                        ctx_len = prev_v_lat - ctx_v_lat
                        chunk_v_masks[:, :, :ctx_len, :, :] = 0.0
                        
                        # If Internal or Reference Audio, capture the generated audio and write it to the Master Track!
                        if not has_audio_input:
                            ctx_a_frames = ((ctx_len - 1) * 8) + 1 if ctx_len > 0 else 0
                            ctx_a_len = get_audio_latents_func(ctx_a_frames, int(current_fps)) if ctx_a_frames > 0 else 0
                            if ctx_a_len > 0:
                                lock_len = min(ctx_a_len, chunk_a_masks.shape[2])
                                chunk_a_masks[:, :, :lock_len] = 0.0

                    # --- CONDITIONING PURGE (The Global Desync Fix) ---
                    # Text-to-Video works because it has no c_concat. Reference Mode breaks because the UNet 
                    # blindly concatenates the 17-frame image to EVERY chunk, stretching the video tensor!
                    # We MUST strip the image conditioning from Chunk 2+ so it relies purely on the video context.
                    
                    # 1. Safely isolate and unpack the specific shot condition (No for-loop needed!)
                    p_tensor, p_dict_orig = final_positive[s]
                    p_dict = p_dict_orig.copy()
                    if chunk_start_sec > shot_start_sec and "c_concat" in p_dict:
                        del p_dict["c_concat"]
                    chunk_pos = [[p_tensor, p_dict]]
                        
                    # 2. Safely iterate through the global negative conditions
                    chunk_neg = []
                    for n in final_negative:
                        n_dict = n[1].copy()
                        if chunk_start_sec > shot_start_sec and "c_concat" in n_dict:
                            del n_dict["c_concat"]
                        chunk_neg.append([n[0], n_dict])
                        
                    # Apply the purged conditioning to the guider for this specific chunk!
                    primary_guider.set_conds(chunk_pos, chunk_neg)
                    base_cb = latent_preview.prepare_callback(primary_guider.model_patcher, primary_sigmas.shape[-1] - 1, x0_output)
                    
                    # Update the callback string so the ComfyUI progress bar reflects the exact chunk!
                    callback = wrap_callback(base_cb, f"Shot {s+1}/{num_prompts} (Chunk {chunk_idx}/{total_chunks_in_shot})")
                    
                    # SLICE THE GLOBAL NOISE: Anchors the UNet's context to the absolute timeline!
                    abs_v_start = base_v_lat + ctx_v_lat
                    abs_a_start = base_a_lat + ctx_a_lat
                    chunk_noise_obj = BaseSlicedNoise(global_base_v_noise, global_base_a_noise, abs_v_start, abs_a_start)
                    
                    # --- THE MISSING DICTIONARY & NESTED TENSOR FIX ---
                    # Restore the lines that package the audio and video into ComfyUI's required formats
                    av_samples = comfy.nested_tensor.NestedTensor((chunk_v_samples, chunk_a_samples))
                    av_masks = comfy.nested_tensor.NestedTensor((chunk_v_masks, chunk_a_masks))
                    current_latent = {"samples": av_samples}
                    
                    if hasattr(model_to_use.model, "likeness_state"):
                        model_to_use.model.likeness_state["is_shot_start"] = (chunk_start_sec == shot_start_sec)
                        
                    sampled_chunk = primary_guider.sample(chunk_noise_obj.generate_noise(current_latent), current_latent["samples"], primary_sampler, primary_sigmas, denoise_mask=av_masks, callback=callback, disable_pbar=disable_pbar, seed=seed_number)
                    
                    # INTERNAL AUDIO FIX: Unbind both Video AND Audio!
                    res_v, res_a = sampled_chunk.unbind()
                    res_v = res_v.to(device)
                    res_a = res_a.to(device)
                    
                    # BATCH DIMENSION FAILSAFE: PyTorch unbind occasionally drops the batch dimension!
                    # Force tensors back to 5D (Video) and 4D (Audio) to protect the frame slicing math!
                    if res_v.ndim == 4:
                        res_v = res_v.unsqueeze(0)
                    if res_a.ndim == 3:
                        res_a = res_a.unsqueeze(0)
                    
                    # Inject the generated chunk back into the isolated shot array
                    if chunk_start_sec == shot_start_sec:
                        pass_v_samples[:, :, prev_v_lat:curr_v_lat] = res_v
                        
                        # If Internal or Reference Audio, capture the generated audio and write it to the Master Track!
                        if not has_audio_input:
                            actual_a_len = min(res_a.shape[2], curr_a_lat - prev_a_lat)
                            # CLAMP FIX: Prevent Out-Of-Bounds PyTorch Crash at the edge of the timeline!
                            actual_a_len = min(actual_a_len, audio_samples.shape[2] - true_a_start)
                            
                            if actual_a_len > 0:
                                pass_a_samples[:, :, prev_a_lat:prev_a_lat + actual_a_len] = res_a[:, :, :actual_a_len]
                                audio_samples[:, :, true_a_start : true_a_start + actual_a_len] = res_a[:, :, :actual_a_len]
                            
                    else:
                        gen_v_len = curr_v_lat - prev_v_lat
                        
                        # --- PURE HARD PASTE ---
                        pass_v_samples[:, :, prev_v_lat:curr_v_lat] = res_v[:, :, -gen_v_len:]
                        
                        # If Internal or Reference Audio, lock the context overlap so it doesn't hallucinate over the past!
                        if not has_audio_input:
                            gen_a_len = curr_a_lat - prev_a_lat
                            actual_gen_a = min(gen_a_len, res_a.shape[2])
                            
                            # CLAMP FIX: Safely anchor the right-bound of the audio slice so it never overflows!
                            true_a_end = min(true_a_start + req_a_len, audio_samples.shape[2])
                            actual_gen_a = min(actual_gen_a, true_a_end)
                            
                            if actual_gen_a > 0:
                                pass_a_samples[:, :, prev_a_lat:prev_a_lat + actual_gen_a] = res_a[:, :, -actual_gen_a:]
                                audio_samples[:, :, true_a_end - actual_gen_a : true_a_end] = res_a[:, :, -actual_gen_a:]
                    
                    chunk_idx += 1
                        
                # End of Autoregressive Loop for Shot S
                isolated_v_shots.append(pass_v_samples)
                isolated_a_shots.append(pass_a_samples)
            
            # 5. Concatenate the pristine, 0-indexed isolated shots end-to-end to form the timeline
            global_v_samples = torch.cat(isolated_v_shots, dim=2)
            global_a_samples = torch.cat(isolated_a_shots, dim=2)
            sampled_tensor = comfy.nested_tensor.NestedTensor((global_v_samples, global_a_samples)).to(device)
        
        else:
            # PURE PASSTHROUGH only triggers if ALL generation and upscaling stages are explicitly disabled!
            pure_image_passthrough = (latent_override is None) and (images is not None) and not (spatial_upscale or temporal_upscale)
            
            if pure_image_passthrough:
                print("\n--- Primary Sampling Bypassed: Pure Image Passthrough Detected ---")
                print("-> Routing input image(s) directly to pixel-level post-processing.")
                sampled_tensor = None
            elif (latent_override is None) and (images is not None):
                print("\n--- Primary Sampling Bypassed: Encoding Image(s) for Downstream Upscalers ---")
                # Safely format standard [Frames, Height, Width, Channels] into the 5D VAE structure
                img_in = images.unsqueeze(0) if images.ndim == 4 else images
                if img_in.shape[-1] > 3:
                    img_in = img_in[..., :3]
                    
                # Encode the raw images directly into the latent space at their native resolution!
                encoded_v = video_vae.encode(img_in).to(device)
                
                # Dynamically generate a mathematically matching audio tensor so the pipeline doesn't crash
                v_frames_encoded = ((encoded_v.shape[2] - 1) * 8) + 1 if encoded_v.shape[2] > 0 else 0
                req_a_latents = get_audio_latents_func(v_frames_encoded, int(current_fps))
                
                a_override_slice = torch.zeros((batch_size, z_channels, req_a_latents, audio_freq), device=device)
                if audio_samples is not None:
                    copy_len = min(req_a_latents, audio_samples.shape[2])
                    a_override_slice[:, :, :copy_len] = audio_samples[:, :, :copy_len]
                    
                sampled_tensor = comfy.nested_tensor.NestedTensor((encoded_v, a_override_slice)).to(device)
            else:
                print("\n--- Primary Sampling Bypassed (Injecting Override/Base Tensors) ---")
                sampled_tensor = comfy.nested_tensor.NestedTensor((video_samples, audio_samples)).to(device)

        # ==========================================
        # MID-GENERATION STAGE 1 PREVIEW (MP4 MUX)
        # ==========================================
        if enable_preview and (spatial_upscale or temporal_upscale) and not pure_image_passthrough:
            print("\n--- Generating Stage 1 Preview ---")
            import uuid
            uid = node_id if node_id is not None else str(uuid.uuid4())
            
            v_samps_prev, a_samps_prev = sampled_tensor.unbind()
            v_samps_prev = v_samps_prev.to(device)
            a_samps_prev = a_samps_prev.to(device)

            total_splits = 1
            if autoregressive_chunking and chunk_size_seconds < length_in_seconds:
                total_splits = int(round(length_in_seconds / chunk_size_seconds))
            if num_prompts > 1:
                total_splits = num_prompts # Override to ensure strict shot boundaries

            if total_splits > 1:
                if num_prompts == 1:
                    print(f"-> Executing Seamless Sliding Window VAE Decode ({total_splits} chunks) for preview...")
                    v_batch, v_channels, v_latents, v_h, v_w = v_samps_prev.shape
                    
                    temp_tile_len = v_latents // total_splits
                    temp_overlap = 4  # 32 frames of temporal VAE convolution overlap
                    
                    out_h = v_h * 32
                    out_w = v_w * 32
                    total_frames = (v_latents - 1) * 8 + 1
                    
                    preview_video = torch.zeros((v_batch, total_frames, out_h, out_w, 3), dtype=torch.float32)
                    
                    chunk_start = 0
                    valid_end_frame = 0
                    
                    while chunk_start < v_latents:
                        if chunk_start == 0:
                            chunk_end = min(chunk_start + temp_tile_len, v_latents)
                            overlap_start = chunk_start
                        else:
                            overlap_start = max(1, chunk_start - temp_overlap - 1)
                            chunk_end = min(chunk_start + temp_tile_len - (chunk_start - overlap_start), v_latents)
                            
                        if chunk_end <= chunk_start:
                            chunk_end = chunk_start + 1
                            
                        v_tile = v_samps_prev[:, :, overlap_start:chunk_end]
                        v_tile_decoded = video_vae.decode(v_tile).cpu()
                        current_valid_frames = v_tile_decoded.shape[1]
                        
                        if chunk_start == 0:
                            preview_video[:, :current_valid_frames] = v_tile_decoded
                            valid_end_frame = current_valid_frames
                        else:
                            global_start_frame = overlap_start * 8
                            overlap_frames = valid_end_frame - global_start_frame
                            
                            if overlap_frames > 0:
                                feather = torch.linspace(0.0, 1.0, overlap_frames).view(1, -1, 1, 1, 1)
                                prev_overlap = preview_video[:, global_start_frame : valid_end_frame]
                                new_overlap = v_tile_decoded[:, :overlap_frames]
                                
                                blended = prev_overlap * (1.0 - feather) + new_overlap * feather
                                preview_video[:, global_start_frame : valid_end_frame] = blended
                                
                                if current_valid_frames > overlap_frames:
                                    preview_video[:, valid_end_frame : global_start_frame + current_valid_frames] = v_tile_decoded[:, overlap_frames:]
                                    valid_end_frame = global_start_frame + current_valid_frames
                            else:
                                preview_video[:, valid_end_frame : valid_end_frame + current_valid_frames] = v_tile_decoded
                                valid_end_frame += current_valid_frames
                                
                        chunk_start = chunk_end
                else:
                    print(f"-> Executing Pure Isolated Array VAE Decode ({total_splits} chunks - CPU Offloaded)...")
                        
                    # PREVIEW FIX: Use base latents (v_samps_prev) and base boundaries since upscaling hasn't happened yet!
                    def get_preview_shot_boundary(s_idx):
                        if s_idx == 0: return 0
                        sec_val = s_idx * (length_in_seconds / num_prompts)
                        v_lat, _ = get_latent_counts(sec_val)
                        return v_lat
                        
                    frames_per_split_out = (length_in_seconds * current_fps) / total_splits
                    decoded_shots = []
                    for s in range(total_splits):
                        start_lat = get_preview_shot_boundary(s)
                        end_lat = get_preview_shot_boundary(s + 1) if s < total_splits - 1 else v_samps_prev.shape[2]
                            
                        split_latent = v_samps_prev[:, :, start_lat : end_lat]
                        decoded_shot = video_vae.decode(split_latent).cpu()
                        
                        target_shot_frames = int(round((s + 1) * frames_per_split_out)) - int(round(s * frames_per_split_out))
                        if s == 0 and out_ref_frame_count > 0:
                            target_shot_frames += out_ref_frame_count
                                
                        if video_mode == "text-to-video":
                            # USER FIX: T2V has no setup padding. Do NOT drop any frames!
                            pass
                        elif decoded_shot.shape[1] > target_shot_frames:
                            # DIRECTOR MODE CUT: Enforce strict, absolute boundary slicing
                            if s > 0 and not bypass_img_ref:
                                # Shots 2+: Slice EXACTLY the setup frames off the front
                                start_idx = out_ref_frame_count
                                end_idx = min(start_idx + target_shot_frames, decoded_shot.shape[1])
                                decoded_shot = decoded_shot[:, start_idx:end_idx]
                                print(f"   -> Trimmed exactly {out_ref_frame_count} static setup frames from the FRONT of Shot {s+1}")
                            else:
                                # Shot 1: Keep the setup frames and trim any VAE block-padding from the tail
                                decoded_shot = decoded_shot[:, :target_shot_frames]
                                
                        elif decoded_shot.shape[1] < target_shot_frames:
                            pad_len = target_shot_frames - decoded_shot.shape[1]
                            decoded_shot = torch.cat([decoded_shot, decoded_shot[:, -1:].repeat(1, pad_len, 1, 1, 1)], dim=1)
                                
                        decoded_shots.append(decoded_shot)
                    preview_video = torch.cat(decoded_shots, dim=1)
            else:
                preview_video = video_vae.decode(v_samps_prev)

            preview_video = preview_video[0] 
            
            # --- ASYMMETRIC PREVIEW TRIM ---
            preview_video_trim = out_ref_frame_count
            if video_mode == "reference-to-video":
                n_ref_images = max(1, ref_frame_count // duplicate_frames) if duplicate_frames > 0 else 1
                preview_video_trim += (2 * n_ref_images)
                
            num_preview_frames = preview_video.shape[0]
            if preview_video_trim >= num_preview_frames:
                preview_video = preview_video[-1:] 
            elif preview_video_trim > 0:
                preview_video = preview_video[preview_video_trim:]
                
            preview_video = (preview_video.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            temp_dir = folder_paths.get_temp_directory()
            preview_filename = f"stage1_preview_{uid}.mp4"
            preview_path = os.path.join(temp_dir, preview_filename)
            audio_path = os.path.join(temp_dir, f"stage1_preview_audio_{uid}.wav")
            out_sample_rate = int(getattr(audio_vae, "output_sample_rate", audio_vae.first_stage_model.output_sample_rate))
            save_sample_rate = out_sample_rate
            
            try:
                if has_audio_input:
                    print("--- Multiplexing Original Source Audio into Preview ---")
                    wf_out = master_wf[0].cpu()
                    if wf_out.ndim == 3:
                        wf_out = wf_out.squeeze(0)
                    save_sample_rate = sampling_rate 
                else:
                    print("--- Decoding Stage 1 Preview Audio ---")
                    if num_prompts > 1:
                        print("-> Applying Long Mirror Crossfade (fps/2) for seamless preview audio...")
                        sr = int(getattr(audio_vae, "output_sample_rate", audio_vae.first_stage_model.output_sample_rate))
                        a_latents_per_shot = a_samps_prev.shape[2] // num_prompts
                        
                        decoded_a_shots = []
                        for s in range(num_prompts):
                            s_lat = a_samps_prev[:, :, s * a_latents_per_shot : (s + 1) * a_latents_per_shot]
                            s_wf = audio_vae.decode(s_lat).to(device).movedim(-1, 1)
                            decoded_a_shots.append(s_wf)
                            
                        preview_audio_wf = torch.cat(decoded_a_shots, dim=-1)
                        
                        # Dynamic FPS/2 frames of audio overlap (~0.5 seconds)
                        fade_frames = current_fps / 2.0
                        fade_len = int((fade_frames / current_fps) * sr)
                        
                        # Safety clamp to ensure the fade doesn't consume the entire shot
                        shot_len_samples = int((length_in_seconds / num_prompts) * sr)
                        fade_len = min(fade_len, max(1, (shot_len_samples // 2) - 1))
                        
                        current_idx = 0
                        for s in range(num_prompts - 1):
                            current_idx += decoded_a_shots[s].shape[-1]
                            b_idx = current_idx
                            
                            if b_idx >= fade_len and b_idx + fade_len <= preview_audio_wf.shape[-1]:
                                # Extract the hard-cut edges
                                left_audio = preview_audio_wf[..., b_idx - fade_len : b_idx].clone()
                                right_audio = preview_audio_wf[..., b_idx : b_idx + fade_len].clone()
                                
                                # Create mirror extensions to maintain timeline length
                                left_extend = torch.cat([left_audio, left_audio.flip(-1)], dim=-1)
                                right_extend = torch.cat([right_audio.flip(-1), right_audio], dim=-1)
                                
                                # True Cross-Dissolve over the mirrored pads
                                fade_in = torch.linspace(0.0, 1.0, fade_len * 2, device=device, dtype=preview_audio_wf.dtype)
                                blended = left_extend * (1.0 - fade_in) + right_extend * fade_in
                                
                                # Paste the seamless transition back into the timeline
                                preview_audio_wf[..., b_idx - fade_len : b_idx + fade_len] = blended
                    else:
                        preview_audio_wf = audio_vae.decode(a_samps_prev).to(device).movedim(-1, 1)
                        
                    wf_out = preview_audio_wf[0].cpu()
                    if wf_out.ndim == 3:
                        wf_out = wf_out.squeeze(0)
                        
                time_to_drop = out_ref_frame_count / current_fps
                samples_to_drop = int(time_to_drop * save_sample_rate)
                total_samples = wf_out.shape[-1]
                
                if samples_to_drop >= total_samples:
                    wf_out = torch.zeros((wf_out.shape[0], 1), device=wf_out.device, dtype=wf_out.dtype)
                elif samples_to_drop > 0:
                    wf_out = wf_out[..., samples_to_drop:]
                    fade_samples = min(int(0.03 * save_sample_rate), wf_out.shape[-1])
                    if fade_samples > 0:
                        fade_tensor = torch.linspace(0.0, 1.0, fade_samples, device=wf_out.device, dtype=wf_out.dtype)
                        wf_out[..., :fade_samples] *= fade_tensor
                
                torchaudio.save(audio_path, wf_out, save_sample_rate)
            except Exception as e:
                print(f"Warning: Failed to decode preview audio: {e}")

            out_v_height_p = preview_video.shape[1]
            out_v_width_p = preview_video.shape[2]
            
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{out_v_width_p}x{out_v_height_p}",
                "-pix_fmt", "rgb24",
                "-r", str(current_fps),
                "-i", "-", 
            ]
            
            if os.path.exists(audio_path):
                cmd.extend(["-i", audio_path])
                
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "superfast", 
                "-crf", "28", 
                "-pix_fmt", "yuv420p", 
                "-r", str(current_fps) 
            ])

            if os.path.exists(audio_path):
                cmd.extend(["-c:a", "aac", "-b:a", "128k"])

            cmd.append(preview_path)

            try:
                print("--- Encoding Low-Bitrate MP4 Preview ---")
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                process.communicate(input=preview_video.tobytes())
            except FileNotFoundError:
                print("LTXV Custom Error: FFmpeg not found on system path. Cannot generate MP4 preview.")
            except Exception as e:
                print(f"LTXV Custom Error: FFmpeg failed to encode MP4 preview: {e}")

            PromptServer.instance.send_sync("trixope_ltxv_preview", {
                "node": uid, 
                "filename": preview_filename, 
                "type": "temp"
            })
            print(f"--- Stage 1 Preview Sent to UI ---")
            comfy.model_management.soft_empty_cache()

        # ==========================================
        # UNIVERSAL SLIDING WINDOW UPSCALER ENGINE
        # ==========================================
        def process_sliding_window_upscale(pass_name, is_temporal, v_samps, a_samps,
                                           upscaler_model, guider, sampler, sigmas,
                                           noise_obj, eta, bongmath, final_pixels, strengths,
                                           time_scale_factor, width_scale_factor, height_scale_factor, video_vae, current_fps, disable_pbar, seed_number, wrap_callback,
                                           temporal_positive=None, temporal_negative=None, video_mode="text-to-video"):

            v_batch, v_channels, v_frames, v_height, v_width = v_samps.shape
            temp_tile_len = 48
            temp_overlap = 8
            
            sim_start = 0
            num_chunks = 0
            while sim_start < v_frames:
                num_chunks += 1
                
                # SIMULATION FIX: The UI counter must perfectly mimic the Shot-Aware boundaries of the execution loop!
                shot_len_latents = v_frames / num_prompts 
                current_shot = min(int(sim_start // shot_len_latents), num_prompts - 1)
                shot_start = int(current_shot * shot_len_latents)
                shot_end = int((current_shot + 1) * shot_len_latents) if current_shot < num_prompts - 1 else v_frames
                
                if sim_start == shot_start:
                    sim_end = min(sim_start + temp_tile_len, shot_end)
                    sim_overlap = sim_start
                else:
                    sim_overlap = max(shot_start + 1, sim_start - temp_overlap - 1)
                    sim_end = min(sim_start + temp_tile_len - (sim_start - sim_overlap), shot_end)
                    
                if sim_end <= sim_start:
                    sim_end = sim_start + 1
                    
                sim_start = sim_end
            
            global_v_samps_up = None
            chunk_start = 0
            chunk_idx = 1
            sp_encoded_t_local = None
            
            a_mask_locked = torch.zeros_like(a_samps)
            
            while chunk_start < v_frames:
                comfy.model_management.soft_empty_cache()

                # SHOT-AWARE BOUNDARY MATH: Treats the stacked array as isolated slots!
                shot_len_latents = v_frames / num_prompts 
                current_shot = min(int(chunk_start // shot_len_latents), num_prompts - 1)
                shot_start = int(current_shot * shot_len_latents)
                shot_end = int((current_shot + 1) * shot_len_latents) if current_shot < num_prompts - 1 else v_frames
                
                if chunk_start == shot_start:
                    chunk_end = min(chunk_start + temp_tile_len, shot_end)
                    overlap_start = chunk_start
                else:
                    overlap_start = max(shot_start + 1, chunk_start - temp_overlap - 1)
                    chunk_end = min(chunk_start + temp_tile_len - (chunk_start - overlap_start), shot_end)
                    
                # FAILSAFE: Guarantee forward mathematical progress
                if chunk_end <= chunk_start:
                    chunk_end = chunk_start + 1
                    
                v_tile = v_samps[:, :, overlap_start:chunk_end]
                
                center_frame = overlap_start + (chunk_end - overlap_start) / 2.0
                frames_per_shot = v_frames / num_prompts
                shot_idx = int(center_frame / frames_per_shot)
                shot_idx = min(max(shot_idx, 0), num_prompts - 1)

                if is_temporal and temporal_positive is not None and temporal_negative is not None:
                    guider.set_conds([temporal_positive[shot_idx]], temporal_negative)
                else:
                    guider.set_conds([final_positive[shot_idx]], final_negative)

                pixel_start = overlap_start * time_scale_factor
                pixel_end = 1 + (chunk_end - 1) * time_scale_factor
                
                get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
                if get_audio_latents_func is None:
                    raise AttributeError("Audio VAE is missing 'num_of_latents_from_frames' method.")

                a_start = get_audio_latents_func(pixel_start, int(current_fps))
                a_end = get_audio_latents_func(pixel_end, int(current_fps))

                a_tile = a_samps[:, :, a_start:a_end]
                
                device_up = comfy.model_management.get_torch_device()
                model_dtype = next(upscaler_model.parameters()).dtype
                input_dtype = v_tile.dtype
                
                upscaler_model.to(device_up)
                v_tile_up = video_vae.first_stage_model.per_channel_statistics.un_normalize(v_tile.to(dtype=model_dtype, device=device_up))
                v_tile_up = upscaler_model(v_tile_up)
                upscaler_model.cpu()
                v_tile_up = video_vae.first_stage_model.per_channel_statistics.normalize(v_tile_up).to(dtype=input_dtype, device=device)
                
                print(f"\n--- {pass_name} Chunk {chunk_idx}/{num_chunks} (Dimensions: {v_tile_up.shape[4]*32}x{v_tile_up.shape[3]*32} | Shot {shot_idx+1}) ---")
                
                v_mask_tile = torch.ones((v_batch, v_tile_up.shape[2], v_tile_up.shape[3], v_tile_up.shape[4]), device=device, dtype=torch.float32)

                if chunk_start == 0:
                    v_mask_tile[:, :1, :, :] = 0.0

                if global_v_samps_up is None:
                    if is_temporal:
                        total_out_frames = (v_frames * 2) - 1
                        global_v_samps_up = torch.zeros((v_batch, v_tile_up.shape[1], total_out_frames, v_tile_up.shape[3], v_tile_up.shape[4]), device=device, dtype=input_dtype)
                    else:
                        global_v_samps_up = torch.zeros((v_batch, v_tile_up.shape[1], v_frames, v_tile_up.shape[3], v_tile_up.shape[4]), device=device, dtype=input_dtype)

                if chunk_start > 0:
                    overlap_in_frames = chunk_start - overlap_start
                    if is_temporal:
                        overlap_out_frames = overlap_in_frames * 2 - 1
                        global_start = overlap_start * 2
                    else:
                        overlap_out_frames = overlap_in_frames
                        global_start = overlap_start

                    feather = torch.linspace(0.0, 1.0, overlap_out_frames, device=device, dtype=torch.float32)
                    v_mask_tile[:, :overlap_out_frames, :, :] = feather.view(1, -1, 1, 1)

                    if sigmas is None:
                        prev_overlap = global_v_samps_up[:, :, global_start : global_start + overlap_out_frames]
                        curr_overlap = v_tile_up[:, :, :overlap_out_frames]
                        alpha = feather.view(1, 1, -1, 1, 1)
                        v_tile_up[:, :, :overlap_out_frames] = prev_overlap * (1.0 - alpha) + curr_overlap * alpha
                    else:
                        v_tile_up[:, :, :overlap_out_frames] = global_v_samps_up[:, :, global_start : global_start + overlap_out_frames]

                if is_temporal:
                    am_tile = a_mask_locked[:, :, a_start:a_end]
                    
                    current_latent_tile = {
                        "samples": comfy.nested_tensor.NestedTensor((v_tile_up, a_tile)),
                        "noise_mask": comfy.nested_tensor.NestedTensor((v_mask_tile.unsqueeze(1), am_tile)),
                        "sample_rate": sampling_rate,
                        "type": "audio"
                    }
                else:
                    am_tile = a_mask_locked[:, :, a_start:a_end]
                    current_latent_tile = {
                        "samples": comfy.nested_tensor.NestedTensor((v_tile_up, a_tile)),
                        "noise_mask": comfy.nested_tensor.NestedTensor((v_mask_tile.unsqueeze(1), am_tile)),
                        "sample_rate": sampling_rate,
                        "type": "audio"
                    }
                
                latent_image_tile = current_latent_tile["samples"]

                if sigmas is not None:
                    x0_output = {}
                    base_cb = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
                    callback = wrap_callback(base_cb, pass_name)

                    chunk_seed_number = seed_number + chunk_idx
                    chunk_noise_obj = Noise_RandomNoise(chunk_seed_number)
                    
                    sampled_chunk = guider.sample(chunk_noise_obj.generate_noise(current_latent_tile), latent_image_tile, sampler, sigmas, denoise_mask=current_latent_tile["noise_mask"], callback=callback, disable_pbar=disable_pbar, seed=chunk_seed_number)
                    sampled_v_tile = sampled_chunk.unbind()[0].to(device)
                else:
                    sampled_v_tile = v_tile_up

                # BATCH DIMENSION FAILSAFE
                if sampled_v_tile.ndim == 4:
                    sampled_v_tile = sampled_v_tile.unsqueeze(0)

                if chunk_start == 0:
                    global_v_samps_up[:, :, :sampled_v_tile.shape[2]] = sampled_v_tile
                else:
                    overlap_len = min(overlap_out_frames, sampled_v_tile.shape[2])

                    prev_chunk_end = global_v_samps_up[:, :, global_start : global_start + overlap_len]
                    new_chunk_start = sampled_v_tile[:, :, :overlap_len]

                    blend_curve = torch.linspace(0.0, 1.0, overlap_len, device=device, dtype=torch.float32).view(1, -1, 1, 1)
                    blended_overlap = prev_chunk_end * (1.0 - blend_curve) + new_chunk_start * blend_curve

                    global_v_samps_up[:, :, global_start : global_start + overlap_len] = blended_overlap

                    if sampled_v_tile.shape[2] > overlap_len:
                        global_v_samps_up[:, :, global_start + overlap_len : global_start + sampled_v_tile.shape[2]] = sampled_v_tile[:, :, overlap_len:]

                chunk_start = chunk_end
                chunk_idx += 1

            return global_v_samps_up

        # ==========================================
        # 6.5 UPSCALING PASS (SPATIAL)
        # ==========================================
        if spatial_upscale and not pure_image_passthrough:
            if spatial_upscaler is None:
                raise ValueError("Spatial upscaler model is required if spatial_upscale is True.")
            
            # THE ANCESTRAL GUARD: Prevent haziness with gradient masks!
            safe_sampler_name = spatial_sampler
            #if "ancestral" in safe_sampler_name.lower() or "sde" in safe_sampler_name.lower():
            #    print(f"\n--- LTXV OVERRIDE: Forcing '{safe_sampler_name}' to 'euler' for Spatial Upscale to prevent boundary haziness! ---")
            #    safe_sampler_name = "euler"

            upsample_sampler = build_custom_sampler(safe_sampler_name, eta, bongmath)
            sigmas_list = re.findall(r"[-+]?(?:\d*\.*\d+)", spatial_sigmas)
            spatial_sigmas_tensor = torch.FloatTensor([float(i) for i in sigmas_list])

            for stage in range(1, spatial_passes + 1):
                v_samps, a_samps = sampled_tensor.unbind()
                v_samps = v_samps.to(device)
                a_samps = a_samps.to(device)

                global_v_samps_up = process_sliding_window_upscale(
                    pass_name=f"Spatial Upscale Pass {stage}/{spatial_passes}",
                    is_temporal=False, v_samps=v_samps, a_samps=a_samps,
                    upscaler_model=spatial_upscaler, guider=upsample_guider, sampler=upsample_sampler, sigmas=spatial_sigmas_tensor,
                    noise_obj=noise_obj, eta=eta, bongmath=bongmath, final_pixels=final_pixels, strengths=strengths,
                    time_scale_factor=time_scale_factor, width_scale_factor=width_scale_factor, height_scale_factor=height_scale_factor, video_vae=video_vae, current_fps=current_fps, disable_pbar=disable_pbar, seed_number=seed_number, wrap_callback=wrap_callback
                )

                sampled_tensor = comfy.nested_tensor.NestedTensor((global_v_samps_up, a_samps)).to(device)

        # ==========================================
        # 7. THE POST-CLEANSE HARD OVERWRITE
        # ==========================================
        final_latent = None
        video_out_latent = None
        audio_out_latent = None

        if not pure_image_passthrough:
            unbound_samples = sampled_tensor.unbind()
            final_video_samples = unbound_samples[0].to(device)
            final_audio_samples = unbound_samples[1].clone().to(device)

            if has_audio_ref or has_audio_input:
                max_latents = final_audio_samples.shape[2]
                master_max = master_latents.shape[2]
                
                if has_audio_ref:
                    lock_a = min(setup_total_latents, max_latents, master_max)
                    if lock_a > 0:
                        final_audio_samples[:, :, :lock_a, :] = master_latents[:, :, :lock_a, :]

                if has_audio_input:
                    start_c = min(setup_total_latents, max_latents)
                    target_len = max_latents - start_c
                    available_master = master_max - start_c
                    copy_len = min(target_len, available_master)
                    
                    if copy_len > 0:
                        final_audio_samples[:, :, start_c : start_c + copy_len, :] = \
                            master_latents[:, :, start_c : start_c + copy_len, :]

            v_batch, _, v_frames, v_height, v_width = final_video_samples.shape
            
            final_v_mask = torch.ones((v_batch, v_frames, v_height, v_width), dtype=torch.float32, device=device)
            final_a_mask = torch.ones_like(final_audio_samples)
            
            final_latent = {
                "samples": comfy.nested_tensor.NestedTensor((final_video_samples, final_audio_samples)),
                "noise_mask": comfy.nested_tensor.NestedTensor((final_v_mask.unsqueeze(1), final_a_mask)),
                "sample_rate": sampling_rate,
                "type": "audio"
            }
            
            video_out_latent = {
                "samples": final_video_samples,
                "noise_mask": final_v_mask 
            }
            
            audio_out_latent = {
                "samples": final_audio_samples,
                "noise_mask": final_a_mask,
                "sample_rate": sampling_rate,
                "type": "audio"
            }

        # ==========================================
        # 7.5 ISOLATED TEMPORAL UPSCALE PASS (NATIVE NODES + SLIDING WINDOW)
        # ==========================================
        if temporal_upscale and not pure_image_passthrough:
            if temporal_upscaler is None:
                raise ValueError("Temporal upscaler model is required if temporal_upscale is True.")
                
            print("\n--- Running Temporal Upscale Pass ---")
            import sys
            
            # Universal helper function to safely instantiate and execute ComfyUI nodes
            def run_node(class_name, **kwargs):
                import sys
                nodes_module = sys.modules.get("nodes")
                if not nodes_module or class_name not in getattr(nodes_module, "NODE_CLASS_MAPPINGS", {}):
                    raise ValueError(f"Required node '{class_name}' not found in ComfyUI.")
                cls = nodes_module.NODE_CLASS_MAPPINGS[class_name]
                func_name = getattr(cls, "FUNCTION", "execute")
                return getattr(cls(), func_name)(**kwargs)

            # Store the original reference frame count before the optimization wipes it!
            original_ref_frame_count = out_ref_frame_count

            # =========================================================
            # 0. THE "LOB IT OFF" OPTIMIZATION (REFERENCE MODE ONLY)
            # =========================================================
            temp_video_mode = video_mode
            if video_mode in ["reference", "reference-to-video"]:
                print("-> LOBBING OFF static setup latents to optimize Temporal Upscale...")
                
                # EXACT MATH: Calculate exact latents injected during setup
                v_setup_base = ((out_ref_frame_count - 1) // 8) if out_ref_frame_count > 0 else 0
                
                get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
                # Number of frames to drop is exactly out_ref_frame_count
                a_setup_base = get_audio_latents_func(out_ref_frame_count, int(current_fps)) if v_setup_base > 0 else 0
                
                # Permanently slice the setup latents off the tensors for the upscale pass!
                if final_video_samples.shape[2] > v_setup_base:
                    final_video_samples = final_video_samples[:, :, v_setup_base:]
                if final_audio_samples.shape[2] > a_setup_base:
                    final_audio_samples = final_audio_samples[:, :, a_setup_base:]
                    
                # Trick the rest of the temporal pass into acting exactly like Text-to-Video!
                temp_video_mode = "text-to-video"
                
                # Tell the final audio/video muxer there are no setup frames left to drop!
                out_ref_frame_count = 0

            # ---------------------------------------------------------
            # 1. GLOBAL PREP: AUDIO ALIGNMENT & STRETCH (DUAL-PATH)
            # ---------------------------------------------------------
            print(f"-> Preparing Audio Alignment (Isolated Mode: {temp_video_mode})...")
            
            is_source_audio = (audio is not None and audio_select == "source")
            
            if is_source_audio:
                print("-> SOURCE AUDIO DETECTED: Executing Phase Vocoder Stretch & Base-FPS Conditioning...")
                import torchaudio.functional as TA_F
                input_audio = audio
                source_wf = input_audio["waveform"].to(device, dtype=torch.float32)
                source_sr = input_audio["sample_rate"]
                
                is_batched = source_wf.ndim == 3
                wf_sq = source_wf.squeeze(0) if is_batched else source_wf
                
                rate = 0.5
                fft_size = 2048
                hop_size = fft_size // 4
                win_length = fft_size
                window = torch.hann_window(win_length, device=device)
                
                complex_spectogram = torch.stft(wf_sq, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True)
                phase_advance = torch.linspace(0, math.pi * hop_size, complex_spectogram.shape[1], device=device)[..., None]
                stretched_spectogram = TA_F.phase_vocoder(complex_spectogram, rate, phase_advance)
                stretched_wf_sq = torch.istft(stretched_spectogram, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window)
                
                stretched_wf = stretched_wf_sq.unsqueeze(0) if is_batched else stretched_wf_sq
                
                target_length = int(source_wf.shape[-1] / rate)
                if stretched_wf.shape[-1] > target_length:
                    stretched_wf = stretched_wf[..., :target_length]
                elif stretched_wf.shape[-1] < target_length:
                    pad_amount = target_length - stretched_wf.shape[-1]
                    stretched_wf = torch.nn.functional.pad(stretched_wf, (0, pad_amount))
                    
                # --- TEMPORAL SYNC FIX ---
                # The "Lob It Off" math drops 16 frames of setup, leaving 1 static frame behind.
                # We MUST pad the stretched audio with 1 frame of silence so the song starts perfectly on frame 1!
                if video_mode in ["reference", "reference-to-video"]:
                    v_setup_base = ((original_ref_frame_count - 1) // 8) if original_ref_frame_count > 0 else 0
                    remaining_silence_frames = original_ref_frame_count - (v_setup_base * 8)
                    if remaining_silence_frames > 0:
                        # Multiply by 2 because we are in the 48fps upscaled temporal timeline!
                        upscaled_silence_frames = remaining_silence_frames * 2
                        silence_samples = int((upscaled_silence_frames / (current_fps * 2)) * source_sr)
                        silence_pad = torch.zeros((stretched_wf.shape[0], stretched_wf.shape[1], silence_samples), device=device, dtype=stretched_wf.dtype)
                        stretched_wf = torch.cat([silence_pad, stretched_wf], dim=-1)
                    
                shifted_audio = {"waveform": stretched_wf.cpu(), "sample_rate": source_sr}
                
                import sys
                nodes_module = sys.modules.get("nodes")
                cls = nodes_module.NODE_CLASS_MAPPINGS["LTXVAudioVAEEncode"]
                audio_latent_out = getattr(cls(), "execute")(audio=shifted_audio, audio_vae=audio_vae)[0]
                stretched_audio_latents = audio_latent_out["samples"].to(device)
                
                cond_fps = float(current_fps)
            else:
                print("-> INTERNAL AUDIO DETECTED: Bypassing Stretch & Executing Target-FPS Conditioning...")
                base_audio_latents = final_audio_samples
                cond_fps = float(current_fps * 2)

            import node_helpers
            cond_out_pos = node_helpers.conditioning_set_values(final_positive, {"frame_rate": cond_fps})
            cond_out_neg = node_helpers.conditioning_set_values(final_negative, {"frame_rate": cond_fps})
            
            crop_pos = node_helpers.conditioning_set_values(cond_out_pos, {"keyframe_idxs": None, "guide_attention_entries": None})
            crop_neg = node_helpers.conditioning_set_values(cond_out_neg, {"keyframe_idxs": None, "guide_attention_entries": None})

            # ---------------------------------------------------------
            # 2. GLOBAL INJECTION SETUP (Inlined LTXVImgToVideoInplace)
            # ---------------------------------------------------------
            print("-> Preparing LTXVImgToVideoInplace Global Injections (Strength: 1.0, Bypass: False)...")
            v_batch, v_channels, v_frames_base, v_height, v_width = final_video_samples.shape
            total_out_frames = (v_frames_base * 2) - 1
            
            global_injection_mask = torch.ones((v_batch, 1, total_out_frames, 1, 1), device=device, dtype=torch.float32)
            global_injection_samples = torch.zeros((v_batch, v_channels, total_out_frames, v_height, v_width), device=device, dtype=final_video_samples.dtype)

            shot_duration = length_in_seconds / num_prompts

            # We must redefine the index grabber so it doesn't accidentally add the lobbed-off frames back in!
            def get_shot_start_latent_idx(s_idx, shot_dur):
                if s_idx == 0:
                    return 0
                sec_val = s_idx * shot_dur
                v_lat, _ = get_latent_counts(sec_val)
                if video_mode in ["reference", "reference-to-video"]:
                    # We lobbed off v_setup_base, so we must subtract it from subsequent shot indices!
                    return max(0, (v_lat - 1) - v_setup_base)
                return max(0, v_lat - 1)

            for s in range(num_prompts):
                v_idx = get_shot_start_latent_idx(s, shot_duration)
                v_lat_up = v_idx * 2
                frames_to_encode = None
                
                # Because temp_video_mode is strictly 'text' or 'image' now, the logic is ultra-clean!
                if temp_video_mode in ["text", "text-to-video"]:
                    base_latent = final_video_samples[:, :, v_idx:v_idx+1]
                    if base_latent.shape[2] > 0:
                        frames_to_encode = video_vae.decode(base_latent)
                elif temp_video_mode in ["image", "image-to-video"]:
                    if first_frame is not None:
                        img_idx = min(s, first_frame.shape[0] - 1)
                        frames_to_encode = first_frame[img_idx:img_idx+1]
                    else:
                        base_latent = final_video_samples[:, :, v_idx:v_idx+1]
                        if base_latent.shape[2] > 0:
                            frames_to_encode = video_vae.decode(base_latent)

                # Universal Flattening & Encoding
                if frames_to_encode is not None:
                    if frames_to_encode.ndim == 5:
                        b_, f_, h_, w_, c_ = frames_to_encode.shape
                        frames_to_encode = frames_to_encode.view(b_ * f_, h_, w_, c_)

                    # Slice to exactly 1 anchor frame!
                    frames_to_encode = frames_to_encode[:1]

                    t_width = v_width * 32
                    t_height = v_height * 32
                    if frames_to_encode.shape[1] != t_height or frames_to_encode.shape[2] != t_width:
                        frames_to_encode = comfy.utils.common_upscale(frames_to_encode.movedim(-1, 1), t_width, t_height, "bilinear", "center").movedim(1, -1)
                    
                    encoded_t = video_vae.encode(frames_to_encode[:, :, :, :3]).to(device)
                    inject_len = encoded_t.shape[2]
                    
                    if v_lat_up + inject_len <= total_out_frames:
                        global_injection_samples[:, :, v_lat_up : v_lat_up + inject_len] = encoded_t
                        global_injection_mask[:, :, v_lat_up : v_lat_up + inject_len] = 0.0
                    else:
                        rem = total_out_frames - v_lat_up
                        if rem > 0:
                            global_injection_samples[:, :, v_lat_up : v_lat_up + rem] = encoded_t[:, :, :rem]
                            global_injection_mask[:, :, v_lat_up : v_lat_up + rem] = 0.0

            # ---------------------------------------------------------
            # 3. SLIDING WINDOW & CONTEXT LOOKBACK SETUP
            # ---------------------------------------------------------
            temporal_context_seconds = 5.0
            # Convert 5 seconds into exact latent counts (15 latents at 24fps)
            temp_overlap = max(1, int((temporal_context_seconds * current_fps) / 8))
            
            # VRAM Safety Cap: Max latents per pass (Context + New Frames)
            # 22 latents = ~7.3 seconds of base video -> extremely safe for 24GB VRAM
            max_vram_latents = 40
            new_latents_per_chunk = max(2, max_vram_latents - temp_overlap)
            
            global_v_temporal = torch.zeros((v_batch, v_channels, total_out_frames, v_height, v_width), device=device, dtype=final_video_samples.dtype)

            # ---------------------------------------------------------
            # 3.5 CHUNK SIMULATION (FOR UI PROGRESS BAR)
            # ---------------------------------------------------------
            sim_start = 0
            num_chunks = 0
            while sim_start < v_frames_base:
                num_chunks += 1
                if sim_start == 0:
                    sim_end = min(sim_start + max_vram_latents, v_frames_base)
                else:
                    sim_end = min(sim_start + new_latents_per_chunk, v_frames_base)
                    
                if sim_end <= sim_start:
                    sim_end = sim_start + 1
                    
                if sim_end >= v_frames_base:
                    break
                    
                # Step back by 3 base latents (6 upscaled frames) to create a wide, buttery crossfade!
                sim_start = sim_end - 3
                if sim_start < 0: sim_start = 0

            chunk_start = 0
            chunk_idx = 1
            valid_global_end = 0  # <--- Initialize our boundary tracker!
            v_time_scale = video_vae.downscale_index_formula[0]
            get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))

            # --- MODEL 3 OVERRIDE FIX ---
            # Correctly map the temporal sampler to the fully patched model3_to_use!
            temporal_model = model3_to_use

            # --- THE TEXTURE LOCK: Global Noise Anchoring ---
            print("-> Anchoring Global Noise Tensor to freeze background textures...")
            generator = torch.manual_seed(seed_number)
            global_noise_tensor = torch.randn((v_batch, v_channels, total_out_frames, v_height, v_width), device=device, dtype=torch.float32, generator=generator)
            
            class SlicedNoise:
                def __init__(self, global_noise, start_idx, seed):
                    self.global_noise = global_noise
                    self.start_idx = start_idx
                    self.seed = seed
                    
                def generate_noise(self, input_latent):
                    latent_image = input_latent["samples"]
                    batch_inds = input_latent.get("batch_index", None)
                    
                    # 1. Generate the standard nested noise (handles audio & video shapes perfectly)
                    base_noise = comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)
                    
                    # 2. Unpack the nested noise
                    v_noise, a_noise = base_noise.unbind()
                    
                    # 3. Swap the random video noise for our locked global slice!
                    length = v_noise.shape[2]
                    v_noise_sliced = self.global_noise[:, :, self.start_idx : self.start_idx + length].clone().to(v_noise.device)
                    if v_noise_sliced.shape[2] < length:
                        pad_len = length - v_noise_sliced.shape[2]
                        v_noise_sliced = torch.nn.functional.pad(v_noise_sliced, (0, 0, 0, 0, 0, pad_len))
                    
                    # 4. Repackage the NestedTensor so the UNet doesn't crash!
                    return comfy.nested_tensor.NestedTensor((v_noise_sliced, a_noise))

            # ---------------------------------------------------------
            # 4. CHUNKING LOOP (Restored Seamless Sliding Window)
            # ---------------------------------------------------------
            while chunk_start < v_frames_base:
                comfy.model_management.soft_empty_cache()

                if chunk_start == 0:
                    overlap_start = 0
                    chunk_end = min(chunk_start + max_vram_latents, v_frames_base)
                else:
                    # Look back 5 seconds into the past!
                    overlap_start = max(0, chunk_start - temp_overlap)
                    chunk_end = min(chunk_start + new_latents_per_chunk, v_frames_base)
                    
                if chunk_end <= chunk_start:
                    chunk_end = chunk_start + 1

                # Slice Video
                v_tile = final_video_samples[:, :, overlap_start:chunk_end]

                # Map & Slice Audio to Timeline (Dynamic Path)
                tile_in_frames = chunk_end - overlap_start
                tile_out_frames = (tile_in_frames * 2) - 1
                up_overlap_start = overlap_start * 2
                
                pixel_start = up_overlap_start * v_time_scale
                pixel_out_frames = 1 + (tile_out_frames - 1) * v_time_scale
                pixel_end = pixel_start + pixel_out_frames - 1
                
                # Helper to grab absolute latents securely from frame indices
                def get_abs_a_lat(f_idx, fps):
                    return get_audio_latents_func(int(f_idx + 1), int(fps)) if f_idx > 0 else 0
                
                if is_source_audio:
                    # SOURCE PATH: Use stretched latents and original fps to preserve external song timing
                    target_fps = int(current_fps)
                    a_start = get_abs_a_lat(pixel_start, target_fps)
                    a_end_ideal = get_abs_a_lat(pixel_end, target_fps)
                    req_a_len = a_end_ideal - a_start
                    
                    a_end = min(a_start + req_a_len, stretched_audio_latents.shape[2])
                    a_tile = stretched_audio_latents[:, :, a_start:a_end]
                else:
                    # INTERNAL PATH: Use original latents and target fps to perfectly lock generated lips
                    target_fps = int(current_fps * 2)
                    a_start = get_abs_a_lat(pixel_start, target_fps)
                    a_end_ideal = get_abs_a_lat(pixel_end, target_fps)
                    req_a_len = a_end_ideal - a_start
                    
                    a_end = min(a_start + req_a_len, base_audio_latents.shape[2])
                    a_tile = base_audio_latents[:, :, a_start:a_end]

                # Universal Safety Pad
                if a_tile.shape[2] > req_a_len:
                    a_tile = a_tile[:, :, :req_a_len]
                elif a_tile.shape[2] < req_a_len:
                    pad_len = req_a_len - a_tile.shape[2]
                    a_tile = torch.nn.functional.pad(a_tile, (0,0,0,pad_len), value=0)

                up_device = comfy.model_management.get_torch_device()
                model_dtype = next(temporal_upscaler.parameters()).dtype
                input_dtype = v_tile.dtype
                
                memory_required = comfy.model_management.module_size(temporal_upscaler)
                memory_required += math.prod(v_tile.shape) * 3000.0
                comfy.model_management.free_memory(memory_required, up_device)
                
                try:
                    temporal_upscaler.to(up_device)
                    v_tile_ready = v_tile.to(dtype=model_dtype, device=up_device)
                    v_tile_unnorm = video_vae.first_stage_model.per_channel_statistics.un_normalize(v_tile_ready)
                    v_upsampled_raw = temporal_upscaler(v_tile_unnorm)
                finally:
                    temporal_upscaler.cpu()
                    
                v_upsampled = video_vae.first_stage_model.per_channel_statistics.normalize(v_upsampled_raw)
                v_upsampled = v_upsampled.to(dtype=input_dtype, device=device)

                # Native 5D Gradient Video Mask for chunk overlaps
                v_noise_mask = torch.ones((v_batch, 1, v_upsampled.shape[2], 1, 1), dtype=torch.float32, device=device)
                
                if chunk_start > 0:
                    overlap_in_frames = chunk_start - overlap_start
                    overlap_out_frames = overlap_in_frames * 2 - 1
                    global_start = overlap_start * 2
                    
                    # 1. HARD LOCK THE CONTEXT: Denoise mask = 0.0. No feathering!
                    v_noise_mask[:, :, :overlap_out_frames] = 0.0
                        
                    # 2. PASTE THE PAST: Inject the previously generated timeline into the current tile
                    v_upsampled[:, :, :overlap_out_frames] = global_v_temporal[:, :, global_start : global_start + overlap_out_frames]

                # Slice the global locked frames to fit the current chunk
                v_inject_tile = global_injection_samples[:, :, up_overlap_start : up_overlap_start + tile_out_frames]
                m_inject_tile = global_injection_mask[:, :, up_overlap_start : up_overlap_start + tile_out_frames]
                
                # Overwrite the upsampled frames with the locked frames and mask out the noise
                v_upsampled = torch.where(m_inject_tile == 0.0, v_inject_tile, v_upsampled)
                v_noise_mask = torch.min(v_noise_mask, m_inject_tile)

                # Native 4D Audio Mask (0.0 strength)
                a_noise_mask = torch.zeros((v_batch, 1, a_tile.shape[2], a_tile.shape[3]), dtype=torch.float32, device=device)
                
                concat_latent = {
                    "samples": comfy.nested_tensor.NestedTensor((v_upsampled, a_tile)),
                    "noise_mask": comfy.nested_tensor.NestedTensor((v_noise_mask, a_noise_mask)),
                    "sample_rate": sampling_rate,
                    "type": "audio"
                }

                # Update UI Progress Bar
                PromptServer.instance.send_sync("ltxv_eta_update", {
                    "node_id": node_id, "step": 1, "total_steps": 1, "chunk": chunk_idx,
                    "total_chunks": num_chunks, "global_step": chunk_idx,
                    "total_global_steps": num_chunks, "pass_name": "Temporal Upscale Pass",
                    "is_face_restore": bool(restore_faces)
                })
                
                print(f"-> Temporal Upscale Chunk {chunk_idx}/{num_chunks}...")

                # 1. Setup the CFG Guider
                temporal_guider = comfy.samplers.CFGGuider(temporal_model)
                temporal_guider.set_cfg(1.0)
                
                center_latent = overlap_start + ((chunk_end - overlap_start) / 2.0)
                shot_len_latents = v_frames_base / num_prompts
                shot_idx = int(center_latent // shot_len_latents)
                shot_idx = min(max(shot_idx, 0), num_prompts - 1)
                
                chunk_pos = [crop_pos[shot_idx]]
                chunk_neg = [crop_neg[shot_idx]] if len(crop_neg) > shot_idx else crop_neg
                temporal_guider.set_conds(chunk_pos, chunk_neg)

                # 2. Setup the Sampler & Scheduler
                temporal_sampler = comfy.samplers.sampler_object("euler_cfg_pp")
                temporal_model_sampling = temporal_model.get_model_object("model_sampling")

                # 3. Exact KSampler Math (4 Steps)
                node_steps = 4
                node_denoise = temporal_denoise
                total_schedule_steps = int(node_steps / node_denoise)
                total_sigmas = comfy.samplers.calculate_sigmas(temporal_model_sampling, "linear_quadratic", total_schedule_steps)
                temporal_sigmas_tensor = total_sigmas[-(node_steps + 1):].to(device)

                chunk_seed_number = seed_number + chunk_idx

                # 4. Extract Globally Anchored Noise!
                chunk_noise_obj = SlicedNoise(global_noise_tensor, up_overlap_start, chunk_seed_number)
                noise = chunk_noise_obj.generate_noise(concat_latent)

                # 5. Execute the Denoise Pass using the TEMPORAL Guider (Not Primary!)
                sampled_latent_raw = temporal_guider.sample(
                    noise, 
                    concat_latent["samples"], 
                    temporal_sampler, 
                    temporal_sigmas_tensor, 
                    denoise_mask=concat_latent["noise_mask"], 
                    callback=None, 
                    disable_pbar=disable_pbar, 
                    seed=seed_number
                )
                sampled_latent = {"samples": sampled_latent_raw}

                unbound_latents = sampled_latent["samples"].unbind()
                sampled_v_tile = unbound_latents[0].to(device)

                # BATCH DIMENSION FAILSAFE
                if sampled_v_tile.ndim == 4:
                    sampled_v_tile = sampled_v_tile.unsqueeze(0)

                if chunk_start == 0:
                    global_v_temporal[:, :, :sampled_v_tile.shape[2]] = sampled_v_tile
                else:
                    # EXACT HARD PASTE: Because the noise is globally anchored, the textures match perfectly!
                    new_frames = sampled_v_tile.shape[2] - overlap_out_frames
                    if new_frames > 0:
                        paste_start = up_overlap_start + overlap_out_frames
                        global_v_temporal[:, :, paste_start : paste_start + new_frames] = sampled_v_tile[:, :, overlap_out_frames:]

                # SEVER THE INFINITE LOOP: If we hit the absolute end of the video, we are done!
                if chunk_end >= v_frames_base:
                    break

                # Step back by 3 base latents (6 upscaled frames) to trigger the overlap!
                chunk_start = chunk_end - 3
                if chunk_start < 0: chunk_start = 0
                chunk_idx += 1

            final_video_samples = global_v_temporal
            current_fps *= 2
            
            # THE CRITICAL SYNC FIX: If lob-it-off ran, setup frames are gone. But we disabled it above, 
            # so we just need to ensure the variable tracks correctly!
            out_ref_frame_count = 0

            for p in final_positive:
                p[1]["frame_rate"] = float(current_fps)
            for n in final_negative:
                n[1]["frame_rate"] = float(current_fps)

            v_batch, _, v_frames, v_height, v_width = final_video_samples.shape
            final_v_mask = torch.ones((v_batch, v_frames, v_height, v_width), dtype=torch.float32, device=device)
            final_a_mask = torch.ones_like(final_audio_samples)

            final_latent = {
                "samples": comfy.nested_tensor.NestedTensor((final_video_samples, final_audio_samples)),
                "noise_mask": comfy.nested_tensor.NestedTensor((final_v_mask.unsqueeze(1), final_a_mask)),
                "sample_rate": sampling_rate,
                "type": "audio"
            }
            video_out_latent = {
                "samples": final_video_samples,
                "noise_mask": final_v_mask
            }
            audio_out_latent = {
                "samples": final_audio_samples,
                "noise_mask": final_a_mask,
                "sample_rate": sampling_rate,
                "type": "audio"
            }

        # ==========================================
        # 8. INTEGRATED VAE DECODE, RESTORE & POST-SLICE A/V
        # ==========================================
        out_image = None
        out_audio = None
        out_video = None
        
        if decode:
            print(f"\n--- Running Integrated Decode & Slicer ({num_prompts} shots) ---")
            
            fr_device = comfy.model_management.get_torch_device()
            face_helper = None
            loaded_facerestore_model = None
            
            if restore_faces and facerestore_model != "None":
                try:
                    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
                    from torchvision.transforms.functional import normalize
                    from basicsr.utils.registry import ARCH_REGISTRY
                    from comfy_extras.chainner_models import model_loading
                    
                    print("\n--- Face Restoration Engine Online ---")
                    
                    model_path = folder_paths.get_full_path("trixope_facerestore", facerestore_model)
                    if "codeformer" in facerestore_model.lower():
                        print(f'\tLoading CodeFormer: {facerestore_model}')
                        codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
                            dim_embd=512,
                            codebook_size=1024,
                            n_head=8,
                            n_layers=9,
                            connect_list=["32", "64", "128", "256"],
                        ).to(fr_device)
                        checkpoint = torch.load(model_path)["params_ema"]
                        codeformer_net.load_state_dict(checkpoint)
                        loaded_facerestore_model = codeformer_net.eval()
                    else:
                        print(f'\tLoading FaceRestore Model: {facerestore_model}')
                        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
                        loaded_facerestore_model = model_loading.load_state_dict(sd).eval().to(fr_device)
                        
                    face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model=facedetection, save_ext='png', use_parse=True, device=fr_device)
                    
                    if hasattr(face_helper, 'face_det') and face_helper.face_det is not None:
                        for param in face_helper.face_det.parameters():
                            param.data = param.data.to(fr_device)
                        for buffer in face_helper.face_det.buffers():
                            buffer.data = buffer.data.to(fr_device)
                            
                    if hasattr(face_helper, 'face_parse') and face_helper.face_parse is not None:
                        for param in face_helper.face_parse.parameters():
                            param.data = param.data.to(fr_device)
                        for buffer in face_helper.face_parse.buffers():
                            buffer.data = buffer.data.to(fr_device)
                        
                except Exception as e:
                    print(f"\nWARNING: Face restoration initialization failed: {e}")
                    restore_faces = False

            if pure_image_passthrough:
                print("-> PURE PASSTHROUGH: Bypassing VAE, Face Restore, and Audio Slicing entirely.")
                out_image = images.cpu()
                out_audio = audio
                out_ref_frame_count = 0
            else:
                # ---------------------------------------------------------
                # STANDARD DECODE PATH: VAE, Face Restore, Sync & Slice
                # ---------------------------------------------------------
                total_splits = 1
                if autoregressive_chunking and chunk_size_seconds < length_in_seconds:
                    total_splits = int(round(length_in_seconds / chunk_size_seconds))
                if num_prompts > 1:
                    total_splits = num_prompts

                if total_splits > 1:
                    if num_prompts == 1:
                        print(f"-> Executing Seamless Sliding Window VAE Decode ({total_splits} chunks - CPU Offloaded)...")
                        v_batch, v_channels, v_latents, v_h, v_w = final_video_samples.shape
                        temp_tile_len = v_latents // total_splits
                        temp_overlap = 4  
                        
                        out_h = v_h * 32
                        out_w = v_w * 32
                        total_frames = (v_latents - 1) * 8 + 1
                        decoded_video = torch.zeros((v_batch, total_frames, out_h, out_w, 3), dtype=torch.float32)
                        
                        chunk_start = 0
                        valid_end_frame = 0
                        while chunk_start < v_latents:
                            if chunk_start == 0:
                                chunk_end = min(chunk_start + temp_tile_len, v_latents)
                                overlap_start = chunk_start
                            else:
                                overlap_start = max(1, chunk_start - temp_overlap - 1)
                                chunk_end = min(chunk_start + temp_tile_len - (chunk_start - overlap_start), v_latents)
                                
                            if chunk_end <= chunk_start: chunk_end = chunk_start + 1
                                
                            v_tile = final_video_samples[:, :, overlap_start:chunk_end]
                            v_tile_decoded = video_vae.decode(v_tile).cpu()
                            current_valid_frames = v_tile_decoded.shape[1]
                            
                            if chunk_start == 0:
                                decoded_video[:, :current_valid_frames] = v_tile_decoded
                                valid_end_frame = current_valid_frames
                            else:
                                global_start_frame = overlap_start * 8
                                overlap_frames = valid_end_frame - global_start_frame
                                
                                if overlap_frames > 0:
                                    feather = torch.linspace(0.0, 1.0, overlap_frames).view(1, -1, 1, 1, 1)
                                    prev_overlap = decoded_video[:, global_start_frame : valid_end_frame]
                                    new_overlap = v_tile_decoded[:, :overlap_frames]
                                    blended = prev_overlap * (1.0 - feather) + new_overlap * feather
                                    decoded_video[:, global_start_frame : valid_end_frame] = blended
                                    
                                    if current_valid_frames > overlap_frames:
                                        decoded_video[:, valid_end_frame : global_start_frame + current_valid_frames] = v_tile_decoded[:, overlap_frames:]
                                        valid_end_frame = global_start_frame + current_valid_frames
                                else:
                                    decoded_video[:, valid_end_frame : valid_end_frame + current_valid_frames] = v_tile_decoded
                                    valid_end_frame += current_valid_frames
                                    
                            chunk_start = chunk_end
                    else:
                        print(f"-> Executing Pure Isolated Array VAE Decode ({total_splits} chunks - CPU Offloaded)...")
                        latents_per_split = final_video_samples.shape[2] // total_splits
                        frames_per_split_out = (length_in_seconds * current_fps) / total_splits
                        decoded_shots = []
                        for s in range(total_splits):
                            split_latent = final_video_samples[:, :, s * latents_per_split : (s + 1) * latents_per_split]
                            decoded_shot = video_vae.decode(split_latent).cpu()
                            
                            target_shot_frames = int(round((s + 1) * frames_per_split_out)) - int(round(s * frames_per_split_out))
                            if s == 0 and out_ref_frame_count > 0:
                                target_shot_frames += out_ref_frame_count
                                
                            if decoded_shot.shape[1] > target_shot_frames:
                                # DIRECTOR MODE CUT: Enforce strict, absolute boundary slicing
                                if s > 0 and not bypass_img_ref:
                                    # Shots 2+: Slice EXACTLY the setup frames off the front
                                    start_idx = out_ref_frame_count
                                    end_idx = min(start_idx + target_shot_frames, decoded_shot.shape[1])
                                    decoded_shot = decoded_shot[:, start_idx:end_idx]
                                    print(f"   -> Trimmed exactly {out_ref_frame_count} static setup frames from the FRONT of Shot {s+1}")
                                else:
                                    # Shot 1: Keep the setup frames and trim any VAE block-padding from the tail
                                    decoded_shot = decoded_shot[:, :target_shot_frames]
                                    
                            elif decoded_shot.shape[1] < target_shot_frames:
                                pad_len = target_shot_frames - decoded_shot.shape[1]
                                decoded_shot = torch.cat([decoded_shot, decoded_shot[:, -1:].repeat(1, pad_len, 1, 1, 1)], dim=1)
                                
                            decoded_shots.append(decoded_shot)
                        decoded_video = torch.cat(decoded_shots, dim=1)
                else:
                    decoded_video = video_vae.decode(final_video_samples).cpu()
                    
                decoded_video = decoded_video.view(v_batch * decoded_video.shape[1], decoded_video.shape[2], decoded_video.shape[3], 3)

                if restore_faces and face_helper is not None and loaded_facerestore_model is not None:
                    print(f"-> Restoring faces in video (Optimized In-Place Memory Processing)...")
                    for f in range(decoded_video.shape[0]):
                        frame_tensor = decoded_video[f]
                        frame_np = (frame_tensor.numpy() * 255.0).astype(np.uint8)
                        frame_bgr = frame_np[:, :, ::-1]
                        
                        face_helper.clean_all()
                        face_helper.read_image(frame_bgr)
                        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                        face_helper.align_warp_face()
                        
                        for idx, cropped_face in enumerate(face_helper.cropped_faces):
                            cropped_face_t = cropped_face.astype(np.float32) / 255.0
                            cropped_face_t = cv2.cvtColor(cropped_face_t, cv2.COLOR_BGR2RGB)
                            cropped_face_t = torch.from_numpy(cropped_face_t.transpose(2, 0, 1)).float()
                            from torchvision.transforms.functional import normalize
                            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                            cropped_face_t = cropped_face_t.unsqueeze(0).to(fr_device)
                            
                            try:
                                with torch.no_grad():
                                    output = loaded_facerestore_model(cropped_face_t, w=codeformer_fidelity)[0]
                                    output = output.squeeze(0).float().cpu().clamp_(-1, 1)
                                    output = (output + 1) / 2.0
                                    output_np = output.numpy().transpose(1, 2, 0)
                                    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
                                    restored_face_raw = (output_bgr * 255.0).round().astype(np.uint8)
                                    final_restored_face = restored_face_raw
                                    
                                    if face_restore_color_match:
                                        orig_lab = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2LAB).astype(np.float32)
                                        rest_lab = cv2.cvtColor(final_restored_face, cv2.COLOR_BGR2LAB).astype(np.float32)
                                        for c in range(3):
                                            orig_mean, orig_std = orig_lab[:,:,c].mean(), orig_lab[:,:,c].std()
                                            rest_mean, rest_std = rest_lab[:,:,c].mean(), rest_lab[:,:,c].std()
                                            rest_lab[:,:,c] = (rest_lab[:,:,c] - rest_mean) * (orig_std / (rest_std + 1e-6)) + orig_mean
                                        rest_lab = np.clip(rest_lab, 0, 255).astype(np.uint8)
                                        final_restored_face = cv2.cvtColor(rest_lab, cv2.COLOR_LAB2BGR)
                                        
                                    if face_restore_edge_blur:
                                        h, w = final_restored_face.shape[:2]
                                        mask = np.zeros((h, w, 3), dtype=np.float32)
                                        pad = int(h * 0.12)
                                        mask[pad:h-pad, pad:w-pad] = 1.0
                                        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=pad*0.5, sigmaY=pad*0.5)
                                        final_restored_face = (final_restored_face * mask + cropped_face.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
                                face_helper.add_restored_face(final_restored_face)
                            except Exception as error:
                                face_helper.add_restored_face(cropped_face)
                                
                        face_helper.get_inverse_affine(None)
                        pasted_img_bgr = face_helper.paste_faces_to_input_image()
                        
                        if face_restore_blend < 1.0:
                            final_img_bgr = cv2.addWeighted(frame_bgr, 1.0 - face_restore_blend, pasted_img_bgr, face_restore_blend, 0)
                        else:
                            final_img_bgr = pasted_img_bgr
                            
                        restored_img_rgb = final_img_bgr[:, :, ::-1]
                        decoded_video[f] = torch.from_numpy(restored_img_rgb.astype(np.float32) / 255.0)
                        if f > 0 and f % 16 == 0:
                            comfy.model_management.soft_empty_cache()
                
                # --- EXACT AUDIO DECODE & ALIGNMENT ---
                a_latent_samples = final_audio_samples
                if a_latent_samples.is_nested:
                    a_latent_samples = a_latent_samples.unbind()[-1]
                
                sample_rate = int(getattr(audio_vae, "output_sample_rate", audio_vae.first_stage_model.output_sample_rate))
                num_frames = decoded_video.shape[0]
                
                # RESTORED MISSING VARIABLES: These dictate exactly how many setup frames to drop!
                audio_trim_frames = out_ref_frame_count if video_mode != "text-to-video" else 0
                video_trim = out_ref_frame_count if video_mode != "text-to-video" else 0
                
                if video_mode == "reference-to-video":
                    n_ref_images = max(1, ref_frame_count // duplicate_frames) if duplicate_frames > 0 else 1
                    additional_video_trim = 2 * n_ref_images
                    video_trim += additional_video_trim
                    
                time_to_drop = audio_trim_frames / current_fps
                samples_to_drop = int(time_to_drop * sample_rate)
                base_fps = current_fps / 2.0 if temporal_upscale else current_fps

                print("--- Decoding Master Audio Track ---")
                
                if has_audio_input and not temporal_upscale:
                    print("-> Bypassing VAE Decode: Using pristine source waveform directly to prevent truncation...")
                    waveform = torchaudio.functional.resample(master_wf.clone(), sampling_rate, sample_rate).to(device)
                    # For src audio, the length is already perfectly matched to the timeline
                else:
                    print("-> Executing Safe Chunked Audio Decode...")
                    # Restore the chunked decode to prevent VAE sequence limits for generated audio!
                    get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
                    max_audio_lats = get_audio_latents_func(int(20.0 * current_fps), int(current_fps))
                    
                    if a_latent_samples.shape[2] > max_audio_lats:
                        wf_chunks = []
                        for i in range(0, a_latent_samples.shape[2], max_audio_lats):
                            chunk = a_latent_samples[:, :, i:i+max_audio_lats]
                            wf_chunks.append(audio_vae.decode(chunk).to(a_latent_samples.device).movedim(-1, 1))
                        waveform = torch.cat(wf_chunks, dim=-1)
                    else:
                        waveform = audio_vae.decode(a_latent_samples).to(a_latent_samples.device).movedim(-1, 1)

                    # Only apply the VAE warmup trim if we actually decoded through the VAE!
                    exact_target_samples = int((length_in_seconds + time_to_drop) * sample_rate)
                    if waveform.shape[-1] > exact_target_samples:
                        extra_samps = waveform.shape[-1] - exact_target_samples
                        waveform = waveform[..., extra_samps:]
                    elif waveform.shape[-1] < exact_target_samples:
                        pad_len = exact_target_samples - waveform.shape[-1]
                        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
                        
                # Paste the pristine Voice Clone over the VAE setup audio so the original voice is perfectly clear
                if has_audio_ref and not temporal_upscale:
                    pristine_wf = torchaudio.functional.resample(master_wf.clone(), sampling_rate, sample_rate).to(waveform.device)
                    if pristine_wf.shape[1] < waveform.shape[1]:
                        pristine_wf = pristine_wf.repeat(1, waveform.shape[1], 1)
                    elif pristine_wf.shape[1] > waveform.shape[1]:
                        waveform = waveform.repeat(1, pristine_wf.shape[1], 1)
                        
                    region_a_samps_out = int((region_a_frames / base_fps) * sample_rate)
                    limit = min(region_a_samps_out, pristine_wf.shape[-1], waveform.shape[-1])
                    if limit > 0: waveform[..., :limit] = pristine_wf[..., :limit]

                # --- APPLY RAZOR BLADE TO INTERNAL AUDIO ---
                if not has_audio_input and num_prompts > 1:
                    print("-> Executing Isolated Shot Audio Decode for perfect multi-shot sync...")
                    frames_per_split_out = (length_in_seconds * current_fps) / total_splits
                    trimmed_audio_chunks = []
                    
                    shot_duration = length_in_seconds / num_prompts
                    total_a_latents = a_latent_samples.shape[2]
                    
                    for s in range(total_splits):
                        sec_start = s * shot_duration
                        sec_end = (s + 1) * shot_duration
                        
                        _, a_start_lat = get_latent_counts(sec_start)
                        _, a_end_lat = get_latent_counts(sec_end)
                        if s == total_splits - 1:
                            a_end_lat = total_a_latents
                            
                        # 1. Isolate the audio latents for this specific shot
                        s_lat = a_latent_samples[:, :, a_start_lat:a_end_lat]
                        
                        # 2. Decode the isolated latents (prevents VAE convolution shrinkage)
                        chunk_wf = audio_vae.decode(s_lat).to(a_latent_samples.device).movedim(-1, 1)
                        
                        # 3. Determine the exact target frames
                        target_shot_frames = int(round((s + 1) * frames_per_split_out)) - int(round(s * frames_per_split_out))
                        if s == 0 and out_ref_frame_count > 0:
                            target_shot_frames += out_ref_frame_count
                            
                        # 4. Convert target video frames to exact audio samples
                        target_samples = int(target_shot_frames * (sample_rate / base_fps))
                        
                        # 5. Trim the exact audio overhang
                        if chunk_wf.shape[-1] > target_samples:
                            extra_samples = chunk_wf.shape[-1] - target_samples
                            
                            # DIRECTOR MODE CUT: Mirrors the video razor blade perfectly
                            if s > 0 and not bypass_img_ref:
                                # For shots 2+ in ref mode, trim from the FRONT
                                chunk_wf = chunk_wf[..., extra_samples:]
                            else:
                                # For all txt mode, and shot 1 in ref mode, trim from the TAIL
                                chunk_wf = chunk_wf[..., :target_samples]
                                
                        elif chunk_wf.shape[-1] < target_samples:
                            pad_len = target_samples - chunk_wf.shape[-1]
                            chunk_wf = torch.nn.functional.pad(chunk_wf, (0, pad_len))
                            
                        trimmed_audio_chunks.append(chunk_wf)
                        
                    # Reconstruct the mathematically pure timeline
                    waveform = torch.cat(trimmed_audio_chunks, dim=-1)

                # --- 2. VIDEO SLICE ---
                if video_trim >= num_frames:
                    out_image = decoded_video[-1:].cpu()
                elif video_trim > 0:
                    out_image = decoded_video[video_trim:].cpu()
                else:
                    out_image = decoded_video.cpu()

                exact_target_frames = int(length_in_seconds * current_fps)
                if out_image.shape[0] > exact_target_frames:
                    out_image = out_image[:exact_target_frames]
                    print(f"-> Trimmed output video to exactly {exact_target_frames} frames.")

            # ==========================================
            # COLORFX POST-PROCESSING INJECTION
            # ==========================================
            if enable_colorfx and out_image is not None:
                print("--- Applying ColorFX Mastering Pass ---")
                processed_images_tensors = []
                iterable = range(out_image.shape[0])
                if TQDM_AVAILABLE: iterable = tqdm(iterable, desc="ColorFX Processing Frames")
                    
                for i in iterable:
                    pil_image = tensor2pil(out_image[i]).convert("RGB")
                    if enable_color_correction:
                        pil_image = self._apply_shadows_highlights(pil_image, shadow_intensity, highlight_intensity, hdr_intensity)
                        pil_image = self._apply_color_enhancements(pil_image, brightness, contrast, saturation, enhance_color)
                    if enable_lut_processing and lut_name != "None":
                        pil_image = self._apply_lut_effect(pil_image, lut_name, lut_strength, lut_log_process)
                    if enable_enhancements:
                        pil_image = self._apply_sharpness_detail(pil_image, sharpness, edge_enhance_strength, detail_enhance_strength)
                    if enable_blur_effects:
                        pil_image = self._apply_blurs(pil_image, blur_radius, gaussian_blur_radius)
                        if radial_blur_strength > 0.0:
                            pil_image = self._apply_radial_blur(pil_image, radial_blur_strength, radial_blur_center_x, radial_blur_center_y, radial_blur_focus_spread, radial_blur_steps)
                    if enable_stylistic_effects:
                        needs_pil = any(v!=0 for v in [chromatic_aberration_r_x, chromatic_aberration_r_y, chromatic_aberration_b_x, chromatic_aberration_b_y]) or chromatic_blur_amount > 0 or scanline_intensity > 0.0 or soft_light_opacity > 0.0
                        if needs_pil:
                            if any(v!=0 for v in [chromatic_aberration_r_x, chromatic_aberration_r_y, chromatic_aberration_b_x, chromatic_aberration_b_y]) or chromatic_blur_amount > 0:
                                pil_image = self._apply_chromatic_aberration(pil_image, chromatic_aberration_r_x, chromatic_aberration_r_y, chromatic_aberration_b_x, chromatic_aberration_b_y, chromatic_blur_amount)
                            if scanline_intensity > 0.0:
                                pil_image = self._apply_scanlines(pil_image, scanline_intensity)
                            if soft_light_opacity > 0.0:
                                pil_image = self._apply_soft_light(pil_image, soft_light_opacity, soft_light_blur_radius)
                    processed_images_tensors.append(pil2tensor(pil_image))
                    
                out_image = torch.cat(processed_images_tensors, dim=0).to(out_image.device)
                if enable_color_correction and gamma != 1.0:
                    out_image = self._apply_gamma_torch(out_image, gamma)
                if enable_stylistic_effects:
                    if simple_film_grain_intensity > 0.0:
                        out_image = self._apply_simple_film_grain_torch(out_image, simple_film_grain_intensity, simple_film_grain_monochrome)
                    if vignette_intensity > 0.0:
                        out_image = self._apply_vignette_torch(out_image, vignette_intensity, vignette_center_x, vignette_center_y)

            # ==========================================
            # AUDIO MASTERING & MUXING (SKIPPED FOR PASSTHROUGH)
            # ==========================================
            if not pure_image_passthrough:
                final_video_length_seconds = out_image.shape[0] / current_fps
                exact_target_samples = int(final_video_length_seconds * sample_rate)
                
                if out_audio is not None and out_audio["waveform"].shape[-1] > exact_target_samples:
                    out_audio["waveform"] = out_audio["waveform"][..., :exact_target_samples]
                    
                print("-> Running Final Audio Mastering Pass (Soft Limiter & Normalization)...")
                threshold = 0.90
                abs_wf = torch.abs(waveform)
                clip_mask = abs_wf > threshold
                if clip_mask.any():
                    waveform[clip_mask] = torch.sign(waveform[clip_mask]) * (threshold + 0.1 * torch.tanh((abs_wf[clip_mask] - threshold) / 0.1))
                    
                max_val = torch.max(torch.abs(waveform))
                if max_val > 0.0:
                    gain_boost = 0.95 / max_val
                    gain_boost = torch.clamp(gain_boost, max=3.0)
                    waveform = waveform * gain_boost
                    
                waveform = torch.clamp(waveform, -1.0, 1.0)
                total_samples = waveform.shape[-1]
                
                if samples_to_drop >= total_samples:
                    empty_wf = torch.zeros((1, 2, 1), device=waveform.device, dtype=torch.float32)
                    out_audio = {"waveform": empty_wf.cpu(), "sample_rate": sample_rate}
                elif samples_to_drop > 0:
                    sliced_wf = waveform[..., samples_to_drop:]
                    fade_samples = min(int(0.03 * sample_rate), sliced_wf.shape[-1])
                    if fade_samples > 0:
                        fade_tensor = torch.linspace(0.0, 1.0, fade_samples, device=sliced_wf.device, dtype=sliced_wf.dtype)
                        sliced_wf[..., :fade_samples] *= fade_tensor
                    
                    final_wf = sliced_wf
                    if final_wf.dim() == 2: final_wf = final_wf.unsqueeze(0)
                    if final_wf.shape[1] == 1: final_wf = final_wf.repeat(1, 2, 1)
                    elif final_wf.shape[1] > 2: final_wf = final_wf[:, :2, :]
                    
                    final_wf = final_wf.to(torch.float32).clamp(-1.0, 1.0).cpu() 
                    out_audio = {"waveform": final_wf, "sample_rate": sample_rate}
                else:
                    final_wf = waveform
                    if final_wf.dim() == 2: final_wf = final_wf.unsqueeze(0)
                    if final_wf.shape[1] == 1: final_wf = final_wf.repeat(1, 2, 1)
                    elif final_wf.shape[1] > 2: final_wf = final_wf[:, :2, :]
                    
                    final_wf = final_wf.to(torch.float32).clamp(-1.0, 1.0).cpu()
                    out_audio = {"waveform": final_wf, "sample_rate": sample_rate}

            if InputImpl is not None and Types is not None and out_image is not None:
                try:
                    out_video = InputImpl.VideoFromComponents(Types.VideoComponents(images=out_image, audio=out_audio, frame_rate=Fraction(current_fps)))
                except Exception as e:
                    print(f"LTXV Custom Warning: Failed to assemble VIDEO type: {e}")
                
            print("--- Decoding & Slicing Complete ---")

        # ==========================================
        # 9. VRAM CLEANUP
        # ==========================================
        def safe_clear_model(m):
            if m is not None:
                try:
                    m.unpatch_model()
                    
                    # NEW: Deep cleanse the 10S hooks to prevent VRAM memory leaks!
                    for block in m.model.diffusion_model.transformer_blocks:
                        if hasattr(block, "attn1") and hasattr(block.attn1, "_10s_trixope_anchor_handle"):
                            block.attn1._10s_trixope_anchor_handle.remove()
                            delattr(block.attn1, "_10s_trixope_anchor_handle")
                except:
                    pass
                
                if hasattr(m, "object_patches") and m.object_patches is not None: 
                    m.object_patches.clear()
                    
                if hasattr(m, "model_options") and m.model_options is not None:
                    # POLITE CLEANUP: Remove the circular SageAttention override, but leave the dictionary intact
                    # so ComfyUI's garbage collector doesn't throw a KeyError on __del__!
                    if "transformer_options" in m.model_options:
                        m.model_options["transformer_options"].pop("optimized_attention_override", None)
                    
                if hasattr(m, "callbacks") and m.callbacks is not None:
                    try:
                        m.callbacks.clear()
                    except:
                        m.callbacks = []

        # 1. Nuke the circular dictionaries on the wrappers
        safe_clear_model(model_to_use)
        
        if model2_to_use is not None and model2_to_use is not model_to_use:
            safe_clear_model(model2_to_use)
            
        if model3_to_use is not None and model3_to_use is not model_to_use and model3_to_use is not model2_to_use:
            safe_clear_model(model3_to_use)

        # 2. Explictly sever the CFG Guiders (which hold model references)
        try: del primary_guider
        except: pass
        try: del upsample_guider
        except: pass
        try: del temporal_guider
        except: pass

        # 3. Delete all local variable pointers
        del model1_primary
        del model_to_use
        del model2_spatial
        del model2_to_use
        del model3_temporal
        del model3_to_use
        del spatial_upscaler
        del temporal_upscaler
        del video_vae
        del audio_vae
        
        try:
            if restore_faces and loaded_facerestore_model is not None:
                del loaded_facerestore_model
                del face_helper
        except:
            pass
            
        try:
            del diffusion_model
        except:
            pass

        # ==========================================
        # 10. DEEP CACHE CLEANSE (API CALL)
        # ==========================================
        if clear_models_and_cache:
            try:
                address = f"{PromptServer.instance.address}:{PromptServer.instance.port}"
                requests.post(
                    f"http://{address.replace('0.0.0.0','127.0.0.1')}/api/free",
                headers={'Content-Type': 'application/json'},
                json={"unload_models": True, "free_memory": True},
                timeout=10
                )
                print("--- Deep Cache & Models Cleared Successfully ---")
            except Exception as e:
                print(f"--- Deep Cache Clearance Failed: {str(e)} ---")
        else:
            print("--- Deep Cache Clearance Bypassed ---")

        return (final_prompt_string_out, final_positive, final_negative, final_latent, video_out_latent, audio_out_latent, out_video, out_image, out_audio, float(current_fps), out_ref_frame_count, node_id)

# ==========================================
# QUEUE INTERCEPTOR ("CONTROL BEFORE GENERATE")
# ==========================================
def trixope_on_prompt_handler(json_data):
    try:
        prompt = json_data.get("prompt", {})
        seed_map = {}
        for k, v in prompt.items():
            if v.get("class_type") == "FilmAuteur_LTX":
                inputs = v.get("inputs", {})
                
                # Now the backend ACTUALLY receives your command!
                control = inputs.get("control_before_generate", "randomize")

                if control == "randomize":
                    new_seed = random.randint(0, 0xffffffffffffffff)
                    inputs["seed_number"] = new_seed
                    seed_map[k] = new_seed
                elif control == "increment":
                    new_seed = inputs.get("seed_number", 0) + 1
                    inputs["seed_number"] = new_seed
                    seed_map[k] = new_seed
                elif control == "decrement":
                    new_seed = max(0, inputs.get("seed_number", 0) - 1)
                    inputs["seed_number"] = new_seed
                    seed_map[k] = new_seed
                # If control == "fixed", we do absolutely nothing! The seed stays pristine.

        if seed_map:
            PromptServer.instance.send_sync("trixope-global-seed", {"seed_map": seed_map})
            
    except Exception as e:
        print(f"LTXV Custom Error in prompt interceptor: {e}")
        
    return json_data

PromptServer.instance.add_on_prompt_handler(trixope_on_prompt_handler)

class LTXVPostSliceAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "drop_first_n_frames": ("INT", {"default": 9, "min": 0, "max": 99999, "step": 1, "tooltip": "Exact number of video frames to drop from the beginning."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01, "tooltip": "Must match your generation FPS to calculate the audio sync."}),
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "slice_av"
    CATEGORY = "LTXV/Custom"

    def slice_av(self, images, drop_first_n_frames, fps, audio=None):
        num_frames = images.shape[0]
        
        if drop_first_n_frames >= num_frames:
            print(f"LTXV Custom Warning: Trying to drop {drop_first_n_frames} frames but only {num_frames} exist. Returning the last frame to prevent a crash.")
            sliced_images = images[-1:] 
        elif drop_first_n_frames > 0:
            sliced_images = images[drop_first_n_frames:]
            print(f"LTXV Custom: Successfully sliced {drop_first_n_frames} video frames.")
        else:
            sliced_images = images

        sliced_audio = None
        
        if audio is not None:
            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate")
            
            if waveform is not None and sample_rate is not None:
                time_to_drop_seconds = drop_first_n_frames / fps
                samples_to_drop = int(time_to_drop_seconds * sample_rate)
                
                total_samples = waveform.shape[-1]
                
                if samples_to_drop >= total_samples:
                    print(f"LTXV Custom Warning: Trying to drop {samples_to_drop} audio samples but only {total_samples} exist.")
                    empty_waveform = torch.zeros((waveform.shape[0], waveform.shape[1], 1), device=waveform.device, dtype=waveform.dtype)
                    sliced_audio = {"waveform": empty_waveform, "sample_rate": sample_rate}
                elif samples_to_drop > 0:
                    sliced_waveform = waveform[..., samples_to_drop:]
                    sliced_audio = {"waveform": sliced_waveform, "sample_rate": sample_rate}
                    print(f"LTXV Custom: Successfully sliced {samples_to_drop} audio samples ({time_to_drop_seconds:.3f} seconds).")
                else:
                    sliced_audio = audio
            else:
                sliced_audio = audio

        return (sliced_images, sliced_audio)
