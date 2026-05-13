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

class FilmAuteur_LTXV:

    @classmethod
    def INPUT_TYPES(cls):
        sampler_names = comfy.samplers.SAMPLER_NAMES
        primary_default = "res_2s" if "res_2s" in sampler_names else ("euler" if "euler" in sampler_names else sampler_names[0])
        upsample_default = "euler_cfg_pp" if "euler_cfg_pp" in sampler_names else ("euler" if "euler" in sampler_names else sampler_names[0])

        return {
            "required": {
                # --- Input/Setup ---
                "clip": ("CLIP",),
                "video_vae": ("VAE", {"tooltip": "The LTXV Video VAE model."}),
                "audio_vae": ("VAE", {"tooltip": "The LTXV Audio VAE model."}),
                "model1_primary": ("MODEL", {"tooltip": "The primary LTXV Model (will be patched if ID-LoRA is active)."}),

                # --- GROUP: Mode Select ---
                "grp_mode": (["▼ Mode Select"], {}),
                "video_mode": (["text", "image", "reference"], {"default": "text"}),
                "image_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "img_compression": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "audio_select": (["internal", "source", "reference"], {"default": "internal"}),
                "identity_guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),

                # --- GROUP: Prompting ---
                "grp_prompting": (["▼ Prompting"], {}),
                "character_descriptions": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "location_description": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "scene_descriptions": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "use_ollama": ("BOOLEAN", {"default": False}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ollama_model": (OLLAMA_MODELS,),

                # --- GROUP: Specs ---
                "grp_specs": (["▼ Video Specs"], {}),
                "seed_number": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "control_before_generate": (["randomize", "fixed", "increment", "decrement"], {"default": "randomize"}),
                "target_width": ("INT", {"default": 1792, "min": 64, "max": 8192, "step": 32}),
                "target_height": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32}),
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
                "spatial_upscale": ("BOOLEAN", {"default": True}),
                "spatial_passes": ("INT", {"default": 2, "min": 1, "max": 2}),
                "spatial_sampler": (sampler_names, {"default": upsample_default}),
                "spatial_cfg": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "spatial_sigmas": ("STRING", {"multiline": False, "default": "0.55, 0.35, 0.15, 0.0"}),
                "temporal_upscale": ("BOOLEAN", {"default": True}),
                "temporal_denoise": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 1.0, "step": 0.01}),
                "restore_faces": ("BOOLEAN", {"default": True}),
                "facerestore_model": (get_trixope_facerestore_models(), {}),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"], {}),
                "codeformer_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "face_restore_color_match": ("BOOLEAN", {"default": True}),
                "face_restore_edge_blur": ("BOOLEAN", {"default": True}),
                "face_restore_blend": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),

                # --- GROUP: Performance ---
                "grp_performance": (["▼ Performance"], {}),
                "enable_fp16_accumulation": ("BOOLEAN", {"default": True}),
                "sage_attention": (sageattn_modes, {"default": "disabled"}),
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
                "images": ("IMAGE", {"tooltip": "Input image batch for image-to-video or reference-to-video."}),
                "audio": ("AUDIO", {"tooltip": "Connect audio here to encode it directly or use it as a voice reference for ID-LoRA."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("STRING", "LATENT", "LATENT", "LATENT", "VIDEO", "IMAGE", "AUDIO", "FLOAT", "INT", "ETA_TRACKER")
    RETURN_NAMES = ("text_prompt(s)", "av_latent", "video_latent", "audio_latent", "video", "images", "audio", "fps", "ref_frame_count", "real_time_eta")
    FUNCTION = "process"
    CATEGORY = "triXope"

    def process(self, clip, video_vae, audio_vae, model1_primary, character_descriptions, location_description, scene_descriptions,
                video_mode, image_strength, img_compression, audio_select, identity_guidance_scale,
                use_ollama, ollama_url, ollama_model,
                seed_number, control_before_generate, target_width, target_height, length_in_seconds, frame_rate,
                primary_sampler_name, primary_cfg, primary_steps,
                eta, bongmath, enable_nag,
                spatial_upscale, spatial_passes, spatial_sampler, spatial_cfg, spatial_sigmas, temporal_upscale, temporal_denoise,
                restore_faces, facerestore_model, facedetection, codeformer_fidelity,
                face_restore_color_match, face_restore_edge_blur, face_restore_blend,
                enable_fp16_accumulation, sage_attention, autoregressive_chunking, chunk_size_seconds, context_window_seconds, chunks_feedforward, clear_models_and_cache, enable_preview,
                model2_spatial=None, spatial_upscaler=None, model3_temporal=None, temporal_upscaler=None,
                images=None, audio=None, unique_id=None, **kwargs):

        unique_id_val = unique_id[0] if isinstance(unique_id, list) else unique_id

        # --- MAP TO INTERNAL VARIABLES TO PRESERVE ALL MATH ---
        bypass_img_ref = (video_mode != "reference")
        bypass_first_frame = (video_mode != "image")
        
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

        raw_prompts = [p.strip() for p in scene_descriptions.split("|") if p.strip()]
        num_prompts = len(raw_prompts)
        if num_prompts == 0:
            raw_prompts = [""]
            num_prompts = 1

        override_char_desc = (not bypass_img_ref) and (image_ref is not None) and use_ollama
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

        if use_ollama:
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

                user_message = {
                    "role": "user",
                    "content": f"This is Shot {i+1} of a multi-shot sequence. Base prompt: {p}\n\nAnalyze the provided reference image grid and generate the final LTX-Video prompt for this specific shot."
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
        for i, p in enumerate(prompt_list):
            modified_prompt = f"{hidden_prefix}{p}{hidden_suffix}"
            final_prompt_strings.append(modified_prompt)
            tokens = clip.tokenize(modified_prompt)
            out = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = out.pop("cond")
            c_dict = out.copy() if isinstance(out, dict) else {}
            c_dict["start_percent"] = 0.0 
            c_dict["end_percent"] = 1.0
            c_dict["frame_rate"] = current_fps 
            final_positive.append([cond, c_dict])

        final_prompt_string_out = " | ".join(final_prompt_strings)

        tokens_neg = clip.tokenize(negative_prompt)
        out_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True, return_dict=True)
        cond_neg = out_neg.pop("cond")
        dict_neg = out_neg.copy() if isinstance(out_neg, dict) else {}
        dict_neg["start_percent"] = 0.000
        dict_neg["end_percent"] = 1.000
        dict_neg["frame_rate"] = current_fps 
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

        video_samples = torch.zeros([batch_size, 128, latent_count, initial_height // 32, initial_width // 32], device=device)
        video_samples = comfy.sample.fix_empty_latent_channels(model1_primary, video_samples, None)
        video_noise_mask = torch.ones((batch_size, video_samples.shape[2], video_samples.shape[3], video_samples.shape[4]), dtype=torch.float32, device=device)

        z_channels = getattr(audio_vae, "latent_channels", audio_vae.first_stage_model.latent_channels)
        audio_freq = getattr(audio_vae, "latent_frequency_bins", audio_vae.first_stage_model.latent_frequency_bins)
        sampling_rate = int(getattr(audio_vae, "sample_rate", audio_vae.first_stage_model.sample_rate))

        get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
        if get_audio_latents_func is None:
            raise AttributeError("Audio VAE is missing 'num_of_latents_from_frames' method.")
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
        use_silence_len = min(num_audio_latents, true_silence_latent.shape[2])
        audio_samples[:, :, :use_silence_len, :] = true_silence_latent[:, :, :use_silence_len, :]
        audio_noise_mask = torch.ones_like(audio_samples)

        # ==========================================
        # 3.4 HELPER: LATENT COUNTER
        # ==========================================
        def get_latent_counts(sec):
            # 0-Index the absolute start
            if sec <= 0.0:
                return 0, 0
                
            # Map continuous seconds to our isolated shot blocks!
            shot_idx = int(sec // shot_seconds)
            if shot_idx >= num_prompts:
                shot_idx = num_prompts - 1
                
            remainder_sec = sec - (shot_idx * shot_seconds)
            
            # If the remainder is functionally zero, we are exactly on a shot boundary!
            if remainder_sec <= 0.001:
                v_lat = shot_idx * latents_per_shot
            else:
                if bypass_img_ref:
                    raw_frames = int((remainder_sec * current_fps) + 1)
                else:
                    raw_frames = int((a * duplicate_frames) + (remainder_sec * current_fps) + 9)
                    
                local_v_lat = int(((raw_frames - 1) // 8) + 1)
                if local_v_lat < 1: local_v_lat = 1
                
                v_lat = (shot_idx * latents_per_shot) + local_v_lat
            
            # Map audio perfectly to the 0-indexed video latents
            synced_frames = int(((v_lat - 1) * 8) + 1) if v_lat > 0 else 1
            get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
            a_lat = get_audio_latents_func(synced_frames, int(current_fps))
            
            if v_lat == 0:
                a_lat = 0
                
            return v_lat, a_lat

        if bypass_img_ref:
            out_ref_frame_count = 1
        else:
            n = ref_frame_count // duplicate_frames
            out_ref_frame_count = (8 * n) + 8 + 1

        # ==========================================
        # 3.5 WAVEFORM MIXING (NATIVE COMFYUI ASSEMBLY)
        # ==========================================
        setup_frames = 0 if bypass_img_ref else (ref_frame_count + duplicate_frames)

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
        setup_total_samples = int(((region_a_frames + region_b_frames) / current_fps) * sampling_rate)

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
            frames_to_inject = min(encoded_t.shape[2], video_samples.shape[2])
            video_samples[:, :, :frames_to_inject] = encoded_t[:, :, :frames_to_inject]

            for i in range(frames_to_inject):
                pixel_idx = min(i * time_scale_factor, max(0, len(strengths) - 1))
                video_noise_mask[:, i, :, :] = 1.0 - strengths[pixel_idx]

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
        diffusion_model = model_to_use.get_model_object("diffusion_model")
        
        def patch_enable_fp16_accum(model):
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
        def patch_disable_fp16_accum(model):
            torch.backends.cuda.matmul.allow_fp16_accumulation = False

        if enable_fp16_accumulation:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                model_to_use.add_callback(comfy.patcher_extension.CallbacksMP.ON_PRE_RUN, patch_enable_fp16_accum)
                model_to_use.add_callback(comfy.patcher_extension.CallbacksMP.ON_CLEANUP, patch_disable_fp16_accum)
        else:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                model_to_use.add_callback(comfy.patcher_extension.CallbacksMP.ON_PRE_RUN, patch_disable_fp16_accum)

        if sage_attention != "disabled":
            model_to_use.model_options["transformer_options"] = model_to_use.model_options.get("transformer_options", {}).copy()
            
            new_attention = get_sage_func(sage_attention)
            def attention_override_sage(func, *args, **kwargs):
                return new_attention.__wrapped__(*args, **kwargs)
            model_to_use.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage

        dim_threshold = 4096
        if chunks_feedforward > 1:
            for idx, block in enumerate(diffusion_model.transformer_blocks):
                patched_ffn = LTXVffnChunkPatch(chunks_feedforward, dim_threshold).__get__(block.ff, block.__class__)
                model_to_use.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.ff.forward", patched_ffn)
        
        # --- HARDCODED NORMALIZED ATTENTION GUIDANCE (NAG) ---
        if enable_nag:
            print("\n--- Injecting Normalized Attention Guidance (NAG) ---")
            nag_scale = 11.0
            nag_alpha = 0.25
            nag_tau = 2.5
            inplace = True
            
            dtype = model_to_use.model.manual_cast_dtype
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
                model_to_use.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.attn2.forward", patched_attn2)

            if getattr(diffusion_model, "audio_caption_projection", None) is not None:
                context_audio = final_negative[0][0].to(compute_device, dtype)
                if vid_split is not None and context_audio.shape[-1] == vid_split + getattr(diffusion_model, "audio_cross_attention_dim", 0):
                    context_audio = context_audio[:, :, vid_split:]
                
                if getattr(diffusion_model, "caption_proj_before_connector", False) and getattr(diffusion_model, "caption_projection_first_linear", False):
                    diffusion_model.audio_caption_projection.to(compute_device)
                    context_audio = diffusion_model.audio_caption_projection(context_audio)
                    diffusion_model.audio_caption_projection.to(offload_device)
                    
                if hasattr(diffusion_model, "audio_embeddings_connector"):
                    diffusion_model.audio_embeddings_connector.to(compute_device)
                    context_audio = diffusion_model.audio_embeddings_connector(context_audio)[0]
                    diffusion_model.audio_embeddings_connector.to(offload_device)
                    
                context_audio = context_audio.view(1, -1, getattr(diffusion_model, "audio_inner_dim", 128))
                for idx, block in enumerate(diffusion_model.transformer_blocks):
                    if hasattr(block, "audio_attn2"):
                        patched_audio_attn2 = LTXVCrossAttentionPatch(context_audio, nag_scale, nag_alpha, nag_tau, inplace=inplace).__get__(block.audio_attn2, block.__class__)
                        model_to_use.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.audio_attn2.forward", patched_audio_attn2)

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

            wf_lora_mapped = wf_lora.movedim(1, -1).to(vae_device)
            try:
                audio_latents_lora = audio_vae.encode(wf_lora_mapped).to(device)
            except AttributeError:
                audio_latents_lora = audio_vae.first_stage_model.encode(wf_lora_mapped).to(device)
            
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
        print(f"\n--- Base Generation Active: Building {num_prompts} Isolated Shot(s) ---")
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
            
            # 1. Create a pristine, empty latent block for this specific shot
            pass_v_samples = torch.zeros([batch_size, 128, v_len, initial_height // 32, initial_width // 32], device=device)
            pass_v_samples = comfy.sample.fix_empty_latent_channels(model1_primary, pass_v_samples, None)
            pass_v_masks = torch.ones_like(pass_v_samples)
            
            pass_a_samples = torch.zeros([batch_size, z_channels, a_len, audio_freq], device=device)
            pass_a_masks = torch.ones_like(pass_a_samples)
            
            # 2. Inject Reference Images uniquely into this shot's slot
            if s == 0 and final_pixels is not None:
                encoded_t = video_vae.encode(final_pixels[:, :, :, :3])
                frames_to_inject = min(encoded_t.shape[2], pass_v_samples.shape[2])
                pass_v_samples[:, :, :frames_to_inject] = encoded_t[:, :, :frames_to_inject]
                for i in range(frames_to_inject):
                    pixel_idx = min(i * time_scale_factor, max(0, len(strengths) - 1))
                    pass_v_masks[:, i, :, :] = 1.0 - strengths[pixel_idx]
            elif first_frame is not None and not bypass_first_frame:
                img_idx = min(s, first_frame.shape[0] - 1)
                img = first_frame[img_idx:img_idx+1]
                img_scaled = comfy.utils.common_upscale(img.movedim(-1, 1), t_width, t_height, "bilinear", "center").movedim(1, -1)
                encoded_img = video_vae.encode(img_scaled[:, :, :, :3]).to(device)
                inject_len = min(encoded_img.shape[2], pass_v_samples.shape[2])
                if inject_len > 0:
                    pass_v_samples[:, :, :inject_len] = encoded_img[:, :, :inject_len]
                    for j in range(inject_len):
                        pass_v_masks[:, j, :, :] = 1.0 - first_frame_str
                        
            # 3. Slice the correct audio chunk for this shot
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
                    print(f"   -> [Shot {s+1}/{num_prompts} - Chunk {chunk_idx}/{total_chunks_in_shot}] 0.0s to {curr_global_sec - shot_start_sec:.2f}s (Base Block)")
                else:
                    context_start_sec = max(shot_start_sec, chunk_start_sec - context_window_seconds)
                    ctx_v_global, ctx_a_global = get_latent_counts(context_start_sec)
                    ctx_v_lat = ctx_v_global - base_v_lat
                    ctx_a_lat = ctx_a_global - base_a_lat
                    print(f"   -> [Shot {s+1}/{num_prompts} - Chunk {chunk_idx}/{total_chunks_in_shot}] {chunk_start_sec - shot_start_sec:.2f}s to {curr_global_sec - shot_start_sec:.2f}s (Context Lookback: {chunk_start_sec - context_start_sec:.2f}s)")
                    
                # Slice the temporary chunk from the isolated shot tensor
                chunk_v_samples = pass_v_samples[:, :, ctx_v_lat:curr_v_lat]
                chunk_a_samples = pass_a_samples[:, :, ctx_a_lat:curr_a_lat]
                chunk_v_masks = pass_v_masks[:, :, ctx_v_lat:curr_v_lat].clone()
                chunk_a_masks = pass_a_masks[:, :, ctx_a_lat:curr_a_lat].clone()
                
                # Lock the context latents so they act purely as un-denoised reference guides!
                if chunk_start_sec > shot_start_sec:
                    chunk_v_masks[:, :, : (prev_v_lat - ctx_v_lat)] = 0.0
                    chunk_a_masks[:, :, : (prev_a_lat - ctx_a_lat)] = 0.0
                    
                av_samples = comfy.nested_tensor.NestedTensor((chunk_v_samples, chunk_a_samples))
                av_masks = comfy.nested_tensor.NestedTensor((chunk_v_masks, chunk_a_masks))
                current_latent = {"samples": av_samples, "noise_mask": av_masks, "sample_rate": sampling_rate, "type": "audio"}
                
                primary_guider.set_conds([final_positive[s]], final_negative)
                base_cb = latent_preview.prepare_callback(primary_guider.model_patcher, primary_sigmas.shape[-1] - 1, x0_output)
                
                # Update the callback string so the ComfyUI progress bar reflects the exact chunk!
                callback = wrap_callback(base_cb, f"Shot {s+1}/{num_prompts} (Chunk {chunk_idx}/{total_chunks_in_shot})")
                
                sampled_chunk = primary_guider.sample(noise_obj.generate_noise(current_latent), current_latent["samples"], primary_sampler, primary_sigmas, denoise_mask=av_masks, callback=callback, disable_pbar=disable_pbar, seed=seed_number)
                
                res_v, res_a = sampled_chunk.unbind()
                res_v = res_v.to(device)
                res_a = res_a.to(device)
                
                # Inject the generated chunk back into the isolated shot array
                if chunk_start_sec == shot_start_sec:
                    pass_v_samples[:, :, prev_v_lat:curr_v_lat] = res_v
                    pass_a_samples[:, :, prev_a_lat:curr_a_lat] = res_a
                else:
                    gen_v_len = curr_v_lat - prev_v_lat
                    gen_a_len = curr_a_lat - prev_a_lat
                    pass_v_samples[:, :, prev_v_lat:curr_v_lat] = res_v[:, :, -gen_v_len:]
                    
                    # --- AUDIO CROSS-DISSOLVE (CHUNKS) ---
                    # ~2.5 frames worth of audio latents
                    a_fade = 6 
                    if gen_a_len > a_fade and prev_a_lat >= a_fade:
                        # Hard paste the bulk of the new audio chunk
                        pass_a_samples[:, :, prev_a_lat:curr_a_lat] = res_a[:, :, -gen_a_len:]
                        
                        # Feather the seam using the generated context!
                        prev_audio = pass_a_samples[:, :, prev_a_lat - a_fade : prev_a_lat].clone()
                        context_audio = res_a[:, :, -gen_a_len - a_fade : -gen_a_len]
                        
                        fade_in = torch.linspace(0.0, 1.0, a_fade, device=device).view(1, 1, -1, 1)
                        pass_a_samples[:, :, prev_a_lat - a_fade : prev_a_lat] = prev_audio * (1.0 - fade_in) + context_audio * fade_in
                    else:
                        pass_a_samples[:, :, prev_a_lat:curr_a_lat] = res_a[:, :, -gen_a_len:]
                
                chunk_idx += 1
                    
            # End of Autoregressive Loop for Shot S
            isolated_v_shots.append(pass_v_samples)
            isolated_a_shots.append(pass_a_samples)
            
        # 5. Concatenate the pristine, 0-indexed isolated shots end-to-end to form the timeline
        global_v_samples = torch.cat(isolated_v_shots, dim=2)
        global_a_samples = torch.cat(isolated_a_shots, dim=2)
        
        # --- AUDIO V-FADE (MULTI-SHOT BOUNDARIES) ---
        if num_prompts > 1:
            print("-> Applying Audio Latent V-Fade at Shot Boundaries...")
            a_fade = 6  # ~2.5 frames of audio latents
            current_a_idx = 0
            
            # Prepare the true silence pad
            silence_pad = true_silence_latent[:, :, :a_fade].clone()
            if silence_pad.shape[2] < a_fade:
                pad_len = a_fade - silence_pad.shape[2]
                silence_pad = torch.nn.functional.pad(silence_pad, (0, 0, 0, pad_len))
                
            for s in range(num_prompts - 1):
                current_a_idx += isolated_a_shots[s].shape[2]
                b_idx = current_a_idx
                
                if b_idx >= a_fade and b_idx + a_fade <= global_a_samples.shape[2]:
                    # Extract the hard-cut audio edges
                    left_audio = global_a_samples[:, :, b_idx - a_fade : b_idx].clone()
                    right_audio = global_a_samples[:, :, b_idx : b_idx + a_fade].clone()
                    
                    # Fade Shot 1 out to silence
                    fade_out = torch.linspace(1.0, 0.0, a_fade, device=device).view(1, 1, -1, 1)
                    global_a_samples[:, :, b_idx - a_fade : b_idx] = left_audio * fade_out + silence_pad * (1.0 - fade_out)
                    
                    # Fade Shot 2 in from silence
                    fade_in = torch.linspace(0.0, 1.0, a_fade, device=device).view(1, 1, -1, 1)
                    global_a_samples[:, :, b_idx : b_idx + a_fade] = right_audio * fade_in + silence_pad * (1.0 - fade_in)

        sampled_tensor = comfy.nested_tensor.NestedTensor((global_v_samples, global_a_samples)).to(device)

        # ==========================================
        # MID-GENERATION STAGE 1 PREVIEW (MP4 MUX)
        # ==========================================
        if enable_preview and (spatial_upscale or temporal_upscale):
            print("\n--- Generating Stage 1 Preview ---")
            import uuid
            uid = node_id if node_id is not None else str(uuid.uuid4())
            
            v_samps_prev, a_samps_prev = sampled_tensor.unbind()
            v_samps_prev = v_samps_prev.to(device)
            a_samps_prev = a_samps_prev.to(device)

            if num_prompts > 1:
                print("-> Executing Pure Isolated Array VAE Decode for pristine preview cuts...")
                latents_per_shot = v_samps_prev.shape[2] // num_prompts
                frames_per_shot_out = (length_in_seconds * current_fps) / num_prompts
                
                decoded_shots = []
                for s in range(num_prompts):
                    shot_latent = v_samps_prev[:, :, s * latents_per_shot : (s + 1) * latents_per_shot]
                    decoded_shot = video_vae.decode(shot_latent)
                    
                    target_shot_frames = int(round((s + 1) * frames_per_shot_out)) - int(round(s * frames_per_shot_out))
                    if s == 0 and out_ref_frame_count > 0:
                        target_shot_frames += out_ref_frame_count
                        
                    if decoded_shot.shape[1] > target_shot_frames:
                        decoded_shot = decoded_shot[:, :target_shot_frames]
                    elif decoded_shot.shape[1] < target_shot_frames:
                        pad_len = target_shot_frames - decoded_shot.shape[1]
                        decoded_shot = torch.cat([decoded_shot, decoded_shot[:, -1:].repeat(1, pad_len, 1, 1, 1)], dim=1)
                        
                    decoded_shots.append(decoded_shot)
                preview_video = torch.cat(decoded_shots, dim=1)
            else:
                preview_video = video_vae.decode(v_samps_prev)

            preview_video = preview_video[0] 
            
            num_preview_frames = preview_video.shape[0]
            if out_ref_frame_count >= num_preview_frames:
                preview_video = preview_video[-1:] 
            elif out_ref_frame_count > 0:
                preview_video = preview_video[out_ref_frame_count:]
                
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
                                           temporal_positive=None, temporal_negative=None, video_mode="text"):

            v_batch, v_channels, v_frames, v_height, v_width = v_samps.shape
            temp_tile_len = 48
            temp_overlap = 8
            
            sim_start = 0
            num_chunks = 0
            while sim_start < v_frames:
                num_chunks += 1
                if sim_start == 0:
                    sim_start = min(sim_start + temp_tile_len, v_frames)
                else:
                    sim_overlap = max(1, sim_start - temp_overlap - 1)
                    sim_start = min(sim_start + temp_tile_len - (sim_start - sim_overlap), v_frames)
            
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
        if spatial_upscale:
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
        unbound_samples = sampled_tensor.unbind()
        final_video_samples = unbound_samples[0].to(device)
        final_audio_samples = unbound_samples[1].clone().to(device)

        if has_audio_ref or has_audio_input:
            max_latents = final_audio_samples.shape[2]
            master_max = master_latents.shape[2]
            
            if has_audio_ref:
                lock_a = min(region_a_latents, max_latents, master_max)
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
        if temporal_upscale:
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

            # =========================================================
            # 0. THE "LOB IT OFF" OPTIMIZATION (REFERENCE MODE ONLY)
            # =========================================================
            temp_video_mode = video_mode
            if video_mode == "reference":
                print("-> LOBBING OFF static setup latents to optimize Temporal Upscale...")
                v_setup_base = max(0, get_latent_counts(0.0)[0] - 1) if not bypass_img_ref else 0
                
                get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
                pixel_start_base = v_setup_base * 8
                a_setup_base = get_audio_latents_func(pixel_start_base + 1, int(current_fps))
                
                # Permanently slice the setup latents off the tensors for the upscale pass!
                if final_video_samples.shape[2] > v_setup_base:
                    final_video_samples = final_video_samples[:, :, v_setup_base:]
                if final_audio_samples.shape[2] > a_setup_base:
                    final_audio_samples = final_audio_samples[:, :, a_setup_base:]
                    
                # Trick the rest of the temporal pass into acting exactly like Text-to-Video!
                temp_video_mode = "text"
                
                # Tell the final audio/video muxer there are no setup frames left to drop!
                out_ref_frame_count = 0 

            # ---------------------------------------------------------
            # 1. GLOBAL PREP: AUDIO ALIGNMENT & STRETCH
            # ---------------------------------------------------------
            print(f"-> Preparing perfectly aligned audio stretch (Isolated Mode: {temp_video_mode})...")
            import torchaudio.functional as TA_F
            
            # Because we lobbed off the reference latents, ALL modes now run the proven pristine logic!
            if audio is not None and audio_select == "source":
                print("-> Using pristine source audio for Temporal Stretch...")
                input_audio = audio
            else:
                print("-> Decoding Base Generation audio for Temporal Stretch (Internal Mode)...")
                curr_audio_wf = audio_vae.decode(final_audio_samples).to("cpu").movedim(-1, 1)
                input_audio = {"waveform": curr_audio_wf, "sample_rate": sampling_rate}

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
                
            shifted_audio = {"waveform": stretched_wf.cpu(), "sample_rate": source_sr}
            
            audio_latent_out = run_node("LTXVAudioVAEEncode", audio=shifted_audio, audio_vae=audio_vae)[0]
            stretched_audio_latents = audio_latent_out["samples"]

            # FORMAT CONDITIONING
            print("-> Formatting Conditioning Metadata...")
            import node_helpers
            
            cond_out_pos = node_helpers.conditioning_set_values(final_positive, {"frame_rate": float(current_fps)})
            cond_out_neg = node_helpers.conditioning_set_values(final_negative, {"frame_rate": float(current_fps)})
            
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
                if video_mode == "reference":
                    # We lobbed off v_setup_base, so we must subtract it from subsequent shot indices!
                    return max(0, (v_lat - 1) - v_setup_base)
                return max(0, v_lat - 1)

            for s in range(num_prompts):
                v_idx = get_shot_start_latent_idx(s, shot_duration)
                v_lat_up = v_idx * 2
                frames_to_encode = None
                
                # Because temp_video_mode is strictly 'text' or 'image' now, the logic is ultra-clean!
                if temp_video_mode == "text":
                    base_latent = final_video_samples[:, :, v_idx:v_idx+1]
                    if base_latent.shape[2] > 0:
                        frames_to_encode = video_vae.decode(base_latent)
                elif temp_video_mode == "image":
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
            # 3. SLIDING WINDOW SETUP
            # ---------------------------------------------------------
            user_tile_len = max(8, int((chunk_size_seconds * current_fps) / 8))
            temp_tile_len = min(user_tile_len, 16) 
            temp_overlap = max(2, temp_tile_len // 4)
            
            global_v_temporal = torch.zeros((v_batch, v_channels, total_out_frames, v_height, v_width), device=device, dtype=final_video_samples.dtype)

            sim_start = 0
            num_chunks = 0
            while sim_start < v_frames_base:
                num_chunks += 1
                if sim_start == 0:
                    sim_start = min(sim_start + temp_tile_len, v_frames_base)
                else:
                    sim_overlap = max(1, sim_start - temp_overlap - 1)
                    sim_start = min(sim_start + temp_tile_len - (sim_start - sim_overlap), v_frames_base)

            chunk_start = 0
            chunk_idx = 1
            v_time_scale = video_vae.downscale_index_formula[0]
            get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))

            temporal_model = model3_temporal if model3_temporal is not None else model1_primary

            # ---------------------------------------------------------
            # 4. CHUNKING LOOP
            # ---------------------------------------------------------
            while chunk_start < v_frames_base:
                comfy.model_management.soft_empty_cache()

                shot_len_latents = v_frames_base / num_prompts 
                current_shot = min(int(chunk_start // shot_len_latents), num_prompts - 1)
                shot_start = int(current_shot * shot_len_latents)
                shot_end = int((current_shot + 1) * shot_len_latents) if current_shot < num_prompts - 1 else v_frames_base
                
                if chunk_start == shot_start:
                    chunk_end = min(chunk_start + temp_tile_len, shot_end)
                    overlap_start = chunk_start
                else:
                    overlap_start = max(shot_start + 1, chunk_start - temp_overlap - 1)
                    chunk_end = min(chunk_start + temp_tile_len - (chunk_start - overlap_start), shot_end)
                    
                if chunk_end <= chunk_start:
                    chunk_end = chunk_start + 1

                # Slice Video
                v_tile = final_video_samples[:, :, overlap_start:chunk_end]

                # Map & Slice Audio to Stretched Timeline
                tile_in_frames = chunk_end - overlap_start
                tile_out_frames = (tile_in_frames * 2) - 1
                up_overlap_start = overlap_start * 2
                
                pixel_start = up_overlap_start * v_time_scale
                pixel_out_frames = 1 + (tile_out_frames - 1) * v_time_scale
                pixel_end = pixel_start + pixel_out_frames
                
                a_start = get_audio_latents_func(pixel_start, int(current_fps))
                req_a_len = get_audio_latents_func(pixel_out_frames, int(current_fps))
                a_end = min(a_start + req_a_len, stretched_audio_latents.shape[2])
                
                a_tile = stretched_audio_latents[:, :, a_start:a_end]
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
                    
                    feather = torch.linspace(0.0, 1.0, overlap_out_frames, device=device, dtype=torch.float32)
                    v_noise_mask[:, :, :overlap_out_frames] = feather.view(1, -1, 1, 1)
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
                
                print(f"-> Executing Inlined Temporal KSampler Chunk {chunk_idx}/{num_chunks}...")
                chunk_seed = seed_number + chunk_idx

                # 1. Setup the CFG Guider
                temporal_guider = comfy.samplers.CFGGuider(temporal_model)
                temporal_guider.set_cfg(1.0)
                temporal_guider.set_conds(crop_pos, crop_neg)

                # 2. Setup the Sampler & Scheduler
                temporal_sampler = comfy.samplers.sampler_object("euler_cfg_pp")
                temporal_model_sampling = temporal_model.get_model_object("model_sampling")

                # 3. Exact KSampler Math (4 Steps)
                node_steps = 4
                node_denoise = temporal_denoise
                total_schedule_steps = int(node_steps / node_denoise)
                total_sigmas = comfy.samplers.calculate_sigmas(temporal_model_sampling, "linear_quadratic", total_schedule_steps)
                temporal_sigmas_tensor = total_sigmas[-(node_steps + 1):].to(device)

                # 4. Generate Random Noise
                chunk_noise_obj = Noise_RandomNoise(chunk_seed)
                noise = chunk_noise_obj.generate_noise(concat_latent)

                # 5. Execute the Denoise Pass
                sampled_latent_raw = temporal_guider.sample(
                    noise, 
                    concat_latent["samples"], 
                    temporal_sampler, 
                    temporal_sigmas_tensor, 
                    denoise_mask=concat_latent["noise_mask"], 
                    callback=None, 
                    disable_pbar=disable_pbar, 
                    seed=chunk_seed
                )
                # Wrap it back up for the Separate node to consume
                sampled_latent = {"samples": sampled_latent_raw}

                unbound_latents = sampled_latent["samples"].unbind()
                sampled_v_tile = unbound_latents[0].to(device)

                if chunk_start == 0:
                    global_v_temporal[:, :, :sampled_v_tile.shape[2]] = sampled_v_tile
                else:
                    global_v_temporal[:, :, global_start : global_start + sampled_v_tile.shape[2]] = sampled_v_tile

                chunk_start = chunk_end
                chunk_idx += 1

            final_video_samples = global_v_temporal
            current_fps *= 2
            out_ref_frame_count = (out_ref_frame_count * 2) - 1

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

            if num_prompts > 1:
                print("-> Executing Pure Isolated Array VAE Decode (CPU Offloaded)...")
                latents_per_shot = final_video_samples.shape[2] // num_prompts
                frames_per_shot_out = (length_in_seconds * current_fps) / num_prompts
                
                decoded_shots = []
                for s in range(num_prompts):
                    shot_latent = final_video_samples[:, :, s * latents_per_shot : (s + 1) * latents_per_shot]
                    # INSTANT CPU OFFLOAD: Prevents VRAM from exploding on long timelines!
                    decoded_shot = video_vae.decode(shot_latent).cpu()
                    
                    target_shot_frames = int(round((s + 1) * frames_per_shot_out)) - int(round(s * frames_per_shot_out))
                    if s == 0 and out_ref_frame_count > 0:
                        target_shot_frames += out_ref_frame_count
                        
                    if decoded_shot.shape[1] > target_shot_frames:
                        decoded_shot = decoded_shot[:, :target_shot_frames]
                    elif decoded_shot.shape[1] < target_shot_frames:
                        pad_len = target_shot_frames - decoded_shot.shape[1]
                        decoded_shot = torch.cat([decoded_shot, decoded_shot[:, -1:].repeat(1, pad_len, 1, 1, 1)], dim=1)
                        
                    decoded_shots.append(decoded_shot)
                decoded_video = torch.cat(decoded_shots, dim=1)
            else:
                # INSTANT CPU OFFLOAD
                decoded_video = video_vae.decode(final_video_samples).cpu()
                
            # Flatten batch dimension for output compatibility
            decoded_video = decoded_video.view(v_batch * decoded_video.shape[1], decoded_video.shape[2], decoded_video.shape[3], 3)

            if restore_faces and face_helper is not None and loaded_facerestore_model is not None:
                print(f"-> Restoring faces in video (Optimized In-Place Memory Processing)...")
                for f in range(decoded_video.shape[0]):
                    frame_tensor = decoded_video[f]
                    # Already on CPU, no extra VRAM jump required
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
                            print(f'\tFailed inference for CodeFormer on frame {f}: {error}')
                            face_helper.add_restored_face(cropped_face)
                            
                    face_helper.get_inverse_affine(None)
                    pasted_img_bgr = face_helper.paste_faces_to_input_image()
                    
                    if face_restore_blend < 1.0:
                        final_img_bgr = cv2.addWeighted(frame_bgr, 1.0 - face_restore_blend, pasted_img_bgr, face_restore_blend, 0)
                    else:
                        final_img_bgr = pasted_img_bgr
                        
                    restored_img_rgb = final_img_bgr[:, :, ::-1]
                    
                    # IN-PLACE RAM OVERWRITE: Destroys the old un-restored frame to prevent RAM doubling!
                    decoded_video[f] = torch.from_numpy(restored_img_rgb.astype(np.float32) / 255.0)
                    
                    # PERIODIC VRAM FLUSH: Clears the CodeFormer GPU buffers every 16 frames
                    if f > 0 and f % 16 == 0:
                        comfy.model_management.soft_empty_cache()
            
            a_latent_samples = final_audio_samples
            if a_latent_samples.is_nested:
                a_latent_samples = a_latent_samples.unbind()[-1]
            
            sample_rate = int(getattr(audio_vae, "output_sample_rate", audio_vae.first_stage_model.output_sample_rate))
            
            base_fps = current_fps / 2.0 if temporal_upscale else current_fps
            
            # Prevent micro-slice audio drift if we lobbed off the reference frames
            if out_ref_frame_count <= 0:
                base_ref_count = 0.0
            else:
                base_ref_count = (out_ref_frame_count + 1) / 2.0 if temporal_upscale else out_ref_frame_count
                
            time_to_drop = base_ref_count / base_fps
            samples_to_drop = int(time_to_drop * sample_rate)

            print("--- Decoding Master Audio Track ---")
            if num_prompts > 1:
                print("-> Executing Isolated Audio Decode with Long Mirror Crossfade (fps/2)...")
                a_latents_per_shot = a_latent_samples.shape[2] // num_prompts
                
                decoded_a_shots = []
                for s in range(num_prompts):
                    s_lat = a_latent_samples[:, :, s * a_latents_per_shot : (s + 1) * a_latents_per_shot]
                    s_wf = audio_vae.decode(s_lat).to(a_latent_samples.device).movedim(-1, 1)
                    decoded_a_shots.append(s_wf)
                    
                waveform = torch.cat(decoded_a_shots, dim=-1)
                
                # Dynamic FPS/2 frames of audio overlap (~0.5 seconds)
                fade_frames = base_fps / 2.0
                fade_len = int((fade_frames / base_fps) * sample_rate)
                
                # Safety clamp to ensure the fade doesn't consume the entire shot
                shot_len_samples = int((length_in_seconds / num_prompts) * sample_rate)
                fade_len = min(fade_len, max(1, (shot_len_samples // 2) - 1))
                
                current_idx = 0
                for s in range(num_prompts - 1):
                    current_idx += decoded_a_shots[s].shape[-1]
                    b_idx = current_idx
                    
                    if b_idx >= fade_len and b_idx + fade_len <= waveform.shape[-1]:
                        # Extract the hard-cut edges
                        left_audio = waveform[..., b_idx - fade_len : b_idx].clone()
                        right_audio = waveform[..., b_idx : b_idx + fade_len].clone()
                        
                        # Create mirror extensions to maintain timeline length
                        left_extend = torch.cat([left_audio, left_audio.flip(-1)], dim=-1)
                        right_extend = torch.cat([right_audio.flip(-1), right_audio], dim=-1)
                        
                        # True Cross-Dissolve over the mirrored pads
                        fade_in = torch.linspace(0.0, 1.0, fade_len * 2, device=waveform.device, dtype=waveform.dtype)
                        blended = left_extend * (1.0 - fade_in) + right_extend * fade_in
                        
                        # Paste the seamless transition back into the timeline
                        waveform[..., b_idx - fade_len : b_idx + fade_len] = blended
            else:
                waveform = audio_vae.decode(a_latent_samples).to(a_latent_samples.device).movedim(-1, 1)

            # --- THE POST-DECODE MASTERING PASS (ANTI-CLIP & NORMALIZE) ---
            print("-> Running Final Audio Mastering Pass (Soft Limiter & Normalization)...")
            
            # 1. Soft Limiter (Anti-Clipping)
            # Smoothly compresses extreme peaks instead of letting them hit the digital ceiling and pop
            threshold = 0.90
            abs_wf = torch.abs(waveform)
            clip_mask = abs_wf > threshold
            if clip_mask.any():
                waveform[clip_mask] = torch.sign(waveform[clip_mask]) * (threshold + 0.1 * torch.tanh((abs_wf[clip_mask] - threshold) / 0.1))
                
            # 2. Auto-Gain / Normalization (Fixes volume drops)
            # Finds the loudest peak and safely boosts the entire track to a cinematic standard
            max_val = torch.max(torch.abs(waveform))
            target_peak = 0.95
            if max_val > 0.0:
                gain_boost = target_peak / max_val
                # Clamp the boost so we don't accidentally blow out quiet ambient noise into loud static
                gain_boost = min(gain_boost, 3.0)
                waveform = waveform * gain_boost
                
            # 3. Absolute Safety Clamp
            waveform = torch.clamp(waveform, -1.0, 1.0)

            if (has_audio_ref or has_audio_input) and not temporal_upscale:
                pristine_wf = torchaudio.functional.resample(master_wf.clone(), sampling_rate, sample_rate).to(waveform.device)
                
                if pristine_wf.shape[1] < waveform.shape[1]:
                    pristine_wf = pristine_wf.repeat(1, waveform.shape[1], 1)
                elif pristine_wf.shape[1] > waveform.shape[1]:
                    waveform = waveform.repeat(1, pristine_wf.shape[1], 1)
                    
                region_a_samps_out = int((region_a_frames / base_fps) * sample_rate)
                setup_total_samps_out = int(((region_a_frames + region_b_frames) / base_fps) * sample_rate)
                
                if has_audio_ref:
                    limit = min(region_a_samps_out, pristine_wf.shape[-1], waveform.shape[-1])
                    if limit > 0:
                        waveform[..., :limit] = pristine_wf[..., :limit]
                
                if has_audio_input:
                    rem_pristine = pristine_wf.shape[-1] - setup_total_samps_out
                    rem_wave = waveform.shape[-1] - setup_total_samps_out
                    limit = min(rem_pristine, rem_wave)
                    if limit > 0:
                        waveform[..., setup_total_samps_out : setup_total_samps_out + limit] = \
                            pristine_wf[..., setup_total_samps_out : setup_total_samps_out + limit]
                    
                region_a_samps_out = int((region_a_frames / base_fps) * sample_rate)
                setup_total_samps_out = int(((region_a_frames + region_b_frames) / base_fps) * sample_rate)
                
                if has_audio_ref:
                    limit = min(region_a_samps_out, pristine_wf.shape[-1], waveform.shape[-1])
                    if limit > 0:
                        waveform[..., :limit] = pristine_wf[..., :limit]
                
                if has_audio_input:
                    rem_pristine = pristine_wf.shape[-1] - setup_total_samps_out
                    rem_wave = waveform.shape[-1] - setup_total_samps_out
                    limit = min(rem_pristine, rem_wave)
                    if limit > 0:
                        waveform[..., setup_total_samps_out : setup_total_samps_out + limit] = \
                            pristine_wf[..., setup_total_samps_out : setup_total_samps_out + limit]

            num_frames = decoded_video.shape[0]
            
            video_trim = out_ref_frame_count
            if temporal_upscale and video_mode == "reference":
                print("-> Applying 1-frame video shift to perfect Reference lip sync...")
                video_trim += 1
            
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
                
            exact_target_samples = int(length_in_seconds * sample_rate)
            if out_audio is not None and out_audio["waveform"].shape[-1] > exact_target_samples:
                out_audio["waveform"] = out_audio["waveform"][..., :exact_target_samples]
                print(f"-> Trimmed output audio to exactly {exact_target_samples} samples.")

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
                if final_wf.dim() == 2:
                    final_wf = final_wf.unsqueeze(0)
                if final_wf.shape[1] == 1:
                    final_wf = final_wf.repeat(1, 2, 1)
                elif final_wf.shape[1] > 2:
                    final_wf = final_wf[:, :2, :]
                
                final_wf = final_wf.to(torch.float32).clamp(-1.0, 1.0).cpu() 
                out_audio = {"waveform": final_wf, "sample_rate": sample_rate}
            else:
                final_wf = waveform
                if final_wf.dim() == 2:
                    final_wf = final_wf.unsqueeze(0)
                if final_wf.shape[1] == 1:
                    final_wf = final_wf.repeat(1, 2, 1)
                elif final_wf.shape[1] > 2:
                    final_wf = final_wf[:, :2, :]
                
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
                m.unpatch_model()
                # Break circular patches safely without destroying the dictionary structure
                if hasattr(m, "object_patches"): 
                    m.object_patches.clear()
                if hasattr(m, "model_options") and "transformer_options" in m.model_options:
                    # Surgically remove only our injected SageAttention override to prevent memory leaks
                    m.model_options["transformer_options"].pop("optimized_attention_override", None)

        safe_clear_model(model_to_use)
        
        if model2_to_use is not None and model2_to_use is not model_to_use:
            safe_clear_model(model2_to_use)
            
        if model3_to_use is not None and model3_to_use is not model_to_use and model3_to_use is not model2_to_use:
            safe_clear_model(model3_to_use)

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

        return (final_prompt_string_out, final_latent, video_out_latent, audio_out_latent, out_video, out_image, out_audio, float(current_fps), out_ref_frame_count, node_id)

# ==========================================
# QUEUE INTERCEPTOR ("CONTROL BEFORE GENERATE")
# ==========================================
def trixope_on_prompt_handler(json_data):
    try:
        prompt = json_data.get("prompt", {})
        seed_map = {}
        for k, v in prompt.items():
            if v.get("class_type") == "FilmAuteur_LTXV":
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
