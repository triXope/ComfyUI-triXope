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

class FilmAuteur_LTXV:

    @classmethod
    def INPUT_TYPES(cls):
        sampler_names = comfy.samplers.SAMPLER_NAMES
        primary_default = "res_2s" if "res_2s" in sampler_names else ("euler" if "euler" in sampler_names else sampler_names[0])
        upsample_default = "euler_ancestral_cfg_pp" if "euler_ancestral_cfg_pp" in sampler_names else ("euler" if "euler" in sampler_names else sampler_names[0])

        mode_options = [
            "manual", 
            "debug/testing", 
            "text-to-video",
            "text-to-video (+ audio in)", 
            "image-to-video",
            "image-to-video (+ audio in)", 
            "reference-to-video (+ audio ref)", 
            "reference-to-video (+ audio in)"
        ]

        return {
            "required": {
                # Setup
                "clip": ("CLIP",),
                "video_vae": ("VAE", {"tooltip": "The LTXV Video VAE model."}),
                "audio_vae": ("VAE", {"tooltip": "The LTXV Audio VAE model."}),
                "primary_model": ("MODEL", {"tooltip": "The primary LTXV Model (will be patched if ID-LoRA is active)."}),
                
                # Prompts
                "character_descriptions": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "", "tooltip": "Provide a detailed description for each character (overridden by image reference)."}),
                "location_description": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "", "tooltip": "Provide a detailed description of the location(s)."}),
                "scene_descriptions": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "", "tooltip": 'Provide a detailed description for each shot, separated by "|" (eg. shot 1 | shot 2 | shot 3). Note: length_in_seconds must be evenly divisible by total number of shots.'}),
                
                # Mode Select
                "simple_mode_select": (mode_options, {"default": "manual", "tooltip": "Override for simplified mode select."}),
                
                # --- GROUP: Input ---
                "grp_input_controls": (["▼ Input"], {}),
                "bypass_img_ref": ("BOOLEAN", {"default": False}),
                "bypass_first_frame": ("BOOLEAN", {"default": False}),
                "load_audio_from_file": ("BOOLEAN", {"default": False}),
                "bypass_audio_ref": ("BOOLEAN", {"default": False}),
                "image_ref_str": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "first_frame_str": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "identity_guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                
                # --- GROUP: Enhance ---
                "grp_ollama_enhance": (["▼ Enhance"], {}),
                "use_ollama": ("BOOLEAN", {"default": False}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ollama_model": (OLLAMA_MODELS,),
                
                # --- GROUP: Timeline ---
                "grp_timeline_controls": (["▼ Timeline"], {}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "target_width": ("INT", {"default": 1792, "min": 64, "max": 8192, "step": 32}),
                "target_height": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32}),
                "length_in_seconds": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                
                # --- GROUP: Sampling ---
                "grp_sampling": (["▼ Sampling"], {}),
                "sampling_stages": ("INT", {"default": 2, "min": 1, "max": 3}),
                "primary_sampler_name": (sampler_names, {"default": primary_default}),
                "primary_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "primary_steps": ("STRING", {"multiline": False, "default": "1.0, 0.995, 0.99, 0.9875, 0.975, 0.65, 0.28, 0.07, 0.0"}),
                "upsample_sampler_name": (sampler_names, {"default": upsample_default}),
                "upsample_cfg": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "upsample_manual_sigmas": ("STRING", {"multiline": False, "default": "0.85, 0.7250, 0.4219, 0.0"}),
                "eta": ("FLOAT", {"default": 0.95, "min": -100.0, "max": 100.0, "step": 0.01, "round": False}),
                "bongmath": ("BOOLEAN", {"default": True}),
                "autoregressive_chunking": ("BOOLEAN", {"default": True}),
                "chunk_size_seconds": ("FLOAT", {"default": 60.0, "min": 5.0, "max": 300.0, "step": 1.0}),
                "context_window_seconds": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 300.0, "step": 1.0}),
                
                # --- GROUP: Refinement ---
                "grp_refinement": (["▼ Refinement"], {}),
                "temporal_upscale": ("BOOLEAN", {"default": False}),
                "restore_faces": ("BOOLEAN", {"default": False}),
                "facerestore_model": (get_trixope_facerestore_models(), {}),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"], {}),
                "codeformer_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "face_restore_color_match": ("BOOLEAN", {"default": True}),
                "face_restore_edge_blur": ("BOOLEAN", {"default": True}),
                "face_restore_blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                # --- GROUP: VRAM ---
                "grp_vram_optimization": (["▼ VRAM"], {}),
                "enable_fp16_accumulation": ("BOOLEAN", {"default": False}),
                "sage_attention": (sageattn_modes, {"default": "disabled"}),
                "chunks": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "model2_opt": ("MODEL", {"tooltip": "Optional model for upsample stages 2 and 3. If disconnected, the main model is used.", "forceInput": True}),
                "spatial_upscaler": ("LATENT_UPSCALE_MODEL", {"tooltip": "Connect the LTXV Spatial Upscale model here to upsample the video latent by 2x.", "forceInput": True}),
                "temporal_upscaler": ("LATENT_UPSCALE_MODEL", {"tooltip": "Connect the LTXV Temporal Upscale model here to double the framerate.", "forceInput": True}),
                "audio_input": ("AUDIO", {"tooltip": "Connect audio here to encode it directly into the latent."}),
                "audio_ref": ("AUDIO", {"tooltip": "Voice reference for ID-LoRA (active if load_audio_from_file is False)."}),
                "image_ref": ("IMAGE", {"tooltip": "Batch of concept images to condition the video globally."}),
                "first_frame(s)": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "LATENT", "LATENT", "LATENT", "VIDEO", "IMAGE", "AUDIO", "FLOAT", "INT")
    RETURN_NAMES = ("text_prompt(s)", "av_latent", "video_latent", "audio_latent", "video", "images", "audio", "fps", "ref_frame_count")
    FUNCTION = "process"
    CATEGORY = "triXope"

    def process(self, clip, video_vae, audio_vae, primary_model, character_descriptions, location_description, scene_descriptions, 
                simple_mode_select, bypass_img_ref, bypass_first_frame, load_audio_from_file, bypass_audio_ref,
                image_ref_str, first_frame_str, identity_guidance_scale,
                use_ollama, ollama_url, ollama_model,
                noise_seed, target_width, target_height, length_in_seconds, frame_rate, 
                sampling_stages, primary_sampler_name, primary_cfg, primary_steps, 
                upsample_sampler_name, upsample_cfg, upsample_manual_sigmas, eta, bongmath,
                autoregressive_chunking, chunk_size_seconds, context_window_seconds, temporal_upscale, 
                restore_faces, facerestore_model, facedetection, codeformer_fidelity, 
                face_restore_color_match, face_restore_edge_blur, face_restore_blend,
                enable_fp16_accumulation, sage_attention, chunks,
                model2_opt=None, spatial_upscaler=None, temporal_upscaler=None, 
                audio_input=None, audio_ref=None, image_ref=None, **kwargs):

        # ==========================================
        # 0. MODE OVERRIDES
        # ==========================================
        if simple_mode_select == "debug/testing":
            sampling_stages = 1
            target_width = max(64, target_width // 4)
            target_height = max(64, target_height // 4)
            length_in_seconds = 5.0
            primary_sampler_name = "euler"
            temporal_upscale = False
            restore_faces = False
            use_ollama = False
        elif simple_mode_select == "text-to-video":
            bypass_img_ref = True
            bypass_first_frame = True
            load_audio_from_file = False
            bypass_audio_ref = True
        elif simple_mode_select == "text-to-video (+ audio in)":
            bypass_img_ref = True
            bypass_first_frame = True
            load_audio_from_file = True
            bypass_audio_ref = True
        elif simple_mode_select == "image-to-video":
            bypass_img_ref = True
            bypass_first_frame = False
            load_audio_from_file = False
            bypass_audio_ref = True
        elif simple_mode_select == "image-to-video (+ audio in)":
            bypass_img_ref = True
            bypass_first_frame = False
            load_audio_from_file = True
            bypass_audio_ref = True
        elif simple_mode_select == "reference-to-video (+ audio ref)":
            bypass_img_ref = False
            bypass_first_frame = True
            load_audio_from_file = False
            bypass_audio_ref = False
        elif simple_mode_select == "reference-to-video (+ audio in)":
            bypass_img_ref = False
            bypass_first_frame = True
            load_audio_from_file = True
            bypass_audio_ref = True

        first_frame = kwargs.get("first_frame(s)")

        current_fps = frame_rate
        decode = True # Hardcoded to True internally 
        
        # Audio Safeguard Variables
        has_audio_ref = not bypass_audio_ref and audio_ref is not None
        has_audio_input = load_audio_from_file and audio_input is not None
        
        # Hardcoded Negative Prompt (Hidden from user)
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

        divisor = 2 ** (sampling_stages - 1)
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

        # ==========================================
        # 1.1 CHARACTER & LOCATION DESCRIPTION INJECTION
        # ==========================================
        override_char_desc = (not bypass_img_ref) and (image_ref is not None)
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

        # ==========================================
        # 1.5 DIRECTOR MODE OVERRIDE
        # ==========================================
        if num_prompts > 1:
            autoregressive_chunking = True # Force chunking if multi-shot is detected
            chunk_size_seconds = length_in_seconds / num_prompts
            print(f"\n--- Multi-Shot Director Mode Active: Timeline synced to {num_prompts} shots ({chunk_size_seconds:.2f}s per shot). ---")

        # ==========================================
        # 0.5 OLLAMA API PROMPT REVAMP (MULTI-SHOT)
        # ==========================================
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
                    # Dynamically map the image in the batch to the shot!
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
            # CRITICAL FIX: Lock prompt to the entire active denoising step schedule
            # This isolates the prompt strictly to its mathematical chunk!
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
            image_ref_processed = process_ref_image(image_ref)
            a += image_ref_processed.shape[0]
            for idx in range(image_ref_processed.shape[0]):
                img = image_ref_processed[idx:idx+1]
                pixel_frames.append(img.repeat(duplicate_frames, 1, 1, 1))
                strengths.extend([image_ref_str] * duplicate_frames)

        if not bypass_first_frame and first_frame is not None:
            first_frame_processed = process_ref_image(first_frame)
            # For setup, only grab the FIRST image of the batch to start the sequence
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
        b = length_in_seconds
        
        if not bypass_img_ref:
            frame_length = int((a * duplicate_frames) + (b * current_fps) + 9)
        else:
            frame_length = int((b * current_fps) + 1)
            
        device = comfy.model_management.intermediate_device()
        batch_size = 1 

        video_samples = torch.zeros([batch_size, 128, ((frame_length - 1) // 8) + 1, initial_height // 32, initial_width // 32], device=device)
        
        # PRISTINE FIX: Apply the structural latent bias ONLY ONCE at the very beginning to prevent ghosting!
        video_samples = comfy.sample.fix_empty_latent_channels(primary_model, video_samples, None)
        
        # Use exact [B, F, H, W] 4D mask sizing to completely bypass ComfyUI's 5D interpolation crash
        video_noise_mask = torch.ones((batch_size, video_samples.shape[2], video_samples.shape[3], video_samples.shape[4]), dtype=torch.float32, device=device)

        z_channels = audio_vae.latent_channels
        audio_freq = audio_vae.latent_frequency_bins
        sampling_rate = int(audio_vae.sample_rate)
        
        num_audio_latents = audio_vae.num_of_latents_from_frames(frame_length, int(current_fps))
        
        total_silence_samples = int(((frame_length / current_fps) + 5.0) * sampling_rate)
        silent_wf = torch.zeros((batch_size, 1, total_silence_samples), dtype=torch.float32, device=device)
        silent_dict = {"waveform": silent_wf, "sample_rate": sampling_rate}
        true_silence_latent = audio_vae.encode(silent_dict).to(device)

        audio_samples = torch.zeros((batch_size, z_channels, num_audio_latents, audio_freq), device=device)
        use_silence_len = min(num_audio_latents, true_silence_latent.shape[2])
        audio_samples[:, :, :use_silence_len, :] = true_silence_latent[:, :, :use_silence_len, :]
        audio_noise_mask = torch.ones_like(audio_samples)

        # ==========================================
        # 3.4 HELPER: LATENT COUNTER
        # ==========================================
        def get_latent_counts(sec):
            if not bypass_img_ref:
                frames = int((a * duplicate_frames) + (sec * current_fps) + 9)
            else:
                frames = int((sec * current_fps) + 1)
            v_lat = ((frames - 1) // 8) + 1
            a_lat = audio_vae.num_of_latents_from_frames(frames, int(current_fps))
            return v_lat, a_lat

        if bypass_img_ref:
            out_ref_frame_count = 1
        else:
            n = ref_frame_count // duplicate_frames
            out_ref_frame_count = (8 * n) + 8 + 1

        # ==========================================
        # 3.5 WAVEFORM MIXING (FLAWLESS ASSEMBLY)
        # ==========================================
        if bypass_img_ref:
            region_a_frames = 1
            region_b_frames = 0
        else:
            region_a_frames = ref_frame_count 
            region_b_frames = duplicate_frames 
            
        region_a_latents = audio_vae.num_of_latents_from_frames(region_a_frames, int(current_fps))
        setup_total_latents = audio_vae.num_of_latents_from_frames(region_a_frames + region_b_frames, int(current_fps))
        
        total_samples = int((frame_length / current_fps) * sampling_rate)
        region_a_samples = int((region_a_frames / current_fps) * sampling_rate)
        setup_total_samples = int(((region_a_frames + region_b_frames) / current_fps) * sampling_rate)

        def extract_and_resample(audio_data, target_sr):
            if isinstance(audio_data, dict):
                wf = audio_data["waveform"]
                orig_sr = audio_data.get("sample_rate", target_sr)
            else:
                wf = audio_data
                orig_sr = target_sr
            if orig_sr != target_sr:
                wf = torchaudio.functional.resample(wf, orig_sr, target_sr)
            return wf

        target_channels = 1
        ref_wf = None
        inp_wf = None
        
        if has_audio_ref:
            ref_wf = extract_and_resample(audio_ref, sampling_rate).to(device)
            target_channels = max(target_channels, ref_wf.shape[1])
            
        if has_audio_input:
            inp_wf = extract_and_resample(audio_input, sampling_rate).to(device)
            target_channels = max(target_channels, inp_wf.shape[1])
            
        master_wf = torch.zeros((batch_size, target_channels, total_samples), dtype=torch.float32, device=device)
        
        if ref_wf is not None:
            if ref_wf.shape[0] != batch_size:
                ref_wf = ref_wf.repeat(batch_size, 1, 1)[:batch_size]
            c = ref_wf.shape[1]
            use_samps = min(ref_wf.shape[2], region_a_samples, total_samples)
            if use_samps > 0:
                master_wf[:, :c, :use_samps] = ref_wf[:, :, :use_samps]
                
        if inp_wf is not None:
            if inp_wf.shape[0] != batch_size:
                inp_wf = inp_wf.repeat(batch_size, 1, 1)[:batch_size]
            c = inp_wf.shape[1]
            remaining_samples = total_samples - setup_total_samples
            if remaining_samples > 0:
                use_samps = min(inp_wf.shape[2], remaining_samples)
                if use_samps > 0:
                    master_wf[:, :c, setup_total_samples:setup_total_samples+use_samps] = inp_wf[:, :, :use_samps]
                    
        master_dict = {"waveform": master_wf, "sample_rate": sampling_rate}
        master_latents = audio_vae.encode(master_dict).to(device)

        use_len = min(num_audio_latents, master_latents.shape[2])
        if use_len > 0:
            audio_samples[:, :, :use_len, :] = master_latents[:, :, :use_len, :]
            
        # Audio Safeguard: Only lock setup frames if an explicit audio ref was provided
        if has_audio_ref:
            lock_a = min(region_a_latents, use_len)
            if lock_a > 0:
                audio_noise_mask[:, :, :lock_a, :] = 0.0
            
        # Audio Safeguard: Only lock input track if explicitly provided
        if has_audio_input:
            start_c = min(setup_total_latents, use_len)
            if start_c < use_len:
                audio_noise_mask[:, :, start_c:use_len, :] = 0.0

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
        # 5. ALL UNET VRAM PATCHING (FP16, Sage, Chunks, LoRA)
        # ==========================================
        model_to_use = primary_model.clone()
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
            new_attention = get_sage_func(sage_attention)
            def attention_override_sage(func, *args, **kwargs):
                return new_attention.__wrapped__(*args, **kwargs)
            model_to_use.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage

        dim_threshold = 4096
        if chunks > 1:
            for idx, block in enumerate(diffusion_model.transformer_blocks):
                patched_ffn = LTXVffnChunkPatch(chunks, dim_threshold).__get__(block.ff, block.__class__)
                model_to_use.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.ff.forward", patched_ffn)
        
        model2_to_use = model2_opt.clone() if model2_opt is not None else model_to_use.clone()
        
        if not bypass_audio_ref and audio_ref is not None:
            audio_wf_lora = extract_and_resample(audio_ref, sampling_rate).to(device)
            audio_dict_lora = {"waveform": audio_wf_lora, "sample_rate": sampling_rate}
            audio_latents_lora = audio_vae.encode(audio_dict_lora)
            
            b_, c_, t_, f_ = audio_latents_lora.shape
            ref_tokens = audio_latents_lora.permute(0, 2, 1, 3).reshape(b_, t_, c_ * f_)
            ref_audio_dict = {"tokens": ref_tokens}

            for i in range(len(final_positive)):
                final_positive[i][1]["ref_audio"] = ref_audio_dict
            for i in range(len(final_negative)):
                final_negative[i][1]["ref_audio"] = ref_audio_dict

            scale = identity_guidance_scale

            def get_post_cfg_function(target_model):
                model_sampling = target_model.get_model_object("model_sampling")
                sigma_start = model_sampling.percent_to_sigma(0.0)
                sigma_end = model_sampling.percent_to_sigma(1.0)
                audio_channels = audio_vae.latent_channels
                
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

            model_to_use.set_model_sampler_post_cfg_function(get_post_cfg_function(model_to_use))
            if model2_opt is not None:
                model2_to_use.set_model_sampler_post_cfg_function(get_post_cfg_function(model2_to_use))

        # ==========================================
        # 6. SAMPLING & UPSCALING LOOP
        # ==========================================
        noise_obj = Noise_RandomNoise(noise_seed)
        
        primary_guider = comfy.samplers.CFGGuider(model_to_use)
        primary_guider.set_cfg(primary_cfg)

        upsample_guider = comfy.samplers.CFGGuider(model2_to_use)
        upsample_guider.set_cfg(upsample_cfg)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        primary_sampler = build_custom_sampler(primary_sampler_name, eta, bongmath)
        
        # --- HYBRID SIGMA/STEP PARSER ---
        # Checks if user entered a single int for steps, or a comma-separated list for manual sigmas
        if ',' in str(primary_steps):
            sigmas_list = re.findall(r"[-+]?(?:\d*\.*\d+)", str(primary_steps))
            primary_sigmas = torch.FloatTensor([float(i) for i in sigmas_list])
        else:
            try:
                p_steps = int(primary_steps)
            except ValueError:
                p_steps = 8
                
            tokens = math.prod(video_samples.shape[2:])
            sigmas = torch.linspace(1.0, 0.0, p_steps + 1)
            max_shift, base_shift, terminal = 2.05, 0.95, 0.1
            x1, x2 = 1024, 4096
            mm_shift = (max_shift - base_shift) / (x2 - x1)
            b_shift = base_shift - mm_shift * x1
            sigma_shift = (tokens) * mm_shift + b_shift

            power = 1
            sigmas = torch.where(sigmas != 0, math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power), 0)
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor_math = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor_math)
            sigmas[non_zero_mask] = stretched
            primary_sigmas = sigmas


        # THE DIRECTOR'S TIMELINE ENGINE
        if not autoregressive_chunking or length_in_seconds <= chunk_size_seconds:
            primary_guider.set_conds([final_positive[0]], final_negative)
            
            av_samples = comfy.nested_tensor.NestedTensor((video_samples, audio_samples))
            av_noise_mask = comfy.nested_tensor.NestedTensor((video_noise_mask.unsqueeze(1), audio_noise_mask))
            current_latent = {"samples": av_samples, "noise_mask": av_noise_mask, "sample_rate": sampling_rate, "type": "audio"}
            latent_image = current_latent["samples"]
            
            x0_output = {}
            callback = latent_preview.prepare_callback(primary_guider.model_patcher, primary_sigmas.shape[-1] - 1, x0_output)

            print(f"\n--- Running Autoregressive Pass 1/1 (Dimensions: {initial_width}x{initial_height}) ---")
            sampled_tensor = primary_guider.sample(noise_obj.generate_noise(current_latent), latent_image, primary_sampler, primary_sigmas, denoise_mask=av_noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)
            sampled_tensor = sampled_tensor.to(device)
            
        else:
            global_v_samples = video_samples
            global_a_samples = audio_samples
            global_v_masks = video_noise_mask.unsqueeze(1) 
            global_a_masks = audio_noise_mask
            
            shot_duration = length_in_seconds / num_prompts
            curr_sec = 0.0
            
            x0_output = {}
            for s in range(num_prompts):
                shot_start_sec = s * shot_duration
                shot_end_sec = (s + 1) * shot_duration
                
                while curr_sec < shot_end_sec:
                    comfy.model_management.soft_empty_cache()
                    chunk_start_sec = curr_sec
                    curr_sec = min(chunk_start_sec + chunk_size_seconds, shot_end_sec)
                    
                    curr_v_latents, curr_a_latents = get_latent_counts(curr_sec)
                    prev_v_latents, prev_a_latents = get_latent_counts(chunk_start_sec)
                    
                    # ISOLATE PROMPT TO CURRENT SHOT!
                    primary_guider.set_conds([final_positive[s]], final_negative)
                    
                    if chunk_start_sec == shot_start_sec:
                        # NEW SHOT: TRIGGER HARD CUT! Set Lookback Context to 0.
                        context_start_sec = chunk_start_sec
                        print(f"\n--- Director Mode: Action! Generating Shot {s+1}/{num_prompts} (0.0s to {curr_sec - chunk_start_sec:.2f}s) ---")
                    else:
                        # CONTINUE SHOT: Apply bounded lookback context.
                        context_start_sec = max(shot_start_sec, chunk_start_sec - context_window_seconds)
                        print(f"\n--- Extending Shot {s+1}: Generating {chunk_start_sec - shot_start_sec:.2f}s to {curr_sec - shot_start_sec:.2f}s (Context Lookback: {chunk_start_sec - context_start_sec:.2f}s) ---")
                        
                    ctx_v_latents, ctx_a_latents = get_latent_counts(context_start_sec)
                    
                    if chunk_start_sec > shot_start_sec:
                        global_v_masks[:, :, ctx_v_latents:prev_v_latents] = 0.0
                        global_a_masks[:, :, ctx_a_latents:prev_a_latents] = 0.0
                    
                    # Slice the exact Window from the global master tensors
                    pass_v_samples = global_v_samples[:, :, ctx_v_latents:curr_v_latents]
                    pass_a_samples = global_a_samples[:, :, ctx_a_latents:curr_a_latents]
                    pass_v_masks = global_v_masks[:, :, ctx_v_latents:curr_v_latents]
                    pass_a_masks = global_a_masks[:, :, ctx_a_latents:curr_a_latents]
                    
                    av_samples = comfy.nested_tensor.NestedTensor((pass_v_samples, pass_a_samples))
                    av_masks = comfy.nested_tensor.NestedTensor((pass_v_masks, pass_a_masks))
                    
                    current_latent = {"samples": av_samples, "noise_mask": av_masks, "sample_rate": sampling_rate, "type": "audio"}
                    latent_image = current_latent["samples"]
                        
                    callback = latent_preview.prepare_callback(primary_guider.model_patcher, primary_sigmas.shape[-1] - 1, x0_output)
                    sampled_chunk = primary_guider.sample(noise_obj.generate_noise(current_latent), latent_image, primary_sampler, primary_sigmas, denoise_mask=av_masks, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)
                    
                    unbound = sampled_chunk.unbind()
                    res_v = unbound[0].to(device)
                    res_a = unbound[1].to(device)
                    
                    if context_start_sec == chunk_start_sec:
                        # Hard Cut Replacement
                        global_v_samples[:, :, prev_v_latents:curr_v_latents] = res_v
                        global_a_samples[:, :, prev_a_latents:curr_a_latents] = res_a
                    else:
                        # Extension Replacement
                        gen_v_len = curr_v_latents - prev_v_latents
                        gen_a_len = curr_a_latents - prev_a_latents
                        global_v_samples[:, :, prev_v_latents:curr_v_latents] = res_v[:, :, -gen_v_len:]
                        global_a_samples[:, :, prev_a_latents:curr_a_latents] = res_a[:, :, -gen_a_len:]
                
            sampled_tensor = comfy.nested_tensor.NestedTensor((global_v_samples, global_a_samples)).to(device)


        # ==========================================
        # UNIVERSAL SLIDING WINDOW UPSCALER ENGINE
        # ==========================================
        def process_sliding_window_upscale(pass_name, is_temporal, v_samps, a_samps,
                                           upscaler_model, guider, sampler, sigmas,
                                           noise_obj, eta, bongmath, final_pixels, strengths,
                                           time_scale_factor, width_scale_factor, height_scale_factor, video_vae, current_fps, disable_pbar, noise_seed):
            
            v_batch, v_channels, v_frames, v_height, v_width = v_samps.shape
            temp_tile_len = 48
            temp_overlap = 8
            
            # Pre-calculate total chunks for console UI
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
            
            # VRAM SAVER: Lock audio completely so the UNet just passes it through untouched during upscales!
            a_mask_locked = torch.zeros_like(a_samps)
            
            while chunk_start < v_frames:
                comfy.model_management.soft_empty_cache()
                
                if chunk_start == 0:
                    chunk_end = min(chunk_start + temp_tile_len, v_frames)
                    overlap_start = chunk_start
                else:
                    overlap_start = max(1, chunk_start - temp_overlap - 1)
                    chunk_end = min(chunk_start + temp_tile_len - (chunk_start - overlap_start), v_frames)
                    
                v_tile = v_samps[:, :, overlap_start:chunk_end]
                
                # UPSCALER PROMPT SYNC: Determine which shot the playhead is currently upscaling!
                center_frame = overlap_start + (chunk_end - overlap_start) / 2.0
                frames_per_shot = v_frames / num_prompts
                shot_idx = int(center_frame / frames_per_shot)
                shot_idx = min(max(shot_idx, 0), num_prompts - 1)
                guider.set_conds([final_positive[shot_idx]], final_negative)
                
                # Audio slice for context only
                pixel_start = overlap_start * time_scale_factor
                pixel_end = 1 + (chunk_end - 1) * time_scale_factor
                a_start = audio_vae.num_of_latents_from_frames(pixel_start, int(current_fps))
                a_end = audio_vae.num_of_latents_from_frames(pixel_end, int(current_fps))
                a_tile = a_samps[:, :, a_start:a_end]
                
                # 1. Neural Net Upscale
                device_up = comfy.model_management.get_torch_device()
                model_dtype = next(upscaler_model.parameters()).dtype
                input_dtype = v_tile.dtype
                
                upscaler_model.to(device_up)
                v_tile_up = video_vae.first_stage_model.per_channel_statistics.un_normalize(v_tile.to(dtype=model_dtype, device=device_up))
                v_tile_up = upscaler_model(v_tile_up)
                upscaler_model.cpu()
                v_tile_up = video_vae.first_stage_model.per_channel_statistics.normalize(v_tile_up).to(dtype=input_dtype, device=device)
                
                print(f"\n--- {pass_name} Chunk {chunk_idx}/{num_chunks} (Dimensions: {v_tile_up.shape[4]*32}x{v_tile_up.shape[3]*32} | Shot {shot_idx+1}) ---")
                
                # Dynamically generate a perfectly sized 3D mask for this exact chunk! (B, F, H, W)
                v_mask_tile = torch.ones((v_batch, v_tile_up.shape[2], v_tile_up.shape[3], v_tile_up.shape[4]), device=device, dtype=torch.float32)
                
                if final_pixels is not None and sp_encoded_t_local is None and not is_temporal:
                    t_width_sp = v_tile_up.shape[4] * width_scale_factor
                    t_height_sp = v_tile_up.shape[3] * height_scale_factor
                    sp_final_pixels = final_pixels.clone()
                    if sp_final_pixels.shape[1] != t_height_sp or sp_final_pixels.shape[2] != t_width_sp:
                        sp_final_pixels = comfy.utils.common_upscale(sp_final_pixels.movedim(-1, 1), t_width_sp, t_height_sp, "bilinear", "center").movedim(1, -1)
                    
                    encoded_frames = video_vae.encode(sp_final_pixels[:, :, :, :3])
                    if encoded_frames.ndim == 4:
                        encoded_frames = encoded_frames.unsqueeze(0)
                    sp_encoded_t_local = encoded_frames.to(device)
                
                if sp_encoded_t_local is not None and not is_temporal:
                    # Base Setup Injection inside sliding window
                    inject_start = max(0, overlap_start)
                    inject_end = min(chunk_end, sp_encoded_t_local.shape[2])
                    if inject_start < inject_end:
                        tile_inj_start = inject_start - overlap_start
                        tile_inj_end = inject_end - overlap_start
                        v_tile_up[:, :, tile_inj_start:tile_inj_end] = sp_encoded_t_local[:, :, inject_start:inject_end]
                        for i in range(inject_start, inject_end):
                            pixel_idx = min(i * time_scale_factor, max(0, len(strengths) - 1))
                            v_mask_tile[:, i - overlap_start, :, :] = 1.0 - strengths[pixel_idx]

                # MULTI-SHOT DIRECTOR CUT: Re-Inject High-Res Reference Frames into Upscaler!
                if not is_temporal and not bypass_first_frame and first_frame is not None and num_prompts > 1 and autoregressive_chunking:
                    shot_duration = length_in_seconds / num_prompts
                    for i in range(1, num_prompts):
                        sec = i * shot_duration
                        v_lat, _ = get_latent_counts(sec)
                        
                        if overlap_start <= v_lat < chunk_end:
                            img_idx = min(i, first_frame.shape[0] - 1)
                            img = first_frame[img_idx:img_idx+1]
                            
                            t_width_sp = v_tile_up.shape[4] * width_scale_factor
                            t_height_sp = v_tile_up.shape[3] * height_scale_factor
                            img_scaled = comfy.utils.common_upscale(img.movedim(-1, 1), t_width_sp, t_height_sp, "bilinear", "center").movedim(1, -1)
                            
                            encoded_img = video_vae.encode(img_scaled[:, :, :, :3])
                            if encoded_img.ndim == 4:
                                encoded_img = encoded_img.unsqueeze(0)
                            encoded_img = encoded_img.to(device)
                            
                            tile_inj_start = v_lat - overlap_start
                            tile_inj_end = min(tile_inj_start + encoded_img.shape[2], v_tile_up.shape[2])
                            
                            if tile_inj_start < tile_inj_end:
                                v_tile_up[:, :, tile_inj_start:tile_inj_end] = encoded_img[:, :, :tile_inj_end - tile_inj_start]
                                for j in range(tile_inj_start, tile_inj_end):
                                    v_mask_tile[:, j, :, :] = 1.0 - first_frame_str

                if global_v_samps_up is None:
                    if is_temporal:
                        total_out_frames = (v_frames * 2) - 1
                        global_v_samps_up = torch.empty((v_batch, v_tile_up.shape[1], total_out_frames, v_tile_up.shape[3], v_tile_up.shape[4]), device=device, dtype=input_dtype)
                    else:
                        global_v_samps_up = torch.empty((v_batch, v_tile_up.shape[1], v_frames, v_tile_up.shape[3], v_tile_up.shape[4]), device=device, dtype=input_dtype)
                        
                # VRAM SAVER & SEAMLESS STITCHING: Feathered Mask Locking!
                if chunk_start > 0:
                    overlap_in_frames = chunk_start - overlap_start
                    if is_temporal:
                        overlap_out_frames = overlap_in_frames * 2 - 1
                        global_start = overlap_start * 2
                    else:
                        overlap_out_frames = overlap_in_frames
                        global_start = overlap_start
                        
                    # Pre-load the upscaler input with perfectly diffused latents from previous chunk
                    v_tile_up[:, :, :overlap_out_frames] = global_v_samps_up[:, :, global_start : global_start + overlap_out_frames]
                    
                    # Feather the mask from 0.0 (locked) to 1.0 (fully diffuse) to seamlessly outpaint
                    feather = torch.linspace(0.0, 1.0, overlap_out_frames, device=device, dtype=torch.float32)
                    v_mask_tile[:, :overlap_out_frames, :, :] = feather.view(1, -1, 1, 1)

                # 2. Diffusion Sample
                am_tile = a_mask_locked[:, :, a_start:a_end]
                
                # Wrap the 4D video mask in a 5D tuple exactly like ComfyUI expects
                current_latent_tile = {
                    "samples": comfy.nested_tensor.NestedTensor((v_tile_up, a_tile)),
                    "noise_mask": comfy.nested_tensor.NestedTensor((v_mask_tile.unsqueeze(1), am_tile)),
                    "sample_rate": sampling_rate,
                    "type": "audio"
                }
                
                latent_image_tile = current_latent_tile["samples"]
                
                x0_output = {}
                callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
                
                sampled_chunk = guider.sample(noise_obj.generate_noise(current_latent_tile), latent_image_tile, sampler, sigmas, denoise_mask=current_latent_tile["noise_mask"], callback=callback, disable_pbar=disable_pbar, seed=noise_seed)
                sampled_v_tile = sampled_chunk.unbind()[0].to(device)

                # 3. Stitch into Global Tensor
                if chunk_start == 0:
                    global_v_samps_up[:, :, :sampled_v_tile.shape[2]] = sampled_v_tile
                else:
                    # Direct overwrite! The feathered mask guarantees frame 0 perfectly matches the global tensor
                    global_v_samps_up[:, :, global_start : global_start + sampled_v_tile.shape[2]] = sampled_v_tile

                chunk_start = chunk_end
                chunk_idx += 1
                
            return global_v_samps_up


        # --- UPSAMPLING PASSES (2 & 3) (SPATIAL) ---
        if sampling_stages > 1:
            if spatial_upscaler is None:
                raise ValueError("Spatial upscaler model is required if sampling_stages > 1.")
            
            upsample_sampler = build_custom_sampler(upsample_sampler_name, eta, bongmath)
            sigmas_list = re.findall(r"[-+]?(?:\d*\.*\d+)", upsample_manual_sigmas)
            upsample_sigmas = torch.FloatTensor([float(i) for i in sigmas_list])

            for stage in range(2, sampling_stages + 1):
                v_samps, a_samps = sampled_tensor.unbind()
                v_samps = v_samps.to(device)
                a_samps = a_samps.to(device)
                
                global_v_samps_up = process_sliding_window_upscale(
                    pass_name=f"Spatial Upscale Pass {stage}",
                    is_temporal=False, v_samps=v_samps, a_samps=a_samps,
                    upscaler_model=spatial_upscaler, guider=upsample_guider, sampler=upsample_sampler, sigmas=upsample_sigmas,
                    noise_obj=noise_obj, eta=eta, bongmath=bongmath, final_pixels=final_pixels, strengths=strengths,
                    time_scale_factor=time_scale_factor, width_scale_factor=width_scale_factor, height_scale_factor=height_scale_factor, video_vae=video_vae, current_fps=current_fps, disable_pbar=disable_pbar, noise_seed=noise_seed
                )
                
                sampled_tensor = comfy.nested_tensor.NestedTensor((global_v_samps_up, a_samps)).to(device)

        # --- TEMPORAL UPSCALING PASS ---
        if temporal_upscale:
            if temporal_upscaler is None:
                raise ValueError("Temporal upscaler model is required if temporal_upscale is True.")
                
            v_samps, a_samps = sampled_tensor.unbind()
            v_samps = v_samps.to(device)
            a_samps = a_samps.to(device)
                
            if sampling_stages == 1:
                upsample_sampler = build_custom_sampler(upsample_sampler_name, eta, bongmath)

            temporal_sigmas = torch.FloatTensor([0.65, 0.35, 0.12, 0.0])
            
            global_v_samps_up = process_sliding_window_upscale(
                pass_name="Temporal Upscale Pass",
                is_temporal=True, v_samps=v_samps, a_samps=a_samps,
                upscaler_model=temporal_upscaler, guider=upsample_guider, sampler=upsample_sampler, sigmas=temporal_sigmas,
                noise_obj=noise_obj, eta=eta, bongmath=bongmath, final_pixels=None, strengths=strengths,
                time_scale_factor=time_scale_factor, width_scale_factor=width_scale_factor, height_scale_factor=height_scale_factor, video_vae=video_vae, current_fps=current_fps, disable_pbar=disable_pbar, noise_seed=noise_seed
            )
            
            sampled_tensor = comfy.nested_tensor.NestedTensor((global_v_samps_up, a_samps)).to(device)
            current_fps *= 2
            
            if bypass_img_ref:
                out_ref_frame_count = 1
            else:
                out_ref_frame_count = ((1 + duplicate_frames) * 2) - 1
        else:
            if bypass_img_ref:
                out_ref_frame_count = 1
            else:
                out_ref_frame_count = (ref_frame_count // duplicate_frames * 8) + 8 + 1

        # ==========================================
        # 7. THE POST-CLEANSE HARD OVERWRITE
        # ==========================================
        unbound_samples = sampled_tensor.unbind()
        final_video_samples = unbound_samples[0].to(device)
        final_audio_samples = unbound_samples[1].clone().to(device)

        # Audio Safeguard Check
        if has_audio_ref or has_audio_input:
            max_latents = final_audio_samples.shape[2]
            
            if has_audio_ref:
                lock_a = min(region_a_latents, max_latents)
                if lock_a > 0:
                    final_audio_samples[:, :, :lock_a, :] = master_latents[:, :, :lock_a, :]

            if has_audio_input:
                start_c = min(setup_total_latents, max_latents)
                if start_c < max_latents:
                    final_audio_samples[:, :, start_c:max_latents, :] = master_latents[:, :, start_c:max_latents, :]

        # Satisfy ComfyUI's output requirements by generating fresh, clean 5D mask tensors for the final outputs!
        v_batch, _, v_frames, v_height, v_width = final_video_samples.shape
        final_v_mask = torch.ones((v_batch, 1, v_frames, v_height, v_width), dtype=torch.float32, device=device)
        final_a_mask = torch.ones_like(final_audio_samples)
        
        final_latent = {
            "samples": comfy.nested_tensor.NestedTensor((final_video_samples, final_audio_samples)),
            "noise_mask": comfy.nested_tensor.NestedTensor((final_v_mask, final_a_mask)),
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
            print("\n--- Running Integrated Decode & Slicer ---")
            
            # --- MEMORY-EFFICIENT FACE RESTORER INITIALIZATION ---
            fr_device = comfy.model_management.get_torch_device()
            face_helper = None
            loaded_facerestore_model = None
            
            if restore_faces and facerestore_model != "None":
                try:
                    from facelib.utils.face_restoration_helper import FaceRestoreHelper
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
                        
                    # CRITICAL FIX: Ensure the helper explicitly uses the active GPU device, not an intermediate placeholder!
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

            temp_tile_len = 48
            temp_overlap = 8
            
            v_time_scale, v_width_scale, v_height_scale = video_vae.downscale_index_formula
            out_v_frames = 1 + (v_frames - 1) * v_time_scale
            out_v_height = v_height * v_height_scale
            out_v_width = v_width * v_width_scale
            
            decoded_video = torch.empty(
                (v_batch, out_v_frames, out_v_height, out_v_width, 3),
                device=final_video_samples.device,
                dtype=final_video_samples.dtype,
            )
            
            chunk_start = 0
            while chunk_start < v_frames:
                comfy.model_management.soft_empty_cache()
                if chunk_start == 0:
                    chunk_end = min(chunk_start + temp_tile_len, v_frames)
                    overlap_start = chunk_start
                else:
                    overlap_start = max(1, chunk_start - temp_overlap - 1)
                    chunk_end = min(chunk_start + temp_tile_len - (chunk_start - overlap_start), v_frames)
                
                tile = final_video_samples[:, :, overlap_start:chunk_end]
                tile_decoded = video_vae.decode(tile) 
                
                tile_out_frames = 1 + (tile.shape[2] - 1) * v_time_scale
                tile_decoded = tile_decoded.view(v_batch, tile_out_frames, out_v_height, out_v_width, 3)
                
                # --- APPLY FACE RESTORE ON THE FLY ---
                if restore_faces and face_helper is not None and loaded_facerestore_model is not None:
                    print(f"-> Restoring faces in frame batch {overlap_start * v_time_scale} to {chunk_end * v_time_scale}...")
                    restored_tile = []
                    for b in range(v_batch):
                        batch_frames = []
                        for f in range(tile_out_frames):
                            frame_tensor = tile_decoded[b, f]
                            frame_np = (frame_tensor.cpu().numpy() * 255.0).astype(np.uint8)
                            frame_bgr = frame_np[:, :, ::-1]
                            
                            face_helper.clean_all()
                            face_helper.read_image(frame_bgr)
                            face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                            face_helper.align_warp_face()
                            
                            for idx, cropped_face in enumerate(face_helper.cropped_faces):
                                cropped_face_t = cropped_face.astype(np.float32) / 255.0
                                cropped_face_t = cv2.cvtColor(cropped_face_t, cv2.COLOR_BGR2RGB)
                                cropped_face_t = torch.from_numpy(cropped_face_t.transpose(2, 0, 1)).float()
                                
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
                                        
                                        # --- COLOR MATCHING (LAB Space Transfer) ---
                                        if face_restore_color_match:
                                            orig_lab = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2LAB).astype(np.float32)
                                            rest_lab = cv2.cvtColor(final_restored_face, cv2.COLOR_BGR2LAB).astype(np.float32)
                                            
                                            for c in range(3):
                                                orig_mean, orig_std = orig_lab[:,:,c].mean(), orig_lab[:,:,c].std()
                                                rest_mean, rest_std = rest_lab[:,:,c].mean(), rest_lab[:,:,c].std()
                                                # Shift restored face colors to perfectly match original frame lighting/hue
                                                rest_lab[:,:,c] = (rest_lab[:,:,c] - rest_mean) * (orig_std / (rest_std + 1e-6)) + orig_mean
                                                
                                            rest_lab = np.clip(rest_lab, 0, 255).astype(np.uint8)
                                            final_restored_face = cv2.cvtColor(rest_lab, cv2.COLOR_LAB2BGR)
                                            
                                        # --- EDGE FEATHERING (Alpha Blur) ---
                                        if face_restore_edge_blur:
                                            h, w = final_restored_face.shape[:2]
                                            mask = np.zeros((h, w, 3), dtype=np.float32)
                                            pad = int(h * 0.12) # 12% internal boundary padding
                                            mask[pad:h-pad, pad:w-pad] = 1.0
                                            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=pad*0.5, sigmaY=pad*0.5)
                                            
                                            final_restored_face = (final_restored_face * mask + cropped_face.astype(np.float32) * (1.0 - mask)).astype(np.uint8)

                                    face_helper.add_restored_face(final_restored_face)
                                except Exception as error:
                                    print(f'\tFailed inference for CodeFormer on frame {f}: {error}')
                                    face_helper.add_restored_face(cropped_face)
                                    
                            face_helper.get_inverse_affine(None)
                            # This paste function uses BiseNet to create occlusion masks, protecting hands and hair.
                            pasted_img_bgr = face_helper.paste_faces_to_input_image()
                            
                            # DEFLICKER BLEND: Globally crossfade the entire restored frame with the original VRAM frame
                            if face_restore_blend < 1.0:
                                final_img_bgr = cv2.addWeighted(frame_bgr, 1.0 - face_restore_blend, pasted_img_bgr, face_restore_blend, 0)
                            else:
                                final_img_bgr = pasted_img_bgr
                                
                            restored_img_rgb = final_img_bgr[:, :, ::-1]
                            batch_frames.append(torch.from_numpy(restored_img_rgb.astype(np.float32) / 255.0).to(tile_decoded.device))
                        restored_tile.append(torch.stack(batch_frames, dim=0))
                    tile_decoded = torch.stack(restored_tile, dim=0)

                # STITCH THE DECODED TILE INTO TIMELINE
                if chunk_start == 0:
                    decoded_video[:, :tile_decoded.shape[1]] = tile_decoded
                else:
                    tile_decoded = tile_decoded[:, 1:] 
                    out_t_start = 1 + overlap_start * v_time_scale
                    out_t_end = out_t_start + tile_decoded.shape[1]
                    
                    overlap_frames = temp_overlap * v_time_scale
                    frame_weights = torch.linspace(0, 1, overlap_frames + 2, device=tile_decoded.device, dtype=tile_decoded.dtype)[1:-1]
                    tile_weights = frame_weights.view(1, -1, 1, 1, 1)
                    after_overlap = out_t_start + overlap_frames
                    
                    decoded_video[:, out_t_start:after_overlap] *= 1 - tile_weights
                    decoded_video[:, out_t_start:after_overlap] += (tile_weights * tile_decoded[:, :overlap_frames])
                    decoded_video[:, after_overlap:out_t_end] = tile_decoded[:, overlap_frames:]
                    
                chunk_start = chunk_end
            
            decoded_video = decoded_video.view(v_batch * out_v_frames, out_v_height, out_v_width, 3)
            
            a_latent_samples = final_audio_samples
            if a_latent_samples.is_nested:
                a_latent_samples = a_latent_samples.unbind()[-1]
            
            sample_rate = int(audio_vae.output_sample_rate)
            time_to_drop = out_ref_frame_count / current_fps
            samples_to_drop = int(time_to_drop * sample_rate)

            # --- MULTI-SHOT AUDIO VAE ISOLATION FILTER ---
            # Decoding latent discontinuities crashes the VAE convolutions. 
            # We fix this by decoding each shot's latents perfectly independently and crossfading the real waveforms!
            if num_prompts > 1 and autoregressive_chunking:
                print("--- Decoding Multi-Shot Audio Tracks ---")
                shot_duration = length_in_seconds / num_prompts
                decoded_waveforms = []
                
                for i in range(num_prompts):
                    if i == 0:
                        start_lat_a = 0
                    else:
                        sec = i * shot_duration
                        _, start_lat_a = get_latent_counts(sec)
                        
                    if i == num_prompts - 1:
                        end_lat_a = a_latent_samples.shape[2]
                    else:
                        sec = (i + 1) * shot_duration
                        _, end_lat_a = get_latent_counts(sec)
                        
                    shot_a_latent = a_latent_samples[:, :, start_lat_a:end_lat_a]
                    shot_wf = audio_vae.decode(shot_a_latent).to(a_latent_samples.device)
                    
                    # Apply a tiny 50ms fade in/out to each individual shot's waveform
                    fade_samps = min(int(0.05 * sample_rate), shot_wf.shape[-1] // 2)
                    if fade_samps > 0:
                        fade_in = torch.linspace(0.0, 1.0, fade_samps, device=shot_wf.device, dtype=shot_wf.dtype)
                        fade_out = torch.linspace(1.0, 0.0, fade_samps, device=shot_wf.device, dtype=shot_wf.dtype)
                        shot_wf[..., :fade_samps] *= fade_in
                        shot_wf[..., -fade_samps:] *= fade_out
                        
                    decoded_waveforms.append(shot_wf)
                    
                waveform = torch.cat(decoded_waveforms, dim=-1)
            else:
                waveform = audio_vae.decode(a_latent_samples).to(a_latent_samples.device)

            num_frames = decoded_video.shape[0]
            if out_ref_frame_count >= num_frames:
                out_image = decoded_video[-1:] 
            elif out_ref_frame_count > 0:
                out_image = decoded_video[out_ref_frame_count:]
            else:
                out_image = decoded_video
                
            total_samples = waveform.shape[-1]
            
            if samples_to_drop >= total_samples:
                empty_wf = torch.zeros((waveform.shape[0], waveform.shape[1], 1), device=waveform.device, dtype=waveform.dtype)
                out_audio = {"waveform": empty_wf, "sample_rate": sample_rate}
            elif samples_to_drop > 0:
                sliced_wf = waveform[..., samples_to_drop:]
                # ELIMINATE INITIAL TRANSIENT AUDIO POP: Apply a 30ms linear fade-in to the master sliced waveform
                fade_samples = min(int(0.03 * sample_rate), sliced_wf.shape[-1])
                if fade_samples > 0:
                    fade_tensor = torch.linspace(0.0, 1.0, fade_samples, device=sliced_wf.device, dtype=sliced_wf.dtype)
                    sliced_wf[..., :fade_samples] *= fade_tensor
                out_audio = {"waveform": sliced_wf, "sample_rate": sample_rate}
            else:
                out_audio = {"waveform": waveform, "sample_rate": sample_rate}
                
            if InputImpl is not None and Types is not None and out_image is not None:
                out_video = InputImpl.VideoFromComponents(Types.VideoComponents(images=out_image, audio=out_audio, frame_rate=Fraction(current_fps)))
                
            print("--- Decoding & Slicing Complete ---")

        # ==========================================
        # 9. VRAM CLEANUP
        # ==========================================
        if model_to_use is not None:
            model_to_use.unpatch_model()
        if model2_to_use is not None and model2_to_use is not model_to_use:
            model2_to_use.unpatch_model()

        # Sever all local Python references to the models before hitting the Free API!
        # If we don't do this, ComfyUI sees these local variables keeping the model alive and throws a Memory Leak warning.
        del primary_model
        del model_to_use
        del model2_opt
        del model2_to_use
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

        return (final_prompt_string_out, final_latent, video_out_latent, audio_out_latent, out_video, out_image, out_audio, float(current_fps), out_ref_frame_count)

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
        # ==========================================
        # 1. SLICE VIDEO (PIXELS)
        # ==========================================
        num_frames = images.shape[0]
        
        if drop_first_n_frames >= num_frames:
            print(f"LTXV Custom Warning: Trying to drop {drop_first_n_frames} frames but only {num_frames} exist. Returning the last frame to prevent a crash.")
            sliced_images = images[-1:] # Keep 1 frame so downstream nodes don't break
        elif drop_first_n_frames > 0:
            sliced_images = images[drop_first_n_frames:]
            print(f"LTXV Custom: Successfully sliced {drop_first_n_frames} video frames.")
        else:
            sliced_images = images

        # ==========================================
        # 2. SLICE AUDIO (WAVEFORM)
        # ==========================================
        sliced_audio = None
        
        if audio is not None:
            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate")
            
            if waveform is not None and sample_rate is not None:
                # Convert dropped frames to exact audio samples based on FPS
                time_to_drop_seconds = drop_first_n_frames / fps
                samples_to_drop = int(time_to_drop_seconds * sample_rate)
                
                total_samples = waveform.shape[-1]
                
                if samples_to_drop >= total_samples:
                    print(f"LTXV Custom Warning: Trying to drop {samples_to_drop} audio samples but only {total_samples} exist.")
                    # Return an empty waveform of the same channel count to prevent crashes
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
