"""triXope ComfyUI nodes — VAE encode/decode and latent files.

Every node class in this module is registered in the pack's top-level
``__init__.py`` and appears in the ComfyUI node menu under ``triXope/latent``.
The module also holds the private helpers those nodes need.
"""

# --- moved from minimax_h3_av_concat.py:AV concat ---------------------------
import torch
import gc
import comfy.model_management
import comfy.nested_tensor
import comfy.nested_tensor as nested_tensor

FPS = 24
AUDIO_LATENT_FPS = 40


# ---- same math as comfy_extras/nodes_minimax_h3.py -------------------------

def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def audio_latent_t(frame_count):
    duration = frame_count / FPS
    return round(duration * AUDIO_LATENT_FPS)


def frame_count_from_video_latent_t(t_v):
    """Invert video_latent_t for the grid case T_v = 5j+2."""
    j = (t_v - 2) // 5
    return max(5, 17 * j + 5)


def is_on_grid(t_v):
    return t_v >= 2 and (t_v - 2) % 5 == 0


class MiniMaxH3AVLatentConcat:
    """
    Join two MiniMax H3 AV latents back-to-back along time.
    Handles latents of different lengths; snaps the joined video latent
    back onto H3's required grid (T_v ≡ 2 mod 5) and re-derives a matching
    audio length so the two streams stay in sync after trimming.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "trim_side": (["b_start", "a_end"], {
                    "default": "b_start",
                    "tooltip": "Which side of the seam to remove frames from "
                               "when snapping the joined length back onto "
                               "H3's grid.",
                }),
                "extra_trim_latents": ("INT", {
                    "default": 0, "min": 0, "max": 500, "step": 5,
                    "tooltip": "Additional latent video frames to remove at "
                               "the seam beyond what's needed to hit the "
                               "grid (must be a multiple of 5 to stay "
                               "grid-aligned). Useful if the seam still "
                               "looks/sounds discontinuous with 0 extra.",
                }),
            },
            "optional": {
                "sync_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Recompute audio length from the model's own "
                               "frame_count->audio_t formula after trimming, "
                               "instead of naively concatenating both audio "
                               "tensors as-is.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "concat"
    CATEGORY = "triXope/latent"
    DESCRIPTION = ("Concatenate two MiniMax H3 AV latents along time, " 
                   "snapping back onto H3's video/audio length grid.")

    def concat(self, latent_a, latent_b, trim_side="b_start",
               extra_trim_latents=0, sync_audio=True):

        nt_a = latent_a["samples"]
        nt_b = latent_b["samples"]

        if not (isinstance(nt_a, nested_tensor.NestedTensor) and
                isinstance(nt_b, nested_tensor.NestedTensor)):
            raise ValueError(
                "MiniMax H3 Concat AV Latent expects H3 joint AV latents "
                "(NestedTensor of [video, audio]), got a plain tensor. "
                "This node isn't for single-modality latents."
            )
        if len(nt_a.tensors) != 2 or len(nt_b.tensors) != 2:
            raise ValueError(
                "Expected exactly [video, audio] in each AV latent."
            )

        video_a, audio_a = nt_a.tensors
        video_b, audio_b = nt_b.tensors

        # --- sanity checks -------------------------------------------------
        if video_a.shape[0] != video_b.shape[0]:
            raise ValueError(
                f"Batch size mismatch: {video_a.shape[0]} vs {video_b.shape[0]}. "
                "Both latents must have the same batch size."
            )
        if video_a.shape[1] != video_b.shape[1] or video_a.shape[3:] != video_b.shape[3:]:
            raise ValueError(
                f"Video latent channel/spatial size mismatch: "
                f"{tuple(video_a.shape)} vs {tuple(video_b.shape)}. "
                "Both clips must share resolution (and therefore the same "
                "width/height you generated them at) to be joined."
            )
        if audio_a.shape[1] != audio_b.shape[1]:
            # dim 1 is the audio VAE's latent channel count (32) -- a
            # mismatch here means these came from different/incompatible
            # audio VAEs, which can't be safely reconciled.
            raise ValueError(
                f"Audio latent channel-count mismatch: {tuple(audio_a.shape)} "
                f"vs {tuple(audio_b.shape)}. These latents look like they "
                f"came from different audio VAEs."
            )

        device = video_a.device
        dtype = video_a.dtype
        video_b = video_b.to(device=device, dtype=dtype)
        audio_b = audio_b.to(device=device, dtype=audio_a.dtype)

        # dim 2 is mono(1)/stereo(2). H3's own nodes always emit stereo (2)
        # here -- a 1 means this latent didn't come from one of the standard
        # H3 latent-producing nodes. Upmix rather than hard-fail, but say so.
        ch_a, ch_b = audio_a.shape[2], audio_b.shape[2]
        if ch_a != ch_b:
            if {ch_a, ch_b} != {1, 2}:
                raise ValueError(
                    f"Audio latent stereo-dim mismatch: {tuple(audio_a.shape)} "
                    f"vs {tuple(audio_b.shape)} -- not a mono/stereo pair, "
                    f"can't reconcile automatically."
                )
            print(f"[MiniMaxH3ConcatAVLatent] warning: mono/stereo mismatch "
                  f"(audio_a shape={tuple(audio_a.shape)}, "
                  f"audio_b shape={tuple(audio_b.shape)}). Upmixing the mono "
                  f"one by repeating its channel. If that's unexpected, "
                  f"check whatever produced the mono latent -- H3's own "
                  f"nodes always emit stereo (dim2=2) here.")
            if ch_a == 1:
                audio_a = audio_a.repeat(1, 1, 2, 1)
            else:
                audio_b = audio_b.repeat(1, 1, 2, 1)

        print(f"[MiniMaxH3ConcatAVLatent] latent_a video={tuple(video_a.shape)} "
              f"audio={tuple(audio_a.shape)} | latent_b video={tuple(video_b.shape)} "
              f"audio={tuple(audio_b.shape)}")

        t_v_a, t_v_b = video_a.shape[2], video_b.shape[2]
        if not is_on_grid(t_v_a):
            print(f"[MiniMaxH3ConcatAVLatent] warning: latent_a video length "
                  f"{t_v_a} isn't on H3's grid (T≡2 mod 5) -- was this "
                  f"latent produced outside EmptyMiniMaxH3LatentAV / the "
                  f"H3 samplers?")
        if not is_on_grid(t_v_b):
            print(f"[MiniMaxH3ConcatAVLatent] warning: latent_b video length "
                  f"{t_v_b} isn't on H3's grid (T≡2 mod 5).")

        # --- concatenate on the correct axis for each stream ---------------
        video_cat = torch.cat([video_a, video_b], dim=2)
        audio_cat = torch.cat([audio_a, audio_b], dim=3)

        t_v_raw = video_cat.shape[2]

        # --- snap the joined video length back onto the grid ---------------
        remainder = (t_v_raw - 2) % 5
        grid_trim = remainder  # minimal trim to land back on T≡2 mod 5

        if extra_trim_latents % 5 != 0:
            raise ValueError(
                f"extra_trim_latents ({extra_trim_latents}) must be a "
                f"multiple of 5 to keep the result on H3's grid."
            )

        total_trim = grid_trim + extra_trim_latents
        t_v_final = t_v_raw - total_trim
        if t_v_final < 2:
            raise ValueError(
                f"Requested trim ({total_trim} latent frames) would shrink "
                f"the joined video below the minimum valid length. Reduce "
                f"extra_trim_latents."
            )

        if total_trim > 0:
            if trim_side == "b_start":
                # remove frames from the start of B, right after the seam
                video_cat = torch.cat(
                    [video_cat[:, :, :t_v_a, ...],
                     video_cat[:, :, t_v_a + total_trim:, ...]],
                    dim=2,
                )
            else:  # "a_end"
                cut_from_a = min(total_trim, t_v_a)
                remaining = total_trim - cut_from_a
                video_cat = torch.cat(
                    [video_cat[:, :, :t_v_a - cut_from_a, ...],
                     video_cat[:, :, t_v_a + remaining:, ...]],
                    dim=2,
                )

        assert video_cat.shape[2] == t_v_final
        if not is_on_grid(t_v_final):
            # Shouldn't happen given the math above, but don't silently
            # hand the sampler a bad shape if it does.
            raise RuntimeError(
                f"Internal error: trimmed video length {t_v_final} is still "
                f"off-grid. Please report this with your input latent shapes."
            )

        # --- keep audio in sync with the trimmed video ----------------------
        if sync_audio:
            frame_count_final = frame_count_from_video_latent_t(t_v_final)
            target_audio_t = audio_latent_t(frame_count_final)
            cur_audio_t = audio_cat.shape[3]
            diff = cur_audio_t - target_audio_t

            if diff > 0:
                # trim audio on the same side as the video trim
                t_a_a = audio_a.shape[3]
                if trim_side == "b_start":
                    cut_start = min(diff, cur_audio_t - t_a_a)
                    audio_cat = torch.cat(
                        [audio_cat[:, :, :, :t_a_a],
                         audio_cat[:, :, :, t_a_a + cut_start:]],
                        dim=3,
                    )
                    # if still too long (diff > b's remaining), trim tail
                    if audio_cat.shape[3] > target_audio_t:
                        audio_cat = audio_cat[:, :, :, :target_audio_t]
                else:
                    cut_end = min(diff, t_a_a)
                    audio_cat = torch.cat(
                        [audio_cat[:, :, :, :t_a_a - cut_end],
                         audio_cat[:, :, :, t_a_a:]],
                        dim=3,
                    )
                    if audio_cat.shape[3] > target_audio_t:
                        audio_cat = audio_cat[:, :, :, cut_end:]
            elif diff < 0:
                # pad with a repeat of the last sample rather than zeros,
                # to avoid a hard silence click at the tail
                pad = audio_cat[:, :, :, -1:].repeat(1, 1, 1, -diff)
                audio_cat = torch.cat([audio_cat, pad], dim=3)

        result = nested_tensor.NestedTensor((video_cat, audio_cat))
        out = dict(latent_a)
        out["samples"] = result
        out.pop("noise_mask", None)
        out.pop("batch_index", None)
        return (out,)

# --- moved from minimax_h3_av_concat.py:list decoder ------------------------
def format_waveform_to_bcs(wf):
    """
    Ensures the audio waveform tensor is strictly formatted as [Batch, Channels, Samples].
    Fixes FFmpeg channel count crashes.
    """
    wf = wf.cpu()
    if wf.ndim == 2:
        # [Samples, Channels] -> [Channels, Samples]
        if wf.shape[0] > wf.shape[1]:
            wf = wf.transpose(0, 1)
        wf = wf.unsqueeze(0)
    elif wf.ndim == 3:
        # [Batch, Samples, Channels] -> [Batch, Channels, Samples]
        if wf.shape[1] > wf.shape[2]:
            wf = wf.movedim(-1, 1)
    elif wf.ndim == 4:
        wf = wf.squeeze(1)
        if wf.ndim == 3 and wf.shape[1] > wf.shape[2]:
            wf = wf.movedim(-1, 1)
    return wf


def get_audio_sample_rate(vae):
    """
    Safely inspects the VAE model to find the true decoded PCM output sample rate
    (e.g., 44100Hz or 48000Hz instead of internal 24000Hz latent rates).
    """
    # 1. Check direct attributes
    if hasattr(vae, "output_sample_rate") and vae.output_sample_rate:
        return int(vae.output_sample_rate)
    
    # 2. Check inner model attributes
    if hasattr(vae, "first_stage_model"):
        fsm = vae.first_stage_model
        if hasattr(fsm, "output_sample_rate") and fsm.output_sample_rate:
            return int(fsm.output_sample_rate)
        if hasattr(fsm, "sample_rate") and fsm.sample_rate:
            return int(fsm.sample_rate)

    if hasattr(vae, "sample_rate") and vae.sample_rate:
        return int(vae.sample_rate)

    # 3. Default high-definition audio fallback
    return 44100


class SafeLatentListDecoder:
    """
    RAM-Safe Latent Decoder (Video + Audio)
    Pre-allocates CPU system RAM and streams latent chunk decoding to prevent VRAM OOM crashes.
    Decodes both video and audio simultaneously from NestedTensor latents.
    """
    INPUT_IS_LIST = True  # Prevents ComfyUI from splitting input lists into multiple runs

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "The VAE model to decode video latents."}),
                "latents": ("*", {"tooltip": "Accepts a List of Latents, Akatz Accumulation object, or Latent dictionary."}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "Optional Audio VAE model (e.g. LTXV Audio VAE) to decode audio latents."}),
                "vram_flush": ("BOOLEAN", {"default": True, "tooltip": "Force VRAM garbage collection after decoding each latent chunk."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "decode_safe"
    CATEGORY = "triXope/latent"

    def decode_safe(self, vae, latents, audio_vae=None, vram_flush=True):
        # 1. Unpack ComfyUI list-wrapped arguments
        vae_model = vae[0] if isinstance(vae, list) else vae
        audio_vae_model = audio_vae[0] if isinstance(audio_vae, list) and len(audio_vae) > 0 else audio_vae
        do_flush = vram_flush[0] if isinstance(vram_flush, list) else vram_flush

        # 2. Recursively flatten and extract all latent dictionaries from input payload
        latent_items = []

        def extract_items(obj):
            if isinstance(obj, list):
                for sub in obj:
                    extract_items(sub)
            elif isinstance(obj, dict):
                if "accum" in obj and isinstance(obj["accum"], list):
                    for sub in obj["accum"]:
                        extract_items(sub)
                elif "samples" in obj:
                    latent_items.append(obj)
                else:
                    latent_items.append(obj)

        extract_items(latents)

        if not latent_items:
            raise ValueError("[SafeLatentDecoder] No latents found in input payload!")

        # 3. Decode latents one by one on GPU & hold decoded video/audio on CPU
        decoded_video_chunks = []
        audio_waveforms = []
        total_pixel_frames = 0
        
        target_audio_vae = audio_vae_model if audio_vae_model is not None else vae_model
        sample_rate = get_audio_sample_rate(target_audio_vae)

        for idx, item in enumerate(latent_items):
            samples = item["samples"] if isinstance(item, dict) and "samples" in item else item

            v_tensor = None
            a_tensor = None

            # Safely unbind NestedTensor (Video + Audio) if present
            if hasattr(samples, "unbind") and not isinstance(samples, torch.Tensor):
                unbound = samples.unbind()
                v_tensor = unbound[0]
                if len(unbound) > 1:
                    a_tensor = unbound[1]
            elif isinstance(samples, torch.Tensor):
                v_tensor = samples
            else:
                try:
                    unbound = samples.unbind()
                    v_tensor = unbound[0]
                    if len(unbound) > 1:
                        a_tensor = unbound[1]
                except Exception:
                    v_tensor = samples

            print(f"[SafeLatentDecoder] Decoding latent clip {idx + 1}/{len(latent_items)} on GPU...")

            # --- Video Decoding ---
            decoded_v = vae_model.decode(v_tensor).cpu()

            # Standardize shape to 4D: [Pixel_Frames, Height, Width, Channels]
            if decoded_v.ndim == 5:
                if decoded_v.shape[-1] == 3:
                    decoded_v = decoded_v.view(-1, decoded_v.shape[2], decoded_v.shape[3], 3)
                else:
                    decoded_v = decoded_v.squeeze(0).permute(1, 2, 3, 0)
            elif decoded_v.ndim == 4 and decoded_v.shape[-1] != 3:
                decoded_v = decoded_v.permute(0, 2, 3, 1)

            num_pixel_frames = decoded_v.shape[0]
            total_pixel_frames += num_pixel_frames
            decoded_video_chunks.append(decoded_v)

            # --- Audio Decoding ---
            if a_tensor is not None and hasattr(target_audio_vae, "decode"):
                try:
                    decoded_a = target_audio_vae.decode(a_tensor)
                    if hasattr(decoded_a, "to"):
                        decoded_a = decoded_a.to(a_tensor.device)

                    # Format waveform to standard [Batch, Channels, Samples]
                    wf = format_waveform_to_bcs(decoded_a)
                    audio_waveforms.append(wf)
                except Exception as e:
                    print(f"[SafeLatentDecoder] Audio decode warning on clip {idx + 1}: {e}")

            # Flush GPU VRAM after each clip
            del v_tensor, a_tensor
            if do_flush:
                gc.collect()
                comfy.model_management.soft_empty_cache()

        # 4. Pre-allocate single CPU Tensor for combined VIDEO
        _, pixel_h, pixel_w, _ = decoded_video_chunks[0].shape
        print(f"[SafeLatentDecoder] Combining {len(decoded_video_chunks)} clips into a single batch of {total_pixel_frames} pixel frames ({pixel_w}x{pixel_h})...")

        out_images = torch.zeros((total_pixel_frames, pixel_h, pixel_w, 3), dtype=torch.float32, device="cpu")

        current_frame_idx = 0
        for chunk in decoded_video_chunks:
            chunk_len = chunk.shape[0]
            out_images[current_frame_idx : current_frame_idx + chunk_len] = chunk
            current_frame_idx += chunk_len

        del decoded_video_chunks

        # 5. Combine AUDIO waveforms end-to-end along time axis (dim=-1)
        out_audio = None
        if audio_waveforms:
            combined_wf = torch.cat(audio_waveforms, dim=-1)
            out_audio = {
                "waveform": combined_wf,
                "sample_rate": sample_rate
            }
            print(f"[SafeLatentDecoder] Combined audio track: {combined_wf.shape[-1]} samples at {sample_rate}Hz.")
        else:
            # Fallback silent audio dict if no audio present
            out_audio = {
                "waveform": torch.zeros((1, 2, 1), dtype=torch.float32, device="cpu"),
                "sample_rate": sample_rate
            }

        if do_flush:
            gc.collect()
            comfy.model_management.soft_empty_cache()

        print(f"[SafeLatentDecoder] Success! Sending {total_pixel_frames} frames + audio track to outputs.\n")
        return (out_images, out_audio)

# --- moved from video_latents.py --------------------------------------------
import os
import re
import gc
import torch
import safetensors.torch
import comfy.model_management as mm
import uuid
import shutil
import numpy as np

class SaveWanVideoLatent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", ),
                "save_path": ("STRING", {"default": "./output/latents", "multiline": False}),
                "prefix": ("STRING", {"default": "video_latents", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("directory_path",)
    FUNCTION = "save_latent"
    CATEGORY = "triXope/latent"
    
    OUTPUT_NODE = True 

    def save_latent(self, samples, save_path, prefix):
        save_path = os.path.normpath(save_path)
        os.makedirs(save_path, exist_ok=True)

        max_index = 0
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{4}})\.latent$")
        
        if os.path.exists(save_path):
            for filename in os.listdir(save_path):
                match = pattern.match(filename)
                if match:
                    index = int(match.group(1))
                    if index > max_index:
                        max_index = index
        
        next_index = max_index + 1
        new_file_name = f"{prefix}_{next_index:04d}.latent"
        full_path = os.path.join(save_path, new_file_name)

        output = {}
        output["latent_tensor"] = samples["samples"]
        output["latent_format_version_0"] = torch.tensor([])

        for k in samples:
            if k != "samples":
                output[k] = samples[k]

        try:
            torch.save(output, full_path)
            print(f"[SaveVideoLatent] Successfully saved latents to: {full_path}")
        except Exception as e:
            print(f"[SaveVideoLatent] Error saving latents: {e}")

        return (save_path,)

class SaveLTXVideoLatent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", ),
                "save_path": ("STRING", {"default": "./output/latents", "multiline": False}),
                "prefix": ("STRING", {"default": "video_latents", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("directory_path",)
    FUNCTION = "save_latent"
    CATEGORY = "triXope/latent"
    
    OUTPUT_NODE = True 

    def save_latent(self, samples, save_path, prefix):
        save_path = os.path.normpath(save_path)
        os.makedirs(save_path, exist_ok=True)

        max_index = 0
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{4}})\.latent$")
        
        if os.path.exists(save_path):
            for filename in os.listdir(save_path):
                match = pattern.match(filename)
                if match:
                    index = int(match.group(1))
                    if index > max_index:
                        max_index = index
        
        next_index = max_index + 1
        new_file_name = f"{prefix}_{next_index:04d}.latent"
        full_path = os.path.join(save_path, new_file_name)

        # --- FLAWLESS STANDARD COMfyUI SAVER ---
        output = {}
        output["latent_tensor"] = samples["samples"]
        if "latent_format_version_0" not in samples:
            output["latent_format_version_0"] = torch.tensor([])
            
        for k in samples:
            if k != "samples":
                output[k] = samples[k]

        try:
            torch.save(output, full_path)
            print(f"[SaveVideoLatent] Successfully saved to: {full_path}")
        except Exception as e:
            print(f"[SaveVideoLatent] Error saving latents: {e}")

        return (save_path,)

class LoadAndConcatVideoLatentsFromDir:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "E:/A1111/output/latents", "multiline": False}),
                "latent_load_cap": ("INT", {"default": -1, "min": -1, "step": 1}),
                "skip_first_latents": ("INT", {"default": 0, "min": 0, "step": 1}),
                "concat_dimension": (["batch (dim=0)", "time/frames (dim=2)"], {"default": "batch (dim=0)"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "load_and_concat"
    CATEGORY = "triXope/latent"

    def _load_single_file(self, file_path):
        try:
            loaded_data = safetensors.torch.load_file(file_path, device="cpu")
        except Exception:
            try:
                loaded_data = torch.load(file_path, map_location="cpu", weights_only=False)
            except Exception as e:
                raise ValueError(f"Failed to load {file_path}: {e}")

        if "latent_tensor" in loaded_data:
            tensor = loaded_data["latent_tensor"]
        elif "samples" in loaded_data:
            tensor = loaded_data["samples"]
        else:
            tensor = loaded_data

        metadata = {}
        if isinstance(loaded_data, dict):
            for k, v in loaded_data.items():
                if k not in ["latent_tensor", "samples"]:
                    metadata[k] = v

        return tensor.to("cpu"), metadata

    def load_and_concat(self, directory_path, latent_load_cap, skip_first_latents, concat_dimension):
        directory_path = os.path.normpath(directory_path)
        
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"[LoadAndConcatVideoLatentsFromDir] Directory not found: {directory_path}")

        valid_files = [f for f in os.listdir(directory_path) if f.endswith(".latent")]
        valid_files.sort()

        if skip_first_latents > 0:
            valid_files = valid_files[skip_first_latents:]
        if latent_load_cap != -1:
            valid_files = valid_files[:latent_load_cap]

        total_files = len(valid_files)
        if total_files == 0:
            raise ValueError("[LoadAndConcatVideoLatentsFromDir] No valid .latent files found.")

        dim = 0 if "dim=0" in concat_dimension else 2
        
        print(f"[LoadAndConcatVideoLatentsFromDir] Analyzing {total_files} files...")
        
        shapes = []
        dtype = None
        final_dict = {}

        for idx, file_name in enumerate(valid_files):
            file_path = os.path.join(directory_path, file_name)
            try:
                tensor, metadata = self._load_single_file(file_path)
                shapes.append(tensor.shape)
                if dtype is None:
                    dtype = tensor.dtype
                if idx == 0:
                    final_dict = metadata
                del tensor, metadata
                gc.collect()
            except Exception as e:
                print(f"[LoadAndConcatVideoLatentsFromDir] Skipping {file_name}: {e}")
                continue

        final_shape = list(shapes[0])
        final_shape[dim] = sum(s[dim] for s in shapes)
        
        final_tensor = torch.zeros(final_shape, dtype=dtype, device="cpu")
        current_idx = 0 

        for idx, file_name in enumerate(valid_files):
            print(f"[LoadAndConcatVideoLatentsFromDir] Loading {idx + 1}/{total_files}: {file_name} ...")
            file_path = os.path.join(directory_path, file_name)
            
            tensor, _ = self._load_single_file(file_path)
            size = tensor.shape[dim]

            slices = [slice(None)] * final_tensor.ndim
            slices[dim] = slice(current_idx, current_idx + size)
            
            final_tensor[tuple(slices)] = tensor
            current_idx += size
            
            del tensor
            gc.collect()

        final_dict["samples"] = final_tensor
        return (final_dict,)

class WanDirectoryVideoDecoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "E:/A1111/output/latents", "multiline": False}),
                "vae": ("WANVAE",),
                "latent_load_cap": ("INT", {"default": -1, "min": -1, "step": 1}),
                "skip_first_latents": ("INT", {"default": 0, "min": 0, "step": 1}),
                "tile_x": ("INT", {"default": 1792, "min": 256, "max": 4096, "step": 16}),
                "tile_y": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode_directory"
    CATEGORY = "triXope/latent"

    def _load_single_file(self, file_path):
        try:
            loaded_data = safetensors.torch.load_file(file_path, device="cpu")
        except Exception:
            loaded_data = torch.load(file_path, map_location="cpu", weights_only=False)

        if "latent_tensor" in loaded_data:
            return loaded_data["latent_tensor"].to("cpu")
        elif "samples" in loaded_data:
            return loaded_data["samples"].to("cpu")
        else:
            return loaded_data.to("cpu")

    def decode_directory(self, directory_path, vae, latent_load_cap, skip_first_latents, tile_x, tile_y):
        directory_path = os.path.normpath(directory_path)
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"[Wan Decoder] Directory not found: {directory_path}")
            
        valid_files = [f for f in os.listdir(directory_path) if f.endswith(".latent")]
        valid_files.sort()

        if skip_first_latents > 0:
            valid_files = valid_files[skip_first_latents:]
        if latent_load_cap != -1:
            valid_files = valid_files[:latent_load_cap]

        total_files = len(valid_files)
        if total_files == 0:
            raise ValueError("[Wan Decoder] No valid .latent files found.")

        print(f"[Wan Decoder] Starting sequential decoding of {total_files} clips...")
        
        device = mm.get_torch_device()
        
        print(f"[Wan Decoder] Moving VAE to {device}...")
        if hasattr(vae, 'model'):
            vae.model.to(device)
        elif hasattr(vae, 'first_stage_model'):
            vae.first_stage_model.to(device)
        
        try:
            if hasattr(vae, 'model'):
                target_dtype = next(vae.model.parameters()).dtype
            else:
                target_dtype = next(vae.first_stage_model.parameters()).dtype
        except Exception:
            target_dtype = torch.bfloat16
            
        print(f"[Wan Decoder] Auto-casting tensors to match VAE dtype: {target_dtype}")

        all_decoded_frames = []

        t_x = tile_x // 8
        t_y = tile_y // 8
        s_x = 256 // 8
        s_y = 256 // 8

        with torch.inference_mode():
            for idx, file_name in enumerate(valid_files):
                print(f"[Wan Decoder] Decoding clip {idx + 1}/{total_files}: {file_name} ...")
                file_path = os.path.join(directory_path, file_name)
                
                tensor = self._load_single_file(file_path)
                
                if len(tensor.shape) == 4:
                    tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)
                elif len(tensor.shape) == 5 and tensor.shape[0] != 1:
                    tensor = tensor[0:1]

                tensor = tensor.to(device=device, dtype=target_dtype)

                decoded = vae.decode(
                    tensor, 
                    device=device, 
                    end_=False, 
                    tiled=True, 
                    tile_size=(t_x, t_y), 
                    tile_stride=(s_x, s_y)
                )
                
                # --- THE FINAL FIX: Safely Format to ComfyUI Image Standard ---
                if isinstance(decoded, tuple) or isinstance(decoded, list):
                    decoded = decoded[0]

                # Remove the batch dimension
                if len(decoded.shape) == 5:
                    decoded = decoded.squeeze(0) # Becomes [Channels, Time, Height, Width]
                    
                # Pivot from [Channels, Time, Height, Width] to [Time, Height, Width, Channels]
                if decoded.shape[0] == 3 or decoded.shape[0] == 4:
                    decoded = decoded.permute(1, 2, 3, 0)

                # Move back to CPU and convert to float32 for VideoCombine
                decoded = decoded.cpu().float()

                # Normalize raw VAE colors from [-1, 1] to [0, 1] 
                if decoded.min() < 0:
                    decoded = (decoded + 1.0) / 2.0
                
                decoded = decoded.clamp(0.0, 1.0)
                    
                all_decoded_frames.append(decoded)

                del tensor, decoded
                gc.collect()
                mm.soft_empty_cache()
                
        if hasattr(vae, 'model'):
            vae.model.to("cpu")

        print("[Wan Decoder] Stitching all clips into final continuous video...")
        # Because we pivoted to [Time, Height, Width, Channels], dim=0 perfectly stitches the timeline!
        final_video = torch.cat(all_decoded_frames, dim=0)
        print(f"[Wan Decoder] Success! Final video shape: {final_video.shape}")
        
        return (final_video,)

import os
import gc
import torch
import safetensors.torch
import comfy.model_management as mm
import uuid
import shutil
import numpy as np

class LTXVDirectoryVideoDecoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "E:/A1111/output/latents", "multiline": False}),
                "vae": ("VAE",),
                "latent_load_cap": ("INT", {"default": -1, "min": -1, "step": 1}),
                "skip_first_latents": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode_directory"
    CATEGORY = "triXope/latent"

    def _load_single_file(self, file_path):
        try:
            loaded_data = safetensors.torch.load_file(file_path, device="cpu")
        except Exception:
            loaded_data = torch.load(file_path, map_location="cpu", weights_only=False)

        if "latent_tensor" in loaded_data:
            tensor = loaded_data["latent_tensor"].to("cpu").clone()
        elif "samples" in loaded_data:
            tensor = loaded_data["samples"].to("cpu").clone()
        else:
            if isinstance(loaded_data, torch.Tensor):
                tensor = loaded_data.to("cpu").clone()
            else:
                tensor = loaded_data.to("cpu").clone()
                
        del loaded_data
        return tensor

    def decode_directory(self, directory_path, vae, latent_load_cap, skip_first_latents):
        directory_path = os.path.normpath(directory_path)
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"[LTXV Decoder] Directory not found: {directory_path}")
            
        valid_files = [f for f in os.listdir(directory_path) if f.endswith(".latent")]
        valid_files.sort()

        if skip_first_latents > 0:
            valid_files = valid_files[skip_first_latents:]
        if latent_load_cap != -1:
            valid_files = valid_files[:latent_load_cap]

        total_files = len(valid_files)
        if total_files == 0:
            raise ValueError("[LTXV Decoder] No valid .latent files found.")

        print(f"[LTXV Decoder] Starting Direct-to-Disk Binary Stream of {total_files} files...")
        
        device = mm.get_torch_device()
        vae.first_stage_model.to(device)
        target_dtype = next(vae.first_stage_model.parameters()).dtype
        
        temp_dir = os.path.join(directory_path, f"_ltxv_temp_{uuid.uuid4().hex[:8]}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # --- THE MASTER BINARY FILE ---
        master_file_path = os.path.join(temp_dir, "master_stitched_video.bin")
        shapes_list = []

        with torch.inference_mode():
            for idx, file_name in enumerate(valid_files):
                print(f"[LTXV Decoder] Decoding & Streaming clip {idx + 1}/{total_files}: {file_name} ...")
                file_path = os.path.join(directory_path, file_name)
                
                tensor = self._load_single_file(file_path)
                
                if tensor.numel() == 0:
                    continue
                
                if len(tensor.shape) == 4:
                    if tensor.shape[0] == 128: tensor = tensor.unsqueeze(0)
                    elif tensor.shape[1] == 128: tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)
                    else: tensor = tensor.unsqueeze(0)
                elif len(tensor.shape) == 5:
                    if tensor.shape[2] == 128: tensor = tensor.permute(0, 2, 1, 3, 4)
                    elif tensor.shape[0] == 128: tensor = tensor.permute(1, 0, 2, 3, 4)

                try:
                    chunk_gpu = tensor.to(device=device, dtype=target_dtype)
                    raw_decoded = vae.first_stage_model.decode(chunk_gpu)
                    
                    if len(raw_decoded.shape) == 5:
                        raw_decoded = raw_decoded.squeeze(0)
                        
                    if raw_decoded.shape[0] in [3, 4]:
                        raw_decoded = raw_decoded.permute(1, 2, 3, 0)
                    elif raw_decoded.shape[1] in [3, 4]:
                        raw_decoded = raw_decoded.permute(0, 2, 3, 1)
                        
                    raw_decoded = (raw_decoded + 1.0) / 2.0
                    decoded_half = raw_decoded.clamp(0.0, 1.0).half() 
                    
                    decoded = decoded_half.cpu()
                    del chunk_gpu, raw_decoded, decoded_half
                    
                except Exception as e:
                    error_msg = str(e)
                    del e 
                    print(f"   -> Direct VRAM decode failed. Falling back to ComfyUI standard decode. Error: {error_msg}")
                    
                    mm.soft_empty_cache()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    gc.collect()
                    
                    decoded_tuple = vae.decode(tensor)
                    decoded = decoded_tuple[0] if isinstance(decoded_tuple, (tuple, list)) else decoded_tuple
                    decoded = decoded.half()
                
                shapes_list.append(decoded.shape)
                
                # --- ZERO RAM OVERHEAD BYPASS ---
                # Converts the tensor to raw binary and appends it directly to the master file
                with open(master_file_path, "ab") as f:
                    decoded.numpy().tofile(f)

                del tensor, decoded
                gc.collect()
                mm.soft_empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
        vae.first_stage_model.to("cpu")
        
        if not shapes_list:
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            raise ValueError("\n\n[LTXV Decoder] CRITICAL ERROR: No valid video data was decoded!\n")
                
        print("[LTXV Decoder] Mapping master file directly into ComfyUI...")
        total_frames = sum(shape[0] for shape in shapes_list)
        final_H, final_W, final_C = shapes_list[0][1], shapes_list[0][2], shapes_list[0][3]
        
        try:
            # Mode 'c' (copy-on-write) allows ComfyUI to read the file safely without modifying your SSD
            mmap_video = np.memmap(master_file_path, dtype=np.float16, mode='c', shape=(total_frames, final_H, final_W, final_C))
        except Exception as e:
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            raise RuntimeError(f"[LTXV Decoder] SSD Error: {e}")

        final_video = torch.from_numpy(mmap_video)
        
        print(f"[LTXV Decoder] Success! Final video streamed entirely from disk. Shape: {final_video.shape}")
        
        return (final_video,)

# --- moved from Trixope_VAE_Encoder.py --------------------------------------
import torch
import comfy.model_management as mm
from comfy.utils import ProgressBar

class Trixope_WanVAEEncodeBatched:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("WANVAE",),
                "per_batch": ("INT", {"default": 16, "min": 4, "step": 4, "tooltip": "Number of new frames to encode per batch. Must be a multiple of 4."}),
                "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Enable tiled encoding to reduce VRAM usage at the cost of speed."}),
                "tile_x": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "tooltip": "Width of the tile in pixels."}),
                "tile_y": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "tooltip": "Height of the tile in pixels."}),
                "tile_stride_x": ("INT", {"default": 256, "min": 32, "max": 4096, "step": 32, "tooltip": "Horizontal overlap between tiles."}),
                "tile_stride_y": ("INT", {"default": 256, "min": 32, "max": 4096, "step": 32, "tooltip": "Vertical overlap between tiles."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "triXope/latent"

    def encode(self, pixels, vae, per_batch, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y):
        if per_batch % 4 != 0:
            per_batch = max(4, (per_batch // 4) * 4)
            print(f"Warning: 'per_batch' was not a multiple of 4. Adjusted to {per_batch}.")
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        original_total_frames = pixels.shape[0]
        OVERLAP_FRAMES = 16

        if original_total_frames < OVERLAP_FRAMES:
            raise ValueError(f"Input video must have at least {OVERLAP_FRAMES} frames for a 16-frame overlap.")

        new_frames_to_process = original_total_frames - OVERLAP_FRAMES
        padding_needed = (per_batch - (new_frames_to_process % per_batch)) % per_batch

        if padding_needed > 0:
            # print(f"Padding video with {padding_needed} frames to ensure all frames are processed.")
            padding = pixels[-1:].repeat(padding_needed, 1, 1, 1)
            pixels = torch.cat([pixels, padding], dim=0)

        total_frames = pixels.shape[0]
        latent_chunks = []
        pbar = ProgressBar(original_total_frames)
        LATENT_OVERLAP = OVERLAP_FRAMES // 4
        memory_frames = None

        for start_idx in range(0, total_frames, per_batch):
            end_idx = min(start_idx + per_batch, total_frames)
            image_chunk = pixels[start_idx:end_idx].clone()
            frames_in_chunk = image_chunk.shape[0]
            if frames_in_chunk == 0: continue

            if memory_frames is not None:
                chunk_with_context = torch.cat([memory_frames, image_chunk], dim=0)
            else:
                if frames_in_chunk < OVERLAP_FRAMES: continue
                chunk_with_context = image_chunk

            vae.to(device)
            image_chunk_processed = chunk_with_context.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3)
            image_chunk_processed = image_chunk_processed * 2.0 - 1.0
            latents = vae.encode(image_chunk_processed, device=device, tiled=enable_vae_tiling, tile_size=(tile_x // vae.upsampling_factor, tile_y // vae.upsampling_factor), tile_stride=(tile_stride_x // vae.upsampling_factor, tile_stride_y // vae.upsampling_factor))
            
            if latents.ndim == 5: latents = latents.squeeze(0)
            
            if memory_frames is not None:
                new_latents = latents[:, LATENT_OVERLAP:, :, :]
            else:
                new_latents = latents
            
            latent_chunks.append(new_latents.cpu())
            
            memory_frames = image_chunk[-OVERLAP_FRAMES:].clone()

            vae.model.clear_cache()
            vae.to(offload_device)
            mm.soft_empty_cache()
            pbar.update(frames_in_chunk)

        final_latents = torch.cat(latent_chunks, dim=1)
        
        expected_latent_len = (original_total_frames + 3) // 4
        final_latents = final_latents[:, :expected_latent_len, :, :]

        return ({"samples": final_latents},)

# --- moved from Trixope_VAE_Decoder.py --------------------------------------
import torch
import comfy.model_management as mm
from comfy.utils import ProgressBar

try:
    from .taehv import TAEHV
except ImportError:
    print("Warning: triXope node could not import TAEHV. Please ensure taehv.py is in the triXope folder.")
    TAEHV = None

class Trixope_WanVAEDecodeBatched:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("WANVAE",),
                "per_batch": ("INT", {"default": 4, "min": 1, "step": 1, "tooltip": "Number of new latent frames to decode per batch."}),
                "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Enable tiled decoding to reduce VRAM usage at the cost of speed."}),
                "tile_x": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "tooltip": "Width of the tile in pixels."}),
                "tile_y": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "tooltip": "Height of the tile in pixels."}),
                "tile_stride_x": ("INT", {"default": 256, "min": 32, "max": 4096, "step": 32, "tooltip": "Horizontal overlap between tiles."}),
                "tile_stride_y": ("INT", {"default": 256, "min": 32, "max": 4096, "step": 32, "tooltip": "Vertical overlap between tiles."}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "triXope/latent"

    def decode(self, samples, vae, per_batch, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        latents = samples["samples"]
        
        total_latent_frames = latents.shape[2]
        image_chunks = []
        pbar = ProgressBar(total_latent_frames)
        
        LATENT_OVERLAP = 2
        IMAGE_OVERLAP = LATENT_OVERLAP * 4

        if total_latent_frames < LATENT_OVERLAP:
            raise ValueError(f"Input latent must have at least {LATENT_OVERLAP} frames for an 8-frame overlap.")
        
        memory_latents = latents[:, :, :LATENT_OVERLAP, :, :].clone()

        for start_idx in range(0, total_latent_frames, per_batch):
            end_idx = min(start_idx + per_batch, total_latent_frames)
            
            # The chunk of "new" latents to process in this iteration
            latent_chunk = latents[:, :, start_idx:end_idx, :, :].clone()
            
            latent_frames_in_chunk = latent_chunk.shape[2]
            if latent_frames_in_chunk == 0: continue

            chunk_with_context = torch.cat([memory_latents, latent_chunk], dim=2)
            
            vae.to(device)
            latent_for_vae = chunk_with_context.to(vae.dtype)

            if TAEHV is not None and isinstance(vae, TAEHV):
                latent_for_vae = latent_for_vae.permute(0, 2, 1, 3, 4)
                images = vae.decode_video(latent_for_vae, parallel=False)
                images = images.permute(0, 2, 1, 3, 4)
            else:
                images = vae.decode(
                    latent_for_vae,
                    device=device,
                    tiled=enable_vae_tiling,
                    tile_size=(tile_x // vae.upsampling_factor, tile_y // vae.upsampling_factor),
                    tile_stride=(tile_stride_x // vae.upsampling_factor, tile_stride_y // vae.upsampling_factor)
                )[0]

            images = images.squeeze(0)

            new_images = images[:, IMAGE_OVERLAP:, :, :]
            
            expected_image_len = latent_frames_in_chunk * 4
            new_images = new_images[:, :expected_image_len, :, :]

            image_chunks.append(new_images.cpu())
            
            memory_latents = chunk_with_context[:, :, -LATENT_OVERLAP:, :, :].clone()
            
            vae.model.clear_cache()
            vae.to(offload_device)
            mm.soft_empty_cache()
            pbar.update(latent_frames_in_chunk)
            
        final_images = torch.cat(image_chunks, dim=1)
        final_images = final_images.clamp_(-1.0, 1.0)
        final_images = (final_images + 1.0) / 2.0
        return (final_images.permute(1, 2, 3, 0),)

# --- moved from Trixope_VAE_Decoder_Native.py -------------------------------
import torch
import comfy.model_management as mm
from comfy.utils import ProgressBar

class Trixope_VAEDecodeBatched_Native:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
                "vae": ("VAE",),
                "per_batch": ("INT", {"default": 8, "min": 1, "step": 1, "tooltip": "Number of latent frames to decode at once. Lower this to reduce VRAM usage."}),
                "latent_overlap": ("INT", {"default": 0, "min": 0, "step": 1, "tooltip": "Frames to overlap for temporal consistency in videos. Set to 0 for standard image batches."}),
                "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Enable tiled decoding to reduce VRAM usage further."}),
                "tile_x": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "tooltip": "The width of the tile in pixels for tiled decoding."}),
                "tile_y": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "tooltip": "The height of the tile in pixels for tiled decoding."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "triXope/latent"

    def decode(self, latents, vae, per_batch, latent_overlap, enable_vae_tiling, tile_x, tile_y):
        samples = latents["samples"]

        # --- FIX: Detect and Reshape 5D Tensors ---
        if samples.ndim == 5:
            # Reshape from (B, C, F, H, W) to a 4D batch of frames (B*F, C, H, W)
            b, c, f, h, w = samples.shape
            samples = samples.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        # --- END FIX ---

        original_total_frames = samples.shape[0]
        pbar = ProgressBar(original_total_frames)
        pixel_chunks = []

        if latent_overlap > 0:
            if original_total_frames < latent_overlap:
                raise ValueError(f"Input latents ({original_total_frames} frames) must be >= latent_overlap ({latent_overlap} frames). Ensure a video latent sequence is connected.")
            
            new_frames_to_process = original_total_frames - latent_overlap
            padding_needed = (per_batch - (new_frames_to_process % per_batch)) % per_batch if new_frames_to_process > 0 else 0
            padded_samples = samples
            if padding_needed > 0:
                padding = samples[-1:].repeat(padding_needed, 1, 1, 1)
                padded_samples = torch.cat([samples, padding], dim=0)

            total_frames = padded_samples.shape[0]
            memory_latents = None
            
            for start_idx in range(0, total_frames, per_batch):
                end_idx = min(start_idx + per_batch, total_frames)
                latent_chunk = padded_samples[start_idx:end_idx].clone()
                if latent_chunk.shape[0] == 0: continue

                chunk_with_context = torch.cat([memory_latents, latent_chunk], dim=0) if memory_latents is not None else latent_chunk
                memory_latents = latent_chunk[-latent_overlap:].clone()
                
                decoded_pixels = vae.decode_tiled(chunk_with_context, tile_x=tile_x//8, tile_y=tile_y//8) if enable_vae_tiling else vae.decode(chunk_with_context)
                
                new_pixels = decoded_pixels[latent_overlap:] if start_idx > 0 else decoded_pixels
                pixel_chunks.append(new_pixels.cpu())

                mm.soft_empty_cache()
                pbar.update(latent_chunk.shape[0])

        else: # Standard Mode (No Overlap)
            for start_idx in range(0, original_total_frames, per_batch):
                end_idx = min(start_idx + per_batch, original_total_frames)
                latent_chunk = samples[start_idx:end_idx]
                if latent_chunk.shape[0] == 0: continue

                decoded_pixels = vae.decode_tiled(latent_chunk, tile_x=tile_x // 8, tile_y=tile_y // 8) if enable_vae_tiling else vae.decode(latent_chunk)
                pixel_chunks.append(decoded_pixels.cpu())
                
                mm.soft_empty_cache()
                pbar.update(latent_chunk.shape[0])

        final_pixels = torch.cat(pixel_chunks, dim=0)
        final_pixels = final_pixels[:original_total_frames]
        
        # This permute now works correctly because final_pixels is guaranteed to be 4D
        final_pixels = final_pixels.permute(0, 2, 3, 1) 
        final_pixels = (final_pixels * 0.5 + 0.5).clamp(0, 1)

        return (final_pixels,)

__all__ = [
    "MiniMaxH3AVLatentConcat",
    "SafeLatentListDecoder",
    "SaveWanVideoLatent",
    "SaveLTXVideoLatent",
    "LoadAndConcatVideoLatentsFromDir",
    "WanDirectoryVideoDecoder",
    "LTXVDirectoryVideoDecoder",
    "Trixope_WanVAEEncodeBatched",
    "Trixope_WanVAEDecodeBatched",
    "Trixope_VAEDecodeBatched_Native",
]
