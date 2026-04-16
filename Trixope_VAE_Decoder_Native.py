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
    CATEGORY = "triXope"

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