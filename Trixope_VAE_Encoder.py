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
    CATEGORY = "triXope"

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