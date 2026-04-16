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
    CATEGORY = "triXope"

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