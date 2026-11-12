# ComfyUI Node for Ultimate SD Upscale by Coyote-A: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111

import logging
from contextlib import contextmanager
import torch
import comfy
from usdu_patch import usdu
from usdu_utils import tensor_to_pil, pil_to_tensor
from modules.processing import StableDiffusionProcessing
import modules.shared as shared
from modules.upscaler import UpscalerData

logger = logging.getLogger(__name__)


@contextmanager
def suppress_logging(level=logging.CRITICAL + 1):
    """Context manager to temporarily suppress logging output."""
    root_logger = logging.getLogger()
    old_level = root_logger.getEffectiveLevel()
    root_logger.setLevel(level)
    try:
        yield
    finally:
        root_logger.setLevel(old_level)

MAX_RESOLUTION = 8192
# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": usdu.USDUMode.LINEAR,
    "Chess": usdu.USDUMode.CHESS,
    "None": usdu.USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": usdu.USDUSFMode.NONE,
    "Band Pass": usdu.USDUSFMode.BAND_PASS,
    "Half Tile": usdu.USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": usdu.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


def USDU_base_inputs():
    required = [
        ("image", ("IMAGE", {"tooltip": "The image to upscale."})),
        # Sampling Params
        ("model", ("MODEL", {"tooltip": "The model to use for image-to-image."})),
        ("positive", ("CONDITIONING", {"tooltip": "The positive conditioning for each tile."})),
        ("negative", ("CONDITIONING", {"tooltip": "The negative conditioning for each tile."})),
        ("vae", ("VAE", {"tooltip": "The VAE model to use for tiles."})),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05, "tooltip": "The factor to upscale the image by."})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The seed to use for image-to-image."})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1, "tooltip": "The number of steps to use for each tile."})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "The CFG scale to use for each tile."})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampler to use for each tile."})),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler to use for each tile."})),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for each tile."})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL", {"tooltip": "The upscaler model for upscaling the image."})),
        ("mode_type", (list(MODES.keys()), {"tooltip": "The tiling order to use for the redraw step."})),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of each tile."})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of each tile."})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the mask."})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply between tiles."})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()), {"tooltip": "The seam fix mode to use."})),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for the seam fix."})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the bands used for the Band Pass seam fix mode."})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the seam fix mask."})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply for the seam fix tiles."})),
        # Misc
        ("force_uniform_tiles", ("BOOLEAN", {"default": True, "tooltip": "Force all tiles to be the same as the set tile size, even when tiles could be smaller. This can help prevent the model from working with irregular tile sizes."})),
        ("tiled_decode", ("BOOLEAN", {"default": False, "tooltip": "Whether to use tiled decoding when decoding tiles."})),
        ("batch_size", ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1, "tooltip": "The number of tiles to process in a batch. Higher values can reduce processing time but use more VRAM. Yields different results than individual tiles. Only affects the main redraw step, not the seam fix step."})),
    ]

    optional = []

    return required, optional


def prepare_inputs(required: list, optional: list = None):
    inputs = {}
    if required:
        inputs["required"] = {}
        for name, type in required:
            inputs["required"][name] = type
    if optional:
        inputs["optional"] = {}
        for name, type in optional:
            inputs["optional"][name] = type
    return inputs


def remove_input(inputs: list, input_name: str):
    for i, (n, _) in enumerate(inputs):
        if n == input_name:
            del inputs[i]
            break


def rename_input(inputs: list, old_name: str, new_name: str):
    for i, (n, t) in enumerate(inputs):
        if n == old_name:
            inputs[i] = (new_name, t)
            break


class UltimateSDUpscale:
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final upscaled image.",)
    DESCRIPTION = "Upscales an image and runs image-to-image on tiles from the input image."

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size=1,
                custom_sampler=None, custom_sigmas=None):
        redraw_mode = MODES[mode_type]
        seam_fix_mode = SEAM_FIX_MODES[seam_fix_mode]

        #
        # Set up A1111 patches
        #

        # Upscaler
        # An object that the script works with
        shared.sd_upscalers[0] = UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        shared.batch_as_tensor = image

        logger.debug("UltimateSDUpscale.upscale() using batch_size=%s", batch_size)
        assert batch_size == 1 or force_uniform_tiles, "batch_size greater than 1 requires force_uniform_tiles to be True; all tiles in the batch must be the same size."

        # Processing
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0], model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, upscale_by, force_uniform_tiles, tiled_decode,
            tile_width, tile_height, redraw_mode, seam_fix_mode,
            custom_sampler, custom_sigmas, batch_size,
        )
        logger.debug("StableDiffusionProcessing created with batch_size=%s", sdprocessing.batch_size)

        # Suppress logging to prevent duplicate tqdm progress bars
        with suppress_logging():
            #
            # Running the script
            #
            script = usdu.Script()
            processed = script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                               mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                               seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=redraw_mode,
                               save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                               seams_fix_type=seam_fix_mode, target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=upscale_by)

        # Return the resulting images
        images = [pil_to_tensor(img) for img in shared.batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)

class UltimateSDUpscaleNoUpscale(UltimateSDUpscale):
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        remove_input(required, "upscale_model")
        remove_input(required, "upscale_by")
        rename_input(required, "image", "upscaled_image")
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final refined image.",)
    DESCRIPTION = "Runs image-to-image on tiles from the input image."

    def upscale(self, upscaled_image, model, positive, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size=1):
        upscale_by = 1.0

        logger.debug("UltimateSDUpscaleNoUpscale.upscale() received batch_size=%s", batch_size)

        return super().upscale(upscaled_image, model, positive, negative, vae, upscale_by, seed,
                               steps, cfg, sampler_name, scheduler, denoise, None,
                               mode_type, tile_width, tile_height, mask_blur, tile_padding,
                               seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                               seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size)
    
class UltimateSDUpscaleCustomSample(UltimateSDUpscale):
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        remove_input(required, "upscale_model")
        optional.append(("upscale_model", ("UPSCALE_MODEL", {"tooltip": "The model to use for upscaling the image. If not provided, a simple Lanczos scaling will be used instead."})))
        optional.append(("custom_sampler", ("SAMPLER", {"tooltip": "A custom sampler to use instead of the built-in ComfyUI sampler specified by sampler_name. Only used if both custom_sampler and custom_sigmas are provided."})))
        optional.append(("custom_sigmas", ("SIGMAS", {"tooltip": "A custom noise schedule to use during sampling. Only used if both custom_sampler and custom_sigmas are provided."})))
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final upscaled image.",)
    DESCRIPTION = "Runs image-to-image on tiles from the input image."

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size=1,
                upscale_model=None,
                custom_sampler=None, custom_sigmas=None):
        return super().upscale(image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size,
                custom_sampler, custom_sigmas)

def match_color_mean_std(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calibrates 'source' RGB distribution to match 'target' channel mean and standard deviation."""
    target_mean = target.mean(dim=(1, 2), keepdim=True)
    target_std = target.std(dim=(1, 2), keepdim=True) + 1e-6
    
    source_mean = source.mean(dim=(1, 2), keepdim=True)
    source_std = source.std(dim=(1, 2), keepdim=True) + 1e-6
    
    normalized = (source - source_mean) / source_std
    matched = normalized * target_std + target_mean
    return torch.clamp(matched, 0.0, 1.0)

class UltimateSDUpscaleMiniMaxH3:
    @classmethod
    def INPUT_TYPES(s):
        samplers = getattr(comfy.samplers, "SAMPLERS", getattr(getattr(comfy.samplers, "KSampler", None), "SAMPLERS", []))
        schedulers = getattr(comfy.samplers, "SCHEDULERS", getattr(getattr(comfy.samplers, "KSampler", None), "SCHEDULERS", []))

        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 8.0, "step": 0.05}),
                "tile_width": ("INT", {"default": 1024, "min": 128, "max": 8192, "step": 16}),
                "tile_height": ("INT", {"default": 1024, "min": 128, "max": 8192, "step": 16}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 16}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": 256, "step": 16}),
                "seam_fix_denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": 128, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(
        self, image, model, positive, negative, vae, upscale_by: float,
        tile_width: int, tile_height: int, tile_padding: int, seam_fix_width: int,
        seam_fix_denoise: float, seam_fix_padding: int, seed: int, steps: int,
        cfg: float, sampler_name: str, scheduler: str, denoise: float,
        upscale_model=None
    ):
        batch_size, h, w, c = image.shape
        target_h = (int(h * upscale_by) // 16) * 16
        target_w = (int(w * upscale_by) // 16) * 16
        
        tile_w = (tile_width // 16) * 16
        tile_h = (tile_height // 16) * 16
        padding = (tile_padding // 16) * 16

        # Base image upsampling using model if provided, falling back to bilinear interpolation
        if upscale_model is not None:
            import comfy_extras.nodes_upscale_model
            upscaler = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel()
            resized_images = upscaler.upscale(upscale_model, image)[0]
            if resized_images.shape[1] != target_h or resized_images.shape[2] != target_w:
                resized_images = torch.nn.functional.interpolate(
                    resized_images.permute(0, 3, 1, 2),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False
                ).permute(0, 2, 3, 1)
        else:
            resized_images = torch.nn.functional.interpolate(
                image.permute(0, 3, 1, 2),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False
            ).permute(0, 2, 3, 1)

        resized_images = torch.nn.functional.interpolate(
            image.permute(0, 3, 1, 2),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False
        ).permute(0, 2, 3, 1)

        output_batch = []

        for b in range(batch_size):
            canvas = resized_images[b:b+1].clone()
            weight_canvas = torch.zeros((1, target_h, target_w, 1), device=canvas.device)
            acc_canvas = torch.zeros_like(canvas)

            y_steps = range(0, max(1, target_h - padding), tile_h - padding)
            x_steps = range(0, max(1, target_w - padding), tile_w - padding)

            for y in y_steps:
                for x in x_steps:
                    y2 = min(y + tile_h, target_h)
                    x2 = min(x + tile_w, target_w)
                    y1 = max(0, y2 - tile_h)
                    x1 = max(0, x2 - tile_w)

                    tile = canvas[:, y1:y2, x1:x2, :]
                    curr_h, curr_w = y2 - y1, x2 - x1

                    # 1. Encode 4D tile [1, H, W, C] using ComfyUI's native pixel normalization
                    latent = vae.encode(tile)
                    latent_samples = latent["samples"] if isinstance(latent, dict) else latent

                    # 2. Reshape 4D latents to 5D [B, C, T=1, H_lat, W_lat] for MiniMax H3
                    if latent_samples.dim() == 4:
                        latent_samples = latent_samples.unsqueeze(2)

                    # 3. Wrap video latent stream with dummy audio stream for MiniMax H3
                    if not getattr(latent_samples, "is_nested", False) and not isinstance(latent_samples, (list, tuple)):
                        b_size = latent_samples.shape[0]
                        audio_dummy = latent_samples.new_zeros((b_size, 32, 2, 65))
                        if hasattr(comfy, "nested_tensor") and hasattr(comfy.nested_tensor, "NestedTensor"):
                            latent_samples = comfy.nested_tensor.NestedTensor((latent_samples, audio_dummy))
                        else:
                            latent_samples = (latent_samples, audio_dummy)

                    # 4. Prepare noise & sample tile
                    noise = comfy.sample.prepare_noise(latent_samples, seed)

                    sampled = comfy.sample.sample(
                        model,
                        noise,
                        steps,
                        cfg,
                        sampler_name,
                        scheduler,
                        positive,
                        negative,
                        latent_samples,
                        denoise=denoise,
                        seed=seed,
                    )

                    sampled_samples = sampled["samples"] if isinstance(sampled, dict) else sampled

                    # 5. Unwrap video latent stream
                    if getattr(sampled_samples, "is_nested", False):
                        video_samples = sampled_samples.unbind()[0]
                    elif isinstance(sampled_samples, (list, tuple)):
                        video_samples = sampled_samples[0]
                    else:
                        video_samples = sampled_samples

                    # 6. Decode tile back to image space
                    decoded_tile = vae.decode(video_samples)
                    if isinstance(decoded_tile, dict):
                        decoded_tile = decoded_tile["samples"]

                    # Dynamically reduce 5D decoded tensor back to 4D [B, H, W, C]
                    while decoded_tile.dim() > 4:
                        squeezed = False
                        for dim_idx in range(1, decoded_tile.dim() - 2):
                            if decoded_tile.shape[dim_idx] == 1:
                                decoded_tile = decoded_tile.squeeze(dim_idx)
                                squeezed = True
                                break
                        if not squeezed:
                            decoded_tile = decoded_tile[:, 0]

                    if decoded_tile.dim() == 4:
                        if decoded_tile.shape[1] < decoded_tile.shape[2] and decoded_tile.shape[1] < decoded_tile.shape[3]:
                            decoded_tile = decoded_tile.permute(0, 2, 3, 1)

                    # 7. Force decoded tile to match original input tile color & luminance profile
                    decoded_tile = match_color_mean_std(decoded_tile, tile)

                    mask = torch.ones((1, curr_h, curr_w, 1), device=canvas.device, dtype=torch.float32)
                    if padding > 0:
                        for p in range(padding):
                            alpha = p / float(padding)
                            mask[:, p, :, :] *= alpha
                            mask[:, curr_h - 1 - p, :, :] *= alpha
                            mask[:, :, p, :] *= alpha
                            mask[:, :, curr_w - 1 - p, :] *= alpha

                    acc_canvas[:, y1:y2, x1:x2, :] += decoded_tile * mask
                    weight_canvas[:, y1:y2, x1:x2, :] += mask

            weight_canvas = torch.clamp(weight_canvas, min=1e-5)
            final_img = acc_canvas / weight_canvas
            output_batch.append(final_img)

        return (torch.cat(output_batch, dim=0),)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscale": UltimateSDUpscale,
    "UltimateSDUpscaleNoUpscale": UltimateSDUpscaleNoUpscale,
    "UltimateSDUpscaleCustomSample": UltimateSDUpscaleCustomSample,
    "UltimateSDUpscaleMiniMaxH3": UltimateSDUpscaleMiniMaxH3,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscale": "Ultimate SD Upscale",
    "UltimateSDUpscaleNoUpscale": "Ultimate SD Upscale (No Upscale)",
    "UltimateSDUpscaleCustomSample": "Ultimate SD Upscale (Custom Sample)",
    "UltimateSDUpscaleMiniMaxH3": "Ultimate SD Upscale (MiniMax H3)",
}
