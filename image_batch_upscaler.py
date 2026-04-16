import torch
import math
from comfy.utils import ProgressBar, common_upscale
import comfy.model_management as model_management
from nodes import MAX_RESOLUTION

class ImageUpscaleToSize:
    """
    This node scales an image to a target size using an upscale model.
    It first downscales the image so that the upscale model's scaling factor
    results in the final desired dimensions. This allows resizing *using* the
    model's algorithm instead of a standard scaler.
    """
    resize_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "images": ("IMAGE",),
                "model_scale_factor": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 64.0, "step": 0.1, "doc": "The multiplier of the upscale model (e.g., 4.0 for a 4x model)."}),
                "final_width": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step": 8, "label": "Final Target Width"}),
                "final_height": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step": 8, "label": "Final Target Height"}),
                "keep_proportion": (["stretch", "crop", "pad"], {"default": "stretch"}),
                "downscale_method": (s.resize_methods, {"default": "lanczos"}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
                "pad_color": ("STRING", {"default": "0, 0, 0", "tooltip": "RGB color for padding, e.g., '0, 0, 0' for black."}),
                "per_batch": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_to_size"
    CATEGORY = "triXope"

    def upscale_to_size(self, upscale_model, images, model_scale_factor, final_width, final_height, keep_proportion, downscale_method, crop_position, pad_color, per_batch):
        if model_scale_factor <= 0:
            raise ValueError("Model Scale Factor must be greater than zero.")

        device = model_management.get_torch_device()
        upscale_model.to(device)

        in_img_batch = images.movedim(-1, 1) # B,C,H,W

        pbar = ProgressBar(in_img_batch.shape[0])
        upscaled_batches = []

        for i in range(0, in_img_batch.shape[0], per_batch):
            sub_batch = in_img_batch[i:i + per_batch]
            _B, _C, H, W = sub_batch.shape

            target_width = final_width if final_width > 0 else round(W * model_scale_factor)
            target_height = final_height if final_height > 0 else round(H * model_scale_factor)

            # Calculate the intermediate dimensions needed before applying the model
            intermediate_width = math.ceil(target_width / model_scale_factor)
            intermediate_height = math.ceil(target_height / model_scale_factor)

            pre_scaled_batch = sub_batch

            if keep_proportion == "stretch":
                pre_scaled_batch = common_upscale(sub_batch, intermediate_width, intermediate_height, downscale_method, "disabled")

            elif keep_proportion == "crop":
                original_aspect = W / H
                target_aspect = target_width / target_height

                if original_aspect > target_aspect:
                    crop_w = int(H * target_aspect)
                    crop_h = H
                else:
                    crop_w = W
                    crop_h = int(W / target_aspect)

                if crop_position == "top": y_start, x_start = 0, (W - crop_w) // 2
                elif crop_position == "bottom": y_start, x_start = H - crop_h, (W - crop_w) // 2
                elif crop_position == "left": y_start, x_start = (H - crop_h) // 2, 0
                elif crop_position == "right": y_start, x_start = (H - crop_h) // 2, W - crop_w
                else: y_start, x_start = (H - crop_h) // 2, (W - crop_w) // 2

                cropped = sub_batch[:, :, y_start:y_start+crop_h, x_start:x_start+crop_w]
                pre_scaled_batch = common_upscale(cropped, intermediate_width, intermediate_height, downscale_method, "disabled")

            elif keep_proportion == "pad":
                ratio = min(intermediate_width / W, intermediate_height / H)
                new_width, new_height = round(W * ratio), round(H * ratio)

                temp_resized = common_upscale(sub_batch, new_width, new_height, downscale_method, "disabled")

                color_vals = [float(x.strip()) / 255.0 for x in pad_color.split(',')]
                padded_batch = torch.zeros((_B, _C, intermediate_height, intermediate_width), dtype=temp_resized.dtype, device=temp_resized.device)
                for c in range(_C):
                    padded_batch[:, c, :, :] = color_vals[c % len(color_vals)]

                pad_top = (intermediate_height - new_height) // 2
                pad_left = (intermediate_width - new_width) // 2

                padded_batch[:, :, pad_top:pad_top+new_height, pad_left:pad_left+new_width] = temp_resized
                pre_scaled_batch = padded_batch
            
            # --- UPSCALE THE PRE-SCALED BATCH ---
            upscaled_sub_batch = upscale_model(pre_scaled_batch.to(device))
            upscaled_batches.append(upscaled_sub_batch.cpu())
            
            pbar.update(upscaled_sub_batch.shape[0])

        upscale_model.cpu()
        
        final_output = torch.cat(upscaled_batches, dim=0).permute(0, 2, 3, 1).cpu()

        return (final_output,)