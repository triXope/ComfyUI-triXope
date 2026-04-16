import os
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

class ToggleFolderSaver:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "folder_path": ("STRING", {"default": "C:/ComfyUI_Output/", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "custom_save"}),
            },
            # Hidden inputs so the node can embed the workflow metadata into the saved PNG
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    # By returning the image, this acts as a pass-through node
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "triXope"

    def save_images(self, images, enabled, folder_path, filename_prefix, prompt=None, extra_pnginfo=None):
        # 1. If disabled, just pass the images through to the next node without saving
        if not enabled:
            return (images,)

        # 2. Ensure the output directory exists
        output_dir = os.path.normpath(folder_path)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                print(f"[ToggleFolderSaver] Failed to create directory {output_dir}: {e}")
                return (images,)

        # 3. Process and save each image in the batch
        for (batch_number, image) in enumerate(images):
            # Convert the ComfyUI tensor to a PIL Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Setup metadata so your ComfyUI workflow is saved inside the PNG file
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            # Find the next available filename by incrementing a counter
            counter = 1
            while True:
                file_name = f"{filename_prefix}_{counter:05d}.png"
                full_path = os.path.join(output_dir, file_name)
                if not os.path.exists(full_path):
                    break
                counter += 1

            # Save the image
            try:
                img.save(full_path, pnginfo=metadata, compress_level=4)
                print(f"[ToggleFolderSaver] Saved image to: {full_path}")
            except Exception as e:
                print(f"[ToggleFolderSaver] Error saving image: {e}")

        # Pass the images through for further use in the workflow
        return (images,)