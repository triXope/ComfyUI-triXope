import torch

class ImageBatchSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1}),
                "inverse_selection": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "filter_batch"
    CATEGORY = "triXope"

    def filter_batch(self, images, select_every_nth, skip_first_images, inverse_selection):
        # 1. Skip the initial frames
        total_images = images.shape[0]
        if skip_first_images >= total_images:
            # Return a single black frame or empty if skip is too large
            # Better to return the last frame to avoid crashing most workflows
            return (images[-1:],)

        remaining_images = images[skip_first_images:]
        num_remaining = remaining_images.shape[0]
        
        # 2. Determine indices
        indices = list(range(0, num_remaining, select_every_nth))
        
        if inverse_selection:
            # Create a set of all indices and remove the 'every nth' ones
            all_indices = set(range(num_remaining))
            selected_set = set(indices)
            final_indices = sorted(list(all_indices - selected_set))
        else:
            final_indices = indices

        # 3. Guard against empty selection
        if not final_indices:
            return (images[-1:],)

        # 4. Extract the selected images
        output_batch = remaining_images[final_indices]
        
        return (output_batch,)