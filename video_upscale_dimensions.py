import os
import torch
import comfy.model_management as model_management
import comfy.utils
import folder_paths
import numpy as np
from PIL import Image
import gc
import time

# Add the missing progress bar initialization function
def init_comfyui_progress():
    """
    Initialize ComfyUI progress bar system for this module.
    This function is called by ComfyUI's progress bar system.
    """
    pass  # No special initialization needed for our progress implementation

class Video_Upscale_To_Dimensions:
    """
    A memory-efficient implementation for upscaling video frames to specific dimensions
    using an upscale model with proper batch processing and memory management.
    """
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic"]
    crop_behaviors = ["stretch", "crop"]
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                    "images": ("IMAGE",),
                    "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                    "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                    "crop_behavior": (s.crop_behaviors,),
                    "upscale_method": (s.upscale_methods,),
                    "device_strategy": (["auto", "load_unload_each_frame", "keep_loaded", "cpu_only"], {"default": "auto"})
                }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_video"
    CATEGORY = "triXope"
    
    # ComfyUI progress bar integration
    def __init__(self):
        self.steps = 0
        self.step = 0
    
    # Important: ComfyUI looks for this method to track progress
    def get_progress_execution(self):
        if self.steps > 0:
            return self.step, self.steps
        return 0, 1
    
    def upscale_video(self, model_name, images, width, height, crop_behavior, upscale_method, device_strategy="auto"):
        """
        Upscale a sequence of images (video frames) efficiently to target dimensions.
        """
        # Load the upscale model
        upscale_model_path = folder_paths.get_full_path("upscale_models", model_name)
        upscale_model = self.load_upscale_model(upscale_model_path)
        
        # Determine the right strategy
        device = model_management.get_torch_device()
        if device_strategy == "auto":
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                reserved_memory = torch.cuda.memory_reserved(0)
                
                if (total_memory - reserved_memory) / total_memory > 0.5:
                    device_strategy = "keep_loaded"
                else:
                    device_strategy = "load_unload_each_frame"
            else:
                device_strategy = "cpu_only"
        
        # Map crop_behavior to ComfyUI's internal crop method
        # "stretch" uses "disabled", "crop" uses "center"
        crop_method = "center" if crop_behavior == "crop" else "disabled"

        # Get dimensions
        num_frames = images.shape[0]
        old_height = images.shape[1]
        old_width = images.shape[2]
        new_width = width
        new_height = height
        
        # Initialize progress tracking
        self.steps = num_frames
        self.step = 0
        
        print(f"Processing video: {num_frames} frames from {old_width}x{old_height} to {new_width}x{new_height} with {device_strategy} strategy")
        
        # Strategy-based upscaling (always batch_size of 1 for simplicity)
        if device_strategy == "cpu_only":
            upscale_model = upscale_model.to("cpu")
            result_frames = self._upscale_on_cpu(upscale_model, images, upscale_method, new_width, new_height, crop_method)
        elif device_strategy == "keep_loaded":
            upscale_model = upscale_model.to(device)
            result_frames = self._upscale_batch_keep_loaded(upscale_model, images, device, upscale_method, new_width, new_height, crop_method)
        else:  # "load_unload_each_frame"
            result_frames = self._upscale_batch_load_unload(upscale_model, images, device, upscale_method, new_width, new_height, crop_method)
        
        # Stack frames back into a single tensor
        return (torch.stack(result_frames),)
    
    def load_upscale_model(self, model_path):
        """Load the upscale model from the given path"""
        from comfy_extras.chainner_models import model_loading
        
        sd = comfy.utils.load_torch_file(model_path)
        upscale_model = model_loading.load_state_dict(sd).eval()
        
        # Free up memory
        del sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return upscale_model
    
    def _upscale_on_cpu(self, upscale_model, images, upscale_method, new_width, new_height, crop_method):
        """Process all frames on CPU to minimize VRAM usage"""
        result_frames = []
        start_time = time.time()
        
        for i in range(images.shape[0]):
            frame = images[i:i+1]
            in_img = frame.movedim(-1, -3)
            
            s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=64, tile_y=64, overlap=8, upscale_amount=upscale_model.scale)
            
            upscaled = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
            samples = upscaled.movedim(-1, 1)
            s = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop=crop_method)
            s = s.movedim(1, -1)
            
            result_frames.append(s[0])
            self.step += 1
            
            elapsed = time.time() - start_time
            eta = (elapsed / self.step * (self.steps - self.step)) if self.step > 0 else 0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            percent = (self.step / self.steps) * 100
            print(f"\r\033[32m|{'█' * int(percent/5)}{' ' * (20-int(percent/5))}| {self.step}/{self.steps} [{percent:.1f}%] - {elapsed_str}<{eta_str}\033[0m", end="", flush=True)
            
            del in_img, s, upscaled, samples
            gc.collect()
        
        print()
        return result_frames
    
    def _upscale_batch_keep_loaded(self, upscale_model, images, device, upscale_method, new_width, new_height, crop_method):
        """Keep model on GPU for entire processing (highest VRAM usage but fastest)"""
        result_frames = []
        start_time = time.time()
        
        for i in range(images.shape[0]):
            frame = images[i:i+1]
            in_img = frame.movedim(-1, -3).to(device)
            
            s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=128, tile_y=128, overlap=8, upscale_amount=upscale_model.scale)
            
            upscaled = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
            samples = upscaled.movedim(-1, 1)
            s = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop=crop_method)
            s = s.movedim(1, -1).cpu()
            
            result_frames.append(s[0])
            self.step += 1
            
            elapsed = time.time() - start_time
            eta = (elapsed / self.step * (self.steps - self.step)) if self.step > 0 else 0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            percent = (self.step / self.steps) * 100
            print(f"\r\033[32m|{'█' * int(percent/5)}{' ' * (20-int(percent/5))}| {self.step}/{self.steps} [{percent:.1f}%] - {elapsed_str}<{eta_str}\033[0m", end="", flush=True)
            
            del in_img, s, upscaled, samples
            torch.cuda.empty_cache()
        
        print()
        return result_frames
    
    def _upscale_batch_load_unload(self, upscale_model, images, device, upscale_method, new_width, new_height, crop_method):
        """Load model to GPU for each frame batch, then move back to CPU (balanced approach)"""
        result_frames = []
        start_time = time.time()
        
        for i in range(images.shape[0]):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            upscale_model = upscale_model.to(device)
            
            frame = images[i:i+1]
            in_img = frame.movedim(-1, -3).to(device)
            
            s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=96, tile_y=96, overlap=8, upscale_amount=upscale_model.scale)
            
            upscaled = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
            samples = upscaled.movedim(-1, 1)
            s = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop=crop_method)
            s = s.movedim(1, -1).cpu()
            
            result_frames.append(s[0])
            self.step += 1
            
            elapsed = time.time() - start_time
            eta = (elapsed / self.step * (self.steps - self.step)) if self.step > 0 else 0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            percent = (self.step / self.steps) * 100
            print(f"\r\033[32m|{'█' * int(percent/5)}{' ' * (20-int(percent/5))}| {self.step}/{self.steps} [{percent:.1f}%] - {elapsed_str}<{eta_str}\033[0m", end="", flush=True)

            del in_img, s, upscaled, samples
            upscale_model = upscale_model.to("cpu")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print()
        return result_frames