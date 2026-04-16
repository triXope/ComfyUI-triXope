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
    CATEGORY = "triXope"
    
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
    CATEGORY = "Latent/Video"
    
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
    CATEGORY = "triXope"

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
    CATEGORY = "triXope"

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
    CATEGORY = "triXope"

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