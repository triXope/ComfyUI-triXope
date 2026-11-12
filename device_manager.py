import torch
import logging
import psutil
import comfy.model_management as mm

logger = logging.getLogger("AutoMultiGPU")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[AutoMultiGPU] %(message)s"))
    logger.addHandler(ch)

def get_available_devices():
    """Returns a list of all available devices (e.g. ['cuda:0', 'cuda:1', 'cpu'])."""
    devices = []
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        for i in range(count):
            devices.append(f"cuda:{i}")
    devices.append("cpu")
    return devices

def get_gpu_status():
    """
    Queries real-time GPU statistics (Total VRAM, Free VRAM, Allocated VRAM, Name).
    Returns a dict keyed by device string (e.g. 'cuda:0').
    """
    status = {}
    if not torch.cuda.is_available():
        return status

    for i in range(torch.cuda.device_count()):
        dev_name = f"cuda:{i}"
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            props = torch.cuda.get_device_properties(i)
            allocated_bytes = torch.cuda.memory_allocated(i)
            status[dev_name] = {
                "index": i,
                "name": props.name,
                "total_gb": round(total_bytes / (1024 ** 3), 2),
                "free_gb": round(free_bytes / (1024 ** 3), 2),
                "allocated_gb": round(allocated_bytes / (1024 ** 3), 2),
            }
        except Exception as e:
            logger.warning(f"Error querying {dev_name}: {e}")
    return status

def select_optimal_devices(mode="Auto: UNet GPU0 / Encoders GPU1", manual_unet=None, manual_clip=None, manual_vae=None):
    """
    Determines optimal device mapping for (UNet, CLIP, VAE) based on the chosen strategy.
    
    Strategies:
    - 'Auto: UNet GPU0 / Encoders GPU1': UNet on cuda:0, CLIP/VAE on cuda:1.
    - 'Auto: Balanced Free VRAM': Heaviest component (UNet) goes to the GPU with most free VRAM.
    - 'Auto: Split Video (GPU0 Sampler / GPU1 VAE)': GPU0 for Sampler/UNet, GPU1 dedicated for VAE.
    - 'Manual Override': Uses manual_unet, manual_clip, manual_vae.
    """
    devs = get_available_devices()
    cuda_devs = [d for d in devs if d.startswith("cuda:")]
    num_gpus = len(cuda_devs)

    if num_gpus < 2 or mode == "Manual Override":
        unet_dev = manual_unet if manual_unet in devs else (cuda_devs[0] if cuda_devs else "cpu")
        clip_dev = manual_clip if manual_clip in devs else (cuda_devs[0] if cuda_devs else "cpu")
        vae_dev = manual_vae if manual_vae in devs else (cuda_devs[0] if cuda_devs else "cpu")
        return unet_dev, clip_dev, vae_dev

    gpu_stats = get_gpu_status()

    if mode == "Auto: Balanced Free VRAM" and gpu_stats:
        # Sort GPUs by free memory descending
        sorted_gpus = sorted(gpu_stats.keys(), key=lambda k: gpu_stats[k]["free_gb"], reverse=True)
        primary_gpu = sorted_gpus[0]
        secondary_gpu = sorted_gpus[1] if len(sorted_gpus) > 1 else sorted_gpus[0]
        
        unet_dev = primary_gpu
        clip_dev = secondary_gpu
        vae_dev = secondary_gpu
    elif mode == "Auto: Split Video (GPU0 Sampler / GPU1 VAE)":
        unet_dev = "cuda:0"
        clip_dev = "cuda:0"
        vae_dev = "cuda:1"
    else:
        # Default: "Auto: UNet GPU0 / Encoders GPU1"
        unet_dev = "cuda:0"
        clip_dev = "cuda:1"
        vae_dev = "cuda:1"

    logger.info(
        f"Auto Distribution Strategy '{mode}' applied:\n"
        f"  - MODEL (UNet/DiT)  -> {unet_dev} (Free: {gpu_stats.get(unet_dev, {}).get('free_gb', '?')} GB)\n"
        f"  - CLIP (TextEnc)   -> {clip_dev} (Free: {gpu_stats.get(clip_dev, {}).get('free_gb', '?')} GB)\n"
        f"  - VAE (Decode/Enc) -> {vae_dev} (Free: {gpu_stats.get(vae_dev, {}).get('free_gb', '?')} GB)"
    )

    return unet_dev, clip_dev, vae_dev
