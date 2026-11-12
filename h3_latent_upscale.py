"""Learned 3D latent upscaling for MiniMax H3 video latents.

Ported from the standalone *Minimax H3 Latent Upscaler (3D)* node so the pack's
``UltimateSDUpscaleMiniMaxH3`` can run its ``latent`` upscale mode in-process.

* Checkpoints live in ``ComfyUI/models/latent_upscale_models`` (registered here,
  so the folder is created/listed like any other model folder).
* The network is a pure 3D-conv resizer that works on the 24-channel H3 video
  latent and preserves the batch/channel/temporal axes exactly.
* The H3 3D VAE compresses both spatial axes 16x, so
  ``latent size * 16 == pixel size`` and target pixel dimensions must stay on a
  16-pixel grid.
* The checkpoint's own latent normalization (mean/std, from the training code)
  is applied around the inference, exactly once, inside :func:`upscale_h3_latent`.

The module only exposes plain helpers; it registers no ComfyUI node.
"""

import glob
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths

LATENT_UPSCALE_FOLDER = "latent_upscale_models"
# Spatial compression factor of the Minimax H3 3D VAE (16x).
VAE_DOWNSAMPLE = 16

if LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, LATENT_UPSCALE_FOLDER),
    )

# Minimax H3 latent normalization stats (24 channels, from the training code).
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]

_PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
_MODEL_CACHE = {}


# ==========================================
# 3D network components (identical to training code)
# ==========================================
def normalization(channels):
    return nn.GroupNorm(32, channels)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h


class LatentResizer3D(nn.Module):
    """Pure-3D backbone: resample the spatial axes, keep B/C/T untouched."""

    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if tuple(size) == tuple(x.shape[-3:]):
            return x

        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb3D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)

        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb3D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        return self.conv_out(x)


# ==========================================
# Model discovery and loading
# ==========================================
def get_models_dir():
    paths = folder_paths.get_folder_paths(LATENT_UPSCALE_FOLDER)
    if paths:
        return paths[0]
    return os.path.join(folder_paths.models_dir, LATENT_UPSCALE_FOLDER)


def latent_upscale_model_names():
    """Checkpoint names (relative to models/latent_upscale_models) for the dropdown."""
    model_dir = get_models_dir()
    files = []
    for ext in ("*.pth", "*.pt", "*.ckpt", "*.safetensors"):
        files.extend(glob.glob(os.path.join(model_dir, "**", ext), recursive=True))
    return sorted(os.path.relpath(f, model_dir).replace("\\", "/") for f in files)


def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


def _convert_state_tensor(tensor, dtype):
    if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.dtype != dtype:
        return tensor.to(dtype=dtype)
    return tensor


def _extract_upscaler_sd(sd):
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd


def _load_raw_sd(path, device, dtype):
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(path, framework="pt", device=str(device)) as f:
            keys = list(f.keys())
            has_prefix = any(k.startswith("upscaler.") for k in keys)
            sd = {}
            for k in keys:
                if has_prefix and not k.startswith("upscaler."):
                    continue
                out_key = k[len("upscaler."):] if has_prefix else k
                sd[out_key] = _convert_state_tensor(f.get_tensor(k), dtype)
        return sd

    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    sd = _extract_upscaler_sd(sd)
    return {k: _convert_state_tensor(v, dtype) for k, v in sd.items()}


def _detect_arch(sd):
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
        "dropout": 0.1, "temporal_every": 2, "temporal_kernel": 5,
    }
    conv_key = "conv_in.weight"
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids, out_ids = set(), set()
    temporal_keys = []
    for k in sd.keys():
        m = re.match(r"in_blocks\.(\d+)\.in_layers\.", k)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r"out_blocks\.(\d+)\.in_layers\.", k)
        if m:
            out_ids.add(int(m.group(1)))
        if k.endswith("dwconv.weight"):
            temporal_keys.append(k)

    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)

    if temporal_keys:
        cfg["temporal_every"] = 2
        cfg["temporal_kernel"] = sd[temporal_keys[0]].shape[2]
    else:
        cfg["temporal_every"] = 0
    return cfg


def load_model(model_name, device, precision):
    cache_key = f"{model_name}::{device}::{precision}"
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        # A cached CUDA model may have been offloaded after a previous run;
        # moving an already-correct model is a no-op in PyTorch.
        return cached.to(device)

    path = os.path.join(get_models_dir(), model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"H3 latent upscale model not found: {path} "
            f"(place checkpoints in {get_models_dir()})"
        )

    dtype = _PRECISION_DTYPES.get(precision, torch.float32)
    sd = _load_raw_sd(path, device, dtype)
    cfg = _detect_arch(sd)

    # Meta construction avoids a second full FP32 CPU model before CUDA inference.
    with torch.device("meta"):
        model = LatentResizer3D(
            in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"],
            out_blocks=cfg["out_blocks"], channels=cfg["channels"],
            dropout=cfg["dropout"], temporal_every=cfg["temporal_every"],
            temporal_kernel=cfg["temporal_kernel"],
        )
    model.load_state_dict(sd, strict=True, assign=True)
    model.eval().requires_grad_(False)
    del sd

    _MODEL_CACHE[cache_key] = model
    print(f"[H3 Upscale] loaded latent upscale model: {model_name} | "
          f"{sum(p.numel() for p in model.parameters()):,} params | "
          f"temporal={'on' if cfg['temporal_every'] > 0 else 'off'} | "
          f"precision={precision} | device={device}")
    return model


def upscale_h3_latent(samples, *, model_name, target_h, target_w,
                      device=None, precision=None, offload_after_upscale=False):
    """Learned spatial resize of an H3 latent to an exact latent H/W.

    ``samples`` must be ``[B, 24, T, H, W]``. Only B/C/T are preserved; the two
    spatial axes are resampled to ``target_h``/``target_w`` latent pixels and may
    only grow. The checkpoint's latent normalization is owned here, so callers
    pass the raw H3 latent exactly as the sampler produced it.
    """
    if not isinstance(samples, torch.Tensor):
        raise TypeError("H3 latent upscale expects a torch.Tensor")
    if samples.dim() != 5:
        raise ValueError(
            f"H3 latent upscale expects BxCxTxHxW, got {tuple(samples.shape)}"
        )
    if samples.shape[1] != len(LATENTS_MEAN):
        raise ValueError(
            f"H3 latent upscale expects {len(LATENTS_MEAN)} latent channels, "
            f"got {samples.shape[1]}"
        )
    if not samples.is_floating_point():
        raise TypeError("H3 latent upscale expects a floating-point latent")
    if not model_name or model_name == "None":
        raise ValueError(
            "Select a latent upscale model (models/latent_upscale_models)"
        )

    src_h, src_w = int(samples.shape[-2]), int(samples.shape[-1])
    target_h, target_w = int(target_h), int(target_w)
    if target_h < src_h or target_w < src_w:
        raise ValueError(
            "H3 latent upscale only supports upscaling: "
            f"target {target_w}x{target_h} is smaller than latent {src_w}x{src_h}"
        )
    if target_h == src_h and target_w == src_w:
        return samples.clone()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    run_device = torch.device(device)
    if run_device.type != "cuda":
        # fp16/bf16 3D convolutions are unreliable (and slow) on CPU.
        precision = "fp32"
    elif precision is None:
        precision = "fp16"
    if precision not in _PRECISION_DTYPES:
        raise ValueError(f"Unsupported latent upscaler precision: {precision!r}")
    compute_dtype = _PRECISION_DTYPES[precision]

    temporal = int(samples.shape[2])
    effective_scale = ((target_h / src_h) + (target_w / src_w)) / 2.0

    work = samples.to(device=run_device, dtype=compute_dtype, copy=True)
    model = load_model(model_name, run_device, precision)
    norm_mean, norm_std = _make_norm_tensors(run_device, compute_dtype)
    with torch.inference_mode():
        work.sub_(norm_mean).div_(norm_std)
        output = model(
            work,
            scale=effective_scale,
            target_size=(temporal, target_h, target_w),
        )
        del work
        output.mul_(norm_std).add_(norm_mean)

    expected = (
        int(samples.shape[0]), int(samples.shape[1]), temporal, target_h, target_w,
    )
    if tuple(output.shape) != expected:
        raise RuntimeError(
            f"H3 latent upscaler returned {tuple(output.shape)}; expected {expected}"
        )
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("H3 latent upscaler returned NaN or Inf values")

    output = output.to(device=samples.device, dtype=samples.dtype)
    if run_device.type == "cuda" and offload_after_upscale:
        model.to("cpu")
        torch.cuda.empty_cache()
    return output


__all__ = [
    "LATENT_UPSCALE_FOLDER",
    "VAE_DOWNSAMPLE",
    "get_models_dir",
    "latent_upscale_model_names",
    "load_model",
    "upscale_h3_latent",
]
