"""triXope ComfyUI nodes — MiniMax H3 and LTXV film generation pipelines.

Every node class in this module is registered in the pack's top-level
``__init__.py`` and appears in the ComfyUI node menu under ``triXope/film``.
The module also holds the private helpers those nodes need.
"""

# --- moved from Film_Auteur_H3.py -------------------------------------------
# Combined MiniMax H3 conditioner and sampler node.
# This node composes the existing H3 FL2VA/Ref2VA conditioner and sliding
# context sampler in sequence, while exposing their controls and outputs from one
# node. The original nodes remain available as independent backups.
# 
import math
import re

import torch
import nodes
import node_helpers
import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_minimax_h3 import (
    _empty_av_latent,
    _resize,
    _encode_ref_audio,
    adapt_canvas,
    CANVAS_MULTIPLE,
    REF_IMAGE_SHORT_EDGE,
    FPS,
)

from .core import (
    AUDIO_LATENT_FPS,
    explicit_transition_mode,
    frames_to_audio_latents,
    parse_timestamp,
    parse_timestamped_beats,
    plan_windows,
    transition_for_overlap,
    transition_offset_tokens,
    remap_keyframes_for_window,
    requested_frames_to_latents,
    select_beats_for_latent_window,
    video_frames_for_latents,
)


_BACKGROUND_REMOVAL_MODELS = {}


def _remove_reference_background(image):
    """Use embedded alpha or BiRefNet to composite an H3 reference onto black."""
    rgb = image[..., :3]
    if image.shape[-1] == 4 and bool(torch.any(image[..., 3:4] < 1.0 - 1e-6)):
        alpha = image[..., 3:4].clamp(0.0, 1.0)
    else:
        import folder_paths
        from comfy.bg_removal_model import load as load_background_removal_model

        available = folder_paths.get_filename_list("background_removal")
        birefnet_models = sorted(name for name in available if "birefnet" in name.lower())
        if not birefnet_models:
            raise RuntimeError(
                "ref_alpha='transparent' needs ComfyUI's BiRefNet model. Install "
                "birefnet.safetensors in ComfyUI/models/background_removal/ and retry."
            )
        model_name = birefnet_models[0]
        model = _BACKGROUND_REMOVAL_MODELS.get(model_name)
        if model is None:
            model_path = folder_paths.get_full_path_or_raise("background_removal", model_name)
            model = load_background_removal_model(model_path)
            if model is None:
                raise RuntimeError(f"ComfyUI could not load background-removal model {model_name!r}")
            _BACKGROUND_REMOVAL_MODELS[model_name] = model
        alpha = model.encode_image(rgb).unsqueeze(-1).clamp(0.0, 1.0)

    # H3's tokenizer and VAE accept RGB only, so remove the background by
    # compositing the foreground onto black before either reference path.
    return rgb * alpha.to(device=rgb.device, dtype=rgb.dtype)


_TIMED_SHOT = re.compile(
    r"\[\s*(?P<start>\d{1,3}:\d{2}(?::\d{2})?(?:\.\d+)?)\s*[-–]\s*"
    r"(?P<end>\d{1,3}:\d{2}(?::\d{2})?(?:\.\d+)?)\s*\]"
)


def _prompt_runtime_seconds(prompt):
    """Read the last end time from generated timed multi-shot prompt content."""
    def seconds(timestamp):
        fields = timestamp.split(":")
        if len(fields) == 2:
            hours, minutes, secs = 0, int(fields[0]), float(fields[1])
        else:
            hours, minutes, secs = int(fields[0]), int(fields[1]), float(fields[2])
        return hours * 3600 + minutes * 60 + secs

    ends = [seconds(match.group("end")) for match in _TIMED_SHOT.finditer(prompt or "")]
    return max(ends, default=0.0)


class H3FL2VAandRef2VA(io.ComfyNode):
    """
    Union of MiniMaxH3ImageToVideo (fl2va keyframes) and
    MiniMaxH3ReferenceToVideo (ref2va references) inputs, in one node.

    - keyframes: a batch of N images anchored at real frame positions, spread
      evenly across the clip (image 1 = first frame, image N = last frame,
      the rest evenly spaced in between). Added to the conditioning as
      `minimax_keyframes`, same mechanism as MiniMaxH3ImageToVideo /
      MiniMaxH3AddGuide.
    - ref_image_batch / ref_videos / ref_video_audios / ref_audios behave
      exactly like MiniMaxH3ReferenceToVideo (identity/motion/voice steering, no
      fixed frame position, added to the conditioning as `minimax_refs`).
    - Both mechanisms can be used together or independently.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3FL2VAandRef2VA",
            display_name="Δ H3 Conditioner",
            description=(
                "Combined fl2va + ref2va conditioning for MiniMax H3. "
                "Supports N evenly-spaced keyframes (first..last) AND <Picture i> / <Video k> / "
                "<Audio j> references in the same conditioning payload."
            ),
            category="triXope/film",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input(
                    "length", default=124, min=5, max=1000008, step=17,
                    tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid "
                            "(124 = ~5s; trained range is ~124-362, longer is untested)",
                ),
                io.Combo.Input("mode", options=["text", "reference", "keyframes", "all"], default="reference",
                               tooltip="text: prompt only; reference: text plus reference inputs, no keyframes; keyframes: text plus keyframes, no reference inputs; all: keyframes and reference inputs together, so a pinned shot keeps its identity references."),

                # --- keyframes (generalised fl2va) ---
                io.Image.Input(
                    "keyframes", optional=True,
                    tooltip="Batch of keyframe images, evenly distributed across the clip. "
                            "Image 1 = first frame, last image = last frame, the rest are "
                            "spaced evenly in between (3 images -> start/middle/end; "
                            "4 images in 9s -> 0s, 3s, 6s, 9s). A single image is the first frame.",
                ),

                # --- from MiniMaxH3ReferenceToVideo (ref2va) ---
                io.Combo.Input(
                    "ref_image_size", options=["match", "max", "mixed"], default="mixed",
                    tooltip="Reference image sizing. 'match' scales each ref (down only, "
                            "keeping aspect) to the generation's pixel area; 'max' uses the "
                            "reference pipeline's 2048px short edge for best identity "
                            "fidelity; 'mixed' keeps character sheets at 'max' and everything "
                            "else at 'match'. Reference tokens ride through every sampling "
                            "step, so 'max' can be several times slower.",
                ),
                io.Combo.Input(
                    "ref_alpha", options=["off", "white", "black", "gray", "transparent"], default="off",
                    tooltip="off applies no alpha compositing or background removal; H3 still aligns image size and VAE-encodes it, so this is not pixel pass-through and alpha is not preserved as alpha. 'transparent' uses input alpha or removes an opaque background with ComfyUI BiRefNet, then composites onto black. Opaque images need a model in models/background_removal.",
                ),
                io.Image.Input(
                    "ref_image_batch", optional=True,
                    tooltip="Single batch of reference images. Frame 1 becomes ref_image_1, "
                            "Frame 2 becomes ref_image_2, etc.",
                ),
                io.Autogrow.Input(
                    "ref_videos", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input(
                            "ref_video", tooltip="Reference video frames at 24 fps (2-15s)"
                        ),
                        names=[f"ref_video_{i}" for i in range(1, 5)], # 1 through 4
                        min=0,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Audio.Input(
                            "ref_video_audio",
                            tooltip="Soundtrack of the same-numbered reference video",
                        ),
                        names=[f"ref_video_audio_{i}" for i in range(1, 5)], # 1 through 4
                        min=0,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Audio.Input("ref_audio", tooltip="Standalone reference audio"),
                        names=[f"ref_audio_{i}" for i in range(1, 5)], # 1 through 4
                        min=0,
                    ),
                ),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(
        cls, clip, vae, audio_vae, prompt, width, height, length,
        mode="reference", keyframes=None,
        ref_image_size="match",
        ref_alpha="off",
        ref_image_batch=None,
        ref_videos=None, ref_video_audios=None, ref_audios=None,
        cross_shot_video_reference=None,
        cross_shot_video_audio=None,
    ) -> io.NodeOutput:
        if mode not in {"text", "reference", "keyframes", "all"}:
            raise ValueError("mode must be text, reference, keyframes, or all")
        if mode == "text":
            keyframes = None
            ref_image_batch = None
            ref_videos = {}
            ref_video_audios = {}
            ref_audios = {}
        elif mode == "reference":
            keyframes = None
        elif mode == "keyframes":
            ref_image_batch = None
            ref_videos = {}
            ref_video_audios = {}
            ref_audios = {}
        # "all" keeps both: a pinned shot still needs its identity references.

        if cross_shot_video_reference is not None and mode in {"reference", "all"}:
            ref_videos = dict(ref_videos or {})
            ref_videos["ref_video_cross_shot"] = cross_shot_video_reference

        runtime_seconds = _prompt_runtime_seconds(prompt)
        timed_length = math.ceil(runtime_seconds * FPS) if runtime_seconds else length
        effective_length = max(length, timed_length)
        if effective_length > length:
            print(f"[H3FL2VAandRef2VA] Extending latent to {runtime_seconds:.3f}s ({effective_length} requested frames) from timed prompt.")
        latent, frame_count = _empty_av_latent(width, height, effective_length)

        # ---------- keyframes: N images evenly spread over the clip ----------
        images = []
        kf_entries = []
        if keyframes is not None and keyframes.shape[0] > 0:
            n_kf = keyframes.shape[0]
            for i in range(n_kf):
                idx = 0 if n_kf == 1 else round(i * (frame_count - 1) / (n_kf - 1))
                # first image: geometry anchor (plain stretch to canvas);
                # the rest: aspect-preserving cover-crop (same as original last_frame)
                img = _resize(keyframes[i:i + 1], width, height, "disabled" if i == 0 else "center")
                images.append(img)
                kf_entries.append({"resolved_frame_index": idx, "image": img})

        # ---------- ref2va: images / videos / audio references ----------
        ref_items = []   # for the tokenizer presentation, in request order
        ref_blocks = []  # for the DiT payload, same order

        # Unpack reference images from the image batch
        raw_ref_images = []

        if ref_image_batch is not None:
            for i in range(ref_image_batch.shape[0]):
                raw_ref_images.append(ref_image_batch[i:i + 1])

        # --- diagnostics: what actually arrived as references ---
        print(f"[H3FL2VAandRef2VA] keyframes={0 if keyframes is None else keyframes.shape[0]} "
              f"ref_images={len(raw_ref_images)} ref_image_size={ref_image_size}")
        _warn_dangling_references(
            prompt, len(raw_ref_images),
            len([name for name in (ref_videos or {}) if name != "ref_video_cross_shot"]),
        )
        for _i, _img in enumerate(raw_ref_images, 1):
            print(f"[H3FL2VAandRef2VA]   ref {_i}: shape={tuple(_img.shape)} "
                  f"min={float(_img.min()):.3f} max={float(_img.max()):.3f} mean={float(_img.mean()):.3f}")

        # "off" intentionally leaves each input tensor untouched here; the
        # normal H3 size alignment and VAE encoding still happen below.
        if ref_alpha == "transparent":
            raw_ref_images = [_remove_reference_background(img) for img in raw_ref_images]
        elif ref_alpha != "off":
            bg_val = {"white": 1.0, "black": 0.0, "gray": 0.5}[ref_alpha]
            flat = []
            for img in raw_ref_images:
                if img.shape[-1] == 4:
                    a = img[..., 3:4]
                    img = img[..., :3] * a + bg_val * (1.0 - a)
                flat.append(img)
            raw_ref_images = flat

        # 'mixed' keeps faces at full fidelity and lets props, sets and the palette chart
        # ride at the cheaper match size; the prompt itself says which pictures are sheets.
        character_pictures = (_character_picture_numbers(prompt)
                              if ref_image_size == "mixed" else set())
        for index, img in enumerate(raw_ref_images, 1):
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match" or (ref_image_size == "mixed" and index not in character_pictures):
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img, tw, th, "disabled")
            z = vae.encode(resized)
            # In off mode, let the vision-token path see the original reference
            # tensor; keep the aligned copy only for H3's required VAE latent.
            tokenizer_image = img if ref_alpha == "off" else resized
            ref_items.append({"type": "image", "data": tokenizer_image})
            ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

        ref_video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
                ref_items.append({"type": "audio"})
            elif name == "ref_video_cross_shot" and cross_shot_video_audio is not None:
                # Voice continuity: the previous shot's own audio rides with its visual tail,
                # so the next shot keeps the same voice, accent and room tone.
                audio_latent = cross_shot_video_audio
                ref_audio_t = int(audio_latent.shape[-1])
                ref_items.append({"type": "audio"})
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({
                "type": "video", "data": qwen_frames,
                "timestamps": [i / 2.0 for i in range(len(sample_idx))],
            })
            ref_blocks.append({
                "kind": "video_audio" if ref_audio_t else "video",
                "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent,
            })

        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        # ---------- tokenize + encode (keyframe images and refs at once) ----------
        tokens = clip.tokenize(prompt, images=images, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        payload = {}
        if kf_entries:
            for kf in kf_entries:
                kf["latent"] = vae.encode(kf.pop("image"))
            payload["minimax_keyframes"] = kf_entries
        if ref_blocks:
            payload["minimax_refs"] = ref_blocks
        if payload:
            cond = node_helpers.conditioning_set_values(cond, payload)

        return io.NodeOutput(cond, latent)


class PositiveOnlyGuider(comfy.samplers.CFGGuider):
    """Match H3's native positive-only BasicGuider sampling path."""

    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})


def _is_h3_latent(samples):
    return (
        getattr(samples, "is_nested", False)
        and len(samples.tensors) == 2
        and samples.tensors[0].ndim == 5
        and samples.tensors[0].shape[1] == 24
        and samples.tensors[1].ndim == 4
        and tuple(samples.tensors[1].shape[1:3]) == (32, 2)
    )


def _keyframe_duration_frames(keyframe):
    """Measure an H3 guide payload in target-frame units for overlap routing."""
    durations = []
    video = keyframe.get("latent")
    if video is not None:
        if video.ndim != 5 or video.shape[1] != 24:
            raise ValueError("H3 video guide latent must have shape [B,24,T,H,W]")
        durations.append(video_frames_for_latents(int(video.shape[2])))
    audio = keyframe.get("audio_latent")
    if audio is not None:
        if audio.ndim != 4 or tuple(audio.shape[1:3]) != (32, 2):
            raise ValueError("H3 audio guide latent must have shape [B,32,2,T]")
        audio_frames = (int(audio.shape[-1]) * FPS + AUDIO_LATENT_FPS - 1) // AUDIO_LATENT_FPS
        durations.append(audio_frames)
    if not durations:
        raise ValueError("H3 keyframe guide must contain a video latent and/or audio latent")
    return max(durations)


def _conditioning_for_window(conditioning, start_frame, end_frame):
    """Return a conditioning copy with guides selected and rebased to a chunk."""
    output = []
    for item in conditioning:
        copied = list(item)
        if len(copied) > 1 and isinstance(copied[1], dict):
            metadata = copied[1]
            guides = metadata.get("minimax_keyframes")
            if guides:
                if not isinstance(guides, (list, tuple)):
                    raise ValueError("minimax_keyframes must be a list of H3 guide records")
                with_durations = [
                    (guide, _keyframe_duration_frames(guide)) for guide in guides
                ]
                local_keyframes = remap_keyframes_for_window(
                    with_durations, start_frame, end_frame
                )
                local_keyframes = _crop_audio_guides(local_keyframes, end_frame - start_frame)
                copied[1] = metadata.copy()
                copied[1]["minimax_keyframes"] = local_keyframes
        output.append(copied)
    return output


def _encode_chunk_conditioning(clip, base_conditioning, chunk_prompt):
    """Re-encode one prompt with the existing H3 references, preserving ref metadata."""
    if len(base_conditioning) != 1:
        raise ValueError("Timestamped mode requires one unscheduled base H3 conditioning entry")
    base_embedding, base_metadata = base_conditioning[0]
    # H3 reference payload is model-level metadata. The embeddings must be
    # freshly encoded because appending old/new token tags can stale pooled and
    # modality-specific conditioning. The reference latents remain unchanged.
    tokens = clip.tokenize(chunk_prompt)
    encoded = clip.encode_from_tokens_scheduled(tokens)
    if len(encoded) != 1:
        raise ValueError("Chunk prompt produced scheduled encodings; disable CLIP hook scheduling")
    chunk_embedding, chunk_metadata = encoded[0]
    if chunk_embedding.ndim != 3 or base_embedding.ndim != 3:
        raise ValueError("Expected H3 conditioning embeddings with shape [batch, tokens, channels]")
    if chunk_embedding.shape[0] != base_embedding.shape[0] or chunk_embedding.shape[2] != base_embedding.shape[2]:
        raise ValueError("Base and chunk H3 conditioning embeddings have incompatible shapes")
    chunk_tags = chunk_metadata.get("minimax_token_tags")
    base_tags = base_metadata.get("minimax_token_tags")
    if chunk_tags is None or base_tags is None:
        raise ValueError("Missing MiniMax H3 modality token tags in conditioning metadata")
    if base_tags.numel() != base_embedding.shape[1] or chunk_tags.numel() != chunk_embedding.shape[1]:
        raise ValueError("MiniMax H3 token tags must align one-to-one with text embedding tokens")
    # Preserve reference/keyframe and other H3 metadata from the base prompt.
    # Only the per-token modality tags need extending to match concatenated text.
    combined_metadata = base_metadata.copy()
    combined_metadata["minimax_token_tags"] = torch.cat(
        (base_tags.reshape(-1), chunk_tags.reshape(-1))
    ).to(device=chunk_embedding.device, dtype=torch.long)
    combined_embedding = torch.cat(
        (base_embedding.to(chunk_embedding.device), chunk_embedding), dim=1
    )
    if combined_embedding.shape[1] != combined_metadata["minimax_token_tags"].numel():
        raise ValueError("Combined H3 token tags do not match the embedding sequence length")
    return [[combined_embedding, combined_metadata]]


def _slice_window(samples, start, end):
    """Slice H3 video and time-matched audio on their separate temporal axes."""
    video, audio = samples.unbind()
    frame_start = video_frames_for_latents(start)
    frame_end = video_frames_for_latents(end)
    audio_start = frames_to_audio_latents(frame_start)
    # Match H3's local chunk audio length (round(local_duration * 40 Hz));
    # converting absolute endpoints separately can differ by one latent frame.
    audio_length = frames_to_audio_latents(frame_end - frame_start)
    audio_end = min(audio_start + audio_length, audio.shape[-1])
    if end == video.shape[2]:
        audio_end = audio.shape[-1]
    if audio_start >= audio_end:
        raise ValueError("Computed an empty H3 audio window; the latent streams do not align")
    chunk = comfy.nested_tensor.NestedTensor((
        video[:, :, start:end].contiguous(),
        audio[..., audio_start:audio_end].contiguous(),
    ))
    return chunk, (audio_start, audio_end)


def _tail_video_context(samples, requested_frames):
    """Extract an H3-aligned video tail for the next shot's visual context."""
    if not _is_h3_latent(samples):
        raise ValueError("Cross-shot context requires a sampled H3 AV latent")
    video, _audio = samples.unbind()
    context_tokens = min(requested_frames_to_latents(requested_frames), video.shape[2])
    return video[:, :, -context_tokens:].contiguous()


def _tail_audio_context(samples, requested_frames):
    """The audio latent that belongs to the same prior-shot tail as _tail_video_context."""
    if not _is_h3_latent(samples):
        raise ValueError("Cross-shot voice context requires a sampled H3 AV latent")
    video, audio = samples.unbind()
    context_tokens = min(requested_frames_to_latents(requested_frames), video.shape[2])
    start_frame = video_frames_for_latents(video.shape[2] - context_tokens)
    audio_start = frames_to_audio_latents(start_frame)
    if audio_start >= audio.shape[-1]:
        return None
    return audio[..., audio_start:].contiguous()


def _tail_video_reference_frames(samples, requested_frames, vae):
    """Decode a prior-shot H3 tail into pixels for H3 video-reference conditioning."""
    video_context = _tail_video_context(samples, requested_frames)
    decoded = vae.decode(video_context)
    if decoded.ndim != 5:
        raise ValueError("H3 video VAE must decode a video reference as a 5D tensor")
    if decoded.shape[-1] in (3, 4):
        frames = decoded[0]
    elif decoded.shape[1] in (3, 4):
        frames = decoded[0].permute(1, 2, 3, 0)
    else:
        raise ValueError("Decoded H3 video reference must have an RGB channel dimension")
    if frames.shape[-1] < 3:
        raise ValueError("Decoded H3 video reference must contain RGB pixels")
    return frames[..., :3].contiguous()


def _conditioning_with_context(conditioning, video_context, audio_context):
    """Append prior generated context as a local-zero H3 keyframe guide."""
    guide = {
        "resolved_frame_index": 0,
        "latent": video_context,
        "audio_latent": audio_context,
    }
    output = []
    for item in conditioning:
        if len(item) < 2 or not isinstance(item[1], dict):
            raise ValueError("H3 continuation requires keyframe metadata in the positive conditioning")
        copied = list(item)
        metadata = copied[1].copy()
        guides = list(metadata.get("minimax_keyframes", []))
        guides.append(guide)
        metadata["minimax_keyframes"] = guides
        if len(copied) > 1:
            copied[1] = metadata
        else:
            copied.append(metadata)
        output.append(copied)
    return output


def _blend_transition(previous, current, time_dim=-1):
    """Crossfade matching overlap tensors along their modality time axis."""
    if previous.shape != current.shape:
        raise ValueError("Crossfade overlap tensors must have matching shapes")
    length = previous.shape[time_dim]
    if length == 0:
        return previous
    weights = torch.linspace(
        0.0, 1.0, length + 2, device=current.device, dtype=torch.float32
    )[1:-1]
    view_shape = [1] * current.ndim
    view_shape[time_dim] = length
    weights = weights.view(view_shape)
    previous = previous.to(device=current.device, dtype=torch.float32)
    current = current.to(dtype=torch.float32)
    return (previous * (1.0 - weights) + current * weights).to(current.dtype)


def _audio_tokens_for_video_prefix(start, prefix_tokens):
    """Map a local H3 video-token prefix to its aligned 40-Hz audio prefix."""
    start_frame = video_frames_for_latents(start)
    prefix_end_frame = video_frames_for_latents(start + prefix_tokens)
    return frames_to_audio_latents(prefix_end_frame - start_frame)


def _append_continuation(assembled_video, assembled_audio, sampled, window, audio_range,
                         previous_window, previous_audio_range, transition, mode):
    """Keep the old overlap for continuation; blend or replace it only on request."""
    start, end = window
    previous_end = previous_window[1]
    audio_start, audio_end = audio_range
    previous_audio_end = previous_audio_range[1]
    video, audio = sampled.unbind()
    video = video.to(device=assembled_video.device, dtype=assembled_video.dtype)
    audio = audio.to(device=assembled_audio.device, dtype=assembled_audio.dtype)
    overlap_tokens = previous_end - start
    overlap_audio = min(max(0, previous_audio_end - audio_start), audio.shape[-1])

    if mode == "crossfade":
        fade_start = transition_offset_tokens(start, overlap_tokens, transition) if transition is not None else 0
        fade_length = max(0, overlap_tokens - fade_start)
        if fade_length:
            global_start = start + fade_start
            assembled_video[:, :, global_start:previous_end] = _blend_transition(
                assembled_video[:, :, global_start:previous_end],
                video[:, :, fade_start:fade_start + fade_length],
                time_dim=2,
            )
        audio_fade_start = min(_audio_tokens_for_video_prefix(start, fade_start), overlap_audio)
        audio_fade_length = overlap_audio - audio_fade_start
        if audio_fade_length:
            assembled_audio[..., audio_start + audio_fade_start:previous_audio_end] = _blend_transition(
                assembled_audio[..., audio_start + audio_fade_start:previous_audio_end],
                audio[..., audio_fade_start:audio_fade_start + audio_fade_length],
            )
        assembled_video[:, :, previous_end:end] = video[:, :, overlap_tokens:]
        assembled_audio[..., previous_audio_end:audio_end] = audio[..., overlap_audio:]
        return assembled_video, assembled_audio

    replace_from = (
        transition_offset_tokens(start, overlap_tokens, transition)
        if mode == "cut" else overlap_tokens
    )
    assembled_video[:, :, start + replace_from:end] = video[:, :, replace_from:]
    audio_replace_start = (
        overlap_audio if mode == "continue"
        else min(_audio_tokens_for_video_prefix(start, replace_from), overlap_audio)
    )

    global_audio_start = audio_start + audio_replace_start
    if global_audio_start < audio_end:
        assembled_audio[..., global_audio_start:audio_end] = audio[..., audio_replace_start:]
    return assembled_video, assembled_audio


class H3SlidingContextSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SlidingContextSampler",
            display_name="Δ H3 Sampler",
            category="triXope/film",
            description=(
                "Sampler replacement for MiniMax H3. Sliding windows continue from generated AV overlap by default; "
                "explicit cut/dissolve cues can change the join."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent"),
                io.Clip.Input("clip", optional=True,
                              tooltip="Connect the same MiniMax H3 CLIP used to encode the static base conditioning when timestamped beats are provided."),
                io.String.Input("story_beats", default="", multiline=True,
                                tooltip="Optional timestamp script: [00:00:05-00:00:12] beat or [00:00:12] beat. Keep this script out of the base conditioning prompt."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=1000),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES,
                               default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.SCHEDULER_NAMES,
                               default="beta"),
                io.Int.Input("window_frames", default=124, min=22, max=362, step=17,
                             tooltip="H3-aligned frames per window; 124 frames is about 5.2 seconds."),
                io.Int.Input("overlap_frames", default=39, min=22, max=192, step=17,
                             tooltip="H3-aligned overlap; 39 frames is about 1.6 seconds."),
                io.String.Input("transition_prompt", default="", multiline=True,
                                tooltip="Paste any transition wording from the actual positive prompt here for mode detection. This text does not alter conditioning; encoded positive text cannot be read back by the sampler."),
                io.Combo.Input("transition_mode", options=["auto", "continue", "crossfade"], default="auto",
                               tooltip="Auto continues by default; explicit timestamped story-beat cues override the join. transition_prompt can request a global cut/crossfade. Continue ignores cues; crossfade always blends."),
                io.Boolean.Input("sliding_windows", default=True,
                                 tooltip="Off runs one H3 sampling pass. On samples sequential windows and carries the previous AV overlap as preserved context."),
                io.Boolean.Input("enforce_cuts", default=True,
                                 tooltip="When enabled, a story-beat start within a sliding overlap forces a hard cut there; previous AV context is not carried across that boundary."),
            ],
            outputs=[io.Latent.Output(display_name="output")],
        )

    @classmethod
    def execute(cls, model, positive, latent, seed, steps, sampler_name,
                scheduler, window_frames, overlap_frames, sliding_windows,
                transition_mode="auto", transition_prompt="", clip=None, story_beats="", enforce_cuts=False) -> io.NodeOutput:
        if "samples" not in latent or not _is_h3_latent(latent["samples"]):
            raise ValueError(
                "Expected MiniMax H3 AV latent: NestedTensor(video [B,24,T,H,W], audio [B,32,2,T]). "
                "Connect MiniMaxH3ImageToVideo or MiniMaxH3ReferenceToVideo."
            )
        if latent["samples"].tensors[0].shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")

        beats = parse_timestamped_beats(story_beats) if story_beats.strip() else None
        if transition_mode not in {"auto", "continue", "crossfade"}:
            raise ValueError("transition_mode must be auto, continue, or crossfade")
        prompt_transition = explicit_transition_mode(transition_prompt)
        if beats is not None and clip is None:
            raise ValueError("Connect the MiniMax H3 CLIP input to encode timestamped story beats")

        if sliding_windows:
            if window_frames <= overlap_frames:
                raise ValueError("window_frames must be larger than overlap_frames")
            if latent.get("noise_mask") is not None:
                raise ValueError("Sliding continuation does not yet support a pre-existing H3 noise mask")
            window_tokens = requested_frames_to_latents(window_frames)
            overlap_tokens = requested_frames_to_latents(overlap_frames)
            windows = plan_windows(
                latent["samples"].tensors[0].shape[2], window_tokens, overlap_tokens
            )
        else:
            windows = [(0, latent["samples"].tensors[0].shape[2])]

        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, steps
        ).cpu()
        latent_base = latent.copy()
        latent_base["samples"] = comfy.sample.fix_empty_latent_channels(
            model,
            latent["samples"],
            latent.get("downscale_ratio_spacial"),
            latent.get("downscale_ratio_temporal"),
        )
        original_samples = latent_base["samples"]
        assembled_video = None
        assembled_audio = None
        audio_ranges = []

        for index, (start, end) in enumerate(windows):
            print(f"[H3SlidingContextSampler] Generating chunk {index + 1}/{len(windows)}")
            chunk_positive = positive
            if beats is not None:
                chunk_prompt = select_beats_for_latent_window(beats, start, end)
                if chunk_prompt:
                    if enforce_cuts:
                        chunk_prompt += ("\nEvery listed beat start is a mandatory hard cut: immediately cut to the next "
                                         "shot at its timestamp; do not dissolve, crossfade, or blend shots.")
                    chunk_positive = _encode_chunk_conditioning(clip, positive, chunk_prompt)

            transition = None
            join_mode = "continue"
            if sliding_windows and index:
                overlap_tokens = windows[index - 1][1] - start
                start_frame = video_frames_for_latents(start)
                overlap_end_frame = video_frames_for_latents(windows[index - 1][1])
                if enforce_cuts and beats:
                    cut = next((beat for beat in beats if start_frame <= int(round(beat["start"] * FPS)) < overlap_end_frame), None)
                    if cut is None:
                        cut = next((beat for beat in beats if start_frame <= int(round(beat["start"] * FPS)) <= overlap_end_frame), None)
                    if cut is not None:
                        join_mode, transition = "cut", int(round(cut["start"] * FPS))
                if join_mode == "continue" and transition_mode == "crossfade":
                    join_mode = "crossfade"
                elif join_mode == "continue" and transition_mode == "auto":
                    cue = transition_for_overlap(beats, start_frame, overlap_end_frame) if beats else None
                    if cue is not None:
                        join_mode, transition = cue["mode"], cue["frame"]
                    elif prompt_transition is not None:
                        join_mode, transition = prompt_transition, start_frame

            if sliding_windows:
                start_frame = video_frames_for_latents(start)
                end_frame = video_frames_for_latents(end)
                chunk_positive = _conditioning_for_window(
                    chunk_positive, start_frame, end_frame
                )
                chunk, audio_range = _slice_window(original_samples, start, end)
                audio_ranges.append(audio_range)
                chunk_latent = latent_base.copy()
                chunk_latent["samples"] = chunk
                if sliding_windows and index and join_mode in {"continue", "cut", "crossfade"}:
                    previous_end = windows[index - 1][1]
                    audio_start, audio_end = audio_range
                    previous_audio_end = audio_ranges[index - 1][1]
                    overlap_tokens = previous_end - start
                    context_tokens = overlap_tokens
                    if join_mode != "continue":
                        context_tokens = (
                            transition_offset_tokens(start, overlap_tokens, transition)
                            if transition is not None else 0
                        )
                    context_audio = min(
                        max(0, previous_audio_end - audio_start),
                        audio_end - audio_start,
                        _audio_tokens_for_video_prefix(start, context_tokens),
                    )
                    if context_tokens or context_audio:
                        if assembled_video is None or assembled_audio is None:
                            raise RuntimeError("Missing prior H3 output required for continuation")
                        chunk_positive = _conditioning_with_context(
                            chunk_positive,
                            assembled_video[:, :, start:start + context_tokens],
                            assembled_audio[..., audio_start:audio_start + context_audio],
                        )
            else:
                chunk_latent = latent_base

            guider = PositiveOnlyGuider(model)
            guider.set_conds(chunk_positive)

            run_seed = (seed + index) % (1 << 64)
            noise = comfy.sample.prepare_noise(
                chunk_latent["samples"], run_seed, chunk_latent.get("batch_index")
            )
            callback_data = {}
            callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, callback_data)
            result = guider.sample(
                noise,
                chunk_latent["samples"],
                sampler,
                sigmas,
                denoise_mask=chunk_latent.get("noise_mask"),
                callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                seed=run_seed,
            )
            if sliding_windows:
                video, audio = result.unbind()
                start, end = windows[index]
                audio_start, audio_end = audio_ranges[index]
                video, audio = video.to(torch.float32), audio.to(torch.float32)
                if index == 0:
                    original_video, original_audio = original_samples.unbind()
                    assembled_video = original_video.to(video.device, dtype=torch.float32).clone()
                    assembled_audio = original_audio.to(audio.device, dtype=torch.float32).clone()
                    assembled_video[:, :, start:end] = video
                    assembled_audio[..., audio_start:audio_end] = audio
                else:
                    assembled_video, assembled_audio = _append_continuation(
                        assembled_video,
                        assembled_audio,
                        result,
                        windows[index],
                        audio_ranges[index],
                        windows[index - 1],
                        audio_ranges[index - 1],
                        transition,
                        join_mode,
                    )
        if sliding_windows:
            original_video, original_audio = original_samples.unbind()
            output_samples = comfy.nested_tensor.NestedTensor((
                assembled_video.to(dtype=original_video.dtype),
                assembled_audio.to(dtype=original_audio.dtype),
            ))
        else:
            output_samples = result

        out = latent_base.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = output_samples.to(comfy.model_management.intermediate_device())
        return io.NodeOutput(out)


_SHOT_HEADER = re.compile(r"\[\s*shot\s+(?P<number>\d+)\s*\]", re.IGNORECASE)
_SHOT_PROMPT_HEADER = re.compile(r"\[\s*shot\s+(?P<number>\d+)\s*\]", re.IGNORECASE)
_PROMPT_SECTION_HEADER = re.compile(
    r"(?im)^[   ]*(?P<name>subject_definitions|summary|retention_analysis|detailed_description|"
    r"integrated_multimodal_description|overall_soundscape|non_diegetic_music)[     ]*:[    ]*"
)
_TIME_MARKER = re.compile(
    r"\[\s*(?P<start>\d{1,3}:\d{2}(?::\d{2})?(?:[.,]\d+)?)"
    r"(?:\s*[-–]\s*(?P<end>\d{1,3}:\d{2}(?::\d{2})?(?:[.,]\d+)?))?\s*\]"
)


def _unwrap_input(value):
    """Unwrap the single-item list ComfyUI supplies for list-input schemas."""
    while isinstance(value, (list, tuple)):
        if not value:
            return None
        if len(value) != 1:
            raise ValueError("Expected one connected value; shot_lengths is the only multi-value input")
        value = value[0]
    return value


def _shot_length_values(value):
    """Flatten the screenwriter's OUTPUT_IS_LIST integer seconds array."""
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if value is None or value == [] or value == [None]:
        return None
    if not isinstance(value, (list, tuple)):
        value = [value]
    try:
        lengths = [int(item) for item in value]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("shot_lengths must be the screenwriter's integer seconds list") from error
    if not lengths or any(seconds <= 0 for seconds in lengths):
        raise ValueError("shot_lengths must contain one positive duration in seconds per shot")
    return lengths


def _seconds(value):
    return parse_timestamp(value.replace(",", "."))


def _format_shot_time(seconds):
    seconds = max(0.0, seconds)
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remainder = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{remainder:06.3f}"
    minutes = int(seconds // 60)
    remainder = seconds % 60
    return f"{minutes:02d}:{remainder:06.3f}"


def _read_beats(text):
    """Parse one shot's timestamp markers, including several beats on one line."""
    markers = list(_TIME_MARKER.finditer(text))
    beats = []
    for index, marker in enumerate(markers):
        end_of_text = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        description = text[marker.end():end_of_text].strip(" \t\r\n.;")
        if not description:
            raise ValueError("Each story-beat timestamp needs a description")
        start = _seconds(marker.group("start"))
        end = _seconds(marker.group("end")) if marker.group("end") else None
        if end is not None and end <= start:
            raise ValueError("A story-beat end timestamp must follow its start")
        beats.append({"start": start, "end": end, "text": description})
    if not beats and text.strip():
        raise ValueError("Story-beat text must use [MM:SS.mmm] timestamps")
    starts = [beat["start"] for beat in beats]
    if starts != sorted(starts) or len(starts) != len(set(starts)):
        raise ValueError("Timestamps within each shot must be unique and ascending from 00:00.000")
    for index, beat in enumerate(beats[:-1]):
        next_start = beats[index + 1]["start"]
        if beat["end"] is None:
            beat["end"] = next_start
        elif beat["end"] > next_start:
            raise ValueError("Story beats within a shot may not overlap")
    return beats


def _local_beat_script(beats, duration_seconds):
    lines = []
    for beat in beats:
        start, end = beat["start"], beat["end"]
        if start >= duration_seconds:
            raise ValueError("A story-beat timestamp falls beyond its shot_lengths duration")
        if end is not None:
            if end > duration_seconds:
                raise ValueError("A story-beat end timestamp falls beyond its shot_lengths duration")
            if end <= start:
                raise ValueError("A story beat has no duration inside its shot")
            marker = f"[{_format_shot_time(start)}-{_format_shot_time(end)}]"
        else:
            marker = f"[{_format_shot_time(start)}]"
        lines.append(f"{marker} {beat['text']}")
    return "\n".join(lines)


def _infer_legacy_shot_durations(text):
    """Infer ordered durations from the screenwriter's cumulative one-range-per-shot output."""
    if not text.strip() or _SHOT_HEADER.search(text):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    ranges = []
    for line in lines:
        markers = list(_TIME_MARKER.finditer(line))
        if len(markers) != 1 or markers[0].group("end") is None:
            return None
        ranges.append((
            _seconds(markers[0].group("start")),
            _seconds(markers[0].group("end")),
        ))
    if not ranges or any(end <= start for start, end in ranges):
        return None
    for previous, current in zip(ranges, ranges[1:]):
        if abs(current[0] - previous[1]) > 0.01:
            return None
    return [end - start for start, end in ranges]


def _parse_shot_beats(text, durations):
    """Group labeled micro-beats or convert the screenwriter's legacy global ranges."""
    if not text.strip():
        return [""] * len(durations)

    headers = list(_SHOT_HEADER.finditer(text))
    shot_beats = [[] for _ in durations]
    if headers:
        if text[:headers[0].start()].strip():
            raise ValueError("Put story-beat content after a [Shot N] header")
        seen = set()
        for index, header in enumerate(headers):
            shot_number = int(header.group("number"))
            shot_index = shot_number - 1
            if shot_index < 0 or shot_index >= len(durations) or shot_index in seen:
                raise ValueError("[Shot N] headers must uniquely match the shot_lengths array")
            seen.add(shot_index)
            end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
            shot_beats[shot_index] = _read_beats(text[header.end():end])
        if seen != set(range(len(durations))):
            raise ValueError("Provide one [Shot N] story-beat group for every shot_lengths entry")
        if any(not beats for beats in shot_beats):
            raise ValueError("Each [Shot N] group must contain at least one timestamped story beat")
        if any(beats[0]["start"] != 0 for beats in shot_beats):
            raise ValueError("Each shot's first story beat must begin at [00:00.000]")
    else:
        # Backwards compatibility: the screenwriter currently emits one globally
        # timestamped range per shot, in the same order as its duration array.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        parsed_lines = []
        for line in lines:
            beats = _read_beats(line)
            parsed_lines.extend(beats)
        if len(parsed_lines) == len(durations) and all(beat["end"] is not None for beat in parsed_lines):
            offset = 0.0
            for shot_index, (beat, duration) in enumerate(zip(parsed_lines, durations)):
                if abs(beat["start"] - offset) > 0.01:
                    raise ValueError("Legacy cumulative story-beat ranges must start at each shot boundary")
                local_end = beat["end"] - offset
                if abs(local_end - duration) > 0.01:
                    raise ValueError("Legacy story-beat range lengths must match the shot_lengths array")
                shot_beats[shot_index] = [{
                    "start": 0.0,
                    "end": local_end,
                    "text": beat["text"],
                }]
                offset += duration
        elif len(durations) == 1:
            shot_beats[0] = parsed_lines
        else:
            raise ValueError(
                "For multiple shots, use [Shot N] groups with shot-local timestamps, "
                "or connect the screenwriter's one-range-per-shot story_beats output"
            )
    if any(beats and beats[0]["start"] != 0 for beats in shot_beats):
        raise ValueError("Each shot's first story beat must begin at [00:00.000]")
    return [
        _local_beat_script(beats, duration)
        for beats, duration in zip(shot_beats, durations)
    ]


_PIPE_SPLIT = re.compile("[     ]*[|][  ]*")


def _split_pipe_sections(text):
    """Split a text box on bare pipes; one section means the text is not split."""
    raw = str(text or "")
    if "|" not in raw:
        return [raw]
    sections = [section.strip() for section in _PIPE_SPLIT.split(raw)]
    sections = [section for section in sections if section]
    if len(sections) <= 1:
        return [raw]
    return sections


def _plan_prompt_sections(prompt, story_beats, shot_lengths, length_seconds):
    """Expand pipe-split prompts into independent (prompt, durations, beats) runs.

    Without a pipe this reproduces the previous single-run behaviour exactly. Each
    pipe-separated section runs to completion on its own prompt, own beats and own
    shot structure before the next section starts from a clean slate.
    """
    sections = _split_pipe_sections(prompt)
    beats_text = str(story_beats or "").strip()
    if len(sections) <= 1:
        durations = _shot_length_values(shot_lengths)
        lengths_connected = durations is not None
        if durations is None:
            durations = _infer_legacy_shot_durations(beats_text)
            lengths_connected = durations is not None
        if durations is None:
            shot_count = len(_SHOT_HEADER.findall(beats_text)) or 1
            durations = [float(length_seconds)] * shot_count
        beat_scripts = _parse_shot_beats(beats_text, durations)
        return [(sections[0], durations, beat_scripts)], lengths_connected

    beat_sections = _split_pipe_sections(beats_text) if beats_text else [""] * len(sections)
    if beats_text and len(beat_sections) != len(sections):
        raise ValueError(
            f"prompt has {len(sections)} pipe-separated sections but story_beats has "
            f"{len(beat_sections)}; give story_beats the same number of pipe-separated "
            "sections, or leave story_beats empty"
        )
    plan = []
    for section_prompt, section_beats in zip(sections, beat_sections):
        durations = _infer_legacy_shot_durations(section_beats)
        if durations is None:
            shot_count = len(_SHOT_HEADER.findall(section_beats)) or 1
            durations = [float(length_seconds)] * shot_count
        plan.append((section_prompt, durations, _parse_shot_beats(section_beats, durations)))
    return plan, False


_CHARACTER_NOTE_TEXT = "Only the identity and wardrobe are used, never the four-panel sheet layout."
_PALETTE_NOTE_TEXT = "Only its colors and tonal look are used, never the swatch layout."
_CHARACTER_DEFINITION_RE = re.compile(
    r"<Picture\s+(\d+)>[^\n]*never the four-panel sheet layout", re.IGNORECASE)
_SUBJECT_LABEL_RE = re.compile(r"<Subject\s+(\d+)>", re.IGNORECASE)
_SOURCE_LABEL_RE = re.compile(r"<(?:Picture|Video|Audio)\s+\d+>", re.IGNORECASE)
_DEFINITION_LINE_RE = re.compile(
    r"^\s*<Subject\s+(?P<subject>\d+)>\s+is\s+(?:the\s+)?(?P<name>.+?)\s+"
    r"(?:environment\s+)?in\s+<(?:Picture|Video)\s+\d+>\s*[,:]?\s*(?P<traits>.*)$",
    re.IGNORECASE,
)
_DEF_SUBJECT_RE = re.compile(r"^\s*<Subject\s+(\d+)>", re.IGNORECASE)


def _character_picture_numbers(prompt):
    """<Picture N> numbers the prompt itself marks as character sheets.

    Only the screenwriter's character definitions carry the "never the four-panel sheet
    layout" note, so the conditioner can size character references at full fidelity and
    leave props, locations and the palette chart at the cheaper match size.
    """
    return {int(number) for number in _CHARACTER_DEFINITION_RE.findall(str(prompt or ""))}


def _warn_dangling_references(prompt, image_count, video_count):
    """Report prompt citations that have no matching reference attached."""
    pictures = [int(n) for n in re.findall(r"<Picture\s+(\d+)>", str(prompt or ""))]
    if pictures and max(pictures) > image_count:
        print(f"[H3FL2VAandRef2VA] \u26a0\ufe0f prompt cites <Picture {max(pictures)}> but only "
              f"{image_count} reference image(s) are attached; connect them in registry order, "
              "or switch the prompt to text mode.")
    videos = [int(n) for n in re.findall(r"<Video\s+(\d+)>", str(prompt or ""))]
    if videos and max(videos) > video_count:
        print(f"[H3FL2VAandRef2VA] \u26a0\ufe0f prompt cites <Video {max(videos)}> but only "
              f"{video_count} reference video(s) are attached.")


def _plain_subject_name(name):
    name = re.sub(r"\s+", " ", str(name or "")).strip(" .,")
    return "the color palette" if re.search(r"(?i)color\s+palette", name) else name


def _text_only_prompt(prompt):
    """Rewrite a screenwriter prompt for text-only conditioning.

    Text mode attaches nothing, so every <Picture N>/<Video N> citation dangles and the
    model is told to consult references that are not there. Expanding each subject into its
    name and repeating the identity prose inline is the only identity signal a
    text-to-video render can get, so it must survive verbatim in every shot.
    """
    text = str(prompt or "")
    if not _SUBJECT_LABEL_RE.search(text):
        return text
    names, lines = {}, []
    for line in text.splitlines():
        match = _DEFINITION_LINE_RE.match(line)
        if not match:
            lines.append(line)
            continue
        subject = match.group("subject")
        name = _plain_subject_name(match.group("name"))
        traits = match.group("traits")
        traits = traits.replace(_CHARACTER_NOTE_TEXT, "").replace(_PALETTE_NOTE_TEXT, "")
        traits = re.sub(r"\s+", " ", traits).strip(" .,;")
        # Outside the definition line the palette reads better as "the palette above"
        # than as a repeated "the color palette ... follow the color palette".
        names[subject] = "the palette above" if name == "the color palette" else name
        if name == "the color palette":
            lines.append(f"The color palette: {traits}." if traits else "The color palette is defined by the film's grade.")
        else:
            lines.append(f"{name}: {traits}." if traits else f"{name}.")
    text = "\n".join(lines)
    text = _SUBJECT_LABEL_RE.sub(lambda m: names.get(m.group(1), m.group(0)), text)
    text = _SOURCE_LABEL_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:])", r"\1", text)
    return text


def _input_present(value):
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value)
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape):
        return int(shape[0]) > 0
    return True


def _filter_definitions(block, used_subjects, shot_total):
    """Keep only the definitions this shot or its beats actually cite.

    Defining the whole cast in every shot invites off-screen identities into the frame and
    spends vision tokens on them. Renumbering stays safe because <Picture N> is fixed by
    registry position while <Subject N> is prompt-local. The palette definition always
    survives: the shared style block refers to it in every shot.
    """
    if not used_subjects or shot_total <= 1:
        return block
    lines = block.splitlines()
    header, body = "", []
    if lines and not _DEF_SUBJECT_RE.match(lines[0]):
        header, lines = lines[0], lines[1:]
    for line in lines:
        match = _DEF_SUBJECT_RE.match(line)
        if (match and int(match.group(1)) not in used_subjects
                and "color palette" not in line.lower()):
            continue
        body.append(line)
    return "\n".join(([header] if header else []) + body).strip()


def _beat_text(beats):
    """Flatten one shot's parsed beats back into text for the definition filter."""
    if not beats:
        return ""
    if isinstance(beats, str):
        return beats
    parts = []
    for item in beats:
        if isinstance(item, (list, tuple)):
            parts.extend(str(piece) for piece in item[1:] if isinstance(piece, str))
        elif isinstance(item, dict):
            parts.append(str(item.get("text") or item.get("beat") or ""))
        elif isinstance(item, str):
            parts.append(item)
    return " ".join(part for part in parts if part)


def _prompt_for_shot(prompt, shot_number, extra_text=""):
    """Keep shared identities/style/audio and only this shot's action description."""
    text = prompt or ""
    section_headers = list(_PROMPT_SECTION_HEADER.finditer(text))
    integrated_header = next(
        (header for header in section_headers
         if header.group("name").lower() == "integrated_multimodal_description"),
        None,
    )
    if integrated_header is None:
        return text

    section_end = next(
        (header.start() for header in section_headers if header.start() > integrated_header.start()),
        len(text),
    )
    shot_headers = list(_SHOT_PROMPT_HEADER.finditer(text, integrated_header.end(), section_end))
    if not shot_headers:
        return text

    selected_index = next(
        (index for index, header in enumerate(shot_headers)
         if int(header.group("number")) == int(shot_number)),
        None,
    )
    if selected_index is None:
        selected_shot = f"[Shot {shot_number}]"
    else:
        selected_end = (
            shot_headers[selected_index + 1].start()
            if selected_index + 1 < len(shot_headers) else section_end
        )
        selected_shot = text[shot_headers[selected_index].start():selected_end].strip()

    used_subjects = {
        int(number) for number in re.findall(
            r"<Subject\s+(\d+)>", selected_shot + " " + str(extra_text or ""))
    }

    shared_parts = []
    if section_headers:
        preamble = text[:section_headers[0].start()].strip()
        if preamble:
            shared_parts.append(preamble)

    # Keep character/prop/location definitions, but not whole-story summaries or
    # other narrative sections that could reintroduce an unrelated shot's action.
    for index, header in enumerate(section_headers):
        name = header.group("name").lower()
        next_end = section_headers[index + 1].start() if index + 1 < len(section_headers) else len(text)
        if name == "subject_definitions":
            shared_parts.append(_filter_definitions(
                text[header.start():next_end].strip(), used_subjects, len(shot_headers)))
        elif name in {"overall_soundscape", "non_diegetic_music"}:
            shared_parts.append(text[header.start():next_end].strip())

    style = text[integrated_header.end():shot_headers[0].start()].strip()
    if style:
        shared_parts.append(style)
    shared_parts.append(selected_shot)
    return "\n\n".join(part for part in shared_parts if part)


def _frame_grid(requested_frames):
    """Round a requested duration up to H3's valid 17k+5 pixel-frame grid."""
    requested_frames = max(22, int(requested_frames))
    groups = max(1, (requested_frames - 5 + 16) // 17)
    return 5 + 17 * groups


FRAME_ALIGNMENT = 17


def _seconds_to_frames(seconds):
    """Seconds -> frames at 24 fps, snapped up to H3's 17k+5 frame grid.

    max(5, round(seconds * 24)) + (5 - (max(5, round(seconds * 24)) % 17)) % 17
    """
    frames = max(5, int(round(float(seconds) * FPS)))
    return frames + (5 - (frames % FRAME_ALIGNMENT)) % FRAME_ALIGNMENT


def _automatic_window_frames(duration_seconds, overlap_frames):
    """Fit one shot and reserve at least a valid 5-token step beyond overlap."""
    shot_frames = _frame_grid(math.ceil(float(duration_seconds) * FPS))
    minimum_window = _frame_grid(overlap_frames) + 17
    return min(362, max(shot_frames, minimum_window))


def _drive_audio_segment(audio, start_seconds, frame_count):
    """Cut one shot's slice of the drive audio and pad it to the shot's frame span."""
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    total_samples = int(waveform.shape[-1])
    start = max(0, int(round(float(start_seconds) * sample_rate)))
    needed = int(round(frame_count / FPS * sample_rate))
    if needed <= 0:
        raise ValueError("audio_drive needs a shot with a positive frame count")
    if start >= total_samples:
        raise ValueError(
            f"drive_audio is shorter than the screenplay timeline: the shot starting at "
            f"{float(start_seconds):.2f}s needs audio through {(start + needed) / sample_rate:.2f}s, "
            f"but only {total_samples / sample_rate:.2f}s is connected"
        )
    segment = waveform[..., start:start + needed]
    if segment.shape[-1] < needed:
        print(
            f"[FilmAuteurH3] audio_drive: drive audio ends at {total_samples / sample_rate:.2f}s; "
            f"padding the last {needed - segment.shape[-1]} samples of this shot with silence."
        )
        segment = torch.nn.functional.pad(segment, (0, needed - segment.shape[-1]))
    return {"waveform": segment, "sample_rate": sample_rate}


def _audio_drive_guide(audio_vae, audio, start_seconds, frame_count, audio_axis):
    """Encode a shot's drive audio as an H3 audio guide pinned to the shot's first frame."""
    segment = _drive_audio_segment(audio, start_seconds, frame_count)
    audio_latent, _latent_frames = _encode_ref_audio(audio_vae, segment)
    audio_latent = audio_latent[..., :audio_axis]
    if audio_latent.shape[-1] < audio_axis:
        pad = audio_axis - audio_latent.shape[-1]
        audio_latent = torch.nn.functional.pad(audio_latent, (0, pad))
    return {"resolved_frame_index": 0, "audio_latent": audio_latent.contiguous()}


def _crop_audio_guides(guides, window_frames):
    """Keep only each audio guide's slice of a sliding window so windows hear their own audio.

    Guides arrive with a window-local anchor after remapping, so an audio guide
    that spans the whole shot starts at a negative offset for later windows. The
    crop rebases it to the window start and trims its latent rows to the window's
    audio span, which is the same span _slice_window cuts from the target audio.
    """
    cropped = []
    for guide in guides:
        audio = guide.get("audio_latent")
        if audio is None or guide.get("latent") is not None:
            cropped.append(guide)
            continue
        anchor = int(guide["resolved_frame_index"])
        guide_frames = int(math.ceil(audio.shape[-1] * FPS / AUDIO_LATENT_FPS))
        local_start = max(anchor, 0)
        local_end = min(anchor + guide_frames, window_frames)
        if local_end <= local_start:
            continue
        audio_start = frames_to_audio_latents(local_start - anchor)
        audio_end = min(audio_start + frames_to_audio_latents(local_end - local_start), audio.shape[-1])
        if audio_end <= audio_start:
            continue
        sliced = dict(guide)
        sliced["resolved_frame_index"] = local_start
        sliced["audio_latent"] = audio[..., audio_start:audio_end].contiguous()
        cropped.append(sliced)
    return cropped


def _apply_audio_guide(conditioning, guide):
    """Pin an encoded audio guide onto the H3 conditioning payload."""
    guides = list(conditioning[0][1].get("minimax_keyframes", []))
    guides.append(guide)
    return node_helpers.conditioning_set_values(conditioning, {"minimax_keyframes": guides})


def _replace_audio_stream(latent, audio_latent):
    """Return an H3 AV latent whose audio stream is the drive audio instead of generated audio."""
    samples = latent["samples"]
    if not _is_h3_latent(samples):
        raise ValueError("audio_drive expected a sampled H3 AV latent")
    video, audio = samples.unbind()
    source = audio_latent.to(device=audio.device, dtype=audio.dtype)
    target = int(audio.shape[-1])
    if source.shape[-1] < target:
        source = torch.nn.functional.pad(source, (0, target - source.shape[-1]))
    source = source[..., :target]
    if source.shape[0] != audio.shape[0]:
        source = source.expand(audio.shape[0], -1, -1, -1)
    replaced = dict(latent)
    replaced["samples"] = comfy.nested_tensor.NestedTensor((video, source.contiguous()))
    return replaced


def _still_av_latent(latent):
    """Collapse an H3 AV latent to MiniMax H3's still form: one video latent frame.

    The VAE has a dedicated single-latent decode path (a decoder fine-tuned for it
    when the checkpoint ships one, otherwise a clip of copies keeping the first
    real frame), and the audio stream keeps the two latent frames that one video
    frame spans so the packed sequence stays an audio+video sequence.
    """
    video, audio = latent["samples"].unbind()
    still_video = torch.zeros(
        (video.shape[0], video.shape[1], 1, video.shape[3], video.shape[4]),
        dtype=video.dtype, device=video.device,
    )
    still_audio = torch.zeros(
        (audio.shape[0], audio.shape[1], audio.shape[2], frames_to_audio_latents(1)),
        dtype=audio.dtype, device=audio.device,
    )
    return {"samples": comfy.nested_tensor.NestedTensor((still_video, still_audio))}


def _decode_still(vae, latent):
    """Decode a one-frame H3 AV latent into a ComfyUI IMAGE batch."""
    video, _audio = latent["samples"].unbind()
    pixels = vae.decode(video)
    if pixels.ndim == 5:
        pixels = pixels[:, 0]
    elif pixels.ndim == 3:
        pixels = pixels.unsqueeze(0)
    return pixels.float().clamp_(0.0, 1.0).cpu()


class FilmAuteurH3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="triXope FilmAuteurH3",
            display_name="Δ Film Auteur (H3)",
            category="triXope/film",
            description=(
                "Build MiniMax H3 text/reference/keyframe conditioning, then sample it "
                "in one pass or with sliding AV context, timestamped beats and transitions. "
                "The original conditioner and sampler nodes remain available separately."
            ),
            is_input_list=True,
            inputs=[
                io.Model.Input("model"),
                # Inputs from H3FL2VAandRef2VA.
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input(
                    "prompt", multiline=True, dynamic_prompts=True,
                    tooltip=(
                        "One prompt is one complete render. Put a | between prompts to run each "
                        "one entirely on its own: that section finishes all of its shots, prints "
                        "its results and starts the next section with a clean slate."
                    ),
                ),
                io.Combo.Input(
                    "output_mode", options=["video", "image"], default="video",
                    tooltip=(
                        "video renders each shot as a clip. image renders one still per shot "
                        "as a single MiniMax H3 video latent frame and returns it on the image "
                        "output; length_seconds, shot durations, sliding windows and audio_drive "
                        "do not apply to a still."
                    ),
                ),
                io.Int.Input("width", default=1792, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input(
                    "length_seconds", default=5, min=1, max=3600, step=1,
                    tooltip=(
                        "Requested clip length in seconds. Converted to frames at 24 fps and "
                        "snapped to H3's 17k+5 grid: max(5, round(seconds * 24)) padded up to "
                        "the next 17k+5 (5s -> 124 frames). A timed prompt can extend this "
                        "automatically. Ignored in image mode."
                    ),
                ),
                io.Combo.Input(
                    "mode", options=["text", "reference", "keyframes", "all"], default="reference",
                    tooltip=(
                        "text: prompt only (the prompt is rewritten without <Picture N>/<Video N> "
                        "citations and repeats each identity inline); reference: text plus reference "
                        "inputs; keyframes: text plus keyframes; all: keyframes and references "
                        "together. Matches the original conditioner."
                    ),
                ),
                io.Audio.Input(
                    "drive_audio", optional=True,
                    tooltip=(
                        "Audio that drives the video when audio_drive is on. Each shot takes its "
                        "own slice of the timeline (shot start to shot end); a short track is "
                        "padded with silence, and a track that ends before a shot starts raises "
                        "an error."
                    ),
                ),
                io.Image.Input(
                    "keyframes", optional=True,
                    tooltip=(
                        "Batch of keyframe images evenly distributed over the generated clip. "
                        "The first image anchors the start; one image is placed at frame zero."
                    ),
                ),
                io.Combo.Input(
                    "ref_image_size", options=["match", "max", "mixed"], default="mixed",
                    tooltip=(
                        "Reference image sizing: match the generation pixel area, max uses the "
                        "reference pipeline's 2048px short-edge target, and mixed (default) keeps "
                        "character sheets at max while props, locations and the palette use match."
                    ),
                ),
                io.Combo.Input(
                    "ref_alpha", options=["off", "white", "black", "gray", "transparent"], default="off",
                    tooltip="off applies no alpha compositing or background removal; H3 still aligns image size and VAE-encodes it, so this is not pixel pass-through and alpha is not preserved as alpha. 'transparent' uses input alpha or removes an opaque background with ComfyUI BiRefNet, then composites onto black. Opaque images need a model in models/background_removal.",
                ),
                io.Boolean.Input(
                    "audio_drive", default=False,
                    tooltip=(
                        "Off lets MiniMax H3 generate its own soundtrack. On pins the connected "
                        "drive_audio to the AV latent as an H3 audio guide, so the generated "
                        "video is conditioned on that audio and the returned audio is the "
                        "provided track instead of a generated one."
                    ),
                ),
                io.Image.Input(
                    "ref_image_batch", optional=True,
                    tooltip="Reference image batch, mapped to MiniMax H3 picture references.",
                ),
                io.Autogrow.Input(
                    "ref_videos", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps (2-15s)"),
                        names=[f"ref_video_{i}" for i in range(1, 5)],
                        min=0,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Audio.Input(
                            "ref_video_audio", tooltip="Soundtrack of the same-numbered reference video"
                        ),
                        names=[f"ref_video_audio_{i}" for i in range(1, 5)],
                        min=0,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Audio.Input("ref_audio", tooltip="Standalone reference audio"),
                        names=[f"ref_audio_{i}" for i in range(1, 5)],
                        min=0,
                    ),
                ),
                # Inputs from H3SlidingContextSampler.
                io.String.Input(
                    "story_beats", default="", multiline=True,
                    tooltip=(
                        "Preferred format: one [Shot N] group per shot, with per-shot timestamps "
                        "starting at [00:00.000]. Also accepts the screenwriter's legacy cumulative "
                        "one-range-per-shot output when shot_lengths is connected. When the prompt is "
                        "split with |, split story_beats with the same number of | so each prompt "
                        "carries its own beats; leave it empty to render the prompts without beats."
                    ),
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xffffffffffffffff,
                    control_after_generate=True,
                ),
                io.Int.Input("steps", default=20, min=1, max=1000),
                io.Combo.Input(
                    "sampler_name", options=comfy.samplers.SAMPLER_NAMES,
                    default="euler",
                ),
                io.Combo.Input(
                    "scheduler", options=comfy.samplers.SCHEDULER_NAMES, default="beta",
                ),
                io.Int.Input(
                    "shot_lengths", optional=True, force_input=True,
                    tooltip=(
                        "Connect the screenwriter's shot_lengths INT-list output (durations in seconds). "
                        "Each duration is paired with one [Shot N] beat group and generated independently."
                    ),
                ),
                io.Boolean.Input(
                    "cross_shot_context", default=True,
                    tooltip=(
                        "When enabled, the previous shot's generated video tail is added as a "
                        "video reference for the next shot, not pinned to its opening frames. "
                        "Audio, prompt text, and shot outputs remain independent."
                    ),
                ),
                io.Int.Input(
                    "cross_shot_context_frames", default=48, min=22, max=192, step=17,
                    tooltip=(
                        "Requested prior-shot context in frames (24 fps); rounded up to H3's "
                        "17k+5 frame grid. 48 frames is about 2 seconds, enough to carry wardrobe, "
                        "casting and light without teaching the model to repeat the previous action. "
                        "The available tail is used if the previous shot is shorter."
                    ),
                ),
                io.Boolean.Input(
                    "voice_continuity", default=True,
                    tooltip=(
                        "When enabled, the previous shot's audio rides along with the cross-shot "
                        "video reference, so a character keeps the same voice, accent and room tone "
                        "from shot to shot. Only applies while cross_shot_context is on."
                    ),
                ),
                io.Boolean.Input(
                    "sliding_windows", default=True,
                    tooltip=(
                        "Off runs one H3 sampling pass. On samples sequential windows and "
                        "carries prior AV overlap as context."
                    ),
                ),
                io.Combo.Input(
                    "window_frames_mode", options=["automatic", "manual"], default="automatic",
                    tooltip="Automatic uses each shot's duration, rounded to H3's frame grid (124 frames for 5 seconds); manual uses window_frames.",
                ),
                io.Int.Input(
                    "window_frames", default=124, min=22, max=362, step=17,
                    tooltip="Manual H3-aligned frames per sliding window; automatic mode derives this from each shot's length.",
                ),
                io.Int.Input(
                    "overlap_frames", default=39, min=22, max=192, step=17,
                    tooltip="H3-aligned context overlap between windows; 22 frames is approximately 0.9 seconds.",
                ),
                io.Combo.Input(
                    "transition_mode", options=["auto", "continue", "crossfade"], default="auto",
                    tooltip=(
                        "Auto follows explicit timestamped cues; continue ignores them; "
                        "crossfade always blends sliding-window joins."
                    ),
                ),
                io.String.Input(
                    "transition_prompt", default="", multiline=True,
                    tooltip=(
                        "Optional transition wording from the positive prompt for global "
                        "cut/crossfade detection; it does not alter conditioning."
                    ),
                ),
                io.Boolean.Input(
                    "enforce_cuts", default=True,
                    tooltip=(
                        "Each shot is always generated and returned independently, so shot outputs "
                        "are hard-cut boundaries. When enabled, reinforce the hard cut in the "
                        "conditioning prompt; beat timestamps within a shot never create cuts."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive", is_output_list=True),
                io.Latent.Output(display_name="sampled shot latents", is_output_list=True),
                io.Latent.Output(display_name="initial shot latents", is_output_list=True),
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        audio_vae,
        prompt,
        width,
        height,
        length_seconds,
        model,
        seed,
        steps,
        sampler_name,
        scheduler,
        window_frames,
        overlap_frames,
        sliding_windows,
        shot_lengths=None,
        window_frames_mode="automatic",
        output_mode="video",
        mode="reference",
        keyframes=None,
        ref_image_size="match",
        ref_alpha="off",
        audio_drive=False,
        drive_audio=None,
        ref_image_batch=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
        story_beats="",
        transition_prompt="",
        transition_mode="auto",
        enforce_cuts=True,
        cross_shot_context=True,
        cross_shot_context_frames=48,
        voice_continuity=True,
    ) -> io.NodeOutput:
        # V3 list-input mode is required to receive the screenwriter's list-valued
        # INT output as one shot-length array. Unwrap all other single inputs.
        values = {
            "clip": clip, "vae": vae, "audio_vae": audio_vae, "prompt": prompt,
            "width": width, "height": height, "length_seconds": length_seconds, "model": model,
            "output_mode": output_mode,
            "seed": seed, "steps": steps, "sampler_name": sampler_name,
            "scheduler": scheduler, "window_frames": window_frames,
            "overlap_frames": overlap_frames, "sliding_windows": sliding_windows,
            "window_frames_mode": window_frames_mode, "mode": mode,
            "keyframes": keyframes, "ref_image_size": ref_image_size,
            "ref_alpha": ref_alpha, "ref_image_batch": ref_image_batch,
            "ref_videos": ref_videos, "ref_video_audios": ref_video_audios,
            "ref_audios": ref_audios, "story_beats": story_beats,
            "transition_prompt": transition_prompt, "transition_mode": transition_mode,
            "enforce_cuts": enforce_cuts,
            "cross_shot_context": cross_shot_context,
            "cross_shot_context_frames": cross_shot_context_frames,
            "voice_continuity": voice_continuity,
            "audio_drive": audio_drive, "drive_audio": drive_audio,
        }
        values = {name: _unwrap_input(value) for name, value in values.items()}
        if values["output_mode"] not in {"video", "image"}:
            raise ValueError("output_mode must be video or image")
        if values["mode"] == "text":
            detached = [label for label in
                        ("ref_image_batch", "ref_videos", "ref_video_audios", "ref_audios", "keyframes")
                        if _input_present(values.get(label))]
            if detached:
                print("[FilmAuteurH3] \u26a0\ufe0f mode=text ignores " + ", ".join(detached)
                      + "; the prompt is rewritten without reference labels and repeats each "
                        "identity inline. Use mode=reference or all to render from them.")
        elif values["mode"] == "keyframes" and _input_present(values.get("ref_image_batch")):
            print("[FilmAuteurH3] \u26a0\ufe0f mode=keyframes ignores reference images and videos; "
                  "use mode=all to keep both.")
        image_mode = values["output_mode"] == "image"
        shot_plan, lengths_connected = _plan_prompt_sections(
            values["prompt"], values["story_beats"], shot_lengths, values["length_seconds"]
        )
        prompt_sections = len(_split_pipe_sections(values["prompt"])) > 1
        if prompt_sections:
            print(
                f"[FilmAuteurH3] prompt split into {len(shot_plan)} pipe-separated prompts; "
                "each one renders completely before the next begins."
            )
            if _shot_length_values(shot_lengths) is not None:
                print(
                    "[FilmAuteurH3] shot_lengths is ignored while the prompt is split; "
                    "each prompt uses its own story_beats and length_seconds."
                )
        length_frames = _seconds_to_frames(values["length_seconds"])
        if not lengths_connected and not image_mode and not prompt_sections:
            print(
                f"[FilmAuteurH3] length_seconds={values['length_seconds']}s "
                f"-> {length_frames} frames (24 fps, 17k+5 grid)"
            )

        if values["window_frames_mode"] not in {"automatic", "manual"}:
            raise ValueError("window_frames_mode must be automatic or manual")
        if (
            values["sliding_windows"]
            and values["window_frames_mode"] == "manual"
            and values["window_frames"] <= values["overlap_frames"]
        ):
            raise ValueError("Manual window_frames must be larger than overlap_frames")

        positive_outputs = []
        sampled_outputs = []
        initial_outputs = []
        image_outputs = []
        renders = []
        for section_index, (section_prompt, section_durations, section_beats) in enumerate(shot_plan, 1):
            for shot_index, (duration_seconds, shot_beats) in enumerate(
                zip(section_durations, section_beats), 1
            ):
                renders.append({
                    "prompt": _prompt_for_shot(section_prompt, shot_index, _beat_text(shot_beats)),
                    "duration": float(duration_seconds),
                    "beats": shot_beats,
                    "section": section_index,
                    "sections": len(shot_plan),
                    "shot": shot_index,
                    "shots": len(section_durations),
                })
        total_shots = len(renders)
        audio_drive = bool(values["audio_drive"]) and not image_mode
        if image_mode and values["audio_drive"]:
            print(
                "[FilmAuteurH3] image mode renders a still, so audio_drive is ignored "
                "and no drive_audio is needed."
            )
        drive_audio = values["drive_audio"]
        if audio_drive and drive_audio is None:
            raise ValueError(
                "audio_drive is enabled but drive_audio is not connected; connect the audio "
                "that should drive the video or turn audio_drive off"
            )
        shot_start_seconds = 0.0
        for render_index, render in enumerate(renders, 1):
            duration_seconds = render["duration"]
            shot_beats = render["beats"]
            shot_index = render["shot"]
            shot_prompt = render["prompt"]
            if values["mode"] == "text":
                shot_prompt = _text_only_prompt(shot_prompt)
            progress = (
                f"Prompt {render['section']}/{render['sections']}, "
                f"Shot {shot_index}/{render['shots']}"
                if prompt_sections else f"Shot {render_index}/{total_shots}"
            )
            if image_mode:
                requested_frames = 1
            else:
                requested_frames = (
                    length_frames
                    if not lengths_connected
                    else max(5, math.ceil(duration_seconds * FPS))
                )
            if values["window_frames_mode"] == "automatic":
                active_window_frames = _automatic_window_frames(
                    duration_seconds, values["overlap_frames"]
                )
            else:
                active_window_frames = int(values["window_frames"])

            if values["enforce_cuts"] and shot_index > 1:
                shot_prompt += (
                    "\nThis shot begins after a hard cut: it is an independent new shot with its "
                    "own framing, not a continuation of the previous shot's action. Any prior "
                    "footage supplied as a reference is used only for the same casting, wardrobe, "
                    "location and light, never to continue that action."
                )
            if image_mode:
                print(
                    f"[FilmAuteurH3] Generating {progress} "
                    "(single still frame)"
                )
            else:
                print(
                    f"[FilmAuteurH3] Generating {progress} "
                    f"({duration_seconds}s; window_frames={active_window_frames}; "
                    f"sliding_windows={bool(values['sliding_windows'])})"
                )

            cross_shot_video_reference = None
            cross_shot_video_audio = None
            if (values["cross_shot_context"] and render_index > 1 and shot_index > 1
                    and not image_mode and values["mode"] in {"reference", "all"}):
                previous_samples = sampled_outputs[-1]["samples"]
                cross_shot_video_reference = _tail_video_reference_frames(
                    previous_samples,
                    int(values["cross_shot_context_frames"]),
                    values["vae"],
                )
                if values["voice_continuity"]:
                    cross_shot_video_audio = _tail_audio_context(
                        previous_samples, int(values["cross_shot_context_frames"])
                    )

            conditioned = H3FL2VAandRef2VA.execute(
                clip=values["clip"],
                vae=values["vae"],
                audio_vae=values["audio_vae"],
                prompt=shot_prompt,
                width=values["width"],
                height=values["height"],
                length=requested_frames,
                mode=values["mode"],
                keyframes=values["keyframes"],
                ref_image_size=values["ref_image_size"],
                ref_alpha=values["ref_alpha"],
                ref_image_batch=values["ref_image_batch"],
                ref_videos=values["ref_videos"],
                ref_video_audios=values["ref_video_audios"],
                ref_audios=values["ref_audios"],
                cross_shot_video_reference=cross_shot_video_reference,
                cross_shot_video_audio=cross_shot_video_audio,
            )
            positive, initial_latent = conditioned.args
            if image_mode:
                initial_latent = _still_av_latent(initial_latent)
            shot_positive = positive
            shot_audio_guide = None
            if audio_drive:
                shot_frame_count = video_frames_for_latents(
                    initial_latent["samples"].tensors[0].shape[2]
                )
                audio_axis = int(initial_latent["samples"].tensors[1].shape[-1])
                shot_audio_guide = _audio_drive_guide(
                    values["audio_vae"],
                    drive_audio,
                    shot_start_seconds,
                    shot_frame_count,
                    audio_axis,
                )
                shot_positive = _apply_audio_guide(shot_positive, shot_audio_guide)
                print(
                    f"[FilmAuteurH3] audio_drive: {progress} audio source is the provided "
                    f"track {shot_start_seconds:.2f}s-{shot_start_seconds + shot_frame_count / FPS:.2f}s"
                )
            sampled = H3SlidingContextSampler.execute(
                model=values["model"],
                positive=shot_positive,
                latent=initial_latent,
                clip=values["clip"],
                story_beats=shot_beats,
                seed=(int(values["seed"]) + render_index - 1) % (1 << 64),
                steps=values["steps"],
                sampler_name=values["sampler_name"],
                scheduler=values["scheduler"],
                window_frames=active_window_frames,
                overlap_frames=values["overlap_frames"],
                transition_prompt=values["transition_prompt"],
                transition_mode=values["transition_mode"],
                sliding_windows=bool(values["sliding_windows"]) and not image_mode,
                # Beat timestamps are micro-direction only. Each shot is sampled
                # independently, so only the shot boundary is a hard cut.
                enforce_cuts=False,
            )
            (sampled_latent,) = sampled.args
            if shot_audio_guide is not None:
                sampled_latent = _replace_audio_stream(
                    sampled_latent, shot_audio_guide["audio_latent"]
                )
            if image_mode:
                image_outputs.append(_decode_still(values["vae"], sampled_latent))
            positive_outputs.append(shot_positive)
            sampled_outputs.append(sampled_latent)
            initial_outputs.append(initial_latent)
            shot_start_seconds += float(duration_seconds)

        images = (
            torch.cat(image_outputs, dim=0)
            if image_outputs
            else torch.zeros((0, 1, 1, 3))
        )
        return io.NodeOutput(positive_outputs, sampled_outputs, initial_outputs, images)


class FilmAuteurH3Extension(ComfyExtension):
    async def get_node_list(self):
        return [FilmAuteurH3]


async def comfy_entrypoint() -> FilmAuteurH3Extension:
    return FilmAuteurH3Extension()

# --- moved from filmauteur_ltx.py:FilmAuteur_LTX ----------------------------
import torch
import math
import comfy.utils
import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import comfy.sample
import comfy.patcher_extension
import re
import latent_preview
import urllib.request
import json
import base64
import requests
import os
import cv2
import folder_paths
from io import BytesIO
from PIL import Image
import numpy as np
import torchaudio
import types
import logging
import av
import random

# ==========================================
# COLORFX INJECTIONS
# ==========================================
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageChops, ImageDraw
import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from scipy.interpolate import RegularGridInterpolator
    SCIPY_AVAILABLE = True
    print("INFO: Scipy found. LUT application will be optimized.")
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: Scipy not found. LUT application will use a slower pixel-by-pixel method if enabled.")

try:
    dir_luts = os.path.join(folder_paths.models_dir, "luts")
    os.makedirs(dir_luts, exist_ok=True)
    if "luts" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["luts"] = ([dir_luts], {".cube"})
    else:
        folder_paths.folder_names_and_paths["luts"][1].add(".cube")
        if dir_luts not in folder_paths.folder_names_and_paths["luts"][0]:
             folder_paths.folder_names_and_paths["luts"][0].append(dir_luts)
except Exception as e:
    pass

def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

from server import PromptServer
from comfy.ldm.modules.attention import wrap_attn, optimized_attention, attention_pytorch

from fractions import Fraction
try:
    from comfy_api.latest import InputImpl, Types
except ImportError:
    InputImpl = None
    Types = None

# ==========================================
# CUSTOM TRIXOPE DIRECTORY SETUP
# ==========================================
current_dir = os.path.dirname(os.path.realpath(__file__))
trixope_facerestore_dir = os.path.join(current_dir, "facerestore_models")
os.makedirs(trixope_facerestore_dir, exist_ok=True)
folder_paths.folder_names_and_paths["trixope_facerestore"] = ([trixope_facerestore_dir], folder_paths.supported_pt_extensions)

def get_trixope_facerestore_models():
    models = folder_paths.get_filename_list("trixope_facerestore")
    return models if models else ["None"]

# ==========================================
# OLLAMA API FETCH (Builds the Dropdown at Boot)
# ==========================================
def get_ollama_models():
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as response:
            data = json.loads(response.read().decode('utf-8'))
            models = [model['name'] for model in data.get('models', [])]
            return models if models else ["llama3.2-vision:latest", "llava:latest"]
    except Exception:
        return ["llama3.2-vision:latest", "(Start Ollama & Restart ComfyUI)"]

OLLAMA_MODELS = get_ollama_models()

# ==========================================
# SAGE ATTENTION CORE
# ==========================================
sageattn_modes = ["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3", "sageattn3_per_block_mean"]

def get_sage_func(sage_attention, allow_compile=False):
    import logging
    logging.info(f"Using sage attention mode: {sage_attention}")
    from sageattention import sageattn
    if sage_attention == "auto":
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda++":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout)
    elif "sageattn3" in sage_attention:
        from sageattn3 import sageattn3_blackwell
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD", **kwargs):
            q, k, v = [x.transpose(1, 2) if tensor_layout == "NHD" else x for x in (q, k, v)]
            out = sageattn3_blackwell(q, k, v, is_causal=is_causal, attn_mask=attn_mask, per_block_mean=(sage_attention == "sageattn3_per_block_mean"))
            return out.transpose(1, 2) if tensor_layout == "NHD" else out

    if not allow_compile:
        sage_func = torch.compiler.disable()(sage_func)

    @wrap_attn
    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        if kwargs.get("low_precision_attention", True) is False:
            return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        if skip_reshape:
            b, _, _, dim_head = q.shape
            tensor_layout="HND"
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(
                lambda t: t.view(b, -1, heads, dim_head),
                (q, k, v),
            )
            tensor_layout="NHD"
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        else:
            if skip_output_reshape:
                out = out.transpose(1, 2)
            else:
                out = out.reshape(b, -1, heads * dim_head)
        return out
    return attention_sage

# ==========================================
# CHUNK FFN CORE
# ==========================================
def ffn_chunked_forward(self_module, x):
    if x.shape[1] > self_module.dim_threshold:
        chunk_size = x.shape[1] // self_module.num_chunks
        for i in range(self_module.num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self_module.num_chunks - 1 else x.shape[1]
            x[:, start_idx:end_idx] = self_module.net(x[:, start_idx:end_idx])
        return x
    else:
        return self_module.net(x)

class LTXVffnChunkPatch:
    def __init__(self, num_chunks, dim_threshold=4096):
        self.num_chunks = num_chunks
        self.dim_threshold = dim_threshold
    def __get__(self, obj, objtype=None):
        def wrapped_forward(self_module, *args, **kwargs):
            self_module.num_chunks = self.num_chunks
            self_module.dim_threshold = self.dim_threshold
            return ffn_chunked_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_forward, obj)

# ==========================================
# NORMALIZED ATTENTION GUIDANCE (NAG) CORE
# ==========================================
def _compute_attention(self, query, context, attn_precision=None, transformer_options={}):
    k = self.k_norm(self.to_k(context)).to(query.dtype)
    v = self.to_v(context).to(query.dtype)
    x = comfy.ldm.modules.attention.optimized_attention(query, k, v, heads=self.heads, attn_precision=attn_precision, transformer_options=transformer_options).flatten(2)
    del k, v
    return x

def nag_attention(self, query, context_positive, nag_context, attn_precision=None, transformer_options={}):
    x_positive = _compute_attention(self, query, context_positive, attn_precision, transformer_options)
    x_negative = _compute_attention(self, query, nag_context, attn_precision, transformer_options)
    return x_positive, x_negative

def normalized_attention_guidance(self, x_positive, x_negative):
    if self.inplace:
        nag_guidance = x_negative.mul_(self.nag_scale - 1).neg_().add_(x_positive, alpha=self.nag_scale)
    else:
        nag_guidance = x_positive * self.nag_scale - x_negative * (self.nag_scale - 1)

    del x_negative

    norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
    norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True)

    scale = norm_guidance / norm_positive
    torch.nan_to_num_(scale, nan=10.0)
    mask = scale > self.nag_tau
    del scale

    adjustment = (norm_positive * self.nag_tau) / (norm_guidance + 1e-7)
    del norm_positive, norm_guidance

    nag_guidance.mul_(torch.where(mask, adjustment, 1.0))
    del mask, adjustment

    if self.inplace:
        nag_guidance.sub_(x_positive).mul_(self.nag_alpha).add_(x_positive)
    else:
        nag_guidance = nag_guidance * self.nag_alpha + x_positive * (1 - self.nag_alpha)
    del x_positive

    return nag_guidance

def ltxv_crossattn_forward_nag(self, x, context, mask=None, transformer_options={}, **kwargs):
    if context.shape[0] == 1:
        x_pos, context_pos = x, context
        x_neg, context_neg = None, None
    else:
        x_pos, x_neg = torch.chunk(x, 2, dim=0)
        context_pos, context_neg = torch.chunk(context, 2, dim=0)

    q_pos = self.q_norm(self.to_q(x_pos))
    del x_pos

    x_positive, x_negative = nag_attention(self, q_pos, context_pos, self.nag_context, attn_precision=self.attn_precision, transformer_options=transformer_options)
    del context_pos, q_pos

    x_pos_out = normalized_attention_guidance(self, x_positive, x_negative)
    del x_positive, x_negative

    if x_neg is not None and context_neg is not None:
        q_neg = self.q_norm(self.to_q(x_neg))
        k_neg = self.k_norm(self.to_k(context_neg))
        v_neg = self.to_v(context_neg)

        x_neg_out = comfy.ldm.modules.attention.optimized_attention(q_neg, k_neg, v_neg, heads=self.heads, attn_precision=self.attn_precision, transformer_options=transformer_options)
        out = torch.cat([x_pos_out, x_neg_out], dim=0)
    else:
        out = x_pos_out

    if self.to_gate_logits is not None:
        gate_logits = self.to_gate_logits(x)  
        b, t, _ = out.shape
        out = out.view(b, t, self.heads, self.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits) 
        out = out * gates.unsqueeze(-1)
        out = out.view(b, t, self.heads * self.dim_head)

    return self.to_out(out)

class LTXVCrossAttentionPatch:
    def __init__(self, context, nag_scale, nag_alpha, nag_tau, inplace=True):
        self.nag_context = context
        self.nag_scale = nag_scale
        self.nag_alpha = nag_alpha
        self.nag_tau = nag_tau
        self.inplace = inplace

    def __get__(self, obj, objtype=None):
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.nag_context = self.nag_context
            self_module.nag_scale = self.nag_scale
            self_module.nag_alpha = self.nag_alpha
            self_module.nag_tau = self.nag_tau
            self_module.inplace = self.inplace
            return ltxv_crossattn_forward_nag(self_module, *args, **kwargs)
        return types.MethodType(wrapped_attention, obj)

# ==========================================
# HELPER DECODE FUNCTIONS
# ==========================================
def compute_chunk_boundaries(chunk_start: int, temporal_tile_length: int, temporal_overlap: int, total_latent_frames: int):
    if chunk_start == 0:
        chunk_end = min(chunk_start + temporal_tile_length, total_latent_frames)
        overlap_start = chunk_start
    else:
        overlap_start = max(1, chunk_start - temporal_overlap - 1)
        extra_frames = chunk_start - overlap_start
        chunk_end = min(chunk_start + temporal_tile_length - extra_frames, total_latent_frames)
    return overlap_start, chunk_end

def calculate_temporal_output_boundaries(overlap_start: int, time_scale_factor: int, tile_out_frames: int):
    out_t_start = 1 + overlap_start * time_scale_factor
    out_t_end = out_t_start + tile_out_frames
    return out_t_start, out_t_end

class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent.get("batch_index", None)
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)

def encode_single_frame(output_file, image_array: np.ndarray, crf):
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height = image_array.shape[0]
        stream.width = image_array.shape[1]
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()

def decode_single_frame(video_file):
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")

def preprocess_compression(image: torch.Tensor, crf=29):
    if crf == 0:
        return image
    image_array = (image[:(image.shape[0] // 2) * 2, :(image.shape[1] // 2) * 2] * 255.0).byte().cpu().numpy()
    with BytesIO() as output_file:
        encode_single_frame(output_file, image_array, crf)
        video_bytes = output_file.getvalue()
    with BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    tensor = torch.tensor(image_array, dtype=image.dtype, device=image.device) / 255.0
    return tensor

def safe_vae_decode(vae_model, latent_tile):
    with torch.no_grad():
        comfy.model_management.soft_empty_cache()
        v_lats = latent_tile.shape[2] if (hasattr(latent_tile, "shape") and latent_tile.ndim >= 3) else 16
        if hasattr(vae_model, "decode_tiled"):
            for tile_size in [512, 256, 128]:
                try:
                    # Keep tile_t equal to full temporal length so VAE tiling is purely spatial (X/Y)
                    return vae_model.decode_tiled(
                        latent_tile,
                        tile_x=tile_size,
                        tile_y=tile_size,
                        tile_t=v_lats,
                        overlap=2
                    )
                except Exception as e:
                    comfy.model_management.soft_empty_cache()
        return vae_model.decode(latent_tile)

RESOLUTION_OPTIONS = [
    "--- 𝟭:𝟭 𝗔𝘀𝗽𝗲𝗰𝘁 𝗥𝗮𝘁𝗶𝗼 (𝗦𝗾𝘂𝗮𝗿𝗲) ---",
    "512\u00A0 × 512\u00A0 — Fast low-VRAM generation",
    "768\u00A0 × 768\u00A0 — Balanced resolution sweet spot",
    "1024 × 1024 — Standard high-definition square layout",
    "1536 × 1536 — Pro fidelity upscale target",
    "--- 𝟰:𝟯 𝗔𝘀𝗽𝗲𝗰𝘁 𝗥𝗮𝘁𝗶𝗼 (𝗦𝘁𝗮𝗻𝗱𝗮𝗿𝗱 𝗟𝗮𝗻𝗱𝘀𝗰𝗮𝗽𝗲) ---",
    "512\u00A0 × 384\u00A0 — Low-resolution draft option",
    "1024 × 768\u00A0 — Exact native classic frame",
    "1536 × 1152 — Sharp high-resolution landscape grid",
    "2048 × 1536 — Ultra-high density presentation resolution",
    "--- 𝟯:𝟰 𝗔𝘀𝗽𝗲𝗰𝘁 𝗥𝗮𝘁𝗶𝗼 (𝗦𝘁𝗮𝗻𝗱𝗮𝗿𝗱 𝗣𝗼𝗿𝘁𝗿𝗮𝗶𝘁) ---",
    "384\u00A0 × 512\u00A0 — Low-resolution vertical layout",
    "768\u00A0 × 1024 — Exact vertical presentation shape",
    "1152 × 1536 — High-definition mobile portrait preview",
    "1536 × 2048 — Maximum density vertical layout",
    "--- 𝟭𝟲:𝟵 𝗔𝘀𝗽𝗲𝗰𝘁 𝗥𝗮𝘁𝗶𝗼 (𝗪𝗶𝗱𝗲𝘀𝗰𝗿𝗲𝗲𝗻) ---",
    "1152 × 640\u00A0 — High-speed widescreen draft approximation",
    "1792 × 1024 — Close standard HD variation",
    "2048 × 1152 — Exact 16:9 math configuration",
    "3840 × 2176 — Immersive 4K UHD approximation",
    "--- 𝟵:𝟭𝟲 𝗔𝘀𝗽𝗲𝗰𝘁 𝗥𝗮𝘁𝗶𝗼 (𝗩𝗲𝗿𝘁𝗶𝗰𝗮𝗹 𝗪𝗶𝗱𝗲𝘀𝗰𝗿𝗲𝗲𝗻) ---",
    "640\u00A0 × 1152 — Fast vertical mobile approximation",
    "1024 × 1792 — Optimized social reel layout",
    "1152 × 2048 — Exact native 9:16 configuration",
    "2176 × 3840 — Ultra-high definition mobile layout",
    "--- 𝟮𝟭:𝟵 𝗔𝘀𝗽𝗲𝗰𝘁 𝗥𝗮𝘁𝗶𝗼 (𝗖𝗶𝗻𝗲𝗺𝗮𝘁𝗶𝗰 𝗨𝗹𝘁𝗿𝗮𝘄𝗶𝗱𝗲) ---",
    "896\u00A0 × 384\u00A0 — Low-VRAM horizontal cinematic draft",
    "1792 × 768\u00A0 — Exact cinematic widescreen resolution",
    "2688 × 1152 — Immersive high-definition panoramic target",
    "3584 × 1536 — Maximum resolution multi-window layout",
    "--- 𝟵:𝟮𝟭 𝗔𝘀𝗽𝗲𝗰𝘁 𝗥𝗮𝘁𝗶𝗼 (𝗩𝗲𝗿𝘁𝗶𝗰𝗮𝗹 𝗨𝗹𝘁𝗿𝗮𝘄𝗶𝗱𝗲) ---",
    "384\u00A0 × 896\u00A0 — Tall smartphone display layout",
    "768\u00A0 × 1792 — Exact vertical cinematic layout",
    "1152 × 2688 — Ultra-tall narrow video layout",
    "1536 × 3584 — Maximum resolution panoramic portrait"
]

class FilmAuteur_LTX:

    @classmethod
    def INPUT_TYPES(cls):
        lut_files = ["None"]
        try:
            raw_files = folder_paths.get_filename_list("luts")
            if raw_files:
                 lut_files.extend([f for f in raw_files if f.lower().endswith('.cube')])
        except Exception as e: pass
        
        sampler_names = comfy.samplers.SAMPLER_NAMES
        primary_default = "euler_ancestral" if "euler_ancestral" in sampler_names else sampler_names[0]
        upsample_default = "euler_cfg_pp" if "euler_cfg_pp" in sampler_names else sampler_names[0]

        return {
            "required": {
                # --- Input/Setup ---
                "clip": ("CLIP",),
                "video_vae": ("VAE", {"tooltip": "The LTXV Video VAE model."}),
                "audio_vae": ("VAE", {"tooltip": "The LTXV Audio VAE model."}),

                # --- GROUP: Mode Activation ---
                "grp_mode": (["▼ Mode Activation"], {}),
                "primary_sampling": ("BOOLEAN", {"default": True}),
                "spatial_upscale": ("BOOLEAN", {"default": True}),
                "temporal_upscale": ("BOOLEAN", {"default": True}),
                "restore_faces": ("BOOLEAN", {"default": True}),
                "enable_colorfx": ("BOOLEAN", {"default": True}),
                "enable_preview": ("BOOLEAN", {"default": True}),
                "enable_final_video": ("BOOLEAN", {"default": True}),
                "video_mode": (["text-to-video", "image-to-video", "reference-to-video"], {"default": "text-to-video"}),
                "image_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "img_compression": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "audio_select": (["internal", "source", "reference"], {"default": "internal"}),
                "identity_guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),

                # --- GROUP: Prompting ---
                "grp_prompting": (["▼ Prompting"], {}),
                "character_descriptions": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "location_description": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "scene_descriptions": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "use_ollama": (["disabled", "enhance prompt", "song-to-script"], {"default": "disabled"}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ollama_model": (OLLAMA_MODELS,),

                # --- GROUP: Specs ---
                "grp_specs": (["▼ Video Specs"], {}),
                "seed_number": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "control_before_generate": (["randomize", "fixed", "increment", "decrement"], {"default": "randomize"}),
                "target_resolution": (RESOLUTION_OPTIONS, {"default": "1792 × 768\u00A0 — Exact cinematic widescreen resolution"}),
                "length_in_seconds": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),

                # --- GROUP: Primary Sampling ---
                "grp_sampling": (["▼ Primary Sampling"], {}),
                "primary_sampler_name": (sampler_names, {"default": primary_default}),
                "primary_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "primary_steps": ("STRING", {"multiline": False, "default": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.881203, 0.863321, 0.841251, 0.820089, 0.655, 0.381875, 0.0"}),
                "eta": ("FLOAT", {"default": 0.95, "min": -100.0, "max": 100.0, "step": 0.01, "round": False}),
                "bongmath": ("BOOLEAN", {"default": True}),
                "enable_nag": ("BOOLEAN", {"default": True}),

                # --- GROUP: Upscale & Refine ---
                "grp_refinement": (["▼ Upscale & Refine"], {}),
                "spatial_passes": ("INT", {"default": 1, "min": 1, "max": 2}),
                "spatial_sampler": (sampler_names, {"default": upsample_default}),
                "spatial_cfg": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "spatial_sigmas": ("STRING", {"multiline": False, "default": "0.55, 0.35, 0.15, 0.0"}),
                "temporal_denoise": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 1.0, "step": 0.01}),
                "facerestore_model": (get_trixope_facerestore_models(), {}),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"], {}),
                "codeformer_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "face_restore_color_match": ("BOOLEAN", {"default": True}),
                "face_restore_edge_blur": ("BOOLEAN", {"default": True}),
                "face_restore_blend": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),

                # --- GROUP: Color FX (Master Parent) ---
                "grp_cfx_main": (["▼ Color FX"], {}),
                "enable_color_correction": ("BOOLEAN", {"default": True}),
                "enable_lut_processing": ("BOOLEAN", {"default": True}),
                "enable_enhancements": ("BOOLEAN", {"default": True}),
                "enable_blur_effects": ("BOOLEAN", {"default": True}),
                "enable_stylistic_effects": ("BOOLEAN", {"default": True}),
                
                # --- SUBGROUP: Color FX (Correction) ---
                "grp_cfx_color": (["▼ Color Correction"], {}),
                "hdr_intensity": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "shadow_intensity": ("FLOAT", {"default": 0.10, "min": 0.00, "max": 2.00, "step": 0.01}),
                "highlight_intensity": ("FLOAT", {"default": 0.20, "min": 0.00, "max": 2.00, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.00, "min": 0.10, "max": 5.00, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 3.00, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 0.75, "min": 0.00, "max": 3.00, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 0.90, "min": 0.00, "max": 3.00, "step": 0.01}),
                "enhance_color": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 3.00, "step": 0.01}),

                # --- SUBGROUP: Color FX (LUTs) ---
                "grp_cfx_lut": (["▼ LUT Processing"], {}),
                "lut_name": (lut_files, {"default": "None"}),
                "lut_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lut_log_process": ("BOOLEAN", {"default": True}),

                # --- SUBGROUP: Color FX (Enhancements) ---
                "grp_cfx_enhancements": (["▼ Enhancements"], {}),
                "sharpness": ("FLOAT", {"default": 3.0, "min": -2.0, "max": 5.0, "step": 0.1}),
                "edge_enhance_strength": ("FLOAT", {"default": 0.10, "min": 0.00, "max": 1.00, "step": 0.01}),
                "detail_enhance_strength": ("FLOAT", {"default": 0.20, "min": 0.00, "max": 1.00, "step": 0.01}),

                # --- SUBGROUP: Color FX (Blur) ---
                "grp_cfx_blur": (["▼ Blur Effects"], {}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "gaussian_blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.1}),
                "radial_blur_strength": ("FLOAT", {"default": 32.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "radial_blur_center_x": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "radial_blur_center_y": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "radial_blur_focus_spread": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "radial_blur_steps": ("INT", {"default": 10, "min": 1, "max": 32, "step": 1}),

                # --- SUBGROUP: Color FX (Stylistic) ---
                "grp_cfx_stylistic": (["▼ Stylistic Effects"], {}),
                "chromatic_aberration_r_x": ("FLOAT", {"default": 1.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_aberration_r_y": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_aberration_b_x": ("FLOAT", {"default": -1.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_aberration_b_y": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_blur_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "simple_film_grain_intensity": ("FLOAT", {"default": 0.04, "min": 0.00, "max": 1.00, "step": 0.01}),
                "simple_film_grain_monochrome": ("BOOLEAN", {"default": True}),
                "scanline_intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0}),
                "vignette_intensity": ("FLOAT", {"default": 0.40, "min": 0.00, "max": 1.00, "step": 0.01}),
                "vignette_center_x": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "vignette_center_y": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "soft_light_opacity": ("FLOAT", {"default": 0.30, "min": 0.00, "max": 1.00, "step": 0.01}),
                "soft_light_blur_radius": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1}),

                # --- GROUP: Performance ---
                "grp_performance": (["▼ Performance"], {}),
                "enable_fp16_accumulation": ("BOOLEAN", {"default": True}),
                "sage_attention": (sageattn_modes, {"default": "auto"}),
                "autoregressive_chunking": ("BOOLEAN", {"default": True}),
                "chunk_size_seconds": ("FLOAT", {"default": 10.0, "min": 5.0, "max": 300.0, "step": 1.0}),
                "context_window_seconds": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 300.0, "step": 1.0}),
                "chunks_feedforward": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "clear_models_and_cache": ("BOOLEAN", {"default": True}),

                # --- GROUP: Output ---
                "grp_output": (["▼ Output"], {}),
            },
            "optional": {
                "model1_primary": ("MODEL", {"tooltip": "The primary LTXV Model (will be patched if ID-LoRA is active).", "forceInput": True}),
                "model1.5_extend": ("MODEL", {"tooltip": "Optional model for single shot extensions.", "forceInput": True}),
                "model2_spatial": ("MODEL", {"tooltip": "Optional model for upsample stages 2 and 3. If disconnected, model1_primary is used.", "forceInput": True}),
                "spatial_upscaler": ("LATENT_UPSCALE_MODEL", {"tooltip": "Connect the LTXV Spatial Upscale model here to upsample the video latent by 2x.", "forceInput": True}),
                "model3_temporal": ("MODEL", {"tooltip": "Optional model for the temporal upscaler. If disconnected, model1_primary is used.", "forceInput": True}),
                "temporal_upscaler": ("LATENT_UPSCALE_MODEL", {"tooltip": "Connect the LTXV Temporal Upscale model here to double the framerate.", "forceInput": True}),
                "positive": ("CONDITIONING", {"tooltip": "Optional override for the positive text conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Optional override for the negative text conditioning."}),
                "image(s)": ("IMAGE", {"tooltip": "Input image batch for image-to-video or reference-to-video."}),
                "audio": ("AUDIO", {"tooltip": "Connect audio here to encode it directly or use it as a voice reference for ID-LoRA."}),
                "latent_override": ("LATENT", {"tooltip": "Optional latent override. Bypasses generation if primary_sampling is off, or acts as the base for Video-to-Video if on."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("STRING", "CONDITIONING", "CONDITIONING", "LATENT", "LATENT", "LATENT", "VIDEO", "IMAGE", "AUDIO", "FLOAT", "INT", "ETA_TRACKER")
    RETURN_NAMES = ("text_prompt(s)", "positive", "negative", "av_latent", "video_latent", "audio_latent", "video", "images", "audio", "fps", "ref_frame_count", "real_time_eta")
    FUNCTION = "process"
    CATEGORY = "triXope/film"

    def _read_cube_file(self, filepath):
        lines = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'): lines.append(line)
        lut_size = 0
        domain_min, domain_max = np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)
        table_data_lines = []
        for line in lines:
            if line.startswith('LUT_3D_SIZE'): lut_size = int(line.split()[-1])
            elif line.startswith('DOMAIN_MIN'): domain_min = np.array([float(x) for x in line.split()[1:]], dtype=np.float32)
            elif line.startswith('DOMAIN_MAX'): domain_max = np.array([float(x) for x in line.split()[1:]], dtype=np.float32)
            elif len(line.split()) == 3 and not line.startswith('TITLE'): table_data_lines.append(list(map(float, line.split())))
        return np.array(table_data_lines, dtype=np.float32).reshape(lut_size, lut_size, lut_size, 3), lut_size, domain_min, domain_max

    def _apply_lut_to_image_scipy(self, image_np_rgb_0_1, lut_data):
        lut_table, lut_size, _, _ = lut_data
        r_axis = g_axis = b_axis = np.linspace(0, 1, lut_size)
        interp_R = RegularGridInterpolator((b_axis, g_axis, r_axis), lut_table[..., 0], bounds_error=False)
        interp_G = RegularGridInterpolator((b_axis, g_axis, r_axis), lut_table[..., 1], bounds_error=False)
        interp_B = RegularGridInterpolator((b_axis, g_axis, r_axis), lut_table[..., 2], bounds_error=False)
        pts = image_np_rgb_0_1[..., [2, 1, 0]].reshape(-1, 3)
        return np.clip(np.stack([interp_R(pts), interp_G(pts), interp_B(pts)], axis=-1).reshape(image_np_rgb_0_1.shape), 0.0, 1.0)

    def _apply_lut_effect(self, pil_image, lut_name, strength, log_process):
        if strength == 0.0 or lut_name == "None": return pil_image
        lut_path = folder_paths.get_full_path("luts", lut_name)
        if not lut_path or not SCIPY_AVAILABLE: return pil_image
        lut_data = self._read_cube_file(lut_path)
        img_np = np.array(pil_image, dtype=np.float32) / 255.0
        orig = img_np.copy()
        if log_process: img_np = np.power(np.clip(img_np, 1e-5, 1.0), 1.0 / 2.2)
        out_np = self._apply_lut_to_image_scipy(img_np, lut_data)
        if log_process: out_np = np.power(np.clip(out_np, 0.0, 1.0), 2.2)
        blended = np.clip((1.0 - strength) * orig + strength * out_np, 0.0, 1.0)
        return Image.fromarray((blended * 255).astype(np.uint8))

    def _apply_shadows_highlights(self, pil_image, shadow_adj, highlight_adj, hdr_intensity):
        if shadow_adj == 0.0 and highlight_adj == 0.0 and hdr_intensity == 1.0: return pil_image
        img_array = np.array(pil_image, dtype=np.float32) / 255.0
        if hdr_intensity != 1.0:
            mean = np.mean(img_array)
            img_array = mean + (img_array - mean) * (1.0 + (hdr_intensity - 1.0) * 0.5) 
        lum = np.clip(0.299 * img_array[..., 0] + 0.587 * img_array[..., 1] + 0.114 * img_array[..., 2], 0.0, 1.0)[..., np.newaxis]
        if shadow_adj > 0.0: img_array += (1.0 - lum) * shadow_adj
        if highlight_adj > 0.0: img_array -= lum * highlight_adj
        return Image.fromarray((np.clip(img_array, 0.0, 1.0) * 255).astype(np.uint8))

    def _apply_color_enhancements(self, pil_image, b, c, s, ec):
        if b != 1.: pil_image = ImageEnhance.Brightness(pil_image).enhance(b)
        if c != 1.: pil_image = ImageEnhance.Contrast(pil_image).enhance(c)
        if s * ec != 1.: pil_image = ImageEnhance.Color(pil_image).enhance(s * ec)
        return pil_image

    def _apply_sharpness_detail(self, pil_image, sh, ee, de):
        if sh != 0.: pil_image = ImageEnhance.Sharpness(pil_image).enhance(1. + sh)
        if ee > 0.: pil_image = Image.blend(pil_image, pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE), ee)
        if de > 0.: pil_image = Image.blend(pil_image, pil_image.filter(ImageFilter.DETAIL), de)
        return pil_image

    def _apply_blurs(self, pil_image, br, gbr):
        if br > 0: pil_image = pil_image.filter(ImageFilter.BoxBlur(br))
        if gbr > 0.: pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=gbr))
        return pil_image

    def _apply_chromatic_aberration(self, pil_image, rx, ry, bx, by, blur):
        img = np.array(pil_image).astype(np.float32)
        rc, gc, bc = img[...,0], img[...,1], img[...,2]
        if blur > 0.: 
            bk = int(blur * 2) * 2 + 1
            rc, bc = cv2.GaussianBlur(rc, (bk, bk), blur), cv2.GaussianBlur(bc, (bk, bk), blur)
        rs = cv2.warpAffine(rc, np.float32([[1, 0, rx], [0, 1, ry]]), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
        bs = cv2.warpAffine(bc, np.float32([[1, 0, bx], [0, 1, by]]), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
        return Image.fromarray(np.clip(cv2.merge([rs, gc, bs]), 0, 255).astype(np.uint8))

    def _apply_radial_blur(self, pil_image, strength, cx_r, cy_r, spread, steps):
        img = np.array(pil_image).astype(np.float32) / 255.
        h, w = img.shape[:2]; cx, cy = int(w * cx_r), int(h * cy_r)
        max_d = np.sqrt(max(cx**2+cy**2, (w-cx)**2+cy**2, cx**2+(h-cy)**2, (w-cx)**2+(h-cy)**2))
        X, Y = np.meshgrid(np.arange(w) - cx, np.arange(h) - cy)
        rad_mask = np.sqrt(X**2 + Y**2) / max(1., max_d)
        blurred, cur_blur = [], strength
        for _ in range(steps):
            k = int(cur_blur)
            blurred.append(cv2.GaussianBlur(img, (k | 1, k | 1), 0) if k > 0 else img.copy())
            cur_blur /= spread
        final = np.zeros_like(img)
        for i in range(steps):
            m_i = np.dstack([np.clip((rad_mask - i / steps) * steps, 0, 1)] * 3)
            final = blurred[steps - 1 - i] * m_i + (final if i > 0 else img) * (1. - m_i)
        return Image.fromarray(np.clip(final * 255, 0, 255).astype(np.uint8))

    def _apply_scanlines(self, pil_image, intensity):
        img = np.array(pil_image)
        scan = np.ones(img.shape[:2], dtype=img.dtype)
        scan[::2, :] = 1. - intensity
        return Image.fromarray((img * np.dstack([scan]*3)).astype(np.uint8))

    def _apply_soft_light(self, pil_image, opacity, blur_radius):
        blur = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return Image.blend(pil_image, ImageChops.soft_light(pil_image, blur), opacity)

    def _apply_gamma_torch(self, tensor_images, gamma_value):
        if gamma_value == 1.0: return tensor_images
        return torch.pow(tensor_images.clamp(1e-5, 1.0), 1.0 / max(gamma_value, 0.01))

    def _apply_simple_film_grain_torch(self, tensor_images, intensity, monochrome):
        if intensity == 0.: return tensor_images
        B, H, W, C = tensor_images.shape
        noise = torch.randn((B, H, W, 1 if monochrome else C), device=tensor_images.device) * intensity
        if monochrome and C > 1: noise = noise.repeat(1, 1, 1, C)
        return (tensor_images + noise).clamp(0.0, 1.0)

    def _apply_vignette_torch(self, tensor_images, intensity, cx_r, cy_r):
        if intensity == 0.: return tensor_images
        B, H, W, C = tensor_images.shape
        cx, cy = W * cx_r, H * cy_r
        Y, X = torch.meshgrid(torch.arange(H, device=tensor_images.device) - cy, torch.arange(W, device=tensor_images.device) - cx, indexing='ij')
        dist = torch.sqrt(X**2 + Y**2)
        max_dist = torch.sqrt(torch.tensor(max(cx**2+cy**2, (W-cx)**2+cy**2, cx**2+(H-cy)**2, (W-cx)**2+(H-cy)**2), device=tensor_images.device))
        v_mask = (1.0 - (dist / max(1.0, max_dist))**2 * intensity).clamp(0.0, 1.0).unsqueeze(0).unsqueeze(-1)
        return (tensor_images * v_mask).clamp(0.0, 1.0)

    def process(self, clip, video_vae, audio_vae, character_descriptions, location_description, scene_descriptions,
                primary_sampling, spatial_upscale, temporal_upscale, restore_faces, enable_colorfx, video_mode, image_strength, img_compression, audio_select, identity_guidance_scale,
                use_ollama, ollama_url, ollama_model,
                seed_number, control_before_generate, target_resolution, length_in_seconds, frame_rate,
                primary_sampler_name, primary_cfg, primary_steps,
                eta, bongmath, enable_nag,
                spatial_passes, spatial_sampler, spatial_cfg, spatial_sigmas, temporal_denoise,
                facerestore_model, facedetection, codeformer_fidelity,
                face_restore_color_match, face_restore_edge_blur, face_restore_blend,
                enable_color_correction, hdr_intensity, shadow_intensity, highlight_intensity, gamma, brightness, contrast, saturation, enhance_color,
                enable_lut_processing, lut_name, lut_strength, lut_log_process,
                enable_enhancements, sharpness, edge_enhance_strength, detail_enhance_strength,
                enable_blur_effects, blur_radius, gaussian_blur_radius, radial_blur_strength, radial_blur_center_x, radial_blur_center_y, radial_blur_focus_spread, radial_blur_steps,
                enable_stylistic_effects, chromatic_aberration_r_x, chromatic_aberration_r_y, chromatic_aberration_b_x, chromatic_aberration_b_y, chromatic_blur_amount, simple_film_grain_intensity, simple_film_grain_monochrome, scanline_intensity, vignette_intensity, vignette_center_x, vignette_center_y, soft_light_opacity, soft_light_blur_radius,
                enable_fp16_accumulation, sage_attention, autoregressive_chunking, chunk_size_seconds, context_window_seconds, chunks_feedforward, clear_models_and_cache, enable_preview, enable_final_video=True,
                model1_primary=None, model2_spatial=None, spatial_upscaler=None, model3_temporal=None, temporal_upscaler=None,
                positive=None, negative=None, audio=None, latent_override=None, unique_id=None, **kwargs):

        images = kwargs.get("image(s)", None)

        unique_id_val = unique_id[0] if isinstance(unique_id, list) else unique_id

        bypass_img_ref = (video_mode != "reference-to-video")
        bypass_first_frame = (video_mode != "image-to-video")
        
        image_ref = images if not bypass_img_ref else None
        first_frame = images if not bypass_first_frame else None
        
        image_ref_str = image_strength
        first_frame_str = image_strength

        if not primary_sampling and audio is not None:
            load_audio_from_file = True
            bypass_audio_ref = True
        elif audio_select == "internal":
            load_audio_from_file = False
            bypass_audio_ref = True
        elif audio_select == "source":
            load_audio_from_file = True
            bypass_audio_ref = True
        elif audio_select == "reference":
            load_audio_from_file = False
            bypass_audio_ref = False
        else:
            load_audio_from_file = False
            bypass_audio_ref = True

        current_fps = frame_rate
        decode = True 
        
        # ==========================================
        # ETA TRACKER MATH & CALLBACK FACTORY
        # ==========================================
        node_id = unique_id_val
        
        # 1. Simulate Base Chunks & Steps
        base_chunks = 0
        if not autoregressive_chunking or length_in_seconds <= chunk_size_seconds:
            base_chunks = 1
        else:
            shot_dur = length_in_seconds / max(1, len([p for p in scene_descriptions.split("|") if p.strip()]))
            for s in range(max(1, len([p for p in scene_descriptions.split("|") if p.strip()]))):
                curr = 0.0
                end = (s + 1) * shot_dur
                while curr < end:
                    curr = min(curr + chunk_size_seconds, end)
                    base_chunks += 1

        if ',' in str(primary_steps):
            base_steps = max(1, len(re.findall(r"[-+]?(?:\d*\.*\d+)", str(primary_steps))) - 1)
        else:
            try:
                base_steps = max(1, int(float(str(primary_steps).strip())))
            except ValueError:
                base_steps = 8
                    
        # 2. Simulate Upscaler Chunks & Steps
        temp_tile_len = 48
        temp_overlap = 8

        v_frames_base = int((length_in_seconds * current_fps) + 1)
        latent_frames_base = ((v_frames_base - 1) // 8) + 1
        
        def count_sliding_chunks(total_latent_frames):
            return len(list(range(0, total_latent_frames, temp_tile_len - temp_overlap)))

        sp_chunks = count_sliding_chunks(latent_frames_base) * spatial_passes if spatial_upscale else 0
        sp_steps = max(1, len(re.findall(r"[-+]?(?:\d*\.*\d+)", str(spatial_sigmas))) - 1) if spatial_upscale else 0

        tm_chunks = 0
        tm_steps = 0

        total_chunks = base_chunks + sp_chunks + tm_chunks
        total_global_steps = (base_chunks * (base_steps + 1)) + (sp_chunks * sp_steps) + (tm_chunks * tm_steps)
        
        current_chunk_idx = [0]
        current_global_step = [0]

        def wrap_callback(base_cb, pass_name):
            current_chunk_idx[0] += 1
            chunk_num = current_chunk_idx[0]
            def custom_cb(step, x0, x, total_steps_in_pass):
                current_global_step[0] += 1
                PromptServer.instance.send_sync("ltxv_eta_update", {
                    "node_id": node_id,
                    "step": int(step) + 1,
                    "total_steps": int(total_steps_in_pass),
                    "chunk": int(chunk_num),
                    "total_chunks": int(total_chunks),
                    "global_step": int(current_global_step[0]),
                    "total_global_steps": int(total_global_steps),
                    "pass_name": pass_name,
                    "is_face_restore": bool(restore_faces) 
                })
                if base_cb is not None:
                    base_cb(step, x0, x, total_steps_in_pass)
            return custom_cb
        
        # Audio Safeguard Variables
        has_audio_ref = not bypass_audio_ref and audio is not None
        has_audio_input = load_audio_from_file and audio is not None

        negative_prompt = "music, background music, soundtrack, worst quality, deformed, glitch, static, bad teeth, deformed teeth, blurry, soft focus, out of focus, smooth, plastic, washed out, hazy, illustration, painting, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, unreadable texts, text, watermarks, 3d render, cgi"

        def build_custom_sampler(name, eta_val, bongmath_val):
            if name.startswith("res_") or name.startswith("rk_"):
                try:
                    return comfy.samplers.ksampler("rk_beta", {
                        "rk_type": name,
                        "eta": eta_val,
                        "eta_substep": eta_val,
                        "BONGMATH": bongmath_val,
                        "sampler_mode": "standard"
                    })
                except Exception:
                    pass 
            return comfy.samplers.sampler_object(name)

        # Parse Resolution Dropdown
        parsed_width = 1792
        parsed_height = 768
        
        if "×" in target_resolution and not target_resolution.startswith("---"):
            try:
                dims = target_resolution.split("—")[0]
                parsed_width = int(dims.split("×")[0].strip())
                parsed_height = int(dims.split("×")[1].strip())
            except Exception as e:
                print(f"LTXV Custom Warning: Failed to parse resolution '{target_resolution}'. Defaulting to 1792x768.")
        else:
            print(f"LTXV Custom Warning: Category Spacer selected instead of resolution. Defaulting to 1792x768.")

        target_width = parsed_width
        target_height = parsed_height

        divisor = (2 ** spatial_passes) if spatial_upscale else 1
        initial_width = target_width // divisor
        initial_height = target_height // divisor

        expected_width = (initial_width // 32) * 32
        expected_height = (initial_height // 32) * 32

        def process_ref_image(img, method="maintain aspect ratio"):
            import torch.nn.functional as F
            N, H, W, C = img.shape
            if H == expected_height and W == expected_width:
                return img

            t_nchw = img.permute(0, 3, 1, 2)
            
            if method == "stretch to fit":
                resized = F.interpolate(t_nchw, size=(expected_height, expected_width), mode="bilinear", align_corners=False)
            elif method == "maintain aspect ratio":
                ratio = min(expected_width / W, expected_height / H)
                new_w = max(32, (int(W * ratio) // 32) * 32)
                new_h = max(32, (int(H * ratio) // 32) * 32)
                inner = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
                
                pad_l = (expected_width - new_w) // 2
                pad_t = (expected_height - new_h) // 2
                resized = F.pad(inner, (pad_l, expected_width - new_w - pad_l, pad_t, expected_height - new_h - pad_t), mode="constant", value=0)
            else:
                resized = F.interpolate(t_nchw, size=(expected_height, expected_width), mode="bilinear", align_corners=False)

            return resized.permute(0, 2, 3, 1)

        # --- LYRICS-TO-SCRIPT: WHISPER TRANSCRIPTION & DYNAMIC GAP FILLING ---
        if use_ollama == "song-to-script":
            if audio is None:
                print("\nLTXV Custom Warning: 'song-to-script' selected but no audio provided. Falling back to manual scene_descriptions.")
            else:
                print("\n--- Transcribing Audio & Calculating Mathematical Timeline Cadence ---")
                try:
                    from transformers import pipeline
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    waveform = audio["waveform"][0]
                    sample_rate = audio["sample_rate"]
                    target_sr = 16000

                    if sample_rate != target_sr:
                        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)

                    if waveform.dim() == 2 and waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0)
                    waveform = waveform.squeeze()

                    model_name = "openai/whisper-large-v3"
                    print(f"-> Loading {model_name} (Extracting Timestamps)...")
                    
                    transcriber = pipeline(
                        "automatic-speech-recognition",
                        model=model_name,
                        device=device,
                        chunk_length_s=30,
                        return_timestamps=True,
                        ignore_warning=True
                    )
                    
                    audio_np = waveform.cpu().numpy()
                    result = transcriber(
                        {"sampling_rate": target_sr, "raw": audio_np},
                        generate_kwargs={"task": "transcribe", "language": "english"}
                    )
                    
                    # 1. Clean the chunks and extract strict lyric timestamps
                    valid_chunks = [c for c in result.get("chunks", []) if c["text"].strip()]
                    num_lyric_prompts = len(valid_chunks)
                    
                    if num_lyric_prompts > 0:
                        total_lyric_duration = 0.0
                        
                        for c in valid_chunks:
                            start_t, end_t = c["timestamp"]
                            word_count = max(1, len(c["text"].split()))
                            
                            if end_t is None: 
                                end_t = start_t + (word_count * 0.5)
                                
                            raw_dur = end_t - start_t
                            max_physical_duration = (word_count * 0.6) + 1.0 # ~0.6s per word + 1s trailing breath/resonance
                            
                            clamped_dur = min(raw_dur, max_physical_duration)
                            total_lyric_duration += clamped_dur
                            
                            c["mid_t"] = start_t + (clamped_dur / 2.0)
                            
                        total_lyric_duration = max(1.0, total_lyric_duration)
                        
                        # 1. Average length of a lyric-based prompt
                        avg_lyric_prompt_duration = total_lyric_duration / num_lyric_prompts
                        
                        # 2. Total length of song minus lyrics
                        remaining_audio = max(0.0, length_in_seconds - total_lyric_duration)
                        
                        # 3. Calculate remaining instrumental prompts needed
                        instrumental_prompts_needed = int(round(remaining_audio / avg_lyric_prompt_duration))
                        
                        # 4. Final mathematical target count
                        target_shots = num_lyric_prompts + instrumental_prompts_needed
                    else:
                        print("-> No lyrics detected. Falling back to default chunk duration.")
                        target_shots = max(1, int(math.ceil(length_in_seconds / chunk_size_seconds)))
                        
                    target_shots = max(1, target_shots)
                    actual_shot_duration = length_in_seconds / target_shots
                    
                    print(f"   * Lyrics Detected: {num_lyric_prompts} phrases")
                    print(f"   * Total Lyric Duration: {total_lyric_duration:.2f}s")
                    print(f"   * Average Pacing: {avg_lyric_prompt_duration:.2f}s per lyric prompt")
                    print(f"   * Remaining Instrumental Audio: {remaining_audio:.2f}s")
                    print(f"   * Interpolated Instrumental Prompts Needed: {instrumental_prompts_needed}")
                    print(f"-> Final Timeline Mathematically Locked to: {target_shots} total slots ({actual_shot_duration:.2f}s each)")
                    
                    # Initialize empty timeline slots
                    timeline_slots = [""] * target_shots
                    
                    # 2. Map transcribed chunks to their respective mathematical time slots
                    for chunk in valid_chunks:
                        text = chunk["text"].strip()
                        mid_t = chunk["mid_t"]
                        
                        slot_idx = int(mid_t // actual_shot_duration)
                        
                        if slot_idx < target_shots:
                            if timeline_slots[slot_idx] == "":
                                timeline_slots[slot_idx] = text
                            else:
                                timeline_slots[slot_idx] += f" {text}"
                                
                    # Clean up Whisper VRAM immediately
                    del transcriber
                    comfy.model_management.soft_empty_cache()
                    
                    # --- 3. THE OLLAMA GAP-FILLING PASS ---
                    print("\n-> Analyzing Timeline Gaps and Generating Instrumental Scenes...")
                    
                    timeline_context = []
                    for i, text in enumerate(timeline_slots):
                        status = text if text != "" else "[INSTRUMENTAL GAP]"
                        timeline_context.append(f"Shot {i+1} ({i * actual_shot_duration:.1f}s - {(i+1) * actual_shot_duration:.1f}s): {status}")
                        
                    context_string = "\n".join(timeline_context)
                    
                    gap_system_prompt = """You are a music video director. I will give you a timeline of shots. Some shots contain sung lyrics, and some are marked as [INSTRUMENTAL GAP]. 
Your job is to read the lyrics to understand the theme, location, and narrative. Then, invent highly visual, descriptive scene actions to fill in EVERY [INSTRUMENTAL GAP]. 
Do NOT alter or rewrite the shots that already have lyrics. Only fill in the gaps.
Output the final sequence of shots exactly in the format "Shot X: [Description or Lyric]". Do not add any extra conversation, intro, or outro."""
                    
                    payload = {
                        "model": ollama_model,
                        "messages": [
                            {"role": "system", "content": gap_system_prompt},
                            {"role": "user", "content": f"Here is the timeline. Fill in the [INSTRUMENTAL GAP] slots with visual scenes that match the vibe of the surrounding lyrics:\n\n{context_string}"}
                        ],
                        "stream": False
                    }
                    
                    req = urllib.request.Request(f"{ollama_url}/api/chat", data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=120) as response:
                        chat_result = json.loads(response.read().decode('utf-8'))
                        gap_filled_text = chat_result.get('message', {}).get('content', '').strip()
                        
                        # 4. Safely parse Ollama's output back into the script array
                        final_phrases = []
                        for line in gap_filled_text.split('\n'):
                            line = line.strip()
                            if re.match(r'^(?:Shot\s*)?\d+[\s:\-]', line, re.IGNORECASE):
                                parts = re.split(r'[:\-]', line, maxsplit=1)
                                if len(parts) > 1:
                                    final_phrases.append(parts[1].strip())
                                    
                        # Failsafe: if Ollama hallucinated the wrong number of shots, fall back to generic placeholders
                        if len(final_phrases) == target_shots:
                            scene_descriptions = " | ".join(final_phrases)
                        else:
                            print(f"-> Warning: Ollama returned {len(final_phrases)} shots instead of {target_shots}. Using generic instrumental placeholders.")
                            filled_slots = [text if text != "" else "Camera sweeps across the scene as instrumental music plays." for text in timeline_slots]
                            scene_descriptions = " | ".join(filled_slots)
                            
                    print(f"-> Gap-Filling Complete! Generated exactly {target_shots} temporal shots synced to the track.")
                    
                except ImportError:
                    print("\nLTXV Custom Error: 'transformers' pipeline missing. Falling back to manual scenes.")
                except Exception as e:
                    print(f"\nLTXV Custom Error during timestamp transcription: {e}. Falling back to manual scenes.")

        raw_prompts = [p.strip() for p in scene_descriptions.split("|") if p.strip()]
        num_prompts = len(raw_prompts)
        if num_prompts == 0:
            raw_prompts = [""]
            num_prompts = 1

        override_char_desc = (not bypass_img_ref) and (image_ref is not None) and (use_ollama in ["enhance prompt", "song-to-script"])
        c_desc = character_descriptions.strip()
        l_desc = location_description.strip()

        new_prompts = []
        for p in raw_prompts:
            prefix = ""
            if not override_char_desc and c_desc:
                prefix += f"Character(s): {c_desc}. "
            if l_desc:
                prefix += f"Location: {l_desc}. "
            new_prompts.append(prefix + p)
        raw_prompts = new_prompts

        if num_prompts > 1:
            shot_duration_int = max(1, round(length_in_seconds / num_prompts))
            ideal_length = float(shot_duration_int * num_prompts)
            
            if ideal_length != length_in_seconds:
                print(f"\n--- Auto-Rounding Timeline: Adjusted total duration from {length_in_seconds}s to {ideal_length}s for perfect {shot_duration_int}s shots. ---")
                length_in_seconds = ideal_length

            autoregressive_chunking = True 
            chunk_size_seconds = length_in_seconds / num_prompts
            print(f"\n--- Multi-Shot Director Mode Active: Timeline synced to {num_prompts} shots ({chunk_size_seconds:.2f}s per shot). ---")

        if use_ollama in ["enhance prompt", "song-to-script"]:
            print(f"\n--- Querying Ollama ({ollama_model}) for Multi-Shot Enhancement ---")

            def create_grid_b64(tensor_list):
                pil_images = []
                for t in tensor_list:
                    img_arr = t[0].cpu().numpy()
                    img_arr = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_arr)
                    pil_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                    pil_images.append(pil_img)

                if not pil_images:
                    return None

                num_imgs = len(pil_images)
                if num_imgs == 1:
                    grid = pil_images[0]
                else:
                    cols = 2
                    rows = math.ceil(num_imgs / 2)
                    w = max(img.width for img in pil_images)
                    h = max(img.height for img in pil_images)
                    grid = Image.new('RGB', (cols * w, rows * h))
                    for i, img in enumerate(pil_images):
                        x = (i % cols) * w
                        y = (i // cols) * h
                        grid.paste(img, (x, y))

                buffered = BytesIO()
                grid.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode("utf-8")

            system_prompt = """You write prompts for LTX Video. Output one single flowing paragraph only — no preamble, no label, no explanation, no markdown, no variations. Begin writing immediately.
I will provide you with a base prompt and a single reference image collage containing the Subject, Object, and/or Location. Your job is to seamlessly combine them into a single, highly detailed, flowing paragraph.
Do not describe the initial pose or orientation of the subject from the reference image - only if included in the base prompt.
Do not leave out a single detail that is included in the base prompt.

CORE FORMAT:
- Single flowing paragraph, present tense, no line breaks
- 8–14 descriptive sentences scaled to clip length
- Specificity wins — LTX handles complexity, do not oversimplify
- Block the scene like a director: name positions (left/right), distances (foreground/background), facing directions
- Every sentence should contain at least one verb driving action or motion

REQUIRED ELEMENTS — write in this order, woven into natural sentences:

1. SHOT + CINEMATOGRAPHY
Open with shot scale and camera position. Examples: close-up, medium shot, wide establishing shot, low angle, Dutch tilt, over-the-shoulder, overhead, POV. Match detail level to shot scale — close-ups need more texture detail than wide shots.

2. SCENE + ATMOSPHERE
Location, time of day, weather, colour palette, surface textures, atmosphere (fog, rain, dust, smoke, particles). Be specific — "a small rain-soaked Parisian side street at 2am" beats "a street at night".

3. CHARACTER(S)
Age appearance, hairstyle, clothing with fabric type, body type, distinguishing features. Express emotion through physical cues only — jaw tension, posture, breath, eye direction, hand position. Never use abstract labels like "sad" or "nervous".

4. ACTION SEQUENCE
Write action as a clear temporal flow from beginning to end. Name who moves, what moves, how they move, and at what pace. Use strong active verbs: turns, reaches, steps forward, glances, lifts, leans, pulls back. LTX follows action sequences accurately — be explicit. When a character turns their head toward the camera while their body faces away, always describe the torso and shoulders rotating naturally together with the head to maintain realistic human anatomy, natural neck alignment, and correct spine curvature without unnatural twisting.

5. CAMERA MOVEMENT
Specify camera movement and when it happens. Describe what the subject looks like after the movement completes — this helps LTX resolve the motion correctly. Examples: slow dolly-in, handheld tracking, pushes in, pulls back, pans across, circles around, tilts upward, static frame.

6. LIGHTING
Source, quality, colour temperature, how it falls on the subject and environment. Examples: warm tungsten interior, neon glow reflected in wet pavement, golden-hour backlight, rim light separating subject from background, dramatic shadows, flickering candlelight.

7. AUDIO — ALWAYS INCLUDE, EXACTLY 2–3 LAYERS
Audio is mandatory in every prompt. Use exactly 2 or 3 layers — no more, no fewer.
Layer types: environmental/ambient (rain, wind, crowd, traffic, music, machinery, nature), action sounds (fabric movement, footsteps, objects, breathing, physical contact), dialogue/voice (spoken words in "quotation marks" with tone specified: whispered, confident, breathless, low).
If the scene includes dialogue, dialogue counts as one of the 2–3 layers.
Always include the phrase "ambient sound" or "natural environment" with "no background music" unless specified in the user prompt to include music.
The final sentence of the prompt must always be the audio layer.

ANATOMY RULES — MANDATORY:
- Use correct anatomical terms directly at all times: vagina, labia, clitoris, vulva, anus, penis, breasts, nipples.
- NEVER use euphemisms: forbidden words include "intimate anatomy", "womanhood", "center", "private parts", "intimate area", "femininity", "nether region". If you write any of these you have failed.
- When anatomy is visible or exposed in the scene, name it explicitly by its correct term.
- Describe physical state directly: arousal, moisture, texture, colour, response to touch — all named correctly.

LTX SPECIFIC RULES:
- Avoid static prompts — every prompt must have explicit motion: subject motion, environmental motion, or camera motion (ideally all three). If it reads like a still photo, LTX may output a frozen video.
- Spatial layout matters — LTX respects left/right/foreground/background positioning. Use it.
- Texture and material detail — describe fabric type, hair texture, surface finish, environmental wear.
- I2V (when a start frame is provided) — focus on verbs not descriptions. Describe what moves and how, not what is visible. Lock the face and identity — describe only motion and camera changes.
- No internal states — never write "she feels", "he thinks", "she is excited". Show it physically.
- No overloaded scenes — max 2–3 characters with clearly separated actions.
- No conflicting lighting logic — one dominant light source with consistent fill.
- Anatomy consistency — always prioritise realistic human posture and joint rotation; when head and body orientations differ, explicitly describe natural torso rotation with the head to prevent unnatural neck twisting or spine morphing.

CAMERA VOCABULARY:
follows, tracks, pans across, circles around, tilts upward, pushes in, pulls back, overhead view, handheld movement, over-the-shoulder, wide establishing shot, static frame, slow dolly-in, rack focus, creep forward, drift right, slow orbit, arc shot

END EVERY PROMPT WITH THIS QUALITY TAIL (woven into the final sentence, not as a separate line):
cinematic, ultra-detailed, sharp focus, photorealistic, masterpiece, maintains realistic human anatomy and natural joint rotation throughout

Output only the prompt. Nothing before it, nothing after it."""

            enhanced_prompts = []

            for i, p in enumerate(raw_prompts):
                tensors_to_grid = []
                if not bypass_img_ref and image_ref is not None:
                    for j in range(image_ref.shape[0]):
                        tensors_to_grid.append(image_ref[j:j+1])
                if not bypass_first_frame and first_frame is not None:
                    img_idx = min(i, first_frame.shape[0] - 1)
                    tensors_to_grid.append(first_frame[img_idx:img_idx+1])

                grid_b64 = create_grid_b64(tensors_to_grid) if tensors_to_grid else None

                # --- CINEMATIC SHOT VARIETY ENGINE ---
                # A carefully curated rotation of focal distances to guarantee dynamic editing pacing
                shot_scales = [
                    "wide establishing shot",
                    "intimate close-up",
                    "medium tracking shot",
                    "low angle extreme wide shot",
                    "medium close-up",
                    "extreme close-up macro shot",
                    "wide sweeping crane shot"
                ]
                # Rotate through the list based on the current shot index
                assigned_scale = shot_scales[i % len(shot_scales)]

                if use_ollama == "song-to-script":
                    user_message = {
                        "role": "user",
                        "content": f"This is Shot {i+1} of a music video sequence. MANDATORY CAMERA FRAMING: You must frame this specific shot as a **{assigned_scale}**.\n\nThe prompt below contains the characters, the location, and the specific sung lyric for this exact shot:\n\n{p}\n\nAct as a music video director. Generate a cohesive, highly detailed visual prompt that blends the characters and location with the emotion of the lyric. CRITICAL INSTRUCTION: You MUST explicitly describe the main character actively singing, performing, or lip-syncing the words in the scene. Do not just describe their mood; describe them physically singing. You must also explicitly describe the camera perspective using the {assigned_scale}."
                    }
                else:
                    user_message = {
                        "role": "user",
                        "content": f"This is Shot {i+1} of a multi-shot sequence. MANDATORY CAMERA FRAMING: You must frame this specific shot as a **{assigned_scale}**.\n\nBase prompt: {p}\n\nAnalyze the provided reference image grid and generate the final LTX-Video prompt. You must explicitly describe the camera perspective using the {assigned_scale}."
                    }

                if grid_b64:
                    user_message["images"] = [grid_b64]

                payload = {
                    "model": ollama_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        user_message
                    ],
                    "stream": False
                }

                try:
                    req = urllib.request.Request(f"{ollama_url}/api/chat", data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=120) as response:
                        result = json.loads(response.read().decode('utf-8'))
                        ollama_prompt = result.get('message', {}).get('content', '').strip()
                        if ollama_prompt:
                            enhanced_prompts.append(ollama_prompt)
                            print(f"-> Shot {i+1} Enhanced:\n{ollama_prompt}\n")
                        else:
                            enhanced_prompts.append(p)
                except urllib.error.HTTPError as e:
                    try:
                        error_body = e.read().decode('utf-8')
                        print(f"-> Ollama HTTP Error {e.code}: {error_body}")
                    except:
                        print(f"-> Ollama HTTP Error {e.code}: {e.reason}")
                    print(f"-> CRASH FIX: Falling back to original prompt for Shot {i+1}.")
                    enhanced_prompts.append(p)
                except Exception as e:
                    print(f"-> Ollama API Error: {e}. Falling back to original prompt for Shot {i+1}.")
                    enhanced_prompts.append(p)

            prompt_list = enhanced_prompts
            print("---------------------------------------------------------")
        else:
            prompt_list = raw_prompts

        # ==========================================
        # 0. DEFINE HARDCODED VARIABLES & HIDDEN TEXT
        # ==========================================
        duplicate_frames = 8  
        hidden_prefix = ""
        hidden_suffix = " Shot on 85mm lens, f/8 aperture, raw DSLR footage, ultra-sharp focus, 8k resolution, hyperrealistic, intricate details, cinematic lighting."

        time_scale_factor, height_scale_factor, width_scale_factor = video_vae.downscale_index_formula

        # ==========================================
        # 1. TEXT CONDITIONING (ISOLATED FOR HARD CUTS)
        # ==========================================
        final_positive = []
        final_prompt_strings = []
        
        # --- POSITIVE OVERRIDE ---
        if positive is not None:
            print("\n--- Positive Conditioning Override Detected ---")
            final_prompt_strings.append("--- Custom Positive Conditioning Override Active ---")
            for i in range(num_prompts):
                # Safely map external conditioning elements to our internal multi-shot arrays
                idx = min(i, len(positive) - 1)
                cond_t, cond_d = positive[idx][0], positive[idx][1].copy()
                cond_d["frame_rate"] = float(current_fps)
                final_positive.append([cond_t, cond_d])
        else:
            for i, p in enumerate(prompt_list):
                modified_prompt = f"{hidden_prefix}{p}{hidden_suffix}"
                final_prompt_strings.append(modified_prompt)
                tokens = clip.tokenize(modified_prompt)
                out = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
                cond = out.pop("cond")
                c_dict = out.copy() if isinstance(out, dict) else {}
                c_dict["start_percent"] = 0.0 
                c_dict["end_percent"] = 1.0
                c_dict["frame_rate"] = float(current_fps)
                final_positive.append([cond, c_dict])

        final_prompt_string_out = " | ".join(final_prompt_strings)

        # --- NEGATIVE OVERRIDE ---
        if negative is not None:
            print("--- Negative Conditioning Override Detected ---")
            final_negative = []
            for i in range(num_prompts):
                idx = min(i, len(negative) - 1)
                cond_t, cond_d = negative[idx][0], negative[idx][1].copy()
                cond_d["frame_rate"] = float(current_fps)
                final_negative.append([cond_t, cond_d])
        else:
            tokens_neg = clip.tokenize(negative_prompt)
            out_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True, return_dict=True)
            cond_neg = out_neg.pop("cond")
            dict_neg = out_neg.copy() if isinstance(out_neg, dict) else {}
            dict_neg["start_percent"] = 0.000
            dict_neg["end_percent"] = 1.000
            dict_neg["frame_rate"] = float(current_fps)
            final_negative = [[cond_neg, dict_neg]]

        # ==========================================
        # 2. CALCULATE 'A' AND ASSEMBLE SETUP REFERENCES
        # ==========================================
        pixel_frames = []
        strengths = []
        a = 0

        if not bypass_img_ref and image_ref is not None:
            if img_compression > 0:
                compressed_images = []
                for i in range(image_ref.shape[0]):
                    compressed_images.append(preprocess_compression(image_ref[i], img_compression))
                image_ref = torch.stack(compressed_images)
                
            image_ref_processed = process_ref_image(image_ref)
            a += image_ref_processed.shape[0]
            for idx in range(image_ref_processed.shape[0]):
                img = image_ref_processed[idx:idx+1]
                pixel_frames.append(img.repeat(duplicate_frames, 1, 1, 1))
                strengths.extend([image_ref_str] * duplicate_frames)

        if not bypass_first_frame and first_frame is not None:
            if img_compression > 0:
                compressed_images = []
                for i in range(first_frame.shape[0]):
                    compressed_images.append(preprocess_compression(first_frame[i], img_compression))
                first_frame = torch.stack(compressed_images)
                
            first_frame_processed = process_ref_image(first_frame)
            pixel_frames.append(first_frame_processed[0:1])
            strengths.append(first_frame_str)

        ref_frame_count = len(strengths)

        if len(strengths) > 0:
            final_pixels = torch.cat(pixel_frames, dim=0)
        else:
            final_pixels = None

        # ==========================================
        # 3. EXACT FRAME MATH & BASE LATENT GENERATION
        # ==========================================
        shot_seconds = length_in_seconds / num_prompts
        
        if not bypass_img_ref:
            raw_frames_per_shot = (a * duplicate_frames) + (shot_seconds * current_fps) + 9
        else:
            raw_frames_per_shot = (shot_seconds * current_fps) + 1
            
        latents_per_shot = int(((raw_frames_per_shot - 1) // 8) + 1)
        if latents_per_shot < 1: latents_per_shot = 1
        
        frames_per_shot = int(((latents_per_shot - 1) * 8) + 1)
        
        # Multiply by num_prompts to allocate perfectly segregated blocks
        latent_count = latents_per_shot * num_prompts
        frame_length = frames_per_shot * num_prompts
        
        device = comfy.model_management.intermediate_device()
        batch_size = 1 

        # --- LATENT OVERRIDE EXTRACTION ---
        v_override = None
        a_override = None
        if latent_override is not None:
            print("\n--- Latent Override Detected ---")
            samples = latent_override.get("samples", latent_override) 
            if getattr(samples, "is_nested", False) or isinstance(samples, comfy.nested_tensor.NestedTensor):
                v_override, a_override = samples.unbind()
            elif isinstance(samples, (list, tuple)) and len(samples) == 2:
                v_override, a_override = samples[0], samples[1]
            else:
                v_override = samples
                
        # --- APPLY OVERRIDE TO TIMELINE MATH ---
        if v_override is not None:
            print("-> Overriding timeline and resolution math to match injected tensor.")
            video_samples = v_override.to(device).clone()
            batch_size, _, latent_count, v_h, v_w = video_samples.shape
            initial_height = v_h * 32
            initial_width = v_w * 32
            frame_length = int(((latent_count - 1) * 8) + 1)
            length_in_seconds = frame_length / current_fps
            pure_latents_per_shot = latent_count // num_prompts
            setup_latents = 0
            shot_seconds = length_in_seconds / num_prompts
        else:
            video_samples = torch.zeros([batch_size, 128, latent_count, initial_height // 32, initial_width // 32], device=device)
            video_samples = comfy.sample.fix_empty_latent_channels(model1_primary, video_samples, None)
            
        video_noise_mask = torch.ones((batch_size, video_samples.shape[2], video_samples.shape[3], video_samples.shape[4]), dtype=torch.float32, device=device)

        z_channels = getattr(audio_vae, "latent_channels", audio_vae.first_stage_model.latent_channels)
        audio_freq = getattr(audio_vae, "latent_frequency_bins", audio_vae.first_stage_model.latent_frequency_bins)
        sampling_rate = int(getattr(audio_vae, "sample_rate", audio_vae.first_stage_model.sample_rate))

        get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
        if get_audio_latents_func is None:
            raise AttributeError("Audio VAE is missing 'num_of_latents_from_frames' method.")
            
        if a_override is not None:
            num_audio_latents = a_override.shape[2]
        else:
            num_audio_latents = get_audio_latents_func(frame_length, int(current_fps))

        total_silence_samples = int(((frame_length / current_fps) + 5.0) * sampling_rate)
        silent_wf = torch.zeros((batch_size, 2, total_silence_samples), dtype=torch.float32, device=device)
        
        # --- SAFE AUDIO VAE ENCODE (DEVICE MISMATCH FIX) ---
        vae_device = audio_vae.first_stage_model.device if hasattr(audio_vae.first_stage_model, "device") else comfy.model_management.get_torch_device()
        silent_wf_mapped = silent_wf.movedim(1, -1).to(vae_device)
        
        try:
            true_silence_latent = audio_vae.encode(silent_wf_mapped).to(device)
        except AttributeError:
            true_silence_latent = audio_vae.first_stage_model.encode(silent_wf_mapped).to(device)
        
        audio_samples = torch.zeros((batch_size, z_channels, num_audio_latents, audio_freq), device=device)
        
        # DIRECTOR FIX: 3D Audio mask prevents KSampler from bleeding audio noise into visual latents!
        audio_noise_mask = torch.ones((batch_size, num_audio_latents, audio_freq), dtype=torch.float32, device=device)

        # ==========================================
        # 3.4 HELPER: LATENT COUNTER
        # ==========================================
        def get_latent_counts(sec):
            if sec <= 0.0:
                return 0, 0
                
            shot_idx = int(sec // shot_seconds)
            if shot_idx >= num_prompts:
                shot_idx = num_prompts - 1
                
            remainder_sec = sec - (shot_idx * shot_seconds)
            
            # ACCORDION FIX: Calculate exact isolated audio length for one full shot to prevent continuous drift!
            get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
            base_a_lat_per_shot = get_audio_latents_func(frames_per_shot, int(current_fps))
            
            if remainder_sec <= 0.001:
                v_lat = shot_idx * latents_per_shot
                a_lat = shot_idx * base_a_lat_per_shot
            else:
                if bypass_img_ref:
                    raw_frames = int((remainder_sec * current_fps) + 1)
                else:
                    raw_frames = int((a * duplicate_frames) + (remainder_sec * current_fps) + 9)
                    
                local_v_lat = int(((raw_frames - 1) // 8) + 1)
                if local_v_lat < 1: local_v_lat = 1
                
                v_lat = (shot_idx * latents_per_shot) + local_v_lat
                
                local_synced_frames = int(((local_v_lat - 1) * 8) + 1) if local_v_lat > 0 else 1
                local_a_lat = get_audio_latents_func(local_synced_frames, int(current_fps))
                
                a_lat = (shot_idx * base_a_lat_per_shot) + local_a_lat
                
            return v_lat, a_lat

        if not primary_sampling:
            out_ref_frame_count = 0
        elif bypass_img_ref:
            out_ref_frame_count = 1
        else:
            n = ref_frame_count // duplicate_frames
            out_ref_frame_count = (8 * n) + 8 + 1

        # ==========================================
        # 3.5 WAVEFORM MIXING (NATIVE COMFYUI ASSEMBLY)
        # ==========================================
        setup_frames = 0 if bypass_img_ref else out_ref_frame_count

        if bypass_img_ref:
            region_a_frames = 1
            region_b_frames = 0
        else:
            region_a_frames = ref_frame_count 
            region_b_frames = duplicate_frames 

        get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
        setup_total_latents = get_audio_latents_func(setup_frames, int(current_fps))
        
        vae_sample_rate = getattr(audio_vae, "audio_sample_rate", getattr(audio_vae, "sample_rate", 24000))
        sampling_rate = vae_sample_rate

        total_samples = int((frame_length / current_fps) * sampling_rate)
        region_a_samples = int((region_a_frames / current_fps) * sampling_rate)

        setup_total_samples = int((setup_frames / current_fps) * sampling_rate)

        master_wf = torch.zeros((batch_size, 2, total_samples), dtype=torch.float32, device=device)
        master_latents = torch.zeros((batch_size, z_channels, num_audio_latents, audio_freq), device=device)

        def encode_native(audio_dict):
            orig_wf = audio_dict["waveform"].to(device)
            orig_sr = audio_dict.get("sample_rate", sampling_rate)
            if orig_sr != sampling_rate:
                wf = torchaudio.functional.resample(orig_wf, orig_sr, sampling_rate)
            else:
                wf = orig_wf

            if wf.dim() == 2:
                wf = wf.unsqueeze(1).repeat(1, 2, 1)
            elif wf.dim() == 3 and wf.shape[1] == 1:
                wf = wf.repeat(1, 2, 1)
            elif wf.dim() == 3 and wf.shape[1] > 2:
                wf = wf[:, :2, :]

            if wf.shape[0] != batch_size:
                wf = wf.repeat(batch_size, 1, 1)[:batch_size]

            wf_mapped = wf.movedim(1, -1).to(vae_device)
            try:
                latents = audio_vae.encode(wf_mapped).to(device)
            except AttributeError:
                latents = audio_vae.first_stage_model.encode(wf_mapped).to(device)

            return wf, latents

        silent_wf = torch.zeros((batch_size, 2, total_samples), dtype=torch.float32, device=device)
        silent_dict = {"waveform": silent_wf, "sample_rate": sampling_rate}
        _, true_silence_latent = encode_native(silent_dict)

        use_silence_len = min(num_audio_latents, true_silence_latent.shape[2])
        if use_silence_len > 0:
            master_latents[:, :, :use_silence_len, :] = true_silence_latent[:, :, :use_silence_len, :]

        if has_audio_input:
            inp_wf, inp_latents = encode_native(audio)

            rem_latents = num_audio_latents - setup_total_latents
            if rem_latents > 0:
                use_lats = min(inp_latents.shape[2], rem_latents)
                if use_lats > 0:
                    master_latents[:, :, setup_total_latents:setup_total_latents+use_lats, :] = inp_latents[:, :, :use_lats]

            rem_samps = total_samples - setup_total_samples
            if rem_samps > 0:
                use_s = min(inp_wf.shape[2], rem_samps)
                if use_s > 0:
                    master_wf[:, :, setup_total_samples:setup_total_samples+use_s] = inp_wf[:, :, :use_s]

        if has_audio_ref:
            ref_wf, ref_latents = encode_native(audio)

            use_lats = min(ref_latents.shape[2], setup_total_latents, num_audio_latents)
            if use_lats > 0:
                master_latents[:, :, :use_lats, :] = ref_latents[:, :, :use_lats]

            use_s = min(ref_wf.shape[2], setup_total_samples, total_samples)
            if use_s > 0:
                master_wf[:, :, :use_s] = ref_wf[:, :, :use_s]

        use_len = min(num_audio_latents, master_latents.shape[2])
        if use_len > 0:
            audio_samples[:, :, :use_len, :] = master_latents[:, :, :use_len, :]

        if has_audio_ref:
            lock_a = min(setup_total_latents, use_len)
            if lock_a > 0:
                audio_noise_mask[:, :lock_a, :] = 0.0

        if has_audio_input:
            # We can use the shorthand [:] to zero out the entire tensor safely
            audio_noise_mask[:] = 0.0

        # ==========================================
        # 4. VIDEO INJECTION & MASKING (PRIMARY PASS)
        # ==========================================
        if final_pixels is not None:
            t_width = video_samples.shape[4] * width_scale_factor
            t_height = video_samples.shape[3] * height_scale_factor

            pass1_pixels = final_pixels.clone()
            if pass1_pixels.shape[1] != t_height or pass1_pixels.shape[2] != t_width:
                pass1_pixels = comfy.utils.common_upscale(pass1_pixels.movedim(-1, 1), t_width, t_height, "bilinear", "center").movedim(1, -1)

            encoded_t = video_vae.encode(pass1_pixels[:, :, :, :3])
            
            # Inject into Shot 1 (Default behavior)
            frames_to_inject = min(encoded_t.shape[2], video_samples.shape[2])
            video_samples[:, :, :frames_to_inject] = encoded_t[:, :, :frames_to_inject]

            for i in range(frames_to_inject):
                pixel_idx = min(i * time_scale_factor, max(0, len(strengths) - 1))
                video_noise_mask[:, i, :, :] = 1.0 - strengths[pixel_idx]
                
            if not bypass_img_ref and num_prompts > 1:
                shot_duration = length_in_seconds / num_prompts
                for s in range(1, num_prompts):
                    sec = s * shot_duration
                    v_lat, _ = get_latent_counts(sec)
                    inject_len = min(encoded_t.shape[2], video_samples.shape[2] - v_lat)
                    if inject_len > 0:
                        video_samples[:, :, v_lat : v_lat + inject_len] = encoded_t[:, :, :inject_len]
                        
                        for i in range(inject_len):
                            pixel_idx = min(i * time_scale_factor, max(0, len(strengths) - 1))
                            video_noise_mask[:, v_lat + i, :, :] = 1.0 - strengths[pixel_idx]
                            
                        print(f"-> Identity Setup Block injected into Shot {s+1} at {sec:.2f}s (Latent Frame {v_lat})")

                if has_audio_input:
                    for s in range(1, num_prompts):
                        sec = s * shot_duration
                        _, a_lat = get_latent_counts(sec)
                        inject_a_len = min(setup_total_latents, audio_samples.shape[2] - a_lat)
                        if inject_a_len > 0:
                            audio_samples[:, :, a_lat : a_lat + inject_a_len, :] = master_latents[:, :, :inject_a_len, :]
                            audio_noise_mask[:, a_lat : a_lat + inject_a_len, :] = 0.0

        # 4.5 MULTI-SHOT DIRECTOR CUT INJECTIONS
        if not bypass_first_frame and first_frame is not None and num_prompts > 1 and autoregressive_chunking:
            shot_duration = length_in_seconds / num_prompts
            for i in range(1, num_prompts):
                img_idx = min(i, first_frame.shape[0] - 1)
                img = first_frame[img_idx:img_idx+1]

                img_scaled = comfy.utils.common_upscale(img.movedim(-1, 1), t_width, t_height, "bilinear", "center").movedim(1, -1)
                encoded_img = video_vae.encode(img_scaled[:, :, :, :3])
                if encoded_img.ndim == 4:
                    encoded_img = encoded_img.unsqueeze(0)
                encoded_img = encoded_img.to(device)

                sec = i * shot_duration
                v_lat, _ = get_latent_counts(sec)

                inject_len = min(encoded_img.shape[2], video_samples.shape[2] - v_lat)
                if inject_len > 0:
                    video_samples[:, :, v_lat : v_lat + inject_len] = encoded_img[:, :, :inject_len]
                    for j in range(inject_len):
                        video_noise_mask[:, v_lat + j, :, :] = 1.0 - first_frame_str
                    print(f"-> Shot {i+1} Reference Frame Injected at {sec:.2f}s (Latent Frame {v_lat})")

        # ==========================================
        # 5. ALL UNET VRAM PATCHING (FP16, Sage, Chunks, NAG, LoRA)
        # ==========================================
        model_to_use = model1_primary.clone()
        model2_to_use = model2_spatial.clone() if model2_spatial is not None else model_to_use.clone()
        model3_to_use = model3_temporal.clone() if model3_temporal is not None else model_to_use.clone()

        # Gather all unique models that require patching
        models_to_patch = [model_to_use]
        if model2_spatial is not None: models_to_patch.append(model2_to_use)
        if model3_temporal is not None: models_to_patch.append(model3_to_use)

        for target_model in models_to_patch:
            diffusion_model = target_model.get_model_object("diffusion_model")
            
            def patch_enable_fp16_accum(model):
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            def patch_disable_fp16_accum(model):
                torch.backends.cuda.matmul.allow_fp16_accumulation = False

            if enable_fp16_accumulation:
                if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                    target_model.add_callback(comfy.patcher_extension.CallbacksMP.ON_PRE_RUN, patch_enable_fp16_accum)
                    target_model.add_callback(comfy.patcher_extension.CallbacksMP.ON_CLEANUP, patch_disable_fp16_accum)
            else:
                if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                    target_model.add_callback(comfy.patcher_extension.CallbacksMP.ON_PRE_RUN, patch_disable_fp16_accum)

            if sage_attention != "disabled":
                target_model.model_options["transformer_options"] = target_model.model_options.get("transformer_options", {}).copy()
                
                new_attention = get_sage_func(sage_attention)
                def attention_override_sage(func, *args, **kwargs):
                    return new_attention.__wrapped__(*args, **kwargs)
                target_model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage

            dim_threshold = 4096
            if chunks_feedforward > 1:
                for idx, block in enumerate(diffusion_model.transformer_blocks):
                    patched_ffn = LTXVffnChunkPatch(chunks_feedforward, dim_threshold).__get__(block.ff, block.__class__)
                    target_model.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.ff.forward", patched_ffn)
            
            # --- HARDCODED NORMALIZED ATTENTION GUIDANCE (NAG) ---
            if enable_nag:
                # To prevent printing multiple times, only print on the first model
                if target_model is model_to_use:
                    print("\n--- Injecting Normalized Attention Guidance (NAG) ---")
                nag_scale = 11.0
                nag_alpha = 0.25
                nag_tau = 2.5
                inplace = True
                
                dtype = target_model.model.manual_cast_dtype
                if dtype is None:
                    dtype = diffusion_model.dtype
                    
                compute_device = comfy.model_management.get_torch_device()
                offload_device = comfy.model_management.unet_offload_device()
                
                context_video = final_negative[0][0].to(compute_device, dtype)
                vid_split = getattr(diffusion_model, "cross_attention_dim", None)
                if vid_split is not None and context_video.shape[-1] == vid_split + getattr(diffusion_model, "audio_cross_attention_dim", 0):
                    context_video = context_video[:, :, :vid_split]
                    
                if getattr(diffusion_model, "caption_proj_before_connector", False) and getattr(diffusion_model, "caption_projection_first_linear", False):
                    diffusion_model.caption_projection.to(compute_device)
                    context_video = diffusion_model.caption_projection(context_video)
                    diffusion_model.caption_projection.to(offload_device)
                    
                if hasattr(diffusion_model, "video_embeddings_connector"):
                    diffusion_model.video_embeddings_connector.to(compute_device)
                    context_video = diffusion_model.video_embeddings_connector(context_video)[0]
                    diffusion_model.video_embeddings_connector.to(offload_device)
                    
                context_video = context_video.view(1, -1, diffusion_model.inner_dim)
                
                for idx, block in enumerate(diffusion_model.transformer_blocks):
                    patched_attn2 = LTXVCrossAttentionPatch(context_video, nag_scale, nag_alpha, nag_tau, inplace=inplace).__get__(block.attn2, block.__class__)
                    target_model.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.attn2.forward", patched_attn2)

        model2_to_use = model2_spatial.clone() if model2_spatial is not None else model_to_use.clone()
        model3_to_use = model3_temporal.clone() if model3_temporal is not None else model_to_use.clone()

        # ==========================================
        # ID-LORA VOICE CLONING (CFG OVERRIDE)
        # ==========================================
        actual_cond_audio = None
        if has_audio_ref:
            actual_cond_audio = audio
        elif has_audio_input:
            actual_cond_audio = audio 
            
        if actual_cond_audio is not None:
            sample_rate_ref = actual_cond_audio["sample_rate"]
            vae_sample_rate = getattr(audio_vae, "audio_sample_rate", 44100)
            if vae_sample_rate != sample_rate_ref:
                wf_lora = torchaudio.functional.resample(actual_cond_audio["waveform"], sample_rate_ref, vae_sample_rate)
            else:
                wf_lora = actual_cond_audio["waveform"]

            max_samps = int(20.0 * vae_sample_rate)
            if wf_lora.shape[2] > max_samps:
                lats = []
                for i in range(0, wf_lora.shape[2], max_samps):
                    chunk_wf = wf_lora[:, :, i:i+max_samps]
                    if chunk_wf.shape[2] < 4000:
                        chunk_wf = torch.nn.functional.pad(chunk_wf, (0, 4000 - chunk_wf.shape[2]))
                    
                    wf_lora_mapped = chunk_wf.movedim(1, -1).to(vae_device)
                    try: lats.append(audio_vae.encode(wf_lora_mapped).to(device))
                    except AttributeError: lats.append(audio_vae.first_stage_model.encode(wf_lora_mapped).to(device))
                audio_latents_lora = torch.cat(lats, dim=2)
            else:
                max_samps = int(20.0 * vae_sample_rate)
                if wf_lora.shape[2] > max_samps:
                    lats = []
                    for i in range(0, wf_lora.shape[2], max_samps):
                        chunk_wf = wf_lora[:, :, i:i+max_samps]
                        if chunk_wf.shape[2] < 2400:
                            continue
                        wf_lora_mapped = chunk_wf.movedim(1, -1).to(vae_device)
                        try: lats.append(audio_vae.encode(wf_lora_mapped).to(device))
                        except AttributeError: lats.append(audio_vae.first_stage_model.encode(wf_lora_mapped).to(device))
                    audio_latents_lora = torch.cat(lats, dim=2)
                else:
                    wf_lora_mapped = wf_lora.movedim(1, -1).to(vae_device)
                    try: audio_latents_lora = audio_vae.encode(wf_lora_mapped).to(device)
                    except AttributeError: audio_latents_lora = audio_vae.first_stage_model.encode(wf_lora_mapped).to(device)
            
            b_, c_, t_, f_ = audio_latents_lora.shape
            ref_tokens = audio_latents_lora.permute(0, 2, 1, 3).reshape(b_, t_, c_ * f_)
            ref_audio_dict = {"tokens": ref_tokens}

            for i in range(len(final_positive)):
                final_positive[i][1]["ref_audio"] = ref_audio_dict
            for i in range(len(final_negative)):
                final_negative[i][1].pop("ref_audio", None)

            scale = identity_guidance_scale

            def get_post_cfg_function(target_model):
                model_sampling = target_model.get_model_object("model_sampling")
                sigma_start = model_sampling.percent_to_sigma(0.0)
                sigma_end = model_sampling.percent_to_sigma(1.0)
                audio_channels = getattr(audio_vae, "latent_channels", getattr(audio_vae.first_stage_model, "latent_channels", 128))
                
                def post_cfg_function(args):
                    if scale == 0:
                        return args["denoised"]

                    sigma = args["sigma"]
                    sigma_ = sigma[0].item()
                    if sigma_ > sigma_start or sigma_ < sigma_end:
                        return args["denoised"]

                    cond_pred = args["cond_denoised"]
                    cond = args["cond"]
                    cfg_result = args["denoised"]
                    model_options = args["model_options"].copy()
                    x = args["input"]

                    is_nested = isinstance(cfg_result, comfy.nested_tensor.NestedTensor)
                    is_video_only = (not is_nested) and (len(cfg_result.shape) > 1 and cfg_result.shape[1] == 128)
                    if is_video_only:
                        return cfg_result

                    noref_cond = []
                    for entry in cond:
                        new_entry = entry.copy()
                        mc = new_entry.get("model_conds", {}).copy()
                        mc.pop("ref_audio", None)
                        new_entry["model_conds"] = mc
                        noref_cond.append(new_entry)

                    (pred_noref,) = comfy.samplers.calc_cond_batch(args["model"], [noref_cond], x, sigma, model_options)

                    if is_nested:
                        cfg_v, cfg_a = cfg_result.unbind()
                        cond_v, cond_a = cond_pred.unbind()
                        noref_v, noref_a = pred_noref.unbind()
                        new_a = cfg_a + (cond_a - noref_a) * scale
                        return comfy.nested_tensor.NestedTensor((cfg_v, new_a))
                    else:
                        is_audio_only = (len(cfg_result.shape) > 1 and cfg_result.shape[1] == audio_channels)
                        if is_audio_only:
                            return cfg_result + (cond_pred - pred_noref) * scale
                    return cfg_result + (cond_pred - pred_noref) * scale
                return post_cfg_function

            if has_audio_ref:
                model_to_use.set_model_sampler_post_cfg_function(get_post_cfg_function(model_to_use))
                model2_to_use.set_model_sampler_post_cfg_function(get_post_cfg_function(model2_to_use))
                model3_to_use.set_model_sampler_post_cfg_function(get_post_cfg_function(model3_to_use))

        # ==========================================
        # 6. SAMPLING & UPSCALING LOOP
        # ==========================================
        noise_obj = Noise_RandomNoise(seed_number)
        
        primary_guider = comfy.samplers.CFGGuider(model_to_use)
        primary_guider.set_cfg(primary_cfg)

        upsample_guider = comfy.samplers.CFGGuider(model2_to_use)
        upsample_guider.set_cfg(spatial_cfg)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        primary_sampler = build_custom_sampler(primary_sampler_name, eta, bongmath)
        
        if ',' in str(primary_steps):
            sigmas_list = re.findall(r"[-+]?(?:\d*\.*\d+)", str(primary_steps))
            primary_sigmas = torch.FloatTensor([float(i) for i in sigmas_list])
        else:
            try:
                p_steps = int(float(str(primary_steps).strip()))
                p_steps = max(1, p_steps)
            except ValueError:
                p_steps = 8

            if autoregressive_chunking:
                active_sec = min(length_in_seconds, chunk_size_seconds + context_window_seconds)
            else:
                active_sec = length_in_seconds
                
            pass_frames = int((active_sec * current_fps) + 1)
            pass_latents = ((pass_frames - 1) // 8) + 1
            active_tokens = pass_latents * (initial_height // 32) * (initial_width // 32)
            
            sigmas = torch.linspace(1.0, 0.0, p_steps + 1)
            max_shift, base_shift, terminal = 2.05, 0.95, 0.1
            x1, x2 = 1024, 4096
            mm_shift = (max_shift - base_shift) / (x2 - x1)
            b_shift = base_shift - mm_shift * x1
            
            sigma_shift = (active_tokens) * mm_shift + b_shift
            sigma_shift = min(sigma_shift, 13.5)

            power = 1
            sigmas = torch.where(sigmas != 0, math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power), 0)
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor_math = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor_math)
            sigmas[non_zero_mask] = stretched
            primary_sigmas = sigmas

        # ==========================================
        # 6.5 UNIFIED AUTOREGRESSIVE & ISOLATED SAMPLING LOOP
        # ==========================================
        pure_image_passthrough = False
        
        # --- INITIALIZE V2V BUFFERS TO PREVENT UNBOUND LOCAL ERRORS ---
        v2v_master_frames = None
        v2v_master_latents = None
        
        if primary_sampling:
            print(f"\n--- Base Generation Active: Building {num_prompts} Isolated Shot(s) ---")
            
            # --- BASE TEXTURE & TIMING LOCK ---
            # Generate pristine global noise for the entire video to prevent sliding hallucinations!
            global_base_v_noise = comfy.sample.prepare_noise(video_samples, seed_number, None).to(device)
            global_base_a_noise = comfy.sample.prepare_noise(audio_samples, seed_number, None).to(device)
            
            class BaseSlicedNoise:
                def __init__(self, v_noise, a_noise, v_start, a_start):
                    self.v_noise = v_noise
                    self.a_noise = a_noise
                    self.v_start = v_start
                    self.a_start = a_start
                    
                def generate_noise(self, input_latent):
                    v_len = input_latent["samples"].unbind()[0].shape[2]
                    a_len = input_latent["samples"].unbind()[1].shape[2]
                    
                    v_slice = self.v_noise[:, :, self.v_start : self.v_start + v_len].clone()
                    a_slice = self.a_noise[:, :, self.a_start : self.a_start + a_len].clone()
                    
                    # ROUNDING FAILSAFE: Pad the noise tensors if multi-shot math exceeds the master timeline!
                    if v_slice.shape[2] < v_len:
                        pad_len = v_len - v_slice.shape[2]
                        v_slice = torch.nn.functional.pad(v_slice, (0, 0, 0, 0, 0, pad_len))
                        
                    if a_slice.shape[2] < a_len:
                        pad_len = a_len - a_slice.shape[2]
                        a_slice = torch.nn.functional.pad(a_slice, (0, 0, 0, pad_len))
                        
                    return comfy.nested_tensor.NestedTensor((v_slice, a_slice))

            isolated_v_shots = []
            isolated_a_shots = []
            
            shot_duration = length_in_seconds / num_prompts
            
            # 0-Indexed local timeline mappers
            def sec_to_v_idx(sec):
                if sec <= 0.0: return 0
                frames = int(sec * current_fps) + 1 if bypass_img_ref else int(a * duplicate_frames) + int(sec * current_fps) + 9
                return ((frames - 1) // 8) + 1
                
            def sec_to_a_idx(sec):
                if sec <= 0.0: return 0
                frames = int(sec * current_fps) + 1 if bypass_img_ref else int(a * duplicate_frames) + int(sec * current_fps) + 9
                synced_frames = int(((((frames - 1) // 8) + 1) - 1) * 8) + 1
                get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
                return get_audio_latents_func(synced_frames, int(current_fps))

            v_len = sec_to_v_idx(shot_duration)
            a_len = sec_to_a_idx(shot_duration)
            
            x0_output = {}
            for s in range(num_prompts):
                shot_start_sec = s * shot_duration
                shot_end_sec = (s + 1) * shot_duration
                
                # Simulate the chunking math to get the exact total chunks for this shot
                total_chunks_in_shot = 0
                temp_sec = shot_start_sec
                while temp_sec < shot_end_sec:
                    total_chunks_in_shot += 1
                    if not autoregressive_chunking:
                        temp_sec = shot_end_sec
                    else:
                        temp_sec = min(temp_sec + chunk_size_seconds, shot_end_sec)
                
                print(f"\n-> Generating Isolated Shot {s+1}/{num_prompts} (Total Chunks: {total_chunks_in_shot})...")
                
                # Map absolute global latents to determine exact local length
                base_v_lat, base_a_lat = get_latent_counts(shot_start_sec)
                end_v_lat, end_a_lat = get_latent_counts(shot_end_sec)
                
                v_len = end_v_lat - base_v_lat
                a_len = end_a_lat - base_a_lat
                
                # 1. Pull the initial latents and masks from the master setup tensors (Enables V2V & Image Injections!)
                pass_v_samples = torch.zeros([batch_size, 128, v_len, initial_height // 32, initial_width // 32], device=device)
                pass_v_masks = torch.ones_like(pass_v_samples)
                
                actual_v_len = min(end_v_lat, video_samples.shape[2]) - base_v_lat
                if actual_v_len > 0:
                    pass_v_samples[:, :, :actual_v_len] = video_samples[:, :, base_v_lat:base_v_lat+actual_v_len]
                    # FIX: video_noise_mask is 4D [B, F, H, W]. We slice it and unsqueeze the channel dimension to broadcast to 5D!
                    pass_v_masks[:, :, :actual_v_len, :, :] = video_noise_mask[:, base_v_lat:base_v_lat+actual_v_len, :, :].unsqueeze(1)
                
                pass_a_samples = torch.zeros([batch_size, z_channels, a_len, audio_freq], device=device)
                # ENFORCE 3D AUDIO MASK [Batch, Frames, Freq]
                pass_a_masks = torch.ones((batch_size, a_len, audio_freq), dtype=torch.float32, device=device)
                
                # 2. Slice the correct audio chunk for this shot
                audio_content_offset = setup_total_latents if has_audio_input else 0
                a_start_master = base_a_lat + audio_content_offset
                a_end_master = min(end_a_lat + audio_content_offset, audio_samples.shape[2])
                actual_a_len = a_end_master - a_start_master
                if actual_a_len > 0:
                    pass_a_samples[:, :, :actual_a_len] = audio_samples[:, :, a_start_master:a_end_master]
                    # 3D INDEXING: Drop the channel dimension so the slices match perfectly!
                    pass_a_masks[:, :actual_a_len, :] = audio_noise_mask[:, a_start_master:a_end_master, :]
                    print(f"   [DEBUG-AUDIO-COND] shot={s+1} shot_start_sec={shot_start_sec:.2f} "
                      f"setup_total_latents={setup_total_latents} base_a_lat={base_a_lat} "
                      f"a_start_master={a_start_master} a_end_master={a_end_master} "
                      f"audio_samples.shape={tuple(audio_samples.shape)}")
                    
                # 4. Autoregressive Inner Chunking Loop!
                curr_global_sec = shot_start_sec
                chunk_idx = 1
                
                # Master pixel tracking buffer used exclusively for true V2V re-encoding and perceptual crossfades
                v2v_master_frames = None
                
                while curr_global_sec < shot_end_sec:
                    comfy.model_management.soft_empty_cache()
                    chunk_start_sec = curr_global_sec
                    
                    # Respect UI Settings for chunking boundaries
                    if not autoregressive_chunking:
                        curr_global_sec = shot_end_sec
                    else:
                        curr_global_sec = min(chunk_start_sec + chunk_size_seconds, shot_end_sec)
                    
                    curr_v_global, curr_a_global = get_latent_counts(curr_global_sec)
                    prev_v_global, prev_a_global = get_latent_counts(chunk_start_sec)
                    
                    # Zero-indexed local array matching
                    curr_v_lat = curr_v_global - base_v_lat
                    curr_a_lat = curr_a_global - base_a_lat
                    prev_v_lat = prev_v_global - base_v_lat
                    prev_a_lat = prev_a_global - base_a_lat
                    
                    # =========================================================================
                    # PATHWAY A: THE EXACT WORKFLOW V2V CHAINING ENGINE (SINGLE SHOT ONLY)
                    # =========================================================================
                    if num_prompts == 1:
                        import comfy_extras.nodes_lt as nodes_lt
                        import kornia
                        import copy
                        fr_device = comfy.model_management.get_torch_device()
                        
                        # Dynamically calculate local pixel heights/widths directly from the active latent dimensions
                        v2v_h = pass_v_samples.shape[3] * 32
                        v2v_w = pass_v_samples.shape[4] * 32
                        
                        # Calculate exact reference frame count matching your workflow nodes (~73 frames at 24fps)
                        ref_frames_count = int(((round((3 * current_fps - 1) / 8)) * 8) + 1)
                        
                        if chunk_start_sec == shot_start_sec:
                            # --- CHUNK 1: GENERATE CONSOLIDATED BASE CLIP (USES MAIN UI SETTINGS) ---
                            print(f"   -> [V2V Chain - Chunk 1] Generating initial base clip: 0.0s to {curr_global_sec:.2f}s")
                            
                            chunk_v_samples = pass_v_samples[:, :, :curr_v_lat].clone()
                            chunk_v_masks = torch.ones((batch_size, 1, curr_v_lat, initial_height // 32, initial_width // 32), device=device)
                            
                            if video_noise_mask is not None:
                                chunk_v_masks[:, 0, :min(curr_v_lat, video_noise_mask.shape[1]), :, :] = video_noise_mask[:, :curr_v_lat, :, :]
                                
                            _, true_a_start = get_latent_counts(shot_start_sec)
                            _, true_a_end = get_latent_counts(curr_global_sec)
                            req_a_len = true_a_end - true_a_start
                            chunk_a_samples = audio_samples[:, :, true_a_start:true_a_start+req_a_len].clone()
                            audio_mask_val = 0.0 if (has_audio_input or has_audio_ref) else 1.0
                            chunk_a_masks = torch.full((batch_size, 1, req_a_len, audio_samples.shape[3]), audio_mask_val, device=device)
                            
                            chunk_sigmas = primary_sigmas.to(device)
                            p_tensor, p_dict = final_positive[s][0], final_positive[s][1].copy()
                            n_tensor, n_dict = final_negative[0][0], final_negative[0][1].copy()
                            
                            primary_guider.set_conds([[p_tensor, p_dict]], [[n_tensor, n_dict]])
                            chunk_noise_obj = Noise_RandomNoise(seed_number)
                            
                            base_cb = latent_preview.prepare_callback(primary_guider.model_patcher, chunk_sigmas.shape[-1] - 1, x0_output)
                            callback = wrap_callback(base_cb, f"V2V Base Pass (Chunk {chunk_idx})")
                            
                            av_samples = comfy.nested_tensor.NestedTensor((chunk_v_samples, chunk_a_samples))
                            av_masks = comfy.nested_tensor.NestedTensor((chunk_v_masks, chunk_a_masks))
                            current_latent = {"samples": av_samples}
                            
                            if hasattr(model_to_use.model, "likeness_state"):
                                model_to_use.model.likeness_state["is_shot_start"] = True
                                
                            sampled_chunk = primary_guider.sample(chunk_noise_obj.generate_noise(current_latent), current_latent["samples"], primary_sampler, chunk_sigmas, denoise_mask=av_masks, callback=callback, disable_pbar=disable_pbar, seed=seed_number)
                            res_v, res_a = sampled_chunk.unbind()
                            
                            pass_v_samples[:, :, :curr_v_lat] = res_v.to(device)
                            if not has_audio_input and res_a.shape[2] > 0:
                                pass_a_samples[:, :, :curr_a_lat] = res_a[:, :, :curr_a_lat].to(device)
                                
                            print("   -> Initializing V2V latent-space chaining...")
                            v2v_master_latents = res_v.clone().to(device)
                            v2v_master_frames = None
                            
                        else:
                            # --- CHUNK 2+: ROLLING ISOLATED LATENT CHAINING TIMELINE EXTENSION ---
                            print(f"   -> [V2V Chain - Chunk {chunk_idx}] Extending timeline from {chunk_start_sec:.2f}s to {curr_global_sec:.2f}s")
                            
                            # 1. Extract trailing reference frames directly from the latent buffer
                            ref_lat_len = ((ref_frames_count - 1) // 8) + 1
                            video_latent_ref = v2v_master_latents[:, :, -ref_lat_len:].clone().to(device)
                            
                            # 3. LTXVAudioVideoMask padding configuration
                            delta_v_lats = curr_v_lat - prev_v_lat
                            
                            # --- STYLE RETENTION REFERENCE RE-INJECTION ---
                            # Check if reference-to-video mode is active and reference latents are available
                            inject_ref = (video_mode == "reference-to-video" and final_pixels is not None)
                            inject_len = encoded_t.shape[2] if inject_ref else 0
                            
                            TAIL_GUARD_LATS = 10
                            total_required_v = ref_lat_len + delta_v_lats + TAIL_GUARD_LATS + inject_len
                            
                            # Allocate a fresh padded latent canvas
                            chunk_v_samples = torch.zeros([batch_size, 128, total_required_v, initial_height // 32, initial_width // 32], device=device)
                            chunk_v_samples[:, :, :ref_lat_len] = video_latent_ref
                            
                            # Set strict 5D noise mask: freeze lookback reference at 0.0, open new pads to 1.0
                            padded_v_masks = torch.ones((batch_size, 1, total_required_v, initial_height // 32, initial_width // 32), device=device)
                            padded_v_masks[:, :, :ref_lat_len, :, :] = 0.0
                            
                            # If reference mode is active, append reference latents to the guard tail as a constant contextual anchor
                            if inject_ref:
                                inject_start_idx = ref_lat_len + delta_v_lats + TAIL_GUARD_LATS
                                chunk_v_samples[:, :, inject_start_idx : inject_start_idx + inject_len] = encoded_t.to(device)
                                for i in range(inject_len):
                                    pixel_idx = min(i * time_scale_factor, max(0, len(strengths) - 1))
                                    padded_v_masks[:, :, inject_start_idx + i, :, :] = 1.0 - strengths[pixel_idx]
                            
                            lookback_offset_sec = ref_frames_count / current_fps
                            lookback_start_sec = max(shot_start_sec, chunk_start_sec - lookback_offset_sec)
                            
                            _, lookback_a_global = get_latent_counts(lookback_start_sec)
                            lookback_a_lat = max(0, lookback_a_global - base_a_lat)
                            
                            v_frames_matching = ((total_required_v - 1) * 8) + 1
                            req_a_latents_matching = get_audio_latents_func(v_frames_matching, int(current_fps))
                            
                            chunk_a_samples = torch.zeros((batch_size, z_channels, req_a_latents_matching, audio_freq), device=device)
                            chunk_a_masks = torch.zeros((batch_size, 1, req_a_latents_matching, audio_samples.shape[3]), device=device)
                            
                            available_a_len = min(req_a_latents_matching, audio_samples.shape[2] - lookback_a_lat)
                            if available_a_len > 0:
                                chunk_a_samples[:, :, :available_a_len] = audio_samples[:, :, lookback_a_lat : lookback_a_lat + available_a_len]
                            
                            extension_model = kwargs.get("model1.5_extend")
                            if extension_model is None:
                                extension_model = model_to_use
                                
                            ext_model_clone = extension_model.clone()
                            ms_obj = ext_model_clone.get_model_object("model_sampling")
                            if hasattr(ms_obj, "shift"):
                                ms_obj.shift = 13.0
                            
                            ext_sigmas = comfy.samplers.calculate_sigmas(ms_obj, "linear_quadratic", 8).to(device)
                            ext_sampler = comfy.samplers.sampler_object("euler_ancestral")
                            
                            # --- DEEP ISOLATION TO BLOCK CONDITIONING LEAKAGE ---
                            import copy
                            p_tensor, p_dict_orig = final_positive[s]
                            p_dict = {k: (v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)) for k, v in p_dict_orig.items()}
                            p_dict.pop("c_concat", None)
                            p_dict.pop("guide_attention_entries", None)
                            p_dict.pop("keyframe_idxs", None)
                            p_dict["start_percent"] = 0.0
                            
                            n_tensor, n_dict_orig = final_negative[0]
                            n_dict = {k: (v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)) for k, v in n_dict_orig.items()}
                            n_dict.pop("c_concat", None)
                            n_dict.pop("guide_attention_entries", None)
                            n_dict.pop("keyframe_idxs", None)
                            n_dict["start_percent"] = 0.0
                            
                            # 4. LTXVAddLatentGuide Keyframe Registration
                            scale_factors = video_vae.downscale_index_formula
                            p_cond_box, n_cond_box, guided_latent, guided_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                                positive=[[p_tensor, p_dict]], negative=[[n_tensor, n_dict]],
                                frame_idx=0, latent_image=chunk_v_samples, noise_mask=padded_v_masks,
                                guiding_latent=video_latent_ref, strength=1.0, scale_factors=scale_factors
                            )
                            
                            ext_guider = comfy.samplers.CFGGuider(ext_model_clone)
                            ext_guider.set_cfg(primary_cfg)
                            ext_guider.set_conds(p_cond_box, n_cond_box)
                            
                            chunk_noise_obj = Noise_RandomNoise(seed_number + chunk_idx)
                            base_cb = latent_preview.prepare_callback(ext_guider.model_patcher, ext_sigmas.shape[-1] - 1, x0_output)
                            callback = wrap_callback(base_cb, f"V2V Extension Pass (Chunk {chunk_idx})")
                            
                            av_samples = comfy.nested_tensor.NestedTensor((guided_latent, chunk_a_samples))
                            av_masks = comfy.nested_tensor.NestedTensor((guided_mask, chunk_a_masks))
                            current_latent = {"samples": av_samples}
                            
                            if hasattr(ext_model_clone.model, "likeness_state"):
                                ext_model_clone.model.likeness_state["is_shot_start"] = False
                                
                            sampled_chunk = ext_guider.sample(chunk_noise_obj.generate_noise(current_latent), current_latent["samples"], ext_sampler, ext_sigmas, denoise_mask=av_masks, callback=callback, disable_pbar=disable_pbar, seed=seed_number + chunk_idx)
                            res_v, res_a = sampled_chunk.unbind()
                            
                            if res_v.ndim == 4:
                                res_v = res_v.unsqueeze(0)
                                
                            res_v = res_v.to(v2v_master_latents.device)
                            
                            # 5. Latent Space Crossfade
                            blend_src = v2v_master_latents[:, :, -ref_lat_len:].clone().to(device)
                            blend_dst = res_v[:, :, :ref_lat_len].clone().to(device)
                            
                            alpha = torch.linspace(0, 1, ref_lat_len + 2, device=device, dtype=res_v.dtype)[1:-1]
                            alpha = alpha.view(1, 1, -1, 1, 1)
                            blended_latent = (1.0 - alpha) * blend_src + alpha * blend_dst
                            
                            # Overwrite trailing reference frames in latent space
                            v2v_master_latents[:, :, -ref_lat_len:] = blended_latent
                            
                            # Keep only the reliable middle section; drop the guard tail entirely
                            kept_new_lats = delta_v_lats
                            v2v_master_latents = torch.cat((v2v_master_latents, res_v[:, :, ref_lat_len : ref_lat_len + kept_new_lats]), dim=2)
                            
                            # Dynamically update the global master array with the latent sequence
                            pass_v_samples = v2v_master_latents.clone()
                            
                            if not has_audio_input and res_a.shape[2] > 0:
                                lookback_audio_len = prev_a_lat - lookback_a_lat
                                pass_a_samples[:, :, prev_a_lat:curr_a_lat] = res_a[:, :, lookback_audio_len:lookback_audio_len + (curr_a_lat - prev_a_lat)].to(device)
                                audio_samples[:, :, lookback_a_lat:curr_a_lat] = pass_a_samples[:, :, lookback_a_lat:curr_a_lat].to(audio_samples.device)

                    # =========================================================================
                    # PATHWAY B: ORIGINAL UNTOUCHED SLIDING LOOP (MULTI-SHOT DIRECTORS CUTS)
                    # =========================================================================
                    else:
                        if chunk_start_sec == shot_start_sec:
                            ctx_v_lat = 0
                            ctx_a_lat = 0
                            context_start_sec = shot_start_sec
                            print(f"   -> [Shot {s+1}/{num_prompts} - Chunk {chunk_idx}/{total_chunks_in_shot}] 0.0s to {curr_global_sec - shot_start_sec:.2f}s (Base Block)")
                        else:
                            context_start_sec = max(shot_start_sec, chunk_start_sec - context_window_seconds)
                            ctx_v_global, ctx_a_global = get_latent_counts(context_start_sec)
                            ctx_v_lat = ctx_v_global - base_v_lat
                            ctx_a_lat = ctx_a_global - base_a_lat
                            print(f"   -> [Shot {s+1}/{num_prompts} - Chunk {chunk_idx}/{total_chunks_in_shot}] {chunk_start_sec - shot_start_sec:.2f}s to {curr_global_sec - shot_start_sec:.2f}s (Sliding Context Lookback Window)")
                            
                        chunk_v_samples = pass_v_samples[:, :, ctx_v_lat:curr_v_lat].clone()
                        chunk_v_masks = torch.ones_like(chunk_v_samples)
                        ctx_len = prev_v_lat - ctx_v_lat
                        if chunk_start_sec > shot_start_sec and ctx_len > 0:
                            chunk_v_masks[:, :, :ctx_len, :, :] = 0.0

                        # audio_content_offset = -((16 * s)) if (has_audio_input and s > 0) else -1
                        audio_content_offset = 0
                        _, true_a_start = get_latent_counts(context_start_sec)
                        _, true_a_end = get_latent_counts(curr_global_sec)
                        req_a_len = true_a_end - true_a_start
                        true_a_start = max(0, true_a_start + audio_content_offset)
                        print(f" [DEBUG-TRUE-A] s={s} shot_start_sec={shot_start_sec} audio_content_offset={audio_content_offset} true_a_start={true_a_start}")
                        
                        available_lats = max(0, audio_samples.shape[2] - true_a_start)
                        slice_len = min(req_a_len, available_lats)
                        
                        chunk_a_samples = audio_samples[:, :, true_a_start : true_a_start + slice_len].clone()
                        chunk_a_masks = audio_noise_mask[:, true_a_start : true_a_start + slice_len, :].clone()
                        
                        if chunk_a_samples.shape[2] < req_a_len:
                            pad_len = req_a_len - chunk_a_samples.shape[2]
                            chunk_a_samples = torch.nn.functional.pad(chunk_a_samples, (0,0,0,pad_len), value=0)
                            chunk_a_masks = torch.nn.functional.pad(chunk_a_masks, (0,0,0,pad_len), value=0)

                        if has_audio_input or has_audio_ref:
                            chunk_a_masks[:] = 0.0
                        else:
                            chunk_a_masks[:] = 1.0
                        chunk_sigmas = primary_sigmas.to(device)

                        p_tensor, p_dict_orig = final_positive[s]
                        p_dict = p_dict_orig.copy()
                        p_dict["start_percent"] = 0.0
                        
                        v_cond = torch.zeros((1, 128, chunk_v_samples.shape[2], chunk_v_samples.shape[3], chunk_v_samples.shape[4]), device=device)
                        if chunk_start_sec > shot_start_sec and ctx_len > 0:
                            v_cond[:, :, :ctx_len, :, :] = pass_v_samples[:, :, ctx_v_lat:prev_v_lat].clone()
                        else:
                            if "c_concat" in p_dict:
                                v_cond[:, :, :1, :, :] = p_dict["c_concat"][:, :, :1, :, :]
                                
                        p_dict["c_concat"] = v_cond
                        chunk_pos = [[p_tensor, p_dict]]
                            
                        chunk_neg = []
                        for n in final_negative:
                            n_dict = n[1].copy()
                            n_dict["start_percent"] = 0.0
                            n_dict["c_concat"] = torch.zeros_like(v_cond)
                            chunk_neg.append([n[0], n_dict])
                            
                        primary_guider.set_conds(chunk_pos, chunk_neg)
                        base_cb = latent_preview.prepare_callback(primary_guider.model_patcher, chunk_sigmas.shape[-1] - 1, x0_output)
                        callback = wrap_callback(base_cb, f"Shot {s+1}/{num_prompts} (Chunk {chunk_idx}/{total_chunks_in_shot})")
                        
                        abs_v_start = base_v_lat + ctx_v_lat
                        abs_a_start = base_a_lat + ctx_a_lat
                        chunk_noise_obj = BaseSlicedNoise(global_base_v_noise, global_base_a_noise, abs_v_start, abs_a_start)
                        
                        av_samples = comfy.nested_tensor.NestedTensor((chunk_v_samples, chunk_a_samples))
                        av_masks = comfy.nested_tensor.NestedTensor((chunk_v_masks, chunk_a_masks))
                        current_latent = {"samples": av_samples}
                        
                        if hasattr(model_to_use.model, "likeness_state"):
                            model_to_use.model.likeness_state["is_shot_start"] = (chunk_start_sec == shot_start_sec)
                            
                        print(f"   [DEBUG-AV-LEN] s={s} chunk_v_samples.shape={tuple(chunk_v_samples.shape)} "
                              f"chunk_a_samples.shape={tuple(chunk_a_samples.shape)} req_a_len={req_a_len} "
                              f"v_cond.shape={tuple(v_cond.shape)}")
                        sampled_chunk = primary_guider.sample(chunk_noise_obj.generate_noise(current_latent), current_latent["samples"], primary_sampler, chunk_sigmas, denoise_mask=av_masks, callback=callback, disable_pbar=disable_pbar, seed=seed_number)
                        res_v, res_a = sampled_chunk.unbind()
                        print(f"   [DEBUG-AV-OUT] s={s} res_v.shape={tuple(res_v.shape)} res_a.shape={tuple(res_a.shape)}")

                        # DEBUG: self-consistency check, shot 2 only (s==1). Decodes the model's RAW, UNTRIMMED
                        # output -- before the razor blade, before final assembly, before any offset math --
                        # and pairs the video with ITS OWN generated audio (not the external pristine track).
                        # If lip-sync looks right here, the bug is entirely downstream of generation (razor
                        # blade / extraction / offset). If it's already wrong here, the bug is upstream, in
                        # what we feed the model as conditioning -- and no downstream offset tuning can fix it.
                        if s == 1:
                            try:
                                import subprocess as _dbg_subprocess
                                dbg_dir = folder_paths.get_temp_directory()
                                dbg_video_path = os.path.join(dbg_dir, "debug_shot2_raw_video.mp4")
                                dbg_audio_path = os.path.join(dbg_dir, "debug_shot2_raw_audio.wav")
                                dbg_mux_path = os.path.join(dbg_dir, "debug_shot2_SELF_CONSISTENCY.mp4")

                                dbg_v_pixels = video_vae.decode(res_v).cpu()
                                dbg_v_h = res_v.shape[3] * 32
                                dbg_v_w = res_v.shape[4] * 32
                                dbg_v_pixels = dbg_v_pixels.view(-1, dbg_v_h, dbg_v_w, 3)

                                dbg_a_wf = audio_vae.decode(res_a).to(device).movedim(-1, 1)
                                dbg_a_wf = dbg_a_wf[0].cpu()
                                if dbg_a_wf.ndim == 3:
                                    dbg_a_wf = dbg_a_wf.squeeze(0)
                                dbg_sample_rate = int(getattr(audio_vae, "output_sample_rate", audio_vae.first_stage_model.output_sample_rate))
                                torchaudio.save(dbg_audio_path, dbg_a_wf, dbg_sample_rate)

                                v_cmd = [
                                    "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                                    "-s", f"{dbg_v_w}x{dbg_v_h}", "-pix_fmt", "rgb24",
                                    "-r", str(current_fps), "-i", "-",
                                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                                    "-pix_fmt", "yuv420p", dbg_video_path
                                ]
                                v_proc = _dbg_subprocess.Popen(v_cmd, stdin=_dbg_subprocess.PIPE, stdout=_dbg_subprocess.DEVNULL, stderr=_dbg_subprocess.DEVNULL)
                                v_np = (dbg_v_pixels.numpy() * 255.0).clip(0, 255).astype(np.uint8)
                                v_proc.stdin.write(v_np.tobytes())
                                v_proc.stdin.close()
                                v_proc.wait()

                                mux_cmd = [
                                    "ffmpeg", "-y", "-i", dbg_video_path, "-i", dbg_audio_path,
                                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                                    "-shortest", dbg_mux_path
                                ]
                                _dbg_subprocess.run(mux_cmd, stdout=_dbg_subprocess.DEVNULL, stderr=_dbg_subprocess.DEVNULL)

                                print(f"   [DEBUG-SELF-CONSISTENCY] Saved RAW shot-2 output ({dbg_v_pixels.shape[0]} "
                                      f"untrimmed frames, video paired with its OWN generated audio) to: {dbg_mux_path}")
                            except Exception as e:
                                print(f"   [DEBUG-SELF-CONSISTENCY] Failed to save debug clip: {e}")

                        res_v = res_v.to(device)
                        res_a = res_a.to(device)
                        if res_v.ndim == 4: res_v = res_v.unsqueeze(0)
                        if res_a.ndim == 3: res_a = res_a.unsqueeze(0)
                        
                        gen_v_len = curr_v_lat - prev_v_lat
                        if chunk_start_sec > shot_start_sec and ctx_len > 0:
                            new_lats = res_v[:, :, ctx_len:]
                            pass_v_samples[:, :, prev_v_lat:curr_v_lat] = new_lats
                        else:
                            pass_v_samples[:, :, prev_v_lat:curr_v_lat] = res_v[:, :, -gen_v_len:]
                        
                        # TESTING: pass_a_samples is normally pre-filled from the pristine source track and only
                        # overwritten here `if not has_audio_input` -- with a source track provided, that write
                        # never runs, so pass_a_samples stays pristine regardless of FORCE_GENERATED_AUDIO_FOR_TESTING
                        # further down (that flag only changes which code path DECODES pass_a_samples; it doesn't
                        # change what's IN it). This flag makes the write actually happen, so the final output
                        # reflects what the model generated. Conditioning/generation itself is untouched -- the
                        # model still sees the real song either way, this only changes what gets kept afterward.
                        FORCE_GENERATED_AUDIO_WRITEBACK = True
                        if not has_audio_input or FORCE_GENERATED_AUDIO_WRITEBACK:
                            gen_a_len = curr_a_lat - prev_a_lat
                            actual_gen_a = min(gen_a_len, res_a.shape[2])
                            true_a_end_pos = min(true_a_start + req_a_len, audio_samples.shape[2])
                            actual_gen_a = min(actual_gen_a, true_a_end_pos)
                            if actual_gen_a > 0:
                                pass_a_samples[:, :, prev_a_lat:prev_a_lat + actual_gen_a] = res_a[:, :, -actual_gen_a:]
                                audio_samples[:, :, true_a_end_pos - actual_gen_a : true_a_end_pos] = res_a[:, :, -actual_gen_a:]
                    
                    chunk_idx += 1
                        
                # End of Autoregressive Loop for Shot S
                # =========================================================
                # THE LATENT RAZOR BLADE (EXACT MATHEMATICAL TIMELINE LOCK)
                # =========================================================
                RAZOR_BLADE_ENABLED = False  # TEMP DIAGNOSTIC: shots keep their full raw length (warm-up
                                              # frames included, nothing trimmed to match shot_duration).
                                              # Set back to True to restore normal trimming.
                print(f"   [DEBUG-PRE-RAZOR] shot={s+1} pass_v_samples.shape={tuple(pass_v_samples.shape)} "
                      f"pass_a_samples.shape={tuple(pass_a_samples.shape)}")

                if not RAZOR_BLADE_ENABLED:
                    print(f"   -> LATENT RAZOR BLADE: DISABLED (diagnostic). Shot {s+1} kept RAW/untrimmed.")
                    isolated_v_shots.append(pass_v_samples)
                    isolated_a_shots.append(pass_a_samples)
                    continue

                target_v_frames = int(round(shot_duration * current_fps))
                
                target_v_latents = ((target_v_frames - 1) // 8) + 1
                if ((target_v_latents - 1) * 8 + 1) < target_v_frames:
                    target_v_latents += 1
                    
                target_a_latents = get_audio_latents_func(target_v_frames, int(current_fps))
                
                if s == 0:
                    shot1_target_frames = target_v_frames + (out_ref_frame_count if not bypass_img_ref else 0)
                    shot1_target_latents = ((shot1_target_frames - 1) // 8) + 1
                    if ((shot1_target_latents - 1) * 8 + 1) < shot1_target_frames:
                        shot1_target_latents += 1
                        
                    shot1_target_a = get_audio_latents_func(shot1_target_frames, int(current_fps))
                    
                    if pass_v_samples.shape[2] > shot1_target_latents:
                        pass_v_samples = pass_v_samples[:, :, :shot1_target_latents]
                        pass_a_samples = pass_a_samples[:, :, :shot1_target_a]
                        print(f"   -> LATENT RAZOR BLADE: Sliced overhang off Shot 1. Locked to {shot1_target_latents} latents.")
                else:
                    if not bypass_img_ref and out_ref_frame_count > 0:
                        setup_latents = out_ref_frame_count // 8
                        a_skip_latents = get_audio_latents_func(setup_latents * 8, int(current_fps))
                        
                        # FIX: pass_a_samples only contains model-generated audio (with real warm-up baked in,
                        # confirmed by the shot-2 self-consistency test) when has_audio_input is False. When a
                        # source track is provided, pass_a_samples is pristine audio copied in before generation
                        # (line ~1984) and NEVER overwritten by res_a (see the `if not has_audio_input:` gate
                        # above) -- it has no warm-up to skip. Applying a_skip_latents to it was deleting ~17
                        # latents of real, correctly-positioned song content from every shot 2+, which is what
                        # compounded into the shot-over-shot drift.
                        a_skip_for_audio = a_skip_latents if not has_audio_input else 0
                        
                        if pass_v_samples.shape[2] >= setup_latents + target_v_latents:
                            pass_v_samples = pass_v_samples[:, :, setup_latents : setup_latents + target_v_latents]
                            pass_a_samples = pass_a_samples[:, :, a_skip_for_audio : a_skip_for_audio + target_a_latents]
                            print(f"   -> LATENT RAZOR BLADE: Sliced {setup_latents} setup latents & overhang off Shot {s+1}. Locked to {target_v_latents} latents.")
                            print(f"   [DEBUG-AUDIO-RAZOR] shot={s+1} setup_latents={setup_latents} a_skip_latents={a_skip_latents} "
                                  f"a_skip_for_audio={a_skip_for_audio} has_audio_input={has_audio_input} "
                                  f"target_v_latents={target_v_latents} target_a_latents={target_a_latents} "
                                  f"pass_v_samples.shape={tuple(pass_v_samples.shape)} pass_a_samples.shape_after_slice={tuple(pass_a_samples.shape)}")
                        else:
                            pass_v_samples = pass_v_samples[:, :, setup_latents:]
                            pass_a_samples = pass_a_samples[:, :, a_skip_for_audio:]
                    else:
                        if pass_v_samples.shape[2] > target_v_latents:
                            pass_v_samples = pass_v_samples[:, :, :target_v_latents]
                            pass_a_samples = pass_a_samples[:, :, :target_a_latents]

                isolated_v_shots.append(pass_v_samples)
                isolated_a_shots.append(pass_a_samples)
            
            # 5. Concatenate the pristine, 0-indexed isolated shots end-to-end to form the timeline
            global_v_samples = torch.cat(isolated_v_shots, dim=2)
            global_a_samples = torch.cat(isolated_a_shots, dim=2)
            sampled_tensor = comfy.nested_tensor.NestedTensor((global_v_samples, global_a_samples)).to(device)
        
        else:
            # PURE PASSTHROUGH only triggers if ALL generation and upscaling stages are explicitly disabled!
            pure_image_passthrough = (latent_override is None) and (images is not None) and not (primary_sampling or spatial_upscale or temporal_upscale or restore_faces or enable_colorfx)
            
            if pure_image_passthrough:
                print("\n--- Primary Sampling Bypassed: Pure Image Passthrough Detected ---")
                print("-> Routing input image(s) directly to pixel-level post-processing.")
                sampled_tensor = None
            elif (latent_override is None) and (images is not None):
                print("\n--- Primary Sampling Bypassed: Smooth Causal VAE Encoding Input Image(s) ---")
                img_in = images if images.ndim == 4 else images.squeeze(0)
                if img_in.shape[-1] > 3:
                    img_in = img_in[..., :3]

                total_input_frames = img_in.shape[0]

                # Attempt unbroken continuous VAE encoding to preserve 100% temporal continuity
                try:
                    with torch.no_grad():
                        encoded_v = video_vae.encode(img_in).to(device="cpu", dtype=torch.float16)
                    print("-> Smooth full-pass VAE encoding complete.")
                except Exception as e:
                    print(f"-> Full-pass VAE encode failed ({e}). Switching to Causal Overlapped Chunk Encoding...")
                    
                    # Overlapped Chunk Encoder aligned to 8k + 1 frames (33 frames = 5 latents)
                    v_chunks = []
                    chunk_size = 33  # Matches 8k + 1 alignment (5 latents)
                    overlap = 8      # 8 frames overlap = 1 latent overlap
                    step = chunk_size - overlap

                    start_idx = 0
                    while start_idx < total_input_frames:
                        end_idx = min(start_idx + chunk_size, total_input_frames)
                        img_slice = img_in[start_idx:end_idx]

                        if img_slice.shape[0] < 9 and len(v_chunks) > 0:
                            break

                        with torch.no_grad():
                            lat_chunk = video_vae.encode(img_slice).to(device="cpu", dtype=torch.float16)

                        if start_idx == 0:
                            v_chunks.append(lat_chunk)
                        else:
                            # Drop the 1 overlapping latent frame at boundary to preserve unbroken causal motion
                            v_chunks.append(lat_chunk[:, :, 1:])

                        if end_idx >= total_input_frames:
                            break

                        start_idx += step
                        import gc
                        gc.collect()
                        comfy.model_management.soft_empty_cache()

                    encoded_v = torch.cat(v_chunks, dim=2)

                # Dynamically generate matching audio latent canvas for downstream pipeline
                v_frames_encoded = ((encoded_v.shape[2] - 1) * 8) + 1 if encoded_v.shape[2] > 0 else 0
                req_a_latents = get_audio_latents_func(v_frames_encoded, int(current_fps))

                a_override_slice = torch.zeros((batch_size, z_channels, req_a_latents, audio_freq), device=device)
                if audio_samples is not None:
                    copy_len = min(req_a_latents, audio_samples.shape[2])
                    a_override_slice[:, :, :copy_len] = audio_samples[:, :, :copy_len]

                sampled_tensor = comfy.nested_tensor.NestedTensor((encoded_v, a_override_slice)).to(device)

        # ==========================================
        # DEBUG: LATENT SAVER HELPER
        # ==========================================
        def debug_save_latent(tensor_data, stage_name):
            try:
                import os
                import torch
                import folder_paths
                temp_dir = folder_paths.get_temp_directory()
                safe_name = stage_name.replace(" ", "_").replace("/", "_").lower()
                filepath = os.path.join(temp_dir, f"ltx_av_temp_{safe_name}.latent")
                
                to_save = {}
                # Handle dictionary packaging (like final_latent)
                if isinstance(tensor_data, dict):
                    for k, v in tensor_data.items():
                        if hasattr(v, "unbind"):
                            v_unb = v.unbind()
                            to_save[f"{k}_v"] = v_unb[0].cpu()
                            to_save[f"{k}_a"] = v_unb[1].cpu()
                        elif isinstance(v, torch.Tensor):
                            to_save[k] = v.cpu()
                        else:
                            to_save[k] = v
                # Handle raw NestedTensor packaging (like sampled_tensor)
                else:
                    v_unb = tensor_data.unbind()
                    to_save["samples_v"] = v_unb[0].cpu()
                    to_save["samples_a"] = v_unb[1].cpu()
                    
                torch.save(to_save, filepath)
                print(f"-> [DEBUG] Saved {stage_name} latent to {filepath}")
            except Exception as e:
                print(f"-> [DEBUG] Failed to save {stage_name} latent: {e}")

        if not pure_image_passthrough:
            debug_save_latent(sampled_tensor, "stage1_base")

        # ==========================================
        # MID-GENERATION STAGE 1 PREVIEW (MP4 MUX)
        # ==========================================
        if enable_preview and (spatial_upscale or temporal_upscale) and not pure_image_passthrough:
            print("\n--- Generating Stage 1 Preview (Direct Streaming to Disk) ---")
            import uuid
            uid = node_id if node_id is not None else str(uuid.uuid4())
            
            v_samps_prev, a_samps_prev = sampled_tensor.unbind()
            v_samps_prev = v_samps_prev.to(device)
            a_samps_prev = a_samps_prev.to(device)

            total_splits = 1
            if autoregressive_chunking and chunk_size_seconds < length_in_seconds:
                total_splits = int(round(length_in_seconds / chunk_size_seconds))
            if num_prompts > 1:
                total_splits = num_prompts # Override to ensure strict shot boundaries

            # --- 1. AUDIO DECODE & SAVE ---
            # We decode the audio first so the file is ready for FFmpeg to multiplex
            temp_dir = folder_paths.get_temp_directory()
            preview_filename = f"stage1_preview_{uid}.mp4"
            preview_path = os.path.join(temp_dir, preview_filename)
            audio_path = os.path.join(temp_dir, f"stage1_preview_audio_{uid}.wav")
            out_sample_rate = int(getattr(audio_vae, "output_sample_rate", audio_vae.first_stage_model.output_sample_rate))
            save_sample_rate = out_sample_rate
            
            try:
                if has_audio_input:
                    print("--- Multiplexing Original Source Audio into Preview ---")
                    wf_out = master_wf[0].cpu()
                    if wf_out.ndim == 3: wf_out = wf_out.squeeze(0)
                    save_sample_rate = sampling_rate 
                else:
                    print("--- Decoding Stage 1 Preview Audio ---")
                    if num_prompts > 1:
                        sr = int(getattr(audio_vae, "output_sample_rate", audio_vae.first_stage_model.output_sample_rate))
                        a_latents_per_shot = a_samps_prev.shape[2] // num_prompts
                        decoded_a_shots = []
                        for s in range(num_prompts):
                            s_lat = a_samps_prev[:, :, s * a_latents_per_shot : (s + 1) * a_latents_per_shot]
                            s_wf = audio_vae.decode(s_lat).to(device).movedim(-1, 1)
                            decoded_a_shots.append(s_wf)
                            
                        preview_audio_wf = torch.cat(decoded_a_shots, dim=-1)
                        
                        fade_frames = current_fps / 2.0
                        fade_len = int((fade_frames / current_fps) * sr)
                        shot_len_samples = int((length_in_seconds / num_prompts) * sr)
                        fade_len = min(fade_len, max(1, (shot_len_samples // 2) - 1))
                        
                        current_idx = 0
                        for s in range(num_prompts - 1):
                            current_idx += decoded_a_shots[s].shape[-1]
                            b_idx = current_idx
                            if b_idx >= fade_len and b_idx + fade_len <= preview_audio_wf.shape[-1]:
                                left_audio = preview_audio_wf[..., b_idx - fade_len : b_idx].clone()
                                right_audio = preview_audio_wf[..., b_idx : b_idx + fade_len].clone()
                                left_extend = torch.cat([left_audio, left_audio.flip(-1)], dim=-1)
                                right_extend = torch.cat([right_audio.flip(-1), right_audio], dim=-1)
                                fade_in = torch.linspace(0.0, 1.0, fade_len * 2, device=device, dtype=preview_audio_wf.dtype)
                                blended = left_extend * (1.0 - fade_in) + right_extend * fade_in
                                preview_audio_wf[..., b_idx - fade_len : b_idx + fade_len] = blended
                    else:
                        preview_audio_wf = audio_vae.decode(a_samps_prev).to(device).movedim(-1, 1)
                        
                    wf_out = preview_audio_wf[0].cpu()
                    if wf_out.ndim == 3: wf_out = wf_out.squeeze(0)
                        
                time_to_drop = out_ref_frame_count / current_fps
                samples_to_drop = int(time_to_drop * save_sample_rate)
                
                if samples_to_drop >= wf_out.shape[-1]:
                    wf_out = torch.zeros((wf_out.shape[0], 1), device=wf_out.device, dtype=wf_out.dtype)
                elif samples_to_drop > 0:
                    wf_out = wf_out[..., samples_to_drop:]
                    fade_samples = min(int(0.03 * save_sample_rate), wf_out.shape[-1])
                    if fade_samples > 0:
                        fade_tensor = torch.linspace(0.0, 1.0, fade_samples, device=wf_out.device, dtype=wf_out.dtype)
                        wf_out[..., :fade_samples] *= fade_tensor
                
                torchaudio.save(audio_path, wf_out, save_sample_rate)
            except Exception as e:
                print(f"Warning: Failed to decode preview audio: {e}")

            # --- 2. SETUP FFMPEG STREAMING PIPE ---
            v_batch, v_channels, v_latents, v_h, v_w = v_samps_prev.shape
            out_v_height_p = v_h * 32
            out_v_width_p = v_w * 32
            
            if not primary_sampling:
                preview_video_trim = 0
            else:
                preview_video_trim = out_ref_frame_count
                if video_mode == "reference-to-video":
                    n_ref_images = max(1, ref_frame_count // duplicate_frames) if duplicate_frames > 0 else 1
                    preview_video_trim += (2 * n_ref_images)

            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{out_v_width_p}x{out_v_height_p}",
                "-pix_fmt", "rgb24",
                "-r", str(current_fps),
                "-i", "-", 
            ]
            if os.path.exists(audio_path): cmd.extend(["-i", audio_path])
            cmd.extend([
                "-c:v", "libx264", "-preset", "superfast", "-crf", "28", 
                "-pix_fmt", "yuv420p", "-r", str(current_fps) 
            ])
            if os.path.exists(audio_path): cmd.extend(["-c:a", "aac", "-b:a", "128k"])
            cmd.append(preview_path)

            print("--- Encoding Low-Bitrate MP4 Preview (RAM-Safe VAE Streaming) ---")
            process = None
            try:
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"LTXV Custom Error: FFmpeg failed to start: {e}")

            # --- 3. VAE DECODE DIRECTLY TO FFMPEG ---
            # By passing frames straight to FFmpeg and deleting them, we bypass the 100GB RAM spike entirely!
            frames_skipped = 0
            
            def write_frames_to_pipe(decoded_tensor):
                nonlocal frames_skipped, process
                if process is None: return
                if decoded_tensor.ndim == 5: decoded_tensor = decoded_tensor.squeeze(0)
                
                chunk_frames = decoded_tensor.shape[0]
                start_idx = 0
                
                if frames_skipped < preview_video_trim:
                    remaining_skip = preview_video_trim - frames_skipped
                    if remaining_skip >= chunk_frames:
                        frames_skipped += chunk_frames
                        return
                    else:
                        start_idx = remaining_skip
                        frames_skipped += remaining_skip
                
                if start_idx < chunk_frames:
                    valid = decoded_tensor[start_idx:]
                    # Send bytes in tiny chunks to keep numpy conversion ultra-lightweight
                    write_chunk_size = 32
                    for i in range(0, valid.shape[0], write_chunk_size):
                        sub_chunk = valid[i : i + write_chunk_size]
                        chunk_np = (sub_chunk.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                        try:
                            process.stdin.write(chunk_np.tobytes())
                        except BrokenPipeError:
                            break

            if process is not None:
                if num_prompts == 1:
                    temp_tile_len = 16 
                    temp_overlap = 4  
                    chunk_start = 0
                    valid_end_frame = 0
                    prev_raw_overlap = None
                    
                    while chunk_start < v_latents:
                        if chunk_start == 0:
                            chunk_end = min(chunk_start + temp_tile_len, v_latents)
                            overlap_start = chunk_start
                        else:
                            overlap_start = max(1, chunk_start - temp_overlap - 1)
                            chunk_end = min(chunk_start + temp_tile_len - (chunk_start - overlap_start), v_latents)
                            
                        if chunk_end <= chunk_start: chunk_end = chunk_start + 1
                            
                        v_tile = v_samps_prev[:, :, overlap_start:chunk_end]
                        v_tile_decoded = safe_vae_decode(video_vae, v_tile).cpu()
                        if v_tile_decoded.ndim == 5:
                            v_tile_decoded = v_tile_decoded.squeeze(0)
                            
                        current_valid_frames = v_tile_decoded.shape[0]
                        
                        if chunk_start == 0:
                            raw_chunk = v_tile_decoded
                            valid_end_frame = current_valid_frames
                            use_len = 0
                        else:
                            global_start_frame = overlap_start * 8
                            overlap_frames = valid_end_frame - global_start_frame
                            
                            if overlap_frames > 0 and prev_raw_overlap is not None:
                                use_len = min(overlap_frames, prev_raw_overlap.shape[0], v_tile_decoded.shape[0])
                                feather = torch.linspace(0.0, 1.0, use_len).view(-1, 1, 1, 1)
                                prev_overlap = prev_raw_overlap[-use_len:]
                                new_overlap = v_tile_decoded[:use_len]
                                blended = prev_overlap * (1.0 - feather) + new_overlap * feather
                                raw_chunk = torch.cat([blended, v_tile_decoded[use_len:]], dim=0)
                            else:
                                raw_chunk = v_tile_decoded
                                use_len = 0
                                
                            valid_end_frame += (current_valid_frames - use_len)

                        next_overlap_start = max(1, chunk_end - temp_overlap - 1)
                        next_global_start = next_overlap_start * 8
                        overlap_frames_for_next = valid_end_frame - next_global_start
                        
                        if chunk_end < v_latents and overlap_frames_for_next > 0:
                            prev_raw_overlap = raw_chunk[-overlap_frames_for_next:].clone()
                            frames_to_pipe = raw_chunk[:-overlap_frames_for_next]
                        else:
                            prev_raw_overlap = None
                            frames_to_pipe = raw_chunk
                            
                        write_frames_to_pipe(frames_to_pipe)
                        
                        del v_tile_decoded
                        del raw_chunk
                        del frames_to_pipe
                        comfy.model_management.soft_empty_cache()
                        chunk_start = chunk_end
                else:
                    out_h = out_v_height_p
                    out_w = out_v_width_p
                    video_trim = out_ref_frame_count if video_mode != "text-to-video" else 0
                    if video_mode == "reference-to-video":
                        n_ref_images = max(1, ref_frame_count // duplicate_frames) if duplicate_frames > 0 else 1
                        video_trim += (2 * n_ref_images)

                    frames_per_split_out = (length_in_seconds * current_fps) / total_splits
                    start_lat = 0
                    
                    for s in range(total_splits):
                        target_shot_frames = int(round((s + 1) * frames_per_split_out)) - int(round(s * frames_per_split_out))
                        
                        if s == 0:
                            shot_target_frames = target_shot_frames + (out_ref_frame_count if not bypass_img_ref else 0)
                        else:
                            shot_target_frames = target_shot_frames
                        
                        shot_lats = ((shot_target_frames - 1) // 8) + 1
                        if ((shot_lats - 1) * 8 + 1) < shot_target_frames:
                            shot_lats += 1
                            
                        end_lat = min(start_lat + shot_lats, v_samps_prev.shape[2])
                        if s == total_splits - 1:
                            end_lat = v_samps_prev.shape[2]
                            
                        split_latent = v_samps_prev[:, :, start_lat : end_lat]
                        
                        if split_latent.shape[2] > 0:
                            decoded_shot = safe_vae_decode(video_vae, split_latent).cpu()
                            decoded_shot = decoded_shot.view(-1, out_h, out_w, 3)
                        else:
                            break
                        
                        # Only shot 0 carries un-trimmed warm-up -- shots 1+ were already sliced by the razor blade.
                        if s == 0 and video_trim > 0:
                            decoded_shot = decoded_shot[video_trim:]
                                
                        if decoded_shot.shape[0] > target_shot_frames:
                            decoded_shot = decoded_shot[:target_shot_frames]
                        elif decoded_shot.shape[0] < target_shot_frames:
                            pad_len = target_shot_frames - decoded_shot.shape[0]
                            decoded_shot = torch.cat([decoded_shot, decoded_shot[-1:].repeat(pad_len, 1, 1, 1)], dim=0)
                                
                        write_frames_to_pipe(decoded_shot)
                        del decoded_shot 
                        start_lat = end_lat

                try:
                    process.stdin.close()
                    process.wait()
                except Exception as e:
                    pass

            PromptServer.instance.send_sync("trixope_ltxv_preview", {
                "node": uid, 
                "filename": preview_filename, 
                "type": "temp"
            })
            print(f"--- Stage 1 Preview Sent to UI ---")
            comfy.model_management.soft_empty_cache()

        # ==========================================
        # UNIVERSAL SLIDING WINDOW UPSCALER ENGINE
        # ==========================================
        def process_sliding_window_upscale(pass_name, is_temporal, v_samps, a_samps,
                                           upscaler_model, guider, sampler, sigmas,
                                           noise_obj, eta, bongmath, final_pixels, strengths,
                                           time_scale_factor, width_scale_factor, height_scale_factor, video_vae, current_fps, disable_pbar, seed_number, wrap_callback,
                                           temporal_positive=None, temporal_negative=None, video_mode="text-to-video"):

            v_batch, v_channels, v_frames, v_height, v_width = v_samps.shape

            # Dynamically scale temporal tile size to fit VRAM after 2x spatial expansion
            if not is_temporal:
                # Spatial pass quadruples spatial tokens; cap temporal tile length to 16 latents
                temp_tile_len = 16
                temp_overlap = 4
            else:
                temp_tile_len = 32
                temp_overlap = 8
            
            sim_start = 0
            num_chunks = 0
            while sim_start < v_frames:
                num_chunks += 1
                
                shot_len_latents = v_frames / num_prompts 
                current_shot = min(int(sim_start // shot_len_latents), num_prompts - 1)
                shot_start = int(current_shot * shot_len_latents)
                shot_end = int((current_shot + 1) * shot_len_latents) if current_shot < num_prompts - 1 else v_frames
                
                if sim_start == shot_start:
                    sim_end = min(sim_start + temp_tile_len, shot_end)
                    sim_overlap = sim_start
                else:
                    sim_overlap = max(shot_start + 1, sim_start - temp_overlap - 1)
                    sim_end = min(sim_start + temp_tile_len - (sim_start - sim_overlap), shot_end)
                    
                if sim_end <= sim_start:
                    sim_end = sim_start + 1
                    
                sim_start = sim_end
            
            global_v_samps_up = None
            chunk_start = 0
            chunk_idx = 1
            sp_encoded_t_local = None
            
            a_mask_locked = torch.zeros_like(a_samps)
            
            device_up = comfy.model_management.get_torch_device()
            target_model = upscaler_model.model if hasattr(upscaler_model, "model") else upscaler_model
            target_model.to(device_up)

            while chunk_start < v_frames:
                comfy.model_management.soft_empty_cache()

                # SHOT-AWARE BOUNDARY MATH: Treats the stacked array as isolated slots!
                shot_len_latents = v_frames / num_prompts 
                current_shot = min(int(chunk_start // shot_len_latents), num_prompts - 1)
                shot_start = int(current_shot * shot_len_latents)
                shot_end = int((current_shot + 1) * shot_len_latents) if current_shot < num_prompts - 1 else v_frames
                
                if chunk_start == shot_start:
                    chunk_end = min(chunk_start + temp_tile_len, shot_end)
                    overlap_start = chunk_start
                else:
                    overlap_start = max(shot_start + 1, chunk_start - temp_overlap - 1)
                    chunk_end = min(chunk_start + temp_tile_len - (chunk_start - overlap_start), shot_end)
                    
                # FAILSAFE: Guarantee forward mathematical progress
                if chunk_end <= chunk_start:
                    chunk_end = chunk_start + 1
                    
                v_tile = v_samps[:, :, overlap_start:chunk_end]
                
                center_frame = overlap_start + (chunk_end - overlap_start) / 2.0
                frames_per_shot = v_frames / num_prompts
                shot_idx = int(center_frame / frames_per_shot)
                shot_idx = min(max(shot_idx, 0), num_prompts - 1)

                if is_temporal and temporal_positive is not None and temporal_negative is not None:
                    guider.set_conds([temporal_positive[shot_idx]], temporal_negative)
                else:
                    guider.set_conds([final_positive[shot_idx]], final_negative)

                pixel_start = overlap_start * time_scale_factor
                pixel_end = 1 + (chunk_end - 1) * time_scale_factor
                
                get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
                if get_audio_latents_func is None:
                    raise AttributeError("Audio VAE is missing 'num_of_latents_from_frames' method.")

                a_start = get_audio_latents_func(pixel_start, int(current_fps))
                a_end = get_audio_latents_func(pixel_end, int(current_fps))

                a_tile = a_samps[:, :, a_start:a_end]
                
                device_up = comfy.model_management.get_torch_device()
                target_model = upscaler_model.model if hasattr(upscaler_model, "model") else upscaler_model
                model_dtype = next(target_model.parameters()).dtype
                input_dtype = v_tile.dtype

                target_model = upscaler_model.model if hasattr(upscaler_model, "model") else upscaler_model
                v_tile_unnorm = video_vae.first_stage_model.per_channel_statistics.un_normalize(v_tile.to(dtype=model_dtype, device=device_up))
                v_tile_raw = target_model(v_tile_unnorm)
                v_tile_up = video_vae.first_stage_model.per_channel_statistics.normalize(v_tile_raw).to(dtype=input_dtype, device=device)

                # Flush VRAM allocated by the upscaler before entering the diffusion sampler
                comfy.model_management.soft_empty_cache()

                print(f"\n--- {pass_name} Chunk {chunk_idx}/{num_chunks} ...")
                
                v_mask_tile = torch.ones((v_batch, v_tile_up.shape[2], v_tile_up.shape[3], v_tile_up.shape[4]), device=device, dtype=torch.float32)

                if chunk_start == 0:
                    v_mask_tile[:, :1, :, :] = 0.0

                if global_v_samps_up is None:
                    total_out_frames = (v_frames * 2) - 1 if is_temporal else v_frames
                    global_v_samps_up = torch.zeros(
                        (v_batch, v_tile_up.shape[1], total_out_frames, v_tile_up.shape[3], v_tile_up.shape[4]),
                        device=device,
                        dtype=input_dtype
                    )

                if chunk_start > 0:
                    overlap_in_frames = chunk_start - overlap_start
                    if is_temporal:
                        overlap_out_frames = overlap_in_frames * 2 - 1
                        global_start = overlap_start * 2
                    else:
                        overlap_out_frames = overlap_in_frames
                        global_start = overlap_start

                    feather = torch.linspace(0.0, 1.0, overlap_out_frames, device=device, dtype=torch.float32)
                    v_mask_tile[:, :overlap_out_frames, :, :] = feather.view(1, -1, 1, 1)

                    if sigmas is None:
                        prev_overlap = global_v_samps_up[:, :, global_start : global_start + overlap_out_frames]
                        curr_overlap = v_tile_up[:, :, :overlap_out_frames]
                        alpha = feather.view(1, 1, -1, 1, 1)
                        v_tile_up[:, :, :overlap_out_frames] = prev_overlap * (1.0 - alpha) + curr_overlap * alpha
                    else:
                        v_tile_up[:, :, :overlap_out_frames] = global_v_samps_up[:, :, global_start : global_start + overlap_out_frames]

                device_up = comfy.model_management.get_torch_device()

                # 1. Slice audio mask tensor from locked canvas
                am_tile = a_mask_locked[:, :, a_start:a_end]

                # 2. Synchronize all video and audio tensors to GPU device
                v_tile_up = v_tile_up.to(device=device_up, dtype=input_dtype)
                a_tile = a_tile.to(device=device_up)
                v_mask_tile = v_mask_tile.to(device=device_up, dtype=torch.float32)
                am_tile = am_tile.to(device=device_up, dtype=torch.float32)

                # 3. Assemble nested latent structure
                current_latent_tile = {
                    "samples": comfy.nested_tensor.NestedTensor((v_tile_up, a_tile)),
                    "noise_mask": comfy.nested_tensor.NestedTensor((v_mask_tile.unsqueeze(1), am_tile)),
                    "sample_rate": sampling_rate,
                    "type": "audio"
                }
                
                latent_image_tile = current_latent_tile["samples"]

                if sigmas is not None:
                    x0_output = {}
                    base_cb = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
                    callback = wrap_callback(base_cb, pass_name)

                    chunk_seed_number = seed_number + chunk_idx
                    chunk_noise_obj = Noise_RandomNoise(chunk_seed_number)
                    
                    sampled_chunk = guider.sample(chunk_noise_obj.generate_noise(current_latent_tile), latent_image_tile, sampler, sigmas, denoise_mask=current_latent_tile["noise_mask"], callback=callback, disable_pbar=disable_pbar, seed=chunk_seed_number)
                    sampled_v_tile = sampled_chunk.unbind()[0].to(device)
                else:
                    sampled_v_tile = v_tile_up

                # BATCH DIMENSION FAILSAFE
                if sampled_v_tile.ndim == 4:
                    sampled_v_tile = sampled_v_tile.unsqueeze(0)

                if chunk_start == 0:
                    global_v_samps_up[:, :, :sampled_v_tile.shape[2]] = sampled_v_tile
                else:
                    overlap_len = min(overlap_out_frames, sampled_v_tile.shape[2])
                    prev_chunk_end = global_v_samps_up[:, :, global_start : global_start + overlap_len]
                    new_chunk_start = sampled_v_tile[:, :, :overlap_len]
                    
                    blend_curve = torch.linspace(0.0, 1.0, overlap_len, device=device, dtype=torch.float32).view(1, -1, 1, 1)
                    blended_overlap = prev_chunk_end * (1.0 - blend_curve) + new_chunk_start * blend_curve
                    
                    global_v_samps_up[:, :, global_start : global_start + overlap_len] = blended_overlap
                    
                    if sampled_v_tile.shape[2] > overlap_len:
                        global_v_samps_up[:, :, global_start + overlap_len : global_start + sampled_v_tile.shape[2]] = sampled_v_tile[:, :, overlap_len:]

                # Clean up chunk garbage immediately
                import gc
                del sampled_v_tile, v_tile_up
                gc.collect()
                comfy.model_management.soft_empty_cache()

                chunk_start = chunk_end
                chunk_idx += 1

            target_model.cpu()
            comfy.model_management.soft_empty_cache()
            return global_v_samps_up

        # ==========================================
        # 6.5 UPSCALING PASS (SPATIAL)
        # ==========================================
        if spatial_upscale and not pure_image_passthrough:
            if spatial_upscaler is None:
                raise ValueError("Spatial upscaler model is required if spatial_upscale is True.")
            
            safe_sampler_name = spatial_sampler

            upsample_sampler = build_custom_sampler(safe_sampler_name, eta, bongmath)
            sigmas_list = re.findall(r"[-+]?(?:\d*\.*\d+)", spatial_sigmas)
            spatial_sigmas_tensor = torch.FloatTensor([float(i) for i in sigmas_list])

            for stage in range(1, spatial_passes + 1):
                v_samps, a_samps = sampled_tensor.unbind()
                v_samps = v_samps.to(device)
                a_samps = a_samps.to(device)

                global_v_samps_up = process_sliding_window_upscale(
                    pass_name=f"Spatial Upscale Pass {stage}/{spatial_passes}",
                    is_temporal=False, v_samps=v_samps, a_samps=a_samps,
                    upscaler_model=spatial_upscaler, guider=upsample_guider, sampler=upsample_sampler, sigmas=spatial_sigmas_tensor,
                    noise_obj=noise_obj, eta=eta, bongmath=bongmath, final_pixels=final_pixels, strengths=strengths,
                    time_scale_factor=time_scale_factor, width_scale_factor=width_scale_factor, height_scale_factor=height_scale_factor, video_vae=video_vae, current_fps=current_fps, disable_pbar=disable_pbar, seed_number=seed_number, wrap_callback=wrap_callback
                )

                sampled_tensor = comfy.nested_tensor.NestedTensor((global_v_samps_up, a_samps)).to(device)
                debug_save_latent(sampled_tensor, f"stage2_spatial_pass_{stage}")

        # ==========================================
        # 7. THE POST-CLEANSE HARD OVERWRITE
        # ==========================================
        final_latent = None
        video_out_latent = None
        audio_out_latent = None

        if not pure_image_passthrough:
            unbound_samples = sampled_tensor.unbind()
            final_video_samples = unbound_samples[0].to(device)
            final_audio_samples = unbound_samples[1].clone().to(device)

            if has_audio_ref or has_audio_input:
                max_latents = final_audio_samples.shape[2]
                master_max = master_latents.shape[2]
                
                if has_audio_ref:
                    lock_a = min(setup_total_latents, max_latents, master_max)
                    if lock_a > 0:
                        final_audio_samples[:, :, :lock_a, :] = master_latents[:, :, :lock_a, :]

                if has_audio_input:
                    start_c = min(setup_total_latents, max_latents)
                    target_len = max_latents - start_c
                    available_master = master_max - start_c
                    copy_len = min(target_len, available_master)
                    
                    if copy_len > 0:
                        final_audio_samples[:, :, start_c : start_c + copy_len, :] = \
                            master_latents[:, :, start_c : start_c + copy_len, :]

            v_batch, _, v_frames, v_height, v_width = final_video_samples.shape
            
            final_v_mask = torch.ones((v_batch, v_frames, v_height, v_width), dtype=torch.float32, device=device)
            # ENFORCE 3D FINAL AUDIO MASK [Batch, Frames, Freq]
            final_a_mask = torch.ones((v_batch, final_audio_samples.shape[2], final_audio_samples.shape[3]), dtype=torch.float32, device=device)
            
            final_latent = {
                "samples": comfy.nested_tensor.NestedTensor((final_video_samples, final_audio_samples)),
                "noise_mask": comfy.nested_tensor.NestedTensor((final_v_mask.unsqueeze(1), final_a_mask)),
                "sample_rate": sampling_rate,
                "type": "audio"
            }
            
            video_out_latent = {
                "samples": final_video_samples,
                "noise_mask": final_v_mask 
            }
            
            audio_out_latent = {
                "samples": final_audio_samples,
                "noise_mask": final_a_mask,
                "sample_rate": sampling_rate,
                "type": "audio"
            }
            debug_save_latent(final_latent, "stage2.5_post_cleanse")

        # ==========================================
        # 7.5 ISOLATED TEMPORAL UPSCALE PASS (NATIVE NODES + SLIDING WINDOW)
        # ==========================================
        if temporal_upscale and not pure_image_passthrough:
            if temporal_upscaler is None:
                raise ValueError("Temporal upscaler model is required if temporal_upscale is True.")
                
            print("\n--- Running Temporal Upscale Pass ---")
            import sys
            
            # Universal helper function to safely instantiate and execute ComfyUI nodes
            def run_node(class_name, **kwargs):
                import sys
                nodes_module = sys.modules.get("nodes")
                if not nodes_module or class_name not in getattr(nodes_module, "NODE_CLASS_MAPPINGS", {}):
                    raise ValueError(f"Required node '{class_name}' not found in ComfyUI.")
                cls = nodes_module.NODE_CLASS_MAPPINGS[class_name]
                func_name = getattr(cls, "FUNCTION", "execute")
                return getattr(cls(), func_name)(**kwargs)

            # =========================================================
            # 0. DYNAMIC PREP: SINGLE-SHOT VS MULTI-SHOT ROUTING
            # =========================================================
            temp_video_mode = video_mode
            is_source_audio = (audio is not None and audio_select == "source")
            
            if num_prompts == 1:
                print("-> SINGLE-SHOT DETECTED: Routing to Audio-Locked Temporal Upscale...")
                # The "Lob It Off" Optimization is perfectly safe for single shots!
                if video_mode in ["reference", "reference-to-video"]:
                    print("   -> LOBBING OFF static setup latents to optimize Temporal Upscale...")
                    v_setup_base = ((out_ref_frame_count - 1) // 8) if out_ref_frame_count > 0 else 0
                    get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
                    a_setup_base = get_audio_latents_func(out_ref_frame_count, int(current_fps)) if v_setup_base > 0 else 0
                    
                    if final_video_samples.shape[2] > v_setup_base:
                        final_video_samples = final_video_samples[:, :, v_setup_base:]
                    if final_audio_samples.shape[2] > a_setup_base:
                        final_audio_samples = final_audio_samples[:, :, a_setup_base:]
                        
                    temp_video_mode = "text-to-video"
                    out_ref_frame_count = 0
            else:
                print("-> MULTI-SHOT DETECTED: Routing to Pure Visual Interpolation Passthrough...")
                # Base generation perfected the sync. We bypass Lob-It-Off to preserve the hard cuts!
                temp_video_mode = "text-to-video"

            # ---------------------------------------------------------
            # 1. GLOBAL PREP: AUDIO ALIGNMENT & STRETCH (DUAL-PATH)
            # ---------------------------------------------------------
            print(f"-> Preparing Audio Alignment (Isolated Mode: {temp_video_mode})...")
            
            # Only apply the Phase Vocoder stretch if it's a single shot with source audio!
            use_stretch_path = (num_prompts == 1 and is_source_audio)
            
            if use_stretch_path:
                print("-> SOURCE AUDIO DETECTED: Executing Phase Vocoder Stretch & Base-FPS Conditioning...")
                import torchaudio.functional as TA_F
                input_audio = audio
                source_wf = input_audio["waveform"].to(device, dtype=torch.float32)
                source_sr = input_audio["sample_rate"]
                
                is_batched = source_wf.ndim == 3
                wf_sq = source_wf.squeeze(0) if is_batched else source_wf
                
                rate = 0.5
                fft_size = 2048
                hop_size = fft_size // 4
                win_length = fft_size
                window = torch.hann_window(win_length, device=device)
                
                complex_spectogram = torch.stft(wf_sq, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True)
                phase_advance = torch.linspace(0, math.pi * hop_size, complex_spectogram.shape[1], device=device)[..., None]
                stretched_spectogram = TA_F.phase_vocoder(complex_spectogram, rate, phase_advance)
                stretched_wf_sq = torch.istft(stretched_spectogram, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window)
                
                stretched_wf = stretched_wf_sq.unsqueeze(0) if is_batched else stretched_wf_sq
                
                target_length = int(source_wf.shape[-1] / rate)
                if stretched_wf.shape[-1] > target_length:
                    stretched_wf = stretched_wf[..., :target_length]
                elif stretched_wf.shape[-1] < target_length:
                    pad_amount = target_length - stretched_wf.shape[-1]
                    stretched_wf = torch.nn.functional.pad(stretched_wf, (0, pad_amount))
                    
                if video_mode in ["reference", "reference-to-video"]:
                    # In single-shot, we lobbed off the setup frames. We must pad the audio so it starts perfectly!
                    v_setup_base = ((out_ref_frame_count - 1) // 8) if out_ref_frame_count > 0 else 0
                    remaining_silence_frames = out_ref_frame_count - (v_setup_base * 8)
                    if remaining_silence_frames > 0:
                        upscaled_silence_frames = remaining_silence_frames * 2
                        silence_samples = int((upscaled_silence_frames / (current_fps * 2)) * source_sr)
                        silence_pad = torch.zeros((stretched_wf.shape[0], stretched_wf.shape[1], silence_samples), device=device, dtype=stretched_wf.dtype)
                        stretched_wf = torch.cat([silence_pad, stretched_wf], dim=-1)
                    
                shifted_audio = {"waveform": stretched_wf.cpu(), "sample_rate": source_sr}
                
                import sys
                nodes_module = sys.modules.get("nodes")
                cls = nodes_module.NODE_CLASS_MAPPINGS["LTXVAudioVAEEncode"]
                audio_latent_out = getattr(cls(), "execute")(audio=shifted_audio, audio_vae=audio_vae)[0]
                audio_for_upscale = audio_latent_out["samples"].to(device)
                cond_fps = float(current_fps)
            else:
                print("-> INTERNAL / MULTI-SHOT DETECTED: Bypassing Stretch & Executing Target-FPS Passthrough...")
                audio_for_upscale = final_audio_samples
                cond_fps = float(current_fps * 2)

            import node_helpers
            cond_out_pos = node_helpers.conditioning_set_values(final_positive, {"frame_rate": cond_fps})
            cond_out_neg = node_helpers.conditioning_set_values(final_negative, {"frame_rate": cond_fps})
            
            crop_pos = node_helpers.conditioning_set_values(cond_out_pos, {"keyframe_idxs": None, "guide_attention_entries": None})
            crop_neg = node_helpers.conditioning_set_values(cond_out_neg, {"keyframe_idxs": None, "guide_attention_entries": None})

            # ---------------------------------------------------------
            # 2. GLOBAL INJECTION SETUP (Inlined LTXVImgToVideoInplace)
            # ---------------------------------------------------------
            print("-> Preparing LTXVImgToVideoInplace Global Injections (Strength: 1.0, Bypass: False)...")
            v_batch, v_channels, v_frames_base, v_height, v_width = final_video_samples.shape
            total_out_frames = (v_frames_base * 2) - 1
            
            global_injection_mask = torch.ones((v_batch, 1, total_out_frames, 1, 1), device=device, dtype=torch.float32)
            global_injection_samples = torch.zeros((v_batch, v_channels, total_out_frames, v_height, v_width), device=device, dtype=final_video_samples.dtype)

            shot_duration = length_in_seconds / num_prompts

            def get_shot_start_latent_idx(s_idx, shot_dur):
                if s_idx == 0:
                    return 0
                sec_val = s_idx * shot_dur
                v_lat, _ = get_latent_counts(sec_val)
                if num_prompts == 1 and video_mode in ["reference", "reference-to-video"]:
                    # We only subtract the v_setup_base if we actually lobbed it off (Single Shot!)
                    return max(0, (v_lat - 1) - v_setup_base)
                return max(0, v_lat - 1)

            for s in range(num_prompts):
                v_idx = get_shot_start_latent_idx(s, shot_duration)
                v_lat_up = v_idx * 2
                frames_to_encode = None
                
                if temp_video_mode in ["text", "text-to-video"]:
                    base_latent = final_video_samples[:, :, v_idx:v_idx+1]
                    if base_latent.shape[2] > 0:
                        frames_to_encode = video_vae.decode(base_latent)
                elif temp_video_mode in ["image", "image-to-video"]:
                    if first_frame is not None:
                        img_idx = min(s, first_frame.shape[0] - 1)
                        frames_to_encode = first_frame[img_idx:img_idx+1]
                    else:
                        base_latent = final_video_samples[:, :, v_idx:v_idx+1]
                        if base_latent.shape[2] > 0:
                            frames_to_encode = video_vae.decode(base_latent)

                if frames_to_encode is not None:
                    if frames_to_encode.ndim == 5:
                        b_, f_, h_, w_, c_ = frames_to_encode.shape
                        frames_to_encode = frames_to_encode.view(b_ * f_, h_, w_, c_)

                    frames_to_encode = frames_to_encode[:1]

                    t_width = v_width * 32
                    t_height = v_height * 32
                    if frames_to_encode.shape[1] != t_height or frames_to_encode.shape[2] != t_width:
                        frames_to_encode = comfy.utils.common_upscale(frames_to_encode.movedim(-1, 1), t_width, t_height, "bilinear", "center").movedim(1, -1)
                    
                    encoded_t = video_vae.encode(frames_to_encode[:, :, :, :3]).to(device)
                    inject_len = encoded_t.shape[2]
                    
                    if v_lat_up + inject_len <= total_out_frames:
                        global_injection_samples[:, :, v_lat_up : v_lat_up + inject_len] = encoded_t
                        global_injection_mask[:, :, v_lat_up : v_lat_up + inject_len] = 0.0
                    else:
                        rem = total_out_frames - v_lat_up
                        if rem > 0:
                            global_injection_samples[:, :, v_lat_up : v_lat_up + rem] = encoded_t[:, :, :rem]
                            global_injection_mask[:, :, v_lat_up : v_lat_up + rem] = 0.0

            # ---------------------------------------------------------
            # 3. SLIDING WINDOW & CONTEXT LOOKBACK SETUP
            # ---------------------------------------------------------
            temporal_context_seconds = 5.0
            temp_overlap = max(1, int((temporal_context_seconds * current_fps) / 8))
            
            max_vram_latents = 40
            new_latents_per_chunk = max(2, max_vram_latents - temp_overlap)
            
            global_v_temporal = torch.zeros((v_batch, v_channels, total_out_frames, v_height, v_width), device=device, dtype=final_video_samples.dtype)

            sim_start = 0
            num_chunks = 0
            while sim_start < v_frames_base:
                num_chunks += 1
                if sim_start == 0:
                    sim_end = min(sim_start + max_vram_latents, v_frames_base)
                else:
                    sim_end = min(sim_start + new_latents_per_chunk, v_frames_base)
                    
                if sim_end <= sim_start: sim_end = sim_start + 1
                if sim_end >= v_frames_base: break
                
                sim_start = sim_end - 3
                if sim_start < 0: sim_start = 0

            chunk_start = 0
            chunk_idx = 1
            valid_global_end = 0  
            v_time_scale = video_vae.downscale_index_formula[0]
            get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))

            temporal_model = model3_to_use

            print("-> Anchoring Global Noise Tensor to freeze background textures...")
            generator = torch.manual_seed(seed_number)
            global_noise_tensor = torch.randn((v_batch, v_channels, total_out_frames, v_height, v_width), device=device, dtype=torch.float32, generator=generator)
            
            class SlicedNoise:
                def __init__(self, global_noise, start_idx, seed):
                    self.global_noise = global_noise
                    self.start_idx = start_idx
                    self.seed = seed
                    
                def generate_noise(self, input_latent):
                    latent_image = input_latent["samples"]
                    batch_inds = input_latent.get("batch_index", None)
                    base_noise = comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)
                    v_noise, a_noise = base_noise.unbind()
                    length = v_noise.shape[2]
                    v_noise_sliced = self.global_noise[:, :, self.start_idx : self.start_idx + length].clone().to(v_noise.device)
                    if v_noise_sliced.shape[2] < length:
                        pad_len = length - v_noise_sliced.shape[2]
                        v_noise_sliced = torch.nn.functional.pad(v_noise_sliced, (0, 0, 0, 0, 0, pad_len))
                    return comfy.nested_tensor.NestedTensor((v_noise_sliced, a_noise))

            # ---------------------------------------------------------
            # 4. CHUNKING LOOP
            # ---------------------------------------------------------
            while chunk_start < v_frames_base:
                comfy.model_management.soft_empty_cache()

                if chunk_start == 0:
                    overlap_start = 0
                    chunk_end = min(chunk_start + max_vram_latents, v_frames_base)
                else:
                    overlap_start = max(0, chunk_start - temp_overlap)
                    chunk_end = min(chunk_start + new_latents_per_chunk, v_frames_base)
                    
                if chunk_end <= chunk_start: chunk_end = chunk_start + 1

                v_tile = final_video_samples[:, :, overlap_start:chunk_end]

                tile_in_frames = chunk_end - overlap_start
                tile_out_frames = (tile_in_frames * 2) - 1
                up_overlap_start = overlap_start * 2
                
                pixel_start = up_overlap_start * v_time_scale
                pixel_out_frames = 1 + (tile_out_frames - 1) * v_time_scale
                pixel_end = pixel_start + pixel_out_frames - 1
                
                def get_abs_a_lat(f_idx, fps):
                    return get_audio_latents_func(int(f_idx + 1), int(fps)) if f_idx > 0 else 0
                
                # Dynamic Audio Slicing Path
                if use_stretch_path:
                    # Single-Shot Stretched Track
                    a_start = get_abs_a_lat(pixel_start, int(current_fps))
                    req_a_len = get_abs_a_lat(pixel_out_frames, int(current_fps))
                else:
                    # Multi-Shot / Internal Native Track
                    target_fps = int(current_fps * 2)
                    a_start = get_abs_a_lat(pixel_start, target_fps)
                    req_a_len = get_abs_a_lat(pixel_out_frames, target_fps)
                
                req_a_len = max(1, req_a_len)
                a_end = min(a_start + req_a_len, audio_for_upscale.shape[2])
                a_tile = audio_for_upscale[:, :, a_start:a_end]

                if a_tile.shape[2] > req_a_len:
                    a_tile = a_tile[:, :, :req_a_len]
                elif a_tile.shape[2] < req_a_len:
                    pad_len = req_a_len - a_tile.shape[2]
                    a_tile = torch.nn.functional.pad(a_tile, (0,0,0,pad_len), value=0)

                up_device = comfy.model_management.get_torch_device()
                target_temp_model = temporal_upscaler.model if hasattr(temporal_upscaler, "model") else temporal_upscaler
                model_dtype = next(target_temp_model.parameters()).dtype
                input_dtype = v_tile.dtype

                memory_required = comfy.model_management.module_size(target_temp_model)
                memory_required += math.prod(v_tile.shape) * 3000.0
                comfy.model_management.free_memory(memory_required, up_device)

                try:
                    target_temp_model.to(up_device)
                    v_tile_ready = v_tile.to(dtype=model_dtype, device=up_device)
                    v_tile_unnorm = video_vae.first_stage_model.per_channel_statistics.un_normalize(v_tile_ready)
                    v_upsampled_raw = target_temp_model(v_tile_unnorm)
                finally:
                    target_temp_model.cpu()
                    
                v_upsampled = video_vae.first_stage_model.per_channel_statistics.normalize(v_upsampled_raw)
                v_upsampled = v_upsampled.to(dtype=input_dtype, device=device)

                v_noise_mask = torch.ones((v_batch, 1, v_upsampled.shape[2], 1, 1), dtype=torch.float32, device=device)
                
                if chunk_start > 0:
                    overlap_in_frames = chunk_start - overlap_start
                    overlap_out_frames = overlap_in_frames * 2 - 1
                    global_start = overlap_start * 2
                    
                    v_noise_mask[:, :, :overlap_out_frames] = 0.0
                    v_upsampled[:, :, :overlap_out_frames] = global_v_temporal[:, :, global_start : global_start + overlap_out_frames]

                v_inject_tile = global_injection_samples[:, :, up_overlap_start : up_overlap_start + tile_out_frames]
                m_inject_tile = global_injection_mask[:, :, up_overlap_start : up_overlap_start + tile_out_frames]
                
                v_upsampled = torch.where(m_inject_tile == 0.0, v_inject_tile, v_upsampled)
                v_noise_mask = torch.min(v_noise_mask, m_inject_tile)

                a_noise_mask = torch.zeros((v_batch, 1, a_tile.shape[2], a_tile.shape[3]), dtype=torch.float32, device=device)
                
                concat_latent = {
                    "samples": comfy.nested_tensor.NestedTensor((v_upsampled, a_tile)),
                    "noise_mask": comfy.nested_tensor.NestedTensor((v_noise_mask, a_noise_mask)),
                    "sample_rate": sampling_rate,
                    "type": "audio"
                }

                PromptServer.instance.send_sync("ltxv_eta_update", {
                    "node_id": node_id, "step": 1, "total_steps": 1, "chunk": chunk_idx,
                    "total_chunks": num_chunks, "global_step": chunk_idx,
                    "total_global_steps": num_chunks, "pass_name": "Temporal Upscale Pass",
                    "is_face_restore": bool(restore_faces)
                })
                
                print(f"-> Temporal Upscale Chunk {chunk_idx}/{num_chunks}...")

                temporal_guider = comfy.samplers.CFGGuider(temporal_model)
                temporal_guider.set_cfg(1.0)
                
                center_latent = overlap_start + ((chunk_end - overlap_start) / 2.0)
                shot_len_latents = v_frames_base / num_prompts
                shot_idx = int(center_latent // shot_len_latents)
                shot_idx = min(max(shot_idx, 0), num_prompts - 1)
                
                chunk_pos = [crop_pos[shot_idx]]
                chunk_neg = [crop_neg[shot_idx]] if len(crop_neg) > shot_idx else crop_neg
                temporal_guider.set_conds(chunk_pos, chunk_neg)

                temporal_sampler = comfy.samplers.sampler_object("euler_cfg_pp")
                temporal_model_sampling = temporal_model.get_model_object("model_sampling")

                node_steps = 4
                node_denoise = temporal_denoise
                total_schedule_steps = int(node_steps / node_denoise)
                total_sigmas = comfy.samplers.calculate_sigmas(temporal_model_sampling, "linear_quadratic", total_schedule_steps)
                temporal_sigmas_tensor = total_sigmas[-(node_steps + 1):].to(device)

                chunk_seed_number = seed_number + chunk_idx

                chunk_noise_obj = SlicedNoise(global_noise_tensor, up_overlap_start, chunk_seed_number)
                noise = chunk_noise_obj.generate_noise(concat_latent)

                sampled_latent_raw = temporal_guider.sample(
                    noise, 
                    concat_latent["samples"], 
                    temporal_sampler, 
                    temporal_sigmas_tensor, 
                    denoise_mask=concat_latent["noise_mask"], 
                    callback=None, 
                    disable_pbar=disable_pbar, 
                    seed=seed_number
                )
                sampled_latent = {"samples": sampled_latent_raw}

                unbound_latents = sampled_latent["samples"].unbind()
                sampled_v_tile = unbound_latents[0].to(device)

                if sampled_v_tile.ndim == 4:
                    sampled_v_tile = sampled_v_tile.unsqueeze(0)

                if chunk_start == 0:
                    global_v_temporal[:, :, :sampled_v_tile.shape[2]] = sampled_v_tile
                else:
                    new_frames = sampled_v_tile.shape[2] - overlap_out_frames
                    if new_frames > 0:
                        paste_start = up_overlap_start + overlap_out_frames
                        global_v_temporal[:, :, paste_start : paste_start + new_frames] = sampled_v_tile[:, :, overlap_out_frames:]

                if chunk_end >= v_frames_base:
                    break

                chunk_start = chunk_end - 3
                if chunk_start < 0: chunk_start = 0
                chunk_idx += 1

            final_video_samples = global_v_temporal
            current_fps *= 2
            
            # In multi-shot, setup frames weren't lobbed off, so we mathematically double them here so the final VAE cuts them perfectly!
            if num_prompts > 1 and out_ref_frame_count > 0:
                out_ref_frame_count = (out_ref_frame_count * 2) - 1

            for p in final_positive:
                p[1]["frame_rate"] = float(current_fps)
            for n in final_negative:
                n[1]["frame_rate"] = float(current_fps)

            v_batch, _, v_frames, v_height, v_width = final_video_samples.shape
            final_v_mask = torch.ones((v_batch, v_frames, v_height, v_width), dtype=torch.float32, device=device)
            # ENFORCE 3D TEMPORAL AUDIO MASK [Batch, Frames, Freq]
            final_a_mask = torch.ones((v_batch, final_audio_samples.shape[2], final_audio_samples.shape[3]), dtype=torch.float32, device=device)

            final_latent = {
                "samples": comfy.nested_tensor.NestedTensor((final_video_samples, final_audio_samples)),
                "noise_mask": comfy.nested_tensor.NestedTensor((final_v_mask.unsqueeze(1), final_a_mask)),
                "sample_rate": sampling_rate,
                "type": "audio"
            }
            video_out_latent = {
                "samples": final_video_samples,
                "noise_mask": final_v_mask
            }
            audio_out_latent = {
                "samples": final_audio_samples,
                "noise_mask": final_a_mask,
                "sample_rate": sampling_rate,
                "type": "audio"
            }
            debug_save_latent(final_latent, "stage3_temporal")

        # ==========================================
        # 8. INTEGRATED VAE DECODE, RESTORE & POST-SLICE A/V
        # ==========================================
        out_image = None
        out_audio = None
        out_video = None
        
        # Flush diffusion models from VRAM before VAE decoding begins
        comfy.model_management.free_memory(12 * 1024 * 1024 * 1024, device)
        comfy.model_management.soft_empty_cache()

        if decode:
            print(f"\n--- Running Integrated Decode & Slicer ({num_prompts} shots) ---")
            
            fr_device = comfy.model_management.get_torch_device()
            face_helper = None
            loaded_facerestore_model = None
            
            if restore_faces and facerestore_model != "None":
                try:
                    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
                    from torchvision.transforms.functional import normalize
                    from basicsr.utils.registry import ARCH_REGISTRY
                    from comfy_extras.chainner_models import model_loading
                    
                    print("\n--- Face Restoration Engine Online ---")
                    
                    model_path = folder_paths.get_full_path("trixope_facerestore", facerestore_model)
                    if "codeformer" in facerestore_model.lower():
                        print(f'\tLoading CodeFormer: {facerestore_model}')
                        codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
                            dim_embd=512,
                            codebook_size=1024,
                            n_head=8,
                            n_layers=9,
                            connect_list=["32", "64", "128", "256"],
                        ).to(fr_device)
                        checkpoint = torch.load(model_path)["params_ema"]
                        codeformer_net.load_state_dict(checkpoint)
                        loaded_facerestore_model = codeformer_net.eval()
                    else:
                        print(f'\tLoading FaceRestore Model: {facerestore_model}')
                        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
                        loaded_facerestore_model = model_loading.load_state_dict(sd).eval().to(fr_device)
                        
                    face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model=facedetection, save_ext='png', use_parse=True, device=fr_device)
                    
                    if hasattr(face_helper, 'face_det') and face_helper.face_det is not None:
                        for param in face_helper.face_det.parameters():
                            param.data = param.data.to(fr_device)
                        for buffer in face_helper.face_det.buffers():
                            buffer.data = buffer.data.to(fr_device)
                            
                    if hasattr(face_helper, 'face_parse') and face_helper.face_parse is not None:
                        for param in face_helper.face_parse.parameters():
                            param.data = param.data.to(fr_device)
                        for buffer in face_helper.face_parse.buffers():
                            buffer.data = buffer.data.to(fr_device)
                        
                except Exception as e:
                    print(f"\nWARNING: Face restoration initialization failed: {e}")
                    restore_faces = False

            if pure_image_passthrough:
                print("-> PURE PASSTHROUGH: Bypassing VAE, Face Restore, and Audio Slicing entirely.")
                out_image = images.cpu()
                out_audio = audio
                out_ref_frame_count = 0
            else:
                # ---------------------------------------------------------
                # RAM-SAFE MASTERING ENGINE: VAE, Face Restore, ColorFX, Sync & Slice
                # ---------------------------------------------------------
                total_splits = 1
                if autoregressive_chunking and chunk_size_seconds < length_in_seconds:
                    total_splits = int(round(length_in_seconds / chunk_size_seconds))
                if num_prompts > 1:
                    total_splits = num_prompts

                v_batch, v_channels, v_latents, v_h, v_w = final_video_samples.shape
                out_h = v_h * 32
                out_w = v_w * 32

                if not primary_sampling:
                    video_trim = 0
                    audio_trim_frames = 0
                    if pure_image_passthrough or not (spatial_upscale or temporal_upscale):
                        exact_target_frames = images.shape[0] if images is not None else int(length_in_seconds * current_fps)
                    else:
                        # Dynamically match the exact total frames decoded from the latent dimension
                        exact_target_frames = ((v_latents - 1) * 8) + 1
                else:
                    exact_target_frames = int(length_in_seconds * current_fps)
                    audio_trim_frames = out_ref_frame_count if video_mode != "text-to-video" else 0
                    video_trim = out_ref_frame_count if video_mode != "text-to-video" else 0
                    if video_mode == "reference-to-video":
                        n_ref_images = max(1, ref_frame_count // duplicate_frames) if duplicate_frames > 0 else 1
                        video_trim += (2 * n_ref_images)

                print(f"-> Executing RAM-Optimized Mastering Engine ({total_splits} chunks - CPU Offloaded)...")

                try:
                    # Pre-allocate the ONLY global return tensor to prevent 2x RAM spikes during torch.cat!
                    # MUST USE ZEROS: torch.empty leaves NaNs in memory which causes FFmpeg to render pure black voids!
                    out_image = torch.zeros((exact_target_frames, out_h, out_w, 3), dtype=torch.uint8, device='cpu')
                except RuntimeError as e:
                    print(f"CRITICAL ERROR: Insufficient RAM to hold {exact_target_frames} frames. Your hardware cannot assemble a video of this length in memory.")
                    raise e

                current_out_frame = 0
                
                if not primary_sampling and images is not None:
                    audio_trim_frames = 0
                    video_trim = 0
                else:
                    audio_trim_frames = out_ref_frame_count if video_mode != "text-to-video" else 0
                    video_trim = out_ref_frame_count if video_mode != "text-to-video" else 0
                    if video_mode == "reference-to-video":
                        n_ref_images = max(1, ref_frame_count // duplicate_frames) if duplicate_frames > 0 else 1
                        video_trim += (2 * n_ref_images)

                def process_frame_chunk(chunk_tensor):
                    if restore_faces and face_helper is not None and loaded_facerestore_model is not None:
                        for f in range(chunk_tensor.shape[0]):
                            frame_np = (chunk_tensor[f].numpy() * 255.0).astype(np.uint8)
                            frame_bgr = frame_np[:, :, ::-1]
                            
                            face_helper.clean_all()
                            face_helper.read_image(frame_bgr)
                            face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                            face_helper.align_warp_face()
                            
                            from torchvision.transforms.functional import normalize
                            for idx, cropped_face in enumerate(face_helper.cropped_faces):
                                cropped_face_t = cropped_face.astype(np.float32) / 255.0
                                cropped_face_t = cv2.cvtColor(cropped_face_t, cv2.COLOR_BGR2RGB)
                                cropped_face_t = torch.from_numpy(cropped_face_t.transpose(2, 0, 1)).float()
                                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                                cropped_face_t = cropped_face_t.unsqueeze(0).to(fr_device)
                                
                                try:
                                    with torch.no_grad():
                                        output = loaded_facerestore_model(cropped_face_t, w=codeformer_fidelity)[0]
                                        output = output.squeeze(0).float().cpu().clamp_(-1, 1)
                                        output = (output + 1) / 2.0
                                        output_np = output.numpy().transpose(1, 2, 0)
                                        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
                                        restored_face_raw = (output_bgr * 255.0).round().astype(np.uint8)
                                        final_restored_face = restored_face_raw
                                        
                                        if face_restore_color_match:
                                            orig_lab = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2LAB).astype(np.float32)
                                            rest_lab = cv2.cvtColor(final_restored_face, cv2.COLOR_BGR2LAB).astype(np.float32)
                                            for c in range(3):
                                                orig_mean, orig_std = orig_lab[:,:,c].mean(), orig_lab[:,:,c].std()
                                                rest_mean, rest_std = rest_lab[:,:,c].mean(), rest_lab[:,:,c].std()
                                                rest_lab[:,:,c] = (rest_lab[:,:,c] - rest_mean) * (orig_std / (rest_std + 1e-6)) + orig_mean
                                            rest_lab = np.clip(rest_lab, 0, 255).astype(np.uint8)
                                            final_restored_face = cv2.cvtColor(rest_lab, cv2.COLOR_LAB2BGR)
                                            
                                        if face_restore_edge_blur:
                                            h, w = final_restored_face.shape[:2]
                                            mask = np.zeros((h, w, 3), dtype=np.float32)
                                            pad = int(h * 0.12)
                                            mask[pad:h-pad, pad:w-pad] = 1.0
                                            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=pad*0.5, sigmaY=pad*0.5)
                                            final_restored_face = (final_restored_face * mask + cropped_face.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
                                    face_helper.add_restored_face(final_restored_face)
                                except Exception as error:
                                    face_helper.add_restored_face(cropped_face)
                                    
                            face_helper.get_inverse_affine(None)
                            pasted_img_bgr = face_helper.paste_faces_to_input_image()
                            
                            if face_restore_blend < 1.0:
                                final_img_bgr = cv2.addWeighted(frame_bgr, 1.0 - face_restore_blend, pasted_img_bgr, face_restore_blend, 0)
                            else:
                                final_img_bgr = pasted_img_bgr
                                
                            restored_img_rgb = final_img_bgr[:, :, ::-1]
                            chunk_tensor[f] = torch.from_numpy(restored_img_rgb.astype(np.float32) / 255.0)

                    if enable_colorfx:
                        if TQDM_AVAILABLE: 
                            iterable = tqdm(range(chunk_tensor.shape[0]), desc="ColorFX Chunk")
                        else:
                            iterable = range(chunk_tensor.shape[0])
                            
                        processed_images_tensors = []
                        for i in iterable:
                            pil_image = tensor2pil(chunk_tensor[i:i+1]).convert("RGB")
                            if enable_color_correction:
                                pil_image = self._apply_shadows_highlights(pil_image, shadow_intensity, highlight_intensity, hdr_intensity)
                                pil_image = self._apply_color_enhancements(pil_image, brightness, contrast, saturation, enhance_color)
                            if enable_lut_processing and lut_name != "None":
                                pil_image = self._apply_lut_effect(pil_image, lut_name, lut_strength, lut_log_process)
                            if enable_enhancements:
                                pil_image = self._apply_sharpness_detail(pil_image, sharpness, edge_enhance_strength, detail_enhance_strength)
                            if enable_blur_effects:
                                pil_image = self._apply_blurs(pil_image, blur_radius, gaussian_blur_radius)
                                if radial_blur_strength > 0.0:
                                    pil_image = self._apply_radial_blur(pil_image, radial_blur_strength, radial_blur_center_x, radial_blur_center_y, radial_blur_focus_spread, radial_blur_steps)
                            if enable_stylistic_effects:
                                needs_pil = any(v!=0 for v in [chromatic_aberration_r_x, chromatic_aberration_r_y, chromatic_aberration_b_x, chromatic_aberration_b_y]) or chromatic_blur_amount > 0 or scanline_intensity > 0.0 or soft_light_opacity > 0.0
                                if needs_pil:
                                    if any(v!=0 for v in [chromatic_aberration_r_x, chromatic_aberration_r_y, chromatic_aberration_b_x, chromatic_aberration_b_y]) or chromatic_blur_amount > 0:
                                        pil_image = self._apply_chromatic_aberration(pil_image, chromatic_aberration_r_x, chromatic_aberration_r_y, chromatic_aberration_b_x, chromatic_aberration_b_y, chromatic_blur_amount)
                                    if scanline_intensity > 0.0:
                                        pil_image = self._apply_scanlines(pil_image, scanline_intensity)
                                    if soft_light_opacity > 0.0:
                                        pil_image = self._apply_soft_light(pil_image, soft_light_opacity, soft_light_blur_radius)
                            processed_images_tensors.append(pil2tensor(pil_image))
                            
                        chunk_tensor = torch.cat(processed_images_tensors, dim=0).to(chunk_tensor.device)
                        if enable_color_correction and gamma != 1.0:
                            chunk_tensor = self._apply_gamma_torch(chunk_tensor, gamma)
                        if enable_stylistic_effects:
                            if simple_film_grain_intensity > 0.0:
                                chunk_tensor = self._apply_simple_film_grain_torch(chunk_tensor, simple_film_grain_intensity, simple_film_grain_monochrome)
                            if vignette_intensity > 0.0:
                                chunk_tensor = self._apply_vignette_torch(chunk_tensor, vignette_intensity, vignette_center_x, vignette_center_y)
                                
                    return chunk_tensor

                if not primary_sampling and images is not None and not (spatial_upscale or temporal_upscale):
                    print("-> Direct Image Processing: Running Face Restore / ColorFX directly on input image batch...")
                    proc_imgs = process_frame_chunk(images.clone())
                    out_image = (proc_imgs * 255.0).clamp(0, 255).to(torch.uint8)
                    out_audio = audio
                else:
                    if num_prompts == 1:
                        # OPTIMIZATION: If the single-shot V2V pixel buffer exists, reuse it to skip a secondary VAE decoding pass[cite: 1]
                        if v2v_master_frames is not None:
                            print("-> [Memory Optimization] Reusing pre-compiled V2V pixel buffer. Bypassing final VAE decoding loop to free system RAM.")
                            total_pixel_frames = v2v_master_frames.shape[0]
                            slice_size = 16 # Streams frames in tight blocks to keep memory overhead minimal
                            frame_idx = 0
                            
                            while frame_idx < total_pixel_frames:
                                end_idx = min(frame_idx + slice_size, total_pixel_frames)
                                chunk_to_process = v2v_master_frames[frame_idx:end_idx].clone()
                                
                                # Cleanly slice off setup/reference frames on the initial step if needed
                                if frame_idx == 0 and video_trim > 0:
                                    chunk_to_process = chunk_to_process[video_trim:]
                                    
                                processed_chunk = process_frame_chunk(chunk_to_process)
                                frames_to_paste = processed_chunk.shape[0]
                                end_paste = min(current_out_frame + frames_to_paste, exact_target_frames)
                                actual_paste = end_paste - current_out_frame
                                
                                if actual_paste > 0:
                                    out_image[current_out_frame : end_paste] = (processed_chunk[:actual_paste] * 255.0).clamp(0, 255).to(torch.uint8)
                                    
                                current_out_frame += actual_paste
                                frame_idx = end_idx
                                
                                # Aggressive garbage collection sweeps after each slice
                                del chunk_to_process, processed_chunk
                                import gc
                                gc.collect()
                                comfy.model_management.soft_empty_cache()
                                
                            # Free the master pixel accumulation buffer completely
                            try:
                                del v2v_master_latents
                            except (NameError, UnboundLocalError):
                                pass
                            import gc
                            gc.collect()
                        else:
                            # Fallback pipeline if the pixel buffer wasn't initialized
                            temp_tile_len = 16 
                            temp_overlap = 4  
                            chunk_start = 0
                            valid_end_frame = 0
                            prev_raw_overlap = None
                            
                            while chunk_start < v_latents:
                                if chunk_start == 0:
                                    chunk_end = min(chunk_start + temp_tile_len, v_latents)
                                    overlap_start = chunk_start
                                else:
                                    overlap_start = max(1, chunk_start - temp_overlap - 1)
                                    chunk_end = min(chunk_start + temp_tile_len - (chunk_start - overlap_start), v_latents)
                                    
                                if chunk_end <= chunk_start: chunk_end = chunk_start + 1
                                    
                                v_tile = final_video_samples[:, :, overlap_start:chunk_end]
                                v_tile_decoded = safe_vae_decode(video_vae, v_tile).cpu()
                                v_tile_decoded = v_tile_decoded.view(-1, out_h, out_w, 3)
                                current_valid_frames = v_tile_decoded.shape[0]
                                
                                if chunk_start == 0:
                                    raw_chunk = v_tile_decoded
                                    valid_end_frame = current_valid_frames
                                    use_len = 0
                                else:
                                    global_start_frame = overlap_start * 8
                                    overlap_frames = valid_end_frame - global_start_frame
                                    
                                    if overlap_frames > 0 and prev_raw_overlap is not None:
                                        use_len = min(overlap_frames, prev_raw_overlap.shape[0], v_tile_decoded.shape[0])
                                        feather = torch.linspace(0.0, 1.0, use_len).view(-1, 1, 1, 1)
                                        prev_overlap = prev_raw_overlap[-use_len:]
                                        new_overlap = v_tile_decoded[:use_len]
                                        blended = prev_overlap * (1.0 - feather) + new_overlap * feather
                                        raw_chunk = torch.cat([blended, v_tile_decoded[use_len:]], dim=0)
                                    else:
                                        raw_chunk = v_tile_decoded
                                        use_len = 0
                                        
                                    valid_end_frame += (current_valid_frames - use_len)

                                next_overlap_start = max(1, chunk_end - temp_overlap - 1)
                                next_global_start = next_overlap_start * 8
                                overlap_frames_for_next = valid_end_frame - next_global_start
                                
                                if chunk_end < v_latents and overlap_frames_for_next > 0:
                                    prev_raw_overlap = raw_chunk[-overlap_frames_for_next:].clone()
                                    chunk_to_process = raw_chunk[:-overlap_frames_for_next]
                                else:
                                    prev_raw_overlap = None
                                    chunk_to_process = raw_chunk

                                if chunk_start == 0 and video_trim > 0:
                                    chunk_to_process = chunk_to_process[video_trim:]
                                    
                                processed_chunk = process_frame_chunk(chunk_to_process)
                                frames_to_paste = processed_chunk.shape[0]
                                end_paste = min(current_out_frame + frames_to_paste, exact_target_frames)
                                actual_paste = end_paste - current_out_frame
                                
                                if actual_paste > 0:
                                    out_image[current_out_frame : end_paste] = (processed_chunk[:actual_paste] * 255.0).clamp(0, 255).to(torch.uint8)
                                    
                                current_out_frame += actual_paste
                                
                                del v_tile_decoded, raw_chunk, chunk_to_process, processed_chunk
                                comfy.model_management.soft_empty_cache()
                                chunk_start = chunk_end
                                
                        if current_out_frame < exact_target_frames and current_out_frame > 0:
                            pad_len = exact_target_frames - current_out_frame
                            out_image[current_out_frame : exact_target_frames] = out_image[current_out_frame - 1].unsqueeze(0).repeat(pad_len, 1, 1, 1)

                    else:
                        frames_per_split_out = (length_in_seconds * current_fps) / total_splits
                        start_lat = 0
                        print(f"   [DEBUG-SHOT] num_prompts={num_prompts} total_splits={total_splits} "
                              f"out_ref_frame_count={out_ref_frame_count} bypass_img_ref={bypass_img_ref} "
                              f"video_trim={video_trim} final_video_samples.shape={tuple(final_video_samples.shape)}")
                        
                        for s in range(total_splits):
                            target_shot_frames = int(round((s + 1) * frames_per_split_out)) - int(round(s * frames_per_split_out))
                            
                            if s == 0:
                                shot_target_frames = target_shot_frames + (out_ref_frame_count if not bypass_img_ref else 0)
                            else:
                                shot_target_frames = target_shot_frames
                            
                            shot_lats = ((shot_target_frames - 1) // 8) + 1
                            if ((shot_lats - 1) * 8 + 1) < shot_target_frames:
                                shot_lats += 1
                                
                            end_lat = min(start_lat + shot_lats, final_video_samples.shape[2])
                            if s == total_splits - 1:
                                end_lat = final_video_samples.shape[2]
                                
                            split_latent = final_video_samples[:, :, start_lat : end_lat]
                            
                            if split_latent.shape[2] > 0:
                                decoded_shot = safe_vae_decode(video_vae, split_latent).cpu()
                                decoded_shot = decoded_shot.view(-1, out_h, out_w, 3)
                            else:
                                break
                            
                            print(f"   [DEBUG-SHOT] s={s} start_lat={start_lat} end_lat={end_lat} "
                                  f"target_shot_frames={target_shot_frames} shot_target_frames={shot_target_frames} "
                                  f"decoded_shot.shape[0]_before_trim={decoded_shot.shape[0]}")
                            
                            # Only shot 0 carries un-trimmed warm-up -- shots 1+ were already sliced by the razor blade.
                            if s == 0 and video_trim > 0:
                                decoded_shot = decoded_shot[video_trim:]

                            if decoded_shot.shape[0] > target_shot_frames:
                                decoded_shot = decoded_shot[:target_shot_frames]
                            elif decoded_shot.shape[0] < target_shot_frames:
                                pad_len = target_shot_frames - decoded_shot.shape[0]
                                decoded_shot = torch.cat([decoded_shot, decoded_shot[-1:].repeat(pad_len, 1, 1, 1)], dim=0)

                            processed_chunk = process_frame_chunk(decoded_shot)
                            
                            frames_to_paste = processed_chunk.shape[0]
                            end_paste = min(current_out_frame + frames_to_paste, exact_target_frames)
                            actual_paste = end_paste - current_out_frame
                            
                            if actual_paste > 0:
                                out_image[current_out_frame : end_paste] = (processed_chunk[:actual_paste] * 255.0).clamp(0, 255).to(torch.uint8)
                                
                            current_out_frame += actual_paste
                            
                            del decoded_shot
                            del processed_chunk
                            comfy.model_management.soft_empty_cache()
                            start_lat = end_lat

                # --- EXACT AUDIO DECODE & ALIGNMENT ---
                a_latent_samples = final_audio_samples
                if getattr(a_latent_samples, "is_nested", False):
                    a_latent_samples = a_latent_samples.unbind()[-1]
                
                sample_rate = int(getattr(audio_vae, "output_sample_rate", audio_vae.first_stage_model.output_sample_rate))
                
                time_to_drop = audio_trim_frames / current_fps
                samples_to_drop = int(time_to_drop * sample_rate)
                base_fps = current_fps / 2.0 if temporal_upscale else current_fps

                print("--- Decoding Master Audio Track ---")
                
                # TESTING: force the generated-audio path even when a source track was provided, so we can
                # check whether shot N's video lines up with the audio the MODEL actually generated for that
                # shot, rather than the pristine source track (which sidesteps any per-shot audio issues by
                # construction). Set back to False once this test is done.
                FORCE_GENERATED_AUDIO_FOR_TESTING = True
                
                if has_audio_input and not temporal_upscale and not FORCE_GENERATED_AUDIO_FOR_TESTING:
                    print("-> Bypassing VAE Decode: Using pristine source waveform directly to prevent truncation...")
                    waveform = torchaudio.functional.resample(master_wf.clone(), sampling_rate, sample_rate).to(device)
                else:
                    print("-> Executing Safe Chunked Audio Decode...")
                    get_audio_latents_func = getattr(audio_vae, "num_of_latents_from_frames", getattr(audio_vae.first_stage_model, "num_of_latents_from_frames", None))
                    max_audio_lats = get_audio_latents_func(int(20.0 * current_fps), int(current_fps))
                    
                    if a_latent_samples.shape[2] > max_audio_lats:
                        wf_chunks = []
                        for i in range(0, a_latent_samples.shape[2], max_audio_lats):
                            chunk = a_latent_samples[:, :, i:i+max_audio_lats]
                            wf_chunks.append(audio_vae.decode(chunk).to(a_latent_samples.device).movedim(-1, 1))
                        waveform = torch.cat(wf_chunks, dim=-1)
                    else:
                        waveform = audio_vae.decode(a_latent_samples).to(a_latent_samples.device).movedim(-1, 1)

                    exact_target_samples = int((length_in_seconds + time_to_drop) * sample_rate)
                    if waveform.shape[-1] > exact_target_samples:
                        extra_samps = waveform.shape[-1] - exact_target_samples
                        waveform = waveform[..., extra_samps:]
                    elif waveform.shape[-1] < exact_target_samples:
                        pad_len = exact_target_samples - waveform.shape[-1]
                        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
                        
                if has_audio_ref and not temporal_upscale:
                    pristine_wf = torchaudio.functional.resample(master_wf.clone(), sampling_rate, sample_rate).to(waveform.device)
                    if pristine_wf.shape[1] < waveform.shape[1]:
                        pristine_wf = pristine_wf.repeat(1, waveform.shape[1], 1)
                    elif pristine_wf.shape[1] > waveform.shape[1]:
                        waveform = waveform.repeat(1, pristine_wf.shape[1], 1)
                        
                    region_a_samps_out = int((region_a_frames / base_fps) * sample_rate)
                    limit = min(region_a_samps_out, pristine_wf.shape[-1], waveform.shape[-1])
                    if limit > 0: waveform[..., :limit] = pristine_wf[..., :limit]

                if not has_audio_input and num_prompts > 1:
                    print("-> Executing Isolated Shot Audio Decode for perfect multi-shot sync...")
                    frames_per_split_out = (length_in_seconds * current_fps) / total_splits
                    trimmed_audio_chunks = []
                    
                    shot_duration = length_in_seconds / num_prompts
                    total_a_latents = a_latent_samples.shape[2]
                    
                    for s in range(total_splits):
                        sec_start = s * shot_duration
                        sec_end = (s + 1) * shot_duration
                        
                        _, a_start_lat = get_latent_counts(sec_start)
                        _, a_end_lat = get_latent_counts(sec_end)
                        if s == total_splits - 1:
                            a_end_lat = total_a_latents
                            
                        s_lat = a_latent_samples[:, :, a_start_lat:a_end_lat]
                        chunk_wf = audio_vae.decode(s_lat).to(a_latent_samples.device).movedim(-1, 1)
                        
                        target_shot_frames = int(round((s + 1) * frames_per_split_out)) - int(round(s * frames_per_split_out))
                        target_samples = int(target_shot_frames * (sample_rate / base_fps))
                        
                        setup_samples = 0
                        if out_ref_frame_count > 0 and not bypass_img_ref:
                            setup_samples = int(out_ref_frame_count * (sample_rate / base_fps))
                            
                        # 1. Trim the tail overhang
                        if chunk_wf.shape[-1] > target_samples + setup_samples:
                            chunk_wf = chunk_wf[..., :target_samples + setup_samples]
                            
                        # 2. AUDIO PURIFICATION: Amputate the setup silence from the front of every shot!
                        if setup_samples > 0:
                            if chunk_wf.shape[-1] > setup_samples:
                                chunk_wf = chunk_wf[..., setup_samples:]
                            else:
                                chunk_wf = torch.zeros((chunk_wf.shape[0], chunk_wf.shape[1], target_samples), device=chunk_wf.device, dtype=chunk_wf.dtype)
                                
                        if chunk_wf.shape[-1] < target_samples:
                            pad_len = target_samples - chunk_wf.shape[-1]
                            chunk_wf = torch.nn.functional.pad(chunk_wf, (0, pad_len))
                            
                        trimmed_audio_chunks.append(chunk_wf)
                        
                    waveform = torch.cat(trimmed_audio_chunks, dim=-1)

            # ==========================================
            # AUDIO MASTERING & MUXING (SKIPPED FOR PASSTHROUGH)
            # ==========================================
            if not primary_sampling:
                print("-> Base Generation OFF: Passing connected audio directly to output...")
                out_audio = audio
            elif not pure_image_passthrough:
                final_video_length_seconds = out_image.shape[0] / current_fps
                exact_target_samples = int(final_video_length_seconds * sample_rate)
                
                if out_audio is not None and out_audio["waveform"].shape[-1] > exact_target_samples:
                    out_audio["waveform"] = out_audio["waveform"][..., :exact_target_samples]
                    
                print("-> Running Final Audio Mastering Pass (Soft Limiter & Normalization)...")
                threshold = 0.90
                abs_wf = torch.abs(waveform)
                clip_mask = abs_wf > threshold
                if clip_mask.any():
                    waveform[clip_mask] = torch.sign(waveform[clip_mask]) * (threshold + 0.1 * torch.tanh((abs_wf[clip_mask] - threshold) / 0.1))
                    
                max_val = torch.max(torch.abs(waveform))
                if max_val > 0.0:
                    gain_boost = 0.95 / max_val
                    gain_boost = torch.clamp(gain_boost, max=3.0)
                    waveform = waveform * gain_boost
                    
                waveform = torch.clamp(waveform, -1.0, 1.0)
                total_samples = waveform.shape[-1]
                
                if samples_to_drop >= total_samples:
                    empty_wf = torch.zeros((1, 2, 1), device=waveform.device, dtype=torch.float32)
                    out_audio = {"waveform": empty_wf.cpu(), "sample_rate": sample_rate}
                elif samples_to_drop > 0:
                    sliced_wf = waveform[..., samples_to_drop:]
                    fade_samples = min(int(0.03 * sample_rate), sliced_wf.shape[-1])
                    if fade_samples > 0:
                        fade_tensor = torch.linspace(0.0, 1.0, fade_samples, device=sliced_wf.device, dtype=sliced_wf.dtype)
                        sliced_wf[..., :fade_samples] *= fade_tensor
                    
                    final_wf = sliced_wf
                    if final_wf.dim() == 2: final_wf = final_wf.unsqueeze(0)
                    if final_wf.shape[1] == 1: final_wf = final_wf.repeat(1, 2, 1)
                    elif final_wf.shape[1] > 2: final_wf = final_wf[:, :2, :]
                    
                    final_wf = final_wf.to(torch.float32).clamp(-1.0, 1.0).cpu() 
                    out_audio = {"waveform": final_wf, "sample_rate": sample_rate}
                else:
                    final_wf = waveform
                    if final_wf.dim() == 2: final_wf = final_wf.unsqueeze(0)
                    if final_wf.shape[1] == 1: final_wf = final_wf.repeat(1, 2, 1)
                    elif final_wf.shape[1] > 2: final_wf = final_wf[:, :2, :]
                    
                    final_wf = final_wf.to(torch.float32).clamp(-1.0, 1.0).cpu()
                    out_audio = {"waveform": final_wf, "sample_rate": sample_rate}

            # --- NEW MASTER PREVIEW EMBED WITH ACTIVE PLACEHOLDERS ---
            if enable_final_video and out_image is not None and not pure_image_passthrough:
                print("\n--- Generating Master Final Video Render Preview ---")
                import subprocess, uuid
                uid = node_id if node_id is not None else str(uuid.uuid4())
                final_filename = f"final_render_{uid}.mp4"
                temp_dir = folder_paths.get_temp_directory()
                final_path = os.path.join(temp_dir, final_filename)
                final_audio_path = os.path.join(temp_dir, f"final_render_audio_{uid}.wav")

                if os.path.exists(final_audio_path):
                    try: os.remove(final_audio_path)
                    except: pass

                if out_audio is not None and "waveform" in out_audio:
                    try: 
                        audio_wf_2d = out_audio["waveform"][0] if out_audio["waveform"].ndim == 3 else out_audio["waveform"]
                        torchaudio.save(final_audio_path, audio_wf_2d.cpu(), out_audio["sample_rate"])
                    except Exception as ae: print(f"Warning: Final audio file write issue: {ae}")

                cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{out_w}x{out_h}", "-pix_fmt", "rgb24", "-r", str(current_fps), "-i", "-"]
                if os.path.exists(final_audio_path) and os.path.getsize(final_audio_path) > 0: 
                    cmd.extend(["-i", final_audio_path])
                cmd.extend(["-c:v", "libx264", "-preset", "faster", "-crf", "22", "-pix_fmt", "yuv420p", "-r", str(current_fps)])
                if os.path.exists(final_audio_path) and os.path.getsize(final_audio_path) > 0: 
                    cmd.extend(["-c:a", "aac", "-b:a", "192k"])
                cmd.append(final_path)

                try:
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    for i in range(0, out_image.shape[0], 32):
                        chunk_data = (out_image[i : i + 32].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8).tobytes()
                        try:
                            proc.stdin.write(chunk_data)
                        except BrokenPipeError:
                            print("-> FFmpeg final render pipe closed unexpectedly mid-stream.")
                            break
                        
                    proc.stdin.close()
                    proc.wait() # Safely block until the stream render completes entirely
                        
                    if proc.returncode == 0:
                        PromptServer.instance.send_sync("trixope_ltxv_final_video", {"node": uid, "filename": final_filename, "type": "temp"})
                        print("--- Master Final Video Render Preview Sent to UI ---")
                    else:
                        print(f"-> FFmpeg Final Render failed with return code {proc.returncode}.")
                except Exception as e: 
                    print(f"Warning: Master video compilation pipeline issue: {e}")

            if InputImpl is not None and Types is not None and out_image is not None:
                try:
                    out_video = InputImpl.VideoFromComponents(Types.VideoComponents(images=out_image, audio=out_audio, frame_rate=Fraction(current_fps)))
                except Exception as e:
                    print(f"LTXV Custom Warning: Failed to assemble VIDEO type: {e}")
                
            print("--- Decoding & Slicing Complete ---")

        # ==========================================
        # 9. VRAM CLEANUP
        # ==========================================
        def safe_clear_model(m):
            if m is not None:
                try:
                    m.unpatch_model()
                    
                    for block in m.model.diffusion_model.transformer_blocks:
                        if hasattr(block, "attn1") and hasattr(block.attn1, "_10s_trixope_anchor_handle"):
                            block.attn1._10s_trixope_anchor_handle.remove()
                            delattr(block.attn1, "_10s_trixope_anchor_handle")
                except:
                    pass
                
                if hasattr(m, "object_patches") and m.object_patches is not None: 
                    m.object_patches.clear()
                    
                if hasattr(m, "model_options") and m.model_options is not None:
                    if "transformer_options" in m.model_options:
                        m.model_options["transformer_options"].pop("optimized_attention_override", None)
                    
                if hasattr(m, "callbacks") and m.callbacks is not None:
                    try:
                        m.callbacks.clear()
                    except:
                        m.callbacks = []

        # 1. Nuke the circular dictionaries on the wrappers
        safe_clear_model(model_to_use)
        
        if model2_to_use is not None and model2_to_use is not model_to_use:
            safe_clear_model(model2_to_use)
            
        if model3_to_use is not None and model3_to_use is not model_to_use and model3_to_use is not model2_to_use:
            safe_clear_model(model3_to_use)

        # 2. Explictly sever the CFG Guiders (which hold model references)
        try: del primary_guider
        except Exception: pass
        try: del upsample_guider
        except: pass
        try: del temporal_guider
        except: pass

        # 3. Delete all local variable pointers
        del model1_primary
        del model_to_use
        del model2_spatial
        del model2_to_use
        del model3_temporal
        del model3_to_use
        del spatial_upscaler
        del temporal_upscaler
        del video_vae
        del audio_vae
        
        try:
            if restore_faces and loaded_facerestore_model is not None:
                del loaded_facerestore_model
                del face_helper
        except:
            pass
            
        try:
            del diffusion_model
        except:
            pass

        # ==========================================
        # 10. DEEP CACHE CLEANSE (API CALL)
        # ==========================================
        if clear_models_and_cache:
            try:
                address = f"{PromptServer.instance.address}:{PromptServer.instance.port}"
                requests.post(
                    f"http://{address.replace('0.0.0.0','127.0.0.1')}/api/free",
                headers={'Content-Type': 'application/json'},
                json={"unload_models": True, "free_memory": True},
                timeout=10
                )
                print("--- Deep Cache & Models Cleared Successfully ---")
            except Exception as e:
                print(f"--- Deep Cache Clearance Failed: {str(e)} ---")
        else:
            print("--- Deep Cache Clearance Bypassed ---")

        if out_image is not None and out_image.dtype == torch.uint8:
            out_image = out_image.to(torch.float32) / 255.0

        return (final_prompt_string_out, final_positive, final_negative, final_latent, video_out_latent, audio_out_latent, out_video, out_image, out_audio, float(current_fps), out_ref_frame_count, node_id)

# ==========================================
# QUEUE INTERCEPTOR ("CONTROL BEFORE GENERATE")
# ==========================================
def trixope_on_prompt_handler(json_data):
    try:
        prompt = json_data.get("prompt", {})
        seed_map = {}
        for k, v in prompt.items():
            if v.get("class_type") == "FilmAuteur_LTX":
                inputs = v.get("inputs", {})
                
                control = inputs.get("control_before_generate", "randomize")

                if control == "randomize":
                    new_seed = random.randint(0, 0xffffffffffffffff)
                    inputs["seed_number"] = new_seed
                    seed_map[k] = new_seed
                elif control == "increment":
                    new_seed = inputs.get("seed_number", 0) + 1
                    inputs["seed_number"] = new_seed
                    seed_map[k] = new_seed
                elif control == "decrement":
                    new_seed = max(0, inputs.get("seed_number", 0) - 1)
                    inputs["seed_number"] = new_seed
                    seed_map[k] = new_seed

        if seed_map:
            PromptServer.instance.send_sync("trixope-global-seed", {"seed_map": seed_map})
            
    except Exception as e:
        print(f"LTXV Custom Error in prompt interceptor: {e}")
        
    return json_data

PromptServer.instance.add_on_prompt_handler(trixope_on_prompt_handler)

# --- moved from H3_ultimate_upscale.py --------------------------------------
import contextlib
import logging
import os
import re
import sys
from contextlib import contextmanager
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import comfy
import comfy.sample
import comfy.samplers
from .usdu_patch import usdu
from .usdu_utils import tensor_to_pil, pil_to_tensor
from modules.processing import StableDiffusionProcessing
import modules.shared as shared
from modules.upscaler import UpscalerData
from .dlss5.settings import (
    DLSS_MODEL_PRESETS,
    MOTION_MODES,
    NR_PRESETS,
    NR_STYLES,
    UPSCALING_LABELS,
)

logger = logging.getLogger(__name__)

# ==========================================
# COLOR FX (ported from FilmAuteur_LTX)
# ==========================================
import folder_paths
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageChops, ImageDraw

try:
    from scipy.interpolate import RegularGridInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    dir_luts = os.path.join(folder_paths.models_dir, "luts")
    os.makedirs(dir_luts, exist_ok=True)
    if "luts" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["luts"] = ([dir_luts], {".cube"})
    else:
        folder_paths.folder_names_and_paths["luts"][1].add(".cube")
        if dir_luts not in folder_paths.folder_names_and_paths["luts"][0]:
            folder_paths.folder_names_and_paths["luts"][0].append(dir_luts)
except Exception:
    pass


def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

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

def match_color_mean_std(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(Legacy, no longer used by the tile pass.) Global per-channel mean/std match.
    Kept for backward compatibility; it flattens contrast, so the tile pass now uses match_low_frequency()."""
    target_mean = target.mean(dim=(1, 2), keepdim=True)
    target_std = target.std(dim=(1, 2), keepdim=True) + 1e-6
    source_mean = source.mean(dim=(1, 2), keepdim=True)
    source_std = source.std(dim=(1, 2), keepdim=True) + 1e-6
    normalized = (source - source_mean) / source_std
    return torch.clamp(normalized * target_std + target_mean, 0.0, 1.0)


# ----------------------------------------------------------------------------------------------
# Image-space helpers (all tensors are ComfyUI IMAGE layout: [B, H, W, C], float 0..1)
# ----------------------------------------------------------------------------------------------
def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur. Large sigmas are computed on a downscaled copy so it stays fast."""
    if sigma < 0.3:
        return x
    b, h, w, c = x.shape
    t = x.permute(0, 3, 1, 2)
    f = 1
    while sigma / f > 6.0 and min(h // (f * 2), w // (f * 2)) >= 8:
        f *= 2
    if f > 1:
        t = F.interpolate(t, size=(max(1, h // f), max(1, w // f)), mode="area")
    s = sigma / f
    r = max(1, int(np.ceil(3.0 * s)))
    ax = torch.arange(-r, r + 1, device=t.device, dtype=t.dtype)
    k = torch.exp(-(ax ** 2) / (2.0 * s * s))
    k = k / k.sum()
    ch = t.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(ch, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(ch, 1, 1, 1)
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), kx, groups=ch)
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), ky, groups=ch)
    if f > 1:
        t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1)


def _resize_bhwc(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if x.shape[1] == h and x.shape[2] == w:
        return x
    t = x.permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    return torch.clamp(t.permute(0, 2, 3, 1), 0.0, 1.0)


def match_low_frequency(source: torch.Tensor, target: torch.Tensor, sigma: float) -> torch.Tensor:
    """Make source's broad colour/lighting match target's WITHOUT touching fine detail.
    (Replaces the old global mean/std match, which flattened contrast and crushed micro-detail.)"""
    return torch.clamp(source + _gaussian_blur(target, sigma) - _gaussian_blur(source, sigma), 0.0, 1.0)


def restore_detail(decoded: torch.Tensor, source: torch.Tensor, amount: float, sigma: float = None) -> torch.Tensor:
    """Detail guard: wherever the diffused result has LESS local high-frequency energy than the
    pre-diffusion source, blend the source's high frequencies back in. Where diffusion added
    detail, the diffused result is left untouched, so the pass can only add or keep detail."""
    if amount <= 0.0:
        return decoded
    h, w = decoded.shape[1], decoded.shape[2]
    if sigma is None:
        sigma = max(1.0, min(h, w) * 0.002)
    hp_d = decoded - _gaussian_blur(decoded, sigma)
    hp_s = source - _gaussian_blur(source, sigma)
    e_d = _gaussian_blur(hp_d.abs().mean(dim=-1, keepdim=True), sigma * 3.0)
    e_s = _gaussian_blur(hp_s.abs().mean(dim=-1, keepdim=True), sigma * 3.0)
    deficit = torch.clamp((e_s - e_d) / (e_s + 1e-4), 0.0, 1.0)
    return torch.clamp(decoded + (amount * deficit) * (hp_s - hp_d), 0.0, 1.0)


def refine_edges_and_textures(tensor: torch.Tensor, amount: float = 0.35, radius: int = 2) -> torch.Tensor:
    """Luminance-only multi-scale unsharp with soft coring. Chroma is untouched (no colour fringing),
    radius is honoured and scales with resolution."""
    h, w = tensor.shape[1], tensor.shape[2]
    x = tensor[..., :3]
    scale = max(1.0, min(h, w) / 1024.0)
    s_fine = max(0.8, radius * 0.5 * scale)
    luma = (0.299 * x[..., 0:1] + 0.587 * x[..., 1:2] + 0.114 * x[..., 2:3])
    g1 = _gaussian_blur(luma, s_fine)
    g2 = _gaussian_blur(luma, s_fine * 3.0)
    fine = luma - g1
    medium = g1 - g2
    mag = fine.abs()
    core = mag / (mag + 0.003)  # soft coring: ignore the very faintest noise, keep real texture
    delta = amount * (1.2 * fine * core + 0.4 * medium)
    out = torch.clamp(x + delta, 0.0, 1.0)
    if tensor.shape[-1] > 3:
        out = torch.cat([out, tensor[..., 3:]], dim=-1)
    return out


def create_face_blend_mask(height: int, width: int, feather: float = 0.35,
                           device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Circular/elliptical smoothstep mask [1,H,W,1]: 1.0 in the middle, 0.0 at the crop border."""
    yy = torch.linspace(-1.0, 1.0, height, device=device).view(height, 1)
    xx = torch.linspace(-1.0, 1.0, width, device=device).view(1, width)
    r = torch.sqrt(xx * xx + yy * yy)
    t = torch.clamp((1.0 - r) / max(feather, 1e-3), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).unsqueeze(0).unsqueeze(-1)


def instance_blend_mask(inst: np.ndarray, fx: int, fy: int, x1: int, y1: int, crop_h: int, crop_w: int,
                        device) -> torch.Tensor:
    """Place a SAM instance mask (bbox at fx,fy) into the crop (origin x1,y1), grow it a little and feather it.
    Only this region is replaced by the face pass; the rest of the crop (shoulders, body, background) is untouched."""
    full = np.zeros((crop_h, crop_w), np.float32)
    mh, mw = inst.shape
    ox, oy = fx - x1, fy - y1
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    cw_, ch_ = min(mw - sx0, crop_w - dx0), min(mh - sy0, crop_h - dy0)
    if cw_ > 0 and ch_ > 0:
        full[dy0:dy0 + ch_, dx0:dx0 + cw_] = inst[sy0:sy0 + ch_, sx0:sx0 + cw_]
    k = max(3, int(0.03 * max(crop_h, crop_w)))
    full = cv2.dilate(full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    full = cv2.GaussianBlur(full, (0, 0), sigmaX=k * 1.5)
    return torch.from_numpy(np.clip(full, 0.0, 1.0)).view(1, crop_h, crop_w, 1).to(device)


def _cover_resize(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Resize to fully cover (h, w) keeping aspect, then centre-crop. No stretching."""
    xh, xw = x.shape[1], x.shape[2]
    sc = max(h / float(xh), w / float(xw))
    nh, nw = max(h, int(round(xh * sc))), max(w, int(round(xw * sc)))
    y = _resize_bhwc(x, nh, nw)
    t, l = (nh - h) // 2, (nw - w) // 2
    return y[:, t:t + h, l:l + w, :]


def warp_master_to_box(img: torch.Tensor, alpha: torch.Tensor, mbox, tbox, out_h: int, out_w: int):
    """Resample the master face so that ITS face box lands exactly on the TARGET face box (anisotropic scale +
    shift). Fixes the double edge you get when two crops are merely 'cover'-resized without being aligned.
    mbox/tbox = (x, y, w, h) in master-image / output pixels."""
    mbx, mby, mbw, mbh = [float(v) for v in mbox]
    tbx, tby, tbw, tbh = [float(v) for v in tbox]
    im = img[0].detach().cpu().numpy().astype(np.float32)
    al = alpha[0, ..., 0].detach().cpu().numpy().astype(np.float32)
    f = 1.0 / max(mbw / max(tbw, 1e-3), mbh / max(tbh, 1e-3))
    if f < 0.75:  # master is much larger than the target: area-downscale first to avoid aliasing
        nw, nh = max(8, int(im.shape[1] * f)), max(8, int(im.shape[0] * f))
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        al = cv2.resize(al, (nw, nh), interpolation=cv2.INTER_AREA)
        mbx, mby, mbw, mbh = mbx * f, mby * f, mbw * f, mbh * f
    sx, sy = mbw / max(tbw, 1e-3), mbh / max(tbh, 1e-3)
    M = np.array([[sx, 0.0, mbx - tbx * sx], [0.0, sy, mby - tby * sy]], dtype=np.float32)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    im_w = cv2.warpAffine(im, M, (out_w, out_h), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    al_w = cv2.warpAffine(al, M, (out_w, out_h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (torch.from_numpy(im_w).unsqueeze(0).to(img.device),
            torch.from_numpy(al_w).unsqueeze(0).unsqueeze(-1).to(img.device))


def flow_align_np(master: np.ndarray, target: np.ndarray, alpha: np.ndarray = None, max_frac: float = 0.08):
    """Dense optical-flow registration (float32 HxWx3, 0..1, same size). Warps `master` so its internal
    features (eyes, nostrils, creases) land on `target`'s, which a face-box alignment alone cannot do.
    Displacement is clamped and smoothed so it can only make small, plausible corrections."""
    H, W = target.shape[:2]

    def prep(x):
        g = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
        return cv2.GaussianBlur(g, (0, 0), max(1.0, max(H, W) * 0.002))

    tg, mg = prep(target), prep(master)
    try:
        flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(tg, mg, None)
    except Exception:
        flow = cv2.calcOpticalFlowFarneback(tg, mg, None, 0.5, 4, 25, 3, 7, 1.5, 0)
    lim = max_frac * max(H, W)
    mag = np.sqrt((flow ** 2).sum(-1, keepdims=True))
    flow = flow * np.minimum(1.0, lim / np.maximum(mag, 1e-6))
    flow = cv2.GaussianBlur(flow, (0, 0), max(2.0, max(H, W) * 0.012))
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mx, my = xs + flow[..., 0], ys + flow[..., 1]
    out = cv2.remap(master.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    al = None
    if alpha is not None:
        al = cv2.remap(alpha.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, al


def flow_align(master: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor):
    out, al = flow_align_np(master[0].detach().cpu().numpy(), target[0].detach().cpu().numpy(),
                            alpha[0, ..., 0].detach().cpu().numpy())
    return (torch.from_numpy(out).unsqueeze(0).to(master.device),
            torch.from_numpy(al).unsqueeze(0).unsqueeze(-1).to(master.device))


def border_taper(h: int, w: int, frac: float = 0.08, device=torch.device("cpu")) -> torch.Tensor:
    """[1,h,w,1] window that is exactly 0 on the crop border and 1 beyond `frac` of the size, so a crop
    can never leave a visible rectangle behind."""
    def ramp(n):
        d = torch.minimum(torch.arange(n, device=device), torch.arange(n, device=device).flip(0)).float()
        t = torch.clamp(d / max(1.0, frac * n), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    return (ramp(h).view(h, 1) * ramp(w).view(1, w)).view(1, h, w, 1)


def _center_square(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[1], x.shape[2]
    side = min(h, w)
    t, l = (h - side) // 2, (w - side) // 2
    return x[:, t:t + side, l:l + side, :]


def compose_face_detail(decoded: torch.Tensor, source: torch.Tensor, sigma: float) -> torch.Tensor:
    """Structure from the SOURCE (everything coarser than `sigma`), fine detail from whichever of
    source/diffused has more local energy. The diffusion pass can add detail but can never blur the
    face or move its features (horn edges, eyes...), whatever the denoise strength."""
    lp_s = _gaussian_blur(source, sigma)
    hp_s = source - lp_s
    hp_d = decoded - _gaussian_blur(decoded, sigma)
    e_s = _gaussian_blur(hp_s.abs().mean(dim=-1, keepdim=True), sigma * 1.5)
    e_d = _gaussian_blur(hp_d.abs().mean(dim=-1, keepdim=True), sigma * 1.5)
    w = torch.clamp(0.5 + (e_d - e_s) / (e_d + e_s + 1e-4), 0.0, 1.0)
    return torch.clamp(lp_s + w * hp_d + (1.0 - w) * hp_s, 0.0, 1.0)


def source_lock_scale(denoise: float, follow: bool = True) -> float:
    """How strongly the 'keep the source' corrections (low-frequency match, detail restore, edge protect,
    tone/chroma lock) are allowed to act. They are right for light refinement, but at higher denoise the model
    re-draws the subject slightly displaced from the source, and pulling the old image back in then leaves a
    second, ghosted copy. Full strength up to 0.25, fading to zero at 0.45 and above."""
    if not follow:
        return 1.0
    return float(np.clip(1.0 - (denoise - 0.25) / 0.20, 0.0, 1.0))


def lock_chroma(decoded: torch.Tensor, source: torch.Tensor, amount: float) -> torch.Tensor:
    """Keep the diffused LUMINANCE but take the colour (B-Y, R-Y) from the source. The VAE misregisters chroma
    on high-contrast edges, which shows up as a purple/magenta fringe on dark-on-light silhouettes."""
    if amount <= 0.0:
        return decoded
    s_ = source[..., :3]
    ys = 0.299 * s_[..., 0:1] + 0.587 * s_[..., 1:2] + 0.114 * s_[..., 2:3]
    cb, cr = s_[..., 2:3] - ys, s_[..., 0:1] - ys
    yd = 0.299 * decoded[..., 0:1] + 0.587 * decoded[..., 1:2] + 0.114 * decoded[..., 2:3]
    r, b = yd + cr, yd + cb
    g = yd - (0.299 * cr + 0.114 * cb) / 0.587
    locked = torch.clamp(torch.cat([r, g, b], dim=-1), 0.0, 1.0)
    return decoded + (locked - decoded) * amount


def edge_protect_weight(source: torch.Tensor, lo: float = 0.025, hi: float = 0.10, grow: int = 6,
                        feather: float = 4.0) -> torch.Tensor:
    """[B,H,W,1] weight, ~1 on strong silhouette edges of the SOURCE (and a few px around them), 0 elsewhere."""
    ys = 0.299 * source[..., 0:1] + 0.587 * source[..., 1:2] + 0.114 * source[..., 2:3]
    l = _gaussian_blur(ys, 1.2)
    gx, gy = torch.zeros_like(l), torch.zeros_like(l)
    gx[:, :, 1:-1] = (l[:, :, 2:] - l[:, :, :-2]) * 0.5
    gy[:, 1:-1] = (l[:, 2:] - l[:, :-2]) * 0.5
    g = torch.sqrt(gx * gx + gy * gy)
    e = torch.clamp((g - lo) / (hi - lo), 0.0, 1.0)
    e = F.max_pool2d(e.permute(0, 3, 1, 2), kernel_size=2 * grow + 1, stride=1, padding=grow).permute(0, 2, 3, 1)
    return torch.clamp(_gaussian_blur(e, feather), 0.0, 1.0)


def seamless_blend_np(dst: np.ndarray, src: np.ndarray, mask: np.ndarray, thresh: float = 0.5, pad: int = 16):
    """Poisson-clone `src` into `dst` inside mask>thresh (float HxWx3 0..1, mask HxW). Inside the mask the result keeps
    src's detail but is shifted so its colour/brightness matches dst exactly along the boundary, so there is no halo,
    box or tint step. dst is untouched outside the mask. Returns None if the mask is unusable."""
    H, W = dst.shape[:2]
    m = (mask > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if int(m.sum()) < 64 or float(m.mean()) > 0.98:
        return None
    ys, xs = np.where(m > 0)
    imin, imax, jmin, jmax = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    lenx, leny = imax - imin, jmax - jmin
    if lenx < 4 or leny < 4:
        return None
    d8 = (np.clip(dst, 0, 1) * 255).astype(np.uint8)
    s8 = (np.clip(src, 0, 1) * 255).astype(np.uint8)
    dp = cv2.copyMakeBorder(d8, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    sp = cv2.copyMakeBorder(s8, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    mp = cv2.copyMakeBorder(m * 255, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    center = (int(jmin + pad + leny // 2), int(imin + pad + lenx // 2))
    out = cv2.seamlessClone(cv2.cvtColor(sp, cv2.COLOR_RGB2BGR), cv2.cvtColor(dp, cv2.COLOR_RGB2BGR), mp, center, cv2.NORMAL_CLONE)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)[pad:pad + H, pad:pad + W].astype(np.float32) / 255.0
    mb = m.astype(np.float32)[..., None]
    return dst * (1.0 - mb) + out * mb


def seamless_face_blend(dst: torch.Tensor, src: torch.Tensor, mask: torch.Tensor):
    """Torch wrapper (tensors [1,H,W,3] / [1,H,W,1]). Returns None on any failure so the caller can alpha-blend."""
    try:
        out = seamless_blend_np(dst[0, ..., :3].detach().cpu().numpy().astype(np.float32),
                                src[0, ..., :3].detach().cpu().numpy().astype(np.float32),
                                mask[0, ..., 0].detach().cpu().numpy().astype(np.float32))
        if out is None:
            return None
        return torch.from_numpy(out).unsqueeze(0).to(device=dst.device, dtype=dst.dtype)
    except Exception as e:
        logger.warning(f"[H3 Upscale] seamless face blend failed ({e}); using alpha blend")
        return None


def create_tile_weight(curr_h: int, curr_w: int, padding: int, top: bool, bottom: bool,
                       left: bool, right: bool, device) -> torch.Tensor:
    """Feather ONLY the edges that touch a neighbouring tile. (The old code feathered the outer canvas
    border too, which drove edge pixels to ~0 weight and produced black/dark border lines.)"""
    wy = torch.ones(curr_h, device=device)
    wx = torch.ones(curr_w, device=device)
    p = min(padding, curr_h // 2, curr_w // 2)
    if p > 0:
        ramp = (torch.arange(p, device=device, dtype=torch.float32) + 1.0) / (p + 1.0)
        if top:
            wy[:p] = torch.minimum(wy[:p], ramp)
        if bottom:
            wy[curr_h - p:] = torch.minimum(wy[curr_h - p:], ramp.flip(0))
        if left:
            wx[:p] = torch.minimum(wx[:p], ramp)
        if right:
            wx[curr_w - p:] = torch.minimum(wx[curr_w - p:], ramp.flip(0))
    return (wy.view(1, curr_h, 1, 1) * wx.view(1, 1, curr_w, 1))


def normalize_decoded(decoded) -> torch.Tensor:
    """Whatever the (video) VAE returns -> [B, H, W, 3]."""
    if isinstance(decoded, dict):
        decoded = decoded["samples"]
    while decoded.dim() > 4:          # [B, T, H, W, C] -> first frame
        decoded = decoded[:, 0]
    if decoded.dim() == 4 and decoded.shape[-1] > 4 and decoded.shape[1] <= 4:
        decoded = decoded.permute(0, 2, 3, 1)   # channels-first -> channels-last
    return decoded[..., :3]


# ----------------------------------------------------------------------------------------------
# Face detection
# ----------------------------------------------------------------------------------------------
_CASCADE_CACHE = {}
_YOLO_CACHE = {}


def _get_cascade(name: str):
    if name not in _CASCADE_CACHE:
        c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        _CASCADE_CACHE[name] = None if c.empty() else c
    return _CASCADE_CACHE[name]


def _nms(boxes, iou_thresh: float = 0.35, contain_thresh: float = 0.7):
    """boxes: list of (x, y, w, h, score). Larger boxes win; a box is dropped if it overlaps a kept box
    (IoU) or sits mostly inside one (fragments of a head cut by a sliding-window edge)."""
    boxes = sorted(boxes, key=lambda b: (b[2] * b[3], b[4]), reverse=True)
    keep = []
    for b in boxes:
        bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
        dup = False
        for k in keep:
            kx1, ky1, kx2, ky2 = k[0], k[1], k[0] + k[2], k[1] + k[3]
            iw = max(0, min(bx2, kx2) - max(bx1, kx1))
            ih = max(0, min(by2, ky2) - max(by1, ky1))
            inter = iw * ih
            union = b[2] * b[3] + k[2] * k[3] - inter
            if (union > 0 and inter / union > iou_thresh) or (b[2] * b[3] > 0 and inter / (b[2] * b[3]) > contain_thresh):
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep


_FACE_MASKS = {}  # (x, y, w, h) -> uint8 instance mask of exactly that box (SAM3 only)
_FACE_SYM = {}    # (x, y, w, h) -> left/right symmetry of that face's mask (1.0 = frontal, low = profile)


def mask_symmetry(inst: np.ndarray) -> float:
    """IoU of a mask with its own left/right mirror image inside its bounding box. A frontal face is nearly
    symmetric (about 0.85-1.0); a profile view is not (about 0.4-0.6)."""
    m = (inst > 0)
    f = m[:, ::-1]
    return float(np.logical_and(m, f).sum()) / float(max(np.logical_or(m, f).sum(), 1))


def _finalize(boxes):
    boxes = _nms(boxes)
    out = []
    for b in boxes:
        key = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        if len(b) > 5 and b[5] is not None:
            _FACE_MASKS[key] = b[5]
            _FACE_SYM[key] = mask_symmetry(b[5])
        elif len(b) > 6 and b[6] is not None:
            _FACE_SYM[key] = b[6]
        out.append(key)
    return sorted(out, key=lambda f: f[2] * f[3], reverse=True)


def detect_faces_opencv(image_tensor: torch.Tensor, confidence: float = 0.35):
    """Haar fallback. Human faces only. Runs on a downscaled, contrast-equalised copy
    (dark subjects are invisible to Haar otherwise), frontal + both profiles."""
    img = (image_tensor[0, ..., :3].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    H, W = img.shape[:2]
    s = min(1.0, 1600.0 / max(H, W))
    small = cv2.resize(img, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA) if s < 1.0 else img
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    min_n = int(np.clip(round(confidence * 20), 3, 12))
    min_size = max(24, int(min(gray.shape[:2]) * 0.04))

    found = []
    for name, flip in (("haarcascade_frontalface_default.xml", False),
                       ("haarcascade_frontalface_alt2.xml", False),
                       ("haarcascade_profileface.xml", False),
                       ("haarcascade_profileface.xml", True)):
        casc = _get_cascade(name)
        if casc is None:
            continue
        g = cv2.flip(gray, 1) if flip else gray
        rects = casc.detectMultiScale(g, scaleFactor=1.05, minNeighbors=min_n, minSize=(min_size, min_size))
        for (x, y, w, h) in rects:
            if flip:
                x = g.shape[1] - x - w
            if not (0.75 <= w / float(h) <= 1.33):
                continue
            found.append((x / s, y / s, w / s, h / s, 1.0))
    return _finalize(found)


def _find_yolo_model(model_path: str):
    candidates = []
    try:
        import folder_paths
        base = os.path.join(folder_paths.models_dir, "ultralytics")
    except Exception:
        base = None
    p = (model_path or "").strip().strip('"')
    if p.lower() == "none":
        p = ""
    if p:
        candidates.append(p)
        if base:
            for sub in ("bbox", "segm", ""):
                candidates.append(os.path.join(base, sub, p))
    elif base:
        import glob
        for sub in ("bbox", ""):
            candidates += sorted(glob.glob(os.path.join(base, sub, "*face*.pt")))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def detect_faces_yolo(image_tensor: torch.Tensor, model_path: str = "", confidence: float = 0.35,
                      window: int = 1024, overlap: float = 0.25):
    """Ultralytics YOLO detector. YOLO resizes its input to ~640px, so on a multi-thousand-pixel canvas
    small heads shrink to a few pixels and are missed. We therefore run one global pass (big heads) plus
    overlapping full-resolution windows (small heads) and merge the results."""
    path = _find_yolo_model(model_path)
    if path is None:
        return []
    try:
        from ultralytics import YOLO
        if path not in _YOLO_CACHE:
            _YOLO_CACHE[path] = YOLO(path)
        model = _YOLO_CACHE[path]
        img = (image_tensor[0, ..., :3].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        bgr = img[:, :, ::-1].copy()
        H, W = bgr.shape[:2]
        found = []

        def run(crop, ox, oy, imgsz, edges):
            res = model(crop, conf=float(confidence), imgsz=int(imgsz), verbose=False)[0]
            ch, cw = crop.shape[:2]
            for box, sc in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = [float(v) for v in box]
                # drop boxes cut by an interior window edge (the neighbouring window sees them whole)
                if (edges[0] and x1 < 3) or (edges[1] and y1 < 3) or (edges[2] and x2 > cw - 3) or (edges[3] and y2 > ch - 3):
                    continue
                found.append((ox + x1, oy + y1, x2 - x1, y2 - y1, float(sc)))

        run(bgr, 0, 0, 1280, (False, False, False, False))
        if max(H, W) > window * 1.25:
            step = max(64, int(window * (1.0 - overlap)))
            xs = sorted(set(list(range(0, max(1, W - window), step)) + [max(0, W - window)]))
            ys = sorted(set(list(range(0, max(1, H - window), step)) + [max(0, H - window)]))
            for y0 in ys:
                for x0 in xs:
                    crop = bgr[y0:y0 + window, x0:x0 + window]
                    edges = (x0 > 0, y0 > 0, x0 + crop.shape[1] < W, y0 + crop.shape[0] < H)
                    run(crop, x0, y0, window, edges)
        return _finalize(found)
    except Exception as e:
        logger.warning(f"[H3 Upscale] YOLO face detection failed ({e}); falling back.")
        return []


_SAM3_CACHE = {}
_SAM3_CKPT_URL = "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"  # same mirror the RMBG pack uses


def _find_sam3_bpe():
    """Locate bpe_simple_vocab_16e6.txt.gz. Different sam3 layouts (pip, vendored, RMBG pack) keep it in
    different places, so search around the imported package, then fall back to downloading it."""
    import glob
    name = "bpe_simple_vocab_16e6.txt.gz"
    roots = []
    try:
        import sam3 as _sam3
        pkg_dir = os.path.dirname(os.path.abspath(_sam3.__file__))
        roots += [pkg_dir, os.path.dirname(pkg_dir), os.path.dirname(os.path.dirname(pkg_dir))]
    except Exception:
        pass
    try:
        import folder_paths
        roots.append(os.path.join(folder_paths.models_dir, "sam3"))
        roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
    except Exception:
        pass
    for r in roots:
        for cand in (os.path.join(r, "assets", name), os.path.join(r, name)):
            if os.path.isfile(cand):
                return cand
    for r in roots[:2]:  # shallow recursive search inside the package itself
        hits = glob.glob(os.path.join(r, "**", name), recursive=True)
        if hits:
            return hits[0]
    # last resort: download the small vocab file next to the weights
    try:
        import folder_paths
        d = os.path.join(folder_paths.models_dir, "sam3", "assets")
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, name)
        if not os.path.isfile(dest):
            from torch.hub import download_url_to_file
            download_url_to_file("https://raw.githubusercontent.com/facebookresearch/sam3/main/sam3/assets/" + name, dest)
        return dest
    except Exception as e:
        logger.warning(f"[H3 Upscale] could not locate/download the SAM3 BPE vocab ({e})")
        return None


def _load_sam3_direct():
    """Load Meta's SAM3 image model directly (no RMBG pack). Needs the `sam3` python package:
    either pip-installed, or copied to custom_nodes/triXope/vendor/sam3. Weights = models/sam3/sam3.pt
    (auto-downloaded if missing). Returns a Sam3Processor or None."""
    if "proc" in _SAM3_CACHE:
        return _SAM3_CACHE["proc"]
    try:
        vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
        if os.path.isdir(vendor) and vendor not in sys.path:
            sys.path.insert(0, vendor)
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as e:
        logger.info(f"[H3 Upscale] direct SAM3 unavailable ({type(e).__name__}: {e})")
        return None
    try:
        import folder_paths
        import comfy.model_management as mm
        dev = mm.get_torch_device()
        device_str = "cuda" if dev.type == "cuda" else "cpu"
        ckpt = None
        try:
            ckpt = folder_paths.get_full_path("sam3", "sam3.pt")
        except Exception:
            pass
        if not ckpt or not os.path.isfile(ckpt):
            d = os.path.join(folder_paths.models_dir, "sam3")
            os.makedirs(d, exist_ok=True)
            ckpt = os.path.join(d, "sam3.pt")
            if not os.path.isfile(ckpt):
                logger.info(f"[H3 Upscale] downloading SAM3 weights to {ckpt} ...")
                from torch.hub import download_url_to_file
                download_url_to_file(_SAM3_CKPT_URL, ckpt)
        model = build_sam3_image_model(
            bpe_path=_find_sam3_bpe(), device=device_str, eval_mode=True, checkpoint_path=ckpt, load_from_HF=False,
            enable_segmentation=True, enable_inst_interactivity=False,
        )
        proc = Sam3Processor(model, device=device_str)
        _SAM3_CACHE["proc"] = proc
        _SAM3_CACHE["device"] = device_str
        return proc
    except Exception as e:
        logger.warning(f"[H3 Upscale] SAM3 failed to load directly ({type(e).__name__}: {e})")
        return None


def _free_sam3_direct():
    """Drop the direct SAM3 model so its VRAM goes back to the diffusion model."""
    _SAM3_CACHE.pop("proc", None)
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_sam3_node():
    """Fallback: reuse the RMBG pack's SAM3Segment node through ComfyUI's registry, if that pack is installed."""
    if "inst" not in _SAM3_CACHE:
        try:
            import nodes
            cls = nodes.NODE_CLASS_MAPPINGS.get("SAM3Segment")
        except Exception:
            cls = None
        _SAM3_CACHE["inst"] = cls() if cls is not None else None
    return _SAM3_CACHE["inst"]


def detect_faces_sam3(image_tensor: torch.Tensor, prompt: str = "head", confidence: float = 0.35,
                      window: int = 1536, overlap: float = 0.25):
    """Text-prompted detection with SAM3 ("head", "minotaur head", "face"...). Works on creatures.
    Backend 1: Meta's `sam3` package loaded directly. Backend 2: the RMBG pack's SAM3Segment node.
    SAM3 works at ~1000px internally, so we run one global pass plus overlapping windows for small heads.
    Returns None if neither backend is available."""
    from PIL import Image
    text = (prompt or "").strip() or "head"
    img = image_tensor[..., :3].contiguous()
    H, W = img.shape[1], img.shape[2]
    proc = _load_sam3_direct()
    node = None if proc is not None else _get_sam3_node()
    if proc is None and node is None:
        return None

    def crop_boxes(crop):
        """-> list of (x1, y1, x2, y2) in crop pixel coordinates."""
        if proc is not None:
            pil = Image.fromarray((crop[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8))
            use_amp = _SAM3_CACHE.get("device") == "cuda"
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
            with torch.inference_mode(), ctx:
                state = proc.set_image(pil)
                proc.reset_all_prompts(state)
                proc.set_confidence_threshold(float(confidence), state)
                state = proc.set_text_prompt(text, state)
            # Use the instance MASKS (what the RMBG node outputs, known good) and derive boxes from them,
            # rather than state["boxes"], which came out wrong with the pack's vendored sam3.
            masks = state.get("masks")
            if masks is None or masks.numel() == 0:
                return []
            m = masks.reshape(-1, masks.shape[-2], masks.shape[-1]) > 0
            out = []
            for mi in m:
                rows = torch.where(mi.any(dim=1))[0]
                cols = torch.where(mi.any(dim=0))[0]
                if rows.numel() == 0 or cols.numel() == 0:
                    continue
                r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
                out.append((float(c0), float(r0), float(c1), float(r1), mi[r0:r1, c0:c1].cpu().numpy().astype(np.uint8)))
            return out
        with torch.no_grad():
            _, masks, _ = node.segment(crop, text, "Auto", confidence_threshold=float(confidence),
                                       output_mode="Separate", background="Alpha", unload_model=False)
        out = []
        for m in masks:
            mb = m.float().cpu().numpy() > 0.5
            ys, xs = np.where(mb)
            if len(xs) >= 64:
                r0, r1, c0, c1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
                out.append((float(c0), float(r0), float(c1), float(r1), mb[r0:r1, c0:c1].astype(np.uint8)))
        return out

    found = []

    def run(x0, y0, x1, y1, edges):
        ch, cw = y1 - y0, x1 - x0
        for (bx1, by1, bx2, by2, bmask) in crop_boxes(img[:, y0:y1, x0:x1, :]):
            if (edges[0] and bx1 < 3) or (edges[1] and by1 < 3) or (edges[2] and bx2 > cw - 3) or (edges[3] and by2 > ch - 3):
                continue  # cut by an interior window edge; a neighbouring window sees it whole
            if max(bx2 - bx1, by2 - by1) > 0.85 * min(H, W):
                continue  # implausibly large for a head (whole-panel / whole-image blob)
            if bx2 - bx1 >= 16 and by2 - by1 >= 16:
                found.append((x0 + bx1, y0 + by1, bx2 - bx1, by2 - by1, 1.0, bmask))

    try:
        run(0, 0, W, H, (False, False, False, False))
        if max(H, W) > window * 1.25:
            step = max(64, int(window * (1.0 - overlap)))
            xs_ = sorted(set(list(range(0, max(1, W - window), step)) + [max(0, W - window)]))
            ys_ = sorted(set(list(range(0, max(1, H - window), step)) + [max(0, H - window)]))
            for y0 in ys_:
                for x0 in xs_:
                    x1, y1 = min(W, x0 + window), min(H, y0 + window)
                    run(x0, y0, x1, y1, (x0 > 0, y0 > 0, x1 < W, y1 < H))
    finally:
        if proc is not None:
            _free_sam3_direct()
    return _finalize(found)


def mask_to_boxes(mask: torch.Tensor, img_h: int, img_w: int, index: int = 0):
    """Each connected blob in a MASK becomes one 'face' (use for creature heads: SAM / GroundingDINO / hand-painted)."""
    if mask is None:
        return []
    m = mask
    if m.dim() == 4:
        m = m[..., 0]
    if m.dim() == 2:
        m = m.unsqueeze(0)
    m = m[min(index, m.shape[0] - 1)].float().cpu().numpy()
    if m.shape != (img_h, img_w):
        mh0, mw0 = m.shape
        if abs((mw0 / float(mh0)) - (img_w / float(img_h))) > 0.02:
            logger.warning(f"[H3 Upscale] face_mask aspect ({mw0}x{mh0}) differs from the image ({img_w}x{img_h}); it will be stretched to fit")
        logger.info(f"[H3 Upscale] face_mask {mw0}x{mh0} resized to the image size at this stage: {img_w}x{img_h}")
        m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_AREA if (mw0 > img_w or mh0 > img_h) else cv2.INTER_LINEAR)
    binary = (m > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w >= 16 and h >= 16 and area >= 256:
            out.append((float(x), float(y), float(w), float(h), 1.0, None, mask_symmetry(labels[y:y + h, x:x + w] == i)))
    return _finalize(out)


def parse_manual_boxes(text: str, img_h: int, img_w: int):
    """'x,y,w,h' per line or ';'-separated. Values <= 1.0 are fractions of the image, otherwise pixels."""
    out = []
    for chunk in re.split(r"[;\n]+", text or ""):
        nums = re.findall(r"-?\d+\.?\d*", chunk)
        if len(nums) < 4:
            continue
        x, y, w, h = [float(v) for v in nums[:4]]
        if max(x, y, w, h) <= 1.0:
            x, y, w, h = x * img_w, y * img_h, w * img_w, h * img_h
        x, y = max(0.0, x), max(0.0, y)
        w, h = min(w, img_w - x), min(h, img_h - y)
        if w >= 16 and h >= 16:
            out.append((x, y, w, h, 1.0))
    return _finalize(out)


def detect_faces(canvas: torch.Tensor, detector: str = "auto", model_path: str = "", confidence: float = 0.35,
                 face_mask=None, mask_index: int = 0, manual_boxes: str = "", sam3_prompt: str = "head"):
    """Priority: explicit MASK > manual boxes > SAM3 (if selected) > YOLO > Haar. Sorted largest -> smallest."""
    H, W = canvas.shape[1], canvas.shape[2]
    _FACE_MASKS.clear()
    _FACE_SYM.clear()
    if face_mask is not None:
        faces = mask_to_boxes(face_mask, H, W, mask_index)
        if faces:
            return faces, "mask"
    if manual_boxes and manual_boxes.strip():
        faces = parse_manual_boxes(manual_boxes, H, W)
        if faces:
            return faces, "manual"
    if detector == "sam3":
        faces = detect_faces_sam3(canvas, sam3_prompt, confidence)
        if faces is None:
            logger.warning("[H3 Upscale] SAM3 unavailable (install the `sam3` package or the RMBG pack); falling back to auto.")
            detector = "auto"
        else:
            return faces, "sam3"
    if detector in ("auto", "yolo"):
        faces = detect_faces_yolo(canvas, model_path, confidence)
        if faces:
            return faces, "yolo"
        if detector == "yolo":
            return [], "yolo"
    return detect_faces_opencv(canvas, confidence), "haar"


def expand_square_box(fx, fy, fw, fh, pad: float, img_w: int, img_h: int):
    """Padded SQUARE crop around the box (no aspect distortion), shifted to stay inside the image."""
    side = int(round(max(fw, fh) * (1.0 + 2.0 * pad)))
    side = max(32, min(side, img_w, img_h))
    cx, cy = fx + fw / 2.0, fy + fh / 2.0
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x1 = max(0, min(x1, img_w - side))
    y1 = max(0, min(y1, img_h - side))
    return x1, y1, x1 + side, y1 + side


def expand_box(fx, fy, fw, fh, pad: float, img_w: int, img_h: int):
    """Padded RECTANGULAR crop: pad is a fraction of the box size on each side, so pad = 0 is the exact box."""
    px, py = fw * pad, fh * pad
    x1, y1 = int(max(0, np.floor(fx - px))), int(max(0, np.floor(fy - py)))
    x2, y2 = int(min(img_w, np.ceil(fx + fw + px))), int(min(img_h, np.ceil(fy + fh + py)))
    if x2 - x1 < 64:
        cx = (x1 + x2) // 2
        x1, x2 = max(0, cx - 32), min(img_w, cx + 32)
    if y2 - y1 < 64:
        cy = (y1 + y2) // 2
        y1, y2 = max(0, cy - 32), min(img_h, cy + 32)
    return x1, y1, x2, y2


# Longest face-crop sequence diffused in a single H3 temporal pass. H3 is trained on roughly
# 124 frames and a long shot at a large face_size would otherwise build one enormous face latent,
# so each track is split into chunks of at most this many crops (fewer when the crop is large).
FACE_SEQUENCE_CHUNK = 48
FACE_SEQUENCE_PIXEL_BUDGET = 64_000_000


def _face_box_iou(a, b) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = [float(v) for v in a[:4]]
    bx, by, bw, bh = [float(v) for v in b[:4]]
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def track_face_boxes(per_frame_faces, max_gap: int = 2, min_score: float = 0.15):
    """Link per-frame detections into temporal tracks.

    `per_frame_faces` is one entry per frame, each a list of (x, y, w, h) boxes. Returns a list of
    {frame_index: box} dicts, largest mean face first. Matching is IoU first with a centre-distance
    fallback relative to the face's own size, so detector jitter and a fast head turn keep one
    identity on one track; a track survives up to `max_gap` consecutively missed detections.
    """
    tracks = []
    for frame_idx, faces in enumerate(per_frame_faces):
        pairs = []
        for track_index, track in enumerate(tracks):
            last_frame = max(track)
            if frame_idx - last_frame > max_gap:
                continue
            last = track[last_frame]
            lcx, lcy = last[0] + last[2] / 2.0, last[1] + last[3] / 2.0
            scale = max(8.0, 0.5 * (last[2] + last[3]))
            for face_index, face in enumerate(faces):
                iou = _face_box_iou(face, last)
                fcx, fcy = face[0] + face[2] / 2.0, face[1] + face[3] / 2.0
                distance = ((lcx - fcx) ** 2 + (lcy - fcy) ** 2) ** 0.5
                if iou < 0.05 and distance > scale:
                    continue
                pairs.append((max(iou, 1.0 - distance / (scale * 1.5)), track_index, face_index))
        pairs.sort(reverse=True)
        used_tracks, used_faces = set(), set()
        for score, track_index, face_index in pairs:
            if score < min_score or track_index in used_tracks or face_index in used_faces:
                continue
            tracks[track_index][frame_idx] = faces[face_index]
            used_tracks.add(track_index)
            used_faces.add(face_index)
        for face_index, face in enumerate(faces):
            if face_index not in used_faces:
                tracks.append({frame_idx: face})
    tracks.sort(key=lambda tr: float(np.mean([b[2] * b[3] for b in tr.values()])), reverse=True)
    return tracks


def largest_alpha_component(alpha: np.ndarray):
    """alpha float HxW -> (x, y, w, h, instance mask uint8 of that box) of the largest opaque blob, or None."""
    binary = (alpha > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = [int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
    if w < 8 or h < 8:
        return None
    return x, y, w, h, (labels[y:y + h, x:x + w] == i).astype(np.uint8)


def reference_cutout_alpha_np(inst: np.ndarray, fx: int, fy: int, x1: int, y1: int, ch: int, cw: int,
                              erode_frac: float = 0.02, feather_frac: float = 0.010) -> np.ndarray:
    """Face cut-out alpha for the reference: the face's own mask placed in the crop, eroded a hair so no
    background fringe survives, then softly feathered."""
    full = np.zeros((ch, cw), np.float32)
    mh, mw = inst.shape
    ox, oy = fx - x1, fy - y1
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    cw_, ch_ = min(mw - sx0, cw - dx0), min(mh - sy0, ch - dy0)
    if cw_ > 0 and ch_ > 0:
        full[dy0:dy0 + ch_, dx0:dx0 + cw_] = inst[sy0:sy0 + ch_, sx0:sx0 + cw_]
    k = max(1, int(erode_frac * max(ch, cw)))
    full = cv2.erode(full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    full = cv2.GaussianBlur(full, (0, 0), sigmaX=max(1.0, feather_frac * max(ch, cw)))
    return np.clip(full, 0.0, 1.0)


def prepare_reference_face(ref: torch.Tensor, pad: float, detector: str, model_path: str, confidence: float, sam3_prompt: str = "head"):
    """Reference image of ANY size/layout -> (tight face crop, cut-out alpha, aspect, face box in the crop).
    1. If the reference already has an alpha channel (e.g. SAM3 'Alpha' output) that cut-out is used as is.
    2. Otherwise the largest face is found (same detector as the main image) and cut out with its own mask.
    The crop is tight (6% margin) and the alpha is eroded+feathered, so nothing but the face reaches the blend.
    Because the master is later warped face-box-to-face-box for every target face, its original size is irrelevant."""
    r = ref[0:1, :, :, :3].clone().float()
    h, w = r.shape[1], r.shape[2]
    margin = 0.06
    found = None            # (fx, fy, fw, fh, instance mask or None)
    if ref.shape[-1] >= 4:
        a_in = ref[0, :, :, 3].float().cpu().numpy()
        if a_in.max() > 0.5 and (a_in < 0.5).any():          # a genuine cut-out, not a fully opaque alpha
            comp = largest_alpha_component(a_in)
            if comp is not None:
                found = comp
                logger.info("[H3 Upscale] reference_face: using its alpha channel as the cut-out")
    if found is None:
        try:
            faces, _ = detect_faces(r, detector, model_path, confidence, sam3_prompt=sam3_prompt)
        except Exception:
            faces = []
        if faces:
            fx, fy, fw, fh = faces[0]
            found = (fx, fy, fw, fh, _FACE_MASKS.get(faces[0]))
    if found is not None:
        fx, fy, fw, fh, inst = found
        x1, y1, x2, y2 = expand_box(fx, fy, fw, fh, margin, w, h)
        ch, cw = y2 - y1, x2 - x1
        if inst is not None:
            al = reference_cutout_alpha_np(inst, fx, fy, x1, y1, ch, cw)
            alpha = torch.from_numpy(al).view(1, ch, cw, 1).to(r.device)
        else:
            alpha = create_face_blend_mask(ch, cw, feather=0.5, device=r.device)
        logger.info(f"[H3 Upscale] reference_face {w}x{h}: tight face crop {cw}x{ch}, cut-out={'mask' if inst is not None else 'ellipse'}")
        return r[:, y1:y2, x1:x2, :], alpha, fw / float(max(fh, 1)), (fx - x1, fy - y1, fw, fh)
    side = min(h, w)
    y1, x1 = (h - side) // 2, (w - side) // 2
    alpha = create_face_blend_mask(side, side, feather=0.5, device=r.device)
    logger.warning("[H3 Upscale] reference_face: no face found, using the centre square")
    return r[:, y1:y1 + side, x1:x1 + side, :], alpha, None, None


def draw_face_debug(canvas: torch.Tensor, boxes, label: str, thumb=None) -> torch.Tensor:
    img = (canvas[0, ..., :3].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8).copy()
    th = max(2, int(min(img.shape[:2]) * 0.003))
    for i, (x1, y1, x2, y2, rx1, ry1, rx2, ry2) in enumerate(boxes):
        col = (0, 255, 0) if i == 0 else (255, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, th)                      # padded square crop that gets diffused
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 255, 255), max(1, th // 2))  # raw detection box
        cv2.putText(img, f"{i}:{label}", (x1 + th * 2, y1 + th * 8), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.6, th * 0.35), col, th, cv2.LINE_AA)
    if thumb is not None:  # show the master/reference face that is being applied (bottom-left)
        t = (thumb[0, ..., :3].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        side = min(256, img.shape[0] // 4, img.shape[1] // 4)
        t = cv2.resize(t, (side, side), interpolation=cv2.INTER_AREA)
        y0, x0 = img.shape[0] - side - 12, 12
        cv2.rectangle(img, (x0 - 3, y0 - 3), (x0 + side + 3, y0 + side + 3), (255, 0, 255), 3)
        img[y0:y0 + side, x0:x0 + side] = t
    return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).to(canvas.device)

class UltimateSDUpscaleMiniMaxH3:
    @classmethod
    def INPUT_TYPES(s):
        samplers = getattr(comfy.samplers, "SAMPLERS", getattr(getattr(comfy.samplers, "KSampler", None), "SAMPLERS", []))
        schedulers = getattr(comfy.samplers, "SCHEDULERS", getattr(getattr(comfy.samplers, "KSampler", None), "SCHEDULERS", []))

        yolo_models = ["None"]
        try:
            _bbox_dir = os.path.join(folder_paths.models_dir, "ultralytics", "bbox")
            if os.path.isdir(_bbox_dir):
                _found = []
                for _root, _dirs, _files in os.walk(_bbox_dir):
                    for _f in _files:
                        if _f.lower().endswith(".pt"):
                            _found.append(os.path.relpath(os.path.join(_root, _f), _bbox_dir).replace("\\", "/"))
                yolo_models += sorted(_found, key=str.lower)
        except Exception:
            pass
        lut_files = ["None"]
        try:
            raw_files = folder_paths.get_filename_list("luts")
            if raw_files:
                lut_files.extend([f for f in raw_files if f.lower().endswith(".cube")])
        except Exception:
            pass
        return {
            "required": {
                "grp_h3_upscale": ("BOOLEAN", {"default": False, "tooltip": "Group header. Click the bar to expand/collapse."}),
                "image": ("IMAGE", {"tooltip": "Single images keep the normal image upscale path. Batches with multiple images are treated as ordered video frames and H3 processes matching tiles as one temporal sequence."}),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "enabled_h3_upscale": ("BOOLEAN", {"default": True, "tooltip": "Master switch for the H3 Upscale group. Off = the group is bypassed: no resize and no tile diffusion; the image goes straight to the next stage."}),
                "upscale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.05}),
                "tile_width": ("INT", {"default": 8192, "min": 128, "max": 8192, "step": 16}),
                "tile_height": ("INT", {"default": 8192, "min": 128, "max": 8192, "step": 16}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 16}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": 256, "step": 16}),
                "seam_fix_denoise": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": 128, "step": 16}),
                "seed": ("INT", {"default": 45, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (samplers, {"default": "euler" if "euler" in samplers else samplers[0]}),
                "scheduler": (schedulers, {"default": "beta" if "beta" in schedulers else schedulers[0]}),
                "denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
                "diffusion_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Opacity of the tile diffusion result over the clean upscale. Lower = less of the model's texture. (denoise = 0 skips the tile pass entirely.)"}),
                "edge_protect": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Restore the clean source along strong silhouette edges (removes jagged/hatched contours)."}),
                "detail_retention": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Restores fine detail wherever diffusion made a region smoother than the upscaled source. 0 = off."}),
                "latent_noise_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.005, "tooltip": "Extra noise added to tile latents. The old hard-coded 0.03 caused a dotted/lattice texture; keep at 0 unless you want it."}),
                "redraw_only": ("BOOLEAN", {"default": True, "tooltip": "Show ONLY the model's redraw: bypasses diffusion_mix, edge_protect, detail_retention, tone_lock and chroma_lock so no part of the source image can ghost through."}),
                "align_to_source": ("BOOLEAN", {"default": False, "tooltip": "Register each redrawn tile onto the source geometry (small clamped warp) so a few pixels of model drift cannot show up as a double image."}),
                "locks_follow_denoise": ("BOOLEAN", {"default": True, "tooltip": "Fade the 'keep the source' corrections (detail restore, edge protect, tone/chroma lock) out as denoise rises: full strength up to 0.25, off from 0.45. Prevents the ghosted second copy of a subject the model has re-drawn."}),
                "grp_face_detail": ("BOOLEAN", {"default": False, "tooltip": "Group header. Click the bar to expand/collapse."}),
                "enable_face_detailer": ("BOOLEAN", {"default": True, "tooltip": "Enable AI Face Detection & Inpaint Pass."}),
                "reference_face": ("IMAGE", {"tooltip": "Optional master reference portrait. When connected, EVERY detected face (largest first) is re-rendered toward this face."}),
                "face_mask": ("MASK", {"tooltip": "Optional. Each white blob = one face/head. Best option for creatures (SAM / GroundingDINO / hand-painted). Overrides auto-detection."}),
                "face_denoise": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.60, "step": 0.01, "tooltip": "Diffusion strength for face inpainting. Raise to ~0.40-0.50 when using reference_face so the identity is fully re-rendered."}),
                "face_padding": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Context added on each side of the head box, as a fraction of the box size (0 = exact box, rectangular crop)."}),
                "face_size": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64, "tooltip": "Minimum working resolution for the face diffusion (larger faces run at native size, up to 2048)."}),
                "face_detector": (["auto", "yolo", "haar", "sam3"], {"default": "sam3", "tooltip": "auto = YOLO if a model is found, else OpenCV Haar (human faces only). sam3 = text-prompted detection through the RMBG pack's SAM3Segment node (works on creatures)."}),
                "face_detector_model": (yolo_models, {"default": "None", "tooltip": "YOLO model from models/ultralytics/bbox (scanned automatically; refresh the node list after adding files). None = use the first *face*.pt found. Only used by the yolo / auto detectors."}),
                "face_confidence": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05, "tooltip": "Detection threshold. Raise it if you get false positives (Haar: stricter neighbours)."}),
                "manual_face_boxes": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional manual boxes, one 'x,y,w,h' per line. Values <=1 are fractions of the image (e.g. 0.12,0.05,0.10,0.14), otherwise pixels. Overrides auto-detection."}),
                "sam3_prompt": ("STRING", {"default": "face", "tooltip": "Text prompt for face_detector = sam3, e.g. 'head', 'face', 'minotaur head'."}),
                "reference_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How much of the master face is mixed into each face crop before diffusion harmonises it."}),
                "reference_pose_match": ("BOOLEAN", {"default": True, "tooltip": "Fade the reference/master mix out on faces whose shape differs from it (profile, back views) to avoid ghosting."}),
                "reference_flow_align": ("BOOLEAN", {"default": True, "tooltip": "Optical-flow register the reference onto each face (eyes/nose/creases) before blending. Removes double-edge ghosting."}),
                "face_structure_lock": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "1 = the face keeps its structure and only gains fine detail from the diffusion pass (never blurrier); 0 = fully use the diffused result."}),
                "face_chroma_lock": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Keep each face's original colour and take only luminance detail from the redraw. Prevents tint / hue steps (blue or orange patches)."}),
                "face_seamless_blend": ("BOOLEAN", {"default": True, "tooltip": "Blend the redrawn face into the image with Poisson (seamless) cloning so colour and brightness match the surroundings along the whole edge. Off = plain feathered alpha blend."}),
                "frontal_only_threshold": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Skip faces whose mask is not left/right symmetric, i.e. profile / side views. 0 = off. About 0.75 keeps frontal faces and drops profiles (needs SAM3 or a face_mask; the console logs each face's score)."}),
                "grp_refine": ("BOOLEAN", {"default": False, "tooltip": "Group header. Click the bar to expand/collapse."}),
                "enable_refiner": ("BOOLEAN", {"default": True, "tooltip": "Enable post-upscale edge & texture refinement."}),
                "refine_amount": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Strength of texture detail enhancement."}),
                "refine_radius": ("INT", {"default": 2, "min": 1, "max": 10, "tooltip": "Pixel radius for edge sharpening (scales with resolution)."}),
                "grp_dlss5": ("BOOLEAN", {"default": False, "tooltip": "Group header. Click the bar to expand/collapse."}),
                "enable_dlss5": ("BOOLEAN", {"default": True, "tooltip": "Enable DLSS 5 Neural Rendering post-pass."}),
                "upscaling_mode": (UPSCALING_LABELS, {"default": UPSCALING_LABELS[0]}),
                "nr_preset": (list(NR_PRESETS), {"default": "Default"}),
                "nr_style": (list(NR_STYLES), {"default": "Default"}),
                "nr_intensity": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "local_tone_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "local_structure_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "skin_structure_strength": ("FLOAT", {"default": 2.0, "min": -1.0, "max": 2.0, "step": 0.01}),
                "automatic_mask": ("BOOLEAN", {"default": True}),
                "dlss_model_preset": (list(DLSS_MODEL_PRESETS), {"default": "M"}),
                "motion": (MOTION_MODES, {"default": "auto"}),
                "scene_change_threshold": ("FLOAT", {"default": 0.24, "min": 0.01, "max": 1.0, "step": 0.01}),
                "warmup_frames": ("INT", {"default": 0, "min": 0, "max": 16}),
                "runtime_dir": ("STRING", {"default": ""}),
                "verify_neural_rendering": ("BOOLEAN", {"default": True}),
                "grp_cfx_main": ("BOOLEAN", {"default": False, "tooltip": "Group header. Click the bar to expand/collapse."}),
                "enable_colorfx": ("BOOLEAN", {"default": True, "tooltip": "Master switch for the Color FX stage (last stage of the node)."}),
                "enable_color_correction": ("BOOLEAN", {"default": True}),
                "enable_lut_processing": ("BOOLEAN", {"default": True}),
                "enable_enhancements": ("BOOLEAN", {"default": True}),
                "enable_blur_effects": ("BOOLEAN", {"default": True}),
                "enable_stylistic_effects": ("BOOLEAN", {"default": True}),
                "grp_cfx_color": ("BOOLEAN", {"default": False, "tooltip": "Group header. Click the bar to expand/collapse."}),
                "tone_lock": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Last step. Pins the final image's broad tone / black level to the clean pre-diffusion upscale, undoing any wash-out from the sampler or DLSS. Fine detail is untouched. 0 = off."}),
                "chroma_lock": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Last step. Takes colour from the clean pre-diffusion upscale and keeps the final luminance. Removes the purple/magenta edge fringe. Face regions are left alone. 0 = off."}),
                "hdr_intensity": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 3.0, "step": 0.01}),
                "shadow_intensity": ("FLOAT", {"default": 0.10, "min": 0.00, "max": 2.00, "step": 0.01}),
                "highlight_intensity": ("FLOAT", {"default": 0.20, "min": 0.00, "max": 2.00, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.00, "min": 0.10, "max": 5.00, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.90, "min": 0.00, "max": 3.00, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 3.00, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 0.90, "min": 0.00, "max": 3.00, "step": 0.01}),
                "enhance_color": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 3.00, "step": 0.01}),
                "grp_cfx_lut": ("BOOLEAN", {"default": False, "tooltip": "Group header. Click the bar to expand/collapse."}),
                "lut_name": (lut_files, {"default": "None"}),
                "lut_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lut_log_process": ("BOOLEAN", {"default": True}),
                "grp_cfx_enhancements": ("BOOLEAN", {"default": False, "tooltip": "Group header. Click the bar to expand/collapse."}),
                "sharpness": ("FLOAT", {"default": 3.0, "min": -2.0, "max": 5.0, "step": 0.1}),
                "edge_enhance_strength": ("FLOAT", {"default": 0.10, "min": 0.00, "max": 1.00, "step": 0.01}),
                "detail_enhance_strength": ("FLOAT", {"default": 0.20, "min": 0.00, "max": 1.00, "step": 0.01}),
                "grp_cfx_blur": ("BOOLEAN", {"default": False, "tooltip": "Group header. Click the bar to expand/collapse."}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "gaussian_blur_radius": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 64.0, "step": 0.1}),
                "radial_blur_strength": ("FLOAT", {"default": 32.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "radial_blur_center_x": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "radial_blur_center_y": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "radial_blur_focus_spread": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "radial_blur_steps": ("INT", {"default": 10, "min": 1, "max": 32, "step": 1}),
                "grp_cfx_stylistic": ("BOOLEAN", {"default": False, "tooltip": "Group header. Click the bar to expand/collapse."}),
                "chromatic_aberration_r_x": ("FLOAT", {"default": 2.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_aberration_r_y": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_aberration_b_x": ("FLOAT", {"default": -2.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_aberration_b_y": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_blur_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "simple_film_grain_intensity": ("FLOAT", {"default": 0.02, "min": 0.00, "max": 1.00, "step": 0.01}),
                "simple_film_grain_monochrome": ("BOOLEAN", {"default": True}),
                "scanline_intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0}),
                "vignette_intensity": ("FLOAT", {"default": 0.40, "min": 0.00, "max": 1.00, "step": 0.01}),
                "vignette_center_x": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "vignette_center_y": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "soft_light_opacity": ("FLOAT", {"default": 0.30, "min": 0.00, "max": 1.00, "step": 0.01}),
                "soft_light_blur_radius": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "face_debug")
    FUNCTION = "upscale"
    CATEGORY = "triXope/film"

    # ------------------------------------------------------------------------------------------
    def _read_cube_file(self, filepath):
        lines = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'): lines.append(line)
        lut_size = 0
        domain_min, domain_max = np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)
        table_data_lines = []
        for line in lines:
            if line.startswith('LUT_3D_SIZE'): lut_size = int(line.split()[-1])
            elif line.startswith('DOMAIN_MIN'): domain_min = np.array([float(x) for x in line.split()[1:]], dtype=np.float32)
            elif line.startswith('DOMAIN_MAX'): domain_max = np.array([float(x) for x in line.split()[1:]], dtype=np.float32)
            elif len(line.split()) == 3 and not line.startswith('TITLE'): table_data_lines.append(list(map(float, line.split())))
        return np.array(table_data_lines, dtype=np.float32).reshape(lut_size, lut_size, lut_size, 3), lut_size, domain_min, domain_max

    def _apply_lut_to_image_scipy(self, image_np_rgb_0_1, lut_data):
        lut_table, lut_size, _, _ = lut_data
        r_axis = g_axis = b_axis = np.linspace(0, 1, lut_size)
        interp_R = RegularGridInterpolator((b_axis, g_axis, r_axis), lut_table[..., 0], bounds_error=False)
        interp_G = RegularGridInterpolator((b_axis, g_axis, r_axis), lut_table[..., 1], bounds_error=False)
        interp_B = RegularGridInterpolator((b_axis, g_axis, r_axis), lut_table[..., 2], bounds_error=False)
        pts = image_np_rgb_0_1[..., [2, 1, 0]].reshape(-1, 3)
        return np.clip(np.stack([interp_R(pts), interp_G(pts), interp_B(pts)], axis=-1).reshape(image_np_rgb_0_1.shape), 0.0, 1.0)

    def _apply_lut_effect(self, pil_image, lut_name, strength, log_process):
        if strength == 0.0 or lut_name == "None": return pil_image
        lut_path = folder_paths.get_full_path("luts", lut_name)
        if not lut_path or not SCIPY_AVAILABLE: return pil_image
        lut_data = self._read_cube_file(lut_path)
        img_np = np.array(pil_image, dtype=np.float32) / 255.0
        orig = img_np.copy()
        if log_process: img_np = np.power(np.clip(img_np, 1e-5, 1.0), 1.0 / 2.2)
        out_np = self._apply_lut_to_image_scipy(img_np, lut_data)
        if log_process: out_np = np.power(np.clip(out_np, 0.0, 1.0), 2.2)
        blended = np.clip((1.0 - strength) * orig + strength * out_np, 0.0, 1.0)
        return Image.fromarray((blended * 255).astype(np.uint8))

    def _apply_shadows_highlights(self, pil_image, shadow_adj, highlight_adj, hdr_intensity):
        if shadow_adj == 0.0 and highlight_adj == 0.0 and hdr_intensity == 1.0: return pil_image
        img_array = np.array(pil_image, dtype=np.float32) / 255.0
        if hdr_intensity != 1.0:
            mean = np.mean(img_array)
            img_array = mean + (img_array - mean) * (1.0 + (hdr_intensity - 1.0) * 0.5) 
        lum = np.clip(0.299 * img_array[..., 0] + 0.587 * img_array[..., 1] + 0.114 * img_array[..., 2], 0.0, 1.0)[..., np.newaxis]
        if shadow_adj > 0.0: img_array += (1.0 - lum) * shadow_adj
        if highlight_adj > 0.0: img_array -= lum * highlight_adj
        return Image.fromarray((np.clip(img_array, 0.0, 1.0) * 255).astype(np.uint8))

    def _apply_color_enhancements(self, pil_image, b, c, s, ec):
        if b != 1.: pil_image = ImageEnhance.Brightness(pil_image).enhance(b)
        if c != 1.: pil_image = ImageEnhance.Contrast(pil_image).enhance(c)
        if s * ec != 1.: pil_image = ImageEnhance.Color(pil_image).enhance(s * ec)
        return pil_image

    def _apply_sharpness_detail(self, pil_image, sh, ee, de):
        if sh != 0.: pil_image = ImageEnhance.Sharpness(pil_image).enhance(1. + sh)
        if ee > 0.: pil_image = Image.blend(pil_image, pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE), ee)
        if de > 0.: pil_image = Image.blend(pil_image, pil_image.filter(ImageFilter.DETAIL), de)
        return pil_image

    def _apply_blurs(self, pil_image, br, gbr):
        if br > 0: pil_image = pil_image.filter(ImageFilter.BoxBlur(br))
        if gbr > 0.: pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=gbr))
        return pil_image

    def _apply_chromatic_aberration(self, pil_image, rx, ry, bx, by, blur):
        img = np.array(pil_image).astype(np.float32)
        rc, gc, bc = img[...,0], img[...,1], img[...,2]
        if blur > 0.: 
            bk = int(blur * 2) * 2 + 1
            rc, bc = cv2.GaussianBlur(rc, (bk, bk), blur), cv2.GaussianBlur(bc, (bk, bk), blur)
        rs = cv2.warpAffine(rc, np.float32([[1, 0, rx], [0, 1, ry]]), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
        bs = cv2.warpAffine(bc, np.float32([[1, 0, bx], [0, 1, by]]), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
        return Image.fromarray(np.clip(cv2.merge([rs, gc, bs]), 0, 255).astype(np.uint8))

    def _apply_radial_blur(self, pil_image, strength, cx_r, cy_r, spread, steps):
        img = np.array(pil_image).astype(np.float32) / 255.
        h, w = img.shape[:2]; cx, cy = int(w * cx_r), int(h * cy_r)
        max_d = np.sqrt(max(cx**2+cy**2, (w-cx)**2+cy**2, cx**2+(h-cy)**2, (w-cx)**2+(h-cy)**2))
        X, Y = np.meshgrid(np.arange(w) - cx, np.arange(h) - cy)
        rad_mask = np.sqrt(X**2 + Y**2) / max(1., max_d)
        blurred, cur_blur = [], strength
        for _ in range(steps):
            k = int(cur_blur)
            blurred.append(cv2.GaussianBlur(img, (k | 1, k | 1), 0) if k > 0 else img.copy())
            cur_blur /= spread
        final = np.zeros_like(img)
        for i in range(steps):
            m_i = np.dstack([np.clip((rad_mask - i / steps) * steps, 0, 1)] * 3)
            final = blurred[steps - 1 - i] * m_i + (final if i > 0 else img) * (1. - m_i)
        return Image.fromarray(np.clip(final * 255, 0, 255).astype(np.uint8))

    def _apply_scanlines(self, pil_image, intensity):
        img = np.array(pil_image)
        scan = np.ones(img.shape[:2], dtype=img.dtype)
        scan[::2, :] = 1. - intensity
        return Image.fromarray((img * np.dstack([scan]*3)).astype(np.uint8))

    def _apply_soft_light(self, pil_image, opacity, blur_radius):
        blur = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return Image.blend(pil_image, ImageChops.soft_light(pil_image, blur), opacity)

    def _apply_gamma_torch(self, tensor_images, gamma_value):
        if gamma_value == 1.0: return tensor_images
        return torch.pow(tensor_images.clamp(1e-5, 1.0), 1.0 / max(gamma_value, 0.01))

    def _apply_simple_film_grain_torch(self, tensor_images, intensity, monochrome):
        if intensity == 0.: return tensor_images
        B, H, W, C = tensor_images.shape
        noise = torch.randn((B, H, W, 1 if monochrome else C), device=tensor_images.device) * intensity
        if monochrome and C > 1: noise = noise.repeat(1, 1, 1, C)
        return (tensor_images + noise).clamp(0.0, 1.0)

    def _apply_vignette_torch(self, tensor_images, intensity, cx_r, cy_r):
        if intensity == 0.: return tensor_images
        B, H, W, C = tensor_images.shape
        cx, cy = W * cx_r, H * cy_r
        Y, X = torch.meshgrid(torch.arange(H, device=tensor_images.device) - cy, torch.arange(W, device=tensor_images.device) - cx, indexing='ij')
        dist = torch.sqrt(X**2 + Y**2)
        max_dist = torch.sqrt(torch.tensor(max(cx**2+cy**2, (W-cx)**2+cy**2, cx**2+(H-cy)**2, (W-cx)**2+(H-cy)**2), device=tensor_images.device))
        v_mask = (1.0 - (dist / max(1.0, max_dist))**2 * intensity).clamp(0.0, 1.0).unsqueeze(0).unsqueeze(-1)
        return (tensor_images * v_mask).clamp(0.0, 1.0)

    def _apply_color_fx(self, images, p):
        """Color FX stage. Same order and maths as FilmAuteur_LTX: shadows/highlights/HDR -> brightness/contrast/
        colour -> LUT -> sharpness/detail -> blurs -> chromatic aberration/scanlines/soft light (PIL), then
        gamma -> film grain -> vignette (tensor). images: [B,H,W,C] float 0..1."""
        alpha = images[..., 3:] if images.shape[-1] > 3 else None
        processed = []
        for i in range(images.shape[0]):
            pil_image = tensor2pil(images[i:i + 1, ..., :3]).convert("RGB")
            if p["enable_color_correction"]:
                pil_image = self._apply_shadows_highlights(pil_image, p["shadow_intensity"], p["highlight_intensity"], p["hdr_intensity"])
                pil_image = self._apply_color_enhancements(pil_image, p["brightness"], p["contrast"], p["saturation"], p["enhance_color"])
            if p["enable_lut_processing"] and p["lut_name"] != "None":
                pil_image = self._apply_lut_effect(pil_image, p["lut_name"], p["lut_strength"], p["lut_log_process"])
            if p["enable_enhancements"]:
                pil_image = self._apply_sharpness_detail(pil_image, p["sharpness"], p["edge_enhance_strength"], p["detail_enhance_strength"])
            if p["enable_blur_effects"]:
                pil_image = self._apply_blurs(pil_image, p["blur_radius"], p["gaussian_blur_radius"])
                if p["radial_blur_strength"] > 0.0:
                    pil_image = self._apply_radial_blur(pil_image, p["radial_blur_strength"], p["radial_blur_center_x"], p["radial_blur_center_y"], p["radial_blur_focus_spread"], p["radial_blur_steps"])
            if p["enable_stylistic_effects"]:
                ca = [p["chromatic_aberration_r_x"], p["chromatic_aberration_r_y"], p["chromatic_aberration_b_x"], p["chromatic_aberration_b_y"]]
                needs_pil = any(v != 0 for v in ca) or p["chromatic_blur_amount"] > 0 or p["scanline_intensity"] > 0.0 or p["soft_light_opacity"] > 0.0
                if needs_pil:
                    if any(v != 0 for v in ca) or p["chromatic_blur_amount"] > 0:
                        pil_image = self._apply_chromatic_aberration(pil_image, ca[0], ca[1], ca[2], ca[3], p["chromatic_blur_amount"])
                    if p["scanline_intensity"] > 0.0:
                        pil_image = self._apply_scanlines(pil_image, p["scanline_intensity"])
                    if p["soft_light_opacity"] > 0.0:
                        pil_image = self._apply_soft_light(pil_image, p["soft_light_opacity"], p["soft_light_blur_radius"])
            processed.append(pil2tensor(pil_image))
        out = torch.cat(processed, dim=0).to(images.device)
        if p["enable_color_correction"] and p["gamma"] != 1.0:
            out = self._apply_gamma_torch(out, p["gamma"])
        if p["enable_stylistic_effects"]:
            if p["simple_film_grain_intensity"] > 0.0:
                out = self._apply_simple_film_grain_torch(out, p["simple_film_grain_intensity"], p["simple_film_grain_monochrome"])
            if p["vignette_intensity"] > 0.0:
                out = self._apply_vignette_torch(out, p["vignette_intensity"], p["vignette_center_x"], p["vignette_center_y"])
        if alpha is not None:
            out = torch.cat([out, alpha.to(out.device, out.dtype)], dim=-1)
        return out

    def _diffuse(self, vae, model, positive, negative, image, steps, cfg, sampler_name, scheduler,
                 denoise, seed, noise_offset=0.0):
        """encode -> (optional noise offset) -> sample -> decode. Returns [B,H,W,3] on the image's device."""
        seed = int(seed) & 0xffffffffffffffff
        latent = vae.encode(image[..., :3])
        latent_samples = latent["samples"] if isinstance(latent, dict) else latent

        # Reshape 4D latents to 5D [B, C, T=1, H_lat, W_lat] for MiniMax H3
        if latent_samples.dim() == 4:
            latent_samples = latent_samples.unsqueeze(2)

        if noise_offset > 0.0:
            latent_samples = latent_samples + torch.randn_like(latent_samples) * noise_offset

        # Wrap the video latent with a dummy audio stream for MiniMax H3
        if not getattr(latent_samples, "is_nested", False) and not isinstance(latent_samples, (list, tuple)):
            b_size = latent_samples.shape[0]
            audio_dummy = latent_samples.new_zeros((b_size, 32, 2, 65))
            if hasattr(comfy, "nested_tensor") and hasattr(comfy.nested_tensor, "NestedTensor"):
                latent_samples = comfy.nested_tensor.NestedTensor((latent_samples, audio_dummy))
            else:
                latent_samples = (latent_samples, audio_dummy)

        noise = comfy.sample.prepare_noise(latent_samples, seed)
        sampled = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_samples, denoise=denoise, seed=seed,
        )
        sampled_samples = sampled["samples"] if isinstance(sampled, dict) else sampled

        if getattr(sampled_samples, "is_nested", False):
            video_samples = sampled_samples.unbind()[0]
        elif isinstance(sampled_samples, (list, tuple)):
            video_samples = sampled_samples[0]
        else:
            video_samples = sampled_samples

        decoded = normalize_decoded(vae.decode(video_samples))
        return decoded.to(device=image.device, dtype=torch.float32)

    def _diffuse_sequence(self, vae, model, positive, negative, images, steps, cfg, sampler_name, scheduler,
                          denoise, seed, noise_offset=0.0):
        """Diffuse an IMAGE batch as one temporal sequence; images are [T,H,W,C]."""
        seed = int(seed) & 0xffffffffffffffff
        frame_count = images.shape[0]
        latent = vae.encode(images[..., :3])
        latent_samples = latent["samples"] if isinstance(latent, dict) else latent

        # ComfyUI's video VAE wrapper converts an IMAGE batch [T,H,W,C] to a
        # video tensor and returns [B,C,T_lat,H_lat,W_lat]. Keep that compressed
        # temporal axis intact. Also accept per-frame 4D encoders for compatibility.
        if latent_samples.dim() == 5:
            if latent_samples.shape[0] != 1 or latent_samples.shape[2] < 1:
                raise ValueError(
                    "H3 video batch expects one encoded temporal sequence in "
                    f"[1,C,T,H,W] layout; received {tuple(latent_samples.shape)}"
                )
        elif latent_samples.dim() == 4 and latent_samples.shape[0] == frame_count:
            # Per-frame image VAE layout [T,C,H,W] -> one H3-style sequence.
            latent_samples = latent_samples.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            raise ValueError(
                "H3 video batch expects VAE encode to return [1,C,T,H,W] video "
                "latents or [T,C,H,W] per-frame latents; "
                f"received {tuple(latent_samples.shape)} for {frame_count} frames"
            )
        latent_frame_count = latent_samples.shape[2]
        if latent_frame_count < 1:
            raise ValueError("H3 video batch VAE encode returned an empty temporal latent")

        # Keep the latent's native [B,C,T,H,W] ordering for the H3 sampler.
        latent_samples = latent_samples.contiguous()

        # VAE wrappers may encode a spatial batch as B>1; this path deliberately
        # represents one ordered sequence, not independent videos.
        if latent_samples.shape[0] != 1:
            raise ValueError(
                "H3 video batch supports one temporal sequence; "
                f"received latent batch size {latent_samples.shape[0]}"
            )
        if noise_offset > 0.0:
            latent_samples = latent_samples + torch.randn_like(latent_samples) * noise_offset

        # Keep the dummy audio stream at the H3 still-image/audio-free default;
        # the issue here is the mismatched visual conditioning rows, not audio.
        audio_dummy = latent_samples.new_zeros((1, 32, 2, 65))
        if hasattr(comfy, "nested_tensor") and hasattr(comfy.nested_tensor, "NestedTensor"):
            latent_samples = comfy.nested_tensor.NestedTensor((latent_samples, audio_dummy))
        else:
            latent_samples = (latent_samples, audio_dummy)

        noise = comfy.sample.prepare_noise(latent_samples, seed)

        # Spatial H3 keyframe/reference latents are packed against the full
        # conditioning layout, but this path samples only a cropped tile. Their
        # rows can therefore disagree with the tile's update mask. The input
        # tile sequence itself supplies the visual source; omit external visual
        # guide latents for this tiled call while retaining audio-only refs.
        def conditioning_for_tile(conditioning):
            if conditioning is None:
                return None
            filtered = []
            removed_visual_guides = 0
            for entry in conditioning:
                if len(entry) > 1 and isinstance(entry[1], dict):
                    metadata = entry[1]
                    keyframes = metadata.get("minimax_keyframes") or []
                    refs = metadata.get("minimax_refs") or []
                    visual_refs = [ref for ref in refs if ref.get("latent") is not None]
                    audio_only_refs = []
                    for ref in refs:
                        if ref.get("latent") is None:
                            audio_only_refs.append(ref)
                        elif ref.get("audio_latent") is not None:
                            audio_only_refs.append({
                                "kind": "audio",
                                "ref_audio_t": ref.get("ref_audio_t", ref["audio_latent"].shape[-1]),
                                "audio_latent": ref["audio_latent"],
                            })
                    if keyframes or visual_refs:
                        entry = list(entry)
                        metadata = metadata.copy()
                        metadata.pop("minimax_keyframes", None)
                        if refs:
                            if audio_only_refs:
                                metadata["minimax_refs"] = audio_only_refs
                            else:
                                metadata.pop("minimax_refs", None)
                        entry[1] = metadata
                        removed_visual_guides += len(keyframes) + len(visual_refs)
                filtered.append(entry)
            if removed_visual_guides:
                logger.warning(
                    "[H3 Upscale] Ignoring %d spatial keyframe/reference guide(s) "
                    "during tiled video diffusion; the input tile supplies the visual source",
                    removed_visual_guides,
                )
            return filtered

        sequence_positive = conditioning_for_tile(positive)
        sequence_negative = conditioning_for_tile(negative)
        sampled = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            sequence_positive, sequence_negative, latent_samples,
            denoise=denoise, seed=seed,
        )
        sampled_samples = sampled["samples"] if isinstance(sampled, dict) else sampled
        if getattr(sampled_samples, "is_nested", False):
            video_samples = sampled_samples.unbind()[0]
        elif isinstance(sampled_samples, (list, tuple)):
            video_samples = sampled_samples[0]
        else:
            video_samples = sampled_samples

        decoded = vae.decode(video_samples)
        if isinstance(decoded, dict):
            decoded = decoded["samples"]
        # Video VAEs may return [B,T,H,W,C] or [B,C,T,H,W]; flatten batch/time
        # back to ComfyUI's [T,H,W,C] image-batch convention.
        if decoded.dim() == 5:
            if decoded.shape[1] in (1, 3, 4) and decoded.shape[-1] > 4:  # [B,C,T,H,W]
                decoded = decoded.permute(0, 2, 3, 4, 1)
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        elif decoded.dim() == 4 and decoded.shape[1] in (1, 3, 4) and decoded.shape[-1] > 4:
            decoded = decoded.permute(0, 2, 3, 1)
        if decoded.dim() != 4 or decoded.shape[0] < 1 or decoded.shape[-1] < 3:
            raise ValueError(
                "H3 video batch decode must return a non-empty image sequence; "
                f"received {tuple(decoded.shape)}"
            )

        # H3's causal VAE temporally compresses arbitrary input batches onto its
        # supported frame grid (e.g. 41 inputs decode as 39 frames). Resample only
        # along time so the output batch still has exactly one frame per input.
        decoded_frame_count = decoded.shape[0]
        if decoded_frame_count != frame_count:
            temporal_first = decoded.unsqueeze(0).permute(0, 4, 1, 2, 3)
            temporal_first = F.interpolate(
                temporal_first,
                size=(frame_count, decoded.shape[1], decoded.shape[2]),
                mode="trilinear",
                align_corners=True,
            )
            decoded = temporal_first.squeeze(0).permute(1, 2, 3, 0)
        return decoded[..., :3].to(device=images.device, dtype=torch.float32)

    def _upscale_video_batch(self, images, vae, model, positive, negative, upscale_by,
                             tile_width, tile_height, tile_padding, steps, cfg, sampler_name,
                             scheduler, denoise, seed, latent_noise_offset, lock_scale,
                             t_mix, t_edge, t_detail, align_to_source, enabled_h3_upscale):
        """Run matching spatial tiles across every frame in one temporal H3 diffusion call."""
        frame_count, target_h, target_w, channels = images.shape
        if denoise < 0.01 or not enabled_h3_upscale:
            return images.clone()

        tile_w = (tile_width // 16) * 16
        tile_h = (tile_height // 16) * 16
        padding = (tile_padding // 16) * 16
        weight_canvas = torch.zeros((1, target_h, target_w, 1), device=images.device)
        acc_canvas = torch.zeros_like(images)
        y_steps = range(0, max(1, target_h - padding), max(16, tile_h - padding))
        x_steps = range(0, max(1, target_w - padding), max(16, tile_w - padding))
        tile_index = 0

        for y in y_steps:
            for x in x_steps:
                y2, x2 = min(y + tile_h, target_h), min(x + tile_w, target_w)
                y1, x1 = max(0, y2 - tile_h), max(0, x2 - tile_w)
                tile_sequence = images[:, y1:y2, x1:x2, :]
                curr_h, curr_w = y2 - y1, x2 - x1
                decoded_sequence = self._diffuse_sequence(
                    vae, model, positive, negative, tile_sequence, steps, cfg,
                    sampler_name, scheduler, denoise, seed + tile_index,
                    noise_offset=latent_noise_offset,
                )
                tile_index += 1
                if decoded_sequence.shape[1:3] != (curr_h, curr_w):
                    decoded_sequence = _resize_bhwc(decoded_sequence, curr_h, curr_w)

                mask = create_tile_weight(
                    curr_h, curr_w, padding,
                    top=y1 > 0, bottom=y2 < target_h, left=x1 > 0, right=x2 < target_w,
                    device=images.device,
                )
                for frame_idx in range(frame_count):
                    decoded_tile = decoded_sequence[frame_idx:frame_idx + 1]
                    source_tile = tile_sequence[frame_idx:frame_idx + 1, ..., :3]
                    if align_to_source:
                        dec_np, _ = flow_align_np(
                            decoded_tile[0].detach().cpu().numpy(),
                            source_tile[0].detach().cpu().numpy(), None, max_frac=0.03,
                        )
                        decoded_tile = torch.from_numpy(dec_np).unsqueeze(0).to(
                            device=decoded_tile.device, dtype=decoded_tile.dtype
                        )
                    decoded_tile = match_low_frequency(
                        decoded_tile, source_tile,
                        sigma=max(8.0, min(curr_h, curr_w) / 16.0) / max(lock_scale, 0.05),
                    )
                    decoded_tile = restore_detail(decoded_tile, source_tile, t_detail * lock_scale)
                    if t_edge > 0.0 and lock_scale > 0.0:
                        decoded_tile = decoded_tile + (
                            source_tile - decoded_tile
                        ) * edge_protect_weight(source_tile) * (t_edge * lock_scale)
                    if t_mix < 1.0:
                        decoded_tile = source_tile + (decoded_tile - source_tile) * t_mix
                    acc_canvas[frame_idx:frame_idx + 1, y1:y2, x1:x2, :3] += decoded_tile * mask
                weight_canvas[:, y1:y2, x1:x2, :] += mask

        weight_canvas = torch.clamp(weight_canvas, min=1e-5)
        final_images = acc_canvas / weight_canvas
        if channels > 3:
            final_images[..., 3:] = images[..., 3:]
        return final_images

    def _face_detail_video(self, frames, vae, model, positive, negative, steps, cfg,
                           sampler_name, scheduler, seed, face_denoise, face_padding, face_size,
                           reference_master, reference_alpha, reference_aspect, reference_box,
                           reference_strength, reference_pose_match, reference_flow_align,
                           face_detector, face_detector_model, face_confidence, manual_face_boxes,
                           face_mask, sam3_prompt, frontal_only_threshold, face_chroma_lock,
                           face_structure_lock, face_seamless_blend, detail_retention):
        """Face pass for an ordered frame batch.

        Detection still runs per frame, but the diffusion does not: detections are linked into
        temporal tracks and each track is diffused as ONE H3 sequence, so the sampler sees the
        face's motion instead of a stack of unrelated stills and a shot costs one face pass per
        face rather than one per frame.
        """
        frame_count, img_h, img_w = frames.shape[0], frames.shape[1], frames.shape[2]
        device = frames.device

        per_frame_faces, per_frame_masks, labels = [], [], []
        for frame_idx in range(frame_count):
            faces, det_label = detect_faces(
                frames[frame_idx:frame_idx + 1], face_detector, face_detector_model,
                face_confidence, face_mask=face_mask, mask_index=frame_idx,
                manual_boxes=manual_face_boxes, sam3_prompt=sam3_prompt,
            )
            if frontal_only_threshold > 0.0 and faces:
                kept = [f for f in faces if _FACE_SYM.get(f, 1.0) >= frontal_only_threshold]
                if len(kept) != len(faces):
                    logger.info("[H3 Upscale] frontal filter (frame %d): kept %d of %d"
                                % (frame_idx, len(kept), len(faces)))
                faces = kept
            per_frame_faces.append(faces)
            # detect_faces clears the mask cache, so copy this frame's masks before the next frame.
            per_frame_masks.append({f: _FACE_MASKS.get(f) for f in faces})
            labels.append(det_label)

        tracks = track_face_boxes(per_frame_faces)
        det_label = labels[0] if labels else "none"
        logger.info("[H3 Upscale] face detailer (video): %d track(s) from %d frame(s), %d detection(s)"
                    % (len(tracks), frame_count, sum(len(f) for f in per_frame_faces)))

        output = frames.clone()
        regions = [torch.zeros((1, img_h, img_w, 1), device=device) for _ in range(frame_count)]
        debug_boxes = [[] for _ in range(frame_count)]
        master_face, master_alpha = reference_master, reference_alpha
        master_aspect, master_box = reference_aspect, reference_box
        faces_done = 0

        for track in tracks:
            idxs = sorted(track)
            boxes = [track[i] for i in idxs]
            rects = [expand_box(fx, fy, fw, fh, face_padding, img_w, img_h)
                     for fx, fy, fw, fh in boxes]
            # One crop size for the whole track: every frame is cropped to the same rectangle scale
            # around its own detection, so the sequence is aligned and no frame is stretched.
            crop_w = min(img_w, max(x2 - x1 for x1, y1, x2, y2 in rects))
            crop_h = min(img_h, max(y2 - y1 for x1, y1, x2, y2 in rects))
            fixed = []
            for fx, fy, fw, fh in boxes:
                x1 = int(round(fx + fw / 2.0 - crop_w / 2.0))
                y1 = int(round(fy + fh / 2.0 - crop_h / 2.0))
                x1 = max(0, min(x1, img_w - crop_w))
                y1 = max(0, min(y1, img_h - crop_h))
                fixed.append((x1, y1, x1 + crop_w, y1 + crop_h))

            long_side, short_side = max(crop_h, crop_w), min(crop_h, crop_w)
            scale = max(min(2048, max(face_size, long_side)) / float(long_side), 512.0 / short_side)
            scale = min(scale, 2048.0 / long_side)
            # The diffusion is chunked, but the prepared crops are held for the whole track, so cap
            # one track's crop stack by a pixel budget (never below 512 px on the short side).
            budget_side = (FACE_SEQUENCE_PIXEL_BUDGET / float(max(len(idxs), 1))) ** 0.5
            scale = min(scale, max(512.0 / short_side, budget_side / long_side))
            work_h = int(min(2048, max(64, np.ceil(crop_h * scale / 32.0) * 32)))
            work_w = int(min(2048, max(64, np.ceil(crop_w * scale / 32.0) * 32)))

            init_list, detail_list, orig_list, mask_list, meta = [], [], [], [], []
            for frame_idx, (fx, fy, fw, fh), (x1, y1, x2, y2) in zip(idxs, boxes, fixed):
                orig_region = frames[frame_idx:frame_idx + 1, y1:y2, x1:x2, :3]
                inst = per_frame_masks[frame_idx].get((fx, fy, fw, fh))
                if inst is not None:
                    crop_mask = instance_blend_mask(inst, fx, fy, x1, y1, crop_h, crop_w, device)
                else:
                    crop_mask = create_face_blend_mask(crop_h, crop_w, feather=0.35, device=device)
                crop_mask = crop_mask * border_taper(crop_h, crop_w, 0.08, device)
                orig_work = _resize_bhwc(orig_region, work_h, work_w)
                mask_work = _resize_bhwc(crop_mask.expand(-1, -1, -1, 3), work_h, work_w)[..., :1]
                init, detail_src = orig_work, orig_work
                if master_face is not None:
                    compat = 1.0
                    if reference_pose_match and master_aspect:
                        ratio = (fw / float(max(fh, 1))) / master_aspect
                        compat = float(np.clip((np.log(1.6) - abs(np.log(max(ratio, 1e-3)))) / (np.log(1.6) - np.log(1.15)), 0.0, 1.0))
                    eff_strength = reference_strength * compat
                    if eff_strength > 0.01:
                        tbox = ((fx - x1) * work_w / float(crop_w), (fy - y1) * work_h / float(crop_h),
                                fw * work_w / float(crop_w), fh * work_h / float(crop_h))
                        if master_box is not None:
                            m, al = warp_master_to_box(master_face, master_alpha, master_box, tbox, work_h, work_w)
                        else:
                            m = _cover_resize(master_face, work_h, work_w)
                            al = _cover_resize(master_alpha.expand(-1, -1, -1, 3), work_h, work_w)[..., :1]
                        m = match_low_frequency(m, orig_work, sigma=max(work_h, work_w) / 10.0)
                        if reference_flow_align:
                            m, al = flow_align(m, orig_work, al)
                        a_mix = mask_work * al * eff_strength
                        init = orig_work * (1.0 - a_mix) + m * a_mix
                        detail_src = init
                init_list.append(init)
                detail_list.append(detail_src)
                orig_list.append(orig_work)
                mask_list.append(crop_mask)
                meta.append((frame_idx, x1, y1, x2, y2))

            init_seq = torch.cat(init_list, dim=0)
            if init_seq.shape[0] == 1:
                # A face that appears for a single frame is a still: keep the still-image path.
                decoded_seq = self._diffuse(vae, model, positive, negative, init_seq, steps, cfg,
                                            sampler_name, scheduler, face_denoise, seed)
                chunk_count = 1
            else:
                chunk_len = max(1, min(FACE_SEQUENCE_CHUNK,
                                       int(FACE_SEQUENCE_PIXEL_BUDGET / float(work_h * work_w))))
                decoded_parts = []
                for start in range(0, init_seq.shape[0], chunk_len):
                    chunk = init_seq[start:start + chunk_len]
                    try:
                        decoded_parts.append(self._diffuse_sequence(
                            vae, model, positive, negative, chunk, steps, cfg,
                            sampler_name, scheduler, face_denoise, seed,
                        ))
                    except Exception as exc:
                        logger.warning("[H3 Upscale] face sequence pass failed (%s); "
                                       "retrying %d face crop(s) as stills" % (exc, chunk.shape[0]))
                        decoded_parts.append(torch.cat([
                            self._diffuse(vae, model, positive, negative, chunk[k:k + 1], steps, cfg,
                                          sampler_name, scheduler, face_denoise, seed)
                            for k in range(chunk.shape[0])
                        ], dim=0))
                decoded_seq = torch.cat(decoded_parts, dim=0)
                chunk_count = len(decoded_parts)
            if decoded_seq.shape[0] != len(idxs):
                # The video VAE may return a different frame count than it was given; map the
                # decoded frames back onto the track so every detection still gets a result.
                last = max(decoded_seq.shape[0] - 1, 0)
                idx_map = [int(round(k * last / float(max(1, len(idxs) - 1)))) for k in range(len(idxs))]
                logger.info("[H3 Upscale] face track: VAE returned %d frame(s) for %d crop(s); resampling"
                            % (decoded_seq.shape[0], len(idxs)))
                decoded_seq = decoded_seq[idx_map]
            logger.info("[H3 Upscale] face track: %d frame(s), crop %dx%d -> work %dx%d, %d temporal pass(es)"
                        % (len(idxs), crop_w, crop_h, work_w, work_h, chunk_count))

            for k, (frame_idx, x1, y1, x2, y2) in enumerate(meta):
                decoded = decoded_seq[k:k + 1]
                if decoded.shape[1] != work_h or decoded.shape[2] != work_w:
                    decoded = _resize_bhwc(decoded, work_h, work_w)
                init, detail_src = init_list[k], detail_list[k]
                dec_np, _ = flow_align_np(decoded[0].detach().cpu().numpy(),
                                          init[0].detach().cpu().numpy(), None, max_frac=0.03)
                decoded = torch.from_numpy(dec_np).unsqueeze(0).to(device=decoded.device, dtype=decoded.dtype)
                decoded = lock_chroma(decoded, init, face_chroma_lock)
                decoded = restore_detail(decoded, detail_src, detail_retention)
                if face_structure_lock > 0.0:
                    composed = compose_face_detail(decoded, detail_src, max(2.0, max(work_h, work_w) * 0.012))
                    decoded = decoded * (1.0 - face_structure_lock) + composed * face_structure_lock
                decoded = match_low_frequency(decoded, orig_list[k], sigma=max(work_h, work_w) / 10.0)

                # Without a reference the largest face (first track) becomes the master, as in the
                # still path, so every other face in the shot matches it.
                if master_face is None and k == 0:
                    master_face = decoded.detach().clone()
                    master_alpha = _resize_bhwc(mask_list[k].to(device).expand(-1, -1, -1, 3), work_h, work_w)[..., :1]
                    master_box = ((boxes[k][0] - x1) * work_w / float(crop_w), (boxes[k][1] - y1) * work_h / float(crop_h),
                                  boxes[k][2] * work_w / float(crop_w), boxes[k][3] * work_h / float(crop_h))
                    master_aspect = boxes[k][2] / float(max(boxes[k][3], 1))

                crop_mask = mask_list[k]
                orig_region = frames[frame_idx:frame_idx + 1, y1:y2, x1:x2, :3]
                restored = _resize_bhwc(decoded, crop_h, crop_w)
                blended = seamless_face_blend(orig_region, restored, crop_mask) if face_seamless_blend else None
                if blended is None:
                    blended = orig_region * (1.0 - crop_mask) + restored * crop_mask
                output[frame_idx:frame_idx + 1, y1:y2, x1:x2, :3] = torch.clamp(blended, 0.0, 1.0)
                regions[frame_idx][:, y1:y2, x1:x2, :] = torch.maximum(
                    regions[frame_idx][:, y1:y2, x1:x2, :], crop_mask.to(device))
                debug_boxes[frame_idx].append((x1, y1, x2, y2, boxes[k][0], boxes[k][1],
                                               boxes[k][0] + boxes[k][2], boxes[k][1] + boxes[k][3]))
                faces_done += 1

        thumb = None if reference_master is None else reference_master * reference_alpha + 0.5 * (1.0 - reference_alpha)
        debug_images = torch.cat([
            draw_face_debug(frames[i:i + 1], debug_boxes[i], det_label, thumb) for i in range(frame_count)
        ], dim=0)
        logger.info("[H3 Upscale] face detailer (video) done: %d face crop(s) re-rendered" % faces_done)
        return output, debug_images, regions

    # ------------------------------------------------------------------------------------------
    def upscale(
        self, image, model, positive, negative, vae, upscale_by: float,
        tile_width: int, tile_height: int, tile_padding: int, seam_fix_width: int,
        seam_fix_denoise: float, seam_fix_padding: int, seed: int, steps: int,
        cfg: float, sampler_name: str, scheduler: str, denoise: float,
        upscale_model=None, reference_face=None,
        enable_dlss5: bool = False,
        upscaling_mode: str = "1x (DLAA / native)",
        nr_preset: str = "Default",
        nr_style: str = "Default",
        nr_intensity: float = 1.0,
        local_tone_strength: float = 1.0,
        local_structure_strength: float = 1.5,
        skin_structure_strength: float = 2.0,
        automatic_mask: bool = True,
        dlss_model_preset: str = "M",
        motion: str = "auto",
        scene_change_threshold: float = 0.24,
        warmup_frames: int = 0,
        runtime_dir: str = "",
        verify_neural_rendering: bool = True,
        enable_refiner: bool = False, refine_amount: float = 0.35, refine_radius: int = 2,
        enable_face_detailer: bool = False, face_denoise: float = 0.25,
        face_padding: float = 0.25, face_size: int = 1024,
        face_mask=None, face_detector: str = "auto", face_detector_model: str = "",
        face_confidence: float = 0.35, manual_face_boxes: str = "",
        reference_strength: float = 0.75, detail_retention: float = 0.6,
        enable_colorfx: bool = True, enable_color_correction: bool = True, enable_lut_processing: bool = True, enable_enhancements: bool = True, enable_blur_effects: bool = True, enable_stylistic_effects: bool = True, hdr_intensity: float = 2.5, shadow_intensity: float = 0.2, highlight_intensity: float = 0.2, gamma: float = 1.0, brightness: float = 0.9, contrast: float = 1.0, saturation: float = 0.9, enhance_color: float = 1.0, lut_name: str = "None", lut_strength: float = 0.5, lut_log_process: bool = True, sharpness: float = 3.0, edge_enhance_strength: float = 0.1, detail_enhance_strength: float = 0.2, blur_radius: int = 0, gaussian_blur_radius: float = 0.0, radial_blur_strength: float = 32.0, radial_blur_center_x: float = 0.5, radial_blur_center_y: float = 0.5, radial_blur_focus_spread: float = 3.0, radial_blur_steps: int = 10, chromatic_aberration_r_x: float = 1.0, chromatic_aberration_r_y: float = 0.0, chromatic_aberration_b_x: float = -1.0, chromatic_aberration_b_y: float = 0.0, chromatic_blur_amount: float = 0.5, simple_film_grain_intensity: float = 0.04, simple_film_grain_monochrome: bool = True, scanline_intensity: float = 0.0, vignette_intensity: float = 0.4, vignette_center_x: float = 0.5, vignette_center_y: float = 0.5, soft_light_opacity: float = 0.3, soft_light_blur_radius: float = 10.0,
        latent_noise_offset: float = 0.0, enabled_h3_upscale: bool = True, tone_lock: float = 1.0, sam3_prompt: str = "head", face_structure_lock: float = 0.8, reference_pose_match: bool = True, reference_flow_align: bool = True, face_chroma_lock: float = 0.9, face_seamless_blend: bool = True, frontal_only_threshold: float = 0.0, locks_follow_denoise: bool = True, redraw_only: bool = False, align_to_source: bool = True, chroma_lock: float = 0.9, edge_protect: float = 0.85, diffusion_mix: float = 1.0,
        **kwargs
    ):
        batch_size, h, w, c = image.shape
        lock_scale = source_lock_scale(denoise, locks_follow_denoise)
        # redraw_only: the model's redraw is the ONLY image that survives. Every blend with the source image
        # (mix, edge/detail restore, tone/chroma lock) is bypassed, so nothing of the old render can ghost through.
        if redraw_only:
            lock_scale = 0.0
            t_mix, t_edge, t_detail, tone_eff, chroma_eff = 1.0, 0.0, 0.0, 0.0, 0.0
            logger.info("[H3 Upscale] redraw_only: all source blending disabled")
        else:
            t_mix, t_edge, t_detail, tone_eff, chroma_eff = diffusion_mix, edge_protect, detail_retention, tone_lock, chroma_lock
        if locks_follow_denoise:
            logger.info(f"[H3 Upscale] denoise {denoise:.2f} -> source-lock strength {lock_scale:.2f}")
        target_h = (int(h * upscale_by) // 16) * 16
        target_w = (int(w * upscale_by) // 16) * 16
        if not enabled_h3_upscale:
            target_h, target_w = h, w

        tile_w = (tile_width // 16) * 16
        tile_h = (tile_height // 16) * 16
        padding = (tile_padding // 16) * 16

        # Base image upsampling using model if provided, falling back to bilinear interpolation
        if not enabled_h3_upscale:
            resized_images = image.clone()
            logger.info("[H3 Upscale] enabled_h3_upscale is off: group bypassed (no resize, no tile diffusion)")
        elif upscale_model is not None:
            import comfy_extras.nodes_upscale_model
            upscaler = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel()
            resized_images = upscaler.upscale(upscale_model, image)[0]
            if resized_images.shape[1] != target_h or resized_images.shape[2] != target_w:
                resized_images = _resize_bhwc(resized_images, target_h, target_w)
        else:
            resized_images = torch.nn.functional.interpolate(
                image.permute(0, 3, 1, 2),
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
                antialias=upscale_by < 1.0,
            ).permute(0, 2, 3, 1).clamp(0.0, 1.0)

        if batch_size > 1:
            # A ComfyUI image batch represents an ordered frame sequence here. Process the same
            # spatial tile from all frames together so H3 can use its temporal video latent.
            final_images = self._upscale_video_batch(
                resized_images, vae, model, positive, negative, upscale_by,
                tile_width, tile_height, tile_padding, steps, cfg, sampler_name,
                scheduler, denoise, seed, latent_noise_offset, lock_scale,
                t_mix, t_edge, t_detail, align_to_source, enabled_h3_upscale,
            )
        else:
            output_batch = []

            for b in range(batch_size):
                canvas = resized_images[b:b+1].clone()
                if denoise < 0.01 or not enabled_h3_upscale:  # tile diffusion disabled: keep the clean upscale (face detailer etc. still run)
                    output_batch.append(canvas)
                    continue
                weight_canvas = torch.zeros((1, target_h, target_w, 1), device=canvas.device)
                acc_canvas = torch.zeros_like(canvas)

                y_steps = range(0, max(1, target_h - padding), max(16, tile_h - padding))
                x_steps = range(0, max(1, target_w - padding), max(16, tile_w - padding))
                tile_index = 0

                for y in y_steps:
                    for x in x_steps:
                        y2 = min(y + tile_h, target_h)
                        x2 = min(x + tile_w, target_w)
                        y1 = max(0, y2 - tile_h)
                        x1 = max(0, x2 - tile_w)

                        tile = canvas[:, y1:y2, x1:x2, :]
                        curr_h, curr_w = y2 - y1, x2 - x1

                        # Different noise per tile (identical noise on every tile prints a repeating pattern)
                        decoded_tile = self._diffuse(
                            vae, model, positive, negative, tile, steps, cfg, sampler_name, scheduler,
                            denoise, seed + tile_index, noise_offset=latent_noise_offset,
                        )
                        tile_index += 1

                        if decoded_tile.shape[1] != curr_h or decoded_tile.shape[2] != curr_w:
                            decoded_tile = _resize_bhwc(decoded_tile, curr_h, curr_w)

                        # Keep the source's broad colour/lighting (frequency-aware, does NOT flatten detail),
                        # then put back any fine detail the diffusion smoothed away.
                        # The model can draw the subject a few pixels away from where the source has it. Register the
                        # redraw onto the source geometry (small, smooth, clamped warp) so any remaining blend, tile
                        # overlap or later pass cannot produce a second, offset copy.
                        if align_to_source:
                            dec_np, _ = flow_align_np(decoded_tile[0].detach().cpu().numpy(),
                                                      tile[0, ..., :3].detach().cpu().numpy(), None, max_frac=0.03)
                            decoded_tile = torch.from_numpy(dec_np).unsqueeze(0).to(device=decoded_tile.device, dtype=decoded_tile.dtype)

                        # tile-to-tile tone consistency: fine scale for light refinement, widening to a plain per-tile
                        # brightness match as the lock fades out (a fine-scale match against a re-drawn subject = ghost)
                        decoded_tile = match_low_frequency(decoded_tile, tile[..., :3],
                                                           sigma=max(8.0, min(curr_h, curr_w) / 16.0) / max(lock_scale, 0.05))
                        decoded_tile = restore_detail(decoded_tile, tile[..., :3], t_detail * lock_scale)
                        if t_edge > 0.0 and lock_scale > 0.0:  # silhouettes come from the clean source
                            decoded_tile = decoded_tile + (tile[..., :3] - decoded_tile) * edge_protect_weight(tile[..., :3]) * (t_edge * lock_scale)
                        if t_mix < 1.0:
                            decoded_tile = tile[..., :3] + (decoded_tile - tile[..., :3]) * t_mix

                        mask = create_tile_weight(
                            curr_h, curr_w, padding,
                            top=y1 > 0, bottom=y2 < target_h, left=x1 > 0, right=x2 < target_w,
                            device=canvas.device,
                        )

                        acc_canvas[:, y1:y2, x1:x2, :3] += decoded_tile * mask
                        weight_canvas[:, y1:y2, x1:x2, :] += mask

                weight_canvas = torch.clamp(weight_canvas, min=1e-5)
                final_img = acc_canvas / weight_canvas
                if canvas.shape[-1] > 3:  # keep alpha from the source if there is one
                    final_img[..., 3:] = canvas[..., 3:]
                output_batch.append(final_img)

            final_images = torch.cat(output_batch, dim=0)

        face_debug_images = final_images
        face_regions = None

        # Optional AI Face Detailer Pass
        if enable_face_detailer:
            # Master face from the reference input (if connected): applied to EVERY face, largest first.
            reference_master, reference_alpha, reference_aspect, reference_box = None, None, None, None
            if reference_face is not None:
                reference_master, reference_alpha, reference_aspect, reference_box = prepare_reference_face(
                    reference_face, face_padding, face_detector, face_detector_model, face_confidence, sam3_prompt
                )
                reference_master = reference_master.to(final_images.device)
                reference_alpha = reference_alpha.to(final_images.device)
                logger.info(f"[H3 Upscale] reference_face connected: master crop {reference_master.shape[2]}x{reference_master.shape[1]}, strength {reference_strength}")
            else:
                logger.info("[H3 Upscale] no reference_face connected: the largest face becomes the master")

        if enable_face_detailer and final_images.shape[0] > 1:
            # Video batch: faces are tracked across frames and diffused as temporal sequences.
            final_images, face_debug_images, face_regions = self._face_detail_video(
                final_images, vae, model, positive, negative, steps, cfg, sampler_name,
                scheduler, seed, face_denoise, face_padding, face_size, reference_master,
                reference_alpha, reference_aspect, reference_box, reference_strength,
                reference_pose_match, reference_flow_align, face_detector, face_detector_model,
                face_confidence, manual_face_boxes, face_mask, sam3_prompt,
                frontal_only_threshold, face_chroma_lock, face_structure_lock,
                face_seamless_blend, detail_retention,
            )
        elif enable_face_detailer:
            output_faces = []
            debug_faces = []
            face_regions = []

            for img_idx in range(final_images.shape[0]):
                canvas_img = final_images[img_idx:img_idx+1].clone()
                img_h, img_w = canvas_img.shape[1], canvas_img.shape[2]

                faces, det_label = detect_faces(
                    canvas_img, face_detector, face_detector_model, face_confidence,
                    face_mask=face_mask, mask_index=img_idx, manual_boxes=manual_face_boxes, sam3_prompt=sam3_prompt,
                )
                if frontal_only_threshold > 0.0 and faces:
                    kept = [f for f in faces if _FACE_SYM.get(f, 1.0) >= frontal_only_threshold]
                    logger.info("[H3 Upscale] frontal filter (symmetry >= %.2f): kept %d of %d (%s)" % (
                        frontal_only_threshold, len(kept), len(faces),
                        ", ".join("%.2f" % _FACE_SYM.get(f, 1.0) for f in faces)))
                    faces = kept
                logger.info(f"[H3 Upscale] face detailer: {len(faces)} face(s) via {det_label}")

                if len(faces) == 0:
                    face_regions.append(None)
                    output_faces.append(canvas_img)
                    debug_faces.append(canvas_img)
                    continue

                # faces are sorted largest -> smallest. With a reference connected it is the master
                # for all of them; otherwise the largest face is rendered first and becomes the master.
                master_face, master_alpha, master_aspect, master_box = reference_master, reference_alpha, reference_aspect, reference_box
                # The face pass works from the CURRENT canvas only. (Pulling structure/detail from the pre-diffusion
                # image while the canvas had been re-drawn put two slightly different versions of the face on top of
                # each other, which is exactly the ghosting.)
                pristine = None
                used_boxes = []
                region = torch.zeros((1, img_h, img_w, 1), device=canvas_img.device)

                for face_idx, (fx, fy, fw, fh) in enumerate(faces):
                    x1, y1, x2, y2 = expand_box(fx, fy, fw, fh, face_padding, img_w, img_h)
                    crop_h, crop_w = y2 - y1, x2 - x1
                    if min(crop_h, crop_w) < 32:
                        continue
                    used_boxes.append((x1, y1, x2, y2, fx, fy, fx + fw, fy + fh))
                    inst_mask = _FACE_MASKS.get((fx, fy, fw, fh))
                    if inst_mask is not None:
                        crop_mask = instance_blend_mask(inst_mask, fx, fy, x1, y1, crop_h, crop_w, canvas_img.device)
                    else:
                        crop_mask = create_face_blend_mask(crop_h, crop_w, feather=0.35, device=canvas_img.device)
                    crop_mask = crop_mask * border_taper(crop_h, crop_w, 0.08, canvas_img.device)

                    orig_region = canvas_img[:, y1:y2, x1:x2, :3]

                    # Working size keeps the crop's aspect ratio: long side >= face_size (or native up to 2048),
                    # short side >= 512, everything a multiple of 32 for the VAE.
                    long_side, short_side = max(crop_h, crop_w), min(crop_h, crop_w)
                    sc = max(min(2048, max(face_size, long_side)) / float(long_side), 512.0 / short_side)
                    sc = min(sc, 2048.0 / long_side)
                    work_h = int(min(2048, max(64, np.ceil(crop_h * sc / 32.0) * 32)))
                    work_w = int(min(2048, max(64, np.ceil(crop_w * sc / 32.0) * 32)))
                    orig_work = _resize_bhwc(orig_region, work_h, work_w)

                    # If a master face exists, mix it in (cover-resized, tone-matched, inside the face mask).
                    init = orig_work
                    # Pose compatibility: a frontal master blended onto a profile/back view only produces
                    # ghosting, so fade the mix out when the face's shape differs from the master's.
                    compat = 1.0
                    if reference_pose_match and master_aspect:
                        ratio = (fw / float(max(fh, 1))) / master_aspect
                        compat = float(np.clip((np.log(1.6) - abs(np.log(max(ratio, 1e-3)))) / (np.log(1.6) - np.log(1.15)), 0.0, 1.0))
                    eff_strength = reference_strength * compat
                    use_master = master_face is not None and eff_strength > 0.01
                    # Detail source: the pristine pre-diffusion upscale (the tile pass may already have smoothed
                    # the face), mixed with the master wherever the master is applied.
                    if pristine is not None:
                        pristine_work = _resize_bhwc(pristine[:, y1:y2, x1:x2, :3].to(orig_work.device), work_h, work_w)
                    else:
                        pristine_work = orig_work
                    detail_src = pristine_work
                    if use_master:
                        tbox = ((fx - x1) * work_w / float(crop_w), (fy - y1) * work_h / float(crop_h),
                                fw * work_w / float(crop_w), fh * work_h / float(crop_h))
                        if master_box is not None:
                            m, al = warp_master_to_box(master_face, master_alpha, master_box, tbox, work_h, work_w)
                        else:
                            m = _cover_resize(master_face, work_h, work_w)
                            al = _cover_resize(master_alpha.expand(-1, -1, -1, 3), work_h, work_w)[..., :1]
                        m = match_low_frequency(m, orig_work, sigma=max(work_h, work_w) / 10.0)
                        if reference_flow_align:
                            m, al = flow_align(m, orig_work, al)
                        cm = _resize_bhwc(crop_mask.to(orig_work.device).expand(-1, -1, -1, 3), work_h, work_w)[..., :1]
                        a_mix = cm * al * eff_strength
                        init = orig_work * (1.0 - a_mix) + m * a_mix
                        detail_src = pristine_work * (1.0 - a_mix) + m * a_mix
                    logger.info(f"[H3 Upscale] face {face_idx}: box {fw}x{fh}, crop {crop_w}x{crop_h}, work {work_w}x{work_h}, "
                                f"mask={'sam' if inst_mask is not None else 'ellipse'}, master={'yes' if use_master else 'no'}, "
                                f"pose_match={compat:.2f}")

                    decoded = self._diffuse(
                        vae, model, positive, negative, init, steps, cfg, sampler_name, scheduler,
                        face_denoise, seed, noise_offset=0.0,
                    )
                    if decoded.shape[1] != work_h or decoded.shape[2] != work_w:
                        decoded = _resize_bhwc(decoded, work_h, work_w)

                    # 1) Register the redraw onto the source geometry. A distilled model can draw eyes/edges several
                    #    pixels away from where the source has them; compositing that without registration doubles them.
                    dec_np, _ = flow_align_np(decoded[0].detach().cpu().numpy(), init[0].detach().cpu().numpy(), None, max_frac=0.03)
                    decoded = torch.from_numpy(dec_np).unsqueeze(0).to(device=decoded.device, dtype=decoded.dtype)
                    # 2) Keep the face's own colour (a hue/brightness step reads as a pasted patch).
                    decoded = lock_chroma(decoded, init, face_chroma_lock)

                    # Never blurrier than the source: keep structure from `init`, add/keep fine detail only.
                    decoded = restore_detail(decoded, detail_src, detail_retention)
                    if face_structure_lock > 0.0:
                        composed = compose_face_detail(decoded, detail_src, max(2.0, max(work_h, work_w) * 0.012))
                        decoded = decoded * (1.0 - face_structure_lock) + composed * face_structure_lock
                    decoded = match_low_frequency(decoded, orig_work, sigma=max(work_h, work_w) / 10.0)

                    # The first (largest) face becomes the master for all the smaller ones.
                    if master_face is None:
                        master_face = decoded.detach().clone()
                        master_alpha = _resize_bhwc(crop_mask.to(master_face.device).expand(-1, -1, -1, 3), work_h, work_w)[..., :1]
                        master_box = ((fx - x1) * work_w / float(crop_w), (fy - y1) * work_h / float(crop_h),
                                      fw * work_w / float(crop_w), fh * work_h / float(crop_h))
                        master_aspect = fw / float(max(fh, 1))

                    restored_face = _resize_bhwc(decoded, crop_h, crop_w)
                    blended = None
                    if face_seamless_blend:   # Poisson blend: matches colour/brightness to the surroundings along the whole edge
                        blended = seamless_face_blend(orig_region, restored_face, crop_mask)
                    if blended is None:
                        blended = orig_region * (1.0 - crop_mask) + restored_face * crop_mask
                    canvas_img[:, y1:y2, x1:x2, :3] = torch.clamp(blended, 0.0, 1.0)
                    region[:, y1:y2, x1:x2, :] = torch.maximum(region[:, y1:y2, x1:x2, :], crop_mask.to(region.device))

                face_regions.append(region)
                output_faces.append(canvas_img)
                debug_faces.append(draw_face_debug(final_images[img_idx:img_idx+1], used_boxes, det_label,
                                                  None if reference_master is None else reference_master * reference_alpha + 0.5 * (1.0 - reference_alpha)))

            final_images = torch.cat(output_faces, dim=0)
            face_debug_images = torch.cat(debug_faces, dim=0)

        # Optional Edge & Texture Refiner Pass
        if enable_refiner:
            final_images = refine_edges_and_textures(final_images, amount=refine_amount, radius=refine_radius)

        # DLSS 5 Neural Rendering Pass
        if enable_dlss5:
            from .dlss5 import DlssOptions, DlssSession, find_runtime
            from .dlss5.imaging import fit_frame, rgba_to_tensor, tensor_to_rgba
            from .dlss5.motion import TemporalGuide
            from .dlss_nodes.common import confirm_feature_18

            layout = find_runtime(runtime_dir.strip() or None)
            options = DlssOptions.create(
                upscaling_mode=upscaling_mode,
                nr_preset=nr_preset,
                nr_style=nr_style,
                nr_intensity=nr_intensity,
                local_tone_strength=local_tone_strength,
                local_structure_strength=local_structure_strength,
                skin_structure_strength=skin_structure_strength,
                automatic_mask=automatic_mask,
                dlss_model_preset=dlss_model_preset,
                motion_mode=motion,
                scene_change_threshold=scene_change_threshold,
                warmup_frames=warmup_frames,
            )

            count, height, width, channels = final_images.shape
            keep_alpha = channels == 4

            with DlssSession(layout, options, input_width=width, input_height=height, frame_count=count) as session:
                guide = TemporalGuide(
                    session.render_width,
                    session.render_height,
                    flow_width=options.flow_width,
                    scene_change_threshold=options.scene_change_threshold,
                    enabled=options.wants_motion(count),
                )

                dlss_result = torch.empty(
                    (count, session.output_height, session.output_width, 4 if keep_alpha else 3),
                    dtype=torch.float32,
                    device=final_images.device
                )

                for index in range(count):
                    rgba = fit_frame(
                        tensor_to_rgba(final_images[index]),
                        session.render_width,
                        session.render_height,
                    )
                    motion_data = guide.process(rgba)
                    enhanced, _pts = session.submit(
                        index=index,
                        rgba=rgba,
                        motion=motion_data.motion,
                        reset=motion_data.reset,
                        pts=index,
                    )
                    dlss_result[index] = rgba_to_tensor(enhanced, keep_alpha=keep_alpha).to(final_images.device)

            confirm_feature_18(session, verify_neural_rendering)
            final_images = dlss_result

        # ---- Color FX (last stage) ----
        if enable_colorfx:
            # (a) tone / chroma lock against the clean pre-diffusion upscale (top of the "Color Correction" subgroup)
            if enable_color_correction and (tone_eff > 0.0 or chroma_eff > 0.0) and final_images.shape[0] == resized_images.shape[0]:
                ref_img = resized_images.to(device=final_images.device, dtype=final_images.dtype)[..., :3]
                if ref_img.shape[1:3] != final_images.shape[1:3]:   # e.g. DLSS5 upscaled the image
                    ref_img = _resize_bhwc(ref_img, final_images.shape[1], final_images.shape[2])
                cc = final_images.clone()
                if tone_eff > 0.0:
                    sig = max(8.0, min(cc.shape[1], cc.shape[2]) / 32.0) / max(lock_scale, 0.05)
                    locked = match_low_frequency(cc[..., :3], ref_img, sig)
                    cc[..., :3] = cc[..., :3] * (1.0 - tone_eff) + locked * tone_eff
                if chroma_eff > 0.0 and lock_scale > 0.0:
                    locked = lock_chroma(cc[..., :3], ref_img, 1.0)
                    wgt = torch.full((cc.shape[0], cc.shape[1], cc.shape[2], 1), float(chroma_eff * lock_scale), device=cc.device)
                    if face_regions is not None and len(face_regions) == cc.shape[0]:
                        for bi, rg in enumerate(face_regions):
                            if rg is not None:
                                rg = rg.to(wgt.device)
                                if rg.shape[1:3] != cc.shape[1:3]:
                                    rg = _resize_bhwc(rg, cc.shape[1], cc.shape[2])
                                wgt[bi:bi + 1] = wgt[bi:bi + 1] * (1.0 - rg)
                    cc[..., :3] = cc[..., :3] + (locked - cc[..., :3]) * wgt
                final_images = cc

            # (b) the Color FX effects (ported from FilmAuteur_LTX)
            final_images = self._apply_color_fx(final_images, dict(
                enable_color_correction=enable_color_correction, enable_lut_processing=enable_lut_processing,
                enable_enhancements=enable_enhancements, enable_blur_effects=enable_blur_effects,
                enable_stylistic_effects=enable_stylistic_effects,
                hdr_intensity=hdr_intensity, shadow_intensity=shadow_intensity, highlight_intensity=highlight_intensity,
                gamma=gamma, brightness=brightness, contrast=contrast, saturation=saturation, enhance_color=enhance_color,
                lut_name=lut_name, lut_strength=lut_strength, lut_log_process=lut_log_process,
                sharpness=sharpness, edge_enhance_strength=edge_enhance_strength, detail_enhance_strength=detail_enhance_strength,
                blur_radius=blur_radius, gaussian_blur_radius=gaussian_blur_radius, radial_blur_strength=radial_blur_strength,
                radial_blur_center_x=radial_blur_center_x, radial_blur_center_y=radial_blur_center_y,
                radial_blur_focus_spread=radial_blur_focus_spread, radial_blur_steps=radial_blur_steps,
                chromatic_aberration_r_x=chromatic_aberration_r_x, chromatic_aberration_r_y=chromatic_aberration_r_y,
                chromatic_aberration_b_x=chromatic_aberration_b_x, chromatic_aberration_b_y=chromatic_aberration_b_y,
                chromatic_blur_amount=chromatic_blur_amount, simple_film_grain_intensity=simple_film_grain_intensity,
                simple_film_grain_monochrome=simple_film_grain_monochrome, scanline_intensity=scanline_intensity,
                vignette_intensity=vignette_intensity, vignette_center_x=vignette_center_x, vignette_center_y=vignette_center_y,
                soft_light_opacity=soft_light_opacity, soft_light_blur_radius=soft_light_blur_radius))

        return (final_images, face_debug_images)

__all__ = [
    "H3FL2VAandRef2VA",
    "H3SlidingContextSampler",
    "FilmAuteurH3",
    "FilmAuteur_LTX",
    "UltimateSDUpscaleMiniMaxH3",
]
