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

from .h3_latent_upscale import latent_upscale_model_names, upscale_h3_latent


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
        ref_image_size="mixed",
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

        # Cross-shot context is generated footage rather than a user reference, so it rides
        # along in every mode, including text, exactly as the original node did.
        if cross_shot_video_reference is not None:
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
        if ref_image_size == "mixed" and raw_ref_images and not character_pictures:
            # Only screenwriter definitions carry the character-sheet note. A hand-written H3
            # prompt, or one from another formatter, would otherwise drop every face to the
            # cheap size with no hint, so treat cited pictures as identity-bearing instead.
            if re.search(r"<Picture\s+\d+>", str(prompt or "")):
                character_pictures = set(range(1, len(raw_ref_images) + 1))
                print("[H3FL2VAandRef2VA] ref_image_size=mixed found no character-sheet note; "
                      "sizing every cited reference at the 2048px character target.")
            else:
                print("[H3FL2VAandRef2VA] ref_image_size=mixed found no <Picture N> citations; "
                      "references use the cheaper match size.")
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


def _denoise_sigmas(model, scheduler, steps, denoise=1.0):
    """Scheduler sigmas trimmed to ComfyUI's KSampler denoise semantics.

    denoise < 1 keeps the full step count but starts further down the schedule: a
    steps / denoise schedule is built and its last steps + 1 sigmas are used, so a
    latent that already carries structure (one handed in from another node) keeps
    part of it instead of being redrawn from noise.
    """
    steps = max(1, int(steps))
    denoise = float(denoise)
    if not 0.0 < denoise <= 1.0:
        raise ValueError("denoise must be greater than 0 and at most 1")
    model_sampling = model.get_model_object("model_sampling")
    if denoise >= 0.9999:
        return comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).cpu()
    schedule_steps = max(steps, int(steps / denoise))
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, schedule_steps).cpu()
    return sigmas[-(steps + 1):]


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
                transition_mode="auto", transition_prompt="", clip=None, story_beats="",
                enforce_cuts=False, denoise=1.0) -> io.NodeOutput:
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
        sigmas = _denoise_sigmas(model, scheduler, steps, denoise)
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
    r"(?:environment\s+)?in\s+<(?P<kind>Picture|Video)\s+(?P<source>\d+)>\s*[,:]?\s*"
    r"(?P<traits>.*)$",
    re.IGNORECASE,
)
_DEF_SUBJECT_RE = re.compile(r"^\s*<Subject\s+(\d+)>", re.IGNORECASE)

_TEXT_ONLY_SOURCE_RE = re.compile(
    r"<(?P<kind>Picture|Video|Audio)\s+(?P<number>\d+)>", re.IGNORECASE)
_TEXT_ONLY_SWATCH_CLAUSE_RE = re.compile(
    r"^a single row of\s+\d+\s+flat color swatches?\s*\((?P<colors>[^)]*)\)\.?\s*",
    re.IGNORECASE,
)


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


def _text_only_prompt(prompt, cast=None):
    """Rewrite a screenwriter prompt into MiniMax H3's text-to-video shape.

    Text mode attaches nothing, so the definitions block and the retention_analysis block
    describe pictures the model never receives: "stay exactly as defined in the reference"
    points at nothing, and identity prose parked in a distant block is weakly attended next
    to the sentence that actually describes the action. H3 follows that sentence, so each
    subject's traits move inline onto its first mention, the reference-only sections are
    dropped, and the prompt keeps the order the screenwriter emits for text-to-video:
    description first, then soundscape, then music.
    """
    text = str(prompt or "")
    if not _SUBJECT_LABEL_RE.search(text) and not _TEXT_ONLY_SOURCE_RE.search(text):
        return text
    cast = _text_only_cast(text) if cast is None else cast
    text = _drop_prompt_section(text, "subject_definitions")
    text = _drop_prompt_section(text, "retention_analysis")
    seen = set()
    text = _text_only_swap(text, cast, seen)

    lead = []
    palette = next((info for info in cast.values() if info["palette"] and info["traits"]), None)
    if palette is not None:
        lead.append(f"The frame's grade: {_capitalized(palette['traits'])}.")
    unmentioned = [info for number, info in sorted(cast.items(), key=lambda item: int(item[0]))
                   if number not in seen and not info["palette"]]
    setting = [info["first"] for info in unmentioned if info["kind"] == "video"]
    in_frame = [info["first"] for info in unmentioned if info["kind"] != "video"]
    if setting:
        lead.append("Setting: " + "; ".join(setting) + ".")
    if in_frame:
        lead.append("In frame: " + "; ".join(in_frame) + ".")

    shot_headers = list(_SHOT_PROMPT_HEADER.finditer(text))
    if shot_headers:
        # _prompt_for_shot joins the style paragraph and the shot line as the last parts, so
        # the description starts at the blank line before the final [Shot N].
        start, cursor = 0, shot_headers[-1].start()
        for _ in range(2):  # walk back past the blank line before the shot, then before the style
            boundary = text.rfind("\n\n", 0, cursor)
            if boundary < 0:
                break
            start, cursor = boundary + 2, boundary
        description = "\n\n".join(lead + [text[start:].strip()])
        sections = text[:start].strip()
        text = "\n\n".join(part for part in (
            f"integrated_multimodal_description:\n{description}", sections) if part)
    elif lead:
        text = "\n\n".join(lead + [text])
    return text.strip()


def _capitalized(text):
    text = str(text or "").strip()
    return text[:1].upper() + text[1:] if text else text


def _text_only_cast(prompt):
    """<Subject N> -> the wording text-to-video should use for it.

    Text-to-video attaches nothing, so `<Picture 1>` is not something the model can look at:
    prose is the only identity signal it gets. Every subject therefore carries its traits for
    the first mention, its plain name for later ones, and the picture/video number it came
    from, so a bare `<Picture 2>` citation inside a shot line still resolves to the right
    character.
    """
    cast = {}
    for line in str(prompt or "").splitlines():
        match = _DEFINITION_LINE_RE.match(line)
        if not match:
            continue
        name = _plain_subject_name(match.group("name"))
        traits = match.group("traits")
        traits = traits.replace(_CHARACTER_NOTE_TEXT, "").replace(_PALETTE_NOTE_TEXT, "")
        colors = ""
        swatches = _TEXT_ONLY_SWATCH_CLAUSE_RE.match(traits.strip())
        if swatches:
            # The swatch row is the chart, not the picture: keep its colours for the palette
            # sentence and never describe the chart itself to a model that cannot see it.
            colors = " ".join(swatches.group("colors").split())
            traits = traits.strip()[swatches.end():]
        traits = re.sub(r"\s+", " ", traits).strip(" .,;")
        palette = "color palette" in name.lower()
        if palette:
            term = f"the palette ({colors})" if colors else "the palette"
            first = term
        else:
            term = name
            first = f"{name} ({traits})" if traits else name
        cast[match.group("subject")] = {
            "name": term,
            "first": first,
            "kind": match.group("kind").lower(),
            "source": match.group("source"),
            "traits": traits,
            "palette": palette,
        }
    return cast


def _text_only_swap(text, cast, seen):
    """Replace subject and source labels, inlining traits on a subject's first mention."""
    if not isinstance(text, str) or not cast:
        return text
    by_source = {f"{info['kind']}{info['source']}": number
                 for number, info in cast.items()}

    def term(number):
        info = cast.get(number)
        if info is None:
            return ""
        if info["palette"] or number in seen:
            return info["name"]
        seen.add(number)
        return info["first"]

    def subject_label(match):
        return term(match.group(1)) if match.group(1) in cast else match.group(0)

    def source_label(match):
        number = by_source.get(f"{match.group('kind').lower()}{match.group('number')}")
        return term(number) if number else ""

    text = _SUBJECT_LABEL_RE.sub(subject_label, text)
    text = _TEXT_ONLY_SOURCE_RE.sub(source_label, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"[ \t]+([,.;:])", r"\1", text)


def _drop_prompt_section(text, name):
    """Remove one `name:` block, header included."""
    headers = list(_PROMPT_SECTION_HEADER.finditer(text))
    for index, header in enumerate(headers):
        if header.group("name").lower() != name:
            continue
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        return re.sub(r"\n{3,}", "\n\n", text[:header.start()] + text[end:]).strip("\n")
    return text


def _text_only_beats(beats, cast):
    """Rewrite one shot's beats with the same inline identity wording as the prompt.

    The beats reach the sampler as per-window direction, so `<Subject N>` there keeps citing
    references text mode never attaches, and a window that reads only a beat loses the
    character description entirely.
    """
    if not cast or not beats:
        return beats
    seen = set()
    if isinstance(beats, str):
        # One shot's parsed beats are a timestamped script, one `[00:00.000] text` line each,
        # not a list. Iterating a string would hand the sampler a list of characters, and every
        # timestamp has to survive byte-identical for the window plan.
        return "\n".join(_text_only_swap(line, cast, seen) for line in beats.splitlines())
    rewritten = []
    for item in beats:
        if isinstance(item, dict):
            rewritten.append({key: _text_only_swap(value, cast, seen)
                              for key, value in item.items()})
        elif isinstance(item, (list, tuple)):
            rewritten.append(type(item)(_text_only_swap(value, cast, seen) for value in item))
        else:
            rewritten.append(_text_only_swap(item, cast, seen))
    return rewritten


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
    if shot_total <= 1:
        return block
    if not used_subjects:
        print("[FilmAuteurH3] this shot cites no <Subject N>; only the palette definition is kept")
    lines = block.splitlines()
    header, body = "", []
    if lines and not _DEF_SUBJECT_RE.match(lines[0]):
        header, lines = lines[0], lines[1:]
    for line in lines:
        match = _DEF_SUBJECT_RE.match(line)
        if (match and int(match.group(1)) not in used_subjects
                and "environment in <video" not in line.lower()
                and "color palette" not in line.lower()):
            continue
        body.append(line)
    return "\n".join(([header] if header else []) + body).strip()


def _retention_section(block):
    """Retention lines for the subjects this shot actually cites.

    MiniMax H3's reference format reads `retention_analysis` to decide what must survive from
    each <Picture N>/<Video N>. Without it the model is free to restyle an identity or rebuild
    a set between shots, which is exactly how a character stops looking like themselves. Only
    subjects left after the per-shot filter are listed, and no reference label is repeated so
    text mode stays readable.
    """
    lines = []
    for line in (block or "").splitlines():
        match = _DEF_SUBJECT_RE.match(line)
        if not match:
            continue
        subject = f"<Subject {match.group(1)}>"
        low = line.lower()
        if "color palette" in low:
            lines.append(
                f"- {subject}: attribute_transfer. Only its colors, contrast and tonal look are "
                "taken from the palette; the swatch layout never appears in the frame."
            )
        elif "environment in <video" in low:
            lines.append(
                f"- {subject}: fully_preserved. The layout, architecture, set dressing, "
                "practicals and light direction stay exactly as established in the location "
                "reference; no new geometry, no relighting and no time-of-day drift."
            )
        elif _CHARACTER_NOTE_TEXT.lower() in low:
            lines.append(
                f"- {subject}: fully_preserved. Face, hair, eye color, skin tone, build, "
                "wardrobe and distinguishing marks stay exactly as defined; no restyling, no "
                "age or ethnicity drift, no wardrobe or hairstyle changes within the shot."
            )
        else:
            lines.append(
                f"- {subject}: fully_preserved. Shape, material, color, scale and wear stay "
                "exactly as defined in the reference; no redesign, recoloring or scale drift."
            )
    return "retention_analysis:\n" + "\n".join(lines) if lines else ""


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
            filtered = _filter_definitions(
                text[header.start():next_end].strip(), used_subjects, len(shot_headers))
            shared_parts.append(filtered)
            retention = _retention_section(filtered)
            if retention:
                shared_parts.append(retention)
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


H3_ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}


def aspect_megapixel_size(aspect_ratio: str, megapixels: float, multiple: int = 32):
    """Pick the height for a megapixel budget, then derive the width from the aspect ratio.

    The width is calculated from the height (21:9 -> width = height * 21/9) and both sides are
    snapped to `multiple`, so every pair is a valid size for H3 and the VAE.
    """
    text = str(aspect_ratio or "").strip()
    rw, rh = H3_ASPECT_RATIOS.get(text, (16, 9))
    area = max(float(megapixels), 1e-3) * 1_000_000.0
    height = max(multiple, int(round(math.sqrt(area * rh / float(rw)) / multiple)) * multiple)
    width = max(multiple, int(round(height * (rw / float(rh)) / multiple)) * multiple)
    limit = int(nodes.MAX_RESOLUTION)
    while max(width, height) > limit and height > multiple:
        height = max(multiple, height - multiple)
        width = max(multiple, int(round(height * (rw / float(rh)) / multiple)) * multiple)
    return int(width), int(height)


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
                io.Combo.Input(
                    "aspect_ratio",
                    options=["1:1 (Square)", "2:3 (Portrait Photo)", "3:2 (Photo)",
                             "3:4 (Portrait Standard)", "4:3 (Standard)",
                             "9:16 (Portrait Widescreen)", "16:9 (Widescreen)", "21:9 (Ultrawide)"],
                    default="16:9 (Widescreen)",
                    tooltip=(
                        "Frame shape of the render. Together with megapixels it sets the output "
                        "size: the height is chosen for the budget, the width follows the ratio "
                        "(21:9 -> width = height * 21/9) and both are snapped to a multiple of 32."
                    ),
                ),
                io.Float.Input(
                    "megapixels", default=0.5, min=0.05, max=8.0, step=0.05,
                    tooltip=(
                        "Output size in megapixels; 0.5 is about 960x544 at 16:9. The aspect "
                        "ratio decides the shape and both sides are snapped to a multiple of 32."
                    ),
                ),
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
                    tooltip=(
                        "Base seed for the whole render. Shot N samples with seed + N - 1 so "
                        "neighbouring shots never share one noise field; to reproduce a single "
                        "shot on its own, set the base seed back by that shot's index - 1."
                    ),
                ),
                io.Int.Input("steps", default=20, min=1, max=1000),
                io.Combo.Input(
                    "sampler_name", options=comfy.samplers.SAMPLER_NAMES,
                    default="euler",
                ),
                io.Combo.Input(
                    "scheduler", options=comfy.samplers.SCHEDULER_NAMES, default="beta",
                ),
                io.Float.Input(
                    "denoise", default=1.0, min=0.01, max=1.0, step=0.01,
                    tooltip=(
                        "Fraction of the sampler schedule each shot runs, exactly like a "
                        "KSampler denoise: the step count is unchanged but sampling starts "
                        "part-way down the scheduler. 1.0 renders from pure noise. Below "
                        "1 whatever structure the incoming latent already holds is kept "
                        "further into the render; on an empty H3 latent it only leaves "
                        "the shot under-denoised, so leave it at 1.0 for plain renders."
                    ),
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
                    "cross_shot_context_frames", default=192, min=22, max=192, step=17,
                    tooltip=(
                        "Requested prior-shot context in frames (24 fps); rounded up to H3's "
                        "17k+5 frame grid. The available tail is used if the previous shot is "
                        "shorter. A high value carries more of the previous shot's casting, set "
                        "and lighting into the next one."
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
        aspect_ratio,
        megapixels,
        length_seconds,
        model,
        seed,
        steps,
        sampler_name,
        scheduler,
        denoise,
        window_frames,
        overlap_frames,
        sliding_windows,
        shot_lengths=None,
        window_frames_mode="automatic",
        output_mode="video",
        mode="reference",
        keyframes=None,
        ref_image_size="mixed",
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
        cross_shot_context_frames=192,
        voice_continuity=True,
    ) -> io.NodeOutput:
        # V3 list-input mode is required to receive the screenwriter's list-valued
        # INT output as one shot-length array. Unwrap all other single inputs.
        values = {
            "clip": clip, "vae": vae, "audio_vae": audio_vae, "prompt": prompt,
            "aspect_ratio": aspect_ratio, "megapixels": megapixels,
            "length_seconds": length_seconds, "model": model,
            "output_mode": output_mode,
            "seed": seed, "steps": steps, "sampler_name": sampler_name,
            "scheduler": scheduler, "denoise": denoise, "window_frames": window_frames,
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
        values["width"], values["height"] = aspect_megapixel_size(
            values["aspect_ratio"], values["megapixels"]
        )
        print(
            f"[FilmAuteurH3] {values['aspect_ratio']} at {float(values['megapixels']):.2f} MP "
            f"-> {values['width']}x{values['height']}"
        )
        if values["output_mode"] not in {"video", "image"}:
            raise ValueError("output_mode must be video or image")
        values["denoise"] = float(values["denoise"])
        if not 0.0 < values["denoise"] <= 1.0:
            raise ValueError("denoise must be greater than 0 and at most 1")
        if values["denoise"] < 0.9999:
            print(
                f"[FilmAuteurH3] denoise {values['denoise']:.2f}: every shot keeps its "
                f"{values['steps']} steps but starts part-way down the scheduler, so the "
                "anything already in the starting latent survives further into the render."
            )
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
                cast = _text_only_cast(shot_prompt)
                shot_beats = _text_only_beats(shot_beats, cast)
                shot_prompt = _text_only_prompt(shot_prompt, cast)
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

            cross_shot_video_reference = None
            cross_shot_video_audio = None
            # The previous shot's generated tail is a video reference for this one, sized by
            # cross_shot_context_frames. It is not pinned to the opening frames, so casting,
            # set and lighting carry over while this shot still starts on its own action.
            if values["cross_shot_context"] and render_index > 1 and shot_index > 1 and not image_mode:
                previous_samples = sampled_outputs[-1]["samples"]
                if not _is_h3_latent(previous_samples):
                    raise ValueError("cross-shot continuity needs a sampled H3 AV latent")
                cross_shot_video_reference = _tail_video_reference_frames(
                    previous_samples,
                    int(values["cross_shot_context_frames"]),
                    values["vae"],
                )
                if values["voice_continuity"]:
                    cross_shot_video_audio = _tail_audio_context(
                        previous_samples, int(values["cross_shot_context_frames"])
                    )
                print(
                    f"[FilmAuteurH3] cross-shot: {progress} carries the previous shot's "
                    f"{int(values['cross_shot_context_frames'])}-frame tail as a video reference."
                )
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
                denoise=values["denoise"],
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
try:
    from .dlss5.settings import (
        DLSS_MODEL_PRESETS,
        MOTION_MODES,
        NR_PRESETS,
        NR_STYLES,
        UPSCALING_LABELS,
    )
    DLSS5_AVAILABLE = True
except ImportError:  # the DLSS 5 pass is optional: only its node needs the package
    DLSS5_AVAILABLE = False
    # Placeholders keep this module, and the upscale node's schema, importable without the
    # package. Asking for enable_dlss5 then raises with the install hint instead of breaking
    # every other node in the file at import time.
    DLSS_MODEL_PRESETS = ("M",)
    MOTION_MODES = ("auto",)
    NR_PRESETS = ("Default",)
    NR_STYLES = ("Default",)
    UPSCALING_LABELS = ("Standard",)

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


# The reference workflow's MVEx_MaskCleanup: erode 12 px off the segmented mask, then grow it back
# 16 px. Eroding first drops the background fringe SAM leaves on the face boundary; growing after
# restores a blend margin. FACE_MASK_TEMPORAL is the majority filter over each track's masks, which
# stands in for the temporal memory of a video tracker.
FACE_MASK_SHRINK = 12
FACE_MASK_GROW = 16
FACE_MASK_TEMPORAL = 2


def place_instance_mask(inst: np.ndarray, fx: int, fy: int, x1: int, y1: int, crop_h: int, crop_w: int) -> np.ndarray:
    """Place a bbox-sized instance mask (bbox at fx,fy) into a crop whose origin is (x1, y1)."""
    crop_h, crop_w = int(round(crop_h)), int(round(crop_w))
    fx, fy, x1, y1 = int(round(fx)), int(round(fy)), int(round(x1)), int(round(y1))
    full = np.zeros((crop_h, crop_w), np.float32)
    mh, mw = inst.shape
    ox, oy = fx - x1, fy - y1
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    cw_, ch_ = min(mw - sx0, crop_w - dx0), min(mh - sy0, crop_h - dy0)
    if cw_ > 0 and ch_ > 0:
        full[dy0:dy0 + ch_, dx0:dx0 + cw_] = inst[sy0:sy0 + ch_, sx0:sx0 + cw_]
    return np.clip(full, 0.0, 1.0)


def cleanup_instance_mask(full: np.ndarray, shrink: int = FACE_MASK_SHRINK, grow: int = FACE_MASK_GROW) -> np.ndarray:
    """Erode then dilate a placed mask (the workflow's shrink_grow cleanup). Returns float32 in [0,1]."""
    binary = (full > 0.5).astype(np.uint8)
    if shrink > 0:
        binary = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * shrink + 1, 2 * shrink + 1)))
    if grow > 0:
        binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    if not binary.any():
        # A face thinner than 2 x shrink (a distant background head) would vanish; keep the raw mask.
        binary = (full > 0.5).astype(np.uint8)
    return binary.astype(np.float32)


def instance_blend_mask(inst: np.ndarray, fx: int, fy: int, x1: int, y1: int, crop_h: int, crop_w: int,
                        device) -> torch.Tensor:
    """Place a SAM instance mask (bbox at fx,fy) into the crop (origin x1,y1), grow it a little and feather it.
    Only this region is replaced by the face pass; the rest of the crop (shoulders, body, background) is untouched."""
    fx, fy, x1, y1 = int(round(fx)), int(round(fy)), int(round(x1)), int(round(y1))
    crop_h, crop_w = int(round(crop_h)), int(round(crop_w))
    full = place_instance_mask(inst, fx, fy, x1, y1, crop_h, crop_w)
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


# How much geometry a frame's redraw may be corrected by. The diffused face is re-drawn for every
# frame, so it drifts against the crop it was conditioned on in rotation, scale and a little shear
# as well as in translation; a translation-only fit leaves that residue behind, and the residue is
# what still reads as the face stuttering once the sliding is gone.
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


def seam_tone_delta(previous: np.ndarray, current: np.ndarray, max_side: int = 160):
    """Coarse tone/colour difference between two renders of the same frames, and how far they drift.

    `previous` is what the clip already shows at those global frames, `current` this chunk's fresh
    render of them ([N,H,W,3] float32 in 0..1). The difference is measured on a small, blurred copy so
    it carries exposure, colour and a little regional gradient - never detail, which is exactly what
    must not be copied from one render into the other. Returns (delta, stats), delta being [h,w,3]
    (previous - current) ready to be added to the new frames, or (None, None) when there is nothing
    to compare. `stats` also carries where the two renders think the content is, with the correlation
    score that estimate is worth, so a geometric drift can be told from a tonal or detail one - and
    told apart from a measurement that simply failed.
    """
    prev = np.asarray(previous, dtype=np.float32)[..., :3]
    cur = np.asarray(current, dtype=np.float32)[..., :3]
    if prev.ndim != 4 or cur.ndim != 4 or prev.shape[1:3] != cur.shape[1:3]:
        return None, None
    count = int(min(prev.shape[0], cur.shape[0]))
    h, w = int(prev.shape[1]), int(prev.shape[2])
    if count <= 0 or h < 2 or w < 2:
        return None, None
    prev, cur = prev[:count], cur[:count]
    scale = float(max_side) / float(max(h, w))
    hh, ww = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
    small_p = np.stack([cv2.resize(f, (ww, hh), interpolation=cv2.INTER_AREA) for f in prev])
    small_c = np.stack([cv2.resize(f, (ww, hh), interpolation=cv2.INTER_AREA) for f in cur])
    sigma = max(1.5, min(hh, ww) / 12.0)
    blur_p = np.stack([cv2.GaussianBlur(f, (0, 0), sigma) for f in small_p])
    blur_c = np.stack([cv2.GaussianBlur(f, (0, 0), sigma) for f in small_c])
    delta = (blur_p - blur_c).mean(axis=0).astype(np.float32)
    before = float(np.abs(small_c - small_p).mean())
    after = float(np.abs(np.clip(small_c + delta, 0.0, 1.0) - small_p).mean())
    # What the coarse measurement throws away: how far the two renders disagree about detail
    # with the exposure and the layout taken out. Tone, detail and position are the three ways a
    # seam can read as a change, and they are logged separately so a report can be read.
    detail = float(np.abs((small_c - blur_c) - (small_p - blur_p)).mean())
    rgb = [float(v) for v in delta.reshape(-1, 3).mean(axis=0)]
    # Where the two renders think the content is. A translation has to be measurable or omitted:
    # phase correlation, which this used to do, has no real peak on a pair of coarse, nearly flat
    # frames and answers differently every time - numbers large enough to look as if the chunks had
    # been cut from different plates. Matching a central patch of this render inside the previous one
    # instead bounds the search to the neighbourhood they could plausibly disagree by, and reports the
    # correlation peak next to the answer so a number can be disbelieved instead of acted on.
    geo_h = max(8, min(320, h))
    geo_w = max(8, int(round(w * geo_h / float(h))))
    geo_p = np.stack([cv2.resize(f, (geo_w, geo_h), interpolation=cv2.INTER_AREA) for f in prev])
    geo_c = np.stack([cv2.resize(f, (geo_w, geo_h), interpolation=cv2.INTER_AREA) for f in cur])

    def translation(index):
        """(dx, dy, peak): this render's frame against the previous render's same frame."""
        try:
            g_p = cv2.cvtColor(np.clip(geo_p[index], 0, 1) * 255.0, cv2.COLOR_RGB2GRAY).astype(np.float32)
            g_c = cv2.cvtColor(np.clip(geo_c[index], 0, 1) * 255.0, cv2.COLOR_RGB2GRAY).astype(np.float32)
        except cv2.error:
            return None, None, 0.0
        g_p = cv2.GaussianBlur(g_p, (0, 0), 1.0)   # kill the finest grain, keep the structure
        g_c = cv2.GaussianBlur(g_c, (0, 0), 1.0)
        ph, pw = g_p.shape
        pad = max(3, int(round(min(ph, pw) * 0.15)))   # as far apart as two renders could sit
        if ph - 2 * pad < 8 or pw - 2 * pad < 8:
            return None, None, 0.0
        patch = g_c[pad:ph - pad, pad:pw - pad]
        # A flat patch has no structure to line up, and a normalised correlation will happily
        # report a confident match anywhere on it. Say so instead.
        if float(patch.std()) < 6.0:
            return None, None, 0.0
        surface = cv2.matchTemplate(g_p, patch, cv2.TM_CCOEFF_NORMED)
        _low, peak, _low_at, at = cv2.minMaxLoc(surface)
        sx, sy = int(at[0]), int(at[1])

        def sub(ym, y0, yp):
            denom = float(ym) - 2.0 * float(y0) + float(yp)
            return 0.0 if abs(denom) < 1e-6 else 0.5 * (float(ym) - float(yp)) / denom

        sub_x = (sub(surface[sy, sx - 1], surface[sy, sx], surface[sy, sx + 1])
                 if 0 < sx < surface.shape[1] - 1 else 0.0)
        sub_y = (sub(surface[sy - 1, sx], surface[sy, sx], surface[sy + 1, sx])
                 if 0 < sy < surface.shape[0] - 1 else 0.0)
        # A peak on the edge of the search means the real offset is outside it, and a weak peak means
        # the two frames have nothing in common to line up: both are reported as "no estimate".
        if float(peak) < 0.5 or sx in (0, surface.shape[1] - 1) or sy in (0, surface.shape[0] - 1):
            return None, None, float(peak)
        return (float(pad - (sx + sub_x)) * w / float(pw),
                float(pad - (sy + sub_y)) * h / float(ph), float(peak))

    attempts = [translation(i) for i in range(count)]
    usable = [(dx, dy) for dx, dy, _peak in attempts if dx is not None]
    peak = max([p for _dx, _dy, p in attempts] or [0.0])
    shift = None
    if usable:
        shift = (float(np.median([d[0] for d in usable])), float(np.median([d[1] for d in usable])))
    return delta, {"before": before, "after": after, "detail": detail, "rgb": rgb,
                   "shift": shift, "match": peak}


def seam_similarity(previous: np.ndarray, current: np.ndarray, max_side: int = 320):
    """How two renders of the same frames sit relative to each other, scale included.

    `seam_tone_delta` reads where the content is by matching a patch, which finds a translation and
    nothing else. A render that redrew the shot a percent larger lands on the frame centre, scores
    a fraction of a pixel and looks aligned - which is exactly how a seam that visibly re-frames
    reads as "no shift". So this fits a full affine (ECC) over the same pair of frames, keeps only
    fits that are close to a similarity (a strong shear or rotation is two renders disagreeing
    about content, not one of them sitting slightly large) and reports the scale, the angle, where
    the frame centre ended up and how far the corners moved. Returns None when nothing was fitted,
    rather than a number made of grain.
    """
    prev = np.asarray(previous, dtype=np.float32)[..., :3]
    cur = np.asarray(current, dtype=np.float32)[..., :3]
    if prev.ndim != 4 or cur.ndim != 4 or prev.shape[1:3] != cur.shape[1:3]:
        return None
    if not hasattr(cv2, "findTransformECC"):
        return None
    count = int(min(prev.shape[0], cur.shape[0]))
    h, w = int(prev.shape[1]), int(prev.shape[2])
    if count <= 0 or h < 16 or w < 16:
        return None
    ratio = float(max_side) / float(max(h, w))
    hh, ww = max(16, int(round(h * ratio))), max(16, int(round(w * ratio)))
    fx, fy = float(w) / float(ww), float(h) / float(hh)

    def gray(frames):
        out = []
        for frame in frames[:count]:
            small = cv2.resize(frame, (ww, hh), interpolation=cv2.INTER_AREA)
            mono = cv2.cvtColor(np.clip(small, 0.0, 1.0) * 255.0, cv2.COLOR_RGB2GRAY)
            out.append(cv2.GaussianBlur(mono, (0, 0), 1.2).astype(np.float32) / 255.0)
        return out

    g_prev, g_cur = gray(prev), gray(cur)
    scores, warps = [], []
    grid = np.diag([fx, fy])
    for i in range(count):
        # A flat pair has no structure to fit and an affine fit on one is noise with a matrix.
        if float(g_prev[i].std()) < 0.02 or float(g_cur[i].std()) < 0.02:
            continue
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            score, warp = cv2.findTransformECC(
                g_prev[i], g_cur[i], warp, cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 3,
            )
        except cv2.error:
            continue
        linear = warp[:, :2].astype(np.float64)
        det = float(np.linalg.det(linear))
        if not np.isfinite(det) or det <= 1e-4:
            continue
        spread = np.linalg.svd(linear, compute_uv=False)
        if float(spread[0]) / max(1e-6, float(spread[-1])) > 1.06:
            continue
        # The fit is in the coordinates of the small copies; carry it back to output pixels by
        # conjugating with the resize, so the matrix applies to the frames as they are.
        full = np.concatenate([
            grid @ linear @ np.linalg.inv(grid),
            (grid @ warp[:, 2].astype(np.float64)).reshape(2, 1),
        ], axis=1).astype(np.float32)
        scores.append(float(score))
        warps.append(full)
    if not warps:
        return None
    warp = np.median(np.stack(warps), axis=0).astype(np.float32)
    linear = warp[:, :2].astype(np.float64)
    det = float(np.linalg.det(linear))
    if not np.isfinite(det) or det <= 1e-4:
        return None
    centre = np.array([w * 0.5, h * 0.5], dtype=np.float64)
    shift = linear @ centre + warp[:, 2] - centre
    corners = np.array([[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]], dtype=np.float64)
    moved = float(np.abs((corners @ linear.T + warp[:, 2]) - corners).max())
    return {
        "scale": float(np.sqrt(det)),
        "rotation": float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))),
        "shift": (float(shift[0]), float(shift[1])),
        "moved": moved,   # how far the worst corner lands from where it was
        "match": float(np.median(scores)),
        "warp": warp,
    }



def seam_framing_split(previous: np.ndarray, current: np.ndarray, max_side: int = 384):
    """How two renders sit against each other, in the middle and at the edge separately.

    A whole-frame fit is dominated by whatever covers the frame, so a subject that lands a few
    percent larger while the world holds reads as a fraction of a percent, and a warp of that size
    changes nothing visible. Splitting the frame into its middle half and the ring around it makes
    those two answers separate numbers: a re-framed shot scales both by the same amount, a re-staged
    subject scales the middle and leaves the ring alone. Each fit is given the whole of its own
    region rather than a small patch, because a free affine over a 128 pixel cell has more freedom
    than that much content has structure - nine of them disagree by ten percent between neighbours
    covering the same shot, which is a property of the fit and not of the picture.

    Returns {"whole", "centre", "edge"}, each {"scale_pct", "shift_px", "match", "moved_px"}, or
    None when nothing could be fitted.
    """
    prev = np.asarray(previous, dtype=np.float32)[..., :3]
    cur = np.asarray(current, dtype=np.float32)[..., :3]
    if prev.ndim != 4 or cur.ndim != 4 or prev.shape[1:3] != cur.shape[1:3]:
        return None
    if not hasattr(cv2, "findTransformECC"):
        return None
    count = int(min(prev.shape[0], cur.shape[0]))
    h, w = int(prev.shape[1]), int(prev.shape[2])
    if count <= 0 or h < 64 or w < 64:
        return None
    ratio = min(1.0, float(max_side) / float(max(h, w)))
    hh, ww = max(64, int(round(h * ratio))), max(64, int(round(w * ratio)))
    fx, fy = float(w) / float(ww), float(h) / float(hh)
    grid = np.diag([fx, fy])

    def grey(frames):
        # Single channel, the way this OpenCV needs it; a colour frame is refused outright by
        # every build before 5.x, and silently refitting nothing looks exactly like agreement.
        out = []
        for frame in frames[:count]:
            part = cv2.resize(frame, (ww, hh), interpolation=cv2.INTER_AREA)
            mono = cv2.cvtColor(np.clip(part, 0.0, 1.0) * 255.0, cv2.COLOR_RGB2GRAY)
            out.append(cv2.GaussianBlur(mono, (0, 0), 1.2).astype(np.float32) / 255.0)
        return out

    g_prev, g_cur = grey(prev), grey(cur)
    middle = np.zeros((hh, ww, 1), dtype=np.uint8)
    middle[hh // 4: hh - hh // 4, ww // 4: ww - ww // 4] = 255
    ring = np.where(middle > 0, np.uint8(0), np.uint8(255)).reshape(hh, ww, 1)
    regions = (("whole", None), ("centre", middle), ("edge", ring))
    out = {}
    centre_full = np.array([w * 0.5, h * 0.5], dtype=np.float64)
    corners = np.array([[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]], dtype=np.float64)
    for name, mask in regions:
        warps, scores = [], []
        for i in range(count):
            if float(g_prev[i].std()) < 0.02 or float(g_cur[i].std()) < 0.02:
                continue
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                score, warp = cv2.findTransformECC(
                    g_prev[i], g_cur[i], warp, cv2.MOTION_AFFINE,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), mask, 3,
                )
            except Exception:
                continue
            linear = warp[:, :2].astype(np.float64)
            det = float(np.linalg.det(linear))
            if not np.isfinite(det) or det <= 1e-4:
                continue
            warps.append(np.concatenate([
                grid @ linear @ np.linalg.inv(grid),
                (grid @ warp[:, 2].astype(np.float64)).reshape(2, 1),
            ], axis=1))
            scores.append(float(score))
        if not warps:
            continue
        warp = np.median(np.stack(warps), axis=0)
        linear = warp[:, :2]
        det = float(np.linalg.det(linear))
        if not np.isfinite(det) or det <= 1e-4:
            continue
        shift = linear @ centre_full + warp[:, 2] - centre_full
        moved = float(np.abs((corners @ linear.T + warp[:, 2]) - corners).max())
        out[name] = {
            "scale_pct": (float(np.sqrt(det)) - 1.0) * 100.0,
            "shift_px": (float(shift[0]), float(shift[1])),
            "match": float(np.median(scores)),
            "moved_px": moved,
        }
    return out or None


PIXEL_BATCH_SEAM_DUMP = True
PIXEL_BATCH_SEAM_DETAIL_MATCH = True


def match_detail_level(frames, gain):
    """Scale a chunk's fine detail to the level of the frames it follows, moving nothing.

    A head drawn at a different sharpness differs from the frame on screen at every pixel by a
    fraction of one, so no translation, scale or framing figure can see it, and it is read as the
    whole picture changing rather than something inside it. The band scaled here is the one the
    seam's own fineness line reads (a blur of 1.2 px on a frame 480 px across on its long side,
    which is `sigma` at full resolution), and there is one figure for the whole chunk: a constant
    on every frame cannot put a step anywhere inside the chunk, and nothing from the neighbouring
    render is mixed in, so this cannot ghost.
    """
    height, width = int(frames.shape[1]), int(frames.shape[2])
    sigma = 1.2 * float(max(height, width)) / 480.0
    for at in range(int(frames.shape[0])):
        frame = frames[at].detach().float().cpu().numpy()
        low = cv2.GaussianBlur(frame, (0, 0), sigma)
        frames[at] = torch.from_numpy(
            np.clip(low + float(gain) * (frame - low), 0.0, 1.0)
        ).to(device=frames.device, dtype=frames.dtype)
    return frames


# Edited footage has hard cuts in it, and a seam on a cut cannot be seen: the picture changes
# completely there, so a step of tone or sharpness has nothing to be compared against. Where a cut
# is found the seam is moved onto it and the two chunks stop sharing frames - no overlap, no context
# frames from the shot before, nothing redrawn - because feeding the previous shot's pixels into the
# next shot is worse than any seam. What still carries across is the grade, matched from frames both
# renders drew for the same timestamps (the previous chunk's look-ahead); tone is left to each shot's
# own source, because matching a new shot's level to the old one's would move it away from what it
# is. A continuous take has no cuts and is left exactly as it was: there is nothing to snap to, and
# the plane matching does the work there instead.
PIXEL_BATCH_SCENE_CUTS = True


def detect_scene_cuts(image, thumb: int = 96, floor: float = 0.12,
                      spike: float = 4.0, gap: int = 3):
    """Frame numbers where a new shot starts, read off the source clip.

    A cut changes every pixel at once; a pan moves them all a little. At a thumbnail 96 pixels across
    a 27 px a frame pan is a pixel and a half of travel, so at that size a cut falls the correlation
    to nothing while a pan leaves almost all of it. The limit is the clip's own median plus a
    multiple of its spread, so footage that moves hard throughout raises its own bar, and each
    candidate must also stand several times above the frames immediately around it - which is what
    tells a cut from a fast passage inside one shot. Returns 0-based frame numbers: frame C is the
    first frame of the new shot.
    """
    if int(image.ndim) != 4:
        return []
    count = int(image.shape[0])
    if count < 4:
        return []
    height, wide = int(image.shape[1]), int(image.shape[2])
    scale = min(1.0, float(thumb) / float(max(1, height, wide)))
    thumbs = []
    # One frame at a time: a clip on the GPU is gigabytes, and the whole of it is never needed -
    # the thumbnail of each frame is a few hundred pixels, and they are all that is compared.
    for at in range(count):
        frame = image[at]
        try:
            frame = frame.detach().float().cpu().numpy()
        except AttributeError:
            frame = np.asarray(frame, dtype=np.float32)
        if (frame.ndim == 3 and frame.shape[0] in (1, 3, 4)
                and frame.shape[-1] not in (1, 3, 4)):
            frame = np.moveaxis(frame, 0, -1)
        frame = np.asarray(frame[..., :3], dtype=np.float32).mean(axis=-1)
        if scale < 1.0:
            frame = cv2.resize(
                frame, (max(8, int(round(wide * scale))), max(8, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA)
        thumbs.append(np.asarray(frame, dtype=np.float32).reshape(-1))
    grey = np.stack(thumbs)
    grey = grey - grey.mean(axis=1, keepdims=True)
    energy = np.sqrt((grey ** 2).sum(axis=1))
    corr = (grey[:-1] * grey[1:]).sum(axis=1) / np.maximum(energy[:-1] * energy[1:], 1e-6)
    diff = np.clip(1.0 - corr, 0.0, 2.0)
    level = float(np.median(diff))
    spread = float(np.median(np.abs(diff - level))) * 1.4826
    limit = max(float(floor), level + spike * max(spread, 0.005))
    cuts = []
    for at in range(1, count):
        value = float(diff[at - 1])
        if value < limit:
            continue
        lo, hi = max(0, at - 1 - gap), min(int(diff.shape[0]), at - 1 + gap + 1)
        near = np.concatenate([diff[lo:at - 1], diff[at:hi]])
        local = float(np.median(near)) if near.size else level
        if value < 3.0 * max(local, 1e-3):
            continue
        if cuts and at - cuts[-1] <= 2:
            # Two frames flagged for one cut: the boundary belongs on the first of them.
            continue
        cuts.append(int(at))
    return cuts


def save_seam_frames(previous, chosen, unused, cut_frame, control=None, folder=None):
    """Write the frames a cut is actually made between, so a seam can be looked at, not inferred.

    Every figure this node reports about a seam is a fit - a translation, a scale, a correlation,
    a detail energy. None of them can say what a viewer sees, and the three things that look the
    same to a fit look completely different in a difference image: a rigid step doubles every
    edge in the frame, a re-drawn frame shows texture difference everywhere with the edges still
    single, and a light that moved shows a soft field with no edges at all. So one seam is worth
    four PNGs: the last frame already on screen, the first frame after the cut, the frame the
    previous render had for the same index and was not used, and the difference between the
    first two amplified eight times so that a sub-pixel step is plain. Written to the pack's
    temp folder rather than the output folder, because these are working frames and not a render.
    A raw difference of two frames a frame apart is the picture moving, which on a shot that
    pans is most of the frame: amplified it is an edge map of the scene and says nothing. So
    `_residual.png` is the same pair with the fit between them undone - the movement taken
    out, so what is left is what moved *differently* - and `_control.png` runs that same
    reading over an ordinary pair from inside the previous render, because a residual means
    nothing against black and everything against what this clip normally leaves.
    Returns {folder, cut, control} with the two residual figures, or None.
    """
    if folder is None:
        try:
            import folder_paths
            folder = os.path.join(folder_paths.get_temp_directory(), "h3_seams")
        except Exception:
            import tempfile
            folder = os.path.join(tempfile.gettempdir(), "h3_seams")
    try:
        os.makedirs(folder, exist_ok=True)
        prev = np.asarray(previous, dtype=np.float32)[..., :3]
        cur = np.asarray(chosen, dtype=np.float32)[..., :3]
        if prev.shape != cur.shape or prev.ndim != 3:
            return None
        height, width = int(prev.shape[0]), int(prev.shape[1])

        def settled(first, second):
            """The two frames with the fit between them undone, and how much difference is left.

            The fit is applied to the second frame and the remainder measured, so a pair that only
            sits differently goes quiet while a pair that was drawn differently keeps its
            difference where it is. The matrix and its inverse are both tried and the quieter one
            kept, because which direction applies is a convention of the OpenCV build rather than
            a property of the frames. Returns (difference, rms) or (None, None).
            """
            fit = seam_similarity(first[None], second[None])
            if fit is None or "warp" not in fit:
                return None, None
            matrix = np.asarray(fit["warp"], dtype=np.float32)
            best = (None, None)
            for candidate in (matrix, cv2.invertAffineTransform(matrix)):
                moved = cv2.warpAffine(second, candidate, (width, height),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                difference = np.abs(np.clip(first, 0.0, 1.0) - np.clip(moved, 0.0, 1.0))
                difference = difference.mean(axis=-1)
                rms = float(np.sqrt(np.mean(np.square(difference))))
                if best[0] is None or rms < best[1]:
                    best = (difference, rms)
            return best

        pictures = [("prev", prev), ("next", cur)]
        if unused is not None:
            alt = np.asarray(unused, dtype=np.float32)[..., :3]
            if alt.shape == prev.shape:
                pictures.append(("unused", alt))
        # Mean over colour, so a slight hue shift does not read as a big difference everywhere.
        pictures.append(("diff", np.repeat(
            (np.abs(cur - prev).mean(axis=-1) * 8.0)[..., None], 3, axis=-1)))
        residual, cut_rms = settled(prev, cur)
        if residual is not None:
            pictures.append(("residual", np.repeat(
                (residual * 8.0)[..., None], 3, axis=-1)))
        control_rms = None
        if control is not None:
            first = np.asarray(control[0], dtype=np.float32)[..., :3]
            second = np.asarray(control[1], dtype=np.float32)[..., :3]
            if first.shape == prev.shape and second.shape == prev.shape:
                ordinary, control_rms = settled(first, second)
                if ordinary is not None:
                    pictures.append(("control", np.repeat(
                        (ordinary * 8.0)[..., None], 3, axis=-1)))
        for name, picture in pictures:
            data = np.clip(np.asarray(picture, dtype=np.float32), 0.0, 1.0) * 255.0
            cv2.imwrite(os.path.join(folder, "h3_seam_%04d_%s.png" % (int(cut_frame), name)),
                        cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_RGB2BGR))
    except Exception as error:   # a dump must never be the reason a render fails
        logger.warning("[H3 Upscale] could not write the frames at the cut: %s", error)
        return None
    return {"folder": folder, "cut": cut_rms, "control": control_rms}


def frame_change_profile(out_frames, src_frames, max_side: int = 128):
    """Frame to frame change of a redraw against the source's own change, in screen order.

    A chunk seam is a jump between two renders and is measured where they meet (seam_tone_delta).
    This measures inside one render, which is where a jump that does not follow the chunking shows
    up. Both sides are averaged down to a small copy first, so the figure carries the picture
    moving rather than which render keeps more grain, and each side is compared against itself so
    ordinary shot motion cancels. Returns (ratio, out_change, src_change), one entry per pair of
    consecutive frames, or None when there is nothing to compare.
    """
    out = np.asarray(out_frames, dtype=np.float32)[..., :3]
    src = np.asarray(src_frames, dtype=np.float32)[..., :3]
    if out.ndim != 4 or src.ndim != 4:
        return None
    count = int(min(out.shape[0], src.shape[0]))
    if count < 3:
        return None

    def changes(frames):
        small = []
        for frame in frames[:count]:
            h, w = int(frame.shape[0]), int(frame.shape[1])
            if h < 8 or w < 8:
                return None
            scale = float(max_side) / float(max(h, w))
            size = (max(8, int(round(w * scale))), max(8, int(round(h * scale))))
            small.append(np.asarray(cv2.resize(frame, size, interpolation=cv2.INTER_AREA),
                                    dtype=np.float32).mean(axis=-1))
        stack = np.stack(small)
        return np.abs(stack[1:] - stack[:-1]).mean(axis=(1, 2)).astype(np.float32)

    out_change = changes(out)
    src_change = changes(src)
    if out_change is None or src_change is None:
        return None
    return out_change / np.maximum(src_change, 1e-4), out_change, src_change


def reconcile_decoded_frames(decoded: torch.Tensor, frame_count: int) -> torch.Tensor:
    """Fit a video VAE's decoded frame count back onto the input frame count, 1:1 by index."""
    decoded_count = int(decoded.shape[0])
    missing = int(frame_count) - decoded_count
    if missing == 0:
        return decoded
    # A causal video VAE drops the last frame or two of the sequence it was given. Stretching the
    # survivors back over the input grid (the old trilinear-over-time path) squashes the whole chunk
    # by missing/frame_count, so output frame j holds content from j - missing*j/frame_count: the
    # drift grows towards the end of a chunk, and every chunk restarts it at zero. Two chunks that
    # share frames then disagree about which frame holds what, by up to `missing` frames of motion,
    # so their overlap can never line up. Padding the trimmed tail back on keeps decoded frame j on
    # input frame j, which is what makes chunk seams register. Only a real frame-count change (a VAE
    # running on another frame grid) is still stretched.
    boundary_trim = max(4, int(frame_count) // 8)
    if 0 < missing <= boundary_trim:
        return torch.cat([decoded, decoded[-1:].expand(missing, *decoded.shape[1:])], dim=0)
    if -boundary_trim <= missing < 0:
        return decoded[:int(frame_count)]
    logger.warning(
        "[H3 Upscale] video batch: the VAE returned %d frame(s) for %d (%+d), too much to "
        "pad, so the sequence is being stretched over the frames it was asked for - its "
        "content now drifts against the neighbouring chunk, which reads as a jump at the "
        "seam. H3 renders on a frame grid of %d + %d k (%d, %d, %d ...): use a chunk "
        "length on that grid",
        decoded_count, int(frame_count), -missing, H3_FRAME_GRID_BASE, H3_FRAME_GRID_STEP,
        H3_FRAME_GRID_BASE, H3_FRAME_GRID_BASE + H3_FRAME_GRID_STEP,
        H3_FRAME_GRID_BASE + 2 * H3_FRAME_GRID_STEP,
    )
    temporal = decoded.unsqueeze(0).permute(0, 4, 1, 2, 3)
    temporal = F.interpolate(
        temporal, size=(int(frame_count), decoded.shape[1], decoded.shape[2]),
        mode="trilinear", align_corners=True,
    )
    return temporal.squeeze(0).permute(1, 2, 3, 0)


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


# YOLO resizes its input to `imgsz` (1280 in the global pass), so feeding it a 4K array buys
# nothing: the detection copy is capped to this long side, which only removes work.
YOLO_DETECT_MAX_SIDE = 1280
# While a face is tracked it is re-found in a padded window around its previous box, at native
# resolution, instead of sweeping the whole frame with the 1024-px grid (up to 16 inferences on a
# 4K canvas). The grid is still what ACQUIRES a face, so it keeps running whenever nothing is
# tracked.
YOLO_HINT_PAD = 1.0
YOLO_GRID_REFRESH = 30
# The global pass only sees big faces, so while a face is tracked its one remaining job is finding
# NEW ones; it runs every Nth frame instead of on every frame, which halves the inferences per
# frame in the common single-face case while still catching an entrance within a sixth of a second.
YOLO_GLOBAL_EVERY = 4
# While nothing is tracked the grid is the only thing that can find a face, but sweeping the whole
# canvas with 15 windows on EVERY frame is the most expensive thing this stage can do: a shot the
# detector never resolves then costs 16 inferences per frame for its entire length. Retry on this
# cadence instead (frame 0, then every 4th frame until something is found).
YOLO_GRID_RETRY = 4
# Face-detail detection cadence, opt-in for long shots: the detector samples every Nth frame and
# the boxes in between are interpolated from the detections on either side. Halves (or thirds) the
# per-frame detection cost and hands the crop path a smoother trajectory as a side effect.
FACE_DETECT_STRIDE_LABELS = ["auto (fit long clips)", "every frame (accurate)",
                             "every 2nd frame (fast)", "every 3rd frame (faster)"]
FACE_DETECT_STRIDE_VALUES = {
    "every frame (accurate)": 1,
    "every 2nd frame (fast)": 2,
    "every 3rd frame (faster)": 3,
}
# Auto stride: keep the tracking scan under roughly this many detected frames per clip, so a long
# clip costs a bounded amount of detection time instead of one SAM3 pass per frame. A 226 frame clip
# scans every 2nd frame, a 1000 frame clip every 6th. The boxes in between detections are
# interpolated, which is what makes a stride affordable: every frame still gets a tracked box.
FACE_DETECT_BUDGET = 180
FACE_DETECT_MAX_STRIDE = 8

# Pixel-mode clips are rendered shot by shot (see pixel_batch_plan), so a long clip cannot run the
# card out of memory. Where a seam is forced, the frames either side of it are sampled with
# PIXEL_BATCH_OVERLAP frames of real context: the next render's head re-renders the previous seam's
# tail as context, and its own tail is re-rendered by the render after it. The seam is therefore a
# cut between two warm renders, never a dissolve of two different redraws - blending those
# double-exposed the frame. pixel_batch_overlap is a widget; a seam costs 2 x overlap re-rendered
# frames, so the context is capped at an eighth of the render.
PIXEL_BATCH_CHUNK = 360
PIXEL_BATCH_OVERLAP = 8
# A shot is one render and only the card can force a split (see pixel_batch_plan). The free VRAM at
# the output frame size is still measured and logged, as the number that decides how much one render
# can hold (auto_pixel_batch_frames). PIXEL_BATCH_CHUNK is the size the estimator falls back to when
# there is no CUDA device to measure.
# The pixel pipeline holds about this many float32 bytes per frame at its peak (source, resized,
# accumulator and output copies), and the reserve is left for the model, the tile diffusion and the
# decoder, which are live at the same time.
PIXEL_BATCH_BYTES_PER_PIXEL = 56.0
PIXEL_BATCH_VRAM_RESERVE = 0.25
PIXEL_BATCH_AUTO_FLOOR = 8
# Auto sizing grows the chunk until its expected peak reaches the free VRAM, so this is only a
# sanity ceiling (the widget's own max), not a memory guess.
PIXEL_BATCH_AUTO_MAX = 4096
# Off: feeding a chunk its own sharp output as context does not work at denoise 1.0. The masked
# sampler holds only whole latents (so with an 8 frame overlap only the first 4 frames are held),
# and the model then re-interprets the sharp frames it did not generate, which shows up as
# ghosting over the following frames plus a cut where the held latents end. Kept behind this
# switch because the plumbing is reusable for a latent-space continuation, where the pinned
# context is the previous chunk's own latents instead of re-encoded pixels.
PIXEL_BATCH_PIN_CONTEXT = False
# On: a chunk's leading overlap frames are replaced, before encoding, by the previous chunk's
# own emitted pixels. They stay context - cut off the output, never emitted twice - but the
# sampler redraws them freely, so nothing holds yesterday's detail in place to disagree with
# today's, which is what ghosted the pin. They carry the one thing a fresh upscale of the
# source cannot: where and how the previous chunk actually put the frame on screen. The
# "seam matched to the previous render" line measures the result, seam by seam.
PIXEL_BATCH_SEAM_GUIDE = True
# Off: each chunk used to be compared with the previous render as a similarity, not only as a
# translation, and warped back onto the clip's framing when the redraw came back a fraction of a
# percent off. Five seams measured with the middle/edge split say there is nothing there to
# correct: every cut reads -0.16 to +0.02% on the whole frame and -0.45 to +0.37% for the middle
# against its edge, tighter than one ordinary pair of frames inside a single render, where the same
# reading swings a few percent. So the seam is not a re-frame, and resampling every frame of every
# chunk to chase half a percent only softens the picture - which is what the "cut 0.045 -> 0.045"
# lines were reporting: the warp did not change the difference at the cut. The measurement and the
# alignment both stay; the switch is here for a shot that really does re-frame across a seam.
PIXEL_BATCH_SEAM_ALIGN = False
# H3 renders on a fixed frame grid: a sequence length is 22 + 17k frames (22, 39, 56 ... the
# node's own windows default to 124 = 22 + 17 * 6, about 5.2 seconds at 24 fps). A length off
# that grid is not refused, it is cut back to the grid at render time, which is why a 51 frame
# chunk comes back as 39: the last twelve frames of the chunk are never rendered. Fitting 39
# decoded frames onto 51 slots then moves content forward in time towards the chunk's tail,
# and every chunk restarts that offset - which is a jump of about twelve frames of motion at
# every seam. On the grid nothing is trimmed, so decoded frame j stays frame j.
H3_FRAME_GRID_BASE = 22
H3_FRAME_GRID_STEP = 17


def h3_aligned_frames(count: int) -> int:
    """Largest H3 frame-grid length (base + step * k) at or below `count`."""
    count = int(count)
    if count < H3_FRAME_GRID_BASE:
        return count
    return (H3_FRAME_GRID_BASE
            + H3_FRAME_GRID_STEP * ((count - H3_FRAME_GRID_BASE) // H3_FRAME_GRID_STEP))


def h3_frame_grid_cover(count: int) -> int:
    """Smallest H3 frame-grid length (base + step * k) at or above `count`."""
    count = int(count)
    if count <= H3_FRAME_GRID_BASE:
        return H3_FRAME_GRID_BASE
    return (H3_FRAME_GRID_BASE
            + H3_FRAME_GRID_STEP * (-(-(count - H3_FRAME_GRID_BASE) // H3_FRAME_GRID_STEP)))

def detect_faces_yolo(image_tensor: torch.Tensor, model_path: str = "", confidence: float = 0.35,
                      window: int = 1024, overlap: float = 0.25, hints=None, allow_grid: bool = True,
                      global_pass: bool = True):
    """Ultralytics YOLO detector. YOLO resizes its input to ~640px, so on a multi-thousand-pixel canvas
    small heads shrink to a few pixels and are missed. We therefore run one global pass (big heads) plus
    overlapping full-resolution windows (small heads) and merge the results.

    `hints` are the previous frame's face boxes. When they are given, one padded window per hinted
    face replaces the sliding grid: the face is looked at in its own neighbourhood, at full
    resolution, at the cost of one inference instead of sixteen. `allow_grid=False` skips the grid
    even without hints (the cheap global pass alone), which keeps a shot without faces from paying
    the sweep on every frame. `global_pass=False` drops the global pass too, for the frames in
    between where the tracked window is all that is needed to keep a known face found.
    """
    path = _find_yolo_model(model_path)
    if path is None:
        return []
    try:
        from ultralytics import YOLO
        if path not in _YOLO_CACHE:
            _YOLO_CACHE[path] = YOLO(path)
        model = _YOLO_CACHE[path]
        frame = image_tensor[0, ..., :3]
        H, W = int(frame.shape[0]), int(frame.shape[1])
        found = []

        def to_bgr(region):
            """HxWx3 tensor (any device) -> uint8 BGR numpy for the detector."""
            arr = (region.detach().float() * 255.0).round().clamp_(0, 255).to(torch.uint8).cpu().numpy()
            return arr[:, :, ::-1].copy()

        # Only the pixels the detector actually looks at are converted. Handing the whole frame to
        # the CPU on every frame cost more than the inferences themselves on a 4K canvas: the global
        # pass now gets a capped copy resized on the frame's own device, each hinted window gets its
        # own small crop, and the full-resolution frame is only materialised when the grid runs.
        native_bgr = None

        def native(region=None):
            nonlocal native_bgr
            if region is None:
                if native_bgr is None:
                    native_bgr = to_bgr(frame)
                return native_bgr
            x1, y1, x2, y2 = region
            if native_bgr is not None:
                return native_bgr[y1:y2, x1:x2]
            return to_bgr(frame[y1:y2, x1:x2])

        def run(crop, ox, oy, imgsz, edges, scale=1.0):
            res = model(crop, conf=float(confidence), imgsz=int(imgsz), verbose=False)[0]
            ch, cw = crop.shape[:2]
            for box, sc in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = [float(v) for v in box]
                # drop boxes cut by an interior window edge (the neighbouring window sees them whole)
                if (edges[0] and x1 < 3) or (edges[1] and y1 < 3) or (edges[2] and x2 > cw - 3) or (edges[3] and y2 > ch - 3):
                    continue
                found.append((ox + x1 * scale, oy + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale, float(sc)))

        # Global pass on a copy capped to the size YOLO samples at anyway (identical boxes, less
        # copying). The resize happens on the frame's device, so only the small copy crosses the bus.
        # Skipped outright while a tracked face only needs its own window (see global_pass).
        if global_pass:
            inv_scale = 1.0
            if max(H, W) > YOLO_DETECT_MAX_SIDE:
                det_scale = YOLO_DETECT_MAX_SIDE / float(max(H, W))
                det_h = max(1, int(round(H * det_scale)))
                det_w = max(1, int(round(W * det_scale)))
                small = torch.nn.functional.interpolate(
                    frame.detach().float().permute(2, 0, 1).unsqueeze(0),
                    size=(det_h, det_w), mode="area",
                )[0].permute(1, 2, 0)
                det_img = to_bgr(small)
                inv_scale = 1.0 / det_scale
            else:
                det_img = native()
            run(det_img, 0, 0, 1280, (False, False, False, False), inv_scale)

        hinted = []
        for h in (hints or []):
            if len(h) >= 4:
                hinted.append((float(h[0]), float(h[1]), float(h[2]), float(h[3])))
        for hx, hy, hw, hh in hinted:
            padx, pady = max(hw, 32.0) * YOLO_HINT_PAD, max(hh, 32.0) * YOLO_HINT_PAD
            x1 = max(0, int(round(hx - padx)))
            y1 = max(0, int(round(hy - pady)))
            x2 = min(W, int(round(hx + hw + padx)))
            y2 = min(H, int(round(hy + hh + pady)))
            if x2 - x1 < 64 or y2 - y1 < 64:
                continue
            crop = native((x1, y1, x2, y2))
            edges = (x1 > 0, y1 > 0, x2 < W, y2 < H)
            run(crop, x1, y1, max(640, min(window, max(crop.shape[:2]))), edges)

        if allow_grid and max(H, W) > window * 1.25:
            step = max(64, int(window * (1.0 - overlap)))
            xs = sorted(set(list(range(0, max(1, W - window), step)) + [max(0, W - window)]))
            ys = sorted(set(list(range(0, max(1, H - window), step)) + [max(0, H - window)]))
            for y0 in ys:
                for x0 in xs:
                    crop = native()[y0:y0 + window, x0:x0 + window]
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
                      window: int = 1536, overlap: float = 0.25, keep_loaded: bool = False):
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
        # keep_loaded=True is what makes video detection one model load instead of one per frame:
        # building the image model reloads a ~1 GB checkpoint every call.
        if proc is not None and not keep_loaded:
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
                 face_mask=None, mask_index: int = 0, manual_boxes: str = "", sam3_prompt: str = "head",
                 hints=None, allow_grid: bool = True, global_pass: bool = True,
                 sam3_keep_loaded: bool = False):
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
        faces = detect_faces_sam3(canvas, sam3_prompt, confidence, keep_loaded=sam3_keep_loaded)
        if faces is None:
            logger.warning("[H3 Upscale] SAM3 unavailable (install the `sam3` package or the RMBG pack); falling back to auto.")
            detector = "auto"
        else:
            return faces, "sam3"
    if detector in ("auto", "yolo"):
        faces = detect_faces_yolo(canvas, model_path, confidence, hints=hints, allow_grid=allow_grid,
                                  global_pass=global_pass)
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


def scale_to_megapixel_target(w: int, h: int, megapixels: float, multiple: int = 32):
    """Aspect-preserving target size (w, h) for a megapixel budget.

    Both sides are snapped to a multiple of `multiple` (32 px, a multiple of the H3 VAE's 16 px
    stride), so every downstream stage sees a grid-aligned size, and the result never shrinks
    below the source because this node only ever upscales. Snapping the two sides independently
    distorts the aspect ratio by more than a percent on small frames, so a short window of
    grid-aligned heights is searched instead and the pair with the least aspect error wins, with
    distance from the requested area as the tie-break.
    """
    if w <= 0 or h <= 0:
        return 0, 0
    budget = max(float(megapixels), 1e-4) * 1_000_000.0
    source_aspect = float(w) / float(h)
    scale = max(math.sqrt(budget / float(w * h)), 1.0)
    # Never shrink: the smallest grid-aligned size that still holds the source.
    min_w = int(math.ceil(w / float(multiple))) * multiple
    min_h = int(math.ceil(h / float(multiple))) * multiple
    base_h = max(min_h, int(round(h * scale / multiple)) * multiple)

    best = None
    for step in range(-6, 7):
        target_h = base_h + step * multiple
        if target_h < min_h:
            continue
        target_w = max(min_w, int(round(source_aspect * target_h / multiple)) * multiple)
        if target_w <= 0:
            continue
        aspect_err = abs(target_w / float(target_h) - source_aspect) / source_aspect
        area_err = abs(target_w * target_h - budget) / budget
        score = area_err + 3.0 * aspect_err
        if best is None or score < best[0]:
            best = (score, target_w, target_h)
    if best is None:
        return min_w, min_h
    return best[1], best[2]


def _dlss_label_scale(label):
    """Numeric scale of an UPSCALING_LABELS entry ("2x (Quality)" -> 2.0), or None."""
    digits = ""
    for char in str(label).strip().lower():
        if char.isdigit() or (char == "." and digits):
            digits += char
        elif digits:
            break
        else:
            return None
    try:
        return float(digits)
    except ValueError:
        return None


def dlss_mode_for_target(w: int, h: int, megapixels: float):
    """The UPSCALING_LABELS entry that covers ``dlss5_scale_to_megapixels``.

    The megapixel target replaces the old upscaling_mode preset with a continuous control, so the
    smallest DLSS mode that can deliver the target is chosen and the pass is then fitted to the
    exact size. 0 means no DLSS upscaling, i.e. the first (native) mode.
    """
    labels = list(UPSCALING_LABELS) or ["Standard"]
    if not megapixels or float(megapixels) <= 0.0:
        return labels[0]
    scales = [(label, _dlss_label_scale(label)) for label in labels]
    scales = [(label, scale) for label, scale in scales if scale]
    if not scales:
        return labels[0]
    needed = math.sqrt(float(megapixels) * 1_000_000.0 / max(1.0, float(w) * float(h)))
    covering = sorted((scale, label) for label, scale in scales if scale >= needed - 1e-6)
    if covering:
        return covering[0][1]
    return max(scales, key=lambda item: item[1])[0]


def dlss_scale_target(w: int, h: int, megapixels: float):
    """DLSS 5 output size for ``dlss5_scale_to_megapixels``, or (0, 0) to leave it alone.

    DLSS has its own megapixel budget, independent of the H3 Upscale group's
    ``scale_to_megapixels``. 0 disables DLSS upscaling and keeps the pass input size, and the
    input is never shrunk, so switching DLSS on cannot lower the resolution. The aspect ratio is
    kept exactly by ``exact_aspect_target``.
    """
    if not megapixels or float(megapixels) <= 0.0:
        return 0, 0
    target_w, target_h = exact_aspect_target(w, h, float(megapixels))
    if target_w <= 0 or target_h <= 0:
        return 0, 0
    if target_w * target_h < w * h * 0.995:   # budget below the pass input: do not shrink
        return 0, 0
    return target_w, target_h


def exact_aspect_target(w: int, h: int, megapixels: float):
    """Integer (w, h) for a megapixel budget with the aspect ratio kept to within half a pixel.

    ``scale_to_megapixel_target`` snaps both sides to the 32 px VAE grid, which the image needs
    while it still goes through a latent. A stage after the decode - the DLSS 5 pass - has no such
    constraint, so its output is sized here instead: the integer pair closest to the budget, with
    the one pixel of slack that keeps the source aspect ratio exact.
    """
    if w <= 0 or h <= 0:
        return 0, 0
    aspect = float(w) / float(h)
    budget = max(float(megapixels), 1e-4) * 1_000_000.0
    target_h = max(1, int(round(math.sqrt(budget / aspect))))
    target_w = max(1, int(round(target_h * aspect)))
    best = None
    for candidate_w in (target_w - 1, target_w, target_w + 1):
        if candidate_w < 1:
            continue
        aspect_err = abs(candidate_w / float(target_h) - aspect) / aspect
        area_err = abs(candidate_w * target_h - budget) / budget
        score = aspect_err * 4.0 + area_err
        if best is None or score < best[0]:
            best = (score, candidate_w, target_h)
    return best[1], best[2]


def _warp_crop_to_roi(crop_np: np.ndarray, mask_np: np.ndarray, matrix, rect):
    """Warp a crop (and its mask) into a frame ROI with the inverse of the crop transform."""
    x0, y0, x1, y1 = rect
    local = np.asarray(matrix, np.float32).copy()
    local[0, 2] -= x0
    local[1, 2] -= y0
    inv = cv2.invertAffineTransform(local)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    warped = cv2.warpAffine(crop_np.astype(np.float32), inv, (x1 - x0, y1 - y0), flags=flags,
                            borderMode=cv2.BORDER_REPLICATE)
    wmask = cv2.warpAffine(mask_np.astype(np.float32), inv, (x1 - x0, y1 - y0), flags=flags,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.clip(warped, 0.0, 1.0), np.clip(wmask, 0.0, 1.0)


# Video faces are handled as TRACKS, not as independent frames: detection runs on a stride, the
# boxes are associated into tracks (greedy overlap, like SAM3_VideoTrack in the reference
# workflow), short gaps are interpolated, the trajectory is smoothed, and every track gets ONE
# crop size for its whole length (like SubjectCrop). A per-frame crop that changes size and
# position is what made the redrawn face breathe and flicker even when the diffusion itself was
# steady.
FACE_TRACK_IOU = 0.3        # association threshold between a detection and a track's last box
FACE_TRACK_MAX_GAP = 2      # frames a track may be continued across a missed detection
FACE_TRACK_SMOOTH = 2       # +- frames of the trajectory moving average
FACE_TRACK_MIN_HITS = 2     # detections a track needs before it is worth redrawing
FACE_CROP_GRID = 16         # crop sides stay on the VAE grid so every frame of a track matches

# Face crops are redrawn at roughly half a megapixel, the size the reference workflow crops to
# (MVEx_SubjectCrop, upscale_megapixels 0.5). A 1024 px working size over a whole clip is what made
# the face pass cost minutes per clip.
FACE_WORK_MEGAPIXELS = 0.5
# Frames per temporal face pass: the workflow samples the entire crop sequence in one KSampler run.
FACE_SEQ_CHUNK = 121
# Faces are detected on a frame capped to this long side, then the boxes are scaled back.
FACE_DETECT_MAX_SIDE = 960


def _face_box_iou(a, b) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = [float(v) for v in a[:4]]
    bx, by, bw, bh = [float(v) for v in b[:4]]
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def _face_aux(box):
    """Instance mask and symmetry that detect_faces recorded for this exact box."""
    key = tuple(box[:4])
    if key in _FACE_MASKS:
        return _FACE_MASKS[key], _FACE_SYM.get(key, 1.0)
    for known in list(_FACE_MASKS):
        if _face_box_iou(known, box) > 0.999:
            return _FACE_MASKS.get(known), _FACE_SYM.get(known, 1.0)
    return None, _FACE_SYM.get(key, 1.0)


def _fill_box_gaps(boxes, max_gap=FACE_TRACK_MAX_GAP):
    """Interpolate a track through short detection gaps between two valid frames."""
    valid = [i for i, box in enumerate(boxes) if box is not None]
    if len(valid) < 2:
        return boxes
    out = list(boxes)
    for start, end in zip(valid, valid[1:]):
        span = end - start
        if 1 < span <= max_gap + 1:
            for step in range(1, span):
                t = step / float(span)
                out[start + step] = tuple(boxes[start][d] * (1.0 - t) + boxes[end][d] * t
                                          for d in range(4))
    return out


def _smooth_box_sequence(boxes, half=FACE_TRACK_SMOOTH):
    """Moving average of (x, y, w, h) over each track's neighbouring valid frames.

    Detection noise of a few pixels per frame is what makes a redrawn face shimmer: the crop
    moves with it even when the model output is identical. Averaging the trajectory removes it
    without lagging a steady pan.
    """
    import numpy as _np
    out = list(boxes)
    valid = [i for i, box in enumerate(boxes) if box is not None]
    if len(valid) < 2:
        return [None if box is None else tuple(float(v) for v in box) for box in out]
    stack = _np.array([boxes[i] for i in valid], dtype=_np.float32)
    for k in range(len(valid)):
        lo, hi = max(0, k - half), min(len(valid), k + half + 1)
        out[valid[k]] = tuple(float(v) for v in stack[lo:hi].mean(axis=0))
    return out


def build_face_tracks(detections, iou_threshold=FACE_TRACK_IOU, max_gap=FACE_TRACK_MAX_GAP):
    """Per-frame detections -> tracks, the job SAM3_VideoTrack does in the reference workflow.

    ``detections`` is one list per frame of ``(x, y, w, h, instance mask, symmetry)``. Each
    detection joins the track it overlaps most (or starts a new one), so one face keeps one
    identity for the whole clip even while its box is re-detected on every stride.
    """
    tracks = []
    for frame_index, frame in enumerate(detections):
        claimed = [False] * len(tracks)
        for det in frame:
            best, best_iou = -1, iou_threshold
            for track_index, track in enumerate(tracks):
                if claimed[track_index] or track["last"] is None:
                    continue
                iou = _face_box_iou(det[:4], track["last"])
                if iou >= best_iou:
                    best, best_iou = track_index, iou
            if best < 0:
                tracks.append({"boxes": [None] * len(detections), "insts": [None] * len(detections),
                               "syms": [None] * len(detections), "last": None})
                best = len(tracks) - 1
                claimed.append(False)
            track = tracks[best]
            claimed[best] = True
            track["boxes"][frame_index] = tuple(float(v) for v in det[:4])
            track["insts"][frame_index] = det[4] if len(det) > 4 else None
            track["syms"][frame_index] = det[5] if len(det) > 5 else None
            track["last"] = track["boxes"][frame_index]
    for track in tracks:
        track["boxes"] = _smooth_box_sequence(_fill_box_gaps(track["boxes"], max_gap=max_gap))
        track["last"] = None
    return tracks


def _inherit_track_masks(track):
    """Give interpolated frames the nearest detection's instance mask and symmetry score.

    A stride leaves frames with a box but no mask of their own. Without this the temporal face pass
    would fall back to an elliptical mask on those frames and lose the mask cleanup and the temporal
    majority filter for the whole track; the score would default to frontal-only-passing 1.0 as well.
    The inherited mask is resized to the interpolated box, so it still lines up with the face.
    """
    boxes, insts, syms = track["boxes"], track["insts"], track["syms"]
    known = [i for i, inst in enumerate(insts) if inst is not None]
    if not known:
        return
    for index, box in enumerate(boxes):
        if box is None:
            continue
        nearest = min(known, key=lambda i: abs(i - index))
        if insts[index] is None:
            src = insts[nearest]
            if src is not None:
                width = max(1, int(round(float(box[2]))))
                height = max(1, int(round(float(box[3]))))
                if src.shape[:2] != (height, width):
                    src = cv2.resize(src, (width, height), interpolation=cv2.INTER_LINEAR)
                insts[index] = src
        if syms[index] is None:
            syms[index] = syms[nearest]


def track_crop_size(boxes, padding: float, img_w: int, img_h: int):
    """One padded crop size for a whole track, on the VAE grid, clamped to the frame."""
    sized = [box for box in boxes if box is not None]
    if not sized:
        return 0, 0
    span = max(max(float(box[2]), float(box[3])) for box in sized)
    side = int(-(-span * (1.0 + 2.0 * float(padding)) // FACE_CROP_GRID) * FACE_CROP_GRID)
    return min(side, int(img_w)), min(side, int(img_h))


def face_work_size(crop_h: int, crop_w: int, face_size: int):
    """Working size for a face crop: never below its native size, at most ``face_size`` on the long side
    and about FACE_WORK_MEGAPIXELS, snapped to the 32 px VAE grid.

    The reference workflow redraws ~0.5 MP crops (SubjectCrop, upscale_megapixels 0.5). Forcing every
    crop up to 1024 px - up to four times the area - was most of the cost of diffusing a whole clip.
    """
    crop_h, crop_w = max(32, int(crop_h)), max(32, int(crop_w))
    long_side = max(crop_h, crop_w)
    area_scale = float(np.sqrt(FACE_WORK_MEGAPIXELS * 1e6 / float(crop_h * crop_w)))
    scale = max(1.0, min(max(64, int(face_size)) / float(long_side), area_scale, 2048.0 / long_side))
    work_h = int(min(2048, max(64, int(np.ceil(crop_h * scale / 32.0)) * 32)))
    work_w = int(min(2048, max(64, int(np.ceil(crop_w * scale / 32.0)) * 32)))
    return work_h, work_w


def centered_crop(box, crop_w: int, crop_h: int, img_w: int, img_h: int):
    """Crop rect of a FIXED size centred on the (smoothed) face box, kept inside the frame."""
    center_x = float(box[0]) + float(box[2]) / 2.0
    center_y = float(box[1]) + float(box[3]) / 2.0
    x1 = int(round(center_x - crop_w / 2.0))
    y1 = int(round(center_y - crop_h / 2.0))
    x1 = max(0, min(x1, int(img_w) - crop_w))
    y1 = max(0, min(y1, int(img_h) - crop_h))
    return x1, y1, x1 + crop_w, y1 + crop_h


def auto_detect_stride(frame_count: int, budget: int = FACE_DETECT_BUDGET,
                       max_stride: int = FACE_DETECT_MAX_STRIDE) -> int:
    """Detection stride that keeps a clip's tracking scan to about ``budget`` sampled frames.

    Short clips are scanned every frame (no behaviour change), long ones every Nth frame. The stride
    is capped so a very long clip still gets several detections per second of a tracked face.
    """
    frame_count = max(1, int(frame_count))
    if frame_count <= budget:
        return 1
    return int(max(1, min(max_stride, -(-frame_count // max(1, budget)))))


def auto_pixel_batch_frames(frame_count: int, frame_h: int, frame_w: int) -> int:
    """Most frames one render can hold at this output frame size, up to PIXEL_BATCH_AUTO_MAX.

    A fixed 360 frames is fine for a 1 MP clip and far too much for a 4K one: the pixel pipeline
    keeps several full-frame tensors alive per frame, so the memory cost of a chunk scales with
    frames x pixels. This is a ceiling on a render, not a batch size: the seams are the source's
    hard cuts (see pixel_batch_plan), and only a shot longer than this figure is split. With a CUDA device the chunk is grown until that cost approaches the memory
    actually free, the largest H3 grid length inside that figure is taken, and the expected peak is
    logged with the limit that bound it and the frame count the memory would have allowed, so the
    size can be checked against the card instead of guessed. Without a CUDA device (or on CPU) the fixed
    PIXEL_BATCH_CHUNK is used.
    """
    frame_count = max(1, int(frame_count))
    frame_h, frame_w = int(frame_h), int(frame_w)
    fallback = min(frame_count, PIXEL_BATCH_CHUNK)
    per_frame = float(max(1, frame_h)) * float(max(1, frame_w)) * PIXEL_BATCH_BYTES_PER_PIXEL
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info(
                "[H3 Upscale] pixel batch auto: no CUDA device to measure, so the chunk stays at "
                "%d frame(s) (%.1f GB expected peak at %dx%d)",
                fallback, fallback * per_frame / 1e9, frame_w, frame_h,
            )
            return fallback
        free, total = torch.cuda.mem_get_info()
    except Exception as e:
        logger.info("[H3 Upscale] pixel batch auto: VRAM query unavailable (%s); using %d frame(s)",
                    e, fallback)
        return fallback
    # Grow the chunk until the peak the pipeline holds for it reaches what the card actually has
    # free, holding PIXEL_BATCH_VRAM_RESERVE back for the model, the tile diffusion and the decoder,
    # which are live beside the batch. The peak is reported either way.
    budget = float(free) * (1.0 - PIXEL_BATCH_VRAM_RESERVE)
    ceiling = min(frame_count, PIXEL_BATCH_AUTO_MAX)
    if per_frame <= 0.0 or budget <= 0.0:
        chunk, limit = fallback, "the fixed chunk (no usable memory figure)"
    else:
        fits = int(budget // per_frame)
        # Floor first, then clamp to the ceiling: a clip shorter than the floor must not be
        # reported as a chunk larger than the clip.
        chunk = int(min(ceiling, max(PIXEL_BATCH_AUTO_FLOOR, fits)))
        if chunk >= frame_count:
            limit = "the whole clip fits"
        elif fits >= ceiling:
            limit = "the %d frame ceiling" % PIXEL_BATCH_AUTO_MAX
        else:
            limit = "the free VRAM"
    # What the memory allowed is not what a render can be: a render is an H3 grid length (see
    # H3_FRAME_GRID_BASE), so take the largest grid length the budget still covers. Both figures
    # are logged, otherwise "53 frames of memory" and "39 frames per render" read as a
    # contradiction, and the step that was refused is the one worth knowing about.
    memory_fits = chunk
    if chunk < frame_count:
        chunk = min(chunk, h3_aligned_frames(chunk))
    peak = chunk * per_frame
    grid_note = ""
    if memory_fits != chunk:
        wider = h3_frame_grid_cover(memory_fits)
        grid_note = (" (%d frame(s) by memory; a render is an H3 grid length, and the next step "
                     "up, %d, would peak at %.1f GB)" % (
                         memory_fits, wider, wider * per_frame / 1e9))
    logger.info(
        "[H3 Upscale] pixel batch: the free VRAM should hold about %d frame(s) of %dx%d (%.1f GB "
        "expected peak, %.0f%% of the %.1f GB free on a %.1f GB card, %.0f MB per frame); %s%s. "
        "Nothing is split on this figure - a shot is one render, and the card decides",
        chunk, frame_w, frame_h, peak / 1e9,
        100.0 * peak / free if free > 0 else 0.0, free / 1e9, total / 1e9,
        per_frame / 1e6, limit, grid_note,
    )
    if peak > budget:
        logger.warning(
            "[H3 Upscale] pixel batch: even %d frame(s) are estimated at %.1f GB, more than the "
            "%.1f GB left after the %.0f%% reserve - expect the card to refuse a render this long, "
            "and the render to be split where it does; close what else is on the card to give it "
            "more room",
            chunk, peak / 1e9, budget / 1e9, PIXEL_BATCH_VRAM_RESERVE * 100.0,
        )
    return chunk



def pixel_batch_plan(frame_count: int, chunk: int, overlap: int, cuts):
    """Where each render starts and stops, and which frames it owns.

    A seam exists only where the source has a hard cut (see detect_scene_cuts). Every shot is
    rendered as itself: a shot one render can hold is rendered in one piece whatever its length,
    and only a shot longer than that is split - those seams keep the overlap and the look matching
    every seam has always had, because a seam inside a shot has nothing under it to hide behind. A
    render is always a grid length (see H3_FRAME_GRID_BASE), so a short shot's render reaches past
    the cut into the next shot; those frames are drawn and dropped, and the seam still lands on the
    cut.

    Returns (plan, seam_frames): one (start, end, emit_from, keep_to, begins_on_cut, overlap) per
    render, and the set of frames a seam was placed on.
    """
    grid = int(H3_FRAME_GRID_BASE)
    frame_count = int(frame_count)
    chunk = max(grid, int(chunk))
    overlap = max(0, min(int(overlap), chunk // 8))
    # A cut can carry a seam only if the shot on each side of it is long enough to be rendered on
    # the grid; a shorter shot is rendered with its neighbour, the cut and all.
    bounds = [0]
    for cut in sorted(int(c) for c in (cuts or ())):
        if cut - bounds[-1] < grid or frame_count - cut < grid:
            continue
        bounds.append(cut)
    bounds.append(frame_count)
    seam_frames = set(bounds[1:-1])

    plan = []
    for a, b in zip(bounds, bounds[1:]):
        length = b - a
        on_cut = a > 0
        need = h3_frame_grid_cover(length)
        if length <= chunk and a + need <= frame_count:
            # One render for the whole shot, even where it runs past the cut: the frames past it
            # belong to the next shot and are drawn and dropped. The overlap is only there for a
            # seam the card forces inside this shot later (see pixel_batch_split_window).
            plan.append((a, a + need, a, b, on_cut, 0 if on_cut else overlap))
            continue
        if length <= chunk and b - need >= 0:
            # The clip ends before this shot's own grid render would, so this one ends on the last
            # frame and backs up over frames already emitted, as the last render of a clip always
            # has. `chunk` can be far larger than the shot (auto), so the window is the shot's own
            # grid length and never the whole clip.
            plan.append((b - need, b, a, b, on_cut, 0 if on_cut else overlap))
            continue
        # Longer than one render holds, or a clip that does not end on the grid and so cannot be
        # one render at all: the fewest grid windows that cover the shot, the last one ending on
        # the shot's own last frame.
        hold = chunk if length > chunk else max(grid, min(chunk, h3_aligned_frames(b)))
        pos, emit_from, first = a, a, True
        while pos < b:
            last = (b - pos) <= hold
            if last:
                end, emit_to = b, b
                if end - pos < hold and b - hold >= a:
                    # A short last part is rendered as a full sequence ending on the shot's own
                    # last frame, so it is not the one render that comes out looking different.
                    end = b
                    pos = b - hold
            else:
                end, emit_to = pos + hold, pos + hold - overlap
            plan.append((max(0, pos), end, emit_from, emit_to, on_cut and first,
                         0 if (on_cut and first) else overlap))
            if last:
                break
            pos, emit_from, first = emit_to - overlap, emit_to, False
    return plan, seam_frames


def _cuda_out_of_memory(exc: BaseException) -> bool:
    """True when a render failed for want of VRAM, which retrying in smaller pieces can fix."""
    return "out of memory" in (str(exc) or "").lower()


def pixel_batch_split_window(entry, frame_count: int, overlap: int):
    """Two render windows that emit exactly what `entry` emitted, or None if it cannot be split.

    Used when the card refuses a render: the shot is untouched, so the seam this adds is the in-shot
    kind, with the overlap, the context and the look matching a seam inside a shot has always had.
    A window carrying more context than the grid needs is first cut back to the shortest render that
    still holds its frames (that costs no seam at all); only then is the span halved. The first half
    keeps the window's own start, so a cut seam stays on its cut; the second half starts a
    context-length into the first half's last frames, exactly like any other in-shot seam.
    """
    start, end, emit_from, keep_to, begins_on_cut, overlap_now = entry
    span = int(keep_to) - int(emit_from)
    minimal = h3_frame_grid_cover(int(keep_to) - int(start))
    if int(end) - int(start) > minimal:
        # The look-ahead the seam is matched on is the first thing to give up: it is context, not
        # output, and dropping it loses nothing but that matching.
        return [(int(start), int(start) + minimal, int(emit_from), int(keep_to),
                 bool(begins_on_cut), int(overlap_now))]
    lead = int(overlap_now) if not begins_on_cut else 0
    half = span // 2
    if span < 2 or h3_frame_grid_cover(max(1, half + lead)) >= int(end) - int(start):
        # Halving would not make either half a shorter render than this one already is: the grid is
        # what the card is really being asked for here, and cutting the file smaller cannot help.
        return None
    mid = int(emit_from) + half
    first_end = int(start) + h3_frame_grid_cover(mid - int(start) + max(1, lead))
    first_end = min(max(first_end, mid + 1), int(frame_count))
    second_start = max(0, mid - int(overlap))
    second_end = min(int(frame_count),
                     second_start + h3_frame_grid_cover(max(1, int(end) - second_start)))
    first = (int(start), first_end, int(emit_from), mid, bool(begins_on_cut), int(overlap_now))
    second = (second_start, second_end, mid, int(keep_to), False, int(overlap))
    return [first, second]

def slice_shared_face_plan(shared, target_h: int, target_w: int):
    """The clip-wide face plan, cut to one pixel chunk and scaled from source to output resolution.

    The plan is detected once on the whole clip's source frames, so every chunk works from the same
    tracks and the same crop size: the redrawn face keeps its scale and identity across a chunk
    boundary instead of each chunk inventing its own. Boxes, crop rectangles and instance masks are
    scaled by the upscale factor; crop sides are re-snapped to the VAE grid so a chunk's crop matches
    the cropper everywhere else in the clip.
    """
    plan = (shared or {}).get("plan")
    if not plan:
        return None
    offset = int((shared or {}).get("offset") or 0)
    count = int((shared or {}).get("frames") or 0)
    source_h = max(1, int((shared or {}).get("source_h") or target_h))
    source_w = max(1, int((shared or {}).get("source_w") or target_w))
    scale_x = float(target_w) / float(source_w)
    scale_y = float(target_h) / float(source_h)
    out = []
    for frame_index in range(max(0, offset), min(len(plan), max(0, offset) + count)):
        entries = []
        for entry in plan[frame_index]:
            bx, by, bw, bh, inst, rect, sym, track_id = entry
            cx1, cy1, cx2, cy2 = rect
            box_w = max(1.0, float(bw) * scale_x)
            box_h = max(1.0, float(bh) * scale_y)
            # Same crop size for every chunk: scale, then snap the sides to the VAE grid and re-centre.
            crop_w = int(max(32, round((cx2 - cx1) * scale_x / FACE_CROP_GRID) * FACE_CROP_GRID))
            crop_h = int(max(32, round((cy2 - cy1) * scale_y / FACE_CROP_GRID) * FACE_CROP_GRID))
            crop_w, crop_h = min(crop_w, int(target_w)), min(crop_h, int(target_h))
            center_x = (cx1 + cx2) / 2.0 * scale_x
            center_y = (cy1 + cy2) / 2.0 * scale_y
            x1 = int(round(center_x - crop_w / 2.0))
            y1 = int(round(center_y - crop_h / 2.0))
            x1 = max(0, min(x1, int(target_w) - crop_w))
            y1 = max(0, min(y1, int(target_h) - crop_h))
            if inst is not None and (abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6):
                inst = cv2.resize(inst, (max(1, int(round(box_w))), max(1, int(round(box_h)))),
                                  interpolation=cv2.INTER_LINEAR)
            entries.append((float(bx) * scale_x, float(by) * scale_y, box_w, box_h, inst,
                            (x1, y1, x1 + crop_w, y1 + crop_h), sym, track_id))
        out.append(entries)
    return out
FACE_CACHE_RAM_MIN = 512 * 1024 * 1024
FACE_CACHE_RAM_MAX = 2 * 1024 * 1024 * 1024


def face_cache_budget_bytes() -> int:
    """RAM allowed for a clip's decoded face crops: an eighth of system RAM, clamped 512 MB - 2 GB."""
    try:
        import os
        total = float(os.sysconf("SC_PHYS_PAGES")) * float(os.sysconf("SC_PAGE_SIZE"))
        if total > 0:
            return int(max(FACE_CACHE_RAM_MIN, min(FACE_CACHE_RAM_MAX, total / 8.0)))
    except Exception:
        pass
    return int(FACE_CACHE_RAM_MIN)


class _ClipFaceCache:
    """Decoded face crops for one clip: held in RAM, or spilt per track to a temp dir past the budget.

    A single pass per track costs time nothing but memory: 600 frames of 0.5 MP crops is a few GB that
    would otherwise sit in RAM for the whole chunk loop. Past the budget the crops are written to a
    temp .npz per track (numpy, no pickle) and read back for the chunk that needs them, so the
    whole-clip guarantee costs disk instead of memory and the loop stays bounded either way.
    """

    def __init__(self, budget_bytes: int = 0, spill: bool = False, label: str = "h3_face_cache"):
        self.budget = int(budget_bytes or 0)
        self.spilled = bool(spill)
        self.label = label
        self.dir = None
        self.ram = {}
        self.files = []          # (first frame, last frame, path, [(frame, face), ...])
        self.count = 0
        self.bytes = 0
        self._buffer = {}
        self._handles = {}       # path -> open memory-mapped .npz, opened once per track file

    @staticmethod
    def _numpy(tensor):
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    def _handle(self, path):
        """Open a spilt track file once and memory-map it, so a chunk reads only its own frames."""
        handle = self._handles.get(path)
        if handle is None:
            handle = np.load(path, allow_pickle=False, mmap_mode="r")
            self._handles[path] = handle
        return handle

    def put(self, key, seq, mask):
        seq_np = self._numpy(seq)
        mask_np = None if mask is None else self._numpy(mask)
        self.bytes += int(seq_np.nbytes) + (0 if mask_np is None else int(mask_np.nbytes))
        self.count += 1
        entry = {"seq": seq_np, "mask": mask_np}
        if self.spilled:
            self._buffer[key] = entry
        else:
            self.ram[key] = entry

    def flush_track(self):
        """Write the buffered track to disk in spill mode; free the buffer either way."""
        if not self.spilled:
            self._buffer = {}
            return
        if not self._buffer:
            return
        import os
        import tempfile
        if self.dir is None:
            self.dir = tempfile.mkdtemp(prefix=self.label + "_")
        keys = sorted(self._buffer)
        arrays = {}
        for index, key in enumerate(keys):
            entry = self._buffer[key]
            arrays["s%d" % index] = entry["seq"]
            arrays["k%d" % index] = np.asarray(key, np.int64)
            if entry["mask"] is not None:
                arrays["m%d" % index] = entry["mask"]
        path = os.path.join(self.dir, "track_%03d.npz" % len(self.files))
        np.savez(path, **arrays)
        self.files.append((int(keys[0][0]), int(keys[-1][0]), path, keys))
        self._buffer = {}

    def slice(self, start: int, end: int):
        """``{"seq": {(frame - start, face): tensor}, "mask": ...}`` for one chunk's frames."""
        start, end = int(start), int(end)
        entries = []
        if not self.spilled:
            entries = [(key, self.ram[key]) for key in sorted(self.ram) if start <= key[0] < end]
        else:
            for first, last, path, file_keys in self.files:
                if last < start:
                    # Chunks only move forward, so this file is behind the render for good: release it
                    # instead of leaving every track of a long clip open.
                    handle = self._handles.pop(path, None)
                    if handle is not None:
                        try:
                            handle.close()
                        except Exception:
                            pass
                    continue
                if first >= end:
                    continue
                data = self._handle(path)
                present = set(data.files)
                for index, key in enumerate(file_keys):
                    if not (start <= key[0] < end):
                        continue
                    mask_key = "m%d" % index
                    # np.array copies just these frames out of the mapping; the rest of the file is
                    # never read, and the mapping stays open for the chunks that follow.
                    entries.append((key, {"seq": np.array(data["s%d" % index]),
                                          "mask": np.array(data[mask_key]) if mask_key in present else None}))
        seq, masks = {}, {}
        for key, entry in entries:
            local = (int(key[0]) - start, int(key[1]))
            seq[local] = torch.from_numpy(entry["seq"])
            if entry["mask"] is not None:
                masks[local] = torch.from_numpy(entry["mask"])
        return {"seq": seq, "mask": masks}

    def summary(self) -> str:
        mb = self.bytes / 1e6
        if self.spilled:
            return "%.0f MB spilt to %s (budget %.0f MB)" % (mb, self.dir, self.budget / 1e6)
        return "%.0f MB in RAM (budget %.0f MB)" % (mb, self.budget / 1e6)

    def __len__(self):
        return int(self.count)

    def close(self):
        import shutil
        for handle in list(self._handles.values()):
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir = None
        self.ram.clear()
        self.files = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


_PIXEL_CROP_UPSCALER = {}


def apply_pixel_upscale(region, upscale_model: str):
    """One face crop through the selected upscale model, which is loaded once and reused.

    The clip-wide face pass builds its crops straight from the SOURCE clip, so without this a face
    would start from a bicubic crop while the rest of the frame came off a learned upscaler: the face
    would be the softest thing in a sharp 4K frame. Any failure falls back to the plain crop.
    """
    if not upscale_model or upscale_model == "None":
        return region
    if _PIXEL_CROP_UPSCALER.get("failed") == upscale_model:
        return region
    try:
        import comfy_extras.nodes_upscale_model
        node = _PIXEL_CROP_UPSCALER.get("node")
        if node is None:
            node = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel()
            _PIXEL_CROP_UPSCALER["node"] = node
        model_obj = _PIXEL_CROP_UPSCALER.get(upscale_model)
        if model_obj is None:
            model_obj = load_pixel_upscale_model(upscale_model)
            _PIXEL_CROP_UPSCALER[upscale_model] = model_obj
        return node.upscale(model_obj, region)[0]
    except Exception as e:
        _PIXEL_CROP_UPSCALER["failed"] = upscale_model
        logger.warning("[H3 Upscale] upscaling face crops with %s failed (%s); using bicubic crops",
                       upscale_model, e)
        return region

FACE_CROP_UPSCALE_BATCH = 4


def apply_pixel_upscale_batch(regions, upscale_model: str,
                              batch_size: int = FACE_CROP_UPSCALE_BATCH):
    """Upscale a track's face crops a few at a time instead of one call per crop.

    An upscale model is fixed-scale, so one call per ~0.5 MP crop leaves most of the card idle: the
    per-call overhead and the small input dominate. Crops of the same size are stacked and upscaled
    together (one call per batch); a leftover single crop still takes the plain path, and crops of a
    different size keep their own group so a stack is always shape-valid.
    """
    if not regions or not upscale_model or upscale_model == "None":
        return regions
    batch_size = max(1, int(batch_size))
    groups = {}
    for index, region in enumerate(regions):
        groups.setdefault(tuple(region.shape[1:]), []).append(index)
    out = list(regions)
    for indices in groups.values():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            if len(chunk) == 1:
                out[chunk[0]] = apply_pixel_upscale(regions[chunk[0]], upscale_model)
                continue
            stacked = torch.cat([regions[index] for index in chunk], dim=0)
            upscaled = apply_pixel_upscale(stacked, upscale_model)
            for offset, index in enumerate(chunk):
                out[index] = upscaled[offset:offset + 1]
    return out


def refine_plan_masks(canvas, plan, stride, detector, model_path, confidence, sam3_prompt,
                      margin: float = 0.35, min_iou: float = 0.15):
    """Re-detect every tracked face inside its OWN crop, at the canvas's resolution.

    The tracking scan runs on a frame capped to FACE_DETECT_MAX_SIDE, so on a 4K upscale a face that
    covers 300 px in the output was a 75 px blob to the detector: its mask is a coarse outline that has
    to be upscaled, and the composite edge shows it. Here the tracked box gives a crop that is already
    at output resolution, so the detector sees the face at (roughly) its full output size and its mask
    comes back at that resolution. Boxes, crop rectangles and track ids are the plan's - only the mask
    is replaced - so the shared, stable geometry survives and the mask gets sharper.

    Detections are made on a stride (a bounded number of scans per track) and copied from the nearest
    scanned frame in between, which is exactly how the tracker already fills the frames it skipped.
    Returns ``(plan, scanned)``.
    """
    if canvas is None or not plan:
        return plan, 0
    frame_count = min(int(canvas.shape[0]), len(plan))
    if frame_count <= 0:
        return plan, 0
    stride = max(1, int(stride))
    sampled = list(range(0, frame_count, stride))
    if sampled[-1] != frame_count - 1:
        sampled.append(frame_count - 1)
    canvas_h, canvas_w = int(canvas.shape[1]), int(canvas.shape[2])
    refined = {}          # track id -> {frame: (box w, box h, mask at that size)}
    scanned = 0
    for frame_index in sampled:
        entries = plan[frame_index]
        if not entries:
            continue
        frame = canvas[frame_index:frame_index + 1, :, :, :3]
        for entry in entries:
            fx, fy, fw, fh = [int(round(float(v))) for v in entry[:4]]
            if fw < 8 or fh < 8:
                continue
            track_id = entry[7] if len(entry) > 7 else -1
            if track_id is None or int(track_id) < 0:
                continue
            x1, y1, x2, y2 = expand_box(fx, fy, fw, fh, margin, canvas_w, canvas_h)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            try:
                faces, _label = detect_faces(
                    frame[:, y1:y2, x1:x2, :3], detector, model_path, confidence,
                    face_mask=None, mask_index=frame_index, manual_boxes="",
                    sam3_prompt=sam3_prompt, sam3_keep_loaded=True,
                )
            except Exception as e:
                logger.warning("[H3 Upscale] output-resolution mask refinement unavailable (%s); "
                               "keeping the tracked masks", e)
                return plan, 0
            scanned += 1
            best, best_iou = None, min_iou
            for face in faces:
                cx, cy, cw, ch = [float(v) for v in face[:4]]
                # The crop offset puts the detection back into canvas coordinates before comparing:
                # another head in the crop must not win just because it is large.
                iou = _face_box_iou((cx + x1, cy + y1, cw, ch), (fx, fy, fw, fh))
                if iou > best_iou:
                    best, best_iou = face, iou
            if best is None:
                continue
            inst, _sym = _face_aux(best)
            if inst is None:
                continue
            mask = cv2.resize(inst, (max(1, fw), max(1, fh)), interpolation=cv2.INTER_LINEAR)
            refined.setdefault(int(track_id), {})[frame_index] = (fw, fh, mask.astype(np.float32))
    if scanned > 0:
        # "auto" resolves to SAM3 too, and _free_sam3_direct() is a no-op if nothing was loaded: release
        # whatever the refinement loaded so the sampler below does not run beside the detector.
        _free_sam3_direct()
    if not refined:
        return plan, 0
    out = []
    replaced = 0
    for frame_index in range(len(plan)):
        entries = []
        for entry in plan[frame_index]:
            track_id = entry[7] if len(entry) > 7 else -1
            masks = refined.get(int(track_id)) if track_id is not None and int(track_id) >= 0 else None
            if not masks:
                entries.append(entry)
                continue
            hit = masks.get(frame_index)
            if hit is not None:
                fw, fh, mask = hit
            else:
                nearest = min(masks, key=lambda f: abs(f - frame_index))
                fx, fy, fw, fh = [int(round(float(v))) for v in entry[:4]]
                mask = cv2.resize(masks[nearest][2], (max(1, fw), max(1, fh)),
                                  interpolation=cv2.INTER_LINEAR)
            replaced += 1
            entries.append((entry[0], entry[1], entry[2], entry[3], mask, entry[5],
                            entry[6] if len(entry) > 6 else 1.0,
                            entry[7] if len(entry) > 7 else -1))
        out.append(entries)
    logger.info("[H3 Upscale] face masks re-detected inside the crop on %d frame(s) at %dx%d "
                "(%d scan(s)) instead of upscaling a %d-side scan's outline",
                replaced, canvas_w, canvas_h, scanned, FACE_DETECT_MAX_SIDE)
    return out, scanned


def plan_video_face_tracks(images, detector: str, model_path: str, confidence: float, face_mask,
                           manual_boxes: str, sam3_prompt: str, stride_label: str,
                           face_padding: float, frontal_only_threshold: float):
    """Detect on a stride, track, gap-fill, smooth and lock one crop per track.

    Returns one entry list per frame, each entry
    ``(x, y, w, h, instance mask, crop rect, symmetry, track id)``, or None when nothing was
    found (the caller then falls back to per-frame detection, so single images and shots without
    faces behave exactly as before).
    """
    frame_count = int(images.shape[0])
    if frame_count < 2:
        return None
    stride = int(FACE_DETECT_STRIDE_VALUES.get(stride_label, 0))
    auto_stride = stride <= 0
    if auto_stride:
        stride = auto_detect_stride(frame_count)
    sampled = list(range(0, frame_count, stride))
    if sampled[-1] != frame_count - 1:
        sampled.append(frame_count - 1)
    img_h, img_w = int(images.shape[1]), int(images.shape[2])

    import time
    _detection_started = time.time()
    detections_by_frame = {}
    for frame_index in sampled:
        # Detect on a capped copy: the canvas here is the already upscaled clip, and running SAM3 over
        # 4K frames is what made the pass appear to hang before any face was rendered. Boxes (and their
        # instance masks) are scaled back to canvas pixels afterwards.
        frame = images[frame_index:frame_index + 1]
        canvas_h, canvas_w = int(images.shape[1]), int(images.shape[2])
        det_scale = 1.0
        long_side = max(canvas_h, canvas_w)
        if long_side > FACE_DETECT_MAX_SIDE:
            det_scale = FACE_DETECT_MAX_SIDE / float(long_side)
            small_h = max(32, int(round(canvas_h * det_scale / 32.0)) * 32)
            small_w = max(32, int(round(canvas_w * det_scale / 32.0)) * 32)
            det_scale = canvas_h / float(small_h)
            frame = _resize_bhwc(frame, small_h, small_w)
        faces, _label = detect_faces(
            frame, detector, model_path, confidence,
            face_mask=face_mask, mask_index=frame_index, manual_boxes=manual_boxes,
            sam3_prompt=sam3_prompt,
            sam3_keep_loaded=True,   # one load for the whole clip
        )
        if frontal_only_threshold > 0.0 and faces:
            faces = [f for f in faces if _FACE_SYM.get(tuple(f[:4]), 1.0) >= frontal_only_threshold]
        scale_x = canvas_w / float(frame.shape[2])
        scale_y = canvas_h / float(frame.shape[1])
        entries = []
        for f in faces:
            # Instance masks and symmetry are keyed by the detection-resolution box, so read them before
            # scaling, then bring the mask up to the scaled box size.
            inst, sym = _face_aux(f)
            bx, by = float(f[0]) * scale_x, float(f[1]) * scale_y
            bw, bh = float(f[2]) * scale_x, float(f[3]) * scale_y
            if inst is not None and (scale_x != 1.0 or scale_y != 1.0):
                inst = cv2.resize(inst, (max(1, int(round(bw))), max(1, int(round(bh)))),
                                  interpolation=cv2.INTER_LINEAR)
            entries.append((bx, by, bw, bh, inst, sym))
        detections_by_frame[frame_index] = entries
    if detector == "sam3":
        _free_sam3_direct()   # hand the detector's VRAM back before diffusion runs
    _detection_elapsed = time.time() - _detection_started
    logger.info("[H3 Upscale] face detection: %d frame(s) scanned in %.1fs (%.2fs/frame)",
                len(sampled), _detection_elapsed, _detection_elapsed / max(1, len(sampled)))
    if not any(detections_by_frame.values()):
        return None

    dense = [detections_by_frame.get(i, []) for i in range(frame_count)]
    tracks = build_face_tracks(dense, max_gap=max(FACE_TRACK_MAX_GAP, stride - 1))
    for track in tracks:
        _inherit_track_masks(track)
    min_hits = FACE_TRACK_MIN_HITS if frame_count >= 4 else 1
    plan = [[] for _ in range(frame_count)]
    kept_tracks = 0
    crop_labels = set()
    for track_id, track in enumerate(tracks):
        if sum(1 for box in track["boxes"] if box is not None) < min_hits:
            continue  # a one-frame false positive is not a face track
        crop_w, crop_h = track_crop_size(track["boxes"], face_padding, img_w, img_h)
        if crop_w < 32 or crop_h < 32:
            continue
        kept_tracks += 1
        crop_labels.add("%dx%d" % (crop_w, crop_h))
        for frame_index, box in enumerate(track["boxes"]):
            if box is None:
                continue
            sym = track["syms"][frame_index]
            sym = 1.0 if sym is None else float(sym)
            if frontal_only_threshold > 0.0 and sym < frontal_only_threshold:
                continue
            plan[frame_index].append((
                box[0], box[1], box[2], box[3], track["insts"][frame_index],
                centered_crop(box, crop_w, crop_h, img_w, img_h), sym, track_id,
            ))
    if not any(plan):
        return None
    if stride > 1:
        logger.info(
            "[H3 Upscale] face detection stride %d (%s): %d of %d frame(s) scanned, boxes "
            "interpolated and masks inherited in between",
            stride, "auto" if auto_stride else "chosen", len(sampled), frame_count,
        )
    logger.info(
        "[H3 Upscale] face tracking: %d track(s) across %d frame(s), detection every %d, "
        "crop %s", kept_tracks, frame_count, stride, ", ".join(sorted(crop_labels)),
    )
    return plan


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

_H3_PIXEL_UPSCALE_MODELS = {}


def load_pixel_upscale_model(model_name):
    """Load an upscale model from models/upscale_models by dropdown name (cached)."""
    if not model_name or model_name == "None":
        return None
    cached = _H3_PIXEL_UPSCALE_MODELS.get(model_name)
    if cached is not None:
        return cached
    import comfy_extras.nodes_upscale_model
    model = comfy_extras.nodes_upscale_model.UpscaleModelLoader().load_model(model_name)[0]
    _H3_PIXEL_UPSCALE_MODELS[model_name] = model
    logger.info(f"[H3 Upscale] loaded pixel upscale model: {model_name}")
    return model


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
        upscale_models = ["None"]
        try:
            upscale_models += list(folder_paths.get_filename_list("upscale_models"))
        except Exception:
            pass
        latent_upscale_models = ["None"]
        try:
            latent_upscale_models += latent_upscale_model_names()
        except Exception:
            pass
        return {
            "required": {
                "grp_h3_upscale": ("BOOLEAN", {"default": False, "tooltip": "Group header. Click the bar to expand/collapse."}),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "vae": ("VAE",),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "Optional H3 audio VAE. The upscaler works on images only, so this input is accepted for graphs that wire it from a shared loader; it is not used here."}),
                "image": ("IMAGE", {"tooltip": "Single images keep the normal image upscale path. Batches with multiple images are treated as ordered video frames and H3 processes matching tiles as one temporal sequence. Needed for h3_upscale_mode = off / pixel, and as the latent-mode source when no latent is connected."}),
                "audio": ("AUDIO", {"tooltip": "When connected, this audio is returned on the audio output unchanged in every mode, taking precedence over the latent's decoded audio. The node renders frames, so it never uses, encodes or decodes it."}),
                "latent": ("LATENT", {"tooltip": "h3_upscale_mode = latent only: the H3 latent that latent_upscale_model resizes before it is decoded. When nothing is connected the image is encoded with the vae instead."}),
                "h3_upscale_mode": (["off", "pixel", "pixel segments", "latent"], {"default": "latent", "tooltip": "off = the H3 Upscale group is bypassed (no resize, no tile diffusion). pixel = upscale/diffuse the image, optionally through upscale_model. pixel segments = the same render, written out one file per render into a folder in the output directory instead of assembled into one batch (for clips too long to hold); it also takes a latent and decodes it one render window at a time. latent = resize the H3 latent with latent_upscale_model, then decode it."}),
                "upscale_model": (upscale_models, {"default": "None", "tooltip": "pixel mode: upscale model from models/upscale_models. None = plain bicubic resize."}),
                "latent_upscale_model": (latent_upscale_models, {"default": "None", "tooltip": "latent mode: H3 latent upscale checkpoint from models/latent_upscale_models."}),
                "scale_to_megapixels": ("FLOAT", {"default": 1.3, "min": 0.1, "max": 16.0, "step": 0.1, "tooltip": "Output size in megapixels: the aspect ratio is kept and both sides are snapped to a multiple of 32 px. The node never shrinks the image."}),
                "tile_width": ("INT", {"default": 8192, "min": 128, "max": 8192, "step": 16}),
                "tile_height": ("INT", {"default": 8192, "min": 128, "max": 8192, "step": 16}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 16}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": 256, "step": 16}),
                "seam_fix_denoise": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": 128, "step": 16}),
                "seed": ("INT", {"default": 45, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "sampler_name": (samplers, {"default": "euler" if "euler" in samplers else samplers[0]}),
                "scheduler": (schedulers, {"default": "simple" if "simple" in schedulers else schedulers[0]}),
                "denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "diffusion_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Opacity of the tile diffusion result over the clean upscale. Lower = less of the model's texture. (denoise = 0 skips the tile pass entirely.)"}),
                "edge_protect": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Restore the clean source along strong silhouette edges (removes jagged/hatched contours)."}),
                "detail_retention": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Restores fine detail wherever diffusion made a region smoother than the upscaled source. 0 = off."}),
                "latent_noise_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.005, "tooltip": "Extra noise added to tile latents. The old hard-coded 0.03 caused a dotted/lattice texture; keep at 0 unless you want it."}),
                "redraw_only": ("BOOLEAN", {"default": True, "tooltip": "Show ONLY the model's redraw: bypasses diffusion_mix, edge_protect, detail_retention, tone_lock and chroma_lock so no part of the source image can ghost through."}),
                "align_to_source": ("BOOLEAN", {"default": False, "tooltip": "Register each redrawn tile onto the source geometry (small clamped warp) so a few pixels of model drift cannot show up as a double image."}),
                "locks_follow_denoise": ("BOOLEAN", {"default": True, "tooltip": "Fade the 'keep the source' corrections (detail restore, edge protect, tone/chroma lock) out as denoise rises: full strength up to 0.25, off from 0.45. Prevents the ghosted second copy of a subject the model has re-drawn."}),
                "pixel_batch_overlap": ("INT", {"default": PIXEL_BATCH_OVERLAP, "min": 0, "max": 512, "step": 1, "tooltip": "pixel mode only. Frames of real context on EACH side of a chunk seam: every chunk is sampled with this many frames before and after the part it emits, so the frames either side of a seam both had real neighbours, and the seam is a cut between them rather than a blend of two different redraws. Costs twice this many frames per seam, so it is capped at an eighth of the chunk (a small chunk still advances most of its frames per render) - lower it if a long clip runs too slowly. 0 = chunks meet with no context."}),
                "grp_face_detail": ("BOOLEAN", {"default": False, "tooltip": "Group header. Click the bar to expand/collapse."}),
                "enable_face_detailer": ("BOOLEAN", {"default": False, "tooltip": "Enable AI Face Detection & Inpaint Pass."}),
                "same_face": ("BOOLEAN", {"default": False, "tooltip": "Off: every detected face is diffused on its own. On: the largest face becomes the master and every other face is drawn toward it, so the cast shares one face (the old behaviour)."}),
                "reference_face": ("IMAGE", {"tooltip": "Optional master reference portrait. When connected, EVERY detected face (largest first) is re-rendered toward this face."}),
                "face_mask": ("MASK", {"tooltip": "Optional. Each white blob = one face/head. Best option for creatures (SAM / GroundingDINO / hand-painted). Overrides auto-detection."}),
                "face_denoise": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.60, "step": 0.01, "tooltip": "Diffusion strength for face inpainting. Raise to ~0.40-0.50 when using reference_face so the identity is fully re-rendered."}),
                "face_padding": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Context added on each side of the head box, as a fraction of the box size (0 = exact box, rectangular crop)."}),
                "face_size": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64, "tooltip": "Maximum working size for the face diffusion. Crops are worked at about 0.5 MP (the reference workflow's crop size), never below their native resolution and never above this on the long side."}),
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
                "frontal_only_threshold": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Skip faces whose mask is not left/right symmetric, i.e. profile / side views. 0 = off. About 0.75 keeps frontal faces and drops profiles (needs SAM3 or a face_mask; the console logs the scores). In a video each frame of a tracked face is scored on its own."}),
                "face_detect_stride": (FACE_DETECT_STRIDE_LABELS,
                                       {"default": FACE_DETECT_STRIDE_LABELS[0],
                                        "tooltip": "Video only. auto scans every frame up to ~180 frames per clip and then widens the stride (capped at every 8th) so a long clip's tracking stays bounded; the named entries force 1 / 2 / 3. The boxes between detections are interpolated, their masks and scores inherited, and the whole track is smoothed. Every track also keeps ONE crop size for its whole length, so the redrawn face no longer jumps in size or position from frame to frame."}),
                "grp_refine": ("BOOLEAN", {"default": False, "tooltip": "Group header. Click the bar to expand/collapse."}),
                "enable_refiner": ("BOOLEAN", {"default": False, "tooltip": "Enable post-upscale edge & texture refinement."}),
                "refine_amount": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Strength of texture detail enhancement."}),
                "refine_radius": ("INT", {"default": 2, "min": 1, "max": 10, "tooltip": "Pixel radius for edge sharpening (scales with resolution)."}),
                "grp_dlss5": ("BOOLEAN", {"default": False, "tooltip": "Group header. Click the bar to expand/collapse."}),
                "enable_dlss5": ("BOOLEAN", {"default": True, "tooltip": "Enable DLSS 5 Neural Rendering post-pass."}),
                "dlss5_scale_to_megapixels": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 32.0, "step": 0.01, "tooltip": "DLSS 5 output size in megapixels, to two decimals (e.g. 2.75). Replaces the old upscaling_mode preset and controls only this pass, separately from the H3 Upscale group's scale_to_megapixels. The node picks the DLSS mode that covers this size, then fits that mode's output to the exact target with the source aspect kept, so the pass lands on the requested megapixel budget; a mode that overshoots the budget is trimmed back to it. 0 = no DLSS upscaling (reconstruct at the input size). The pass input itself is never shrunk."}),
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

    RETURN_TYPES = ("IMAGE", "IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("image", "face_debug", "audio", "segments")
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

    @staticmethod
    def _color_fx_chunk_frames(tensor_images, budget_bytes=256 * 1024 * 1024):
        """Frames per chunk for a bounded temporary tensor in the Color FX pass."""
        frame_bytes = max(1, int(tensor_images.shape[1]) * int(tensor_images.shape[2]) *
                          int(tensor_images.shape[3]) * int(tensor_images.element_size()))
        return max(1, min(int(tensor_images.shape[0]), int(budget_bytes // frame_bytes)))

    def _apply_gamma_torch(self, tensor_images, gamma_value):
        if gamma_value == 1.0: return tensor_images
        exponent = 1.0 / max(gamma_value, 0.01)
        step = self._color_fx_chunk_frames(tensor_images)
        for start in range(0, tensor_images.shape[0], step):
            part = tensor_images[start:start + step]
            part.clamp_(1e-5, 1.0).pow_(exponent)
        return tensor_images

    def _apply_simple_film_grain_torch(self, tensor_images, intensity, monochrome):
        if intensity == 0.: return tensor_images
        B, H, W, C = tensor_images.shape
        step = self._color_fx_chunk_frames(tensor_images)
        for start in range(0, B, step):
            part = tensor_images[start:start + step]
            noise = torch.randn((part.shape[0], H, W, 1 if monochrome else C),
                                device=part.device, dtype=part.dtype) * intensity
            if monochrome and C > 1:
                noise = noise.expand(-1, -1, -1, C)
            part.add_(noise).clamp_(0.0, 1.0)
        return tensor_images

    def _apply_vignette_torch(self, tensor_images, intensity, cx_r, cy_r):
        if intensity == 0.: return tensor_images
        B, H, W, C = tensor_images.shape
        cx, cy = W * cx_r, H * cy_r
        Y, X = torch.meshgrid(torch.arange(H, device=tensor_images.device) - cy, torch.arange(W, device=tensor_images.device) - cx, indexing='ij')
        dist = torch.sqrt(X**2 + Y**2)
        max_dist = torch.sqrt(torch.tensor(max(cx**2+cy**2, (W-cx)**2+cy**2, cx**2+(H-cy)**2, (W-cx)**2+(H-cy)**2), device=tensor_images.device))
        v_mask = (1.0 - (dist / max(1.0, max_dist))**2 * intensity).clamp(0.0, 1.0).unsqueeze(0).unsqueeze(-1)
        step = self._color_fx_chunk_frames(tensor_images)
        for start in range(0, B, step):
            part = tensor_images[start:start + step]
            part.mul_(v_mask).clamp_(0.0, 1.0)
        return tensor_images

    def _apply_color_fx(self, images, p):
        """Color FX stage. Same order and maths as FilmAuteur_LTX: shadows/highlights/HDR -> brightness/contrast/
        colour -> LUT -> sharpness/detail -> blurs -> chromatic aberration/scanlines/soft light (PIL), then
        gamma -> film grain -> vignette (tensor). images: [B,H,W,C] float 0..1.

        The batch can be hundreds of 1080p/4K frames, so every frame is written straight back into
        ``images`` instead of being collected and torch.cat'd into a second full batch, and the tensor
        effects run in bounded chunks.  This removes the multi-gigabyte ``tensor + noise`` allocation
        that exhausted RAM on long latent lists.  RGB is rewritten; an alpha channel is left untouched."""
        with torch.no_grad():
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
                images[i:i + 1, ..., :3] = pil2tensor(pil_image).to(images.device, images.dtype)
            # Tensor effects run on the RGB view only: when the batch carries an alpha
            # channel it must pass through untouched, exactly as the old cat() path did.
            rgb = images[..., :3]
            if p["enable_color_correction"] and p["gamma"] != 1.0:
                self._apply_gamma_torch(rgb, p["gamma"])
            if p["enable_stylistic_effects"]:
                if p["simple_film_grain_intensity"] > 0.0:
                    self._apply_simple_film_grain_torch(rgb, p["simple_film_grain_intensity"], p["simple_film_grain_monochrome"])
                if p["vignette_intensity"] > 0.0:
                    self._apply_vignette_torch(rgb, p["vignette_intensity"], p["vignette_center_x"], p["vignette_center_y"])
        return images

    def _face_init(self, orig_work, crop_mask, fx, fy, fw, fh, x1, y1, crop_w, crop_h, work_h, work_w,
                   master_face, master_alpha, master_box, master_aspect, reference_strength,
                   reference_pose_match, reference_flow_align):
        """Mix a master/reference face into a crop's init (and detail source) inside the face mask.

        Returns ``(init, detail_src, use_master, compat)``. Both the per-frame loop and the whole-track
        temporal pass call this, so the two cannot drift apart.
        """
        init = orig_work
        detail_src = orig_work
        # Pose compatibility: a frontal master blended onto a profile/back view only produces ghosting,
        # so fade the mix out when the face's shape differs from the master's.
        compat = 1.0
        if reference_pose_match and master_aspect:
            ratio = (fw / float(max(fh, 1))) / master_aspect
            compat = float(np.clip((np.log(1.6) - abs(np.log(max(ratio, 1e-3)))) / (np.log(1.6) - np.log(1.15)), 0.0, 1.0))
        eff_strength = reference_strength * compat
        use_master = master_face is not None and eff_strength > 0.01
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
            detail_src = orig_work * (1.0 - a_mix) + m * a_mix
        return init, detail_src, use_master, compat

    def _face_sequence_pass(self, vae, model, positive, negative, canvas, plan, steps, cfg,
                            sampler_name, scheduler, denoise, seed, face_padding, face_size,
                            master_face=None, master_alpha=None, master_box=None, master_aspect=None,
                            reference_strength=1.0, reference_pose_match=True, reference_flow_align=True):
        """Redraw every frame of each face track in ONE temporal diffusion pass.

        This is the reference workflow's core move: it tracks a subject, crops the whole clip to one
        constant crop size and runs a single KSampler over that crop sequence (denoise 1, 20 steps,
        cfg 1), so the model sees a clip instead of N unrelated stills. Diffusing one frame at a time
        was both slow (a full sampler run per frame) and flickery (no temporal context).

        Returns ``(decoded, masks)``, both keyed exactly like the frame loop indexes its plan entries,
        so the loop composites the pass with the very masks it was diffused with. Empty dicts mean the
        caller should keep diffusing per frame.
        """
        groups = {}
        for frame_index, entries in enumerate(plan):
            for face_index, entry in enumerate(entries):
                track_id = entry[7] if len(entry) > 7 else -1
                if track_id is None or int(track_id) < 0:
                    continue
                groups.setdefault(int(track_id), []).append((frame_index, face_index, entry))
        if not groups:
            return {}, {}

        canvas_h, canvas_w = int(canvas.shape[1]), int(canvas.shape[2])
        out = {}
        mask_cache = {}
        for track_id in sorted(groups):
            items = groups[track_id]
            if len(items) < 2:
                continue                      # one frame gains nothing from a temporal pass
            rects = []
            for frame_index, face_index, entry in items:
                fx, fy, fw, fh = [int(round(float(v))) for v in entry[:4]]
                crop = entry[5]
                if crop is not None and crop[2] - crop[0] >= 32 and crop[3] - crop[1] >= 32:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in crop]
                else:
                    x1, y1, x2, y2 = expand_box(fx, fy, fw, fh, face_padding, canvas_w, canvas_h)
                if min(y2 - y1, x2 - x1) < 32:
                    continue
                rects.append((frame_index, face_index, (x1, y1, x2, y2)))
            if len(rects) < 2:
                continue

            # One crop size for the whole track (track_crop_size guarantees it), so the crops form a
            # clean sequence - exactly what MVEx_SubjectCrop feeds the sampler in the workflow.
            first = rects[0][2]
            work_h, work_w = face_work_size(first[3] - first[1], first[2] - first[0], face_size)
            # Phase 1: place each frame's instance mask in the track's one fixed crop and clean it.
            geometry, placed, from_instance = [], [], []
            for frame_index, face_index, (rx1, ry1, rx2, ry2) in rects:
                entry = plan[frame_index][face_index]
                fx, fy, fw, fh = [int(round(float(v))) for v in entry[:4]]
                crop_h, crop_w = ry2 - ry1, rx2 - rx1
                inst_mask = entry[4]
                if inst_mask is not None:
                    mask_np = cleanup_instance_mask(place_instance_mask(inst_mask, fx, fy, rx1, ry1, crop_h, crop_w))
                    from_instance.append(True)
                else:
                    mask_np = create_face_blend_mask(crop_h, crop_w, feather=0.35, device=canvas.device)
                    mask_np = mask_np[0, ..., 0].detach().cpu().numpy().astype(np.float32)
                    from_instance.append(False)
                placed.append(mask_np)
                geometry.append((frame_index, face_index, (rx1, ry1, rx2, ry2), fx, fy, fw, fh, crop_h, crop_w))

            # Phase 2: temporal majority over the track's masks. Per-frame segmentation flickers at the
            # edges and that flicker survives into the composite as a pulsing seam; a video tracker does
            # not have it because it carries memory between frames.
            if all(from_instance) and len(placed) > 2 and FACE_MASK_TEMPORAL > 0:
                stack = np.stack([(m > 0.5).astype(np.uint8) for m in placed])
                smoothed = []
                for index in range(len(stack)):
                    lo = max(0, index - FACE_MASK_TEMPORAL)
                    hi = min(len(stack), index + FACE_MASK_TEMPORAL + 1)
                    smoothed.append((stack[lo:hi].mean(axis=0) >= 0.5).astype(np.float32))
            else:
                smoothed = placed

            crops, keys = [], []
            for (frame_index, face_index, (rx1, ry1, rx2, ry2), fx, fy, fw, fh,
                 crop_h, crop_w), mask_np in zip(geometry, smoothed):
                crop_mask = torch.from_numpy(np.clip(cv2.GaussianBlur(
                    mask_np, (0, 0), sigmaX=max(1.5, 0.015 * max(crop_h, crop_w))), 0.0, 1.0))
                crop_mask = crop_mask.view(1, crop_h, crop_w, 1).to(canvas.device)
                crop_mask = crop_mask * border_taper(crop_h, crop_w, 0.08, canvas.device)
                region = canvas[frame_index:frame_index + 1, ry1:ry2, rx1:rx2, :3]
                if region.shape[1] != work_h or region.shape[2] != work_w:
                    region = _resize_bhwc(region, work_h, work_w)
                # Same init the per-frame loop would build, including any reference-face mix.
                init, _detail_src, _use_master, _compat = self._face_init(
                    region, crop_mask, fx, fy, fw, fh, rx1, ry1, crop_w, crop_h, work_h, work_w,
                    master_face, master_alpha, master_box, master_aspect, reference_strength,
                    reference_pose_match, reference_flow_align,
                )
                crops.append(init)
                keys.append((frame_index, face_index))
                mask_cache[keys[-1]] = crop_mask.detach()

            # denoise trims the schedule to its last fraction, so asking for 20 steps at denoise 0.25
            # runs only 4 sampler iterations - a soft touch-up rather than the full redraw the reference
            # workflow gets from its 20-step pass. Ask for a longer schedule instead, so this pass runs
            # the same number of iterations as a 20-step KSampler at that denoise.
            seq_steps = max(1, int(round(steps / max(denoise, 0.01))))
            logger.info("[H3 Upscale] face track %d: %d frame(s) in one temporal pass at %dx%d, "
                        "%d steps at denoise %.2f",
                        track_id, len(crops), work_w, work_h,
                        max(1, int(round(seq_steps * denoise))), denoise)
            import time
            _pass_started = time.time()
            for start in range(0, len(crops), FACE_SEQ_CHUNK):
                chunk = torch.cat(crops[start:start + FACE_SEQ_CHUNK], dim=0)
                decoded = self._diffuse_sequence(
                    vae, model, positive, negative, chunk, seq_steps, cfg, sampler_name, scheduler,
                    denoise, seed,
                )
                if decoded.shape[1] != work_h or decoded.shape[2] != work_w:
                    decoded = _resize_bhwc(decoded, work_h, work_w)
                for offset in range(min(decoded.shape[0], len(chunk))):
                    out[keys[start + offset]] = decoded[offset:offset + 1]
            logger.info("[H3 Upscale] face track %d temporal pass finished in %.1fs",
                        track_id, time.time() - _pass_started)
        return out, mask_cache

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
                          denoise, seed, noise_offset=0.0, pin_frames=0):
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
        # The leading frames of this sequence are already emitted and on screen, so they are
        # pinned: a zero denoise mask makes the sampler keep their latent exactly as encoded
        # instead of redrawing them, which both stops the context drifting and stops a fresh
        # context from pushing the frames that follow. The count is mapped onto the VAE's
        # compressed temporal grid and rounded down, so a pinned latent can only ever be inside
        # the overlap - never a frame this chunk emits.
        pin_latents = 0
        if pin_frames > 0 and latent_frame_count > 1:
            pin_latents = int(int(pin_frames) * latent_frame_count // max(1, frame_count))
            pin_latents = max(0, min(pin_latents, latent_frame_count - 1))
        if noise_offset > 0.0:
            offset = torch.randn_like(latent_samples) * noise_offset
            if pin_latents > 0:
                # Pinned context must stay exactly what the previous chunk emitted, so the
                # offset applies to the frames this chunk renders, not to the continuation.
                offset[:, :, :pin_latents] = 0.0
            latent_samples = latent_samples + offset

        # Keep the dummy audio stream at the H3 still-image/audio-free default;
        # the issue here is the mismatched visual conditioning rows, not audio.
        audio_dummy = latent_samples.new_zeros((1, 32, 2, 65))
        denoise_mask = None
        if pin_latents > 0:
            video_mask = latent_samples.new_ones(
                (1, 1, latent_frame_count, latent_samples.shape[3], latent_samples.shape[4])
            )
            video_mask[:, :, :pin_latents] = 0.0
            # The dummy audio stream has nothing to continue, so it is pinned whole.
            audio_mask = audio_dummy.new_zeros((1, 1) + tuple(audio_dummy.shape[2:]))
            if hasattr(comfy, "nested_tensor") and hasattr(comfy.nested_tensor, "NestedTensor"):
                denoise_mask = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
            else:
                denoise_mask = (video_mask, audio_mask)
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

        def run_sample(mask):
            extra = {} if mask is None else {"noise_mask": mask}
            # With the pin off this is the same call the node has always made, keyword for
            # keyword, so nothing about plain sampling depends on the pin existing.
            return comfy.sample.sample(
                model, noise, steps, cfg, sampler_name, scheduler,
                sequence_positive, sequence_negative, latent_samples,
                denoise=denoise, seed=seed, **extra,
            )

        sampled = None
        if denoise_mask is not None and not getattr(self, "_pin_context_unsupported", False):
            try:
                sampled = run_sample(denoise_mask)
            except Exception as error:
                # Masking a nested video latent is not universal: some samplers only take a plain
                # 5D mask, or reject one at all. Say so once and keep rendering - an unmasked pass
                # is the old behaviour, not a failure.
                if not getattr(self, "_pin_context_unsupported", False):
                    self._pin_context_unsupported = True
                    logger.warning(
                        "[H3 Upscale] this sampler rejected the seam context mask (%s: %s); "
                        "falling back to re-sampling the overlap frames",
                        type(error).__name__, error,
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if sampled is None:
            sampled = run_sample(None)
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
        # supported frame grid (e.g. 41 inputs decode as 39 frames). Get one frame
        # per input frame back without moving content between frames
        # (see reconcile_decoded_frames).
        decoded_count = int(decoded.shape[0])
        decoded = reconcile_decoded_frames(decoded, frame_count)
        if decoded_count != int(frame_count):
            # Worth saying: a chunk's last frames are the next chunk's first frames, and this
            # is where what the VAE handed back can disagree with what was asked for.
            logger.info(
                "[H3 Upscale] video batch: the VAE returned %d frame(s) for %d in the "
                "sequence (%+d)",
                decoded_count, int(frame_count), decoded_count - int(frame_count),
            )
        return decoded[..., :3].to(device=images.device, dtype=torch.float32)

    def _upscale_video_batch(self, images, vae, model, positive, negative,
                             tile_width, tile_height, tile_padding, steps, cfg, sampler_name,
                             scheduler, denoise, seed, latent_noise_offset, lock_scale,
                             t_mix, t_edge, t_detail, align_to_source, tile_diffusion,
                             pin_frames=0):
        """Run matching spatial tiles across every frame in one temporal H3 diffusion call."""
        frame_count, target_h, target_w, channels = images.shape
        if denoise < 0.01 or not tile_diffusion:
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
                    noise_offset=latent_noise_offset, pin_frames=pin_frames,
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

    def _latent_upscale_images(self, latent, image, vae, scale_to_megapixels, latent_upscale_model):
        """h3_upscale_mode = latent: resize the H3 latent with the learned 3D resizer, then decode.

        The latent input wins; without it the image is encoded with the vae first. Returns
        (frames [T,H,W,3], target_h, target_w) in pixel space.
        """
        if vae is None:
            raise ValueError(
                "h3_upscale_mode = latent needs the vae input to decode the upscaled H3 latent"
            )
        if not latent_upscale_model or latent_upscale_model == "None":
            raise ValueError(
                "h3_upscale_mode = latent needs a latent_upscale_model "
                "(place H3 latent checkpoints in models/latent_upscale_models)"
            )

        samples = latent.get("samples") if isinstance(latent, dict) else latent
        if samples is None:
            if image is None:
                raise ValueError(
                    "h3_upscale_mode = latent needs the latent input "
                    "(or an image to encode with the vae)"
                )
            encoded = vae.encode(image[..., :3])
            samples = encoded["samples"] if isinstance(encoded, dict) else encoded
            source_h, source_w = int(image.shape[1]), int(image.shape[2])
        else:
            source_h = int(samples.shape[-2]) * 16
            source_w = int(samples.shape[-1]) * 16

        if getattr(samples, "is_nested", False):
            samples = samples.unbind()[0]           # video stream of an AV latent
        elif isinstance(samples, (list, tuple)):
            samples = samples[0]
        samples = samples.detach()
        if samples.dim() == 4:                      # [B,C,H,W] still latent -> one frame
            samples = samples.unsqueeze(2)
        if samples.dim() != 5:
            raise ValueError(
                "h3_upscale_mode = latent expects a BxCxTxHxW H3 latent; "
                f"received {tuple(samples.shape)}"
            )

        # Pixel target from scale_to_megapixels, snapped to 32 px (a multiple of the 16 px H3
        # VAE grid), never shrinking.
        target_w, target_h = scale_to_megapixel_target(source_w, source_h, scale_to_megapixels)
        target_h = max(target_h, 16)
        target_w = max(target_w, 16)
        target_h = max(target_h, int(samples.shape[-2]) * 16)
        target_w = max(target_w, int(samples.shape[-1]) * 16)
        target_lat_h = max(1, target_h // 16)
        target_lat_w = max(1, target_w // 16)

        upscaled = upscale_h3_latent(
            samples,
            model_name=latent_upscale_model,
            target_h=target_lat_h,
            target_w=target_lat_w,
        )

        decoded = vae.decode(upscaled)
        if isinstance(decoded, dict):
            decoded = decoded["samples"]
        # Video VAEs may return [B,T,H,W,C] or [B,C,T,H,W]; flatten to ComfyUI's [T,H,W,C].
        if decoded.dim() == 5:
            if decoded.shape[1] in (1, 3, 4) and decoded.shape[-1] > 4:  # [B,C,T,H,W]
                decoded = decoded.permute(0, 2, 3, 4, 1)
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        elif decoded.dim() == 4 and decoded.shape[1] in (1, 3, 4) and decoded.shape[-1] > 4:
            decoded = decoded.permute(0, 2, 3, 1)
        if decoded.dim() != 4 or decoded.shape[-1] < 3:
            raise ValueError(
                "H3 latent decode must return an image sequence; "
                f"received {tuple(decoded.shape)}"
            )

        frames = decoded[..., :3].clamp(0.0, 1.0)
        if frames.shape[1] != target_h or frames.shape[2] != target_w:
            frames = _resize_bhwc(frames, target_h, target_w)
        device = image.device if image is not None else torch.device("cpu")
        logger.info(
            "[H3 Upscale] latent mode: latent %dx%d -> %dx%d, %d frame(s) from %d latent frame(s)",
            int(samples.shape[-1]), int(samples.shape[-2]), target_lat_w, target_lat_h,
            frames.shape[0], int(samples.shape[2]),
        )
        return frames.to(device=device, dtype=torch.float32), target_h, target_w

    # ------------------------------------------------------------------------------------------
    def _decode_latent_audio(self, latent, audio_vae):
        """Decode the audio stream of a nested H3 AV latent into a ComfyUI AUDIO dict.

        Returns ``None`` when no audio can be produced (no latent, no ``audio_vae``, a
        video-only latent, or a VAE that fails to decode) so the ``audio`` output can be
        left unconnected without breaking image-only graphs.
        """
        if audio_vae is None or latent is None:
            return None
        samples = latent.get("samples") if isinstance(latent, dict) else latent
        if samples is None:
            return None
        if not (getattr(samples, "is_nested", False)
                or isinstance(samples, comfy.nested_tensor.NestedTensor)):
            return None
        streams = samples.unbind()
        if len(streams) < 2:
            return None
        audio = streams[1].detach()
        if audio.dim() != 4:
            logger.warning(
                "[H3 Upscale] audio stream has shape %s; expected BxCxTxF, returning no audio",
                tuple(audio.shape),
            )
            return None

        try:
            waveform = audio_vae.decode(audio)
        except Exception as exc:                      # pragma: no cover - VAE specific
            logger.warning("[H3 Upscale] audio decode failed, returning no audio: %s", exc)
            return None
        if isinstance(waveform, dict):
            waveform = waveform.get("waveform")
        if not torch.is_tensor(waveform):
            logger.warning("[H3 Upscale] audio decode returned %r; returning no audio",
                           type(waveform).__name__)
            return None

        waveform = waveform.detach().to(device="cpu", dtype=torch.float32).movedim(-1, 1)
        sample_rate = int(getattr(
            audio_vae, "audio_sample_rate", getattr(audio_vae, "sample_rate", 44100)
        ))
        logger.info(
            "[H3 Upscale] decoded %d audio sample(s) at %d Hz from the H3 audio latent",
            int(waveform.shape[-1]), sample_rate,
        )
        return {"waveform": waveform, "sample_rate": sample_rate}

    # ComfyUI hands a node the whole list of values an upstream produced once INPUT_IS_LIST is set.
    # The upscaler takes that list, renders every latent and merges the results, so its two media
    # outputs are a single IMAGE batch and a single AUDIO value instead of a list of per-latent
    # batches that a downstream video node can only sample one element of.
    INPUT_IS_LIST = True

    @staticmethod
    def _pick_list_value(values, index):
        """ComfyUI's own rule for uneven lists: repeat the last value when the list is short."""
        if not values:
            return None
        return values[index] if len(values) > index else values[-1]

    @staticmethod
    def _conditioning_list(value):
        """The conditioning for each latent/shot, from a CONDITIONING input.

        A conditioning is already a list of ``[tensor, dict]`` entries, so unlike the scalar
        inputs it must not be unwrapped as a one-element list. With INPUT_IS_LIST and a latent
        list, ComfyUI can also hand over one conditioning per latent (occasionally wrapped one
        level deep); each is returned separately so the matching one can be picked per latent.
        """
        def is_entry(item):
            return (isinstance(item, (list, tuple)) and len(item) >= 2
                    and isinstance(item[1], dict))

        if value is None:
            return [None]
        if is_entry(value):                 # a bare [tensor, dict] entry
            return [[value]]
        if not isinstance(value, (list, tuple)) or not value:
            return [value]
        if all(is_entry(entry) for entry in value):
            return [list(value)]            # one conditioning, however many entries it has
        while (isinstance(value, (list, tuple)) and len(value) == 1
               and isinstance(value[0], (list, tuple)) and not is_entry(value[0])):
            first = value[0]
            if all(is_entry(entry) for entry in first):
                break                       # one-element wrapper around one conditioning
            value = first
        if isinstance(value, (list, tuple)) and value and all(is_entry(entry) for entry in value):
            return [list(value)]
        return list(value) if isinstance(value, (list, tuple)) else [value]

    @staticmethod
    def _video_latent_samples(latent):
        """The video stream of an H3 latent as [B, C, T, h, w], or None when there is none."""
        if latent is None:
            return None
        samples = latent.get("samples") if isinstance(latent, dict) else latent
        if samples is None:
            return None
        if getattr(samples, "is_nested", False):
            samples = samples.unbind()[0]          # video stream of an AV latent
        elif isinstance(samples, (list, tuple)):
            samples = samples[0]
        samples = samples.detach()
        if samples.dim() == 4:                     # [B,C,H,W] still latent -> one frame
            samples = samples.unsqueeze(2)
        if samples.dim() != 5:
            raise ValueError(
                "pixel segments from a latent expects a BxCxTxHxW H3 latent; "
                f"received {tuple(samples.shape)}")
        return samples

    @staticmethod
    def _latent_step_at(total, frame):
        """The latent step that holds a pixel frame: 2 steps cover the first 5 frames, then 5
        steps cover every 17. Render lengths are grid lengths, so a window edge is a step edge."""
        total = max(1, int(total))
        frame = max(0, int(frame))
        step = 2 + 5 * ((frame - 5) // 17) if frame >= 5 else 0
        step = max(0, min(step, total - 1))
        while step + 1 < total and video_frames_for_latents(step + 1) <= frame:
            step += 1
        while step > 0 and video_frames_for_latents(step) > frame:
            step -= 1
        return step

    def _decode_latent_window(self, vae, samples, first, last):
        """Pixel frames [first, last) of a video latent, decoded from the steps that hold them.

        Only this window's steps are decoded, so a long shot is decoded one render at a time
        instead of as a whole clip. The steps either side come back as well, which is what makes
        the slice below exact: the decode is aligned on a step edge and the frames before the
        window are simply dropped.
        """
        if vae is None:
            raise ValueError(
                "pixel segments from a latent needs the vae input to decode each window")
        total = int(samples.shape[2])
        first, last = int(first), int(last)
        step_from = self._latent_step_at(total, first)
        step_to = min(total, self._latent_step_at(total, max(first, last - 1)) + 1)
        decoded = vae.decode(samples[:, :, step_from:step_to])
        if isinstance(decoded, dict):
            decoded = decoded["samples"]
        if decoded.dim() == 5:
            if decoded.shape[1] in (1, 3, 4) and decoded.shape[-1] > 4:  # [B,C,T,H,W]
                decoded = decoded.permute(0, 2, 3, 4, 1)
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        elif decoded.dim() == 4 and decoded.shape[1] in (1, 3, 4) and decoded.shape[-1] > 4:
            decoded = decoded.permute(0, 2, 3, 1)
        if decoded.dim() != 4 or decoded.shape[-1] < 3:
            raise ValueError(
                "a latent window decode must return an image sequence; "
                f"received {tuple(decoded.shape)}")
        base = video_frames_for_latents(step_from)
        expected = video_frames_for_latents(step_to) - base
        if int(decoded.shape[0]) != expected:
            logger.warning(
                "[H3 Upscale] pixel segments: latent steps %d-%d decoded to %d frame(s) where "
                "the packing expects %d - the window is taken from what came back",
                step_from, step_to, int(decoded.shape[0]), expected)
        low = max(0, first - base)
        high = min(int(decoded.shape[0]), last - base)
        if high <= low:
            raise ValueError(
                "a latent window decoded no frame for %d-%d (decoded %d from step %d)"
                % (first + 1, last, int(decoded.shape[0]), step_from))
        return decoded[low:high, ..., :3].detach().float().clamp(0.0, 1.0).cpu()

    @staticmethod
    def _segments_folder():
        """A fresh folder in ComfyUI's temp directory for one segmented run."""
        import tempfile, time, uuid
        base = None
        try:
            import folder_paths
            getter = getattr(folder_paths, "get_temp_directory", None)
            base = getter() if callable(getter) else None
        except Exception:
            base = None
        base = base or tempfile.gettempdir()
        os.makedirs(base, exist_ok=True)
        folder = os.path.join(base, "h3_segments", "%s_%s" % (
            time.strftime("%Y%m%d_%H%M%S"), uuid.uuid4().hex[:6]))
        os.makedirs(folder, exist_ok=True)
        return folder

    @staticmethod
    def _write_segment(folder, number, frames, fps=FPS):
        """Encode one render's frames to their own file and let them go.

        H.264 through the PyAV this module already imports. The frames are converted a frame at a
        time, so the peak is one render's frames as they arrive, and the file is closed before the
        next render starts: nothing is ever assembled for the whole clip. The segments join in
        order because every seam between two of them is a cut.
        """
        path = os.path.join(folder, "segment_%04d.mp4" % int(number))
        height, width = int(frames.shape[1]), int(frames.shape[2])
        container = av.open(path, mode="w")
        try:
            stream = container.add_stream("libx264", rate=int(fps))
            stream.width, stream.height = width, height
            # 4:2:0 needs both sides even; the DLSS target can be odd (2525 wide here), and 4:4:4
            # is the profile that holds an odd width.
            stream.pix_fmt = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"
            # One keyframe per second: the timeline editor seeks by rewinding to the keyframe
            # before the frame it wants, so this is what keeps scrubbing a long clip responsive.
            stream.options = {"crf": "16", "preset": "medium",
                              "g": str(int(fps)), "keyint_min": str(int(fps))}
            for at in range(int(frames.shape[0])):
                rgb = (frames[at].detach().float().clamp(0.0, 1.0)
                       .mul(255.0).round().byte().cpu().numpy())
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
        with open(os.path.join(folder, "concat.txt"), "a", encoding="utf-8") as handle:
            handle.write("file '%s'\n" % os.path.basename(path))
        return path

    @staticmethod
    def _clip_buffer(shape, dtype, tag):
        """One buffer for a whole clip: RAM when it fits, a memory-mapped file when it does not.

        Frames are written straight into it as each render finishes, and the mapping is flushed
        after every render, so the part of the clip that is already done leaves RAM instead of
        piling up there. The returned tensor reads through to the file rather than being rebuilt:
        writing the chunks out and loading them back at the end would reassemble exactly the
        tensor that ran out of memory.
        """
        shape = tuple(int(v) for v in shape)
        np_dtype = {torch.float32: np.float32, torch.float16: np.float16}.get(dtype)
        total = int(np.prod(shape)) * int(np.dtype(np_dtype).itemsize if np_dtype else 4)
        try:
            import psutil
            free = int(psutil.virtual_memory().available)
        except Exception:
            free = None
        fits = (total <= 8 * 1024 ** 3) if free is None else (total <= free // 2)
        if np_dtype is None or fits:
            return torch.empty(shape, dtype=dtype), None
        import shutil, tempfile, uuid
        directory = None
        try:
            import folder_paths
            getter = getattr(folder_paths, "get_temp_directory", None)
            directory = getter() if callable(getter) else None
            if not directory:
                directory = os.path.join(
                    getattr(folder_paths, "base_path", tempfile.gettempdir()), "temp")
        except Exception:
            directory = None
        directory = directory or tempfile.gettempdir()
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "h3_%s_%d_%s.raw" % (
            tag.replace(" ", "_"), os.getpid(), uuid.uuid4().hex[:8]))
        free_disk = shutil.disk_usage(directory).free
        if total > free_disk:
            raise RuntimeError(
                "[H3 Upscale] %s needs %.2f GiB of disk and %s has %.2f GiB free - shorten the "
                "clip, or point ComfyUI's temp folder at a bigger drive"
                % (tag, total / 2 ** 30, directory, free_disk / 2 ** 30))
        mapping = np.memmap(path, dtype=np_dtype, mode="w+", shape=shape)
        logger.info(
            "[H3 Upscale] %s: %.2f GiB does not fit in RAM - writing it to %s and flushing it "
            "after every render", tag, total / 2 ** 30, path)
        return torch.from_numpy(np.asarray(mapping)), mapping

    @staticmethod
    def _merge_image_batches(batches):
        """One IMAGE batch holding every latent's frames, in order."""
        if len(batches) == 1:
            return batches[0]
        first = batches[0]
        merged = []
        for index, batch in enumerate(batches):
            if batch.shape[1] != first.shape[1] or batch.shape[2] != first.shape[2]:
                logger.warning(
                    "[H3 Upscale] latent %d produced %dx%d frames, resizing to %dx%d so every "
                    "frame can share one batch",
                    index + 1, batch.shape[2], batch.shape[1], first.shape[2], first.shape[1],
                )
                batch = _resize_bhwc(batch, first.shape[1], first.shape[2])
            merged.append(batch)
        return torch.cat(merged, dim=0)

    @staticmethod
    def _merge_audio(audios):
        """One AUDIO value from every latent's stream, joined along time in frame order."""
        audios = [a for a in audios if isinstance(a, dict) and a.get("waveform") is not None]
        if not audios:
            return None
        if len(audios) == 1:
            return audios[0]
        rates = [int(a.get("sample_rate") or 0) for a in audios]
        if len(set(rates)) > 1:
            logger.warning("[H3 Upscale] latents decoded at different sample rates %s; the merged "
                           "audio keeps the first (%d Hz)", rates, rates[0])
        return {"waveform": torch.cat([a["waveform"] for a in audios], dim=-1),
                "sample_rate": audios[0].get("sample_rate")}

    def upscale(self, **kwargs):
        """Render every input latent and return ONE image batch and ONE audio value.

        Without INPUT_IS_LIST ComfyUI ran this node once per latent and the image output became a
        list of batches: the preview cycled through them and a receiving video node only ever saw
        the last one. The latents are now rendered here, in order, and concatenated, so N latents
        are N groups of frames of the same video rather than N separate videos.
        """
        latents = kwargs.pop("latent", None)
        images = kwargs.pop("image", None)
        # pixel / pixel segments: a connected audio is only carried to the audio output. It is
        # popped here so _upscale_pipeline never sees it (that function has no audio parameter).
        audio_passthrough = kwargs.pop("audio", None)
        if isinstance(audio_passthrough, (list, tuple)) and len(audio_passthrough) == 1:
            audio_passthrough = audio_passthrough[0]
        # positive / negative are already lists (a conditioning is a list of entries), so the
        # generic unwrap below would turn a one-entry conditioning into [tensor, dict]; and with
        # a latent list the sampler supplies one conditioning per latent. Split them out first.
        positive = kwargs.pop("positive", None)
        negative = kwargs.pop("negative", None)
        positive_list = self._conditioning_list(positive)
        negative_list = self._conditioning_list(negative)
        # Every other input is a single value wrapped in a one-element list; unwrap it. A list with
        # more than one entry is a genuine list of upstream values and is left alone.
        inputs = {key: (value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value)
                  for key, value in kwargs.items()}

        latent_list = list(latents) if isinstance(latents, (list, tuple)) else (
            [] if latents is None else [latents])
        image_list = list(images) if isinstance(images, (list, tuple)) else (
            [] if images is None else [images])
        if not latent_list:
            latent_list = [None]
        if not image_list:
            image_list = [None]
        count = max(len(latent_list), len(image_list))
        # A connected audio always wins: it is the caller's own stream. It goes out over a
        # latent's decoded audio in latent / off, and is the only audio in the pixel modes,
        # which have none of their own.
        passthrough_audio = audio_passthrough
        if passthrough_audio is not None:
            # It replaces the latent's decoded stream, so there is nothing left to decode:
            # no audio VAE is the same path an image-only graph takes.
            inputs["audio_vae"] = None
            logger.info("[H3 Upscale] audio: the connected audio is passed straight through "
                        "(latent audio decode skipped)")

        results = [
            self._upscale_one(
                image=self._pick_list_value(image_list, index),
                latent=self._pick_list_value(latent_list, index),
                positive=self._pick_list_value(positive_list, index),
                negative=self._pick_list_value(negative_list, index),
                **inputs,
            )
            for index in range(count)
        ]
        if len(results) == 1:
            result = results[0]
            if passthrough_audio is not None:
                return (result[0], result[1], passthrough_audio, result[3])
            return result

        images_out = self._merge_image_batches([result[0] for result in results])
        debug_out = self._merge_image_batches([result[1] for result in results])
        logger.info("[H3 Upscale] %d latent(s) merged into one batch of %d frame(s)",
                    len(results), images_out.shape[0])
        audio_out = (passthrough_audio if passthrough_audio is not None
                     else self._merge_audio([result[2] for result in results]))
        return (images_out, debug_out, audio_out,
                ";".join(result[3] for result in results if result[3]))

    # A ComfyUI image batch is one tensor through the whole pixel pipeline: resized, diffused tile by
    # tile, face-detailed and colour-graded. A few hundred 4K frames of that is what runs a card out
    # of memory, so long batches are rendered in chunks of PIXEL_BATCH_CHUNK frames that meet at a
    # seam with PIXEL_BATCH_OVERLAP frames of real context on each side of it: the next chunk
    # re-renders the previous seam's tail as context, and its own tail is context for the seam after
    # it. Frames are never rebuilt into the output twice, so a seam is a cut between two renders
    # that both saw real neighbours (dissolving two different redraws double-exposed the frame).
    def _shared_face_plan(self, kwargs, image):
        """Track the clip's faces ONCE, on the source frames, for every pixel chunk to share.

        Chunking used to give each chunk its own detection pass, so the same face could change track
        id and crop size at a chunk boundary and the redraw changed scale with it. This plan is scaled
        into each chunk's output resolution (slice_shared_face_plan), keeping one track, one crop size
        and one mask per face for the whole clip - and it costs one detection pass, not one per chunk.
        """
        if not kwargs.get("enable_face_detailer") or kwargs.get("same_face"):
            return None
        if image is None or int(image.shape[0]) < 2:
            return None
        try:
            plan = plan_video_face_tracks(
                image,
                kwargs.get("face_detector") or "auto",
                kwargs.get("face_detector_model") or "",
                float(kwargs.get("face_confidence") or 0.35),
                kwargs.get("face_mask"),
                kwargs.get("manual_face_boxes") or "",
                kwargs.get("sam3_prompt") or "face",
                kwargs.get("face_detect_stride") or FACE_DETECT_STRIDE_LABELS[0],
                float(kwargs.get("face_padding") or 0.25),
                float(kwargs.get("frontal_only_threshold") or 0.0),
            )
        except Exception as e:
            logger.warning("[H3 Upscale] clip-wide face plan unavailable (%s); each chunk detects "
                           "its own faces", e)
            return None
        if plan is None:
            logger.info("[H3 Upscale] pixel batch: no tracked face in the clip, chunks fall back to "
                        "per-frame detection")
            return None
        tracks = {entry[7] for frame in plan for entry in frame}
        logger.info("[H3 Upscale] pixel batch: one face plan for the whole clip (%d frame(s), "
                    "%d track(s)); every chunk reuses its tracks and crop sizes",
                    int(image.shape[0]), len(tracks))
        return {"plan": plan, "source_h": int(image.shape[1]), "source_w": int(image.shape[2])}

    def _clip_face_cache(self, kwargs, image, shared):
        """Diffuse every tracked face ONCE, over the whole clip, before any pixel chunk runs.

        A chunk only owns its own frames, so a per-chunk face pass can never sample one track as one
        sequence: every chunk restarts the sampler and the track's later half has no memory of the
        earlier half's grain or micro-detail. This pass crops each track out of the SOURCE clip,
        upscales those crops to output resolution (a face crop is ~0.5 MP, not a full 4K frame), re-detects
        the masks inside them (refine_plan_masks), and runs the track's whole crop sequence through one
        temporal diffusion - the reference workflow's core move. The chunks then only composite the cached
        crops, so the face is identical on both sides of a chunk boundary and the seam has nothing
        left to show.

        Returns ``(seq, masks)`` keyed by ``(frame index, face index)`` in clip coordinates, held on the
        CPU, or None when the caller should let each chunk diffuse its own faces.
        """
        if shared is None or kwargs.get("same_face"):
            return None
        src_plan = shared.get("plan") or []
        if not src_plan:
            return None
        vae, model = kwargs.get("vae"), kwargs.get("model")
        if vae is None or model is None:
            return None
        target_w, target_h = scale_to_megapixel_target(
            int(image.shape[2]), int(image.shape[1]),
            float(kwargs.get("scale_to_megapixels") or 1.0),
        )
        out_plan = slice_shared_face_plan(dict(shared, offset=0, frames=len(src_plan)),
                                          target_h, target_w)
        if not out_plan:
            return None
        groups = {}
        for frame_index, entries in enumerate(out_plan):
            for face_index, entry in enumerate(entries):
                track_id = entry[7] if len(entry) > 7 else -1
                if track_id is None or int(track_id) < 0:
                    continue
                groups.setdefault(int(track_id), []).append((frame_index, face_index, entry))
        if not groups:
            return None

        detector = kwargs.get("face_detector") or "auto"
        model_path = kwargs.get("face_detector_model") or ""
        confidence = float(kwargs.get("face_confidence") or 0.35)
        sam3_prompt = kwargs.get("sam3_prompt") or "face"
        face_padding = float(kwargs.get("face_padding") or 0.25)
        face_size = int(kwargs.get("face_size") or 1024)
        face_denoise = float(kwargs.get("face_denoise") or 0.25)
        seed = int(kwargs.get("seed", 0) or 0)
        master = alpha = mbox = maspect = None
        if kwargs.get("reference_face") is not None:
            try:
                master, alpha, maspect, mbox = prepare_reference_face(
                    kwargs["reference_face"], face_padding, detector, model_path, confidence, sam3_prompt,
                )
                master = master.to(image.device)
                alpha = alpha.to(image.device)
            except Exception as e:
                logger.warning("[H3 Upscale] reference face unavailable for the clip face pass (%s)", e)
                master = alpha = mbox = maspect = None

        # Decide RAM vs disk before running: the crop geometry is known from the plan, so the store
        # never has to migrate what it already holds.
        estimate = 0
        for track_items in groups.values():
            if len(track_items) < 2:
                continue
            rect = track_items[0][2][5]
            crop_w = max(0, int(rect[2]) - int(rect[0]))
            crop_h = max(0, int(rect[3]) - int(rect[1]))
            estimate += len(track_items) * crop_w * crop_h * 16   # 3 colour channels + 1 mask, float32
        budget = face_cache_budget_bytes()
        store = _ClipFaceCache(budget, spill=estimate > budget)
        upscale_model_name = kwargs.get("upscale_model") or "None"
        if upscale_model_name != "None":
            logger.info("[H3 Upscale] clip-wide face pass: crops upscaled with %s, not bicubic",
                        upscale_model_name)
        tracks_done = 0
        for track_id in sorted(groups):
            items = groups[track_id]
            if len(items) < 2:
                continue                      # one frame gains nothing from a temporal pass
            x1, y1, x2, y2 = [int(round(float(v))) for v in items[0][2][5]]
            crop_w, crop_h = x2 - x1, y2 - y1
            if min(crop_w, crop_h) < 32:
                continue
            crops, plan_entries, keys = [], [], []
            for frame_index, face_index, entry in items:
                cx1, cy1, cx2, cy2 = [int(round(float(v))) for v in entry[5]]
                if (cx2 - cx1) != crop_w or (cy2 - cy1) != crop_h:
                    continue                  # edge clamping moved this frame's crop; leave it to the chunk
                # Crop the SOURCE frame with the source plan's rect and upscale just that region: the
                # face then arrives at the sampler at output resolution without materialising a full frame.
                src_entries = src_plan[frame_index]
                src_entry = src_entries[face_index] if face_index < len(src_entries) else entry
                sx1, sy1, sx2, sy2 = [int(round(float(v))) for v in (src_entry[5] or entry[5])]
                if sx2 - sx1 < 8 or sy2 - sy1 < 8:
                    continue
                crops.append(image[frame_index:frame_index + 1, sy1:sy2, sx1:sx2, :3])
                fx, fy, fw, fh = [float(v) for v in entry[:4]]
                # Crop-space entry: the sampler crops the whole "canvas" frame, i.e. the face crop itself.
                plan_entries.append((fx - x1, fy - y1, fw, fh, entry[4], (0, 0, crop_w, crop_h),
                                     entry[6] if len(entry) > 6 else 1.0, int(track_id)))
                keys.append((frame_index, face_index))
            if len(crops) < 2:
                continue
            if upscale_model_name != "None":
                # Same upscaler the pixel pass uses, in batches: a fixed-scale model called once per
                # crop spends most of its time on call overhead rather than on the face.
                crops = apply_pixel_upscale_batch(crops, upscale_model_name)
            crop_canvas = torch.cat([_resize_bhwc(region, crop_h, crop_w) for region in crops],
                                    dim=0).to(image.device)
            crop_plan = [[entry] for entry in plan_entries]
            refine_stride = auto_detect_stride(len(crop_plan))
            crop_plan, _scanned = refine_plan_masks(
                crop_canvas, crop_plan, refine_stride, detector, model_path, confidence, sam3_prompt,
            )
            try:
                decoded, masks = self._face_sequence_pass(
                    vae, model, kwargs.get("positive"), [], crop_canvas, crop_plan,
                    int(kwargs.get("steps") or 20), 1.0,
                    kwargs.get("sampler_name") or "euler",
                    kwargs.get("scheduler") or "normal",
                    face_denoise, seed, face_padding, face_size,
                    master_face=master, master_alpha=alpha, master_box=mbox, master_aspect=maspect,
                    reference_strength=float(kwargs.get("reference_strength") or 0.75),
                    reference_pose_match=bool(kwargs.get("reference_pose_match", True)),
                    reference_flow_align=bool(kwargs.get("reference_flow_align", True)),
                )
            except Exception as e:
                logger.warning("[H3 Upscale] clip-wide face pass unavailable (%s); each chunk diffuses "
                               "its own faces", e)
                return None
            try:
                for index, key in enumerate(keys):
                    if (index, 0) in decoded:
                        store.put(key, decoded[(index, 0)], masks.get((index, 0)))
                store.flush_track()
            except OSError as e:
                # A full or read-only temp disk must not fail the render: fall back to what the chunks
                # did before the clip pass existed.
                logger.warning("[H3 Upscale] face crop cache could not be written to disk (%s); each "
                               "chunk diffuses its own faces", e)
                store.close()
                return None
            tracks_done += 1
            del crops, crop_canvas, plan_entries, decoded, masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if store.bytes <= 0:
            store.close()
            return None
        logger.info("[H3 Upscale] clip-wide face pass: %d track(s), %d face frame(s) sampled once for "
                    "the whole clip (%s); the chunks only composite them",
                    tracks_done, len(store), store.summary())
        return store

    def _lock_chunk_tone(self, images_part, previous, local_from, local_to, overlap, chunk_no):
        """Match a chunk's opening frames to the pixels the previous chunk already emitted.

        The chunk re-renders the same global frames as its leading `overlap` frames (they are its
        context, and the previous chunk emitted them), so that pair is the one place the two renders
        can be compared. The difference is measured there at a coarse scale and its inverse is added
        to the frames this chunk emits, fading out over the block: the seam carries the tone the clip
        already has, and the chunk's own look takes over again well before its end, so a correction
        cannot accumulate along the clip. It moves no pixels - unlike a dissolve it cannot double a
        redraw - and unlike the source-side locks it still applies at denoise 1.0, where those are
        deliberately switched off. Nothing here can make two independent samples identical; it keeps
        the seam from reading as a pump in exposure or colour. It is also only applied when the
        difference in question is exposure at all: if a coarse correction cannot account for the
        seam, the two renders disagree about content and the frames are left as rendered.
        """
        available = min(int(previous.shape[0]), int(local_from))
        if available <= 0 or local_to <= local_from:
            return
        delta, stats = seam_tone_delta(
            previous[previous.shape[0] - available:].detach().float().cpu().numpy(),
            images_part[local_from - available:local_from].detach().float().cpu().numpy(),
        )
        if delta is None:
            return
        block = images_part[local_from:local_to]
        frames, h, w = int(block.shape[0]), int(block.shape[1]), int(block.shape[2])
        # Only act when the difference really is exposure. Two renders that placed or drew the content
        # differently leave a residual just as large after a coarse correction, and adding that
        # correction anyway moves the first frames of the block without closing the seam - measured on
        # real seams, a genuine exposure step collapses under it (0.060 -> 0.000) while content
        # disagreement does not (0.142 -> 0.092). The second kind is left untouched, and says so.
        tonal = stats["before"] >= 0.01 and stats["after"] <= stats["before"] * 0.6
        shift = stats.get("shift")
        match = float(stats.get("match") or 0.0)
        where = ("unmeasurable" if shift is None else "%+.2f,%+.2f px" % (shift[0], shift[1]))
        logger.info(
            "[H3 Upscale] pixel chunk %d: seam matched to the previous render - tone drift %.3f -> "
            "%.3f (rgb %+.3f/%+.3f/%+.3f), detail %.3f, position %s (match %.2f)%s",
            chunk_no, stats["before"], stats["after"], stats["rgb"][0], stats["rgb"][1],
            stats["rgb"][2], float(stats.get("detail") or 0.0), where, match,
            "" if tonal else ", left alone (not an exposure difference)",
        )
        # A couple of pixels of shift is what an independent redraw costs: the decoder is sampled
        # again at denoise 0.25, so the best integer-pixel alignment of two renders never lands on
        # exactly 0,0. Measured on real seams, ordinary redraw jitter is well under 2 px while a
        # chunk that lost frames to the H3 grid showed tens of pixels. Only the second kind is
        # worth a warning, and the bar scales with the frame because the reading is in output pixels.
        limit = max(2.0, 0.005 * float(min(h, w)))
        if shift is not None and max(abs(shift[0]), abs(shift[1])) >= limit:
            logger.warning(
                "[H3 Upscale] pixel chunk %d: the redraw sits %+.2f,%+.2f px off the previous chunk "
                "(match %.2f, %.3f residual left after the tone lock) - more than a redraw's own "
                "resampling can account for at this size, so the two renders disagree about where "
                "the content is; check the upscale target and align_to_source",
                chunk_no, shift[0], shift[1], match, stats["after"],
            )
        if not tonal:
            return
        delta_t = torch.from_numpy(delta).to(device=block.device, dtype=block.dtype)
        if delta_t.shape[0] != h or delta_t.shape[1] != w:
            delta_t = torch.nn.functional.interpolate(
                delta_t.permute(2, 0, 1).unsqueeze(0), size=(h, w),
                mode="bilinear", align_corners=False,
            )[0].permute(1, 2, 0)
        # Full strength at the seam, eased out across the block: the next seam measures whatever
        # drift is left rather than inheriting a bias.
        fade = max(1, min(frames, max(int(2 * overlap), frames // 2)))
        step = (torch.arange(frames, device=block.device, dtype=block.dtype).clamp(max=fade - 1)
                / max(1.0, float(fade - 1)))
        weights = (1.0 - (step * step * (3.0 - 2.0 * step))).view(frames, 1, 1, 1)
        images_part[local_from:local_to, ..., :3] = torch.clamp(
            block[..., :3] + delta_t * weights, 0.0, 1.0,
        )

    def _align_chunk_to_previous(self, images_part, previous, local_from, local_to, chunk_no):
        """Put a chunk's frames back on the framing the clip already has.

        Every chunk is an independent sample of the same shot, and H3 re-decides where the frame
        sits, not only its detail - by a fraction of a percent of scale, which a translation fit
        cannot see and the eye sees immediately at the cut. This fits the similarity against the
        previous render's frames of the same global indices and, when they disagree by more than a
        redraw's own resampling costs, warps the whole block onto them. One constant transform for
        the block, so nothing moves inside the chunk: it can add no motion of its own, only remove
        the step at the cut. It is written back only when it measurably closes the gap there;
        otherwise the frames are left exactly as rendered and the line says so.
        """
        if not PIXEL_BATCH_SEAM_ALIGN or previous is None:
            return
        available = min(int(previous.shape[0]), int(local_to - local_from))
        if available <= 0:
            return
        # The trailing look-ahead is rendered here too, and the next seam is measured against
        # it, so it has to carry the framing this chunk ends up with. Correct only the frames
        # that are emitted and the next comparison meets this render again at its old framing,
        # reads it as aligned, and the step simply moves on to the next seam.
        stop = int(min(int(images_part.shape[0]), int(local_to) + int(previous.shape[0])))
        block = images_part[local_from:stop]
        frames, h, w = int(block.shape[0]), int(block.shape[1]), int(block.shape[2])
        block_np = block.detach().float().cpu().numpy()
        fit = seam_similarity(previous.detach().float().cpu().numpy()[:available],
                              block_np[:available])
        if fit is None:
            return
        scale, shift, moved = float(fit["scale"]), fit["shift"], float(fit["moved"])
        # A quarter of a percent of scale is already 2 px at the corner of a 1792 wide frame, and
        # that is where the eye catches it. Below this a correction would cost a resample of every
        # emitted frame to move content by less than the redraw's own jitter.
        worth = moved >= max(1.0, 0.0015 * float(max(h, w))) or max(abs(shift[0]),
                                                                abs(shift[1])) >= 1.5
        readings = ("%.2f%% scale, %+.2f deg, %+.2f,%+.2f px at the centre (fit %.2f, worst corner "
                    "%.1f px)" % ((scale - 1.0) * 100.0, float(fit["rotation"]),
                                  shift[0], shift[1], float(fit["match"]), moved))
        # The fit also has to be a framing nudge rather than two renders disagreeing about the
        # shot: past a degree, a percent and a half of scale, or a correlation that low,
        # "putting the frames back" would be moving them where the clip never was.
        trusted = (abs(scale - 1.0) <= 0.015 and abs(float(fit["rotation"])) <= 1.0
                   and float(fit["match"]) >= 0.90)
        if not worth or not trusted:
            # Always say which: a seam that does nothing and says nothing is indistinguishable
            # from one that was never measured, which is the loop this whole block exists to
            # break out of.
            logger.info(
                "[H3 Upscale] pixel chunk %d: the redraw came back at %s - left as rendered "
                "(%s)",
                chunk_no, readings,
                ("the renders disagree about the shot, not its framing" if not trusted
                 else "inside what two renders of the same frames cost each other"),
            )
            return
        warped = torch.empty_like(block)
        warp_m = fit["warp"]
        for lo in range(0, frames, 8):
            hi = min(frames, lo + 8)
            out_np = np.stack([
                # ECC returns the warp in the other convention: it has to be applied with
                # WARP_INVERSE_MAP to land the frames on the previous render. Applying it
                # the plain way doubles the difference instead of removing it (measured:
                # 0.031 -> 0.051 wrong way, 0.031 -> 0.0003 right way).
                cv2.warpAffine(frame, warp_m, (w, h),
                               flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)
                for frame in block_np[lo:hi]
            ])
            warped[lo:hi] = torch.from_numpy(np.clip(out_np, 0.0, 1.0)).to(
                device=block.device, dtype=block.dtype)
        prev_np = previous.detach().float().cpu().numpy()[:available]
        _delta, before = seam_tone_delta(prev_np, block_np[:available])
        _delta, after = seam_tone_delta(prev_np, warped[:available].detach().float().cpu().numpy())
        # The cut's residual is mostly the redraw's own texture, and no geometric correction moves
        # that: demanding that the warp visibly close it threw away every real re-frame this node
        # has measured - 0.05 to 0.29% of scale, up to 0.37 degrees, 9 px at the corner, five
        # seams in a row, all logged as "left as rendered". So the framing the fit read is what
        # decides, and this only checks that the warp did not cost picture on the way.
        if before is None or after is None:
            return
        if float(after["before"]) > float(before["before"]) * 1.01:
            logger.info(
                "[H3 Upscale] pixel chunk %d: the redraw came back at %s - left as rendered, "
                "warping it there costs detail (%.3f -> %.3f)",
                chunk_no, readings, float(before["before"]), float(after["before"]),
            )
            return
        images_part[local_from:stop] = warped
        logger.info(
            "[H3 Upscale] pixel chunk %d: the redraw came back at %s - warped onto the previous "
            "render, cut %.3f -> %.3f", chunk_no, readings,
            float(before["before"]), float(after["before"]),
        )
        del block_np

    def _upscale_one(self, *args, **kwargs):
        """Render one image batch; a batch is split only where the source has a hard cut, or where one
        render cannot hold a shot."""
        image = kwargs.get("image")
        latent = kwargs.get("latent")
        mode = str(kwargs.get("h3_upscale_mode") or "off").strip().lower()
        # off / latent: the pipeline renders the whole clip itself, and it is the pipeline that
        # knows whether it has a latent to work from instead of frames. This comes before anything
        # reads image.shape, so a latent-only graph reaches that path instead of failing here.
        if mode not in ("pixel", "pixel segments"):
            result = self._upscale_pipeline(*args, **kwargs)
            return (result[0], result[1], result[2], "")
        # pixel segments = the pixel render, written out one file per render instead of assembled.
        segments = mode == "pixel segments"
        latent_samples = self._video_latent_samples(latent) if (segments and image is None) else None
        if image is None and latent_samples is None:
            raise ValueError(
                "h3_upscale_mode = %s needs the image input, or a latent in pixel segments: a "
                "latent is only decoded window by window there (otherwise use "
                "h3_upscale_mode = latent)" % mode)
        segments_dir = self._segments_folder() if segments else ""
        segments_made = 0
        debug_skipped = False
        if latent_samples is not None:
            frame_count = video_frames_for_latents(int(latent_samples.shape[2]))
            source_h = int(latent_samples.shape[-2]) * 16
            source_w = int(latent_samples.shape[-1]) * 16
        else:
            frame_count = int(image.shape[0])
            source_h, source_w = int(image.shape[1]), int(image.shape[2])
        target_w, target_h = scale_to_megapixel_target(
            source_w, source_h, float(kwargs.get("scale_to_megapixels") or 1.0),
        )
        # Not a batch size: a shot is one render, and the card is what can refuse it (see the retry
        # in the render loop below). The free-VRAM figure is still measured and logged, because it
        # is the number to free memory up to - it says how much one render of this output size is
        # expected to hold before the card objects.
        room = auto_pixel_batch_frames(frame_count, target_h, target_w)
        # Stay on H3's frame grid (see H3_FRAME_GRID_BASE): an off-grid render is silently cut back
        # by the model, and a cut chunk cannot be laid back onto the frames it asked for without
        # moving content against its neighbours.
        chunk = h3_aligned_frames(PIXEL_BATCH_AUTO_MAX)
        if latent_samples is not None:
            # A window is decoded as well as rendered, so it has to be bounded before the first
            # decode: the card's own estimate is the only figure here that says what will fit.
            # The OOM retry in the loop still splits a window that turns out to be too big.
            per_window = int(room) if int(room or 0) > 0 else 96
            chunk = h3_aligned_frames(max(H3_FRAME_GRID_BASE,
                                           min(per_window, PIXEL_BATCH_AUTO_MAX)))
        overlap = kwargs.get("pixel_batch_overlap")
        overlap = PIXEL_BATCH_OVERLAP if overlap is None else int(overlap)
        # The seam is cut `overlap` frames in from both edges of the pair of chunks that meet
        # there, so every seam re-renders 2 x overlap frames. Cap the context at an eighth of the
        # chunk: a fixed default would otherwise swamp a small auto chunk (chunk 50 with overlap 22
        # advanced 6 frames per render, eight times the work), so every render now advances at least
        # three quarters of its chunk.
        overlap = max(0, min(overlap, chunk // 8))
        # Hard cuts in the source, read once for the whole clip (see detect_scene_cuts), and every
        # render's window and the frames it owns (see pixel_batch_plan). A seam exists only where
        # the source has a hard cut; one is placed anywhere else only where the clip cannot be one
        # render (a length that is not on the H3 grid) or where the card refuses a render, which the
        # retry in the loop below splits.
        if latent_samples is not None:
            logger.info(
                "[H3 Upscale] pixel segments: %d frame(s) read from the latent one render at a "
                "time - one latent is one shot, so there is no cut to scan for and no clip-wide "
                "face plan; seams are only where a render window ends", frame_count)
        scene_cuts = ([] if latent_samples is not None or not PIXEL_BATCH_SCENE_CUTS
                      else detect_scene_cuts(image))
        plan, seam_frames = pixel_batch_plan(frame_count, chunk, overlap, scene_cuts)
        if (not segments and len(plan) == 1 and plan[0][2] == 0
                and plan[0][3] == frame_count):
            # No cut and the whole clip is one render: nothing to seam, nothing to plan for.
            result = self._upscale_pipeline(*args, **kwargs)
            return (result[0], result[1], result[2], "")

        # One plan for the whole clip (see _shared_face_plan): every chunk then works the same tracks
        # with the same crop size instead of re-detecting and re-sizing per chunk.
        shared_face = (None if latent_samples is not None
                       else self._shared_face_plan(kwargs, image))
        # One diffusion pass per track for the whole clip, before the loop: a chunk boundary then never
        # splits a track's sampler run. Figures the pass cannot build (no vae/model, same_face) fall back
        # to the per-chunk pass, unchanged.
        clip_face = self._clip_face_cache(kwargs, image, shared_face) if shared_face is not None else None
        base_seed = int(kwargs.get("seed", 0) or 0)
        # One buffer for the whole clip, filled as each render finishes. A list of chunks plus a
        # final torch.cat holds the finished clip twice over at the moment of the copy, and on a
        # long clip that second copy is the whole budget: 5000 frames at 2.7 MP is 160 GiB, which
        # no machine holds twice.
        images_out = None
        images_map = None   # set when the batch lives in a file instead of in RAM
        debug_out = None
        debug_map = None
        written = 0
        audio, index = None, 0
        tail = None   # last frames already emitted, the pixels the next chunk re-renders
        look_ahead = None   # frames this chunk rendered past its block and did not emit

        if PIXEL_BATCH_SCENE_CUTS:
            if not scene_cuts:
                logger.info(
                    "[H3 Upscale] pixel batch: no hard cut in these %d frames - this is one "
                    "continuous take, so no seam is placed on it; every seam is one that a shot "
                    "too long for a single render forces",
                    frame_count,
                )
            else:
                logger.info(
                    "[H3 Upscale] pixel batch: %d hard cut(s) in the source at frame %s - a seam "
                    "goes on each one that can carry it, and nowhere else",
                    len(scene_cuts), ", ".join(str(int(cut) + 1) for cut in scene_cuts),
                )
                loose = [int(cut) + 1 for cut in scene_cuts if int(cut) not in seam_frames]
                if loose:
                    logger.info(
                        "[H3 Upscale] pixel batch: frame %s is a hard cut with no seam on it - a "
                        "shot that short is not a render of its own, so it is rendered with its "
                        "neighbour and the cut is drawn inside that render",
                        ", ".join(str(int(cut)) for cut in loose),
                    )
        logger.info(
            "[H3 Upscale] pixel batch: %d frame(s) in %d render(s) - one render per shot, and a "
            "shot is split only where the card refuses one (the free VRAM is estimated to hold "
            "about %d frame(s)), with %d frame(s) of context on each side of a seam inside a shot; "
            "%d seam(s) sit on a hard cut and share no frame across it",
            frame_count, len(plan), room,
            overlap, len(seam_frames),
        )
        plan = list(plan)
        pos = 0
        # A while loop rather than a for: a render the card refuses is split in two and re-queued
        # here (see the except below), so the plan can grow while it is being walked.
        while pos < len(plan):
            (start, end, emit_from, keep_to, began_on_cut, overlap_now) = plan[pos]

            # The window's own frames: sliced from the batch, or decoded from the latent's steps.
            if latent_samples is not None:
                window = self._decode_latent_window(
                    kwargs.get("vae"), latent_samples, start, end)
            else:
                window = image[start:end]
            part = dict(kwargs)
            part["image"] = window
            if segments:
                # One pixel render per segment; the writing stays out here.
                part["h3_upscale_mode"] = "pixel"
            # Two chunks must not share one noise field: the same noise over different content prints
            # the same grain twice.
            part["seed"] = base_seed + start
            detail_gain = 1.0
            if (tail is not None and not began_on_cut
                    and (PIXEL_BATCH_PIN_CONTEXT or PIXEL_BATCH_SEAM_GUIDE)):
                # Continue from what is already on screen: the pipeline puts these frames in at
                # output resolution where the chunk's own leading frames would be, so the seam
                # is a continuation rather than a second opinion about the same pixels. As a
                # guide they are context the sampler redraws; the pin additionally holds their
                # latents (see PIXEL_BATCH_PIN_CONTEXT for why that is off by default).
                part["seam_prefix"] = tail
                if PIXEL_BATCH_PIN_CONTEXT:
                    part["pin_frames"] = max(0, emit_from - start)
                if PIXEL_BATCH_SEAM_GUIDE:
                    # Only the frames the tail actually holds: a backed-up last chunk skips
                    # frames further back than that, and they have no guide to copy.
                    part["seam_guide_frames"] = min(int(tail.shape[0]), max(0, emit_from - start))
            if shared_face is not None:
                part["shared_face_plan"] = dict(shared_face, offset=start, frames=end - start)
            if clip_face is not None:
                # Re-keyed to this chunk's local frames, and spilt tracks are read back here, so the
                # pipeline's loop finds them by (img_idx, face_idx) as if it had run its own pass.
                part["precomputed_face_cache"] = clip_face.slice(start, end)
            index += 1
            logger.info("[H3 Upscale] pixel chunk %d: frames %d-%d of %d",
                        index, start + 1, end, frame_count)
            if began_on_cut:
                logger.info(
                    "[H3 Upscale] pixel chunk %d: starts on a hard cut - nothing from the shot "
                    "before it is fed in as context, the seam carries no overlap, no positional "
                    "alignment and no tone lock, so it is a cut between two shots rather than two "
                    "opinions about one picture",
                    index,
                )
            if keep_to < end:
                logger.info(
                    "[H3 Upscale] pixel chunk %d: emits frames %d-%d and draws %d frame(s) past "
                    "them without emitting them (%s)",
                    index, emit_from + 1, keep_to, end - keep_to,
                    "the shot ends there, and a render is a grid length, so it runs on into the "
                    "next shot and drops it"
                    if (keep_to == frame_count or keep_to in seam_frames) else
                    "the look-ahead the next seam is matched on",
                )
            try:
                images_part, debug_part, audio_part = self._upscale_pipeline(*args, **part)
            except Exception as exc:
                halves = (pixel_batch_split_window(plan[pos], frame_count, overlap)
                          if _cuda_out_of_memory(exc) else None)
                if halves is None:
                    raise
                if len(halves) == 1:
                    logger.warning(
                        "[H3 Upscale] pixel chunk %d: the card could not hold frames %d-%d (%d "
                        "frame(s)) - %s. Re-rendering frames %d-%d as a %d frame(s) window with "
                        "no context outside them - no seam, it is the same frames",
                        index, start + 1, end, end - start,
                        str(exc).strip().splitlines()[0][:200], halves[0][0] + 1, halves[0][1],
                        halves[0][1] - halves[0][0],
                    )
                else:
                    logger.warning(
                        "[H3 Upscale] pixel chunk %d: the card could not hold frames %d-%d (%d "
                        "frame(s)) - %s. Splitting this render at frame %d and retrying, as two "
                        "renders that share %d frame(s) of context",
                        index, start + 1, end, end - start,
                        str(exc).strip().splitlines()[0][:200], halves[1][0] + 1, overlap,
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                plan[pos:pos + 1] = halves
                index -= 1
                continue

            # Emit only the frames this chunk owns. Its first `overlap` frames are the context the
            # previous seam was cut on and were emitted there already; its last `overlap` frames are
            # context for the next seam, which re-renders them with frames of its own following. So
            # every frame comes out of exactly one render, and a seam is a cut between two renders
            # that both saw real neighbours - not a dissolve of two different redraws, which is what
            # doubled the content and left the seam looking torn.
            local_from = max(0, emit_from - start)
            local_to = max(local_from, min(int(images_part.shape[0]), keep_to - start))
            # What the cut actually costs. The seam line above compares the context frames both
            # chunks render, which is the only pair that shares global frames - but those frames
            # are thrown away by both, so they cannot show a jump. The pair that meets in the
            # output is the previous chunk's trailing look-ahead (frames it rendered and did not
            # emit) against the frames this chunk emits for the same indices: one is what the
            # clip would have shown had the chunk been longer, the other is what it does show.
            cut_gap = None
            cut_limit = max(2.0, 0.005 * float(min(int(images_part.shape[1]),
                                                   int(images_part.shape[2]))))
            if look_ahead is not None and local_to > local_from:
                cut_take = min(int(look_ahead.shape[0]), int(local_to - local_from))
                if cut_take > 0:
                    _cut_delta, cut = seam_tone_delta(
                        look_ahead[:cut_take].detach().float().cpu().numpy(),
                        images_part[local_from:local_from + cut_take]
                        .detach().float().cpu().numpy(),
                    )
                    if cut is not None:
                        cut_shift = cut.get("shift")
                        cut_where = ("unmeasurable" if cut_shift is None else
                                     "%+.2f,%+.2f px" % (cut_shift[0], cut_shift[1]))
                        logger.info(
                            "[H3 Upscale] pixel chunk %d: cut before frame %d - the previous "
                            "render's look-ahead and the frames this one emits there are %s "
                            "apart (match %.2f), detail %.3f, tone %.3f",
                            index, int(emit_from) + 1, cut_where,
                            float(cut.get("match") or 0.0), float(cut.get("detail") or 0.0),
                            float(cut.get("before") or 0.0),
                        )
                        # How far apart the two renders sit at the same frame, kept because it
                        # decides whose step the one playback takes actually is: a pair that agrees
                        # here cannot have a step at the cut that the previous render would not
                        # have taken just as well without any boundary in the way.
                        if cut_shift is not None:
                            cut_gap = float(max(abs(cut_shift[0]), abs(cut_shift[1])))
                        if cut_gap is not None and cut_gap >= cut_limit:
                            logger.warning(
                                "[H3 Upscale] pixel chunk %d: the cut before frame %d moves the "
                                "picture %s (match %.2f) - a shift this size is visible at "
                                "the seam, and it is between frames of two different renders",
                                index, int(emit_from) + 1, cut_where,
                                float(cut.get("match") or 0.0),
                            )
            # What the cut does to the movement. Every figure above compares two renders of one
            # timestamp, and that pair cannot see a step in the motion itself: if both renders
            # agree at frame N, the jump from the clip's frame N-1 into N is no larger than the
            # step the previous render was already taking there. So this measures the pair
            # playback actually cuts on - the last frame already on screen against the first
            # frame this chunk emits - beside the same step taken twice inside the previous
            # render. A chunk that restarts the motion, or slips it, shows up here as a step
            # well past its neighbours or as a step in the wrong direction, while every
            # same-timestamp fit still reads as the two renders agreeing.
            if tail is not None and int(tail.shape[0]) >= 3 and local_to > local_from:
                tail_np = tail.detach().float().cpu().numpy()
                here = images_part[local_from:local_from + 1].detach().float().cpu().numpy()
                # Across a cut the frame before is another shot, so every motion figure here would
                # be measuring the edit, not the seam. The same-timestamp pair above still measures
                # the renders against each other, which is the only thing a cut seam can be asked.
                step = None if began_on_cut else seam_similarity(tail_np[-1:], here)
                if step is None and not began_on_cut:
                    # A free affine over a pair a fifth of the frame apart does not always
                    # converge, and a missing line reads as a clean seam.
                    logger.info(
                        "[H3 Upscale] pixel chunk %d: the step into frame %d could not be "
                        "fitted - two frames with too little in common to line up, which a big "
                        "movement and a different picture both look like from here; the frames "
                        "themselves are dumped",
                        index, int(emit_from) + 1,
                    )
                if step is not None and step.get("shift") is not None:
                    cut_x, cut_y = float(step["shift"][0]), float(step["shift"][1])
                    cut_px = float(np.hypot(cut_x, cut_y))
                    inside = []
                    for at in range(int(tail_np.shape[0]) - 3, int(tail_np.shape[0]) - 1):
                        part = seam_similarity(tail_np[at:at + 1], tail_np[at + 1:at + 2])
                        if part is not None and part.get("shift") is not None:
                            inside.append((float(part["shift"][0]), float(part["shift"][1])))
                    if inside:
                        travels = sorted(float(np.hypot(x, y)) for x, y in inside)
                        # Where the picture was going before the cut, as one vector: a step that
                        # keeps its size but turns is a jump as well, and a per-pair magnitude
                        # would call it ordinary motion.
                        was_x = float(np.median([x for x, _y in inside]))
                        was_y = float(np.median([y for _x, y in inside]))
                        was_px = float(np.hypot(was_x, was_y))
                        typical = float(np.median(travels))
                        turn = None
                        if cut_px >= 0.2 and was_px >= 0.2:
                            cosine = (cut_x * was_x + cut_y * was_y) / (cut_px * was_px)
                            turn = float(np.degrees(np.arccos(float(np.clip(cosine, -1.0, 1.0)))))
                        logger.info(
                            "[H3 Upscale] pixel chunk %d: the step into frame %d is %.2f px "
                            "(%+.2f,%+.2f, scale %+.2f%%, match %.2f) where this clip was "
                            "stepping %.2f to %.2f px a frame inside the previous render - %.2fx"
                            "%s",
                            index, int(emit_from) + 1, cut_px, cut_x, cut_y,
                            (float(step["scale"]) - 1.0) * 100.0, float(step["match"]),
                            travels[0], travels[-1], cut_px / max(1e-3, typical),
                            ("" if turn is None else
                             ", %+.0f deg from the way it was going" % turn),
                        )
                        # The pair above already says where the two renders put this frame. Under
                        # the limit, the step into it is the one the previous render was taking
                        # anyway, so it cannot be a jump the cut made however far it sits from the
                        # two pairs behind it - and a warning here would be the instrument
                        # contradicting itself.
                        inherited = cut_gap is not None and cut_gap < cut_limit
                        restarts = cut_px >= 1.5 and cut_px >= 1.8 * typical
                        # A cut that repeats the frame the clip already showed is the other half
                        # of the same failure: the motion stops dead for a frame and picks up
                        # again, which reads as a hitch at every seam and looks nothing like a
                        # stall in any single frame pair.
                        stalled = typical >= 1.0 and cut_px <= 0.4 * typical
                        turned = turn is not None and turn >= 60.0 and cut_px >= 1.0 and was_px >= 1.0
                        if (restarts or turned or stalled) and not inherited:
                            logger.warning(
                                "[H3 Upscale] pixel chunk %d: the motion does not carry through "
                                "the cut into frame %d - a %.2f px step where the frames before "
                                "it step %.2f to %.2f px, %s. Playback reads that as the whole "
                                "frame jumping, while comparing two renders at the same "
                                "timestamp reads as agreement - a seam can measure calm and "
                                "still show this",
                                index, int(emit_from) + 1, cut_px, travels[0], travels[-1],
                                ("the picture holds still for a frame" if stalled else
                                 "turning away from where the picture was going" if turned
                                 else "far larger than its own neighbours"),
                            )
                        elif inherited:
                            logger.info(
                                "[H3 Upscale] pixel chunk %d: that step is the previous render's "
                                "own, not one the cut made - the two renders put frame %d within "
                                "%.2f px of each other, so the clip steps this far there whether "
                                "the chunk boundary falls on it or not; only a pair that disagreed "
                                "at that frame could be moved by the seam",
                                index, int(emit_from) + 1, float(cut_gap),
                            )
                # Every figure above fits one frame onto another, and they can all agree while the
                # whole picture changes how fine it is - the one change that touches every pixel
                # without moving any of them, and the one that reads as the whole frame shifting
                # rather than something inside it. So count the detail each frame holds, frame by
                # frame across the cut, with the same reading inside the previous render as the
                # scale: a pan speeding up softens a shot gradually and cannot put one frame on the
                # wrong side of a boundary.
                def _fineness(frames):
                    held = []
                    for frame in frames:
                        fh, fw = int(frame.shape[0]), int(frame.shape[1])
                        if fh < 8 or fw < 8:
                            return None
                        scale = 480.0 / float(max(fh, fw))
                        if scale < 1.0:
                            frame = cv2.resize(
                                frame, (max(8, int(round(fw * scale))),
                                        max(8, int(round(fh * scale)))),
                                interpolation=cv2.INTER_AREA)
                        grey = np.asarray(frame, dtype=np.float32)[..., :3].mean(axis=-1)
                        held.append(float(np.abs(grey - cv2.GaussianBlur(
                            grey, (0, 0), 1.2)).mean()))
                    return held or None

                held_before = _fineness(tail_np[-3:])
                held_after = _fineness(
                    images_part[local_from:min(int(local_to), int(local_from) + 4)]
                    .detach().float().cpu().numpy())
                # The same frames drawn twice. The previous chunk rendered this chunk's head frames
                # as the look-ahead it threw away, and those pixels are still in hand, so they can
                # be compared with what this chunk drew for the same timestamps: any difference
                # between those two is the render, because content cannot change between two draws
                # of one frame - where the frames before the cut differ by content as well as by
                # look. That pair is what the correction is driven by.
                held_same = None
                if look_ahead is not None and held_after:
                    take = min(int(look_ahead.shape[0]), len(held_after))
                    if take >= 1:
                        held_same = _fineness(look_ahead[:take].detach().float().cpu().numpy())
                same_ratio, same_pct = None, None
                if held_same:
                    twice = float(np.mean(held_after[:len(held_same)]))
                    was = float(np.mean(held_same))
                    same_ratio = was / max(1e-5, twice)
                    same_pct = (twice / max(1e-5, was) - 1.0) * 100.0
                if held_before and held_after and len(held_before) >= 2:
                    head = float(np.mean(held_after[:2]))
                    level = float(np.mean(held_before))
                    inside_steps = [(held_before[at + 1] / max(1e-5, held_before[at]) - 1.0) * 100.0
                                    for at in range(len(held_before) - 1)]
                    if PIXEL_BATCH_SEAM_DETAIL_MATCH and tail is not None:
                        # What the picture held before the cut over what this chunk's head holds:
                        # one figure for the whole render, bounded, and only worth acting on when
                        # it is larger than the frame-to-frame wobble the line measures anyway. The
                        # same-timestamp pair is preferred, because it is the only one where
                        # content is not a suspect - and across a cut it is the only one that can be
                        # used at all: the frames before are another shot, so their sharpness is
                        # that shot's content, not this render's look.
                        if began_on_cut and same_ratio is None:
                            wanted = 1.0
                        else:
                            wanted = (same_ratio if same_ratio is not None
                                      else level / max(1e-5, head))
                        wanted = min(1.35, max(1.0 / 1.35, wanted))
                        if abs(wanted - 1.0) >= 0.03:
                            detail_gain = wanted
                    logger.info(
                        "[H3 Upscale] pixel chunk %d: fine detail held by each frame across the "
                        "cut into frame %d - %s | %s; the frames after the cut hold %+.0f%% of "
                        "what the ones they follow hold, where frame to frame inside the previous "
                        "render the same reading moves at most %.0f%%",
                        index, int(emit_from) + 1,
                        " ".join("%.4f" % v for v in held_before),
                        " ".join("%.4f" % v for v in held_after),
                        (head / max(1e-5, level) - 1.0) * 100.0,
                        max(abs(v) for v in inside_steps),
                    )
                    if same_pct is not None:
                        logger.info(
                            "[H3 Upscale] pixel chunk %d: the head frames drawn twice - the "
                            "previous chunk's look-ahead holds %s and this one's hold %s at the "
                            "same %d frame(s), so this render is %s than the one it continues. "
                            "Both sides are the same timestamps, so this difference is the render; "
                            "the frames before the cut differ by %+.0f%% (%s)%s",
                            index,
                            " ".join("%.4f" % v for v in held_same),
                            " ".join("%.4f" % v for v in held_after[:len(held_same)]),
                            len(held_same),
                            ("%+.0f%% finer" % same_pct) if same_pct >= 0 else
                            ("%+.0f%% softer" % same_pct),
                            (head / max(1e-5, level) - 1.0) * 100.0,
                            ("the shot before this one, not the look" if began_on_cut
                             else "content can move that too"),
                            ("" if detail_gain == 1.0 else
                             ", and the grade is matched from the pair above"),
                        )
                    elif began_on_cut:
                        logger.info(
                            "[H3 Upscale] pixel chunk %d: the cut falls exactly on this seam, so no "
                            "frame was drawn by both renders and the grade is not matched here - "
                            "there is nothing on the other side of it to match to",
                            index,
                        )
                    if detail_gain != 1.0:
                        logger.info(
                            "[H3 Upscale] pixel chunk %d: its fine detail is being scaled by %.2fx "
                            "across all %d emitted frame(s), one figure for the chunk, so the whole "
                            "render is delivered at the sharpness of the frames it follows - "
                            "nothing moves, nothing inside the chunk flickers, and nothing from the "
                            "previous render is mixed in",
                            index, detail_gain, int(local_to - local_from),
                        )
                if PIXEL_BATCH_SEAM_DUMP:
                    dumped = save_seam_frames(
                        tail_np[-1], here[0],
                        (None if look_ahead is None else
                         look_ahead[0:1].detach().float().cpu().numpy()[0]),
                        int(emit_from) + 1,
                        control=((tail_np[-2], tail_np[-1]) if int(tail_np.shape[0]) >= 2
                                 else None),
                    )
                    if dumped is not None:
                        held = dumped.get("cut")
                        ordinary = dumped.get("control")
                        if held is not None and ordinary is not None and not began_on_cut:
                            logger.info(
                                "[H3 Upscale] pixel chunk %d: with the fit between the two frames "
                                "at the cut undone they still differ by %.4f, against %.4f for an "
                                "ordinary pair inside the previous render - %.2fx what this clip "
                                "does between two frames there, so %s",
                                index, float(held), float(ordinary),
                                float(held) / max(1e-4, float(ordinary)),
                                ("past the 1.5x a re-drawn picture costs, so the cut is changing "
                                 "the picture rather than moving it" if
                                 float(held) >= 1.5 * float(ordinary) else
                                 "short of the 1.5x a re-drawn picture costs, so these two frames "
                                 "are the same picture with its fine texture sampled again and "
                                 "what is seen at the cut is not them disagreeing"),
                            )
                        if index == 2:
                            logger.info(
                                "[H3 Upscale] pixel chunk %d: the frames every cut is made "
                                "between are in %s - h3_seam_NNNN_prev.png (last frame on screen), "
                                "_next.png (first frame after the cut), _unused.png (what the "
                                "previous render had there and this one did not use), _diff.png "
                                "(their raw difference at 8x), _residual.png (the same pair with "
                                "the fit between them undone, so movement is taken out and what "
                                "is left is what moved differently) and _control.png (that same "
                                "reading over an ordinary pair inside the previous render, which "
                                "is what _residual.png has to be judged against)",
                                index, dumped.get("folder"),
                            )
            # Where this chunk sits relative to the framing the clip already has - scale included,
            # because a re-framed shot lands on the centre and reads as no shift at all. Corrected
            # before the frames are emitted, and before the look-ahead is taken from them, so the
            # next seam is measured against what is actually on screen.
            if look_ahead is not None and local_to > local_from and not began_on_cut:
                # Where the difference sits, and what an ordinary frame pair costs on this clip.
                # The control is two adjacent frames from inside the previous render: same shot,
                # same motion, no seam anywhere near it. Whatever the cut reads beyond that is the
                # cut's own difference, with this clip's noise floor printed beside it.
                split = seam_framing_split(
                    look_ahead.detach().float().cpu().numpy(),
                    images_part[local_from:local_to].detach().float().cpu().numpy(),
                )
                control = []
                look_np = look_ahead.detach().float().cpu().numpy()
                for pair in range(min(3, int(look_np.shape[0]) - 1)):
                    parts = seam_framing_split(look_np[pair:pair + 1], look_np[pair + 1:pair + 2])
                    if parts:
                        control.append(parts)
                if split is None:
                    logger.info(
                        "[H3 Upscale] pixel chunk %d: framing at the cut: nothing to fit - the "
                        "frames on either side are too flat to compare", index,
                    )
                else:
                    def _split_of(parts, key):
                        return float(parts[key]["scale_pct"]) if key in parts else float("nan")
                    whole = _split_of(split, "whole")
                    middle = _split_of(split, "centre")
                    edge = _split_of(split, "edge")
                    fit = max((float(split[k]["match"]) for k in split), default=0.0)
                    moved = max((float(split[k]["moved_px"]) for k in split), default=0.0)
                    logger.info(
                        "[H3 Upscale] pixel chunk %d: framing across the cut: whole %+.2f%%, "
                        "middle %+.2f%%, edge %+.2f%% (up to %.1f px at a corner, fit %.2f) - "
                        "middle against edge %+.2f%%",
                        index, whole, middle, edge, moved, fit, middle - edge,
                    )
                    if control:
                        # A single ordinary pair is not a noise floor: on a shot with real camera
                        # motion one pair can read several percent on its own (a fit that slides off
                        # the structure it locked onto), which would make the cut look calm against
                        # it for no reason. Three pairs, and the spread printed, so the comparison
                        # is against what this clip actually does frame to frame.
                        contrasts = sorted(_split_of(p, "centre") - _split_of(p, "edge")
                                           for p in control)
                        wholes = sorted(_split_of(p, "whole") for p in control)
                        fits = sorted(max(float(v["match"]) for v in p.values()) for p in control)
                        c_low, c_high = contrasts[0], contrasts[-1]
                        c_middle = float(np.median(contrasts))
                        span = max(abs(c_low), abs(c_high))
                        contrast = middle - edge
                        if abs(contrast) <= span:
                            verdict = ("no larger than the biggest an ordinary pair reads here, so "
                                       "this cut is not re-framing the middle against its edge")
                        else:
                            verdict = ("larger than anything an ordinary pair reads here, so this "
                                       "cut is re-framing the middle against its edge")
                        logger.info(
                            "[H3 Upscale] pixel chunk %d: the same reading for %d pair(s) of frames "
                            "inside the previous render: whole %+.2f%%, middle against edge %+.2f%% "
                            "(spread %+.2f%% to %+.2f%%, fit %.2f to %.2f) - this cut reads "
                            "%+.2f%%, which is %s",
                            index, len(control), float(np.median(wholes)), c_middle, c_low, c_high,
                            fits[0], fits[-1], contrast, verdict,
                        )
                self._align_chunk_to_previous(images_part, look_ahead, local_from, local_to, index)
            # Where the picture moves inside this one render, against the source's own motion at
            # those frames. A seam is a jump between two renders and is measured above; a jump that
            # recurs inside a single render cannot be fixed by aligning chunks to anything, so it is
            # worth a number instead of another guess. H3's sequences come in 17 frame steps (see
            # H3_FRAME_GRID_STEP), so a jump belonging to the sampler would land on that period, and
            # the frame it lands on is printed.
            if local_to - local_from >= 3:
                profile = frame_change_profile(
                    images_part[local_from:local_to].detach().float().cpu().numpy(),
                    window[local_from:local_to].detach().float().cpu().numpy(),
                )
                if profile is not None:
                    change_ratio, out_change, src_change = profile
                    order = np.argsort(change_ratio)[::-1]
                    worst = int(order[0])
                    worst_frame = int(start) + int(local_from) + worst + 2
                    second = ("then at frame %d (%.2fx)" % (
                        int(start) + int(local_from) + int(order[1]) + 2,
                        float(change_ratio[order[1]])) if change_ratio.size > 1 else "no second pair")
                    logger.info(
                        "[H3 Upscale] pixel chunk %d: sharpest change inside the render is at "
                        "frame %d - the redraw moves %.2fx as much as the source does there "
                        "(%.3f vs %.3f), %s; typical pair %.2fx",
                        index, worst_frame, float(change_ratio[worst]),
                        float(out_change[worst]), float(src_change[worst]), second,
                        float(np.median(change_ratio)),
                    )
                    if float(change_ratio[worst]) >= 3.0 and float(out_change[worst]) >= 0.01:
                        logger.warning(
                            "[H3 Upscale] pixel chunk %d: the picture jumps at frame %d, inside a "
                            "single render (%.2fx the source's change there) - that is the "
                            "sampler's own temporal step, not a chunk seam",
                            index, worst_frame, float(change_ratio[worst]),
                        )
            if local_to > local_from:
                if tail is not None and not began_on_cut:
                    self._lock_chunk_tone(images_part, tail, local_from, local_to, overlap_now, index)
                if detail_gain != 1.0:
                    images_part = match_detail_level(images_part, detail_gain)
                piece = images_part[local_from:local_to]
                if segments:
                    # Its own file, closed before the next render starts: nothing is assembled, so
                    # the run costs one render of RAM plus whatever the segments take on disk.
                    if images_out is None:
                        images_out = torch.empty(
                            (0,) + tuple(int(v) for v in piece.shape[1:]), dtype=piece.dtype)
                    segments_made += 1
                    self._write_segment(segments_dir, segments_made, piece)
                else:
                    if images_out is None:
                        # The size is only known once a render has come back, and every render of
                        # the clip has the same one, so the first one sizes the batch.
                        images_out, images_map = self._clip_buffer(
                            (frame_count,) + tuple(int(v) for v in piece.shape[1:]), piece.dtype,
                            "pixel batch")
                    at = int(emit_from)
                    images_out[at:at + int(piece.shape[0])] = piece
                    if images_map is not None:
                        # This render's frames are on disk now, not waiting in RAM for the end.
                        images_map.flush()
                written += int(piece.shape[0])
                piece = None
            look_ahead = None
            cut_frames = min(int(overlap_now), int(images_part.shape[0]) - local_to)
            if cut_frames > 0 and end < frame_count:
                look_ahead = images_part[local_to:local_to + cut_frames].clone()
            tail_frames = min(int(overlap_now), local_to - local_from)
            tail = (images_part[local_to - tail_frames:local_to].clone()
                    if tail_frames > 0 else None)
            # Its frames are in the output buffer now, and the tail above is a copy of the few
            # that matter: drop this render before the next one allocates its own.
            images_part = None
            if segments and debug_part is not None:
                # The segments are the output; a debug clip would be another whole batch.
                debug_skipped = True
            elif debug_part is not None:
                debug_to = max(local_from, min(int(debug_part.shape[0]), keep_to - start))
                if debug_to > local_from:
                    piece = debug_part[local_from:debug_to]
                    if debug_out is None:
                        debug_out, debug_map = self._clip_buffer(
                            (frame_count,) + tuple(int(v) for v in piece.shape[1:]), piece.dtype,
                            "face debug")
                    at = int(emit_from)
                    debug_out[at:at + int(piece.shape[0])] = piece
                    if debug_map is not None:
                        debug_map.flush()
                    piece = None
            if audio is None:
                audio = audio_part   # one clip: decode the latent's audio once, not once per chunk
            pos += 1
            del part
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if images_out is None:
            # An empty plan, or every render refused: an empty batch rather than a failed one.
            shape = ((0,) + tuple(int(v) for v in image.shape[1:]) if image is not None
                     else (0, int(source_h), int(source_w), 3))
            images_out = torch.empty(shape, dtype=torch.float32, device="cpu")
            logger.warning("[H3 Upscale] pixel batch: none of the %d frame(s) came out of a "
                           "render - returning an empty batch", frame_count)
        elif written != frame_count:
            logger.warning("[H3 Upscale] pixel batch: the renders covered %d of %d frame(s) - "
                           "the rest of the batch is blank", written, frame_count)
        if segments:
            logger.info(
                "[H3 Upscale] pixel batch: %d frame(s) written as %d segment(s) in %s - join them "
                "in order with: ffmpeg -f concat -safe 0 -i concat.txt -c copy joined.mp4",
                written, segments_made, segments_dir)
            if debug_skipped:
                logger.info("[H3 Upscale] pixel segments: the face debug frames are not written - "
                            "the segment files are the output")
        else:
            logger.info(
                "[H3 Upscale] pixel batch: %d frame(s) filled into one buffer from %d chunk(s)",
                int(images_out.shape[0]), index)
        if images_map is not None:
            logger.info(
                "[H3 Upscale] pixel batch: the batch reads through to %s, flushed after every "
                "render - it stays on disk while the graph runs and can be deleted afterwards",
                images_map.filename)
        if clip_face is not None:
            clip_face.close()   # drop the temp dir (spilt mode) now, not at some later collection
        return (images_out, debug_out, audio, segments_dir)

    def _upscale_pipeline(
        self, model, positive, vae, scale_to_megapixels: float,
        tile_width: int, tile_height: int, tile_padding: int, seam_fix_width: int,
        seam_fix_denoise: float, seam_fix_padding: int, seed: int, steps: int,
        sampler_name: str, scheduler: str, denoise: float,
        image=None, latent=None, audio_vae=None,
        h3_upscale_mode: str = "pixel", upscale_model: str = "None",
        latent_upscale_model: str = "None", reference_face=None,
        enable_dlss5: bool = False,
        dlss5_scale_to_megapixels: float = 1.3,
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
        enable_face_detailer: bool = False, same_face: bool = False, face_denoise: float = 0.25,
        face_detect_stride: str = FACE_DETECT_STRIDE_LABELS[0],
        face_padding: float = 0.25, face_size: int = 1024,
        face_mask=None, face_detector: str = "auto", face_detector_model: str = "",
        face_confidence: float = 0.35, manual_face_boxes: str = "",
        reference_strength: float = 0.75, detail_retention: float = 0.6,
        enable_colorfx: bool = True, enable_color_correction: bool = True, enable_lut_processing: bool = True, enable_enhancements: bool = True, enable_blur_effects: bool = True, enable_stylistic_effects: bool = True, hdr_intensity: float = 2.5, shadow_intensity: float = 0.2, highlight_intensity: float = 0.2, gamma: float = 1.0, brightness: float = 0.9, contrast: float = 1.0, saturation: float = 0.9, enhance_color: float = 1.0, lut_name: str = "None", lut_strength: float = 0.5, lut_log_process: bool = True, sharpness: float = 3.0, edge_enhance_strength: float = 0.1, detail_enhance_strength: float = 0.2, blur_radius: int = 0, gaussian_blur_radius: float = 0.0, radial_blur_strength: float = 32.0, radial_blur_center_x: float = 0.5, radial_blur_center_y: float = 0.5, radial_blur_focus_spread: float = 3.0, radial_blur_steps: int = 10, chromatic_aberration_r_x: float = 1.0, chromatic_aberration_r_y: float = 0.0, chromatic_aberration_b_x: float = -1.0, chromatic_aberration_b_y: float = 0.0, chromatic_blur_amount: float = 0.5, simple_film_grain_intensity: float = 0.04, simple_film_grain_monochrome: bool = True, scanline_intensity: float = 0.0, vignette_intensity: float = 0.4, vignette_center_x: float = 0.5, vignette_center_y: float = 0.5, soft_light_opacity: float = 0.3, soft_light_blur_radius: float = 10.0,
        pixel_batch_overlap: int = PIXEL_BATCH_OVERLAP,
        pin_frames: int = 0, seam_guide_frames: int = 0, seam_prefix=None,
        shared_face_plan=None, precomputed_face_cache=None,
        latent_noise_offset: float = 0.0, tone_lock: float = 1.0, sam3_prompt: str = "head", face_structure_lock: float = 0.8, reference_pose_match: bool = True, reference_flow_align: bool = True, face_chroma_lock: float = 0.9, face_seamless_blend: bool = True, frontal_only_threshold: float = 0.0, locks_follow_denoise: bool = True, redraw_only: bool = False, align_to_source: bool = True, chroma_lock: float = 0.9, edge_protect: float = 0.85, diffusion_mix: float = 1.0,
        **kwargs
    ):
        # H3 tile diffusion always runs without guidance (cfg 1.0) and the node has no
        # negative conditioning input.
        negative = []
        cfg = 1.0
        mode = str(h3_upscale_mode or "off").strip().lower()
        if mode not in ("off", "pixel", "latent"):
            raise ValueError(
                "h3_upscale_mode must be 'off', 'pixel' or 'latent'; "
                f"received {h3_upscale_mode!r}"
            )
        if mode == "off" and image is None:
            raise ValueError("h3_upscale_mode = off needs the image input")
        if mode == "pixel" and image is None:
            raise ValueError("h3_upscale_mode = pixel needs the image input")
        if mode == "pixel" and denoise >= 0.01 and vae is None:
            raise ValueError(
                "h3_upscale_mode = pixel with denoise > 0 needs the vae input "
                "(set denoise to 0 for a pure model upscale)"
            )
        if image is not None:
            batch_size, h, w, c = image.shape
        else:
            batch_size, h, w, c = 1, 0, 0, 3
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
        target_w, target_h = scale_to_megapixel_target(w, h, scale_to_megapixels)
        tile_diffusion = mode != "off"

        tile_w = (tile_width // 16) * 16
        tile_h = (tile_height // 16) * 16
        padding = (tile_padding // 16) * 16

        if mode == "latent":
            # Learned 3D latent resize, then decode: the base image comes out of the latent.
            resized_images, target_h, target_w = self._latent_upscale_images(
                latent, image, vae, scale_to_megapixels, latent_upscale_model,
            )
            batch_size, c = resized_images.shape[0], resized_images.shape[-1]
        else:
            target_w, target_h = scale_to_megapixel_target(w, h, scale_to_megapixels)
            if mode == "off":
                target_h, target_w = h, w
                resized_images = image.clone()
                logger.info("[H3 Upscale] h3_upscale_mode = off: group bypassed (no resize, no tile diffusion)")
            elif upscale_model and upscale_model != "None":
                import comfy_extras.nodes_upscale_model
                upscaler = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel()
                resized_images = upscaler.upscale(load_pixel_upscale_model(upscale_model), image)[0]
                if resized_images.shape[1] != target_h or resized_images.shape[2] != target_w:
                    resized_images = _resize_bhwc(resized_images, target_h, target_w)
            else:
                resized_images = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                    antialias=False,
                ).permute(0, 2, 3, 1).clamp(0.0, 1.0)

        # Continue the clip where it stopped instead of from a fresh upscale of the same source:
        # the leading frames of this chunk were emitted by the previous one, so that chunk's own
        # output is the only context that matches what is on screen. Pinning stops the sampler
        # from turning them into a new render (see _diffuse_sequence); as a guide the sampler is
        # free to redraw them, and what it inherits is their position on screen.
        guide_frames = max(0, int(seam_guide_frames or 0))
        if seam_prefix is not None and batch_size > 1 and (pin_frames > 0 or guide_frames > 0):
            want = int(pin_frames) if pin_frames > 0 else guide_frames
            take = min(int(seam_prefix.shape[0]), want, batch_size)
            if take > 0 and tuple(seam_prefix.shape[1:3]) == tuple(resized_images.shape[1:3]):
                resized_images = resized_images.clone()
                resized_images[:take] = seam_prefix[seam_prefix.shape[0] - take:].to(
                    device=resized_images.device, dtype=resized_images.dtype)
                if pin_frames > 0:
                    pin_frames = take
                    logger.info(
                        "[H3 Upscale] seam context: continuing from %d already-emitted "
                        "frame(s) instead of re-upscaling the source", take,
                    )
                else:
                    logger.info(
                        "[H3 Upscale] seam guide: %d leading frame(s) are the previous "
                        "chunk's own output, redrawn as context", take,
                    )
            else:
                logger.warning(
                    "[H3 Upscale] seam context is %s but this chunk renders %s; re-sampling the "
                    "overlap instead of continuing from it",
                    tuple(seam_prefix.shape[1:3]), tuple(resized_images.shape[1:3]),
                )
                pin_frames, guide_frames = 0, 0
        else:
            pin_frames = 0

        if batch_size > 1:
            # A ComfyUI image batch represents an ordered frame sequence here. Process the same
            # spatial tile from all frames together so H3 can use its temporal video latent.
            final_images = self._upscale_video_batch(
                resized_images, vae, model, positive, negative,
                tile_width, tile_height, tile_padding, steps, cfg, sampler_name,
                scheduler, denoise, seed, latent_noise_offset, lock_scale,
                t_mix, t_edge, t_detail, align_to_source, tile_diffusion,
                pin_frames=pin_frames,
            )
        else:
            output_batch = []

            for b in range(batch_size):
                canvas = resized_images[b:b+1].clone()
                if denoise < 0.01 or not tile_diffusion:  # tile diffusion disabled: keep the clean upscale (face detailer etc. still run)
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
            elif same_face:
                logger.info("[H3 Upscale] no reference_face connected: the largest face becomes the master")
            else:
                logger.info("[H3 Upscale] no reference_face connected: every detected face is diffused on its own")

        if enable_face_detailer:
            # A video batch is detected on a stride, tracked, gap-filled and smoothed once up
            # front, and every track keeps ONE crop size for its whole length (see
            # plan_video_face_tracks). The frame loop below then renders those stable crops, so
            # the redrawn face no longer jumps in size or position from frame to frame. A single
            # image, a shot without faces, or a failed plan still goes through per-frame
            # detection, exactly as before.
            face_plan = None
            if shared_face_plan is not None and final_images.shape[0] > 1:
                face_plan = slice_shared_face_plan(
                    shared_face_plan, int(final_images.shape[1]), int(final_images.shape[2]),
                )
            elif final_images.shape[0] > 1:
                try:
                    face_plan = plan_video_face_tracks(
                        final_images, face_detector, face_detector_model, face_confidence, face_mask,
                        manual_face_boxes, sam3_prompt, face_detect_stride, face_padding,
                        frontal_only_threshold,
                    )
                except Exception as e:
                    logger.warning(f"[H3 Upscale] face tracking unavailable ({e}); detecting per frame")
                    face_plan = None
            output_faces = []
            debug_faces = []
            face_regions = []

            # The reference workflow redraws a whole tracked subject in one KSampler pass over its crop
            # sequence. Do the same: diffuse every track once, up front, and let the frame loop below
            # composite those results. The self-master (same_face) path has to stay sequential, and a
            # connected reference face mixes into each frame's init, so both keep the per-frame path.
            face_seq_cache = {}
            face_mask_cache = {}
            # Chunked pixel mode diffuses each track once for the whole clip up front (_clip_face_cache)
            # and passes the decoded crops in: this chunk composites them and must not restart the sampler.
            if precomputed_face_cache:
                face_seq_cache = dict(precomputed_face_cache.get("seq") or {})
                face_mask_cache = dict(precomputed_face_cache.get("mask") or {})
            if (face_plan is not None and final_images.shape[0] > 1 and not same_face
                    and not face_seq_cache):
                try:
                    face_seq_cache, face_mask_cache = self._face_sequence_pass(
                        vae, model, positive, negative, final_images, face_plan,
                        steps, cfg, sampler_name, scheduler, face_denoise, seed,
                        face_padding, face_size,
                        master_face=reference_master, master_alpha=reference_alpha,
                        master_box=reference_box, master_aspect=reference_aspect,
                        reference_strength=reference_strength,
                        reference_pose_match=reference_pose_match,
                        reference_flow_align=reference_flow_align,
                    )
                except Exception as e:
                    logger.warning(f"[H3 Upscale] temporal face pass unavailable ({e}); diffusing per frame")
                    face_seq_cache, face_mask_cache = {}, {}
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            for img_idx in range(final_images.shape[0]):
                canvas_img = final_images[img_idx:img_idx+1].clone()
                img_h, img_w = canvas_img.shape[1], canvas_img.shape[2]

                if face_plan is not None:
                    entries = face_plan[img_idx]
                    det_label = "track"
                else:
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
                    entries = [(f[0], f[1], f[2], f[3], _FACE_MASKS.get(tuple(f[:4])), None,
                                _FACE_SYM.get(tuple(f[:4]), 1.0), -1) for f in faces]
                faces = [(entry[0], entry[1], entry[2], entry[3]) for entry in entries]
                logger.info(f"[H3 Upscale] face detailer: {len(entries)} face(s) via {det_label}")

                if len(entries) == 0:
                    face_regions.append(None)
                    output_faces.append(canvas_img)
                    debug_faces.append(canvas_img)
                    continue

                # faces are sorted largest -> smallest. With a reference connected it is the master
                # for all of them; with same_face on, the largest face rendered first becomes the
                # master; with same_face off, every face is re-rendered on its own (no master).
                master_face, master_alpha, master_aspect, master_box = reference_master, reference_alpha, reference_aspect, reference_box
                # The face pass works from the CURRENT canvas only. (Pulling structure/detail from the pre-diffusion
                # image while the canvas had been re-drawn put two slightly different versions of the face on top of
                # each other, which is exactly the ghosting.)
                pristine = None
                used_boxes = []
                region = torch.zeros((1, img_h, img_w, 1), device=canvas_img.device)

                for face_idx, entry in enumerate(entries):
                    # A tracked entry carries smoothed boxes and a fixed crop rect, so its numbers can
                    # be fractional; every pixel position below has to be an integer.
                    fx, fy, fw, fh = [int(round(float(v))) for v in entry[:4]]
                    entry_mask, entry_crop = entry[4], entry[5]
                    # A tracked entry carries the track's fixed crop; a per-frame detection is
                    # expanded from its own box, exactly as before.
                    if entry_crop is not None and entry_crop[2] - entry_crop[0] >= 32 and entry_crop[3] - entry_crop[1] >= 32:
                        x1, y1, x2, y2 = [int(round(float(v))) for v in entry_crop]
                    else:
                        x1, y1, x2, y2 = expand_box(fx, fy, fw, fh, face_padding, img_w, img_h)
                    crop_h, crop_w = y2 - y1, x2 - x1
                    if min(crop_h, crop_w) < 32:
                        continue
                    used_boxes.append((x1, y1, x2, y2, fx, fy, fx + fw, fy + fh))
                    inst_mask = entry_mask
                    # The temporal pass already cleaned and temporally smoothed this track's masks; reuse
                    # them so the composite edge matches the mask the pass was diffused with.
                    cached_mask = face_mask_cache.get((img_idx, face_idx))
                    if cached_mask is not None:
                        crop_mask = cached_mask.to(device=canvas_img.device, dtype=torch.float32)
                    else:
                        if inst_mask is not None:
                            crop_mask = instance_blend_mask(inst_mask, fx, fy, x1, y1, crop_h, crop_w, canvas_img.device)
                        else:
                            crop_mask = create_face_blend_mask(crop_h, crop_w, feather=0.35, device=canvas_img.device)
                        crop_mask = crop_mask * border_taper(crop_h, crop_w, 0.08, canvas_img.device)

                    orig_region = canvas_img[:, y1:y2, x1:x2, :3]

                    work_h, work_w = face_work_size(crop_h, crop_w, face_size)
                    orig_work = _resize_bhwc(orig_region, work_h, work_w)

                    # If a master face exists, mix it in (cover-resized, tone-matched, inside the face mask).
                    # The same helper builds the temporal pass's init, so both paths start from the same pixels.
                    init, detail_src, use_master, compat = self._face_init(
                        orig_work, crop_mask, fx, fy, fw, fh, x1, y1, crop_w, crop_h, work_h, work_w,
                        master_face, master_alpha, master_box, master_aspect, reference_strength,
                        reference_pose_match, reference_flow_align,
                    )
                    logger.info(f"[H3 Upscale] face {face_idx}: box {fw}x{fh}, crop {crop_w}x{crop_h}, work {work_w}x{work_h}, "
                                f"mask={'sam' if inst_mask is not None else 'ellipse'}, master={'yes' if use_master else 'no'}, "
                                f"pose_match={compat:.2f}")

                    cached = face_seq_cache.get((img_idx, face_idx))
                    if cached is not None:
                        # Already redrawn as part of this track's temporal pass; reusing it is what makes
                        # the pass one model run per track instead of one per frame.
                        decoded = cached.to(device=canvas_img.device, dtype=torch.float32)
                    else:
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
                    if same_face and master_face is None:
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
            if not DLSS5_AVAILABLE:
                raise RuntimeError(
                    "enable_dlss5 needs the dlss5 package beside this one (dlss5/settings.py), "
                    "the DLSS runtime and RTX hardware. Turn enable_dlss5 off to upscale "
                    "without it."
                )
            from .dlss5 import DlssOptions, DlssSession, find_runtime
            from .dlss5.imaging import fit_frame, rgba_to_tensor, tensor_to_rgba
            from .dlss5.motion import TemporalGuide
            from .dlss_nodes.common import confirm_feature_18

            layout = find_runtime(runtime_dir.strip() or None)
            dlss_mode = dlss_mode_for_target(final_images.shape[2], final_images.shape[1],
                                             dlss5_scale_to_megapixels)
            options = DlssOptions.create(
                upscaling_mode=dlss_mode,
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
            # dlss5_scale_to_megapixels owns the pass output size and is independent of the H3
            # Upscale group's scale_to_megapixels, so DLSS can add resolution on top of an
            # upscale. The chosen DLSS mode covers that budget, and the resize lands on it.
            megapixel_w, megapixel_h = dlss_scale_target(
                width, height, dlss5_scale_to_megapixels)
            if megapixel_w and megapixel_h and (
                    dlss_result.shape[1] != megapixel_h
                    or dlss_result.shape[2] != megapixel_w):
                logger.info("[H3 Upscale] DLSS5: %dx%d -> %dx%d (dlss5_scale_to_megapixels %.2f MP, "
                            "exact aspect)" % (dlss_result.shape[2], dlss_result.shape[1],
                                               megapixel_w, megapixel_h,
                                               dlss5_scale_to_megapixels))
                dlss_result = _resize_bhwc(dlss_result, megapixel_h, megapixel_w)
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

        return (final_images, face_debug_images, self._decode_latent_audio(latent, audio_vae))

__all__ = [
    "H3FL2VAandRef2VA",
    "H3SlidingContextSampler",
    "FilmAuteurH3",
    "FilmAuteur_LTX",
    "UltimateSDUpscaleMiniMaxH3",
]
