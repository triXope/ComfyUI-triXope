"""triXope ComfyUI nodes — Video assembly, trimming and upscaling.

Every node class in this module is registered in the pack's top-level
``__init__.py`` and appears in the ComfyUI node menu under ``triXope/video``.
The module also holds the private helpers those nodes need.
"""

from __future__ import annotations

# --- moved from video_timeline_editor.py ------------------------------------
# ComfyUI Video Timeline Editor backend node.

import asyncio
import bisect
import glob
import json
import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from fractions import Fraction
from typing import Any

import av
import torch
from comfy_api.latest import ComfyExtension, io


CUT_THRESHOLD = 0.12
# Preview frames are served downscaled to the monitor and kept as JPEG bytes per node, so
# playback and scrubbing refetch a frame instead of decoding and compositing it again.
PREVIEW_JPEG_QUALITY = 78
PREVIEW_CACHE_ENTRIES = 160
_MEDIA_CACHE: dict[str, dict[str, Any]] = {}
_EXPORT_CACHE: dict[str, tuple[str, float]] = {}
_routes_registered = False


class VideoTimelineEditor(io.ComfyNode):
    # Keep legacy prompt validators aware that this side-effect node is executable.
    OUTPUT_NODE = True

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="triXope VideoTimelineEditor",
            display_name="Δ Video Timeline Editor",
            category="triXope/video",
            description="Cut and trim connected image/audio tracks, preview playback, then export an MP4.",
            inputs=[
                io.Image.Input("images", optional=True, tooltip="Image sequence source available on either video timeline lane."),
                io.Image.Input("images2", optional=True, tooltip="Optional overlay image sequence."),
                io.Audio.Input("audio", optional=True, tooltip="Optional primary audio track."),
                io.Audio.Input("audio2", optional=True, tooltip="Optional additional audio track to mix."),
                io.String.Input("timeline_json", default="", socketless=True, extra_dict={"hidden": True}),
                io.String.Input("ui_action", default="inspect", socketless=True, extra_dict={"hidden": True}),
                io.String.Input("video", default="", optional=True, tooltip="Segments folder(s), a joined MP4, or a ';' list of them: the editor plays and edits the clips from disk instead of holding every frame in memory."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
            outputs=[],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor | None = None,
        timeline_json: str = "",
        ui_action: str = "inspect",
        audio: dict[str, Any] | None = None,
        images2: torch.Tensor | None = None,
        audio2: dict[str, Any] | None = None,
        video: str = "",
    ) -> io.NodeOutput:
        try:
            state = _decode_timeline_payload(timeline_json)
            fps = int(state.get("media", {}).get("fps", 24))
        except (TypeError, ValueError, json.JSONDecodeError):
            fps = 24
        if fps not in (8, 12, 16, 24, 25, 30, 48, 50, 60):
            raise ValueError("FPS must be one of 8, 12, 16, 24, 25, 30, 48, 50, or 60.")
        if images is not None and (images.ndim != 4 or images.shape[0] < 1 or images.shape[-1] not in (3, 4) or images.shape[1] < 2 or images.shape[2] < 2):
            raise ValueError("Primary images must have shape [B,H,W,3 or 4] with at least one frame and dimensions of at least 2x2.")
        video_value = str(video or "").strip()
        if images is None and images2 is None and not video_value:
            raise ValueError("Connect an image sequence, or the segments/video source, to the editor.")
        if images2 is not None and (images2.ndim != 4 or images2.shape[0] < 1 or images2.shape[-1] not in (3, 4) or images2.shape[1] < 2 or images2.shape[2] < 2):
            raise ValueError("Overlay images must have shape [B,H,W,3 or 4] with at least one frame and dimensions of at least 2x2.")

        audio_waveform, sample_rate = _validate_audio_input(audio, "primary")
        audio_waveform2, sample_rate2 = _validate_audio_input(audio2, "secondary")

        node_id = str(cls.hidden.unique_id)
        if not node_id or node_id == "None":
            raise RuntimeError("ComfyUI did not provide the hidden unique node ID required by the timeline editor.")
        source = None
        if images is None and video_value:
            paths = _video_paths(video_value)
            if not paths:
                raise ValueError(
                    "The segments/video input found no video files to load: %s" % video_value[:300])
            source = _MediaSource(paths)
        elif video_value:
            print("[VideoTimeline] images and segments/video are both connected; using the images.")
        previous = _MEDIA_CACHE.get(node_id)
        if previous is not None:
            if previous.get("source") is not None:
                previous["source"].close()
            if previous.get("prefetch_source") is not None:
                previous["prefetch_source"].close()
        media = {
            "images": images.detach().to(device="cpu") if images is not None else None,
            "images2": images2.detach().to(device="cpu") if images2 is not None else None,
            "audio": audio_waveform.detach().to(device="cpu") if audio_waveform is not None else None,
            "audio2": audio_waveform2.detach().to(device="cpu") if audio_waveform2 is not None else None,
            "sample_rate": sample_rate,
            "sample_rate2": sample_rate2,
            "source": source,
            "cuts": None,
        }
        _MEDIA_CACHE[node_id] = media
        stats = _media_stats(media, fps)

        from server import PromptServer

        _register_routes()
        PromptServer.instance.send_sync(
            "video_timeline_editor.media_info",
            {"node_id": node_id, "fps": fps, **stats},
        )
        if ui_action != "export":
            return io.NodeOutput()

        export_token = _render_timeline_export(media, timeline_json)
        _schedule_export_expiry(export_token)
        PromptServer.instance.send_sync(
            "video_timeline_editor.exported",
            {"node_id": node_id, "token": export_token},
        )
        return io.NodeOutput()


def _media_stats(media: dict[str, Any], fps: int) -> dict[str, int]:
    """Frame counts, source rates and canvas size for one editor's cached media."""
    images = media.get("images")
    images2 = media.get("images2")
    audio = media.get("audio")
    audio2 = media.get("audio2")
    sample_rate = int(media.get("sample_rate") or 0)
    sample_rate2 = int(media.get("sample_rate2") or 0)
    source = media.get("source")
    image_frames = int(source.total) if source is not None else (int(images.shape[0]) if images is not None else 0)
    image_frames2 = int(images2.shape[0]) if images2 is not None else 0
    audio_samples = int(audio.shape[-1]) if audio is not None else 0
    audio_samples2 = int(audio2.shape[-1]) if audio2 is not None else 0
    if source is not None:
        image_fps = float(source.fps)
        image_width, image_height = int(source.width), int(source.height)
    else:
        image_fps = _resolve_image_fps(image_frames, audio_samples, sample_rate, fps)
        canvas_source = images if images is not None else images2
        image_width = int(canvas_source.shape[2]) if canvas_source is not None else 0
        image_height = int(canvas_source.shape[1]) if canvas_source is not None else 0
    image_fps2 = _resolve_image_fps(image_frames2, audio_samples2, sample_rate2, fps)
    return {
        "image_frames": image_frames,
        "image_fps": image_fps,
        "image_width": image_width,
        "image_height": image_height,
        "image_timeline_frames": round(image_frames * fps / image_fps) if image_frames else 0,
        "image2_frames": image_frames2,
        "image2_fps": image_fps2,
        "image2_timeline_frames": round(image_frames2 * fps / image_fps2) if image_frames2 else 0,
        "audio_frames": (audio_samples * fps + sample_rate - 1) // sample_rate if audio_samples else 0,
        "audio_samples": audio_samples,
        "sample_rate": sample_rate,
        "audio2_frames": (audio_samples2 * fps + sample_rate2 - 1) // sample_rate2 if audio_samples2 else 0,
        "audio2_samples": audio_samples2,
        "sample_rate2": sample_rate2,
    }


def _video_paths(value: str) -> list[str]:
    """The clips named by a segments folder list: a joined MP4 when present, else the segments."""
    paths: list[str] = []
    for part in str(value or "").split(";"):
        part = part.strip().strip("\"'")
        if not part:
            continue
        if os.path.isdir(part):
            joined = os.path.join(part, "joined.mp4")
            if os.path.isfile(joined):
                paths.append(joined)
            else:
                paths.extend(sorted(glob.glob(os.path.join(part, "segment_*.mp4"))))
        elif os.path.isfile(part):
            paths.append(part)
    return paths


def _count_video_frames(path: str) -> int:
    """Packet count of one clip: exact for these CFR H.264 segments and far cheaper than decoding."""
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            return sum(1 for packet in container.demux(stream) if packet.dts is not None)
    except Exception:  # noqa: BLE001 - the caller falls back to the duration
        return 0


def _probe_video(path: str) -> dict[str, Any]:
    """Frame count, rate and size of one clip, read from the container rather than by decoding."""
    with av.open(path) as container:
        streams = container.streams.video
        if not streams:
            raise ValueError("No video stream in %s" % path)
        stream = streams[0]
        rate = float(stream.average_rate or stream.base_rate or 0) or 24.0
        frames = int(stream.frames or 0)
        if frames <= 0:
            # A packet count is exact for these CFR H.264 segments and needs no decoding. A
            # duration estimate can be a frame long, which used to run past the real end.
            frames = _count_video_frames(path)
        if frames <= 0:
            seconds = None
            if stream.duration is not None and stream.time_base is not None:
                seconds = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                seconds = float(container.duration) / 1_000_000.0
            if not seconds:
                raise ValueError("Could not read the length of %s" % path)
            frames = int(round(seconds * rate))
        if frames <= 0:
            raise ValueError("Could not read the length of %s" % path)
        return {
            "path": path,
            "frames": frames,
            "fps": rate,
            "width": int(stream.width),
            "height": int(stream.height),
        }


class _MediaSource:
    """Frames read from the segment files, decoded one at a time as the editor asks for them.

    One or more clips play as a single sequence: the frame table says which file holds a frame,
    and only one decoder is open, so the editor holds a frame or two instead of the whole clip.
    Sequential reads (playback, export) decode forward; a seek rewinds to the keyframe before
    the frame and decodes up to it.
    """

    def __init__(self, paths: list[str]):
        self.segments = [_probe_video(path) for path in paths]
        self.starts: list[int] = []
        total = 0
        for segment in self.segments:
            self.starts.append(total)
            total += segment["frames"]
        self.total = total
        self.fps = float(self.segments[0]["fps"])
        self.width = int(self.segments[0]["width"])
        self.height = int(self.segments[0]["height"])
        self._container = None
        self._stream = None
        self._segment = -1
        self._next = -1
        self._last: tuple[int | None, Any] = (None, None)
        self._last_frame = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._close()

    def _close(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            finally:
                self._container = None
                self._stream = None
                self._segment = -1
                self._next = -1

    def _open(self, segment: int):
        self._close()
        container = av.open(self.segments[segment]["path"])
        self._container = container
        self._stream = container.streams.video[0]
        self._segment = segment
        self._next = self.starts[segment]
        return container

    def frame(self, index: int) -> Any:
        index = int(index)
        with self._lock:
            if self._last[0] == index:
                return self._last[1]
        frame = self._decoded(index)
        if frame is None:
            raise IndexError("Frame %d could not be read from the loaded clips." % index)
        tensor = _frame_tensor(frame)
        with self._lock:
            self._last = (index, tensor)
        return tensor

    def frame_rgb(self, index: int, width: int = 0):
        """One decoded frame as uint8 RGB, scaled in the decoder when a preview width is given."""
        frame = self._decoded(int(index))
        if frame is None:
            raise IndexError("Frame %d could not be read from the loaded clips." % int(index))
        if width and 0 < int(width) < frame.width:
            height = max(1, round(frame.height * int(width) / frame.width))
            frame = frame.reformat(width=int(width), height=height, format="rgb24")
        return frame.to_ndarray(format="rgb24")

    def _decoded(self, index: int):
        if index < 0 or index >= self.total:
            raise IndexError("Frame %d is outside the loaded clips (0..%d)." % (index, self.total - 1))
        with self._lock:
            segment = bisect.bisect_right(self.starts, index) - 1
            if self._container is None or self._segment != segment or self._next < 0 or self._next > index:
                return self._seek(segment, index - self.starts[segment])
            return self._advance_to(index)

    def _advance_to(self, index: int):
        while self._next < index:
            if self._decode_one() is None:
                break
        frame = self._decode_one()
        return frame if frame is not None else self._last_frame

    def _decode_one(self):
        try:
            frame = next(self._container.decode(video=0))
        except (StopIteration, av.error.EOFError):
            return None
        self._next += 1
        self._last_frame = frame
        return frame

    def _seek(self, segment: int, local: int):
        container = self._open(segment)
        if local <= 0:
            frame = self._decode_one()
            return frame if frame is not None else self._last_frame
        try:
            time_base = float(self._stream.time_base)
            container.seek(int(local / self.fps / time_base), stream=self._stream, backward=True)
        except Exception:  # noqa: BLE001 - no seekable timestamps: read forward from the top
            return self._advance_to(self.starts[segment] + local)
        frame = self._decode_one()
        if frame is None:
            self._open(segment)
            return self._advance_to(self.starts[segment] + local)
        at = self._local_index(frame)
        if at > local:
            self._open(segment)
            frame = self._decode_one()
            if frame is None:
                return self._last_frame
            at = 0
        while at < local:
            frame = self._decode_one()
            if frame is None:
                return self._last_frame
            at += 1
        return frame

    def _local_index(self, frame) -> int:
        if frame.pts is None:
            return max(0, self._next - self._segment_start() - 1)
        return int(round(float(frame.pts * frame.time_base) * self.fps))

    def _segment_start(self) -> int:
        return self.starts[self._segment] if self._segment >= 0 else 0


def _frame_tensor(frame) -> Any:
    """One decoded frame as HWC float RGB, the shape the editor's composite expects."""
    array = frame.to_ndarray(format="rgb24")
    return torch.from_numpy(array).to(torch.float32).div_(255.0)


def _frame_in_range(media: dict[str, Any], name: str | None, index: int) -> bool:
    if name is None:
        return False
    source = media.get("source") if name == "images" else None
    if source is not None:
        return 0 <= int(index) < source.total
    tensor = media.get(name)
    return tensor is not None and 0 <= int(index) < int(tensor.shape[0])


def _media_frame(media: dict[str, Any], name: str | None, index: int,
                 file_source: _MediaSource | None = None) -> Any:
    if name is None:
        return None
    source = (file_source if file_source is not None else media.get("source")) if name == "images" else None
    if source is not None:
        return source.frame(int(index))
    tensor = media.get(name)
    return tensor[int(index)].detach().to(device="cpu", dtype=torch.float32)


def _detect_file_cuts(source: _MediaSource, threshold: float = CUT_THRESHOLD) -> list[int]:
    """Hard cuts across the loaded clips, decoded at 48x48 so the whole scan stays small."""
    cuts = set(int(start) for start in source.starts[1:])
    previous = None
    for info, start in zip(source.segments, source.starts):
        with av.open(info["path"]) as container:
            for at, frame in enumerate(container.decode(video=0)):
                small = frame.reformat(width=48, height=48, format="rgb24").to_ndarray()
                sample = torch.from_numpy(small).to(torch.float32).div_(255.0)
                if previous is not None and float((sample - previous).abs().mean()) >= threshold:
                    cuts.add(int(start) + at)
                previous = sample
    return sorted(cut for cut in cuts if 0 < cut < source.total)


def _untouched_timeline(timeline: dict[str, list[dict[str, int]]], stats: dict[str, int], fps: int) -> bool:
    """True when the edit is the whole clip with no overlay and no audio, so export can copy it."""
    images = timeline.get("images") or []
    return (
        not timeline.get("images2") and not timeline.get("audio") and not timeline.get("audio2")
        and len(images) == 1
        and images[0].get("source", "images") == "images"
        and images[0]["start"] == 0
        and images[0]["source_start"] == 0
        and images[0]["end"] == stats["image_frames"]
        and images[0]["source_end"] == stats["image_frames"]
        and abs(float(stats["image_fps"]) - float(fps)) < 1e-6
    )


def _remux_source(source: _MediaSource, output_path: str) -> None:
    """Copy a single loaded clip's packets into the export without decoding or re-encoding."""
    if len(source.segments) != 1:
        raise ValueError("only a single-file clip can be copied")
    with av.open(source.segments[0]["path"]) as container, av.open(output_path, mode="w") as output:
        source_stream = container.streams.video[0]
        output_stream = output.add_stream(template=source_stream)
        for packet in container.demux(source_stream):
            if packet.dts is None:
                continue
            packet.stream = output_stream
            output.mux(packet)


def _render_timeline_export(media: dict[str, Any], timeline_payload: Any) -> str:
    """Encode the current timeline from cached media and return a one-shot token.

    This never touches the execution queue, so exporting the edit as it stands
    cannot re-run the workflow or regenerate the connected media. The work is
    CPU-bound and safe to call from a worker thread; the caller schedules expiry.
    """
    state = _decode_timeline_payload(timeline_payload)
    if not isinstance(state, dict):
        state = {}
    try:
        fps = int((state.get("media") or {}).get("fps", 24))
    except (AttributeError, TypeError, ValueError):
        fps = 24
    if fps not in (8, 12, 16, 24, 25, 30, 48, 50, 60):
        raise ValueError("FPS must be one of 8, 12, 16, 24, 25, 30, 48, 50, or 60.")

    images = media.get("images")
    images2 = media.get("images2")
    audio_waveform = media.get("audio")
    audio_waveform2 = media.get("audio2")
    sample_rate = int(media.get("sample_rate") or 0)
    sample_rate2 = int(media.get("sample_rate2") or 0)
    if images is None and images2 is None and media.get("source") is None:
        raise ValueError("Connect an image sequence, or the segments/video source, to the editor.")

    stats = _media_stats(media, fps)
    timeline = _parse_timeline(timeline_payload, stats["image_frames"], stats["audio_frames"], stats["image2_frames"], stats["audio2_frames"])
    if audio_waveform2 is not None and not timeline["audio2"] and stats["audio2_frames"]:
        timeline["audio2"] = [{"start": 0, "end": stats["audio2_frames"], "source_start": 0, "source_end": stats["audio2_frames"], "source": "audio2"}]

    audio_source_limits = {
        "audio": (stats["audio_samples"], sample_rate),
        "audio2": (stats["audio2_samples"], sample_rate2),
    }
    for lane, clips in (("audio", timeline["audio"]), ("audio2", timeline["audio2"])):
        for source_name, (sample_count, source_rate) in audio_source_limits.items():
            matching = [clip for clip in clips if clip.get("source", lane) == source_name]
            if matching and sample_count and source_rate:
                _validate_audio_timeline(matching, sample_count, source_rate, fps)
    image_clips = timeline["images"]
    image_clips2 = timeline["images2"]
    audio_clips = timeline["audio"]
    audio_clips2 = timeline["audio2"]
    total_frames = max((clip["end"] for track in timeline.values() if isinstance(track, list) for clip in track), default=0)
    if total_frames <= 0:
        raise ValueError("Timeline is empty; keep at least one image or audio frame.")
    if not image_clips and not image_clips2:
        raise ValueError("At least one image segment is required to create a video.")

    source = media.get("source")
    width, height = int(stats["image_width"]), int(stats["image_height"])
    file_descriptor, output_path = tempfile.mkstemp(prefix="comfy_video_timeline_", suffix=".mp4")
    os.close(file_descriptor)
    export_token = uuid.uuid4().hex
    if source is not None and _untouched_timeline(timeline, stats, fps):
        try:
            _remux_source(source, output_path)
        except Exception as exc:  # noqa: BLE001 - an unusual file simply falls back to encoding
            print("[VideoTimeline] the untouched clip could not be copied (%s); encoding it instead." % exc)
            try:
                os.remove(output_path)
            except OSError:
                pass
        else:
            _EXPORT_CACHE[export_token] = (output_path, time.time())
            return export_token
    try:
        container = av.open(output_path, mode="w")
    except Exception:
        os.remove(output_path)
        raise
    try:
        video_stream = container.add_stream("libx264", rate=Fraction(fps, 1))
        video_stream.width = width + (width % 2)
        video_stream.height = height + (height % 2)
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"preset": "medium", "crf": "18"}

        audio_stream = None
        audio_sources = {
            "audio": (audio_waveform, sample_rate),
            "audio2": (audio_waveform2, sample_rate2),
        }
        active_audio_sources = {name: source for name, source in audio_sources.items() if source[0] is not None}
        output_sample_rate = max((source[1] for source in active_audio_sources.values()), default=0)
        output_channels = max((int(source[0].shape[1]) for source in active_audio_sources.values()), default=1)
        audio_layout = "mono" if output_channels == 1 else "stereo"
        if active_audio_sources:
            audio_stream = container.add_stream("aac", rate=output_sample_rate, layout=audio_layout)

        for timeline_frame in range(total_frames):
            primary = _timeline_image_frame(images, images2, image_clips, timeline_frame, stats["image_fps"], stats["image2_fps"], fps, "images", source)
            overlay = _timeline_image_frame(images, images2, image_clips2, timeline_frame, stats["image_fps"], stats["image2_fps"], fps, "images2", source)
            pixels = _composite_image_frames(primary, overlay, video_stream.width, video_stream.height)
            frame = av.VideoFrame.from_ndarray(pixels.numpy(), format="rgb24")
            frame.pts = timeline_frame
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)
        if audio_stream is not None:
            _encode_audio(container, audio_stream, active_audio_sources, [audio_clips, audio_clips2], output_sample_rate, fps, total_frames, audio_layout)
    except Exception:
        container.close()
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise
    else:
        container.close()

    _EXPORT_CACHE[export_token] = (output_path, time.time())
    return export_token


def _schedule_export_expiry(token: str) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.call_later(900, _expire_export, token)


def _expire_export(token: str) -> None:
    entry = _EXPORT_CACHE.pop(token, None)
    if entry is not None:
        try:
            os.remove(entry[0])
        except OSError:
            pass


def _register_routes() -> None:
    global _routes_registered
    if _routes_registered:
        return
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/video_timeline_editor/frame")
    async def get_frame(request: web.Request) -> web.Response:
        requested_node_id = request.query.get("node_id", "")
        media = _MEDIA_CACHE.get(requested_node_id)
        if media is None:
            return web.json_response(
                {
                    "error": f"No media is cached for timeline node {requested_node_id}. Click Load media to refresh it.",
                },
                status=404,
                headers={"Cache-Control": "no-store"},
            )
        try:
            frame_index = int(request.query["frame"]) if "frame" in request.query else None
            overlay_index = int(request.query["frame2"]) if "frame2" in request.query else None
            frame_index2 = int(request.query["frame_v2"]) if "frame_v2" in request.query else None
            overlay_index2 = int(request.query["frame2_v2"]) if "frame2_v2" in request.query else None
        except ValueError:
            return web.json_response({"error": "Frame indices must be integers."}, status=400)
        try:
            preview_width = max(0, int(request.query.get("w", 0) or 0))
            quality = max(40, min(95, int(request.query.get("q", PREVIEW_JPEG_QUALITY) or PREVIEW_JPEG_QUALITY)))
        except ValueError:
            return web.json_response({"error": "Preview width and quality must be integers."}, status=400)
        media_source = media.get("source")
        images = media.get("images")
        images2 = media.get("images2")
        if images is None and images2 is None and media_source is None:
            return web.json_response({"error": "No connected image sequence is available."}, status=404)
        # A prefetch request decodes on its own reader. Sharing the playhead's reader pulled it
        # forward to the previewed frame, and the next visible frame then had to seek back to a
        # keyframe; with many short segments those seeks dominated playback.
        is_prefetch = request.query.get("p", "") in ("1", "true")
        file_source = media_source
        if is_prefetch and media_source is not None:
            file_source = media.get("prefetch_source")
            if file_source is None:
                try:
                    file_source = _MediaSource([segment["path"] for segment in media_source.segments])
                    media["prefetch_source"] = file_source
                except Exception:  # noqa: BLE001 - a preview that cannot prefetch is not an error
                    file_source = media_source
        primary_index = frame_index if frame_index is not None else overlay_index
        overlay_primary_index = frame_index2 if frame_index2 is not None else overlay_index2
        primary_name = "images" if frame_index is not None else "images2" if overlay_index is not None else None
        overlay_name = "images" if frame_index2 is not None else "images2" if overlay_index2 is not None else None
        if primary_index is not None and not _frame_in_range(media, primary_name, primary_index):
            return web.json_response({"error": "V1 frame index is outside the connected images."}, status=416)
        if overlay_primary_index is not None and not _frame_in_range(media, overlay_name, overlay_primary_index):
            return web.json_response({"error": "V2 frame index is outside the connected images."}, status=416)
        if media_source is not None and images is None:
            canvas_width, canvas_height = int(media_source.width), int(media_source.height)
        else:
            canvas_shape = images if images is not None else images2
            canvas_width, canvas_height = int(canvas_shape.shape[2]), int(canvas_shape.shape[1])
        preview_width = min(preview_width, canvas_width) if preview_width else 0
        cache = media.setdefault("preview_cache", OrderedDict())
        cache_key = (frame_index, overlay_index, frame_index2, overlay_index2, preview_width, quality)
        body = cache.get(cache_key)
        if body is None:
            def build_preview() -> bytes:
                from io import BytesIO
                from PIL import Image

                if file_source is not None and primary_index is not None and overlay_primary_index is None:
                    # The common playback path: scale the decoded frame in the decoder instead of
                    # converting the full-size frame to floats, compositing it and resizing after.
                    image = Image.fromarray(file_source.frame_rgb(primary_index, preview_width))
                else:
                    v1 = (_media_frame(media, primary_name, primary_index, file_source)
                          if primary_index is not None else None)
                    v2 = (_media_frame(media, overlay_name, overlay_primary_index, file_source)
                          if overlay_primary_index is not None else None)
                    rgb = _composite_image_frames(v1, v2, canvas_width, canvas_height)
                    image = Image.fromarray(rgb.numpy())
                    if preview_width and preview_width < image.width:
                        height = max(1, round(image.height * preview_width / image.width))
                        image = image.resize((preview_width, height), Image.BILINEAR)
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=quality)
                return buffer.getvalue()

            # Decode and encode off the event loop: on it, one frame request stalled every
            # other request (cache hits included), which showed up as stutter under load.
            body = await asyncio.get_running_loop().run_in_executor(None, build_preview)
            cache[cache_key] = body
            while len(cache) > PREVIEW_CACHE_ENTRIES:
                cache.popitem(last=False)
        else:
            cache.move_to_end(cache_key)
        return web.Response(body=body, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @PromptServer.instance.routes.get("/video_timeline_editor/cuts")
    async def get_cuts(request: web.Request) -> web.Response:
        node_id = request.query.get("node_id", "")
        media = _MEDIA_CACHE.get(node_id)
        if media is None:
            return web.json_response({"error": "No media is cached for this timeline. Click Load media, then try again."}, status=404)

        cuts = {
            "images": _detect_hard_cuts(media.get("images")),
            "images2": _detect_hard_cuts(media.get("images2")),
        }
        media_source = media.get("source")
        if media_source is not None:
            if media.get("cuts") is None:
                media["cuts"] = await asyncio.get_running_loop().run_in_executor(
                    None, _detect_file_cuts, media_source)
            cuts["images"] = media["cuts"]
        return web.json_response({"cuts": cuts, "threshold": CUT_THRESHOLD})

    @PromptServer.instance.routes.get("/video_timeline_editor/audio")
    async def get_audio(request: web.Request) -> web.Response:
        media = _MEDIA_CACHE.get(request.query.get("node_id", ""))
        track = request.query.get("track", "audio")
        audio_key = "audio2" if track == "audio2" else "audio"
        rate_key = "sample_rate2" if track == "audio2" else "sample_rate"
        if media is None or media[audio_key] is None:
            return web.Response(status=404, text=f"No connected {track} is available.")
        waveform_data = media[audio_key]
        waveform = waveform_data[0].float().clamp(-1, 1).mul(32767).short().transpose(0, 1).contiguous().numpy()
        channels = min(int(waveform_data.shape[1]), 2)
        return web.Response(
            body=waveform[:, :channels].astype("<i2", copy=False).tobytes(),
            content_type="application/octet-stream",
            headers={"X-Sample-Rate": str(media[rate_key]), "X-Channels": str(channels), "Cache-Control": "no-store"},
        )

    @PromptServer.instance.routes.get("/video_timeline_editor/export/{token}")
    async def download_export(request: web.Request) -> web.StreamResponse:
        token = request.match_info["token"]
        entry = _EXPORT_CACHE.pop(token, None)
        if entry is None or not os.path.isfile(entry[0]):
            return web.Response(status=404, text="Export is missing or has already been downloaded.")
        path = entry[0]
        filename = request.query.get("filename", "video_timeline.mp4")
        filename = os.path.basename(filename).replace("\"", "") or "video_timeline.mp4"
        response = web.FileResponse(path, headers={
            "Content-Type": "video/mp4",
            "Content-Disposition": f'attachment; filename="{filename}"',
        })

        asyncio.get_running_loop().call_later(300, lambda: os.path.exists(path) and os.remove(path))
        return response

    @PromptServer.instance.routes.post("/video_timeline_editor/export_now")
    async def export_now(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return web.json_response({"error": "Export requests must be JSON."}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "Export requests must be a JSON object."}, status=400)
        node_id = str(body.get("node_id", ""))
        media = _MEDIA_CACHE.get(node_id)
        if media is None:
            return web.json_response(
                {"error": f"No media is cached for timeline node {node_id}. Queue this node once so it loads its media, then export as often as you like."},
                status=404,
                headers={"Cache-Control": "no-store"},
            )
        timeline_payload = body.get("timeline", body.get("timeline_json", ""))
        try:
            token = await asyncio.get_running_loop().run_in_executor(None, _render_timeline_export, media, timeline_payload)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400, headers={"Cache-Control": "no-store"})
        except Exception as exc:  # noqa: BLE001 - surface render failures to the editor
            return web.json_response({"error": f"Could not render the timeline: {exc}"}, status=500, headers={"Cache-Control": "no-store"})
        _schedule_export_expiry(token)
        return web.json_response({"token": token}, headers={"Cache-Control": "no-store"})

    _routes_registered = True


def _register_routes_if_server_available() -> None:
    """Register routes during custom-node import when ComfyUI has created its server.

    comfy_entrypoint retries registration for loaders that import this module before
    the PromptServer singleton is ready.
    """
    try:
        from server import PromptServer
    except ImportError:
        return
    if getattr(PromptServer, "instance", None) is not None:
        _register_routes()


def _resolve_image_fps(
    image_frames: int,
    audio_samples: int,
    sample_rate: int,
    timeline_fps: int,
) -> float:
    # Infer source FPS from matching audio duration when available; otherwise
    # interpret the image sequence at the selected timeline rate.
    if image_frames and audio_samples and sample_rate:
        image_fps = image_frames * sample_rate / audio_samples
    else:
        image_fps = float(timeline_fps)
    if not 0 < image_fps < 1000:
        raise ValueError("Source FPS must be greater than 0 and less than 1000.")
    return image_fps


def _validate_audio_input(audio: dict[str, Any] | None, label: str) -> tuple[torch.Tensor | None, int]:
    if audio is None:
        return None, 0
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", audio.get("sampler_rate", 0)))
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError(f"{label.capitalize()} audio must have waveform shape [B,C,T].")
    if waveform.shape[0] < 1 or waveform.shape[1] < 1 or sample_rate < 1:
        raise ValueError(f"{label.capitalize()} audio must include a channel and a valid sample rate.")
    return waveform, sample_rate


def _timeline_image_frame(
    images: torch.Tensor | None,
    images2: torch.Tensor | None,
    clips: list[dict[str, int]],
    timeline_frame: int,
    image_fps: float,
    image2_fps: float,
    fps: int,
    lane: str,
    source: Any = None,
) -> torch.Tensor | None:
    clip = _clip_at(clips, timeline_frame)
    if clip is None:
        return None
    origin = clip.get("source", lane)
    source_images = images2 if origin == "images2" else images
    source_fps = image2_fps if origin == "images2" else image_fps
    source_index = clip["source_start"] + int((timeline_frame - clip["start"]) * source_fps / fps)
    source_index = min(source_index, clip["source_end"] - 1)
    if source_images is None:
        return source.frame(source_index) if source is not None and origin == "images" else None
    return source_images[source_index].detach().to(device="cpu", dtype=torch.float32)


def _detect_hard_cuts(images: torch.Tensor | None, threshold: float = CUT_THRESHOLD, batch_size: int = 64) -> list[int]:
    if images is None or images.shape[0] < 2:
        return []
    cuts: list[int] = []
    previous = None
    for start in range(0, images.shape[0], batch_size):
        frames = images[start:start + batch_size, ..., :3].to(device="cpu", dtype=torch.float32).clamp(0, 1)
        samples = torch.nn.functional.adaptive_avg_pool2d(frames.permute(0, 3, 1, 2), (48, 48))
        if previous is not None:
            difference = (samples[0] - previous).abs().mean().item()
            if difference >= threshold:
                cuts.append(start)
        if samples.shape[0] > 1:
            differences = (samples[1:] - samples[:-1]).abs().mean(dim=(1, 2, 3))
            cuts.extend((torch.nonzero(differences >= threshold, as_tuple=False).flatten() + start + 1).tolist())
        previous = samples[-1]
    return cuts


def _composite_image_frames(primary: torch.Tensor | None, overlay: torch.Tensor | None, width: int, height: int) -> torch.Tensor:
    canvas = torch.zeros((height, width, 3), dtype=torch.float32)
    if primary is not None:
        image = primary[..., :3].clamp(0, 1)
        h, w = min(height, image.shape[0]), min(width, image.shape[1])
        canvas[:h, :w] = image[:h, :w]
    if overlay is not None:
        image = overlay
        if image.shape[-1] >= 4:
            alpha = image[..., 3:4].clamp(0, 1)
        else:
            alpha = torch.ones_like(image[..., :1])
        rgb = image[..., :3].clamp(0, 1)
        h, w = min(height, rgb.shape[0]), min(width, rgb.shape[1])
        canvas[:h, :w] = rgb[:h, :w] * alpha[:h, :w] + canvas[:h, :w] * (1 - alpha[:h, :w])
    return canvas.clamp(0, 1).mul(255).round().to(torch.uint8)


def _clip_at(clips: list[dict[str, int]], timeline_frame: int) -> dict[str, int] | None:
    for clip in reversed(clips):
        if clip["start"] <= timeline_frame < clip["end"]:
            return clip
    return None


def _encode_audio(
    container,
    stream,
    audio_sources: dict[str, tuple[torch.Tensor, int]],
    lanes: list[list[dict[str, int]]],
    sample_rate: int,
    fps: int,
    total_frames: int,
    layout: str,
) -> None:
    sample_count = (total_frames * sample_rate + fps - 1) // fps
    channels = 1 if layout == "mono" else 2
    rendered = torch.zeros((channels, sample_count), dtype=torch.float32)
    for clips in lanes:
        for clip in clips:
            source_name = clip.get("source", "audio")
            source_entry = audio_sources.get(source_name)
            if source_entry is None:
                continue
            waveform, source_rate = source_entry
            source = waveform[0].detach().to(device="cpu", dtype=torch.float32)
            if source_rate != sample_rate:
                target_length = max(1, round(source.shape[-1] * sample_rate / source_rate))
                source = torch.nn.functional.interpolate(source.unsqueeze(0), size=target_length, mode="linear", align_corners=False).squeeze(0)

            destination_start = clip["start"] * sample_rate // fps
            destination_end = min(clip["end"] * sample_rate // fps, sample_count)
            source_start = clip["source_start"] * sample_rate // fps
            source_end = min(clip["source_end"] * sample_rate // fps, source.shape[-1])
            count = min(destination_end - destination_start, source_end - source_start)
            if count <= 0:
                continue
            segment = source[:, source_start:source_start + count]
            if segment.shape[0] < channels:
                segment = segment.expand(channels, -1)
            rendered[:, destination_start:destination_start + count] += segment[:channels]
    rendered.clamp_(-1.0, 1.0)
    pcm = rendered.mul(32767).to(torch.int16).contiguous()
    for offset in range(0, pcm.shape[-1], 1024):
        # PyAV's packed s16 layout expects one array row with interleaved channels.
        # pcm is channel-first, so transpose samples/channels before flattening.
        chunk = pcm[:, offset:offset + 1024].transpose(0, 1).contiguous().view(1, -1)
        frame = av.AudioFrame.from_ndarray(chunk.numpy(), format="s16", layout=layout)
        frame.sample_rate = sample_rate
        frame.time_base = Fraction(1, sample_rate)
        frame.pts = offset
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)


def _validate_audio_timeline(clips: list[dict[str, int]], sample_count: int, sample_rate: int, fps: int) -> None:
    max_sample = int(sample_count)
    max_frame = (max_sample * fps + sample_rate - 1) // sample_rate
    for clip in clips:
        source_start = clip["source_start"] * sample_rate // fps
        if (
            clip["source_start"] >= max_frame
            or clip["source_end"] > max_frame
            or source_start >= max_sample
        ):
            raise ValueError("Audio clip exceeds the connected audio duration at the selected FPS.")


def _decode_timeline_payload(payload: Any) -> dict[str, Any]:
    data = payload
    for _ in range(8):
        if isinstance(data, str):
            if not data.strip():
                return {}
            try:
                data = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ValueError("Timeline data is invalid JSON. Reload media in the editor and export again.") from exc
            continue
        if isinstance(data, dict):
            if "images" in data or "audio" in data:
                return data
            for key in ("timeline_json", "timeline", "value", "data"):
                if key in data:
                    data = data[key]
                    break
            else:
                return data
            continue
        if isinstance(data, (list, tuple)):
            # Empty arrays can be emitted for a blank hidden V3 widget value.
            if not data:
                return {}
            # Some ComfyUI widget serializers pass the hidden field in a value array.
            decoded = []
            timelines = []
            for item in data:
                try:
                    candidate = _decode_timeline_payload(item)
                except ValueError:
                    continue
                decoded.append(candidate)
                if "images" in candidate or "audio" in candidate:
                    timelines.append(candidate)
            if len(timelines) == 1:
                return timelines[0]
            if len(timelines) > 1:
                raise ValueError("Timeline data contains multiple timeline objects.")
            if len(decoded) == 1:
                return decoded[0]
            if not decoded:
                return {}
        break
    raise ValueError(f"Timeline data must be an object, not {type(data).__name__}. Reload media in the editor and export again.")


def _parse_timeline(payload: Any, max_images: int, max_audio: int, max_images2: int = 0, max_audio2: int = 0) -> dict[str, list[dict[str, int]]]:
    if not payload or (isinstance(payload, str) and not payload.strip()):
        data = {}
    else:
        data = _decode_timeline_payload(payload)
    if not any(track in data for track in ("images", "audio", "images2", "audio2")):
        data = {
            "images": [{"start": 0, "end": max_images, "source_start": 0, "source_end": max_images, "source": "images"}] if max_images else [],
            "audio": [{"start": 0, "end": max_audio, "source_start": 0, "source_end": max_audio, "source": "audio"}] if max_audio else [],
            "images2": [{"start": 0, "end": max_images2, "source_start": 0, "source_end": max_images2, "source": "images2"}] if max_images2 else [],
            "audio2": [{"start": 0, "end": max_audio2, "source_start": 0, "source_end": max_audio2, "source": "audio2"}] if max_audio2 else [],
        }

    result: dict[str, list[dict[str, int]]] = {}
    limits = {
        "images": {"images": max_images, "images2": max_images2},
        "images2": {"images": max_images, "images2": max_images2},
        "audio": {"audio": max_audio, "audio2": max_audio2},
        "audio2": {"audio": max_audio, "audio2": max_audio2},
    }
    if "images2" not in data and max_images2:
        data["images2"] = [{"start": 0, "end": max_images2, "source_start": 0, "source_end": max_images2, "source": "images2"}]
    if "audio2" not in data and max_audio2:
        data["audio2"] = [{"start": 0, "end": max_audio2, "source_start": 0, "source_end": max_audio2, "source": "audio2"}]
    for track, source_media_limits in limits.items():
        items = data.get(track, [])
        if not isinstance(items, list):
            raise ValueError(f"Timeline {track} track must be a list.")
        clips: list[dict[str, int]] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Each {track} clip must be an object.")
            values = {key: item.get(key) for key in ("start", "end", "source_start", "source_end")}
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values.values()):
                raise ValueError(f"{track.capitalize()} clip positions must be integer frames.")
            clip = {key: int(value) for key, value in values.items()}
            source_name = item.get("source", track)
            if source_name not in source_media_limits:
                raise ValueError(f"{track.capitalize()} clips must reference a source of the same media type.")
            clip["source"] = source_name
            source_limit = source_media_limits[source_name]
            if clip["start"] < 0 or clip["end"] <= clip["start"]:
                raise ValueError(f"{track.capitalize()} clip must have a positive duration and nonnegative position.")
            if source_limit <= 0 or clip["source_start"] < 0 or clip["source_end"] > source_limit or clip["source_end"] <= clip["source_start"]:
                raise ValueError(f"{track.capitalize()} clip must stay within its connected media length ({source_limit} frames).")
            if source_name.startswith("audio") and clip["end"] - clip["start"] != clip["source_end"] - clip["source_start"]:
                raise ValueError(f"{track.capitalize()} clip timeline and source durations must match in frames.")
            clips.append(clip)
        # Clips on a lane may overlap and may rejoin the source anywhere at a cut. Overlapping
        # image clips stack: the clip that starts last covers the ones underneath it (see
        # _clip_at), with the images2 lane composited over images. Overlapping audio clips mix.
        clips.sort(key=lambda clip: (clip["start"], clip["end"], clip["source_start"]))
        result[track] = clips
    return result


class VideoTimelineExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VideoTimelineEditor]


async def comfy_entrypoint() -> VideoTimelineExtension:
    # Register before ComfyUI mounts its custom aiohttp routes.
    _register_routes()
    return VideoTimelineExtension()


# Register at module import as well as from the V3 entrypoint. This covers loaders
# that discover the legacy node mapping but do not invoke the V3 entrypoint hook.
_register_routes_if_server_available()

# --- moved from video_upscale_dimensions.py ---------------------------------
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
    CATEGORY = "triXope/video"
    
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

# --- moved from minimax_h3_av_concat.py:MultiClipTrimmer --------------------
import torch

class MultiClipTrimmer:
    """
    Trims concatenated image and audio batches down to an exact target duration 
    per clip segment. Automatically calculates any excess frames (e.g. 362 - 360 = 2)
    and drops them from the START of each shot to eliminate VAE warmup degradation.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "clip_count": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1}),
                "clip_duration": ("FLOAT", {"default": 15.0, "min": 0.1, "max": 600.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "trim_clips"
    CATEGORY = "triXope/video"

    def trim_clips(self, images: torch.Tensor, clip_count: int, clip_duration: float, fps: float, audio: dict = None):
        target_frames_per_clip = int(round(clip_duration * fps))
        total_frames = images.shape[0]
        raw_frames_per_clip = total_frames // clip_count

        trimmed_images_list = []
        for i in range(clip_count):
            segment_start = i * raw_frames_per_clip
            segment_end = (i + 1) * raw_frames_per_clip if i < clip_count - 1 else total_frames
            segment_len = segment_end - segment_start

            # Automatically calculate excess frames for this shot and trim them from the HEAD
            excess_frames = max(0, segment_len - target_frames_per_clip)
            start_idx = segment_start + excess_frames
            end_idx = start_idx + min(target_frames_per_clip, segment_len - excess_frames)

            trimmed_images_list.append(images[start_idx:end_idx])

        out_images = torch.cat(trimmed_images_list, dim=0)

        out_audio = None
        if audio is not None and "waveform" in audio:
            waveform = audio["waveform"]
            sample_rate = audio.get("sample_rate", 44100)
            target_samples_per_clip = int(round(clip_duration * sample_rate))
            total_samples = waveform.shape[-1]
            raw_samples_per_clip = total_samples // clip_count

            trimmed_audio_list = []
            for i in range(clip_count):
                segment_start_s = i * raw_samples_per_clip
                segment_end_s = (i + 1) * raw_samples_per_clip if i < clip_count - 1 else total_samples
                segment_len_s = segment_end_s - segment_start_s

                # Automatically calculate excess audio samples and trim them from the HEAD
                excess_samples = max(0, segment_len_s - target_samples_per_clip)
                start_s = segment_start_s + excess_samples
                end_s = start_s + min(target_samples_per_clip, segment_len_s - excess_samples)

                if waveform.ndim == 3:
                    trimmed_audio_list.append(waveform[:, :, start_s:end_s])
                elif waveform.ndim == 2:
                    trimmed_audio_list.append(waveform[:, start_s:end_s])
                else:
                    trimmed_audio_list.append(waveform[start_s:end_s])

            out_waveform = torch.cat(trimmed_audio_list, dim=-1)
            out_audio = {
                "waveform": out_waveform,
                "sample_rate": sample_rate
            }

        return (out_images, out_audio)

# --- moved from filmauteur_ltx.py:LTXVPostSliceAV ---------------------------
class LTXVPostSliceAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "drop_first_n_frames": ("INT", {"default": 9, "min": 0, "max": 99999, "step": 1, "tooltip": "Exact number of video frames to drop from the beginning."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01, "tooltip": "Must match your generation FPS to calculate the audio sync."}),
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "slice_av"
    CATEGORY = "triXope/video"

    def slice_av(self, images, drop_first_n_frames, fps, audio=None):
        num_frames = images.shape[0]
        
        if drop_first_n_frames >= num_frames:
            print(f"LTXV Custom Warning: Trying to drop {drop_first_n_frames} frames but only {num_frames} exist. Returning the last frame to prevent a crash.")
            sliced_images = images[-1:] 
        elif drop_first_n_frames > 0:
            sliced_images = images[drop_first_n_frames:]
            print(f"LTXV Custom: Successfully sliced {drop_first_n_frames} video frames.")
        else:
            sliced_images = images

        sliced_audio = None
        
        if audio is not None:
            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate")
            
            if waveform is not None and sample_rate is not None:
                time_to_drop_seconds = drop_first_n_frames / fps
                samples_to_drop = int(time_to_drop_seconds * sample_rate)
                
                total_samples = waveform.shape[-1]
                
                if samples_to_drop >= total_samples:
                    print(f"LTXV Custom Warning: Trying to drop {samples_to_drop} audio samples but only {total_samples} exist.")
                    empty_waveform = torch.zeros((waveform.shape[0], waveform.shape[1], 1), device=waveform.device, dtype=waveform.dtype)
                    sliced_audio = {"waveform": empty_waveform, "sample_rate": sample_rate}
                elif samples_to_drop > 0:
                    sliced_waveform = waveform[..., samples_to_drop:]
                    sliced_audio = {"waveform": sliced_waveform, "sample_rate": sample_rate}
                    print(f"LTXV Custom: Successfully sliced {samples_to_drop} audio samples ({time_to_drop_seconds:.3f} seconds).")
                else:
                    sliced_audio = audio
            else:
                sliced_audio = audio

        return (sliced_images, sliced_audio)

# --- moved from nodes.py:VideoPreview ---------------------------------------
class triXope_VideoPreview:
    @classmethod
    def INPUT_TYPES(s):
        target_dir = os.path.join(folder_paths.get_input_directory(), "_film\\output\\video")
        video_list = []

        if os.path.exists(target_dir):
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    if file.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
                        video_list.append(file)

        return {"required": {"video": ("STRING",),}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "triXope/video"

    def execute(self, video):
        video_path = os.path.join(folder_paths.get_input_directory(), "_film\\output\\video", video)
        video_name = os.path.basename(video_path)
        video_path_name = os.path.dirname(video_path)
        print(f"Selected video: {video_name}")
        print(f"Full video path: {video_path}")
        return {"ui": {"video": [video_name, video_path_name]}}

__all__ = [
    "VideoTimelineEditor",
    "Video_Upscale_To_Dimensions",
    "MultiClipTrimmer",
    "LTXVPostSliceAV",
    "triXope_VideoPreview",
]
