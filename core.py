"""Pure helpers for MiniMax H3 temporal windows and timestamped story beats."""
from __future__ import annotations

import re

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24
AUDIO_LATENT_FPS = 40
_TIMESTAMPED_BEAT = re.compile(
    r"^\s*\[\s*(?P<start>\d{1,3}:\d{2}(?::\d{2})?(?:\.\d+)?)"
    r"\s*(?:[-–]\s*(?P<end>\d{1,3}:\d{2}(?::\d{2})?(?:\.\d+)?))?"
    r"\s*\]\s*(?P<text>.+?)\s*$"
)
_TIMESTAMP = re.compile(r"^\d{1,3}:\d{2}(?::\d{2})?(?:\.\d+)?$")
_CROSSFADE_CUE = re.compile(r"\b(?:cross[\s-]?fade|dissolve)\b", re.IGNORECASE)
_CUT_CUE = re.compile(r"\b(?:cut\s+to|hard\s+cut|scene\s+cut|new\s+scene)\b", re.IGNORECASE)


def parse_timestamp(value: str) -> float:
    """Parse MM:SS[.sss] or HH:MM:SS[.sss] as seconds."""
    if not _TIMESTAMP.fullmatch(value):
        raise ValueError(f"Invalid timestamp {value!r}; use MM:SS or HH:MM:SS")
    parts = value.split(":")
    if len(parts) == 2:
        hours, minutes, seconds = 0, int(parts[0]), float(parts[1])
    else:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid timestamp {value!r}; minutes and seconds must be below 60")
    return hours * 3600 + minutes * 60 + seconds


def format_timestamp(seconds: float) -> str:
    """Format a non-negative time as HH:MM:SS.mmm."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:06.3f}"


def parse_timestamped_beats(script: str) -> list[dict]:
    """Parse explicit or next-marker-ended timestamped story beats.

    Supported lines: ``[00:00:05-00:00:12] beat`` and ``[00:00:12] beat``.
    A beat without an end stops at the next beat start; the last such beat
    extends to the end of whichever output chunk contains it.
    """
    beats = []
    for line_number, line in enumerate(script.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _TIMESTAMPED_BEAT.match(line)
        if match is None:
            raise ValueError(
                f"Line {line_number}: expected [MM:SS-MM:SS] beat or [MM:SS] beat"
            )
        start = parse_timestamp(match.group("start"))
        end_text = match.group("end")
        end = parse_timestamp(end_text) if end_text else None
        if end is not None and end <= start:
            raise ValueError(f"Line {line_number}: beat end must be after its start")
        beats.append({"start": start, "end": end, "text": match.group("text")})
    if not beats:
        raise ValueError("Provide at least one timestamped story beat")
    starts = [beat["start"] for beat in beats]
    if starts != sorted(starts) or len(set(starts)) != len(starts):
        raise ValueError("Story beats must have unique timestamps in ascending order")
    for index, beat in enumerate(beats[:-1]):
        if beat["end"] is None:
            beat["end"] = beats[index + 1]["start"]
        elif beat["end"] > beats[index + 1]["start"]:
            raise ValueError(f"Beat at {format_timestamp(beat['start'])} overlaps the next beat")
    return beats


def select_beats_for_window(beats: list[dict], window_start: float,
                            window_end: float) -> str:
    """Clip intersecting beats and label their chunk-local and global times."""
    if window_start < 0 or window_end <= window_start:
        raise ValueError("window must have a non-negative start and positive duration")
    selected = []
    for beat in beats:
        start = max(beat["start"], window_start)
        end = min(beat["end"] if beat["end"] is not None else window_end, window_end)
        if start >= end:
            continue
        selected.append(
            f"[local {format_timestamp(start - window_start)}-{format_timestamp(end - window_start)}; "
            f"global {format_timestamp(start)}-{format_timestamp(end)}] {beat['text']}"
        )
    if not selected:
        return ""
    header = (
        f"Generate this chunk from global time {format_timestamp(window_start)} "
        f"to {format_timestamp(window_end)}. Follow only these beats in order; "
        "local timestamps are relative to this chunk."
    )
    return header + "\n" + "\n".join(selected)


def explicit_transition_mode(text: str) -> str | None:
    """Return a transition explicitly named in a beat, or None for continuation."""
    if _CROSSFADE_CUE.search(text):
        return "crossfade"
    if _CUT_CUE.search(text):
        return "cut"
    return None


def transition_for_overlap(beats: list[dict], start_frame: int, end_frame: int) -> dict | None:
    """Find the first explicitly cued transition whose beat begins in an overlap."""
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("overlap must have non-negative, ordered frame boundaries")
    for beat in beats:
        mode = explicit_transition_mode(beat["text"])
        frame = int(round(beat["start"] * FPS))
        if mode is not None and start_frame <= frame < end_frame:
            return {"mode": mode, "frame": frame}
    return None


def transition_offset_tokens(window_start: int, overlap_tokens: int, transition_frame: int) -> int:
    """Map an absolute 24-fps transition frame to a local H3 overlap token offset."""
    if window_start < 0 or overlap_tokens < 0 or transition_frame < 0:
        raise ValueError("window, overlap, and transition positions must be non-negative")
    start_frame = video_frames_for_latents(window_start)
    end_frame = video_frames_for_latents(window_start + overlap_tokens)
    if not start_frame <= transition_frame <= end_frame:
        raise ValueError("transition frame must lie inside the window overlap")
    low, high = 0, overlap_tokens
    while low < high:
        middle = (low + high) // 2
        frame = video_frames_for_latents(window_start + middle) - start_frame
        if frame < transition_frame - start_frame:
            low = middle + 1
        else:
            high = middle
    return low


def video_frames_for_latents(latent_frames: int) -> int:
    """Return the H3 pixel-frame span represented by a video latent time axis."""
    if latent_frames < 0:
        raise ValueError("latent_frames must be non-negative")
    return sum(FRAME_PER_TOKEN[i % len(FRAME_PER_TOKEN)] for i in range(latent_frames))


def select_beats_for_latent_window(beats: list[dict], start: int, end: int) -> str:
    """Select beats by an H3 latent window's absolute video-time boundaries."""
    if start < 0 or end <= start:
        raise ValueError("latent window must have a non-negative start and positive duration")
    start_seconds = video_frames_for_latents(start) / FPS
    end_seconds = video_frames_for_latents(end) / FPS
    return select_beats_for_window(beats, start_seconds, end_seconds)


def remap_keyframes_for_window(keyframes_with_durations: list[tuple[dict, int]],
                               window_start: int, window_end: int) -> list[dict]:
    """Route every guide interval intersecting a chunk and rebase its anchor.

    Guide latents are retained intact; crossing clips are not cropped because
    arbitrary slices are not guaranteed to align with H3's temporal latent grid.
    """
    if window_start < 0 or window_end <= window_start:
        raise ValueError("window must have a non-negative start and positive duration")
    remapped = []
    for keyframe, duration_frames in keyframes_with_durations:
        frame_index = keyframe.get("resolved_frame_index")
        if not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("H3 keyframes require a non-negative integer resolved_frame_index")
        if not isinstance(duration_frames, int) or duration_frames <= 0:
            raise ValueError("H3 keyframe guide duration must be a positive integer")
        guide_end = frame_index + duration_frames
        if frame_index < window_end and guide_end > window_start:
            local_keyframe = keyframe.copy()
            local_keyframe["resolved_frame_index"] = frame_index - window_start
            remapped.append(local_keyframe)
    return remapped


def frames_to_audio_latents(frame_count: int) -> int:
    """Map a 24-fps video frame boundary to H3's 40-Hz audio latent grid."""
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    return (frame_count * AUDIO_LATENT_FPS + FPS // 2) // FPS


def requested_frames_to_latents(requested_frames: int) -> int:
    """Round to H3's supported 17k+5 frame length pattern (minimum 22 here)."""
    if requested_frames < 22:
        raise ValueError("window and overlap must be at least 22 frames")
    groups = max(1, (requested_frames - 5 + 16) // 17)
    return 2 + 5 * groups


def plan_windows(total_latent_frames: int, window_latent_frames: int,
                 overlap_latent_frames: int) -> list[tuple[int, int]]:
    """Plan overlapping [start, end) windows on the H3 temporal token grid."""
    if total_latent_frames < 2 or (total_latent_frames - 2) % 5:
        raise ValueError("expected an H3 video latent length of 2 + 5k temporal tokens")
    if window_latent_frames < 7 or (window_latent_frames - 2) % 5:
        raise ValueError("window length must be 2 + 5k temporal tokens")
    if overlap_latent_frames < 7 or (overlap_latent_frames - 2) % 5:
        raise ValueError("overlap must be 2 + 5k temporal tokens")
    step = window_latent_frames - overlap_latent_frames
    if step < 5 or step % 5:
        raise ValueError("overlap must be smaller than the window by at least 5 tokens")

    windows = []
    start = 0
    while True:
        end = min(start + window_latent_frames, total_latent_frames)
        windows.append((start, end))
        if end == total_latent_frames:
            return windows
        start += step