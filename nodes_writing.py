"""triXope ComfyUI nodes — Screenwriting, story and prompt authoring.

Every node class in this module is registered in the pack's top-level
``__init__.py`` and appears in the ComfyUI node menu under ``triXope/writing``.
The module also holds the private helpers those nodes need.
"""

# --- moved from screenplay_formatter.py -------------------------------------
import math
import re
import json
import gc
import os
import urllib.request
import urllib.error
import html
import time
import queue
import random
import threading


_SCREENPLAY_RUN = threading.local()


_SCRIPT_OVERRIDE_MODES = ("off", "text", "audio")

# Bigger Whisper checkpoints win when they are already cached locally, so a normal
# install never has to download anything mid-workflow.
_TRANSCRIPT_MODEL_ORDER = ("large-v3", "large-v2", "medium", "small", "base", "tiny")
_TRANSCRIPT_SOUND_TAG_RE = re.compile(r"\[[^\]]*\]")
_UTTERANCE_GAP_SECONDS = 0.9
_TRANSCRIPT_SAMPLE_RATE = 16000
# Whisper skips low-confidence windows; a sung opening over a loud intro is exactly what it
# drops, so the gate is relaxed and the audio before the first line is decoded again.
_TRANSCRIPT_NO_SPEECH_THRESHOLD = 0.85
_TRANSCRIPT_LEAD_RECOVERY_SECONDS = 1.0
_TRANSCRIPT_RECOVERY_MIN_LOGPROB = -1.0
_TRANSCRIPT_RECOVERY_MAX_COMPRESSION = 2.4
_TRANSCRIPT_SILENCE_RMS = 1e-4
# Gaps at least this long have no words over them (intros, breaks, outros) and become their own
# shot so the picture keeps moving through the instrumental stretches.
_TRANSCRIPT_INSTRUMENTAL_MIN_SECONDS = 2.0
_TIMELINE_MAX_SHOT_SECONDS = 12.0


# ---------------------------------------------------------------------------
#  Blank-script mode: raw_script empty + script_override_mode=off
# ---------------------------------------------------------------------------
# The writer field is optional. When it is left blank the model invents an
# entire original screenplay instead of returning empty outputs, so a user can
# queue the node cold and still get a script, shot list and beats to work from.
_RANDOM_SCREENPLAY_MIN_PAGES = 3
_RANDOM_SCREENPLAY_MAX_PAGES = 12

_RANDOM_GENRES = (
    "neo-noir", "heist thriller", "slow-burn horror", "folk horror", "science fiction",
    "space opera", "western", "courtroom drama", "romantic comedy", "coming-of-age",
    "road movie", "political satire", "psychological thriller", "war drama",
    "family drama", "surrealist fable", "sports underdog story", "period mystery",
)

_RANDOM_STORY_SHAPES = (
    "linear and chronological",
    "one extended flashback that catches up with the present in the final act",
    "two timelines that only make sense once the last scene connects them",
    "told backwards from the aftermath, revealing the cause in the final scene",
    "a framing device in the present that the main story is told from",
    "one location, one continuous night, with the backstory revealed through what the characters admit",
)

_RANDOM_SCREENPLAY_PROMPT = """You are an award-winning screenwriter writing a complete, original, production-ready screenplay from scratch. Nothing is supplied to you; every creative decision is yours. Invent a story nobody has told before.

THE BRIEF (already randomised, follow it exactly):
- Genre: @@GENRE@@
- Structure: @@SHAPE@@
- Target length: @@PAGES@@ pages, roughly @@MINUTES@@ minutes of screen time (about @@WORDS@@ words)
- Cast: @@CASTLINE@@ No stock or generic names, and no two names that could be confused with one another.

STORY REQUIREMENTS:
1. Tell a complete three-act story with a beginning, a middle and an end: a hook that establishes a protagonist with a concrete want and a specific obstacle, a midpoint that changes what is at stake, and a climax that resolves the central conflict.
2. It does not have to be linear. Flashbacks, intercut timelines, framing devices and withheld information are all welcome, but every thread must come together and be paid off by the final scene. Nothing important can be left dangling.
3. The story must be coherent on a first read: consistent cause and effect, characters acting from recognisable motives, and a world with its own internal logic.
4. Be specific and surprising. Invent particular jobs, places, objects, rituals, slang and small physical details instead of generic ones. Avoid cliché, coincidence that solves the plot, twist endings that contradict earlier scenes, and moralising narration.

FORMATTING RULES (mandatory - this script is parsed mechanically, so follow them exactly):
- Scene headings on their own line, in capitals, starting with INT. or EXT., then the location, then a dash and DAY, NIGHT, DAWN, DUSK, CONTINUOUS or LATER. Example: INT. BUS DEPOT - NIGHT
- A character cue on its own line, in capitals, with nothing else on that line, directly above that character's dialogue.
- Dialogue lines immediately under the cue. Use a parenthetical on its own line, in (parentheses), only when the delivery matters.
- Action and description in plain present tense. No camera directions, no shot numbers, no montage headings, no scene numbers.
- Open with FADE IN: and close the story with FADE OUT.
- Do not use markdown, bullet points, bold text, section headings, notes, commentary or any text outside the screenplay itself. No title page.

OUTPUT: the screenplay only, starting with FADE IN: and ending with FADE OUT."""

_GENERATED_SCREENPLAY_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*$")
_SCENE_HEADING_RE = re.compile(r"^(?:INT\.|EXT\.|INT/EXT\.|EXT/INT\.|INT\./EXT\.|EST\.)", re.IGNORECASE)
_FADE_OUT_RE = re.compile(r"^FADE OUT\.?$", re.IGNORECASE)


def _random_screenplay_prompt():
    """Build a randomised brief for an original three-act screenplay.

    Returns (prompt, info); every call reseeds from system entropy so repeated
    runs produce a different genre, structure and length.
    """
    rng = random.Random()
    genre = rng.choice(_RANDOM_GENRES)
    shape = rng.choice(_RANDOM_STORY_SHAPES)
    pages = rng.randint(_RANDOM_SCREENPLAY_MIN_PAGES, _RANDOM_SCREENPLAY_MAX_PAGES)
    cast = rng.randint(1, 6)
    # A solo lead must not read as "1 or so speaking characters".
    cast_line = ("A single speaking character, with a distinctive, memorable name." if cast == 1
                 else f"{cast} or so speaking characters, each with a distinctive, memorable name.")
    prompt = (_RANDOM_SCREENPLAY_PROMPT
              .replace("@@GENRE@@", genre)
              .replace("@@SHAPE@@", shape)
              .replace("@@PAGES@@", str(pages))
              .replace("@@MINUTES@@", str(pages))
              .replace("@@WORDS@@", str(pages * 190))
              .replace("@@CASTLINE@@", cast_line))
    info = {
        "genre": genre,
        "shape": shape,
        "pages": pages,
        "cast": cast,
        # ~190 words per page plus title beats; keeps the token budget ahead of the ask.
        "num_predict": min(8000, pages * 420 + 800),
    }
    return prompt, info


def _clean_generated_screenplay(text):
    """Strip fences and any chatty preamble so the screenplay parser starts clean."""
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in raw.split("\n")]
    lines = [line for line in lines if not _GENERATED_SCREENPLAY_FENCE_RE.match(line.strip())]
    first = next((i for i, line in enumerate(lines) if _SCENE_HEADING_RE.match(line.strip())), None)
    if first:
        dropped = [line for line in lines[:first] if line.strip()]
        if dropped:
            print(f"     [Random Screenplay] Dropped {len(dropped)} line(s) of preamble before the first scene heading.")
        lines = lines[first:]
    # Models often close with a friendly line; the screenplay itself ends at FADE OUT.
    last = next((i for i in range(len(lines) - 1, -1, -1) if _FADE_OUT_RE.match(lines[i].strip())), None)
    if last is not None:
        trailing = [line for line in lines[last + 1:] if line.strip()]
        if trailing:
            print(f"     [Random Screenplay] Dropped {len(trailing)} line(s) after FADE OUT.")
        lines = lines[:last + 1]
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip("\n"))
    return out



def _script_override_mode(value):
    """Normalise script_override_mode; a legacy boolean maps to text/off."""
    if isinstance(value, bool):
        return "text" if value else "off"
    mode = str(value or "off").strip().lower()
    if mode not in _SCRIPT_OVERRIDE_MODES:
        raise ValueError("script_override_mode must be off, text, or audio")
    return mode


def _transcript_model_name():
    requested = os.environ.get("TRX_ASR_MODEL", "").strip()
    if requested:
        return requested
    cache = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    for name in _TRANSCRIPT_MODEL_ORDER:
        if os.path.isfile(os.path.join(cache, name + ".pt")):
            return name
    return "base"


def _transcript_device():
    requested = os.environ.get("TRX_ASR_DEVICE", "").strip()
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _audio_to_asr_samples(audio):
    """ComfyUI AUDIO -> mono float32 samples at 16 kHz, the rate Whisper expects."""
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        raise ValueError("script_override_audio must be a ComfyUI AUDIO input")
    import torch
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().float().cpu().contiguous()
    mono = waveform if waveform.ndim == 1 else waveform.reshape(-1, waveform.shape[-1]).mean(dim=0)
    sample_rate = int(audio.get("sample_rate") or 0)
    if sample_rate and sample_rate != 16000:
        import librosa
        mono = torch.from_numpy(librosa.resample(mono.numpy(), orig_sr=sample_rate, target_sr=16000))
    return mono.contiguous().numpy().astype("float32")


def _visible_transcript_text(raw):
    """Segment text with sound tags and the sung marker removed, or "" when nothing is left."""
    return _collapse(_TRANSCRIPT_SOUND_TAG_RE.sub(" ", str(raw or "")).replace("\u266a", " ")).strip()


def _whisper_segments(result):
    """Whisper's raw result -> the timed {start, end, text} records the node works with."""
    return [
        {"start": float(segment.get("start") or 0.0),
         "end": float(segment.get("end") or 0.0),
         "text": str(segment.get("text") or "")}
        for segment in (result.get("segments") or [])
    ]


def _recover_lead_in(model, samples, segments, fp16):
    """Decode the audio before the first detected line again, looking for lines Whisper dropped.

    Whisper discards windows it rates as no-speech or low-confidence, and the first phrase of a
    song over a loud intro is exactly that kind of window, so the opening lines of a track go
    missing. The leading slice is decoded with those gates off, then each recovered line is kept
    only if the decoder itself was confident and did not loop, so real lines come back without
    inventing lyrics for a silent or purely instrumental intro.
    """
    import numpy as np

    lead_end = float(segments[0]["start"]) if segments else float(len(samples)) / _TRANSCRIPT_SAMPLE_RATE
    if lead_end < _TRANSCRIPT_LEAD_RECOVERY_SECONDS:
        return []
    clip = samples[:max(1, int(lead_end * _TRANSCRIPT_SAMPLE_RATE))]
    clip_rms = float(np.sqrt(np.mean(np.square(clip, dtype=np.float64))))
    whole_rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) or 1.0
    if clip_rms < max(_TRANSCRIPT_SILENCE_RMS, 0.1 * whole_rms):
        print("[ScreenplayFormatter] Audio opens with silence; nothing precedes the first line.")
        return []
    print(f"[ScreenplayFormatter] Re-decoding the first {lead_end:.1f}s of audio for dropped opening lines...")
    result = model.transcribe(
        clip, task="transcribe", verbose=False, fp16=fp16,
        condition_on_previous_text=False,
        no_speech_threshold=1.0, logprob_threshold=-2.0,
    )
    recovered = []
    for segment in result.get("segments") or []:
        if not _visible_transcript_text(segment.get("text")):
            continue
        if float(segment.get("avg_logprob") or 0.0) < _TRANSCRIPT_RECOVERY_MIN_LOGPROB:
            continue
        if float(segment.get("compression_ratio") or 1.0) >= _TRANSCRIPT_RECOVERY_MAX_COMPRESSION:
            continue
        recovered.append({"start": float(segment.get("start") or 0.0),
                          "end": float(segment.get("end") or 0.0),
                          "text": str(segment.get("text") or "")})
    if recovered:
        print(f"[ScreenplayFormatter] Recovered {len(recovered)} opening segment(s) the first pass dropped.")
    else:
        print("[ScreenplayFormatter] The opening audio held no extra words to recover.")
    return recovered


def _transcribe_audio(audio, model_name=None):
    """Whisper-transcribe a ComfyUI AUDIO input into timed segments plus the clip's real duration."""
    samples = _audio_to_asr_samples(audio)
    if samples.size == 0:
        raise ValueError(
            "script_override_audio is silent; connect an audio track with speech or lyrics"
        )
    try:
        import whisper
    except ImportError as error:
        raise RuntimeError(
            "script_override_audio needs the openai-whisper package in ComfyUI's python; "
            "install it with pip install openai-whisper in ComfyUI_windows_portable\\\\python_embeded"
        ) from error
    name = model_name or _transcript_model_name()
    device = _transcript_device()
    duration = float(samples.size) / _TRANSCRIPT_SAMPLE_RATE
    print(f"[ScreenplayFormatter] Transcribing speech/lyrics with Whisper '{name}' on {device}...")
    model = whisper.load_model(name, device=device)
    fp16 = device.startswith("cuda")
    try:
        result = model.transcribe(
            samples, task="transcribe", verbose=False, fp16=fp16,
            condition_on_previous_text=False,
            no_speech_threshold=_TRANSCRIPT_NO_SPEECH_THRESHOLD,
        )
        segments = _recover_lead_in(model, samples, _whisper_segments(result), fp16) + _whisper_segments(result)
    finally:
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    print(f"[ScreenplayFormatter] Whisper returned {len(segments)} timed segment(s) over {duration:.2f}s of audio.")
    return {"segments": segments, "duration": duration}


def _transcript_groups(segments, spoken_cue="SPEAKER", lyric_cue="SINGER", gap=_UTTERANCE_GAP_SECONDS):
    """Whisper segments -> cue groups of individually timed lines, speech and song kept apart.

    Speech becomes a CHARACTER cue plus DIALOGUE lines; lines Whisper marks as sung (`\u266a`,
    `(singing)`, a music tag) get their own cue so the sung part stays distinct. Whisper does not
    diarise voices, so every speaker shares one cue instead of inventing names the character
    pipeline would then treat as real people.
    """
    groups = []
    current = None
    for segment in segments:
        raw = str(segment.get("text") or "")
        text = _visible_transcript_text(raw)
        if not text:
            current = None
            continue
        tags = " ".join(_TRANSCRIPT_SOUND_TAG_RE.findall(raw)).lower()
        sung = "\u266a" in raw or "singing" in tags or "music" in tags
        cue = lyric_cue if sung else spoken_cue
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        if current is None or current["cue"] != cue or start - current["end"] > gap:
            current = {"cue": cue, "lines": [], "end": start}
            groups.append(current)
        current["lines"].append({"text": text, "start": start, "end": end})
        current["end"] = end
    return groups


def _groups_to_screenplay(groups):
    """Cue groups -> screenplay text the STEP 1 parser understands."""
    return "\n\n".join(
        "\n".join([group["cue"]] + [line["text"] for line in group["lines"]])
        for group in groups if group["lines"]
    )


def _transcript_to_screenplay(segments, spoken_cue="SPEAKER", lyric_cue="SINGER",
                              gap=_UTTERANCE_GAP_SECONDS):
    """Turn Whisper segments into screenplay text the STEP 1 parser understands."""
    return _groups_to_screenplay(_transcript_groups(segments, spoken_cue, lyric_cue, gap))


def _audio_timeline(groups, duration):
    """Lay the transcript out over the real audio length, gap by gap.

    Every group of lines becomes one timed item, and any stretch without words that is long
    enough to read as an intro, break or outro becomes an instrumental item of its own, so the
    whole timeline covers the audio from 0.0 to its exact duration with no holes.
    """
    duration = max(0.0, float(duration))
    items = []
    cursor = 0.0
    for group in groups:
        first = float(group["lines"][0]["start"]) if group["lines"] else cursor
        start = min(max(cursor, first), duration)
        end = min(max(start, float(group["end"])), duration)
        if start - cursor >= _TRANSCRIPT_INSTRUMENTAL_MIN_SECONDS:
            items.append({"kind": "instrumental", "cue": "", "start": cursor, "end": start, "lines": []})
        else:
            start = cursor
        if end > start:
            items.append({"kind": "dialogue", "cue": group["cue"], "start": start, "end": end,
                          "lines": list(group["lines"])})
            cursor = end
    if duration - cursor >= _TRANSCRIPT_INSTRUMENTAL_MIN_SECONDS:
        items.append({"kind": "instrumental", "cue": "", "start": cursor, "end": duration, "lines": []})
    elif items and duration > items[-1]["end"]:
        items[-1]["end"] = duration
    return _split_long_timeline_items(items)


def _split_long_timeline_items(items, max_seconds=_TIMELINE_MAX_SHOT_SECONDS):
    """Split single items longer than one shot may run, keeping the timeline contiguous.

    A long sung run is cut on its own line times; an instrumental stretch is cut into even
    chunks, so one item never forces an over-long shot in the shot list.
    """
    split = []
    for item in items:
        span = float(item["end"]) - float(item["start"])
        if span <= max_seconds:
            split.append(item)
            continue
        chunks = max(2, int(math.ceil(span / float(max_seconds))))
        step = span / chunks
        if item["kind"] != "dialogue":
            for index in range(chunks):
                start = float(item["start"]) + step * index
                end = float(item["end"]) if index == chunks - 1 else start + step
                split.append({"kind": item["kind"], "cue": item["cue"], "start": start, "end": end, "lines": []})
            continue
        buckets = [[] for _ in range(chunks)]
        for line in item["lines"]:
            index = int((float(line["start"]) - float(item["start"])) // step)
            buckets[min(chunks - 1, max(0, index))].append(line)
        start = float(item["start"])
        for index, lines in enumerate(buckets):
            end = float(item["end"]) if index == chunks - 1 else float(item["start"]) + step * (index + 1)
            if not lines:
                start = end
                continue
            split.append({"kind": "dialogue", "cue": item["cue"], "start": start, "end": end, "lines": lines})
            start = end
    return split


def _timeline_lengths(items, duration):
    """Whole-second shot lengths that add up to the audio's real length.

    Each shot starts from its own span rounded to the nearest second (never below the node's
    minimum), then the total is nudged onto the audio duration by handing the leftover seconds
    to the shots whose span was rounded down hardest.
    """
    spans = [max(0.0, float(item["end"]) - float(item["start"])) for item in items]
    lengths = [max(MIN_SHOT_SECONDS, int(round(span))) for span in spans]
    target = max(MIN_SHOT_SECONDS, int(round(float(duration))))
    guard = 0
    while lengths and sum(lengths) < target and guard < 1000:
        index = max(range(len(lengths)), key=lambda i: spans[i] - lengths[i])
        lengths[index] += 1
        guard += 1
    while lengths and sum(lengths) > target and guard < 1000:
        longer = [i for i, value in enumerate(lengths) if value > MIN_SHOT_SECONDS]
        if not longer:
            break
        index = max(longer, key=lambda i: lengths[i] - spans[i])
        lengths[index] -= 1
        guard += 1
    return lengths

def _check_processing_interrupted():
    """Raise ComfyUI's interrupt exception at a cooperative cancellation point."""
    try:
        import comfy.model_management as model_management
    except ImportError:
        return
    model_management.throw_exception_if_processing_interrupted()


def _is_processing_interrupted(error):
    try:
        import comfy.model_management as model_management
    except ImportError:
        return False
    interrupt_type = getattr(model_management, "InterruptProcessingException", None)
    return isinstance(interrupt_type, type) and isinstance(error, interrupt_type)


def _release_comfyui_cached_memory():
    """Clear unused allocator cache only; do not unload models owned by other nodes."""
    try:
        import comfy.model_management as model_management
    except ImportError:
        return
    empty_cache = getattr(model_management, "soft_empty_cache", None)
    if callable(empty_cache):
        try:
            empty_cache()
            print("🧹 [ScreenplayFormatter] Released unused ComfyUI GPU allocator cache.")
        except Exception as error:
            print(f"⚠️ [ScreenplayFormatter] Could not release ComfyUI GPU allocator cache: {error}")


def _stream_ollama_request(req, timeout, response_queue, response_holder, cancelled):
    """Read Ollama's NDJSON stream in a worker so the node can poll Stop promptly."""
    pieces, thinking_pieces, metadata = [], [], {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_holder.append(response)
            for raw_line in response:
                if cancelled.is_set():
                    return
                line = raw_line.strip()
                if not line:
                    continue
                item = json.loads(line.decode("utf-8"))
                if item.get("error"):
                    raise RuntimeError(str(item["error"]))
                if item.get("response"):
                    pieces.append(str(item["response"]))
                if item.get("thinking"):
                    thinking_pieces.append(str(item["thinking"]))
                if item.get("done"):
                    metadata = {key: item[key] for key in ("done_reason", "eval_count") if key in item}
                    break
        response_queue.put(("response", {
            "response": "".join(pieces), "thinking_chars": sum(map(len, thinking_pieces)), **metadata,
        }))
    except BaseException as error:
        if not cancelled.is_set():
            response_queue.put(("error", error))


KIND_CHARACTER = "character"
KIND_PROP = "prop"
KIND_LOCATION = "location"
KIND_ORDER = {KIND_CHARACTER: 0, KIND_PROP: 1, KIND_LOCATION: 2}

MIN_SHOT_SECONDS = 3
# MiniMax H3's own tooltip documents a trained range of about 124-362 frames, and 124
# frames is ~5.2s. A shorter shot therefore renders outside the range the model was
# trained on, which is exactly where limbs, faces and acting fall apart.
MIN_H3_SHOT_SECONDS = 5
WORDS_PER_PAGE = 250
SECONDS_PER_PAGE = 60
SCREENPLAY_PAGE_LINES = 55
SCREENPLAY_LINE_WIDTH = 60
STORY_BEAT_BATCH_SIZE = 20
MAX_REF_IMAGES = 9999
# The H3 conditioner exposes four autogrow reference-video slots, so the screenwriter
# must warn at the same limit instead of at three.
MAX_REF_VIDEOS = 4

# Shared blocks of the shot list: they ride in every shot of the Film Auteur (H3) node.
SHOT_LIST_STYLE = "A cinematic live-action film with motivated, naturalistic lighting and a controlled color grade."
SHOT_LIST_SOUNDSCAPE = "Natural ambient sound that fits each environment."

# The color-palette reference image is appended AFTER the last prop in the image batch
# (<Picture N+1>). Flip to True if you connect it as the FIRST image instead.
PALETTE_AT_START = False

LIST_DELIM = "\n\n| \n\n"

CHARACTER_SHEET_LAYOUT = (
    "An image divided by three vertical bars into four synchronized panels side by side: "
    "the first panel a full-body front view; the second panel a full-body side profile view, held in strict "
    "side profile\u2014the subject's face and body stay turned toward frame-left from edge to edge; "
    "the third panel a full-body rear view; the fourth panel a medium close-up of the subject's face. "
    "Each view contains the identical subject, fully preserved\u2014face, hair, eyes, chest, figure, "
    "proportions, and wardrobe are identical in every panel, the same person."
)
STILL_FRAME_SENTENCE = "The shot is a single frozen still frame: the camera is locked off and nothing in the frame moves."
STILL_SOUNDSCAPE = "Faint, quiet studio room tone."
NO_PEOPLE_SENTENCE = "No people, animals or creatures appear anywhere in the frame."
CHARACTER_NOTE = "Only the identity and wardrobe are used, never the four-panel sheet layout."

PROP_FIELD_KEYS = ["Type", "Material & Texture", "Size & Scale", "Color & Visual", "Condition & Wear",
                   "Narrative Significance", "Distinguishable Features"]
LOCATION_FIELD_KEYS = ["Style & Era", "Scale & Layout", "Color Palette", "Lighting & Atmosphere",
                       "Key Props & Set Dressing", "Condition & Traits"]

RELATIONSHIP_MARKERS = ["fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"]

# =============================================================================
#  SYSTEM PROMPTS  (placeholders use @@TOKEN@@ so JSON braces never need escaping)
# =============================================================================

# ---------------------------------------------------------------- character_list
CHARACTER_STAGE1_PROMPT = """You are an expert Casting Director and Screenplay Analyst. Your task is to analyze screenplays or scripts provided by the user and extract complete, fully detailed character descriptions for every character that appears in the narrative.

PIPELINE CONTEXT:
Characters are the FIRST assets in the generation order (characters -> props -> locations). Each character is generated as a reference image and is then cited in every shot by its exact name, so every name must be exact, unique and stable.

CORE RULES & REQUIREMENTS:
1. Mandatory Standardized Name Header: EVERY character entry MUST explicitly begin with `Name: [EXACT CHARACTER NAME] - `.
2. Verbatim Name Consistency: Use the exact, full name verbatim across all prompts, in UPPERCASE (e.g., "JOHN DOE"). Never shorten names, use nicknames, or alter spellings anywhere. One character = one name, and no character's name may be contained inside another character's name.
3. Comprehensive Extraction: Identify every named or prominent character in the script and generate a full physical profile for each. Do not list props, locations or anonymous background crowds as characters.

OUTPUT FORMAT FOR EACH ENTRY:
Name: [EXACT CHARACTER NAME] - Facial structure: [...]; Hair style: [...]; Hair color: [...]; Wardrobe: [...]; Distinguishing traits: [...]; Eye shape: [...]; Eye color: [...]; Facial hair: [...]; Build: [...]; Height: [...]; Age: [...]; Ethnicity: [...]; Weight: [...] |
4. Required Attributes: Label every attribute exactly as written above and keep exactly that order on the single line after the name, separated by `; `. The first five labels carry the character's identity on camera and must be the most concrete and specific: bone structure, nose, jaw and skin tone; hair length, texture and parting; the exact garments, colors, fabrics and footwear worn; scars, tattoos, posture and gait. Measurements follow. Never leave a label empty: infer a plausible, consistent value.
5. Visible Details Only: Describe only what a camera can see. Do not include backstory, personality or narrative role.
6. Spacing & Separation Rules:
   - Each character entry must end with a pipe character (`|`).
   - Follow the pipe character with a double line break so that a blank line separates each subject, and the next character begins on a new line.
   - Never use a pipe character inside an entry.
7. Inference & Consistency: Where specific physical details are not explicitly stated in the script, infer realistic, context-appropriate attributes that fit the character's role, era, and tone of the screenplay.

OUTPUT EXECUTION: Do not include conversational greetings, meta-commentary, notes, or concluding summaries. Output only the formatted character profiles starting immediately with the first character.

KNOWN CHARACTER DEFINITIONS:
@@KNOWN@@
"""

CHARACTER_STAGE2_PROMPT = """You are an expert AI Concept Artist and Prompt Engineer specializing in video keyframe preparation for MiniMax H3. You transform character profiles into the visual BODY of a MiniMax-H3 text-to-video (T2VA) prompt that renders a four-panel character sheet as one frozen still frame.

HOW YOUR OUTPUT IS USED (the software adds this scaffolding itself; never reproduce it):
  integrated_multimodal_description: [Shot 1] <style opening> <four-panel layout sentence> <YOUR BODY> <still-frame sentence>
  overall_soundscape: <room tone>
  non_diegetic_music: N/A
The four-panel layout sentence (three vertical bars dividing the image into four synchronized panels: full-body front, full-body strict side profile facing frame-left, full-body rear, medium close-up of the face, with the identical subject in every panel) is added automatically. Do NOT write it, and never describe panels, bars or layout.

MANDATORY ENTRY FORMAT:
Name: [EXACT CHARACTER NAME] - [BODY] |

CORE RULES FOR THE BODY:
1. Mandatory Standardized Name Header: EVERY entry MUST start with `Name: [EXACT CHARACTER NAME] - ` using the exact name from the profile. Never put the name inside the BODY.
2. Neutral Studio Background: The character is isolated on a plain, solid neutral gray or white backdrop with zero background clutter.
3. Absolute Identity Consistency: Write the body so it describes ONE person whose face, hair, eyes, chest, figure, proportions and wardrobe never change.
4. Plain visible prose: One paragraph, present tense, English, only things a camera can see. No backstory, personality, action, props held, other people, text or scene.
5. Cover, in this order: <Subject physical profile> (age, height/build, ethnicity, skin tone, facial structure, hair style and color, eye shape and color, facial hair, distinguishing traits) <Wardrobe> (every garment, color, material, condition) <Pose & Lighting> (neutral standing pose, soft even studio lighting, sharp focus) <Aesthetic Drivers> (photorealistic detail, texture, color grading).
6. STRICT PIPE RULES: Each entry MUST terminate with a single pipe character (`|`). Never use a pipe inside the body.

OUTPUT EXECUTION: Output only the entries, no greetings, notes or summaries.
"""

# ---------------------------------------------------------------- prop_list
PROP_STAGE1_PROMPT = """You are an expert Prop Master and Screenplay Analyst. Your task is to analyze screenplays or scripts provided by the user and extract complete, fully detailed descriptions for every crucial object, prop, or key item that plays a significant role in the narrative.

PIPELINE CONTEXT:
Props are the SECOND assets in the generation order (characters -> props -> locations). Each prop is generated as an isolated reference image and is then cited by exact name in location prompts and in shots. The name you choose is the only link between the prop image and every prompt that uses it. Costume pieces, set pieces, architectural elements and vegetation count as props when they are visually important and can be shown as an isolated object.

CHARACTER NAMES (already assigned; never reuse them as prop names):
@@CHARACTERS@@

CORE RULES & REQUIREMENTS:
1. Mandatory Standardized Name Header: EVERY prop entry MUST explicitly begin with `Object Name: [EXACT PROP NAME] - `.
2. Exact & Consistent Object Naming: Derive a single, precise, standardized name for every object/prop in UPPERCASE and use that exact string without variation across all outputs. Names must be distinct, and no prop name may be contained inside another prop's name (do not use both ALTAR and RITUAL ALTAR).

OUTPUT FORMAT FOR EACH ENTRY:
Object Name: [EXACT PROP NAME] - [Full detailed visual description, material, colors, size, and physical condition.] |
3. Comprehensive Extraction: Identify every crucial, recurring, or plot-driving object in the script and generate a full physical and functional profile for each.
4. Strict Object Isolation: Describe the item in complete isolation.
   - No Characters/Subjects: Do NOT mention or refer to any characters, people, hands, bodies, or owners (e.g., write "engraved with initials" instead of "held by Arthur" or "owned by the detective").
   - No Surrounding Context/Proximity: Do NOT mention any surrounding background, location, furniture, or adjacent objects (e.g., do not write "resting on a wooden table" or "placed next to a lamp").
   - Explicit Exception: Only describe a person or subject if the object itself intrinsically depicts one (e.g., the subject printed on a photograph, painted on a portrait, carved into a statue, or printed on an ID card).
5. Required Attributes: Every object description must explicitly and thoroughly detail:
   - Object Name / Item Title
   - Type / Category (e.g., Prop, Weapon, Tool, Document, Memento, Technology)
   - Material & Texture (e.g., Polished brass, weathered wood, scuffed leather, digital glass)
   - Size & Scale (e.g., 4 inches long, pocket-sized, 2-foot diameter, tabletop scale)
   - Color & Visual Appearance (e.g., Matte black with glowing blue runes, faded crimson)
   - Condition & Wear (e.g., Pristine, heavily dented, dried water spots, rusted hinges)
   - Narrative Significance / Function (e.g., Unlocks the main vault mechanism, serves as the central clue, acts as a cipher device)
   - Distinguishable Features (e.g., Engraved initials "J.C.", flickering power indicator, weighted base)
6. Spacing & Separation Rules:
   - Each object entry must end with a pipe character (`|`).
   - Follow the pipe character with a double line break so that a blank line separates each subject, and the next object begins on a new line.
   - Do NOT place a pipe character (`|`) after the final object entry, and never use a pipe inside an entry.
7. Inference & Consistency: Where specific physical details are not explicitly stated in the script, infer realistic, visually rich attributes that fit the story's genre, era, and narrative tone while remaining strictly isolated to the standalone object.

OUTPUT SCHEMA & LAYOUT:
Object Name: <NAME> - Type: <Category> - Material & Texture: <Material> - Size & Scale: <Size> - Color & Visual: <Color> - Condition & Wear: <Condition> - Narrative Significance: <Significance> - Distinguishable Features: <Features> |

OUTPUT EXECUTION: Do not include conversational greetings, meta-commentary, notes, or concluding summaries. Output only the formatted object profiles starting immediately with the first object.
"""

# ---------------------------------------------------------------- location_list
LOCATION_STAGE1_PROMPT = """You are an expert Production Designer and Location Scout. Your task is to analyze screenplays or scripts provided by the user and extract complete, fully detailed location descriptions for every distinct setting that appears in the narrative.

PIPELINE CONTEXT:
Locations are the THIRD assets in the generation order (characters -> props -> locations). Each location is generated as an empty-set reference still (later used as a reference video) from the props that are named in its description, and is then cited in shots by its exact name.

KNOWN PROPS (exact names, registry order):
@@PROPS@@

CORE RULES & REQUIREMENTS:
1. Mandatory Standardized Name Header: EVERY location entry MUST explicitly begin with `Location Name: [EXACT LOCATION NAME] - `. The name is everything between `Location Name:` and the first ` - `; do not insert a second descriptor between the name and `Style & Era:`.
2. Verbatim Location Naming: If a location exists in KNOWN LOCATION DEFINITIONS, you MUST use its EXACT name verbatim across all outputs.

OUTPUT FORMAT FOR EACH ENTRY:
Location Name: [EXACT LOCATION NAME] - [Full production design description, environment type, architecture, lighting, weather, and mood.] |
3. Comprehensive Extraction: Identify every unique environment, set, or location in the script and generate a full production design profile for each.
4. Required Attributes: Every location description must explicitly and thoroughly detail:
   - Location Name & Type (e.g., INT. / EXT. Name)
   - Architectural Style & Era (e.g., 1980s Brutalist, Gothic Revival, Modern Minimalist)
   - Spatial Scale & Layout (e.g., Cramped basement, cavernous high-ceiling warehouse)
   - Color Palette & Tonal Aesthetic (e.g., Desaturated blues, warm earth tones, neon-drenched)
   - Lighting & Atmosphere (e.g., Harsh fluorescent flicker, golden hour sunlight, rain-slicked haze)
   - Key Props & Set Dressing (e.g., Weathered oak desk, exposed pipes, vintage vinyl collection)
   - Condition & Distinguishable Features (e.g., Water-stained ceiling, peeling wallpaper, rust, cracked pavement)
5. Props Rule (CRITICAL): In Key Props & Set Dressing, name every KNOWN PROP that physically appears in this location by its EXACT UPPERCASE name, and ONLY the ones that truly appear (each named prop is attached as a reference image when the location is generated). Describe any other set dressing in plain lowercase words. Locations are shown EMPTY: never name or describe characters, crowds or creatures.
6. Spacing & Separation Rules:
   - Each location entry must end with a pipe character (`|`).
   - Follow the pipe character with a double line break so that a blank line separates each subject, and the next location begins on a new line.
   - Do NOT place a pipe character (`|`) after the final location entry, and never use a pipe inside an entry.
7. Inference & Consistency: Where specific environmental details are not explicitly stated in the script, infer realistic, visually rich design elements that fit the scene's mood, era, and visual narrative.

OUTPUT SCHEMA & LAYOUT:
Location Name: <Name (INT/EXT)> - Style & Era: <Architectural Style/Era> - Scale & Layout: <Spatial Scale/Layout> - Color Palette: <Palette> - Lighting & Atmosphere: <Lighting/Atmosphere> - Key Props & Set Dressing: <Props/Dressing> - Condition & Traits: <Condition/Features> |

OUTPUT EXECUTION: Do not include conversational greetings, meta-commentary, notes, or concluding summaries. Output only the formatted location profiles starting immediately with the first location.

KNOWN LOCATION DEFINITIONS:
@@KNOWN@@
"""

LOCATION_STAGE2_PROMPT = """# Role & Objective
You are an expert Production Designer and MiniMax-H3 prompt engineer. You turn ONE location profile into a prompt in MiniMax-H3's official FULL-REFERENCE (Ref2VA) six-section format. The location is rendered as a single frozen still frame (a 5-frame clip) of the EMPTY set, built from reference images of the props that appear in it. The finished clip is later used as a location reference VIDEO (<Video N>) for shots.

# Props, Generation Order & Labels
Assets are generated in the fixed order CHARACTERS -> PROPS -> LOCATIONS, and every reference IMAGE has a FIXED picture number equal to its position in the registry (characters first, then props). ALL reference images are always attached, whether or not this prompt uses them. This location is shown EMPTY, so you may name ONLY props (never characters, never other locations). Props with their fixed picture numbers:
@@PROPS@@
- Each prop you name becomes a <Subject N> (numbered per prompt, restarting at 1, in registry order) backed by its FIXED reference image <Picture N>: the number shown in the list above, which never changes (with the registry order MARA, BRASS LENS, SPIRAL STAIRCASE, naming BRASS LENS and SPIRAL STAIRCASE yields <Subject 1>/<Picture 2> and <Subject 2>/<Picture 3>).
- A prop or character this location does not use is never mentioned: no name and no picture number.
- YOU NEVER WRITE THE NUMBERS. Write each prop's EXACT registry NAME in UPPERCASE every time you refer to it; the software replaces names with labels and builds the citations automatically.
- Name ONLY props that genuinely appear in this location (the profile's Key Props & Set Dressing lists them). Describe all other set dressing in plain lowercase words.
- Give every named prop a specific role and position in the set.

# Output Format (STRICT)
Plain text; exactly these six section headers, in this order, each on its own line followed by its content; no markdown, no JSON, no code fences, no commentary, never the vertical bar character:
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

## subject_definitions
One line per named prop, in registry order, formatted exactly `NAME: traits` (max ~35 words): only what the reference image supplies visually (shape, material, color, condition, distinguishing features, taken from the registry description). No placement or actions, and never another asset's name. If no known prop appears, write the single word `none`.

## summary
One short paragraph beginning with `[reference generation]`: a single static wide shot of the empty location and which prop supplies which part of it. Refer to props by NAME.

## retention_analysis
One line per named prop: `NAME (appears in [Shot 1]): <marker> - <what is retained and how it is used>`. Markers (fixed English): fully_preserved, partially_preserved (still used, some traits change, e.g., scaled up into architecture), attribute_transfer (its characteristics move to a different identifiable element, e.g., torn cloth becoming hanging drapes), weak_reference (only broad similarity kept). If no prop is named write `none`.

## detailed_description
1. A style opening of one or two sentences on its own line BEFORE the shot marker: a full-frame, professional architectural landscape photograph with studio lighting, plus the architectural style/era, color palette and lighting from the profile.
2. Then the marker `[Shot 1]` (no timestamp) followed by ONE paragraph of roughly 250-400 words: a locked-off, eye-level wide static shot that shows the whole space: scale, layout, architecture, depth, lighting and atmosphere, condition and distinguishing features, and each named prop (by NAME) with its position, role and look.
3. Describe only what is visible, as observable detail. State "No people, animals or creatures appear anywhere in the frame." and "The shot is a single frozen still frame: the camera is locked off and nothing in the frame moves."

## overall_soundscape
One sentence of ambient sound that fits the space.

## non_diegetic_music
`N/A`

# Format Example (structure only, shortened: the real detailed_description runs roughly 250-400 words; never reuse this content)
subject_definitions:
BRASS LENS: A large faceted brass-framed lens with green-tinted glass and a tarnished rim.
SPIRAL STAIRCASE: A tight wrought-iron spiral staircase with rust-flecked steps and a thin handrail.
summary:
[reference generation] A single static wide shot of the empty lighthouse lantern room, with BRASS LENS as the central lamp and SPIRAL STAIRCASE as the descending access.
retention_analysis:
BRASS LENS (appears in [Shot 1]): fully_preserved - the faceted green-tinted lens and tarnished brass frame are retained as the central lamp.
SPIRAL STAIRCASE (appears in [Shot 1]): fully_preserved - the rust-flecked iron spiral is retained, descending through the floor opening at frame-right.
detailed_description:
A full-frame, professional architectural landscape photograph with studio lighting in a weathered maritime style with a cold teal and tarnished brass palette.
[Shot 1] A locked-off, eye-level wide shot frames the circular lantern room from the doorway, its curved glass walls dark with salt. BRASS LENS fills the center on a stone plinth, its green-tinted facets catching a thin teal light, while SPIRAL STAIRCASE winds down through a round floor opening at frame-right, its rust-flecked iron steps fading into shadow. No people, animals or creatures appear anywhere in the frame. The shot is a single frozen still frame: the camera is locked off and nothing in the frame moves.
overall_soundscape:
A low wind hum presses against the glass.
non_diegetic_music:
N/A

@@ASPECT_NOTE@@
"""

# Location reference videos: every generated location prompt ends with this camera block,
# so the empty-set reference clip is one frozen instant seen from a full orbit.
LOCATION_REFERENCE_CAMERA_SUFFIX = """The camera completes one smooth 360-degree orbit around the main subject in a single consistent direction. Maintain a steady camera distance and height, returning to the starting viewpoint and framing without reversing. Preserve the background geometry and background objects in their original world positions, orientations and shapes, with their natural appearance intact. Camera parallax and changing occlusion reveal the scene from successive viewpoints. No cuts, zoom, morphing or added objects.

One frozen instant. Only the camera moves. Preserve every person and object in the same world position, orientation, shape and pose throughout. Airborne objects remain suspended at their captured height and angle. Faces, hands, clothing and liquids remain frozen with their natural appearance intact. Nothing moves independently. No continued action."""

# ---------------------------------------------------------------- shot_list : pass 1
SHOT_BREAKDOWN_PROMPT = """# Role & Objective
You are an expert AI Audio-Visual Technical Director and First Assistant Director preparing a MiniMax-H3 production. In this pass you analyze the user's screenplay against the ASSET REGISTRY and break it down into a numbered sequence of self-contained single-take shots: which registered assets are visible, how the camera behaves, how long the shot runs and what continuity must hold. Later passes turn your breakdown into a shot list and timestamped story beats, so be precise.

# Asset Registry & Generation Order
Every reference asset was generated in this fixed order and is cited by its EXACT NAME: CHARACTERS first, then PROPS, then LOCATIONS.
- Characters and props are reference IMAGES with FIXED picture numbers equal to their position in the registry: the first character is always <Picture 1>, the second always <Picture 2>, and so on through the props. A color-palette reference image follows the last prop.
- Locations are reference VIDEOS with fixed numbers (<Video 1>, <Video 2>, ... in registry order).
- ALL reference files are attached to every shot at all times, so these numbers never change from shot to shot. The `characters`, `props` and `location` fields only say which assets are actually USED in the shot; an asset that is not used is simply never mentioned (its picture number appears nowhere in that shot's brief).

@@REGISTRY@@

# Core Breakdown Rules & Best Practices
1. Universal Exact Entity Naming (CRITICAL): Always use the EXACT, verbatim registry NAME of every character, prop and location. Never use nicknames, truncated names, plurals or aliases (if a character is named "JOHN DOE", never write "John", "Johnny", "Doe" or "the detective" for them).
2. Production Brief Mindset: Treat the generation like a professional production brief. Translate abstract adjectives (e.g., "she looks confident") into explicit, observable physical behaviors (e.g., "she walks steadily, maintaining eye contact, and straightens her jacket").
3. Character Presence: List in `characters` EVERY registered character who is visible or audible on-screen in the shot. A character present in a scene stays listed in every consecutive shot of that scene unless an explicit action line removes them.
4. Prop Selection: List in `props` only registered props that are clearly visible in the frame of this shot. Do not list a prop that is merely mentioned in passing: every listed asset is defined and tracked in that shot's brief, so list only what the camera actually sees.
5. Location: Set `location` to the one registered location where the shot takes place (empty string if none fits).
6. Extended 7-Shot Bidirectional Context & Continuity: For every shot N, evaluate a 7-shot sliding window: the 3 preceding shots (N-3, N-2, N-1) AND the 3 upcoming shots (N+1, N+2, N+3). Ensure visual continuity of lighting, environmental tone, character trajectory, camera vector alignment and action momentum across the whole sequence.
7. Subject Persistence & Positional Locking:
   - Zero Disappearing Characters: any character present in a scene remains present across consecutive shots of that scene. Characters do not vanish when the camera angle changes.
   - Positional & Seating State Preservation: maintain exact postures and object interactions across angle changes. A character seated on a throne stays seated on that exact throne in later shots unless an action line says they stood or moved.
   - Spatial Coordinates: define relative spatial geometry between characters, props and furniture (e.g., "LONA stays on the altar at frame-center while SHADOW KING stands 4 feet to frame-left").
8. Zero Meta-References (CRITICAL): never refer to other shots by number and never write "as seen in Shot 1". In `action` and `continuity`, explicitly state every active character's posture, spatial coordinates and the lighting condition so each shot is unambiguous.
9. One Shot = One Continuous Take: describe no timestamps, cuts or dissolves inside a shot; a cut starts a new shot.
10. On-Screen Text Rule: any text readable on screen (signs, labels, banners, UI) must be enclosed in standard English double quotes inside `action` and enumerated exactly.
11. Spoken Dialogue: record every spoken line verbatim in `action` as `SPEAKER NAME says in <delivery>: "line"`; for narration over a silent on-screen character write `in an off-screen voiceover`. Keep dialogue within about 2.5 words per second of the shot duration. Every spoken line belongs to exactly ONE shot: write it only in the shot where it is delivered and never repeat it in the shots that lead into or react to it.
12. Fixed vs Variable Mapping: begin `continuity` with `FIXED:` (the assets, postures, positions and lighting that must remain consistent, including those carried over from the 3 preceding shots) followed by `VARIABLE:` (what changes in this shot, including what the 3 following shots will expect). Write each half as shot-local state a camera can see (posture, position in frame, wardrobe state, prop in hand, light direction, screen direction) in about 25 words; never restate an asset's face, hair, body or build, because appearance lives in the subject definitions.
13. Camera Language Precision: use formal camera operations (Zoom In/Out, Push In/Pull Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down, Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly/Strongly, POV, Roll Clockwise/Counterclockwise) as [Motion Type] + [Amplitude] + [Speed], plus shot size and angle. For a static shot, explicitly state the framing constraint. Never let a shot repeat the previous shot's shot size AND angle: change at least one of them, either one full size step (e.g. medium shot to close-up) or at least 30 degrees of angle (e.g. eye-level to low angle or over-the-shoulder), and name both in literal, comparable words. Cross the cut on the same side of the action line so screen direction never mirrors. Target aspect ratio is @@ASPECT@@.
14. Dynamic Shot Duration: set `duration_seconds` to an integer of at least @@MIN@@ seconds, proportional to the visual action and spoken dialogue. There is no maximum duration; do not truncate a shot or dialogue to fit a time cap. Target roughly 60 seconds per formatted screenplay page (about 55 screenplay lines), with dialogue paced at a natural speaking rate and holds/actions allowed time as needed.

# Output Format (STRICT)
Output EXACTLY ONE single-line JSON object per shot, in screenplay order, and nothing else:
{"shot": <INTEGER starting at 1>, "location": "<EXACT LOCATION NAME or empty string>", "characters": ["<EXACT NAME>", ...], "props": ["<EXACT NAME>", ...], "camera": "<formal camera operation + shot size + angle>", "source_blocks": [<one or more local source-block IDs>], "duration_seconds": <INTEGER >= @@MIN@@>, "action": "<observable action with spatial placements, verbatim dialogue and quoted on-screen text>", "continuity": "FIXED: <...> VARIABLE: <...>"}
- One object per line, valid JSON, no trailing commas, no markdown code fences, no commentary.
- Source blocks are numbered and tagged ACTION, DIALOGUE, SCENE_HEADING, CHARACTER, PARENTHETICAL, TRANSITION or SHOT_HEADER. Scene headings, character cues and parentheticals provide context; do not create a shot for each such line. Every ACTION block and DIALOGUE block must be covered by one or more shots, with exact spoken dialogue preserved verbatim. Usually use one visual beat per short action paragraph; split only when a paragraph contains distinct sequential visual beats. A shot may cover several consecutive dialogue blocks in the same continuous exchange; dialogue lines do not dictate camera cuts. Use `source_blocks` to list every covered local block ID.
- Complete the whole current screenplay page before moving to the next page. Never omit or truncate script content.
- Never use the vertical bar character anywhere.

CHARACTER DEFINITIONS (user supplied):
@@CHAR_DEFS@@

LOCATION DEFINITIONS (user supplied):
@@LOC_DEFS@@
"""


# ---------------------------------------------------------------- story_beats
STORY_BEATS_PROMPT = """# Role & Objective
You are an expert film editor and MiniMax-H3 prompt engineer. Turn the SHOT BREAKDOWN below into timestamped story beats: exactly ONE line per shot, in the same order. A line is the shot label `[Shot N]` followed by several `[MM:SS.mmm] beat` entries on that same line. Every description of a character, prop or location lives in the shot-list prompt, so a beat NEVER describes an asset's appearance; it only names which asset is on screen.

# Naming
@@REGISTRY@@
Write each asset's EXACT registry NAME in UPPERCASE every time you refer to it; the software replaces names with the right <Subject N> labels. Anything not in the registry is described in plain lowercase words.

# Beat Rules (MiniMax-H3 verbiage)
1. One line per shot, formatted `[Shot N] [00:00.000] first beat [00:01.000] second beat ...`. Timestamps are shot-local (every shot restarts at 00:00), written as `[MM:SS.mmm]`, strictly ascending, and inside that shot's `duration_seconds`. Never write an end timestamp: each beat runs until the next beat or the end of the shot.
2. The first beat of every shot MUST start at `[00:00.000]`.
3. Split the shot into beats of roughly two to three seconds: about one beat per 2-3 seconds of duration (a 5-second shot gets 2-3 beats, a 12-second shot 4-6). Each beat is one observable action clause of roughly 8-20 words in present tense: what an asset does, where it moves, and the intention or feeling that drives it (a breath before a line, a weight shift, a beat of hesitation, a glance). Never re-state a pose an earlier beat already holds: describe the change or the sustained behaviour instead.
4. Beats are micro-direction inside ONE continuous shot: never write timestamps in the prose, never mention shots or cuts, never write "the camera cuts to", and do not repeat camera moves the `camera` field already covers.
5. Name the assets on screen on the first beat of the shot and whenever a beat changes who is acting; a beat that continues the same action does not need to repeat every name. Describe positions as explicit spatial relationships (e.g., "at frame-left"), and keep postures and positions consistent with the neighboring beats.
6. Dialogue: write each spoken line verbatim inside its beat as `SPEAKER NAME says: "exact line"`; for a voiceover write `SPEAKER NAME says in an off-screen voiceover: "exact line"` and, if the speaker is visible on screen, add "while their lips remain completely closed" right after the line. The software converts these to the official speaker-ID syntax. Keep the same speaker spelled the same way every time. Each line is delivered exactly once in the whole sequence: never repeat it in another shot's beats and never anticipate it before its own shot.
7. Any text readable on screen goes in standard English double quotes, spelled exactly.
8. Never refer to other shots by number and never use the vertical bar character.

# Output Format (STRICT)
Exactly one line per shot in the form `[Shot N] [00:00.000] beat [00:01.000] beat`, nothing else (no headers, no markdown, no commentary).

SHOT BREAKDOWN (one JSON object per shot, in order, with its planned duration):
@@BREAKDOWN@@
"""

# ---------------------------------------------------------------- color_palette
PALETTE_STAGE_PROMPT = """You are an expert Colorist and Director of Photography designing the master color palette of a film. Analyze the screenplay and the location profiles and define ONE master palette. It will be rendered as a flat swatch chart image and attached as a color/style reference to every shot, so it must capture the look of the whole film.

CORE RULES:
1. Derive every color from the screenplay's settings, mood, lighting, materials and wardrobe, and from the location profiles below. Be visual and specific: hue, saturation, brightness and undertone, never abstract adjectives.
2. Define 5 to 8 swatches, ordered from the darkest shadow to the brightest highlight, followed by the key accent colors (fire, magic, signal colors, skin and flesh tones if they matter to the look).
3. Name each swatch with a short, simple, evocative English color name (e.g., "obsidian black", "hellfire amber", "necrotic green", "bone white"). No hex codes, no brand names.
4. The palette is colors only: no objects, no people, no text, no textures.
5. Also write ONE grade statement (one or two sentences, visual only): overall contrast, saturation, black level, highlight roll-off, grain and tonal look of the film.

OUTPUT FORMAT (STRICT): output exactly ONE single-line valid JSON object and nothing else (no markdown fences, no commentary, never the vertical bar character):
{"swatches": [{"name": "<color name>", "color": "<precise visual description of the color>", "role": "<what it is used for>"}, ...], "grade": "<grade statement>"}

LOCATION PROFILES (for reference):
@@LOCATIONS@@
"""


# =============================================================================
#  Asset registry  (single source of truth for names and label order)
# =============================================================================
_LOC_PREFIX_RE = re.compile(
    r"^(?:INT\.?\s*/\s*EXT\.?|EXT\.?\s*/\s*INT\.?|INT\.?|EXT\.?|I/E\.?|EST\.?)\s+", re.IGNORECASE)


def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().upper()


def _akey(asset):
    return f"{asset['kind']}:{asset['key']}"


def _short_desc(text, max_words=30):
    words = [w for w in re.sub(r"\s+", " ", str(text or "")).strip().split(" ") if w]
    if len(words) <= max_words:
        return " ".join(words)
    cut = " ".join(words[:max_words])
    # prefer ending on a phrase boundary when one exists in the back half
    idx = max(cut.rfind(";"), cut.rfind(","), cut.rfind("."))
    if idx > len(cut) * 0.5:
        cut = cut[:idx]
    return cut.rstrip(",;:- ")


# Identity-critical order for a character summary. The sheet render, the shot list and the
# story beats all read this one string, so the face and the wardrobe lead and the
# demographics trail: a word budget can then only cut the least visual attributes.
CHARACTER_IDENTITY_FIELDS = (
    ("facial structure", True), ("hair style", True), ("hair color", True),
    ("wardrobe", True), ("wardrobe / primary attire", True), ("distinguishing traits", True),
    ("distinguishable physical traits", True), ("eye shape", False), ("eye color", False),
    ("facial hair", False), ("build", False), ("height", False), ("age", False),
    ("ethnicity", False), ("weight", False),
)

_IDENTITY_LABEL_RE = re.compile(r"(?:^|;[ \t]*)([A-Za-z$][A-Za-z$&' /]{1,40}?)\s*:\s*")


def _identity_fields(text):
    """Labelled character profile ("Facial structure: ...; Wardrobe: ...") -> {label: value}."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    out, matches = {}, list(_IDENTITY_LABEL_RE.finditer(t))
    for index, match in enumerate(matches):
        label = match.group(1).strip().lower()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(t)
        value = t[match.end():end].strip().strip("[]").strip(" .;,")
        if label and value and label not in out:
            out[label] = value
    return out


def _canonical_identity(asset, max_words=60):
    """One fixed, face-first identity sentence per character, reused verbatim everywhere.

    Stage 1 writes the profile as labelled attributes, so this rebuilds the summary in
    identity order instead of truncating the raw profile: front-loaded demographics used
    to eat the word budget before the face, wardrobe and marks were reached. A free-form
    description (characters_json) still works through the phrase-boundary fallback.
    """
    if not isinstance(asset, dict):
        return _short_desc(asset, max_words)
    if asset.get("identity"):
        return _collapse(asset["identity"])
    desc = str(asset.get("description") or "")
    fields = _identity_fields(desc)
    if len(fields) < 3:
        return _short_desc(desc, max_words)
    parts, used = [], 0
    for key, essential in CHARACTER_IDENTITY_FIELDS:
        value = _collapse(fields.get(key, "")).strip(" .;,")
        if not value:
            continue
        budget = max_words - used
        if budget <= 2:
            break
        # Secondary attributes give up their words first so the face and wardrobe stay whole,
        # and keep their label because a bare "58" or "grey" means nothing in a prompt.
        if not essential:
            value = f"{key}: {_short_desc(value, min(budget - 2, 8))}"
        if len(value.split()) > budget:
            value = _short_desc(value, budget)
        if not value:
            continue
        parts.append(value)
        used += len(value.split())
        if used >= max_words - 3:
            break
    return _short_desc("; ".join(parts), max_words) if parts else _short_desc(desc, max_words)


def _asset_summary(asset, max_words=35):
    """Clean visual summary of an asset (no field labels, no narrative-only fields)."""
    if asset["kind"] == KIND_CHARACTER:
        # A character keeps one fixed identity string: it is the sentence every shot
        # repeats, so it must not be cut down to a caller's word budget.
        return _canonical_identity(asset, max(50, int(max_words) + 15))
    d = asset.get("description", "")
    text = ""
    if asset["kind"] == KIND_PROP:
        f = _parse_fields(d, PROP_FIELD_KEYS)
        parts = [f.get(k, "") for k in ("material & texture", "color & visual", "size & scale",
                                        "condition & wear", "distinguishable features")]
        text = "; ".join(p.strip(" .") for p in parts if p.strip(" ."))
    elif asset["kind"] == KIND_LOCATION:
        f = _parse_fields(d, LOCATION_FIELD_KEYS)
        parts = [f.get(k, "") for k in ("style & era", "scale & layout", "color palette", "lighting & atmosphere")]
        text = "; ".join(p.strip(" .") for p in parts if p.strip(" ."))
    if not text.strip():
        text = d
    return _short_desc(text, max_words)


def _collapse(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()

_SPOKEN_LINE_RE = re.compile(r'([A-Z][A-Z0-9 .\'\-]{1,40}?)\s+says(\s+in\s+[^:"]+?)?\s*:\s*"([^"]+)"')
# Spoken lines as they reach the beats: any quote style (straight or curly, single or
# double) and either `says:` or `says,` before the line, e.g. `CINDY says in a soft tone: 'line'`.
_D_SPOKEN_RE = re.compile(
    r'([A-Z][A-Z0-9 .\'\u2019\-]{1,40}?)\s+says\s*'
    r'(in\s+[^:"\u201c\u201d\u2018\u2019]{1,60}?)?\s*[:,]?\s*'
    r'(?:[\u201c"](?P<dq>[^\u201d"]+)[\u201d"]|[\u2018\'](?P<sq>[^\u2019\']+)[\u2019\'])'
)
_BEAT_MARKER_RE = re.compile(
    r"\[\s*(?P<start>\d{1,3}:\d{2}(?::\d{2})?(?:[.,]\d+)?)"
    r"(?:\s*[-\u2013]\s*\d{1,3}:\d{2}(?::\d{2})?(?:[.,]\d+)?)?\s*\]"
)


def _clock_seconds(value):
    """Parse MM:SS[.mmm] or HH:MM:SS[.mmm] into seconds."""
    parts = str(value or "").strip().replace(",", ".").split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid beat timestamp {value!r}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as error:
        raise ValueError(f"Invalid beat timestamp {value!r}") from error
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    hours, minutes, seconds = numbers
    return hours * 3600 + minutes * 60 + seconds


def _parse_beat_entries(text):
    """'[00:00.000] a [00:01.500] b' -> [(0.0, 'a'), (1.5, 'b')]; unmarked text is one beat at zero."""
    text = str(text or "")
    markers = list(_BEAT_MARKER_RE.finditer(text))
    if not markers:
        body = _collapse(text).strip(" .,;:")
        return [(0.0, body)] if body else []
    entries = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        body = text[marker.end():end].strip(" \t\r\n.,;:")
        if body:
            entries.append((_clock_seconds(marker.group("start")), body))
    return entries


def _fit_beats_to_duration(entries, duration_seconds):
    """Clamp beat starts into one shot: first at 00:00.000, strictly ascending, inside the duration."""
    duration = max(0.001, float(duration_seconds))
    fitted = []
    for start, text in entries:
        start = min(max(0.0, float(start)), duration - 0.001)
        if not fitted:
            start = 0.0
        elif start <= fitted[-1][0]:
            start = fitted[-1][0] + 0.001
        if start >= duration:
            continue
        fitted.append((start, text))
    return fitted


def _shot_list_action(shot):
    """Action prose for one shot-list line, without the spoken lines the beats carry."""
    text = str(shot.get("action") or shot.get("desc") or "")
    # Strip a spoken clause only when it opens its sentence, so an all-caps asset
    # name ending the previous sentence ("... in WHEELCHAIR. BLAKE says ...") survives.
    kept = []
    for sentence in re.split(r"(?<=[.!?])[  ]+", text):
        sentence = sentence.strip()
        match = _SPOKEN_LINE_RE.match(sentence)
        if match is None:
            kept.append(sentence)
            continue
        remainder = sentence[match.end():].strip()
        if remainder:
            kept.append(remainder)
    text = re.sub(r"<d>.*?</d>", "", " ".join(kept), flags=re.DOTALL | re.IGNORECASE)
    return _collapse(text).strip(" .,;:")


_QUOTE_CHARS = "\"'\u201c\u201d\u2018\u2019"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _speech_key(text):
    """Quote-free, whitespace-collapsed, casefolded text for comparing spoken lines."""
    value = str(text or "")
    for char in _QUOTE_CHARS:
        value = value.replace(char, " ")
    return _collapse(value).casefold()


def _quotes_line(text, line):
    """True when `text` already states `line` verbatim, whatever quotes wrap it."""
    key = _speech_key(line).strip(" .,;:!?")
    return bool(key) and key in _speech_key(text)


def _drop_spoken_line(text, line):
    """Remove the sentence that quotes `line`; the other sentences of the shot survive."""
    if not _quotes_line(text, line):
        return text
    kept = [sentence for sentence in _SENTENCE_SPLIT_RE.split(str(text or "").strip())
            if sentence and not _quotes_line(sentence, line)]
    return _collapse(" ".join(kept)).strip()


def _synthesized_dialogue(block):
    """A DIALOGUE source block -> `SPEAKER says in <delivery>: "line"` in the official wording."""
    line = _collapse(block.get("text")).strip()
    if not line:
        return ""
    speaker = re.sub(r"\s*\(.*?\)", "", str(block.get("speaker", ""))).strip().upper()
    if not speaker:
        return f'"{line}"'
    prefix = f"{speaker} says in an off-screen voiceover" if block.get("voiceover") else f"{speaker} says"
    return f'{prefix}: "{line}"'


def _resolve_shot_coverage(page_shots, source_page):
    """Give every ACTION and DIALOGUE block exactly one shot, and drop repeated prose.

    * A spoken line stays in the FIRST shot whose action already quotes it; later shots
      that quote the same line lose that clause, so the line is delivered exactly once.
    * A line no shot quotes is appended verbatim to the LAST shot that claims its block,
      so overlapping `source_blocks` can never make two shots say the same line.
    * A shot whose remaining action repeats an earlier shot's action is dropped, so an
      action block is staged once instead of twice.

    Returns the kept shots in source order and the set of covered source-block IDs.
    """
    ordered = [shot for _, shot in sorted(
        enumerate(page_shots), key=lambda pair: (min(pair[1]["_content_ids"] or [0]), pair[0]))]

    covered = set()
    for block_id, block in enumerate(source_page, 1):
        if block["type"] != "DIALOGUE":
            continue
        quoters = [shot for shot in ordered if _quotes_line(shot.get("action"), block["text"])]
        if quoters:
            covered.add(block_id)
            for duplicate in quoters[1:]:
                duplicate["action"] = _drop_spoken_line(duplicate["action"], block["text"])
            continue
        claimants = [shot for shot in ordered if block_id in shot["_content_ids"]]
        if claimants:
            clause = _synthesized_dialogue(block)
            if clause:
                claimants[-1]["action"] = f'{claimants[-1]["action"]} {clause}'.strip()
            covered.add(block_id)

    kept, seen = [], set()
    for shot in ordered:
        key = _collapse(shot.get("action")).casefold()
        if not key or key in seen:
            covered.update(shot["_block_ids"])
            continue
        seen.add(key)
        kept.append(shot)
        covered.update(shot["_block_ids"])
    return kept, covered


_CONTINUITY_FIXED_RE = re.compile(r"(?is)\bfixed\s*:\s*(.*?)(?=\bvariable\s*:|$)")


def _continuity_clause(shot, max_words=28):
    """The `FIXED:` half of a shot's continuity mapping, trimmed for a shot-list line.

    The breakdown pass already works out what has to survive the cut (posture, wardrobe
    state, prop in hand, light direction, screen direction); that text used to be
    computed and then never sent to the model.
    """
    text = _collapse(shot.get("continuity") or "")
    match = _CONTINUITY_FIXED_RE.search(text)
    if match:
        text = match.group(1)
    elif re.search(r"(?i)\bvariable\s*:", text):
        return ""
    text = text.strip(" .;,")
    if not text or text.lower().startswith("preserve the current scene state"):
        return ""
    return _short_desc(text, max_words)


def _shot_prompt_line(number, camera, action, continuity=""):
    """'[Shot N] <camera phrase> as <action>' — one shot-list line in MiniMax-H3 phrasing."""
    body = _lower_first(_collapse(action).strip(" ."))
    cam = _collapse(camera).strip(" .")
    if not cam:
        line = f"[Shot {number}] {body}."
    else:
        cam = _lower_first(cam)
        if re.match(r"^(?:the\s+camera|the\s+shot|camera)\b", cam):
            lead = cam[0].upper() + cam[1:]
            line = f"[Shot {number}] {lead}. {body[0].upper() + body[1:]}." if body else f"[Shot {number}] {lead}."
        else:
            lead = cam if re.match(r"^(?:a|an|the)\b", cam) else f"{'an' if cam[0].lower() in 'aeiou' else 'a'} {cam}"
            line = f"[Shot {number}] {lead} as {body}." if body else f"[Shot {number}] {lead}."
    keep = _collapse(continuity).strip(" .;,")
    if keep:
        line += f" Continuity holds: {_lower_first(keep)}."
    return line


def _definition_line(asset, label, traits):
    """One '<Subject n> is NAME in <Picture|Video n>, traits.' line of subject_definitions."""
    traits = str(traits or "").strip(" .")
    if asset["kind"] == KIND_LOCATION:
        # written without the INT./EXT. prefix so name-based image filters do not match a location
        loc_name = asset["aliases"][0] if asset["aliases"] else asset["name"]
        head = f"{label['subject']} is the {loc_name} environment in {label['source']}"
    else:
        head = f"{label['subject']} is {asset['name']} in {label['source']}"
    line = f"{head}, {traits}." if traits else f"{head}."
    if asset["kind"] == KIND_CHARACTER:
        line += " " + CHARACTER_NOTE
    return line


class _Registry:
    """Ordered asset registry: characters -> props -> locations (insertion order inside each kind)."""

    def __init__(self):
        self._assets = []
        self._seq = 0
        self.palette = None  # {"swatches": [{"name","color","role"}], "grade": str}

    def picture_numbers(self):
        """FIXED <Picture N> positions (all references are always connected): characters, then props,
        in registry order; the color palette image follows the last prop (or leads, see PALETTE_AT_START)."""
        imgs = self.ordered({KIND_CHARACTER, KIND_PROP})
        offset = 1 if (self.palette and PALETTE_AT_START) else 0
        nums = {_akey(a): i + 1 + offset for i, a in enumerate(imgs)}
        palette_no = (1 if PALETTE_AT_START else len(imgs) + 1) if self.palette else None
        return nums, palette_no

    def video_numbers(self):
        """FIXED <Video N> positions: locations in registry order."""
        return {_akey(a): i + 1 for i, a in enumerate(self.ordered({KIND_LOCATION}))}

    @staticmethod
    def _key(kind, name):
        n = _norm(name)
        if kind == KIND_LOCATION:
            n = _norm(_LOC_PREFIX_RE.sub("", n))
        return n

    def add(self, kind, name, description="", display=None):
        name_n = _norm(name)
        if not name_n:
            return None
        key = self._key(kind, name_n)
        for a in self._assets:
            if a["kind"] == kind and a["key"] == key:
                if description and not a["description"]:
                    a["description"] = str(description).strip()
                return a
        aliases = []
        if kind == KIND_LOCATION:
            short = _norm(_LOC_PREFIX_RE.sub("", name_n))
            if short and short != name_n:
                aliases.append(short)
        asset = {
            "kind": kind, "name": name_n, "key": key,
            "display": (display or name).strip() if (display or name) else name_n,
            "description": str(description or "").strip(),
            "aliases": aliases, "seq": self._seq,
        }
        self._seq += 1
        self._assets.append(asset)
        return asset

    def ordered(self, kinds=None):
        items = [a for a in self._assets if kinds is None or a["kind"] in kinds]
        return sorted(items, key=lambda a: (KIND_ORDER[a["kind"]], a["seq"]))

    def names(self, kinds=None):
        out = []
        for a in self.ordered(kinds):
            if a["name"] not in out:
                out.append(a["name"])
        return out

    def lookup(self, name, kinds=None):
        n = _norm(name)
        if not n:
            return None
        pool = self.ordered(kinds)
        for a in pool:
            if n == a["name"] or n in [_norm(x) for x in a["aliases"]]:
                return a
        loose = [a for a in pool if n in a["name"] or a["name"] in n]
        return loose[0] if len(loose) == 1 else None

    def collisions(self):
        """Names contained in other names confuse substring-based reference filters."""
        out = []
        pool = self.ordered()
        for a in pool:
            for b in pool:
                if a is not b and a["name"] != b["name"] and a["name"] in b["name"]:
                    out.append((a["name"], b["name"]))
        return out


def _compile_matcher(assets):
    variants = []
    for a in assets:
        for v in [a["name"]] + list(a["aliases"]):
            variants.append((v, a))
    if not variants:
        return None, {}
    variants.sort(key=lambda t: len(t[0]), reverse=True)
    lookup, parts = {}, []
    for v, a in variants:
        lookup.setdefault(_norm(v), a)
        parts.append(r"\s+".join(re.escape(w) for w in v.split()))
    rx = re.compile(r"(?<![A-Za-z0-9])(?:%s)(?![A-Za-z0-9])" % "|".join(parts), re.IGNORECASE)
    return rx, lookup


def _registry_listing(registry, kinds=None, words=45, with_palette=False):
    nums, pal_no = registry.picture_numbers()
    vids = registry.video_numbers()
    groups = [
        (KIND_CHARACTER, "CHARACTERS (reference images, fixed <Picture N>)"),
        (KIND_PROP, "PROPS (reference images, fixed <Picture N>)"),
        (KIND_LOCATION, "LOCATIONS (reference videos, fixed <Video N>)"),
    ]
    out = ["ASSET REGISTRY (global order: characters, then props, then locations; every reference is always attached):"]
    for kind, title in groups:
        if kinds and kind not in kinds:
            continue
        out.append(title + ":")
        items = registry.ordered({kind})
        if not items:
            out.append("  (none)")
        for a in items:
            tag = f"<Video {vids[_akey(a)]}>" if kind == KIND_LOCATION else f"<Picture {nums[_akey(a)]}>"
            d = _asset_summary(a, words)
            out.append(f"  - {a['name']} [{tag}]" + (f": {d}" if d else ""))
    if with_palette and pal_no:
        out.append(f"COLOR PALETTE REFERENCE (reference image <Picture {pal_no}>, always attached, handled automatically): "
                   + ", ".join(s["name"] for s in registry.palette["swatches"]))
    return "\n".join(out)


def _label_preview(registry, assets, include_palette=False):
    """Label table for a set of used assets: Subject numbers per prompt, Picture/Video numbers fixed."""
    nums, pal_no = registry.picture_numbers()
    vids = registry.video_numbers()
    ordered = sorted(assets, key=lambda a: (KIND_ORDER[a["kind"]], a["seq"]))
    bits = []
    for n, a in enumerate(ordered, 1):
        src = f"<Video {vids[_akey(a)]}>" if a["kind"] == KIND_LOCATION else f"<Picture {nums[_akey(a)]}>"
        bits.append(f"{a['name']} = <Subject {n}>/{src}")
    if include_palette and pal_no:
        bits.append(f"COLOR PALETTE = <Subject {len(ordered) + 1}>/<Picture {pal_no}>")
    return "; ".join(bits)


# =============================================================================
#  Text utilities
# =============================================================================
_DIALOGUE_RE = re.compile(r"(<d>.*?</d>)", re.DOTALL | re.IGNORECASE)
_SUBJ_RE = re.compile(r"<Subject\s+(\d+)>", re.IGNORECASE)
_SRC_RE = re.compile(r"<(?:Picture|Video|Audio)\s+\d+>", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]+|\d+[.)])\s*")
_MARKER_RE = re.compile(r"\b(%s)\b" % "|".join(RELATIONSHIP_MARKERS), re.IGNORECASE)
_SHOT1_RE = re.compile(r"\[\s*Shot\s*1\s*\]", re.IGNORECASE)
_SECTION_KEYS = ["subject_definitions", "summary", "retention_analysis",
                 "detailed_description", "overall_soundscape", "non_diegetic_music"]
_SECTION_RE = re.compile(
    r"(?im)^[ \t>*#_`\-]*(?:\d+[.)][ \t]*)?(subject[ _-]*definitions|summary|retention[ _-]*analysis|"
    r"detailed[ _-]*description|integrated[ _-]*multimodal[ _-]*description|overall[ _-]*soundscape|"
    r"non[ _-]*diegetic[ _-]*music)[ \t*_`]*(?::[ \t*_`]*|[ \t]*(?=$))")


def _map_outside_dialogue(text, fn):
    parts = _DIALOGUE_RE.split(text or "")
    return "".join(p if i % 2 else fn(p) for i, p in enumerate(parts))


def _strip_fences(text):
    t = str(text or "")
    t = re.sub(r"```[a-zA-Z]*", "", t)
    return t.replace("```", "").strip()


def _clean_block(text):
    t = str(text or "").replace("\r", "")
    t = t.replace("**", "").replace("|", "-")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _parse_sections(text):
    t = _strip_fences(text)
    matches = list(_SECTION_RE.finditer(t))
    out = {}
    for i, m in enumerate(matches):
        name = re.sub(r"[ _-]+", "_", m.group(1).lower())
        if name == "integrated_multimodal_description":
            name = "detailed_description"
        end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
        out.setdefault(name, t[m.end():end].strip())
    return out


def _parse_fields(text, keys):
    """'Label: value - Label: value' text -> {lowercase label: value} (tolerates unknown labels and '$' typos)."""
    t = str(text or "").strip()
    wanted = {k.lower() for k in keys}
    out = {}
    # segment on ' - ' only where a 'Label:' follows, so hyphens inside values are kept
    segments = re.split(r"\s[-\u2013\u2014]\s+(?=[A-Z$][A-Za-z$&' /]{1,40}:\s)", t)
    for seg in segments:
        m = re.match(r"^\s*([A-Za-z$][A-Za-z$&' /]{1,40}?)\s*:\s*(.*)$", seg, re.DOTALL)
        if not m:
            continue
        label = m.group(1).replace("$", "s").strip().lower()
        val = m.group(2).strip().rstrip("|").strip()
        val = re.sub(r"\s*[-\u2013\u2014]\s*$", "", val).strip()
        if label in wanted:
            out.setdefault(label, val)
    if out:
        return out
    # fallback: scan for the known keys anywhere in the text
    alt = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
    rx = re.compile(r"(?i)(?<![A-Za-z])(%s)\s*:\s*" % alt)
    ms = list(rx.finditer(" " + t))
    t2 = " " + t
    for i, m in enumerate(ms):
        end_i = ms[i + 1].start() if i + 1 < len(ms) else len(t2)
        val = t2[m.end():end_i].strip().rstrip("|").strip()
        val = re.sub(r"\s*[-\u2013\u2014]\s*$", "", val).strip()
        out.setdefault(m.group(1).lower(), val)
    return out


def _parse_header_entries(text, keys):
    """'Name: X - description |' style entries -> [(NAME, description), ...] (names uppercased)."""
    out = []
    if not text:
        return out
    alt = "|".join(re.escape(k) for k in keys)
    for chunk in _strip_fences(text).split("|"):
        c = chunk.strip()
        if not c:
            continue
        m = re.match(r"(?is)^\W*(?:%s)\s*:\s*([^\n]+?)\s+[-\u2013\u2014]\s+(.*)$" % alt, c)
        if m:
            name, desc = m.group(1), m.group(2)
        else:
            m2 = re.match(r"(?is)^\W*(?:%s)\s*:\s*([^-\n\u2013\u2014]+)[-\u2013\u2014:]?\s*(.*)$" % alt, c)
            if not m2:
                continue
            name, desc = m2.group(1), m2.group(2)
        name = name.strip().strip("*").strip()
        if name:
            out.append((name, desc.strip()))
    return out


def _parse_json_objects(text):
    t = _strip_fences(text)
    t = re.sub(r",\s*([}\]])", r"\1", t)
    dec = json.JSONDecoder()
    i, out = 0, []
    while True:
        j = t.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(t[j:])
            if isinstance(obj, dict):
                out.append(obj)
            i = j + end
        except Exception:
            i = j + 1
    return out


def _with_trigger(text, trigger_words):
    tw = (trigger_words or "").strip().rstrip(".")
    text = (text or "").strip()
    if not tw:
        return text
    return f"{tw}. {text}".strip()


# =============================================================================
#  Label resolution
# =============================================================================
def _scan_mentions(text, rx, lookup, found, exclude=None):
    if rx is None or not text:
        return

    def scan(p):
        for m in rx.finditer(p):
            a = lookup.get(_norm(m.group(0)))
            if a and (exclude is None or _akey(a) != exclude):
                found[_akey(a)] = a
        return p

    _map_outside_dialogue(text, scan)


def _apply_labels(text, rx, lookup, labels, skip_key=None):
    if rx is None or not text:
        return text or ""

    def rep(m):
        a = lookup.get(_norm(m.group(0)))
        if a and _akey(a) in labels and _akey(a) != skip_key:
            return labels[_akey(a)]["subject"]
        return m.group(0)

    return _map_outside_dialogue(text, lambda p: rx.sub(rep, p))


def _parse_asset_lines(block, rx, lookup):
    """{asset_key: (asset, text_after_name)} for lines whose FIRST asset mention identifies the line's asset."""
    out = {}
    if not block or rx is None:
        return out
    for raw in block.splitlines():
        line = _BULLET_RE.sub("", raw.strip())
        if not line or line.lower().strip(". ") in ("none", "n/a"):
            continue
        # Generated definitions are formatted as NAME: traits. Resolve the leading
        # name before scanning the traits, which may mention another asset name.
        candidate = line.split(":", 1)[0].strip()
        a = lookup.get(_norm(candidate))
        if a is None:
            m = rx.search(line)
            if not m:
                continue
            a = lookup.get(_norm(m.group(0)))
            if a is None:
                continue
            rest = line[m.end():]
        else:
            rest = line[len(candidate):]
        if _akey(a) not in out:
            out[_akey(a)] = (a, rest)
    return out


def _clean_traits(rest):
    t = (rest or "").strip()
    t = re.sub(r"^\s*(?:\([^)]*\))?\s*(?:[:\-\u2013\u2014]|\bis\b|\bare\b)\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*(?:the\s+)?environment\b\s*[:,\-]?\s*", "", t, flags=re.IGNORECASE)
    t = _SRC_RE.sub("", t)
    t = re.sub(r"\b(?:in|from)\s*[,;:]\s*", "", t)
    return _collapse(t).strip(" .,;:-")


def _clean_retention(rest):
    r = rest or ""
    m = _MARKER_RE.search(r)
    marker = m.group(1).lower() if m else "fully_preserved"
    reason = r[m.end():] if m else r
    reason = re.sub(r"^\s*[-\u2013\u2014:]+\s*", "", reason)
    if not m:
        reason = re.sub(r"^\s*\([^)]*\)\s*[:\-]?\s*", "", reason)
    return marker, _collapse(reason).strip(" .,;:-")


def _demote_labels(sec, rx, lookup):
    """If the model wrote <Subject N> numbers anyway, turn them back into asset names so numbering can be recomputed."""
    sec = dict(sec)
    if rx is None:
        return sec
    label_to_name = {}
    new_lines = []
    for line in sec.get("subject_definitions", "").splitlines():
        m = _SUBJ_RE.search(line)
        if m:
            mm = rx.search(line[m.end():])
            if mm:
                label_to_name[m.group(1)] = lookup[_norm(mm.group(0))]["name"]
        n = re.match(
            r"^\s*(?:[-*]\s*)?<Subject\s+\d+>\s*(?:is|=|:)?\s*(?:the\s+)?(.+?)\s+in\s+<(?:Picture|Video|Audio)\s+\d+>\s*[,:;]?\s*(.*)$",
            line, re.IGNORECASE)
        new_lines.append(f"{n.group(1)}: {n.group(2)}" if n else line)
    sec["subject_definitions"] = "\n".join(new_lines)
    if label_to_name:
        for k in list(sec):
            sec[k] = _SUBJ_RE.sub(lambda m: label_to_name.get(m.group(1), m.group(0)), sec[k])
    return sec


def _lower_first(text):
    t = text.strip()
    if not t:
        return t
    first = t.split(" ", 1)[0]
    if first in ("A", "An", "The") or (len(first) > 1 and first[0].isupper() and first[1:].islower()):
        return t[0].lower() + t[1:]
    return t


def _to_t2va_text(style, body, sound, music, trigger_words=""):
    inner = " ".join(p.strip() for p in (style, body) if p and p.strip())
    inner = _with_trigger(inner, trigger_words)
    return ("integrated_multimodal_description: [Shot 1] " + inner + "\n\n"
            "overall_soundscape: " + sound + "\n\n"
            "non_diegetic_music: " + music)


_NO_PEOPLE_RE = re.compile(r"\b(?:no|without)\s+(?:humans?|people|persons?)\b|\b(?:void|empty)\s+of\s+(?:all\s+)?people\b",
                           re.IGNORECASE)


def _assemble_reference_prompt(sections, registry, allowed_kinds, trigger_words="", still=False,
                               no_people=False, default_soundscape="Natural ambient sound that fits the scene.",
                               include_palette=False, force_assets=None):
    """
    Turn a name-based six-section draft into the final MiniMax-H3 prompt.

    * Which assets are used is decided by which registry NAMES appear in the draft.
    * <Subject N> / <Picture N> / <Video N> are assigned HERE (never by the LLM): registry order is
      characters -> props -> locations; characters+props are pictures, locations are videos.
    * With zero used assets the prompt is emitted in the base (no-reference) T2VA format instead.
    Returns dict(text, used, labels, warnings).
    """
    warnings = []
    assets = registry.ordered(allowed_kinds)
    rx, lookup = _compile_matcher(assets)

    sec = {k: _clean_block(sections.get(k, "")) for k in _SECTION_KEYS}
    sec = _demote_labels(sec, rx, lookup)
    for k in sec:
        sec[k] = _SUBJ_RE.sub("", _SRC_RE.sub("", sec[k]))

    defs = _parse_asset_lines(sec["subject_definitions"], rx, lookup)
    rets = _parse_asset_lines(sec["retention_analysis"], rx, lookup)

    # --- which assets are really used (a line's own leading name does not count as a use)
    found = {}
    for text in (sec["summary"], sec["detailed_description"], sec["overall_soundscape"], sec["non_diegetic_music"]):
        _scan_mentions(text, rx, lookup, found)
    for key, (a, rest) in list(defs.items()) + list(rets.items()):
        _scan_mentions(rest, rx, lookup, found, exclude=key)
    force_keys = {_akey(a) for a in (force_assets or [])}
    used = [a for a in assets if _akey(a) in found or _akey(a) in force_keys]
    for key, (a, _) in defs.items():
        if _akey(a) not in found and _akey(a) not in force_keys:
            warnings.append(f"{a['name']} was defined but never used in the shot; dropped")

    # --- labels: Subject = per-prompt order (characters, props, locations, palette); Picture/Video = FIXED
    # positions in the global registry, because every reference is always connected
    nums, pal_no = registry.picture_numbers()
    vids = registry.video_numbers()
    labels = {}
    for n, a in enumerate(used, 1):
        src = f"<Video {vids[_akey(a)]}>" if a["kind"] == KIND_LOCATION else f"<Picture {nums[_akey(a)]}>"
        labels[_akey(a)] = {"subject": f"<Subject {n}>", "source": src, "asset": a}
    palette = registry.palette if (include_palette and registry.palette and pal_no) else None
    pal_label = ({"subject": f"<Subject {len(used) + 1}>", "source": f"<Picture {pal_no}>"} if palette else None)

    def L(t):
        return _apply_labels(t, rx, lookup, labels)

    # --- detailed description (style opening, single [Shot 1])
    dd = L(sec["detailed_description"])
    m = _SHOT1_RE.search(dd)
    if m:
        style, body = _collapse(dd[:m.start()]), _collapse(dd[m.end():])
    else:
        style, body = "", _collapse(dd)
        warnings.append("detailed_description had no [Shot 1] marker; one was added")
    if no_people and not _NO_PEOPLE_RE.search(body):
        body = f"{body} {NO_PEOPLE_SENTENCE}".strip()
    if still and "frozen still frame" not in body.lower():
        body = f"{body} {STILL_FRAME_SENTENCE}".strip()

    sound = _collapse(L(sec["overall_soundscape"])) or default_soundscape
    music = _collapse(L(sec["non_diegetic_music"]))
    if not music or music.lower().strip(". ") in ("none", "n/a", "na"):
        music = "N/A"

    # --- no references -> base T2VA format
    if not used and not palette:
        return {"text": _to_t2va_text(style, body, sound, music, trigger_words),
                "used": [], "labels": labels, "warnings": warnings}

    # --- subject_definitions
    def_lines = []
    for a in used:
        traits = ""
        if _akey(a) in defs:
            traits = _apply_labels(_clean_traits(defs[_akey(a)][1]), rx, lookup, labels, skip_key=_akey(a))
        if not traits:
            traits = _asset_summary(a, 30)
            if _akey(a) not in defs:
                warnings.append(f"{a['name']} had no definition line; used its registry description")
        def_lines.append(_definition_line(a, labels[_akey(a)], _lower_first(traits).strip(" .")))

    # --- retention_analysis
    ret_lines = []
    for a in used:
        lb = labels[_akey(a)]
        marker, reason = ("fully_preserved", "")
        if _akey(a) in rets:
            marker, reason = _clean_retention(rets[_akey(a)][1])
            reason = L(reason)
        if not reason:
            reason = "the referenced appearance is retained"
        ret_lines.append(f"{lb['subject']} (appears in [Shot 1]): {marker} - {_lower_first(reason).strip(' .')}.")

    if palette:
        names = ", ".join(s["name"] for s in palette["swatches"])
        pdef = (f"{pal_label['subject']} is the color palette reference in {pal_label['source']}, a single row of "
                f"{len(palette['swatches'])} flat color swatches ({names}).")
        grade = (palette.get("grade") or "").strip().rstrip(".")
        if grade:
            pdef += f" {grade}."
        pdef += " Only its colors and tonal look are used, never the swatch layout."
        def_lines.append(pdef)
        ret_lines.append(f"{pal_label['subject']} (appears in [Shot 1]): attribute_transfer - its colors, contrast and tonal grade "
                         f"are transferred to the lighting, set, wardrobe and skin tones across the entire frame; the swatch layout never appears.")

    # --- summary
    summary = _collapse(L(sec["summary"]))
    summary = re.sub(r"^\s*\[[^\]]*\]\s*", "", summary)
    if not summary:
        summary = "A single continuous shot built from " + ", ".join(labels[_akey(a)]["subject"] for a in used) + "."
    if palette:
        summary = f"{summary} {pal_label['subject']} sets the color palette for the whole frame.".strip()
    summary = "[reference generation] " + summary

    style = _with_trigger(style, trigger_words)
    if palette:
        style = ((style + " ") if style else "") + f"The color palette, contrast and grade of the whole frame follow {pal_label['subject']}."
    dd_text = (style + "\n" if style else "") + "[Shot 1] " + body

    text = "\n\n".join([
        "subject_definitions:\n" + "\n".join(def_lines),
        "summary:\n" + summary,
        "retention_analysis:\n" + "\n".join(ret_lines),
        "detailed_description:\n" + dd_text,
        "overall_soundscape:\n" + sound,
        "non_diegetic_music:\n" + music,
    ])
    return {"text": text, "used": used, "labels": labels, "warnings": warnings}


# ---------------------------------------------------- camera continuity (no jump cuts)
# Two neighbouring shots that keep BOTH the same shot size and the same camera angle read as
# a jump cut: every shot is generated from the same style block and the same subject
# definitions, so identical camera text renders nearly identical frames. The breakdown model
# writes each `camera` field on its own, so nothing stops it from repeating a setup;
# _vary_camera() is the deterministic backstop that runs before the shot list is assembled.
_CAMERA_SIZES = ("extreme wide", "wide", "full", "medium wide", "medium",
                 "medium close", "close", "extreme close")
_CAMERA_SIZE_PATTERNS = (
    ("extreme wide", r"extreme[\s-]+wide(?:[\s-]+shot)?|\bE\.?W\.?S\.?\b|aerial\s+(?:shot|view)"),
    ("extreme close", r"extreme[\s-]+close(?:[\s-]*up)?|\bE\.?C\.?U\.?\b|macro\s+shot"),
    ("medium close", r"medium[\s-]+close(?:[\s-]*up)?|\bM\.?C\.?U\.?\b"),
    ("medium wide", r"medium[\s-]+wide(?:[\s-]+shot)?"),
    ("close", r"close(?:[\s-]*up)?|\bC\.?U\.?\b|tight\s+(?:shot|framing|on)\b|insert\s+shot|detail\s+shot"),
    ("full", r"full[\s-]+(?:body\s+)?shot|cowboy\s+shot"),
    ("wide", r"(?:wide|long|establishing)[\s-]+(?:angle\s+)?shot|\bwide\b"),
    ("medium", r"medium[\s-]+(?:shot|two[\s-]?shot)|\bmid[\s-]+shot\b|\bmedium\b"),
)
_CAMERA_ANGLE_PATTERNS = (
    ("low angle", r"low[\s-]+angle|worm'?s[\s-]*eye|from\s+below|looking\s+up\b"),
    ("high angle", r"high[\s-]+angle|top[\s-]+down|overhead|from\s+above|looking\s+down\b"),
    ("over-the-shoulder", r"over[\s-]+the[\s-]+shoulder|\bO\.?T\.?S\.?\b"),
    ("dutch angle", r"dutch[\s-]+(?:angle|tilt)|canted\s+(?:framing|angle|horizon)"),
    ("eye-level", r"eye[\s-]*level|at\s+eye\s+height|straight[\s-]+on|dead[\s-]+on"),
)
_CAMERA_SIZE_TEXT = {
    "extreme wide": "extreme wide shot", "wide": "wide shot", "full": "full shot",
    "medium wide": "medium wide shot", "medium": "medium shot",
    "medium close": "medium close-up", "close": "close-up", "extreme close": "extreme close-up",
}
# Inline wording replaces an existing phrase; the tail wording is appended to a camera line
# that never named an angle, so both read as normal camera language.
_CAMERA_ANGLE_TEXT = {"eye-level": "eye-level", "low angle": "low-angle", "high angle": "high-angle",
                      "over-the-shoulder": "over-the-shoulder", "dutch angle": "Dutch-angle"}
_CAMERA_ANGLE_TAIL = {"eye-level": "at eye level", "low angle": "from a low angle",
                      "high angle": "from a high angle", "over-the-shoulder": "over-the-shoulder",
                      "dutch angle": "at a Dutch angle"}
# Neutral reframes only: they change the angle without crossing to the other side of the
# action line, so screen direction never mirrors across the cut.
_CAMERA_ANGLE_ALTERNATES = {
    "eye-level": ("low angle", "high angle", "over-the-shoulder"),
    "low angle": ("eye-level", "high angle", "over-the-shoulder"),
    "high angle": ("eye-level", "low angle", "over-the-shoulder"),
    "over-the-shoulder": ("low angle", "eye-level", "high angle"),
    "dutch angle": ("eye-level", "low angle", "high angle"),
}
_CAMERA_TIGHT_WORDS = re.compile(
    r"\b(?:hands?|fingers?|thumbs?|eyes?|eyelids?|face|lips?|mouth|jaw|tears?|smiles?|frowns?|"
    r"knuckles?|grip|ring|coin|needle|blade|screen|photograph|photo|letter|key|bullets?|blood|"
    r"wounds?|scars?|tattoos?|symbol|whispers?|kiss)\b")
_CAMERA_WIDE_WORDS = re.compile(
    r"\b(?:rooms?|halls?|hallways?|corridors?|streets?|alleys?|fields?|valleys?|desert|cities|city|"
    r"skyline|sky|crowds?|army|forest|hills?|ocean|sea|river|horizon|landscape|warehouse|rooftop|"
    r"buildings?|arena|stadium|stage|walkway|platform|bridges?|squares?|marketplace)\b")


def _camera_size(camera):
    """(canonical shot size, match) for a camera phrase; medium when it names no size."""
    for name, pattern in _CAMERA_SIZE_PATTERNS:
        found = re.search(pattern, camera, re.IGNORECASE)
        if found:
            return name, found
    return "medium", None


def _camera_angle(camera):
    """(canonical camera angle, match) for a camera phrase; eye-level when none is named."""
    for name, pattern in _CAMERA_ANGLE_PATTERNS:
        found = re.search(pattern, camera, re.IGNORECASE)
        if found:
            return name, found
    return "eye-level", None


def _camera_fits_action(shot, size):
    """False when the shot's own action text rules this framing out."""
    action = str(shot.get("action") or shot.get("desc") or "").lower()
    tight = bool(_CAMERA_TIGHT_WORDS.search(action))
    wide = bool(_CAMERA_WIDE_WORDS.search(action))
    index = _CAMERA_SIZES.index(size)
    if index >= _CAMERA_SIZES.index("medium close") and wide and not tight:
        return False
    if index <= _CAMERA_SIZES.index("full") and tight and not wide:
        return False
    return True


def _camera_step(size, direction):
    """Next rung on the size ladder, reversing direction at either end."""
    index = _CAMERA_SIZES.index(size) + direction
    if 0 <= index < len(_CAMERA_SIZES):
        return _CAMERA_SIZES[index], direction
    return _CAMERA_SIZES[_CAMERA_SIZES.index(size) - direction], -direction


def _match_case(replacement, original):
    return replacement[:1].upper() + replacement[1:] if original[:1].isupper() else replacement


def _reframe_camera(camera, size_match, angle_match, new_size, new_angle):
    """Rewrite only the size and/or angle wording; motion, amplitude and speed stay intact."""
    text = camera.strip()
    edits, tails = [], []
    if new_size:
        if size_match is not None:
            edits.append((size_match.span(), _match_case(_CAMERA_SIZE_TEXT[new_size], size_match.group())))
        else:
            tails.append(_CAMERA_SIZE_TEXT[new_size])
    if new_angle:
        if angle_match is not None:
            edits.append((angle_match.span(), _match_case(_CAMERA_ANGLE_TEXT[new_angle], angle_match.group())))
        else:
            tails.append(_CAMERA_ANGLE_TAIL[new_angle])
    if not edits and not tails:
        return camera
    for (start, end), replacement in sorted(edits, key=lambda edit: edit[0][0], reverse=True):
        text = text[:start] + replacement + text[end:]
    body = text.rstrip(" .,;:")
    if not body:
        return camera
    if tails:
        body = f"{body}, {', '.join(tails)}"
    return body + "."


def _vary_camera(shots, warnings=None):
    """Keep consecutive shots from sharing both a shot size and a camera angle.

    A cut that keeps the framing and the angle unchanged reads as a jump cut. When two
    neighbours agree on both, one of them moves: the angle when the action fixes the size,
    otherwise one rung on the size ladder. The rewrite happens in place so the shot list, the
    story beats and the logs all describe the same framing. Returns the same shot list.
    """
    previous, previous_number, direction = None, None, 1
    for shot in shots:
        camera = _collapse(shot.get("camera") or "")
        if not camera:                      # no camera text: nothing to preserve or repeat
            previous, previous_number = None, None
            continue
        size, size_match = _camera_size(camera)
        angle, angle_match = _camera_angle(camera)
        if previous is not None and (size, angle) == previous:
            stepped, next_direction = _camera_step(size, direction)
            new_size = new_angle = None
            if _camera_fits_action(shot, stepped):
                new_size, direction = stepped, next_direction
            else:
                new_angle = _CAMERA_ANGLE_ALTERNATES[angle][0]
            reframed = _reframe_camera(camera, size_match, angle_match, new_size, new_angle)
            shot["camera"] = reframed
            size, size_match = _camera_size(reframed)
            angle, angle_match = _camera_angle(reframed)
            if warnings is not None:
                warnings.append(
                    f"shot {shot.get('shot', '?')} repeated the framing of shot {previous_number} "
                    f"({previous[0]}, {previous[1]}); reframed to {size}, {angle}")
        previous, previous_number = (size, angle), shot.get("shot")
    return shots


def _continuation_camera(previous_camera):
    """Camera text for a shot the breakdown had to synthesize: step away from its neighbour."""
    previous_camera = _collapse(previous_camera)
    if not previous_camera:
        return "Static shot, medium shot, at eye level."
    size, _ = _camera_size(previous_camera)
    angle, _ = _camera_angle(previous_camera)
    stepped, _ = _camera_step(size, 1)
    return f"Static shot, {_CAMERA_SIZE_TEXT[stepped]}, {_CAMERA_ANGLE_TAIL[angle]}."


def _assemble_shot_list(shots, registry, trigger_words="", style=SHOT_LIST_STYLE,
                        soundscape=SHOT_LIST_SOUNDSCAPE):
    """
    Build the MiniMax-H3 shot list: shared definitions and style, then ONE [Shot N] line per shot.

    * <Subject N> is assigned HERE, once for the whole list (characters -> props -> locations ->
      palette), so shot lines and story beats always share the same labels.
    * Only assets the shots actually use are defined; nothing else is mentioned.
    * Used assets come from each shot's own cast plus any registry name written in its text.
    Returns dict(text, used, labels, warnings).
    """
    warnings = []
    # Settle the framing before anything reads it, so two neighbouring shots never share
    # both a shot size and an angle (that is what renders as a jump cut).
    _vary_camera(shots, warnings)
    assets = registry.ordered()
    asset_keys = {_akey(a) for a in assets}
    rx, lookup = _compile_matcher(assets)

    used = {}
    for shot in shots:
        for key in ("characters", "props", "locations"):
            for asset in shot.get(key) or []:
                if isinstance(asset, dict) and _akey(asset) in asset_keys:
                    used.setdefault(_akey(asset), asset)
        found = {}
        for text in (shot.get("action") or "", shot.get("camera") or ""):
            _scan_mentions(text, rx, lookup, found)
        for key, asset in found.items():
            used.setdefault(key, asset)
    used_list = [a for a in assets if _akey(a) in used]

    nums, pal_no = registry.picture_numbers()
    vids = registry.video_numbers()
    labels = {}
    for n, a in enumerate(used_list, 1):
        source = f"<Video {vids[_akey(a)]}>" if a["kind"] == KIND_LOCATION else f"<Picture {nums[_akey(a)]}>"
        labels[_akey(a)] = {"subject": f"<Subject {n}>", "source": source, "asset": a}
    palette = registry.palette if (registry.palette and pal_no) else None
    pal_label = ({"subject": f"<Subject {len(used_list) + 1}>", "source": f"<Picture {pal_no}>"}
                 if palette else None)

    def L(value):
        return _apply_labels(value, rx, lookup, labels)

    def_lines = [
        _definition_line(a, labels[_akey(a)], _lower_first(L(_clean_traits(_asset_summary(a, 35)))))
        for a in used_list
    ]

    shot_lines = []
    for index, shot in enumerate(shots, 1):
        number = int(shot.get("shot") or index)
        action = _shot_list_action(shot)
        if not action:
            warnings.append(f"shot {number} has no action text; its line carries only the camera")
        shot_lines.append(_shot_prompt_line(
            number, L(_collapse(shot.get("camera") or "")), L(action), L(_continuity_clause(shot))))

    style_text = _collapse(style)
    if palette:
        names = ", ".join(s["name"] for s in palette["swatches"])
        pdef = (f"{pal_label['subject']} is the color palette reference in {pal_label['source']}, a single row of "
                f"{len(palette['swatches'])} flat color swatches ({names}).")
        grade = (palette.get("grade") or "").strip().rstrip(".")
        if grade:
            pdef += f" {grade}."
        pdef += " Only its colors and tonal look are used, never the swatch layout."
        def_lines.append(pdef)
        style_text = ((style_text + " ") if style_text else "") + (
            f"The color palette, contrast and grade of the whole frame follow {pal_label['subject']}.")

    blocks = []
    trigger = _collapse(trigger_words).rstrip(".")
    if trigger:
        blocks.append(trigger)
    blocks.append("subject_definitions:\n" + ("\n".join(def_lines) or "none"))
    blocks.append("integrated_multimodal_description:\n" + "\n".join([style_text] + shot_lines))
    blocks.append("overall_soundscape:\n" + _collapse(soundscape))
    blocks.append("non_diegetic_music:\nN/A")
    return {"text": "\n\n".join(blocks), "used": used_list, "labels": labels, "warnings": warnings}


def _simulate_reference_filter(text, registry):
    """Approximates a name-based reference filter: image assets whose names appear anywhere in the prompt."""
    up = _norm(text)
    return [a for a in registry.ordered({KIND_CHARACTER, KIND_PROP}) if a["name"] in up]


# =============================================================================
#  Still-frame entries (characters / props: base T2VA format, no references)
# =============================================================================
def _build_character_entry(body, trigger_words):
    style = ("Live-action, photorealistic studio character-sheet photograph with soft, even professional "
             "studio lighting against a solid neutral backdrop.")
    inner = " ".join([CHARACTER_SHEET_LAYOUT, str(body).strip(), STILL_FRAME_SENTENCE])
    return _to_t2va_text(style, inner, STILL_SOUNDSCAPE, "N/A", trigger_words)


def _build_prop_entry(display, desc, trigger_words):
    f = _parse_fields(desc, PROP_FIELD_KEYS)

    def sent(label, key):
        v = f.get(key, "").strip().rstrip(".")
        return f"{label}: {v}." if v else ""

    kind = f.get("type", "").strip().rstrip(".")
    bits = [f"The object is {display}" + (f" ({kind})" if kind else "") + "."]
    for label, key in (("Material and texture", "material & texture"), ("Size and scale", "size & scale"),
                       ("Color and appearance", "color & visual"), ("Condition and wear", "condition & wear"),
                       ("Distinguishing features", "distinguishable features")):
        s = sent(label, key)
        if s:
            bits.append(s)
    if not f:
        bits = [f"The object is {display}: {str(desc).strip().rstrip('.')}."]
    style = ("Live-action, photorealistic studio product-portrait photograph with professional studio lighting; "
             "the full frame shows only this single object against a solid neutral-colored backdrop, exactly as "
             "described, with no other items, objects, or people.")
    return _to_t2va_text(style, " ".join(bits + [STILL_FRAME_SENTENCE]), STILL_SOUNDSCAPE, "N/A", trigger_words)


def _build_palette_entry(palette, trigger_words):
    ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
    sw = palette["swatches"][:8]
    parts = []
    for i, s in enumerate(sw):
        color = s["color"].strip().rstrip(".")
        name = s["name"].strip()
        parts.append(f"the {ordinals[i]} swatch {color} ({name})" if color.lower() != name.lower() else f"the {ordinals[i]} swatch {name}")
    layout = (f"An image divided by thin vertical bars into {len(sw)} equal swatches side by side, each a perfectly flat, evenly lit, "
              f"uniform rectangle of one single solid color with crisp straight edges and no gradients, text, labels, logos, textures, "
              f"objects, or people: " + "; ".join(parts) + ".")
    style = "Live-action, photorealistic studio color-palette reference chart with flat, even, shadowless lighting."
    return _to_t2va_text(style, " ".join([layout, STILL_FRAME_SENTENCE]), STILL_SOUNDSCAPE, "N/A", trigger_words)


class ScreenplayFormatterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auto_capitalize_headers": ("BOOLEAN", {"default": True}),
                "output_format": (["Standard Plain Text", "HTML Visual Script", "Fountain Syntax", "Print-Ready (US Letter)"], {"default": "Standard Plain Text"}),
                "default_aspect_ratio": (["16:9", "2.39:1", "4:3", "9:16", "1:1"], {"default": "16:9"}),
                "use_ollama_ai": ("BOOLEAN", {"default": False}),
                "ollama_model": ("STRING", {"default": "llama3"}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "script_override_mode": (["off", "text", "audio"], {
                    "default": "off",
                    "tooltip": "off uses raw_script; text uses script_override_text; audio transcribes script_override_audio into a screenplay first.",
                }),
                "trigger_words": ("STRING", {"default": "", "multiline": False}),
                "raw_script": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Leave this blank with script_override_mode set to off and the Ollama model writes a complete original screenplay on its own: a random genre, unique characters and a random length, always a full three-act story with a beginning, middle and end. Requires use_ollama_ai."
                }),
                "characters_json": ("STRING", {"multiline": True, "default": "[]"}),
                "locations_json": ("STRING", {"multiline": True, "default": "[]"}),
                "props_json": ("STRING", {"multiline": True, "default": "[]"}),
                "lock_last_generation": ("BOOLEAN", {"default": False}),
                "stored_generation": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "script_override_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Screenplay text used when script_override_mode is text.",
                }),
                "script_override_audio": ("AUDIO", {
                    "tooltip": "Audio transcribed into a screenplay when script_override_mode is audio: speech becomes dialogue, sung lines and lyrics become a separate singer part. The clip's length sets story_length and every shot length, and stretches without words become instrumental shots. Speakers are not diarised, so all speech shares one cue.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "INT", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("script", "shot_list", "story_beats", "story_length", "shot_count", "shot_lengths", "character_list", "prop_list", "location_list", "color_palette", "asset_list")
    OUTPUT_IS_LIST = (False, False, False, False, False, True, False, False, False, False, False)
    FUNCTION = "process_screenplay"
    CATEGORY = "triXope/writing"

    @staticmethod
    def _decode_generation_cache(stored_generation, output_format=None):
        try:
            cache = json.loads(stored_generation or "")
            outputs = cache.get("outputs") if isinstance(cache, dict) else None
            cached_format = cache.get("output_format") if isinstance(cache, dict) else None
            if output_format == "Print-Ready (US Letter)" and cached_format != output_format:
                return None
            if cached_format and output_format and cached_format != output_format:
                return None
            if isinstance(outputs, list) and len(outputs) == len(ScreenplayFormatterNode.RETURN_NAMES):
                return tuple(outputs)
        except (TypeError, ValueError):
            pass
        return None

    @staticmethod
    def _node_output(outputs, stored_generation):
        return {"ui": {"stored_generation": [stored_generation]}, "result": tuple(outputs)}

    def process_screenplay(self, auto_capitalize_headers, output_format, default_aspect_ratio, use_ollama_ai, ollama_model, ollama_url, script_override_mode, trigger_words, raw_script, characters_json, locations_json, lock_last_generation=False, stored_generation="", script_override_text="", props_json="[]", script_override_audio=None):
        # Do not unload anything when this node was already stopped before it started.
        _check_processing_interrupted()
        previous_models = getattr(_SCREENPLAY_RUN, "ollama_models", None)
        _SCREENPLAY_RUN.ollama_models = set()
        try:
            result = self._process_screenplay(
                auto_capitalize_headers, output_format, default_aspect_ratio, use_ollama_ai, ollama_model,
                ollama_url, script_override_mode, trigger_words, raw_script, characters_json, locations_json,
                lock_last_generation, stored_generation, script_override_text, props_json, script_override_audio)
        except BaseException as error:
            if _is_processing_interrupted(error):
                self._cleanup_screenplay_run(ollama_url, "workflow stopped")
            raise
        else:
            # Ollama keeps models resident by default after successful requests. Unload
            # only models this invocation requested, then release unused ComfyUI cache.
            self._cleanup_screenplay_run(ollama_url, "node completed")
            return result
        finally:
            if previous_models is None:
                try:
                    del _SCREENPLAY_RUN.ollama_models
                except AttributeError:
                    pass
            else:
                _SCREENPLAY_RUN.ollama_models = previous_models

    def _cleanup_screenplay_run(self, url, reason):
        """Release resources requested by this node, without unloading other ComfyUI models."""
        requested_models = set(getattr(_SCREENPLAY_RUN, "ollama_models", set()))
        if not requested_models:
            return
        print(f"[ScreenplayFormatter] {reason}; releasing node-requested Ollama models and cached GPU memory.")
        for requested_model in sorted(requested_models):
            if requested_model:
                self._unload_ollama_model(requested_model, url)
        _release_comfyui_cached_memory()

    @staticmethod
    def _unload_ollama_model(model, url):
        """Unload only the Ollama model requested by this node after its execution."""
        endpoint = f"{str(url).rstrip('/')}/api/generate"
        payload = json.dumps({
            "model": str(model).strip(), "prompt": "", "keep_alive": 0, "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(endpoint, data=payload, headers={
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                response.read()
            print(f"🧹 [ScreenplayFormatter] Requested Ollama to unload '{model}'.")
        except Exception as error:
            print(f"⚠️ [ScreenplayFormatter] Could not request Ollama to unload '{model}': {error}")

    def _process_screenplay(self, auto_capitalize_headers, output_format, default_aspect_ratio, use_ollama_ai, ollama_model, ollama_url, script_override_mode, trigger_words, raw_script, characters_json, locations_json, lock_last_generation=False, stored_generation="", script_override_text="", props_json="[]", script_override_audio=None):
        _check_processing_interrupted()
        if lock_last_generation:
            cached_outputs = self._decode_generation_cache(stored_generation, output_format)
            if cached_outputs is not None:
                print("🔒 [ScreenplayFormatter] Replaying cached generation; screenplay and LLM stages skipped.")
                return self._node_output(cached_outputs, stored_generation)

        print("\n" + "=" * 65)
        print("🎬 [ScreenplayFormatter] STARTING SCREENPLAY PROCESSING")
        print("=" * 65)

        try:
            character_defs = json.loads(characters_json) if characters_json else []
        except Exception as e:
            print(f"⚠️ [ScreenplayFormatter] Warning: Failed to parse characters_json: {e}")
            character_defs = []

        try:
            location_defs = json.loads(locations_json) if locations_json else []
        except Exception as e:
            print(f"⚠️ [ScreenplayFormatter] Warning: Failed to parse locations_json: {e}")
            location_defs = []

        try:
            prop_defs = json.loads(props_json) if props_json else []
        except Exception as e:
            print(f"⚠️ [ScreenplayFormatter] Warning: Failed to parse props_json: {e}")
            prop_defs = []

        char_dict = {c['name'].strip().upper(): c['description'].strip() for c in character_defs if c.get('name')}
        loc_dict = {l['name'].strip().upper(): l for l in location_defs if l.get('name')}
        prop_dict = {p['name'].strip().upper(): p.get('description', '').strip()
                     for p in prop_defs if p.get('name')}

        # LLM availability is resolved before the script source is picked, because a blank
        # raw_script with script_override_mode=off is filled in by the model further down.
        use_llm = bool(use_ollama_ai and str(ollama_model or "").strip() and ollama_model != "None / Ollama Offline")
        tw = (trigger_words or "").strip()

        # Determine the active script source from script_override_mode (off/text/audio).
        override_mode = _script_override_mode(script_override_mode)
        audio_timeline = None
        if override_mode == "audio":
            if script_override_audio is None:
                raise ValueError(
                    "script_override_mode is audio but script_override_audio is not connected; "
                    "connect the audio to transcribe or set the mode to off or text"
                )
            transcript = _transcribe_audio(script_override_audio)
            transcript_segments = transcript["segments"]
            audio_duration = float(transcript["duration"])
            transcript_groups = _transcript_groups(transcript_segments)
            audio_timeline = _audio_timeline(transcript_groups, audio_duration)
            active_script = _groups_to_screenplay(transcript_groups).strip()
            script_source_name = "Wired Audio (script_override_audio -> Whisper transcript)"
            if not active_script:
                print("⚠️ [ScreenplayFormatter] Whisper found no speech or lyrics in script_override_audio.")
            else:
                print(f"├─ Transcribed {len(transcript_segments)} segment(s) over {audio_duration:.2f}s:")
                for transcript_line in active_script.splitlines()[:12]:
                    print(f"│    {transcript_line}")
                if len(active_script.splitlines()) > 12:
                    print("│    ...")
                silent = sum(1 for item in audio_timeline if item["kind"] == "instrumental")
                print(f"├─ Audio Timeline: {len(audio_timeline)} shot(s), {silent} instrumental stretch(es), "
                      "story length locked to the audio")
        elif override_mode == "text" and script_override_text and script_override_text.strip():
            active_script = script_override_text.strip()
            script_source_name = "Wired External Input (script_override_text)"
        else:
            if override_mode == "text":
                print("⚠️ [ScreenplayFormatter] script_override_mode is text but script_override_text is empty; using raw_script.")
            active_script = raw_script.strip()
            script_source_name = "Text Area (raw_script)"
            if not active_script and override_mode == "off":
                if use_llm:
                    active_script, script_source_name = self._generate_random_screenplay(ollama_model, ollama_url)
                else:
                    print("⚠️ [ScreenplayFormatter] raw_script is blank and script_override_mode is off, but "
                          "Ollama is unavailable, so no screenplay can be invented; returning blank outputs.")

        print(f"├─ Script Source: {script_source_name}")
        print(f"├─ Active Script Length: {len(active_script.splitlines())} lines")
        print(f"├─ Selected Output Format: {output_format}")
        print(f"├─ Target Aspect Ratio: {default_aspect_ratio}")
        print(f"├─ Ollama AI Acceleration: {'ENABLED' if use_ollama_ai else 'DISABLED'} (Model: {ollama_model if use_ollama_ai else 'N/A'})")
        print(f"├─ Active Character Profiles: {len(char_dict)}")
        print(f"├─ Active Prop Profiles: {len(prop_dict)}")
        print(f"└─ Active Location Profiles: {len(loc_dict)}")

        if not active_script:
            print("\n⚠️ [ScreenplayFormatter] Active script is empty. Returning blank outputs.")
            print("=" * 65 + "\n")
            blank_outputs = ("", "", "", 0, 0, [], "", "", "", "", "")
            return self._node_output(blank_outputs, stored_generation)

        # -------------------------------------------------------------
        # STEP 1: SCRIPT PARSING & STRUCTURE FORMATTING
        # -------------------------------------------------------------
        print("\n[STEP 1/5] 📄 Parsing and Structuring Screenplay...")
        lines = active_script.split('\n')
        
        scene_heading_re = re.compile(r'^(INT\.|EXT\.|INT/EXT\.|EXT/INT\.|INT\./EXT\.|EST\.)', re.IGNORECASE)
        transition_re = re.compile(r'^(FADE IN:|FADE OUT\.|CUT TO:|DISSOLVE TO:|SMASH CUT TO:|FADE TO BLACK\.)', re.IGNORECASE)
        parenthetical_re = re.compile(r'^\(.*\)$')
        
        blocks = []
        i = 0
        n = len(lines)
        
        while i < n:
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            if scene_heading_re.match(line):
                text = line.upper() if auto_capitalize_headers else line
                blocks.append(('SCENE_HEADING', text))
                i += 1
                continue
                
            if transition_re.match(line) or (line.isupper() and line.endswith(':')):
                text = line.upper() if auto_capitalize_headers else line
                blocks.append(('TRANSITION', text))
                i += 1
                continue
                
            is_char = False
            if line.isupper() and len(line) < 40 and not line.endswith('.'):
                next_idx = i + 1
                while next_idx < n and not lines[next_idx].strip():
                    next_idx += 1
                if next_idx < n:
                    next_line = lines[next_idx].strip()
                    if next_line.startswith('(') or not next_line.isupper():
                        is_char = True
            
            if is_char:
                char_name = line.upper() if auto_capitalize_headers else line
                blocks.append(('CHARACTER', char_name))
                i += 1
                
                while i < n:
                    curr = lines[i].strip()
                    if not curr:
                        break
                    if parenthetical_re.match(curr):
                        blocks.append(('PARENTHETICAL', curr))
                    elif curr.isupper() and len(curr) < 40 and not curr.endswith('.') and not parenthetical_re.match(curr):
                        break
                    elif scene_heading_re.match(curr) or transition_re.match(curr):
                        break
                    else:
                        blocks.append(('DIALOGUE', curr))
                    i += 1
                continue
                
            if parenthetical_re.match(line):
                blocks.append(('PARENTHETICAL', line))
                i += 1
                continue
                
            if line.isupper() and len(line) < 45:
                blocks.append(('SHOT_HEADER', line.upper() if auto_capitalize_headers else line))
                i += 1
                continue

            blocks.append(('ACTION', line))
            i += 1

        plain_text = self._to_plain_text(blocks)
        fountain_text = self._to_fountain(blocks)
        html_markup = self._to_html(blocks)
        print(f"  └─ Complete! Identified and aligned {len(blocks)} screenplay elements.")

        # -------------------------------------------------------------
        # PIPELINE ORDER: characters -> props -> locations -> shots
        # Every stage registers its assets in one registry so later prompts
        # know the exact names and the label order (characters, props, locations).
        # -------------------------------------------------------------
        registry = _Registry()

        # -------------------------------------------------------------
        # STEP 2: CHARACTERS (base T2VA format, text only -> <Picture> assets)
        # -------------------------------------------------------------
        print("\n[STEP 2/6] 👤 Generating Character Concept Prompts (MiniMax-H3 T2VA)...")
        character_entries = self._build_character_entries(active_script, char_dict, registry, use_llm, ollama_model, ollama_url, tw)
        print(f"  └─ Complete! Generated concept prompt entries for {len(character_entries)} character(s).")

        # -------------------------------------------------------------
        # STEP 3: PROPS (base T2VA format, text only -> <Picture> assets)
        # -------------------------------------------------------------
        print("\n[STEP 3/6] 🔍 Generating Prop & Object Concept Prompts (MiniMax-H3 T2VA)...")
        prop_entries = self._build_prop_entries(active_script, registry, use_llm, ollama_model, ollama_url, tw, prop_dict)
        print(f"  └─ Complete! Generated studio portrait prompt entries for {len(prop_entries)} prop(s).")

        # -------------------------------------------------------------
        # STEP 4: LOCATIONS (full-reference format, props as references -> <Video> assets)
        # -------------------------------------------------------------
        print("\n[STEP 4/6] 📍 Generating Location Prompts (MiniMax-H3 full-reference)...")
        location_entries = self._build_location_entries(active_script, loc_dict, registry, use_llm, ollama_model, ollama_url, tw)
        print(f"  └─ Complete! Generated location prompt entries for {len(location_entries)} location(s).")

        for a, b in registry.collisions():
            print(f"  ⚠️ [Registry] '{a}' is contained inside '{b}'. Name-based reference filters may attach both; consider renaming.")

        # -------------------------------------------------------------
        # STEP 5: COLOR PALETTE (base T2VA format; image reference attached to every shot)
        # -------------------------------------------------------------
        print("\n[STEP 5/6] 🎨 Generating Color Palette Reference Prompt (MiniMax-H3 T2VA)...")
        color_palette = self._build_color_palette(active_script, registry, use_llm, ollama_model, ollama_url, tw)
        print(f"  └─ Complete! Palette of {len(registry.palette['swatches'])} swatches defined.")

        # -------------------------------------------------------------
        # STEP 6: STORY (shot list + timestamped story beats)
        # -------------------------------------------------------------
        print("\n[STEP 6/6] 🎬 Generating Shot List & Story Beats (MiniMax-H3)...")
        shot_list_text, story_beats, shot_lengths = self._build_story(
            blocks, active_script, char_dict, loc_dict, registry, default_aspect_ratio, use_llm,
            ollama_model, ollama_url, tw, audio_timeline=audio_timeline)
        shot_count = len(shot_lengths)
        story_length = int(sum(shot_lengths))
        print(f"  └─ Complete! {shot_count} story beats, total runtime {story_length}s.")

        character_list = LIST_DELIM.join(character_entries)
        prop_list = LIST_DELIM.join(prop_entries)
        location_list = LIST_DELIM.join(location_entries)

        selected_script = plain_text
        if output_format == "HTML Visual Script":
            selected_script = html_markup
        elif output_format == "Fountain Syntax":
            selected_script = fountain_text
        elif output_format == "Print-Ready (US Letter)":
            selected_script = self._to_print_html(blocks)

        # Asset names in generation order: characters -> props -> locations
        asset_list = ", ".join(registry.names())

        self._print_reference_order(registry)

        print("\n" + "=" * 65)
        print("✅ [ScreenplayFormatter] ALL PROCESSES FINISHED SUCCESSFULLY!")
        print(f"  ├─ Output Script Length: {len(selected_script)} chars")
        print(f"  ├─ Output Shot List: {len(shot_list_text)} chars")
        print(f"  ├─ Output Story Beats: {shot_count} beats")
        print(f"  ├─ Story Length: {story_length}s")
        print(f"  ├─ Calculated Shot Lengths: {shot_lengths}")
        print(f"  ├─ Output Asset List: {asset_list}")
        print(f"  ├─ Output Character Prompts: {len(character_entries)} entries")
        print(f"  ├─ Output Prop Prompts: {len(prop_entries)} entries")
        print(f"  └─ Output Location Prompts: {len(location_entries)} entries")
        print("=" * 65 + "\n")

        outputs = (selected_script, shot_list_text, story_beats, story_length, shot_count, shot_lengths, character_list, prop_list, location_list, color_palette, asset_list)
        cache = json.dumps({"version": 2, "output_format": output_format, "outputs": outputs}, ensure_ascii=False)
        return self._node_output(outputs, cache)

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------
    def _generate_random_screenplay(self, model, url):
        """Ask the model for an original screenplay when the writer field is blank."""
        prompt, info = _random_screenplay_prompt()
        print(f"  ├─ [Random Screenplay] Blank raw_script with override off: asking {model} for an original "
              f"{info['genre']} screenplay ({info['pages']} pages, {info['cast']} speaking characters, {info['shape']})...")
        text = self._call_ollama_api(prompt, model, url, timeout=900, num_predict=info["num_predict"])
        cleaned = _clean_generated_screenplay(text)
        if not cleaned:
            print("  └─ ⚠️ [Random Screenplay] The model returned no screenplay; the script stays empty.")
            return "", "Text Area (raw_script)"
        print(f"  └─ [Random Screenplay] Generated an original screenplay: {len(cleaned.splitlines())} lines, "
              f"{len(cleaned)} chars.")
        return cleaned, f"LLM Original (blank raw_script) - {info['genre']}, {info['pages']} pages"

    def _call_ollama_api(self, prompt, model, url, timeout=300, num_predict=None, think=None):
        endpoint = f"{url.rstrip('/')}/api/generate"
        request = {"model": model, "prompt": prompt, "stream": True}
        if num_predict is not None:
            request["options"] = {"num_predict": int(num_predict)}
        if think is not None:
            request["think"] = think
        payload = json.dumps(request).encode("utf-8")
        attempt, backoff = 0, 1

        while True:
            _check_processing_interrupted()
            attempt += 1
            req = urllib.request.Request(endpoint, data=payload, headers={
                "Content-Type": "application/json",
                "Accept": "application/x-ndjson",
            })
            response_queue = queue.Queue(maxsize=1)
            response_holder = []
            cancelled = threading.Event()
            worker = threading.Thread(
                target=_stream_ollama_request,
                args=(req, timeout, response_queue, response_holder, cancelled),
                daemon=True,
                name="ScreenplayFormatter-Ollama",
            )
            try:
                print(f"     [Ollama API] Sending payload to {endpoint} (Model: {model}; attempt {attempt})...")
                worker.start()
                active_models = getattr(_SCREENPLAY_RUN, "ollama_models", None)
                if active_models is not None:
                    active_models.add(str(model).strip())
                while True:
                    _check_processing_interrupted()
                    try:
                        result_type, result = response_queue.get(timeout=0.1)
                        break
                    except queue.Empty:
                        continue
                if result_type == "error":
                    raise result
                if isinstance(result, dict):
                    raw_out = str(result.get("response", "") or "").strip()
                    if not raw_out:
                        print("     [Ollama API] Empty response text "
                              f"(thinking chars: {result.get('thinking_chars', 0)}; "
                              f"done_reason: {result.get('done_reason', 'unknown')}; "
                              f"eval_count: {result.get('eval_count', 'unknown')}).")
                else:
                    raw_out = str(result or "").strip()
                print(f"     [Ollama API] Response received successfully ({len(raw_out)} chars).")
                return raw_out
            except Exception as error:
                try:
                    _check_processing_interrupted()
                except BaseException:
                    cancelled.set()
                    for response in response_holder:
                        try:
                            response.close()
                        except Exception:
                            pass
                    raise
                cancelled.set()
                for response in response_holder:
                    try:
                        response.close()
                    except Exception:
                        pass
                print(f"     [Ollama API] ❌ Connection failed or timed out: {error}")
                if not (isinstance(error, urllib.error.HTTPError) and error.code == 500):
                    return ""
            except BaseException:
                cancelled.set()
                for response in response_holder:
                    try:
                        response.close()
                    except Exception:
                        pass
                raise

            print(f"     [Ollama API] Retrying the same payload in {backoff}s (next attempt {attempt + 1}).")
            remaining = backoff
            while remaining > 0:
                _check_processing_interrupted()
                interval = min(0.1, remaining)
                time.sleep(interval)
                remaining -= interval
            backoff = min(backoff * 2, 30)

    @staticmethod
    def _display_name(name):
        n = str(name or "").strip()
        if not n.isupper():
            return n
        small = {"of", "the", "and", "a", "an", "in", "on", "at", "to", "for"}
        words = n.lower().split(" ")
        return " ".join(w if (i and w in small) else w.capitalize() for i, w in enumerate(words))

    # ------------------------------------------------------------------
    # COLOR PALETTE  (a swatch chart; one image attached to every shot)
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_swatches(raw):
        out = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if isinstance(item, dict):
                name = _collapse(item.get("name", ""))
                color = _collapse(item.get("color", "")) or name
                role = _collapse(item.get("role", ""))
            else:
                name = color = _collapse(item)
                role = ""
            if name:
                out.append({"name": name.replace("|", "-"), "color": color.replace("|", "-"), "role": role})
        return out[:8]

    def _palette_fallback(self, registry):
        phrases = []
        for a in registry.ordered({KIND_LOCATION}):
            f = _parse_fields(a["description"], LOCATION_FIELD_KEYS)
            for part in re.split(r"[,;]|\band\b", f.get("color palette", "")):
                p = _collapse(part).strip(" .")
                if p and p.lower() not in [x.lower() for x in phrases]:
                    phrases.append(p)
        swatches = [{"name": p, "color": p, "role": ""} for p in phrases[:8]]
        if len(swatches) >= 3:
            return {"swatches": swatches, "grade": ""}
        defaults = [("deep shadow", "a very dark, near-black neutral with a slight cool undertone"),
                    ("charcoal", "a dark neutral grey"),
                    ("mid grey", "a balanced mid-tone neutral grey"),
                    ("warm highlight", "a soft warm off-white"),
                    ("muted accent", "a desaturated mid-saturation accent hue"),
                    ("skin tone", "a natural medium flesh tone")]
        return {"swatches": [{"name": n, "color": c, "role": ""} for n, c in defaults], "grade": ""}

    def _build_color_palette(self, script, registry, use_llm, model, url, tw):
        palette = None
        if use_llm:
            print("  ├─ [Palette Pipeline] Running Colorist master palette design...")
            locs = "\n".join(f"- {a['name']}: {_asset_summary(a, 40)}" for a in registry.ordered({KIND_LOCATION})) or "(none)"
            prompt = PALETTE_STAGE_PROMPT.replace("@@LOCATIONS@@", locs)
            res = self._call_ollama_api(f"{prompt}\n\nSCREENPLAY:\n{script}", model, url)
            for obj in _parse_json_objects(res):
                sw = self._clean_swatches(obj.get("swatches"))
                if len(sw) >= 3:
                    palette = {"swatches": sw, "grade": _collapse(obj.get("grade", "")).replace("|", "-")}
                    break
            if palette is None:
                print("  ├─ ⚠️ Palette design returned nothing usable; deriving the palette from location profiles...")
        if palette is None:
            palette = self._palette_fallback(registry)
        registry.palette = palette
        return _build_palette_entry(palette, tw)

    def _print_reference_order(self, registry):
        nums, pal_no = registry.picture_numbers()
        vids = registry.video_numbers()
        print("\n[Reference Order] Connect reference IMAGES in this exact order (all are always attached):")
        rows = [(nums[_akey(a)], a["name"]) for a in registry.ordered({KIND_CHARACTER, KIND_PROP})]
        if pal_no:
            rows.append((pal_no, "COLOR PALETTE"))
        for n, name in sorted(rows):
            print(f"  <Picture {n}> = {name}")
        print("Reference VIDEOS (locations):")
        for a in registry.ordered({KIND_LOCATION}):
            print(f"  <Video {vids[_akey(a)]}> = {a['name']}")
        n_img = len(rows)
        n_vid = len(vids)
        if n_img > MAX_REF_IMAGES:
            print(f"  ⚠️ {n_img} reference images exceed MiniMax-H3's limit of {MAX_REF_IMAGES} images per generation.")
        if n_vid > MAX_REF_VIDEOS:
            print(f"  ⚠️ {n_vid} reference videos exceed MiniMax-H3's limit of {MAX_REF_VIDEOS} videos per generation.")

    # ------------------------------------------------------------------
    # CHARACTERS  (two stages, as before; Stage 2 now writes only the visual body,
    # the official T2VA structure and the four-panel layout sentence are added in code)
    # ------------------------------------------------------------------
    def _character_body_fallback(self, desc):
        return (f"{str(desc).strip().rstrip('.')}. Neutral standing pose, soft even studio lighting against a plain solid "
                f"neutral gray background with zero background clutter, sharp focus, photorealistic, hyper-detailed texture, "
                f"clean character concept design.")

    def _build_character_entries(self, script, char_dict, registry, use_llm, model, url, tw):
        profiles = []
        stage1_text = ""
        if use_llm:
            print("  ├─ [Character Pipeline] Running Stage 1: Extraction & Attribute Mapping...")
            prompt = CHARACTER_STAGE1_PROMPT.replace("@@KNOWN@@", json.dumps(char_dict))
            stage1_text = self._call_ollama_api(f"{prompt}\n\nSCREENPLAY:\n{script}", model, url)
            profiles = _parse_header_entries(stage1_text, ["Name"])
            if not profiles:
                print("  ├─ ⚠️ Stage 1 returned no parsable characters. Falling back to supplied character definitions...")
        known = {_norm(n) for n, _ in profiles}
        for name, desc in char_dict.items():
            if _norm(name) not in known:
                profiles.append((name, desc))
        if not profiles:
            return []

        bodies = {}
        if use_llm and stage1_text:
            print("  ├─ [Character Pipeline] Running Stage 2: Multi-Panel MiniMax-H3 Concept Prompting...")
            profile_text = "\n\n".join(f"Name: {n} - {d} |" for n, d in profiles)
            res2 = self._call_ollama_api(f"{CHARACTER_STAGE2_PROMPT}\n\nCHARACTER PROFILES:\n{profile_text}", model, url)
            bodies = {_norm(n): b for n, b in _parse_header_entries(res2, ["Name"])}

        entries = []
        for name, desc in profiles:
            asset = registry.add(KIND_CHARACTER, name, desc, display=name)
            if asset is None:
                continue
            body = _clean_block(bodies.get(asset["name"], ""))
            if not body:
                if use_llm and stage1_text:
                    print(f"  ├─ ⚠️ No Stage 2 body for {asset['name']}; using template body.")
                body = self._character_body_fallback(desc)
            entries.append(_build_character_entry(body, tw))
        return entries

    # ------------------------------------------------------------------
    # PROPS  (Stage 1 extraction; the visual prompt is composed in code from the visual fields)
    # ------------------------------------------------------------------
    def _build_prop_entries(self, script, registry, use_llm, model, url, tw, known_props=None):
        raw = []
        if use_llm:
            print("  ├─ [Prop Pipeline] Running Stage 1: Prop Master Object Extraction...")
            chars = ", ".join(registry.names({KIND_CHARACTER})) or "(none)"
            prompt = PROP_STAGE1_PROMPT.replace("@@CHARACTERS@@", chars)
            res = self._call_ollama_api(f"{prompt}\n\nSCREENPLAY:\n{script}", model, url)
            raw = _parse_header_entries(res, ["Object Name", "Prop Name", "Prop", "Object"])
            if not raw and not known_props:
                print("  ├─ ⚠️ Prop extraction returned nothing parsable.")
                return []
        elif not known_props:
            print("  ├─ Props need local LLM extraction; no prop prompts generated (Ollama disabled).")
            return []

        existing = {_Registry._key(KIND_PROP, name) for name, _ in raw}
        for name, desc in (known_props or {}).items():
            key = _Registry._key(KIND_PROP, name)
            if key not in existing:
                raw.append((name, desc))
                existing.add(key)

        entries = []
        for name, desc in raw:
            character = registry.lookup(name, {KIND_CHARACTER})
            if character and character["name"] == _norm(name):
                print(f"  ├─ ⚠️ Prop '{name}' duplicates a character name; skipped.")
                continue
            asset = registry.add(KIND_PROP, name, desc, display=self._display_name(name))
            if asset is None:
                continue
            entries.append(_build_prop_entry(asset["display"], desc, tw))
        return entries

    # ------------------------------------------------------------------
    # LOCATIONS  (Stage 1 profile -> Stage 2 full-reference prompt; props named in the prompt are the references)
    # ------------------------------------------------------------------
    def _prop_listing(self, registry, words=25):
        items = registry.ordered({KIND_PROP})
        if not items:
            return "(none)"
        return "\n".join(f"- {a['name']}: {_asset_summary(a, words)}" for a in items)

    def _prop_listing_fixed(self, registry, words=40):
        nums, _ = registry.picture_numbers()
        items = registry.ordered({KIND_PROP})
        if not items:
            return "(none)"
        return "\n".join(f"- {a['name']} [<Picture {nums[_akey(a)]}>]: {_asset_summary(a, words)}" for a in items)

    def _build_location_entries(self, script, loc_dict, registry, use_llm, model, url, tw):
        profiles = []
        if use_llm:
            print("  ├─ [Location Pipeline] Running Stage 1: Production Designer & Location Scout Extraction...")
            prompt = (LOCATION_STAGE1_PROMPT.replace("@@PROPS@@", self._prop_listing(registry))
                      .replace("@@KNOWN@@", json.dumps(loc_dict)))
            res = self._call_ollama_api(f"{prompt}\n\nSCREENPLAY:\n{script}", model, url)
            profiles = _parse_header_entries(res, ["Location Name", "Location", "Scene Name"])
            if not profiles:
                print("  ├─ ⚠️ Stage 1 returned no parsable locations. Falling back to supplied location definitions...")
        have = {_Registry._key(KIND_LOCATION, n) for n, _ in profiles}
        for name, data in loc_dict.items():
            if _Registry._key(KIND_LOCATION, name) not in have:
                where = "interior" if str(data.get("int_ext", "INT")).upper().startswith("INT") else "exterior"
                desc = f"{str(data.get('description', '')).strip().rstrip('.')} ({where}, {str(data.get('time_of_day', 'DAY')).lower()})"
                profiles.append((name, desc))

        entries = []
        for name, desc in profiles:
            asset = registry.add(KIND_LOCATION, name, desc, display=name)
            if asset is None:
                continue
            text = None
            if use_llm:
                print(f"  ├─ [Location Pipeline] Running Stage 2 for {asset['name']}...")
                text = self._write_location_prompt(asset, desc, registry, model, url, tw)
            if text is None:
                text = self._location_prompt_fallback(asset, desc, registry, tw)
            entries.append(f"{text.rstrip()}\n\n{LOCATION_REFERENCE_CAMERA_SUFFIX}")
        return entries

    def _write_location_prompt(self, asset, profile, registry, model, url, tw):
        system = (LOCATION_STAGE2_PROMPT.replace("@@PROPS@@", self._prop_listing_fixed(registry, 40))
                  .replace("@@ASPECT_NOTE@@", ""))
        base = f"{system}\n\nLOCATION PROFILE:\nLocation Name: {asset['name']} - {profile}"
        for attempt in range(2):
            note = "" if attempt == 0 else ("\n\nYour previous answer did not follow the six-section format. Output exactly the six "
                                            "sections (subject_definitions, summary, retention_analysis, detailed_description, "
                                            "overall_soundscape, non_diegetic_music) with a [Shot 1] marker inside detailed_description.")
            res = self._call_ollama_api(base + note, model, url)
            sections = _parse_sections(res)
            if len(sections.get("detailed_description", "").split()) >= 40:
                out = _assemble_reference_prompt(sections, registry, {KIND_PROP}, tw, still=True, no_people=True,
                                                 default_soundscape="Low ambient room tone that fits the space.")
                self._log_assembly(asset["name"], out)
                self._warn_stray_names(out["text"], registry, {KIND_CHARACTER, KIND_LOCATION}, asset["name"])
                return out["text"]
        print(f"  ├─ ⚠️ Stage 2 failed for {asset['name']}; using template location prompt.")
        return None

    def _location_prompt_fallback(self, asset, profile, registry, tw):
        f = _parse_fields(profile, LOCATION_FIELD_KEYS)
        rx, lookup = _compile_matcher(registry.ordered({KIND_PROP}))
        found = {}
        _scan_mentions(profile, rx, lookup, found)
        props = [a for a in registry.ordered({KIND_PROP}) if _akey(a) in found]

        style_bits = [b for b in (f.get("style & era", ""), f.get("color palette", ""), f.get("lighting & atmosphere", "")) if b]
        style = "A full-frame, professional architectural landscape photograph with studio lighting"
        if style_bits:
            style += ", " + "; ".join(x.rstrip(".") for x in style_bits)
        style += "."
        body = ["A locked-off, eye-level wide static shot frames the whole space."]
        for label, key in (("The space", "scale & layout"), ("Key dressing", "key props & set dressing"),
                           ("Condition", "condition & traits")):
            v = f.get(key, "").strip().rstrip(".")
            if v:
                body.append(f"{label}: {v}.")
        if not f:
            body.append(str(profile).strip().rstrip(".") + ".")
        sections = {
            "subject_definitions": "\n".join(f"{p['name']}: {_asset_summary(p, 30)}" for p in props) or "none",
            "summary": "[reference generation] A single static wide shot of the empty location"
                       + (", using " + ", ".join(p["name"] for p in props) + " as visual anchors." if props else "."),
            "retention_analysis": "\n".join(f"{p['name']} (appears in [Shot 1]): fully_preserved - the referenced prop is retained as set dressing."
                                            for p in props) or "none",
            "detailed_description": style + "\n[Shot 1] " + " ".join(body),
            "overall_soundscape": "Low ambient room tone that fits the space.",
            "non_diegetic_music": "N/A",
        }
        out = _assemble_reference_prompt(sections, registry, {KIND_PROP}, tw, still=True, no_people=True,
                                         default_soundscape="Low ambient room tone that fits the space.")
        self._log_assembly(asset["name"], out)
        return out["text"]

    # ------------------------------------------------------------------
    # SHOTS  (Pass 1: breakdown -> Pass 2: one full-reference brief per shot)
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_duration(value):
        try:
            return max(MIN_SHOT_SECONDS, int(round(float(value))))
        except (TypeError, ValueError, OverflowError):
            return max(MIN_SHOT_SECONDS, 8)

    def _shot_duration(self, shot):
        d = shot.get("duration")
        if d:
            return self._clamp_duration(d)
        words = len(str(shot.get("action", "")).split())
        return self._clamp_duration(round(words / 2.5) + 3)

    def _as_name_list(self, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [v.strip() for v in re.split(r"[;,]", value) if v.strip()]
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    def _screenplay_source_blocks(self, script):
        """Group screenplay lines into typed visual/action and dialogue blocks."""
        lines = script.splitlines()
        blocks = []
        action_lines, dialogue_lines = [], []
        current_character = ""
        voiceover = False
        in_character_context = False

        def add_block(kind, pending):
            if pending:
                blocks.append({
                    "type": kind,
                    "text": " ".join(text for _, text in pending).strip(),
                    "start_line": pending[0][0],
                    "end_line": pending[-1][0],
                })
                pending.clear()

        def flush_action():
            add_block("ACTION", action_lines)

        def flush_dialogue():
            if dialogue_lines:
                blocks.append({
                    "type": "DIALOGUE",
                    "text": " ".join(text for _, text in dialogue_lines).strip(),
                    "start_line": dialogue_lines[0][0],
                    "end_line": dialogue_lines[-1][0],
                    "speaker": current_character,
                    "voiceover": voiceover,
                })
                dialogue_lines.clear()

        scene_heading_re = re.compile(r'^(INT\.|EXT\.|INT/EXT\.|EXT/INT\.|INT\./EXT\.|I/E\.|EST\.)', re.IGNORECASE)
        transition_re = re.compile(r'^(FADE IN:|FADE OUT\.|CUT TO:|DISSOLVE TO:|SMASH CUT TO:|FADE TO BLACK\.)', re.IGNORECASE)
        parenthetical_re = re.compile(r'^\(.*\)$')

        for index, raw in enumerate(lines):
            line_no = index + 1
            line = raw.strip()
            if not line:
                flush_action()
                flush_dialogue()
                in_character_context = False
                current_character = ""
                continue

            if scene_heading_re.match(line):
                flush_action()
                flush_dialogue()
                blocks.append({"type": "SCENE_HEADING", "text": line, "start_line": line_no, "end_line": line_no})
                in_character_context = False
                current_character = ""
                continue

            if transition_re.match(line) or (line.isupper() and line.endswith(":")):
                flush_action()
                flush_dialogue()
                blocks.append({"type": "TRANSITION", "text": line, "start_line": line_no, "end_line": line_no})
                in_character_context = False
                current_character = ""
                continue

            is_character = False
            if line.isupper() and len(line) < 40 and not line.endswith("."):
                next_index = index + 1
                while next_index < len(lines) and not lines[next_index].strip():
                    next_index += 1
                if next_index < len(lines):
                    next_line = lines[next_index].strip()
                    if next_line.startswith("(") or not next_line.isupper():
                        is_character = True
            if is_character:
                flush_action()
                flush_dialogue()
                blocks.append({"type": "CHARACTER", "text": line, "start_line": line_no, "end_line": line_no})
                current_character = re.sub(r'\s*\(.*?\)', '', line).strip()
                voiceover = bool(re.search(r'\((?:V\.?O\.?|O\.?S\.?)\)', line))
                in_character_context = True
                continue

            if parenthetical_re.match(line):
                flush_action()
                flush_dialogue()
                blocks.append({
                    "type": "PARENTHETICAL", "text": line, "start_line": line_no, "end_line": line_no,
                    "speaker": current_character, "voiceover": voiceover,
                })
                continue

            if in_character_context:
                dialogue_lines.append((line_no, line))
                continue

            if line.isupper() and len(line) < 45:
                flush_action()
                blocks.append({"type": "SHOT_HEADER", "text": line, "start_line": line_no, "end_line": line_no})
                continue

            flush_dialogue()
            action_lines.append((line_no, line))

        flush_action()
        flush_dialogue()
        return blocks

    def _breakdown_shots(self, script, char_dict, loc_dict, registry, aspect, model, url):
        print("  ├─ [Shot Pipeline] Running Pass 1: screenplay block breakdown...")
        source_blocks = self._screenplay_source_blocks(script)
        pages, page = [], []
        page_lines = 0
        previous_type = None
        for block in source_blocks:
            block_type = block["type"]
            width = SCREENPLAY_LINE_WIDTH
            if block_type in {"CHARACTER", "PARENTHETICAL", "DIALOGUE"}:
                indent = {"CHARACTER": 28, "PARENTHETICAL": 20, "DIALOGUE": 14}[block_type]
                width = max(1, SCREENPLAY_LINE_WIDTH - indent)
            wrapped_lines = max(1, (len(block["text"]) + width - 1) // width)
            if block_type == "ACTION" and previous_type not in {None, "ACTION"}:
                wrapped_lines += 1
            elif block_type in {"SCENE_HEADING", "TRANSITION", "SHOT_HEADER"}:
                if previous_type is not None:
                    wrapped_lines += 1
                wrapped_lines += 1
            elif block_type == "CHARACTER" and previous_type not in {"CHARACTER", "PARENTHETICAL", "DIALOGUE"}:
                wrapped_lines += 1
            if page and page_lines + wrapped_lines > SCREENPLAY_PAGE_LINES:
                pages.append((page, page_lines))
                page, page_lines = [], 0
            page.append(block)
            page_lines += wrapped_lines
            previous_type = block_type
        if page:
            pages.append((page, page_lines))

        prompt = (SHOT_BREAKDOWN_PROMPT.replace("@@REGISTRY@@", _registry_listing(registry, words=30, with_palette=True))
                  .replace("@@ASPECT@@", aspect).replace("@@MIN@@", str(MIN_H3_SHOT_SECONDS))
                  .replace("@@CHAR_DEFS@@", json.dumps(char_dict)).replace("@@LOC_DEFS@@", json.dumps(loc_dict)))
        shots = []
        current_location = None
        for page_no, (source_page, formatted_page_lines) in enumerate(pages, 1):
            _check_processing_interrupted()
            source_text = "\n".join(
                f"{block_id}: {block['type']} (source lines {block['start_line']}-{block['end_line']}): {block['text']}"
                for block_id, block in enumerate(source_page, 1)
            )
            page_words = sum(len(block["text"].split()) for block in source_page)
            target_seconds = max(MIN_SHOT_SECONDS,
                                 round(formatted_page_lines * SECONDS_PER_PAGE / SCREENPLAY_PAGE_LINES))
            request = (f"{prompt}\n\nSCREENPLAY PAGE {page_no}/{len(pages)}; approximate pacing target {target_seconds}s. "
                       "Use short action paragraphs as likely visual beats; split a paragraph only when it contains distinct sequential visual changes. "
                       "Do not make camera cuts just because there is a new dialogue line; keep a short exchange together when it plays in one setup. Make every shot differ from the shot before it in shot size or camera angle; never repeat both. "
                       "Use source_blocks to cover each ACTION and DIALOGUE block exactly; scene headings, character cues, parentheticals and transitions are context, not separate required shots. "
                       "Use around 15-30 words and 1-3 formatted lines as soft targets for action beats, never as truncation limits. "
                       "Do not truncate, omit, or rewrite spoken dialogue.\n"
                       f"PAGE SOURCE BLOCKS (local ID, type, source lines, exact text):\n{source_text}")
            response = self._call_ollama_api(request, model, url, timeout=900)
            page_shots = []
            for obj in _parse_json_objects(response):
                ids = obj.get("source_blocks", obj.get("source_lines", []))
                if not isinstance(ids, (list, tuple)):
                    ids = [ids]
                block_ids = sorted({int(value) for value in ids
                                    if str(value).isdigit() and 1 <= int(value) <= len(source_page)})
                if not block_ids or not any(key in obj for key in ("action", "characters", "props", "camera")):
                    continue

                def assets(key, kind):
                    found = [registry.lookup(name, {kind}) for name in self._as_name_list(obj.get(key))]
                    unique = {_akey(asset): asset for asset in found if asset is not None}
                    return sorted(unique.values(), key=lambda asset: (KIND_ORDER[asset["kind"]], asset["seq"]))

                context_blocks = [source_page[block_id - 1] for block_id in block_ids]
                for context_block in context_blocks:
                    if context_block["type"] == "SCENE_HEADING":
                        heading = context_block["text"].upper()
                        current_location = next((asset for asset in registry.ordered({KIND_LOCATION})
                                                 if asset["key"] in heading or asset["name"] in heading), None)
                content_blocks = [block for block in context_blocks if block["type"] in {"ACTION", "DIALOGUE"}]
                if not content_blocks:
                    continue
                locations = assets("location", KIND_LOCATION) or assets("locations", KIND_LOCATION)
                if not locations and current_location is not None:
                    locations = [current_location]
                source_content = " ".join(block["text"] for block in content_blocks)
                page_shots.append({
                    "source_line": min(block["start_line"] for block in content_blocks),
                    "source_text": source_content,
                    "characters": assets("characters", KIND_CHARACTER), "props": assets("props", KIND_PROP),
                    "locations": locations, "camera": str(obj.get("camera", "")).strip(),
                    "duration": obj.get("duration_seconds"),
                    "action": str(obj.get("action", "")).strip() or source_content,
                    "continuity": str(obj.get("continuity", "")).strip(), "_from_llm": True,
                    "_block_ids": block_ids,
                    "_content_ids": [block_id for block_id in block_ids
                                     if source_page[block_id - 1]["type"] in {"ACTION", "DIALOGUE"}],
                })

            page_shots, covered = _resolve_shot_coverage(page_shots, source_page)

            # Preserve every action paragraph and dialogue exchange if the model misses a block.
            for block_id, block in enumerate(source_page, 1):
                if block["type"] == "SCENE_HEADING":
                    heading = block["text"].upper()
                    current_location = next((asset for asset in registry.ordered({KIND_LOCATION})
                                             if asset["key"] in heading or asset["name"] in heading), None)
                if block_id in covered or block["type"] not in {"ACTION", "DIALOGUE"}:
                    continue
                # A shot already staged this text even though it did not claim the block ID.
                if any(_quotes_line(shot.get("action"), block["text"]) for shot in page_shots):
                    continue
                text = _synthesized_dialogue(block) if block["type"] == "DIALOGUE" else block["text"]
                if not text:
                    continue
                page_shots.append({
                    "source_line": block["start_line"], "source_text": block["text"],
                    "characters": [], "props": [], "locations": [current_location] if current_location else [],
                    "camera": _continuation_camera(page_shots[-1].get("camera") if page_shots else ""),
                    "duration": MIN_SHOT_SECONDS,
                    "action": text,
                    "continuity": "FIXED: preserve the current scene state. VARIABLE: show this action beat or dialogue exchange.",
                    "_from_llm": False,
                })

            page_shots.sort(key=lambda shot: shot["source_line"])
            raw_durations = [max(MIN_H3_SHOT_SECONDS, self._shot_duration(shot)) for shot in page_shots]
            target_seconds = max(target_seconds, sum(raw_durations))
            extra = target_seconds - sum(raw_durations)
            weights = [max(1, len(shot["source_text"].split())) for shot in page_shots]
            weight_total = sum(weights) or 1
            fractional = [extra * weight / weight_total for weight in weights]
            increments = [int(value) for value in fractional]
            remainder = extra - sum(increments)
            for index in sorted(range(len(weights)), key=lambda i: fractional[i] - increments[i], reverse=True)[:remainder]:
                increments[index] += 1
            for shot, duration, increment in zip(page_shots, raw_durations, increments):
                shot["shot"] = len(shots) + 1
                shot["duration"] = duration + increment
                shots.append(shot)
        return shots

    def _shot_brief(self, s, words=60):
        names = lambda items: ", ".join(a["name"] for a in items) or "-"
        return (f"Shot {s['shot']}: location={names(s['locations'])}; characters={names(s['characters'])}; "
                f"props={names(s['props'])}; camera={s['camera'] or '-'}; action={_short_desc(s['action'], words)}")

    # ---------------- heuristic / template paths ----------------
    def _camera_sentence(self, text):
        gear, lens, angle, movement = self._infer_camera_specs(text)
        size = {"Close-up": "a close-up shot", "Wide Angle": "a wide shot", "Medium Shot": "a medium shot",
                "OTS (Over The Shoulder)": "an over-the-shoulder shot", "Dutch Angle": "a Dutch-angle shot",
                "Low Angle": "a low-angle shot", "High Angle": "a high-angle shot"}.get(angle, "an eye-level shot")
        move = {"Static": "The camera holds a locked-off static shot",
                "Tracking": "The camera follows the action in a tracking shot with small amplitude at moderate speed",
                "Dolly In/Out": "The camera pushes in with small amplitude at slow speed",
                "Handheld / Micro-jitter": "The camera shakes slightly",
                "Pan": "The camera pans with small amplitude at slow speed",
                "Tilt": "The camera tilts with small amplitude at slow speed"}.get(movement, "The camera holds a locked-off static shot")
        return f"{move}, framed as {size}."

    def _dialogue_to_d_syntax(self, text, speakers=None):
        """'NAME says [in ...]: "line"' -> 'NAME (S1) says, <d>[English] line</d>' with speaker IDs shared across calls."""
        speakers = speakers if speakers is not None else {}

        def rep(m):
            who = m.group(1).strip()
            sid = speakers.setdefault(who, len(speakers) + 1)
            delivery = (m.group(2) or "")
            line = (m.group("dq") or m.group("sq") or "").strip()
            if "voiceover" in delivery.lower():
                says = "says in an off-screen voiceover"
            elif delivery.strip():
                says = "says " + delivery.strip()
            else:
                says = "says"
            return f'{who} (S{sid}) {says}, <d>[English] {line}</d>'

        return _D_SPOKEN_RE.sub(rep, text)

    def _shots_from_blocks(self, blocks, aspect, registry):
        shots = []
        state = {"loc": None, "chars": set(), "desc": [], "dialogue": [], "curr_char": ""}
        char_assets = registry.ordered({KIND_CHARACTER})
        prop_assets = registry.ordered({KIND_PROP})
        loc_assets = registry.ordered({KIND_LOCATION})

        def finalize():
            if not state["desc"] and not state["dialogue"]:
                return
            desc = " ".join(state["desc"])
            dialogue = " ".join(state["dialogue"])
            text_up = f"{desc} {dialogue}".upper()
            chars = [a for a in char_assets if a["name"] in text_up or a["name"] in state["chars"]]
            props = [a for a in prop_assets if a["name"] in text_up][:max(0, MAX_REF_IMAGES - len(chars))]
            locs = [state["loc"]] if state["loc"] else []
            action = desc
            if dialogue:
                action = f"{desc} {dialogue}".strip()
            shots.append({
                "shot": len(shots) + 1, "characters": chars, "props": props, "locations": locs,
                "camera": self._camera_sentence(desc),
                "duration": None, "action": action, "desc": desc, "dialogue": dialogue, "continuity": "", "_from_llm": False,
            })
            state["desc"], state["dialogue"], state["chars"] = [], [], set()

        for btype, text in blocks:
            if btype == 'SCENE_HEADING':
                finalize()
                heading = text.upper()
                state["loc"] = None
                for a in loc_assets:
                    if a["key"] in heading or a["name"] in heading:
                        state["loc"] = a
                        break
            elif btype == 'SHOT_HEADER':
                finalize()
                state["desc"].append(f"[{text}]")
            elif btype == 'ACTION':
                if state["dialogue"]:
                    finalize()
                state["desc"].append(text)
            elif btype == 'CHARACTER':
                cue = text.strip().upper()
                state["voiceover"] = bool(re.search(r"\((?:V\.?O\.?|O\.?S\.?)\)", cue))
                state["curr_char"] = re.sub(r"\s*\(.*?\)", "", cue).strip()
                state["chars"].add(state["curr_char"])
            elif btype == 'DIALOGUE':
                delivery = " in an off-screen voiceover" if state.get("voiceover") else ""
                state["dialogue"].append(f'{state["curr_char"]} says{delivery}: "{text}"')
        finalize()
        return shots

    def _timeline_shots(self, items, duration, registry):
        """Timeline items -> shots, one per utterance group or instrumental stretch.

        Each shot's length comes from the transcript timing and the whole list adds up to the
        audio's real length, so the rendered video lands on the words that belong to it.
        """
        characters = {a["name"].upper(): a for a in registry.ordered({KIND_CHARACTER})}
        lengths = _timeline_lengths(items, duration)
        upcoming = [""] * len(items)
        nxt = ""
        for index in range(len(items) - 1, -1, -1):
            upcoming[index] = nxt
            if items[index]["kind"] == "dialogue" and items[index].get("cue"):
                nxt = items[index]["cue"]
        shots, last_cue = [], ""
        for index, (item, seconds) in enumerate(zip(items, lengths), 1):
            cue = item.get("cue") or last_cue or upcoming[index - 1]
            last_cue = cue or last_cue
            actor = characters.get(str(cue).upper())
            if item["kind"] == "dialogue":
                spoken = " ".join(line["text"] for line in item["lines"])
                desc = f"{cue} performs the line in the scene."
                dialogue = f'{cue} says: "{spoken}"'
                action = f"{desc} {dialogue}"
                camera = self._camera_sentence(spoken)
            else:
                dialogue = ""
                if index == 1:
                    passage = "The scene opens on the instrumental"
                elif index == len(items):
                    passage = "The closing instrumental plays the scene out"
                elif items[index - 2]["kind"] == "instrumental":
                    passage = "The instrumental continues"
                else:
                    passage = "The instrumental break carries the shot"
                desc = (f"{passage}: the camera holds on the scene and the surrounding action while "
                        "the music plays, with no dialogue.")
                action = desc
                camera = self._camera_sentence("")
            shots.append({
                "shot": index, "characters": [actor] if actor else [], "props": [], "locations": [],
                "camera": camera, "duration": seconds, "action": action, "desc": desc,
                "dialogue": dialogue, "continuity": "", "_from_llm": False,
            })
        return shots

    # ---------------- story: shot list + beats ----------------
    @staticmethod
    def _timestamp(seconds):
        """MM:SS.mmm beat stamp (H3 rewinds every shot to 00:00.000); HH:MM:SS above an hour."""
        total_ms = max(0, int(round(float(seconds) * 1000)))
        hours, remainder = divmod(total_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        whole_seconds, milliseconds = divmod(remainder, 1000)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"
        return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"

    def _beat_fallback(self, shot, idx):
        """One template beat for a shot; the spoken line is added once, never repeated."""
        desc = str(shot.get("desc") or "").strip()
        dialogue = str(shot.get("dialogue") or "").strip()
        if dialogue:
            prose = _short_desc(desc, 26).rstrip(".") if desc else ""
            return f"{prose}. {dialogue}" if prose else dialogue
        action = _short_desc(str(shot.get("action") or ""), 45).rstrip(".")
        return f"{action}." if action.strip() else ""

    def _write_story_beats(self, shots, registry, model, url):
        total = len(shots)
        print(f"     [Story Beats] Total story beats to generate: {total}")
        beats = {}
        for batch_offset in range(0, total, STORY_BEAT_BATCH_SIZE):
            batch = shots[batch_offset:batch_offset + STORY_BEAT_BATCH_SIZE]
            first_shot = batch[0]["shot"]
            last_shot = batch[-1]["shot"]
            print(f"     [Story Beats] Requesting shots {first_shot}–{last_shot} of {total}...")
            compact = []
            for shot in batch:
                compact.append(json.dumps({
                    "shot": shot["shot"], "duration_seconds": self._shot_duration(shot),
                    "location": [a["name"] for a in shot["locations"]],
                    "characters": [a["name"] for a in shot["characters"]],
                    "props": [a["name"] for a in shot["props"]],
                    "camera": shot["camera"], "action": shot["action"], "continuity": shot["continuity"],
                }, ensure_ascii=False))
            system = (STORY_BEATS_PROMPT.replace("@@REGISTRY@@", _registry_listing(registry, words=20))
                      .replace("@@BREAKDOWN@@", "\n".join(compact)))
            res = self._call_ollama_api(system, model, url, timeout=600)
            reported = []
            for line in _strip_fences(res).splitlines():
                line = line.strip()
                if not line:
                    continue
                shot_id, rest = 0, line
                if line.startswith("[") and "]" in line:
                    head, rest = line[1:].split("]", 1)
                    head = head.strip().lower()
                    if head.startswith("shot"):
                        head = head[4:].strip()
                    if head.isdigit():
                        shot_id = int(head)
                if not shot_id:
                    for separator in (":", ")", "."):
                        head, _, rest = line.partition(separator)
                        if head.strip().isdigit():
                            shot_id = int(head.strip())
                            break
                if shot_id:
                    reported.append((shot_id, rest.strip()))
                elif reported and _BEAT_MARKER_RE.search(line):
                    previous_id, previous = reported[-1]
                    reported[-1] = (previous_id, f"{previous} {line}".strip())

            # Some models number each batch from 1, while others preserve global IDs.
            batch_uses_global_ids = any(first_shot <= shot_id <= last_shot and shot_id > len(batch)
                                        for shot_id, _ in reported)
            batch_beats = {}
            for shot_id, text in reported:
                if batch_uses_global_ids:
                    global_id = shot_id if first_shot <= shot_id <= last_shot else None
                else:
                    global_id = first_shot + shot_id - 1 if 1 <= shot_id <= len(batch) else None
                if global_id is not None and global_id not in batch_beats:
                    batch_beats[global_id] = text
            beats.update(batch_beats)
            parsed_ids = sorted(batch_beats)
            parsed_label = f"shot IDs {parsed_ids[0]}–{parsed_ids[-1]}" if parsed_ids else "no shot IDs"
            print(f"     [Story Beats] Parsed {len(batch_beats)}/{len(batch)} beats ({parsed_label}).")
            print(f"     [Story Beats] Batch {first_shot}–{last_shot}: parsed {len(batch_beats)}/{len(batch)} beats.")
        print(f"     [Story Beats] Parsed {len(beats)}/{total} beats total.")
        return beats

    def _build_story(self, blocks, script, char_dict, loc_dict, registry, aspect, use_llm, model, url, tw,
                     audio_timeline=None):
        shots = []
        lengths = []
        if audio_timeline:
            # The wired audio already decided where every line and every instrumental stretch
            # sits, so the timeline replaces the shot breakdown and fixes each shot's length.
            duration = float(audio_timeline[-1]["end"])
            print(f"  ├─ [Audio Timeline] Laying {len(audio_timeline)} timeline beat(s) over {duration:.2f}s of audio...")
            shots = self._timeline_shots(audio_timeline, duration, registry)
            lengths = [shot["duration"] for shot in shots]
            if use_llm:
                print("  ├─ [Audio Timeline] Shot boundaries follow the audio; the LLM writes the beats inside them.")
            print(f"  └─ [Audio Timeline] {len(shots)} shot(s) totalling {sum(lengths)}s for "
                  f"{duration:.2f}s of audio "
                  f"({sum(1 for item in audio_timeline if item['kind'] == 'instrumental')} instrumental)")
        else:
            if use_llm:
                shots = self._breakdown_shots(script, char_dict, loc_dict, registry, aspect, model, url)
                if not shots:
                    print("  ├─ ⚠️ Shot breakdown bypassed or unfulfilling. Switching to heuristic fallback...")
            if not shots:
                print("  ├─ Running rule-based heuristic shot generator...")
                shots = self._shots_from_blocks(blocks, aspect, registry)
            lengths = [max(MIN_H3_SHOT_SECONDS, self._shot_duration(s)) for s in shots]
        if not shots:
            return "", "", []

        # ---- shot list: shared definitions and style, then one [Shot N] line per shot
        out = _assemble_shot_list(shots, registry, tw)
        self._log_assembly("Shot list", out)
        shot_list_text = out["text"]

        # ---- story beats: shot-local micro-beats that reuse the shot list's <Subject N> labels
        raw = {}
        if use_llm:
            print("  ├─ [Story Pipeline] Writing timestamped story beats...")
            raw = self._write_story_beats(shots, registry, model, url)
        rx, lookup = _compile_matcher(registry.ordered())
        speakers = {}
        lines = []
        for idx, shot in enumerate(shots):
            text = raw.get(shot["shot"]) or raw.get(idx + 1) or ""
            if not text:
                if use_llm:
                    print(f"  ├─ ⚠️ No story beat returned for shot {shot['shot']}; using template beat.")
                text = self._beat_fallback(shot, idx)
            parsed = _parse_beat_entries(text)
            entries = _fit_beats_to_duration(parsed, lengths[idx])
            if len(entries) < len(parsed):
                print(f"  ├─ ⚠️ Shot {shot['shot']}: dropped {len(parsed) - len(entries)} beat(s) outside its {lengths[idx]}s duration.")
            rendered = []
            for start, beat in entries:
                beat = self._dialogue_to_d_syntax(_clean_block(beat), speakers)
                beat = _collapse(_SRC_RE.sub("", _apply_labels(beat, rx, lookup, out["labels"])))
                if beat:
                    rendered.append(f"[{self._timestamp(start)}] {beat}")
            if not rendered:
                only = _collapse(self._beat_fallback(shot, idx)) or "the shot holds on the scene"
                rendered = [f"[{self._timestamp(0.0)}] {only}"]
            lines.append(f"[Shot {shot['shot']}] " + " ".join(rendered))
        return shot_list_text, "\n".join(lines), lengths

    # ---------------- logging helpers ----------------
    def _log_assembly(self, label, out):
        if out["used"]:
            table = "; ".join(f"{a['name']} = {out['labels'][_akey(a)]['subject']}/{out['labels'][_akey(a)]['source']}" for a in out["used"])
        else:
            table = "no references (base T2VA format)"
        print(f"  │   🏷️ {label}: {table}")
        for w in out["warnings"]:
            print(f"  │   ⚠️ {label}: {w}")

    def _warn_stray_names(self, text, registry, kinds, owner):
        up = _norm(text)
        for a in registry.ordered(kinds):
            if a["name"] != owner and a["name"] in up:
                print(f"  │   ⚠️ {owner}: prompt text contains '{a['name']}', which a name-based reference filter may attach.")

    def _infer_camera_specs(self, text):
        t = text.lower()
        gear = "Tripod"
        lens = "35mm"
        angle = "Eye Level"
        movement = "Static"

        if "close-up" in t or "close up" in t or "tight" in t:
            lens = "85mm"
            angle = "Close-up"
        elif "wide" in t or "establishing" in t or "landscape" in t:
            lens = "24mm"
            angle = "Wide Angle"
        elif "medium" in t:
            lens = "50mm"
            angle = "Medium Shot"
        elif "over-the-shoulder" in t or "ots" in t:
            lens = "50mm"
            angle = "OTS (Over The Shoulder)"

        if "tracking" in t or "follows" in t or "running" in t:
            gear = "Steadicam"
            movement = "Tracking"
        elif "dolly" in t or "pushes in" in t or "pulls back" in t:
            gear = "Dolly"
            movement = "Dolly In/Out"
        elif "handheld" in t or "shaky" in t or "steps up" in t:
            gear = "Handheld"
            movement = "Handheld / Micro-jitter"
        elif "pan" in t or "swivels" in t:
            movement = "Pan"
        elif "tilt" in t:
            movement = "Tilt"

        if "dutch" in t or "tilted angle" in t:
            angle = "Dutch Angle"
        elif "low angle" in t or "looking up" in t:
            angle = "Low Angle"
        elif "high angle" in t or "looking down" in t:
            angle = "High Angle"

        return gear, lens, angle, movement

    def _wrap_text(self, text, width, indent_spaces):
        words = text.split()
        lines = []
        curr_line = ""
        indent = " " * indent_spaces
        
        for word in words:
            if not curr_line:
                curr_line = word
            elif len(curr_line) + 1 + len(word) <= width:
                curr_line += " " + word
            else:
                lines.append(indent + curr_line)
                curr_line = word
        if curr_line:
            lines.append(indent + curr_line)
        return "\n".join(lines)

    def _to_plain_text(self, blocks):
        formatted_lines = []
        prev_type = None
        
        for btype, text in blocks:
            if btype in ('SCENE_HEADING', 'SHOT_HEADER'):
                if prev_type is not None:
                    formatted_lines.append("")
                formatted_lines.append(text)
                formatted_lines.append("")
            elif btype == 'TRANSITION':
                if prev_type is not None:
                    formatted_lines.append("")
                formatted_lines.append(" " * 52 + text)
                formatted_lines.append("")
            elif btype == 'CHARACTER':
                if prev_type not in ('CHARACTER', 'PARENTHETICAL', 'DIALOGUE'):
                    formatted_lines.append("")
                formatted_lines.append(" " * 28 + text)
            elif btype == 'PARENTHETICAL':
                formatted_lines.append(self._wrap_text(text, 30, 20))
            elif btype == 'DIALOGUE':
                formatted_lines.append(self._wrap_text(text, 36, 14))
            elif btype == 'ACTION':
                if prev_type is not None and prev_type != 'ACTION':
                    formatted_lines.append("")
                formatted_lines.append(self._wrap_text(text, SCREENPLAY_LINE_WIDTH, 0))
            
            prev_type = btype

        return "\n".join(formatted_lines)

    def _to_print_html(self, blocks):
        """Render a paginated US Letter screenplay preview with screenplay typography."""
        lines = self._to_plain_text(blocks).splitlines()
        pages = []
        start = 0
        while start < len(lines):
            end = min(start + SCREENPLAY_PAGE_LINES, len(lines))
            if end < len(lines):
                lower_bound = max(start + 1, end - 8)
                safe_break = next((i for i in range(end, lower_bound - 1, -1)
                                   if lines[i - 1].strip() == ""), None)
                if safe_break is not None:
                    end = safe_break
            pages.append(lines[start:end])
            start = end
        if not pages:
            pages = [[]]

        css = (
            "<style>"
            ".screenplay-pages{display:flex;flex-direction:column;align-items:center;gap:16px;background:#303030;padding:12px 0;}"
            ".screenplay-page{position:relative;box-sizing:border-box;width:8.5in;height:11in;min-height:11in;"
            "padding:1in 1in 1in 1.5in;background:#fff;color:#111;overflow:hidden;box-shadow:0 2px 12px #0008;"
            "font:12pt/11.75pt 'Courier Prime','Courier New',Courier,monospace;}"
            ".screenplay-page-number{position:absolute;top:.5in;right:1in;font:12pt/12pt 'Courier Prime','Courier New',Courier,monospace;}"
            ".screenplay-page-content{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;line-height:11.75pt;}"
            "@media print{.screenplay-pages{display:block;padding:0;background:#fff}.screenplay-page{margin:0;box-shadow:none;page-break-after:always;break-after:page}.screenplay-page:last-child{page-break-after:auto;break-after:auto}}"
            "@page{size:letter;margin:0}"
            "</style>"
        )
        rendered = [css, '<div class="screenplay-pages">']
        for page_number, page_lines in enumerate(pages, 1):
            page_text = html.escape("\n".join(page_lines))
            rendered.append(
                f'<section class="screenplay-page"><div class="screenplay-page-number">{page_number}.</div>'
                f'<pre class="screenplay-page-content">{page_text}</pre></section>'
            )
        rendered.append("</div>")
        return "".join(rendered)

    def _to_fountain(self, blocks):
        out = []
        prev = None
        for btype, text in blocks:
            if btype in ('SCENE_HEADING', 'SHOT_HEADER'):
                out.append(f"\n{text}\n")
            elif btype == 'TRANSITION':
                out.append(f"\n> {text}\n")
            elif btype == 'CHARACTER':
                out.append(f"\n{text}")
            elif btype == 'PARENTHETICAL':
                out.append(f"{text}")
            elif btype == 'DIALOGUE':
                out.append(f"{text}")
            elif btype == 'ACTION':
                if prev != 'ACTION':
                    out.append(f"\n{text}")
                else:
                    out.append(text)
            prev = btype
        return "\n".join(out).strip()

    def _to_html(self, blocks):
        html = [
            '<div style="font-family: \'Courier New\', Courier, monospace; background: #181818; color: #e0e0e0; padding: 12px; line-height: 1.35; font-size: 11px; border: 1px solid #333; border-radius: 4px; box-sizing: border-box;">'
        ]
        
        for btype, text in blocks:
            if btype == 'SCENE_HEADING':
                html.append(f'<div style="font-weight: bold; color: #4cc9f0; margin-top: 14px; margin-bottom: 4px; text-transform: uppercase;">{text}</div>')
            elif btype == 'SHOT_HEADER':
                html.append(f'<div style="font-weight: bold; color: #f72585; margin-top: 10px; margin-bottom: 4px; text-transform: uppercase;">{text}</div>')
            elif btype == 'TRANSITION':
                html.append(f'<div style="text-align: right; margin-top: 8px; margin-bottom: 8px; font-weight: bold; color: #b5179e;">{text}</div>')
            elif btype == 'CHARACTER':
                html.append(f'<div style="margin-left: 38%; margin-top: 10px; margin-bottom: 2px; font-weight: bold; color: #70e000; text-transform: uppercase;">{text}</div>')
            elif btype == 'PARENTHETICAL':
                html.append(f'<div style="margin-left: 28%; max-width: 45%; margin-bottom: 2px; color: #aaaaaa; font-style: italic;">{text}</div>')
            elif btype == 'DIALOGUE':
                html.append(f'<div style="margin-left: 20%; max-width: 55%; margin-bottom: 8px; color: #ffffff;">{text}</div>')
            elif btype == 'ACTION':
                html.append(f'<div style="margin-top: 4px; margin-bottom: 4px; width: 100%; color: #cccccc;">{text}</div>')
                
        html.append('</div>')
        return "".join(html)

class ShotLengthSelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1}),
            },
            "optional": {
                "shot_length": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("selected_length",)
    INPUT_IS_LIST = (False, True)
    FUNCTION = "select_length"
    CATEGORY = "triXope/writing"

    def select_length(self, shot_index, shot_length=None):
        # Unpack shot_index if ComfyUI passed it wrapped inside a list
        if isinstance(shot_index, (list, tuple)):
            shot_index = shot_index[0] if len(shot_index) > 0 else 1

        if not shot_length:
            return (5,)

        # Unpack shot_length if nested inside an extra outer list
        if isinstance(shot_length, (list, tuple)) and len(shot_length) > 0 and isinstance(shot_length[0], (list, tuple)):
            shot_length = shot_length[0]

        idx = max(1, int(shot_index)) - 1
        if idx < len(shot_length):
            selected = shot_length[idx]
        else:
            selected = shot_length[-1]

        if isinstance(selected, (list, tuple)):
            selected = selected[0]

        final_val = max(5, min(9999, int(selected)))
        return (final_val,)

class ShotLengthBufferNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stored_array": ("STRING", {
                    "default": "[5]",
                    "multiline": True
                }),
            },
            "optional": {
                "shot_length": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("shot_length",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = (False, True)
    FUNCTION = "hold_array"
    CATEGORY = "triXope/writing"

    def hold_array(self, stored_array, shot_length=None):
        # Unpack stored_array if wrapped in a list by ComfyUI
        if isinstance(stored_array, (list, tuple)):
            stored_array = stored_array[0] if len(stored_array) > 0 else "[5]"

        # Check if a live shot_length array is connected
        if shot_length is not None and len(shot_length) > 0:
            if isinstance(shot_length[0], (list, tuple)):
                active_list = [int(x) for x in shot_length[0]]
            else:
                active_list = [int(x) for x in shot_length]

            active_list = [max(5, min(1000008, x)) for x in active_list]
            
            # Send updated JSON to frontend to update stored_array widget
            return {
                "ui": {"stored_array": [json.dumps(active_list)]},
                "result": (active_list,)
            }

        # Fallback to last known saved array when disconnected
        try:
            fallback_list = json.loads(stored_array)
            if not isinstance(fallback_list, list) or len(fallback_list) == 0:
                fallback_list = [5]
        except Exception:
            fallback_list = [5]

        fallback_list = [max(5, min(1000008, int(x))) for x in fallback_list]
        return (fallback_list,)

# --- moved from character_craft.py ------------------------------------------
import sys

RACE_LIST = [
    "Australoid", "Bugbear", "Centaur", "Dhampir", "Dragonborn", "Dwarf", "Elf", "Extraterrestrial", "Fairy", "Giant", "Gnome", "Goblin", "Goliath", "Half-Elf", "Half-Orc", "Halfling", "Hobgoblin", "Human - Mongoloid/Asian", "Human - Negroid/Black",
    "Human - White/Caucasian", "Kobold", "Lizardfolk", "Minotaur", "Orc", "Satyr", "Tiefling", "Vampire", "Werewolf", "Other/Unknown"
]

ROLE_LIST = [
    "Protagonist", "Protagonist's Helper", "Sidekick", "Guardian", "Mentor",
    "Catalyst", "Impact", "Antagonist", "Antagonist's Helper", "Skeptic",
    "Obstacle", "Goal", "Minor Character", "Other/Unknown"
]

EYECOLOR_LIST = [
    "Amber", "Blue", "Brown", "Gray", "Green", "Hazel", "Red", "Yellow",
    "Black", "Other/Unknown"
]

SKINTONE_LIST = [
    "Fair", "Light", "Medium", "Olive", "Tan", "Brown", "Dark Brown",
    "Black", "Other/Unknown"
]

BODYTYPE_LIST = [
    "Athletic", "Ectomorph", "Mesomorph", "Endomorph", "Rectangle", "Triangle", "Spoon",
    "Hourglass", "Top Hourglass", "Bottom Hourglass", "Inverted Triangle",
    "Round or Oval", "Diamond", "Other/Unknown"
]

POSTURE_LIST = [
    "Healthy", "Kyphosis", "Flat Back", "Swayback (Lordosis)",
    "Forward Neck or Head", "Other/Unknown"
]

FACESHAPE_LIST = [
    "Oval", "Round", "Square", "Diamond", "Heart", "Pear", "Oblong",
    "Other/Unknown"
]

HAIRSTYLE_LIST = [
    "Afro", "Asymmetric Cut", "Beehive", "Bald", "Bangs", "Big Hair", "Blowout",
    "Bob Cut", "Bouffant", "Bowl Cut", "Braid", "Brush Cut", "Bun", "Bunches",
    "Burr", "Businessman", "Butch Cut", "Buzz Cut", "Caesar Cut", "Chignon",
    "Chonmage", "Comb Over", "Comma Hair", "Conk", "Cornrows", "Crew Cut",
    "Crop", "Crown Braid", "Croydon Facelift", "Curtained Hair", "Devilock",
    "Dido Flip", "Double Buns", "Dreadlocks", "Duck's Ass", "Eton Crop",
    "Extensions", "Fade", "Fallera Hairdo", "Fauxhawk", "Feather Cut",
    "Feathered Hair", "Finger Waves", "Fishtail Hair", "Flattop",
    "Flipped-up Ends", "Fontange", "French Braid", "French Twist",
    "Fringe (Bangs)", "Frosted Tips", "Full Crown", "Half Crown", "Half Updo",
    "Harvard Clip", "High and Tight", "Highlights", "Hime Cut",
    "Historic Hairstyle", "Hi-Top Fade", "Induction Cut", "Ivy League",
    "Jewfro", "Jheri Curl", "Layered Hair", "Liberty Spikes", "Line Up", "Lob",
    "Long Hair", "Marcel Waves", "Mod Cut", "Mohawk", "Mop-Top", "Mullet",
    "Natural", "Odango", "Oseledets", "Pageboy", "Payot", "Perm", "Pigtails",
    "Pixie Cut", "Pompadour", "Ponyhawk", "Ponytail", "Princeton",
    "Professional Cut", "Psychobilly Wedge", "Queue", "Quiff", "The Rachel",
    "Rattail", "Razor Cut", "Regular Haircut", "Regular Taper Cut", "Ringlets",
    "Shag Cut", "Shape-Up", "Shingle Bob", "Short Back and Sides",
    "Short Brush Cut", "Short Hair", "Slicked-Back", "Spiky Hair",
    "Standard Haircut", "Step Cut", "Surfer Hair", "Taper Cut", "Tail on Back",
    "Tonsure", "Undercut", "Updo", "Waves", "Weave", "Wings", "Other/Unknown"
]

HAIRCOLOR_LIST = [
    "Auburn", "Black", "Blonde", "Blue", "Brown", "Brunette", "Copper", "Gray",
    "Green", "None/Bald", "Pink", "Purple", "Red", "Silver", "White", "Other/Unknown"
]

WARDROBE_LIST = [
    "Streetwear", "Ethnic Fashion", "Formal Office Wear", "Business Casual",
    "Corporate Powerhouse", "Evening Black Tie", "Sports Wear", "Girly",
    "Rockstar", "Preppy", "Skateboarder", "Jock", "Maternity", "Goth",
    "Lolita", "Gothic Lolita", "Hip-Hop", "Chave Culture", "Kawaii",
    "Cowgirl/boy", "Lagenlook", "Fashion", "Scene Fashion", "Girl Next Door",
    "Casual", "Geeky", "Military", "Retro", "Flapper (20’s look)", "Tomboy",
    "Garconne Look", "Vacation (Resort)", "Artsy Fashion", "Grunge", "Punk",
    "Boho/Bohemian", "Psychedelic", "Cosplay", "Haute Couture", "Modest",
    "Rave", "Flamboyant", "Adventurer", "Ankara", "Other/Unknown"
]

LIFESKILL_LIST = [
    "Active Listening", "Adaptability", "Blending In", "Breath Control",
    "Charm", "Clairvoyance", "Communication", "Creativity", "Customer +",
    "Decisiveness", "Empathy", "Senses +", "Foraging", "Haggling",
    "Hospitality", "Humorous", "Leadership", "Lip-Reading", "Logical",
    "Lying", "Management", "Memorization", "Mimicking", "Multitasking",
    "Musicality", "Navigation", "Organization", "Problem-Solving", "Psychic",
    "Psychotic", "Public Speaking", "Regeneration", "Self-Defense",
    "Sharpshooting", "Sleight-of-Hand", "Strength", "Survival Skills",
    "Swifted-Footed", "Teamwork", "Telekinesis", "Throws Voice", "Trustworthy", "Other/Unknown"
]

LIFESTRENGTH_LIST = [
    "Brave", "Reckless", "Idealistic", "Naive", "Funny", "Unserious",
    "Confident", "Arrogant", "Careful", "Timid", "Virtuous", "Rigid",
    "Open-Minded", "Overthinking", "Determined", "Stubborn", "Loyal",
    "Insular", "Humble", "Passive", "Patient", "Unproductive", "Energetic",
    "Exhausting", "Independent", "Poor Teamwork", "Imaginative",
    "Unrealistic", "Charismatic", "Self-Absorbed", "Emotional", "Illogical",
    "Proud", "Vain", "Driven", "Greedy", "Logical", "Low Empathy", "Honest",
    "Blunt", "Decisive", "Impulsive", "Leadership", "Bossy", "Focused",
    "Addictive", "Curious", "Aimless", "Kind", "Unassertive", "Connoisseur",
    "Glutton", "Content", "Unambitious", "Principled", "Intolerant",
    "Charming", "Deceitful", "Adaptable", "Unstructured", "Realistic",
    "Cynical", "Other/Unknown"
]

HOBBY_LIST = [
    "None", "Animal Watch", "Art", "Astronomy", "Billiards", "Bowling", "Camping",
    "Collecting", "Cooking", "Crafts", "Dance", "Darts", "Explosives",
    "Fishing", "Gaming", "Gambling", "Gardening", "Hunting", "Magic", "Movies",
    "Music", "Parkour", "Photography", "Puzzles", "Racing", "Reading",
    "Sewing", "Skating", "Sports/Fitness", "Travelling", "Writing",
    "Other/Unknown"
]

MUSICALINTEREST_LIST = [
    "None", "Alternative", "Blues", "Children's Music", "Classical", "Comedy",
    "Country", "Dance", "Disco", "Easy Listening", "Electronic", "Enka",
    "Folk", "Hip-Hop/Rap", "Holiday", "Indie", "Industrial", "Inspirational",
    "Instrumental", "Jazz", "Latin", "Opera", "Pop", "Progressive",
    "R&B/Soul", "Reggae", "Rock", "World", "Other/Unknown"
]

BADHABIT_LIST = [
    "None", "Abusive", "Alcoholic", "Avoids Eyes", "Belches", "Breaks Promises",
    "Cheats", "Chews Tobacco", "Chews Loudly", "Cracks Joints", "Daydreams",
    "Farts", "Fidgets", "Freeloads", "Gambles", "Gossips", "Grinds Teeth",
    "Hoarder", "Hums", "Indecisive", "Leaves Lid Up", "Licks Lips", "Litters",
    "Lies", "Mimicking", "Multitasks", "Nail Bites", "Negative",
    "Night Snacks", "Nose Picks", "Obsesses", "Overeats", "Over-medicates",
    "Oversleeps", "Overspends", "Picks Scabs", "Picks Teeth", "Plays w/ Hair",
    "Pops Gum", "Pops Zits", "Procrastinates", "Public Affection", "Sedentary",
    "Slang", "Slouches", "Smokes", "Speeds", "Spits", "Steals", "Stereotypes",
    "Swears", "Taps Foot", "Tardy", "Views Porn", "Violent", "Other/Unknown"
]

FAITH_LIST = [
    "Non-Spiritual", "Atheism", "Baha'i", "Buddhism", "Candomble", "Christianity", "Hinduism",
    "Islam", "Jainism", "Jehovah's Witness", "Judaism", "Mormonism",
    "Paganism", "Rastafari", "Santeria", "Shinto", "Sikhism", "Spiritualism",
    "Taoism", "Unitarianism", "Zoroastrianism", "Other/Unknown"
]

PET_LIST = [
    "None", "Alpaca", "Ant Farm", "Bird", "Cat", "Dog", "Ferret", "Fish",
    "Frog/Toad", "Gecko", "Gerbil", "Goat", "Guinea Pig", "Hamster", "Hedgehog",
    "Hermit Crab", "Horse", "Iguana", "Mantis", "Mouse", "Newt", "Pig",
    "Rabbit", "Rat", "Salamander", "Sheep", "Skunk", "Snake", "Spider",
    "Stick-Bug", "Turtle", "Other/Unknown"
]

PERSONALITYTYPE_LIST = [
    "The Architect/Thinker", "The Champion", "The Commander", "The Composer",
    "The Counselor", "The Craftsman", "The Dynamo/Doer", "The Healer/Idealist",
    "The Inspector", "The Mastermind", "The Performer",
    "The Protector/Nurturer", "The Provider", "The Supervisor",
    "The Teacher/Giver", "The Visionary", "Other/Unknown"
]

PHOBIA_LIST = [
    "None", "Achluophobia - darkness", "Acrophobia - heights",
    "Aerophobia - flying", "Algophobia - pain", "Alektorophobia - chickens",
    "Agoraphobia - crowds", "Aichmophobia - needles",
    "Amaxophobia - riding in a car", "Androphobia - men",
    "Anginophobia - choking", "Anthophobia - flowers",
    "Anthropophobia - people", "Aphenphosmphobia - being touched",
    "Arachnophobia - spiders", "Arithmophobia - numbers",
    "Astraphobia - thunder & lightning", "Ataxophobia - disorder",
    "Atelophobia - imperfection", "Atychiphobia - failure",
    "Autophobia - being alone", "Bacteriophobia - bacteria",
    "Barophobia - gravity", "Bathmophobia - steep slopes",
    "Batrachophobia - amphibians", "Belonephobia - pins & needles",
    "Bibliophobia - books", "Botanophobia - plants", "Cacophobia - ugliness",
    "Catagelophobia - being ridiculed", "Catoptrophobia - mirrors",
    "Chionophobia - snow", "Chromophobia - colors",
    "Chronomentrophobia - clocks", "Claustrophobia - confined spaces",
    "Coulrophobia - clowns", "Cyberphobia - computers", "Cynophobia - dogs",
    "Dendrophobia - trees", "Dentophobia - dentists", "Domatophobia - houses",
    "Dystychiphobia - accidents", "Ecophobia - home", "Elurophobia - cats",
    "Entomophobia - insects", "Ephebiphobia - teenagers",
    "Equinophobia - horses", "Gamophobia - marriage", "Genuphobia - knees",
    "Glossophobia - public speaking", "Gynophobia - women", "Heliophobia - sun",
    "Hemophobia - blood", "Herpetophobia - reptiles", "Hydrophobia - water",
    "Hypochondria - illness", "Iatrophobia - doctors",
    "Insectophobia - insects", "Koinoniphobia - rooms full of people",
    "Leukophobia - color white", "Lilapsophobia - tornadoes & hurricanes",
    "Lockiophobia - childbirth", "Mageirocophobia - cooking",
    "Megalophobia - large things", "Melanophobia - color black",
    "Microphobia - small things", "Mysophobia - dirt & germs",
    "Necrophobia - death or dead things", "Noctiphobia - night",
    "Nosocomephobia - hospitals", "Nyctophobia - the dark",
    "Obesophobia - gaining weight", "Octophobia - figure 8",
    "Ombrophobia - rain", "Ophidiophobia - snakes", "Ornithophobia - birds",
    "Papyrophobia - paper", "Pathophobia - disease", "Pedophobia - children",
    "Philophobia - love", "Phobophobia - phobias", "Podophobia - feet",
    "Pogonophobia - beards", "Porphyrophobia - color purple",
    "Pteridophobia - ferns", "Pteromerhanophobia - flying", "Pyrophobia - fire",
    "Samhainophobia - Halloween", "Scolionophobia - school",
    "Selenophobia - the moon", "Sociophobia - social evaluation",
    "Somniphobia - sleep", "Tachophobia - speed", "Technophobia - technology",
    "Tonitrophobia - thunder", "Trypanophobia - needles or injections",
    "Venustraphobia - beautiful women", "Verminophobia - germs",
    "Wiccaphobia - witches & witchcraft", "Xenophobia - strangers or foreigners",
    "Zoophobia - animals", "Other/Unknown"
]

OCCUPATION_LIST = [
    "Unemployed", "Computers & Technology", "Health Care", "Education & Social Services",
    "Arts & Communication", "Trades & Transportation",
    "Management, Business, & Finance", "Architecture & Engineering",
    "Farming & Agriculture", "Science", "Hospitality & Tourism",
    "Law & Enforcement", "Other/Unknown"
]

INCOME_LIST = [
    "$0 - $10,000", "$10,000 - $20,000", "$20,000 - $30,000",
    "$30,000 - $40,000", "$40,000 - $50,000", "$50,000 - $60,000",
    "$60,000 - $70,000", "$70,000 - $80,000", "$80,000 - $90,000",
    "$90,000 - $100,000", "$100,000 - $200,000", "$200,000 - $300,000",
    "$300,000 - $400,000", "$400,000 - $500,000", "$500,000 - $600,000",
    "$600,000 - $700,000", "$700,000 - $800,000", "$800,000 - $900,000",
    "$900,000 - $1,000,000", "greater than $1,000,000", "Other/Unknown"
]

SEXUALORIENTATION_LIST = [
    "Straight", "Gay", "Bi-Sexual", "Other/Unknown"
]

MARITALSTATUS_LIST = [
    "Married", "Single", "Divorced", "Separated", "Widowed", "Other/Unknown"
]

RELATIONTOPROTAGONIST_LIST = [
    "Self", "Grandparent", "Parent", "Sibling", "Child", "Cousin",
    "Aunt/Uncle", "Niece/Nephew", "Significant Other", "Other Relative",
    "Friend", "Nemesis", "Other/Unknown"
]

FIRSTAPPEARANCE_LIST = [
    "Act I (Beginning)", "Act II (Middle)", "Act III (End)", "Exposition",
    "Inciting Incident", "Rising Action", "Climax", "Falling Action",
    "Moment of Final Suspense", "Resolution", "Other/Unknown"
]


class TriXopeCharacterCraft:
    """
    A custom node for ComfyUI to craft character details.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off"}),
                "Name": ("STRING", {"multiline": False, "default": ""}),
                "Age": ("INT", {"default": 24, "min": 0, "max": 10000, "step": 1}),
                "Race": (RACE_LIST, {"default": RACE_LIST[0]}),
                "Role": (ROLE_LIST, {"default": ROLE_LIST[0]}),
                "Eye Color": (EYECOLOR_LIST, {"default": EYECOLOR_LIST[0]}),
                "Skin Tone": (SKINTONE_LIST, {"default": SKINTONE_LIST[0]}),
                "Body Type": (BODYTYPE_LIST, {"default": BODYTYPE_LIST[0]}),
                "Posture": (POSTURE_LIST, {"default": POSTURE_LIST[0]}),
                "Face Shape": (FACESHAPE_LIST, {"default": FACESHAPE_LIST[0]}),
                "Hair Style": (HAIRSTYLE_LIST, {"default": HAIRSTYLE_LIST[0]}),
                "Hair Color": (HAIRCOLOR_LIST, {"default": HAIRCOLOR_LIST[0]}),
                "Wardrobe": (WARDROBE_LIST, {"default": WARDROBE_LIST[0]}),
                "Life Skill": (LIFESKILL_LIST, {"default": LIFESKILL_LIST[0]}),
                "Life Strength": (LIFESTRENGTH_LIST, {"default": LIFESTRENGTH_LIST[0]}),
                "Hobby": (HOBBY_LIST, {"default": HOBBY_LIST[0]}),
                "Musical Interest": (MUSICALINTEREST_LIST, {"default": MUSICALINTEREST_LIST[0]}),
                "Bad Habit": (BADHABIT_LIST, {"default": BADHABIT_LIST[0]}),
                "Faith": (FAITH_LIST, {"default": FAITH_LIST[0]}),
                "Pet": (PET_LIST, {"default": PET_LIST[0]}),
                "Personality Type": (PERSONALITYTYPE_LIST, {"default": PERSONALITYTYPE_LIST[0]}),
                "Phobia": (PHOBIA_LIST, {"default": PHOBIA_LIST[0]}),
                "Occupation": (OCCUPATION_LIST, {"default": OCCUPATION_LIST[0]}),
                "Income": (INCOME_LIST, {"default": INCOME_LIST[0]}),
                "Sexual Orientation": (SEXUALORIENTATION_LIST, {"default": SEXUALORIENTATION_LIST[0]}),
                "Marital Status": (MARITALSTATUS_LIST, {"default": MARITALSTATUS_LIST[0]}),
                "Relation To Protagonist": (RELATIONTOPROTAGONIST_LIST, {"default": RELATIONTOPROTAGONIST_LIST[0]}),
                "First Appearance": (FIRSTAPPEARANCE_LIST, {"default": FIRSTAPPEARANCE_LIST[0]}),
                "Additional Details": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("character_info",)
    FUNCTION = "generate_character"
    CATEGORY = "triXope/writing"

    def generate_character(self, **kwargs):
        if not kwargs.get("enabled", True):
            return ("",)
        name = kwargs.get("Name")
        age = kwargs.get("Age")
        race = kwargs.get("Race")
        role = kwargs.get("Role")
        eye_color = kwargs.get("Eye Color")
        skin_tone = kwargs.get("Skin Tone")
        body_type = kwargs.get("Body Type")
        posture = kwargs.get("Posture")
        face_shape = kwargs.get("Face Shape")
        hair_style = kwargs.get("Hair Style")
        hair_color = kwargs.get("Hair Color")
        wardrobe = kwargs.get("Wardrobe")
        life_skill = kwargs.get("Life Skill")
        life_strength = kwargs.get("Life Strength")
        hobby = kwargs.get("Hobby")
        musical_interest = kwargs.get("Musical Interest")
        bad_habit = kwargs.get("Bad Habit")
        faith = kwargs.get("Faith")
        pet = kwargs.get("Pet")
        personality_type = kwargs.get("Personality Type")
        phobia = kwargs.get("Phobia")
        occupation = kwargs.get("Occupation")
        income = kwargs.get("Income")
        sexual_orientation = kwargs.get("Sexual Orientation")
        marital_status = kwargs.get("Marital Status")
        relation_to_protagonist = kwargs.get("Relation To Protagonist")
        first_appearance = kwargs.get("First Appearance")
        details = kwargs.get("Additional Details")

        output_text = (
            f"Name: {name}\n"
            f"Age: {age}\n"
            f"Race: {race}\n"
            f"Role: {role}\n"
            f"Eye Color: {eye_color}\n"
            f"Skin Tone: {skin_tone}\n"
            f"Body Type: {body_type}\n"
            f"Posture: {posture}\n"
            f"Face Shape: {face_shape}\n"
            f"Hair Style: {hair_style}\n"
            f"Hair Color: {hair_color}\n"
            f"Wardrobe: {wardrobe}\n"
            f"Life Skill: {life_skill}\n"
            f"Life Strength: {life_strength}\n"
            f"Hobby: {hobby}\n"
            f"Musical Interest: {musical_interest}\n"
            f"Bad Habit: {bad_habit}\n"
            f"Faith: {faith}\n"
            f"Pet: {pet}\n"
            f"Personality Type: {personality_type}\n"
            f"Phobia: {phobia}\n"
            f"Occupation: {occupation}\n"
            f"Income: {income}\n"
            f"Sexual Orientation: {sexual_orientation}\n"
            f"Marital Status: {marital_status}\n"
            f"Relation To Protagonist: {relation_to_protagonist}\n"
            f"First Appearance: {first_appearance}\n"
            f"Additional Details: {details}"
        )
        
        return (output_text,)

# --- moved from story_select.py ---------------------------------------------
import sys

GENRE_LIST = [
    "Absurdist/Surreal/Whimsical", "Action", "Adventure", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Historical (non fiction)", "Historical (fiction)", "Horror", "Magical Realism",
    "Mystery", "Paranoid", "Philosophical", "Political", "Romance", "Saga", "Satire",
    "Science Fiction", "Slice of Life", "Social", "Speculative", "Thriller", "Urban",
    "Western", "Other"
]

STRUCTURE_LIST = [
    "Milieu - a strange and outlandish setting",
    "Idea - a question is raised and answered",
    "Character - the study of ones character",
    "Event - world change through cataclysmic events",
    "Other"
]

FLOW_LIST = [
    "Linear / Chronological",
    "Nonlinear / Unsequential",
    "Mirror (Forward then reverse or vice versa)",
    "Other"
]

TIME_PERIOD_LIST = [
    "Stone Age", "Ancient Egypt", "Ancient China", "Bronze Age (c.3000 BC – c.1050 BC)",
    "Iron Age (c.1050 BC – c.500 AD)", "Archaic Greece (776 BC – 480 BC)",
    "Persian Empires (550 BC – 651 AD)", "Protohistory", "Ancient History",
    "Classical Antiquity (480 BC – 476 AD)", "Post-Classical History",
    "Middle Ages (Europe, 476 – 1453)", "Dark Ages (Europe, 476 – 800)",
    "Viking Age (Scandinavia, Europe, 793 – 1066)",
    "The Renaissance (Europe, c.1300 – c.1600)",
    "Age of Discovery (or Exploration) (Europe, c.1400 – 1770)",
    "Baroque (New World, 1600–1750)", "Golden Age of Piracy (1650 – 1730)",
    "Age of Enlightenment (Europe, 18th century)", "Romantic era (1770–1850)",
    "Napoleonic era (1799–1815)", "Industrial Revolution (18th and 19th centuries)",
    "Victorian era (United Kingdom, 1837 – 1901)", "Machine Age (1880–1945)",
    "Age of Oil (after 1901)", "Edwardian era (United Kingdom, 1901 – 1914)",
    "World War I (1914–1918)", "Roaring Twenties (1920–1929)",
    "Great Depression (1929–1939)", "Jet Age (1940s)", "World War II (1939–1945)",
    "Atomic Age (after 1945)", "Cold War (1945–1991)", "Korean War (1950–1953)",
    "Digital Revolution (1950s–present)", "Vietnam War (1955–1975)",
    "Space Age (after 1957)", "The Multimedia Age (1987–present)",
    "Bosnian War (1992–1995)", "The Social Age (1996–present)",
    "War in Afghanistan (2001–2014)", "War in Iraq (2003–2011)",
    "The Big Data age (2001–present)", "The Future", "Other/Unknown"
]

SETTING_LIST = [
    "Abandoned Factory", "Abandoned Mine", "Abandoned Office", "Abandoned Quarry", "Abandoned Road",
    "Abandoned Subway", "Abandoned Warehouse", "Abbey", "Academy", "Active Volcano",
    "Aerie", "Aero Club", "Aerodrome", "Agricultural Equipment Center",
    "Agricultural Experiments Facility", "Agricultural Service Center",
    "Air Cargo Terminal", "Air Passenger Terminal", "Air Purification Plant",
    "Air Show", "Airborne Aircraft", "Airborne Research Facility", "Aircraft Bunker",
    "Aircraft Carrier", "Aircraft Charter", "Aircraft Graveyard", "Aircraft Museum",
    "Aircraft on the Ground", "Aircraft Repair Facility", "Aircraft Service Center",
    "Aircraft Test & Simulations Facility", "Airline Offices", "Airpark",
    "Airport Control Tower", "Airport Hangar", "Airship", "Airstrip",
    "Alarm Monitoring Center", "Alter", "Amphibious Vehicle", "Amphitheatre",
    "Amusement Park", "An Abandoned Ship", "Anchorage", "Anchorage or Mooring",
    "Ancient Battleground", "Ancient Ruins", "Animal Hospital", "Animal Shelter",
    "Animal Trail", "Animal Trainer", "Annex", "Anomalous Mine",
    "Antique Aircraft Airfield", "Apartment Tower", "Aquarium", "Aquatic Preserve",
    "Aqueduct", "Arch Bridge", "Archaeology Dig Site", "Archipelago",
    "Arctic Fishing Cabin", "Arena", "Armory", "Arsenal", "Art Gallery",
    "Art Institute", "Art Museum", "Art School", "Arctic Expedition HQ",
    "Auction House", "Audio Recording Studio", "Auto Body Shop", "Auto Dealership",
    "Auto Parts Shop", "Automobile Graveyard", "Automobile Manufacturing Complex",
    "Automobile Museum", "Automobile Race Track", "Automobile Tunnel", "Aviary",
    "Aviation Museum", "Badlands", "Bail Bond House", "Ballet Studio", "Ballroom",
    "Banquet Facility", "Bar", "Barbershop", "Barge", "Barn", "Barracks",
    "Baseball Park", "Baseball Stadium", "Basement", "Basin", "Basketball Arena",
    "Bathhouse", "Battle Field", "Battleship", "Bay", "Bayou", "Beach House",
    "Beached Shipwreck", "Beam Bridge", "Beauty Salon", "Bed & Breakfast",
    "Beneath a Bridge", "Bermuda Triangle", "Bible Institute", "Billiard Hall",
    "Bluffs", "Boat Dealership", "Boat House", "Boat Ramp", "Boat Taking on Water",
    "Bog", "Bookshop", "Border Crossing", "Botanical Garden", "Bowling Alley",
    "Box Canyon", "Brewery", "Bridge", "Broadcast Tower", "Bull Fighting Ring",
    "Bungalow", "Bunker", "Burial Mound", "Bus Depot", "Bus Station", "Bus Stop",
    "Bus Terminal", "Busy Intersection", "Butcher Shop", "Butte", "Cabaret/Supper Club",
    "Cabin", "Cable Company", "Cable Television Company", "Cable-Stayed Bridge",
    "Cafeteria", "Caldera", "Campground", "Camping Area", "Canal", "Candy Factory",
    "Cantilever Bridge", "Canyon Mouth", "Cape", "Car Wash", "Cargo Aircraft",
    "Cargo Container", "Cargo Plane", "Cargo Terminal", "Carnival", "Casino",
    "Castle", "Cathedral", "Causeway", "Cave", "Caverns", "Cemetery", "Cenote",
    "Central Park", "Chalet", "Chamber of Commerce", "Chapel", "Chasm", "Checkpoint",
    "Chemical Dump", "Chemical Plant", "Chemical Storage Facility",
    "Chemical Tanker Ship", "Chicken Farm", "Children's Hospital", "Children's Museum",
    "Chinese Junk (Sailing Ship)", "Chiropractic Clinic", "Church", "Church Camp",
    "Church Tower", "Cinema", "Circus", "Citadel", "Citrus Grove", "City",
    "City Center", "City Hall", "Civic Center", "Clandestine Drug Lab", "Cliff",
    "Clinic", "Clock Tower", "Club House", "Coal Mine", "Coal Processing Plant",
    "Coal-Fired Power Plant", "Coastal Game Preserve", "Coastal Island",
    "Coastal Marsh", "Coastal Resort", "Coastal Shallows", "Coastline",
    "Cocktail Lounge", "Coffee House", "Collapsed Bridge", "Collapsed Mine",
    "Collapsed Tunnel", "Collection Agency", "College", "Colonial Style Residence",
    "Combination Office/Warehouse", "Commercial Construction Area",
    "Community Building", "Community College Complex", "Commuter Train",
    "Concert Hall", "Concrete, Sand, & Gravel Company", "Condominium Tower",
    "Conference", "Construction Crane", "Container Ship", "Convenience Store",
    "Convent", "Convention Center", "Coral Reef", "Correctional Institute Airport",
    "Correctional Institution", "Country Club", "Country Store", "Courthouse", "Cove",
    "Covered Bridge", "Cow Camp", "Crater", "Credit Union", "Crematorium",
    "Crystal Mine", "Culinary Institute", "Cultural Museum", "Culvert", "Cypress Swamp",
    "Dairy Farm", "Dam", "Dance Club", "Dance Hall", "Dance Studio",
    "Data Storage & Processing Center", "Deep Canyon", "Deep Canyon Bridge",
    "Deep Earth Exploration Facility", "Deep in the Earth", "Deep Sea Trench",
    "Deep Sea Tugboat (Ship)", "Deep-Sea Research Submarine", "Delta", "Dense Jungle",
    "Dental Center", "Department of Corrections Medical Center", "Department Store",
    "Desert", "Desert Basin", "Desert Flats", "Desert Hills", "Desert Isle",
    "Deserted Island", "Destroyer", "Detention Facility", "Diamond Field", "Dike",
    "Diner", "Dirigible", "Distillery", "District Office", "Ditch, or Small Canal",
    "Dive Boat", "Dog Racing Track", "Drag Strip", "Drainage Canal",
    "Drive-In Movie Theatre", "Drive-in Restaurant",
    "Drug & Alcohol Abuse Treatment Facility", "Drug Laboratory", "Dry Cleaner",
    "Dry Dock", "Dry Flood Channel", "Dry Gulch or Canyon", "Dry Riverbed", "Dungeon",
    "Election Headquarters", "Electric Company", "Electrical Storm",
    "Electronics Manufacturer", "Elementary School", "Elevated Train", "Embassy",
    "Emergency Response Center", "Emergency Shelter", "Engine House (Train)",
    "Engine Repair Shop", "Environmental Rally", "Esplanade", "Estate", "Estuary",
    "Excavation", "Exhibition Center", "Exotic Restaurant", "Experimental Facility",
    "Experimental Station", "Experimental Vehicle", "Exploration Submarine",
    "Extra-Dimensional Gate", "Factory", "Fair", "Fallen Log Foot Bridge",
    "Fallout Shelter", "Falls", "Farm", "Farm Equipment Dealership",
    "Fast Food Restaurant", "Ferris Wheel", "Ferry Boat", "Ferry Embarkation Point",
    "Ferryboat", "Festival", "Filtration Plant", "Fine Dining Restaurant",
    "Fire Academy", "Fire Department", "Fire Station", "Fire Watchtower",
    "Fireworks Display", "Fish Farm", "Fish Hatchery", "Fishing Boat", "Fishing Lodge",
    "Fishing Pier", "Fishing Trawler (Small Ship)", "Fitness Center",
    "Fixed Radar Site", "Fjord", "Flats", "Flea Market", "Flight School",
    "Floating Casino", "Floating Crane (Ship)", "Floating Dock",
    "Floating Drydock (Ship)", "Flooded Bridge", "Flooded Building, or Structure",
    "Flooded River", "Florist or Flower shop", "Foot Bridge", "Football Stadium",
    "Foothills", "Foreign Restaurant", "Forest", "Forest Fire", "Forested Hills",
    "Fort", "Fortified Bridge", "Fortified Underground Complex", "Fortress",
    "Fraternity House", "Freight Train", "Freighter (ship)", "Frontier Style Residence",
    "Funeral Home", "Funeral in Progress", "Gambling Den", "Game Management Area",
    "Game Preserve", "Gap", "Garage", "Garden", "Gas Company", "Gas Station",
    "Gas-Fired Power Plant", "Gate", "General Cargo Ship", "Geological Park",
    "Geothermal Electric Plant", "Geothermal Park", "Geyser Park", "Ghost Town",
    "Glacier", "Glass-Bottomed Boat", "Gold Mine", "Gorge", "Government Academy",
    "Government Office Building", "Government Office Complex", "Government Offices",
    "Grain Elevator", "Grain Mill", "Grain Storage Bin", "Grassy Heights",
    "Gravel Road", "Greenhouse", "Grocery Store", "Grotto", "Grove", "Guard House",
    "Guard Station", "Gulch", "Gun Shop", "Gymnasium", "Gymnastics Center",
    "Halfway House", "Harbor", "Harbormasters Office", "Haunted House", "Haven",
    "Health Club", "Heaven", "Helicopter", "Helicopter Charter",
    "Helicopter Flight School", "Heliport", "Hell", "High Speed Patrol Boat",
    "Hiking Trails", "Hilltop", "Hi-Rise Construction Site", "Historic District",
    "Historic Monument", "History Museum", "Hole, or Open Pit",
    "Holiday or Major Public Gathering", "Hollow", "Homestead", "Horn", "Horse Park",
    "Horse Racing Track", "Hospital", "Hot Air Balloon", "Hot Springs", "Hotel/Casino",
    "Hunter/Killer Submarine", "Hunting Lodge", "Ice Pack", "Ice Plain", "Iceberg",
    "Icebreaker Ship", "Illegal Fighting Pits", "Import/Export Warehouse",
    "In the Eye of a Hurricane", "Inactive Volcano", "Incline Mine",
    "Incomplete Bridge", "Indian Burial Grounds", "Indian Reservation", "Indoor Arena",
    "Indoor Gun Range", "Industrial Equipment Supply", "Industrial Parts Plant",
    "Industrial Plant", "Inlet", "Inn", "Institute", "Insurance Office",
    "Intelligence Agency", "Interchange", "International Airport",
    "Interstate or Highways", "Interstellar Exploration Facility",
    "Intervention Center", "Island Keys", "Jail", "Jet Ski Terminal", "Jetty",
    "Jewel Mine", "Jewelry Shop", "Junction", "Jungle", "Jungle Isle", "Junkyard",
    "Karate Dojo", "Kennel", "Kennel Club", "Kindergarten &/or Daycare", "Knob",
    "Labor Union", "Laboratory", "Lagoon", "Lake", "Lakefront Cabin", "Landfill",
    "Landing Craft (Ship)", "Landing Strip", "Large Apartment Complex", "Large Canyon",
    "Large City Sewers", "Large Mesa", "Large Office Building", "Large Waterfall",
    "Laundromat", "Lava Field", "Law Library", "Law Offices", "Levee", "Library",
    "Light Buoy", "Light House", "Lighted Data Tower", "Linguistics Institute",
    "Liquor Store", "Listening Post", "Loading Dock", "Loch", "Lodge", "Log Cabin",
    "Long Range Patrol Ship", "Lookout Tower", "Lumber Mill", "Machine Shop",
    "Major Concert", "Man-made Wonder", "Mansion", "Manufacturing Plant", "Marina",
    "Marine & Boat Storage Facility", "Marine Repair Facility", "Marine Sanctuary",
    "Marine Service & Fueling Station", "Maritime Museum", "Marketplace", "Marsh",
    "Martial Arts Dojo", "Martial Arts Facility", "Martial Arts Studio", "Mausoleum",
    "Mausoleum Crypt", "Maze", "Meadow", "Media Ceremony/ Event", "Medical Center",
    "Medical Laboratory", "Medical Library", "Medical School", "Medium Boat",
    "Medium Office Building", "Meeting House", "Megalithic Ruin", "Memorial Library",
    "Memorial Park", "Mental Health Center", "Merchant Ship", "Metal Works Mill",
    "Microwave Relay Tower", "Microwave Transmitter Tower", "Military Academy",
    "Military Airbase", "Military Bombing Range", "Military Cruiser",
    "Military Facility", "Military Installation", "Military Institute",
    "Military-Industrial Complex", "Mill", "Mine", "Missile Base",
    "Mission (Church or Social)",
    "Mixed Jungle and Rice Paddies on Hillside Terraces", "Mobile Home",
    "Mobile Home park", "Mobile Radar Site", "Modern Bank", "Modern Hotel", "Monastery",
    "Monument", "Moon", "Morgue", "Mosque", "Motel", "Motocross or Dirt Bike Raceway",
    "Motor Speedway", "Motorcycle Dealership", "Motorcycle Raceway", "Mound",
    "Mountain Climbing Expedition", "Mountain Meadow", "Mountain Pass", "Mountain Peak",
    "Mountain Range", "Mountain Spring", "Mountains", "Mountainside Cliff",
    "Mountainside Terraces", "Mountaintop", "Movie Filming Location", "Movie Studio",
    "Movie Theatre", "Moving Bridge", "Municipal Airport", "Munitions Storage Facility",
    "Music Hall", "Music Hall of Fame", "Museum", "Museum of Natural Science",
    "Narrow Canyon", "National Historic Landmark", "National Park",
    "National Park Complex", "Native American Reservation", "Native Holy Grounds",
    "Natural Bridge", "Natural Overlook", "Natural Spring", "Natural Wonder",
    "Naval Station", "Naval Supply Ship", "Neck", "Newspaper", "Nightclub",
    "Nomad Camp", "Nuclear Attack Submarine", "Nuclear Plant",
    "Nuclear Shelter Complex", "Nuclear Testing Ground", "Nunnery", "Nursing Home",
    "Oasis", "Observation Tower", "Observatory", "Ocean Front Bar/Tavern",
    "Ocean Liner", "Ocean Storm", "Oceanfront Condominium", "Office Complex",
    "Office Tower", "Oil & Gas Platform (Ocean)", "Oil & Gas Rig", "Oil Terminal",
    "Oilfield", "Old Bank", "Old Hotel", "Old Sailing Vessel", "Old Theatre",
    "Old Town", "Old Town Historical District", "Old U-Boat (Submarine)", "Open Pit",
    "Open Pit Mine", "Opera Hall", "Orchard", "Ossuary", "Outcropping",
    "Outdoor Concert", "Outer Space", "Outpost",
    "Overnight Automobile/Passenger Ferry Ship", "Package Terminal", "Packing Plant",
    "Pagoda", "Palace", "Parachute School", "Paranormal Institute",
    "Park & Recreation Facility", "Parking Garage", "Parking Lot", "Parliament",
    "Partially Buried Pyramid", "Passenger Cruise Ship", "Passenger Train",
    "Patrol Vessel", "Paved Airstrip", "Pavilion", "Pawnbroker",
    "Performing Arts Awards Ceremony", "Performing Arts Theatre", "Pet Shop",
    "Petroleum Manufacturing Facility", "Petroleum Storage Facility", "Photography Lab",
    "Picnic Area", "Pier", "Pipeline Terminal", "Pizza Palace", "Placer Mine",
    "Plains or Grassland", "Plant Nursery", "Plastics Factory", "Plateau", "Playground",
    "Plaza", "Point", "Police Station", "Police Sub-Station", "Political Party HQ",
    "Political Rally", "Polluted Waterway", "Polo Club", "Pond", "Pontoon Bridge",
    "Pool", "Pool Hall", "Post Office", "Power Collection Array", "Power Substation",
    "Prep School", "Preparatory School", "Primitive Village", "Print Shop", "Prison",
    "Prison Camp", "Prison Medical Center", "Private Beach", "Private Golf Course",
    "Private Home", "Private Island", "Private Library", "Private Park",
    "Private School", "Processing Facility", "Protest March",
    "Psychic Research Facility", "Pub", "Public Beach", "Public Ceremony",
    "Public Gathering", "Public Golf Course", "Public Library", "Public Park",
    "Publishing House", "Putt-Putt Golf Course", "Race Track", "Racetrack",
    "Radio Station", "Radio Telescope", "Radioactive Mine",
    "Radioactive Waste Storage Facility", "Raft", "Rail Yard", "Railcar Wash",
    "Railway", "Railway Bridge", "Railway Platform", "Railway Station",
    "Railway Tunnel", "Ranch", "Ranger Station", "Rapids", "Recovery Ward",
    "Recreation Area", "Recreational Vehicle", "Recreational Vehicle Park",
    "Red-light District", "Reef", "Refrigerated Ship",
    "Refrigerated Storage Facility", "Regional Medical Center", "Rehabilitation Center",
    "Religious Rally", "Religious Retreat", "Religious Sanctuary", "Remote Island",
    "Remote Listening Post", "Remote Research Facility", "Remote Research Station",
    "Rental Agency", "Research Facility", "Research Laboratory", "Research Vessel",
    "Reservoir", "Residential Construction Area", "Resort",
    "Resort, Golf Course, & Country Club", "Restaurant", "Retail Store",
    "Retirement Community", "Rice Paddy", "Ridge", "Rifle & Pistol Range",
    "Rift Valley", "River", "River Canyon", "River Ford", "River Island", "River Lock",
    "River Tunnel", "Riverbank", "Riverbend", "Riverboat (Paddlewheel)", "Road",
    "Road Fork", "Road Paving Company", "Road Tunnel", "Roadside Store", "Rock Garden",
    "Rock Quarry", "Rock Tower", "Rocky Desert", "Rocky Ridges", "Rooftop", "Rookery",
    "Rope Bridge", "Roundhouse (Train)", "Rugged Coastline", "Ruins", "Safari Preserve",
    "Safe House", "Sailboat", "Salt Flats", "Salt Mine", "Sand & Gravel Pit",
    "Sandbar", "Sanitarium", "Sargasso Sea", "Satellite",
    "Satellite Communications Center", "Science & Industry Museum", "Science Institute",
    "Science Laboratory", "Science Museum", "Scientific Library", "Scrap Metal Yard",
    "Scrap Yard", "Sculpture Garden", "Sea Banks", "Sea Canyon", "Sea Cargo Terminal",
    "Sea Going Hydrofoil (Ship)", "Sea Passenger Terminal", "Sea Trench", "Seadrome",
    "Seaplane Base", "Secondary School", "Secret Military Facility", "Secret Stronghold",
    "Secret Underground Stronghold", "Security Center", "Sewer Company",
    "Sewer Entrance", "Sewers", "Shaft Mine", "Shanty Town", "Shantytown",
    "Ship Taking on Water", "Shipping Warehouse", "Shipwreck", "Shipyard", "Shoals",
    "Shoals & Banks", "Shopping Center", "Shopping Mall", "Shrine", "Silver Mine",
    "Sink", "Sinkhole", "Sinking Ship", "Skateboard Rink", "Skating Rink", "Ski Area",
    "Ski Lift", "Ski Resort", "Slaughterhouse", "Small Airport",
    "Small Apartment Complex", "Small Canyon", "Small Day Passenger Ship", "Small Mesa",
    "Small Office Building", "Small Open Boat", "Small Town", "Small Waterfall",
    "Smooth Pyramid", "Snowfield", "Snowy Crevasse", "Solar Observatory",
    "Solar Power Plant", "Solar System Exploration Facility", "Sorority House", "Spa",
    "Space Landing Facility", "Space Launch Facility", "Space Station", "Spacecraft",
    "Spacecraft Repair Facility", "Spanish Mission", "Speedboat", "Sports Arena",
    "Sports Hall of Fame", "Stables", "Stadium", "State Park", "Station", "Steamship",
    "Steel Manufacturing Plant", "Steel Mill", "Stepped Pyramid", "Stockyard", "Store",
    "Strait or Sound", "Street", "Strip Club", "Strip Mall", "Strip Mine",
    "Structural Rubble", "Studio Back Lot", "Sub-glacial Cave", "Subterranean River",
    "Subway", "Summer Camp", "Survivalist Enclave", "Suspension Bridge", "Swamp",
    "Swimming Pool", "Synagogue", "Tanker (Ship)", "Tanning Salon", "Tavern",
    "Taxi Company", "Technology Gallery", "Telecommunication Center",
    "Teleconferencing Center", "Telephone Company", "Television Recording Studio",
    "Television Station", "Temple", "Tents", "Terrarium", "Test Range",
    "Textiles Factory", "Textiles Mill", "The Docks", "Theme Park", "Tire Dump",
    "Toll Booth", "Toll Road", "Tomb", "Tool Shop", "Tour Bus", "Town Bridge",
    "Town Square", "Townhouse", "Toxic Waste Storage Facility", "Trading Post",
    "Traffic Control Tower", "Trail Crossing", "Trailer", "Trailer park", "Trailhead",
    "Train Depot", "Train Museum", "Train Station", "Training Facility",
    "Trash Company", "Trash Dump", "Travel Agency", "Treatment Center", "Tree Nursery",
    "Trolley", "Truck Terminal", "Tug Boat", "Tundra", "Tunnel", "Tunnel Mine",
    "Ultralight Aircraft School", "Underground Bunker Complex", "Underground Garage",
    "Underground Mall", "Underground Storage Facility", "Underground Stronghold",
    "Undersea Tunnel", "Underwater - Deep Sea", "Underwater Caves & Caverns",
    "Underwater -Coastal", "Underwater Facility", "Underwater Kelp Farm",
    "Underwater Pen or Contained Area", "Underwater Preserve",
    "Underwater Research Facility", "Union Hall", "University",
    "University Experimental Farm", "Unpaved Airstrip", "Unusual Rock Formation",
    "Urban Fire", "Urban Street", "Valley", "Vast Antennae Array", "Vault",
    "Vehicle Tire Store", "Victorian Residence", "Villa", "Village", "Vineyards",
    "Volunteer Fire Department", "War Museum", "Warehouse", "Warehouse District",
    "Warm Springs", "Washed-Out Bridge", "Washed-Out Road",
    "Waste Water Treatment Plant", "Wastewater Treatment Plant", "Watchtower",
    "Water Buoy", "Water Garden", "Water Hole", "Water Landing", "Water Park",
    "Water Passage", "Water Pumping Station", "Water Purification Plant", "Water Tower",
    "Water Trail", "Waterfront Commercial", "Waterfront Industrial",
    "Waterfront Restaurant", "Watermill", "Watershed", "Waterworks", "Wax Museum",
    "Weapons Laboratory", "Wedding Hall", "Wedding in Progress", "Welcome Center",
    "Welding Equipment Store", "Well", "Well House", "Whaling Boat", "Whitewater Rapids",
    "Wide Canyon", "Wild Animal Park", "Wilderness Park", "Wildlife Refuge",
    "Wildlife Rehabilitation Center", "Wildlife Sanctuary", "Windmill", "Winery",
    "Woodshop", "Workshop", "Worlds Fair", "Wreck", "Yacht", "Yacht & Country Club",
    "Yacht Center", "Yacht Harbor", "Yacht Show", "Youth Camp", "Youth Center", "Zoo",
    "Other/Unknown"
]

ART_STYLE_LIST = [
    "1980s dark fantasy", "3d-model", "abstract expressionism", "abstract", "advertising", "alien", "analog film",
    "anime", "architectural", "art deco", "art nouveau", "automotive",
    "biomechanical cyberpunk", "biomechanical", "cinematic", "collage", "comic book",
    "constructivist", "corporate", "craft clay", "cubist", "cybernetic robot",
    "cybernetic", "cyberpunk cityscape", "digital art", "disco", "dreamscape",
    "dystopian", "enhance", "expressionist", "fairy tale", "fantasy art",
    "fashion editorial", "film noir", "flat papercut", "food photography",
    "futuristic", "glamour", "gothic", "gourmet food photography", "graffiti",
    "grunge", "hdr", "horror", "hyperrealism", "impressionist",
    "iphone photographic", "isometric", "kawaii", "kirigami", "line art",
    "long exposure", "lovecraftian", "lowpoly", "luxury", "macabre", "manga",
    "metropolis", "minimalist", "monochrome", "nautical", "neon noir", "neonpunk",
    "origami", "paper mache", "paper quilling", "papercut collage",
    "papercut shadow box", "photographic", "pixel art", "pointillism", "pop art",
    "psychedelic", "real estate", "renaissance", "retail", "retro cyberpunk",
    "retro futurism", "sci-fi", "silhouette", "space", "stacked papercut",
    "stained glass", "steampunk", "surrealist", "techwear fashion", "texture",
    "thick layered papercut", "tilt-shift", "tribal", "typography", "vaporwave",
    "watercolor", "zentangle", "Other"
]

class TriXopeStorySelect:
    """
    A custom node for ComfyUI to select story elements and generate a prompt.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Genre": (GENRE_LIST, {"default": GENRE_LIST[0]}),
                "Structure": (STRUCTURE_LIST, {"default": STRUCTURE_LIST[0]}),
                "Flow": (FLOW_LIST, {"default": FLOW_LIST[0]}),
                "Time Period": (TIME_PERIOD_LIST, {"default": TIME_PERIOD_LIST[0]}),
                "Setting": (SETTING_LIST, {"default": SETTING_LIST[0]}),
                "Art Style": (ART_STYLE_LIST, {"default": ART_STYLE_LIST[0]}),
                "Premise": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("story_prompt",)
    FUNCTION = "generate_story"
    CATEGORY = "triXope/writing"

    def generate_story(self, **kwargs):
        # We use **kwargs to capture keys with spaces like "Time Period" and "Art Style"
        genre = kwargs.get("Genre")
        structure = kwargs.get("Structure")
        flow = kwargs.get("Flow")
        time_period = kwargs.get("Time Period")
        setting = kwargs.get("Setting")
        art_style = kwargs.get("Art Style")
        premise = kwargs.get("Premise")

        output_text = (
            f"Genre: {genre}\n"
            f"Structure: {structure}\n"
            f"Flow: {flow}\n"
            f"Time Period: {time_period}\n"
            f"Setting: {setting}\n"
            f"Art Style: {art_style}\n"
            f"Premise: {premise}"
        )
        
        return (output_text,)

# --- moved from prompt_split.py ---------------------------------------------
class TriXopePromptSplitSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "first part | second part | third part"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text", "prompt_count")
    FUNCTION = "select_prompt"
    CATEGORY = "triXope/writing"

    def select_prompt(self, text: str, index: int):
        parts = [p.strip() for p in text.split('|')]
        parts = [p for p in parts if p]
        if not parts:
            return ("", 0)
        safe_index = index % len(parts)
        selected_part = parts[safe_index]
        return (selected_part, len(parts))

print("### Loading: triXope Prompt Split & Select ###")

# --- moved from batch_ollama_writer.py --------------------------------------
import os
import json
import requests
import time

class TriXopeBatchOllamaWriter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts_text": ("STRING", {"multiline": True, "default": ""}),
                "delimiter": ("STRING", {"default": "|", "multiline": False}),
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ollama_model": ("STRING", {"default": "gemma4:26b"}),
                "output_dir": ("STRING", {"default": "./ComfyUI/output/FilmAuteur"}),
                "filename": ("STRING", {"default": "final_shot_list.txt"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("saved_file_path", "file_contents", "processed_count")
    FUNCTION = "process_batch"
    CATEGORY = "triXope/writing"

    def process_batch(self, prompts_text, delimiter, system_instruction, ollama_url, ollama_model, output_dir, filename, seed):
        # Prepare output path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if not filename.endswith(".txt"):
            filename += ".txt"
        file_path = os.path.join(output_dir, filename)

        # Split input text into individual shot prompts
        raw_prompts = [p.strip() for p in prompts_text.split(delimiter) if p.strip()]
        total_prompts = len(raw_prompts)
        
        print(f"\n🚀 [TriXope Batch] Starting batch processing for {total_prompts} shots...")

        # Overwrite/Initialize the output file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("")

        # Loop internally inside Python (Zero ComfyUI / Browser memory footprint during run)
        with open(file_path, "a", encoding="utf-8") as f:
            for idx, raw_prompt in enumerate(raw_prompts, start=1):
                start_time = time.time()
                
                # Build combined prompt for Ollama
                full_user_prompt = f"{system_instruction}\n\nUSER PROMPT:\n{raw_prompt}" if system_instruction.strip() else raw_prompt
                
                payload = {
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": full_user_prompt}],
                    "stream": False,
                    "keep_alive": -1,
                }
                if seed > 0:
                    payload["options"] = {"seed": seed + idx}

                try:
                    res = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=300)
                    if res.status_code == 200:
                        enhanced_text = res.json().get("message", {}).get("content", "").strip()
                    else:
                        enhanced_text = f"[ERROR Shot {idx}]: HTTP {res.status_code}"
                except Exception as e:
                    enhanced_text = f"[ERROR Shot {idx}]: {str(e)}"

                # Clean leading/trailing markdown if present
                if enhanced_text.startswith("```"):
                    enhanced_text = enhanced_text.strip("`").strip()

                # Write directly to disk and flush buffer immediately
                f.write(enhanced_text + "\n|\n")
                f.flush()

                elapsed = time.time() - start_time
                print(f"  └─ ✅ Shot [{idx}/{total_prompts}] finished in {elapsed:.2f}s")

        print(f"🎉 [TriXope Batch] Completed all {total_prompts} shots! File saved to: {file_path}\n")

        # Read the complete file contents once at the very end to pass downstream safely
        with open(file_path, "r", encoding="utf-8") as f:
            file_contents = f.read()

        return (file_path, file_contents, total_prompts)

__all__ = [
    "ScreenplayFormatterNode",
    "ShotLengthSelectorNode",
    "ShotLengthBufferNode",
    "TriXopeCharacterCraft",
    "TriXopeStorySelect",
    "TriXopePromptSplitSelect",
    "TriXopeBatchOllamaWriter",
]
