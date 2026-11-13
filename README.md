# triXope — ComfyUI Custom Node Pack

triXope is a ComfyUI custom-node pack built around a script-to-screen film pipeline:
screenwriting and shot planning (MiniMax **H3** or **LTXV**), joint audio/video
conditioning and sliding-context sampling, batched VAE work, latent file handling,
upscaling/face restoration, colour grading, and a timeline editor for the final cut.

The pack registers **62 nodes**. They are grouped into nine node-menu categories, all
under the base category `triXope`:

| Category | Module | Nodes |
|---|---|---|
| `triXope/audio` | [nodes_audio.py](nodes_audio.py) | 2 |
| `triXope/film` | [nodes_film.py](nodes_film.py) | 5 |
| `triXope/image` | [nodes_image.py](nodes_image.py) | 6 |
| `triXope/io` | [nodes_io.py](nodes_io.py) | 8 |
| `triXope/latent` | [nodes_latent.py](nodes_latent.py) | 10 |
| `triXope/loaders` | [nodes_loaders.py](nodes_loaders.py) | 5 |
| `triXope/utilities` | [nodes_utilities.py](nodes_utilities.py) | 14 |
| `triXope/video` | [nodes_video.py](nodes_video.py) | 5 |
| `triXope/writing` | [nodes_writing.py](nodes_writing.py) | 7 |

Every node class lives in the module for its category, and every node's ComfyUI
category matches that module (`CATEGORY = "triXope/<category>"`, or `category=` in the
V3 `io.Schema`). Registration happens once, in [`__init__.py`](__init__.py), through
`NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`; menu names keep the pack's
`Δ` prefix.

## Contents

- [Installation](#installation)
- [Repository layout](#repository-layout)
- [Node reference](#node-reference)
  - [audio](#trixopeaudio--audio) · [film](#trixopefilm--film-pipelines) · [image](#trixopeimage--image) · [io](#trixopeio--inputoutput) · [latent](#trixopelatent--latents-and-vae) · [loaders](#trixopeloaders--loaders) · [utilities](#trixopeutilities--utilities) · [video](#trixopevideo--video) · [writing](#trixopewriting--writing)
- [Support modules](#support-modules)
- [Frontend scripts](#frontend-scripts)
- [Unregistered and legacy classes](#unregistered-and-legacy-classes)
- [Notes and known quirks](#notes-and-known-quirks)

## Installation

The pack installs into `ComfyUI/custom_nodes/ComfyUI-triXope`, with `__init__.py` and the
category modules at that single package level — ComfyUI loads the folder itself as the custom
node package.

**Option A — ComfyUI Manager (recommended):** open *Manager → Custom Nodes Manager*, search
for `triXope` (the pack is listed as **ComfyUI-triXope**), install it, then restart ComfyUI.
Manager installs into `custom_nodes/ComfyUI-triXope` and handles the restart prompt.

**Option B — manual clone:**

```bash
cd ComfyUI/custom_nodes
git clone <repository-url> ComfyUI-triXope
```

To install from a downloaded archive instead, extract it so the result is
`ComfyUI/custom_nodes/ComfyUI-triXope/__init__.py` — an extra nesting level
(`custom_nodes/ComfyUI-triXope/ComfyUI-triXope/…`) leaves the nodes undetected.

Then restart ComfyUI. The nodes appear in the right-click menu under `triXope/…`.

Install the Python requirements of the nodes you intend to use:

```bash
pip install -r requirements.txt
```

This covers the pack's third-party packages; ComfyUI already supplies `torch`,
`torchvision`, `torchaudio`, `numpy`, `Pillow` and `aiohttp`. Optional extras (face
restoration, SageAttention, SAM3) are commented out at the bottom of the file. The table
below says which nodes need what:

| Dependency | Needed by |
|---|---|
| `torch`, `torchvision`, `torchaudio`, `numpy`, `Pillow` | nearly every node |
| `opencv-python` (`cv2`) | ColorFX, HSV match, H3 Upscaler, FilmAuteur (LTX) |
| `safetensors` | latent load/save nodes |
| `av` (PyAV) | Video Timeline Editor, FilmAuteur (LTX) |
| `requests` | Batch Ollama Writer, FilmAuteur (LTX) prompting |
| `tqdm` | Efficient Image Batch Loader |
| `imageio-ffmpeg` **or** `ffmpeg` on `PATH` | Panorama 360 Rotate |
| `psutil` | `device_manager.py` (Universal Loader) |
| Ollama server | Optional: Batch Ollama Writer, Screenwriter, Character/Story nodes' AI options. The Screenwriter can use a connected `clip` or a `Local: ...` model instead. |
| `openai-whisper` | Screenwriter audio-to-screenplay mode |
| `ultralytics` YOLO models, ComfyUI BiRefNet, SAM3 (RMBG pack) | H3 Upscaler face detection, `ref_alpha` background removal |
| `taehv.py` (optional, next to `__init__.py`) | fast path in VAE Decode Batched (WanVideo) |
| `repositories/ultimate_sd_upscale` | fetched automatically on first start if missing |
| DLSS 5 runtime (`dlss5/`, `install_runtime.py`) | H3 Upscaler DLSS 5 stage |

## Repository layout

| Path | Purpose |
|---|---|
| [`__init__.py`](__init__.py) | registration tables, pack startup, `/gemini/get_files` route |
| [`nodes_audio.py`](nodes_audio.py) | `triXope/audio` node classes |
| [`nodes_film.py`](nodes_film.py) | `triXope/film` node classes (H3 + LTXV pipelines) |
| [`nodes_image.py`](nodes_image.py) | `triXope/image` node classes |
| [`nodes_io.py`](nodes_io.py) | `triXope/io` node classes |
| [`nodes_latent.py`](nodes_latent.py) | `triXope/latent` node classes |
| [`nodes_loaders.py`](nodes_loaders.py) | `triXope/loaders` node classes |
| [`nodes_utilities.py`](nodes_utilities.py) | `triXope/utilities` node classes |
| [`nodes_video.py`](nodes_video.py) | `triXope/video` node classes |
| [`nodes_writing.py`](nodes_writing.py) | `triXope/writing` node classes |
| [`utils.py`](utils.py), [`common.py`](common.py), [`core.py`](core.py), [`context_utils.py`](context_utils.py) | shared helpers (any-type plumbing, H3 AV grid math, timestamp/beat parsing) |
| [`device_manager.py`](device_manager.py), [`autogpu_pytorch.py`](autogpu_pytorch.py), [`gguf_support.py`](gguf_support.py) | multi-GPU device routing, GGUF loaders |
| [`usdu_patch.py`](usdu_patch.py), [`usdu_utils.py`](usdu_utils.py), [`usdu_nodes.py`](usdu_nodes.py) | Ultimate SD Upscale integration |
| [`crop_model_patch.py`](crop_model_patch.py), [`install_runtime.py`](install_runtime.py) | ControlNet patch cropping, DLSS runtime installer |
| [`js/`](js) | browser-side extensions (see [Frontend scripts](#frontend-scripts)) |
| [`requirements.txt`](requirements.txt) | Python package requirements for the node modules |
| [`test_screenplay_formatter.py`](test_screenplay_formatter.py) | unit tests for the screenwriter module |

---

# Node reference

Each entry lists the registered node id, its menu name, the module it lives in, its
sockets and what it does. `*` means the socket accepts any type.

## `triXope/audio` — Audio

Audio analysis and audio file output. Module: [nodes_audio.py](nodes_audio.py).

### Δ Audio To Tablature 🎵 — `triXope AudioToTablature`

Detects pitches in a connected audio clip, maps them onto a guitar fretboard and renders
tablature as an image, as text and as a synthesized audio preview. Standard tuning
(`E2 A2 D3 G3 B3 E4`) is used, and each note is placed on the simplest fret that can play
it.

- **Inputs:** `audio` (AUDIO), `time_step` (FLOAT, default `0.1` — analysis hop in seconds),
  `confidence_threshold` (FLOAT `0`–`1`, default `0.1`), `notes_per_measure` (INT 1–16,
  default `16`), `measures_per_line` (INT 1–10, default `4`), `note_duration` (FLOAT
  `0.05`–`1.0`, default `0.25`).
- **Outputs:** `tablature_image` (IMAGE), `tablature_text` (STRING), `audio_preview` (AUDIO).
- **Notes:** accepts 1-D, 2-D `[channels, samples]` and 3-D `[batch, channels, samples]`
  audio. Any failure returns an error image/text plus a quarter-second of silence instead of
  raising, so workflows keep running.

### Δ Audio File Saver — `triXope AudioFileSaver`

Saves AUDIO output to FLAC and reports the files back to the UI as playable results.

- **Inputs:** `audio` (AUDIO), `output_path` (STRING, defaults to the ComfyUI output
  directory), `filename_prefix` (STRING, default `audio`); hidden `prompt` / `extra_pnginfo`.
- **Outputs:** none (output node); returns UI `audio` entries.
- **Notes:** one FLAC per batch item, `%batch_num%` in the prefix is substituted per item,
  and the workflow prompt/extra metadata is embedded as Vorbis comments unless
  `--disable-metadata` is set.

## `triXope/film` — Film pipelines

The MiniMax H3 and LTXV generation stacks. Module: [nodes_film.py](nodes_film.py).

### Δ H3 Conditioner — `triXope H3FL2VAandRef2VA`

Union of ComfyUI's MiniMax H3 image-to-video (`fl2va`) and reference-to-video (`ref2va`)
conditioning in a single node: N evenly spaced keyframes **and** picture/video/audio
references can be combined in one conditioning payload. Implemented on the ComfyUI V3
`io.ComfyNode` API.

- **Inputs:** `clip` (CLIP), `vae` (VAE), `audio_vae` (VAE), `prompt` (STRING, multiline,
  dynamic prompts), `width` (default `1344`, step 32), `height` (default `768`, step 32),
  `length` (frames at 24 fps, default `124`, snapped to H3's `17k+5` grid), `mode`
  (`text` / `reference` / `keyframes` / `all`, default `reference`; `all` keeps keyframes
  and references together so a pinned shot keeps its identity references), `keyframes`
  (optional IMAGE batch), `ref_image_size` (`match` / `max` / `mixed`, default `mixed`:
  character sheets at full fidelity, props and the palette chart at `match`), `ref_alpha`
  (`off` / `white` / `black` /
  `gray` / `transparent`), `ref_image_batch` (optional IMAGE), and autogrow `ref_videos`,
  `ref_video_audios`, `ref_audios` (up to four each).
- **Outputs:** `positive` (CONDITIONING), latent (LATENT — the matching empty AV latent).
- **Notes:** keyframes are anchored at real frame positions (first image = first frame, last
  image = last frame, the rest evenly spaced); references ride through every sampling step as
  `minimax_refs`, which is why `ref_image_size = max` can be several times slower.
  `ref_alpha = transparent` uses the image alpha, or removes an opaque background with
  ComfyUI BiRefNet, then composites onto black — alpha is never preserved as alpha. In
  `text` mode nothing is attached, so the node warns when the prompt cites a `<Picture N>`
  or `<Video N>` that has no matching reference connected.

### Δ H3 Sampler — `triXope H3SlidingContextSampler`

Sampler replacement for MiniMax H3. Sliding windows continue from the generated AV overlap by
default; explicit cut/dissolve cues can change the join, and `sliding_windows` off runs a
single H3 sampling pass instead.

- **Inputs:** `model` (MODEL), `positive` (CONDITIONING), `latent` (LATENT), `seed`, `steps`
  (default 20), `sampler_name` (default `euler`), `scheduler` (default `beta`),
  `window_frames` (default `124`, step 17), `overlap_frames` (default `39`, step 17),
  `story_beats` (timestamped beat script), `transition_mode` (`auto` / `continue` /
  `crossfade`), `transition_prompt` (transition wording copied from the positive prompt for
  mode detection only), `sliding_windows` (BOOLEAN, default on), `enforce_cuts` (BOOLEAN,
  default on), and an optional `clip` (required only when `story_beats` is used, because beats
  are encoded on the fly).
- **Outputs:** `output` (LATENT).
- **Notes:** rejects non-H3 latents (expects a `NestedTensor` video `[B,24,T,H,W]` + audio
  `[B,32,2,T]`) and batch sizes above 1. Timestamped beats (`[00:00:05-00:00:12] beat` or
  `[00:00:12] beat`) can force hard cuts inside a sliding overlap when `enforce_cuts` is on;
  `transition_prompt` can request a global cut or crossfade.

### Δ Film Auteur (H3) — `triXope FilmAuteurH3`

The single-node H3 film pipeline: builds conditioning, plans shots from the screenwriter's
output, and samples either in one pass or with sliding AV context per shot. Declared with
`is_input_list=True`, so the screenwriter's list-valued `shot_lengths` arrives as one array.

- **Inputs:** `model`, `clip`, `vae`, `audio_vae`, `prompt` (multiline; `|` splits it into
  independent prompt sections that run one after another), `output_mode` (`video` / `image`),
  `aspect_ratio` (`1:1 (Square)`, `2:3 (Portrait Photo)`, `3:2 (Photo)`,
  `3:4 (Portrait Standard)`, `4:3 (Standard)`, `9:16 (Portrait Widescreen)`,
  `16:9 (Widescreen)` default, `21:9 (Ultrawide)`), `megapixels` (default `0.5`; the height is
  picked for the budget, the width follows the ratio — 21:9 is `height * 21/9` — and both sides
  are snapped to a multiple of 32), `length_seconds` (default 5, converted to frames at
  24 fps and snapped to `17k+5`), `mode` (`text` / `reference` / `keyframes` / `all`,
  default `reference`), `drive_audio` (optional AUDIO), `keyframes` (optional IMAGE),
  `ref_image_size` (`match` / `max` / `mixed`, default `mixed`), `ref_alpha`,
  `audio_drive` (BOOLEAN), `ref_image_batch` (optional), autogrow `ref_videos`,
  `ref_video_audios`, `ref_audios`, `story_beats`, `seed`, `steps`, `sampler_name`,
  `scheduler`, `shot_lengths` (optional INT list input), `cross_shot_context`,
  `cross_shot_context_frames` (default `48`, about 2 s of prior footage), `voice_continuity`
  (BOOLEAN, default on: the previous shot's audio rides with the cross-shot reference so a
  voice and room tone carry across the cut), `sliding_windows`, `window_frames_mode`
  (`automatic` / `manual`), `window_frames`, `overlap_frames`, `transition_mode`,
  `transition_prompt`, `enforce_cuts`.
- **Outputs:** `positive` (CONDITIONING list), `sampled shot latents` (LATENT list),
  `initial shot latents` (LATENT list), `image` (IMAGE).
- **Notes:** each shot is conditioned, sampled and returned independently, so shot outputs
  are hard-cut boundaries. `audio_drive` pins the connected track to the AV latent as an H3
  audio guide, and the returned audio is that track rather than a generated one. In `image`
  mode one still per shot is produced as a single H3 video latent frame; `length_seconds`,
  shot durations, sliding windows and audio drive do not apply. With `prompt` split on `|`,
  `shot_lengths` is ignored and each section uses its own `story_beats`/`length_seconds`.
  In `text` mode the prompt is rewritten for text-only conditioning: `<Subject N>` and
  `<Picture N>` citations become the asset names and every identity is repeated inline,
  because no references exist to cite. Reference mode keeps only the definitions a shot (or
  its beats) actually mentions, so off-screen cast cannot bleed into the frame. The hard-cut
  sentence added by `enforce_cuts` also states that any prior footage supplied as a
  reference is used for casting, wardrobe, location and light only, not to continue the
  previous action.

### Δ Film Auteur (LTX) — `triXope FilmAuteur_LTX`

The monolithic LTXV film pipeline: prompting (optionally through Ollama), primary sampling,
spatial/temporal latent upscaling, face restoration, colour FX, preview and final video — all
in one node with a grouped widget interface (the `grp_*` dummy widgets drive the collapsible
sections in `js/filmauteur_groups.js`).

- **Required inputs:** `clip`, `video_vae`, `audio_vae`, plus the grouped settings:
  - *Mode Activation*: `primary_sampling`, `spatial_upscale`, `temporal_upscale`,
    `restore_faces`, `enable_colorfx`, `enable_preview`, `enable_final_video`, `video_mode`
    (`text-to-video` / `image-to-video` / `reference-to-video`), `image_strength`,
    `img_compression`, `audio_select` (`internal` / `source` / `reference`),
    `identity_guidance_scale`.
  - *Prompting*: `character_descriptions`, `location_description`, `scene_descriptions`,
    `use_ollama` (`disabled` / `enhance prompt` / `song-to-script`), `ollama_url`,
    `ollama_model`.
  - *Video Specs*: `seed_number`, `control_before_generate`, `target_resolution`,
    `length_in_seconds`, `frame_rate`.
  - *Primary Sampling*: `primary_sampler_name`, `primary_cfg`, `primary_steps` (a comma
    separated sigma list), `eta`, `bongmath`, `enable_nag`.
  - *Upscale & Refine*: `spatial_passes`, `spatial_sampler`, `spatial_cfg`, `spatial_sigmas`,
    `temporal_denoise`, `facerestore_model`, `facedetection`, `codeformer_fidelity`,
    `face_restore_color_match`, `face_restore_edge_blur`, `face_restore_blend`.
  - *Color FX*: master toggles plus the Color Correction, LUT, Enhancements, Blur and
    Stylistic subgroups.
  - *Performance*: `enable_fp16_accumulation`, `sage_attention`, `autoregressive_chunking`,
    `chunk_size_seconds`, `context_window_seconds`, `chunks_feedforward`,
    `clear_models_and_cache`.
- **Optional inputs:** `model1_primary`, `model1.5_extend`, `model2_spatial`,
  `spatial_upscaler` (LATENT_UPSCALE_MODEL), `model3_temporal`, `temporal_upscaler`,
  `positive`, `negative`, `image(s)`, `audio`, `latent_override`.
- **Outputs:** `text_prompt(s)`, `positive`, `negative`, `av_latent`, `video_latent`,
  `audio_latent`, `video`, `images`, `audio`, `fps`, `ref_frame_count`, `real_time_eta`
  (`ETA_TRACKER`).
- **Notes:** the node installs a prompt handler so the UI can hand it the submitted prompt
  (`trixope_on_prompt_handler`). Face-restore models are read from a pack-local
  `facerestore_models/` folder registered as `trixope_facerestore`. `real_time_eta` is meant
  for the LTXV Real-Time ETA display node.

### Δ H3 Upscaler — `triXope UltimateSDUpscaleMiniMaxH3`

Ultimate-SD-style tile upscaling aimed at H3 output, with face detail, refinement, DLSS 5
neural rendering and a full colour stage, all in one node. Batches with more than one image
are treated as ordered video frames, so matching tiles are processed as a temporal sequence.

- **Required inputs:** `model`, `positive`, `vae`, plus the `grp_h3_upscale` group header.
- **Optional inputs:** `audio_vae` (decodes the `audio` output from the H3 latent), `image`,
  `audio` (returned unchanged on the `audio` output in every mode, ahead of the latent's decoded
  audio), `latent`
  (the H3 latent for `latent` mode and the source of the `audio` output), then the
  *H3 Upscale* settings:
  - `h3_upscale_mode` — `off` (the old `enabled_h3_upscale` off state: no resize, no tile
    diffusion), `pixel` (resize/diffuse the `image`; a long clip is rendered shot by shot so the card
    does not run out of memory. `pixel_batch_overlap` (default 8) is the frames of real context on
    **each** side of a seam: a chunk re-renders that many frames before the part it emits, and its own
    last frames are re-rendered by the next chunk, so every frame either side of a seam was sampled
    with real neighbours instead of from a cold start. Each frame is emitted by exactly one chunk, so
    a seam is a cut between two warm renders and never a blend of two different redraws, which
    double-exposed the content; the segment costs twice `pixel_batch_overlap` frames per seam, so the
    context is capped at an eighth of the chunk and every render still advances at least three
    quarters of its frames. A clip that does not divide evenly ends on a full-length render
    rather than a short remainder: the last chunk backs up over frames it has already emitted
    and skips them when it cuts its block, so the final seam is not a stub sequence that looks
    different from the rest. Every render is a grid length (22 + 17 k frames: 39, 56,
    73 ...), because an off-grid length is silently cut back to the grid - a 51 frame render produces 39, and the twelve
    frames it asked for are never made. A clip whose own length is not on the grid ends on
    it by backing the last render up over frames it has already emitted. On the grid nothing is trimmed, so decoded frame j stays frame j and the two
    sides of a seam register. A shortfall of a frame or two from
    the VAE is padded on the tail; a large one is reported as what it is, a sequence stretched
    over frames it does not have, whose content drifts against the neighbouring chunk.
    Nothing is sized from a frame count any more: a shot is one render, and if the card
    refuses it the render is cut back or split and retried (the log says where, and why).
    The free VRAM at the output resolution is still measured and logged - a quarter is
    held back for the model and the decoder - because it is the number that says how
    much one render should hold. Every render after the first that meets an in-shot
    seam also *starts* from the frames already on screen:
    its leading overlap frames are replaced, before encoding, by the previous chunk's own output
    instead of a fresh upscale of the source, so the render continues from what the clip shows and
    inherits where that render placed the content rather than placing it again. They are context,
    not pinned frames - the chunk redraws them and they are cut from its output, so nothing old is
    held in place for the redraw to disagree with, which is what ghosted the pinned variant
    (`PIXEL_BATCH_PIN_CONTEXT` stays off). A guide narrows a seam's own re-placement; it cannot
    make two independent samples identical, and each seam still reports the offset that is left.
    Every chunk after the first also keeps the tone of what is already on screen:
    it re-renders the frames the previous seam emitted anyway, so those are compared, and the
    coarse colour/exposure difference is added back over the frames it emits, fading out before the
    chunk ends so nothing accumulates along the clip. It moves no pixels, so it cannot double a
    redraw, and unlike the source-side locks it still applies at denoise 1.0. A correction is
    only applied when it accounts for the difference: a real exposure step collapses under it,
    while two renders that placed or drew the content differently do not, and those seams are
    left exactly as rendered and logged as such - moving frames to chase a difference that is
    not exposure only adds a change of our own at every boundary. Each seam logs three
    separate measurements - the tone drift before and after the lock, a detail figure, and where
    the two renders think the content sits with the correlation score behind that estimate - so a
    seam can be read as tonal, as detail or as geometry instead of guessed at. Those frames are
    both chunks' context and neither one's output, though, so a second line measures the pair
    that actually meets in the clip: the previous render's discarded look-ahead against the
    frames the next render emits for the same indices (`cut before frame N`), which is the only
    reading that can show a jump between two chunks. A jump that recurs inside one render is
    not a seam at all, so each chunk also reports the sharpest frame to frame change it made
    against the source's own motion there (`sharpest change inside the render`, with the
    typical pair beside it) - H3's sequences come in 17 frame steps, and a jump on that step
    belongs to the sampler, which no alignment of chunks removes. An offset that
    cannot be measured (too little structure to line up, or nothing in common) is reported as
    unmeasurable rather than given a number. The offset only raises a warning when it exceeds
    half a percent of the frame's short side (and at least 2 px): an independent redraw's own
    resampling never lands on exactly 0,0, so a pixel or two is the ordinary cost of a seam,
    while a chunk that lost frames to the grid drifts by tens. All of those numbers read a
    translation, though, and a chunk that redraws the frame a fraction of a percent larger lands on
    the frame centre: it measures a fraction of a pixel and still re-frames the picture at the cut,
    which is what the eye catches. So each chunk is also fitted against the previous render as a
    similarity - scale, angle, offset - and the same fit is run over the middle half of the frame
    and over the ring around it: a re-framed shot scales the two together, while a re-staged
    subject scales the middle and leaves the ring alone, and a subject a few percent larger moves
    the whole-frame reading by a fraction of a percent, which is what a re-framing looks like when
    it is not one. Beside it the same reading is printed for three adjacent pairs from inside the
    previous render - pairs with no seam anywhere near them - along with the spread they span, so a
    cut is compared against what this clip actually does frame to frame instead of against one
    lucky sample: on a shot with real camera motion a single pair can read several percent on its
    own, and subtracting that from the cut manufactures a difference out of the control. Five seams
    of a 180 frame clip read -0.16 to +0.02% on the whole frame and -0.45 to +0.37% for the middle
    against its edge, tighter than the ordinary pairs printed beside them, so that footage is not
    re-framed at its cuts at all - and `PIXEL_BATCH_SEAM_ALIGN` is off by default, because
    resampling every frame of every chunk to chase half a percent only softens the picture. The
    similarity fit, its trust bounds and the warp stay behind that switch for a shot that does
    re-frame. What is left at a cut that measures like this is drawn rather than placed - the same
    frame rendered twice, with the detail and sometimes the object re-invented at the same size -
    and no warp, global or local, removes it: the correlation line is where it shows (0.90-0.95
    across a cut against 0.94-0.99 between frames inside one render), and the lever is what the
    second render was allowed to re-invent, not what is done to its frames afterwards.
    Two runs at denoise 0.15 and 0.10 withdraw that reading, though: the cut's tone figure fell
    from 0.040-0.050 at 0.25 to 0.028-0.033 and then 0.020-0.025, its detail figure from
    0.027-0.030 to 0.018-0.021 over the last two, and the jump stayed the same size to the eye.
    What a seam costs measurably scales with how much freedom the redraw had; what is seen does
    not, so it is not the redraw re-inventing the frame. Every figure above also compares two
    renders of one timestamp, and that pair cannot see a step in the movement at all: if the two
    renders agree at frame N, the cut from the clip's frame N-1 into N is no larger than the step
    the previous render was already taking there. So each seam also reports the pair that playback
    cuts on - the last frame already on screen against the first frame this chunk emits - beside
    the same step taken twice inside the previous render (`the step into frame N is X px ...
    where this clip was stepping A to B px a frame`, with the angle against the direction the
    picture was going). A chunk that restarts the motion, slips it forward, repeats a frame or
    turns away lands outside that range and is warned about, which is the one seam failure the
    same-timestamp measurements are blind to by construction; a seam that reads calm there too is
    a difference in what was drawn rather than in the movement or the placement. That comparison
    has one honest limit: when the two renders agree at the timestamp, the step across the cut is
    the previous render's own step by construction, so a pan that is speeding up prints a ratio
    above 1 and one that is slowing down prints a ratio below, with no seam involved. Measured on a
    180 frame render panning 17 to 27 px a frame, all five cuts had the two renders agreeing at
    the timestamp to 0.27-1.73 px, so all five steps were the previous render's own - and three of
    them nevertheless sat outside their neighbours' range (0.56x, 1.29x, 1.88x), because the ratio
    was following the pan's acceleration and not a seam. The line now says which case it is: under
    the agreement limit it prints `that step is the previous render's own, not one the cut made`,
    and only a cut whose two renders disagree at the timestamp can raise the warning.
    Because a fit cannot say which of those a viewer is looking at, `PIXEL_BATCH_SEAM_DUMP` (on
    by default, no widget) writes six PNGs per cut into `temp/h3_seams`: the last frame on
    screen, the first frame after the cut, the frame the previous render had for that index and
    this one did not use, their raw difference at 8x, that same pair with the fit between them
    undone (also at 8x), and the last of those run again over an ordinary pair from inside the
    previous render as a control. The raw difference is only readable on a shot that holds still:
    two frames of a pan a frame apart already differ everywhere, so at 8x it fills the frame with
    edges and shows nothing - a 180 frame clip panning near 20 px a frame does exactly that - and
    a step and a redraw are then indistinguishable in it. `_residual.png` is the one to look at:
    the movement measured in the pair is warped out first, so a cut that only placed its content
    differently goes dark while a cut that changed it keeps its difference where the change is,
    and its figure is printed against the control's, which is what this clip leaves between two
    frames anyway. Read it as a ratio against a 1.5x bar: below it the two frames are the same
    picture with its texture sampled again, at or above it the cut is changing the picture, and
    the number is printed beside the sentence so the one can be checked against the other. On the
    render above, the three cuts whose fit converged read 1.49x, 1.35x and 1.13x - all the same
    picture. On a synthetic pair, moved only measured 1.2x its control, an object appearing
    measured 4.9x with the brightest part of the residual sitting on the object. A cut whose fit
    does not converge says so rather than printing nothing, because a missing line reads as a
    clean seam.
    One change no pair of frames can see is the one that touches every pixel without moving any
    of them: the whole picture arriving finer or softer than the frames it follows. Every figure
    above fits one frame onto another and reads calm straight through it, which is exactly what a
    chunk head drawn with a different amount of blur or grain looks like - and a viewer calls that
    the whole frame shifting, not something inside it. So each seam also prints the fine detail
    each frame holds, frame by frame across the cut (`fine detail held by each frame across the
    cut into frame N - 0.0145 0.0145 0.0145 | 0.0120 0.0120 0.0145 0.0145`), beside the most the
    same reading moves between frames inside the previous render. A pan that is speeding up
    softens a shot gradually and cannot put one frame on the wrong side of a cut; a head that is
    an impulse - softer for two frames, back to the level it came from by the fourth - is the
    shape this line exists to catch. On synthetic frames a pure pan read +0%, and a two frame
    head softened by a 0.55 sigma blur read -17% with that recovery visible in the same line.
    The first render to carry the line answered it: the heads were not softer but **finer**, at
    two of the five cuts - +19% into frame 36 and +15% into frame 129, where frame to frame
    inside the previous render the same reading moves 2% and 1% - marginal at 67 (+4%) and absent
    at 98 and 160 (+0% and +3%, against 5% wobble in the frames themselves there). It is also
    sustained: the whole of the chunk after the cut sits at the new level rather than the first
    two frames. So a render carries one sharpness for its whole length and the next render can
    carry a different one, which arrives at the cut as the whole picture changing while every
    position, scale and tone figure stays calm - and which no pair-fitting measurement can see,
    because it moves nothing. `PIXEL_BATCH_SEAM_DETAIL_MATCH` (on by default, no widget) acts on
    it: each chunk's fine detail, in that same 1.2 px at 480 px band, is scaled by one figure for
    the whole chunk - the ratio of the frames before the cut to this chunk's head, bounded to
    0.74-1.35x and skipped below 3% - before the frames are emitted, so the look-ahead and the
    tail handed to the next chunk carry it too. One constant over every frame cannot put a step
    inside the chunk, and no pixels from the neighbouring render are mixed in, so it cannot
    flicker, ghost or soften a seam into a dissolve; it only stops the picture arriving at a
    different sharpness exactly at a cut. On synthetic frames at 1792x768 a head 15% finer and
    one 16% softer both came back to within 1.3% of the level they followed. It is a correction
    to what the redraw chose, not to what it drew: the frames either side of a cut are still
    different draws of the same picture, and a seam that jumps where no match was logged is a
    different mechanism than this one.
    The figure that decided whether to act was the one with a suspect in it: the frames before a
    cut are different frames from those after it, so a pan moving into more detail can raise that
    number without any render difference at all. The previous chunk's look-ahead is that chunk's
    own pixels for the frames this one emits, so the same timestamps are rendered twice and both
    draws are in hand at the cut. `the head frames drawn twice` prints exactly those two and is
    what the match is driven by, since content cannot differ between two draws of one frame:
    what is left is the render. The across-cut figure is still printed beside it, and a content
    change that moved only that one now leaves the correction at 1.00x.
    Every one of those measurements is trying to make two renders of one shot agree. On footage
    that has been cut there is a cheaper answer, and it is the only one now used: put the seams on
    the cuts. `PIXEL_BATCH_SCENE_CUTS` (on by default, no widget) reads the source clip for hard
    cuts once per batch - each frame reduced to a 96 px thumbnail, frames compared by correlation,
    where a cut falls to nothing and a 27 px a frame pan is a pixel and a half of travel - and
    keeps a candidate only if it clears the clip's own median plus a multiple of its spread **and**
    stands three times above the frames immediately around it, which is what tells a cut from a
    fast passage inside one shot. The cuts are logged by frame number, or the camera says plainly
    that the clip is one continuous take and no seam is placed on it at all.
    A seam then goes on each cut that can carry one, and nowhere else. Auto does not size a batch
    any more: the clip is split into its shots, and a shot is one render, so the shot decides where
    renders end and not a frame count. A continuous take is one render with no seam in it anywhere -
    unless the clip cannot be one render at all: a render must be an H3 grid length (22 + 17k), so a
    180 frame clip is a 175 frame render plus a short tail render, and where that fall lands on a
    cut it lands on the cut. The other thing that can split a shot is the card: a render that runs
    out of memory is cut back to the shortest render that still holds its frames (giving up the
    look-ahead context, which costs no seam), or halved at a frame boundary with the usual overlap
    and look matching, and retried - the log says which, and why. A VRAM estimate never places a
    seam; it only prints how many frames the free memory is expected to hold.
    Those in-shot seams keep the overlap and the look matching described above: a seam inside a shot
    is the only kind with nothing under it to hide behind. Across a cut nothing crosses - no
    overlap, no `seam_prefix`/`pin_frames`/`seam_guide_frames` from the shot before, no positional
    alignment and no tone lock, since a new shot's level follows its own source and matching it to
    the old shot's would move it away from what it is. Two shots meeting at a hard cut are a cut:
    there is no pair of frames that has to agree, and nothing is matched across it. A cut is skipped
    only where a shot is too short to be a render of its own (22 frames, the H3 grid base); that cut
    is then drawn inside a render that straddles it, and the log names the frame.
    Every render is a grid length, so a short shot's render runs a few frames past its cut into the
    next shot - drawn and dropped, exactly like the look-ahead at any other seam, with the seam
    still on the cut - and the last render of a clip ends on the last frame and backs up over frames
    already emitted rather than coming back off the grid. The cost of a cut seam is therefore that
    handful of dropped frames and nothing else: no shot is ever redrawn with another shot's content
    inside it, and no frame is ever blended between two shots.
    Faces are tracked **once for the
    whole clip** before chunking and the plan is
    scaled into every chunk, so a face keeps one track, one crop size and one mask across chunk
    boundaries. Each track is also
    diffused **once for the whole clip**, before the chunks run: its face crops are upscaled to the
    output resolution with the selected `upscale_model` in small batches (bicubic when it is `None`,
    and a failure falls back to bicubic), the masks are re-detected inside those crops (a big
    upscale then gets output-resolution masks instead of an upscaled outline), and the whole track is
    sampled as one sequence, which the chunks only composite. The decoded crops stay in RAM up to a
    budget (an eighth of system RAM, clamped to 512 MB - 2 GB) and spill per track to a temp
    directory beyond it, memory-mapped so each chunk reads only the frames it needs), `latent`
    (learned 3D resize of the `latent` with `latent_upscale_model`, then decode it with the `vae`).
  - `upscale_model` (dropdown from `models/upscale_models`, `pixel` mode; `None` = bicubic),
    `latent_upscale_model` (dropdown from `models/latent_upscale_models`, `latent` mode),
    `scale_to_megapixels` (output size; aspect ratio kept, both sides snapped to a multiple of
    32 px, never shrinking), `tile_width`/`tile_height`, `tile_padding`,
    `seam_fix_width`/`seam_fix_denoise`/`seam_fix_padding`, `seed`, `steps`,
    `sampler_name`, `scheduler`, `denoise`, `diffusion_mix`, `edge_protect`,
    `detail_retention`, `latent_noise_offset`, `redraw_only`, `align_to_source`,
    `locks_follow_denoise`. Tile diffusion always samples at `cfg` 1.0 and there is no
    negative conditioning input.
  - *Face Detail*: `enable_face_detailer`, `same_face` (default off: every detected face is
    diffused on its own, with no shared master; on: the largest face becomes the master and the
    smaller faces are drawn toward it, which is the old behaviour), `reference_face` (IMAGE), `face_mask` (MASK),
    `face_denoise`, `face_padding`, `face_size`, `face_detector`
    (`auto` / `yolo` / `haar` / `sam3`), `face_detector_model`, `face_confidence`,
    `manual_face_boxes`, `sam3_prompt`, `reference_strength`, `reference_pose_match`,
    `reference_flow_align`, `face_structure_lock`, `face_chroma_lock`, `face_seamless_blend`,
    `frontal_only_threshold`, `face_detect_stride`. A video batch is **tracked**: faces are
    detected on `face_detect_stride` - `auto` scans every frame up to about 180 frames per clip
    and then widens the stride (capped at every 8th), so a long clip's tracking time stays
    bounded - then associated into tracks by box overlap (every gap between detections
    interpolated, each trajectory smoothed, and the interpolated frames inherit the nearest
    detection's instance mask and frontal score so mask cleanup and the temporal filter still
    apply), and every track keeps ONE crop size for its whole
    length, so the redrawn face no longer jumps in size and position from frame to frame. Every
    track is then redrawn in ONE temporal diffusion pass over all of its frames - the reference
    workflow's single KSampler over the whole crop sequence - which is both far faster than one
    sampler run per frame and temporally coherent; registration, colour/detail locks, blending
    and pasting still happen per frame. Crops are worked at about 0.5 MP (never below their
    native size, never above `face_size` on the long side) and detection runs on a copy capped at
    960 px, with the boxes and masks scaled back to the canvas. SAM3 is loaded once for the whole
    clip instead of once per frame, every track's masks go through the reference pack's cleanup (12 px
    shrunk off the edge, 16 px grown back) and a +/-2 frame majority filter before they are composited,
    and the diffusion pass runs the node's step count inside the denoise window: 20 steps at denoise
    0.25 is 20 sampler iterations, not 4. The log reports the detection time, the per-track pass time
    and the step count it actually ran, so a slow stage is visible instead of guessed. A single image, a shot without
    faces, a failed plan and the self-mastering `same_face` mode fall back to per-frame
    diffusion. Only the pixels the detector actually looks at are converted off the GPU.
  - *Refine*: `enable_refiner`, `refine_amount`, `refine_radius`.
  - *DLSS 5*: `enable_dlss5`, `dlss5_scale_to_megapixels`, `nr_preset`, `nr_style`, `nr_intensity`,
    `local_tone_strength`, `local_structure_strength`, `skin_structure_strength`,
    `automatic_mask`, `dlss_model_preset`, `motion`, `scene_change_threshold`,
    `warmup_frames`, `runtime_dir`, `verify_neural_rendering`, `dlss5_scale_to_megapixels`.
    `dlss5_scale_to_megapixels` replaces the old `upscaling_mode` preset: a continuous megapixel
    target for this pass alone, independent of the H3 Upscale group's `scale_to_megapixels`, so
    stacking DLSS on top of an upscale adds real resolution. The node picks the smallest DLSS mode
    that covers the target and then fits the result to it with the source aspect kept
    (`exact_aspect_target`). 0 disables DLSS upscaling and keeps the input size; the pass never
    shrinks the image.
  - *Color FX*: `enable_colorfx`, stage toggles, `tone_lock`, `chroma_lock`,
    `hdr_intensity`, `shadow_intensity`, `highlight_intensity`, `gamma`, `brightness`,
    `contrast`, `saturation`, `enhance_color`, `lut_name`, `lut_strength`, `lut_log_process`,
    `sharpness`, `edge_enhance_strength`, `detail_enhance_strength`, blur radii,
    chromatic aberration, grain, scanlines, vignette and soft-light controls.
- **Outputs:** `image` (IMAGE), `face_debug` (IMAGE), `audio` (AUDIO — the audio stream of a
  nested H3 AV latent decoded with `audio_vae`; `None` when `latent` is not connected, is
  video-only, or no `audio_vae` is wired, so image-only graphs are unaffected; a connected
  `audio` input is returned unchanged in every mode, ahead of the latent's decoded audio).
  Several latents arriving at once are rendered in order and merged: `image` and `face_debug`
  are one batch holding every latent's frames, and `audio` is one waveform with the segments
  joined along time - never a list of per-latent batches.
- **Notes:** the last two stages (DLSS 5, Color FX) are post-passes on the decoded image;
  `tone_lock` and `chroma_lock` pin the final broad tone/colour to the clean pre-diffusion
  upscale, and `locks_follow_denoise` fades those "keep the source" corrections out as
  denoise rises. YOLO face models are scanned from `models/ultralytics/bbox`, LUTs from
  `models/luts` (`.cube`). All groups start collapsed and are driven by
  `js/h3_upscale_groups.js`, which detects the node by the presence of `grp_h3_upscale` rather
  than by name.

## `triXope/image` — Image

Image processing, colour and preview. Module: [nodes_image.py](nodes_image.py).

### Δ ColorFX 🎬 — `triXope ColorFX`

A complete grading stage in one node: colour correction, LUT application, sharpening/detail
enhancement, blur effects and stylistic film effects, each behind its own enable switch.

- **Inputs:** `images` (IMAGE), `enable`, `enable_color_correction`, `enable_lut_processing`,
  `enable_enhancements`, `enable_blur_effects`, `enable_stylistic_effects`, plus group
  widgets: *Color Correction* (`grp_color_correction`, `hdr_intensity`, `shadow_intensity`,
  `highlight_intensity`, `gamma`, `brightness`, `contrast`, `saturation`, `enhance_color`),
  *LUT* (`grp_lut`, `lut_name`, `lut_strength`, `lut_log_process`), *Enhancements*
  (`grp_enhancements`, `sharpness`, `edge_enhance_strength`, `detail_enhance_strength`),
  *Blur* (`grp_blur`, `blur_radius`, `gaussian_blur_radius`, `radial_blur_strength`,
  `radial_blur_center_x`/`_y`, `radial_blur_focus_spread`, `radial_blur_steps`) and
  *Stylistic* (`grp_stylistic`, chromatic aberration R/B offsets, `chromatic_blur_amount`,
  `simple_film_grain_intensity`, `simple_film_grain_monochrome`, `scanline_intensity`,
  `vignette_intensity`, `vignette_center_x`/`_y`, `soft_light_opacity`,
  `soft_light_blur_radius`).
- **Outputs:** `IMAGE`.
- **Notes:** LUTs are `.cube` files from `models/luts`; `lut_log_process` applies the LUT in
  log space for film-style grades. `js/colorfx_groups.js` turns the `grp_*` widgets into
  collapsible section bars.

### Δ HSV 🎨 — `triXope HistogramMatchHSV`

Matches the contrast and saturation of a target batch to a reference image by histogram
(CDF) matching in HSV space.

- **Inputs:** `reference_image` (IMAGE — only the first frame is used), `target_image`
  (IMAGE), `mix_factor` (FLOAT `0`–`1`, default `1.0`), `match_hue` (BOOLEAN, default off).
- **Outputs:** `IMAGE`.
- **Notes:** saturation and value are matched per target frame; `match_hue` additionally
  shifts each frame's hue by the difference of the mean hues, and `mix_factor` blends the
  matched result back over the original.

### Δ Upscale Image to Size (Model) — `triXope ImageUpscaleToSize`

Uses an upscale **model** to reach an exact target size: the image is first downscaled so the
model's scale factor lands on the requested dimensions, keeping the model's own resampling
character instead of a plain resize.

- **Inputs:** `upscale_model` (UPSCALE_MODEL), `images` (IMAGE), `model_scale_factor`
  (default `4.0`), `final_width` / `final_height` (default `1024`; `0` = derive from the
  model factor), `keep_proportion` (`stretch` / `crop` / `pad`), `downscale_method`
  (`nearest-exact`, `bilinear`, `area`, `bicubic`, `lanczos`), `crop_position`
  (`center` / `top` / `bottom` / `left` / `right`), `pad_color` (RGB string, default
  `0, 0, 0`), `per_batch` (default 8).
- **Outputs:** `IMAGE`.
- **Notes:** the model is moved to the torch device for the run and returned to CPU
  afterwards; progress is reported through ComfyUI's progress bar.

### Δ Image Batch Filter — `triXope ImageBatchSelector`

Keeps or drops frames from an image batch on a fixed stride.

- **Inputs:** `images`, `select_every_nth` (default 1), `skip_first_images` (default 0),
  `inverse_selection` (BOOLEAN, default off).
- **Outputs:** `IMAGE`.
- **Notes:** `inverse_selection` returns everything *except* the strided frames. If the
  selection would be empty (or `skip_first_images` exceeds the batch), the last frame is
  returned so downstream nodes do not crash.

### Δ Panorama 360 Rotate — `triXope Pano360Rotate`

Turns a single equirectangular panorama into a rotating flat-frame sequence by driving
FFmpeg's `scroll` + `v360` filters: a 2-second horizontal pan rendered at the requested
output size.

- **Inputs:** `image` (IMAGE — first frame is used), `width` (default `4096`, 256–8192),
  `height` (default `2048`, 128–4096).
- **Outputs:** `images` (IMAGE sequence — one frame per FFmpeg output frame).
- **Notes:** requires FFmpeg, taken from `imageio-ffmpeg` if installed or from `PATH`.
  Frames are read from FFmpeg's rawvideo stdout and normalised to `0..1`.

### Δ Image Sequence Preview — `triXope Image Sequence Preview`

Previews an image batch as an animated WebP in the node body.

- **Inputs:** `fps` (FLOAT, default `24.0`), optional `images` (IMAGE).
- **Outputs:** none (output node); returns UI `images` plus an `animated`/size caption.
- **Notes:** frames are thumbnailed to at most 960×540 (LANCZOS) before being written to the
  temp directory as a single lossy WebP animation with `duration = 1000/fps` per frame.

## `triXope/io` — Input/output

File, path, text and image output. Module: [nodes_io.py](nodes_io.py).

### Δ Load Text — `triXope LoadText`

Loads a screenplay or lyrics file from any path into a STRING.

- **Inputs:** `directory` (STRING, default `_film/input/text`), `file` (wildcard `*` filled
  by the frontend from the directory scan).
- **Outputs:** `OUTPUT` (STRING).
- **Notes:** resolve order is: absolute path, then relative to ComfyUI's `input` directory,
  then `abspath`. Recognised extensions: `.data`, `.env`, `.info`, `.json`, `.log`, `.text`,
  `.txt`, `.yaml`, `.yml`; decodings are tried in order `utf-8`, `latin-1`, `cp1252`,
  `utf-16`, `utf-16-le`. The scan is served by `GET /triXope/scan_directory`, which also
  returns the resolved directory.

### Δ Filename Extractor — `triXope FilenameFromDirectory`

Returns the file name at a given index of a directory, using the same alphabetical order as
the directory listing.

- **Inputs:** `directory_path` (STRING), `file_index` (INT, default 0).
- **Outputs:** STRING (file name only, no path).
- **Notes:** raises `ValueError` for a non-directory path and `IndexError` when the index is
  out of range.

### Δ File Copier — `triXope File Copier`

Copies one file to a destination folder, keeping metadata, and returns the new full path.

- **Inputs:** `source` (STRING, force input), `destination` (STRING — a folder),
  `use_absolute_paths` (BOOLEAN, default off; when off both paths are resolved with
  `abspath`).
- **Outputs:** `filepath` (STRING).
- **Notes:** output node. Creates the destination directory, validates read/write
  permissions and raises descriptive errors when the source is missing or the copy fails.

### Δ Path Combiner — `triXope PathCombinerNode`

Joins a base path with up to ten optional subfolders and emits ten ready-to-use paths.

- **Inputs:** `base_path` (STRING, default `.`), `subfolder_1` … `subfolder_10` (optional
  STRING).
- **Outputs:** `path_1` … `path_10` (STRING).
- **Notes:** strings are stripped of quotes, newlines and leading/trailing slashes before
  joining; results are absolute and normalised to Windows-style backslashes.

### Δ Create Directory — `triXope EnsureDirectoryNode`

Creates a directory (including parents) and returns its absolute path.

- **Inputs:** `directory_path` (STRING, default `./my_output_folder`).
- **Outputs:** `absolute_path` (STRING).
- **Notes:** output node, so it runs even with nothing connected; existing directories are
  reported and left alone.

### Δ Text Saver — `triXope TextSaver`

Writes text chunks to disk, optionally appending, and releases the whole file only once a
loop is finished.

- **Inputs:** `path` (STRING, default `./output`), `filename` (STRING, default
  `saved_text`), `text` (STRING, multiline), `append` (BOOLEAN), `total_iterations` (INT,
  default 1), optional `current_iteration` (INT).
- **Outputs:** `output` (STRING).
- **Notes:** adds `.txt` when missing and an internal per-file counter is used when
  `current_iteration` is unlinked, so the node returns `""` for every iteration except the
  last, which returns the accumulated file contents (keeps loop memory flat).

### Δ Save Image — `triXope ToggleFolderSaver`

Saves the image batch to an arbitrary folder when enabled and passes the images through.

- **Inputs:** `images` (IMAGE), `enabled` (BOOLEAN, default on), `folder_path` (STRING),
  `filename_prefix` (STRING, default `custom_save`); hidden `prompt` / `extra_pnginfo`.
- **Outputs:** `images` (IMAGE).
- **Notes:** writes PNGs with the workflow embedded as PNG metadata and picks the next free
  `prefix_00001.png`-style name; when disabled, it is a pure pass-through.

### Δ Directory File Selector 📂 (Live) — `triXope DirectoryFileSelectorGemini`

Resolves `directory` + `file` into a full path, with a frontend dropdown that re-scans the
directory whenever the path changes.

- **Inputs:** `directory` (STRING), `file` (wildcard `*`).
- **Outputs:** `file_path` (STRING).
- **Notes:** `GET /triXope/get_files` returns the sorted file list for a directory (registered
  only once, defensively). Invalid directories, empty selections and missing files return an
  empty string with a console message instead of raising.

## `triXope/latent` — Latents and VAE

VAE encode/decode and latent file handling. Module: [nodes_latent.py](nodes_latent.py).

### Δ MiniMax H3 AV Latent Concat — `triXope MiniMaxH3AVLatentConcat`

Joins two MiniMax H3 joint audio/video latents along time and snaps the result back onto H3's
length grid.

- **Inputs:** `latent_a` (LATENT), `latent_b` (LATENT), `trim_side` (`b_start` / `a_end` —
  which side of the seam loses frames), `extra_trim_latents` (INT, multiple of 5, default 0),
  optional `sync_audio` (BOOLEAN, default on).
- **Outputs:** `latent` (LATENT).
- **Notes:** expects `NestedTensor` latents with exactly `[video, audio]`; the joined video
  length is snapped to `T_v ≡ 2 (mod 5)`. With `sync_audio` the audio length is recomputed from
  the model's own `frame_count → audio_t` formula so both streams stay in sync. Silent
  mono/stereo mismatches are upmixed with a warning; channel-count or resolution mismatches
  raise.

### Δ Latent List Decoder — `triXope SafeLatentListDecoder`

RAM-safe decoder for lists of latents that returns stitched video frames plus audio.

- **Inputs:** `vae` (VAE), `latents` (`*` — a latent dict, a list of latents, or an
  Akatz-style accumulation object), optional `audio_vae` (VAE), optional `vram_flush`
  (BOOLEAN, default on).
- **Outputs:** `images` (IMAGE), `audio` (AUDIO).
- **Notes:** declares `INPUT_IS_LIST = True` so ComfyUI does not split the list into separate
  runs; it flattens the payload, decodes one latent at a time on the GPU while holding frames
  on CPU, and can flush VRAM between chunks. The audio sample rate is read from the audio VAE
  with a 44100 Hz fallback.

### Δ Save WAN Latent — `triXope SaveWanVideoLatent`

Saves a LATENT to `<prefix>_0001.latent` in a folder, continuing an existing numbering.

- **Inputs:** `samples` (LATENT), `save_path` (STRING, default `./output/latents`),
  `prefix` (STRING, default `video_latents`).
- **Outputs:** `directory_path` (STRING).
- **Notes:** writes a `torch.save` dict containing `latent_tensor`, every other key of the
  incoming latent, and `latent_format_version_0`.

### Δ Save LTXV Latent — `triXope SaveLTXVideoLatent`

LTXV counterpart of the WAN saver with the same numbering scheme and outputs.

- **Inputs:** `samples` (LATENT), `save_path` (STRING), `prefix` (STRING).
- **Outputs:** `directory_path` (STRING).
- **Notes:** only adds `latent_format_version_0` when the incoming latent does not already
  carry it, so LTXV latents keep their own format marker.

### Δ Load Video Latent — `triXope LoadAndConcatVideoLatentsFromDir`

Loads every `.latent` file in a folder (sorted) and concatenates them into a single latent.

- **Inputs:** `directory_path` (STRING), `latent_load_cap` (INT, `-1` = all),
  `skip_first_latents` (INT), `concat_dimension` (`batch (dim=0)` / `time/frames (dim=2)`).
- **Outputs:** `latent` (LATENT).
- **Notes:** reads safetensors first, then `torch.load`; accepts `latent_tensor` or `samples`
  keys. A first pass records shapes and dtypes, then one pre-allocated CPU tensor is filled
  file by file and garbage-collected as it goes, which keeps peak RAM low. Metadata from the
  first file is carried onto the result.

### Δ Wan Directory Video Decoder — `triXope WanDirectoryVideoDecoder`

Decodes a folder of `.latent` clips with a Wan VAE and stitches them into one video.

- **Inputs:** `directory_path` (STRING), `vae` (WANVAE), `latent_load_cap` (INT, `-1` = all),
  `skip_first_latents` (INT), `tile_x` (default `1792`), `tile_y` (default `768`).
- **Outputs:** `image` (IMAGE).
- **Notes:** moves the VAE to the torch device, casts each clip to the VAE dtype, decodes with
  tiling (`tile × 8` latent, 256-pixel stride), normalises `[-1,1]` output to `0..1` and
  concatenates along the time axis.

### Δ LTXV Directory Video Decoder — `triXope LTXVDirectoryVideoDecoder`

LTXV counterpart of the directory decoder, written for low system RAM.

- **Inputs:** `directory_path` (STRING), `vae` (VAE), `latent_load_cap` (INT),
  `skip_first_latents` (INT).
- **Outputs:** `image` (IMAGE).
- **Notes:** decodes each clip through `vae.first_stage_model`, writes fp16 frames straight to
  a temporary binary file inside the latent folder, then memory-maps that file into the output
  tensor — so the full video is never held in RAM. Falls back to `vae.decode` if the direct
  decode fails; the temp folder is removed on error.

### Δ VAE Encode Batched (WanVideo) — `triXope WanVAEEncodeBatched`

Encodes a long frame sequence in overlapping batches so Wan VAE encoding fits in VRAM.

- **Inputs:** `pixels` (IMAGE), `vae` (WANVAE), `per_batch` (multiple of 4, default 16),
  `enable_vae_tiling`, `tile_x`, `tile_y`, `tile_stride_x`, `tile_stride_y`.
- **Outputs:** LATENT (`{"samples": …}`).
- **Notes:** keeps 16 frames of context between batches and drops the duplicated latent
  frames; pads the tail by repeating the last frame and trims latents to
  `ceil(frames/4)`. `per_batch` is rounded down to a multiple of 4. Requires at least 16
  input frames.

### Δ VAE Decode Batched (WanVideo) — `triXope WanVAEDecodeBatched`

Decodes a Wan video latent in overlapping latent-frame batches.

- **Inputs:** `samples` (LATENT), `vae` (WANVAE), `per_batch` (new latent frames per batch,
  default 4), `enable_vae_tiling`, `tile_x`, `tile_y`, `tile_stride_x`, `tile_stride_y`.
- **Outputs:** `IMAGE`.
- **Notes:** carries two latent frames (8 pixel frames) of overlap between batches and
  discards the duplicated decoded frames; payload is mapped from `[-1,1]` to `0..1`. Uses the
  TAEHV fast path when the VAE is a TAEHV instance and `taehv.py` is present.

### Δ VAE Decode Batched (Native) — `triXope VAEDecodeBatched_Native`

Batched decoder for any standard ComfyUI VAE, with optional temporal overlap.

- **Inputs:** `latents` (LATENT), `vae` (VAE), `per_batch` (default 8), `latent_overlap`
  (default 0), `enable_vae_tiling`, `tile_x`, `tile_y`.
- **Outputs:** `IMAGE`.
- **Notes:** reshapes 5-D `[B,C,F,H,W]` video latents into a 4-D frame batch; when
  `latent_overlap > 0` the previous chunk's tail is prepended as context and the duplicated
  frames are dropped, which keeps long videos temporally consistent under low VRAM.

## `triXope/loaders` — Loaders

Model and image loaders. Module: [nodes_loaders.py](nodes_loaders.py).

### Δ Universal Loader (MultiGPU) — `triXope UniversalLoaderMultiGPU`

All-in-one loader for UNet/DiT, one or two text encoders, an image/video VAE and an optional
audio VAE, with automatic multi-GPU distribution.

- **Inputs:** `unet_name` (safetensors or `.gguf`), `clip_name1`, `clip_name2` (second encoder
  optional; `none` disables), `vae_name`, `audio_vae_name`, `model_type` (architecture list
  from `autogpu_pytorch.FULL_MODEL_TYPES`, default `auto`), `distribution_mode`
  (`Auto: UNet GPU0 / Encoders GPU1`, `Auto: Balanced Free VRAM`, `Manual Override`),
  optional `weight_dtype` (`default`, `fp8_e4m3fn`, `fp8_e4m3fn_fast`, `fp8_e5m2`,
  `float16`, `bfloat16`, `float32`) and `manual_unet_device` / `manual_clip_device` /
  `manual_vae_device`.
- **Outputs:** `MODEL`, `CLIP`, `VAE`, `AUDIO_VAE`.
- **Notes:** GGUF weights go through `gguf_support` (dequant dtype from `weight_dtype`),
  safetensors through `comfy.sd.load_diffusion_model` with `load_device`/`offload_device` set
  per device plan. The CLIP type is derived from `model_type` (falling back to the UNet file
  name), and both encoders are loaded as one CLIP when two are selected.

### Δ Efficient Image Batch Loader — `triXope EfficientImageBatchLoader`

Loads a numbered frame sequence into one IMAGE batch using a memory-mapped scratch file, so
large sequences do not need to fit in RAM as Python objects.

- **Inputs:** `directory_path` (STRING), `start_index` (INT, default 1), `frame_count` (INT,
  default 100), `file_pattern` (printf-style, default `frame_%04d.png`).
- **Outputs:** `IMAGE`.
- **Notes:** dimensions are taken from the first frame; every frame is written into a
  `np.memmap` temp file and the batch is exposed as a torch tensor sharing that memory.
  Raises `FileNotFoundError` when the pattern matches nothing.

### Δ Load (Character) Image — `triXope Load (Character) Image`

Image picker limited to `input/_film/input/characters`.

- **Inputs:** `image` (combo of the images found there, with upload enabled).
- **Outputs:** `IMAGE`, `filename` (STRING, without extension).
- **Notes:** applies EXIF orientation, converts to RGB and normalises to `0..1`;
  `IS_CHANGED` hashes the file so edits re-trigger execution.

### Δ Load (Outfit) Image — `triXope Load (Outfit) Image`

Same loader for `input/_film/input/outfits` (inputs, outputs and behaviour identical to the
character loader).

### Δ Load (Scene) Image — `triXope Load (Scene) Image`

Same loader for `input/_film/input/scenes` (inputs, outputs and behaviour identical to the
character loader).

## `triXope/utilities` — Utilities

Switches, routing, selection and display helpers. Module: [nodes_utilities.py](nodes_utilities.py).

### Δ Any Cast — `triXope AnyCast`

Presents any incoming value as a chosen ComfyUI type, so a `*` output can feed a typed input.

- **Inputs:** `TYPE` (combo: `*`, `STRING`, `INT`, `FLOAT`, `BOOLEAN`, `IMAGE`, `LATENT`,
  `MASK`, `NOISE`, `SAMPLER`, `SIGMAS`, `GUIDER`, `MODEL`, `CLIP`, `VAE`, `CONDITIONING`,
  `PYLIST`), optional `ANY` (`*`).
- **Outputs:** `*`.
- **Notes:** `STRING`, `INT`, `FLOAT` and `BOOLEAN` also convert the value; booleans accept
  numbers (> 0 is true) or numeric strings. A `None` input returns an `ExecutionBlocker`, which
  stops the downstream branch instead of failing the run. The socket label is updated live by
  `js/anycast.js`.

### Δ Any Switch — `triXope AnySwitch`

Chooses between two `*` inputs with graph-aware lazy evaluation.

- **Inputs:** `select` (INT 1–2), `mode` (`Smart Fallback` / `Strict Select`), optional
  `any_1` / `any_2` (`*`, lazy).
- **Outputs:** `*`.
- **Notes:** in *Smart Fallback* it returns the first non-empty input from `select` onwards
  (empty means `None` or an empty context dict), so a missing branch falls through to the next
  one; in *Strict Select* only the selected socket is used. `check_lazy_status` only asks for
  sockets that are actually wired, which avoids errors on unplugged inputs.

### Δ Any Switch Inversed — `triXope AnySwitchInversed`

Routes one value to exactly one of two outputs.

- **Inputs:** `select` (INT 1–2), optional `any_value` (`*`).
- **Outputs:** `any_1`, `any_2` (`*`).
- **Notes:** the unselected output is `None` — deliberate, so it can act as a graph-level
  branch gate.

### Δ Text Switch — `triXope text_switch`

Picks one of up to 100 text inputs by index.

- **Inputs:** `index` (INT 0–9 selection field), optional `text0` … `text99` (STRING, force
  input).
- **Outputs:** `text` (STRING).
- **Notes:** on ComfyUI builds new enough to support lazy evaluation the unselected text
  sockets are not evaluated.

### Δ Prompt Counter — `triXope CurrentPromptDisplay`

Shows "working on x to y of z" as a neon banner while a loop runs, and passes the signal on.

- **Inputs:** optional `start`, `end`, `total` (INT/FLOAT/STRING, force input) and `signal`
  (`*`, non-lazy so it executes inside the loop).
- **Outputs:** `signal` (`*`, passthrough), `text` (plain STRING sentence).
- **Notes:** output node; `IS_CHANGED` returns `NaN` so it re-runs every iteration. The styled
  banner is delivered through the UI payload (`js/neon_display.js`).

### Δ Realtime Loop Tracker — `triXope RealtimeLoopTracker`

Reports loop progress (index/total) to the node body in real time.

- **Inputs:** optional `index` (INT, force input), `total` (INT, force input); hidden
  `unique_id`.
- **Outputs:** `index` (INT).
- **Notes:** output node that sends a `loop_tracker_update` websocket event per iteration
  (`js/eta_tracker.js` renders elapsed/remaining time). Unwired, it quietly returns 0.

### Δ LTXV Real-Time ETA — `triXope LTXV_ETA_Display`

Display sink for the `real_time_eta` (`ETA_TRACKER`) output of Film Auteur (LTX).

- **Inputs:** `real_time_eta` (ETA_TRACKER, force input); hidden `unique_id`.
- **Outputs:** none (output node).
- **Notes:** the timing work happens in the producer; this node only anchors the tracker in
  the graph so the frontend can draw it.

### Δ Group Monitor — `triXope GroupMonitor`

Shows which workflow group is currently executing.

- **Inputs:** optional `passthrough` (`*`).
- **Outputs:** `passthrough` (`*`).
- **Notes:** frontend-only display; the passthrough exists so the node is not pruned when it
  is unconnected. `js/GroupMonitor.js` watches ComfyUI's `executing` messages.

### Δ Final Frame Selector — `triXope FinalFrameSelector`

Returns the last frame of a batch as a single-image batch.

- **Inputs:** `images` (IMAGE).
- **Outputs:** `IMAGE`.
- **Notes:** a 3-D single image is returned as a 1-frame batch; unexpected dimensionality
  returns `None`.

### Δ First Frame Selector — `triXope FirstFrameSelector`

Returns the first frame of a batch as a single-image batch (same handling as the final-frame
selector).

- **Inputs:** `images` (IMAGE).
- **Outputs:** `IMAGE`.

### Δ Smart Reference Filter — `triXope SmartReferenceFilter`

Keeps only the reference images whose character appears in the current shot prompt.

- **Inputs:** `split_prompt` (STRING, force input), `ref_image_batch` (IMAGE),
  `character_names` (STRING, multiline, comma separated, default `JOHN DOE, JANE DOE, ANY`).
- **Outputs:** `filtered_ref_image` (IMAGE), `has_character` (BOOLEAN).
- **Notes:** name *i* is matched against batch frame *i*; `ANY` always matches. With no
  matches it returns a zero-filled batch of the original shape and `False`, so the reference
  branch can be detected downstream rather than skipped.

### Δ Smart Conditioning Selector — `triXope SmartConditioningSelector`

Switches between a reference-conditioned branch and a plain text-to-video branch based on the
prompt.

- **Inputs:** `text_prompt` (STRING, force input), `pos_with_ref` (CONDITIONING),
  `latent_with_ref` (LATENT), `pos_no_ref` (CONDITIONING), `latent_no_ref` (LATENT),
  `character_names` (STRING, multiline, default `BLAKE, SYLVIA, CARY, KARAN`).
- **Outputs:** `positive` (CONDITIONING), `latent` (LATENT).
- **Notes:** if any listed name appears in the prompt (case-insensitive) the reference branch
  is returned, otherwise the clean branch; the decision is logged.

### Δ String Cleaner — `triXope StringCleaner`

Flattens a multi-line string into a single line by removing spaces, newlines and carriage
returns.

- **Inputs:** `text` (STRING, multiline).
- **Outputs:** STRING.

### Δ Group Highlighter — `triXope WorkflowGroupHighlighter`

Marks which workflow group is running, for the canvas highlight overlay.

- **Inputs:** none.
- **Outputs:** none (output node).
- **Notes:** has no compute; its execution is the signal `js/group_highlighter.js` listens for.

## `triXope/video` — Video

Video assembly, trimming and upscaling. Module: [nodes_video.py](nodes_video.py).

### Δ Video Timeline Editor — `triXope VideoTimelineEditor`

A full multi-track cut/trim/export editor: two image lanes, two audio lanes, scrubbing preview
and MP4 export, all driven from the node's own canvas UI.

- **Inputs:** optional `images` (IMAGE), `images2` (IMAGE), `audio` (AUDIO), `audio2`
  (AUDIO), optional `video` (STRING: the H3 Upscaler's `segments` folder list, a joined MP4,
  or a `;`-separated list of either), plus the hidden `timeline_json` / `ui_action` strings and
  the hidden `prompt`/`extra_pnginfo`/`unique_id`.
- **Outputs:** none (output node).
- **Notes:** caches the connected media per node id and pushes frame/sample/fps statistics to
  the frontend (`video_timeline_editor.media_info`). Every execution rebuilds the editor's
  timeline from the media it just received — one full-length clip per connected lane — so a
  second run never keeps clips, playback position or undo history from the run before it;
  the edit you leave in the node is what gets saved with the workflow. **Export** posts the
  timeline as it stands to `POST /video_timeline_editor/export_now`, which renders that edit
  from the cached media with PyAV (`libx264`, mixed audio) to a temp MP4 the UI downloads — the
  workflow is never queued, so exporting again after a cut or trim cannot regenerate upstream
  footage. The queued `ui_action: export` path is kept for compatibility and shares the same
  renderer. FPS is validated against `8, 12, 16, 24, 25, 30, 48, 50, 60`; image batches must
  be `[B,H,W,3|4]` with at least 2×2 pixels. Clips on one lane may overlap or jump to any
  source frame at a cut: overlapping image clips stack so the clip that starts last is
  drawn on top, the `images2` lane is composited over `images`, and overlapping audio
  clips from both audio lanes are summed into the mix. When `video` is connected and no
  `images` is, the editor reads the clips from disk instead of caching frames: the frame table
  is probed once from the containers, preview frames are decoded on demand through one open
  decoder, cut detection decodes at 48x48, and export re-encodes the timeline from the files -
  or copies a single untouched clip's packets without decoding at all. The files play as one
  continuous sequence in list order, so the H3 Upscaler's `segments` output wires straight in.
  The segment writer places a keyframe every second, which is what keeps seeking in a long
  clip cheap. Preview frames are served downscaled to the monitor and cached per node, and
  playback prefetches a rolling window of them, so the monitor follows the playhead instead of
  waiting on a full-size JPEG per frame. Routes are registered both at
  import time and from the V3 entry point. The
  expand icon at the right end of the toolbar row above the monitor shows the preview full
  screen (browser Fullscreen API, with a fixed overlay fallback); Esc, the ✕ button or a
  double-click returns to the timeline, and the enlarged monitor keeps receiving frames and
  playing back. The timeline zooms to 100–1600% with the `＋` / `－` buttons above it, the `+` /
  `-` keys, or the mouse wheel over the tracks: the point under the cursor stays put, the ruler
  re-ticks, and the lane labels stay pinned while the track canvas scrolls sideways.

### Δ Video Upscale To Dimensions — `triXope Video_Upscale_To_Dimensions`

Memory-aware frame-by-frame upscaling of a video batch to exact dimensions with an upscale
model.

- **Inputs:** `model_name` (upscale_models), `images` (IMAGE), `width` (default `1024`),
  `height` (default `1024`), `crop_behavior` (`stretch` / `crop`),
  `upscale_method` (`nearest-exact`, `bilinear`, `area`, `bicubic`), `device_strategy`
  (`auto`, `load_unload_each_frame`, `keep_loaded`, `cpu_only`).
- **Outputs:** `IMAGE`.
- **Notes:** each frame is tiled through `comfy.utils.tiled_scale` with a tile size that
  depends on the strategy (64 px on CPU, 128 px keep-loaded, 96 px load/unload), then resized
  to the target. `auto` picks keep-loaded when more than half of VRAM is free. Progress and an
  ETA line are printed per frame.

### Δ Multi-Clip Duration Trimmer — `triXope MultiClipTrimmer`

Trims a concatenated multi-shot image/audio sequence to an exact per-clip duration, removing
the excess frames from the **start** of each shot (the usual VAE warm-up frames).

- **Inputs:** `images` (IMAGE), `clip_count` (INT 1–64, default 2), `clip_duration` (FLOAT
  seconds, default 15), `fps` (FLOAT, default 24), optional `audio` (AUDIO).
- **Outputs:** `images` (IMAGE), `audio` (AUDIO).
- **Notes:** the batch is split into `clip_count` equal segments; each segment keeps its last
  `clip_duration × fps` frames, and audio is trimmed by the same amount of time so sync is
  preserved.

### Δ LTXV Trim A/V — `triXope LTXVPostSliceAV`

Drops leading video frames and the matching audio samples from a generated clip.

- **Inputs:** `images` (IMAGE), `drop_first_n_frames` (INT, default 9), `fps` (FLOAT — must
  match the generation fps), optional `audio` (AUDIO).
- **Outputs:** `images` (IMAGE), `audio` (AUDIO).
- **Notes:** audio is cut at `frames / fps` seconds. If more frames than exist are requested,
  the last frame is kept and a warning is printed; audio that would be fully consumed becomes a
  single silent sample.

### Δ Video Preview — `triXope VideoPreview`

Plays a video file from the pack's film output folder inside the node.

- **Inputs:** `video` (STRING; the frontend lists `.mp4`, `.mov`, `.avi`, `.webm` files from
  `input/_film/output/video`).
- **Outputs:** none (output node); returns the file name and folder to the UI, which renders a
  `<video>` widget.

## `triXope/writing` — Writing

Screenwriting, story and prompt authoring. Module: [nodes_writing.py](nodes_writing.py).

### 🎬 Δ H3 Screenwriter — `triXope ScreenplayFormatterNode`

Turns a raw screenplay (typed, loaded, or transcribed from audio) into every prompt artefact
the film nodes need: formatted script, shot list, timestamped story beats, shot lengths, asset
lists and a colour palette.

- **Inputs:** `raw_script` (STRING, multiline), `output_format` (`Standard Plain Text`,
  `HTML Visual Script`, `Fountain Syntax`, `Print-Ready (US Letter)`), `default_aspect_ratio`
  (`16:9`, `2.39:1`, `4:3`, `9:16`, `1:1`), `auto_capitalize_headers` (BOOLEAN),
  `use_ollama_ai` (BOOLEAN), `ollama_model` (STRING, default `llama3`), `ollama_url`,
  `script_override_mode` (`off` / `text` / `audio`), `trigger_words` (STRING),
  `characters_json`, `locations_json`, `props_json` (multiline JSON), `lock_last_generation`
  (BOOLEAN), `stored_generation` (multiline STRING cache); optional `clip` (CLIP),
  `script_override_text` (STRING, force input) and `script_override_audio` (AUDIO).
- **Outputs:** `script`, `shot_list`, `story_beats`, `story_length`, `shot_count`,
  `shot_lengths` (**list** of INT), `character_list`, `prop_list`, `location_list`,
  `color_palette`, `asset_list`.
- **Notes:** shot lengths are generated per shot with a 5-second minimum (H3's 124-frame
  trained floor) and no maximum; only the audio-timeline path keeps its 12-second ceiling.
  With `script_override_audio` the clip owns the total: `story_length` is the shortest whole
  second length whose MiniMax H3 17k+5 frame count still covers every frame of the clip, and the
  last shot is extended to reach it, so the script never ends before its audio.
  The node keeps a JSON generation cache in its UI payload so
  `lock_last_generation` can restore the previous result (a cached result is discarded when
  the output format changes or when switching to Print-Ready). In `audio` mode the connected
  clip sets `story_length` and every shot length, speech becomes dialogue under one `SPEAKER`
  cue, sung lines become a `SINGER` part and repeats without words become instrumental shots.
- Local LLM: the optional `clip` input accepts a ComfyUI text-generation model loaded with
  `CLIP Loader` (Gemma 3/4, Qwen 3.5), and an `ollama_model` value beginning with `Local:`
  downloads that model into `models/text_encoders` on first run and runs it in-process. Both
  bypass Ollama. ComfyUI's native Gemma 4 path covers E2B, E4B, 12B and 31B; only the first
  three have Comfy-Org single-file downloads, so those are the Gemma entries offered here.
  The 26B A4B is a mixture-of-experts model: ComfyUI's Gemma 4 text encoder has no MoE
  support and Comfy-Org publishes no 26B file, so it cannot be loaded as a CLIP text
  encoder here, native or GGUF. Run it through the Ollama backend instead - for example
  the `gemma4:26b-a4b-it-q4_K_M` tag. For Qwen3.5, the
  picker includes 4B bf16 and 9B int8; 9B is the largest text encoder Comfy-Org publishes
  (there is no Comfy-Org 27B repack). Native quantized safetensors load through the same
  path. `.gguf` text encoders are not handled by ComfyUI's built-in `CLIP Loader`: install a
  GGUF-capable CLIP loader that supports the architecture (for example a ComfyUI-GGUF fork
  with Qwen3.5 support) and connect its `CLIP` output to the Screenwriter's optional `clip`
  input. The `Local:` picker intentionally stays on ComfyUI-native single files so it does
  not add a third-party loader dependency.
- Camera continuity: before the shot list is assembled, any shot that repeats its
  neighbour's shot size *and* camera angle is reframed — one rung on the size ladder, or a
  neutral angle change (low/high/over-the-shoulder) when the shot's own action fixes the
  framing — so no two back-to-back shots render the same setup as a jump cut. Each rewrite is
  logged as a warning by the node.
- Identity: each character profile is requested as labelled attributes in face-first order
  (facial structure, hair, wardrobe and distinguishing marks before height, age and weight),
  and that one identity string is repeated verbatim for the character in the shot list and
  the story beats instead of being truncated with the demographics first.
- Continuity: every shot line also carries the `FIXED:` half of the breakdown's continuity
  mapping (posture, position, wardrobe state, prop in hand, light direction), so the
  cross-cut state the breakdown pass worked out actually reaches the render.
- Story beats are written at roughly one beat per 2-3 seconds with 8-20 words each and
  describe intent as well as action (a breath, a weight shift, a hesitation) instead of
  restating a pose every second; assets are named on a shot's first beat and whenever the
  acting subject changes.
  `js/screenplay_editor.js` provides the large editing UI, asset panels and preview.
- Blank-script mode: with `raw_script` empty and `script_override_mode` set to `off`, the Ollama
  model writes a complete original screenplay on its own instead of returning empty outputs —
  a random genre (18), a random story shape (linear, flashback, converging timelines, told
  backwards, framing device, single continuous night), a random cast size and a random length
  of 3-12 pages, always as a finished three-act script with a beginning, middle and end. The
  result is stripped of markdown fences and any preamble or closing chatter, then parsed by the
  normal pipeline, so the shot list, beats and asset prompts follow from it as usual. Requires
  `use_ollama_ai` (otherwise the previous blank-output behaviour is kept), and everything else
  in the node is untouched.

### ⏱️ Δ Shot Length Selector — `triXope ShotLengthSelectorNode`

Picks one shot duration out of a list-valued `shot_lengths` connection.

- **Inputs:** `shot_index` (INT, 1-based, default 1), optional `shot_length` (INT, force
  input, received as a list).
- **Outputs:** `selected_length` (INT).
- **Notes:** clamps to 5–9999, and falls back to the last duration when the index overshoots
  (which is how a loop over shots stays valid); returns 5 when nothing is connected.

### 💾 Δ Shot Length Buffer / Holder — `triXope ShotLengthBufferNode`

Holds the last received shot-length array in a widget so it survives between runs.

- **Inputs:** `stored_array` (STRING JSON, default `[5]`), optional `shot_length` (INT, force
  input).
- **Outputs:** `shot_length` (INT, **list**).
- **Notes:** when a live array is connected it clamps every value to 5–1000008 and mirrors the
  result back into the `stored_array` widget; when disconnected it re-emits the stored array.

### Δ Character Craft — `triXope CharacterCraft`

Builds a complete, prompt-ready character description from structured pickers.

- **Inputs:** `enabled` (BOOLEAN), `Name`, `Age`, `Race`, `Role`, `Eye Color`, `Skin Tone`,
  `Body Type`, `Posture`, `Face Shape`, `Hair Style`, `Hair Color`, `Wardrobe`, `Life Skill`,
  `Life Strength`, `Hobby`, `Musical Interest`, `Bad Habit`, `Faith`, `Pet`,
  `Personality Type`, `Phobia`, `Occupation`, `Income`, `Sexual Orientation`,
  `Marital Status`, `Relation To Protagonist`, `First Appearance`, `Additional Details`.
- **Outputs:** `character_info` (STRING).
- **Notes:** emits `Label: value` lines in a fixed order; disabled, it returns an empty string
  so it can stay in a shared workflow.

### Δ Story Craft — `triXope StorySelect`

Assembles a story prompt from genre/structure/setting/art-style pickers.

- **Inputs:** `Genre`, `Structure`, `Flow`, `Time Period`, `Setting`, `Art Style`,
  `Premise` (STRING, multiline).
- **Outputs:** `story_prompt` (STRING).
- **Notes:** output is a plain `Field: value` block — the art-style and setting lists are
  long, curated, LLM-prompt-oriented vocabularies (hundreds of entries each).

### Δ Prompt Split & Select — `triXope PromptSplitSelect`

Splits pipe-separated prompt text and returns one entry plus the entry count.

- **Inputs:** `text` (STRING, multiline, default `first part | second part | third part`),
  `index` (INT, default 0).
- **Outputs:** `text` (STRING), `prompt_count` (INT).
- **Notes:** empty entries are dropped and the index wraps modulo the entry count, so loops
  can drive it directly; an empty input returns `("", 0)`.

### Δ Batch Ollama Writer — `triXope BatchOllamaWriter`

Feeds a list of shot prompts to a local Ollama model one at a time and writes the answers to a
single file.

- **Inputs:** `prompts_text` (STRING, multiline), `delimiter` (STRING, default `|`),
  `system_instruction` (STRING, multiline), `ollama_url` (default
  `http://127.0.0.1:11434`), `ollama_model` (default `gemma4:26b`), `output_dir` (default
  `./ComfyUI/output/FilmAuteur`), `filename` (default `final_shot_list.txt`), `seed` (INT).
- **Outputs:** `saved_file_path` (STRING), `file_contents` (STRING), `processed_count` (INT).
- **Notes:** calls `/api/chat` with `stream: false` and `keep_alive: -1`, strips markdown
  fences, appends each answer followed by `|`, flushes per shot and reports per-shot timing in
  the console. HTTP or connection errors are written into the file as `[ERROR Shot N]` rather
  than aborting the batch. The whole file is read once at the end for the downstream outputs.

---

# Support modules

These files are not node modules; they provide the helpers the category modules import.

### [`utils.py`](utils.py)

Any-type plumbing shared by the utility nodes: `AnyType` (a `str` subclass that compares
unequal to everything, used as the `*` socket type), `FlexibleOptionalInputType` (dynamic
optional-input dictionary used by Any Switch), `ByPassTypeTuple`, plus small dictionary/JSON
and path helpers (`is_dict_value_falsy`, `get_dict_value`, `set_dict_value`, `dict_has_key`,
`load_json_file`, `save_json_file`, `path_exists`, `file_exists`, `remove_path`, `abspath`).

### [`common.py`](common.py)

MiniMax H3 joint-AV latent maths: the 24 fps / 40 Hz audio-latent frame grid
(`FRAMES_PER_GROUP = 17`, `FRAME_BASE = 5`, `LATENTS_PER_GROUP = 5`, `LATENT_BASE = 2`),
`FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`, snapping helpers (`snap_latents`, `snap_frames`,
`on_grid`), conversions (`latents_to_frames`, `frames_to_latents`, `frames_to_audio_t`,
`frame_at_latent`), NestedTensor packing (`unpack_av`, `pack_av`, `slice_av_tail`) and
`CANVAS_MULTIPLE = 32`.

### [`core.py`](core.py)

Screenplay-to-shot timing used by the H3 nodes: `parse_timestamp` / `format_timestamp`,
`parse_timestamped_beats`, `select_beats_for_window`, `select_beats_for_latent_window`,
`explicit_transition_mode` (cut/crossfade wording detection), `transition_for_overlap`,
`transition_offset_tokens`, `video_frames_for_latents`, `frames_to_audio_latents`,
`requested_frames_to_latents`, `remap_keyframes_for_window` and `plan_windows` (sliding-window
planning).

### [`context_utils.py`](context_utils.py)

Context-dictionary helpers (`new_context`, `merge_new_context`, `get_context_return_tuple`,
`get_orig_context_return_tuple`, `is_context_empty`) used by Any Switch to treat an empty
context object as "no value".

### [`device_manager.py`](device_manager.py)

GPU discovery and routing for the multi-GPU loaders: `get_available_devices`,
`select_optimal_devices` (auto UNet/encoder split, balanced-free-VRAM mode, manual override)
and `get_gpu_status` (VRAM reporting through `psutil` + `comfy.model_management`).

### [`autogpu_pytorch.py`](autogpu_pytorch.py)

The pure-PyTorch multi-GPU engine: `AutoMultiGPU_PyTorch` (an unregistered node class),
`FULL_MODEL_TYPES` (the architecture list exposed by Universal Loader) and
`resolve_clip_type` (maps a model architecture, or the UNet file name, to a ComfyUI CLIP
type). GGUF helpers are re-exported from `gguf_support`.

### [`gguf_support.py`](gguf_support.py)

GGUF support: `is_gguf_available`, `is_gguf_file`, `get_combined_unet_list`,
`get_combined_clip_list`, `resolve_unet_path`, `resolve_clip_path`,
`load_unet_gguf_on_device`, `load_clip_gguf_on_device`.

### [`install_runtime.py`](install_runtime.py)

Installer/downloader for the DLSS 5 neural-rendering runtime used by the H3 Upscaler
(release archive from the `Merserk/dlss5-visual-enhancer` project, extracted into `dlss5/`).
It also prints the third-party distribution terms of the bundled components.

### [`crop_model_patch.py`](crop_model_patch.py)

`ModelPatchCropper` plus `crop_model_cond`: crops ControlNet-style patches to a region and
restores the original patch state afterwards.

### [`usdu_patch.py`](usdu_patch.py), [`usdu_utils.py`](usdu_utils.py), [`usdu_nodes.py`](usdu_nodes.py)

Integration with the original Ultimate SD Upscale script (`repositories/ultimate_sd_upscale`,
downloaded by `__init__.py` on first start): the script loader/patch, tensor↔PIL helpers and
the upstream node definitions (see below — those nodes are not registered by this pack).

### [`gradio.py`](gradio.py)

Empty stub module so the Ultimate SD Upscale script's optional `gradio` import succeeds in a
ComfyUI environment.

---

# Frontend scripts

The pack ships browser-side extensions. They live in the [`js/`](js) folder that
`WEB_DIRECTORY` points at; each entry lists the script and the node it targets, as written in
the script itself.

| Script | Target |
|---|---|
| [`js/anycast.js`](js/anycast.js) | `triXope AnyCast` — live output-socket relabelling |
| [`js/DirectoryFileSelector.js`](js/DirectoryFileSelector.js) | `triXope DirectoryFileSelectorGemini` — directory scan/dropdown |
| [`js/load_text.js`](js/load_text.js) | `triXope LoadText` — text file browser |
| [`js/colorfx_groups.js`](js/colorfx_groups.js) | `triXope ColorFX` — collapsible group bars, serialised open/closed state |
| [`js/filmauteur_groups.js`](js/filmauteur_groups.js) | `triXope FilmAuteur_LTX` — collapsible groups and remembered node sizes |
| [`js/h3_upscale_groups.js`](js/h3_upscale_groups.js) | the H3 Upscaler, detected by its `grp_h3_upscale` widget rather than by name |
| [`js/film_auteur_h3_editor.js`](js/film_auteur_h3_editor.js) | `triXope FilmAuteurH3` — connection/slot layout helpers |
| [`js/screenplay_editor.js`](js/screenplay_editor.js) | `triXope ScreenplayFormatterNode` — screenplay editor, asset panels, preview |
| [`js/video_timeline.js`](js/video_timeline.js) | `triXope VideoTimelineEditor` — timeline UI, waveform, export |
| [`js/triXope_VideoPreview.js`](js/triXope_VideoPreview.js) | video preview widget |
| [`js/GroupMonitor.js`](js/GroupMonitor.js) | `triXope GroupMonitor` — executing-group readout |
| [`js/group_highlighter.js`](js/group_highlighter.js) | `triXope WorkflowGroupHighlighter` — canvas overlay |
| [`js/eta_tracker.js`](js/eta_tracker.js) | `triXope RealtimeLoopTracker` — loop ETA overlay |
| [`js/neon_display.js`](js/neon_display.js) | prompt-counter display node |
| [`js/ltxv_eta_tracker.js`](js/ltxv_eta_tracker.js) | LTXV ETA tracker display |

`WEB_DIRECTORY = "./js"` in [`__init__.py`](__init__.py) serves this folder. Note that the
scripts import ComfyUI's modules with relative paths (`../../scripts/app.js`, and
`../../../scripts/app.js` in `js/triXope_VideoPreview.js`), which assume they sit two (or three)
folders deep inside `js/`, not directly in it.

---

# Unregistered and legacy classes

These classes exist in the tree but are **not** in `NODE_CLASS_MAPPINGS`, so they never appear
in the node menu:

| Class | File | Notes |
|---|---|---|
| `PreviewAudioTabNode` | [`PreviewAudioTab.py`](PreviewAudioTab.py) | AUDIO preview into the ComfyUI audio tab (`CATEGORY = "trIxope"`) |
| `AutoMultiGPU_PyTorch` | [`autogpu_pytorch.py`](autogpu_pytorch.py) | standalone PyTorch loader variant; `UniversalLoaderMultiGPU` supersedes it |
| `ModelPatchCropper` | [`crop_model_patch.py`](crop_model_patch.py) | ControlNet patch cropping helper |
| `UltimateSDUpscale`, `UltimateSDUpscaleNoUpscale`, `UltimateSDUpscaleCustomSample`, `UltimateSDUpscaleMiniMaxH3` | [`usdu_nodes.py`](usdu_nodes.py) | upstream Ultimate SD Upscale nodes (the registered MiniMax H3 upscaler is the one in `nodes_film.py`) |

---

# Notes and known quirks

- **Registration is centralised.** `__init__.py` imports the nine category modules with
  `import *`; each module's `__all__` lists only its registered node classes, so the mapping
  table stays the single source of truth.
- **The standalone H3 files were older and have been dropped.** `H3_FL2VA_Ref2VA.py` and
  `h3_sliding_context.py` contained earlier, superseded copies of the H3 conditioner and
  sampler (different `ref_alpha` options, different sampler/scheduler defaults, no cross-shot
  video reference). They are gone; `nodes_film.py` carries only the newer implementations from
  `Film_Auteur_H3.py`, which is also what `FilmAuteurH3` itself calls.
- **Startup side effects.** Importing `__init__.py` downloads the Ultimate SD Upscale script
  when `repositories/ultimate_sd_upscale` is empty, adjusts `sys.path`/`sys.modules` to avoid
  clashing with ComfyUI's own `utils`/`modules` packages, and registers
  `GET /gemini/get_files`.
- **`nodes_film.py` carries a vendored DLSS 5 import** (`from .dlss5.settings import …`) at
  module scope, and `nodes_latent.py` imports `taehv` behind a `try/except`. Neither `dlss5/`
  nor `taehv.py` is part of this tree, so the DLSS stage requires the runtime installed by
  `install_runtime.py`.
- **Some frontend checks do not match the registered ids.** `js/ltxv_eta_tracker.js` and
  `js/neon_display.js` compare against the bare class names (`LTXV_ETA_Display`,
  `TriXopeCurrentPromptDisplay`), and `js/triXope_VideoPreview.js` compares against
  `triXope_VideoPreview`, while the registered id is `triXope VideoPreview` — those overlays
  only attach if the ids are aligned.
- **Path handling is Windows-oriented in places** (`PathCombinerNode` normalises to `\`,
  `VideoPreview` builds its folder with `_film\\output\\video`).
- **Tests.** `test_screenplay_formatter.py` loads the screenwriting module by file path and
  expects `nodes_writing.py`.
