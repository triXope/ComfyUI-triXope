import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "triXope VideoTimelineEditor";
const FPS_VALUES = [8, 12, 16, 24, 25, 30, 48, 50, 60];
const SNAP_PIXELS = 12;
const IMAGE_TRACKS = ["images", "images2"];
const AUDIO_TRACKS = ["audio", "audio2"];
const TIMELINE_TRACKS = [...IMAGE_TRACKS, ...AUDIO_TRACKS];
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const sourceTrack = (track, source) => source || track;
const sourceLimit = (info, track, fps) => {
  if (IMAGE_TRACKS.includes(track)) return imageTrackInfo(info, track).frames;
  const audio = audioTrackInfo(info, track);
  return Math.ceil(audio.samples * fps / Math.max(1, audio.sampleRate));
};
const widget = (node, name) => node.widgets?.find((item) => item.name === name);
const timelineSelection = (node) => {
  const selection = node._timelineSelection;
  return Array.isArray(selection) ? selection : selection ? [selection] : [];
};
const timelineSourceScale = (info, track, clip, fps) => IMAGE_TRACKS.includes(track)
  ? imageTrackInfo(info, sourceTrack(track, clip.source)).fps / fps
  : 1;
const linkedClipPartners = (timeline, track, clip, linkAV = true) => {
  const otherTracks = IMAGE_TRACKS.includes(track) ? AUDIO_TRACKS : IMAGE_TRACKS;
  return otherTracks.flatMap((otherTrack) => (timeline[otherTrack] || [])
    .map((otherClip, index) => ({ track: otherTrack, clips: timeline[otherTrack], clip: otherClip, index }))
    .filter(({ clip: otherClip }) => {
      const hasExplicitLink = Boolean(clip.av_link_id || otherClip.av_link_id);
      if (hasExplicitLink) return Boolean(clip.av_link_id && clip.av_link_id === otherClip.av_link_id);
      return linkAV && !clip.av_unlinked && !otherClip.av_unlinked
        && otherClip.start === clip.start && otherClip.end === clip.end;
    }));
};

function selectedClipRefs(timeline, selection) {
  return selection.flatMap(({ track, index }) => {
    const clip = timeline[track]?.[index];
    return clip ? [{ track, clips: timeline[track], clip, index }] : [];
  });
}

function clipsAreLinked(left, right, linkAV) {
  if (left.clip.av_link_id && left.clip.av_link_id === right.clip.av_link_id) return true;
  return linkAV && !left.clip.av_unlinked && !right.clip.av_unlinked
    && left.clip.start === right.clip.start && left.clip.end === right.clip.end;
}

function selectionLinkState(timeline, selection, linkAV = true) {
  const selected = selectedClipRefs(timeline, selection);
  const videos = selected.filter(({ track }) => IMAGE_TRACKS.includes(track));
  const audios = selected.filter(({ track }) => AUDIO_TRACKS.includes(track));
  if (videos.length && audios.length) {
    const linkedVideos = videos.filter((video) => audios.some((audio) => clipsAreLinked(video, audio, linkAV))).length;
    const linkedAudios = audios.filter((audio) => videos.some((video) => clipsAreLinked(video, audio, linkAV))).length;
    if (linkedVideos === videos.length && linkedAudios === audios.length) return "linked";
    if (linkedVideos || linkedAudios) return "mixed";
    return "unlinked";
  }
  return selected.some((item) => linkedClipPartners(timeline, item.track, item.clip, linkAV).length > 0)
    ? "linked"
    : "unlinked";
}

function updateSelectionLinks(timeline, selection, linkAV) {
  const selected = selectedClipRefs(timeline, selection);
  const state = selectionLinkState(timeline, selection, linkAV);
  const linked = state === "linked";
  if (linked) {
    const unlink = new Set(selected.map(({ clip }) => clip));
    for (const item of selected) {
      for (const partner of linkedClipPartners(timeline, item.track, item.clip, linkAV)) unlink.add(partner.clip);
    }
    for (const clip of unlink) {
      delete clip.av_link_id;
      clip.av_unlinked = true;
    }
    return "unlinked";
  }

  const videos = selected.filter(({ track }) => IMAGE_TRACKS.includes(track));
  const audios = selected.filter(({ track }) => AUDIO_TRACKS.includes(track));
  if (!videos.length || !audios.length) return "needs-pair";
  const linkPairs = [];
  const unmatchedAudio = new Set(audios);
  for (const video of videos) {
    const audio = audios.find((candidate) => unmatchedAudio.has(candidate)
      && candidate.clip.start === video.clip.start && candidate.clip.end === video.clip.end)
      || audios.find((candidate) => unmatchedAudio.has(candidate));
    if (!audio) break;
    unmatchedAudio.delete(audio);
    linkPairs.push([video, audio]);
  }
  for (const [video, audio] of linkPairs) {
    const linkId = `av-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    video.clip.av_link_id = linkId;
    audio.clip.av_link_id = linkId;
    delete video.clip.av_unlinked;
    delete audio.clip.av_unlinked;
  }
  return linkPairs.length ? "linked" : "needs-pair";
}

function linkedBoundaryLimits(partners, edge, info, fps) {
  return partners.reduce((limits, partner) => {
    const mediaLimit = trackMediaLimit(info, partner.track, fps, partner.clip);
    const sourceScale = timelineSourceScale(info, partner.track, partner.clip, fps);
    const partnerIndex = partner.clips.indexOf(partner.clip);
    if (partnerIndex < 0) return { minDelta: 1, maxDelta: 0 };
    const partnerLimits = boundaryDeltaLimits(partner.clips, partnerIndex, edge, mediaLimit, sourceScale);
    return {
      minDelta: Math.max(limits.minDelta, partnerLimits.minDelta),
      maxDelta: Math.min(limits.maxDelta, partnerLimits.maxDelta),
    };
  }, { minDelta: -Infinity, maxDelta: Infinity });
}

function applyDetectedCuts(timeline, cutsBySource, info, fps, linkAV) {
  const linkPairsBefore = new Map();
  for (const track of IMAGE_TRACKS) {
    for (const clip of timeline[track] || []) {
      const partners = linkedClipPartners(timeline, track, clip, linkAV);
      if (partners.length) linkPairsBefore.set(clip, { track, partners });
    }
  }
  const cutsByFrame = new Map();
  let splitCount = 0;
  for (const track of IMAGE_TRACKS) {
    const originalClips = timeline[track] || [];
    const updated = [];
    for (const clip of originalClips) {
      const source = sourceTrack(track, clip.source);
      const sourceFps = imageTrackInfo(info, source).fps;
      const cuts = [...new Set(cutsBySource[source] || [])].filter(Number.isInteger).sort((a, b) => a - b);
      let pieces = [clip];
      for (const sourceCut of cuts) {
        const nextPieces = [];
        for (const piece of pieces) {
          if (sourceCut <= piece.source_start || sourceCut >= piece.source_end) {
            nextPieces.push(piece);
            continue;
          }
          const timelineCut = piece.start + Math.round((sourceCut - piece.source_start) * fps / sourceFps);
          if (timelineCut <= piece.start || timelineCut >= piece.end) {
            nextPieces.push(piece);
            continue;
          }
          const leftPiece = { ...piece, end: timelineCut, source_end: sourceCut };
          const rightPiece = { ...piece, start: timelineCut, source_start: sourceCut };
          if (piece.av_link_id) {
            leftPiece.av_link_id = `${piece.av_link_id}-a`;
            rightPiece.av_link_id = `${piece.av_link_id}-b`;
          }
          nextPieces.push(leftPiece, rightPiece);
          cutsByFrame.set(timelineCut, true);
          splitCount++;
        }
        pieces = nextPieces;
      }
      updated.push(...pieces);
    }
    timeline[track] = updated.sort((a, b) => a.start - b.start);
  }
  for (const { partners } of linkPairsBefore.values()) {
    for (const partner of partners) {
      const originalStart = partner.clip.start;
      const originalEnd = partner.clip.end;
      const audioPieces = timeline[partner.track] || [];
      for (const frame of [...cutsByFrame.keys()].sort((a, b) => a - b)) {
        if (frame <= originalStart || frame >= originalEnd) continue;
        const index = audioPieces.findIndex((piece) => piece.start >= originalStart && piece.end <= originalEnd && piece.start < frame && frame < piece.end);
        if (index < 0) continue;
        audioPieces.splice(index, 1, ...cutAt([audioPieces[index]], frame));
      }
    }
  }
  for (const [videoClip, { track: videoTrack, partners }] of linkPairsBefore) {
    const videoPieces = (timeline[videoTrack] || []).filter((piece) => piece.start >= videoClip.start && piece.end <= videoClip.end);
    if (videoPieces.length < 2) continue;
    const videoLinkIds = videoPieces.map((piece) => piece.av_link_id || `av-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    videoPieces.forEach((piece, index) => { piece.av_link_id = videoLinkIds[index]; delete piece.av_unlinked; });
    for (const partner of partners) {
      const audioPieces = timeline[partner.track] || [];
      for (const videoPiece of videoPieces) {
        const audioPiece = audioPieces.find((piece) => piece.start === videoPiece.start && piece.end === videoPiece.end);
        if (audioPiece) {
          audioPiece.av_link_id = videoPiece.av_link_id;
          delete audioPiece.av_unlinked;
        }
      }
    }
  }
  return splitCount;
}
const nodeById = (id) => app.graph?._nodes_by_id?.[String(id)] || app.graph?._nodes_by_id?.[Number(id)] || app.graph?.getNodeById?.(Number(id));
const isTimelineNode = (node) => node?.comfyClass === NODE_TYPE || node?.type === NODE_TYPE;

function mediaInfo(node) {
  return node._timelineMediaInfo || { image_frames: 0, image2_frames: 0, audio_frames: 0, audio2_frames: 0, audio_samples: 0, audio2_samples: 0, sample_rate: 0, sample_rate2: 0 };
}

function imageTrackInfo(info, track) {
  return track === "images2"
    ? { frames: Number(info.image2_frames || 0), fps: Number(info.image2_fps || 24), rate: "image2_fps" }
    : { frames: Number(info.image_frames || 0), fps: Number(info.image_fps || 24), rate: "image_fps" };
}

function audioTrackInfo(info, track) {
  return track === "audio2"
    ? { samples: Number(info.audio2_samples || 0), sampleRate: Number(info.sample_rate2 || 0) }
    : { samples: Number(info.audio_samples || 0), sampleRate: Number(info.sample_rate || 0) };
}

function trackMediaLimit(info, track, fps, clip = null) {
  return sourceLimit(info, sourceTrack(track, clip?.source), fps);
}

function emptyTimeline(info, fps) {    const imageFrames = Number(info.image_frames || 0);

  const image2Frames = Number(info.image2_frames || 0);
  const imageFps = Number(info.image_fps || fps);
  const image2Fps = Number(info.image2_fps || fps);    const imageTimelineFrames = Number(info.image_timeline_frames || Math.round(imageFrames * fps / imageFps));
  const image2TimelineFrames = Number(info.image2_timeline_frames || Math.round(image2Frames * fps / image2Fps));
  const audioFrames = Math.ceil(Number(info.audio_samples || 0) * fps / Math.max(1, Number(info.sample_rate || 1)));
  const audio2Frames = Math.ceil(Number(info.audio2_samples || 0) * fps / Math.max(1, Number(info.sample_rate2 || 1)));
  return {
    media: {
      image_frames: imageFrames, image_fps: imageFps, audio_samples: Number(info.audio_samples || 0), sample_rate: Number(info.sample_rate || 0),
      image2_frames: image2Frames, image2_fps: image2Fps, audio2_samples: Number(info.audio2_samples || 0), sample_rate2: Number(info.sample_rate2 || 0), fps,
    },
    images: imageFrames ? [{ start: 0, end: imageTimelineFrames, source_start: 0, source_end: imageFrames, source: "images" }] : [],
    images2: image2Frames ? [{ start: 0, end: image2TimelineFrames, source_start: 0, source_end: image2Frames, source: "images2" }] : [],
    audio: audioFrames ? [{ start: 0, end: audioFrames, source_start: 0, source_end: audioFrames, source: "audio" }] : [],
    audio2: audio2Frames ? [{ start: 0, end: audio2Frames, source_start: 0, source_end: audio2Frames, source: "audio2" }] : [],
  };
}

function storedTimeline(node) {
  try {
    const value = JSON.parse(widget(node, "timeline_json")?.value || "");
    if (Array.isArray(value.images) && Array.isArray(value.audio)) {
      value.images2 ||= [];
      value.audio2 ||= [];
      for (const track of TIMELINE_TRACKS) {
        for (const clip of value[track]) {
          clip.source ||= track;
          if (value.link_av === false) clip.av_unlinked = true;
        }
      }
      value.link_av = true;
      return value;
    }
  } catch { /* An empty or malformed workflow value is initialized from connected media. */ }
  return null;
}

function mediaMatches(timeline, info) {
  return Number(timeline?.media?.image_frames || 0) === Number(info.image_frames || 0)
    && Number(timeline?.media?.image_fps || 0) === Number(info.image_fps || 0)
    && Number(timeline?.media?.audio_samples || 0) === Number(info.audio_samples || 0)
    && Number(timeline?.media?.sample_rate || 0) === Number(info.sample_rate || 0)
    && Number(timeline?.media?.image2_frames || 0) === Number(info.image2_frames || 0)
    && Number(timeline?.media?.image2_fps || 0) === Number(info.image2_fps || 0)
    && Number(timeline?.media?.audio2_samples || 0) === Number(info.audio2_samples || 0)
    && Number(timeline?.media?.sample_rate2 || 0) === Number(info.sample_rate2 || 0);
}

function selectedFps(node) {
  const saved = storedTimeline(node);
  const fps = Number(saved?.media?.fps || 24);
  return FPS_VALUES.includes(fps) ? fps : 24;
}

function getTimeline(node, info = mediaInfo(node)) {
  const saved = storedTimeline(node);
  if (saved && (!(info.image_frames || info.image2_frames || info.audio_samples || info.audio2_samples) || mediaMatches(saved, info))) return saved;
  const timeline = emptyTimeline(info, Number(node._timelineFps || selectedFps(node)));
  timeline.media.fps = Number(node._timelineFps || selectedFps(node));
  return timeline;
}

function saveTimeline(node, timeline, recordUndo = true) {
  const info = mediaInfo(node);
  timeline.media = {
    image_frames: Number(info.image_frames || 0),
    image_width: Number(info.image_width || 0),
    image_height: Number(info.image_height || 0),
    image_fps: Number(info.image_fps || node._timelineFps || 24),
    audio_samples: Number(info.audio_samples || 0),
    sample_rate: Number(info.sample_rate || 0),
    image2_frames: Number(info.image2_frames || 0),
    image_width: Number(info.image_width || 0),
    image_height: Number(info.image_height || 0),
    image2_fps: Number(info.image2_fps || node._timelineFps || 24),
    audio2_samples: Number(info.audio2_samples || 0),
    sample_rate2: Number(info.sample_rate2 || 0),
    fps: Number(node._timelineFps || 24),
  };
  const stateWidget = widget(node, "timeline_json");
  timeline.link_av = true;
  const serialized = JSON.stringify(timeline);
  const current = stateWidget?.value || "";
  if (recordUndo && stateWidget && current && current !== serialized) {
    const undo = node._timelineUndo || [];
    if (undo.at(-1) !== current) undo.push(current);
    if (undo.length > 100) undo.shift();
    node._timelineUndo = undo;
    node._timelineRedo = [];
  }
  if (stateWidget) stateWidget.value = serialized;
  node._timelineLastSaved = serialized;
  node.graph?.change?.();
}

function undoTimeline(node) {
  const undo = node._timelineUndo || [];
  if (!undo.length) return;
  const current = widget(node, "timeline_json")?.value || "";
  const previous = undo.pop();
  node._timelineLinkAV = JSON.parse(previous).link_av !== false;
  const redo = node._timelineRedo || [];
  if (current && redo.at(-1) !== current) redo.push(current);
  node._timelineRedo = redo;
  const stateWidget = widget(node, "timeline_json");
  if (!stateWidget) return;
  stateWidget.value = previous;
  node._timelineLastSaved = previous;
  node._timelineSelection = null;
  node.graph?.change?.();
  render(node);
}

function redoTimeline(node) {
  const redo = node._timelineRedo || [];
  if (!redo.length) return;
  const current = widget(node, "timeline_json")?.value || "";
  const next = redo.pop();
  node._timelineLinkAV = JSON.parse(next).link_av !== false;
  const undo = node._timelineUndo || [];
  if (current && undo.at(-1) !== current) undo.push(current);
  node._timelineUndo = undo;
  node._timelineRedo = redo;
  const stateWidget = widget(node, "timeline_json");
  if (stateWidget) stateWidget.value = next;
  node._timelineLastSaved = next;
  node._timelineSelection = null;
  node.graph?.change?.();
  render(node);
}

function timelineEnd(timeline) {
  let end = 1;
  for (const track of TIMELINE_TRACKS) {
    for (const clip of timeline[track] || []) {
      if (clip.end > end) end = clip.end;
    }
  }
  return end;
}

function timecode(frame, fps) {
  const total = frame / fps;
  const minutes = Math.floor(total / 60);
  const seconds = Math.floor(total % 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}:${String(frame % fps).padStart(2, "0")}`;
}

function mediaUrl(path, node, query = {}) {
  const params = new URLSearchParams({ node_id: String(node.id), ...query });
  return `/video_timeline_editor/${path}?${params}`;
}

const PREVIEW_CACHE_MAX = 480;
const PREVIEW_AHEAD = 48;
const PREVIEW_WORKERS = 3;
const PREVIEW_JPEG_QUALITY = 78;

function previewWidth(monitor) {
  const width = monitor?.closest?.(".timeline-monitor")?.clientWidth || monitor?.clientWidth || 0;
  return Math.max(320, Math.min(1280, Math.round(width || 640)));
}

function previewQuery(frameMap) {
  const query = {};
  const primary = frameMap.images;
  const overlay = frameMap.images2;
  if (primary) query[primary.source === "images2" ? "frame2" : "frame"] = primary.index;
  if (overlay) query[overlay.source === "images2" ? "frame2_v2" : "frame_v2"] = overlay.index;
  return query;
}

// Prefetch requests carry p=1 so the server decodes them on its second reader, but their bytes
// are the visible frame's bytes: store them under the URL the monitor will ask for, otherwise
// the JS cache fills with entries no visible request can use and evicts the ones it could.
function cachedPreviewUrl(url) {
  const parsed = new URL(url, "http://timeline.local");
  parsed.searchParams.delete("p");
  return `${parsed.pathname}${parsed.search}`;
}

function previewCache(node) {
  return node._timelinePreviewCache || (node._timelinePreviewCache = new Map());
}

function clearPreviewCache(node) {
  node._timelinePrefetchQueue = [];
  node._timelinePrefetchSeen?.clear();
  const cache = node._timelinePreviewCache;
  if (!cache) return;
  for (const objectUrl of cache.values()) {
    if (objectUrl !== node._timelineFrameObjectUrl) URL.revokeObjectURL(objectUrl);
  }
  cache.clear();
}

function rememberPreview(node, url, blob) {
  const cache = previewCache(node);
  const existing = cache.get(url);
  if (existing) return existing;
  const objectUrl = URL.createObjectURL(blob);
  cache.set(url, objectUrl);
  while (cache.size > PREVIEW_CACHE_MAX) {
    const [oldUrl, oldObjectUrl] = cache.entries().next().value;
    cache.delete(oldUrl);
    if (oldObjectUrl !== node._timelineFrameObjectUrl) URL.revokeObjectURL(oldObjectUrl);
  }
  return objectUrl;
}

async function runPrefetchWorker(node) {
  const queue = node._timelinePrefetchQueue;
  const seen = node._timelinePrefetchSeen;
  if (!queue || !seen) return;
  if ((node._timelinePrefetching || 0) >= PREVIEW_WORKERS) return;
  node._timelinePrefetching = (node._timelinePrefetching || 0) + 1;
  try {
    while (queue.length && node._timelinePlaying) {
      const url = queue.shift();
      const cachedUrl = cachedPreviewUrl(url);
      if (previewCache(node).has(cachedUrl)) { seen.delete(url); continue; }
      try {
        let response = await api.fetchApi(url);
        if (response.status === 404) response = await fetch(url, { credentials: "same-origin" });
        if (response.ok) rememberPreview(node, cachedUrl, await response.blob());
      } catch { /* prefetch is best effort; the visible frame path reports errors */ }
      finally { seen.delete(url); }
    }
  } finally {
    node._timelinePrefetching = Math.max(0, (node._timelinePrefetching || 1) - 1);
  }
}

function prefetchPreview(node, timeline, frame, width, duration) {
  if (!node._timelinePlaying) return;
  const queue = node._timelinePrefetchQueue || (node._timelinePrefetchQueue = []);
  if (queue.length > PREVIEW_AHEAD * 2) return;
  const seen = node._timelinePrefetchSeen || (node._timelinePrefetchSeen = new Set());
  const cache = previewCache(node);
  const info = mediaInfo(node);
  const fps = Number(node._timelineFps || 24);
  const ahead = Math.min(PREVIEW_AHEAD, Math.max(0, (duration ?? timelineEnd(timeline)) - frame));
  for (let offset = 1; offset <= ahead; offset++) {
    const frameMap = timelineImagesAt(timeline, info, frame + offset, fps);
    if (!Object.keys(frameMap).length) continue;
    const query = previewQuery(frameMap);
    query.w = width;
    query.q = PREVIEW_JPEG_QUALITY;
    // Marks the request as prefetch: the server decodes it on a second reader, so it cannot
    // pull the decoder the visible frames are reading away from the playhead.
    query.p = 1;
    const url = mediaUrl("frame", node, query);
    if (cache.has(cachedPreviewUrl(url)) || seen.has(url)) continue;
    seen.add(url);
    queue.push(url);
  }
  for (let worker = 0; worker < PREVIEW_WORKERS; worker++) void runPrefetchWorker(node);
}

function requestMonitorFrame(node, url, sourceFrame) {
  const monitor = node._timelineRoot?.querySelector(".timeline-monitor img");
  const cached = previewCache(node).get(url);
  if (cached && monitor) {
    node._timelinePendingFrame = null;
    node._timelineFramePendingUrl = null;
    node._timelineFrameFailedUrl = null;
    monitor.dataset.frameUrl = url;
    monitor.src = cached;
    return;
  }
  const requestId = (node._timelineFrameRequestId || 0) + 1;
  node._timelineFrameRequestId = requestId;
  node._timelinePendingFrame = { url, sourceFrame, requestId };
  node._timelineFramePendingUrl = url;
  if (!node._timelineFrameLoading) void loadNextMonitorFrame(node);
}

async function loadNextMonitorFrame(node) {
  const request = node._timelinePendingFrame;
  if (!request || node._timelineFrameLoading) return;
  node._timelinePendingFrame = null;
  node._timelineFrameLoading = true;
  const controller = new AbortController();
  node._timelineFrameAbort = controller;
  const monitor = node._timelineRoot?.querySelector(".timeline-monitor img");
  try {
    let response = await api.fetchApi(request.url, { signal: controller.signal });
    if (response.status === 404) {
      const apiDetail = (await response.clone().text()).slice(0, 200);
      const legacyResponse = await fetch(request.url, { signal: controller.signal, credentials: "same-origin" });
      if (legacyResponse.ok) response = legacyResponse;
      else {
        const legacyDetail = (await legacyResponse.text()).slice(0, 200);
        throw new Error(`HTTP ${response.status} from ${response.url || "API route"}${apiDetail ? ` (${apiDetail})` : ""}; HTTP ${legacyResponse.status} from ${legacyResponse.url || "legacy route"}${legacyDetail ? ` (${legacyDetail})` : ""}`);
      }
    }
    if (!response.ok) {
      const detail = (await response.text()).slice(0, 300);
      throw new Error(`HTTP ${response.status}${detail ? `: ${detail}` : ""}`);
    }
    const blob = await response.blob();
    if (request.requestId === node._timelineFrameRequestId && monitor) {
      const objectUrl = rememberPreview(node, request.url, blob);
      node._timelineFrameObjectUrl = objectUrl;
      monitor.onerror = () => setStatus(node, `Frame ${request.sourceFrame} was returned but the browser could not decode it.`, "error");
      monitor.src = objectUrl;
      node._timelineFramePendingUrl = null;
      node._timelineFrameFailedUrl = null;
      if (node._timelineStatus?.text?.startsWith("Frame ")) setStatus(node, "", "info");
    }
  } catch (error) {
    if (error.name !== "AbortError" && request.requestId === node._timelineFrameRequestId) {
      node._timelineFrameFailedUrl = request.url;
      node._timelineFramePendingUrl = null;
      setStatus(node, `Frame ${request.sourceFrame} preview failed: ${error.message || error}`, "error");
    }
  } finally {
    if (node._timelineFramePendingUrl === request.url && request.requestId !== node._timelineFrameRequestId) node._timelineFramePendingUrl = null;
    node._timelineFrameLoading = false;
    if (node._timelineFrameAbort === controller) node._timelineFrameAbort = null;
    if (node._timelinePendingFrame) void loadNextMonitorFrame(node);
  }
}


function updateQueueStatus(node, result) {
  if (result === false) {
    const errors = app.extensionManager?.lastNodeErrors || app.lastNodeErrors;
    const nodeError = errors && Object.values(errors).flatMap((item) => item.errors || [item])[0];
    const message = nodeError?.message || nodeError?.details || app.extensionManager?.lastExecutionError?.exception_message || app.lastExecutionError?.exception_message;
    throw new Error(message ? `ComfyUI rejected the timeline prompt: ${message}` : "ComfyUI could not queue the timeline prompt. Check the ComfyUI execution error details.");
  }
  return result;
}

async function queueNode(node, action) {
  const actionWidget = widget(node, "ui_action");
  node._timelineQueuedAction = action;
  if (actionWidget) actionWidget.value = "inspect";
  const stateWidget = widget(node, "timeline_json");
  const timeline = getTimeline(node);

  if (stateWidget) stateWidget.value = JSON.stringify(timeline);
  node.graph?.change?.();
  try {
    const result = await app.queuePrompt(0, 1, [String(node.id)]);
    return updateQueueStatus(node, result);
  } finally {
    node._timelineQueuedAction = null;
    if (actionWidget && actionWidget.value !== "inspect") {
      actionWidget.value = "inspect";
      node.graph?.change?.();
    }
  }
}

function button(label, title, callback) {
  const element = document.createElement("button");
  element.type = "button";
  element.textContent = label;
  element.title = title;
  Object.assign(element.style, { border: "1px solid #454b55", background: "#252a31", color: "#e7e9ed", borderRadius: "4px", padding: "5px 8px", cursor: "pointer", fontSize: "12px" });
  element.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); callback(event); });
  return element;
}

// Outward diagonal arrows, matching the usual "expand / maximize" glyph.
const ENLARGE_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>';

function iconButton(icon, label, title, callback) {
  const element = button("", title, callback);
  element.innerHTML = icon;
  element.setAttribute("aria-label", label);
  Object.assign(element.style, { display: "inline-flex", alignItems: "center", justifyContent: "center", width: "28px", height: "28px", padding: "0", color: "#dfe3e9", lineHeight: "0" });
  return element;
}

function setStatus(node, text, kind = "info") {
  const colors = { info: "#adb4bf", success: "#73d99c", error: "#ff8383" };
  node._timelineStatus = { text, color: colors[kind] || colors.info };
  const status = node._timelineRoot?.querySelector(".timeline-status");
  if (status) { status.textContent = text; status.style.color = node._timelineStatus.color; }
}

function cutAt(clips, frame, sourceFramesPerTimelineFrame = () => 1) {
  return clips.flatMap((clip) => {
    if (frame <= clip.start || frame >= clip.end) return [clip];
    const sourceCut = clip.source_start + Math.round((frame - clip.start) * sourceFramesPerTimelineFrame(clip));
    if (sourceCut <= clip.source_start || sourceCut >= clip.source_end) return [clip];
    return [{ ...clip, end: frame, source_end: sourceCut }, { ...clip, start: frame, source_start: sourceCut }];
  });
}


function adjacentBoundary(clips, index, edge) {
  const clip = clips[index];
  if (edge === "start" && index > 0 && clips[index - 1].end === clip.start) return clips[index - 1];
  if (edge === "end" && index + 1 < clips.length && clips[index + 1].start === clip.end) return clips[index + 1];
  return null;
}

function boundaryDeltaLimits(clips, index, edge, mediaLimit, sourceFramesPerTimelineFrame = 1) {
  const clip = clips[index];
  const previous = clips[index - 1];
  const next = clips[index + 1];
  return {
    minDelta: edge === "start"
      ? Math.max(-clip.start, (previous?.end ?? 0) - clip.start, Math.ceil(-clip.source_start / sourceFramesPerTimelineFrame))
      : Math.max(clip.start + 1 - clip.end, Math.ceil((clip.source_start + 1 - clip.source_end) / sourceFramesPerTimelineFrame)),
    maxDelta: edge === "start"
      ? Math.min(clip.end - clip.start - 1, Math.floor((clip.source_end - clip.source_start - 1) / sourceFramesPerTimelineFrame))
      : Math.min(next ? next.start - clip.end : Infinity, Math.floor((mediaLimit - clip.source_end) / sourceFramesPerTimelineFrame)),
  };
}

function moveBoundary(clips, index, edge, requestedDelta, mediaLimit, sourceFramesPerTimelineFrame = 1) {
  const clip = clips[index];
  const { minDelta, maxDelta } = boundaryDeltaLimits(clips, index, edge, mediaLimit, sourceFramesPerTimelineFrame);
  const timelineDelta = clamp(requestedDelta, minDelta, maxDelta);
  const sourceDelta = Math.round(timelineDelta * sourceFramesPerTimelineFrame);
  if (edge === "start") {
    clip.start += timelineDelta;
    clip.source_start += sourceDelta;
  } else {
    clip.end += timelineDelta;
    clip.source_end += sourceDelta;
  }
  return timelineDelta;
}

function clipStartLimits(clips, clip, originalClips) {
  const original = originalClips.get(clip);
  if (!original) return { minStart: 0, maxStart: 0 };
  const duration = original.end - original.start;
  const others = [...originalClips.entries()]
    .filter(([other]) => other !== clip)
    .map(([, value]) => value)
    .sort((a, b) => a.start - b.start);
  const previous = [...others].reverse().find((other) => other.end <= original.start);
  const next = others.find((other) => other.start >= original.end);
  const minStart = Math.max(0, previous?.end ?? 0);
  const maxStart = next ? Math.max(minStart, next.start - duration) : Infinity;
  return { minStart, maxStart };
}

function moveClip(clips, clip, requestedStart, originalClips) {
  const original = originalClips.get(clip);
  if (!original) return 0;
  const duration = original.end - original.start;
  const { minStart, maxStart } = clipStartLimits(clips, clip, originalClips);
  const safeStart = clamp(Math.round(requestedStart), minStart, maxStart);
  clip.start = safeStart;
  clip.end = safeStart + duration;
  clips.sort((a, b) => a.start - b.start);
  return safeStart;
}


async function loadAudioPreview(node, timeline, fps) {
  const signature = `${fps}:${timelineEnd(timeline)}:${JSON.stringify([timeline.audio, timeline.audio2])}`;
  if (node._timelineAudio?.signature === signature) return Promise.resolve();
  if (node._timelineAudioLoading) return node._timelineAudioLoadPromise;
  const generation = (node._timelineAudioGeneration || 0) + 1;
  node._timelineAudioSignature = signature;
  node._timelineAudioGeneration = generation;
  node._timelineAudioLoading = true;
  const loadPromise = (async () => {
    try {
      if (!node._timelineAudioRaw) node._timelineAudioRaw = {};
      for (const track of AUDIO_TRACKS) {
        const trackInfo = audioTrackInfo(mediaInfo(node), track);
        if (!trackInfo.samples || node._timelineAudioRaw[track]) continue;
        const response = await api.fetchApi(`/video_timeline_editor/audio?node_id=${encodeURIComponent(node.id)}&track=${track}`);
        if (!response.ok) {
          const detail = (await response.text()).slice(0, 300);
          throw new Error(`${track} preview route returned HTTP ${response.status}${detail ? `: ${detail}` : ""}`);
        }
        const pcm = new Int16Array(await response.arrayBuffer());
        const channels = Number(response.headers.get("X-Channels") || 1);
        const sampleRate = Number(response.headers.get("X-Sample-Rate") || 0);
        const samples = Array.from({ length: channels }, () => new Float32Array(Math.floor(pcm.length / channels)));
        for (let index = 0; index < samples[0].length; index++) {
          for (let channel = 0; channel < channels; channel++) samples[channel][index] = pcm[index * channels + channel] / 32768;
        }
        node._timelineAudioRaw[track] = { channels, sampleRate, samples };
      }
      if (generation !== node._timelineAudioGeneration || !node._timelineAudioContext) return;
          const endFrame = timelineEnd(timeline);
      const sampleRate = Math.max(...AUDIO_TRACKS.map((track) => node._timelineAudioRaw[track]?.sampleRate || 0));
      const activeTracks = AUDIO_TRACKS.filter((track) => node._timelineAudioRaw[track]);
      if (!activeTracks.length || !sampleRate) return;
      const channels = Math.max(...activeTracks.map((track) => node._timelineAudioRaw[track].channels));
      const rendered = node._timelineAudioContext.createBuffer(channels, Math.max(1, Math.ceil(endFrame * sampleRate / fps)), sampleRate);
      for (const lane of AUDIO_TRACKS) {
        for (const clip of timeline[lane] || []) {
          const sourceTrackName = sourceTrack(lane, clip.source);
          const raw = node._timelineAudioRaw[sourceTrackName];
          if (!raw) continue;
          const ratio = raw.sampleRate / sampleRate;
          const sourceStart = Math.floor(clip.source_start * raw.sampleRate / fps);
          const sourceEnd = Math.min(raw.samples[0].length, Math.ceil(clip.source_end * raw.sampleRate / fps));
          const destinationStart = Math.floor(clip.start * sampleRate / fps);
          const sourceCount = Math.max(0, sourceEnd - sourceStart);
          const count = Math.max(0, Math.min(Math.ceil(sourceCount * sampleRate / raw.sampleRate), rendered.length - destinationStart));
          if (!count) continue;
          for (let channel = 0; channel < channels; channel++) {
            const sourceChannel = Math.min(channel, raw.channels - 1);
            const sourceData = raw.samples[sourceChannel];
            const target = rendered.getChannelData(channel);
            for (let index = 0; index < count; index++) {
              const sourcePosition = sourceStart + index * ratio;
              const first = Math.min(sourceData.length - 1, Math.floor(sourcePosition));
              const second = Math.min(sourceData.length - 1, first + 1);
              const fraction = sourcePosition - first;
              target[destinationStart + index] += sourceData[first] * (1 - fraction) + sourceData[second] * fraction;
            }
          }
        }
      }
      for (let channel = 0; channel < channels; channel++) {
        const data = rendered.getChannelData(channel);
        for (let index = 0; index < data.length; index++) data[index] = clamp(data[index], -1, 1);
      }
      if (generation !== node._timelineAudioGeneration) return;
      node._timelineAudio = { context: node._timelineAudioContext, buffer: rendered, fps, endFrame, signature };
      if (node._timelinePlaying) {
        node._timelinePlaybackStartFrame = node._timelinePlayhead || 0;
        node._timelinePlaybackStartTime = performance.now();
        startAudio(node);
      }
    } catch (error) {
      setStatus(node, `Audio preview unavailable: ${error.message || error}`, "error");
      throw error;
    } finally {
      if (generation === node._timelineAudioGeneration) node._timelineAudioLoading = false;
    }
  })();
  node._timelineAudioLoadPromise = loadPromise;
  loadPromise.then(() => {
    if (node._timelineAudioLoadPromise === loadPromise) node._timelineAudioLoadPromise = null;
  }, () => {
    if (node._timelineAudioLoadPromise === loadPromise) node._timelineAudioLoadPromise = null;
  });
  return loadPromise;
}

function startAudio(node) {
  const audio = node._timelineAudio;
  if (!audio || !node._timelinePlaying || node._timelineAudioSource) return;
  const offset = (node._timelinePlayhead || 0) / audio.fps;
  if (offset >= audio.buffer.duration) return;
  const source = audio.context.createBufferSource();
  source.buffer = audio.buffer;
  source.connect(audio.context.destination);
  node._timelineAudioStartedAt = audio.context.currentTime;
  node._timelineAudioStartFrame = node._timelinePlayhead || 0;
  node._timelineAudioSource = source;
  source.onended = () => { if (node._timelineAudioSource === source) node._timelineAudioSource = null; };
  source.start(0, offset);
}

function stopAudio(node) {
  try { node._timelineAudioSource?.stop(); } catch { /* Already stopped. */ }
  node._timelineAudioSource = null;
  node._timelineAudioStartedAt = null;
}

function stopPlayback(node) {
  if (node._timelineAnimation) cancelAnimationFrame(node._timelineAnimation);
  node._timelineAnimation = null;
  node._timelinePlaying = false;
  node._timelinePlaybackStartFrame = node._timelinePlayhead || 0;
  node._timelinePlaybackStartTime = performance.now();
  node._timelinePrefetchQueue = [];
  stopAudio(node);
}

function updatePlaybackDisplay(node, frame = node._timelinePlayhead || 0) {
  const timeline = node._timelineRenderedTimeline;
  const duration = node._timelineRenderedDuration || 1;
  const fps = node._timelineFps || 24;
  node._timelinePlayhead = clamp(frame, 0, duration);
  const time = node._timelineRoot?.querySelector(".timeline-time");
  if (time) time.textContent = `${timecode(node._timelinePlayhead, fps)} / ${timecode(duration, fps)}`;
  for (const line of node._timelineRoot?.querySelectorAll(".timeline-playhead") || []) {
    line.style.left = `${node._timelinePlayhead / duration * 100}%`;
  }
  if (timeline) updateMonitor(node, timeline, duration);
}

function startPlayback(node, direction = 1) {
  stopAudio(node);
  const timeline = getTimeline(node);
  const fps = Number(node._timelineFps || 24);
  const duration = timelineEnd(timeline);
  node._timelinePlaybackDirection = direction < 0 ? -1 : 1;
  if (node._timelinePlaybackDirection > 0 && node._timelinePlayhead >= duration) node._timelinePlayhead = 0;
  if (node._timelinePlaybackDirection < 0 && node._timelinePlayhead <= 0) node._timelinePlayhead = duration;
  node._timelinePlaying = true;
  updatePlaybackDisplay(node, node._timelinePlayhead || 0);
  if (node._timelineAnimation) cancelAnimationFrame(node._timelineAnimation);
  node._timelineAnimation = null;
  node._timelinePlaybackStartFrame = node._timelinePlayhead || 0;
  node._timelinePlaybackStartTime = performance.now();
  const Context = window.AudioContext || window.webkitAudioContext;
  if (node._timelinePlaybackDirection > 0 && (mediaInfo(node).audio_samples || mediaInfo(node).audio2_samples)) {
    if (!Context) {
      setStatus(node, "Audio preview is not supported by this browser.", "error");
    } else {
      node._timelineAudioContext ||= new Context();
      node._timelineAudioContext.resume().catch((error) => setStatus(node, `Audio preview unavailable: ${error.message || error}`, "error"));
      const preview = loadAudioPreview(node, timeline, fps);
      if (preview && typeof preview.catch === "function") preview.catch(() => {});
      if (node._timelineAudio?.fps === fps) startAudio(node);
    }
  }
  const tick = (now) => {
    if (!node._timelinePlaying) return;
    let frame;
    const audio = node._timelineAudio;
    if (node._timelineAudioSource && audio && node._timelineAudioStartedAt !== null) {
      frame = node._timelineAudioStartFrame + Math.floor((audio.context.currentTime - node._timelineAudioStartedAt) * fps);
    } else {
      frame = node._timelinePlaybackStartFrame + node._timelinePlaybackDirection * Math.floor((now - node._timelinePlaybackStartTime) * fps / 1000);
    }
    if (frame >= duration || frame <= 0 && node._timelinePlaybackDirection < 0) {
      updatePlaybackDisplay(node, node._timelinePlaybackDirection < 0 ? 0 : duration);
      stopPlayback(node);
      const play = node._timelineRoot?.querySelector(".timeline-play-button");
      if (play) play.textContent = "▶ Play";
      return;
    }
    updatePlaybackDisplay(node, frame);
    node._timelineAnimation = requestAnimationFrame(tick);
  };
  const play = node._timelineRoot?.querySelector(".timeline-play-button");
  if (play) play.textContent = "❚❚ Pause";
  node._timelineAnimation = requestAnimationFrame(tick);
}

function pausePlayback(node) {
  updatePlaybackDisplay(node, node._timelinePlayhead || 0);
  stopPlayback(node);
  const play = node._timelineRoot?.querySelector(".timeline-play-button");
  if (play) play.textContent = "▶ Play";
}

function snapCandidates(timeline, playhead, excludeClip = null) {
  const candidates = new Set([playhead]);
  for (const track of TIMELINE_TRACKS) {
    for (const clip of timeline[track] || []) {
      if (clip === excludeClip) continue;
      candidates.add(clip.start);
      candidates.add(clip.end);
    }
  }
  return [...candidates].filter(Number.isFinite);
}

function snapFrame(value, candidates, threshold) {
  let closest = value;
  let distance = threshold + 1;
  for (const boundary of candidates) {
    const candidateDistance = Math.abs(boundary - value);
    if (candidateDistance < distance) {
      closest = boundary;
      distance = candidateDistance;
    }
  }
  return distance <= threshold ? closest : value;
}

function snapClipStart(start, duration, candidates, threshold, minStart = 0, maxStart = Infinity) {
  let closest = start;
  let closestDistance = threshold + 1;
  for (const boundary of candidates) {
    for (const alignedStart of [boundary, boundary - duration]) {
      if (alignedStart < minStart || alignedStart > maxStart) continue;
      const distance = Math.abs(alignedStart - start);
      if (distance < closestDistance) {
        closest = alignedStart;
        closestDistance = distance;
      }
    }
  }
  return closestDistance <= threshold ? closest : start;
}

function snapThreshold(track, duration) {
  return Math.max(1, Math.ceil(duration * SNAP_PIXELS / Math.max(1, track.clientWidth)));
}

function laneAtPointerY(laneRects, clientY, fallback) {
  const laneTop = Math.min(...laneRects.map(({ rect }) => rect.top));
  const laneBottom = Math.max(...laneRects.map(({ rect }) => rect.bottom));
  if (clientY < laneTop || clientY > laneBottom) return fallback;
  return laneRects.reduce((nearest, lane) => {
    const distance = Math.abs(clientY - (lane.rect.top + lane.rect.bottom) / 2);
    return !nearest || distance < nearest.distance ? { name: lane.name, distance } : nearest;
  }, null)?.name || fallback;
}

function snapBoundary(value, candidates, threshold, minValue = -Infinity, maxValue = Infinity) {
  const legalCandidates = candidates.filter((candidate) => candidate >= minValue && candidate <= maxValue);
  return snapFrame(value, legalCandidates, threshold);
}

// Playback resolves one clip per lane per frame, and the 48-frame prefetch resolves 48 more.
// Copying and reversing the whole lane for each lookup made the editor slower the more clips
// the timeline held; the cursor remembers the clip the last frame landed in, so a step forward
// is one comparison and a jump costs one pass over the lane, with no allocation.
const CLIP_CURSORS = new WeakMap();

function clipCursor(timeline, track) {
  let cursors = CLIP_CURSORS.get(timeline);
  if (!cursors) CLIP_CURSORS.set(timeline, (cursors = {}));
  return cursors[track] || (cursors[track] = { index: 0 });
}

function clipAt(clips, frame, cursor) {
  if (!clips.length) return null;
  const index = Math.min(cursor.index, clips.length - 1);
  const current = clips[index];
  if (current && current.start <= frame && frame < current.end) return current;
  if (current && frame >= current.end) {
    for (let at = index + 1; at < clips.length; at++) {
      const clip = clips[at];
      if (clip.start > frame) break;
      if (frame < clip.end) { cursor.index = at; return clip; }
    }
  } else {
    for (let at = index - 1; at >= 0; at--) {
      const clip = clips[at];
      if (clip.start <= frame && frame < clip.end) { cursor.index = at; return clip; }
    }
  }
  for (let at = 0; at < clips.length; at++) {
    const clip = clips[at];
    if (clip.start <= frame && frame < clip.end) { cursor.index = at; return clip; }
  }
  return null;
}

function timelineImagesAt(timeline, info, frame, fps) {
  const frames = {};
  for (const track of IMAGE_TRACKS) {
    const clip = clipAt(timeline[track] || [], frame, clipCursor(timeline, track));
    if (!clip) continue;
    const source = sourceTrack(track, clip.source);
    const sourceInfo = imageTrackInfo(info, source);
    if (!sourceInfo.frames) continue;
    frames[track] = {
      source,
      index: clamp(clip.source_start + Math.floor((frame - clip.start) * sourceInfo.fps / fps), clip.source_start, clip.source_end - 1),
    };
  }
  return frames;
}

function updateMonitor(node, timeline, duration) {
  const monitor = node._timelineRoot?.querySelector(".timeline-monitor img");
  if (!monitor) return;
  const frame = clamp(node._timelinePlayhead || 0, 0, duration);
  const currentMediaInfo = mediaInfo(node);
  if (timeline.images?.some((clip) => clip.source === "images2") || timeline.images2?.some((clip) => clip.source === "images2")) {
    currentMediaInfo.image2_frames = Number(currentMediaInfo.image2_frames || timeline.media?.image2_frames || 0);
  }
  if (timeline.images?.some((clip) => clip.source !== "images2") || timeline.images2?.some((clip) => clip.source !== "images2")) {
    currentMediaInfo.image_frames = Number(currentMediaInfo.image_frames || timeline.media?.image_frames || 0);
  }
  const frameMap = timelineImagesAt(timeline, currentMediaInfo, frame, Number(node._timelineFps || 24));
  if (!Object.keys(frameMap).length) {
    node._timelineFrameRequestId = (node._timelineFrameRequestId || 0) + 1;
    node._timelinePendingFrame = null;
    node._timelineFramePendingUrl = null;
    node._timelineFrameAbort?.abort();
    monitor.removeAttribute("src");
    monitor.dataset.frameUrl = "";
    node._timelineFrameFailedUrl = null;
    return;
  }
  const primaryFrame = frameMap.images;
  const overlayFrame = frameMap.images2;
  const query = previewQuery(frameMap);
  query.w = previewWidth(monitor);
  if (node._timelinePreviewWidth && node._timelinePreviewWidth !== query.w) clearPreviewCache(node);
  node._timelinePreviewWidth = query.w;
  query.q = PREVIEW_JPEG_QUALITY;
  const url = mediaUrl("frame", node, query);
  const displayFrame = primaryFrame?.index ?? overlayFrame?.index;
  if (monitor.dataset.frameUrl !== url || (!monitor.getAttribute("src") && node._timelineFrameFailedUrl !== url && !node._timelineFramePendingUrl)) {
    monitor.dataset.frameUrl = url;
    requestMonitorFrame(node, url, displayFrame);
  }
  prefetchPreview(node, timeline, frame, query.w, duration);
}

function resizeTimelineNode(node) {
  node.setDirtyCanvas?.(true, true);
}

// ---------------------------------------------------------------- monitor full screen
// The monitor element itself is enlarged rather than a copy of it: it stays inside the node
// root, so the frame requests, the cached object URL and updateMonitor() keep addressing the
// same <img> while the view is full screen.
const MONITOR_FULLSCREEN_STYLE = {
  width: "100%", height: "100%", maxHeight: "100vh", aspectRatio: "auto", minHeight: "0",
  margin: "0", border: "none", borderRadius: "0", background: "#05070a",
  position: "fixed", inset: "0", top: "0", right: "0", bottom: "0", left: "0", zIndex: "2147483000",
};

function monitorExitButton(onClick) {
  const exit = document.createElement("button");
  exit.type = "button";
  exit.className = "timeline-monitor-exit";
  exit.textContent = "✕";
  exit.title = "Exit full screen (Esc)";
  Object.assign(exit.style, { position: "absolute", top: "12px", right: "12px", zIndex: "1", width: "34px", height: "34px", padding: "0", color: "#f2f5f8", background: "rgba(10, 14, 18, 0.72)", border: "1px solid rgba(255, 255, 255, 0.28)", borderRadius: "6px", font: "15px/1 sans-serif", cursor: "pointer" });
  exit.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); onClick(); });
  return exit;
}

async function exitMonitorFullscreen(node) {
  const monitor = node._timelineFullscreenMonitor;
  if (!monitor) return;
  const wasNative = Boolean(node._timelineFullscreenNative);
  node._timelineFullscreenMonitor = null;
  node._timelineFullscreenNative = false;
  if (node._timelineFullscreenKeyHandler) {
    document.removeEventListener("keydown", node._timelineFullscreenKeyHandler, true);
    node._timelineFullscreenKeyHandler = null;
  }
  if (node._timelineFullscreenChangeHandler) {
    document.removeEventListener("fullscreenchange", node._timelineFullscreenChangeHandler);
    node._timelineFullscreenChangeHandler = null;
  }
  monitor.querySelector(".timeline-monitor-exit")?.remove();
  monitor.style.cssText = node._timelineFullscreenStyle || "";
  node._timelineFullscreenStyle = "";
  if (wasNative && document.fullscreenElement === monitor && document.exitFullscreen) {
    try { await document.exitFullscreen(); } catch { /* the element is already detached */ }
  }
  node.setDirtyCanvas?.(true, true);
}

async function enterMonitorFullscreen(node, monitor) {
  if (node._timelineFullscreenMonitor) await exitMonitorFullscreen(node);
  node._timelineFullscreenMonitor = monitor;
  node._timelineFullscreenStyle = monitor.style.cssText;
  Object.assign(monitor.style, MONITOR_FULLSCREEN_STYLE);
  monitor.append(monitorExitButton(() => { void exitMonitorFullscreen(node); }));
  // Esc is the way out for the CSS overlay; the browser handles it natively in real full screen.
  node._timelineFullscreenKeyHandler = (event) => {
    if (event.key !== "Escape") return;
    event.preventDefault();
    event.stopPropagation();
    void exitMonitorFullscreen(node);
  };
  document.addEventListener("keydown", node._timelineFullscreenKeyHandler, true);
  node._timelineFullscreenChangeHandler = () => {
    if (node._timelineFullscreenNative && !document.fullscreenElement) void exitMonitorFullscreen(node);
  };
  document.addEventListener("fullscreenchange", node._timelineFullscreenChangeHandler);
  if (typeof monitor.requestFullscreen === "function") {
    try {
      await monitor.requestFullscreen({ navigationUI: "hide" });
      node._timelineFullscreenNative = document.fullscreenElement === monitor;
    } catch {
      node._timelineFullscreenNative = false;
    }
  }
  if (!node._timelineFullscreenNative) {
    setStatus(node, "Full-screen monitor. Esc, \u2715 or a double-click returns to the timeline.", "info");
  }
}

async function toggleMonitorFullscreen(node, monitor) {
  if (node._timelineFullscreenMonitor === monitor) await exitMonitorFullscreen(node);
  else await enterMonitorFullscreen(node, monitor);
}

const TIMELINE_ZOOM_MIN = 1;
const TIMELINE_ZOOM_MAX = 16;
const TIMELINE_ZOOM_STEP = 1.25;
const clampTimelineZoom = (value) => Math.min(TIMELINE_ZOOM_MAX, Math.max(TIMELINE_ZOOM_MIN, Number(value) || TIMELINE_ZOOM_MIN));

// Zoom is a width multiplier on the track canvas: clips, ruler and playhead are all percentage
// positioned, so widening the canvas scales them without touching any edit math. The frame under
// the cursor stays put by restoring scrollLeft around the same ratio after the rebuild.
function zoomTimeline(node, factor, anchorX = null) {
  const current = clampTimelineZoom(node._timelineZoom);
  const next = clampTimelineZoom(current * factor);
  if (Math.abs(next - current) < 0.001) return;
  const scroller = node._timelineScroller;
  if (scroller) {
    const rect = scroller.getBoundingClientRect();
    const canvasRect = scroller.firstElementChild?.getBoundingClientRect();
    const point = anchorX == null ? rect.left + rect.width / 2 : anchorX;
    const ratio = canvasRect ? clamp((point - canvasRect.left) / Math.max(1, canvasRect.width), 0, 1) : 0.5;
    node._timelineZoomAnchor = { ratio, offset: point - rect.left };
  }
  node._timelineZoom = next;
  render(node);
}

function render(node) {
  const root = node._timelineRoot;
  if (!root) return;
  // The root is rebuilt below, which would drop an enlarged monitor out of the layout.
  if (node._timelineFullscreenMonitor) void exitMonitorFullscreen(node);
  const info = mediaInfo(node);
  const fps = Number(node._timelineFps || 24);
  const zoom = clampTimelineZoom(node._timelineZoom);
  node._timelineZoom = zoom;
  const zoomAnchor = node._timelineZoomAnchor;
  node._timelineZoomAnchor = null;
  const timeline = getTimeline(node, info);
  const laneLabels = { images: "v1", images2: "v2", audio: "a1", audio2: "a2" };
  for (const track of TIMELINE_TRACKS) timeline[track] ||= [];

  const duration = timelineEnd(timeline);
  node._timelineRenderedTimeline = timeline;
  node._timelineRenderedDuration = duration;
  root.replaceChildren();
  const renderInfo = { ...info, image_width: info.image_width || node._timelineMediaInfo?.image_width, image_height: info.image_height || node._timelineMediaInfo?.image_height };

  const toolbar = document.createElement("div");
  Object.assign(toolbar.style, { display: "flex", alignItems: "center", gap: "6px", flexWrap: "wrap", marginBottom: "7px" });
  const fpsSelect = document.createElement("select");
  fpsSelect.setAttribute("aria-label", "FPS selector");
  fpsSelect.style.cssText = "background:#252a31;color:#eee;border:1px solid #454b55;border-radius:4px;padding:4px";
  for (const value of FPS_VALUES) {
    const option = document.createElement("option"); option.value = value; option.textContent = `${value} FPS`; option.selected = value === fps; fpsSelect.append(option);
  }
  fpsSelect.addEventListener("change", () => {
    node._timelineWasPlayingOnFpsChange = Boolean(node._timelinePlaying);
    const nextFps = Number(fpsSelect.value);
    const frameScale = nextFps / fps;
    for (const track of TIMELINE_TRACKS) {
      timeline[track] = (timeline[track] || []).map((clip) => ({
        ...clip,
        start: Math.round(clip.start * frameScale),
        end: Math.round(clip.end * frameScale),
        ...(IMAGE_TRACKS.includes(track) ? {} : {
          source_start: Math.round(clip.source_start * frameScale),
          source_end: Math.round(clip.source_end * frameScale),
        }),
      }));
    }
    node._timelineFps = nextFps;
    node._timelinePlayhead = Math.round((node._timelinePlayhead || 0) * frameScale);
    if (node._timelinePlaying) pausePlayback(node);
    node._timelineAudio = null;
    node._timelineAudioRaw = null;
    node._timelineAudioGeneration = (node._timelineAudioGeneration || 0) + 1;
    node._timelineAudioLoading = false;
    node._timelineAudioSignature = null;
    node._timelineAudioLoadPromise = null;
    saveTimeline(node, timeline);
    render(node);
    if (node._timelineWasPlayingOnFpsChange) {
      node._timelineWasPlayingOnFpsChange = false;
      startPlayback(node);
    }
  });
  const playButton = button(node._timelinePlaying ? "❚❚ Pause" : "▶ Play", "Play or pause the timeline in real time", () => {
    if (node._timelinePlaying) pausePlayback(node);
    else startPlayback(node);
  });
  playButton.className = "timeline-play-button";
  const timeDisplay = document.createElement("span");
  timeDisplay.className = "timeline-time";
  timeDisplay.style.cssText = "font:12px monospace;color:#d5d9e0;min-width:115px";
  const currentFrame = clamp(node._timelinePlayhead || 0, 0, duration);
  timeDisplay.textContent = `${timecode(currentFrame, fps)} / ${timecode(duration, fps)}`;
  const zoomOutButton = button("－", "Zoom the timeline out (+ / - keys, or the mouse wheel over the timeline)", () => zoomTimeline(node, 1 / TIMELINE_ZOOM_STEP));
  zoomOutButton.className = "timeline-zoom-out";
  const zoomInButton = button("＋", "Zoom the timeline in (+ / - keys, or the mouse wheel over the timeline)", () => zoomTimeline(node, TIMELINE_ZOOM_STEP));
  zoomInButton.className = "timeline-zoom-in";
  const zoomLabel = document.createElement("span");
  zoomLabel.className = "timeline-zoom";
  zoomLabel.title = "Timeline zoom: + / - keys, the mouse wheel, or the buttons";
  zoomLabel.style.cssText = "font:11px monospace;color:#c5cad2;min-width:38px;text-align:center";
  zoomLabel.textContent = `${Math.round(zoom * 100)}%`;
  for (const [element, disabled] of [[zoomOutButton, zoom <= TIMELINE_ZOOM_MIN], [zoomInButton, zoom >= TIMELINE_ZOOM_MAX]]) {
    element.disabled = disabled;
    element.style.opacity = disabled ? "0.45" : "1";
  }
  const selectedForLinking = selectedClipRefs(timeline, timelineSelection(node));
  const selectionState = selectionLinkState(timeline, timelineSelection(node), node._timelineLinkAV !== false);
  const canLinkSelection = (selectedForLinking.some(({ track }) => IMAGE_TRACKS.includes(track))
    && selectedForLinking.some(({ track }) => AUDIO_TRACKS.includes(track)))
    || selectionState === "linked" || selectionState === "mixed";
  const linkButton = button(
    !selectedForLinking.length ? "Select clips to link" : selectionState === "linked" ? "🔗 A/V linked · click to unlink" : selectionState === "mixed" ? "◐ Mixed · click to link selection" : "🔓 A/V unlinked · click to link",
    "Select one video clip and one or more audio clips with Shift-click, then link or unlink those clips",
    () => {
      const result = updateSelectionLinks(timeline, timelineSelection(node), node._timelineLinkAV !== false);
      if (result === "needs-pair") {
        setStatus(node, "Shift-click at least one video clip and one audio clip to link them.", "info");
        return;
      }
      saveTimeline(node, timeline);
      render(node);
    },
  );
  linkButton.disabled = !canLinkSelection;
  linkButton.style.opacity = linkButton.disabled ? "0.5" : "1";
  const cut = button("✂️ Cut", "Split image and audio clips at the playhead", () => {
    for (const track of TIMELINE_TRACKS) {
      timeline[track] = cutAt(timeline[track] || [], node._timelinePlayhead || 0, (clip) => IMAGE_TRACKS.includes(track) ? imageTrackInfo(info, sourceTrack(track, clip.source)).fps / fps : 1);
    }
    saveTimeline(node, timeline); render(node);
  });
  const findCutsButton = button("Find cuts", "Detect hard cuts in connected image sequences and split matching clips", async () => {
    findCutsButton.disabled = true;
    setStatus(node, "Finding hard cuts…", "info");
    try {
      const response = await api.fetchApi(`/video_timeline_editor/cuts?node_id=${encodeURIComponent(node.id)}`);
      if (!response.ok) throw new Error((await response.text()).slice(0, 300) || `HTTP ${response.status}`);
      const result = await response.json();
      const added = applyDetectedCuts(timeline, result.cuts || {}, info, fps, node._timelineLinkAV !== false);
      if (!added) {
        setStatus(node, `No hard cuts found${Number.isFinite(result.threshold) ? ` (threshold ${result.threshold})` : ""}.`, "info");
        return;
      }
      saveTimeline(node, timeline);
      render(node);
      setStatus(node, `Added ${added} hard-cut${added === 1 ? "" : "s"}${node._timelineLinkAV !== false ? "; linked audio was split at the same points" : ""}.`, "success");
    } catch (error) {
      setStatus(node, error.message || String(error), "error");
    } finally {
      findCutsButton.disabled = false;
    }
  });
  const exportButton = button("Export", "Render the current timeline to MP4 without re-running the workflow", () => exportTimeline(node, timeline));
  Object.assign(exportButton.style, { background: "#176b46", borderColor: "#23875b", fontWeight: "600" });
  const undoButton = button("↶ Undo", "Undo the last timeline edit", () => undoTimeline(node));
  undoButton.disabled = !(node._timelineUndo || []).length;
  undoButton.style.opacity = undoButton.disabled ? "0.5" : "1";
  const redoButton = button("↷ Redo", "Redo the last undone edit", () => redoTimeline(node));
  redoButton.disabled = !(node._timelineRedo || []).length;
  redoButton.style.opacity = redoButton.disabled ? "0.5" : "1";
  toolbar.append(fpsSelect, playButton, timeDisplay, zoomOutButton, zoomInButton, zoomLabel, linkButton, cut, findCutsButton, undoButton, redoButton, exportButton);

  const status = document.createElement("div");
  status.className = "timeline-status";
  status.style.cssText = "font-size:11px;color:#adb4bf;min-height:15px;margin-bottom:5px";
  if (node._timelineStatus) { status.textContent = node._timelineStatus.text; status.style.color = node._timelineStatus.color; }
  const monitor = document.createElement("div");
  monitor.className = "timeline-monitor";
  const aspectRatio = Number(renderInfo.image_width) > 0 && Number(renderInfo.image_height) > 0
    ? Number(renderInfo.image_width) / Number(renderInfo.image_height)
    : Number(node._timelineAspectRatio) > 0 ? Number(node._timelineAspectRatio) : 16 / 9;
  Object.assign(monitor.style, { width: "100%", height: "auto", aspectRatio: String(aspectRatio), minHeight: "180px", marginBottom: "8px", background: "#090b0e", border: "1px solid #343942", borderRadius: "4px", display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden", flex: "none" });
  const monitorImage = document.createElement("img");
  monitorImage.alt = "";
  monitorImage.setAttribute("aria-hidden", "true");
  monitorImage.addEventListener("error", () => monitorImage.removeAttribute("src"));
  Object.assign(monitorImage.style, { width: "100%", height: "100%", maxWidth: "100%", maxHeight: "100%", objectFit: "contain", color: "transparent", fontSize: "0" });
  monitor.append(monitorImage);
  monitor.addEventListener("dblclick", (event) => {
    if (event.target?.closest?.(".timeline-monitor-exit")) return;  // the close button handles itself
    event.preventDefault();
    event.stopPropagation();
    void toggleMonitorFullscreen(node, monitor);
  });
  const enlargeButton = iconButton(ENLARGE_ICON, "Enlarge monitor", "Show the monitor full screen; Esc, ✕ or a double-click returns to the timeline", () => { void toggleMonitorFullscreen(node, monitor); });
  enlargeButton.className = "timeline-enlarge-button";
  enlargeButton.style.marginLeft = "auto";
  toolbar.append(enlargeButton);
  if (!info.image_frames && !info.image2_frames) {
    const hint = document.createElement("div");
    hint.textContent = "Connect an image sequence to Images 1 or Images 2, then click Load media to populate the tracks.";
    hint.style.cssText = "color:#aeb5c0;font-size:12px;padding:12px";
    const loadButton = button("Load media", "Inspect the connected image and audio sequences", async () => {
      setStatus(node, "Loading connected media…", "info");
      try { await queueNode(node, "inspect"); }
      catch (error) { setStatus(node, error.message || String(error), "error"); }
    });
    root.append(toolbar, status, monitor, hint, loadButton);
    resizeTimelineNode(node);
    return;
  }

  const ruler = document.createElement("div");
  Object.assign(ruler.style, { height: "25px", position: "relative", borderBottom: "1px solid #535a66", marginLeft: "46px", cursor: "crosshair" });
  const tickStep = Math.max(1, Math.ceil(duration / (8 * zoom)));
  for (let frame = 0; frame <= duration; frame += tickStep) {
    const tick = document.createElement("span"); tick.textContent = timecode(frame, fps);
    const isFinalTick = frame === duration;
    Object.assign(tick.style, { position: "absolute", left: `${frame / duration * 100}%`, bottom: "1px", color: "#aeb4be", font: "10px monospace", borderLeft: "1px solid #646c79", paddingLeft: "3px", transform: isFinalTick ? "translateX(calc(-100% - 3px))" : "none", borderLeftStyle: isFinalTick ? "none" : "solid", borderRight: isFinalTick ? "1px solid #646c79" : "none", paddingLeft: isFinalTick ? "0" : "3px", paddingRight: isFinalTick ? "3px" : "0" });
    ruler.append(tick);
  }

  const viewport = document.createElement("div");
  Object.assign(viewport.style, { position: "relative", width: "100%", overflowX: zoom > 1 ? "auto" : "hidden", overflowY: "hidden" });
  const canvas = document.createElement("div");
  Object.assign(canvas.style, { position: "relative", width: `${zoom * 100}%`, minWidth: "100%" });
  canvas.append(ruler);
  viewport.append(canvas);
  node._timelineScroller = viewport;
  viewport.addEventListener("wheel", (event) => {
    if (!event.deltaY) return;
    event.preventDefault();
    zoomTimeline(node, Math.exp(-event.deltaY * 0.0015), event.clientX);
  }, { passive: false });

  const lanes = document.createElement("div");
  Object.assign(lanes.style, { display: "flex", flexDirection: "column", gap: "7px", marginTop: "5px" });
  const tracks = {};
  const laneOrder = ["images2", "images", "audio", "audio2"];
  for (const name of laneOrder) {
    const labelText = laneLabels[name];
    const lane = document.createElement("div");
    Object.assign(lane.style, { display: "grid", gridTemplateColumns: "40px 1fr", alignItems: "center", gap: "6px", minHeight: "36px" });
    const label = document.createElement("span"); label.textContent = labelText; label.style.cssText = "position:sticky;left:0;z-index:6;background:#171a1f;padding:2px 0;font-size:11px;color:#d4d8df";
    const track = document.createElement("div"); track.className = `timeline-track ${name}`;
    Object.assign(track.style, { height: "31px", position: "relative", background: "#171a1f", borderRadius: "4px", outline: "1px solid #343942", touchAction: "none" });
    lane.append(label, track); lanes.append(lane); tracks[name] = track;
  }
  const frameFromX = (element, clientX) => {
    const rect = element.getBoundingClientRect();
    return clamp(Math.round((clientX - rect.left) / Math.max(1, rect.width) * duration), 0, duration);
  };
  const updatePlayhead = (frame) => {
    node._timelinePlayhead = clamp(frame, 0, duration);
    timeDisplay.textContent = `${timecode(node._timelinePlayhead, fps)} / ${timecode(duration, fps)}`;
    for (const track of Object.values(tracks)) {
      let line = track.querySelector(":scope > .timeline-playhead");
      if (!line) {
        line = document.createElement("div"); line.className = "timeline-playhead";
        Object.assign(line.style, { position: "absolute", top: "-7px", bottom: "-6px", width: "2px", background: "#f0c04a", zIndex: "8", cursor: "ew-resize" });
        const knob = document.createElement("div"); Object.assign(knob.style, { position: "absolute", left: "-5px", top: "0", width: "10px", height: "10px", borderRadius: "50%", background: "#f0c04a" });
        line.append(knob); track.append(line);
        line.addEventListener("pointerdown", (event) => scrub(track, event));
      }
      line.style.left = `${node._timelinePlayhead / duration * 100}%`;
    }
    updateMonitor(node, timeline, duration);
  };
  const scrub = (area, event) => {
    event.preventDefault(); event.stopPropagation(); updatePlayhead(frameFromX(area, event.clientX));
    const move = (moveEvent) => updatePlayhead(frameFromX(area, moveEvent.clientX));
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up, { once: true });
  };
  for (const area of [ruler, ...Object.values(tracks)]) {
    area.addEventListener("pointerdown", (event) => { if (event.target === area) scrub(area, event); });
  }

  for (const trackName of TIMELINE_TRACKS) {
    const track = tracks[trackName]; const clips = timeline[trackName] || [];
    const mediaLimit = trackMediaLimit(info, trackName, fps);
    clips.forEach((clip, index) => {
      const block = document.createElement("div"); block.dataset.clipIndex = String(index);
      const selected = timelineSelection(node).some((item) => item.track === trackName && item.index === index);
      const isImage = IMAGE_TRACKS.includes(trackName);
      Object.assign(block.style, { position: "absolute", top: "3px", height: "25px", left: `${clip.start / duration * 100}%`, width: `${Math.max((clip.end - clip.start) / duration * 100, 0.6)}%`, background: isImage ? (trackName === "images2" ? "#365c92" : "#285c7d") : (trackName === "audio2" ? "#76573d" : "#674587"), border: selected ? "2px solid #f0c04a" : "1px solid #7190a9", boxSizing: "border-box", borderRadius: "4px", cursor: "grab", zIndex: selected ? "5" : "2" });
      const caption = document.createElement("span"); caption.style.cssText = "display:block;overflow:hidden;white-space:nowrap;text-align:center;font:10px/21px monospace;color:white;pointer-events:none"; caption.textContent = `${index + 1} · ${clip.end - clip.start}f`; block.append(caption);
      for (const edge of ["start", "end"]) {
        const handle = document.createElement("div");
        Object.assign(handle.style, { position: "absolute", top: "-1px", bottom: "-1px", width: "9px", [edge === "start" ? "left" : "right"]: "0", cursor: "ew-resize", background: "rgba(255,255,255,.28)", borderRadius: "4px" });
        handle.title = `Drag ${trackName} ${edge} edge to trim or extend`;
        handle.addEventListener("pointerdown", (event) => {
          event.preventDefault(); event.stopPropagation(); node._timelineSelection = [{ track: trackName, index, edge }];
          const originX = event.clientX;
          const originEdge = edge === "start" ? clip.start : clip.end;
          const originalClips = new Map(clips.map((item) => [item, { ...item }]));
          const linkedPartners = linkedClipPartners(timeline, trackName, clip, node._timelineLinkAV !== false);
          const partnerSnapshots = linkedPartners.map((partner) => ({ ...partner, original: { ...partner.clip } }));
          const snapRange = snapThreshold(track, duration);
                const sourceInfo = IMAGE_TRACKS.includes(trackName) ? imageTrackInfo(info, sourceTrack(trackName, clip.source)) : null;
          handle.setPointerCapture(event.pointerId);
          const onMove = (moveEvent) => {
            if (!handle.hasPointerCapture(moveEvent.pointerId)) return;
            for (const [target, snapshot] of originalClips) Object.assign(target, snapshot);
            for (const partner of partnerSnapshots) Object.assign(partner.clip, partner.original);
            const delta = Math.round((moveEvent.clientX - originX) / Math.max(1, track.clientWidth) * duration);
            const rawEdge = originEdge + delta;
            const sourceScale = sourceInfo ? sourceInfo.fps / fps : 1;
            const mediaLimit = trackMediaLimit(info, trackName, fps, clip);
            const ownLimits = boundaryDeltaLimits(clips, index, edge, mediaLimit, sourceScale);
            const partnerLimits = linkedBoundaryLimits(partnerSnapshots, edge, info, fps);
            const minDelta = Math.max(ownLimits.minDelta, partnerLimits.minDelta);
            const maxDelta = Math.min(ownLimits.maxDelta, partnerLimits.maxDelta);
            const snapSet = snapCandidates(timeline, node._timelinePlayhead || 0, clip);
            const candidate = snapBoundary(rawEdge, snapSet, snapRange, originEdge + minDelta, originEdge + maxDelta);
            const legalDelta = clamp(candidate - originEdge, minDelta, maxDelta);
            const appliedDelta = moveBoundary(clips, index, edge, legalDelta, mediaLimit, sourceScale);            for (const partner of partnerSnapshots) {
              const partnerScale = timelineSourceScale(info, partner.track, partner.clip, fps);
              const partnerIndex = partner.clips.indexOf(partner.clip);
              if (partnerIndex >= 0) moveBoundary(partner.clips, partnerIndex, edge, appliedDelta, trackMediaLimit(info, partner.track, fps, partner.clip), partnerScale);
              const element = partnerElements.get(partner.clip);
              if (element) {
                element.style.left = `${partner.clip.start / duration * 100}%`;
                element.style.width = `${Math.max((partner.clip.end - partner.clip.start) / duration * 100, 0.6)}%`;
              }
            }
            for (const [clipIndex, current] of clips.entries()) {
              const element = track.querySelector(`[data-clip-index="${clipIndex}"]`); if (!element) continue;
              element.style.left = `${current.start / duration * 100}%`; element.style.width = `${Math.max((current.end - current.start) / duration * 100, 0.6)}%`;
              const text = element.querySelector("span"); if (text) text.textContent = `${clipIndex + 1} · ${current.end - current.start}f`;
            }
          };
          const onUp = (upEvent) => { if (handle.hasPointerCapture(upEvent.pointerId)) handle.releasePointerCapture(upEvent.pointerId); handle.removeEventListener("pointermove", onMove); handle.removeEventListener("pointerup", onUp); saveTimeline(node, timeline); render(node); };
          handle.addEventListener("pointermove", onMove); handle.addEventListener("pointerup", onUp);
        });
        block.append(handle);
      }
      block.addEventListener("pointerdown", (event) => {
        if (event.target !== block && event.target !== caption) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.shiftKey) {
          const selectedClips = timelineSelection(node);
          const selectedIndex = selectedClips.findIndex((item) => item.track === trackName && item.index === index);
          if (selectedIndex >= 0) selectedClips.splice(selectedIndex, 1);
          else selectedClips.push({ track: trackName, index, edge: null });
          node._timelineSelection = selectedClips;
          render(node);
          return;
        }
        node._timelineSelection = [{ track: trackName, index, edge: null }];
        const originX = event.clientX;
        const laneNames = IMAGE_TRACKS.includes(trackName) ? IMAGE_TRACKS : AUDIO_TRACKS;
        const laneSnapshots = new Map(laneNames.map((name) => [name, (timeline[name] || []).map((item) => [item, { ...item }])]));
        const original = { ...clip };
        const linkedPartners = linkedClipPartners(timeline, trackName, clip, node._timelineLinkAV !== false);
        const partnerSnapshots = linkedPartners.map((partner) => ({ ...partner, original: { ...partner.clip } }));
        const partnerElements = new Map(partnerSnapshots.map((partner) => [partner.clip, tracks[partner.track].querySelector(`[data-clip-index="${partner.index}"]`)]));
        const elements = new Map(laneNames.flatMap((name) => [...tracks[name].querySelectorAll("[data-clip-index]")].map((element) => [timeline[name][Number(element.dataset.clipIndex)], element])));
        const onMove = (moveEvent) => {
          if (moveEvent.pointerId !== event.pointerId) return;
          moveEvent.preventDefault();
          moveEvent.stopPropagation();
          for (const name of laneNames) {
            const targetClips = timeline[name];
            targetClips.splice(0, targetClips.length);
            for (const [item, snapshot] of laneSnapshots.get(name)) {
              Object.assign(item, snapshot);
              targetClips.push(item);
            }
          }
          for (const partner of partnerSnapshots) Object.assign(partner.clip, partner.original);
          const laneRects = laneNames.map((name) => ({ name, rect: tracks[name].getBoundingClientRect() }));
          const destination = laneAtPointerY(laneRects, moveEvent.clientY, trackName);
          if (destination !== trackName) {
            const originLane = timeline[trackName];
            originLane.splice(originLane.indexOf(clip), 1);
            timeline[destination].push(clip);
            timeline[destination].sort((a, b) => a.start - b.start);
          }
          const targetClips = timeline[destination];
          const targetSnapshots = new Map(targetClips.filter((item) => item !== clip).map((item) => [item, { ...item }]));
          targetSnapshots.set(clip, original);
          const rect = tracks[destination].getBoundingClientRect();
          const snapRange = snapThreshold(tracks[destination], duration);
          const snapSet = snapCandidates(timeline, node._timelinePlayhead || 0, clip);
          const requestedStart = original.start + Math.round((moveEvent.clientX - originX) / Math.max(1, rect.width) * duration);
          const limits = clipStartLimits(targetClips, clip, targetSnapshots);
          let minDelta = limits.minStart - original.start;
          let maxDelta = limits.maxStart - original.start;
          for (const partner of partnerSnapshots) {
            const partnerClips = partner.clips.filter((item) => item !== partner.clip);
            const previous = [...partnerClips].filter((item) => item.end <= partner.original.start).sort((a, b) => b.start - a.start)[0];
            const next = partnerClips.filter((item) => item.start >= partner.original.end).sort((a, b) => a.start - b.start)[0];
            minDelta = Math.max(minDelta, Math.max(0, previous?.end ?? 0) - partner.original.start);
            maxDelta = Math.min(maxDelta, (next ? Math.max(previous?.end ?? 0, next.start - (partner.original.end - partner.original.start)) : Infinity) - partner.original.start);
          }
          const minStart = original.start + minDelta;
          const maxStart = original.start + maxDelta;
          const constrainedStart = clamp(requestedStart, minStart, maxStart);
          const snappedStart = snapClipStart(constrainedStart, original.end - original.start, snapSet, snapRange, minStart, maxStart);
          const actualStart = moveClip(targetClips, clip, snappedStart, targetSnapshots);
          const appliedDelta = actualStart - original.start;
          for (const partner of partnerSnapshots) {
            partner.clip.start = partner.original.start + appliedDelta;
            partner.clip.end = partner.original.end + appliedDelta;
            partner.clips.sort((a, b) => a.start - b.start);
            const element = partnerElements.get(partner.clip);
            if (element) {
              element.style.left = `${partner.clip.start / duration * 100}%`;
              element.style.width = `${Math.max((partner.clip.end - partner.clip.start) / duration * 100, 0.6)}%`;
            }
          }
          node._timelineSelection = [{ track: destination, index: targetClips.indexOf(clip), edge: null }];
          for (const name of laneNames) {
            timeline[name].forEach((item, itemIndex) => {
              const element = elements.get(item);
              if (!element) return;
              tracks[name].append(element);
              element.dataset.clipIndex = String(itemIndex);
              element.style.left = `${item.start / duration * 100}%`;
              element.style.width = `${Math.max((item.end - item.start) / duration * 100, 0.6)}%`;
            });
          }
          block.dataset.lastStart = String(actualStart);
        };
        const onUp = (upEvent) => {
          if (upEvent.pointerId !== event.pointerId) return;
          window.removeEventListener("pointermove", onMove);
          window.removeEventListener("pointerup", onUp);
          window.removeEventListener("pointercancel", onUp);
          saveTimeline(node, timeline); render(node);
        };
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        window.addEventListener("pointercancel", onUp);
      });
      track.append(block);
    });
  }

  const controls = document.createElement("div"); Object.assign(controls.style, { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "5px", marginTop: "7px", minHeight: "24px" });
  const selected = timelineSelection(node).filter((item) => timeline[item.track]?.[item.index]);
  if (selected.length) {
    const title = document.createElement("span");
    title.textContent = selected.length === 1
      ? `${laneLabels[selected[0].track]} segment ${selected[0].index + 1} · ${selected[0].edge || "end"} edge`
      : `${selected.length} clips selected`;
    title.style.cssText = "font-size:11px;color:#c5cad2;margin-right:3px"; controls.append(title);
    if (selected.length === 1) {
      const selectedClip = timeline[selected[0].track][selected[0].index];
      for (const step of [-10, -1, 1, 10]) {
        const edge = selected[0].edge || "end"; const direction = step < 0 ? "←" : "→";
        controls.append(button(`${direction}${Math.abs(step)}f`, `Move selected ${edge} edge ${Math.abs(step)} frames`, () => {
          const currentClip = timeline[selected[0].track][selected[0].index];
            const sourceScale = timelineSourceScale(info, selected[0].track, currentClip, fps);
          const limit = trackMediaLimit(info, selected[0].track, fps, currentClip);
          const clips = timeline[selected[0].track];
          const partners = linkedClipPartners(timeline, selected[0].track, currentClip, node._timelineLinkAV !== false);
          const limits = boundaryDeltaLimits(clips, selected[0].index, edge, limit, sourceScale);
          const partnerLimits = linkedBoundaryLimits(partners, edge, info, fps);
          const delta = clamp(step, Math.max(limits.minDelta, partnerLimits.minDelta), Math.min(limits.maxDelta, partnerLimits.maxDelta));
          const applied = moveBoundary(clips, selected[0].index, edge, delta, limit, sourceScale);
          for (const partner of partners) {
            const partnerIndex = partner.clips.indexOf(partner.clip);
            if (partnerIndex >= 0) moveBoundary(partner.clips, partnerIndex, edge, applied, trackMediaLimit(info, partner.track, fps, partner.clip), timelineSourceScale(info, partner.track, partner.clip, fps));
          }
          saveTimeline(node, timeline); render(node);
        }));
      }
    }
    controls.append(button(selected.length === 1 ? "Delete segment" : "Delete selected clips", "Remove selected clips from their tracks", () => {
      for (const track of TIMELINE_TRACKS) {
        const indexes = selected.filter((item) => item.track === track).map((item) => item.index).sort((a, b) => b - a);
        for (const index of indexes) timeline[track].splice(index, 1);
      }
      node._timelineSelection = null; saveTimeline(node, timeline); render(node);
    }));
  } else {
    const hint = document.createElement("span"); hint.textContent = "Drag the playhead to scrub. Cut splits tracks; drag a clip body to reposition and an edge to trim. Undo/redo is available in the toolbar."; hint.style.cssText = "font-size:11px;color:#aeb5c0"; controls.append(hint);
  }
  canvas.append(lanes);
  root.append(toolbar, status, monitor, viewport, controls);
  if (zoomAnchor && zoom > 1) {
    viewport.scrollLeft = Math.max(0, zoomAnchor.ratio * canvas.getBoundingClientRect().width - zoomAnchor.offset);
  }
  resizeTimelineNode(node);
  updatePlayhead(node._timelinePlayhead || 0);
  if (node._timelinePlaying && node._timelineAnimation === null) {
    const play = root.querySelector(".timeline-play-button");
    if (play) play.textContent = "❚❚ Pause";
    startPlayback(node, node._timelinePlaybackDirection || 1);
  } else if (node._timelinePlaying) {
    const play = root.querySelector(".timeline-play-button");
    if (play) play.textContent = "❚❚ Pause";
  }
}

async function exportTimeline(node, timeline) {
  node._timelineSaveHandle = null;
  node._timelineDownloadName = "video_timeline.mp4";
  if (window.showSaveFilePicker) {
    try {
      node._timelineSaveHandle = await window.showSaveFilePicker({ suggestedName: "video_timeline.mp4", types: [{ description: "MP4 video", accept: { "video/mp4": [".mp4"] } }] });
    } catch (error) { if (error.name === "AbortError") return; setStatus(node, error.message || String(error), "error"); return; }
  } else {
    const suggested = window.prompt("Choose a filename for the exported MP4", "video_timeline.mp4");
    if (suggested === null) return;
    node._timelineDownloadName = suggested.trim() || "video_timeline.mp4";
  }
  if (node._timelinePlaying) pausePlayback(node);
  saveTimeline(node, timeline);
  // Render straight from the node's cached media: the workflow is left untouched,
  // so exporting the edit as it stands never regenerates upstream footage.
  setStatus(node, "Rendering the current timeline…", "info");
  node._timelineExporting = true;
  try {
    const response = await api.fetchApi("/video_timeline_editor/export_now", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: String(node.id), timeline }),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error || `Could not render the timeline (${response.status}).`);
    await saveExport(node, result.token);
  } catch (error) {
    node._timelineExporting = false;
    node._timelineSaveHandle = null;
    setStatus(node, error.message || String(error), "error");
  }
}

app.registerExtension({
  name: "Comfy.VideoTimelineEditor",
  nodeCreated(node) {
    if (!isTimelineNode(node)) return;
    const root = document.createElement("div");
    Object.assign(root.style, { width: "100%", height: "100%", minWidth: "0", minHeight: "0", overflow: "auto", padding: "8px", boxSizing: "border-box", color: "#e7e9ed", fontFamily: "sans-serif", userSelect: "none" });
    node._timelineRoot = root;
    const minHeight = () => {
      const ratio = Number(node._timelineAspectRatio) > 0 ? Number(node._timelineAspectRatio) : 16 / 9;
      const width = Math.max(240, Number(node.size?.[0] || 560) - 28);
      return Math.ceil(Math.max(180, width / ratio) + 330);
    };
    const domWidget = node.addDOMWidget("timeline_editor", "video_timeline_editor", root, { hideOnZoom: false, getMinHeight: minHeight });
    domWidget.serialize = false;
    domWidget.options.getMinHeight = minHeight;
    for (const name of ["timeline_json", "ui_action"]) {
      const hiddenWidget = widget(node, name);
      if (hiddenWidget) { hiddenWidget.computeSize = () => [0, -4]; hiddenWidget.type = "timeline-hidden-state"; }
    }
    const actionWidget = widget(node, "ui_action");
    if (actionWidget) {
      actionWidget.value = "inspect";
      actionWidget.serialize = false;
      actionWidget.serializeValue = () => {
        const action = node._timelineQueuedAction || "inspect";
        node._timelineQueuedAction = null;
        return action;
      };
    }
    node._timelineQueuedAction = null;
    node._timelineFps = selectedFps(node);
    node._timelineLinkAV = true;
    render(node);
    resizeTimelineNode(node);
    node._timelineKeyHandler = (event) => {
      const selectedNodes = app.canvas?.selected_nodes || {};
      const selected = Object.values(selectedNodes).includes(node) || app.canvas?.current_node === node;
      if (!selected || event.altKey || event.ctrlKey || event.metaKey) return;
      const target = event.target;
      if (target instanceof HTMLElement && (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))) return;
      const key = event.key.toLowerCase();
      if ([" ", "arrowleft", "arrowright", "j", "k", "l", "+", "=", "-", "_"].includes(key)) event.preventDefault();
      if (event.repeat && [" ", "j", "k", "l", "+", "=", "-", "_"].includes(key)) return;
      if (key === "k") {
        event.preventDefault();
        pausePlayback(node);
      } else if (key === " ") {
        event.preventDefault();
        if (node._timelinePlaying) pausePlayback(node);
        else startPlayback(node);
      } else if (key === "j") {
        event.preventDefault();
        startPlayback(node, -1);
      } else if (key === "l") {
        event.preventDefault();
        startPlayback(node, 1);
      } else if (key === "arrowleft" || key === "arrowright") {
        event.preventDefault();
        if (node._timelinePlaying) pausePlayback(node);
        updatePlaybackDisplay(node, (node._timelinePlayhead || 0) + (key === "arrowright" ? 1 : -1));
      } else if (key === "+" || key === "=") {
        event.preventDefault();
        zoomTimeline(node, TIMELINE_ZOOM_STEP);
      } else if (key === "-" || key === "_") {
        event.preventDefault();
        zoomTimeline(node, 1 / TIMELINE_ZOOM_STEP);
      }
    };
    window.addEventListener("keydown", node._timelineKeyHandler);
    node.onRemoved = ((original) => function onRemoved() {
      stopPlayback(node);
      void exitMonitorFullscreen(node);
      node._timelineAudioGeneration = (node._timelineAudioGeneration || 0) + 1;
      if (node._timelineKeyHandler) window.removeEventListener("keydown", node._timelineKeyHandler);
      node._timelineAudioContext?.close?.();
      node._timelineAudioContext = null;
      node._timelineFrameRequestId = (node._timelineFrameRequestId || 0) + 1;
      node._timelinePendingFrame = null;
      node._timelineFrameAbort?.abort();
      clearPreviewCache(node);
      if (node._timelineFrameObjectUrl) URL.revokeObjectURL(node._timelineFrameObjectUrl);
      node._timelineFrameObjectUrl = null;
      original?.apply(this, arguments);
    })(node.onRemoved);
  },
});

api.addEventListener("video_timeline_editor.media_info", (event) => {
  const info = event.detail || {};
  const node = nodeById(info.node_id);
  if (!isTimelineNode(node)) return;
  const wasPlaying = Boolean(node._timelinePlaying);
  node._timelineLinkAV = true;
  node._timelineMediaInfo = { ...info, image_width: Number(info.image_width || 0), image_height: Number(info.image_height || 0) };
  clearPreviewCache(node);
  node._timelineAspectRatio = Number(info.image_width) > 0 && Number(info.image_height) > 0 ? Number(info.image_width) / Number(info.image_height) : 16 / 9;
  // Reaching the node means the graph just produced new footage, so the editor starts over
  // from that media: clips, selection, undo history and decoded preview audio from the
  // previous run all belong to media that no longer exists.
  const fps = FPS_VALUES.includes(Number(info.fps)) ? Number(info.fps) : selectedFps(node);
  node._timelineFps = fps;
  node._timelineSelection = null;
  node._timelineUndo = [];
  node._timelineRedo = [];
  node._timelinePlayhead = 0;
  node._timelineAudio = null;
  node._timelineAudioRaw = null;
  node._timelineAudioSignature = null;
  node._timelineAudioLoadPromise = null;
  node._timelineAudioLoading = false;
  node._timelineAudioGeneration = (node._timelineAudioGeneration || 0) + 1;
  const initialTimeline = emptyTimeline(info, fps);
  initialTimeline.media.fps = fps;
  saveTimeline(node, initialTimeline, false);
  render(node);
  if (wasPlaying && !node._timelinePlaying) startPlayback(node);
});

api.addEventListener("video_timeline_editor.exported", async (event) => {
  const result = event.detail || {};
  const node = nodeById(result.node_id);
  if (!isTimelineNode(node)) return;
  await saveExport(node, result.token);
});

async function saveExport(node, token) {
  try {
    const filename = node._timelineSaveHandle?.name || node._timelineDownloadName || "video_timeline.mp4";
    const response = await api.fetchApi(`/video_timeline_editor/export/${encodeURIComponent(token)}?filename=${encodeURIComponent(filename)}`);
    if (!response.ok) throw new Error("Could not retrieve the rendered MP4.");
    const blob = await response.blob();
    if (node._timelineSaveHandle) {
      const writable = await node._timelineSaveHandle.createWritable();
      await writable.write(blob);
      await writable.close();
      node._timelineSaveHandle = null;
      setStatus(node, "Video saved to the selected location.", "success");
    } else {
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = node._timelineDownloadName || "video_timeline.mp4";
      node._timelineDownloadName = null;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 30_000);
      setStatus(node, "Video download started. Set the browser to ask where to save downloads to choose its location.", "success");
    }
  } catch (error) {
    node._timelineSaveHandle = null;
    setStatus(node, error.message || String(error), "error");
  } finally {
    node._timelineExporting = false;
  }
}
