import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (
    explicit_transition_mode,
    frames_to_audio_latents,
    transition_for_overlap,
    transition_offset_tokens,
    parse_timestamp,
    parse_timestamped_beats,
    plan_windows,
    requested_frames_to_latents,
    remap_keyframes_for_window,
    select_beats_for_latent_window,
    select_beats_for_window,
    video_frames_for_latents,
)


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.device = "cpu"
        self.dtype = "float32"
        self.writes = []

    def to(self, device=None, dtype=None):
        return self

    def __getitem__(self, key):
        return _FakeTensor(self.shape)

    def __setitem__(self, key, value):
        self.writes.append((key, value))


class _FakeNestedSamples:
    def __init__(self, video, audio):
        self.video = video
        self.audio = audio

    def unbind(self):
        return self.video, self.audio


class H3ContinuationAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nodes_path = Path(__file__).resolve().parents[1] / "nodes.py"
        tree = ast.parse(nodes_path.read_text())
        wanted = {"_append_continuation", "_audio_tokens_for_video_prefix"}
        functions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        namespace = {
            "transition_offset_tokens": transition_offset_tokens,
            "video_frames_for_latents": video_frames_for_latents,
            "frames_to_audio_latents": frames_to_audio_latents,
            "_blend_transition": lambda previous, current, time_dim=-1: previous,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(nodes_path), "exec"), namespace)
        cls.append_continuation = staticmethod(namespace["_append_continuation"])

    def test_all_join_modes_return_the_assembled_av_pair(self):
        cases = (
            ("continue", None),
            ("cut", 128),
            ("crossfade", None),
        )
        for mode, transition in cases:
            with self.subTest(mode=mode):
                assembled_video = _FakeTensor((1, 24, 80, 2, 2))
                assembled_audio = _FakeTensor((1, 32, 2, 400))
                sampled = _FakeNestedSamples(
                    _FakeTensor((1, 24, 37, 2, 2)),
                    _FakeTensor((1, 32, 2, 80)),
                )
                result = self.append_continuation(
                    assembled_video,
                    assembled_audio,
                    sampled,
                    (37, 74),
                    (207, 287),
                    (0, 49),
                    (0, 227),
                    transition,
                    mode,
                )
                self.assertEqual(result, (assembled_video, assembled_audio))
                self.assertTrue(assembled_video.writes)
                self.assertTrue(assembled_audio.writes)


class H3WindowPlanningTests(unittest.TestCase):
    def test_native_124_frame_latent_shape(self):
        self.assertEqual(video_frames_for_latents(37), 124)
        self.assertEqual(frames_to_audio_latents(124), 207)

    def test_rounds_frame_request_to_h3_pattern(self):
        self.assertEqual(requested_frames_to_latents(124), 37)
        self.assertEqual(video_frames_for_latents(requested_frames_to_latents(125)), 141)

    def test_plans_overlapping_aligned_windows_without_gaps(self):
        windows = plan_windows(102, 37, 12)
        self.assertEqual(windows[0], (0, 37))
        self.assertEqual(windows[-1][1], 102)
        for previous, current in zip(windows, windows[1:]):
            self.assertEqual(previous[1] - current[0], 12)
            self.assertEqual(current[0] % 5, 0)

    def test_rejects_overlap_that_leaves_no_progress(self):
        with self.assertRaises(ValueError):
            plan_windows(42, 17, 17)

    def test_parses_explicit_and_implicit_beat_ranges(self):
        beats = parse_timestamped_beats(
            "[00:00:05-00:00:12] A train enters the station\n"
            "[00:12] The passenger boards\n"
            "# comment ignored"
        )
        self.assertEqual(beats[0], {"start": 5.0, "end": 12.0, "text": "A train enters the station"})
        self.assertEqual(beats[1], {"start": 12.0, "end": None, "text": "The passenger boards"})

    def test_selects_intersecting_beats_and_rebases_to_chunk_time(self):
        beats = parse_timestamped_beats(
            "[00:00:00-00:00:10] Establish the city\n"
            "[00:00:10-00:00:20] Reveal the bridge"
        )
        selected = select_beats_for_window(beats, 8.0, 16.0)
        self.assertIn("local 00:00:00.000-00:00:02.000", selected)
        self.assertIn("global 00:00:08.000-00:00:10.000", selected)
        self.assertIn("local 00:00:02.000-00:00:08.000", selected)
        self.assertIn("Reveal the bridge", selected)

    def test_routes_beats_using_global_h3_latent_window_times(self):
        beats = parse_timestamped_beats(
            "[00:00:00-00:00:05] Opening\n"
            "[00:00:05-00:00:10] Middle\n"
            "[00:00:10-00:00:15] Ending"
        )
        # 37 H3 tokens represent 124 video frames; starting at token 37
        # therefore selects the story beats around global 00:00:05.167.
        selected = select_beats_for_latent_window(beats, 37, 74)
        self.assertIn("global 00:00:05.167-00:00:10.000", selected)
        self.assertIn("Middle", selected)
        self.assertIn("Ending", selected)
        self.assertNotIn("Opening", selected)

    def test_keyframes_are_filtered_and_rebased_for_each_overlap_window(self):
        first_guide = {"resolved_frame_index": 110, "latent": object(), "audio_latent": object()}
        second_guide = {"resolved_frame_index": 170, "latent": object()}
        guides = [(first_guide, 1), (second_guide, 1)]

        first = remap_keyframes_for_window(guides, 0, 124)
        second = remap_keyframes_for_window(guides, 85, 209)
        third = remap_keyframes_for_window(guides, 170, 294)

        self.assertEqual([g["resolved_frame_index"] for g in first], [110])
        self.assertEqual([g["resolved_frame_index"] for g in second], [25, 85])
        self.assertEqual([g["resolved_frame_index"] for g in third], [0])
        self.assertIs(first[0]["latent"], first_guide["latent"])
        self.assertIs(first[0]["audio_latent"], first_guide["audio_latent"])
        self.assertEqual(first_guide["resolved_frame_index"], 110)

    def test_keyframe_clip_overlapping_boundary_is_kept_whole_and_rebased(self):
        guide = {"resolved_frame_index": 100, "latent": object()}
        mapped = remap_keyframes_for_window([(guide, 40)], 120, 200)
        self.assertEqual(mapped[0]["resolved_frame_index"], -20)
        self.assertIs(mapped[0]["latent"], guide["latent"])

    def test_keyframe_outside_windows_is_filtered(self):
        guide = {"resolved_frame_index": 30, "latent": object()}
        self.assertEqual(remap_keyframes_for_window([(guide, 5)], 40, 80), [])

    def test_rejects_invalid_keyframe_positions_and_durations(self):
        with self.assertRaises(ValueError):
            remap_keyframes_for_window([({"resolved_frame_index": -1}, 1)], 0, 10)
        with self.assertRaises(ValueError):
            remap_keyframes_for_window([({"resolved_frame_index": 1}, 0)], 0, 10)

    def test_rejects_invalid_or_unsorted_timestamps(self):
        with self.assertRaises(ValueError):
            parse_timestamp("00:61")
        with self.assertRaises(ValueError):
            parse_timestamped_beats("[00:10] later\n[00:05] earlier")
        with self.assertRaises(ValueError):
            parse_timestamped_beats("unstructured line")
        with self.assertRaises(ValueError):
            parse_timestamped_beats("[00:00-00:10] overlaps\n[00:05-00:15] next")

    def test_transition_cues_are_explicit_and_default_to_continuation(self):
        self.assertIsNone(explicit_transition_mode("The same runner continues forward"))
        self.assertEqual(explicit_transition_mode("Dissolve to a quiet street"), "crossfade")
        self.assertEqual(explicit_transition_mode("Cut to a new scene"), "cut")

    def test_story_beat_transition_is_routed_only_when_its_start_is_in_overlap(self):
        beats = parse_timestamped_beats(
            "[00:00:05] Motion continues\n"
            "[00:00:06] Cut to a new scene\n"
            "[00:00:08] Dissolve to night"
        )
        self.assertEqual(transition_for_overlap(beats, 144, 192), {"mode": "cut", "frame": 144})
        self.assertEqual(transition_for_overlap(beats, 192, 240), {"mode": "crossfade", "frame": 192})
        self.assertIsNone(transition_for_overlap(beats, 0, 120))

    def test_transition_frame_maps_to_first_h3_token_at_or_after_cue(self):
        self.assertEqual(transition_offset_tokens(37, 12, 124), 0)
        self.assertEqual(transition_offset_tokens(37, 12, 128), 1)
        with self.assertRaises(ValueError):
            transition_offset_tokens(37, 12, 200)

    def test_open_ended_final_beat_continues_across_later_chunks(self):
        beats = parse_timestamped_beats("[00:00:30] The storm continues")
        first = select_beats_for_window(beats, 28.0, 35.0)
        later = select_beats_for_window(beats, 35.0, 42.0)
        self.assertIn("The storm continues", first)
        self.assertIn("The storm continues", later)
        self.assertIn("local 00:00:00.000-00:00:07.000", later)


if __name__ == "__main__":
    unittest.main()