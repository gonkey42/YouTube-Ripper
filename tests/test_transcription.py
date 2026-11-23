import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import transcription
from media_types import Transcript, TranscriptSegment, TranscriptionOptions


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.factory = Mock()
        self.model = self.factory.return_value
        self.model.transcribe.return_value = (
            iter([SimpleNamespace(start=1.25, end=2.75, text=" Bonjour. ")]),
            SimpleNamespace(language="fr"),
        )
        self.patches = [
            patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=self.factory)}),
            patch.dict("os.environ", {}, clear=True),
            patch.object(transcription, "_model", None),
            patch.object(transcription, "_model_key", None),
            patch.object(transcription, "_mlx_model_holder", None),
            patch.object(transcription.platform, "system", return_value="Linux"),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_auto_detection_retains_times_language_and_source(self):
        result = transcription.transcribe(Path("audio.m4a"), TranscriptionOptions())

        self.factory.assert_called_once_with("turbo", device="cpu", compute_type="int8")
        self.model.transcribe.assert_called_once_with(
            "audio.m4a", beam_size=5, language=None, vad_filter=True
        )
        self.assertEqual(result, Transcript((TranscriptSegment(1.25, 2.75, "Bonjour."),), "fr", "Local Whisper"))

    def test_explicit_language_and_model_override_environment(self):
        with patch.dict("os.environ", {"YOUTUBE_RIPPER_WHISPER_MODEL": "tiny"}):
            transcription.transcribe(Path("audio.wav"), TranscriptionOptions(model="small", language="fr"))

        self.factory.assert_called_once_with("small", device="cpu", compute_type="int8")
        self.assertEqual(self.model.transcribe.call_args.kwargs["language"], "fr")

    def test_environment_model_device_and_compute_type_are_supported(self):
        with patch.dict("os.environ", {
            "YOUTUBE_RIPPER_WHISPER_MODEL": "medium",
            "YOUTUBE_RIPPER_WHISPER_DEVICE": "cuda",
            "YOUTUBE_RIPPER_WHISPER_COMPUTE_TYPE": "float16",
        }):
            transcription.transcribe(Path("audio.wav"), TranscriptionOptions())

        self.factory.assert_called_once_with("medium", device="cuda", compute_type="float16")

    def test_cache_reuses_only_current_model(self):
        for model in ("base", "base", "small", "base"):
            transcription.transcribe(Path("audio.wav"), TranscriptionOptions(model=model))

        self.assertEqual([call.args[0] for call in self.factory.call_args_list], ["base", "small", "base"])

    def test_cache_includes_device_and_compute_type(self):
        transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        with patch.dict("os.environ", {"YOUTUBE_RIPPER_WHISPER_DEVICE": "cuda"}):
            transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
            with patch.dict("os.environ", {"YOUTUBE_RIPPER_WHISPER_COMPUTE_TYPE": "float16"}):
                transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.assertEqual(self.factory.call_count, 3)

    def test_lazy_inference_remains_locked_until_segments_are_consumed(self):
        def segments():
            self.assertTrue(transcription._model_lock.locked())
            yield SimpleNamespace(start=0, end=1, text="Hello")
            self.assertTrue(transcription._model_lock.locked())

        self.model.transcribe.return_value = (segments(), SimpleNamespace(language="en"))
        transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.assertFalse(transcription._model_lock.locked())

    def test_inference_error_releases_lock(self):
        def segments():
            raise RuntimeError("Cannot decode audio")
            yield

        self.model.transcribe.return_value = (segments(), SimpleNamespace(language="en"))
        with self.assertRaisesRegex(RuntimeError, "Cannot decode audio"):
            transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.assertFalse(transcription._model_lock.locked())

    def test_failed_replacement_does_not_leave_stale_model(self):
        transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.factory.side_effect = RuntimeError("Model unavailable")
        with self.assertRaisesRegex(RuntimeError, "Model unavailable"):
            transcription.transcribe(Path("audio.wav"), TranscriptionOptions(model="small"))
        self.assertIsNone(transcription._model)
        self.assertIsNone(transcription._model_key)
        self.assertFalse(transcription._model_lock.locked())


class MlxTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.holder = SimpleNamespace(model=None, model_path=None)
        self.mlx = Mock()
        self.mlx.transcribe.return_value = {
            "segments": [{"start": 1.25, "end": 2.75, "text": " Bonjour. "}],
            "language": "fr",
        }
        patches = [
            patch.dict(sys.modules, {
                "mlx_whisper": self.mlx,
                "mlx_whisper.transcribe": SimpleNamespace(ModelHolder=self.holder),
            }),
            patch.dict("os.environ", {}, clear=True),
            patch.object(transcription.platform, "system", return_value="Darwin"),
            patch.object(transcription.platform, "machine", return_value="arm64"),
            patch.object(transcription, "_model", None),
            patch.object(transcription, "_model_key", None),
            patch.object(transcription, "_mlx_model_holder", None),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_apple_silicon_defaults_to_mlx_with_timed_segments(self):
        result = transcription.transcribe(Path("video.mp4"), TranscriptionOptions())
        self.mlx.transcribe.assert_called_once_with(
            "video.mp4", path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language=None, verbose=None
        )
        self.assertEqual(result, Transcript((TranscriptSegment(1.25, 2.75, "Bonjour."),), "fr", "Local Whisper (MLX)"))

    def test_all_supported_model_names_map_to_mlx_repositories(self):
        for name, repo in transcription.MLX_MODELS.items():
            with self.subTest(model=name):
                transcription.transcribe(Path("audio.wav"), TranscriptionOptions(model=name, language="fr"))
                self.assertEqual(self.mlx.transcribe.call_args.kwargs["path_or_hf_repo"], repo)
                self.assertEqual(self.mlx.transcribe.call_args.kwargs["language"], "fr")
        self.assertEqual(transcription.MLX_MODELS["turbo"], "mlx-community/whisper-large-v3-turbo")

    def test_environment_backend_and_model_overrides(self):
        with patch.dict("os.environ", {"YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND": "faster-whisper"}):
            self.assertEqual(transcription.resolve_backend(TranscriptionOptions()), "faster-whisper")
            self.assertEqual(transcription.resolve_backend(TranscriptionOptions(backend="mlx")), "mlx")
        with patch.dict("os.environ", {"YOUTUBE_RIPPER_WHISPER_MODEL": "small"}):
            transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.assertEqual(self.mlx.transcribe.call_args.kwargs["path_or_hf_repo"], transcription.MLX_MODELS["small"])

    def test_other_platforms_default_to_faster_whisper_and_reject_explicit_mlx(self):
        for system, machine in (("Linux", "aarch64"), ("Darwin", "x86_64"), ("Windows", "AMD64")):
            with self.subTest(system=system), patch.object(transcription.platform, "system", return_value=system), patch.object(transcription.platform, "machine", return_value=machine):
                self.assertEqual(transcription.resolve_backend(TranscriptionOptions()), "faster-whisper")
                with self.assertRaisesRegex(RuntimeError, "Apple Silicon"):
                    transcription.resolve_backend(TranscriptionOptions(backend="mlx"))

    def test_invalid_backend_and_mlx_model_fail_clearly(self):
        with patch.dict("os.environ", {"YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND": "bogus"}):
            with self.assertRaisesRegex(ValueError, "backend"):
                transcription.resolve_backend(TranscriptionOptions())
        with self.assertRaisesRegex(ValueError, "MLX model"):
            transcription.transcribe(Path("audio.wav"), TranscriptionOptions(model="bogus"))

    def test_mlx_failure_is_actionable_and_does_not_fallback(self):
        self.mlx.transcribe.side_effect = RuntimeError("Missing model weights")
        with patch.dict(sys.modules, {"faster_whisper": None}):
            with self.assertRaisesRegex(RuntimeError, "MLX transcription failed.*Missing model weights.*Hugging Face"):
                transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.assertFalse(transcription._model_lock.locked())

    def test_missing_mlx_dependency_explains_install_command(self):
        with patch.dict(sys.modules, {"mlx_whisper": None}):
            with self.assertRaisesRegex(RuntimeError, "uv sync --locked"):
                transcription.transcribe(Path("audio.wav"), TranscriptionOptions())

    def test_mlx_cache_is_preserved_for_same_model_and_replaced_for_new_model(self):
        cached = object()
        self.holder.model = cached
        self.holder.model_path = transcription.MLX_MODELS["turbo"]
        def inference(*args, **kwargs):
            self.assertTrue(transcription._model_lock.locked())
            if kwargs["path_or_hf_repo"] == transcription.MLX_MODELS["turbo"]:
                self.assertIs(self.holder.model, cached)
            else:
                self.assertIsNone(self.holder.model)
            return {"segments": [], "language": "en"}
        self.mlx.transcribe.side_effect = inference
        transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        transcription.transcribe(Path("audio.wav"), TranscriptionOptions(model="small"))

    def test_switching_backends_releases_previous_model(self):
        transcription._model = object()
        transcription._model_key = ("base", "cpu", "int8")
        transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.assertIsNone(transcription._model)
        self.assertIsNone(transcription._model_key)
        self.holder.model = object()
        self.holder.model_path = transcription.MLX_MODELS["base"]
        faster = Mock()
        faster.WhisperModel.return_value.transcribe.return_value = ([], SimpleNamespace(language="en"))
        with patch.dict(sys.modules, {"faster_whisper": faster}):
            transcription.transcribe(Path("audio.wav"), TranscriptionOptions(backend="faster-whisper"))
        self.assertIsNone(self.holder.model)
        self.assertIsNone(self.holder.model_path)


class TranscriptExportTests(unittest.TestCase):
    def export(self, transcript, formats=("srt", "vtt")):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "nested"
            files = transcription.write_transcript_files(
                transcript,
                title="Example Video",
                url="https://youtu.be/abc123",
                stem="Example Video [abc123]",
                output_dir=output_dir,
                subtitle_formats=formats,
            )
            return files, {kind: (output_dir / filename).read_text(encoding="utf-8") for kind, filename in files.items()}

    def test_text_has_metadata_and_subtitles_retain_timing(self):
        transcript = Transcript((TranscriptSegment(1.25, 2.75, "Bonjour."),), "fr", "Local Whisper")
        files, contents = self.export(transcript)

        self.assertEqual(files, {
            "text": "Example Video [abc123].txt",
            "srt": "Example Video [abc123].srt",
            "vtt": "Example Video [abc123].vtt",
        })
        self.assertIn("Example Video\n\nSource: https://youtu.be/abc123", contents["text"])
        self.assertIn("Transcript source: Local Whisper\nLanguage: fr", contents["text"])
        self.assertIn("Bonjour.\n", contents["text"])
        self.assertEqual(contents["srt"], "1\n00:00:01,250 --> 00:00:02,750\nBonjour.\n\n")
        self.assertEqual(contents["vtt"], "WEBVTT\n\n00:00:01.250 --> 00:00:02.750\nBonjour.\n\n")

    def test_text_only_is_default_and_preserves_caption_source(self):
        transcript = Transcript((TranscriptSegment(0, 1, "Hello."),), "en", "YouTube automatic captions")
        files, contents = self.export(transcript, formats=())
        self.assertEqual(set(files), {"text"})
        self.assertIn("Transcript source: YouTube automatic captions", contents["text"])

    def test_empty_speech_creates_valid_empty_subtitles(self):
        transcript = Transcript((TranscriptSegment(0, 1, " \n "),), None, "Local Whisper")
        _, contents = self.export(transcript)
        self.assertIn("No speech detected", contents["text"])
        self.assertIn("Language: unknown", contents["text"])
        self.assertEqual(contents["srt"], "")
        self.assertEqual(contents["vtt"], "WEBVTT\n\n")

    def test_multiline_cues_escape_markup_and_remove_internal_blank_lines(self):
        transcript = Transcript((TranscriptSegment(0, 2, " <b>Hello & привет</b>\r\n\r\n next -->\x00 "),), "en", "Local Whisper")
        _, contents = self.export(transcript)
        expected = "&lt;b&gt;Hello &amp; привет&lt;/b&gt;\nnext --&gt;\n\n"
        self.assertTrue(contents["srt"].endswith(expected))
        self.assertTrue(contents["vtt"].endswith(expected))
        self.assertNotIn("\x00", contents["vtt"])

    def test_rounding_carries_across_minutes_and_hours(self):
        transcript = Transcript((TranscriptSegment(59.9996, 3599.9996, "Carry."),), "en", "Local Whisper")
        _, contents = self.export(transcript)
        self.assertIn("00:01:00,000 --> 01:00:00,000", contents["srt"])
        self.assertIn("00:01:00.000 --> 01:00:00.000", contents["vtt"])

    def test_negative_start_and_zero_length_cues_get_valid_times(self):
        transcript = Transcript((TranscriptSegment(-1, 0, "Hello."), TranscriptSegment(1, 1, "Again.")), "en", "Local Whisper")
        _, contents = self.export(transcript)
        self.assertIn("00:00:00,000 --> 00:00:00,001", contents["srt"])
        self.assertIn("2\n00:00:01,000 --> 00:00:01,001", contents["srt"])

    def test_bad_export_input_is_rejected_before_writing(self):
        transcript = Transcript((TranscriptSegment(0, float("inf"), "Hello."),), "en", "Local Whisper")
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            args = dict(title="Example", url="url", stem="example", output_dir=output_dir)
            with self.assertRaisesRegex(ValueError, "finite"):
                transcription.write_transcript_files(transcript, **args, subtitle_formats=("srt",))
            with self.assertRaisesRegex(ValueError, "formats"):
                transcription.write_transcript_files(transcript, **args, subtitle_formats=("html",))
            args["stem"] = "../outside"
            with self.assertRaisesRegex(ValueError, "filename"):
                transcription.write_transcript_files(transcript, **args)
            self.assertEqual(list(output_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
