import sys
import weakref
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
        self.model.transcribe.side_effect = lambda *args, **kwargs: (
            iter([SimpleNamespace(start=1.25, end=2.75, text=" Bonjour. ")]),
            SimpleNamespace(language="fr"),
        )
        self.patches = [
            patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=self.factory)}),
            patch.dict("os.environ", {}, clear=True),
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

    def test_explicit_language_is_passed_and_model_environment_is_ignored(self):
        with patch.dict("os.environ", {"YOUTUBE_RIPPER_WHISPER_MODEL": "tiny"}):
            transcription.transcribe(Path("audio.wav"), TranscriptionOptions(language="fr"))

        self.factory.assert_called_once_with("turbo", device="cpu", compute_type="int8")
        self.assertEqual(self.model.transcribe.call_args.kwargs["language"], "fr")

    def test_environment_device_and_compute_type_are_supported(self):
        with patch.dict("os.environ", {
            "YOUTUBE_RIPPER_WHISPER_DEVICE": "cuda",
            "YOUTUBE_RIPPER_WHISPER_COMPUTE_TYPE": "float16",
        }):
            transcription.transcribe(Path("audio.wav"), TranscriptionOptions())

        self.factory.assert_called_once_with("turbo", device="cuda", compute_type="float16")

    def test_model_is_loaded_fresh_for_each_call(self):
        transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        transcription.transcribe(Path("audio.wav"), TranscriptionOptions())

        self.assertEqual(self.factory.call_count, 2)

    def test_lazy_inference_remains_locked_until_segments_are_consumed(self):
        def segments():
            self.assertTrue(transcription._model_lock.locked())
            yield SimpleNamespace(start=0, end=1, text="Hello")
            self.assertTrue(transcription._model_lock.locked())

        self.model.transcribe.side_effect = None
        self.model.transcribe.return_value = (segments(), SimpleNamespace(language="en"))
        result = transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.assertEqual(result.segments, (TranscriptSegment(0, 1, "Hello"),))
        self.assertFalse(transcription._model_lock.locked())

    def test_inference_error_releases_lock(self):
        def segments():
            raise RuntimeError("Cannot decode audio")
            yield

        self.model.transcribe.side_effect = None
        self.model.transcribe.return_value = (segments(), SimpleNamespace(language="en"))
        with self.assertRaisesRegex(RuntimeError, "Cannot decode audio"):
            transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.assertFalse(transcription._model_lock.locked())

    def test_model_load_error_releases_lock(self):
        self.factory.side_effect = RuntimeError("Model unavailable")
        with self.assertRaisesRegex(RuntimeError, "Model unavailable"):
            transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.assertFalse(transcription._model_lock.locked())


class MlxTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.holder = SimpleNamespace(model=None, model_path=None)
        self.mlx = Mock()

        def inference(*args, **kwargs):
            # The real library caches the loaded model on ModelHolder.
            self.assertTrue(transcription._model_lock.locked())
            self.holder.model = object()
            self.holder.model_path = kwargs["path_or_hf_repo"]
            return {
                "segments": [{"start": 1.25, "end": 2.75, "text": " Bonjour. "}],
                "language": "fr",
            }

        self.mlx.transcribe.side_effect = inference
        # "import mlx.core as mx" resolves to the parent module's attribute.
        self.mx = Mock()
        mlx_parent = SimpleNamespace(core=self.mx)
        patches = [
            patch.dict(sys.modules, {
                "mlx": mlx_parent,
                "mlx.core": self.mx,
                "mlx_whisper": self.mlx,
                "mlx_whisper.transcribe": SimpleNamespace(ModelHolder=self.holder),
            }),
            patch.dict("os.environ", {}, clear=True),
            patch.object(transcription.platform, "system", return_value="Darwin"),
            patch.object(transcription.platform, "machine", return_value="arm64"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def assert_released(self):
        self.assertIsNone(self.holder.model)
        self.assertIsNone(self.holder.model_path)
        self.mx.clear_cache.assert_called_once_with()
        self.assertFalse(transcription._model_lock.locked())

    def test_apple_silicon_defaults_to_mlx_turbo_with_timed_segments(self):
        result = transcription.transcribe(Path("video.mp4"), TranscriptionOptions())
        self.mlx.transcribe.assert_called_once_with(
            "video.mp4", path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language=None, verbose=None
        )
        self.assertEqual(result, Transcript((TranscriptSegment(1.25, 2.75, "Bonjour."),), "fr", "Local Whisper (MLX)"))

    def test_language_is_passed_and_model_environment_is_ignored(self):
        with patch.dict("os.environ", {"YOUTUBE_RIPPER_WHISPER_MODEL": "small"}):
            transcription.transcribe(Path("audio.wav"), TranscriptionOptions(language="fr"))
        self.assertEqual(self.mlx.transcribe.call_args.kwargs["path_or_hf_repo"], transcription.MLX_MODEL_REPO)
        self.assertEqual(self.mlx.transcribe.call_args.kwargs["language"], "fr")

    def test_successful_call_unloads_model_and_clears_cache(self):
        transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.assert_released()

    def test_failing_call_unloads_model_and_clears_cache(self):
        class Model:
            pass

        state = {}

        def inference(*args, **kwargs):
            # A local in the failing frame stands in for the loaded model.
            model = Model()
            state["model"] = weakref.ref(model)
            self.holder.model = object()
            self.holder.model_path = kwargs["path_or_hf_repo"]
            raise RuntimeError("Cannot decode audio")

        def clear_cache():
            # The release must run after the except block ends, or the
            # traceback still pins the failing frame and nothing is freed.
            state["freed_before_release"] = state["model"]() is None
            state["locked_during_release"] = transcription._model_lock.locked()

        self.mlx.transcribe.side_effect = inference
        self.mx.clear_cache.side_effect = clear_cache
        # A captured log record would hold the traceback, so stub the logger.
        with patch.object(transcription.logger, "exception") as log_exception:
            with self.assertRaisesRegex(RuntimeError, "MLX transcription failed.*Cannot decode audio") as caught:
                transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        log_exception.assert_called_once()
        self.assert_released()
        self.assertTrue(state["freed_before_release"])
        self.assertTrue(state["locked_during_release"])
        # A chained cause would keep the failed call's frames, and the model, alive.
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_failing_release_keeps_the_transcript(self):
        self.mx.clear_cache.side_effect = RuntimeError("Metal unavailable")
        with self.assertLogs(transcription.logger, level="ERROR"):
            result = transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.assertEqual(result.segments, (TranscriptSegment(1.25, 2.75, "Bonjour."),))
        self.assertFalse(transcription._model_lock.locked())

    def test_failing_release_does_not_mask_the_transcription_error(self):
        self.mlx.transcribe.side_effect = RuntimeError("Missing model weights")
        self.mx.clear_cache.side_effect = RuntimeError("Metal unavailable")
        with self.assertLogs(transcription.logger, level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "MLX transcription failed.*Missing model weights"):
                transcription.transcribe(Path("audio.wav"), TranscriptionOptions())

    def test_environment_backend_override_is_honoured(self):
        faster = Mock()
        faster.WhisperModel.return_value.transcribe.return_value = ([], SimpleNamespace(language="en"))
        with patch.dict("os.environ", {"YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND": "faster-whisper"}):
            self.assertEqual(transcription.resolve_backend(), "faster-whisper")
            with patch.dict(sys.modules, {"faster_whisper": faster}):
                result = transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.assertEqual(result.source, "Local Whisper")
        self.mlx.transcribe.assert_not_called()
        self.mx.clear_cache.assert_not_called()
        with patch.dict("os.environ", {"YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND": "mlx"}):
            self.assertEqual(transcription.resolve_backend(), "mlx")

    def test_other_platforms_default_to_faster_whisper_and_reject_explicit_mlx(self):
        for system, machine in (("Linux", "aarch64"), ("Darwin", "x86_64"), ("Windows", "AMD64")):
            with self.subTest(system=system), patch.object(transcription.platform, "system", return_value=system), patch.object(transcription.platform, "machine", return_value=machine):
                self.assertEqual(transcription.resolve_backend(), "faster-whisper")
                with patch.dict("os.environ", {"YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND": "mlx"}):
                    with self.assertRaisesRegex(RuntimeError, "Apple Silicon.*YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND"):
                        transcription.resolve_backend()

    def test_invalid_environment_backend_fails_clearly(self):
        with patch.dict("os.environ", {"YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND": "bogus"}):
            with self.assertRaisesRegex(ValueError, "backend"):
                transcription.resolve_backend()

    def test_mlx_failure_is_actionable_and_does_not_fallback(self):
        self.mlx.transcribe.side_effect = RuntimeError("Missing model weights")
        with patch.dict(sys.modules, {"faster_whisper": None}), self.assertLogs(transcription.logger, level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "MLX transcription failed.*Missing model weights.*Hugging Face"):
                transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.assertFalse(transcription._model_lock.locked())

    def test_missing_mlx_dependency_explains_install_command(self):
        with patch.dict(sys.modules, {"mlx_whisper": None}):
            with self.assertRaisesRegex(RuntimeError, "uv sync --locked"):
                transcription.transcribe(Path("audio.wav"), TranscriptionOptions())
        self.assertFalse(transcription._model_lock.locked())


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
