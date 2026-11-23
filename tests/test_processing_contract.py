"""Exercise options and result files across the HTTP and processing boundaries."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import app as webapp
from media_types import Transcript, TranscriptSegment, TranscriptionOptions
import ripper


URL = "https://youtu.be/abcdefghijk"
INFO = ripper.MediaInfo("Example", "abcdefghijk")
TRANSCRIPT = Transcript((TranscriptSegment(0, 1.25, "Hello world."),), "en", "YouTube uploaded captions")


class ProcessingContractTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name)
        for target, value in (("OUTPUT_DIR", self.output), ("dependency_issues", [])):
            patcher = patch.object(ripper, target, return_value=value) if target == "dependency_issues" else patch.object(ripper, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_caption_text_exports_all_formats_without_media_or_whisper(self):
        options = TranscriptionOptions(prefer_captions=True, subtitle_formats=("srt", "vtt"))
        with (
            patch.object(ripper, "fetch_captions", return_value=({"id": INFO.video_id, "title": INFO.title}, TRANSCRIPT)),
            patch.object(ripper, "download_audio") as download,
            patch.object(ripper.transcription, "transcribe") as transcribe,
        ):
            events = list(ripper.process(URL, "text", options=options))
        download.assert_not_called()
        transcribe.assert_not_called()
        files = next(data for kind, data in events if kind == "result")
        self.assertEqual(set(files), {"text", "srt", "vtt"})
        for filename in files.values():
            self.assertTrue((self.output / filename).is_file())
        self.assertIn(TRANSCRIPT.source, (self.output / files["text"]).read_text())
        self.assertIn("00:00:00,000 --> 00:00:01,250", (self.output / files["srt"]).read_text())

    def test_missing_captions_fall_back_to_selected_engine_and_model(self):
        media = self.output / "audio.m4a"
        media.write_bytes(b"audio")
        options = TranscriptionOptions(model="small", language="en", prefer_captions=True, backend="faster-whisper")
        with (
            patch.object(ripper, "fetch_captions", return_value=None),
            patch.object(ripper, "download_audio", return_value=ripper.DownloadedMedia(media, INFO)),
            patch.object(ripper.transcription, "transcribe", return_value=TRANSCRIPT) as transcribe,
        ):
            events = list(ripper.process(URL, "text", options=options))
        transcribe.assert_called_once_with(media, options)
        self.assertTrue(any(kind == "result" for kind, _ in events))
        self.assertFalse(media.exists())

    def test_video_with_captions_downloads_video_but_skips_transcription(self):
        media = self.output / "video.webm"
        media.write_bytes(b"video")
        with (
            patch.object(ripper, "fetch_captions", return_value=({"id": INFO.video_id, "title": INFO.title}, TRANSCRIPT)),
            patch.object(ripper, "download_video", return_value=ripper.DownloadedMedia(media, INFO)),
            patch.object(ripper.transcription, "transcribe") as transcribe,
        ):
            events = list(ripper.process(URL, "video_text", options=TranscriptionOptions(prefer_captions=True)))
        transcribe.assert_not_called()
        files = next(data for kind, data in events if kind == "result")
        self.assertEqual(files["video"], "video.webm")
        self.assertIn("text", files)

    def test_caption_fallback_reports_missing_ffmpeg_without_starting_download(self):
        with (
            patch.object(ripper, "fetch_captions", return_value=None),
            patch.object(ripper, "dependency_issues", return_value=["ffmpeg missing"]),
            patch.object(ripper, "download_audio") as download,
        ):
            events = list(ripper.process(URL, "text", options=TranscriptionOptions(prefer_captions=True)))
        download.assert_not_called()
        self.assertIn(("error", "ffmpeg missing"), events)
        self.assertFalse(any(kind == "result" for kind, _ in events))

    def test_transcription_failure_never_reports_success(self):
        media = self.output / "audio.m4a"
        media.write_bytes(b"audio")
        with (
            patch.object(ripper, "download_audio", return_value=ripper.DownloadedMedia(media, INFO)),
            patch.object(ripper.transcription, "transcribe", side_effect=RuntimeError("model unavailable")),
        ):
            events = list(ripper.process(URL, "both", options=TranscriptionOptions(backend="faster-whisper")))
        self.assertIn(("error", "Transcription failed: model unavailable"), events)
        self.assertFalse(any(kind == "result" for kind, _ in events))


class RequestContractTests(unittest.TestCase):
    def setUp(self):
        self.client = webapp.app.test_client()
        patcher = patch.object(webapp, "dependency_issues", return_value=[])
        self.check = patcher.start()
        self.addCleanup(patcher.stop)

    def test_options_reach_processor_and_subtitle_links_reach_stream(self):
        settings = {"source": "captions", "language": "en", "model": "small", "backend": "mlx", "subtitle_formats": ["srt", "vtt"]}
        with patch.object(ripper, "process", return_value=[("result", {"text": "a.txt", "srt": "a.srt", "vtt": "a.vtt"})]) as process:
            response = self.client.post("/rip", json={"url": URL, "mode": "text", "transcription": settings})
            payload = json.loads(response.get_data(as_text=True).removeprefix("data: ").strip())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(payload["data"]), {"text", "srt", "vtt"})
        process.assert_called_once_with(URL, "text", "1080p", TranscriptionOptions("small", "en", True, ("srt", "vtt"), "mlx"))
        self.check.assert_called_once_with(require_ffmpeg=False)

    def test_invalid_settings_do_not_start_processing(self):
        with patch.object(ripper, "process") as process:
            for setting in ({"model": "arbitrary/model"}, {"language": []}, {"backend": "unknown"}, {"subtitle_formats": "srt"}, {"subtitle_formats": ["pdf"]}):
                with self.subTest(setting=setting):
                    response = self.client.post("/rip", json={"url": URL, "transcription": setting})
                    self.assertEqual(response.status_code, 400)
        process.assert_not_called()

    def test_local_source_uses_english_and_the_explicit_engine(self):
        for backend in (None, "mlx", "faster-whisper"):
            with self.subTest(backend=backend), patch.object(ripper, "process", return_value=[]) as process:
                settings = {"source": "local"}
                if backend is not None:
                    settings["backend"] = backend
                response = self.client.post("/rip", json={"url": URL, "mode": "text", "transcription": settings})
                response.get_data()
                self.assertEqual(response.status_code, 200)
                options = process.call_args.args[3]
                self.assertEqual(options.language, "en")
                self.assertEqual(options.backend, backend or "mlx")
                self.assertFalse(options.prefer_captions)

    def test_dependency_failures_return_actionable_service_error(self):
        self.check.return_value = ["Install ffmpeg"]
        with patch.object(ripper, "process") as process:
            response = self.client.post("/rip", json={"url": URL})
        self.assertEqual(response.status_code, 503)
        self.assertIn("Install ffmpeg", response.get_data(as_text=True))
        process.assert_not_called()

    def test_page_renders_new_controls_and_dependency_warning(self):
        self.check.return_value = ["Install Node.js"]
        response = self.client.get("/")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        for text in ("transcription-backend", "value=\"mlx\"", "Install Node.js"):
            self.assertIn(text, html)
        self.assertNotIn('subtitle-format', html)
        self.assertNotIn('id="language"', html)
        self.assertNotIn('value="auto"', html)
        self.assertIn('value="local">Transcribe locally', html)
        self.assertIn('value="mlx" selected>Apple MLX', html)


if __name__ == "__main__":
    unittest.main()
