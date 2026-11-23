import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from subprocess import CalledProcessError
from media_types import Transcript, TranscriptSegment, TranscriptionOptions
from unittest.mock import patch

import ripper
from yt_dlp.utils import DownloadError


class TranscriptOutputTests(unittest.TestCase):
    def setUp(self):
        check = patch.object(ripper, "dependency_issues", return_value=[])
        check.start()
        self.addCleanup(check.stop)

    def test_output_stem_includes_video_id(self):
        info = ripper.MediaInfo(title="Example: Video?", video_id="abc123")

        self.assertEqual(ripper._output_stem(info), "Example Video [abc123]")

    def test_generate_text_uses_video_id_in_filename(self):
        info = ripper.MediaInfo(title="Example Video", video_id="abc123")

        with TemporaryDirectory() as temp_dir:
            with patch.object(ripper, "OUTPUT_DIR", Path(temp_dir)):
                files = ripper._write_transcript(Transcript((TranscriptSegment(0, 1, "hello"),), "en", "Local Whisper"), info, "https://youtu.be/abc123", TranscriptionOptions())
                text_path = Path(temp_dir) / files["text"]
                text_content = text_path.read_text(encoding="utf-8")

        self.assertEqual(text_path.name, "Example Video [abc123].txt")
        self.assertIn("hello\n", text_content)

    def test_download_audio_uses_reported_path_with_bracketed_video_id(self):
        info = ripper.MediaInfo(title="Example Video", video_id="abc123")

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            fallback_path = output_dir / "Example Video [abc123].opus"
            fallback_path.write_bytes(b"audio")

            with (
                patch.object(ripper, "OUTPUT_DIR", output_dir),
                patch.object(ripper, "_fetch_media_info", return_value=info),
                patch.object(ripper.yt_dlp, "YoutubeDL") as youtube_dl,
            ):
                youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {"filepath": str(fallback_path)}
                downloaded = ripper.download_audio("https://youtu.be/abc123")

        self.assertEqual(downloaded.path, fallback_path)
        self.assertEqual(downloaded.info, info)

    def test_download_audio_retries_without_cookies_after_cookie_download_failure(self):
        info = ripper.MediaInfo(title="Example Video", video_id="abc123")
        seen_opts = []

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            output_path = output_dir / "Example Video [abc123].m4a"

            class FakeYoutubeDL:
                def __init__(self, opts):
                    seen_opts.append(opts)

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return False

                def extract_info(self, url, download):
                    if len(seen_opts) == 1:
                        raise DownloadError("ERROR: [youtube] abc123: No video formats found!")
                    output_path.write_bytes(b"audio")
                    return {"filepath": str(output_path)}

            with (
                patch.object(ripper, "OUTPUT_DIR", output_dir),
                patch.object(ripper, "_fetch_media_info", return_value=info),
                patch.object(ripper.yt_dlp, "YoutubeDL", FakeYoutubeDL),
                patch.dict("os.environ", {"YOUTUBE_RIPPER_COOKIES_FROM_BROWSER": "chrome:Profile 1"}, clear=True),
            ):
                downloaded = ripper.download_audio("https://youtu.be/abc123")

        self.assertEqual(downloaded.path, output_path)
        self.assertIn("cookiesfrombrowser", seen_opts[0])
        self.assertNotIn("cookiesfrombrowser", seen_opts[1])
        self.assertNotIn("cookiefile", seen_opts[1])

    def test_text_mode_returns_text_file_result(self):
        segments = Transcript((TranscriptSegment(0, 1, "Hello world."),), "en", "Local Whisper")

        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "example.m4a"
            audio_path.write_bytes(b"audio")

            with (
                patch.object(ripper, "OUTPUT_DIR", Path(temp_dir)),
                patch.object(
                    ripper,
                    "download_audio",
                    return_value=ripper.DownloadedMedia(
                        path=audio_path,
                        info=ripper.MediaInfo(title="Example Video", video_id="abc123"),
                    ),
                ),
                patch.object(ripper, "_run_transcription_with_keepalive", return_value=[("_done", segments)]),
            ):
                events = list(ripper.process("https://youtu.be/abc123", "text"))

        self.assertIn(("status", "Generating transcript files..."), events)
        self.assertIn(("result", {"text": "Example Video [abc123].txt"}), events)

    def test_video_text_transcribes_video_directly_without_extra_audio_conversion(self):
        segments = Transcript((TranscriptSegment(0, 1, "Hello from video."),), "en", "Local Whisper")

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            video_path = output_dir / "Example Video [abc123].mp4"
            video_path.write_bytes(b"video")
            temp_audio_path = output_dir / "Example Video [abc123].transcription.m4a"
            temp_audio_path.write_bytes(b"audio")
            info = ripper.MediaInfo(title="Example Video", video_id="abc123")

            with (
                patch.object(ripper, "OUTPUT_DIR", output_dir),
                patch.object(
                    ripper,
                    "_run_video_download_with_progress",
                    return_value=[("_done", ripper.DownloadedMedia(path=video_path, info=info))],
                ) as video_download,
                patch.object(ripper, "download_audio") as audio_download,
                patch.object(ripper, "extract_audio_from_video", return_value=temp_audio_path) as extract_audio,
                patch.object(ripper, "_run_transcription_with_keepalive", return_value=[("_done", segments)]) as transcribe,
            ):
                events = list(ripper.process("https://youtu.be/abc123", "video_text"))

        video_download.assert_called_once_with("https://youtu.be/abc123", "1080p")
        audio_download.assert_not_called()
        extract_audio.assert_not_called()
        transcribe.assert_called_once_with(video_path, TranscriptionOptions())
        self.assertIn(("result", {"video": "Example Video [abc123].mp4", "text": "Example Video [abc123].txt"}), events)

    def test_extract_audio_from_video_cleans_partial_file_on_ffmpeg_failure(self):
        with TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "Example Video [abc123].mp4"
            video_path.write_bytes(b"video")
            temp_audio_path = Path(temp_dir) / "Example Video [abc123].transcription.m4a"
            temp_audio_path.write_bytes(b"partial")

            with patch.object(
                ripper.subprocess,
                "run",
                side_effect=CalledProcessError(1, ["ffmpeg"], stderr="failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "ffmpeg audio extraction failed"):
                    ripper.extract_audio_from_video(video_path)

            self.assertFalse(temp_audio_path.exists())

    def test_transcribe_to_text_file_emits_done_path(self):
        segments = Transcript((TranscriptSegment(0, 1, "Hello helper."),), "en", "Local Whisper")
        info = ripper.MediaInfo(title="Helper Video", video_id="abc123")

        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.m4a"
            audio_path.write_bytes(b"audio")

            with (
                patch.object(ripper, "OUTPUT_DIR", Path(temp_dir)),
                patch.object(ripper, "_run_transcription_with_keepalive", return_value=[("_done", segments)]),
            ):
                events = list(ripper._transcribe_to_text_file(audio_path, info, "https://youtu.be/abc123"))

        self.assertTrue(any(kind == "status" and text.startswith("Transcribing with") for kind, text in events))
        self.assertIn(("status", "Generating transcript files..."), events)
        self.assertIn(("_done", {"text": "Helper Video [abc123].txt"}), events)


if __name__ == "__main__":
    unittest.main()
