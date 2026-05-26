import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ripper


class TranscriptOutputTests(unittest.TestCase):
    def test_output_stem_includes_video_id(self):
        info = ripper.MediaInfo(title="Example: Video?", video_id="abc123")

        self.assertEqual(ripper._output_stem(info), "Example Video [abc123]")

    def test_generate_text_uses_video_id_in_filename(self):
        info = ripper.MediaInfo(title="Example Video", video_id="abc123")

        with TemporaryDirectory() as temp_dir:
            with patch.object(ripper, "OUTPUT_DIR", Path(temp_dir)):
                text_path = ripper.generate_text("hello\n", info)
                text_content = text_path.read_text(encoding="utf-8")

        self.assertEqual(text_path.name, "Example Video [abc123].txt")
        self.assertEqual(text_content, "hello\n")

    def test_download_audio_fallback_matches_bracketed_video_id_literal(self):
        info = ripper.MediaInfo(title="Example Video", video_id="abc123")

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            fallback_path = output_dir / "Example Video [abc123].opus"
            fallback_path.touch()

            with (
                patch.object(ripper, "OUTPUT_DIR", output_dir),
                patch.object(ripper, "_fetch_media_info", return_value=info),
                patch.object(ripper.yt_dlp, "YoutubeDL") as youtube_dl,
            ):
                youtube_dl.return_value.__enter__.return_value.download.return_value = None
                downloaded = ripper.download_audio("https://youtu.be/abc123")

        self.assertEqual(downloaded.path, fallback_path)
        self.assertEqual(downloaded.info, info)

    def test_text_mode_returns_text_file_result(self):
        segments = [SimpleNamespace(text="Hello world.")]

        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "example.m4a"
            audio_path.touch()

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

        self.assertIn(("status", "Generating text file..."), events)
        self.assertIn(("result", {"text": "Example Video [abc123].txt"}), events)

    def test_video_text_downloads_video_once_and_extracts_temp_audio(self):
        segments = [SimpleNamespace(text="Hello from video.")]

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
                patch.object(ripper, "_run_transcription_with_keepalive", return_value=[("_done", segments)]),
            ):
                events = list(ripper.process("https://youtu.be/abc123", "video_text"))

        video_download.assert_called_once_with("https://youtu.be/abc123", "1080p")
        audio_download.assert_not_called()
        extract_audio.assert_called_once_with(video_path)
        self.assertFalse(temp_audio_path.exists())
        self.assertIn(("result", {"video": "Example Video [abc123].mp4", "text": "Example Video [abc123].txt"}), events)


if __name__ == "__main__":
    unittest.main()
