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


if __name__ == "__main__":
    unittest.main()
