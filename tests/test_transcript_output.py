import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ripper


class TranscriptOutputTests(unittest.TestCase):
    def test_text_mode_returns_text_file_result(self):
        segments = [SimpleNamespace(text="Hello world.")]

        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "example.m4a"
            audio_path.touch()

            with (
                patch.object(ripper, "OUTPUT_DIR", Path(temp_dir)),
                patch.object(ripper, "download_audio", return_value=(audio_path, "Example Video")),
                patch.object(ripper, "_run_transcription_with_keepalive", return_value=[("_done", segments)]),
            ):
                events = list(ripper.process("https://youtu.be/abc123", "text"))

        self.assertIn(("status", "Generating text file..."), events)
        self.assertIn(("result", {"text": "Example Video.txt"}), events)


if __name__ == "__main__":
    unittest.main()
