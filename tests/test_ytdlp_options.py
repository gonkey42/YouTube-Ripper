import os
import unittest
from pathlib import Path
from unittest.mock import patch

import ripper


class YtDlpOptionsTests(unittest.TestCase):
    def test_uses_cookie_file_when_configured(self):
        with patch.dict(
            os.environ,
            {
                "YOUTUBE_RIPPER_COOKIES": "/tmp/youtube-cookies.txt",
                "YOUTUBE_RIPPER_COOKIES_FROM_BROWSER": "chrome:Profile 1",
            },
            clear=True,
        ):
            opts = ripper._yt_dlp_opts({"quiet": True})

        self.assertEqual(opts["cookiefile"], "/tmp/youtube-cookies.txt")
        self.assertNotIn("cookiesfrombrowser", opts)
        self.assertTrue(opts["quiet"])

    def test_uses_browser_cookies_when_no_cookie_file_is_configured(self):
        with patch.dict(
            os.environ,
            {"YOUTUBE_RIPPER_COOKIES_FROM_BROWSER": "firefox:Default"},
            clear=True,
        ):
            opts = ripper._yt_dlp_opts()

        self.assertEqual(opts["cookiesfrombrowser"], ("firefox", "Default", None, None))

    def test_defaults_to_local_chrome_profile_when_unconfigured(self):
        with patch.dict(os.environ, {}, clear=True):
            opts = ripper._yt_dlp_opts()

        self.assertEqual(opts["cookiesfrombrowser"], ("chrome", "Profile 1", None, None))

    def test_configured_output_dir_uses_environment(self):
        with patch.dict(os.environ, {"YOUTUBE_RIPPER_OUTPUT_DIR": "/tmp/ripper-output"}, clear=True):
            self.assertEqual(ripper._configured_output_dir(), Path("/tmp/ripper-output"))

    def test_whisper_settings_use_environment_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ripper._whisper_model_name(), "base")
            self.assertEqual(ripper._whisper_device(), "cpu")
            self.assertEqual(ripper._whisper_compute_type(), "int8")


if __name__ == "__main__":
    unittest.main()
