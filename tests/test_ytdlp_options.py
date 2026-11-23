import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from yt_dlp.utils import DownloadError

import ripper


class YtDlpOptionsTests(unittest.TestCase):
    def test_prefers_current_homebrew_node_for_youtube_challenges(self):
        with TemporaryDirectory() as temp_dir:
            node_path = Path(temp_dir) / "node"
            node_path.touch()
            node_path.chmod(0o755)

            with (
                patch.object(ripper, "HOMEBREW_NODE_PATH", node_path, create=True),
                patch.dict(os.environ, {}, clear=True),
            ):
                opts = ripper._yt_dlp_opts()

        self.assertEqual(opts["js_runtimes"], {"node": {"path": str(node_path)}})

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

    def test_can_disable_cookies_for_retry(self):
        with patch.dict(os.environ, {"YOUTUBE_RIPPER_COOKIES_FROM_BROWSER": "firefox:Default"}, clear=True):
            opts = ripper._yt_dlp_opts(use_cookies=False)

        self.assertNotIn("cookiesfrombrowser", opts)
        self.assertNotIn("cookiefile", opts)

    def test_fetch_media_info_retries_without_cookies_after_cookie_format_failure(self):
        seen_opts = []

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
                return {"title": "Example Video", "id": "abc123"}

        with (
            patch.dict(os.environ, {"YOUTUBE_RIPPER_COOKIES_FROM_BROWSER": "chrome:Profile 1"}, clear=True),
            patch.object(ripper.yt_dlp, "YoutubeDL", FakeYoutubeDL),
        ):
            info = ripper._fetch_media_info("https://youtu.be/abc123")

        self.assertEqual(info, ripper.MediaInfo(title="Example Video", video_id="abc123"))
        self.assertIn("cookiesfrombrowser", seen_opts[0])
        self.assertNotIn("cookiesfrombrowser", seen_opts[1])
        self.assertNotIn("cookiefile", seen_opts[1])

    def test_configured_output_dir_uses_environment(self):
        with patch.dict(os.environ, {"YOUTUBE_RIPPER_OUTPUT_DIR": "/tmp/ripper-output"}, clear=True):
            self.assertEqual(ripper._configured_output_dir(), Path("/tmp/ripper-output"))


if __name__ == "__main__":
    unittest.main()
