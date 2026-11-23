import queue
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ripper


class DownloadValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_dir = Path(self.temp_dir.name)
        self.info = ripper.MediaInfo(title="Example", video_id="abc123")
        self.output_patch = patch.object(ripper, "OUTPUT_DIR", self.output_dir)
        self.output_patch.start()
        self.addCleanup(self.output_patch.stop)
        self.info_patch = patch.object(ripper, "_fetch_media_info", return_value=self.info)
        self.info_patch.start()
        self.addCleanup(self.info_patch.stop)

    def run_download(self, function, result, hook_updates=()):
        def execute(opts, action):
            class FakeYoutubeDL:
                def extract_info(self, url, download):
                    self.assert_download = download
                    for hook_name, update in hook_updates:
                        for hook in opts[hook_name]:
                            hook(update)
                    return result
            return action(FakeYoutubeDL())
        with patch.object(ripper, "_run_ytdlp_with_cookie_fallback", side_effect=execute):
            return function("https://youtu.be/abc123")

    def test_missing_output_fails_in_both_downloaders(self):
        for function in (ripper.download_video, ripper.download_audio):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(RuntimeError, "nonempty media file"):
                    self.run_download(function, {"filepath": str(self.output_dir / "missing.mp4")})

    def test_empty_output_fails_in_both_downloaders(self):
        path = self.output_dir / "empty.mp4"
        path.touch()
        for function in (ripper.download_video, ripper.download_audio):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(RuntimeError, "nonempty media file"):
                    self.run_download(function, {"filepath": str(path)})

    def test_stale_unreported_files_are_not_used(self):
        for extension in ("mp4", "m4a", "opus"):
            (self.output_dir / f"Example [abc123].{extension}").write_bytes(b"old content")
        for function in (ripper.download_video, ripper.download_audio):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(RuntimeError, "no completed media file"):
                    self.run_download(function, {})

    def test_video_uses_actual_fallback_extension(self):
        path = self.output_dir / "Example [abc123].webm"
        path.write_bytes(b"downloaded video")
        media = self.run_download(ripper.download_video, {
            "requested_downloads": [{"filepath": str(path)}],
        })
        self.assertEqual(media.path, path)
        self.assertEqual(media.info, self.info)

    def test_converted_audio_uses_final_download_metadata(self):
        source = self.output_dir / "Example [abc123].webm"
        path = self.output_dir / "Example [abc123].m4a"
        source.write_bytes(b"source")
        path.write_bytes(b"converted audio")
        media = self.run_download(ripper.download_audio, {
            "filepath": str(source), "requested_downloads": [{"filepath": str(path)}],
        })
        self.assertEqual(media.path, path)

    def test_finished_move_hook_supplies_final_path(self):
        path = self.output_dir / "Example [abc123].m4a"
        path.write_bytes(b"converted audio")
        media = self.run_download(ripper.download_audio, {}, [("postprocessor_hooks", {
            "status": "finished", "postprocessor": "MoveFiles", "info_dict": {"filepath": str(path)},
        })])
        self.assertEqual(media.path, path)

    def test_progress_hook_source_file_does_not_count_as_completed_output(self):
        path = self.output_dir / "Example [abc123].f137.mp4"
        path.write_bytes(b"video without merged audio")
        with self.assertRaisesRegex(RuntimeError, "no completed media file"):
            self.run_download(ripper.download_video, {}, [("progress_hooks", {
                "status": "finished", "filename": str(path),
            })])

    def test_audio_generator_emits_progress_before_completed_media(self):
        path = self.output_dir / "Example [abc123].m4a"
        path.write_bytes(b"audio")
        media = ripper.DownloadedMedia(path, self.info)

        def download(url, progress_queue):
            progress_queue.put({"status": "downloading", "percent": 25, "speed": "1MB/s", "eta": "0:03"})
            progress_queue.put({"status": "processing"})
            return media

        with patch.object(ripper, "download_audio", side_effect=download):
            events = list(ripper._run_audio_download_with_progress("https://youtu.be/abc123"))
        self.assertIn(("status", "Downloading audio... 25% (1MB/s, ETA 0:03)"), events)
        self.assertIn(("status", "Processing audio with ffmpeg..."), events)
        self.assertEqual(events[-1], ("_done", media))

    def test_percent_resets_between_streams_and_cookie_retry(self):
        path = self.output_dir / "Example [abc123].mp4"
        path.write_bytes(b"video")
        media = ripper.DownloadedMedia(path, self.info)

        def download(url, quality, progress_queue):
            for phase, percent in (("video", 98), ("audio", 1), ("audio", 5)):
                progress_queue.put({"status": "downloading", "phase": phase, "percent": percent,
                                    "speed": "?", "eta": "?"})
            progress_queue.put({"status": "starting"})
            progress_queue.put({"status": "downloading", "phase": "audio", "percent": 0,
                                "speed": "?", "eta": "?"})
            return media

        with patch.object(ripper, "download_video", side_effect=download):
            events = list(ripper._run_video_download_with_progress("url", "1080p"))
        for percent in (98, 1, 5, 0):
            self.assertIn(("status", f"Downloading video... {percent}%"), events)

    def test_audio_hooks_report_unknown_total_without_fake_zero_percent(self):
        path = self.output_dir / "audio.m4a"
        path.write_bytes(b"audio")
        progress_queue = queue.Queue()
        media = self.run_download(
            lambda url: ripper.download_audio(url, progress_queue),
            {"filepath": str(path)},
            [("progress_hooks", {"status": "downloading", "downloaded_bytes": 123})],
        )
        self.assertEqual(media.path, path)
        updates = [progress_queue.get_nowait() for _ in range(progress_queue.qsize())]
        self.assertIsNone(updates[-1]["percent"])

    def test_idle_download_sends_keepalive(self):
        path = self.output_dir / "audio.m4a"
        path.write_bytes(b"audio")
        media = ripper.DownloadedMedia(path, self.info)
        original_queue = queue.Queue

        class IdleOnceQueue(original_queue):
            idle = True

            def get(self, *args, **kwargs):
                if self.idle:
                    self.idle = False
                    raise queue.Empty
                return super().get(*args, **kwargs)

        with (
            patch.object(ripper.queue, "Queue", IdleOnceQueue),
            patch.object(ripper, "download_audio", return_value=media),
        ):
            events = list(ripper._run_audio_download_with_progress("url"))
        self.assertIn(("status", "Still downloading audio..."), events)
        self.assertEqual(events[-1], ("_done", media))

    def test_idle_merge_sends_merge_keepalive(self):
        path = self.output_dir / "video.mp4"
        path.write_bytes(b"video")
        media = ripper.DownloadedMedia(path, self.info)
        original_queue = queue.Queue

        class IdleAfterMergeQueue(original_queue):
            idle_next = False

            def get(self, *args, **kwargs):
                if self.idle_next:
                    self.idle_next = False
                    raise queue.Empty
                update = super().get(*args, **kwargs)
                if update and update.get("status") == "merging":
                    self.idle_next = True
                return update

        def download(url, quality, progress_queue):
            progress_queue.put({"status": "merging"})
            return media

        with (
            patch.object(ripper.queue, "Queue", IdleAfterMergeQueue),
            patch.object(ripper, "download_video", side_effect=download),
        ):
            events = list(ripper._run_video_download_with_progress("url", "1080p"))
        self.assertIn(("status", "Still merging video and audio streams..."), events)
        self.assertEqual(events[-1], ("_done", media))

    def test_downloader_exception_returns_error_without_success(self):
        with patch.object(ripper, "download_audio", side_effect=RuntimeError("ffmpeg is missing")):
            events = list(ripper._run_audio_download_with_progress("url"))
        self.assertEqual(events, [("error", "Audio download failed: ffmpeg is missing")])

    def test_generator_reports_invalid_output_without_done_event(self):
        media = ripper.DownloadedMedia(self.output_dir / "missing.mp4", self.info)
        with patch.object(ripper, "download_video", return_value=media):
            events = list(ripper._run_video_download_with_progress("url", "720p"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "error")
        self.assertIn("nonempty media file", events[0][1])


if __name__ == "__main__":
    unittest.main()
