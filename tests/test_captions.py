import io
import json
import unittest

import captions


VIDEO_URL = "https://youtu.be/example"


def track(language="en", *, translated=None, headers=None):
    url = f"https://www.youtube.com/api/timedtext?lang={language}&fmt=json3"
    if translated:
        url += f"&tlang={translated}"
    return {"ext": "json3", "url": url, "http_headers": headers or {}}


def cue(text="Hello world", start=1000, duration=2000, **extra):
    return {"tStartMs": start, "dDurationMs": duration, "segs": [{"utf8": text}], **extra}


def payload(*events):
    return json.dumps({"events": list(events)}).encode()


class FakeYtdlp:
    def __init__(self, info, bodies=None):
        self.info = info
        self.bodies = bodies or {}
        self.requests = []
        self.extractions = []
        self.read_sizes = []
        self.closed_responses = []
        self.options = None

    def run(self, options, action):
        self.options = options
        return action(self)

    def extract_info(self, url, download):
        self.extractions.append((url, download))
        return self.info

    def urlopen(self, request):
        self.requests.append(request)
        body = self.bodies.get(request.url, payload(cue()))
        if isinstance(body, Exception):
            raise body
        owner = self

        class Response(io.BytesIO):
            def read(self, size=-1):
                owner.read_sizes.append(size)
                return super().read(size)

            def close(self):
                owner.closed_responses.append(True)
                super().close()

        return Response(body)


class CaptionFetchTests(unittest.TestCase):
    def fetch(self, info, language="en", bodies=None):
        fake = FakeYtdlp(info, bodies)
        return captions.fetch_captions(VIDEO_URL, language, fake.run), fake

    def test_uploaded_preferred_and_metadata_preserved_without_media_download(self):
        uploaded, automatic = track(), track("en-auto")
        info = {"id": "example", "title": "Example", "subtitles": {"en": [uploaded]}, "automatic_captions": {"en": [automatic]}}
        result, fake = self.fetch(info)
        self.assertIs(result[0], info)
        self.assertEqual(result[1].source, "YouTube uploaded captions")
        self.assertEqual(result[1].language, "en")
        self.assertEqual(fake.requests[0].url, uploaded["url"])
        self.assertEqual(fake.extractions, [(VIDEO_URL, False)])
        self.assertTrue(fake.options["skip_download"])
        self.assertTrue(fake.options["noplaylist"])
        self.assertEqual(fake.options["socket_timeout"], captions.CAPTION_TIMEOUT)
        self.assertEqual(fake.read_sizes, [captions.MAX_CAPTION_BYTES + 1])
        self.assertTrue(fake.closed_responses)

    def test_automatic_used_when_uploaded_language_missing(self):
        result, _ = self.fetch({"subtitles": {"fr": [track("fr")]}, "automatic_captions": {"en": [track()]}})
        self.assertEqual(result[1].source, "YouTube automatic captions")

    def test_explicit_language_missing_does_not_choose_another_language(self):
        result, fake = self.fetch({"subtitles": {"fr": [track("fr")]}})
        self.assertIsNone(result)
        self.assertFalse(fake.requests)

    def test_language_case_normalization_and_regional_matching(self):
        result, _ = self.fetch({"subtitles": {"en-US": [track("en-US")]}}, "EN")
        self.assertEqual(result[1].language, "en-us")
        result, _ = self.fetch({"subtitles": {"en-US": [track("en-US")]}}, "en-GB")
        self.assertIsNone(result)

    def test_exact_language_preferred_to_regional_variant(self):
        result, fake = self.fetch({"subtitles": {"en-US": [track("en-US")], "en": [track()]}})
        self.assertEqual(result[1].language, "en")
        self.assertEqual(fake.requests[0].url, track()["url"])

    def test_auto_language_uses_original_and_prefers_matching_uploaded(self):
        info = {
            "subtitles": {"en": [track()], "fr": [track("fr")]},
            "automatic_captions": {"de": [track("fr", translated="de")], "fr-orig": [track("fr-auto")]},
        }
        result, fake = self.fetch(info, None)
        self.assertEqual(result[1].source, "YouTube uploaded captions")
        self.assertEqual(result[1].language, "fr")
        self.assertEqual(fake.requests[0].url, track("fr")["url"])

    def test_auto_language_never_picks_arbitrary_translation(self):
        result, fake = self.fetch({"automatic_captions": {"en": [track("fr", translated="en")]}}, None)
        self.assertIsNone(result)
        self.assertFalse(fake.requests)

    def test_auto_language_reads_source_from_original_audio_format(self):
        info = {"formats": [{"format_note": "French (original)", "language": "fr"}], "subtitles": {"en": [track()], "fr": [track("fr")]}}
        result, _ = self.fetch(info, None)
        self.assertEqual(result[1].language, "fr")

    def test_auto_language_uses_metadata_and_refuses_unrelated_uploaded(self):
        result, _ = self.fetch({"language": "es", "subtitles": {"en": [track()]}}, None)
        self.assertIsNone(result)

    def test_explicit_language_does_not_use_automatic_translation(self):
        result, fake = self.fetch({"automatic_captions": {"en": [track("fr", translated="en")]}})
        self.assertIsNone(result)
        self.assertFalse(fake.requests)

    def test_none_empty_and_unsupported_tracks_fall_back(self):
        for info in (None, {}, {"subtitles": {}}, {"subtitles": {"en": [{"ext": "vtt", "url": "https://example.com/a.vtt"}]}}):
            with self.subTest(info=info):
                result, fake = self.fetch(info)
                self.assertIsNone(result)
                self.assertFalse(fake.requests)

    def test_malformed_empty_and_failed_caption_requests_fall_back(self):
        for body in (b"not json", b"[]", b"{}", b"null", payload(), payload(cue("  \n")), OSError("expired")):
            with self.subTest(body=body):
                result, _ = self.fetch({"subtitles": {"en": [track()]}}, bodies={track()["url"]: body})
                self.assertIsNone(result)

    def test_bad_uploaded_track_tries_automatic(self):
        result, fake = self.fetch(
            {"subtitles": {"en": [track()]}, "automatic_captions": {"en": [track("en-auto")]}},
            bodies={track()["url"]: b"bad"},
        )
        self.assertEqual(len(fake.requests), 2)
        self.assertEqual(result[1].source, "YouTube automatic captions")

    def test_headers_merge_with_track_overrides(self):
        info = {"http_headers": {"Referer": VIDEO_URL, "User-Agent": "common"}, "subtitles": {"en": [track(headers={"User-Agent": "specific"})]}}
        result, fake = self.fetch(info)
        self.assertIsNotNone(result)
        self.assertEqual(fake.requests[0].headers["Referer"], VIDEO_URL)
        self.assertEqual(fake.requests[0].headers["User-Agent"], "specific")

    def test_oversized_caption_track_falls_back_and_read_is_bounded(self):
        result, fake = self.fetch({"subtitles": {"en": [track()]}}, bodies={track()["url"]: b" " * (captions.MAX_CAPTION_BYTES + 2)})
        self.assertIsNone(result)
        self.assertEqual(fake.read_sizes, [captions.MAX_CAPTION_BYTES + 1])

    def test_extraction_failure_falls_back(self):
        def failed(options, action):
            raise RuntimeError("video unavailable")

        self.assertIsNone(captions.fetch_captions(VIDEO_URL, "en", failed))


class Json3Tests(unittest.TestCase):
    def test_timestamps_and_multiple_text_parts(self):
        event = cue(start=1250, duration=2300)
        event["segs"] = [{"utf8": "Hello"}, {"utf8": "\nworld"}]
        result = captions._parse_json3(payload(event))
        self.assertEqual([(s.start, s.end, s.text) for s in result], [(1.25, 3.55, "Hello world")])

    def test_rolling_snapshot_prefix_updates_and_suffix_overlap(self):
        result = captions._parse_json3(payload(
            cue("Hello there", 0, 3000, wWinId=1),
            cue("Hello there friend", 1000, 3000, wWinId=1),
            cue("there friend welcome", 2000, 3000, wWinId=1),
        ))
        self.assertEqual([(s.start, s.end, s.text) for s in result], [(0, 2, "Hello there friend"), (2, 5, "welcome")])

    def test_append_preserves_real_repetition(self):
        result = captions._parse_json3(payload(cue("No", 0, 3000, wWinId=1), cue(" no", 1000, 2000, wWinId=1, aAppend=1)))
        self.assertEqual([s.text for s in result], ["No no"])

    def test_append_punctuation_has_no_extra_space(self):
        result = captions._parse_json3(payload(cue("Hello", 0, 3000, wWinId=1), cue(", world", 1000, 2000, wWinId=1, aAppend=1)))
        self.assertEqual([s.text for s in result], ["Hello, world"])

    def test_identical_spoken_cues_preserved_after_window_expires(self):
        result = captions._parse_json3(payload(cue("Go go", 0, 1000, wWinId=1), cue("Go go", 2000, 1000, wWinId=1)))
        self.assertEqual([s.text for s in result], ["Go go", "Go go"])

    def test_uploaded_repetition_without_window_is_preserved(self):
        result = captions._parse_json3(payload(cue("Go", 0, 3000), cue("Go", 1000, 3000)))
        self.assertEqual([s.text for s in result], ["Go", "Go"])

    def test_separate_windows_are_not_deduplicated(self):
        result = captions._parse_json3(payload(cue("Hello", 0, 3000, wWinId=1), cue("Hello", 1000, 3000, wWinId=2)))
        self.assertEqual([s.text for s in result], ["Hello", "Hello"])

    def test_invalid_times_and_structure_are_ignored(self):
        result = captions._parse_json3(payload(None, {}, cue(start=-1), cue(duration=0), cue(start=True), cue(duration=float("inf")), cue(start="1000"), cue()))
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
