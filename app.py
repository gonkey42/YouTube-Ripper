"""Flask server for YouTube Ripper with SSE streaming."""

import json
import os
import re
from ipaddress import ip_address, ip_network
from pathlib import Path

from flask import Flask, Response, abort, render_template, request, send_from_directory

from diagnostics import dependency_issues
from media_types import TranscriptionOptions
import ripper

app = Flask(__name__)
DEFAULT_ALLOWED_CLIENTS = "127.0.0.0/8,::1/128,100.64.0.0/10,fd7a:115c:a1e0::/48"
LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
    "ru": "Russian", "uk": "Ukrainian", "ar": "Arabic", "hi": "Hindi",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
}

# Simple URL validation: must look like a YouTube URL
YT_PATTERN = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/live/)[\w\-]+"
)


def _allowed_client_networks():
    configured = os.environ.get("YOUTUBE_RIPPER_ALLOWED_CLIENTS", DEFAULT_ALLOWED_CLIENTS)
    networks = []
    for item in configured.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            networks.append(ip_network(item, strict=False))
        except ValueError:
            continue
    return tuple(networks)


def _remote_addr_allowed(remote_addr: str | None) -> bool:
    if not remote_addr:
        return False

    try:
        addr = ip_address(remote_addr)
    except ValueError:
        return False

    if getattr(addr, "ipv4_mapped", None):
        addr = addr.ipv4_mapped

    return any(addr in network for network in _allowed_client_networks())


@app.before_request
def restrict_clients():
    if not _remote_addr_allowed(request.remote_addr):
        abort(403)


@app.route("/")
def index():
    return render_template("index.html", dependency_issues=dependency_issues())


@app.route("/files/<path:filename>")
def files(filename):
    if Path(filename).name != filename:
        abort(404)

    file_path = ripper.OUTPUT_DIR / filename
    if not file_path.is_file():
        abort(404)

    return send_from_directory(ripper.OUTPUT_DIR, filename, as_attachment=True)


@app.route("/rip", methods=["POST"])
def rip():
    data = request.get_json()
    if not isinstance(data, dict) or not data:
        return Response("Missing JSON body", status=400)

    url = data.get("url", "")
    if not isinstance(url, str):
        return Response("URL must be text", status=400)
    url = url.strip()
    mode = data.get("mode", "both")

    if not url:
        return Response("Missing URL", status=400)

    if not YT_PATTERN.match(url):
        return Response("Invalid YouTube URL", status=400)

    if mode not in ("audio", "text", "both", "video", "video_text"):
        return Response("Invalid mode", status=400)

    quality = data.get("quality", "1080p")
    if quality not in ("480p", "720p", "1080p", "best"):
        quality = "1080p"

    settings = data.get("transcription", {})
    if not isinstance(settings, dict):
        return Response("Invalid transcription settings", status=400)
    language = settings.get("language", "en")
    source = settings.get("source", "local")
    formats = settings.get("subtitle_formats", [])
    if not isinstance(language, str) or language not in LANGUAGES:
        return Response("Invalid transcription language", status=400)
    if source not in ("local", "whisper", "captions"):
        return Response("Invalid transcript source", status=400)
    if not isinstance(formats, list) or any(fmt not in ("srt", "vtt") for fmt in formats):
        return Response("Invalid subtitle formats", status=400)
    options = TranscriptionOptions(
        language=None if language == "auto" else language,
        prefer_captions=source == "captions",
        subtitle_formats=tuple(dict.fromkeys(formats)),
    )
    issues = dependency_issues(require_ffmpeg=not (mode == "text" and options.prefer_captions))
    if issues:
        return Response("\n".join(issues), status=503, mimetype="text/plain")

    def generate():
        for msg_type, msg_data in ripper.process(url, mode, quality, options):
            if msg_type.startswith("_"):
                continue
            payload = json.dumps({"type": msg_type, "data": msg_data})
            yield f"data: {payload}\n\n"

    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


if __name__ == "__main__":
    for issue in dependency_issues():
        app.logger.warning(issue)
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("YOUTUBE_RIPPER_HOST", "0.0.0.0")
    port = int(os.environ.get("YOUTUBE_RIPPER_PORT", "4039"))
    app.run(host=host, port=port, debug=debug)
