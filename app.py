"""Flask server for YouTube Ripper with SSE streaming."""

import json
import re
from pathlib import Path

from flask import Flask, Response, abort, render_template, request, send_from_directory

import ripper

app = Flask(__name__)

# Simple URL validation: must look like a YouTube URL
YT_PATTERN = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/live/)[\w\-]+"
)


@app.route("/")
def index():
    return render_template("index.html")


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
    if not data:
        return Response("Missing JSON body", status=400)

    url = data.get("url", "").strip()
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

    def generate():
        for msg_type, msg_data in ripper.process(url, mode, quality):
            if msg_type.startswith("_"):
                continue
            payload = json.dumps({"type": msg_type, "data": msg_data})
            yield f"data: {payload}\n\n"

    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


if __name__ == "__main__":
    import os

    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("YOUTUBE_RIPPER_HOST", "0.0.0.0")
    port = int(os.environ.get("YOUTUBE_RIPPER_PORT", "4039"))
    app.run(host=host, port=port, debug=debug)
