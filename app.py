"""Flask server for YouTube Ripper with SSE streaming."""

import json
import re

from flask import Flask, Response, render_template, request, send_from_directory

import ripper

app = Flask(__name__)

# Simple URL validation: must look like a YouTube URL
YT_PATTERN = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/live/)[\w\-]+"
)


@app.route("/")
def index():
    return render_template("index.html")


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
            payload = json.dumps({"type": msg_type, "data": msg_data})
            yield f"data: {payload}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory("output", filename, as_attachment=True)


if __name__ == "__main__":
    import os

    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=4039, debug=debug)
