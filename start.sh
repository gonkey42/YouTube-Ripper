#!/bin/bash
# Start YouTube Ripper Flask server
# Called by launchd plist

cd /Users/hal9000/claudebot/tools/youtube-ripper

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

exec /Users/hal9000/.local/bin/uv run python app.py
