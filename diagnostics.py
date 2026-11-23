"""Fast local checks for the executables used by downloads."""

import os
from pathlib import Path
import re
import shutil
import subprocess


HOMEBREW_NODE_PATH = Path("/opt/homebrew/opt/node/bin/node")
CHECK_TIMEOUT_SECONDS = 2


def _version_output(executable: str, flag: str, label: str, remedy: str) -> tuple[str, str | None]:
    try:
        result = subprocess.run(
            [executable, flag],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CHECK_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", f"{label} did not respond within {CHECK_TIMEOUT_SECONDS} seconds. {remedy}"
    except OSError:
        return "", f"{label} could not run at {executable}. {remedy}"
    if result.returncode != 0:
        return "", f"{label} failed its version check at {executable} (exit {result.returncode}). {remedy}"
    return result.stdout.strip(), None


def dependency_issues(*, require_ffmpeg: bool = True) -> list[str]:
    """Return actionable local setup problems, without network or cookie access.

    Match ripper's runtime selection: prefer executable Homebrew Node, then PATH.
    Run checks afresh so installing a dependency fixes the next request immediately.
    """
    issues = []
    if require_ffmpeg:
        remedy = "Install or repair ffmpeg and put it on PATH; on macOS, run `brew install ffmpeg`."
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            issues.append(f"ffmpeg is required for audio extraction and merging video. {remedy}")
        else:
            output, issue = _version_output(ffmpeg, "-version", "ffmpeg", remedy)
            if issue:
                issues.append(issue)
            elif not output.startswith("ffmpeg version "):
                issues.append(f"ffmpeg returned an unrecognized version at {ffmpeg}. {remedy}")

    remedy = "Install or upgrade Node.js to version 22 or newer; on macOS, run `brew install node` or `brew upgrade node`."
    node = str(HOMEBREW_NODE_PATH) if os.access(HOMEBREW_NODE_PATH, os.X_OK) else shutil.which("node")
    if not node:
        issues.append(f"Node.js 22 or newer is required for YouTube downloads. {remedy}")
    else:
        output, issue = _version_output(node, "--version", "Node.js", remedy)
        if issue:
            issues.append(issue)
        else:
            version = re.fullmatch(r"v?(\d+)\.\d+\.\d+(?:[-+][\w.-]+)?", output)
            if not version:
                issues.append(f"Node.js returned an unrecognized version at {node}. {remedy}")
            elif int(version.group(1)) < 22:
                issues.append(f"Node.js {output} at {node} is too old for YouTube downloads. {remedy}")
    return issues
