"""Fetch the audio track of a YouTube video with yt-dlp.

Used by the server's `POST /youtube`, so the web UI can take a link instead of
an uploaded file. Needs `ffmpeg` and `ffprobe` on the PATH to convert the
downloaded stream to MP3.
"""

import random
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

# Only YouTube is accepted. yt-dlp can fetch from thousands of sites (and from
# arbitrary URLs through its generic extractor), which would turn a public
# server into a proxy that can reach its own internal network.
YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
)

MAX_DURATION_S = 15 * 60  # Same cap as POST /transcribe/midi.

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class YoutubeError(Exception):
    """A failure whose message is safe to show the user.

    `status` is the HTTP status the server should answer with.
    """

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def validate_youtube_url(url: str) -> str:
    """Return `url` normalized to an absolute http(s) YouTube link.

    Accepts links pasted without a scheme (`youtu.be/abc`). Raises
    `YoutubeError` for anything that isn't a YouTube host.
    """
    url = url.strip()
    if not url:
        raise YoutubeError("Paste a YouTube link first.")
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    # `hostname` is what a client would connect to, so the classic
    # `https://youtube.com@evil.example/` trick resolves to evil.example here.
    if parsed.scheme not in ("http", "https") or parsed.hostname not in YOUTUBE_HOSTS:
        raise YoutubeError("That doesn't look like a YouTube link.")
    return url


def random_user_agent() -> str:
    """A plausible YouTube Android app User-Agent, varied per request."""
    app_versions = [
        "19.09.37",
        "19.08.36",
        "19.07.39",
        "19.06.41",
        "19.05.34",
        "19.04.38",
    ]
    build_ids = [
        "AP31.240517.022",
        "AP31.240617.007",
        "AP31.240717.014",
        "BP1A.250405.005",
        "BP2A.250605.031",
    ]
    device_models = [
        "Pixel 6",
        "Pixel 7",
        "Pixel 8",
        "Pixel 9",
        "SM-S928B",
        "SM-G998B",
    ]
    return (
        f"com.google.android.youtube/{random.choice(app_versions)} "
        f"(Linux; U; Android 15; {random.choice(device_models)} "
        f"Build/{random.choice(build_ids)}) gzip"
    )


def _clean(message: str) -> str:
    """yt-dlp error text without colour codes and the `ERROR: ` prefix."""
    message = _ANSI_RE.sub("", message).strip()
    return message.removeprefix("ERROR:").strip()


def download_audio(
    url: str, out_dir: Path, max_duration_s: float = MAX_DURATION_S
) -> tuple[Path, str]:
    """Download the audio of the YouTube video at `url` into `out_dir` as MP3.

    `url` must already have passed `validate_youtube_url`. Returns the file's
    path and the video's title. Raises `YoutubeError` on any failure.
    """
    import yt_dlp
    from yt_dlp.utils import DownloadError, PostProcessingError

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise YoutubeError(
            "Fetching YouTube audio needs ffmpeg installed on the server "
            "(https://ffmpeg.org/download.html).",
            status=503,
        )

    options = {
        "format": "bestaudio/best",
        # A fixed name: the title can hold characters a filesystem rejects,
        # and the caller gets it separately.
        "outtmpl": str(out_dir / "audio.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "0",
            }
        ],
        "noplaylist": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "tv_embedded"],
            }
        },
        "http_headers": {"User-Agent": random_user_agent()},
        "quiet": True,
        "no_warnings": True,
        "no_color": True,
        "noprogress": True,
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            # Look before downloading, so a two-hour video is turned away
            # without fetching it.
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise YoutubeError("Could not read that video.", status=502)
            if info.get("_type") == "playlist" or "entries" in info:
                raise YoutubeError("That is a playlist, paste a single video link.")
            if info.get("is_live"):
                raise YoutubeError("Live streams can't be transcribed.")
            duration = info.get("duration")
            if duration is not None and duration > max_duration_s:
                raise YoutubeError(
                    f"That video is {duration / 60:.1f} minutes long; "
                    f"the limit is {max_duration_s / 60:.0f} minutes.",
                    status=413,
                )
            ydl.process_ie_result(info, download=True)
    except YoutubeError:
        raise
    except (DownloadError, PostProcessingError) as e:
        raise YoutubeError(
            f"Could not fetch audio from YouTube: {_clean(str(e))}", status=502
        ) from e

    path = out_dir / "audio.mp3"
    if not path.is_file():
        raise YoutubeError("YouTube download produced no audio.", status=502)
    return path, str(info.get("title") or "youtube")
