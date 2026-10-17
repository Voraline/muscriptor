import random
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

YoutubeHosts = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
)

MaxDurationS = 15 * 60

AnsiPattern = re.compile(r"\x1b\[[0-9;]*m")


class YoutubeError(Exception):
    def __init__(self, Message: str, Status: int = 400):
        super().__init__(Message)
        self.Status = Status


def ValidateYoutubeUrl(Url: str) -> str:
    Url = Url.strip()
    if not Url:
        raise YoutubeError("Paste a YouTube link first.")
    if "://" not in Url:
        Url = "https://" + Url
    Parsed = urlparse(Url)
    if Parsed.scheme not in ("http", "https") or Parsed.hostname not in YoutubeHosts:
        raise YoutubeError("That doesn't look like a YouTube link.")
    return Url


def GetRandomUserAgent() -> str:
    AppVersions = [
        "19.09.37",
        "19.08.36",
        "19.07.39",
        "19.06.41",
        "19.05.34",
        "19.04.38",
    ]
    BuildIds = [
        "AP31.240517.022",
        "AP31.240617.007",
        "AP31.240717.014",
        "BP1A.250405.005",
        "BP2A.250605.031",
    ]
    DeviceModels = [
        "Pixel 6",
        "Pixel 7",
        "Pixel 8",
        "Pixel 9",
        "SM-S928B",
        "SM-G998B",
    ]
    return (
        f"com.google.android.youtube/{random.choice(AppVersions)} "
        f"(Linux; U; Android 15; {random.choice(DeviceModels)} "
        f"Build/{random.choice(BuildIds)}) gzip"
    )


def CleanMessage(Message: str) -> str:
    Message = AnsiPattern.sub("", Message).strip()
    return Message.removeprefix("ERROR:").strip()


def DownloadAudio(
    Url: str, OutDir: Path, MaxDuration: float = MaxDurationS
) -> tuple[Path, str]:
    import yt_dlp
    from yt_dlp.utils import DownloadError, PostProcessingError

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise YoutubeError(
            "Fetching YouTube audio needs ffmpeg installed on the server "
            "(https://ffmpeg.org/download.html).",
            Status=503,
        )

    YdlOptions = {
        "format": "bestaudio/best",
        "outtmpl": str(OutDir / "audio.%(ext)s"),
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
        "http_headers": {"User-Agent": GetRandomUserAgent()},
        "quiet": True,
        "no_warnings": True,
        "no_color": True,
        "noprogress": True,
    }

    try:
        with yt_dlp.YoutubeDL(YdlOptions) as Ydl:
            Info = Ydl.extract_info(Url, download=False)
            if Info is None:
                raise YoutubeError("Could not read that video.", Status=502)
            if Info.get("_type") == "playlist" or "entries" in Info:
                raise YoutubeError("That is a playlist, paste a single video link.")
            if Info.get("is_live"):
                raise YoutubeError("Live streams can't be transcribed.")
            Duration = Info.get("duration")
            if Duration is not None and Duration > MaxDuration:
                raise YoutubeError(
                    f"That video is {Duration / 60:.1f} minutes long; "
                    f"the limit is {MaxDuration / 60:.0f} minutes.",
                    Status=413,
                )
            Ydl.process_ie_result(Info, download=True)
    except YoutubeError:
        raise
    except (DownloadError, PostProcessingError) as Error:
        raise YoutubeError(
            f"Could not fetch audio from YouTube: {CleanMessage(str(Error))}",
            Status=502,
        ) from Error

    AudioPath = OutDir / "audio.mp3"
    if not AudioPath.is_file():
        raise YoutubeError("YouTube download produced no audio.", Status=502)
    return AudioPath, str(Info.get("title") or "youtube")
