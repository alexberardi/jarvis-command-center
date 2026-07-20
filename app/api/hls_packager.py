"""Standard-HLS packager for camera streams (one ffmpeg per active camera).

Why this exists
---------------
go2rtc's built-in HLS is a low-latency, *session-based* variant meant for its own
JS player: the media playlist never advances ``EXT-X-MEDIA-SEQUENCE``, segments
are evicted within a few seconds, and each fetch mints a new session id. Standard
HLS players (iOS AVPlayer, Android ExoPlayer, expo-av) — and a stateless proxy —
keep requesting segment ``n=0``, which go2rtc has evicted, so they ``404`` forever
and show black.

Instead, per active camera we run ONE ffmpeg that reads go2rtc's RTSP re-serve of
the stream (go2rtc exposes *any* registered stream over RTSP on :8554, even a
WebRTC-only Nest source) and writes **spec-compliant rolling HLS** into a temp dir
that command center serves as static files. The mobile URL contract is unchanged.

Two things make it robust against the flaky battery-Nest WebRTC source:

- **Pre-warm before ffmpeg.** The Nest WebRTC producer emits corrupt/no frames
  while establishing; starting ffmpeg against it makes it come up audio-only.
  A manager thread first polls go2rtc's frame endpoint until a real frame lands,
  then starts ffmpeg against the now-warm producer.
- **Exactly one ffmpeg, respawn only on death.** An earlier "respawn if no video"
  watchdog stampeded consumers and *destabilized* the source (PPS corruption). One
  steady consumer rides through the packet loss and produces watchable video.

Audio ``OPUS -> AAC`` is done here in ffmpeg (go2rtc's built-in Opus->AAC has a
recurring muted-track bug); video is lightly re-encoded with regular keyframes and
a compatible pixel format.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import quote, urlparse

logger = logging.getLogger("uvicorn")


class _ProcLike(Protocol):
    """The subset of subprocess.Popen the packager uses (fakeable in tests)."""

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


# go2rtc RTSP host is derived from the HTTP base URL (same container, port 8554).
GO2RTC_BASE_URL: str = os.getenv("GO2RTC_URL", "http://jarvis-go2rtc:1984")
_GO2RTC_RTSP_PORT: int = int(os.getenv("GO2RTC_RTSP_PORT", "8554"))

# Segment tuning. 2s segments, keep 6 in the window.
_HLS_SEGMENT_SECONDS: int = 2
_HLS_LIST_SIZE: int = 6

# A 2s video segment is ~200-500KB; an audio-only (AAC) 2s segment is ~35KB.
_VIDEO_MIN_BYTES: int = 120_000
_SEGMENT_STALE_SECONDS: float = 12.0

# Pre-warm: poll go2rtc frame.jpeg until the producer delivers a real frame.
_PREWARM_TIMEOUT_SECONDS: float = 45.0
_PREWARM_POLL_SECONDS: float = 2.0

# Respawn-on-death policy (never respawn faster than this — avoids storms).
_RESPAWN_BACKOFF_SECONDS: float = 3.0
_MONITOR_POLL_SECONDS: float = 3.0

# Injection point so tests never touch real ffmpeg.
_spawn = subprocess.Popen  # overridable in tests


@dataclass
class _Packager:
    stream_name: str
    directory: str
    process: _ProcLike | None = None  # None until the manager launches ffmpeg
    watchdog: bool = True
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None


_packagers: dict[str, _Packager] = {}
_lock = threading.RLock()


def _rtsp_url(stream_name: str) -> str:
    host = urlparse(GO2RTC_BASE_URL).hostname or "jarvis-go2rtc"
    return f"rtsp://{host}:{_GO2RTC_RTSP_PORT}/{stream_name}"


def build_ffmpeg_cmd(stream_name: str, directory: str) -> list[str]:
    """The ffmpeg invocation that turns a go2rtc stream into standard HLS."""
    gop = _HLS_SEGMENT_SECONDS * 30  # ~30fps; keyframe every segment
    return [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-rtsp_transport", "tcp", "-err_detect", "ignore_err",
        "-i", _rtsp_url(stream_name),
        # Video: re-encode with regular keyframes (Nest H264 lacks steady IDR
        # frames) and a broadly-compatible pixel format (source is full-range
        # yuvj420p, which some players render as black / wrong colors).
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        # Audio: Opus -> AAC for HLS/MPEG-TS + iOS compatibility.
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-f", "hls",
        "-hls_time", str(_HLS_SEGMENT_SECONDS),
        "-hls_list_size", str(_HLS_LIST_SIZE),
        "-hls_flags", "delete_segments+append_list+independent_segments+omit_endlist",
        "-hls_segment_type", "mpegts",
        "-hls_segment_filename", os.path.join(directory, "seg_%d.ts"),
        os.path.join(directory, "stream.m3u8"),
    ]


def _launch(stream_name: str, directory: str) -> _ProcLike:
    os.makedirs(directory, exist_ok=True)  # tolerate a removed dir
    logger.info("HLS packager: starting ffmpeg for stream=%s", stream_name)
    return _spawn(
        build_ffmpeg_cmd(stream_name, directory),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _prewarm(pkg: _Packager) -> bool:
    """Poll go2rtc's frame endpoint until the producer delivers a real frame."""
    url = f"{GO2RTC_BASE_URL}/api/frame.jpeg?src={quote(pkg.stream_name)}"
    deadline = time.monotonic() + _PREWARM_TIMEOUT_SECONDS
    while not pkg._stop.is_set() and time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                if resp.status == 200 and len(resp.read(4096)) > 2000:
                    return True
        except Exception:
            pass
        if pkg._stop.wait(_PREWARM_POLL_SECONDS):
            break
    return False


def _manage(pkg: _Packager) -> None:
    """Warm the producer, then run one ffmpeg, respawning only if it dies."""
    _prewarm(pkg)  # best-effort; ffmpeg will still try if it times out
    if pkg._stop.is_set():
        return
    pkg.process = _launch(pkg.stream_name, pkg.directory)
    while not pkg._stop.wait(_MONITOR_POLL_SECONDS):
        proc = pkg.process
        if proc is not None and proc.poll() is not None:
            logger.warning(
                "HLS packager: ffmpeg stream=%s exited (code=%s); respawning",
                pkg.stream_name, proc.poll(),
            )
            if pkg._stop.wait(_RESPAWN_BACKOFF_SECONDS):
                break
            pkg.process = _launch(pkg.stream_name, pkg.directory)


def start(stream_name: str, watchdog: bool = True) -> str:
    """Start (or reuse) the HLS packager for ``stream_name``; return its dir."""
    with _lock:
        existing = _packagers.get(stream_name)
        if existing is not None and (
            existing.process is None or existing.process.poll() is None
        ):
            return existing.directory
        if existing is not None:
            _teardown(existing)

        directory = tempfile.mkdtemp(prefix=f"hls-{_safe(stream_name)}-")
        pkg = _Packager(stream_name=stream_name, directory=directory, watchdog=watchdog)
        if watchdog:
            pkg._thread = threading.Thread(
                target=_manage, args=(pkg,), daemon=True,
                name=f"hls-{stream_name}",
            )
            pkg._thread.start()
        else:
            # Test / no-thread mode: launch ffmpeg directly (no pre-warm thread).
            pkg.process = _launch(stream_name, directory)
        _packagers[stream_name] = pkg
        return directory


def directory_for(stream_name: str) -> str | None:
    with _lock:
        pkg = _packagers.get(stream_name)
        return pkg.directory if pkg is not None else None


def stop(stream_name: str) -> bool:
    with _lock:
        pkg = _packagers.pop(stream_name, None)
    if pkg is None:
        return False
    _teardown(pkg)
    return True


def _teardown(pkg: _Packager) -> None:
    pkg._stop.set()
    proc = pkg.process
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        except Exception as e:
            logger.warning("HLS packager: error stopping ffmpeg stream=%s: %s", pkg.stream_name, e)
    shutil.rmtree(pkg.directory, ignore_errors=True)
    logger.info("HLS packager: stopped stream=%s", pkg.stream_name)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40]


def _newest_video_bytes(directory: str) -> int | None:
    """Size of the newest *recent* segment; 0 if stalled; None if none yet."""
    try:
        segs = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.startswith("seg_") and f.endswith(".ts")
        ]
    except OSError:
        return None
    if not segs:
        return None
    newest = max(segs, key=lambda p: os.path.getmtime(p))
    if time.time() - os.path.getmtime(newest) > _SEGMENT_STALE_SECONDS:
        return 0
    return os.path.getsize(newest)


def wait_for_video(stream_name: str, timeout: float = 40.0) -> bool:
    """Block until a *video-sized* segment appears (real video is flowing).

    Waiting for the playlist alone isn't enough: ffmpeg can emit audio-only
    segments during warm-up, so this waits for actual video before returning so
    the phone gets a picture on its first playlist fetch.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        directory = directory_for(stream_name)
        if directory is not None:
            size = _newest_video_bytes(directory)
            if size is not None and size >= _VIDEO_MIN_BYTES:
                return True
        time.sleep(0.5)
    return False
