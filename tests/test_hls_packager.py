"""Unit tests for the ffmpeg HLS packager (no real ffmpeg is spawned)."""
import os

import pytest

from app.api import hls_packager


class _FakeProc:
    """Stands in for subprocess.Popen."""

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.alive = True
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.killed = True
        self.alive = False

    def wait(self, timeout=None):
        self.alive = False
        return 0


@pytest.fixture
def fake_spawn(monkeypatch):
    spawned: list[_FakeProc] = []

    def _spawn(cmd, **kwargs):
        p = _FakeProc(cmd, **kwargs)
        spawned.append(p)
        return p

    monkeypatch.setattr(hls_packager, "_spawn", _spawn)
    yield spawned
    # clean up any packagers a test left running
    for name in list(hls_packager._packagers):
        hls_packager.stop(name)


def test_ffmpeg_cmd_shape():
    cmd = hls_packager.build_ffmpeg_cmd("cam_front", "/tmp/x")
    assert cmd[0] == "ffmpeg"
    assert "rtsp://jarvis-go2rtc:8554/cam_front" in cmd          # go2rtc RTSP re-serve
    assert cmd[cmd.index("-c:a") + 1] == "aac"                   # Opus -> AAC
    assert cmd[cmd.index("-f") + 1] == "hls"
    assert "delete_segments" in cmd[cmd.index("-hls_flags") + 1]  # rolling window
    assert cmd[-1] == "/tmp/x/stream.m3u8"


def test_start_creates_dir_and_spawns(fake_spawn):
    directory = hls_packager.start("cam_a", watchdog=False)
    assert os.path.isdir(directory)
    assert hls_packager.directory_for("cam_a") == directory
    assert len(fake_spawn) == 1


def test_start_is_idempotent_while_alive(fake_spawn):
    d1 = hls_packager.start("cam_a", watchdog=False)
    d2 = hls_packager.start("cam_a", watchdog=False)
    assert d1 == d2
    assert len(fake_spawn) == 1  # reused, not respawned


def test_stop_kills_and_removes_dir(fake_spawn):
    directory = hls_packager.start("cam_a", watchdog=False)
    assert hls_packager.stop("cam_a") is True
    assert hls_packager.directory_for("cam_a") is None
    assert not os.path.exists(directory)
    assert fake_spawn[0].terminated is True


def test_stop_unknown_stream_is_false(fake_spawn):
    assert hls_packager.stop("nope") is False


def test_start_replaces_dead_process(fake_spawn):
    hls_packager.start("cam_a", watchdog=False)
    fake_spawn[0].alive = False  # ffmpeg died
    d2 = hls_packager.start("cam_a", watchdog=False)
    assert len(fake_spawn) == 2  # respawned
    assert os.path.isdir(d2)


def test_wait_for_video_needs_video_sized_segment(fake_spawn):
    directory = hls_packager.start("cam_a", watchdog=False)
    assert hls_packager.wait_for_video("cam_a", timeout=0.4) is False  # no segments
    # audio-only sized segment does NOT count as video
    with open(os.path.join(directory, "seg_0.ts"), "wb") as f:
        f.write(b"\x00" * 40_000)
    assert hls_packager.wait_for_video("cam_a", timeout=0.4) is False
    # a video-sized segment does
    with open(os.path.join(directory, "seg_1.ts"), "wb") as f:
        f.write(b"\x00" * 300_000)
    assert hls_packager.wait_for_video("cam_a", timeout=1.0) is True


def test_newest_video_bytes(tmp_path):
    assert hls_packager._newest_video_bytes(str(tmp_path)) is None  # nothing yet
    (tmp_path / "seg_0.ts").write_bytes(b"\x00" * 40_000)
    assert hls_packager._newest_video_bytes(str(tmp_path)) == 40_000  # recent audio-only
    (tmp_path / "seg_1.ts").write_bytes(b"\x00" * 400_000)
    assert hls_packager._newest_video_bytes(str(tmp_path)) == 400_000  # recent video
