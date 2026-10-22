import os
import re
import shutil
import subprocess
from pathlib import Path

from . import config


def media_path(video_id: str) -> Path:
    p = Path(config.MEDIA_DIR) / video_id
    if not p.is_file() or p.suffix.lower() not in config.VIDEO_EXTS or "/" in video_id or video_id.startswith("."):
        raise FileNotFoundError(video_id)
    return p


def list_media() -> list[str]:
    root = Path(config.MEDIA_DIR)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in config.VIDEO_EXTS)


def duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def extract_frames(path: Path, fps: float, out_dir: Path, on_progress=None) -> list[Path]:
    """한 번의 ffmpeg 패스로 fps 간격 프레임 추출. i번째(0-base) 프레임 시각 ≈ i / fps.

    on_progress(sec): ffmpeg -progress 의 out_time 을 초 단위로 전달 (진행률 표시용).
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-nostats", "-progress", "pipe:1", "-i", str(path),
           "-vf", f"fps={fps},scale=384:-2", "-q:v", "3", str(out_dir / "%06d.jpg")]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        # out_time_us / out_time_ms 둘 다 마이크로초 (ffmpeg 의 오래된 명명 버그)
        if on_progress and (line.startswith("out_time_us=") or line.startswith("out_time_ms=")):
            try:
                on_progress(int(line.split("=", 1)[1]) / 1e6)
            except ValueError:
                pass
    err = proc.stderr.read() if proc.stderr else ""
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=err)
    return sorted(out_dir.glob("*.jpg"))


# ---------- 자막 사이드카 (.srt / .vtt) ----------
_TS = re.compile(r"(\d+):(\d+):(\d+)[.,](\d+)")


def _ts(s: str) -> float:
    m = _TS.search(s)
    if not m:
        return 0.0
    h, mi, se, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")[:3]) / 1000


def find_sidecar(path: Path) -> Path | None:
    for ext in (".srt", ".vtt", ".ko.srt", ".ko.vtt"):
        p = path.with_suffix(ext)
        if p.is_file():
            return p
    return None


def parse_sidecar(path: Path) -> list[tuple[float, float, str]]:
    segs: list[tuple[float, float, str]] = []
    block: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines() + [""]:
        line = raw.strip()
        if line:
            block.append(line)
            continue
        if not block:
            continue
        for i, l in enumerate(block):
            if "-->" in l:
                a, b = l.split("-->", 1)
                text = " ".join(block[i + 1 :]).strip()
                text = re.sub(r"<[^>]+>", "", text)
                if text:
                    segs.append((_ts(a), _ts(b), text))
                break
        block = []
    return segs
