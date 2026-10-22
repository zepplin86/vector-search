"""인덱싱 파이프라인. 시각/자막 분리. 순차 워커 스레드 하나로 처리 (CPU 공유)."""
import logging
import queue
import shutil
import threading
from pathlib import Path

import numpy as np

from . import config, db, media, models

log = logging.getLogger("ingest")

_jobs: "queue.Queue[tuple[str, str]]" = queue.Queue()


def enqueue(video_id: str, kind: str) -> None:
    db.set_status(video_id, kind, "queued")
    _jobs.put((video_id, kind))


def _worker() -> None:
    while True:
        video_id, kind = _jobs.get()
        try:
            db.set_status(video_id, kind, "running")
            log.info("ingest start: %s %s", kind, video_id)
            n = ingest_visual(video_id) if kind == "visual" else ingest_subtitle(video_id)
            db.set_status(video_id, kind, "done", count=n)
            log.info("ingest done: %s %s rows=%d", kind, video_id, n)
        except Exception as e:  # noqa: BLE001
            log.exception("ingest failed: %s %s", kind, video_id)
            db.set_status(video_id, kind, "error", error=f"{kind}: {e}")
        finally:
            _jobs.task_done()


def requeue_stale() -> int:
    """재시작으로 메모리 큐가 사라진 경우: DB 에 queued/running 으로 남은 작업을 다시 넣음. 짧은 영상 우선."""
    with db.pool.connection() as conn:
        rows = conn.execute(
            "SELECT id, visual_status, subtitle_status FROM videos "
            "WHERE visual_status IN ('queued','running') OR subtitle_status IN ('queued','running') "
            "ORDER BY duration NULLS LAST"
        ).fetchall()
    n = 0
    for vid, vs, ss in rows:
        for kind, st in (("subtitle", ss), ("visual", vs)):
            if st in ("queued", "running"):
                log.info("requeue stale job: %s %s (was %s)", kind, vid, st)
                enqueue(vid, kind)
                n += 1
    return n


def start_worker() -> None:
    threading.Thread(target=_worker, name="ingest-worker", daemon=True).start()


# ---------- 시각 ----------
def ingest_visual(video_id: str) -> int:
    path = media.media_path(video_id)
    dur = media.duration(path)
    db.set_duration(video_id, dur)

    fps = config.FRAMES_PER_CLIP / config.CLIP_SEC
    frames_dir = Path(config.DATA_DIR) / "frames" / video_id
    thumbs_dir = Path(config.DATA_DIR) / "thumbs" / video_id
    frames = media.extract_frames(path, fps, frames_dir)
    if not frames:
        raise RuntimeError("no frames extracted")
    log.info("visual %s: %d frames extracted, embedding...", video_id, len(frames))

    vecs = models.embed_images([str(f) for f in frames])

    # 프레임 i 시각 = i/fps → 클립 k = floor(t / CLIP_SEC). 클립 내 프레임 평균.
    groups: dict[int, list[int]] = {}
    for i in range(len(frames)):
        k = int((i / fps) // config.CLIP_SEC)
        groups.setdefault(k, []).append(i)

    if thumbs_dir.exists():
        shutil.rmtree(thumbs_dir)
    thumbs_dir.mkdir(parents=True)

    rows = []
    for k, idxs in sorted(groups.items()):
        v = vecs[idxs].mean(axis=0)
        v = v / max(np.linalg.norm(v), 1e-12)
        start = k * config.CLIP_SEC
        end = min(start + config.CLIP_SEC, dur)
        thumb = thumbs_dir / f"{k:06d}.jpg"
        shutil.copyfile(frames[idxs[0]], thumb)
        rows.append((video_id, start, end, f"{video_id}/{thumb.name}", v))

    with db.pool.connection() as conn:
        conn.execute("DELETE FROM clip_embeddings WHERE video_id = %s", (video_id,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO clip_embeddings (video_id, start_sec, end_sec, thumb_path, embedding) VALUES (%s,%s,%s,%s,%s)",
                rows,
            )
    shutil.rmtree(frames_dir, ignore_errors=True)
    return len(rows)


# ---------- 자막 ----------
def _transcribe(path: Path) -> list[tuple[float, float, str]]:
    sidecar = media.find_sidecar(path)
    if sidecar:
        log.info("using sidecar subtitle %s", sidecar)
        return media.parse_sidecar(sidecar)
    log.info("whisper(%s) transcribing %s", config.WHISPER_MODEL, path.name)
    segments, _info = models.asr().transcribe(str(path), language=config.WHISPER_LANG, vad_filter=True)
    out = []
    for s in segments:
        if s.text.strip():
            out.append((s.start, s.end, s.text.strip()))
            if len(out) % 50 == 0:
                log.info("whisper %s: %d segments, at %.0fs", path.name, len(out), s.end)
    return out


def _chunk(segs: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    chunks, buf, start = [], [], None
    for s, e, t in segs:
        if start is None:
            start = s
        buf.append(t)
        if e - start >= config.CHUNK_SEC:
            chunks.append((start, e, " ".join(buf)))
            buf, start = [], None
    if buf:
        chunks.append((start, segs[-1][1], " ".join(buf)))
    return chunks


def ingest_subtitle(video_id: str) -> int:
    path = media.media_path(video_id)
    db.set_duration(video_id, media.duration(path))
    segs = _transcribe(path)
    chunks = _chunk(segs)
    if not chunks:
        raise RuntimeError("no speech / subtitle found")
    vecs = models.embed_passages([c[2] for c in chunks])
    rows = [(video_id, s, e, t, v) for (s, e, t), v in zip(chunks, vecs)]
    with db.pool.connection() as conn:
        conn.execute("DELETE FROM subtitle_chunks WHERE video_id = %s", (video_id,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO subtitle_chunks (video_id, start_sec, end_sec, text, embedding) VALUES (%s,%s,%s,%s,%s)",
                rows,
            )
    return len(rows)
