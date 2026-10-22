"""인덱싱 파이프라인. 시각/자막 분리. 순차 워커 스레드 하나로 처리 (CPU 공유).

진행률: 메모리 딕셔너리 하나(_progress)에 작업별 단계/퍼센트 기록. /api/progress 가 그대로 노출.
"""
import logging
import queue
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from . import config, db, media, models

log = logging.getLogger("ingest")

_jobs: "queue.Queue[tuple[str, str]]" = queue.Queue()

# ---------- 진행률 ----------
_plock = threading.Lock()
_progress: dict[str, dict] = {}   # "video:kind" → 상태
_FINISHED_TTL = 10.0              # done/error 를 보드에 남겨두는 시간(초)
_seq = 0                          # 큐 순서 (대기열 표시용)

# 단계별 가중치 (전체 퍼센트 계산). 합 = 1.0
_WEIGHTS = {
    "visual":   [("model", 0.05), ("frames", 0.20), ("embed", 0.70), ("save", 0.05)],
    "subtitle": [("model", 0.05), ("transcribe", 0.60), ("embed", 0.30), ("save", 0.05)],
}
PHASE_LABEL = {
    "model": "모델 로드", "frames": "프레임 추출", "transcribe": "음성 인식",
    "embed": "임베딩", "save": "DB 저장", "queued": "대기",
}


def _key(video_id: str, kind: str) -> str:
    return f"{video_id}:{kind}"


def _overall(kind: str, phase: str, phase_pct: float | None) -> float | None:
    """단계 가중치 누적 → 0~100. 현재 단계 퍼센트 모르면 단계 시작점."""
    acc = 0.0
    for name, w in _WEIGHTS[kind]:
        if name == phase:
            return round((acc + w * (phase_pct or 0) / 100) * 100, 1)
        acc += w
    return None


def report(video_id: str, kind: str, phase: str, done: float | None = None, total: float | None = None,
           detail: str | None = None) -> None:
    """워커가 단계 진입/진행 때 호출. done/total 주면 단계 퍼센트 계산."""
    pct = None
    if done is not None and total:
        pct = max(0.0, min(100.0, done / total * 100))
    with _plock:
        p = _progress.setdefault(_key(video_id, kind), {"video_id": video_id, "kind": kind})
        if p.get("phase") != phase:
            p["phase_started"] = time.time()
        p.update({
            "status": "running", "phase": phase, "phase_label": PHASE_LABEL.get(phase, phase),
            "phase_pct": round(pct, 1) if pct is not None else None,
            "done": done, "total": total, "detail": detail,
            "pct": _overall(kind, phase, pct), "updated": time.time(),
        })


def _set(video_id: str, kind: str, **fields) -> None:
    with _plock:
        p = _progress.setdefault(_key(video_id, kind), {"video_id": video_id, "kind": kind})
        p.update(fields, updated=time.time())


def snapshot() -> list[dict]:
    """/api/progress 응답. running → queued(순서) → finished. 오래된 finished 는 제거."""
    now = time.time()
    with _plock:
        for k in [k for k, p in _progress.items()
                  if p.get("status") in ("done", "error") and now - p.get("finished", now) > _FINISHED_TTL]:
            del _progress[k]
        out = []
        for p in _progress.values():
            q = dict(p)
            q["elapsed"] = round(now - p["started"], 1) if p.get("started") else None
            q["phase_elapsed"] = round(now - p["phase_started"], 1) if p.get("phase_started") else None
            out.append(q)
    order = {"running": 0, "queued": 1, "done": 2, "error": 2}
    out.sort(key=lambda p: (order.get(p.get("status"), 3), p.get("seq", 0)))
    return out


# ---------- 큐 / 워커 ----------
def enqueue(video_id: str, kind: str) -> None:
    global _seq
    db.set_status(video_id, kind, "queued")
    with _plock:
        _seq += 1
        _progress[_key(video_id, kind)] = {
            "video_id": video_id, "kind": kind, "status": "queued", "phase": "queued",
            "phase_label": PHASE_LABEL["queued"], "pct": 0, "seq": _seq, "updated": time.time(),
        }
    _jobs.put((video_id, kind))


def _worker() -> None:
    while True:
        video_id, kind = _jobs.get()
        try:
            db.set_status(video_id, kind, "running")
            _set(video_id, kind, status="running", started=time.time())
            log.info("ingest start: %s %s", kind, video_id)
            n = ingest_visual(video_id) if kind == "visual" else ingest_subtitle(video_id)
            db.set_status(video_id, kind, "done", count=n)
            _set(video_id, kind, status="done", pct=100, phase="done", phase_label="완료",
                 detail=f"{n} rows", finished=time.time())
            log.info("ingest done: %s %s rows=%d", kind, video_id, n)
        except Exception as e:  # noqa: BLE001
            log.exception("ingest failed: %s %s", kind, video_id)
            db.set_status(video_id, kind, "error", error=f"{kind}: {e}")
            _set(video_id, kind, status="error", phase="error", phase_label="오류", detail=str(e)[:200],
                 finished=time.time())
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

    report(video_id, "visual", "model", detail=config.VISUAL_MODEL)
    models.visual()

    fps = config.FRAMES_PER_CLIP / config.CLIP_SEC
    frames_dir = Path(config.DATA_DIR) / "frames" / video_id
    thumbs_dir = Path(config.DATA_DIR) / "thumbs" / video_id
    report(video_id, "visual", "frames", 0, dur, detail="ffmpeg")
    frames = media.extract_frames(
        path, fps, frames_dir,
        on_progress=lambda t: report(video_id, "visual", "frames", t, dur, detail=f"{t:.0f}s / {dur:.0f}s"),
    )
    if not frames:
        raise RuntimeError("no frames extracted")
    log.info("visual %s: %d frames extracted, embedding...", video_id, len(frames))

    report(video_id, "visual", "embed", 0, len(frames), detail=f"0 / {len(frames)} frames")
    vecs = models.embed_images(
        [str(f) for f in frames],
        on_progress=lambda i, n: report(video_id, "visual", "embed", i, n, detail=f"{i} / {n} frames"),
    )

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

    report(video_id, "visual", "save", detail=f"{len(rows)} clips")
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
def _transcribe(video_id: str, path: Path, dur: float) -> list[tuple[float, float, str]]:
    sidecar = media.find_sidecar(path)
    if sidecar:
        log.info("using sidecar subtitle %s", sidecar)
        report(video_id, "subtitle", "transcribe", 1, 1, detail=f"자막 파일 {sidecar.name}")
        return media.parse_sidecar(sidecar)
    report(video_id, "subtitle", "model", detail=f"whisper {config.WHISPER_MODEL}")
    asr = models.asr()
    log.info("whisper(%s) transcribing %s", config.WHISPER_MODEL, path.name)
    report(video_id, "subtitle", "transcribe", 0, dur, detail=f"whisper {config.WHISPER_MODEL}")
    segments, _info = asr.transcribe(str(path), language=config.WHISPER_LANG, vad_filter=True)
    out = []
    for s in segments:
        if s.text.strip():
            out.append((s.start, s.end, s.text.strip()))
            if len(out) % 50 == 0:
                log.info("whisper %s: %d segments, at %.0fs", path.name, len(out), s.end)
        report(video_id, "subtitle", "transcribe", s.end, dur, detail=f"{s.end:.0f}s / {dur:.0f}s · {len(out)} seg")
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
    dur = media.duration(path)
    db.set_duration(video_id, dur)
    segs = _transcribe(video_id, path, dur)
    chunks = _chunk(segs)
    if not chunks:
        raise RuntimeError("no speech / subtitle found")
    report(video_id, "subtitle", "model", detail=config.TEXT_MODEL)
    models.text_model()
    report(video_id, "subtitle", "embed", 0, len(chunks), detail=f"0 / {len(chunks)} chunks")
    vecs = models.embed_passages(
        [c[2] for c in chunks],
        on_progress=lambda i, n: report(video_id, "subtitle", "embed", i, n, detail=f"{i} / {n} chunks"),
    )
    rows = [(video_id, s, e, t, v) for (s, e, t), v in zip(chunks, vecs)]
    report(video_id, "subtitle", "save", detail=f"{len(rows)} chunks")
    with db.pool.connection() as conn:
        conn.execute("DELETE FROM subtitle_chunks WHERE video_id = %s", (video_id,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO subtitle_chunks (video_id, start_sec, end_sec, text, embedding) VALUES (%s,%s,%s,%s,%s)",
                rows,
            )
    return len(rows)
