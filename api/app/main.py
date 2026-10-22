import logging
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db, ingest, media, models, search

def _warmup() -> None:
    """기동 직후 모델 미리 로드. 첫 검색 지연 방지."""
    try:
        models.visual()
        models.text_model()
        logging.getLogger("warmup").info("models loaded")
    except Exception:  # noqa: BLE001
        logging.getLogger("warmup").exception("model warmup failed")


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.open_pool()
    Path(config.DATA_DIR, "thumbs").mkdir(parents=True, exist_ok=True)
    ingest.start_worker()
    ingest.requeue_stale()
    threading.Thread(target=_warmup, name="warmup", daemon=True).start()
    yield
    db.close_pool()


app = FastAPI(title="video-vector-poc", lifespan=lifespan)
STATIC = Path(__file__).parent / "static"
THUMBS = Path(config.DATA_DIR) / "thumbs"
THUMBS.mkdir(parents=True, exist_ok=True)
Path(config.MEDIA_DIR).mkdir(parents=True, exist_ok=True)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True}


# ---------- 영상 목록 / 인덱싱 ----------
@app.get("/api/videos")
def videos():
    with db.pool.connection() as conn:
        rows = conn.execute(
            "SELECT id, duration, visual_status, subtitle_status, visual_count, subtitle_count, error FROM videos"
        ).fetchall()
    st = {r[0]: r for r in rows}
    out = []
    for name in media.list_media():
        r = st.get(name)
        out.append({
            "id": name,
            "duration": r[1] if r else None,
            "visual_status": r[2] if r else "none",
            "subtitle_status": r[3] if r else "none",
            "visual_count": r[4] if r else 0,
            "subtitle_count": r[5] if r else 0,
            "error": r[6] if r else None,
        })
    return out


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    """브라우저용. 파일을 MEDIA_DIR 로 저장 (영상 또는 .srt/.vtt 사이드카)."""
    name = Path(file.filename or "").name
    if not name or name.startswith("."):
        raise HTTPException(400, "bad filename")
    if Path(name).suffix.lower() not in config.VIDEO_EXTS | {".srt", ".vtt"}:
        raise HTTPException(400, "unsupported extension")
    dest = Path(config.MEDIA_DIR) / name
    with open(dest, "wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    return {"saved": name}


@app.post("/api/videos/{video_id}/ingest")
def ingest_video(video_id: str, kind: str = "both"):
    try:
        media.media_path(video_id)
    except FileNotFoundError:
        raise HTTPException(404, "video not found")
    kinds = ["visual", "subtitle"] if kind == "both" else [kind]
    for k in kinds:
        if k not in ("visual", "subtitle"):
            raise HTTPException(400, "kind must be visual|subtitle|both")
        ingest.enqueue(video_id, k)
    return {"queued": kinds}


@app.get("/api/progress")
def progress():
    """진행 중/대기/최근 완료 인덱싱 작업. UI 왼쪽아래 보드가 1초마다 폴링."""
    return ingest.snapshot()


@app.delete("/api/videos/{video_id}/index")
def delete_index(video_id: str):
    with db.pool.connection() as conn:
        conn.execute("DELETE FROM videos WHERE id = %s", (video_id,))
    return {"deleted": video_id}


# ---------- 검색 ----------
DEFAULT_MIN_VISUAL = 0.10    # SigLIP2 p_match = sigmoid(scale*cos+bias). 무관 <0.03, 관련 0.3+
DEFAULT_MIN_SUBTITLE = 0.35  # Qwen3-Embedding cosine. 무관 0.15~0.30, 관련 0.38+ (Whisper 오인식 많으면 낮아짐)


class TextQuery(BaseModel):
    query: str
    video_id: str | None = None
    k: int = 20
    min_visual: float = DEFAULT_MIN_VISUAL
    min_subtitle: float = DEFAULT_MIN_SUBTITLE


def _respond(request: dict, results: list, trace: search.Trace, t0: float) -> dict:
    return {
        "results": results,
        "trace": {"request": request, "total_ms": round((time.perf_counter() - t0) * 1000, 1), "steps": trace.steps},
    }


@app.post("/api/search/visual")
async def search_visual(
    query: str | None = Form(None),
    video_id: str | None = Form(None),
    k: int = Form(20),
    min_score: float = Form(DEFAULT_MIN_VISUAL),
    image: UploadFile | None = File(None),
):
    """텍스트 또는 이미지 질의 → 시각 벡터 테이블."""
    t0 = time.perf_counter()
    trace = search.Trace()
    if image is not None and image.filename:
        with tempfile.NamedTemporaryFile(suffix=Path(image.filename).suffix or ".jpg", delete=True) as tmp:
            tmp.write(await image.read())
            tmp.flush()
            vec = search.embed_visual_image(tmp.name, trace)
        req = {"endpoint": "/api/search/visual", "image": image.filename, "video_id": video_id or None, "k": k, "min_score": min_score}
    elif query:
        vec = search.embed_visual_text(query, trace)
        req = {"endpoint": "/api/search/visual", "query": query, "video_id": video_id or None, "k": k, "min_score": min_score}
    else:
        raise HTTPException(400, "query or image required")
    return _respond(req, search.search_visual(vec, video_id or None, k, trace, min_score), trace, t0)


@app.post("/api/search/subtitle")
def search_subtitle(q: TextQuery):
    t0 = time.perf_counter()
    trace = search.Trace()
    vec = search.embed_text_query(q.query, trace)
    res = search.search_subtitle(vec, q.video_id or None, q.k, trace, q.min_subtitle)
    return _respond({"endpoint": "/api/search/subtitle", **q.model_dump()}, res, trace, t0)


@app.post("/api/search/hybrid")
def search_hybrid(q: TextQuery):
    t0 = time.perf_counter()
    trace = search.Trace()
    res = search.search_hybrid(q.query, q.video_id or None, q.k, trace, q.min_visual, q.min_subtitle)
    return _respond({"endpoint": "/api/search/hybrid", **q.model_dump()}, res, trace, t0)


# ---------- 정적 파일 ----------
app.mount("/media", StaticFiles(directory=config.MEDIA_DIR), name="media")
app.mount("/thumbs", StaticFiles(directory=str(THUMBS)), name="thumbs")
