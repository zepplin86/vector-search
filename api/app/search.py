"""검색. 시각/자막 각각 별도 테이블·별도 질의 인코더. 통합은 RRF.
모든 단계는 Trace 에 기록 → UI 디버그 패널에 그대로 노출."""
import time

import numpy as np

from . import config, db, models

RRF_K = 60
MERGE_BUCKET_SEC = 30.0
EF_SEARCH = 100

SQL_VISUAL = """SELECT video_id, start_sec, end_sec, thumb_path,
       1 - (embedding <=> %(v)s) AS score
FROM clip_embeddings
WHERE %(vid)s::text IS NULL OR video_id = %(vid)s
ORDER BY embedding <=> %(v)s
LIMIT %(k)s"""

SQL_SUBTITLE = """SELECT video_id, start_sec, end_sec, text,
       1 - (embedding <=> %(v)s) AS score
FROM subtitle_chunks
WHERE %(vid)s::text IS NULL OR video_id = %(vid)s
ORDER BY embedding <=> %(v)s
LIMIT %(k)s"""


class Trace:
    def __init__(self) -> None:
        self.steps: list[dict] = []

    def add(self, step: str, **kw) -> None:
        self.steps.append({"step": step, **kw})


def vec_preview(v: np.ndarray) -> dict:
    return {
        "dim": int(v.shape[0]),
        "norm": round(float(np.linalg.norm(v)), 4),
        "head": [round(float(x), 4) for x in v[:8]],
        "note": "pgvector 에는 전체 float 배열이 '[x,y,...]' 리터럴로 바인딩됨",
    }


def _timed_embed(trace: Trace, label: str, model: str, fn):
    t0 = time.perf_counter()
    v = fn()
    trace.add("embed", target=label, model=model, ms=round((time.perf_counter() - t0) * 1000, 1), vector=vec_preview(v))
    return v


def embed_visual_text(q: str, trace: Trace) -> np.ndarray:
    return _timed_embed(trace, "visual/text", f"{config.VISUAL_MODEL} ({config.VISUAL_PRETRAINED}) text encoder",
                        lambda: models.embed_visual_text(q))


def embed_visual_image(path: str, trace: Trace) -> np.ndarray:
    return _timed_embed(trace, "visual/image", f"{config.VISUAL_MODEL} ({config.VISUAL_PRETRAINED}) image encoder",
                        lambda: models.embed_visual_image(path))


def embed_text_query(q: str, trace: Trace) -> np.ndarray:
    return _timed_embed(trace, "subtitle/text", f"{config.TEXT_MODEL} (prompt_name=query)",
                        lambda: models.embed_text_query(q))


def _query(trace: Trace, table: str, sql: str, vec: np.ndarray, video_id: str | None, k: int,
           min_score: float, calibrate=None) -> list[tuple]:
    """rows 마지막 컬럼 = cosine(1 - distance). calibrate(cos) 주어지면 그 값을 score 로 써서 컷."""
    params = {"v": vec, "vid": video_id, "k": k}
    t0 = time.perf_counter()
    with db.pool.connection() as conn:
        conn.execute(f"SET LOCAL hnsw.ef_search = {EF_SEARCH}")
        rows = conn.execute(sql, params).fetchall()
    ms = round((time.perf_counter() - t0) * 1000, 1)
    scored = [(r, float(calibrate(float(r[-1])) if calibrate else r[-1])) for r in rows]
    kept = [(r, sc) for r, sc in scored if sc >= min_score]
    trace.add(
        "pgvector",
        table=table,
        sql=f"SET LOCAL hnsw.ef_search = {EF_SEARCH};\n{sql};",
        params={"v": f"<vector({vec.shape[0]})>", "vid": video_id, "k": k},
        ms=ms,
        row_count=len(rows),
        rows=[[(round(float(x), 4) if isinstance(x, float) else x) for x in r] + ([round(sc, 4)] if calibrate else [])
              for r, sc in scored],
        columns=["video_id", "start_sec", "end_sec", "thumb_path" if table == "clip_embeddings" else "text",
                 "cosine"] + (["p_match"] if calibrate else []),
        min_score=min_score,
        kept=len(kept),
        note="ANN 은 항상 가장 가까운 k개를 돌려줌. 관련 없어도 나옴 → score 컷으로 걸러냄 (앱 레이어)"
             + (" / SigLIP: p_match = sigmoid(scale*cos + bias), 컷은 p_match 기준" if calibrate else ""),
    )
    return kept


def search_visual(vec: np.ndarray, video_id: str | None, k: int, trace: Trace, min_score: float = 0.0) -> list[dict]:
    scale, bias = models.visual_logit_params()
    sig = lambda c: 1.0 / (1.0 + np.exp(-(c * scale + bias)))  # noqa: E731
    rows = _query(trace, "clip_embeddings", SQL_VISUAL, vec, video_id, k, min_score, calibrate=sig)
    return [
        {"source": "visual", "video_id": r[0], "start": r[1], "end": r[2], "thumb": r[3],
         "score": round(p, 4), "cosine": round(float(r[4]), 4)}
        for r, p in rows
    ]


def search_subtitle(vec: np.ndarray, video_id: str | None, k: int, trace: Trace, min_score: float = 0.0) -> list[dict]:
    rows = _query(trace, "subtitle_chunks", SQL_SUBTITLE, vec, video_id, k, min_score)
    return [
        {"source": "subtitle", "video_id": r[0], "start": r[1], "end": r[2], "text": r[3], "score": round(sc, 4)}
        for r, sc in rows
    ]


def thumb_for(video_id: str, sec: float) -> str | None:
    with db.pool.connection() as conn:
        row = conn.execute(
            "SELECT thumb_path FROM clip_embeddings WHERE video_id = %s AND start_sec <= %s ORDER BY start_sec DESC LIMIT 1",
            (video_id, sec),
        ).fetchone()
    return row[0] if row else None


def search_hybrid(query: str, video_id: str | None, k: int, trace: Trace,
                  min_visual: float = 0.0, min_subtitle: float = 0.0) -> list[dict]:
    vis = search_visual(embed_visual_text(query, trace), video_id, 50, trace, min_visual)
    sub = search_subtitle(embed_text_query(query, trace), video_id, 50, trace, min_subtitle)

    t0 = time.perf_counter()
    groups: dict[tuple[str, int], dict] = {}
    contributions: list[dict] = []

    def add(items: list[dict]) -> None:
        for rank, it in enumerate(items, start=1):
            key = (it["video_id"], int(it["start"] // MERGE_BUCKET_SEC))
            g = groups.setdefault(key, {
                "video_id": it["video_id"], "start": it["start"], "end": it["end"],
                "score": 0.0, "sources": [], "text": None, "thumb": None,
            })
            rrf = 1.0 / (RRF_K + rank)
            # 소스별 최고 rank 1개만 반영. 합산하면 5초 클립이 여러 개 들어가는 시각 소스가 항상 자막을 이김.
            best = g.setdefault("_best", {})
            if it["source"] not in best:
                best[it["source"]] = rrf
                g["score"] += rrf
                contributions.append({"bucket": f"{key[0]}#{key[1]}", "source": it["source"], "rank": rank,
                                      "score": it["score"], "rrf": round(rrf, 5)})
            g["start"] = min(g["start"], it["start"])
            g["end"] = max(g["end"], it["end"])
            if it["source"] not in g["sources"]:
                g["sources"].append(it["source"])
            if it.get("text") and not g["text"]:
                g["text"] = it["text"]
            if it.get("thumb") and not g["thumb"]:
                g["thumb"] = it["thumb"]

    add(vis)
    add(sub)
    out = sorted(groups.values(), key=lambda g: g["score"], reverse=True)[:k]
    for g in groups.values():
        g.pop("_best", None)
    for g in out:
        if not g["thumb"]:
            g["thumb"] = thumb_for(g["video_id"], g["start"])
        g["source"] = "+".join(g["sources"])
    trace.add(
        "rrf_merge",
        formula=f"score = Σ_source 1/({RRF_K} + best_rank_in_source)   bucket = (video_id, floor(start / {MERGE_BUCKET_SEC:g}s))",
        ms=round((time.perf_counter() - t0) * 1000, 1),
        buckets=len(groups),
        contributions=contributions[:60],
        merged=[{"bucket": f"{g['video_id']}#{int(g['start'] // MERGE_BUCKET_SEC)}", "score": round(g["score"], 5),
                 "sources": g["sources"], "start": g["start"], "end": g["end"]} for g in out],
    )
    return out
