# claude-vector — 영상/자막 벡터 검색 PoC

pgvector(PostgreSQL) 하나에 **영상(시각) 벡터**와 **자막 벡터**를 별도 테이블·별도 모델로 저장하고,
텍스트/이미지 질의로 구간을 찾아 바로 재생하는 PoC.

```
media/*.mp4 ──┬─ ffmpeg 프레임 ─ SigLIP2 ──────────► clip_embeddings   (vector 768)
              └─ whisper / .srt ─ Qwen3-Embedding ─► subtitle_chunks   (vector 1024)
질의 ─ 같은 모델로 임베딩 ─ cosine top-k ─ (통합: RRF) ─ 결과 클릭 → 해당 구간 재생
```

## 시스템 구성도

![시스템 구성도](docs/architecture.png)

원본: [`docs/architecture.svg`](docs/architecture.svg)

| 구간 | 흐름 |
|---|---|
| 입력 | `media/*.mp4` 를 직접 넣거나, 데스크톱 앱 '영상 추가'로 복사. 같은 이름 `.srt/.vtt` 있으면 Whisper 생략 |
| 인덱싱 (api 워커) | 시각: ffmpeg 프레임(5초 클립·2프레임 평균) → SigLIP2 → `clip_embeddings` vector(768). 자막: 자막파일 or faster-whisper → 45초 청크 → Qwen3-Embedding-0.6B → `subtitle_chunks` vector(1024) |
| 저장 (db) | pgvector pg17. 테이블·모델·차원 분리, 각각 HNSW cosine 인덱스 |
| 검색 (api) | 질의를 같은 모델로 임베딩 → `ORDER BY embedding <=> $q LIMIT k` → 점수 컷. hybrid 는 둘 다 실행 후 앱 레벨 RRF(k=60) 병합 |
| 출력 | `{results, trace}` → 웹 UI / 데스크톱 앱(WKWebView). 결과 클릭 시 해당 구간 재생 |

## 구성

| 서비스 | 내용 |
|---|---|
| `db`  | `pgvector/pgvector:pg17`, 포트 5433, 스키마 `db/init.sql` |
| `api` | FastAPI + 모델(CPU). 인덱싱 워커, 검색 API, 웹 UI. 포트 8000 |
| `desktop/` | pywebview 창. 호스트에서 실행, API에 붙음 |

## 실행

```bash
# 1. 영상 넣기 (같은 이름의 .srt/.vtt 있으면 Whisper 대신 사용)
cp ~/some.mp4 media/

# 2. 서비스 기동 (첫 빌드 수 분, 첫 인덱싱 때 모델 다운로드 ~2GB)
docker compose up -d --build
docker compose logs -f api

# 3-a. 브라우저
open http://localhost:8000

# 3-b. 데스크톱 앱
cd desktop && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

UI 왼쪽에서 영상별 **시각 인덱싱 / 자막 인덱싱** 클릭 → 상태 done 되면 검색.

## 검색 API

```bash
# 시각 (텍스트)
curl -F query="강아지가 뛰는 장면" -F k=10 localhost:8000/api/search/visual
# 시각 (이미지)
curl -F image=@shot.jpg localhost:8000/api/search/visual
# 자막
curl -H 'content-type: application/json' -d '{"query":"환불 정책 설명","k":10}' localhost:8000/api/search/subtitle
# 통합 (RRF)
curl -H 'content-type: application/json' -d '{"query":"강아지","video_id":null,"k":10}' localhost:8000/api/search/hybrid
```

`video_id`(파일명) 주면 그 영상 안에서만 검색.

응답은 `{results: [...], trace: {request, total_ms, steps[]}}`. `steps`에 임베딩(모델·차원·시간), pgvector SQL·파라미터·원본 rows, RRF 병합 내역이 순서대로 들어감. UI 우측 패널이 이걸 그대로 표시.

## 점수 컷 (왜 아무거나 검색해도 결과가 나오나)

ANN 검색은 **항상 가장 가까운 k개**를 돌려줌. 관련 없어도 "덜 먼 것"이 나옴. 그래서 앱에서 cosine 점수 하한으로 걸러냄.

| 파라미터 | 기본 | 비고 |
|---|---|---|
| 시각 `min_score` / `min_visual` | 0.10 | SigLIP2 **매칭 확률** `p = sigmoid(logit_scale·cos + logit_bias)` (scale≈113, bias≈-16.8). 무관 <0.03, 관련 0.3+. cosine 자체는 실사 영상에서 0.10~0.15에 몰려 쓸모없음 |
| 자막 `min_subtitle` | 0.35 | Qwen3-Embedding cosine. 무관 0.15~0.30, 관련 0.38+. Whisper 오인식 많을수록 낮아짐 |

값은 영상·질의에 따라 달라짐. 우측 패널 `rows` 의 score 보고 조정. 0 이면 컷 없음.

## 튜닝 (docker-compose.yml 환경변수)

| 변수 | 기본 | 설명 |
|---|---|---|
| `CLIP_SEC` | 5 | 시각 클립 길이(초) |
| `FRAMES_PER_CLIP` | 2 | 클립당 프레임 수, 평균 풀링 |
| `CHUNK_SEC` | 45 | 자막 청크 길이(초) |
| `VISUAL_MODEL` | ViT-B-16-SigLIP2 | open_clip 모델명. 바꾸면 `clip_embeddings` 차원도 맞춰야 함 |
| `TEXT_MODEL` | Qwen/Qwen3-Embedding-0.6B | 바꾸면 `subtitle_chunks` 차원 맞춰야 함 |
| `WHISPER_MODEL` | small | base/small/medium/large-v3 |
| `WHISPER_LANG` | ko | 비우면 자동 감지 |

차원 바꾸면 `docker compose down -v` 후 재기동 (init.sql 재실행).

## 메모

- 전부 CPU. Docker Desktop 메모리 8GB 이상 권장. 1시간 영상: 시각 ~3분, Whisper small ~5분.
- Mac MPS 쓰려면 api를 호스트에서 직접 실행: `DATABASE_URL=postgresql://vec:vec@localhost:5433/vec MEDIA_DIR=./media DATA_DIR=./data uvicorn app.main:app` (api 디렉토리에서, 모델 코드의 device 수정 필요).
- 자막 검색은 벡터만. 고유명사 정확 매칭 필요하면 FTS(pg_bigm) + RRF 추가.
- SigLIP2 로드 실패 시 `VISUAL_MODEL=ViT-B-16-SigLIP`, `VISUAL_PRETRAINED=webli` (영어 질의 전용) 로 대체.
