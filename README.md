# claude-vector — 영상/자막 벡터 검색 PoC

## 한 줄 요약

영상 폴더에 mp4 를 넣으면, **"강아지가 뛰는 장면"** 이나 **"환불 정책 설명하는 부분"** 같은 말로 검색해서
그 장면을 바로 재생해 주는 실험용 프로그램입니다.

## 이게 어떻게 가능한가 (처음 보는 분을 위한 설명)

컴퓨터는 "강아지"라는 글자와 강아지 사진이 같은 것이라는 걸 모릅니다.
그래서 AI 모델을 써서 **글자도, 사진도, 전부 숫자 목록(벡터)** 으로 바꿉니다.
이 숫자 목록은 "비슷한 의미면 비슷한 숫자"가 되도록 학습되어 있습니다.

- 강아지 사진 → `[0.12, -0.33, 0.80, …]`
- "강아지" 라는 글자 → `[0.11, -0.30, 0.79, …]`  ← 사진과 거의 같은 숫자
- "엑셀 함수" 라는 글자 → `[-0.70, 0.05, -0.22, …]` ← 전혀 다른 숫자

이렇게 숫자로 바꾸는 작업을 **임베딩**이라고 합니다.
검색은 "질문을 숫자로 바꾼 뒤, 저장된 숫자들 중 가장 가까운 것을 찾는 것"입니다.
숫자 목록을 저장하고 "가까운 것"을 빠르게 찾아주는 DB 가 **벡터 DB** 이고, 여기선 PostgreSQL 의 확장 기능인 **pgvector** 를 씁니다.

영상은 두 가지 방법으로 검색합니다.

| 방법 | 무엇을 보나 | 어떻게 숫자로 만드나 | 어떤 질문에 좋은가 |
|---|---|---|---|
| **시각 검색** | 화면에 보이는 것 | 영상을 5초마다 잘라 사진을 뽑고, 사진을 숫자로 변환 (SigLIP2 모델) | "강아지", "바다", "빨간 자동차" 처럼 눈에 보이는 것 |
| **자막 검색** | 말하는 내용 | 음성을 글자로 받아쓴 뒤(Whisper), 45초씩 묶어서 숫자로 변환 (Qwen3-Embedding 모델) | "환불 정책", "가입 방법" 처럼 말로 설명한 것 |
| **통합 검색** | 둘 다 | 위 둘을 각각 검색한 뒤 순위를 합침 (RRF) | 뭐가 나올지 모를 때 |

```
media/*.mp4 ──┬─ ffmpeg 프레임 ─ SigLIP2 ──────────► clip_embeddings   (vector 768)
              └─ whisper / .srt ─ Qwen3-Embedding ─► subtitle_chunks   (vector 1024)
질의 ─ 같은 모델로 임베딩 ─ cosine top-k ─ (통합: RRF) ─ 결과 클릭 → 해당 구간 재생
```

## 시스템 구성도

![시스템 구성도](docs/architecture.png)

원본: [`docs/architecture.svg`](docs/architecture.svg)

| 구간 | 흐름 (쉬운 말) | 세부 |
|---|---|---|
| 입력 | 영상 파일을 `media/` 폴더에 넣음. 자막 파일(.srt/.vtt)이 같은 이름으로 있으면 받아쓰기 생략 | 데스크톱 앱 '영상 추가' 버튼으로 복사도 가능 |
| 인덱싱 | 영상을 조각내서 숫자로 바꿔 DB 에 저장. **시간이 오래 걸림** (1시간 영상 ≈ 10분) | 시각: 5초 클립·2프레임 평균 → SigLIP2 → `clip_embeddings` vector(768). 자막: 자막파일 or faster-whisper → 45초 청크 → Qwen3-Embedding-0.6B → `subtitle_chunks` vector(1024) |
| 저장 | PostgreSQL 에 숫자 목록 + "몇 초부터 몇 초까지" 정보 저장 | pgvector pg17. 테이블·모델·차원 분리, 각각 HNSW cosine 인덱스 |
| 검색 | 질문을 같은 방식으로 숫자로 바꾸고 가장 가까운 것 찾기 | `ORDER BY embedding <=> $q LIMIT k` → 점수 컷. hybrid 는 둘 다 실행 후 앱 레벨 RRF(k=60) 병합 |
| 출력 | 결과 카드 클릭하면 해당 구간부터 재생 | `{results, trace}` → 웹 UI / 데스크톱 앱(WKWebView) |

## 구성

| 서비스 | 역할 |
|---|---|
| `db`  | 숫자(벡터) 저장소. `pgvector/pgvector:pg17`, 포트 5433, 스키마 `db/init.sql` |
| `api` | AI 모델 실행 + 인덱싱 + 검색 + 웹 화면. FastAPI, CPU 전용. 포트 8000 |
| `desktop/` | 웹 화면을 창으로 띄우는 Mac 앱 (pywebview). 호스트에서 실행, API에 붙음 |

## 실행

사전 준비: Docker Desktop 설치, 메모리 8GB 이상 할당.

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

### 화면 사용법

1. 왼쪽 목록에서 영상 옆 **시각 인덱싱 / 자막 인덱싱 / 둘 다** 클릭.
2. 왼쪽아래 작은 보드에 진행 상황이 뜸. 단계는 `모델 로드 → 프레임 추출(또는 음성 인식) → 임베딩 → 저장`. 1초마다 갱신 (`GET /api/progress`).
   첫 실행은 모델 다운로드·로드 때문에 "모델 로드"에서 1~2분 머무는 게 정상.
3. 상태가 **done** 이 되면 위쪽 검색창에 말로 입력하고 검색. 탭으로 시각/자막/통합 선택.
4. 결과 카드 클릭 → 아래 플레이어가 그 구간부터 재생.
5. 오른쪽 패널은 "검색이 내부에서 어떻게 돌았나" (숫자 변환 → SQL → 결과 → 병합) 를 그대로 보여주는 디버그 창. 안 봐도 됨.

## 검색 API (화면 없이 직접 호출할 때)

```bash
# 시각 (텍스트)
curl -F query="강아지가 뛰는 장면" -F k=10 localhost:8000/api/search/visual
# 시각 (이미지)
curl -F image=@shot.jpg localhost:8000/api/search/visual
# 자막
curl -H 'content-type: application/json' -d '{"query":"환불 정책 설명","k":10}' localhost:8000/api/search/subtitle
# 통합 (RRF)
curl -H 'content-type: application/json' -d '{"query":"강아지","video_id":null,"k":10}' localhost:8000/api/search/hybrid
# 인덱싱 진행 상황
curl localhost:8000/api/progress
```

`video_id`(파일명) 주면 그 영상 안에서만 검색. `k` 는 최대 결과 개수.

응답은 `{results: [...], trace: {request, total_ms, steps[]}}`. `steps`에 임베딩(모델·차원·시간), pgvector SQL·파라미터·원본 rows, RRF 병합 내역이 순서대로 들어감. UI 우측 패널이 이걸 그대로 표시.

## 점수 컷 — 왜 아무거나 검색해도 결과가 나오나

벡터 검색은 **"가장 가까운 것 k개"** 를 항상 돌려줍니다. 관련 없는 질문이어도 "그나마 덜 먼 것"이 나옵니다.
그래서 앱에서 "이 점수 미만은 버림" 기준(점수 컷)을 둡니다. 화면 검색창의 **시각 p≥ / 자막≥** 입력칸이 그것입니다.

| 파라미터 | 기본 | 쉬운 설명 | 세부 |
|---|---|---|---|
| 시각 `min_score` / `min_visual` | 0.10 | "이 사진이 질문과 맞을 확률". 무관하면 0.03 이하, 맞으면 0.3 이상 | SigLIP2 매칭 확률 `p = sigmoid(logit_scale·cos + logit_bias)` (scale≈113, bias≈-16.8). cosine 자체는 실사 영상에서 0.10~0.15에 몰려 쓸모없음 |
| 자막 `min_subtitle` | 0.35 | "말한 내용이 질문과 얼마나 비슷한가". 무관하면 0.15~0.30, 맞으면 0.38 이상 | Qwen3-Embedding cosine. Whisper 오인식 많을수록 낮아짐 |

값은 영상·질의에 따라 달라짐. 결과가 안 나오면 낮추고, 엉뚱한 게 섞이면 올리기. 우측 패널 `rows` 의 score 보고 조정. 0 이면 컷 없음.

## 튜닝 (docker-compose.yml 환경변수)

| 변수 | 기본 | 쉬운 설명 | 세부 |
|---|---|---|---|
| `CLIP_SEC` | 5 | 영상을 몇 초 단위로 자를지. 짧을수록 정밀하지만 느리고 DB 커짐 | 시각 클립 길이(초) |
| `FRAMES_PER_CLIP` | 2 | 한 조각에서 사진 몇 장 뽑을지 | 클립당 프레임 수, 평균 풀링 |
| `CHUNK_SEC` | 45 | 자막을 몇 초씩 묶을지 | 자막 청크 길이(초) |
| `VISUAL_MODEL` | ViT-B-16-SigLIP2 | 사진→숫자 변환 모델 | open_clip 모델명. 바꾸면 `clip_embeddings` 차원도 맞춰야 함 |
| `TEXT_MODEL` | Qwen/Qwen3-Embedding-0.6B | 글자→숫자 변환 모델 | 바꾸면 `subtitle_chunks` 차원 맞춰야 함 |
| `WHISPER_MODEL` | small | 받아쓰기 모델 크기. 클수록 정확하지만 느림 | base/small/medium/large-v3 |
| `WHISPER_LANG` | ko | 받아쓰기 언어 | 비우면 자동 감지 |

**바꾼 뒤 해야 할 것**

- `CLIP_SEC` / `FRAMES_PER_CLIP` / `CHUNK_SEC`: DB 구조와 무관. 값 바꾸고 `docker compose up -d` 로 api 재기동 후 영상별 인덱싱 다시 실행 (기존 rows 는 재인덱싱 때 삭제 후 교체).
- `VISUAL_MODEL` / `TEXT_MODEL`: 숫자 목록의 길이(차원)가 바뀌므로 DB 를 새로 만들어야 함. `db/init.sql` 의 `vector(768)` / `vector(1024)` 수정 → `docker compose down -v` → 재기동 (init.sql 재실행).

## 메모

- 전부 CPU. Docker Desktop 메모리 8GB 이상 권장. 1시간 영상: 시각 ~3분, Whisper small ~5분.
- 첫 기동 직후 모델 로드가 느림 (SigLIP ~1분, Qwen ~30초). 인덱싱 보드의 "모델 로드"가 길게 보이는 건 이 때문.
- Mac MPS 쓰려면 api를 호스트에서 직접 실행: `DATABASE_URL=postgresql://vec:vec@localhost:5433/vec MEDIA_DIR=./media DATA_DIR=./data uvicorn app.main:app` (api 디렉토리에서, 모델 코드의 device 수정 필요).
- 자막 검색은 벡터만. 고유명사 정확 매칭 필요하면 FTS(pg_bigm) + RRF 추가.
- SigLIP2 로드 실패 시 `VISUAL_MODEL=ViT-B-16-SigLIP`, `VISUAL_PRETRAINED=webli` (영어 질의 전용) 로 대체.
- 사전 조사 내용(왜 pgvector 인가, 다른 벡터 DB 비교 등)은 `RESEARCH.md`.

## 용어

| 용어 | 뜻 |
|---|---|
| 벡터 / 임베딩 | 글자·사진을 AI 가 숫자 목록으로 바꾼 것. 의미가 비슷하면 숫자도 비슷 |
| 차원 | 숫자 목록의 길이. 시각 768개, 자막 1024개 |
| cosine 유사도 | 두 숫자 목록이 얼마나 같은 방향인지. 1 이면 같음, 0 이면 무관 |
| ANN / HNSW | 수만 개 중 가까운 것을 전부 비교하지 않고 빠르게 찾는 방법. 약간 틀릴 수 있지만 훨씬 빠름 |
| pgvector | PostgreSQL 에 벡터 저장·검색 기능을 붙이는 확장 |
| RRF | 두 검색 결과의 순위를 합치는 공식. 양쪽에서 상위면 높은 점수 |
| Whisper | 음성을 글자로 받아쓰는 OpenAI 모델 |
| SigLIP2 | 사진과 글자를 같은 숫자 공간에 놓는 Google 모델. 글자로 사진 검색 가능 |
| Qwen3-Embedding | 글자를 숫자로 바꾸는 모델. 한국어 지원 |
| 인덱싱 | 영상을 조각내 숫자로 바꿔 DB 에 저장하는 사전 작업 |
