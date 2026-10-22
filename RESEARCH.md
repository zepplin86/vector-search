# 영상 벡터 검색 사전 조사 노트

앱 구현 전 정리한 조사 내용. 구현 상세는 `README.md`.

## 1. 영상 벡터 검색에 필요한 것 (4단계 파이프라인)

1. **전처리 (영상 → 조각)**
   - ffmpeg로 씬 분할 또는 고정 간격(예: 5초) 클립, 키프레임 추출, 오디오 분리
   - 조각마다 `start/end` 타임스탬프 보존 → 검색 결과에서 바로 그 구간 재생
2. **임베딩 모델 (조각 → 벡터)**
   - 시각: CLIP / SigLIP (프레임 단위), VideoCLIP · InternVideo2 (클립 단위, 동작 이해)
   - 오디오/자막: Whisper → 텍스트 → 텍스트 임베딩
   - 통합 상용: Twelve Labs Marengo, Google Vertex Multimodal Embedding
   - 텍스트로 영상을 찾으려면 **텍스트-이미지 공동 공간** 모델 필수 (CLIP 계열)
3. **벡터 DB (저장 + ANN 검색)**
   - 가벼움: pgvector, sqlite-vec, FAISS / 전용: Qdrant, Milvus, Weaviate
   - 메타데이터 필수: `video_id`, `start_sec`, `end_sec`, 썸네일 경로
4. **질의 + 후처리**
   - 질의도 **같은 모델**로 임베딩 → cosine top-k
   - 인접 구간 병합, 영상 단위 집계(max/mean), 선택적 리랭킹

## 2. 벡터 DB 선택: PostgreSQL + pgvector

pgvector = PostgreSQL 확장. `CREATE EXTENSION vector` 한 줄. 별도 제품 아님.

**장점**
- 메타데이터(video_id, 구간, 권한, 카테고리)와 벡터를 한 테이블 → JOIN·필터 SQL로 처리
- 기존 PG 운영 그대로. 백업·권한·트랜잭션 공짜. 전용 DB 도입 시 생기는 메타↔벡터 동기화 문제 없음
- HNSW / IVFFlat, cosine · L2 · inner product 지원

**한계**
- 수백만 벡터 넘으면 Qdrant/Milvus가 속도·메모리 유리
- 고차원(1536+) + 대량이면 인덱스 빌드 느림 (`halfvec`, 차원 축소로 완화). HNSW 인덱스 차원 제한 2000
- 필터 + ANN 조합 시 recall 저하 가능 → 0.8.0+ `hnsw.iterative_scan` 으로 개선

**비교**

| | pgvector | Qdrant | Milvus | Pinecone |
|---|---|---|---|---|
| 규모 | ~수백만 | 수억 | 수십억 | 수억 |
| 필터+ANN | 보통 | 강함 | 강함 | 강함 |
| 하이브리드 FTS | 내장 | 내장(sparse) | 내장 | 내장 |
| 운영 | 기존 PG | docker 1개 | 복잡(etcd, minio) | 관리형 유료 |

**규모 감각**: 영상 1만 개 × 1시간 × 5초 클립 = 720만 시각 벡터(경계선). 자막은 시간당 청크 ~80개 = 80만(여유).
768차원 float32 = 3KB/행 → 700만 행 ≈ 21GB + 인덱스. `shared_buffers`에 인덱스 안 들어가면 느려짐.

**전환 신호**: HNSW 빌드 수 시간 초과 / 필터 걸면 recall 급락 / PG 메모리 압박.
**중간 선택지**: `pgvectorscale`(Timescale, DiskANN) → PG 유지하며 수천만 벡터.
**버전**: PG 15+, pgvector 0.8+ 권장. RDS/Aurora/Cloud SQL/AlloyDB/Azure 전부 기본 제공.

**결론**: PoC는 전부 pgvector. 데이터 쌓인 뒤 측정하고 판단.

## 3. 벡터 추출 과정

영상 → 조각 → 모델 → float 배열.

1. **분할**: PySceneDetect 씬 감지 또는 고정 간격(5초). 조각마다 start/end 기록
2. **프레임 샘플링**: 조각당 1~N장. 224~384px 충분
3. **시각 임베딩**: 프레임 → 이미지 인코더 → 벡터. 여러 장이면 평균(mean pooling) → 조각당 1개
4. **자막 임베딩** (선택): Whisper 전사 → 텍스트 임베딩. 시각과 **다른 공간**이므로 별도 테이블
5. **정규화 + 저장**: L2 normalize → pgvector INSERT

**비용 감각**: SigLIP ViT-B, M-시리즈 CPU 프레임당 ~50ms. 1시간 영상 720장 ≈ 40초. GPU면 수 초.
**정확도 올리기**: 조각당 프레임 3~5장 평균, 큰 모델(ViT-L, SO400M), 자막 벡터 병행.

## 4. 자막 벡터 검색 = 텍스트 RAG

1. **자막 확보**: SRT/VTT 있으면 파싱(타임스탬프 공짜). 없으면 faster-whisper (`large-v3` 한국어 좋음)
2. **청킹**: Whisper 세그먼트(2~10초)는 너무 짧음 → 30~60초 또는 200~300자, 10~20% overlap. 첫 start·마지막 end 보존
3. **임베딩**: 한국어 가능 모델 필수. Qwen3-Embedding, bge-m3, text-embedding-3
4. **하이브리드**: 벡터(의미) + FTS/BM25(고유명사) → RRF 합산. PG 한국어 FTS 약함 → `pg_bigm` 또는 mecab

교육·강의 영상이면 자막 검색만으로 80% 해결. 시각은 그 다음.

## 5. 모델 선택: Qwen3

- **Qwen3-Embedding (텍스트)**: 0.6B / 4B / 8B. 8B가 MTEB 다국어 1위권, 한국어 bge-m3보다 좋음. 0.6B도 bge-m3 수준.
  차원 1024 / 2560 / 4096, MRL로 축소 가능(pgvector 2000 제한). 질의에 instruction 붙이면 정확도↑. `Qwen3-Reranker` 세트. Apache-2.
- **Qwen3-VL-Embedding (시각)**: 이미지/영상 → 벡터, 텍스트 공동 공간. 한국어 질의 자연스러움. SigLIP 대비 10배+ 느림 → 씬 단위로 줄여야 현실적.
- **Qwen3-VL (생성형)**: 프레임 → 캡션 → 텍스트 임베딩. 시각·자막 공간 통일되는 대신 캡션 비용 최대.

**PoC 조합**: 자막 = Qwen3-Embedding-0.6B, 시각 = SigLIP2 (품질 부족하면 Qwen3-VL-Embedding).

## 6. 검색 범위: 영상 하나 vs 전체

같은 테이블, `WHERE` 차이.

- 전체: `ORDER BY embedding <=> $q LIMIT 20`
- 영상 하나: `WHERE video_id = ... ORDER BY ...`
- 영상 단위 결과: 클립 top-200 → `GROUP BY video_id` max 집계

주의: 필터 선택도 높으면(영상 1개 = 전체 0.01%) HNSW 후보 부족 → `hnsw.iterative_scan = relaxed_order` 또는 `video_id` 인덱스로 전수 스캔.
전체 검색은 긴 영상이 결과 독식 → 영상당 결과 수 제한 또는 max 집계.

## 7. 한국어 질의 가능 여부

- **자막 벡터**: 다국어 모델이면 됨. 한↔영 cross-lingual도 어느 정도
- **시각 벡터**: 모델 따라 다름
  - OpenAI CLIP 원본: 영어 전용, 한국어 안 됨
  - SigLIP(webli): 영어 중심, 불안정
  - **SigLIP 2**: 다국어 학습, 한국어 쓸 만함 ← PoC 채택
  - Qwen3-VL-Embedding: 가장 좋음, 무거움
- 우회: 질의만 번역("강아지 뛰는 장면" → "a dog running") 후 영어 CLIP. 실용적.
- 테스트: 한국어 질의 vs 영어 번역 질의 top-10 겹침 70% 이상이면 한국어 직접 사용 OK

## 8. 질의 종류

저장은 프레임(이미지) 임베딩. 질의는 같은 공간에 넣을 수 있으면 뭐든:

| 질의 | 방법 |
|---|---|
| 텍스트 | 텍스트 인코더 |
| 이미지 | 이미지 인코더 (텍스트보다 정확) |
| 영상 클립 | 프레임 벡터 평균 → 유사 영상·중복 탐지 |
| 저장된 구간 | more-like-this |

## 9. 영상 벡터 vs 자막 벡터 — 별개

| | 영상(시각) | 자막 |
|---|---|---|
| 원본 | 프레임 | 전사 텍스트 |
| 모델 | SigLIP2 | Qwen3-Embedding |
| 차원 | 768 | 1024 |
| 잡는 것 | 화면에 보이는 것 | 말한 내용 |

모델 다르면 벡터 공간 다름 → cosine 비교 무의미 → 같은 컬럼 불가. 테이블 두 개, `video_id + start/end`가 연결 고리.

**결합**: 질의 → 시각 top-50 + 자막 top-50 → (video_id, 구간) 묶기 → RRF `1/(60+rank)` 합산 → top-10.
**단순화**: 질의 종류로 라우팅(장면 질의 → 시각, 대사 질의 → 자막).
**예외**: 프레임 캡션 생성 → 텍스트 공간 통일 → 테이블 하나.

## 10. PoC 결정 사항

- Docker compose: `pgvector/pgvector:pg17` + FastAPI(CPU)
- 시각: SigLIP2 ViT-B/16, 5초 클립, 클립당 2프레임 평균 → `clip_embeddings vector(768)`
- 자막: 사이드카 .srt/.vtt 우선, 없으면 faster-whisper small → 45초 청크 → Qwen3-Embedding-0.6B → `subtitle_chunks vector(1024)`
- 검색: 시각(텍스트/이미지) · 자막 · 통합(RRF) 3종, `video_id` 필터
- 데스크톱: pywebview가 API의 웹 UI를 네이티브 창으로 감쌈
- 미구현(다음 단계): 씬 감지, FTS 하이브리드, 리랭커, 영상 단위 집계 뷰
