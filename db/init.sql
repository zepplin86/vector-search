CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE videos (
  id              text PRIMARY KEY,          -- 파일명
  duration        real,
  visual_status   text NOT NULL DEFAULT 'none',   -- none|queued|running|done|error
  subtitle_status text NOT NULL DEFAULT 'none',
  visual_count    int  NOT NULL DEFAULT 0,
  subtitle_count  int  NOT NULL DEFAULT 0,
  error           text,
  updated_at      timestamptz NOT NULL DEFAULT now()
);

-- 영상(시각) 벡터: 프레임 → SigLIP2 (768차원)
CREATE TABLE clip_embeddings (
  id         bigserial PRIMARY KEY,
  video_id   text NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  start_sec  real NOT NULL,
  end_sec    real NOT NULL,
  thumb_path text,
  embedding  vector(768) NOT NULL
);
CREATE INDEX clip_embeddings_video_idx ON clip_embeddings (video_id, start_sec);
CREATE INDEX clip_embeddings_hnsw ON clip_embeddings USING hnsw (embedding vector_cosine_ops);

-- 자막 벡터: 전사 텍스트 청크 → Qwen3-Embedding-0.6B (1024차원)
CREATE TABLE subtitle_chunks (
  id         bigserial PRIMARY KEY,
  video_id   text NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  start_sec  real NOT NULL,
  end_sec    real NOT NULL,
  text       text NOT NULL,
  embedding  vector(1024) NOT NULL
);
CREATE INDEX subtitle_chunks_video_idx ON subtitle_chunks (video_id, start_sec);
CREATE INDEX subtitle_chunks_hnsw ON subtitle_chunks USING hnsw (embedding vector_cosine_ops);
