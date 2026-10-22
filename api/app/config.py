import os


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


DATABASE_URL = _env("DATABASE_URL", "postgresql://vec:vec@localhost:5433/vec")
MEDIA_DIR = _env("MEDIA_DIR", "./media")
DATA_DIR = _env("DATA_DIR", "./data")

VISUAL_MODEL = _env("VISUAL_MODEL", "ViT-B-16-SigLIP2")
VISUAL_PRETRAINED = _env("VISUAL_PRETRAINED", "webli")
TEXT_MODEL = _env("TEXT_MODEL", "Qwen/Qwen3-Embedding-0.6B")
WHISPER_MODEL = _env("WHISPER_MODEL", "small")
WHISPER_LANG = _env("WHISPER_LANG", "ko") or None

CLIP_SEC = float(_env("CLIP_SEC", "5"))
FRAMES_PER_CLIP = int(_env("FRAMES_PER_CLIP", "2"))
CHUNK_SEC = float(_env("CHUNK_SEC", "45"))

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
