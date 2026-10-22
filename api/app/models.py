"""모델 로더. 시각(SigLIP2) / 자막(Qwen3-Embedding) / ASR(faster-whisper) 각각 별도 모델, 별도 벡터 공간."""
import threading

import numpy as np
import torch
from PIL import Image

from . import config

torch.set_num_threads(max(1, torch.get_num_threads()))

_lock = threading.Lock()
_visual = None
_text = None
_asr = None


def visual():
    global _visual
    with _lock:
        if _visual is None:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                config.VISUAL_MODEL, pretrained=config.VISUAL_PRETRAINED
            )
            tokenizer = open_clip.get_tokenizer(config.VISUAL_MODEL)
            model.eval()
            _visual = (model, preprocess, tokenizer)
    return _visual


def text_model():
    global _text
    with _lock:
        if _text is None:
            from sentence_transformers import SentenceTransformer

            _text = SentenceTransformer(
                config.TEXT_MODEL,
                device="cpu",
                tokenizer_kwargs={"padding_side": "left"},
            )
    return _text


def asr():
    global _asr
    with _lock:
        if _asr is None:
            from faster_whisper import WhisperModel

            _asr = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    return _asr


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


# ---------- 시각 ----------
@torch.no_grad()
def embed_images(paths: list[str], batch_size: int = 16) -> np.ndarray:
    model, preprocess, _ = visual()
    out = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i : i + batch_size]])
        v = model.encode_image(batch)
        out.append(v.float().cpu().numpy())
    return _l2(np.concatenate(out, axis=0))


@torch.no_grad()
def embed_visual_text(query: str) -> np.ndarray:
    model, _, tokenizer = visual()
    v = model.encode_text(tokenizer([query]))
    return _l2(v.float().cpu().numpy())[0]


@torch.no_grad()
def visual_logit_params() -> tuple[float, float]:
    """SigLIP 은 sigmoid 손실로 학습됨 → p(match) = sigmoid(scale * cos + bias). cos 자체는 보정 안 된 값."""
    model, _, _ = visual()
    scale = float(model.logit_scale.exp())
    bias = float(getattr(model, "logit_bias", torch.tensor(0.0)))
    return scale, bias


def embed_visual_image(path: str) -> np.ndarray:
    return embed_images([path])[0]


# ---------- 자막 ----------
def embed_passages(texts: list[str]) -> np.ndarray:
    m = text_model()
    return np.asarray(m.encode(texts, normalize_embeddings=True, batch_size=8, convert_to_numpy=True))


def embed_text_query(query: str) -> np.ndarray:
    m = text_model()
    return np.asarray(m.encode([query], prompt_name="query", normalize_embeddings=True, convert_to_numpy=True))[0]
