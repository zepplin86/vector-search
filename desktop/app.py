"""데스크톱 셸. API(docker)가 띄운 UI를 네이티브 창(WKWebView)으로 감쌈.
'영상 추가' 버튼 → 네이티브 파일 다이얼로그 → ./media 로 복사 (같은 이름 .srt/.vtt 있으면 같이)."""
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import webview

URL = os.environ.get("API_URL", "http://localhost:8000")
MEDIA_DIR = Path(os.environ.get("MEDIA_DIR", Path(__file__).resolve().parent.parent / "media"))
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")


class Api:
    def pick_video(self):
        try:
            win = webview.windows[0]
            dialog = getattr(webview, "FileDialog", None)
            kind = dialog.OPEN if dialog else webview.OPEN_DIALOG  # pywebview 5 / 4
            picked = win.create_file_dialog(
                kind, allow_multiple=True,
                file_types=("Video (*.mp4;*.mov;*.mkv;*.webm;*.m4v;*.avi)", "All files (*.*)"),
            ) or []
            MEDIA_DIR.mkdir(parents=True, exist_ok=True)
            added = []
            for src in map(Path, picked):
                if src.suffix.lower() not in VIDEO_EXTS:
                    continue
                shutil.copy2(src, MEDIA_DIR / src.name)
                added.append(src.name)
                for ext in (".srt", ".vtt"):
                    sc = src.with_suffix(ext)
                    if sc.is_file():
                        shutil.copy2(sc, MEDIA_DIR / sc.name)
            return {"added": added}
        except Exception as e:  # noqa: BLE001
            return {"added": [], "error": str(e)}


def wait_api(timeout: float = 60) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"{URL}/api/health", timeout=2)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


if __name__ == "__main__":
    if not wait_api():
        print(f"API not reachable at {URL}. run `docker compose up -d` first.", file=sys.stderr)
        sys.exit(1)
    webview.create_window(
        "Video Vector Search PoC", URL, js_api=Api(),
        width=1320, height=880, min_size=(900, 600),
    )
    webview.start()
