"""全域設定模組。

集中管理路徑、模型、API 設定，可由 .env 覆寫。
所有對外設定皆透過 ``settings`` 取得，避免在程式碼中散落硬編字串。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # 若未安裝 python-dotenv 也不應阻塞匯入
    pass


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
PAGE_IMAGE_DIR = DATA_DIR / "page_images"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
REPORT_DIR = DATA_DIR / "reports"

for _d in (PDF_DIR, PAGE_IMAGE_DIR, PROCESSED_DIR, EVAL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(key, default)
    if isinstance(value, str) and value.strip() == "":
        return default
    return value


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """執行時期設定容器。"""

    vlm_base_url: str = field(default_factory=lambda: _env("VLM_BASE_URL", "http://localhost:8000/v1") or "")
    vlm_api_key: str = field(default_factory=lambda: _env("VLM_API_KEY", "") or "")
    vlm_model: str = field(default_factory=lambda: _env("VLM_MODEL", "qwen2.5-vl-72b-instruct") or "")

    retriever_backend: str = field(default_factory=lambda: (_env("RETRIEVER_BACKEND", "auto") or "auto").lower())

    colpali_model_path: str = field(default_factory=lambda: _env("COLPALI_MODEL_PATH", "./models/colpali-v1.2-merged") or "")
    colpali_index_name: str = field(default_factory=lambda: _env("COLPALI_INDEX_NAME", "finance_report") or "finance_report")

    clip_model_name: str = field(default_factory=lambda: _env("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32") or "openai/clip-vit-base-patch32")

    hf_offline: bool = field(default_factory=lambda: (_env("HF_HUB_OFFLINE", "0") or "0") == "1")
    hf_endpoint: str = field(default_factory=lambda: _env("HF_ENDPOINT", "https://huggingface.co") or "")

    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 4))
    page_render_dpi: int = field(default_factory=lambda: _env_int("PAGE_RENDER_DPI", 200))

    def vlm_ready(self) -> bool:
        return bool(self.vlm_base_url and self.vlm_api_key)


settings = Settings()


def reload_settings() -> Settings:
    """重新從環境變數讀取設定，主要供測試使用。"""
    global settings
    settings = Settings()
    return settings
