"""環境檢查腳本：列出各套件可用性與設定狀態。

執行：
    python scripts/check_setup.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from src.retriever_factory import list_available_backends  # noqa: E402


_PACKAGES = [
    ("fitz", "PyMuPDF", "PDF 渲染"),
    ("PIL", "Pillow", "影像處理"),
    ("reportlab", "reportlab", "測試 PDF 產生"),
    ("byaldi", "byaldi", "ColPali 視覺檢索（主路徑）"),
    ("torch", "torch", "CLIP 備援必備"),
    ("transformers", "transformers", "CLIP 備援必備"),
    ("requests", "requests", "VLM 呼叫"),
    ("dotenv", "python-dotenv", "讀取 .env"),
    ("pydantic", "pydantic", "型別校驗"),
]


def _check(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main() -> int:
    print("=== 套件可用性 ===")
    missing: list[str] = []
    for mod, pkg, desc in _PACKAGES:
        ok = _check(mod)
        mark = "OK" if ok else "缺"
        print(f"  [{mark}] {pkg:<18} - {desc}")
        if not ok:
            missing.append(pkg)

    print("\n=== 視覺檢索後端 ===")
    for name, ok in list_available_backends().items():
        mark = "OK" if ok else "缺"
        print(f"  [{mark}] {name}")

    print("\n=== VLM 設定 ===")
    print(f"  base_url = {settings.vlm_base_url or '(未設定)'}")
    print(f"  model    = {settings.vlm_model or '(未設定)'}")
    print(f"  api_key  = {'已設定' if settings.vlm_api_key else '未設定'}")

    print("\n=== 路徑 ===")
    print(f"  ColPali 模型路徑：{settings.colpali_model_path}")
    print(f"  HF 端點：       {settings.hf_endpoint}")

    if missing:
        print("\n下列套件尚未安裝，請執行 `pip install -r requirements.txt`：")
        for m in missing:
            print(f"  - {m}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
