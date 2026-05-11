"""建立索引：渲染 PDF 並建立視覺檢索索引。

執行：
    python scripts/build_index.py --pdf data/pdfs/sample_finance_report.pdf

流程：
    1. 渲染 PDF 至 data/page_images/
    2. 寫入 data/processed/page_units.jsonl
    3. 依 RETRIEVER_BACKEND 建立對應索引到 data/processed/<backend>_index/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from src.pdf_renderer import render_pdf  # noqa: E402
from src.retriever_factory import get_retriever, list_available_backends  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="建立多模態 RAG 索引")
    parser.add_argument(
        "--pdf",
        default=str(PROJECT_ROOT / "data" / "pdfs" / "sample_finance_report.pdf"),
        help="要索引的 PDF 路徑",
    )
    parser.add_argument(
        "--page-image-dir",
        default=str(PROJECT_ROOT / "data" / "page_images"),
        help="頁面圖檔輸出資料夾",
    )
    parser.add_argument(
        "--processed-dir",
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="中繼與索引存放資料夾",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=settings.page_render_dpi,
        help="渲染解析度（預設 200）",
    )
    parser.add_argument(
        "--backend",
        default=settings.retriever_backend,
        choices=["auto", "byaldi", "clip", "mock"],
        help="檢索後端",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[build_index] 找不到 PDF：{pdf_path}", file=sys.stderr)
        return 2

    page_image_dir = Path(args.page_image_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    page_units_path = processed_dir / "page_units.jsonl"

    print(f"[build_index] 環境後端能力：{list_available_backends()}")

    print(f"[build_index] 渲染 PDF（DPI={args.dpi}）...")
    render = render_pdf(
        pdf_path,
        page_image_dir,
        dpi=args.dpi,
        metadata_path=page_units_path,
    )
    print(f"  -> 渲染完成，共 {render.page_count} 頁")
    toc_pages = [u.page_index for u in render.units if u.is_likely_toc]
    print(f"  -> 疑似目錄頁：{toc_pages or '無'}")
    print(f"  -> 中繼檔：{page_units_path}")

    retriever = get_retriever(
        backend=args.backend,
        colpali_model_path=settings.colpali_model_path,
        colpali_index_name=settings.colpali_index_name,
        clip_model_name=settings.clip_model_name,
        hf_offline=settings.hf_offline,
        hf_endpoint=settings.hf_endpoint,
    )
    index_dir = processed_dir / f"{retriever.name}_index"
    print(f"[build_index] 使用後端 {retriever.name}，索引輸出：{index_dir}")

    if retriever.name == "mock":
        retriever.build_index(
            pdf_path=pdf_path,
            page_image_dir=page_image_dir,
            index_dir=index_dir,
            page_units_path=page_units_path,
        )
    else:
        retriever.build_index(
            pdf_path=pdf_path,
            page_image_dir=page_image_dir,
            index_dir=index_dir,
        )

    summary = {
        "pdf": str(pdf_path).replace("\\", "/"),
        "page_count": render.page_count,
        "page_units_path": str(page_units_path).replace("\\", "/"),
        "index_dir": str(index_dir).replace("\\", "/"),
        "backend": retriever.name,
        "toc_pages": toc_pages,
    }
    (processed_dir / "rag_index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[build_index] 完成，摘要：{processed_dir / 'rag_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
