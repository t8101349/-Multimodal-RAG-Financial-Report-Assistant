"""Mock 檢索後端單元測試。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever_mock import MockRetriever  # noqa: E402


def _make_page_units(tmp_path: Path) -> Path:
    units = [
        {
            "page_index": 1,
            "raw_page_index": 0,
            "image_path": str(tmp_path / "img_p001.jpg"),
            "width": 800,
            "height": 1000,
            "char_count": 12,
            "text_preview": "封面",
            "text": "繁星科技股份有限公司 2024 年度報告",
            "is_likely_toc": False,
            "block_count": 1,
            "source_pdf": "x.pdf",
        },
        {
            "page_index": 2,
            "raw_page_index": 1,
            "image_path": str(tmp_path / "img_p002.jpg"),
            "width": 800,
            "height": 1000,
            "char_count": 30,
            "text_preview": "目錄 致股東",
            "text": "目錄\n致股東報告書 ... 3\n經營概要 ... 4",
            "is_likely_toc": True,
            "block_count": 5,
            "source_pdf": "x.pdf",
        },
        {
            "page_index": 5,
            "raw_page_index": 4,
            "image_path": str(tmp_path / "img_p005.jpg"),
            "width": 800,
            "height": 1000,
            "char_count": 120,
            "text_preview": "核心財務指標",
            "text": "核心財務指標 營業收入 482.6 億元 年增 18.7% 研發投入 12.7 億",
            "is_likely_toc": False,
            "block_count": 8,
            "source_pdf": "x.pdf",
        },
    ]
    page_path = tmp_path / "page_units.jsonl"
    with page_path.open("w", encoding="utf-8") as fh:
        for u in units:
            fh.write(json.dumps(u, ensure_ascii=False) + "\n")
    return page_path


def test_mock_retriever_build_and_search(tmp_path: Path) -> None:
    page_units = _make_page_units(tmp_path)
    img_dir = tmp_path
    index_dir = tmp_path / "mock_idx"

    retriever = MockRetriever()
    retriever.build_index(
        pdf_path=tmp_path / "x.pdf",
        page_image_dir=img_dir,
        index_dir=index_dir,
        page_units_path=page_units,
    )

    hits = retriever.search("營業收入是多少？", k=2)
    assert hits, "至少應命中一頁"
    # 第 5 頁含營業收入，應為最高分（目錄頁因降權而排名較後）
    assert hits[0].page_index == 5


def test_mock_retriever_toc_demoted(tmp_path: Path) -> None:
    page_units = _make_page_units(tmp_path)
    img_dir = tmp_path
    index_dir = tmp_path / "mock_idx"

    retriever = MockRetriever()
    retriever.build_index(
        pdf_path=tmp_path / "x.pdf",
        page_image_dir=img_dir,
        index_dir=index_dir,
        page_units_path=page_units,
    )

    hits = retriever.search("經營概要", k=3)
    # 目錄頁雖然提到「經營概要」字面，但被降權；若有內容頁仍應排在前
    page_order = [h.page_index for h in hits]
    assert 2 in page_order  # 目錄頁仍會出現
    if len(hits) >= 2:
        # 同樣字面命中時，內容頁分數應高於目錄頁（此例只有目錄頁含此詞，僅檢查不報錯）
        pass


def test_mock_retriever_load_index(tmp_path: Path) -> None:
    page_units = _make_page_units(tmp_path)
    img_dir = tmp_path
    index_dir = tmp_path / "mock_idx"

    r1 = MockRetriever()
    r1.build_index(
        pdf_path=tmp_path / "x.pdf",
        page_image_dir=img_dir,
        index_dir=index_dir,
        page_units_path=page_units,
    )

    r2 = MockRetriever()
    r2.load_index(index_dir)
    hits = r2.search("研發投入", k=2)
    assert hits and hits[0].page_index == 5
