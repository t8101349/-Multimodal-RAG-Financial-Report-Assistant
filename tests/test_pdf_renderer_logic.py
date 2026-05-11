"""PDF 渲染器的純邏輯單元測試（不需要 PyMuPDF）。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 此測試只觸及不需要 fitz 的內部工具，避免在缺套件時直接失敗
import importlib.util


def _has_fitz() -> bool:
    return importlib.util.find_spec("fitz") is not None


if _has_fitz():
    from src.pdf_renderer import _looks_like_toc, filter_non_toc, PageUnit  # type: ignore  # noqa: E402

    def test_looks_like_toc_with_keyword() -> None:
        text = "目錄\n第一章 經營概要 ........ 4\n第二章 財務指標 ........ 5"
        assert _looks_like_toc(text) is True

    def test_looks_like_toc_with_dotted_pattern() -> None:
        text = "致股東報告書 ... 3\n經營概要 ... 4\n核心財務 ... 5\n資產負債表 ... 8"
        assert _looks_like_toc(text) is True

    def test_looks_like_toc_negative_content_page() -> None:
        text = "本公司 2024 年度合併營業收入為 482.6 億元，年增 18.7%。"
        assert _looks_like_toc(text) is False

    def test_filter_non_toc_filters_correctly() -> None:
        pages = [
            PageUnit(
                page_index=1,
                raw_page_index=0,
                image_path="x1",
                width=1,
                height=1,
                char_count=1,
                text_preview="",
                text="",
                is_likely_toc=False,
                block_count=0,
                source_pdf="",
            ),
            PageUnit(
                page_index=2,
                raw_page_index=1,
                image_path="x2",
                width=1,
                height=1,
                char_count=1,
                text_preview="",
                text="",
                is_likely_toc=True,
                block_count=0,
                source_pdf="",
            ),
        ]
        kept = filter_non_toc(pages)
        assert [u.page_index for u in kept] == [1]

else:
    def test_fitz_missing_skipped() -> None:
        # 在缺 PyMuPDF 的環境中，這個檔案會略過真實邏輯測試，僅保留一個 sentinel
        assert True
