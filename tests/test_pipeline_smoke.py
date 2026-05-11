"""端到端煙霧測試 - 使用 MockRetriever + MockVLMClient。

此測試不會載入任何重型 ML 模型，因此可在任何環境執行；
目的是檢驗 :mod:`pipeline` 把檢索、目錄頁過濾、提示組合、
VLM 呼叫與遙測紀錄串起來的能力。
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pdf_renderer import PageUnit  # noqa: E402
from src.pipeline import MultimodalRAGPipeline  # noqa: E402
from src.retriever_mock import MockRetriever  # noqa: E402
from src.vlm_client import MockVLMClient  # noqa: E402


def _write_jpeg(path: Path, payload: bytes = b"FAKEJPEGCONTENT") -> None:
    path.write_bytes(payload)


def _prepare_assets(tmp_path: Path) -> tuple[Path, Path, list[PageUnit]]:
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    # 三張假頁面圖
    img1 = img_dir / "x_p001.jpg"
    img2 = img_dir / "x_p002.jpg"
    img5 = img_dir / "x_p005.jpg"
    for p in (img1, img2, img5):
        _write_jpeg(p)

    units = [
        PageUnit(
            page_index=1,
            raw_page_index=0,
            image_path=str(img1),
            width=800,
            height=1000,
            char_count=10,
            text_preview="封面",
            text="繁星科技 2024 年度報告",
            is_likely_toc=False,
            block_count=2,
            source_pdf=str(tmp_path / "x.pdf"),
        ),
        PageUnit(
            page_index=2,
            raw_page_index=1,
            image_path=str(img2),
            width=800,
            height=1000,
            char_count=40,
            text_preview="目錄",
            text="目錄\n致股東報告書 ... 3\n核心財務指標 ... 5",
            is_likely_toc=True,
            block_count=6,
            source_pdf=str(tmp_path / "x.pdf"),
        ),
        PageUnit(
            page_index=5,
            raw_page_index=4,
            image_path=str(img5),
            width=800,
            height=1000,
            char_count=80,
            text_preview="核心財務指標",
            text="核心財務指標 營業收入 482.6 億元 研發投入 12.7 億",
            is_likely_toc=False,
            block_count=10,
            source_pdf=str(tmp_path / "x.pdf"),
        ),
    ]

    # 建立 mock 索引所需的 page_units.jsonl
    units_path = tmp_path / "page_units.jsonl"
    with units_path.open("w", encoding="utf-8") as fh:
        for u in units:
            fh.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")

    return img_dir, units_path, units


def test_pipeline_end_to_end_smoke(tmp_path: Path) -> None:
    img_dir, units_path, units = _prepare_assets(tmp_path)
    index_dir = tmp_path / "mock_idx"

    retriever = MockRetriever()
    retriever.build_index(
        pdf_path=tmp_path / "x.pdf",
        page_image_dir=img_dir,
        index_dir=index_dir,
        page_units_path=units_path,
    )

    vlm = MockVLMClient(
        canned_answer="**結論**：2024 年營收為 482.6 億元（第 5 頁）。"
    )
    pipeline = MultimodalRAGPipeline(
        retriever=retriever,
        vlm=vlm,
        page_units=units,
        default_top_k=2,
        filter_toc=True,
    )

    result = pipeline.query("2024 年的營業收入是多少？")
    assert 5 in result.retrieved_pages
    # 目錄頁不應出現在最終召回（已被 pipeline 過濾）
    assert 2 not in result.retrieved_pages
    assert result.answer.startswith("**結論**")
    # MockVLMClient 的 latency_ms 是固定值；只驗證遙測欄位存在即可
    assert result.latency_total_ms >= 0
    assert result.latency_vlm_ms >= 0
    # 至少送出一筆 chat
    assert vlm.calls
    # user content 中應含至少一張圖片
    user_content = vlm.calls[-1]["messages"][1]["content"]
    image_blocks = [c for c in user_content if c.get("type") == "image_url"]
    assert len(image_blocks) >= 1


def test_pipeline_appends_log(tmp_path: Path) -> None:
    img_dir, units_path, units = _prepare_assets(tmp_path)
    index_dir = tmp_path / "mock_idx"

    retriever = MockRetriever()
    retriever.build_index(
        pdf_path=tmp_path / "x.pdf",
        page_image_dir=img_dir,
        index_dir=index_dir,
        page_units_path=units_path,
    )
    vlm = MockVLMClient()
    pipeline = MultimodalRAGPipeline(retriever, vlm, page_units=units, default_top_k=2)

    log_path = tmp_path / "log.jsonl"
    result = pipeline.query("研發投入是多少？")
    pipeline.append_log(result, log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["question"] == "研發投入是多少？"
    assert data["retrieved_pages"]
