"""評測模組單元測試。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import (  # noqa: E402
    EvalCase,
    extract_cited_pages,
    keyword_hits,
)


def test_extract_cited_pages_basic() -> None:
    text = "依第 5 頁與第 6 頁的核心財務指標表，可確認 2024 年營收 482.6 億元。"
    assert extract_cited_pages(text) == [5, 6]


def test_extract_cited_pages_full_width_digits() -> None:
    text = "請參考第 ５ 頁的圖表。"
    assert extract_cited_pages(text) == [5]


def test_extract_cited_pages_dedupes_and_sorts() -> None:
    text = "第 7 頁、第 5 頁、第 5 頁均有提及。"
    assert extract_cited_pages(text) == [5, 7]


def test_keyword_hits_skips_empty() -> None:
    answer = "2024 年合併營收 482.6 億元，年增 18.7%"
    hits = keyword_hits(answer, ["482.6", "", "億", "完全沒命中"])
    assert "482.6" in hits and "億" in hits and "完全沒命中" not in hits


def test_eval_case_from_dict_round_trip() -> None:
    raw = {
        "qid": "Q99",
        "question": "問句？",
        "expected_pages": [3, 4],
        "expected_keywords": ["甲", "乙"],
    }
    case = EvalCase.from_dict(raw)
    assert case.qid == "Q99"
    assert case.expected_pages == [3, 4]
    assert case.expected_keywords == ["甲", "乙"]
