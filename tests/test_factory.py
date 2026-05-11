"""檢索後端工廠單元測試。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever_factory import get_retriever, list_available_backends  # noqa: E402


def test_list_available_backends_has_mock() -> None:
    status = list_available_backends()
    assert "mock" in status and status["mock"] is True
    assert "byaldi" in status
    assert "clip" in status


def test_get_retriever_mock_forced() -> None:
    r = get_retriever(backend="mock")
    assert r.name == "mock"


def test_get_retriever_auto_falls_back_to_mock() -> None:
    # 在乾淨環境中，auto 應退到第一個可用後端；至少 mock 永遠可用
    r = get_retriever(backend="auto")
    assert r.name in {"byaldi", "clip", "mock"}
