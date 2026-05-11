"""提示詞模組單元測試。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    build_image_payloads,
    build_prompt,
    build_user_text,
)


def test_system_prompt_uses_traditional_chinese_keywords() -> None:
    # 系統提示應包含若干關鍵字，避免改寫時不慎丟失
    must_contain = [
        "繁體中文",
        "目錄頁",
        "結論",
        "證據頁碼",
        "不確定",
        "趨勢",
    ]
    for kw in must_contain:
        assert kw in SYSTEM_PROMPT, f"系統提示缺少關鍵字：{kw}"


def test_build_user_text_handles_no_pages() -> None:
    text = build_user_text("營業收入多少？", [])
    assert "0" in text or "無頁面" in text
    assert "營業收入多少？" in text


def test_build_user_text_with_toc_warning() -> None:
    text = build_user_text("研發投入？", [3, 5, 6], contains_possible_toc=True)
    assert "第 3 頁" in text
    assert "第 5 頁" in text
    assert "第 6 頁" in text
    assert "目錄" in text


def test_build_user_text_without_toc_warning() -> None:
    text = build_user_text("研發投入？", [3, 5, 6], contains_possible_toc=False)
    assert "目錄頁" not in text
    assert "第 3 頁" in text


def test_build_image_payloads_skips_empty() -> None:
    payloads = build_image_payloads(["abc", "", "def"], detail="high")
    assert len(payloads) == 2
    for p in payloads:
        assert p["type"] == "image_url"
        assert p["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert p["image_url"]["detail"] == "high"


def test_build_prompt_messages_shape() -> None:
    bundle = build_prompt(
        "2024 年營收？",
        page_indices=[5, 6],
        image_b64_list=["aGVsbG8="],
        contains_possible_toc=False,
    )
    msgs = bundle.to_messages()
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    user_content = msgs[1]["content"]
    assert user_content[0]["type"] == "text"
    assert user_content[1]["type"] == "image_url"
