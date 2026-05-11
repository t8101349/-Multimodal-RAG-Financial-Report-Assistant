"""多模態 RAG 提示詞模板。

對應原文「生成階段的抗干擾約束」設計：
    - 角色：專業 CFO / 投研 / 審計助手
    - 任務：依命中頁面回答具體問題
    - 抗干擾：忽略目錄、封面、無數據頁
    - 證據偏好：優先表格、圖表與明確數值
    - 不確定性：若證據不足明確說明
    - 輸出格式：結論 / 證據 / 頁碼 / 趨勢解讀
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


SYSTEM_PROMPT = """你是一位資深的企業財報分析助手，服務對象為投研、CFO 與審計人員。
請依使用者提供的「財報頁面截圖」回答問題。請嚴格遵守以下規則：

【證據規則】
1. 優先依賴含有具體數值、表格或圖表的頁面，忽略目錄頁、封面、版權聲明與致辭等噪聲頁面。
2. 引用時請以「(第 X 頁)」標註頁碼；多頁綜合時請逐項標註。
3. 若某項結論需要綜合多頁資料，請明確說明對應證據來源。

【數值規則】
1. 抄寫數字時保留原始單位（元 / 千元 / 百萬元 / 億元）與時期（如本期 / 上期 / 本年度）。
2. 區分同比（YoY）與環比（QoQ），不可混用。
3. 表格中若有「合計 / 小計」列，請優先採用作為彙總值。

【誠實規則】
1. 若提供的頁面不足以回答問題，請明確說「依目前頁面證據不足以判斷 X」。
2. 不得自行編造數字或趨勢；不確定的細節請說明不確定來源。

【輸出格式】（一律使用繁體中文，依下列段落輸出）
- **結論**：以一句話直接回答問題。
- **關鍵數值**：列出對齊到原始單位的核心數字。
- **趨勢解讀**：若問題涉及變化，請依圖表或多期數據解讀趨勢；否則填「不適用」。
- **證據頁碼**：列出本回答所依據的頁碼，例如 `第 5 頁、第 6 頁`。
- **不確定性**：列出尚未能確認的部分；若無，填「無」。
"""


@dataclass
class PromptBundle:
    """打包後的 chat-completions 訊息結構（OpenAI 相容）。"""

    system: str
    user_text: str
    image_payloads: List[dict]

    def to_messages(self) -> List[dict]:
        user_content: List[dict] = [{"type": "text", "text": self.user_text}]
        user_content.extend(self.image_payloads)
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_content},
        ]


def build_user_text(
    question: str,
    page_indices: Iterable[int],
    contains_possible_toc: bool = False,
) -> str:
    """組合使用者訊息的文字部分。

    - 明確告知模型總共有幾頁、各頁人類頁碼
    - 若召回中可能含目錄頁，提醒模型忽略
    """
    pages = list(page_indices)
    page_label = "、".join(f"第 {p} 頁" for p in pages) if pages else "（無頁面）"
    head = (
        f"以下提供 {len(pages)} 張財報頁面截圖（依序為 {page_label}），"
        "請以這些頁面為唯一證據回答下列問題。"
    )
    if contains_possible_toc:
        head += "其中可能混有目錄頁，請忽略目錄頁、直接根據含具體數據的頁面回答。"

    body = (
        f"\n\n問題：{question}\n\n"
        "請依系統提示要求的格式輸出。若多頁資料相互矛盾，請以表格 / 圖表頁為準。"
    )
    return head + body


def build_image_payloads(
    image_b64_list: Iterable[str],
    detail: str = "high",
) -> List[dict]:
    """組合多張圖片成 OpenAI 相容 content 陣列。

    參數：
        image_b64_list: 已編碼的 base64 字串列表（不含 data: 前綴）
        detail: ``low`` / ``high`` / ``auto``
    """
    payloads: List[dict] = []
    for b64 in image_b64_list:
        if not b64:
            continue
        payloads.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": detail,
                },
            }
        )
    return payloads


def build_prompt(
    question: str,
    page_indices: Iterable[int],
    image_b64_list: Iterable[str],
    *,
    contains_possible_toc: bool = False,
    detail: str = "high",
) -> PromptBundle:
    """一鍵組合 system + user 訊息。"""
    return PromptBundle(
        system=SYSTEM_PROMPT,
        user_text=build_user_text(question, page_indices, contains_possible_toc),
        image_payloads=build_image_payloads(image_b64_list, detail=detail),
    )
