"""Mock 檢索後端 - 僅供測試與煙霧驗證使用。

它不會載入任何模型；改以「對頁面文字做關鍵字計分」的方式回傳 Top-K。
這個後端的存在目的有兩個：

    1. 在 Byaldi 與 CLIP 皆缺席的環境中，仍能讓 :mod:`pipeline` 端到端跑通
    2. 在單元測試裡提供可預測的檢索行為

⚠️ 注意：Mock 後端的檢索品質不代表系統實際能力，請勿用於生產評估。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

from .retriever_base import BaseRetriever, RetrievalResult


# 簡單的中文字元 + 英數字斷詞
_TOKEN_PATTERN = re.compile(r"[一-鿿]|[A-Za-z]+|\d+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text or "")


class MockRetriever(BaseRetriever):
    """以頁面文字 Bag-of-Words 計分的關鍵字檢索。"""

    name = "mock"

    def __init__(self) -> None:
        self._records: List[dict] = []
        self._index_dir: Optional[Path] = None

    def build_index(
        self,
        pdf_path: str | Path,
        page_image_dir: str | Path,
        index_dir: str | Path,
        page_units_path: Optional[str | Path] = None,
        **kwargs,
    ) -> None:
        page_image_dir = Path(page_image_dir)
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        # 優先使用 page_units.jsonl 中已抽好的文字
        records: List[dict] = []
        if page_units_path:
            path = Path(page_units_path)
            if path.exists():
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        records.append(
                            {
                                "page_index": int(data["page_index"]),
                                "image_path": data.get("image_path", ""),
                                "text": data.get("text", ""),
                                "is_likely_toc": bool(data.get("is_likely_toc", False)),
                            }
                        )

        if not records:
            # 若無 page_units，退而求其次：以圖檔列表建立佔位記錄，不含文字
            for img in sorted(page_image_dir.glob("*_p*.*")):
                try:
                    page_no = int(img.stem.rsplit("_p", 1)[-1])
                except (ValueError, IndexError):
                    continue
                records.append(
                    {
                        "page_index": page_no,
                        "image_path": str(img).replace("\\", "/"),
                        "text": "",
                        "is_likely_toc": False,
                    }
                )

        if not records:
            raise FileNotFoundError("找不到任何可索引頁面")

        (index_dir / "mock_records.json").write_text(
            json.dumps({"records": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._records = records
        self._index_dir = index_dir

    def load_index(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        rec_path = index_dir / "mock_records.json"
        if not rec_path.exists():
            raise FileNotFoundError(f"找不到 mock 索引：{index_dir}")
        data = json.loads(rec_path.read_text(encoding="utf-8"))
        self._records = data.get("records", [])
        self._index_dir = index_dir

    def search(self, query: str, k: int = 4) -> List[RetrievalResult]:
        if not self._records:
            raise RuntimeError("尚未建立或載入 mock 索引")

        q_tokens = Counter(_tokenize(query))
        if not q_tokens:
            return []

        scored: List[tuple[float, dict]] = []
        for rec in self._records:
            page_tokens = Counter(_tokenize(rec["text"]))
            score = 0.0
            for tok, qcnt in q_tokens.items():
                if tok in page_tokens:
                    score += qcnt * (1.0 + 0.1 * page_tokens[tok])
            # 目錄頁輕度降權，模擬「抗目錄頁」啟發式
            if rec.get("is_likely_toc"):
                score *= 0.4
            if score > 0:
                scored.append((score, rec))

        scored.sort(key=lambda x: -x[0])
        results: List[RetrievalResult] = []
        for score, rec in scored[:k]:
            results.append(
                RetrievalResult(
                    page_index=int(rec["page_index"]),
                    score=float(score),
                    image_path=rec["image_path"],
                    source=self.name,
                    extra={"is_likely_toc": rec.get("is_likely_toc", False)},
                )
            )
        return results
