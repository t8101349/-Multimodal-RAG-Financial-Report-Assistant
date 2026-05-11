"""視覺檢索後端的抽象介面。

所有具體後端（Byaldi、CLIP、Mock）皆需實作 :class:`BaseRetriever`，
這樣 :mod:`pipeline` 與評測層就可以對任一後端做一致的呼叫。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class RetrievalResult:
    """單一檢索命中結果。"""

    page_index: int             # 從 1 開始的人類頁碼
    score: float                # 後端原生分數（越高越相關）
    image_path: str             # 對應頁面圖檔
    source: str                 # 後端來源（例如 byaldi / clip / mock）
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class BaseRetriever(ABC):
    """視覺檢索後端介面。"""

    name: str = "base"

    @abstractmethod
    def build_index(
        self,
        pdf_path: str | Path,
        page_image_dir: str | Path,
        index_dir: str | Path,
        **kwargs,
    ) -> None:
        """建立索引。

        要求所有具體後端在此完成：
            1. 載入模型
            2. 讀取 PDF 或頁面圖檔
            3. 寫入磁碟以供之後 :meth:`search` 載入
        """

    @abstractmethod
    def load_index(self, index_dir: str | Path) -> None:
        """從磁碟載入既有索引。"""

    @abstractmethod
    def search(self, query: str, k: int = 4) -> List[RetrievalResult]:
        """以查詢字串檢索 Top-K 頁面。"""

    def filter_toc(
        self,
        results: List[RetrievalResult],
        toc_pages: Optional[set[int]] = None,
    ) -> List[RetrievalResult]:
        """從檢索結果剔除目錄頁。

        參數：
            results: 原始命中
            toc_pages: 已標記為目錄頁的人類頁碼集合；若為 None 則保留原順序
        """
        if not toc_pages:
            return list(results)
        return [r for r in results if r.page_index not in toc_pages]
