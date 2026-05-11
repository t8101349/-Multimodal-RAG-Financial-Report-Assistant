"""Byaldi + ColPali 視覺檢索後端（主路徑）。

對應原文「Vision-first」主路徑：
    - 使用 ColPali 直接對 PDF 頁面做視覺編碼
    - 由 Byaldi 封裝索引建立、儲存與查詢
    - ``store_collection_with_index=True`` 確保命中結果可回指原圖

若環境未安裝 byaldi，匯入會延後到實際呼叫時才失敗，
方便 :mod:`retriever_factory` 直接做能力探測。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from .retriever_base import BaseRetriever, RetrievalResult


class ByaldiRetriever(BaseRetriever):
    """以 ColPali + Byaldi 為核心的視覺檢索後端。"""

    name = "byaldi"

    def __init__(
        self,
        model_path: str,
        index_name: str = "finance_report",
        verbose: int = 0,
        hf_offline: bool = False,
        hf_endpoint: Optional[str] = None,
    ) -> None:
        if hf_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint

        self.model_path = model_path
        self.index_name = index_name
        self.verbose = verbose
        self._rag = None
        self._page_image_map: dict[int, str] = {}

    # ------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------
    def _ensure_byaldi(self):
        try:
            from byaldi import RAGMultiModalModel  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "尚未安裝 byaldi，無法使用 ColPali 主路徑；"
                "請執行 `pip install byaldi colpali-engine`，或將 RETRIEVER_BACKEND 設為 clip"
            ) from exc
        return RAGMultiModalModel

    def _load_page_image_map(self, page_image_dir: Path) -> None:
        """掃描 page_images 資料夾，建立人類頁碼 -> 圖檔路徑映射。

        檔名約定：``<stem>_p<3 位頁碼>.<ext>`` （由 :mod:`pdf_renderer` 產生）。
        """
        mapping: dict[int, str] = {}
        for img in sorted(page_image_dir.glob("*_p*.*")):
            stem = img.stem
            try:
                page_part = stem.rsplit("_p", 1)[-1]
                page_no = int(page_part)
                mapping[page_no] = str(img).replace("\\", "/")
            except (ValueError, IndexError):
                continue
        self._page_image_map = mapping

    # ------------------------------------------------------------
    # 介面實作
    # ------------------------------------------------------------
    def build_index(
        self,
        pdf_path: str | Path,
        page_image_dir: str | Path,
        index_dir: str | Path,
        overwrite: bool = True,
        **kwargs,
    ) -> None:
        RAGMultiModalModel = self._ensure_byaldi()

        pdf_path = Path(pdf_path)
        page_image_dir = Path(page_image_dir)
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"找不到 ColPali 模型資料夾：{self.model_path}；"
                "請先下載 vidore/colpali-v1.2 並設定 COLPALI_MODEL_PATH"
            )

        rag = RAGMultiModalModel.from_pretrained(self.model_path, verbose=self.verbose)
        rag.index(
            input_path=str(pdf_path),
            index_name=self.index_name,
            store_collection_with_index=True,
            overwrite=overwrite,
        )
        self._rag = rag
        self._load_page_image_map(page_image_dir)

        # 將輔助資訊（檔案、頁碼映射）寫入 index_dir 供 load_index 重建
        side_car = {
            "pdf_path": str(pdf_path).replace("\\", "/"),
            "page_image_dir": str(page_image_dir).replace("\\", "/"),
            "index_name": self.index_name,
            "model_path": self.model_path,
            "page_image_map": self._page_image_map,
        }
        (index_dir / "byaldi_sidecar.json").write_text(
            json.dumps(side_car, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load_index(self, index_dir: str | Path) -> None:
        RAGMultiModalModel = self._ensure_byaldi()
        index_dir = Path(index_dir)

        side_car_path = index_dir / "byaldi_sidecar.json"
        if side_car_path.exists():
            side_car = json.loads(side_car_path.read_text(encoding="utf-8"))
            self.index_name = side_car.get("index_name", self.index_name)
            self._page_image_map = {
                int(k): v for k, v in side_car.get("page_image_map", {}).items()
            }

        self._rag = RAGMultiModalModel.from_index(self.index_name)

    def search(self, query: str, k: int = 4) -> List[RetrievalResult]:
        if self._rag is None:
            raise RuntimeError("尚未建立或載入索引，請先呼叫 build_index 或 load_index")

        raw_results = self._rag.search(query, k=k)
        results: List[RetrievalResult] = []
        for r in raw_results:
            # Byaldi 返回物件含 page_num（1 起算）與 score；base64 為頁面圖
            page_index = int(getattr(r, "page_num", getattr(r, "page", 0)) or 0)
            score = float(getattr(r, "score", 0.0) or 0.0)
            image_path = self._page_image_map.get(page_index, "")
            extra = {}
            base64 = getattr(r, "base64", None)
            if base64:
                extra["base64"] = base64
            results.append(
                RetrievalResult(
                    page_index=page_index,
                    score=score,
                    image_path=image_path,
                    source=self.name,
                    extra=extra,
                )
            )
        return results
