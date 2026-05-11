"""CLIP 視覺檢索後端（備援路徑）。

對應原文中的「雙軌」設計：當 Byaldi / ColPali 主路徑因環境限制而無法使用時，
退回到 HuggingFace ``transformers`` 提供的 CLIP 系列模型，
對每一頁的頁面圖檔做整頁影像向量檢索。

注意：
    - CLIP 對「文檔圖像」並非最佳模型，召回品質不如 ColPali，
      但具備穩定可用、易於部署、不依賴 ColPali 權重等優點。
    - 為了讓中文查詢有意義，預設使用 ``openai/clip-vit-base-patch32``；
      若需要更好的中文語意對齊，可改用 ``OFA-Sys/chinese-clip-vit-base-patch16``，
      此後端會自動偵測模型 ID 是否含 ``chinese``，並選用對應的 Processor。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .retriever_base import BaseRetriever, RetrievalResult


class CLIPRetriever(BaseRetriever):
    """以 HuggingFace CLIP 為核心的頁面圖向量檢索後端。"""

    name = "clip"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._index_dir: Optional[Path] = None
        self._embeddings: Optional[np.ndarray] = None  # 形狀 (N, D)
        self._page_records: List[dict] = []           # 每筆含 page_index 與 image_path

    # ------------------------------------------------------------
    # 模型載入
    # ------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "需要安裝 torch 與 transformers 才能使用 CLIP 備援路徑：\n"
                "    pip install torch transformers"
            ) from exc

        import torch

        is_chinese = "chinese" in self.model_name.lower()
        if is_chinese:
            try:
                from transformers import ChineseCLIPModel, ChineseCLIPProcessor  # type: ignore

                self._model = ChineseCLIPModel.from_pretrained(self.model_name)
                self._processor = ChineseCLIPProcessor.from_pretrained(self.model_name)
            except ImportError:  # pragma: no cover
                # 後備：以一般 CLIP 介面載入
                self._model = CLIPModel.from_pretrained(self.model_name)
                self._processor = CLIPProcessor.from_pretrained(self.model_name)
        else:
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()

    # ------------------------------------------------------------
    # 嵌入工具
    # ------------------------------------------------------------
    def _embed_images(self, image_paths: List[Path]) -> np.ndarray:
        import torch
        from PIL import Image

        self._ensure_model()
        embeddings: List[np.ndarray] = []
        batch_size = 4
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())
        return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 512), dtype=np.float32)

    def _embed_text(self, text: str) -> np.ndarray:
        import torch

        self._ensure_model()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()[0]

    # ------------------------------------------------------------
    # 介面實作
    # ------------------------------------------------------------
    def build_index(
        self,
        pdf_path: str | Path,
        page_image_dir: str | Path,
        index_dir: str | Path,
        **kwargs,
    ) -> None:
        page_image_dir = Path(page_image_dir)
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        page_records: List[dict] = []
        image_paths: List[Path] = []
        for img in sorted(page_image_dir.glob("*_p*.*")):
            stem = img.stem
            try:
                page_no = int(stem.rsplit("_p", 1)[-1])
            except (ValueError, IndexError):
                continue
            page_records.append(
                {"page_index": page_no, "image_path": str(img).replace("\\", "/")}
            )
            image_paths.append(img)

        if not image_paths:
            raise FileNotFoundError(
                f"page_image_dir 中找不到任何頁面圖：{page_image_dir}；"
                "請先執行 pdf_renderer.render_pdf"
            )

        embeddings = self._embed_images(image_paths)

        np.save(index_dir / "clip_embeddings.npy", embeddings)
        (index_dir / "clip_records.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "records": page_records,
                    "pdf_path": str(pdf_path).replace("\\", "/"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self._index_dir = index_dir
        self._embeddings = embeddings
        self._page_records = page_records

    def load_index(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        emb_path = index_dir / "clip_embeddings.npy"
        rec_path = index_dir / "clip_records.json"
        if not emb_path.exists() or not rec_path.exists():
            raise FileNotFoundError(f"找不到 CLIP 索引檔：{index_dir}")

        self._embeddings = np.load(emb_path)
        meta = json.loads(rec_path.read_text(encoding="utf-8"))
        self._page_records = meta.get("records", [])
        self.model_name = meta.get("model_name", self.model_name)
        self._index_dir = index_dir

    def search(self, query: str, k: int = 4) -> List[RetrievalResult]:
        if self._embeddings is None or not self._page_records:
            raise RuntimeError("尚未建立或載入 CLIP 索引")

        query_vec = self._embed_text(query)
        sims = self._embeddings @ query_vec
        top_idx = np.argsort(-sims)[:k]

        results: List[RetrievalResult] = []
        for idx in top_idx:
            record = self._page_records[int(idx)]
            results.append(
                RetrievalResult(
                    page_index=int(record["page_index"]),
                    score=float(sims[int(idx)]),
                    image_path=record["image_path"],
                    source=self.name,
                    extra={"model": self.model_name},
                )
            )
        return results
