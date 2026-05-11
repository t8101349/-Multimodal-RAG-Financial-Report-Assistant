"""端到端多模態 RAG 流水線。

把以下五段整合成單一可呼叫流程：

    1. 載入頁面資產與目錄頁集合
    2. 透過 :class:`BaseRetriever` 召回 Top-K 頁面
    3. 視需要做目錄頁過濾
    4. 把命中頁面圖檔轉為 base64，組合提示詞
    5. 呼叫 VLM 取得結構化回答，並紀錄遙測

回傳物件 :class:`QueryResult` 直接可被評測模組消費。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

from .pdf_renderer import PageUnit, load_page_units
from .prompts import build_prompt
from .retriever_base import BaseRetriever, RetrievalResult
from .vlm_client import MockVLMClient, VLMClient, VLMResponse, encode_image_to_base64


@dataclass
class QueryResult:
    """單次查詢的完整輸出（含遙測）。"""

    question: str
    retrieved_pages: List[int]
    retrieval_scores: List[float]
    retrieval_sources: List[str]
    filtered_toc_pages: List[int]
    answer: str
    latency_total_ms: float
    latency_vlm_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    image_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class MultimodalRAGPipeline:
    """整合檢索與生成的流水線。"""

    def __init__(
        self,
        retriever: BaseRetriever,
        vlm: Union[VLMClient, MockVLMClient],
        page_units: Optional[Sequence[PageUnit]] = None,
        default_top_k: int = 4,
        filter_toc: bool = True,
        detail: str = "high",
    ) -> None:
        self.retriever = retriever
        self.vlm = vlm
        self.page_units = list(page_units) if page_units else []
        self.default_top_k = default_top_k
        self.filter_toc = filter_toc
        self.detail = detail

        self._toc_pages = {
            u.page_index for u in self.page_units if u.is_likely_toc
        }
        self._page_image_map = {u.page_index: u.image_path for u in self.page_units}

    # ------------------------------------------------------------
    # 建立 / 載入
    # ------------------------------------------------------------
    @classmethod
    def from_paths(
        cls,
        retriever: BaseRetriever,
        vlm: Union[VLMClient, MockVLMClient],
        page_units_path: str | Path,
        index_dir: Optional[str | Path] = None,
        default_top_k: int = 4,
        filter_toc: bool = True,
        detail: str = "high",
    ) -> "MultimodalRAGPipeline":
        units = load_page_units(page_units_path)
        if index_dir:
            retriever.load_index(index_dir)
        return cls(
            retriever=retriever,
            vlm=vlm,
            page_units=units,
            default_top_k=default_top_k,
            filter_toc=filter_toc,
            detail=detail,
        )

    # ------------------------------------------------------------
    # 查詢
    # ------------------------------------------------------------
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> QueryResult:
        import time

        k = top_k or self.default_top_k
        t0 = time.perf_counter()

        raw_hits: List[RetrievalResult] = self.retriever.search(question, k=k)

        filtered_toc: List[int] = []
        if self.filter_toc and self._toc_pages:
            kept: List[RetrievalResult] = []
            for r in raw_hits:
                if r.page_index in self._toc_pages:
                    filtered_toc.append(r.page_index)
                else:
                    kept.append(r)
            hits = kept
        else:
            hits = list(raw_hits)

        # 取圖檔路徑（優先用檢索結果的 image_path；否則 fallback 到 page_units）
        image_paths: List[str] = []
        for r in hits:
            path = r.image_path or self._page_image_map.get(r.page_index, "")
            if path:
                image_paths.append(path)

        # 編碼成 base64（若檢索後端已提供 extra["base64"]，優先採用）
        b64_list: List[str] = []
        for r, path in zip(hits, image_paths):
            b64 = r.extra.get("base64") if r.extra else None
            if not b64 and path:
                try:
                    b64 = encode_image_to_base64(path)
                except FileNotFoundError:
                    b64 = ""
            if b64:
                b64_list.append(b64)

        prompt_bundle = build_prompt(
            question=question,
            page_indices=[r.page_index for r in hits],
            image_b64_list=b64_list,
            contains_possible_toc=bool(filtered_toc),
            detail=self.detail,
        )

        vlm_resp: VLMResponse = self.vlm.chat(
            prompt_bundle.to_messages(),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total_ms = (time.perf_counter() - t0) * 1000

        return QueryResult(
            question=question,
            retrieved_pages=[r.page_index for r in hits],
            retrieval_scores=[r.score for r in hits],
            retrieval_sources=[r.source for r in hits],
            filtered_toc_pages=filtered_toc,
            answer=vlm_resp.content,
            latency_total_ms=total_ms,
            latency_vlm_ms=vlm_resp.latency_ms,
            prompt_tokens=vlm_resp.prompt_tokens,
            completion_tokens=vlm_resp.completion_tokens,
            total_tokens=vlm_resp.total_tokens,
            image_paths=image_paths,
        )

    # ------------------------------------------------------------
    # 紀錄寫檔
    # ------------------------------------------------------------
    def append_log(self, result: QueryResult, log_path: str | Path) -> None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
