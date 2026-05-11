"""命令列：對既有索引執行單次或互動式查詢。

執行：
    python scripts/run_query.py --question "2024 年度的總營業收入是多少？"

可加 ``--mock-vlm`` 用於檢驗檢索結果（不會實際呼叫 VLM）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from src.pipeline import MultimodalRAGPipeline  # noqa: E402
from src.retriever_factory import get_retriever  # noqa: E402
from src.vlm_client import MockVLMClient, VLMClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="多模態 RAG 查詢 CLI")
    parser.add_argument("--question", required=True, help="使用者問題")
    parser.add_argument(
        "--page-units",
        default=str(PROJECT_ROOT / "data" / "processed" / "page_units.jsonl"),
    )
    parser.add_argument(
        "--index-dir",
        default="",
        help="若留空則依 rag_index.json 的紀錄自動選擇",
    )
    parser.add_argument(
        "--rag-index",
        default=str(PROJECT_ROOT / "data" / "processed" / "rag_index.json"),
    )
    parser.add_argument("--top-k", type=int, default=settings.retrieval_top_k)
    parser.add_argument("--no-filter-toc", action="store_true", help="不過濾目錄頁")
    parser.add_argument(
        "--mock-vlm",
        action="store_true",
        help="使用 MockVLMClient（避免真實 API 開銷，方便驗證檢索鏈路）",
    )
    parser.add_argument(
        "--log",
        default=str(PROJECT_ROOT / "data" / "reports" / "query_log.jsonl"),
    )
    args = parser.parse_args()

    rag_index_path = Path(args.rag_index)
    backend_name = ""
    if rag_index_path.exists():
        meta = json.loads(rag_index_path.read_text(encoding="utf-8"))
        backend_name = meta.get("backend", "")
        index_dir = Path(args.index_dir) if args.index_dir else Path(meta["index_dir"])
    else:
        index_dir = Path(args.index_dir) if args.index_dir else (
            PROJECT_ROOT / "data" / "processed" / "mock_index"
        )

    retriever = get_retriever(
        backend=backend_name or settings.retriever_backend,
        colpali_model_path=settings.colpali_model_path,
        colpali_index_name=settings.colpali_index_name,
        clip_model_name=settings.clip_model_name,
        hf_offline=settings.hf_offline,
        hf_endpoint=settings.hf_endpoint,
    )

    if args.mock_vlm or not settings.vlm_ready():
        if not args.mock_vlm:
            print("[run_query] 偵測不到 VLM_API_KEY，自動切換至 MockVLMClient")
        vlm = MockVLMClient()
    else:
        vlm = VLMClient(
            base_url=settings.vlm_base_url,
            api_key=settings.vlm_api_key,
            model=settings.vlm_model,
        )

    pipeline = MultimodalRAGPipeline.from_paths(
        retriever=retriever,
        vlm=vlm,
        page_units_path=args.page_units,
        index_dir=index_dir,
        default_top_k=args.top_k,
        filter_toc=not args.no_filter_toc,
    )

    result = pipeline.query(args.question, top_k=args.top_k)
    pipeline.append_log(result, args.log)

    print("==== 問題 ====")
    print(args.question)
    print("==== 召回頁 ====")
    print(result.retrieved_pages, "scores:", [round(s, 4) for s in result.retrieval_scores])
    if result.filtered_toc_pages:
        print(f"（已過濾目錄頁：{result.filtered_toc_pages}）")
    print("==== 回答 ====")
    print(result.answer)
    print(
        f"\n[遙測] 總延遲 {result.latency_total_ms:.1f}ms，"
        f"VLM 延遲 {result.latency_vlm_ms:.1f}ms，"
        f"tokens={result.total_tokens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
