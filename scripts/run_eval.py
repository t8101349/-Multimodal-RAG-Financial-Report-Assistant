"""命令列：對既有索引執行整批評測。

執行：
    python scripts/run_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from src.evaluator import evaluate, load_eval_cases  # noqa: E402
from src.pipeline import MultimodalRAGPipeline  # noqa: E402
from src.retriever_factory import get_retriever  # noqa: E402
from src.vlm_client import MockVLMClient, VLMClient  # noqa: E402


def _format_report(summary, backend: str) -> str:
    s = summary.to_dict()["summary"]
    lines = [
        "# P05 多模態 RAG 評測報告",
        "",
        f"- 後端：`{backend}`",
        f"- 題目數：{s['n']}",
        f"- 檢索命中率 @K：{s['hit_rate_at_k']:.1%}",
        f"- 引用準確率：{s['citation_accuracy']:.1%}",
        f"- 關鍵字召回率：{s['keyword_recall']:.1%}",
        f"- 平均總延遲：{s['avg_latency_total_ms']:.1f} ms",
        f"- 平均 VLM 延遲：{s['avg_latency_vlm_ms']:.1f} ms",
        f"- 平均 tokens：{s['avg_total_tokens']:.0f}",
        "",
        "## 逐題結果",
        "",
        "| QID | 命中 | 引用準確 | 關鍵字 | 召回頁 | 引用頁 | 延遲 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in summary.cases:
        lines.append(
            f"| {c.qid} | {'✅' if c.hit_at_k else '❌'} | "
            f"{c.citation_accuracy:.0%} | {c.keyword_recall:.0%} | "
            f"{c.retrieved_pages} | {c.cited_pages} | "
            f"{c.latency_total_ms:.0f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="批次評測 RAG 流水線")
    parser.add_argument(
        "--cases",
        default=str(PROJECT_ROOT / "data" / "eval" / "reference_questions.jsonl"),
    )
    parser.add_argument(
        "--page-units",
        default=str(PROJECT_ROOT / "data" / "processed" / "page_units.jsonl"),
    )
    parser.add_argument(
        "--rag-index",
        default=str(PROJECT_ROOT / "data" / "processed" / "rag_index.json"),
    )
    parser.add_argument("--top-k", type=int, default=settings.retrieval_top_k)
    parser.add_argument(
        "--mock-vlm",
        action="store_true",
        help="使用 MockVLMClient",
    )
    parser.add_argument(
        "--results",
        default=str(PROJECT_ROOT / "data" / "eval" / "evaluation_results.jsonl"),
    )
    parser.add_argument(
        "--failures",
        default=str(PROJECT_ROOT / "data" / "eval" / "failure_replay.jsonl"),
    )
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "data" / "reports" / "p5_report.md"),
    )
    parser.add_argument(
        "--metrics",
        default=str(PROJECT_ROOT / "data" / "reports" / "p5_metrics.json"),
    )
    args = parser.parse_args()

    cases = load_eval_cases(args.cases)
    print(f"[run_eval] 載入 {len(cases)} 題")

    rag_index_path = Path(args.rag_index)
    if not rag_index_path.exists():
        print(f"[run_eval] 找不到 {rag_index_path}，請先執行 build_index.py", file=sys.stderr)
        return 2
    meta = json.loads(rag_index_path.read_text(encoding="utf-8"))
    backend_name = meta.get("backend", settings.retriever_backend)
    index_dir = Path(meta["index_dir"])

    retriever = get_retriever(
        backend=backend_name,
        colpali_model_path=settings.colpali_model_path,
        colpali_index_name=settings.colpali_index_name,
        clip_model_name=settings.clip_model_name,
        hf_offline=settings.hf_offline,
        hf_endpoint=settings.hf_endpoint,
    )

    if args.mock_vlm or not settings.vlm_ready():
        if not args.mock_vlm:
            print("[run_eval] 未偵測到 VLM 設定，使用 MockVLMClient")
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
    )

    summary = evaluate(
        pipeline,
        cases,
        results_path=args.results,
        failure_path=args.failures,
    )

    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(
        json.dumps(summary.to_dict()["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = _format_report(summary, backend_name)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")

    print(report)
    print(f"\n[run_eval] 報告已寫入 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
