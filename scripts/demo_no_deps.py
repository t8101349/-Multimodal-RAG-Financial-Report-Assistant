"""零依賴煙霧演示：在缺少 PyMuPDF / byaldi / CLIP / VLM 的環境中
也能完整跑一遍「索引 → 多頁召回 → 抗目錄過濾 → 多圖回答 → 評測」流程。

做法：
    1. 直接寫出一份合成的 ``page_units.jsonl``，模擬 :func:`render_pdf` 的輸出
       （頁面文字、頁碼、目錄頁旗標都是合成的）
    2. 用同樣的文字寫出對應的占位「頁面圖」（純文字 .jpg，僅作示意，
       因為 MockVLMClient 不會真正讀檔；MultimodalRAGPipeline 仍會嘗試
       讀檔轉 base64，因此這裡寫入了一段合法的最小 JPEG 位元流）
    3. 用 MockRetriever 建立索引，再以 MockVLMClient 作為 VLM
    4. 跑一輪完整評測，輸出報告

⚠️ 此腳本只能驗證系統「鏈路是否串通」，不代表真實檢索 / 生成品質。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import evaluate, load_eval_cases  # noqa: E402
from src.pipeline import MultimodalRAGPipeline  # noqa: E402
from src.retriever_mock import MockRetriever  # noqa: E402
from src.vlm_client import MockVLMClient  # noqa: E402
from src.pdf_renderer import PageUnit  # noqa: E402


# 最小合法 JPEG（1x1 全白），用於滿足 Pipeline 中 base64 編碼步驟
_TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
    "07090908"
    + "0a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c303132311f27"
    + "39"
    + "3d3833"
    + "3d352e31"
    + "31333"
    + "0ffc0000b08000100010101220000ffc40014000100000000000000000000000000000007ffc40014100100000000000000000000000000000000ffda0008010100003f00d2cf20ffd9"
)


_SYNTHETIC_PAGES: List[dict] = [
    {
        "page_index": 1,
        "text": "繁星科技股份有限公司 2024 年度報告 封面",
        "is_likely_toc": False,
    },
    {
        "page_index": 2,
        "text": (
            "目錄\n"
            "致股東報告書 ............ 3\n"
            "經營概要 ............ 4\n"
            "核心財務指標 ............ 5\n"
            "研發投入趨勢 ............ 6\n"
            "業務板塊收入結構 ............ 7\n"
            "資產負債表 ............ 8\n"
            "重要會計政策附註 ............ 9\n"
            "風險因素 ............ 10\n"
            "企業社會責任 ............ 11\n"
            "後續事項 ............ 12\n"
        ),
        "is_likely_toc": True,
    },
    {
        "page_index": 3,
        "text": (
            "致股東報告書 2024 年合併營業收入 482.6 億元，年增 18.7%。"
            "歸屬母公司業主之淨利為 92.4 億元，年增 15.3%。"
        ),
        "is_likely_toc": False,
    },
    {
        "page_index": 4,
        "text": (
            "經營概要 三大業務板塊：先進製程晶圓代工、特殊製程晶圓代工、封裝測試，"
            "合計佔合併營業收入 96.4%。"
        ),
        "is_likely_toc": False,
    },
    {
        "page_index": 5,
        "text": (
            "核心財務指標 營業收入 482.6 億元 營業毛利 212.4 億元 營業利益 118.3 億元 "
            "稅後淨利 92.4 億元 每股盈餘 EPS 9.24 元 研發投入 12.7 億 "
            "研發投入佔營收比 2.6%"
        ),
        "is_likely_toc": False,
    },
    {
        "page_index": 6,
        "text": (
            "研發投入趨勢 2022 年 8.6 億 2023 年 10.4 億 2024 年 12.7 億 "
            "近三年成長 47.7% 反映集團對先進製程與封裝技術的長期承諾。"
        ),
        "is_likely_toc": False,
    },
    {
        "page_index": 7,
        "text": (
            "業務板塊收入結構 先進製程晶圓代工 248.7 億 51.5% 特殊製程晶圓代工 "
            "126.4 億 26.2% 封裝測試 90.0 億 18.7% 技術授權與工程服務 17.5 億 3.6% "
            "合計 482.6 億 100.0%"
        ),
        "is_likely_toc": False,
    },
    {
        "page_index": 8,
        "text": (
            "資產負債表 流動資產 316.8 億 非流動資產 542.1 億 總資產 858.9 億 "
            "流動負債 192.5 億 非流動負債 180.4 億 總負債 372.9 億 "
            "歸屬母公司業主權益 486.0 億"
        ),
        "is_likely_toc": False,
    },
    {
        "page_index": 9,
        "text": (
            "重要會計政策附註 存貨採加權平均法 固定資產採直線法折舊 "
            "建築物耐用年限 20 至 40 年 機器設備 5 至 10 年 無形資產 38.6 億"
        ),
        "is_likely_toc": False,
    },
    {
        "page_index": 10,
        "text": (
            "風險因素 全球景氣循環導致需求波動 主要客戶集中度偏高 前五大客戶 "
            "62.4% 地緣政治造成關鍵原物料供應風險 匯率波動 美元計價收入比重 78.5%"
        ),
        "is_likely_toc": False,
    },
    {
        "page_index": 11,
        "text": (
            "企業社會責任 推動淨零碳排目標 溫室氣體排放強度下降 16% 通過 ISO 14064-1 "
            "查證 贊助 STEM 教育 培訓 1820 名工程師"
        ),
        "is_likely_toc": False,
    },
    {
        "page_index": 12,
        "text": (
            "後續事項 董事會通過配發 2024 年度現金股利每股 4.5 元 預計 2025 年 6 月除息 "
            "簽訂為期 5 年長期供應合約 預計合約期間累計營收約 850 億"
        ),
        "is_likely_toc": False,
    },
]


def build_synthetic_assets(processed_dir: Path, page_image_dir: Path) -> Path:
    page_image_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    page_units_path = processed_dir / "page_units.jsonl"

    units: List[PageUnit] = []
    for entry in _SYNTHETIC_PAGES:
        img_path = page_image_dir / f"synthetic_p{entry['page_index']:03d}.jpg"
        img_path.write_bytes(_TINY_JPEG)
        units.append(
            PageUnit(
                page_index=entry["page_index"],
                raw_page_index=entry["page_index"] - 1,
                image_path=str(img_path).replace("\\", "/"),
                width=1,
                height=1,
                char_count=len(entry["text"]),
                text_preview=entry["text"][:80],
                text=entry["text"],
                is_likely_toc=entry["is_likely_toc"],
                block_count=1,
                source_pdf="synthetic://demo.pdf",
            )
        )

    with page_units_path.open("w", encoding="utf-8") as fh:
        for u in units:
            fh.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")

    return page_units_path


def main() -> int:
    parser = argparse.ArgumentParser(description="零依賴煙霧演示")
    parser.add_argument(
        "--processed-dir",
        default=str(PROJECT_ROOT / "data" / "processed"),
    )
    parser.add_argument(
        "--page-image-dir",
        default=str(PROJECT_ROOT / "data" / "page_images"),
    )
    parser.add_argument(
        "--cases",
        default=str(PROJECT_ROOT / "data" / "eval" / "reference_questions.jsonl"),
    )
    parser.add_argument(
        "--report-md",
        default=str(PROJECT_ROOT / "data" / "reports" / "p5_demo_report.md"),
    )
    parser.add_argument(
        "--metrics-json",
        default=str(PROJECT_ROOT / "data" / "reports" / "p5_demo_metrics.json"),
    )
    parser.add_argument(
        "--results-jsonl",
        default=str(PROJECT_ROOT / "data" / "eval" / "evaluation_results.jsonl"),
    )
    parser.add_argument(
        "--failure-jsonl",
        default=str(PROJECT_ROOT / "data" / "eval" / "failure_replay.jsonl"),
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    page_image_dir = Path(args.page_image_dir)
    page_units_path = build_synthetic_assets(processed_dir, page_image_dir)
    print(f"[demo] 已建立合成 page_units.jsonl：{page_units_path}")

    retriever = MockRetriever()
    index_dir = processed_dir / "mock_index"
    retriever.build_index(
        pdf_path="synthetic://demo.pdf",
        page_image_dir=page_image_dir,
        index_dir=index_dir,
        page_units_path=page_units_path,
    )
    print(f"[demo] Mock 索引已建立於：{index_dir}")

    rag_index_summary = {
        "pdf": "synthetic://demo.pdf",
        "page_count": len(_SYNTHETIC_PAGES),
        "page_units_path": str(page_units_path).replace("\\", "/"),
        "index_dir": str(index_dir).replace("\\", "/"),
        "backend": "mock",
        "toc_pages": [u["page_index"] for u in _SYNTHETIC_PAGES if u["is_likely_toc"]],
    }
    (processed_dir / "rag_index.json").write_text(
        json.dumps(rag_index_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    vlm = MockVLMClient(
        canned_answer=(
            "**結論**：（Mock VLM）以下為示意回答，請接入真正的 VLM 後重新評估。\n"
            "**關鍵數值**：482.6 億元（營業收入）、92.4 億元（稅後淨利）。\n"
            "**趨勢解讀**：研發投入近三年自 8.6 億成長至 12.7 億。\n"
            "**證據頁碼**：第 5 頁、第 6 頁。\n"
            "**不確定性**：無，因屬 Mock 範例。"
        )
    )

    pipeline = MultimodalRAGPipeline.from_paths(
        retriever=retriever,
        vlm=vlm,
        page_units_path=page_units_path,
        index_dir=index_dir,
        default_top_k=4,
        filter_toc=True,
    )

    cases = load_eval_cases(args.cases)
    summary = evaluate(
        pipeline,
        cases,
        results_path=args.results_jsonl,
        failure_path=args.failure_jsonl,
    )

    metrics = summary.to_dict()["summary"]
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# P05 多模態 RAG 零依賴煙霧演示",
        "",
        "本報告僅驗證鏈路是否串通；檢索採用 MockRetriever（關鍵字計分），",
        "回答採用 MockVLMClient（固定回應）。**指標不代表系統真實品質**。",
        "",
        f"- 題目數：{metrics['n']}",
        f"- 檢索命中率 @K：{metrics['hit_rate_at_k']:.1%}",
        f"- 引用準確率（受 Mock 固定回應影響）：{metrics['citation_accuracy']:.1%}",
        f"- 關鍵字召回率（受 Mock 固定回應影響）：{metrics['keyword_recall']:.1%}",
        f"- 平均總延遲：{metrics['avg_latency_total_ms']:.1f} ms",
        "",
        "## 逐題召回（檢索層真實表現）",
        "",
        "| QID | 標註頁 | 召回頁（Top-K） | 命中 |",
        "|---|---|---|---|",
    ]
    for c in summary.cases:
        lines.append(
            f"| {c.qid} | {c.expected_pages} | {c.retrieved_pages} | "
            f"{'✅' if c.hit_at_k else '❌'} |"
        )
    Path(args.report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_md).write_text("\n".join(lines), encoding="utf-8")

    print("[demo] === 指標 ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"[demo] 報告已寫入：{args.report_md}")
    print(f"[demo] 指標 JSON 已寫入：{args.metrics_json}")
    print(f"[demo] 評測明細：{args.results_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
