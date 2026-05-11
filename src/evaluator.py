"""評測模組。

對應原文的雙層評測框架：

    第一層：檢索是否找對頁面
        - hit@K：標註的證據頁是否出現在召回集
        - 目錄頁過濾準確率
    第二層：回答是否基於正確頁面得出正確結論
        - citation accuracy：回答中引用的頁碼是否落在標註證據頁集合內
        - answer keyword accuracy：回答中是否覆蓋預期關鍵字
        - latency / token 用量

題庫格式（JSONL）：
    {
        "qid": "Q01",
        "question": "2024 年度的總營業收入是多少？",
        "expected_pages": [5],
        "expected_keywords": ["營業收入", "億"]
    }
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .pipeline import MultimodalRAGPipeline, QueryResult


# 在中文回答裡擷取「第 X 頁」與「第 X、Y、Z 頁」型式
_PAGE_REF_PATTERN = re.compile(r"第\s*([0-9０-９]+)\s*頁")
_PAGE_REF_RANGE = re.compile(r"第\s*([0-9０-９]+)\s*[、,，]\s*([0-9０-９]+)")


def _to_int(s: str) -> int:
    table = str.maketrans("０１２３４５６７８９", "0123456789")
    return int(s.translate(table))


def extract_cited_pages(answer: str) -> List[int]:
    """從回答文字裡擷取被引用的頁碼。"""
    pages = []
    for m in _PAGE_REF_PATTERN.finditer(answer or ""):
        try:
            pages.append(_to_int(m.group(1)))
        except ValueError:
            continue
    return sorted(set(pages))


def keyword_hits(answer: str, keywords: Sequence[str]) -> List[str]:
    return [kw for kw in keywords if kw and kw in (answer or "")]


@dataclass
class EvalCase:
    qid: str
    question: str
    expected_pages: List[int] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalCase":
        return cls(
            qid=str(data.get("qid", "")),
            question=str(data.get("question", "")),
            expected_pages=[int(x) for x in data.get("expected_pages", []) or []],
            expected_keywords=list(data.get("expected_keywords", []) or []),
        )


@dataclass
class CaseResult:
    qid: str
    question: str
    retrieved_pages: List[int]
    cited_pages: List[int]
    expected_pages: List[int]
    hit_at_k: bool
    citation_accuracy: float
    keyword_hits: List[str]
    expected_keywords: List[str]
    keyword_recall: float
    answer: str
    latency_total_ms: float
    latency_vlm_ms: float
    total_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalSummary:
    cases: List[CaseResult]
    hit_rate_at_k: float
    citation_accuracy: float
    keyword_recall: float
    avg_latency_total_ms: float
    avg_latency_vlm_ms: float
    avg_total_tokens: float

    def to_dict(self) -> dict:
        return {
            "summary": {
                "n": len(self.cases),
                "hit_rate_at_k": self.hit_rate_at_k,
                "citation_accuracy": self.citation_accuracy,
                "keyword_recall": self.keyword_recall,
                "avg_latency_total_ms": self.avg_latency_total_ms,
                "avg_latency_vlm_ms": self.avg_latency_vlm_ms,
                "avg_total_tokens": self.avg_total_tokens,
            },
            "cases": [c.to_dict() for c in self.cases],
        }


def load_eval_cases(path: str | Path) -> List[EvalCase]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到評測題庫：{path}")
    cases: List[EvalCase] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cases.append(EvalCase.from_dict(json.loads(line)))
    return cases


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def evaluate_case(pipeline: MultimodalRAGPipeline, case: EvalCase) -> CaseResult:
    qr: QueryResult = pipeline.query(case.question)
    cited = extract_cited_pages(qr.answer)

    expected_set = set(case.expected_pages)
    retrieved_set = set(qr.retrieved_pages)
    hit = bool(expected_set & retrieved_set) if expected_set else True

    if expected_set and cited:
        citation_acc = _safe_div(
            len([p for p in cited if p in expected_set]), len(cited)
        )
    elif not expected_set:
        citation_acc = 1.0
    else:
        citation_acc = 0.0

    hits = keyword_hits(qr.answer, case.expected_keywords)
    kw_recall = (
        _safe_div(len(hits), len(case.expected_keywords))
        if case.expected_keywords
        else 1.0
    )

    return CaseResult(
        qid=case.qid,
        question=case.question,
        retrieved_pages=qr.retrieved_pages,
        cited_pages=cited,
        expected_pages=case.expected_pages,
        hit_at_k=hit,
        citation_accuracy=citation_acc,
        keyword_hits=hits,
        expected_keywords=case.expected_keywords,
        keyword_recall=kw_recall,
        answer=qr.answer,
        latency_total_ms=qr.latency_total_ms,
        latency_vlm_ms=qr.latency_vlm_ms,
        total_tokens=qr.total_tokens,
    )


def evaluate(
    pipeline: MultimodalRAGPipeline,
    cases: Iterable[EvalCase],
    *,
    results_path: Optional[str | Path] = None,
    failure_path: Optional[str | Path] = None,
) -> EvalSummary:
    case_results: List[CaseResult] = []
    failures: List[CaseResult] = []

    for case in cases:
        result = evaluate_case(pipeline, case)
        case_results.append(result)
        if not result.hit_at_k or result.keyword_recall < 0.5:
            failures.append(result)

    n = len(case_results)
    summary = EvalSummary(
        cases=case_results,
        hit_rate_at_k=_safe_div(sum(1 for c in case_results if c.hit_at_k), n),
        citation_accuracy=_safe_div(sum(c.citation_accuracy for c in case_results), n),
        keyword_recall=_safe_div(sum(c.keyword_recall for c in case_results), n),
        avg_latency_total_ms=_safe_div(sum(c.latency_total_ms for c in case_results), n),
        avg_latency_vlm_ms=_safe_div(sum(c.latency_vlm_ms for c in case_results), n),
        avg_total_tokens=_safe_div(sum(c.total_tokens for c in case_results), n),
    )

    if results_path:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as fh:
            for c in case_results:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    if failure_path:
        failure_path = Path(failure_path)
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("w", encoding="utf-8") as fh:
            for c in failures:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    return summary
