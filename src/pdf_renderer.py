"""PDF 頁面渲染與資產層。

職責：
    1. 把 PDF 渲染為頁面截圖（JPEG/PNG），確保後續檢索與生成可重複指回同一份原圖
    2. 同步抽取每頁可讀文字（供 BM25 / TF-IDF 備援檢索與目錄頁啟發式辨識）
    3. 產生 ``page_units.jsonl`` 中繼資料：頁碼、檔名、尺寸、字數、區塊密度、來源 PDF
    4. 提供目錄頁啟發式辨識，協助後續檢索層做抗噪過濾

設計重點：
    - 渲染參數（DPI、頁碼起點、輸出格式）必須穩定可重現
    - 中繼資料寫入磁碟，避免每次提問都要重新渲染
    - 解析失敗的單頁不應拖垮整份 PDF 的處理
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


def _import_fitz():
    """延遲匯入 PyMuPDF。

    讓本模組在缺少 ``fitz`` 的環境中仍可被引用（例如使用 :class:`PageUnit`、
    :func:`load_page_units`、:func:`_looks_like_toc` 這些純邏輯工具）。
    只有真正呼叫 :func:`render_pdf` 時才會強制要求 ``fitz``。
    """
    try:
        import fitz  # PyMuPDF
        return fitz
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "需要安裝 PyMuPDF (`pip install PyMuPDF`) 才能渲染 PDF 頁面"
        ) from exc


# --- 中文目錄頁啟發式關鍵字（繁體 + 簡體 + 英文）---
_TOC_KEYWORDS = (
    "目錄", "目录", "Contents", "Table of Contents", "CONTENTS",
    "目  錄", "目  录",
)
_PAGE_NUMBER_LINE = re.compile(r"^\s*[　\.·•\s]*\d{1,4}\s*$")
_DOTTED_LEADER = re.compile(r"\.{3,}|·{3,}|…{1,}")


@dataclass
class PageUnit:
    """單頁資產的中繼資料。"""

    page_index: int                # 從 1 開始的人類頁碼
    raw_page_index: int            # PDF 內部頁索引（從 0 開始）
    image_path: str                # 相對於專案根目錄
    width: int
    height: int
    char_count: int
    text_preview: str              # 前 200 字摘要
    text: str                      # 完整文字（後續用作 BM25 備援檢索）
    is_likely_toc: bool            # 是否疑似目錄頁
    block_count: int               # 文字區塊數
    source_pdf: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RenderResult:
    """整份 PDF 的渲染結果摘要。"""

    pdf_path: str
    pdf_sha1: str
    page_count: int
    units: List[PageUnit] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pdf_path": self.pdf_path,
            "pdf_sha1": self.pdf_sha1,
            "page_count": self.page_count,
            "units": [u.to_dict() for u in self.units],
        }


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_like_toc(text: str) -> bool:
    """判斷一頁是否疑似目錄頁。

    啟發式條件（任一成立即視為目錄頁）：
        1. 出現「目錄 / 目录 / Contents」等明顯標題
        2. 多行以章節名稱 + 點點點 + 頁碼 結尾
        3. 含大量「. . . . 數字」型式
    """
    if not text:
        return False
    head = text[:80]
    if any(k in head for k in _TOC_KEYWORDS):
        return True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    dotted_lines = sum(1 for ln in lines if _DOTTED_LEADER.search(ln))
    page_number_lines = sum(1 for ln in lines if _PAGE_NUMBER_LINE.match(ln))
    if dotted_lines >= 3 or page_number_lines >= 5:
        return True

    return False


def render_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 200,
    image_format: str = "jpg",
    metadata_path: Optional[str | Path] = None,
) -> RenderResult:
    """渲染整份 PDF 為頁面圖檔並輸出中繼資料。

    參數：
        pdf_path: PDF 路徑
        output_dir: 圖檔輸出資料夾
        dpi: 渲染解析度（建議 180~300，財報複雜版面建議 200 起跳）
        image_format: ``jpg`` 或 ``png``
        metadata_path: 中繼 JSONL 輸出路徑；若為 None 則不輸出

    回傳：
        :class:`RenderResult`
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到 PDF：{pdf_path}")

    if image_format.lower() not in {"jpg", "jpeg", "png"}:
        raise ValueError(f"image_format 必須為 jpg / png，目前為 {image_format}")

    ext = "jpg" if image_format.lower() in {"jpg", "jpeg"} else "png"

    fitz = _import_fitz()
    pdf_sha1 = _file_sha1(pdf_path)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    units: List[PageUnit] = []
    stem = pdf_path.stem
    try:
        for raw_idx in range(doc.page_count):
            page = doc.load_page(raw_idx)
            try:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
            except Exception as exc:  # pragma: no cover - 單頁異常不應中斷全流程
                print(f"[render_pdf] 第 {raw_idx + 1} 頁渲染失敗：{exc}")
                continue

            img_name = f"{stem}_p{raw_idx + 1:03d}.{ext}"
            img_path = output_dir / img_name
            pix.save(img_path)

            text = page.get_text("text") or ""
            blocks = page.get_text("blocks") or []
            preview = text.strip().replace("\n", " ")[:200]

            unit = PageUnit(
                page_index=raw_idx + 1,
                raw_page_index=raw_idx,
                image_path=str(img_path).replace("\\", "/"),
                width=pix.width,
                height=pix.height,
                char_count=len(text),
                text_preview=preview,
                text=text,
                is_likely_toc=_looks_like_toc(text),
                block_count=len(blocks),
                source_pdf=str(pdf_path).replace("\\", "/"),
            )
            units.append(unit)
    finally:
        doc.close()

    result = RenderResult(
        pdf_path=str(pdf_path).replace("\\", "/"),
        pdf_sha1=pdf_sha1,
        page_count=len(units),
        units=units,
    )

    if metadata_path:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as fh:
            for u in units:
                fh.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")

    return result


def load_page_units(metadata_path: str | Path) -> List[PageUnit]:
    """從 ``page_units.jsonl`` 讀取頁面資產中繼資料。"""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到中繼檔：{metadata_path}")

    units: List[PageUnit] = []
    with metadata_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            units.append(PageUnit(**data))
    return units


def filter_non_toc(units: Iterable[PageUnit]) -> List[PageUnit]:
    """過濾掉疑似目錄頁，保留實際內容頁。"""
    return [u for u in units if not u.is_likely_toc]
