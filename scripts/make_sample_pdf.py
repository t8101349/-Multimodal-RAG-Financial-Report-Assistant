"""產生測試用繁體中文財報 PDF。

目的：
    - 在沒有真實財報 PDF 時，仍能驗證 :mod:`pipeline` 端到端流程
    - 內容刻意安排目錄頁、表格頁、圖表頁、噪聲頁，
      以便評測「抗目錄」、「跨頁綜合」與「圖表理解」三類能力

需求：``pip install reportlab``

執行：
    python scripts/make_sample_pdf.py --output data/pdfs/sample_finance_report.pdf
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.graphics.charts.barcharts import VerticalBarChart
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "請先安裝 reportlab：pip install reportlab"
    ) from exc


# 候選的繁體中文 TTF 字型（按優先順序）
_CANDIDATE_FONTS = [
    ("MSJH", r"C:\Windows\Fonts\msjh.ttc"),
    ("MSJHBD", r"C:\Windows\Fonts\msjhbd.ttc"),
    ("MingLiU", r"C:\Windows\Fonts\mingliu.ttc"),
    ("KaiU", r"C:\Windows\Fonts\kaiu.ttf"),
    ("SimSun", r"C:\Windows\Fonts\simsun.ttc"),
    ("NotoSansCJKTC", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ("PingFang", "/System/Library/Fonts/PingFang.ttc"),
]


def register_chinese_font() -> str:
    """嘗試在常見系統字型路徑中找到一個可用的中文字型並註冊。"""
    for name, path in _CANDIDATE_FONTS:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception as exc:  # pragma: no cover
                print(f"[字型] 嘗試 {name} ({path}) 失敗：{exc}")
                continue
    raise RuntimeError(
        "找不到可用的中文 TTF 字型。請手動安裝或修改 _CANDIDATE_FONTS。"
    )


def make_styles(font_name: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["BodyText"]
    return {
        "title": ParagraphStyle(
            "title",
            parent=base,
            fontName=font_name,
            fontSize=22,
            leading=28,
            spaceAfter=16,
            alignment=1,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base,
            fontName=font_name,
            fontSize=16,
            leading=22,
            spaceAfter=10,
            textColor=colors.HexColor("#1f2d5a"),
        ),
        "body": ParagraphStyle(
            "body",
            parent=base,
            fontName=font_name,
            fontSize=12,
            leading=18,
            spaceAfter=8,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base,
            fontName=font_name,
            fontSize=10,
            leading=14,
        ),
        "toc": ParagraphStyle(
            "toc",
            parent=base,
            fontName=font_name,
            fontSize=12,
            leading=20,
            spaceAfter=4,
        ),
    }


def make_table(data: list[list[str]], font_name: str, col_widths=None) -> Table:
    table = Table(data, colWidths=col_widths)
    style = TableStyle(
        [
            ("FONT", (0, 0), (-1, -1), font_name, 11),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2d5a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]
    )
    table.setStyle(style)
    return table


def make_rnd_chart(font_name: str) -> Drawing:
    """研發投入近三年趨勢長條圖。"""
    drawing = Drawing(420, 240)
    chart = VerticalBarChart()
    chart.x = 60
    chart.y = 60
    chart.width = 320
    chart.height = 160
    chart.data = [[8.6, 10.4, 12.7]]   # 單位：億元
    chart.categoryAxis.categoryNames = ["2022 年", "2023 年", "2024 年"]
    chart.bars[0].fillColor = colors.HexColor("#3674b5")
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 16
    chart.valueAxis.valueStep = 2
    chart.barLabels.fontName = font_name
    chart.barLabels.fontSize = 10
    chart.barLabelFormat = "%.1f 億"
    chart.barLabels.nudge = 12
    chart.categoryAxis.labels.fontName = font_name
    chart.categoryAxis.labels.fontSize = 11
    chart.valueAxis.labels.fontName = font_name
    chart.valueAxis.labels.fontSize = 10

    drawing.add(chart)
    drawing.add(
        String(
            210,
            10,
            "圖表 1：繁星科技研發投入趨勢（單位：新台幣億元）",
            fontName=font_name,
            fontSize=10,
            textAnchor="middle",
        )
    )
    return drawing


def build_story(font_name: str) -> list:
    styles = make_styles(font_name)
    story: list = []

    # ---------------- Page 1：封面 ----------------
    story.append(Spacer(1, 6 * cm))
    story.append(Paragraph("繁星科技股份有限公司", styles["title"]))
    story.append(Paragraph("2024 年度報告", styles["title"]))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("股票代號：6688　產業類別：半導體", styles["body"]))
    story.append(Paragraph("發行日期：2025 年 3 月 29 日", styles["body"]))
    story.append(PageBreak())

    # ---------------- Page 2：目錄 ----------------
    story.append(Paragraph("目錄", styles["h2"]))
    toc_rows = [
        ("致股東報告書 ............................................................ 3", ),
        ("第一章　經營概要 ........................................................ 4", ),
        ("第二章　核心財務指標 .................................................. 5", ),
        ("第三章　研發投入趨勢 .................................................. 6", ),
        ("第四章　業務板塊收入結構 .......................................... 7", ),
        ("第五章　資產負債表 ...................................................... 8", ),
        ("第六章　重要會計政策附註 .......................................... 9", ),
        ("第七章　風險因素 ........................................................ 10", ),
        ("第八章　企業社會責任 ................................................ 11", ),
        ("第九章　後續事項 ........................................................ 12", ),
    ]
    for (line,) in toc_rows:
        story.append(Paragraph(line, styles["toc"]))
    story.append(PageBreak())

    # ---------------- Page 3：致股東 ----------------
    story.append(Paragraph("致股東報告書", styles["h2"]))
    story.append(
        Paragraph(
            "2024 年度，繁星科技持續強化先進製程研發能量，並深化與全球客戶於 AI "
            "加速器、車用半導體與資料中心領域的合作。本年度集團合併營業收入創歷史新高，"
            "達新台幣 482.6 億元，年增 18.7%；歸屬母公司業主之淨利為 92.4 億元，"
            "年增 15.3%。",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "展望 2025 年，本公司將持續投入新世代封裝技術、提高高毛利產品比重，"
            "並關注地緣政治與匯率波動對成本端的影響。董事會誠摯感謝股東長期以來的支持。",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # ---------------- Page 4：經營概要 ----------------
    story.append(Paragraph("第一章　經營概要", styles["h2"]))
    story.append(
        Paragraph(
            "本集團主要業務包含三大板塊：先進製程晶圓代工、特殊製程晶圓代工，以及"
            "封裝測試。2024 年三大板塊合計佔合併營業收入 96.4%，其餘 3.6% 為"
            "技術授權與工程服務。",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "報告年度內，公司於台南科學園區完成 3 奈米製程量產，並於高雄路竹建立"
            "新一代先進封裝產線，預計 2025 年下半年量產。",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # ---------------- Page 5：核心財務指標表 ----------------
    story.append(Paragraph("第二章　核心財務指標", styles["h2"]))
    fin_table = [
        ["項目（單位：新台幣億元）", "2024 年", "2023 年", "年增率"],
        ["營業收入", "482.6", "406.5", "+18.7%"],
        ["營業毛利", "212.4", "171.8", "+23.6%"],
        ["營業利益", "118.3", "94.7", "+24.9%"],
        ["稅後淨利", "92.4", "80.1", "+15.3%"],
        ["每股盈餘 EPS（元）", "9.24", "8.02", "+15.2%"],
        ["研發投入", "12.7", "10.4", "+22.1%"],
        ["研發投入佔營收比", "2.6%", "2.6%", "+0.0pp"],
    ]
    story.append(
        make_table(fin_table, font_name, col_widths=[6.5 * cm, 3 * cm, 3 * cm, 2.5 * cm])
    )
    story.append(Spacer(1, 0.6 * cm))
    story.append(
        Paragraph(
            "註：上述數字來自合併財務報表，已經會計師查核。",
            styles["small"],
        )
    )
    story.append(PageBreak())

    # ---------------- Page 6：研發投入趨勢 ----------------
    story.append(Paragraph("第三章　研發投入趨勢", styles["h2"]))
    story.append(
        Paragraph(
            "本公司近三年研發投入持續成長，2024 年達 12.7 億元，較 2022 年的 "
            "8.6 億元成長 47.7%，反映集團對先進製程與封裝技術的長期承諾。",
            styles["body"],
        )
    )
    story.append(make_rnd_chart(font_name))
    story.append(PageBreak())

    # ---------------- Page 7：業務板塊收入結構 ----------------
    story.append(Paragraph("第四章　業務板塊收入結構", styles["h2"]))
    rev_table = [
        ["業務板塊", "2024 營收（億元）", "佔比", "年增率"],
        ["先進製程晶圓代工", "248.7", "51.5%", "+24.2%"],
        ["特殊製程晶圓代工", "126.4", "26.2%", "+11.8%"],
        ["封裝測試", "90.0", "18.7%", "+15.6%"],
        ["技術授權與工程服務", "17.5", "3.6%", "+8.1%"],
        ["合計", "482.6", "100.0%", "+18.7%"],
    ]
    story.append(make_table(rev_table, font_name, col_widths=[6 * cm, 4 * cm, 2.5 * cm, 2.5 * cm]))
    story.append(PageBreak())

    # ---------------- Page 8：資產負債表 ----------------
    story.append(Paragraph("第五章　資產負債表（彙總）", styles["h2"]))
    bs_table = [
        ["項目（單位：新台幣億元）", "2024 年末", "2023 年末"],
        ["流動資產", "316.8", "275.4"],
        ["非流動資產", "542.1", "498.6"],
        ["總資產", "858.9", "774.0"],
        ["流動負債", "192.5", "176.2"],
        ["非流動負債", "180.4", "172.9"],
        ["總負債", "372.9", "349.1"],
        ["歸屬母公司業主權益", "486.0", "424.9"],
    ]
    story.append(make_table(bs_table, font_name, col_widths=[7 * cm, 4 * cm, 4 * cm]))
    story.append(PageBreak())

    # ---------------- Page 9：附註 ----------------
    story.append(Paragraph("第六章　重要會計政策附註", styles["h2"]))
    story.append(
        Paragraph(
            "本公司合併財務報表係依國際財務報導準則（IFRSs）編製。存貨採加權平均法"
            "計算成本，並以成本與淨變現價值孰低法評價。固定資產採直線法折舊，"
            "建築物耐用年限 20 至 40 年，機器設備 5 至 10 年。",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "無形資產主要包含技術授權與專利，依據合約年限或預估使用年限攤銷，"
            "目前帳列無形資產 38.6 億元，年增 8.2%。",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # ---------------- Page 10：風險因素 ----------------
    story.append(Paragraph("第七章　風險因素", styles["h2"]))
    story.append(
        Paragraph(
            "本公司主要風險包含：(1) 全球景氣循環導致需求波動；(2) 主要客戶集中度偏高，"
            "前五大客戶營收貢獻佔 62.4%；(3) 地緣政治造成關鍵原物料供應風險；"
            "(4) 匯率波動，以美元計價收入比重達 78.5%。",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "公司已建立避險工具與多元供應鏈，並透過長約鎖定關鍵材料價格以分散風險。",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # ---------------- Page 11：社會責任 ----------------
    story.append(Paragraph("第八章　企業社會責任", styles["h2"]))
    story.append(
        Paragraph(
            "本公司持續推動淨零碳排目標，2024 年合併營運溫室氣體排放強度較基準年下降 16%，"
            "並通過 ISO 14064-1 第三方查證。",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "在社會貢獻方面，公司贊助 STEM 教育及在地產學合作計畫，2024 年累計培訓 "
            "1,820 名工程師。",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # ---------------- Page 12：後續事項 ----------------
    story.append(Paragraph("第九章　後續事項", styles["h2"]))
    story.append(
        Paragraph(
            "2025 年 1 月，本公司董事會通過配發 2024 年度現金股利每股 4.5 元，"
            "預計於 2025 年 6 月除息。",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "2025 年 2 月，本公司與 A 客戶簽訂為期 5 年的長期供應合約，"
            "預計貢獻合約期間累計營收約 850 億元。",
            styles["body"],
        )
    )
    return story


def generate(output_path: Path) -> Path:
    font_name = register_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="繁星科技 2024 年度報告（測試用）",
    )
    story = build_story(font_name)
    doc.build(story)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="產生繁體中文測試財報 PDF")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "pdfs" / "sample_finance_report.pdf"),
        help="PDF 輸出路徑",
    )
    args = parser.parse_args()

    out = Path(args.output)
    print(f"[make_sample_pdf] 開始產生 PDF：{out}")
    generate(out)
    size_kb = out.stat().st_size / 1024
    print(f"[make_sample_pdf] 完成，檔案大小：{size_kb:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
