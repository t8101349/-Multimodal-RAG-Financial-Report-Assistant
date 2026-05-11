# 多模態 RAG 財報助手（P05）

把企業財報、招股書等複雜 PDF 文件組織成一條可檢索、可解釋、可評測的多模態 RAG 流水線：

```
財報 PDF -> 頁面渲染 -> 視覺索引 -> 多頁召回 -> 證據組織 -> 多圖推理 -> 效果評測 -> 成本優化
```

本專案以「Vision-first」為核心：不依賴傳統 OCR 把版面壓成文字串，而是把整頁影像放入視覺索引，再交給多模態大模型做跨頁綜合理解。

---

## 目錄結構

```
project_RAG/
├── README.md                            本檔
├── requirements.txt                     相依套件
├── .env.example                         環境變數範本
├── config.py                            全域設定
├── src/
│   ├── pdf_renderer.py                  PDF 渲染與頁面資產層
│   ├── retriever_base.py                檢索後端介面
│   ├── retriever_byaldi.py              ColPali + Byaldi 主路徑
│   ├── retriever_clip.py                HuggingFace CLIP 備援路徑
│   ├── retriever_mock.py                關鍵字 mock（測試與煙霧）
│   ├── retriever_factory.py             雙軌路由
│   ├── prompts.py                       系統 / 使用者提示詞
│   ├── vlm_client.py                    OpenAI 相容 VLM 客戶端
│   ├── pipeline.py                      端到端流水線
│   └── evaluator.py                     雙層評測
├── scripts/
│   ├── make_sample_pdf.py               產生繁體中文測試 PDF
│   ├── build_index.py                   渲染 + 建索引
│   ├── run_query.py                     單筆查詢 CLI
│   ├── run_eval.py                      批次評測 CLI
│   ├── check_setup.py                   環境檢查
│   └── demo_no_deps.py                  零依賴煙霧演示
├── data/
│   ├── pdfs/                            來源 PDF
│   ├── page_images/                     渲染後頁面圖
│   ├── processed/                       中繼與索引
│   ├── eval/                            題庫與評測結果
│   └── reports/                         報告
└── tests/                               單元 / 煙霧測試
```

---

## 安裝

```bash
# 1. 建立虛擬環境（建議 Python 3.10+）
python -m venv .venv
.\.venv\Scripts\activate         # Windows
# source .venv/bin/activate      # macOS / Linux

# 2. 安裝相依套件
pip install -r requirements.txt

# 3. 設定 .env
copy .env.example .env           # Windows
# cp .env.example .env           # macOS / Linux
# 編輯 .env，至少填入 VLM_BASE_URL 與 VLM_API_KEY
```

`requirements.txt` 將相依分成三組：

| 分組 | 必要性 | 內容 |
| --- | --- | --- |
| 基礎 | 必裝 | `PyMuPDF`, `Pillow`, `numpy`, `requests`, `python-dotenv`, `pydantic`, `reportlab` |
| 視覺檢索主路徑 | 建議 | `byaldi`, `colpali-engine`（依需求自行 `pip install`） |
| 視覺檢索備援 | 建議 | `torch`, `transformers`（CLIP 系列模型） |

---

## 快速開始

### 1. 環境檢查

```bash
python scripts/check_setup.py
```

會列出每一個關鍵套件是否安裝、視覺檢索後端是否可用、VLM 設定是否就緒。

### 2. 產生測試 PDF（可選）

若手邊沒有真實財報，先用內建的繁體中文範例 PDF 跑通流水線：

```bash
python scripts/make_sample_pdf.py
# 輸出：data/pdfs/sample_finance_report.pdf（12 頁，含目錄、表格、圖表）
```

### 3. 建立索引

```bash
python scripts/build_index.py --pdf data/pdfs/sample_finance_report.pdf --backend auto
```

`--backend` 可選：

- `auto`（預設）：先試 Byaldi，失敗後退到 CLIP，最後退到 mock
- `byaldi`：強制使用 ColPali 主路徑
- `clip`：強制使用 CLIP 備援
- `mock`：強制使用關鍵字 mock（僅供測試）

### 4. 提問

```bash
python scripts/run_query.py --question "2024 年度的合併營業收入是多少？"
```

若尚未設定 VLM 環境變數，會自動切換到 `MockVLMClient`，仍可驗證檢索鏈路。

### 5. 跑整批評測

```bash
python scripts/run_eval.py
```

產出：

- `data/reports/p5_report.md`：人類可讀報告
- `data/reports/p5_metrics.json`：機器可讀指標
- `data/eval/evaluation_results.jsonl`：逐題明細
- `data/eval/failure_replay.jsonl`：失敗樣本

### 零依賴煙霧演示（無 PyMuPDF / Byaldi / CLIP / VLM）

```bash
python scripts/demo_no_deps.py
```

直接生成合成 `page_units.jsonl` 與占位頁面圖，跑一輪 mock 索引 + mock VLM 的完整流程；
適合在裝任何重型套件之前確認程式碼結構是否正確。

---

## 設計重點對照

| 文章重點 | 對應模組 | 對應檔案 |
| --- | --- | --- |
| 頁面渲染與視覺索引 | 頁面資產層 | `src/pdf_renderer.py` |
| 多頁召回與目錄抑制 | 雙層過濾（資產層辨識 + 流水線過濾） | `_looks_like_toc` + `MultimodalRAGPipeline.query` |
| Vision-first 雙軌設計 | Byaldi 主路徑 + CLIP 備援 | `retriever_byaldi.py` / `retriever_clip.py` |
| 多圖推理 + 抗目錄 Prompt | 系統與使用者提示模板 | `src/prompts.py` |
| 結構化輸出 | 結論 / 數值 / 趨勢 / 頁碼 / 不確定性 | `src/prompts.py` `SYSTEM_PROMPT` |
| 雙層評測 | 檢索命中率 + 引用準確率 + 關鍵字召回 | `src/evaluator.py` |
| 失敗重播 | `failure_replay.jsonl` | `evaluate()` 內 |

---

## 測試

```bash
pytest tests/ -v
```

目前測試覆蓋：

- `test_prompts.py`：系統提示關鍵字、使用者文字組合、抗目錄旗標、圖片 payload
- `test_evaluator.py`：頁碼擷取（含全形）、關鍵字命中、題目反序列化
- `test_retriever_mock.py`：mock 索引建立、查詢、目錄頁降權、載入
- `test_factory.py`：後端能力列出、強制與 auto 路由
- `test_pdf_renderer_logic.py`：目錄頁啟發式（需 PyMuPDF）
- `test_pipeline_smoke.py`：端到端煙霧（含目錄頁過濾、log 寫入）

所有測試皆設計為「即使缺少重型依賴也能跑」。

---

## 環境變數

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `VLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI 相容 endpoint（DashScope / 本地 vLLM） |
| `VLM_API_KEY` | (空) | 上面 endpoint 對應的 API 金鑰 |
| `VLM_MODEL` | `qwen2.5-vl-72b-instruct` | 模型名稱 |
| `RETRIEVER_BACKEND` | `auto` | `auto` / `byaldi` / `clip` / `mock` |
| `COLPALI_MODEL_PATH` | `./models/colpali-v1.2-merged` | ColPali 本地權重 |
| `COLPALI_INDEX_NAME` | `finance_report` | Byaldi 索引名稱 |
| `CLIP_MODEL_NAME` | `openai/clip-vit-base-patch32` | CLIP 模型（HuggingFace） |
| `HF_HUB_OFFLINE` | `0` | 是否離線載入 HF 模型 |
| `HF_ENDPOINT` | `https://huggingface.co` | HF 鏡像 |
| `RETRIEVAL_TOP_K` | `4` | Top-K 召回 |
| `PAGE_RENDER_DPI` | `200` | PDF 渲染解析度 |

---

## 如何接入真實 VLM

要把 .env.example 複製成 .env 並填入金鑰
`src/vlm_client.py` 的 `VLMClient` 完全相容 OpenAI Chat Completions 規格。

範例（DashScope）：

```bash
export VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export VLM_API_KEY=sk-...
export VLM_MODEL=qwen2.5-vl-72b-instruct
```

範例（本地 vLLM）：

```bash
# 先啟動 vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct --port 8000

export VLM_BASE_URL=http://localhost:8000/v1
export VLM_API_KEY=EMPTY  # vLLM 預設不檢查
export VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

---
## 如何換成自己的財報 PDF
把任何中文財報放到 data/pdfs/your_report.pdf，再執行：

```bash
python scripts/build_index.py --pdf data/pdfs/your_report.pdf --backend auto
```

---

## 工作流程小結

1. `make_sample_pdf.py` 或自備 PDF → `data/pdfs/`
2. `build_index.py` → 產生 `page_images/` + `page_units.jsonl` + `<backend>_index/`
3. `run_query.py` 或 `run_eval.py` → 取用上述產物完成檢索 + 生成
4. 檢視 `data/reports/` 與 `data/eval/` 以追蹤回歸與失敗樣本

如需擴充：

- 新增檢索後端：實作 `BaseRetriever` 並在 `retriever_factory.get_retriever()` 註冊
- 新增評測指標：擴充 `evaluator.evaluate_case`
- 新增提示策略：在 `prompts.py` 增補 `build_prompt` 變體
