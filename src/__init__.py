"""多模態 RAG 財報助手核心套件。

模組總覽：
    pdf_renderer   - PDF 頁面渲染與資產層
    retriever_*    - 視覺檢索後端（Byaldi / CLIP / Mock）
    vlm_client     - OpenAI 相容多模態模型客戶端
    prompts        - 提示詞模板
    pipeline       - 端到端流水線整合
    evaluator      - 評測模組
"""

__all__ = [
    "pdf_renderer",
    "retriever_base",
    "retriever_byaldi",
    "retriever_clip",
    "retriever_factory",
    "retriever_mock",
    "vlm_client",
    "prompts",
    "pipeline",
    "evaluator",
]
