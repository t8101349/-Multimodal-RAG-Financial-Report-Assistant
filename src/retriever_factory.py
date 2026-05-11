"""視覺檢索後端路由器（雙軌設計）。

決策順序（對應 ``RETRIEVER_BACKEND`` 設定）：

    - ``byaldi``：強制使用 ColPali 主路徑；缺套件直接報錯
    - ``clip``：強制使用 CLIP 備援路徑；缺套件直接報錯
    - ``mock``：強制使用測試用後端（不應用於正式檢索）
    - ``auto``：先試 Byaldi，失敗再試 CLIP，最後退到 Mock
"""
from __future__ import annotations

from typing import Optional

from .retriever_base import BaseRetriever


def _try_byaldi(model_path: str, index_name: str, hf_offline: bool, hf_endpoint: str) -> Optional[BaseRetriever]:
    try:
        import byaldi  # noqa: F401
    except ImportError:
        return None
    from .retriever_byaldi import ByaldiRetriever

    return ByaldiRetriever(
        model_path=model_path,
        index_name=index_name,
        hf_offline=hf_offline,
        hf_endpoint=hf_endpoint,
    )


def _try_clip(clip_model_name: str) -> Optional[BaseRetriever]:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return None
    from .retriever_clip import CLIPRetriever

    return CLIPRetriever(model_name=clip_model_name)


def get_retriever(
    backend: str = "auto",
    *,
    colpali_model_path: str = "",
    colpali_index_name: str = "finance_report",
    clip_model_name: str = "openai/clip-vit-base-patch32",
    hf_offline: bool = False,
    hf_endpoint: str = "",
) -> BaseRetriever:
    """依設定回傳一個可用的視覺檢索後端。

    若選擇 ``auto``，會依序嘗試 byaldi -> clip -> mock。
    """
    backend = (backend or "auto").lower()

    if backend == "byaldi":
        retriever = _try_byaldi(colpali_model_path, colpali_index_name, hf_offline, hf_endpoint)
        if retriever is None:
            raise ImportError(
                "RETRIEVER_BACKEND=byaldi 但未安裝 byaldi，請改用 clip 或 auto"
            )
        return retriever

    if backend == "clip":
        retriever = _try_clip(clip_model_name)
        if retriever is None:
            raise ImportError(
                "RETRIEVER_BACKEND=clip 但缺少 torch/transformers，請先安裝"
            )
        return retriever

    if backend == "mock":
        from .retriever_mock import MockRetriever
        return MockRetriever()

    # auto：依序嘗試
    retriever = _try_byaldi(colpali_model_path, colpali_index_name, hf_offline, hf_endpoint)
    if retriever is not None:
        return retriever

    retriever = _try_clip(clip_model_name)
    if retriever is not None:
        return retriever

    from .retriever_mock import MockRetriever
    return MockRetriever()


def list_available_backends() -> dict[str, bool]:
    """探測目前環境中可用的後端，回傳 ``{name: available}``。"""
    status: dict[str, bool] = {"mock": True}

    try:
        import byaldi  # noqa: F401

        status["byaldi"] = True
    except ImportError:
        status["byaldi"] = False

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        status["clip"] = True
    except ImportError:
        status["clip"] = False

    return status
