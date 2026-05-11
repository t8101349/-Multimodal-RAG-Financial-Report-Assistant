"""OpenAI 相容 VLM 客戶端。

不依賴 ``openai`` SDK，直接以 ``requests`` 對接 chat-completions endpoint。
這樣可同時支援：
    - 阿里雲 DashScope 的 OpenAI 相容介面
    - 本地 vLLM
    - 任何符合 OpenAI Chat Completions 規格的閘道

設計目標：
    - 介面單純：給一組 ``messages`` 與圖檔，就能拿到回答
    - 對 429 / 5xx 自動退避重試
    - 紀錄延遲、token 用量、輸入輸出尺寸，供後續評測與成本分析使用
    - 提供 ``MockVLMClient`` 供無 API 環境的測試
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


@dataclass
class VLMResponse:
    """VLM 回應與遙測。"""

    content: str
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def encode_image_to_base64(image_path: str | Path) -> str:
    """讀取頁面圖檔並轉為 base64（不含 data: 前綴）。"""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"找不到圖檔：{image_path}")
    with image_path.open("rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


class VLMClient:
    """OpenAI Chat Completions 相容客戶端。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        if not base_url:
            raise ValueError("VLM base_url 不可為空，請設定 VLM_BASE_URL")
        if not api_key:
            raise ValueError("VLM api_key 不可為空，請設定 VLM_API_KEY")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        extra: Optional[Dict[str, Any]] = None,
    ) -> VLMResponse:
        """送出一次 chat completion 呼叫。"""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra:
            payload.update(extra)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            start = time.perf_counter()
            try:
                resp = requests.post(
                    self._endpoint(),
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                latency_ms = (time.perf_counter() - start) * 1000

                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = RuntimeError(
                        f"VLM 暫時錯誤 (status={resp.status_code})：{resp.text[:200]}"
                    )
                    sleep_s = self.retry_backoff * (attempt + 1)
                    time.sleep(sleep_s)
                    continue

                resp.raise_for_status()
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                usage = data.get("usage", {}) or {}
                return VLMResponse(
                    content=content,
                    model=data.get("model", self.model),
                    latency_ms=latency_ms,
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    total_tokens=int(usage.get("total_tokens", 0) or 0),
                    raw=data,
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(self.retry_backoff * (attempt + 1))

        raise RuntimeError(
            f"VLM 呼叫失敗，已重試 {self.max_retries} 次；最後錯誤：{last_error}"
        )


class MockVLMClient:
    """測試與 dry-run 模式使用的假 VLM。"""

    def __init__(self, canned_answer: Optional[str] = None) -> None:
        self.canned_answer = canned_answer or (
            "**結論**：（Mock 回答）已收到問題與頁面，但目前處於離線模式，"
            "無實際 VLM 推理。\n"
            "**關鍵數值**：無\n"
            "**趨勢解讀**：不適用\n"
            "**證據頁碼**：（無）\n"
            "**不確定性**：所有內容皆由 MockVLMClient 產生，請接入正式 VLM 後再評估。"
        )
        self.calls: List[Dict[str, Any]] = []

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        extra: Optional[Dict[str, Any]] = None,
    ) -> VLMResponse:
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        return VLMResponse(
            content=self.canned_answer,
            model="mock-vlm",
            latency_ms=1.0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            raw={"mock": True},
        )
