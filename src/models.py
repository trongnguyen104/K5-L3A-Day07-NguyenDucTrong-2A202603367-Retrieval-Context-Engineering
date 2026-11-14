"""Adapter chat model (OpenAI-compatible) cấu hình qua biến môi trường.

Dùng cho agent Day 7: trả JSON đã parse để `Controller.decide` quyết định
search hay answer. Không có fallback giả khi model lỗi.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import ChatConfig


class ModelError(RuntimeError):
    """Lỗi gọi model; lan ra ngoài để trace ghi nhận thay vì trả kết quả giả."""


class ChatModel:
    name: str

    def __init__(self, config: ChatConfig | None = None):
        self.config = config or ChatConfig.from_env()
        if not self.config.api_key:
            raise ModelError('Thiếu LLM_API_KEY (hoặc OPENAI_API_KEY) cho chat model')
        self.name = f'openai-compatible:{self.config.model}'
        self.last_usage: dict[str, Any] | None = None
        self._url = self.config.api_base.rstrip('/') + '/chat/completions'

    def complete(self, system: str, user: str) -> str:
        payload = {
            'model': self.config.model,
            'temperature': self.config.temperature,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
        }
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode('utf-8'),
            method='POST',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.config.api_key}',
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as error:  # pragma: no cover - phụ thuộc mạng
            detail = error.read().decode('utf-8', errors='replace')[:300]
            raise ModelError(f'chat API HTTP {error.code}: {detail}') from error
        except (urllib.error.URLError, TimeoutError) as error:  # pragma: no cover - phụ thuộc mạng
            raise ModelError(f'không gọi được chat API: {error}') from error
        usage = body.get('usage')
        self.last_usage = usage if isinstance(usage, dict) else None
        try:
            return body['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as error:
            raise ModelError('phản hồi chat API sai định dạng') from error

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Gọi model và parse JSON; báo lỗi rõ nếu model không trả JSON hợp lệ."""
        raw = self.complete(system, user).strip()
        if raw.startswith('```'):
            raw = raw.strip('`')
            raw = raw.split('\n', 1)[1] if '\n' in raw else raw
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ModelError(f'model không trả JSON hợp lệ: {raw[:200]}') from error
        if not isinstance(parsed, dict):
            raise ModelError('model phải trả một JSON object')
        return parsed
