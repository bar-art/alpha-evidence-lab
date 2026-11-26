"""OpenRouter 结构化输出客户端：响应缓存、调用次数与费用上限；不记录、不导出密钥。

只用标准库发送 HTTPS 请求，不增加运行依赖。缓存键是去掉密钥后的请求体哈希：
相同请求重跑不再计费，并且结果可复现（模型在 temperature=0 时仍可能有非确定性）。
"""

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import LlmSettings, canonical, digest

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
Transport = Callable[[dict[str, Any], str], dict[str, Any]]


MAX_REPAIRS = 2
MAX_FLAG_RETRIES = 2
MAX_OUTPUT_TOKENS = 2000


class LlmBudgetError(ValueError):
    pass


class ProviderFlagged(ValueError):
    """服务商内容过滤器的误判（HTTP 400 flagged）；同一请求重发通常可通过。"""


def schema_problem(output: dict[str, Any], schema: dict[str, Any]) -> str:
    """按本项目用到的 schema 子集检查：必填、无额外字段、类型、枚举。返回问题描述或空串。"""
    properties = schema.get("properties", {})
    missing = [key for key in schema.get("required", []) if key not in output]
    extra = [key for key in output if key not in properties]
    if missing or extra:
        return f"缺少字段 {missing}，多余字段 {extra}"
    for key, rule in properties.items():
        value = output.get(key)
        if rule.get("type") == "string":
            if not isinstance(value, str) or ("enum" in rule and value not in rule["enum"]):
                return f"{key} 不是允许的字符串"
        if rule.get("type") == "array" and (
            not isinstance(value, list) or not all(isinstance(item, str) for item in value)
        ):
            return f"{key} 不是字符串列表"
    return ""


def http_transport(body: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return dict(json.loads(response.read()))
    except urllib.error.HTTPError as error:
        # 只报告状态码和服务端错误摘要；请求头（含密钥）不进入异常信息。
        detail = error.read().decode(errors="replace")[:200]
        kind = ProviderFlagged if error.code == 400 and "flagged" in detail else ValueError
        raise kind(f"OpenRouter HTTP {error.code}：{detail}") from None


class OpenRouterClient:
    def __init__(
        self,
        settings: LlmSettings,
        cache_path: Path,
        seed: int,
        api_key: str | None = None,
        transport: Transport = http_transport,
    ) -> None:
        self.settings, self.cache_path, self.seed = settings, cache_path, seed
        self._api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "")
        self.transport = transport
        self.calls = self.cache_hits = self.repairs = self.flag_retries = 0
        self.cost_usd = 0.0
        self.prompt_tokens = self.completion_tokens = 0
        self._cache: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                self._cache[entry["key"]] = entry

    def complete(
        self, system: str, user: dict[str, Any], schema: dict[str, Any], name: str
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": canonical(user)},
        ]
        body = self._body(messages, schema, name)
        # 缓存键只取首次请求（不含输出长度上限，它不改变合规结果）；修复重试的结果也存在这个键下。
        key = digest({k: v for k, v in body.items() if k != "max_tokens"})
        cached = self._cache.get(key)
        if cached is not None and not schema_problem(cached["output"], schema):
            self.cache_hits += 1
            return dict(cached["output"])
        for repair in range(MAX_REPAIRS + 1):
            content, response = self._send(body)
            try:
                output = dict(json.loads(content))
                problem = schema_problem(output, schema)
            except (TypeError, ValueError):
                output, problem = {}, "不是 JSON 对象" if content else "回复为空"
            if not problem:
                break
            # 格式不合规是通信问题：把错误告诉模型重答，费用照记，不占研究的尝试预算。
            self.repairs += 1
            if repair == MAX_REPAIRS:
                raise ValueError(f"模型连续 {MAX_REPAIRS + 1} 次输出不符合协议：{problem}")
            previous = [{"role": "assistant", "content": content}] if content else []
            messages = [
                *messages,
                *previous,
                {
                    "role": "user",
                    "content": f"Your previous reply was invalid ({problem}). Reply now with "
                    "exactly one JSON object containing only the schema's fields.",
                },
            ]
            body = self._body(messages, schema, name)
        entry = {
            "key": key,
            "model": response.get("model", self.settings.model),
            "provider": response.get("provider"),
            "repairs": repair,
            "request": body,
            "output": output,
            "usage": response.get("usage") or {},
        }
        self._cache[key] = entry
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return output

    def _body(
        self, messages: list[dict[str, str]], schema: dict[str, Any], name: str
    ) -> dict[str, Any]:
        return {
            "model": self.settings.model,
            "messages": messages,
            **(
                {"temperature": self.settings.temperature}
                if self.settings.temperature is not None
                else {}
            ),
            "seed": self.seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
            "reasoning": {"effort": self.settings.reasoning_effort},
            # 输出上限：模型偶尔陷入重复生成；截断后按格式错误重试，而不是无限等待。
            "max_tokens": MAX_OUTPUT_TOKENS,
            # 只路由到支持上述参数（结构化输出、seed）的服务商；返回实际费用。
            "provider": self._routing(),
            "usage": {"include": True},
        }

    def _send(self, body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if self.calls >= self.settings.max_calls:
            raise LlmBudgetError(f"大模型调用次数达到上限 {self.settings.max_calls}，停止研究")
        if self.cost_usd >= self.settings.max_cost_usd:
            raise LlmBudgetError(f"大模型费用达到上限 ${self.settings.max_cost_usd}，停止研究")
        if not self._api_key:
            raise ValueError("缺少 OPENROUTER_API_KEY 环境变量；不会改用规则结果")
        for attempt in range(MAX_FLAG_RETRIES + 1):
            self.calls += 1
            try:
                response = self.transport(body, self._api_key)
                break
            except ProviderFlagged:
                # 过滤器误判：原样重发，次数计入调用；连续被拦截则停止研究。
                self.flag_retries += 1
                if attempt == MAX_FLAG_RETRIES or self.calls >= self.settings.max_calls:
                    raise
        usage = response.get("usage") or {}
        self.cost_usd += float(usage.get("cost") or 0.0)
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        try:
            content = str(response["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError):
            content = ""
        return content, response

    def _routing(self) -> dict[str, Any]:
        routing: dict[str, Any] = {"require_parameters": True}
        if self.settings.provider:
            routing.update(only=[self.settings.provider], allow_fallbacks=False)
        return routing

    def usage(self) -> dict[str, Any]:
        return {
            "model": self.settings.model,
            "provider_routing": self.settings.provider,
            "reasoning_effort": self.settings.reasoning_effort,
            "release_date": str(self.settings.release_date),
            "release_source": self.settings.release_source,
            "api_calls": self.calls,
            "format_repairs": self.repairs,
            "content_filter_retries": self.flag_retries,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }
