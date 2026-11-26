"""OpenRouter 结构化输出客户端：响应缓存、调用次数与费用上限；不记录、不导出密钥。

只用标准库发送 HTTPS 请求，不增加运行依赖。缓存键是去掉密钥后的请求体哈希：
相同请求重跑不再计费，并且结果可复现（模型在 temperature=0 时仍可能有非确定性）。
"""

import json
import math
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


Pricing = dict[str, float]


def valid_cost(value: Any) -> float | None:
    """usage.cost 的合法值；缺失、非数值、负数、NaN/Inf 返回 None（未知，不当成 0）。"""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    cost = float(value)
    return cost if math.isfinite(cost) and cost >= 0 else None


def http_error_category(code: int, flagged: bool) -> str:
    if flagged:
        return "content_filter"
    if code in {401, 403}:
        return "authentication"
    if code == 402:
        return "insufficient_credits"
    if code in {408, 429}:
        return "rate_limit_or_timeout"
    if code >= 500:
        return "provider_error"
    return "request_rejected"


def fetch_pricing(model: str, provider: str | None) -> Pricing | None:
    """从 OpenRouter 公开端点目录读取单价（美元/token）。指定端点时取该端点，否则取各端点最高价。

    读取失败返回 None：此时费用上限只能作为软阈值，报告会写明。
    """
    url = f"https://openrouter.ai/api/v1/models/{model}/endpoints"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            endpoints = json.loads(response.read())["data"]["endpoints"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    matches = [
        item
        for item in endpoints
        if provider is None
        or item.get("tag") == provider
        or str(item.get("provider_name", "")).lower() == provider.lower()
    ]
    try:
        prompt = max(float(item["pricing"]["prompt"]) for item in matches)
        completion = max(float(item["pricing"]["completion"]) for item in matches)
    except (KeyError, TypeError, ValueError):
        return None
    return {"prompt": prompt, "completion": completion}


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
        # 只报告状态码和安全分类：服务商错误正文可能回显密钥或提示词，不进入异常、日志或缓存。
        text = error.read().decode(errors="replace")
        flagged = error.code == 400 and "flagged" in text
        request_id = error.headers.get("x-request-id") if error.headers else None
        kind = ProviderFlagged if flagged else ValueError
        suffix = f"，请求ID {request_id}" if request_id else ""
        raise kind(
            f"OpenRouter HTTP {error.code}（{http_error_category(error.code, flagged)}）{suffix}"
        ) from None


class OpenRouterClient:
    def __init__(
        self,
        settings: LlmSettings,
        cache_path: Path,
        seed: int,
        api_key: str | None = None,
        transport: Transport = http_transport,
        pricing: Pricing | None = None,
    ) -> None:
        self.settings, self.cache_path, self.seed = settings, cache_path, seed
        # 有可信单价时，每次发送前按最坏情况预留费用（硬上限）；否则只能作为软阈值。
        self.pricing = pricing
        self.unknown_cost_calls = 0
        self.unknown_cost_reserved = 0.0
        self.budget_exceeded = False
        self._api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "")
        self.transport = transport
        self.calls = self.cache_hits = self.repairs = self.flag_retries = 0
        self.cached_source_cost = 0.0
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
            # 缓存命中不产生新费用，但原始调用的费用仍单独报告。
            self.cached_source_cost += valid_cost((cached.get("usage") or {}).get("cost")) or 0.0
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
        if self.budget_exceeded:
            # 已付费的响应已保存，但实验不再完整：停止，不把它当作预算内结果使用。
            raise LlmBudgetError(
                f"费用 ${self.committed_cost():.6f} 超过上限 ${self.settings.max_cost_usd}"
                "（budget_exceeded）；已发生的调用已保存，研究不完整"
            )
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

    def committed_cost(self) -> float:
        """已结算费用加上费用未知调用的预留上界。"""
        return self.cost_usd + self.unknown_cost_reserved

    def reservation(self, body: dict[str, Any]) -> float | None:
        """单次请求最坏费用：请求字符数（token数不超过字符数）× 输入单价 + 输出上限 × 输出单价。"""
        if self.pricing is None:
            return None
        prompt_chars = len(json.dumps(body["messages"], ensure_ascii=False)) + len(
            json.dumps(body.get("response_format", {}), ensure_ascii=False)
        )
        return (
            prompt_chars * self.pricing["prompt"]
            + int(body.get("max_tokens") or 0) * (self.pricing["completion"])
        )

    def _send(self, body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if self.budget_exceeded:
            raise LlmBudgetError("已超过费用上限（budget_exceeded），不再发送付费请求")
        if not self._api_key:
            raise ValueError("缺少 OPENROUTER_API_KEY 环境变量；不会改用规则结果")
        reserve = self.reservation(body)
        for attempt in range(MAX_FLAG_RETRIES + 1):
            if self.calls >= self.settings.max_calls:
                raise LlmBudgetError(f"大模型调用次数达到上限 {self.settings.max_calls}，停止研究")
            if reserve is not None:
                if self.committed_cost() + reserve > self.settings.max_cost_usd:
                    raise LlmBudgetError(
                        f"预留最坏费用 ${reserve:.6f} 后将超过上限 ${self.settings.max_cost_usd}，"
                        "请求未发送"
                    )
            elif self.committed_cost() >= self.settings.max_cost_usd:
                raise LlmBudgetError(f"大模型费用达到上限 ${self.settings.max_cost_usd}，停止研究")
            self.calls += 1
            try:
                response = self.transport(body, self._api_key)
                break
            except ProviderFlagged:
                # 过滤器误判：原样重发，次数计入调用；连续被拦截则停止研究。
                self.flag_retries += 1
                if attempt == MAX_FLAG_RETRIES:
                    raise
        usage = response.get("usage") or {}
        cost = valid_cost(usage.get("cost"))
        if cost is None:
            # 费用未知：不当成 0。有可信上界时按预留额计入；没有则无法保证上限，停止。
            self.unknown_cost_calls += 1
            if reserve is None:
                self.budget_exceeded = True
            else:
                self.unknown_cost_reserved += reserve
        else:
            self.cost_usd += cost
        if self.committed_cost() > self.settings.max_cost_usd:
            self.budget_exceeded = True
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
            "unknown_cost_calls": self.unknown_cost_calls,
            "unknown_cost_reserved_usd": round(self.unknown_cost_reserved, 6),
            "cached_source_cost_usd": round(self.cached_source_cost, 6),
            "max_cost_usd": self.settings.max_cost_usd,
            "budget_mode": "hard_reservation" if self.pricing else "soft_threshold",
            "pricing_usd_per_token": self.pricing,
            "budget_exceeded": self.budget_exceeded,
        }
