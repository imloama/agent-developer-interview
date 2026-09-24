#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面试题第 02 期 · 公共模块：极简 LLM 客户端（Mock + 真实 OpenAI 兼容接口）

这个模块存在的意义：让「Agent 循环」的演示**离线可跑、结果可复现**。
默认走 MockModel（脚本化模型），不联网、不花钱、每次输出一致；
需要看真实模型行为时，设置两个环境变量即可切到真实接口：

    export OPENAI_API_KEY=sk-xxx
    export OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4   # 智谱
    export OPENAI_MODEL=glm-4-plus
    python workflow_vs_agent.py --real

兼容任何 OpenAI 格式的接口（智谱 GLM / DeepSeek / Moonshot / OpenAI / 本地 vLLM）。

【重要说明】
MockModel 不是「假装成模型」，而是把「模型决策的不确定性」参量化：
它在理想决策路径上以 noise_rate 的概率走偏。这是为了做定量对比
（不加噪声就没法比较 Agent 与 Workflow 的方差），不是为了模拟真实模型。
真实模型的行为要复杂得多——这一点在文章里会明确说明。
"""

from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.request
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 工具定义（JSON Schema）——第 03 期会专门讲「一个 Tool 该包含哪些信息」
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "查询订单的当前状态与签收时间。任何退款类判断都必须先调用它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号，例如 A1001"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_refund_policy",
            "description": "根据订单的签收时间判断是否还在可退款期内。必须已有 get_order 的结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "signed_date": {"type": "string", "description": "签收日期，格式 YYYY-MM-DD"},
                },
                "required": ["order_id", "signed_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_refund",
            "description": "发起退款。这是**不可逆的写操作**，只有在策略判定可退款时才能调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number", "description": "退款金额（元）"},
                },
                "required": ["order_id", "amount"],
            },
        },
    },
]


@dataclass
class ToolCall:
    name: str
    arguments: dict
    call_id: str = "call-1"


@dataclass
class Reply:
    """模型一次响应的抽象：要么要求调用工具，要么给出最终答复。"""
    content: str | None = None
    tool_call: ToolCall | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def is_tool_call(self) -> bool:
        return self.tool_call is not None


# ---------------------------------------------------------------------------
# 业务侧的工具实现（真实系统里这些是 API 调用，这里用本地状态代替）
# ---------------------------------------------------------------------------

ORDER_DB = {
    "A1001": {"status": "已签收", "signed_date": "2026-09-20", "amount": 299.0},
    "A1002": {"status": "已签收", "signed_date": "2026-08-01", "amount": 599.0},
}

REFUND_WINDOW_DAYS = 7

# 「今天」固定住，保证演示可复现（真实系统里取当前时间）
TODAY = "2026-09-24"


def days_between(d1: str, d2: str) -> int:
    from datetime import date
    y1, m1, dd1 = (int(x) for x in d1.split("-"))
    y2, m2, dd2 = (int(x) for x in d2.split("-"))
    return (date(y2, m2, dd2) - date(y1, m1, dd1)).days


def execute_tool(call: ToolCall, side_effects: list[str] | None = None) -> dict:
    """工具执行层。注意：这一层是**代码**，不是模型——所有写操作都在这里发生。"""
    name, args = call.name, call.arguments
    oid = args.get("order_id", "")

    if name == "get_order":
        row = ORDER_DB.get(oid)
        if not row:
            return {"ok": False, "error_code": "ORDER_NOT_FOUND", "message": f"订单 {oid} 不存在"}
        return {"ok": True, "order_id": oid, **row}

    if name == "check_refund_policy":
        signed = args.get("signed_date", "")
        if not signed:
            return {"ok": False, "error_code": "MISSING_ARG",
                    "message": "缺少 signed_date，请先调用 get_order 取得签收时间",
                    "retryable": True}
        days = days_between(signed, TODAY)
        return {"ok": True, "order_id": oid, "days_since_signed": days,
                "window_days": REFUND_WINDOW_DAYS,
                "refundable": days <= REFUND_WINDOW_DAYS}

    if name == "create_refund":
        if side_effects is not None:
            side_effects.append(f"REFUND_CREATED:{oid}:{args.get('amount')}")
        return {"ok": True, "order_id": oid, "refund_id": f"R-{oid}", "status": "已受理"}

    return {"ok": False, "error_code": "UNKNOWN_TOOL", "message": f"未知工具 {name}"}


# ---------------------------------------------------------------------------
# MockModel：脚本化模型（离线可跑）
# ---------------------------------------------------------------------------

IDEAL_ORDER = ["get_order", "check_refund_policy", "create_refund"]


class MockModel:
    """
    把「模型的工具选择」参量化成：以 noise_rate 的概率偏离理想顺序。

    偏离的三种形态，正对应真实模型在长任务里的三类翻车：
      · 越序：还没拿到订单信息就问策略（参数不全）
      · 重复：把已经调过的工具再调一次（忘了工具结果）
      · 提前写：跳过策略校验直接下单（不可逆操作提前发生）
    """

    def __init__(self, seed: int = 0, noise_rate: float = 0.0, order_id: str = "A1001"):
        self.rng = random.Random(seed)
        self.noise_rate = noise_rate
        self.order_id = order_id

    # 模型「看得见」的只有对话历史里发生的事——这正是它决策的全部依据
    def respond(self, called: list[str], observations: list[dict]) -> Reply:
        known = {c: o for c, o in zip(called, observations)}
        prompt_tokens = 900 + 120 * len(called)         # 粗略模拟：每轮要重发历史

        # --- 噪声：走偏 ---
        if self.rng.random() < self.noise_rate:
            if "get_order" not in called:
                # 越序：直接问策略，但缺 signed_date
                return Reply(
                    tool_call=ToolCall("check_refund_policy", {"order_id": self.order_id}),
                    prompt_tokens=prompt_tokens, completion_tokens=40,
                )
            if "check_refund_policy" not in called:
                # 提前写：不可逆操作在策略校验之前发生
                return Reply(
                    tool_call=ToolCall("create_refund", {"order_id": self.order_id, "amount": 299.0}),
                    prompt_tokens=prompt_tokens, completion_tokens=45,
                )
            # 重复：把已经调过的工具再调一遍
            return Reply(
                tool_call=ToolCall("get_order", {"order_id": self.order_id}),
                prompt_tokens=prompt_tokens, completion_tokens=35,
            )

        # --- 理想路径 ---
        if "get_order" not in called:
            return Reply(tool_call=ToolCall("get_order", {"order_id": self.order_id}),
                         prompt_tokens=prompt_tokens, completion_tokens=35)

        if "check_refund_policy" not in called:
            signed = known.get("get_order", {}).get("signed_date", "")
            return Reply(
                tool_call=ToolCall("check_refund_policy",
                                   {"order_id": self.order_id, "signed_date": signed}),
                prompt_tokens=prompt_tokens, completion_tokens=45,
            )

        policy = known.get("check_refund_policy", {})
        if policy.get("refundable") and "create_refund" not in called:
            return Reply(
                tool_call=ToolCall("create_refund",
                                   {"order_id": self.order_id,
                                    "amount": known.get("get_order", {}).get("amount", 0.0)}),
                prompt_tokens=prompt_tokens, completion_tokens=50,
            )

        days = policy.get("days_since_signed")
        verdict = "可以退款，退款已受理" if policy.get("refundable") else "已超过 7 天无理由退款期，无法退款"
        return Reply(
            content=f"订单 {self.order_id} 签收已 {days} 天，{verdict}。",
            prompt_tokens=prompt_tokens, completion_tokens=60,
        )


# ---------------------------------------------------------------------------
# 真实模型（OpenAI 兼容接口，仅标准库）
# ---------------------------------------------------------------------------

class OpenAIModel:
    def __init__(self, model: str | None = None, temperature: float = 0.0):
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        if not self.key:
            raise RuntimeError("未设置 OPENAI_API_KEY，无法使用真实接口")

    def respond(self, messages: list[dict]) -> Reply:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": TOOLS,
            "temperature": self.temperature,
        }
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"接口返回 {e.code}：{e.read().decode('utf-8', 'ignore')[:300]}") from e

        msg = data["choices"][0]["message"]
        usage = data.get("usage", {})
        tcs = msg.get("tool_calls") or []
        if tcs:
            tc = tcs[0]
            return Reply(
                tool_call=ToolCall(
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"] or "{}"),
                    call_id=tc.get("id", "call-1"),
                ),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
        return Reply(content=msg.get("content", ""),
                     prompt_tokens=usage.get("prompt_tokens", 0),
                     completion_tokens=usage.get("completion_tokens", 0))


def build_model(real: bool, seed: int = 0, noise_rate: float = 0.0, order_id: str = "A1001"):
    if real:
        return OpenAIModel()
    return MockModel(seed=seed, noise_rate=noise_rate, order_id=order_id)


# ---------------------------------------------------------------------------
# 成本估算（示意口径）
# ---------------------------------------------------------------------------

# 举例说明用：真实价格请查各厂商定价页。这里只关心**相对倍数**。
PRICE_IN_PER_1K = 0.002
PRICE_OUT_PER_1K = 0.008


def cost_of(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens / 1000 * PRICE_IN_PER_1K + completion_tokens / 1000 * PRICE_OUT_PER_1K


if __name__ == "__main__":
    # 自检：跑一遍理想路径，确认工具层工作正常
    print("工具层自检：")
    c1 = ToolCall("get_order", {"order_id": "A1001"})
    print("  get_order        →", execute_tool(c1))
    c2 = ToolCall("check_refund_policy", {"order_id": "A1001", "signed_date": "2026-09-15"})
    print("  check_refund_policy →", execute_tool(c2))
    c3 = ToolCall("create_refund", {"order_id": "A1001", "amount": 299.0})
    print("  create_refund    →", execute_tool(c3))

    print("\nMockModel 理想路径（noise_rate=0）：")
    m = MockModel(seed=0, noise_rate=0.0)
    called, obs = [], []
    for _ in range(6):
        r = m.respond(called, obs)
        if r.is_tool_call:
            called.append(r.tool_call.name)
            obs.append(execute_tool(r.tool_call))
            print(f"  → 工具调用 {r.tool_call.name}")
        else:
            print(f"  → 最终答复：{r.content}")
            break

    print("\nMockModel 带噪声（noise_rate=1.0，必然走偏）：")
    m = MockModel(seed=0, noise_rate=1.0)
    r = m.respond([], [])
    print(f"  → 第一个动作：{r.tool_call.name} {r.tool_call.arguments}")
