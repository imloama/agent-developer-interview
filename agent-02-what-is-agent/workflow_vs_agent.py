#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面试题第 02 期 · 实验 1：同一个任务，Workflow 与 Agent 各写一遍

零依赖（可选联网），直接运行：

    python workflow_vs_agent.py            # 用脚本化 MockModel，离线可复现

可选：接真实模型看行为差异
    export OPENAI_API_KEY=sk-xxx
    export OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
    export OPENAI_MODEL=glm-4-plus
    python workflow_vs_agent.py --real

对应题目
  Q1：什么是 AI Agent？它和 Chatbot、Workflow 有什么本质区别？
  Q2：什么场景该用 Agent、什么场景该用 Workflow？为什么很多 Agent 产品最后都变成了 Workflow？

任务（两条实现完全一样）
  用户：「订单 A1001 现在还能退款吗？能的话帮我退掉。」
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from llm import (  # noqa: E402
    IDEAL_ORDER,
    MockModel,
    ToolCall,
    build_model,
    cost_of,
    execute_tool,
)

USER_TEXT = "订单 A1001 现在还能退款吗？能的话帮我退掉。"


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 实现 A：Workflow —— 控制流在代码手里
# ---------------------------------------------------------------------------

def run_workflow(order_id: str = "A1001") -> dict:
    """
    固定流水线。**用不用模型不影响流程**：模型只负责「抽取参数」和「润色文案」，
    它没有任何机会决定「下一步做什么」。
    """
    trace, llm_calls, tin, tout = [], 0, 0, 0
    side_effects: list[str] = []

    # 步骤 1：模型抽取参数（受约束输出，只做抽取，不做决策）
    llm_calls += 1
    tin += 320
    tout += 18
    trace.append(("LLM", "抽取 order_id → 'A1001'（schema 约束，失败则走兜底追问）"))

    # 步骤 2：以下全部由代码决定顺序
    r1 = execute_tool(ToolCall("get_order", {"order_id": order_id}), side_effects)
    trace.append(("TOOL", f"get_order → {r1}"))
    if not r1.get("ok"):
        return _summary("Workflow", trace, llm_calls, tin, tout, side_effects,
                        final="订单不存在，请确认订单号。", ok=False, reason="ORDER_NOT_FOUND")

    r2 = execute_tool(ToolCall("check_refund_policy",
                               {"order_id": order_id, "signed_date": r1["signed_date"]}),
                      side_effects)
    trace.append(("TOOL", f"check_refund_policy → {r2}"))

    # 步骤 3：分支判断 —— 代码里的 if，不是模型的选择
    if not r2["refundable"]:
        trace.append(("CODE", "if not refundable → 直接返回不可退款，不调用 create_refund"))
        final = (f"订单 {order_id} 签收已 {r2['days_since_signed']} 天，"
                 f"超过 {r2['window_days']} 天无理由退款期，无法退款。")
        return _summary("Workflow", trace, llm_calls, tin, tout, side_effects,
                        final=final, ok=True, reason="NOT_REFUNDABLE")

    r3 = execute_tool(ToolCall("create_refund", {"order_id": order_id, "amount": r1["amount"]}),
                      side_effects)
    trace.append(("TOOL", f"create_refund → {r3}"))

    # 步骤 4：模型润色文案（同样只是生成，不决策）
    llm_calls += 1
    tin += 480
    tout += 62
    final = (f"订单 {order_id} 签收已 {r2['days_since_signed']} 天，仍在退款期内，"
             f"退款单 {r3['refund_id']} 已受理，预计 1-3 个工作日到账。")
    trace.append(("LLM", "生成面向用户的回复文案"))

    return _summary("Workflow", trace, llm_calls, tin, tout, side_effects,
                    final=final, ok=True, reason="SUCCESS")


# ---------------------------------------------------------------------------
# 实现 B：Agent —— 控制流在模型手里
# ---------------------------------------------------------------------------

def run_agent(order_id: str = "A1001", model=None, max_steps: int = 8) -> dict:
    """
    循环：模型选工具 → 应用执行 → 结果回填 → 再问模型 → 直到模型给出最终答复。

    注意三件事：
      · 下一步做什么**完全由模型决定**，代码里没有一个 if 决定流程
      · 工具执行、参数校验、写操作全部仍在代码侧（模型只是提出意图）
      · 必须有 max_steps 兜底，否则模型可能一直转圈
    """
    model = model or MockModel(seed=0, noise_rate=0.0, order_id=order_id)
    trace, llm_calls, tin, tout = [], 0, 0, 0
    side_effects: list[str] = []
    called: list[str] = []
    observations: list[dict] = []
    messages = [{"role": "system", "content": "你是订单助手，可使用工具完成退款。"},
                {"role": "user", "content": USER_TEXT}]

    for step in range(1, max_steps + 1):
        reply = model.respond(called, observations)
        llm_calls += 1
        tin += reply.prompt_tokens
        tout += reply.completion_tokens

        if not reply.is_tool_call:
            trace.append(("LLM", f"第 {step} 步：给出最终答复（循环结束条件 = 模型主动收手）"))
            return _summary("Agent", trace, llm_calls, tin, tout, side_effects,
                            final=reply.content or "", ok=True, reason="MODEL_FINISHED")

        tc: ToolCall = reply.tool_call
        trace.append(("LLM", f"第 {step} 步：选择工具 {tc.name}({tc.arguments})"))
        result = execute_tool(tc, side_effects)
        trace.append(("TOOL", f"{tc.name} → {result}"))

        called.append(tc.name)
        observations.append(result)
        # 结果必须回填进消息历史——不然下一轮模型「不记得」已经调过
        messages.append({"role": "assistant", "content": None})
        messages.append({"role": "tool", "content": str(result)})

    trace.append(("RUNTIME", f"达到 max_steps={max_steps}，强制中断"))
    return _summary("Agent", trace, llm_calls, tin, tout, side_effects,
                    final="（未在预算内完成）", ok=False, reason="MAX_STEPS")


def _summary(kind, trace, llm_calls, tin, tout, side_effects, *, final, ok, reason) -> dict:
    return {"kind": kind, "trace": trace, "llm_calls": llm_calls,
            "prompt_tokens": tin, "completion_tokens": tout,
            "cost": cost_of(tin, tout), "side_effects": side_effects,
            "final": final, "ok": ok, "reason": reason}


# ---------------------------------------------------------------------------

def show(res: dict, title: str) -> None:
    hr(title)
    for actor, line in res["trace"]:
        mark = {"LLM": "🧠", "TOOL": "🔧", "CODE": "⚙️ ", "RUNTIME": "🛡️ "}.get(actor, "  ")
        print(f"  {mark} [{actor:<7}] {line}")
    print(f"\n  最终答复：{res['final']}")
    print(f"  副作用（不可逆动作）：{res['side_effects'] or '无'}")


def show_compare(wf: dict, ag: dict) -> None:
    hr("对比：同一个任务、同一个模型、同一批工具")

    print(f"  {'维度':<24} {'Workflow':>14} {'Agent':>14} {'差异':>12}")
    print("  " + "-" * 70)

    rows = [
        ("控制流归谁", "代码", "模型", "← 本质区别"),
        ("模型调用次数", wf["llm_calls"], ag["llm_calls"], ""),
        ("输入 token", wf["prompt_tokens"], ag["prompt_tokens"], ""),
        ("输出 token", wf["completion_tokens"], ag["completion_tokens"], ""),
        ("相对成本", f"{wf['cost']:.5f}", f"{ag['cost']:.5f}", ""),
        ("工具调用次数", sum(1 for a, _ in wf["trace"] if a == "TOOL"),
         sum(1 for a, _ in ag["trace"] if a == "TOOL"), ""),
    ]
    for name, a, b, note in rows:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and isinstance(a, int):
            ratio = f"{b / a:.2f}×" if a else "-"
            print(f"  {name:<24} {a:>14} {b:>14} {ratio:>12}")
        else:
            print(f"  {name:<24} {str(a):>14} {str(b):>14} {note:>12}")

    cost_ratio = ag["cost"] / wf["cost"] if wf["cost"] else float("nan")
    print(f"\n  Agent 成本是 Workflow 的 {cost_ratio:.2f} 倍。")

    print("""
【差异从哪来：不是「Agent 更贵」这种形容词，而是可推导的结构】
  1. 每一轮决策都要重发一次完整上下文（system + 工具定义 + 历史）
     → 输入 token 随轮数**近似平方**增长，而不是线性
     本例：第 k 轮的 prompt_tokens = 900 + 120×k（见 llm.py 的 MockModel）
  2. 每一轮决策就是一次模型调用 → 调用次数 = 决策步数
     Workflow 的模型调用数是**常数**（抽参数 1 次 + 润色 1 次，与分支无关），
     Agent 的调用数 = 它自己走了多少步，且不可预先知晓
  3. 输出 token 也更贵（输出单价通常是输入的 3~4 倍）

  所以「Agent 贵」，贵在**循环次数 × 每轮重发的上下文**，不是贵在单次调用。
  这也直接给出了省钱的两个方向：减少轮数、让前缀可缓存（第 01 期 kv_cache.py）。
""")


def show_function_calling_is_not_agent() -> None:
    hr("反例：用了 Function Calling 就算 Agent 吗？——不算")

    print("""  看这样一条链路（每步都调模型、都用 Function Calling）：

      用户输入 → [LLM 抽取订单号] → get_order → [LLM 生成摘要]
              → [LLM 翻译成英文] → send_email → 结束

  · 用了 Function Calling      ✅
  · 用了 LLM                   ✅
  · 是 Agent 吗？              ❌

  因为**下一步做什么是写死的**：翻译后必然是发邮件。模型的输出只被当作数据，
  从未参与「接下来干什么」的决策。控制流在代码手里 → 它是 Workflow。

  判断 Agent 的唯一硬标准不是「用没用工具」，而是：
      下一次调用哪个工具、什么时候停，是不是由模型决定的？

  换句话说：**Chatbot 决定「说什么」，Workflow 决定「按什么顺序做」，
  Agent 决定「做什么、做几步、什么时候停」。**""")


def show_degeneration(wf: dict, ag: dict) -> None:
    hr("为什么 Agent 产品最后都变成了 Workflow（本题第二问的胜负手）")

    print("""  先把攻击性最强的那句话摆在前面，因为它就是答案：

    「产品化要求确定性」。
    而 Agent 的不确定只能从外面收敛：
      收紧工具集 → 约束调用顺序 → 固定分支条件 → 加 guardrails → 加人工确认
    每收紧一次，模型能自主决定的空间就小一点。收敛到最后，剩下的就是一条
    分支明确、可测试、可回滚的流程——也就是 Workflow。

  ▶ 这不是退步，是工程收敛。而且它有正收益：

    ① 可测试：同样的输入必然走同样的路径，回归测试才有断言可写
    ② 可计费：成本可预测（token 预算 = 轮数上限 × 单轮 token）
    ③ 可审计：每一步都能解释「为什么这么做」，合规与事故复盘才做得下去
    ④ 可回滚：没有不可预测的中间状态
    ⑤ 延迟可控：不再有「这次跑了 13 步」的长尾

  ▶ 所以工业界的终局形态几乎都是**混合体**：

      Workflow 骨架（可枚举、需确定性的部分）
        + Agent 节点（步骤无法预先枚举、需要根据中间结果决策的那一步）

    在本例里就是：主干固定为「查订单 → 判策略 → 若可退则退款 → 回复」，
    只有「该不该退款」这一步留给模型判断（因为规则复杂、边界模糊、会变）。

  ▶ 反过来说，什么情况下**必须**用 Agent（这才是区分候选人水平的地方）：
    · 步骤无法预先枚举（研究、排障、探索式数据分析）
    · 必须根据中间结果才能决定下一步（每步的结果改变后续路径）
    · 工具集动态（不同用户/不同权限看到不同工具，路径无法统一写死）
    · 错误的代价低且可回滚（试错成本 < 写死所有分支的成本）

  ▶ 什么情况下**不该**用 Agent（反向判断题，最见功力）：
    · 步骤可枚举 → 直接写 Workflow，确定性还更便宜
    · 错误代价不可逆且高（资金、删库、发对外公告）→ 必须人工确认或纯代码
    · 延迟预算紧（交互式场景等待 30 秒不可接受）
    · 没有可验证的成功判据 → 无法评估、无法回归、无法上线
    · 输入分布稳定但长尾复杂 → 先试规则引擎，再试微调，最后才考虑 Agent

  面试里能把上面这两张清单讲清楚，比背十遍「Agent = LLM + 工具 + 循环」有效得多。""")


def main() -> None:
    real = "--real" in sys.argv
    print("面试题第 02 期 · 实验 1：同一个任务，Workflow 与 Agent 各写一遍")
    print(f"模型来源：{'真实接口（OpenAI 兼容）' if real else '脚本化 MockModel（离线）'}")

    wf = run_workflow("A1001")
    show(wf, "实现 A｜Workflow：控制流在代码手里")

    if real:
        ag = run_agent("A1001", model=build_model(True))
    else:
        ag = run_agent("A1001", model=MockModel(seed=0, noise_rate=0.0))
    show(ag, "实现 B｜Agent：控制流在模型手里")

    show_compare(wf, ag)
    show_function_calling_is_not_agent()
    show_degeneration(wf, ag)

    hr("复盘")
    print("""  · 四个判据：控制流归谁 / 是否循环直到达成目标 / 能否作用于外部世界 / 是否以结果为优化目标
  · 关键分界线：控制流在代码手里是 Workflow，在模型手里才是 Agent
  · 「用了工具」不构成 Agent——固定链即使每步都调模型也仍是 Workflow
  · 组件清单：Model / Prompt / Memory / Tools / Planning / Execution / Observation / Guardrails
  · Agent → Workflow 的收敛不是退步，是产品化要求确定性
  · 下一节 convergence.py 会用 300 次运行把「收敛」这件事量化出来""")


if __name__ == "__main__":
    main()
