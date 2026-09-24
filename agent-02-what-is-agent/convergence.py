#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面试题第 02 期 · 实验 2：把「Agent 最后变成 Workflow」量化出来

零依赖，直接运行：

    python convergence.py

对应题目
  Q2：为什么很多 Agent 产品最后都变成了 Workflow？

做法
  同一个任务跑三种实现，各 300 次，统计**正确率、不可逆操作的越序次数、
  重复写次数、平均模型调用数、平均成本、路径种类数**：

    A. Workflow（纯代码控制流）
    B. Agent（模型决定控制流，无约束）
    C. Agent + Guardrails（模型仍决定，但运行时强制约束调用顺序）

  期望看到的结论：C 在正确率上追平 A，在路径种类数上也逼近 A——
  这就是「收敛成 Workflow」的定量形态。收敛不是退步，是产品化的必要条件。

  噪声口径说明：MockModel 以 noise_rate 的概率偏离理想顺序，模拟真实模型
  「越序 / 重复 / 提前写」三类失误。这不是在模拟真实模型的行为分布，
  而是把「模型的不确定性」参量化，以便做定量对比。
"""

from __future__ import annotations

import sys
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from llm import MockModel, ToolCall, cost_of, execute_tool  # noqa: E402
from workflow_vs_agent import run_workflow  # noqa: E402

NOISE_RATE = 0.25      # 25% 的决策步会走偏
TRIALS = 300
MAX_STEPS = 8


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 运行时：Agent 循环（可选 guardrails）
# ---------------------------------------------------------------------------

def expected_next(called: list[str], obs: list[dict]) -> str:
    """把「正确的下一步」写成代码 —— 这就是 Guardrails 的核心。"""
    known = dict(zip(called, obs))
    if "get_order" not in called:
        return "get_order"
    if "check_refund_policy" not in called:
        return "check_refund_policy"
    if known.get("check_refund_policy", {}).get("refundable") and "create_refund" not in called:
        return "create_refund"
    return "__finish__"


def args_for(step: str, order_id: str, obs: list[dict]) -> dict:
    """运行时把「该做哪一步」编译成具体参数 —— 参数不再由模型提供，也就不会错。"""
    order_info = next((o for o in obs if isinstance(o, dict) and "signed_date" in o), {})
    if step == "get_order":
        return {"order_id": order_id}
    if step == "check_refund_policy":
        return {"order_id": order_id, "signed_date": order_info.get("signed_date", "")}
    if step == "create_refund":
        return {"order_id": order_id, "amount": order_info.get("amount", 0.0)}
    return {}


def run_agent_once(*, seed: int, noise_rate: float, order_id: str,
                   guardrails: bool) -> dict:
    model = MockModel(seed=seed, noise_rate=noise_rate, order_id=order_id)
    called: list[str] = []
    obs: list[dict] = []
    side_effects: list[str] = []
    llm_calls, cost, interventions = 0, 0.0, 0
    out_of_order_writes, refund_writes = 0, 0
    finished = False
    path: list[str] = []
    final = ""

    for _ in range(MAX_STEPS):
        reply = model.respond(called, obs)
        llm_calls += 1
        cost += cost_of(reply.prompt_tokens, reply.completion_tokens)

        exp = expected_next(called, obs)

        if guardrails:
            # 运行时接管：模型想干的和该干的不一致时，按「该干的」执行并记一次干预
            if exp == "__finish__":
                finished = True
                path.append("FINISH")
                break
            picked = reply.tool_call.name if reply.is_tool_call else "<finish>"
            if picked != exp:
                interventions += 1
            name, args = exp, args_for(exp, order_id, obs)
        else:
            if not reply.is_tool_call:
                finished = True
                path.append("FINISH")
                final = reply.content or ""
                break
            name, args = reply.tool_call.name, reply.tool_call.arguments

        # 记账：不可逆操作是否发生在策略校验之前
        if name == "create_refund":
            refund_writes += 1
            if "check_refund_policy" not in called:
                out_of_order_writes += 1

        result = execute_tool(ToolCall(name, args), side_effects)
        called.append(name)
        obs.append(result)
        path.append(name)

    # 终态判定
    if order_id == "A1001":          # 可退款：应当恰好写一次
        correct = finished and refund_writes == 1 and out_of_order_writes == 0
    else:                            # A1002 不可退款：应当一次都不写
        correct = finished and refund_writes == 0

    return {
        "correct": correct,
        "out_of_order_writes": out_of_order_writes,
        "refund_writes": refund_writes,
        "llm_calls": llm_calls,
        "cost": cost,
        "interventions": interventions,
        "path": "→".join(path) if path else "(空)",
        "final": final,
    }


def run_workflow_once(order_id: str) -> dict:
    r = run_workflow(order_id)
    return {
        "correct": r["ok"],
        "out_of_order_writes": 0,
        "refund_writes": sum(1 for s in r["side_effects"] if s.startswith("REFUND_CREATED")),
        "llm_calls": r["llm_calls"],
        "cost": r["cost"],
        "interventions": 0,
        "path": "→".join(a for a, _ in r["trace"]),
    }


# ---------------------------------------------------------------------------

def benchmark(mode: str, order_id: str) -> dict:
    correct = ooo = writes = calls = 0
    cost = 0.0
    paths: Counter[str] = Counter()
    interventions = 0

    for seed in range(TRIALS):
        if mode == "workflow":
            r = run_workflow_once(order_id)
        else:
            r = run_agent_once(seed=seed, noise_rate=NOISE_RATE,
                               order_id=order_id, guardrails=(mode == "guarded"))
        correct += int(r["correct"])
        ooo += r["out_of_order_writes"]
        writes += r["refund_writes"]
        calls += r["llm_calls"]
        cost += r["cost"]
        interventions += r["interventions"]
        paths[r["path"]] += 1

    return {
        "correct_rate": correct / TRIALS,
        "out_of_order": ooo,
        "writes": writes,
        "avg_calls": calls / TRIALS,
        "avg_cost": cost / TRIALS,
        "path_kinds": len(paths),
        "top_paths": paths.most_common(3),
        "interventions": interventions,
    }


def print_table(order_id: str, label: str, results: dict[str, dict]) -> None:
    hr(f"订单 {order_id}（{label}）· {TRIALS} 次运行，模型决策噪声 {NOISE_RATE:.0%}")

    print(f"  {'方案':<26} {'终态正确率':>10} {'越序写':>8} {'重复写':>8} "
          f"{'平均调用':>9} {'平均成本':>10} {'路径种类':>9}")
    print("  " + "-" * 84)

    names = {
        "workflow": "A. Workflow（纯代码）",
        "agent": "B. Agent（模型决定）",
        "guarded": "C. Agent + Guardrails",
    }
    for key in ("workflow", "agent", "guarded"):
        r = results[key]
        dup = max(0, r["writes"] - TRIALS)
        print(f"  {names[key]:<26} {r['correct_rate']:>10.1%} {r['out_of_order']:>8} "
              f"{dup:>8} {r['avg_calls']:>9.2f} {r['avg_cost']:>10.5f} {r['path_kinds']:>9}")

    print("\n  各方案出现最多的执行路径：")
    for key in ("workflow", "agent", "guarded"):
        r = results[key]
        print(f"    {names[key]}")
        for p, c in r["top_paths"]:
            print(f"      {c / TRIALS:>6.1%}  {p}")

    print(f"\n  C 方案共发生 {results['guarded']['interventions']} 次运行时干预"
          f"（平均 {results['guarded']['interventions'] / TRIALS:.2f} 次/任务）。")
    print("  这 {:.2f} 次干预就是「从 Agent 收敛到 Workflow」的具体动作——".format(
        results["guarded"]["interventions"] / TRIALS))
    print("  每一次都是把模型的一个自主决定替换成代码的确定性判断。")


def main() -> None:
    print("面试题第 02 期 · 实验 2：把「Agent 最后变成 Workflow」量化出来")

    for order_id, label in (("A1001", "可退款订单"), ("A1002", "已过退款期订单")):
        results = {m: benchmark(m, order_id) for m in ("workflow", "agent", "guarded")}
        print_table(order_id, label, results)

    hr("三个结论")
    print("""  ① Agent 的正确率来自约束，不来自模型
     B 方案在 25% 决策噪声下的终态正确率明显低于 A；C 方案把同一批噪声
     全部挡在运行时之外，正确率回到 100%。注意 C 里模型仍然在「做决策」，
     只是它的决定可以被运行时否决——**可靠性由代码提供，不由模型提供**。

  ② 收敛的代价与收益可以同时算出来
     收益：越序写 → 0、重复写 → 0、终态正确率 → 100%
     代价：C 的平均调用数略高于 A（因为要容忍并纠正模型的走偏），
           但显著低于 B（B 会绕圈、会重复调用）
     所以 C 不是「又贵又慢」的妥协，而是**比纯 Agent 更便宜、比纯 Workflow 更灵活**的形态。

  ③ 收敛的终点就是产品形态
     把 C 再收紧一步（干预时不再问模型、直接走代码分支），就得到 A。
     这就是为什么「Agent 产品最后都变成了 Workflow」——不是团队能力不行，
     而是**每一条被验证过的失败路径，最终都会变成一条代码分支**。

【对照真实世界的一句话总结】
  Agent 的可观测价值不在「它自己会做决定」，而在「它能在你没写死的路径上
  给你一个可用的初稿」。一旦这条路径被验证清楚了，就该把它固化成流程——
  把不确定性留给真正不确定的那一步。""")

    hr("复盘：这两道题一起答的样子")
    print("""  Q1（什么是 Agent）：
    四个判据 → 控制流归谁 → 组件八项（含 Guardrails）→ 标准链路
  Q2（该用 Agent 还是 Workflow）：
    可枚举性 + 可验证性 + 错误代价 + 延迟预算 → 四张牌决定形态
    → 产品化必然收敛为「Workflow 骨架 + Agent 节点」
    → 能讲清「什么情况下必须用 Agent」和「什么情况下不该用 Agent」才算答完""")


if __name__ == "__main__":
    main()
