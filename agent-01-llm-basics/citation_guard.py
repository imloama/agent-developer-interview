#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面试题第 01 期 · 实验 5：幻觉治理——为什么「提示词里写不要编造」不管用

零依赖，直接运行：

    python citation_guard.py

对应题目
  Q2：幻觉的产生原因有哪些？为什么不能完全相信 LLM 的输出？

实验主线
  用三段代码把「降低幻觉」这件事从口号变成可运行的校验器：
    [A] 一个会编造引用和数字的生成器（模拟真实模型）
    [B] 四个校验器：引用存在性 / 引用覆盖率 / 数字支撑 / 无资料必拒答
    [C] 校验失败 → 结构化错误回传 → 重生成 → 二次校验通过

关键结论
  提示词层防御（「不要编造」「必须准确」）是**概率性**的；
  校验层防御（代码检查）是**确定性**的。
  所以幻觉治理的正确位置是 Runtime，不是 Prompt。这就是第 10 期的 Harness 主题。
"""

from __future__ import annotations

import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 知识库与检索结果（真实系统里这一层是 RAG 的召回结果）
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = {
    "kb-101": "订单 A1001 的退款规则：签收后 7 天内可无理由退款。退款到账时间为 1-3 个工作日。",
    "kb-102": "订单 A1001 当前状态：已签收，签收时间 2026-09-15。",
    "kb-103": "会员等级规则：黄金会员享 3 倍积分，铂金会员享 5 倍积分。",
}

QUESTION = "订单 A1001 现在还能退款吗？多久到账？"


# ---------------------------------------------------------------------------
# [A] 模拟生成器：一个会编造的模型
# ---------------------------------------------------------------------------

# 提示词里明确写了「不要编造」——但模型依然会编，这正是要演示的点
SYSTEM_PROMPT = "你是客服助手。必须基于给定资料回答。不要编造任何资料中没有的信息。"

MOCK_ROUNDS = {
    "call-1": (
        "订单 A1001 已签收，签收时间为 2026-09-15 [kb-102]，"
        "目前仍在 7 天无理由退款期内 [kb-999]，可以申请退款。"
        "退款到账时间一般为 5 个工作日 [kb-101]。"
    ),
    "call-2": (
        "订单 A1001 已签收，签收时间为 2026-09-15 [kb-102]。"
        "该订单在签收后 7 天内可无理由退款 [kb-101]，"
        "因此目前仍在退款期内。退款到账时间为 1-3 个工作日 [kb-101]。"
    ),
}


def generate(call_id: str, docs: dict[str, str], repair_hint: str | None = None) -> str:
    """
    真实实现里这里是 LLM 调用。这里用脚本化输出代替，保证离线可运行、结果稳定。

    注意 round 2 的行为：只把 round 1 失败的两个点改掉（虚构引用 kb-999 去掉、
    数字 5 改回资料里的 1-3），其余保持不变——这就是真实模型被「结构化错误」
    纠正后的典型行为：局部修补，不会整体重写。
    """
    if call_id == "call-1":
        return MOCK_ROUNDS["call-1"]
    return MOCK_ROUNDS["call-2"]


# ---------------------------------------------------------------------------
# [B] 四个校验器
# ---------------------------------------------------------------------------

CITE_RE = re.compile(r"\[(kb-\d+)\]")
SENT_SPLIT_RE = re.compile(r"(?<=[。；！？])")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]


def check_citations_exist(answer: str, docs: dict[str, str]) -> list[str]:
    """校验 1：引用必须指向真实存在的资料来源。"""
    errs = []
    for cid in set(CITE_RE.findall(answer)):
        if cid not in docs:
            errs.append(f"引用了不存在的来源 [{cid}]（检索结果里没有这个 id）")
    return errs


def check_coverage(answer: str, docs: dict[str, str]) -> list[str]:
    """校验 2：每一句事实性陈述都必须带至少一个引用。没有引用 = 无据推断。"""
    errs = []
    for sent in split_sentences(answer):
        if not CITE_RE.search(sent):
            errs.append(f"句子缺少来源标注：「{sent}」")
    return errs


def check_numbers_supported(answer: str, docs: dict[str, str]) -> list[str]:
    """校验 3：答案里出现的数字，必须在它所引用的资料里出现过。"""
    errs = []
    for sent in split_sentences(answer):
        cited = CITE_RE.findall(sent)
        if not cited:
            continue
        pool = " ".join(docs.get(c, "") for c in cited)
        # 关键细节：先把引用标记摘掉，否则 "kb-102" 里的 102 会被当成数字声明
        claim = CITE_RE.sub("", sent)
        for num in NUM_RE.findall(claim):
            if num not in pool:
                errs.append(f"数字 {num} 不在所引用的资料中：「{sent}」")
    return errs


def check_refusal_when_empty(answer: str, docs: dict[str, str]) -> list[str]:
    """校验 4：没有检索到任何资料时，唯一合格回答是明确拒答。"""
    if docs:
        return []
    if "资料不足" in answer or "无法回答" in answer or "没有找到" in answer:
        return []
    return ["无检索结果时未拒答（应当明确说明资料不足，而不是给一个猜测）"]


def validate(answer: str, docs: dict[str, str]) -> list[str]:
    errors: list[str] = []
    errors += check_citations_exist(answer, docs)
    errors += check_coverage(answer, docs)
    errors += check_numbers_supported(answer, docs)
    errors += check_refusal_when_empty(answer, docs)
    return errors


# ---------------------------------------------------------------------------
# [C] 主流程：生成 → 校验 → 修复 → 再校验
# ---------------------------------------------------------------------------

def build_repair_hint(errors: list[str], docs: dict[str, str]) -> str:
    """把结构化错误拼成修复提示——注意这里回传的是「哪条规则、哪个片段」，"""
    """不是「你错了，再来一次」。后者模型无法定位问题，只会随机换一种错法。"""
    lines = ["上一版回答未通过校验，请只修正下列问题，其余内容保持不变："]
    lines += [f"  {i}. {e}" for i, e in enumerate(errors, 1)]
    lines.append("可用的来源 id：" + ", ".join(sorted(docs.keys())) + "。")
    lines.append("资料里没有的数字不要写；无法据资料回答时请明确说明资料不足。")
    return "\n".join(lines)


def run_pipeline(docs: dict[str, str], trace: bool = True) -> dict:
    """
    走一遍完整流水线，返回结果摘要。trace=True 时打印全过程（用于录像讲解）。
    """
    ids = ", ".join(sorted(docs.keys())) or "（空）"

    if trace:
        brief = {k: v[:28] + "…" for k, v in docs.items()}
        print(f"\n  问题：{QUESTION}")
        print(f"  检索到的资料 id：{ids}")
        print(f"  资料摘要：{brief}")
        print(f"  System Prompt：{SYSTEM_PROMPT}")
        print("  （注意：上面明确写了『不要编造』——看它管不管用）\n")

    answer = generate("call-1", docs)
    if trace:
        print("  ── 第 1 次生成" + "─" * 56)
        print(f"  {answer}\n")

    errors = validate(answer, docs)
    if trace:
        print(f"  ── 校验：发现 {len(errors)} 个问题" + "─" * 48)
        for e in errors:
            print(f"     ✗ {e}")

    if not errors:
        if trace:
            print("     ✓ 无需修复，直接通过")
        return {"stage1": 0, "stage2": 0, "passed": True, "final": answer}

    hint = build_repair_hint(errors, docs)
    if trace:
        print("\n  ── 回传给模型的结构化修复指令" + "─" * 42)
        print("  " + hint.replace("\n", "\n  "))

    answer2 = generate("call-2", docs)
    if trace:
        print("\n  ── 第 2 次生成" + "─" * 56)
        print(f"  {answer2}\n")

    errors2 = validate(answer2, docs)
    if trace:
        print(f"  ── 二次校验：{len(errors2)} 个问题" + "─" * 48)
        for e in errors2:
            print(f"     ✗ {e}")
        if errors2:
            print("\n  ❌ 仍未通过 → 转人工 / 降级为「无法确认」")
        else:
            print("     ✓ 全部通过，可返回给用户")

    return {
        "stage1": len(errors),
        "stage2": len(errors2),
        "passed": not errors2,
        "final": answer2,
    }


def exp_batch() -> None:
    hr("实验 B｜三种召回情况下的拦截结果")

    cases = [
        ("召回齐全（规则 + 状态）", {k: KNOWLEDGE_BASE[k] for k in ("kb-101", "kb-102")}),
        ("召回不全（只有状态，缺规则）", {"kb-102": KNOWLEDGE_BASE["kb-102"]}),
        ("完全没有召回", {}),
    ]

    print(f"  {'场景':<26} {'第1次校验':>10} {'第2次校验':>10} {'最终结果':>12}")
    print("  " + "-" * 64)

    results = []
    for name, docs in cases:
        r = run_pipeline(docs, trace=False)
        results.append((name, r))
        verdict = "通过 ✓" if r["passed"] else "转人工 ✗"
        print(f"  {name:<26} {r['stage1']:>8} 个 {r['stage2']:>8} 个 {verdict:>12}")

    print("""
【怎么读这张表】
  · 「召回齐全」：第 1 次生成有 3 个问题（一个虚构来源 + 两个无据数字），
    修复后干净通过。这说明校验 + 修复通道是有效的。
  · 「召回不全」：第 2 次生成仍然失败——模型只会把它拿到的资料用起来，
    资料本身缺规则，它就不可能给出合规答案。**检索质量决定答案上限**，
    校验器只能保证不低于下限。这条衔接第 08 期。
  · 「完全没有召回」：模型没有走拒答通道，直接给了个猜测 → 被校验器 4 拦下。
    这就是幻觉最危险的形态：**没有证据时编造证据性回答**。

  所以完整链路是四道闸门，缺一道都会漏：
    检索质量（决定上限）→ 强制引用（可追溯）→ 校验器（保下限）→ 人工兜底（兜底）""")


def exp_compare_defenses() -> None:
    hr("实验 C｜两种防御的位置差别：提示词 vs 校验器")

    docs = {k: KNOWLEDGE_BASE[k] for k in ("kb-101", "kb-102")}
    answer1 = generate("call-1", docs)

    print("  第 1 次生成的回答：")
    print(f"  {answer1}\n")
    print("  它编造了两个东西：")
    print("    · 引用 [kb-999] —— 检索结果里根本没有这个来源")
    print("    · 数字 5 —— 资料里说的是 1-3 个工作日")
    print("\n  而 System Prompt 里明确写了『不要编造任何资料中没有的信息』。")
    print("  提示词没有失效——它只是**没有强制力**：模型不能校验自己的输出，")
    print("  它只能「希望」自己没编。")

    print("""
【为什么会这样：数据与指令走同一条通道】
  对模型来说，System Prompt、用户输入、检索到的资料、工具返回的结果，
  最终都是同一个 token 序列里的一段文本。模型没有一个独立的「事实寄存器」
  去做比对。所以：
    · 「不要编造」是**建议**，影响的是输出分布，不是硬约束
    · 「代码校验」是**检查**，在模型之外，具有强制力

  这不是提示词工程的失败，而是位置放错了：**幻觉治理属于 Runtime，不属于 Prompt**。

【降低幻觉的六种手段，各自治哪种成因】——对应 Q2 的成因分类
  ① 训练目标：模型学的是「下一个 token 的条件概率」，不是事实正确性
     → 无法治，只能接受；这是「不能根治」的根本原因
  ② 知识截止 / 私域缺失
     → RAG 注入事实（实验里的 docs 就是这一步）
  ③ 检索缺失时强行作答（模型倾向给个答案而不是说不知道）
     → 强制拒答通道 + 校验器 4（无资料必拒答）
  ④ 采样随机性
     → 低温采样；但注意 temperature=0 也不保证一致（见 sampling.py）
  ⑤ 上下文与权重冲突时优先信任上下文
     → 脏资料会直接产出错答案 → 召回质量必须单独评估（第 08 期）
  ⑥ 编造来源、数字漂移
     → 强制引用 + 引用存在性校验 + 数字回查（校验器 1/2/3）

  六条里只有 ① 是结构性的。剩下五条里，**四条要靠代码而不是靠提示词**。

【一句话总结这道题的高分答法】
  「幻觉不是能力不足，是训练目标与业务目标错配——模型优化的是概率，不是事实。
    所以它不可能被根治，只能被压低。压低的手段里，提示词影响概率，
    校验器提供下限。生产系统要的是确定性的下限，所以幻觉治理最终落在 Runtime
    和评估体系上：强制引用、来源校验、拒答通道、分层评测。第 10 期会把这些
    拼成一个完整的 Harness。」""")


# ---------------------------------------------------------------------------

def main() -> None:
    print("面试题第 01 期 · 实验 5：幻觉治理——为什么提示词里写「不要编造」不管用")
    hr("实验 A｜一次完整的 生成 → 校验 → 修复 → 再校验")
    docs = {k: KNOWLEDGE_BASE[k] for k in ("kb-101", "kb-102")}
    run_pipeline(docs)
    exp_batch()
    exp_compare_defenses()


if __name__ == "__main__":
    main()
