#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面试题第 01 期 · 实验 3：采样参数与「temperature=0 就确定了吗」

零依赖，直接运行：

    python sampling.py

对应追问
  · temperature / top_p / top_k 分别控制什么？一般调哪个？
  · temperature=0 输出就完全可复现了吗？（新增题 N1）
  · Agent 场景应该用什么采样参数？

三个实验
  [1] temperature / top_k / top_p 对同一个分布做了什么
  [2] temperature=0 看着确定，但「同一份 logits」在两种等价求和顺序下会给出不同 argmax
  [3] 浮点累加顺序为什么会在真实硬件上不同，以及工程上怎么兜底
"""

from __future__ import annotations

import math
import random
import struct
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# 实验 1：三个旋钮各管什么
# ---------------------------------------------------------------------------

TOKENS = ["北京", "上海", "广州", "天津", "深圳", "北京烤鸭", "东京"]
LOGITS = [9.2, 8.1, 7.4, 6.2, 5.9, 4.1, 3.8]


def show_dist(label: str, probs: list[float], keep: list[int] | None = None) -> None:
    print(f"  {label}")
    for i, (t, p) in enumerate(zip(TOKENS, probs)):
        if keep is not None and i not in keep:
            continue
        bar = "█" * max(0, int(round(p * 50)))
        print(f"    {t:<8} p={p:7.4f}  {bar}")


def apply_top_k(logits: list[float], k: int) -> list[float]:
    if k >= len(logits):
        return list(logits)
    cutoff = sorted(logits, reverse=True)[k - 1]
    return [v if v >= cutoff else float("-inf") for v in logits]


def apply_top_p(logits: list[float], p: float) -> list[float]:
    """核采样：从高到低累加概率，累计刚好超过 p 就截断（至少保留 1 个）。"""
    probs = softmax(logits)
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    cum, keep = 0.0, []
    for i in order:
        keep.append(i)
        cum += probs[i]
        if cum >= p:
            break
    keep_set = set(keep)
    return [v if i in keep_set else float("-inf") for i, v in enumerate(logits)]


def exp1_knobs() -> None:
    hr("实验 1｜temperature / top_k / top_p 对同一个分布做了什么")

    print(f"  原始 logits：{dict(zip(TOKENS, LOGITS))}\n")
    show_dist("不做任何处理（原始 softmax）：", softmax(LOGITS))
    print()

    for temp in (0.2, 0.7, 1.0, 1.8):
        show_dist(f"temperature = {temp}", softmax([v / temp for v in LOGITS]))
        print()

    kept_k = [i for i, v in enumerate(apply_top_k(LOGITS, 3)) if v != float("-inf")]
    show_dist("top_k = 3 之后再采样（其余概率归零并重新归一化）：",
              softmax(apply_top_k(LOGITS, 3)), keep=kept_k)
    print()

    kept_p = [i for i, v in enumerate(apply_top_p(LOGITS, 0.9)) if v != float("-inf")]
    show_dist("top_p = 0.9（核采样：只保留累计概率前 90% 的那几个）：",
              softmax(apply_top_p(LOGITS, 0.9)), keep=kept_p)

    print("""
【分工，一句话记牢】
  temperature：把分布整体「拉平 / 变尖」。低 → 更确定，高 → 更有创意。
  top_k      ：固定保留 k 个候选。简单，但 k 是绝对数量，不适应分布形状。
  top_p      ：保留累计概率达到 p 的最小候选集。候选数量自适应。

  只调一个即可（通常调 temperature，或 temperature + top_p）。
  同时把三个都调低，是新手常见操作——收益不叠加，只是更难归因。
  模型输出 logits 是分布；temperature=0 时取 argmax，就是「贪心解码」。

【Agent 场景的默认值】
  工具调用 / 结构化输出 / 代码生成 → 0 ~ 0.3（要的是稳定，不是创意）
  需求澄清 / 头脑风暴 / 文案     → 0.7 ~ 1.0
  别用「写作任务保留 0.7、工具调用 0.2」这种全局设置——采样参数应该是
  按**调用类型**分别配置的，写在 Runtime 里（第 07 期会讲）。""")


# ---------------------------------------------------------------------------
# 实验 2：temperature=0 能不能保证可复现
# ---------------------------------------------------------------------------

def to_bf16(x: float) -> float:
    """把 float32 舍入到 bf16（8 位指数 + 7 位尾数）——模拟权重/激活的精度。"""
    i = int.from_bytes(struct.pack(">f", x), "big")
    i = (i + 0x8000) & 0xFFFF0000          # 低 16 位四舍五入
    return struct.unpack(">f", i.to_bytes(4, "big"))[0]


def to_fp32(x: float) -> float:
    """舍入到 fp32 —— 用来模拟「每一步加法后都写回 fp32 累加器」。"""
    return struct.unpack(">f", struct.pack(">f", x))[0]


# 四种真实的归约实现。它们计算的是同一个数学表达式，只是加法顺序不同。
def sum_sequential(v: list[float]) -> float:
    """顺序累加：一维串行 kernel。"""
    acc = 0.0
    for x in v:
        acc = to_fp32(acc + x)
    return acc


def sum_pairwise(v: list[float]) -> float:
    """树形（两两）归约：warp shuffle / 并行归约。"""
    cur = list(v)
    while len(cur) > 1:
        nxt = [to_fp32(cur[i] + cur[i + 1]) for i in range(0, len(cur) - 1, 2)]
        if len(cur) % 2:
            nxt.append(cur[-1])
        cur = nxt
    return cur[0]


def sum_blocked(v: list[float], block: int = 128) -> float:
    """分块累加：块内串行 + 块间归约，最常见的 CUDA kernel 形状。"""
    parts = []
    for i in range(0, len(v), block):
        acc = 0.0
        for x in v[i : i + block]:
            acc = to_fp32(acc + x)
        parts.append(acc)
    acc = 0.0
    for p in parts:
        acc = to_fp32(acc + p)
    return acc


def exp2_not_reproducible() -> None:
    hr("实验 2｜temperature=0 能保证输出可复现吗？——不能")

    random.seed(20260924)
    N = 4096                                       # 模拟 4096 次乘加贡献（一个隐藏维度的累加）

    # 权重/激活精度是 bf16（现代模型默认），累加器是 fp32 —— 这就是误差的来源
    base = [to_bf16(random.gauss(0.0, 1.0)) for _ in range(N)]
    # C 明显更低：base 的求和是负数，放大 1.2 倍让它更负，保证 C 不会成为 argmax
    contribs_C = [to_bf16(v * 1.2) for v in base]

    kernels = {
        "kernel ①（顺序累加）": sum_sequential,
        "kernel ②（树形归约）": sum_pairwise,
        "kernel ③（分块 128）": sum_blocked,
    }

    print(f"  每个候选 token 的 logit = {N} 个 bf16 贡献项之和，累加器为 fp32。")
    print("  真实硬件上，kernel 形状 / 归约顺序 / 分块大小都会改变加法顺序。")
    print("  本实验固定数学表达式不变，只换三种「等价的」加法顺序。\n")

    # ---------- 主表：以 1e-4 的差距展示具体数值 ----------
    GAP_DEMO = 1e-4
    cA = base
    cB = base[:-1] + [base[-1] + GAP_DEMO]
    true_A, true_B = sum(cA), sum(cB)
    true_C = sum(contribs_C)

    print(f"  【演示】令 A 与 B 的真实 logit 差距 = {GAP_DEMO:.0e}\n")
    print(f"  {'kernel':<22} {'logit(A)':>13} {'logit(B)':>13} {'logit(C)':>10} {'argmax':>7}")
    print("  " + "-" * 70)
    for name, fn in kernels.items():
        la, lb, lc = fn(cA), fn(cB), fn(contribs_C)
        am = max(range(3), key=lambda i: (la, lb, lc)[i])
        print(f"  {name:<22} {la:>13.6f} {lb:>13.6f} {lc:>10.2f} {'ABC'[am]:>7}")

    err = max(abs(fn(cA) - true_A) for fn in kernels.values())
    print(f"\n  真值（fp64 精确求和）：A = {true_A:.7f}   B = {true_B:.7f}")
    print(f"  累加顺序引入的最大误差  ：{err:.3e}")
    print(f"  A / B 的真实差距        ：{abs(true_A - true_B):.3e}")
    print(f"  → 误差 ÷ 差距 = {err / abs(true_A - true_B):.2f}，说明"
          f"{'误差还不足以翻转' if err < abs(true_A - true_B) else '误差已经足以翻转'}。")

    # ---------- 扫描：把差距逐步缩小，看翻转点在哪 ----------
    print("\n  【扫描】把 A / B 的差距逐步缩小，观察三个 kernel 是否还一致：\n")
    print(f"  {'A/B 真实差距':>12} {'① 的 argmax':>12} {'② 的 argmax':>12} {'③ 的 argmax':>12} {'一致?':>7}")
    print("  " + "-" * 70)

    flip_gap = None
    for gap in (1e-3, 1e-4, 1e-5, 3e-6, 1e-6):
        cB_i = base[:-1] + [base[-1] + gap]
        picks = [fn(cB_i) > fn(cA) for fn in kernels.values()]   # True = 选 B
        labels = "".join("B" if p else "A" for p in picks)
        same = len(set(picks)) == 1
        if not same and flip_gap is None:
            flip_gap = gap
        print(f"  {gap:>12.0e} {picks[0] and 'B' or 'A':>12} "
              f"{picks[1] and 'B' or 'A':>12} {picks[2] and 'B' or 'A':>12} "
              f"{'一致' if same else '❌ 不一致':>7}")

    if flip_gap is not None:
        print(f"\n  ⚠️ 当 A / B 的真实差距缩小到 {flip_gap:.0e} 时，"
              f"三个 kernel 对同一个问题给出了不同的 argmax。")
        print("     这不是随机采样造成的——本次全程 temperature=0（纯 argmax）、没有采样。")
        print("     原因只有一个：**浮点加法不满足结合律**，(a+b)+c ≠ a+(b+c)。")
    else:
        print("\n  本组参数下未出现翻转，但误差量级已与差距同阶——换 kernel 形状即可翻转。")

    print(f"""
【结论】temperature=0 ≠ 可复现。
  同一份权重、同一份输入、同一套数学公式，在不同 kernel / 不同批次大小 /
  不同分块策略下会得到不同的 logit；只要 top-2 的差距落进误差量级，argmax 就翻转。
  真实模型里 top-2 的 logit 差距经常就在 1e-4 以下，所以这不是理论担忧。

【还会改变数值的现实因素（比 kernel 更常见）】
  1. 累加精度：fp32 累加器 vs fp16 累加器，误差差 2~3 个数量级
  2. 批处理：同一请求和别的请求拼在同一个 batch，归约长度变了 → 数值也变
  3. MoE 路由：负载均衡策略可能把同一个 token 分给不同专家
  4. 后端 / 内核版本：kernel 换一次实现，末位就变；不同推理后端结果也不同
  5. 并行策略：tensor parallel 的切分方式改变归约树形状

【这对工程意味着什么】
  · 不要用「两次调用输出完全相同」当自动化测试的断言——会随机挂。
    正确做法：断言**结构**（JSON schema 合法）、断言**语义**（关键字段命中）、
    或断言**答案属于可接受集合**，而不是字符串相等。
  · 要真正可复现，必须同时固定：kernel 实现、批次大小、并行策略、累加精度、
    模型后端版本。只设 temperature=0 远远不够。
  · 更重要的推论：既然输出天然不确定，**关键路径就不能依赖模型输出做确定性判断**。
    金额、权限、状态、事务——该由代码判断的一律不要交给模型。
    这是 Agent 工程的第一条纪律，第 07 期讲控制流时会展开。""")


# ---------------------------------------------------------------------------

def main() -> None:
    print("面试题第 01 期 · 实验 3：采样参数与「temperature=0 就确定了吗」")
    exp1_knobs()
    exp2_not_reproducible()

    hr("复盘：这三个旋钮和一个陷阱")
    print("""  temperature → 分布平坦度；top_k → 固定数量截断；top_p → 自适应数量截断
  三者只需调一个（或 temperature + top_p）
  temperature=0 ≠ 可复现：浮点加法不满足结合律，误差足以在 logits 接近时翻转 argmax
  → 因此工程上：不确定的输出不能承载确定的判断""")


if __name__ == "__main__":
    main()
