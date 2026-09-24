#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面试题第 01 期 · 实验 4：O(n²) 到底意味着什么

零依赖，直接运行：

    python scaling.py

对应追问
  · Attention 是 O(n²)，那长上下文的成本与延迟曲线是什么形状？（新增题 N2）
  · 上下文窗口已经 1M 了，是不是不用管长度了？
  · 为什么上下文越长效果反而越差？

三个实验
  [1] 数一数一层 Transformer 里，注意力部分和 FFN 部分各花多少计算量
  [2] 求出「注意力开始压过 FFN」的临界长度 n*，理解长上下文为什么是另一个问题
  [3] 真实计时：纯 Python 跑不同长度的注意力，看实测耗时如何随 n 增长

口径说明：只数乘加次数（乘加算 1 次），不仿真硬件。目的是让量级关系可见。
"""

from __future__ import annotations

import math
import random
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 一层 Transformer 的计算量（乘加次数）
# ---------------------------------------------------------------------------

def layer_ops(n: int, d: int) -> tuple[int, int]:
    """
    返回 (注意力相关计算量, FFN + 投影计算量)。

    按标准 decoder-only 一层拆开：
      QKV 投影      : 3 n d²      —— 线性于 n
      注意力 scores : n² d         —— 平方于 n   ← 重点
      加权 V        : n² d         —— 平方于 n   ← 重点
      输出投影 Wo   : n d²
      FFN（d→4d→d）: 8 n d²
    """
    attn = n * n * d * 2                     # scores + 加权 V
    linear = 3 * n * d * d + n * d * d + 8 * n * d * d
    return attn, linear


def exp1_breakdown() -> None:
    hr("实验 1｜一层里注意力 vs 其余部分：谁在涨，涨多快")

    d = 4096
    print(f"设定 d_model = {d}（约 7B / 8B 级模型），只算一层，单位：乘加次数\n")
    print(f"{'n (上下文)':>12} {'注意力 2n²d':>16} {'线性部分 12nd²':>18} {'注意力占比':>10}")
    print("-" * 78)

    for n in (512, 2_048, 8_192, 32_768, 131_072):
        attn, linear = layer_ops(n, d)
        print(f"{n:>12,} {attn:>16.3e} {linear:>18.3e} {attn / (attn + linear):>9.1%}")

    print("""
【读法】
  n 每翻一倍：注意力部分涨 4 倍，线性部分只涨 2 倍。
  短上下文时线性部分是主力（FFN 的参数量远大于注意力）；
  长上下文时注意力反超——而且是平方级反超。

  这就是「窗口开到 1M 不等于成本线性增长」的原因，也是「窗口大就不用管长度」
  这个判断错在哪的原因。""")


def exp2_crossover() -> None:
    hr("实验 2｜临界长度 n*：注意力从什么时候开始主导成本")

    print("  令 2·n²·d = 12·n·d²  →  n* = 6d\n")

    print(f"{'d_model':>10} {'n* = 6d':>12} {'说明':<40}")
    print("-" * 78)
    rows = [
        (768, "1B 级以下，小模型"),
        (2048, "3B 级"),
        (4096, "7B / 8B 级"),
        (8192, "70B 级"),
        (16384, "超大规模 MoE"),
    ]
    for d, note in rows:
        print(f"{d:>10,} {6 * d:>12,} {note:<40}")

    print("""
【结论】临界长度大致是 d_model 的 6 倍（按本实验的计数口径，系数随口径变化）。
  d=4096 的模型，n* ≈ 24.5K：上下文短于它时，算力主要花在 FFN；
  长于它时，算力主要花在注意力。

  实际影响：
    · 32K 上下文 → 注意力和 FFN 相当，两端都要优化
    · 128K 上下文 → 注意力是绝对大头 → 所以有了 FlashAttention、稀疏注意力、
      KV 量化、滑窗注意力这些专门优化
    · 这也解释了为什么「长上下文能力」通常伴随着一整套推理优化一起发布，
      而不是单纯「把窗口调大」。

  面试加分点：说得出「长上下文瓶颈有两层——机制层是 n²，系统层是 KV 显存
  与带宽」，比只说「O(n²)」高一个档。机制层的 n² 在这里，系统层在 kv_cache.py。""")


# ---------------------------------------------------------------------------
# 实验 3：真实计时
# ---------------------------------------------------------------------------

def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def real_attention(q, k, v):
    """教科书实现：scores = QKᵀ/√d → softmax → @V。"""
    n, d = len(q), len(q[0])
    factor = math.sqrt(d)
    kt = [list(col) for col in zip(*k)]
    out = []
    for i in range(n):
        row = [sum(q[i][t] * kt[t][j] for t in range(d)) / factor for j in range(n)]
        w = softmax(row)
        out.append([sum(w[j] * v[j][t] for j in range(n)) for t in range(d)])
    return out


def exp3_real_timing() -> None:
    hr("实验 3｜真实计时：纯 Python 跑不同长度，看实测耗时怎么涨")

    random.seed(2026)
    d = 16
    print(f"d_k = {d}（故意开小，让纯 Python 也能跑完；增长规律与 d 无关）\n")
    print(f"{'n':>6} {'实测耗时(s)':>13} {'相对 n=32':>11} {'理论倍率 (n/32)²':>18}")
    print("-" * 78)

    base_time = None
    for n in (32, 64, 128, 192):
        q = [[random.gauss(0, 1) for _ in range(d)] for _ in range(n)]
        k = [[random.gauss(0, 1) for _ in range(d)] for _ in range(n)]
        v = [[random.gauss(0, 1) for _ in range(d)] for _ in range(n)]

        t0 = time.perf_counter()
        real_attention(q, k, v)
        dt = time.perf_counter() - t0

        if base_time is None:
            base_time = dt
        theoretical = (n / 32) ** 2
        print(f"{n:>6} {dt:>13.4f} {dt / base_time:>10.2f}× {theoretical:>17.2f}×")

    print("""
【结论】实测倍率贴着 (n/32)² 走——这就是 O(n²) 的字面含义。
  n 翻倍 → 耗时约 4 倍；n 涨 10 倍 → 耗时约 100 倍。

  注意：真实推理里，长上下文的表现还有额外两层修正：
    · 好的 kernel（如 FlashAttention）减少了显存往返，常数变小，但阶数不变
    · decode 阶段是 memory-bound，瓶颈在 KV 读取而非算力 → 曲线又不一样
  所以「n²」是机制层的结论，不能直接当延迟预测公式用。面试时把这两层分开讲。""")


# ---------------------------------------------------------------------------

def main() -> None:
    print("面试题第 01 期 · 实验 4：O(n²) 到底意味着什么")
    exp1_breakdown()
    exp2_crossover()
    exp3_real_timing()

    hr("复盘：一条从机制到成本的完整链条")
    print("""  注意力是 n²（机制）
    → 长上下文成本超线性（成本模型）
    → 所以需要 KV Cache 把 decode 降到 O(n)（优化）
    → 所以 KV Cache 命中与否决定账单（工程）
    → 所以 System 区不能放动态内容（设计规范）
    → 所以长上下文还必须配 GQA / 量化 / FlashAttention（系统优化）

  能顺着这条链子从「Self-Attention 是什么」一路讲到「所以我们的 System Prompt
  这样组织」，就是这道题的高分答案。""")
    print("\n提示：把 exp3_real_timing 里的 n 改成 512、1024，亲眼看纯 Python 跑不动——")
    print("      这就是为什么真实推理必须换 CUDA kernel，也是最好的「O(n²) 直观感受」。")


if __name__ == "__main__":
    main()
