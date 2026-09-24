#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面试题第 01 期 · 实验 1：Scaled Dot-Product Attention 从零实现

零依赖（只用标准库 math / random），直接运行：

    python attention.py

对应题目
  Q1：什么是 Transformer？Self-Attention 在算什么？为什么它比 RNN 更适合做 LLM？

四个实验分别回答四个追问
  [1] 注意力到底在算什么      → 打印 Q/K/V 与权重矩阵，肉眼验证「按相关性加权求和」
  [2] 为什么必须除以 sqrt(d_k) → 不缩放时 softmax 会饱和，注意力退化成 one-hot
  [3] 多头到底「多」在哪       → 把 d_model 切成 h 份，每个头看的是不同的关系
  [4] 因果掩码                 → 为什么 GPT 并行训练却不会「偷看答案」

为什么不用 numpy：面试演示 / 录像时最怕环境装不上。标准库跑 3 秒出结果，
比「先在 GPU 上跑一个 FlashAttention」更能说明问题——这是概念题，不是性能题。
"""

from __future__ import annotations

import math
import random
import sys

# Windows 控制台默认不是 UTF-8，重定向一下，避免中文输出乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# 0. 基础运算（手写，刻意不调库，让每一步都看得见）
# ---------------------------------------------------------------------------

def softmax(xs: list[float]) -> list[float]:
    """数值稳定版 softmax。先减最大值，避免 exp 上溢。"""
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    if total == 0:                      # 全被 mask 掉的极端情况
        return [0.0] * len(xs)
    return [e / total for e in exps]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """矩阵乘法 a @ b，形状 (n,k) @ (k,m) -> (n,m)。"""
    n, k, m = len(a), len(b), len(b[0])
    return [[sum(a[i][t] * b[t][j] for t in range(k)) for j in range(m)] for i in range(n)]


def transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*a)]


def entropy(ps: list[float]) -> float:
    """香农熵（自然对数）。熵越低 → 分布越尖锐 → 注意力越像 one-hot。"""
    return -sum(p * math.log(p + 1e-12) for p in ps)


def fmt_matrix(m: list[list[float]], prec: int = 3) -> str:
    return "\n".join("    [" + ", ".join(f"{v:+.{prec}f}" for v in row) + "]" for row in m)


def fmt_vec(v: list[float], prec: int = 3) -> str:
    return "[" + ", ".join(f"{x:+.{prec}f}" for x in v) + "]"


def hr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ---------------------------------------------------------------------------
# 1. 核心公式：Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k)) · V
# ---------------------------------------------------------------------------

def attention(
    q: list[list[float]],
    k: list[list[float]],
    v: list[list[float]],
    *,
    mask: list[list[bool]] | None = None,
    scale: bool = True,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """
    返回 (原始分数, 注意力权重, 输出)。

    q: (n_q, d_k)   k: (n_kv, d_k)   v: (n_kv, d_v)
    四个步骤，一步不多：
      1) scores = Q @ Kᵀ              —— 每个 query 和每个 key 的相似度
      2) scores /= sqrt(d_k)          —— 缩放（实验 2 讲为什么）
      3) mask 屏蔽不该看的位置         —— 实验 4
      4) out = softmax(scores) @ V    —— 按权重把 value 加权汇总
    """
    d_k = len(q[0])
    factor = math.sqrt(d_k) if scale else 1.0

    scores = [[s / factor for s in row] for row in matmul(q, transpose(k))]

    if mask is not None:
        scores = [
            [s if mask[i][j] else float("-inf") for j, s in enumerate(row)]
            for i, row in enumerate(scores)
        ]

    weights = [softmax(row) for row in scores]
    out = matmul(weights, v)
    return scores, weights, out


# ---------------------------------------------------------------------------
# 实验 1：注意力在算什么 —— 3 个 token 的全过程
# ---------------------------------------------------------------------------

def exp1_what_is_attention() -> None:
    hr("实验 1｜注意力到底在算什么：3 个 token，d_k = 4，全部数字可肉眼验算")

    # 手工造一组好解释的向量：
    #   token 0 的 query 指向「第 0 维」，而 token 0 和 token 2 的 key 都在第 0 维有分量
    #   → 预期：token 0 主要关注 token 0 和 token 2，几乎不看 token 1
    q = [
        [1.0, 0.0, 0.0, 0.0],   # token 0 在找什么
        [0.0, 1.0, 0.0, 0.0],   # token 1 在找什么
        [0.0, 0.0, 1.0, 0.0],   # token 2 在找什么
    ]
    k = [
        [1.0, 0.0, 0.0, 0.0],   # token 0 有什么
        [0.0, 1.0, 0.0, 0.0],   # token 1 有什么
        [0.9, 0.1, 0.0, 0.0],   # token 2 有什么（和第 0 维很像）
    ]
    v = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    print("Q（我在找什么）=")
    print(fmt_matrix(q))
    print("K（我有什么）=")
    print(fmt_matrix(k))

    scores, weights, out = attention(q, k, v)

    print("\n步骤 1｜scores = Q·Kᵀ，再除以 sqrt(d_k) = sqrt(4) = 2.0")
    print(fmt_matrix(scores))
    print("    解释：原始点积 token0·token0 = 1.0、token0·token2 = 0.9，除以 2.0 后为 0.50 / 0.45；")
    print("          token0·token1 = 0 → 0。分数越高 = 相关性越强。这就是『注意力』的字面来源。")

    print("\n步骤 2｜按行做 softmax，得到注意力权重（每行和 = 1.0）")
    print(fmt_matrix(weights))
    for i, row in enumerate(weights):
        print(f"    token {i} 的注意力分配：" + "  ".join(f"→token{j}: {w:.3f}" for j, w in enumerate(row))
              + f"   （行和 = {sum(row):.6f}）")

    print("\n步骤 3｜out = 权重 · V，即「按相关性把各 token 的 value 加权汇总」")
    print(fmt_matrix(out))
    w0 = weights[0]
    print("    token 0 的输出：" + fmt_vec(out[0]))
    print(f"    它 = {w0[0]:.3f} × V0 + {w0[1]:.3f} × V1 + {w0[2]:.3f} × V2"
          " —— 这就是『加权求和』的字面含义。")
    print("\n【结论】Self-Attention 没有魔法：一次矩阵乘法算相关性，一次 softmax 归一化，")
    print("        再一次矩阵乘法做加权汇总。所谓『全局感受野』就是 key 覆盖了全部位置。")


# ---------------------------------------------------------------------------
# 实验 2：为什么必须除以 sqrt(d_k)
# ---------------------------------------------------------------------------

def exp2_why_scale() -> None:
    hr("实验 2｜为什么要除以 sqrt(d_k)：不缩放会 softmax 饱和")

    print("推导：若 q、k 各维独立且 ~N(0,1)，则点积 q·k 的方差 = d_k，标准差 = sqrt(d_k)。")
    print("      维度越高，分数差得越开 → softmax 越接近 one-hot → 梯度趋近 0 → 学不动。")
    print("      除以 sqrt(d_k) 恰好把方差拉回 1。\n")
    print("  为了排除随机波动，每个 d_k 取 200 次实验的平均值。\n")

    random.seed(20260924)
    n_key, trials = 8, 200

    print(f"{'d_k':>6} | {'缩放?':<5} | {'最大注意力权重(均值)':>20} | {'熵(均值)':>10} | 解读")
    print("-" * 78)

    for d_k in (4, 16, 64, 256, 1024, 4096):
        for scale in (False, True):
            sum_max, sum_ent = 0.0, 0.0
            for _ in range(trials):
                q = [random.gauss(0, 1) for _ in range(d_k)]
                keys = [[random.gauss(0, 1) for _ in range(d_k)] for _ in range(n_key)]
                raw = [sum(a * b for a, b in zip(q, kk)) for kk in keys]
                factor = math.sqrt(d_k) if scale else 1.0
                w = softmax([r / factor for r in raw])
                sum_max += max(w)
                sum_ent += entropy(w)
            mx, ent = sum_max / trials, sum_ent / trials

            if not scale:
                if mx > 0.9:
                    note = "饱和！注意力退化成 one-hot"
                elif mx > 0.6:
                    note = "已经很尖锐，区分度被压扁"
                else:
                    note = "低维时影响还不明显"
            else:
                note = "稳定在可用区间（熵 ~1.4–2.0）"
            print(f"{d_k:>6} | {'是' if scale else '否':<5} | {mx:>20.4f} | {ent:>10.4f} | {note}")
        print("-" * 78)

    print("""
【读这张表的方法】只看『缩放=否』那一列：
  d_k 从 4 涨到 4096，最大注意力权重从 0.53 一路上涨到 0.99，平均熵从 1.28 掉到 0.03。
  这是**单调恶化**：维度越高，softmax 越接近 one-hot，反向传播到 softmax 输入的
  梯度几乎为 0。另一列（缩放=是）在所有维度上都稳在同一个区间（熵 1.70–1.74）
  ——这就是 sqrt(d_k) 存在的唯一理由。

  面试里说『为了数值稳定』只是及格；
  说出『点积方差是 d_k，缩放到方差 1，否则 softmax 饱和、梯度消失』才是满分。""")


# ---------------------------------------------------------------------------
# 实验 3：多头注意力
# ---------------------------------------------------------------------------

def exp3_multi_head() -> None:
    hr("实验 3｜多头注意力：d_model = 8，切给 h = 2 个头（每头 d_k = 4）")

    random.seed(7)
    d_model, n_head = 8, 2
    d_k = d_model // n_head
    n_tok = 4

    x = [[random.gauss(0, 1) for _ in range(d_model)] for _ in range(n_tok)]
    wq = [[random.gauss(0, 0.5) for _ in range(d_model)] for _ in range(d_model)]
    wk = [[random.gauss(0, 0.5) for _ in range(d_model)] for _ in range(d_model)]
    wv = [[random.gauss(0, 0.5) for _ in range(d_model)] for _ in range(d_model)]

    q_all, k_all, v_all = matmul(x, wq), matmul(x, wk), matmul(x, wv)

    print(f"输入 X: {n_tok} 个 token × {d_model} 维（这里用随机数代替 embedding 输出）")
    print(f"每个头处理 {d_k} 维：head0 取第 0–3 维，head1 取第 4–7 维\n")

    heads_out = []
    for h in range(n_head):
        lo, hi = h * d_k, (h + 1) * d_k
        q = [row[lo:hi] for row in q_all]
        k = [row[lo:hi] for row in k_all]
        v = [row[lo:hi] for row in v_all]
        _, w, out = attention(q, k, v)
        heads_out.append(out)
        print(f"--- head {h}（第 {lo}–{hi - 1} 维）的注意力权重 ---")
        print(fmt_matrix(w, prec=4))
        for i, row in enumerate(w):
            print(f"    token {i} 最关注：token {max(range(len(row)), key=lambda j: row[j])}"
                  f"  (权重 {max(row):.4f})")
        print()

    concat = [heads_out[0][i] + heads_out[1][i] for i in range(n_tok)]
    print("拼接各头输出（real 实现里还会再乘一次输出投影 Wo）：")
    print(fmt_matrix(concat))

    print("\n【结论】两个头在同一份输入上算出了不同的注意力分布——这就是『多头』的收益：")
    print("        不同的头可以在不同的子空间里学不同的关系（语法、指代、位置、共现）。")
    print("        注意：多头不增加参数量级——只是把 d_model 切成 h 份并行算，")
    print("        算完之后拼接、再乘一次输出投影 Wo。")


# ---------------------------------------------------------------------------
# 实验 4：因果掩码
# ---------------------------------------------------------------------------

def exp4_causal_mask() -> None:
    hr("实验 4｜因果掩码：为什么 GPT 能并行训练却不会偷看答案")

    random.seed(11)
    n, d_k = 4, 4
    q = [[random.gauss(0, 1) for _ in range(d_k)] for _ in range(n)]
    k = [[random.gauss(0, 1) for _ in range(d_k)] for _ in range(n)]
    v = [[random.gauss(0, 1) for _ in range(d_k)] for _ in range(n)]

    print("不加掩码（双向注意力，BERT 风格）——每个 token 都能看到右边：")
    _, w_bi, _ = attention(q, k, v)
    print(fmt_matrix(w_bi, prec=4))

    # 因果掩码：位置 i 只能看 j <= i
    mask = [[j <= i for j in range(n)] for i in range(n)]
    print("\n掩码矩阵（True = 允许看）：")
    for i, row in enumerate(mask):
        print(f"    token {i}: " + " ".join("✓" if ok else "✗" for ok in row))

    print("\n加因果掩码（自回归 LM 风格）——注意右上角全为 0，第 0 行只能是 1.0：")
    _, w_causal, _ = attention(q, k, v, mask=mask)
    print(fmt_matrix(w_causal, prec=4))

    print("\n【结论】掩码让位置 i 的输出只依赖 0..i，因此训练时可以把 n 个位置的预测")
    print("        一次性并行算出来（每个位置的『未来』已被屏蔽），而推理时逐个生成。")
    print("        RNN 做不到这一点：第 i 步必须等第 i-1 步算完，这是它的吞吐瓶颈。")


# ---------------------------------------------------------------------------

def main() -> None:
    print("面试题第 01 期 · 实验 1：Scaled Dot-Product Attention 从零实现")
    print("（零依赖，只用 Python 标准库）")
    exp1_what_is_attention()
    exp2_why_scale()
    exp3_multi_head()
    exp4_causal_mask()

    hr("复盘：这四个实验分别对应面试里的哪句话")
    print("""  实验 1 → 「Self-Attention 是按相关性对全局信息加权汇总」
  实验 2 → 「除以 sqrt(d_k) 是为了把点积方差从 d_k 拉回 1，避免 softmax 饱和」
  实验 3 → 「多头 = 在不同子空间并行学不同关系，不是简单变宽」
  实验 4 → 「掩码使训练可并行，这就是 Transformer 比 RNN 快的根本原因」

  再往上一步的工程含义（第 05 期会回扣）：
    · 分数矩阵是 n×n → 注意力成本随序列长度平方增长
    · 每个位置都要和全部位置交互 → 长上下文下 attention 成为瓶颈（见 scaling.py）
    · 因果掩码 + 前缀稳定 → KV Cache 才能跨请求复用（见 kv_cache.py）
""")


if __name__ == "__main__":
    main()
