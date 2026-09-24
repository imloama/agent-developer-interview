#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面试题第 01 期 · 实验 2：KV Cache 与长上下文的成本曲线

零依赖，直接运行：

    python kv_cache.py

对应追问
  · KV Cache 到底是什么？为什么它是推理的必备优化？
  · 为什么上下文越长，成本涨得比长度还快？
  · 为什么「把时间戳塞进 System Prompt」会让成本翻倍？
  · 128K 上下文的 KV Cache 到底占多少显存？

三个实验
  [1] prefill / decode / cache hit 三种状态的计算量差多少
  [2] 前缀稳定性 → KV Cache 命中率 → 真实账单差多少
  [3] KV Cache 显存占用（MHA vs GQA）

成本口径说明：本实验只数「乘加次数」与「元素个数」，不仿真真实硬件。
目的是让量级关系可见，不是预测真实延迟。
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 计算量模型
# ---------------------------------------------------------------------------

def proj_ops(n_tokens: int, d: int, n_layer: int) -> int:
    """给 n 个 token 算 Q/K/V 投影的乘加次数：3 n d² / 层。"""
    return n_layer * 3 * n_tokens * d * d


def attn_ops(n_query: int, n_key: int, d: int, n_layer: int) -> int:
    """注意力主体的乘加次数：n_q × n_k × d（scores）+ n_q × n_k × d（加权 V）。"""
    return n_layer * 2 * n_query * n_key * d


def kv_cache_elems(total_len: int, d: int, n_layer: int, n_kv_head: int, n_head: int) -> int:
    """KV Cache 元素个数（2 = K 和 V 各一份）。GQA 时 KV 头数少于 Q 头数。"""
    per_head = d // n_head
    return n_layer * 2 * total_len * n_kv_head * per_head


# ---------------------------------------------------------------------------
# 实验 1：prefill / decode / 无缓存
# ---------------------------------------------------------------------------

def exp1_prefill_decode() -> None:
    hr("实验 1｜prefill、decode 与「不用缓存」的计算量对比")

    n_ctx, n_new = 2000, 200          # 2000 token 上下文，还要生成 200 token
    d, n_layer = 4096, 32             # 约等于一个 7B 级模型（d=4096，32 层）

    print(f"设定：上下文 {n_ctx} token，生成 {n_new} token，d_model={d}，层数={n_layer}")
    print("      记为 C = 上下文长度，T = 已生成长度\n")

    # --- prefill：一次把 2000 个 token 全部算完 ---
    prefill = proj_ops(n_ctx, d, n_layer) + attn_ops(n_ctx, n_ctx, d, n_layer)

    # --- decode 带缓存：每步只算 1 个新 token 的投影 + 与全部历史的注意力 ---
    with_cache = 0
    for t in range(n_new):
        now = n_ctx + t + 1
        with_cache += proj_ops(1, d, n_layer) + attn_ops(1, now, d, n_layer)

    # --- decode 不带缓存：每步重算全部 token 的投影与全量注意力 ---
    no_cache = 0
    for t in range(n_new):
        now = n_ctx + t + 1
        no_cache += proj_ops(now, d, n_layer) + attn_ops(now, now, d, n_layer)

    print(f"{'阶段':<26} {'乘加次数':>14} {'相对 prefill':>14}")
    print("-" * 78)
    print(f"{'prefill（1 次，全量）':<26} {prefill:>14.3e} {'1.0×':>14}")
    print(f"{'decode（200 步，带缓存）':<26} {with_cache:>14.3e} {with_cache / prefill:>13.3f}×")
    print(f"{'decode（200 步，无缓存）':<26} {no_cache:>14.3e} {no_cache / prefill:>13.3f}×")
    print("-" * 78)
    print(f"『用不用 KV Cache』相差 {no_cache / with_cache:.1f} 倍。")

    print("""
【三种状态的工程含义】
  prefill  ：输入阶段，n×n 的注意力一次算完。计算密集（compute-bound），
             GPU 打满，计费按 input token 计价。
  decode   ：逐 token 生成阶段，每步只算 1 个 query，但要读完整份 KV Cache。
             访存密集（memory-bound），GPU 算力吃不满，计费按 output token 计价。
  cache hit：同一前缀的第二次请求，直接复用已算好的 KV，只需 prefill 新增部分。
             这是唯一能显著降本的状态——前提是前缀没变。

  减一个数量级的话：decode 贵在『每生成一个字都要把整个历史读一遍』，
  prefill 贵在『一次要算 n² 的注意力矩阵』。两者贵的地方不一样。""")


# ---------------------------------------------------------------------------
# 实验 2：前缀稳定性 → 命中率 → 账单
# ---------------------------------------------------------------------------

def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def exp2_prefix_stability() -> None:
    hr("实验 2｜前缀稳定性：一个小小的字符串决定命中率，命中率决定账单")

    # 用「字符数」近似 token 数，只为演示命中逻辑
    system_stable = "你是订单助手，负责查询订单状态并给出可执行建议。" * 20
    tools_block = "[工具定义] get_order / cancel_order / refund_order " * 20
    history = ["第 1 轮用户消息……" * 3] * 4        # 4 轮历史

    # 情景 A：System Prompt 固定，历史逐轮追加 → 前缀稳定
    # 情景 B：System Prompt 里插入了当前时间戳 → 每轮前缀都变
    conv_a, conv_b = [], []
    hits_a, total_a = 0, 0
    hits_b, total_b = 0, 0
    prev_a = prev_b = None

    for turn in range(1, 6):
        hist = "".join(history[: turn - 1])
        a = system_stable + tools_block + hist
        b = f"[当前时间 2026-09-24 09:{turn:02d}] " + system_stable + tools_block + hist

        if prev_a is not None:
            hit = common_prefix_len(prev_a, a)
            hits_a += hit
        if prev_b is not None:
            hit = common_prefix_len(prev_b, b)
            hits_b += hit
        total_a += len(a)
        total_b += len(b)
        conv_a.append(a)
        conv_b.append(b)
        prev_a, prev_b = a, b

    print("情景 A｜System Prompt 固定不变（时间戳放在用户消息里或根本不放）")
    print(f"    请求总量      : {total_a:>8} 字符")
    print(f"    命中前缀总量  : {hits_a:>8} 字符")
    print(f"    前缀命中率    : {hits_a / total_a:>8.1%}")

    print("\n情景 B｜System Prompt 里塞了『当前时间』（每次请求都不同）")
    print(f"    请求总量      : {total_b:>8} 字符")
    print(f"    命中前缀总量  : {hits_b:>8} 字符")
    print(f"    前缀命中率    : {hits_b / total_b:>8.1%}")

    # 计费：举例说明（价格是示意值，真实价格看各厂商定价页）
    miss_price, hit_price = 1.0, 0.25     # 每单位 token 的示意价格
    cost_a = (total_a - hits_a) * miss_price + hits_a * hit_price
    cost_b = (total_b - hits_b) * miss_price + hits_b * hit_price
    print(f"\n按「未命中 1.0 / 命中 0.25」的示意单价估算：")
    print(f"    情景 A 成本 ≈ {cost_a:,.0f}（相对值）")
    print(f"    情景 B 成本 ≈ {cost_b:,.0f}（相对值）")
    print(f"    成本比      ≈ {cost_b / cost_a:.2f}×")

    print("""
【结论】前缀缓存的命中条件只有一个：**从第一个 token 开始逐字节相同**。
  任何一个位于前缀里的动态内容（时间、用户名、随机 ID、检索到的文档）都会让
  它后面的所有内容全部失效。

  工程做法：
    · System 区只放长期稳定的契约（身份、约束、工具定义、固定示例）
    · 动态内容一律后置到 User 区 / 追加在新的一轮里
    · 多轮对话天然满足「前缀稳定」——这是 append-only 消息历史的隐藏收益

  这条结论会在第 05 期《上下文工程》里再次出现：上下文分区不只是「放什么」，
  更是「放在哪个位置」，因为位置决定缓存。""")


# ---------------------------------------------------------------------------
# 实验 3：KV Cache 显存
# ---------------------------------------------------------------------------

def exp3_kv_memory() -> None:
    hr("实验 3｜KV Cache 显存：为什么长上下文这么贵")

    d, n_layer = 4096, 32
    cases = [
        ("7B 级（32 层 / 32 头）", n_layer, d, 32, 32),
        ("70B 级（80 层 / 64 头）", 80, 8192, 64, 64),
    ]

    print(f"{'配置':<24} {'上下文':>9} {'KV Cache (fp16)':>18} {'GQA 8 KV 头后':>16}")
    print("-" * 78)

    for name, layers, dd, n_head, n_kv in cases:
        for ctx in (8_192, 32_768, 131_072):
            full = kv_cache_elems(ctx, dd, layers, n_head, n_head) * 2  # fp16 = 2 字节
            gqa = kv_cache_elems(ctx, dd, layers, 8, n_head) * 2         # GQA: 8 组 KV 头
            print(f"{name:<24} {ctx:>9,} {full / 1024**3:>15.1f} GB {gqa / 1024**3:>13.1f} GB")
        print("-" * 78)

    print("""
【结论】128K 上下文的 KV Cache 在 fp16 下是几十 GB 量级——比很多模型的权重还大。
  所以长上下文真正的门槛不是「模型支不支持」，而是：
    · 显存：能不能装下这么多 KV（→ GQA / MQA / 量化 KV / PagedAttention）
    · 带宽：decode 每步都要把 KV 读一遍，读得越慢生成越慢
    · 注意力本身：n² 的计算量（见 scaling.py）

  GQA（Grouped-Query Attention）把 KV 头数从 32 降到 8，显存直接 /4，
  这就是现代模型默认用 GQA 的原因：几乎不损失质量，换来 4 倍显存。""")


# ---------------------------------------------------------------------------

def main() -> None:
    print("面试题第 01 期 · 实验 2：KV Cache 与长上下文的成本曲线")
    exp1_prefill_decode()
    exp2_prefix_stability()
    exp3_kv_memory()

    hr("复盘：这三个实验结果对应面试里的哪句话")
    print("""  实验 1 → 「KV Cache 把 decode 从 n² 降到 n，是推理的必备优化」
  实验 2 → 「前缀缓存命中要求逐 token 完全相同，所以动态内容必须后置」
  实验 3 → 「长上下文的真实瓶颈是显存与带宽，不是模型能不能读」

  面试里能顺着这条线讲下去的候选人非常少：
    注意力是 n²（机制）→ 所以长上下文成本超线性（成本模型）→
    所以要 KV Cache（优化）→ 所以缓存命中与否决定账单（工程）→
    所以 System 区不能放动态内容（设计规范）。
  这条链子能把机制题回答成工程题。""")


if __name__ == "__main__":
    main()
