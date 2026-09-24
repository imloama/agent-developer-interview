# 第 01 期配套代码：LLM 基础与模型边界

对应文章：[LLM 基础与模型边界：Transformer、Token 与幻觉](https://www.teamagent.site/interview/agent-01-llm-basics/)

零依赖，只用 Python 标准库。逐个运行即可，每个脚本 3 秒内跑完。

| 脚本 | 回答的追问 | 关键输出 |
| :--- | :--- | :--- |
| `attention.py` | Self-Attention 在算什么 / 为什么要除以 sqrt(d_k) / 多头是什么 / 因果掩码干什么 | 3 个 token 的完整注意力矩阵；不缩放时最大权重从 0.53 涨到 0.99、熵从 1.28 掉到 0.03 |
| `kv_cache.py` | KV Cache 为什么必需 / 长上下文成本曲线的两层 / 为什么时间戳不能放 System Prompt | decode 带缓存是不带缓存的 1/2100；前缀命中率 79.1% vs 1.1%，成本差 2.48 倍；128K 上下文 KV 占 64 GB |
| `sampling.py` | temperature/top_k/top_p 各管什么 / temperature=0 能保证可复现吗 | 同一份 logits 在三种「等价」归约顺序下，A/B 差距缩到 1e-6 时 argmax 开始不一致 |
| `scaling.py` | Attention 是 O(n²)，长上下文的成本曲线是什么形状 | 注意力占比从 512 token 的 2.0% 涨到 128K token 的 84.2%；临界长度 n* ≈ 6·d_model；实测倍率贴着 (n/32)² |
| `citation_guard.py` | 幻觉为什么不能根治 / 怎么降低而非消除 | 提示词写了「不要编造」模型照样编；四个校验器拦下 3 个问题，修复后通过；无召回时必须拒答 |

## 快速跑

```bash
python attention.py
python kv_cache.py
python sampling.py
python scaling.py
python citation_guard.py
```

## 建议的录像切分

| 段落 | 脚本 | 讲解重点 |
| :--- | :--- | :--- |
| 1. 注意力到底在算什么 | `attention.py` 实验 1 | 3×3 矩阵肉眼验算，打破「注意力很玄」的印象 |
| 2. 为什么除以 sqrt(d_k) | `attention.py` 实验 2 | 看那张 200 次取均值的表，不缩放时熵一路掉到 0.03 |
| 3. 掩码与并行训练 | `attention.py` 实验 4 | 对比双向/因果两套权重矩阵，解释并行训练的由来 |
| 4. 从机制到成本 | `scaling.py` 实验 1–2 | 注意力占比曲线 + n* ≈ 6d，说明长上下文是另一个问题 |
| 5. 从成本到缓存 | `kv_cache.py` 实验 1–2 | prefill/decode/cache hit 三态 + 前缀稳定性决定账单 |
| 6. 幻觉的结构性成因 | `sampling.py` 实验 2 | 浮点加法不满足结合律，temperature=0 也不可复现 |
| 7. 幻觉治理落在代码里 | `citation_guard.py` | 提示词没有强制力，校验器有 |

## 三个值得在视频里「改一改再跑」的参数

1. `attention.py` → `exp2_why_scale` 里的 `trials`，改成 10 看噪声有多大
2. `scaling.py` → `exp3_real_timing` 里的 n 加一个 `512`，让观众亲眼看到纯 Python 跑不动
3. `sampling.py` → `exp2_not_reproducible` 里的 `N` 从 4096 改到 16384，观察误差随 √N 增长、翻转门槛抬高

## 输出里最该被记住的四个数字

- **0.53 → 0.99**：不缩放时最大注意力权重随 d_k 的上升（softmax 饱和）
- **84.2%**：128K 上下文下注意力占一层计算量的比例
- **64 GB**：128K 上下文、32 层、d=4096 的 fp16 KV Cache 占用（GQA 后 16 GB）
- **2.48×**：System Prompt 里塞时间戳导致的前缀缓存失效成本倍数
