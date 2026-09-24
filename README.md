# agent-developer-interview · 面试题系列配套代码

博客「面试题」栏目（<https://www.teamagent.site/interview/>）各篇文章的配套实验代码。

每篇文章对应一个目录，目录名与文章 slug 一致。

> ⚠️ 这些是**教学演示代码**，不是生产代码。
> 为了把概念讲清楚，它们刻意省略了鉴权、并发控制、错误重试、日志、配置管理等
> 生产系统必备的部分。请勿直接用于生产。

## 环境要求

- Python 3.10+（用到 `list[float]` 这类类型注解语法）
- 无需任何第三方包

Windows 下若控制台中文乱码，脚本已内置 `sys.stdout.reconfigure(encoding="utf-8")` 处理。

## 篇目与代码对应

| 期 | 文章 | 代码目录 |
| :--- | :--- | :--- |
| 01 | [LLM 基础与模型边界：Transformer、Token 与幻觉](https://www.teamagent.site/interview/agent-01-llm-basics/) | [`agent-01-llm-basics/`](agent-01-llm-basics/) |
| 02 | [什么是 AI Agent：定义、判据与组件](https://www.teamagent.site/interview/agent-02-what-is-agent/) | [`agent-02-what-is-agent/`](agent-02-what-is-agent/) |

## 目录结构

```
agent-developer-interview/
├── agent-01-llm-basics/          第 01 期：LLM 基础与模型边界
│   ├── attention.py              Self-Attention 从零实现 + 为什么除以 sqrt(d_k) + 多头 + 因果掩码
│   ├── kv_cache.py               prefill/decode/cache hit 的计算量、前缀稳定性与账单、KV 显存
│   ├── sampling.py               temperature/top_k/top_p + 「temperature=0 为什么不可复现」
│   ├── scaling.py                O(n²) 的操作计数、注意力与 FFN 的成本交叉点、实测耗时
│   └── citation_guard.py         幻觉治理：强制引用 + 四个校验器 + 结构化错误回传
└── agent-02-what-is-agent/       第 02 期：什么是 AI Agent
    ├── llm.py                    极简 LLM 客户端（Mock + 真实 OpenAI 兼容）+ 工具层
    ├── workflow_vs_agent.py      同一任务写两遍：控制流在代码手里 vs 在模型手里
    └── convergence.py            300 次运行量化「为什么 Agent 最后变成 Workflow」
```

## 怎么跑

```bash
cd agent-01-llm-basics
python attention.py
python kv_cache.py
python sampling.py
python scaling.py
python citation_guard.py

cd ../agent-02-what-is-agent
python llm.py                    # 先自检工具层与 MockModel
python workflow_vs_agent.py
python convergence.py
```

接真实模型（可选）：

```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4   # 智谱 GLM
export OPENAI_MODEL=glm-4-plus
python workflow_vs_agent.py --real
```

兼容任何 OpenAI 格式的接口：智谱 GLM / DeepSeek / Moonshot / OpenAI / 本地 vLLM。

## 设计原则

| 原则 | 说明 |
| :--- | :--- |
| **零依赖** | 只用 Python 标准库，不需要 `pip install` 任何东西。`python xxx.py` 直接跑 |
| **离线可跑** | 涉及模型调用的实验默认走脚本化 MockModel，不联网、不花钱、结果可复现 |
| **真实可选** | 需要看真实模型行为时，设 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 即可切到真实接口 |
| **每个实验对应一个结论** | 脚本里的注释明确写出「这个实验要验证正文里的哪一句」 |
| **给数字不给形容词** | 输出打印真实数值与倍率，让结论可验证，而不是「Agent 更贵」这类空话 |

## 后续批次

| 批次 | 篇目 | 对应目录 |
| :--- | :--- | :--- |
| 第 1 批 | 01–02 | `agent-01-llm-basics/`、`agent-02-what-is-agent/` ✅ |
| 第 2 批 | 03–06 | 待建：Function Calling / MCP / 上下文工程 / 记忆系统 |
| 第 3 批 | 07–09 | 待建：控制流与 Runtime / RAG / 复杂架构取舍 |
| 第 4 批 | 10 | 待建：生产化与系统设计 |

命名约定：目录名 = 文章 slug（`agent-<期号>-<主题>`）。
未来接入 Java / Golang 面试分组时，用 `java-` / `golang-` 前缀平级扩展。

## 每篇代码的写法约定

1. 文件头注释写清：**这篇讲什么题、每个实验验证正文里的哪个结论**
2. 每个实验一个小节，`hr()` 打印分隔标题，方便分段阅读与讲解
3. 结论用 `print` 打印成成段文字，包含该结论的工程含义
4. 输出里凡是数字，都在代码里可追溯（不做手工填数）
5. 诚实标注口径：成本模型、token 估算、噪声模拟都写明「这是示意口径」

## 授权

[MIT](LICENSE)
