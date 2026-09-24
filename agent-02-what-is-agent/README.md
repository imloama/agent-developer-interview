# 第 02 期配套代码：什么是 AI Agent

对应文章：[什么是 AI Agent：定义、判据与组件](https://www.teamagent.site/interview/agent-02-what-is-agent/)

零依赖，默认离线可跑。这一期代码的核心价值是**把「Agent 与 Workflow 的区别」
从定义之争变成可测量的数字**。

| 脚本 | 回答的追问 | 关键输出 |
| :--- | :--- | :--- |
| `llm.py` | （公共模块） | 工具层自检 + MockModel 的理想路径/噪声路径演示 |
| `workflow_vs_agent.py` | 什么是 Agent / 和 Workflow 的本质区别 / 用 Function Calling 就算 Agent 吗 | 同一任务两条实现：Agent 成本是 Workflow 的 4.54 倍，调用 4 次 vs 2 次 |
| `convergence.py` | 为什么 Agent 产品最后都变成了 Workflow | 300 次运行：Agent 对不可退款订单发起 76 次越序退款，加 Guardrails 后归零 |

## 快速跑

```bash
python llm.py                    # 先看工具层与 MockModel 的行为
python workflow_vs_agent.py
python convergence.py
```

## 这一期最重要的那张表（`convergence.py` 输出）

订单 A1001（**可退款**），300 次运行，模型决策噪声 25%：

| 方案 | 终态正确率 | 越序写 | 平均调用 | 平均成本 | 路径种类 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| A. Workflow（纯代码） | 100.0% | 0 | 2.00 | 0.00224 | 1 |
| B. Agent（模型决定） | 53.7% | 76 | 4.45 | 0.01159 | 31 |
| C. Agent + Guardrails | 100.0% | 0 | 4.00 | 0.01008 | 1 |

订单 A1002（**已过退款期、不可退款**），同样的 300 次：

| 方案 | 终态正确率 | 越序写 | 平均调用 | 平均成本 | 路径种类 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| A. Workflow（纯代码） | 100.0% | 0 | 1.00 | 0.00078 | 1 |
| B. Agent（模型决定） | 80.3% | 76 | 3.70 | 0.00928 | 23 |
| C. Agent + Guardrails | 100.0% | 0 | 3.00 | 0.00720 | 1 |

**「越序写 = 76」的意思是：不可退款期已过的订单，被 Agent 发起了 76 次退款。**
这不是延迟或体验问题，是直接的资金损失。这就是 Q2 的答案要落到工程约束上的原因。

## 关于 MockModel 的诚实说明

`MockModel` **不是**在模拟真实模型的行为分布。它把「模型决策的不确定性」
参量化成一个 `noise_rate` 参数，以三种方式偏离理想路径：

| 偏离形态 | 对应真实模型的哪类失误 |
| :--- | :--- |
| 越序（未取数据先问策略） | 参数不全就调用工具，靠工具报错才发现 |
| 提前写（跳过策略校验直接下单） | 不可逆操作在验证之前发生 |
| 重复（把调过的工具再调一次） | 忘了工具已经返回过结果，因为结果没回填进上下文 |

这样做的目的是**让方差可比较**——不引入噪声，就没法量化「约束把方差压掉了多少」。
真实模型的行为要复杂得多，但结论方向一致：可靠性由运行时提供，不由模型提供。

## 接真实模型

```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
export OPENAI_MODEL=glm-4-plus
python workflow_vs_agent.py --real
```

注意：接真实模型时 `workflow_vs_agent.py` 会走真实接口，输出不再可复现——
这本身就是第 01 期「temperature=0 不可复现」的现场验证。

## 建议的录像切分

| 段落 | 脚本 | 讲解重点 |
| :--- | :--- | :--- |
| 1. 同一任务写两遍 | `workflow_vs_agent.py` 实现 A / B | 并排看 trace，指出控制流归属的差别 |
| 2. 成本从哪来 | `workflow_vs_agent.py` 对比表 | 输入 token 涨 5.4 倍、成本涨 4.54 倍，推导「轮数 × 重发上下文」 |
| 3. 反例 | `workflow_vs_agent.py` 反例小节 | 用了 Function Calling 仍然不是 Agent |
| 4. 量化收敛 | `convergence.py` 两张表 | 76 次越序退款 → 加 Guardrails 归零 |
| 5. 反向判断题 | `convergence.py` 复盘小节 | 「什么情况下**不该**用 Agent」的四张牌 |

## 值得在视频里「改一改再跑」的参数

1. `convergence.py` → `NOISE_RATE` 从 0.25 改到 0.05 / 0.5，看正确率与干预次数怎么变
2. `convergence.py` → `TRIALS` 从 300 改到 50，观察统计波动
3. `convergence.py` → 把 `expected_next()` 里 `create_refund` 那一行注释掉，看 Guardrails 变成什么样（提示：Agent 就永远不会退款了）
