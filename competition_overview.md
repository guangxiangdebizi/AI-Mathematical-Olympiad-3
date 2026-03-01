# AIMO3 - AI Mathematical Olympiad Progress Prize 3 比赛全面概述

## 一、比赛基本信息

| 项目 | 详情 |
|------|------|
| **比赛名称** | AI Mathematical Olympiad - Progress Prize 3 (AIMO3) |
| **组织方** | XTX Markets (量化交易公司) |
| **平台** | Kaggle |
| **奖金池** | $2,200,000 (220万美元) |
| **启动时间** | 2025年11月 |
| **截止日期** | **2026年4月15日 23:59 UTC** |
| **参赛队伍** | 2711+ 支队伍 |
| **比赛类型** | Code Competition (代码竞赛) |
| **硬件** | H100 GPU (Kaggle 提供) |
| **时间限制** | 每次提交 **9小时** 运行时间 |
| **比赛链接** | https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3 |

## 二、比赛目标

**核心目标**: 构建能够解决奥林匹克数学竞赛难度题目的 AI 模型（开源/开放权重模型）。

比赛给你一组数学题（LaTeX格式），你的模型需要读懂题目、进行数学推理、然后给出正确的整数答案。

## 三、题目详情

- **题目数量**: 110道原创数学题
- **难度范围**: 从国家奥林匹克竞赛水平到 IMO（国际数学奥林匹克）水平
- **覆盖领域**:
  - 📐 代数 (Algebra)
  - 🔢 组合数学 (Combinatorics)
  - 📏 几何 (Geometry)
  - 🧮 数论 (Number Theory)
- **特点**: 所有题目都是原创的，零数据污染风险
- **答案格式**: 5位整数（比AIMO2的3位整数更长，大幅降低了猜测的可能性）
  - 答案需要对结果取模 100000
  - 例如：如果计算结果是 265521，提交 65521
  - 如果计算结果是 -900，提交 99100 (即 -900 mod 100000)

## 四、提交格式与评估

### 提交方式
- **Code Competition**: 你需要提交一个 Kaggle Notebook
- Notebook 在 Kaggle 的 H100 GPU 上运行
- 模型读取测试题目 → 推理 → 输出预测答案

### 数据格式
- **输入**: CSV文件，包含 `id` 和 `problem`（LaTeX格式的数学题）
  - 题目逐个发送给推理服务器（每次一题）
  - 题目顺序在 public leaderboard 阶段会被随机打乱
- **输出**: CSV文件，包含 `id` 和 `answer`（整数，对 10^5 取模）
- **提交架构**: gRPC Gateway/InferenceServer 模式
  - Gateway 读取题目 → 逐题发送 → InferenceServer 返回答案

### 评估指标
- **Accuracy（准确率）**: 预测答案与真实答案精确匹配的比例
- 排名由 private leaderboard 上的准确率决定

### 运行约束
- 使用 H100 GPU（相比AIMO2的T4 GPU，算力翻倍，支持 bfloat16）
- 必须使用**开源/开放权重**的模型
- 不允许联网（submission notebook 无网络访问）
- **9小时**运行时间限制（从源码 `set_response_timeout_seconds(60 * 60 * 9)` 确认）
- 平均每题约 **4.9分钟**（9小时 / 110题）的推理时间

## 五、奖项结构

### 主赛奖金
基于 private leaderboard 排名分配（总计 $2.2M）

### 额外奖项 (Extra Prizes)
AIMO3 首次引入额外奖项：
1. **最难问题奖 (Hardest Problem Prize)**: 奖励解出其他模型最难解的那道题的最佳模型
2. **数学语料库奖 (MathCorpus Prize)**: 奖励发布有助于社区的新数据集
3. **技术报告奖 (Write-up Prizes)**: 奖励最佳技术方案说明
4. **最长领先奖 (Longest Leader Prize)**: 奖励在 public leaderboard 上领先时间最长的团队

## 六、AIMO 历史与背景

| 版本 | 获胜者 | 难度 | 硬件 | 奖金 |
|------|--------|------|------|------|
| AIMO1 | Project Numina (HuggingFace) | 高中竞赛级 | T4 GPU | $1.048M |
| AIMO2 | NemoSkills (NVIDIA) | 国家奥赛级 | T4 GPU | $2.1M |
| AIMO3 | 进行中 | 国家奥赛→IMO级 | H100 GPU | $2.2M |

## 七、历届获胜方案分析

### AIMO1 获胜方案 — Project Numina
1. **基座模型**: DeepSeekMath-Base 7B
2. **训练数据**:
   - NuminaMath-CoT: ~860K 数学题（CoT格式解答）
   - NuminaMath-TIR: ~70K 题（工具集成推理格式，用GPT-4生成）
3. **推理方法**: SC-TIR (Self-Consistency Tool-Integrated Reasoning)
   - 生成多个候选解答
   - 通过代码执行验证
   - 投票选出最一致的答案
4. **训练**: 8×H100 GPU，10小时

### AIMO2 获胜方案 — NemoSkills (NVIDIA)
1. **核心技术**: Nemotron-Math + OpenMathReasoning 数据集
2. **训练数据**:
   - OpenMathReasoning: 306K 唯一数学题，568万解答
   - 三种推理模式：CoT (320万)、TIR (170万)、GenSelect (56.6万)
3. **关键方法**:
   - Chain-of-Thought (CoT): 逐步逻辑推理
   - Tool-Integrated Reasoning (TIR): 集成 Python 代码辅助计算
   - Generation Selection (GenSelect): 生成多个候选解，选最优
4. **成果**: AIME 2024 & 2025 上 100% maj@16 准确率

---

# 八、比赛打法策略

## 第一步：环境准备与基线搭建 (Week 1-2)

### 1.1 环境配置
- [ ] 配置 Kaggle API，下载比赛数据
- [ ] 了解 Kaggle Notebook 的 H100 GPU 环境
- [ ] 研究 submission demo notebook
- [ ] 确认提交格式和时间限制

### 1.2 基线方案
- [ ] 使用现成的开放权重模型（如 Qwen2.5-Math、DeepSeek-Math）跑一个 baseline
- [ ] 了解 vLLM / SGLang 等推理框架在 H100 上的部署
- [ ] 提交第一个 baseline，了解整个流程

## 第二步：模型选择与优化 (Week 2-4)

### 2.1 候选模型
基于 H100 GPU 的算力，可以考虑的开放权重模型：
- **GPT-OSS-120B**: AIMO3 官方提到的可用模型
- **Qwen3-Next / Qwen2.5-Math-72B**: 阿里的数学推理模型
- **DeepSeek-Math / DeepSeek-V3**: 深度求索的数学模型
- **Llama 3.3 / Llama 4**: Meta 的大模型
- **Nemotron-Math**: NVIDIA AIMO2 获胜模型

### 2.2 推理策略
- **Tool-Integrated Reasoning (TIR)**: 让模型生成 Python 代码，执行计算
  - 这是历届获胜方案的核心——模型不只是"想"，还能"算"
  - 用 Python/SymPy 处理复杂数学计算
- **Self-Consistency (SC)**: 多次采样，投票选答案
  - 对同一题生成多个解答
  - 取出现次数最多的答案
- **Agentic Solving**: 多步骤推理代理
  - 模型先分析题目类型
  - 选择合适的推理策略
  - 如果第一次解错，能自我纠正

## 第三步：数据与微调 (Week 3-6)

### 3.1 训练数据收集
- OpenMathReasoning 数据集 (NVIDIA 开源)
- NuminaMath 数据集 (HuggingFace)
- AOPS (Art of Problem Solving) 论坛数据
- 各国数学奥赛真题
- 自己用强模型生成的合成数据

### 3.2 微调策略
- **Stage 1**: CoT SFT — 用数学题+文字解答微调
- **Stage 2**: TIR SFT — 用数学题+代码解答微调
- 可以申请比赛提供的 128×H100 GPU 集群来做微调

## 第四步：推理优化 (Week 5-8)

### 4.1 推理加速
- 使用 vLLM 或 SGLang 进行高效推理
- FP8/INT8 量化减少显存占用
- KV-Cache 优化
- Speculative Decoding

### 4.2 答案提取与验证
- 精确的答案解析（从模型输出中提取整数答案）
- 代码执行沙箱（安全执行模型生成的 Python 代码）
- 答案一致性校验

## 第五步：集成与提交 (Week 7-10)

### 5.1 方案集成
- 多模型 ensemble
- 多种推理策略组合
- 根据题目难度自适应分配推理时间

### 5.2 提交优化
- 确保 notebook 在时间限制内完成
- 合理分配每题的推理时间
- 错误处理和 fallback 机制

---

# 九、关键技术栈

| 组件 | 推荐工具 |
|------|----------|
| **推理框架** | vLLM, SGLang, TensorRT-LLM |
| **训练框架** | PyTorch, TRL, DeepSpeed, Axolotl |
| **数学库** | SymPy, SageMath, NumPy |
| **模型格式** | GGUF, AWQ, GPTQ (量化) |
| **代码执行** | 沙箱化 Python 执行环境 |
| **数据处理** | HuggingFace Datasets, Pandas |

# 十、关键资源链接

- **比赛页面**: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3
- **AIMO官网**: https://aimoprize.com/
- **AIMO1 方案 (Numina)**: https://github.com/project-numina/aimo-progress-prize
- **AIMO2 方案论文**: https://arxiv.org/abs/2504.16891
- **OpenMathReasoning数据集**: https://huggingface.co/datasets/nvidia/OpenMathReasoning
- **NuminaMath数据集**: https://huggingface.co/collections/AI-MO/numinamath-6697df380293bcfdbc1d978c
- **Submission Demo**: https://www.kaggle.com/code/ryanholbrook/aimo-3-submission-demo
