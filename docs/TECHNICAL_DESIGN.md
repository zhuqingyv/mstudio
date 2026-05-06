# mstudio 技术方案

> 版本：v0.2
> 日期：2026-05-06
> 状态：设计中
> 定位：纯本地运行的「数据 → LoRA」黑盒工具，mnemo 子模块

---

## 1. 项目概述

mstudio 是一个**高度特化的数据蒸馏与 LoRA 训练工具**。它只负责一件事：把用户本地的一堆原始数据，自动整理成高质量训练数据，并训练出可加载到本地 Ollama 的 LoRA adapter。

### 1.1 核心原则

- **黑盒化**：用户只感知「输入文件夹」和「输出 LoRA」。中间所有过程（分类、蒸馏、训练、调参）全部被内部吃掉。
- **纯本地**：M2 Ultra 64GB 上完成全部流程，不上云。
- **零 schema 依赖**：不感知上游数据结构。无论原始数据是 `.txt`、`.md`、`.jsonl`、`.db` 还是混合格式，均由内部 Agent 自行理解。
- **专用 Agent**：Agent 不是通用助手，而是「数据蒸馏工蚁」——没有代码执行、没有 Git、没有浏览器、没有 MCP 工具。全部提示词和内部循环只围绕一个使命：高效分类和整理训练数据。
- **因果极简**：用户数据好 → LoRA 效果好；用户数据差 → LoRA 效果差。中间不加任何需要用户决策的环节。

### 1.2 用户视角

```
用户操作：
  $ mstudio train ./my_raw_data/

用户等待...

用户获得：
  ./output/my_raw_data_lora/
  ├── adapter_model.safetensors
  ├── adapter_config.json
  └── Modelfile
```

用户执行 `ollama create my-lora -f ./output/my_raw_data_lora/Modelfile`，即可使用。

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                         mstudio                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐   ┌─────────────────────┐   ┌─────────────┐│
│  │   输入层    │ → │    蒸馏 Agent       │ → │  训练引擎   ││
│  │  Input      │   │  (专用蒸馏工蚁)      │   │  Trainer    ││
│  │  Gateway    │   │                     │   │             ││
│  └─────────────┘   └─────────────────────┘   └─────────────┘│
│         │                   │                      │        │
│         └───────────────────┴──────────────────────┘        │
│                            │                                │
│                            ▼                                │
│                     ┌─────────────┐                         │
│                     │  输出层     │                         │
│                     │  Output     │                         │
│                     │  (LoRA)     │                         │
│                     └─────────────┘                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 输入层（Input Gateway）

- 接收一个本地文件夹路径
- 递归遍历，识别文件类型和编码
- 提取原始文本，统一为内部文档流
- **不解析字段含义，不校验 schema**

### 2.2 蒸馏 Agent（Distillation Agent）

专用 Agent，内部 5 阶段流水线。所有阶段通过标准 JSON 对象传递数据，可循环反馈。

### 2.3 训练引擎（Training Engine）

- **主力**：MLX / mlx-lm（Apple Silicon 原生，内存效率高）
- **输出**：Safetensors adapter，可直接被 Ollama 加载

### 2.4 输出层（Output）

- LoRA adapter 权重
- Ollama Modelfile 模板
- 轻量元数据（样本数、训练时长，供内部诊断，不暴露给用户）

---

## 3. 核心模块设计

### 3.1 蒸馏 Agent（Distillation Agent）

Agent 是 mstudio 的心脏。它不是一个通用 Agent，而是一个**高度特化的数据整理流水线**。所有提示词、循环逻辑、质量判断都围绕「把原始数据变成好训练数据」这一件事。

#### 3.1.1 阶段 ①：发现与解析（Discovery & Parsing）

**职责**：把文件系统里的原始数据，变成统一的文本流。

- 递归遍历输入目录
- 识别文件类型：`.txt`、`.md`、`.jsonl`、`.json`、`.csv`、`.db`、`.sqlite` 等
- 文本文件直接读取；数据库文件用启发式或 LLM 推理提取文本字段
- 统一编码为 UTF-8

**输出**：
```json
{"source": "path/to/file.md", "raw_text": "...", "file_type": "markdown"}
```

#### 3.1.2 阶段 ②：语义理解（Semantic Understanding）

**职责**：让本地 LLM 阅读每份文档，输出语义标签。

模型对每份文档输出：
- `summary`：内容摘要
- `content_types`：**多标签分类**，如 `["knowledge", "anti_pattern"]`
- `quality_score`：信息密度与可训练价值（0-1）
- `key_entities`：关键实体
- `confidence`：模型对自身判断的确信度

**关键设计**：
- `content_types` 不是枚举硬编码，而是模型自由输出的标签字符串
- Agent 后续按标签聚类，新增类型无需改代码，只需新增 prompt 模板
- 低 confidence 的文档进入「待人工确认」队列（但用户不感知，内部降级处理）

#### 3.1.3 阶段 ③：蒸馏与重构（Distillation & Reconstruction）

**职责**：把带标签的原始文档，改写为 instruction-following 训练样本。

**Prompt 模板库**：
根据 `content_types` 匹配对应的重构模板（不是代码 if-else，而是配置化的 prompt）：

| 标签示例 | 重构策略 |
|---------|---------|
| `knowledge` | 改写为问答对 `{prompt, response}` |
| `workflow` | 改写为指令遵循 + CoT `{prompt, response_with_steps}` |
| `preference` | 改写为风格示例对 `{scenario, preferred_output}` |
| `anti_pattern` | 改写为反向指令 `{bad_example, correction}` |
| `tool_usage` | 改写为工具调用模板 `{goal, tool_call_json}` |
| `relation` | 改写为上下文关联对 `{context, related_info}` |

**多标签处理**：
- 一份文档如果有多个标签，可生成多个训练样本（每个标签一个）
- 或生成一个融合多维度信息的复合样本（由模板决定）

**输出**：标准训练样本
```json
{"messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

#### 3.1.4 阶段 ④：质量控制（Quality Control）

**职责**：清洗、去重、平衡、过滤。

- **语义去重**：用本地 embedding 模型计算样本相似度，合并或删除冗余
- **质量过滤**：过滤过短、过泛、信息密度低于阈值的样本
- **矛盾检测**：找出 instruction 相似但 response 矛盾的样本，标记内部告警
- **分布平衡**：确保各 content_type 的样本量不过度倾斜（不追求绝对平均，但防止某一类垄断）

**反馈回路**：
- 如果某类样本质量整体过低，Agent 可触发「回退」：回到阶段 ③ 用更保守的模板重新生成
- 如果整体数据量过少，Agent 可触发「增强」：用本地 LLM 对高价值样本做 paraphrase 扩充

#### 3.1.5 阶段 ⑤：训练准备（Training Preparation）

**职责**：输出训练引擎可直接消费的格式。

- 按 content_type 拆分（为后续「独立 LoRA / 混合 LoRA」实验预留能力）
- 分割 train / val
- 输出 MLX 标准格式 `.jsonl`
- 生成内部诊断文件 `meta.json`（样本统计、类别分布、建议超参）

**输出**：
```
output/
├── train.jsonl
├── val.jsonl
├── meta.json
└── by_category/          # 预留：按类别拆分
    ├── knowledge.jsonl
    ├── preference.jsonl
    └── ...
```

### 3.2 训练引擎（Training Engine）

#### 3.2.1 框架选择

**主力：MLX / mlx-lm**
- Apple Silicon 原生，Unified Memory 零拷贝
- 64GB 内存可舒适训练 14B-32B 模型的 QLoRA
- 输出 `.safetensors` adapter，Ollama 可直接加载

**不选 PEFT+MPS 的理由**：
- MPS 上 fp16 不可用（NaN），bf16 硬件不支持，只能 fp32
- GPU 利用率低（10-15%），训练速度慢 2-5 倍
- bitsandbytes 完全不支持 Mac
- 作为主力框架维护成本过高

#### 3.2.2 基座模型

前期锁定一个**默认基座**（建议 Qwen2.5-14B-Instruct 或 Mistral-7B-Instruct），原因：
- mstudio 是黑盒工具，用户不选择基座
- 多基座支持会增加复杂度，后期再扩展
- Ollama 需预装对应基座模型，默认基座应在 Ollama 生态中下载量高、兼容性好

#### 3.2.3 超参策略

超参不暴露给用户，内部根据数据量、样本长度、类别分布自动推荐：
- 数据量 < 50 条 → rank 4, iters 500
- 数据量 50-500 条 → rank 8-16, iters 1000-3000
- 数据量 > 500 条 → rank 16-32, iters 3000-5000
- 学习率固定 1e-5 ~ 2e-5（MLX 默认值附近）

### 3.3 输出层与 Ollama 集成

训练完成后自动生成 Ollama Modelfile：

```dockerfile
FROM qwen2.5:14b
ADAPTER ./adapters
SYSTEM ""
```

用户执行：
```bash
ollama create my-behavior -f ./output/Modelfile
ollama run my-behavior
```

---

## 4. 极简接口设计

### 4.1 CLI 接口（内部使用或高级用户）

```bash
# 一键训练
mstudio train ./path/to/raw_data/

# 指定输出目录
mstudio train ./path/to/raw_data/ --output ./my_lora/

# 查看内部诊断（可选，默认不显示）
mstudio train ./path/to/raw_data/ --verbose
```

### 4.2 API 接口（为 mnemo 集成预留）

```python
from mstudio import Trainer

result = Trainer.train(
    input_dir="/path/to/raw_data",
    output_dir="/path/to/output"
)
# result.adapter_path -> LoRA 权重路径
# result.modelfile_path -> Ollama Modelfile 路径
```

### 4.3 用户零感知原则

- 无配置面板
- 无中间审查界面
- 无参数调节选项
- 失败时只给出极简原因：「数据量过少」、「数据质量不足」、「训练失败，请检查原始数据」

---

## 5. 多 LoRA 策略（预留，待实验）

当前先按「混合训练一个 LoRA」实现，但架构预留独立扩展能力：

| 策略 | 实现复杂度 | 用户感知 | 预留接口 |
|------|-----------|---------|---------|
| **混合 LoRA** | 低 | 无 | 当前默认 |
| **按类别独立 LoRA** | 中 | 无（内部管理） | `by_category/` 目录已预留 |
| **独立训练 + Merge** | 中 | 无 | 训练后 merge script |
| **运行时 Compose** | 高 | 无 | 需绕过 Ollama，暂不实现 |

**实验计划**：
- Phase 1：混合 LoRA，验证端到端可行性
- Phase 2：对比实验——同样的数据，混合 LoRA vs 独立 LoRA + Merge，评估效果差异
- Phase 3：根据实验结果决定默认策略

---

## 6. 推荐目录结构

```
mstudio/
├── docs/
│   ├── TECHNICAL_DESIGN.md
│   └── RESEARCH.md
├── src/
│   ├── __init__.py
│   ├── cli.py                    # 极简 CLI 入口
│   ├── trainer.py                # 对外 API
│   ├── agent/                    # 蒸馏 Agent（核心）
│   │   ├── __init__.py
│   │   ├── pipeline.py           # 5 阶段流水线编排
│   │   ├── discovery.py          # 阶段 ①：发现解析
│   │   ├── understanding.py      # 阶段 ②：语义理解
│   │   ├── reconstruction.py     # 阶段 ③：蒸馏重构
│   │   ├── quality.py            # 阶段 ④：质量控制
│   │   ├── preparation.py        # 阶段 ⑤：训练准备
│   │   └── prompts/              # Prompt 模板库
│   │       ├── understand.txt
│   │       ├── reconstruct_knowledge.txt
│   │       ├── reconstruct_preference.txt
│   │       └── ...
│   ├── engine/                   # 训练引擎
│   │   ├── __init__.py
│   │   ├── mlx_trainer.py        # MLX 训练封装
│   │   └── config_builder.py     # 超参自动推荐
│   ├── models/                   # 本地模型调用抽象
│   │   ├── __init__.py
│   │   ├── llm_client.py         # 统一 LLM 调用接口
│   │   └── embedding_client.py   # 统一 Embedding 调用接口
│   └── output/                   # 输出层
│       ├── builder.py            # Modelfile 生成
│       └── packer.py             # 打包输出
├── configs/
│   └── default.yaml              # 内部默认配置（用户不感知）
├── tests/
├── pyproject.toml
└── README.md
```

---

## 7. 本地开发路径（M2 Ultra）

### 7.1 环境准备

```bash
# 安装 MLX
pip install mlx-lm

# 安装 Ollama（用于推理和 embedding）
brew install ollama

# 拉取默认基座模型
ollama pull qwen2.5:14b

# 拉取 embedding 模型
ollama pull qwen3-embedding:0.6b
```

### 7.2 验证里程碑

| 阶段 | 目标 | 验证标准 |
|------|------|---------|
| **Week 1** | 跑通蒸馏 Agent 单阶段 | 输入 10 个 `.txt` 文件，Agent 输出结构化训练样本 |
| **Week 2** | 跑通完整 5 阶段流水线 | 输入任意混合格式文件夹，输出 `train.jsonl` + `meta.json` |
| **Week 3** | 跑通 MLX 训练 | 用 Agent 产出的数据训练一个 7B 模型的 LoRA，Ollama 加载验证 |
| **Week 4** | 端到端黑盒验证 | 输入文件夹 → 等待 → Ollama 运行新模型，行为有可见差异 |

---

## 8. 关键未解问题（内部待验证）

以下问题不影响架构设计，但需要在实验中验证：

1. **多标签分类的准确性**：本地 7B/14B 模型对中文/英文混合数据的分类能力是否足够？
2. **混合 vs 独立 LoRA**：同样的多标签数据，训成一个 LoRA 还是按标签拆分成多个再 merge，哪个效果更好？
3. **数据量下限**：Agent 蒸馏出的数据最少需要多少条，才能训出「可感知差异」的 LoRA？
4. **回退与增强策略**：阶段 ④ 的反馈回路（回退重蒸馏、paraphrase 扩充）是否真的能提升最终效果？
5. **Embedding 去重阈值**：语义相似度多高的样本应该被合并？不同内容类型的阈值是否不同？
