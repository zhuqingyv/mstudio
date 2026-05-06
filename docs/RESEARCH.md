# mstudio 技术深度调研

> 范围：文本大模型 LoRA / Adapter 的本地训练、多适配器组合、以及记忆数据蒸馏策略
> 目标硬件：Apple Mac Studio (M2 Ultra, 64GB Unified Memory)
> 约束：纯本地运行，前期不上云；最终作为 mnemo 子模块存在
> 日期：2026-05-06
> 状态：调研中，未定论

---

## 1. 调研背景与边界

### 1.1 核心定位

mstudio 不是独立的 C 端产品，而是 mnemo 记忆内化流水线的「训练工厂」模块。其职责是将 mnemo 本地 SQLite 中沉淀的、经过验证的文本记忆，蒸馏为可直接加载到本地大模型上的 LoRA / Adapter 权重。

### 1.2 用户视角的极简接口

从用户侧看，只暴露两个信息：
1. **「我的本地记忆有多少可用于训练」** —— 系统从 mnemo 数据库自动统计
2. **「我要训练一个什么模型」** —— 选择意图/场景，而非选择技术参数

所有「框架选择、Rank 设定、学习率、数据格式转换、训练配置」全部被工具内部吃掉。

### 1.3 硬件边界

| 硬件 | Apple Mac Studio (M2 Ultra) |
|------|----------------------------|
| CPU/GPU | M2 Ultra (24 核 CPU + 60/76 核 GPU) |
| 内存 | 64GB Unified Memory |
| 训练后端 | Metal Performance Shaders (MPS) / MLX |
| 推理后端 | Ollama (本地) |

**关键约束**：无 NVIDIA GPU，无 CUDA，所有训练与推理必须在 Apple Silicon 本地闭环。

---

## 2. 训练框架调研

### 2.1 候选框架概览

| 框架 | 维护方 | Mac 支持 | 训练方式 | 输出格式 | 生态成熟度 |
|------|--------|---------|---------|---------|-----------|
| **MLX / mlx-lm** | Apple (ml-explore) | ⭐⭐⭐⭐⭐ 原生一等公民 | LoRA / QLoRA | `.safetensors` / `.npz` | 快速增长 |
| **PEFT + Transformers + PyTorch MPS** | HuggingFace | ⭐⭐⭐ 能跑但问题多 | LoRA / QLoRA | `.safetensors` | 最成熟 |
| **llama.cpp** | ggerganov | ⭐⭐⭐⭐ 推理强 | 量化后 LoRA (有限) | GGUF | 推理为主 |
| **LLaMA-Factory** | hiyouga | ⭐⭐ 主要为 CUDA | LoRA / 全量 | `.safetensors` | 极成熟 |
| **axolotl** | OpenAccess AI | ⭐⭐ 主要为 CUDA | LoRA / QLoRA | `.safetensors` | 成熟 |

### 2.2 MLX 详细评估

**核心优势**
- Apple 官方团队维护，专为 Unified Memory 架构设计，零 CPU-GPU 拷贝开销
- 内存效率极高：Mistral-7B QLoRA 仅需约 8GB 内存，64GB 可轻松训练 30B+ 级别模型
- 训练速度：M2 Ultra 上 Llama-7B LoRA 约 475 tokens/s（mlx-examples 实测数据）
- 支持格式：`chat`（多轮对话 JSON）和 `text`（completion）两种数据格式
- 支持 adapter 断点续训、fuse（合并到基座）、量化模型上直接训练

**关键限制**
- 模型生态依赖 HuggingFace 的 MLX 社区转换（`mlx-community` 组织）
- 不支持所有架构（主要支持 Llama、Mistral、Qwen、Gemma、Phi 等）
- 训练功能比 PEFT 精简，部分高级技巧（如 multiple LoRA、Dora）需自研

**数据格式示例**
```jsonl
{"messages": [
  {"role": "user", "content": "在这个项目里怎么部署合约？"},
  {"role": "assistant", "content": "使用 deploy --chain 标志..."}
]}
```

### 2.3 PEFT + PyTorch MPS 详细评估

**核心问题**
- **FP16 不可用**：`mixed_precision="fp16"` 在 MPS 上会直接产生 NaN loss，因为 AMP 梯度缩放未在 MPS 上实现
- **BF16 不可用**：Apple Silicon 硬件无 BF16 指令，PyTorch 会软件模拟，导致精度灾难
- **必须使用 FP32**：`mixed_precision="no"`，训练速度下降约 30-50%
- **bitsandbytes 完全不支持**：8-bit Adam、8-bit 量化优化器均只能在 CUDA 上运行
- **GPU 利用率低**：实测 MPS 后端 GPU 利用率仅 10-15%，MLX 约为其 2 倍以上
- **版本敏感**：torch>=2.3.0, transformers>=4.45.0, peft>=0.18.0 才相对稳定

**结论**：PEFT+MPS 可以作为「兼容性兜底」方案（支持更多模型架构），但不建议作为 Mac 本地主力训练框架。

### 2.4 训练框架决策矩阵

| 维度 | MLX (推荐调研) | PEFT+MPS (备选) |
|------|---------------|----------------|
| 内存效率 | 极高（64GB 可训 30B+） | 中（64GB 可训 7B-13B） |
| 训练速度 | 快（M2 Ultra 475 tok/s @7B） | 慢（FP32 惩罚 + 低利用率） |
| 稳定性 | 高（Apple 原生） | 中（版本敏感、算子 fallback） |
| 模型覆盖 | 中（主流 Decoder-only） | 高（几乎所有 HF 模型） |
| 与 Ollama 衔接 | 需 fuse 或格式转换 | Safetensors 直接兼容 |
| 维护成本 | 低 | 高 |

---

## 3. 基座模型与推理加载调研

### 3.1 本地推理端现状

mnemo 当前已集成 Ollama（用于 embedding 和潜在推理）。mstudio 训练出的 LoRA 必须能被 Ollama 加载，否则无法形成闭环。

### 3.2 Ollama 的 LoRA 支持能力

**已确认支持的功能（截至 Ollama 0.5.x）**
- Modelfile `ADAPTER` 指令可加载两种格式：
  1. **Safetensors adapter 目录**：直接指向训练输出的 adapter 文件夹（含 adapter_model.safetensors + adapter_config.json）
  2. **GGUF adapter 文件**：通过 llama.cpp 的 `convert_lora_to_gguf.py` 转换后的单文件
- 支持架构：Llama（含 Llama 2/3/3.1/3.2）、Mistral（含 Mixtral）、Gemma（含 Gemma 2）
- `ADAPTER .` 支持相对路径，可指向当前目录
- Adapter 通过 `ollama create` 后永久合并到模型变体中，运行时无额外开销

**关键限制**
- Base model 必须和训练时的基座完全一致，否则行为不可预测
- 不支持动态切换：一个 Ollama 模型实例只能绑定固定的 adapter 集合
- 多 adapter 并发需要创建多个模型变体，内存中基座权重无法共享（从 Ollama 视角）
- 不支持运行时热加载：需要重新 `ollama create` 模型变体

**MLX → Ollama 的衔接路径**
```
MLX 训练输出 (adapters/)
    ├── adapter_model.safetensors
    └── adapter_config.json
         ↓
Ollama Modelfile:
    FROM qwen2.5:14b
    ADAPTER ./adapters
    SYSTEM "..."
         ↓
ollama create mstudio-preference -f Modelfile
ollama run mstudio-preference
```

### 3.3 可选基座模型分析

结合 mnemo 的 Agent 场景（文本/代码/推理）和 M2 Ultra 64GB 内存：

| 模型 | 大小 | M2 Ultra 推理 | M2 Ultra 训练 (QLoRA) | Ollama 支持 | 适用场景 |
|------|------|--------------|----------------------|------------|---------|
| Qwen2.5-7B-Instruct | 7B | ✅ 极快 | ✅ 轻松 | ✅ | 通用中文 Agent |
| Qwen2.5-14B-Instruct | 14B | ✅ 快 | ✅ 可行 | ✅ | 高质量中文推理 |
| Qwen2.5-32B-Instruct (4bit) | 32B | ✅ 中等 | ⚠️ 较重但可能 | ✅ | 复杂推理 |
| Mistral-7B-Instruct | 7B | ✅ 极快 | ✅ 轻松 | ✅ | 通用英文 Agent |
| Llama-3.1-8B-Instruct | 8B | ✅ 极快 | ✅ 轻松 | ✅ | 英文/代码 |
| Llama-3.3-70B (4bit) | 70B | ⚠️ 边缘 | ❌ 不可能 | ✅ | 超大模型推理 |
| Gemma-2-9B | 9B | ✅ 快 | ✅ 轻松 | ✅ | Google 生态 |

**关键发现**：64GB Unified Memory 在 MLX QLoRA 下可以训练 14B-32B 级别的模型（取决于量化精度和序列长度），推理可以跑 70B 的 4bit 量化版，但训练 70B 几乎不可能。

---

## 4. 多 LoRA 组合与加载机制调研

### 4.1 需求场景

mnemo 的记忆分类多样（用户偏好、工具调用、避坑知识、工作流程、人物关系等）。用户可能希望：
- 不同类别的记忆训练成**独立的 LoRA**
- 一个基座模型同时加载**多个 LoRA**
- 各 LoRA 之间可独立更新、版本管理

### 4.2 技术路线对比

| 方案 | 原理 | 训练端复杂度 | 推理端复杂度 | 灵活性 | 性能开销 |
|------|------|------------|------------|--------|---------|
| **权重合并 (Merge/Fuse)** | 将多个 LoRA 的 B×A 矩阵线性相加，生成一个等效 LoRA | 低 | 极低（合并后就是一个 LoRA） | 差（合并后不可独立更新） | 无 |
| **顺序加载切换 (Switch)** | 推理时动态加载/卸载不同 LoRA，同一时刻只有一个生效 | 低 | 低 | 中（需管理加载状态） | 低（切换时有加载延迟） |
| **并发组合 (Compose)** | 多个 LoRA 的梯度同时参与前向/反向传播 | 低 | 高 | 高 | 线性增长（N 个 LoRA ≈ N 倍计算） |
| **任务路由 (Task Routing)** | 根据输入特征（如 prompt 前缀）路由到不同 LoRA | 中 | 中 | 高 | 中 |
| **MoE 式门控 (MeteoRA/DLP-LoRA)** | 训练一个轻量门控网络，动态决定每个 token 用哪些 LoRA | 极高 | 极高 | 极高 | 高（需训练 gate） |

### 4.3 学术界方案详情

**LoRAHub / ZipLoRA**
- 用少量样本学习一个系数矩阵，将多个 LoRA 线性融合为一个统一 LoRA
- 优点：融合后推理成本不变
- 缺点：融合过程需要额外的 few-shot 数据来调系数，且会丢失独立更新能力

**LoRA Switch / LoRA Composite (Zhong et al., 2024)**
- Switch：在不同推理步骤（layer 或 timestep）交替激活不同 LoRA
- Composite：在每个步骤聚合多个 LoRA 的输出
- 优点：保留原始权重，灵活组合
- 缺点：需要修改推理引擎（Ollama 目前不支持）

**DLP-LoRA (Dynamic Lightweight Plugin)**
- 引入一个 5M 参数的 mini-MLP 作为插件，基于句子级别动态融合多个 LoRA
- 优点：上下文感知的动态组合
- 缺点：需要额外训练 gate 网络，复杂度高

**vLLM / S-LoRA / Punica**
- 服务端多租户场景下的多 LoRA 并发批处理
- 使用专门的 Batched GEMM kernel 减少开销
- 仅适用于服务端 GPU 部署，与本地 Ollama 无关

### 4.4 对 mstudio 的启示

| 需求 | 推荐方案 | 理由 |
|------|---------|------|
| 追求极简、用户无感知 | **Merge** | 多个小 LoRA 合并成一个，Ollama 直接加载 |
| 需要独立更新、按需切换 | **Switch + Modelfile 变体** | 为不同组合创建不同的 Ollama 模型变体 |
| 精细控制、多维度同时影响 | **Compose（需自研推理端）** | 绕过 Ollama，直接用 MLX 或 Transformers 加载多 LoRA |

**关键矛盾**：Ollama 当前不支持运行时多 LoRA 并发。如果 mstudio 要走「一个模型同时受多个 LoRA 影响」的路线，可能需要：
1. 在训练前将多个记忆类别的数据混合训练成一个 LoRA（最简单）
2. 在导出前将多个独立 LoRA merge 成一个（折中）
3. 绕过 Ollama，自建基于 MLX 的本地推理端（最灵活但工程量大）

---

## 5. 记忆数据分类与蒸馏策略调研

### 5.1 mnemo 记忆结构回顾

mnemo 的记忆条目核心字段：
- `claim_type`: fact / decision / procedure / hypothesis
- `scope`: global / project / session
- `title`, `summary`, `body`: 文本内容
- `tags`: 分类标签
- `feedback`: helpful / misleading / outdated
- `lifecycle`: active / stale / superseded / archived

### 5.2 记忆到训练数据的转换层级

从原始记忆到训练数据，存在多个加工层级：

```
[原始记忆条目] → [筛选与质量评估] → [语义聚类/去重] → [教学式重构] → [指令微调数据]
```

**层级 1：直接拼接（浅层）**
- 把记忆的 title + body 直接作为文本语料，做 Continual Pre-training
- 优点：简单，保留原始知识密度
- 缺点：模型学会的是「续写」，而非「按用户习惯回答」

**层级 2：指令模板化（中层）**
- 根据记忆类型套用模板：
  - fact → "用户询问 X 时，应回答 Y"
  - procedure → "当需要执行 X 时，步骤为 Y"
  - preference → "用户在 X 场景下偏好 Y"
- 优点：结构清晰，训练目标明确
- 缺点：模板化可能导致过度泛化

**层级 3：对话式重构（深层）**
- 用 LLM Agent 将记忆扩展为模拟对话对：
  - 输入：记忆的 context
  - 输出：多轮 "用户问题 - 符合记忆的行为回答"
- 优点：最接近真实使用场景，训练效果最佳
- 缺点：蒸馏成本高，需要本地运行 VLM/LLM 生成数据

### 5.3 按记忆类别差异化的训练策略

用户提出的记忆分类及对应的 LoRA 设计思考：

| 记忆类别 | 特征 | 训练数据形式 | LoRA 规模建议 | 更新频率 |
|---------|------|------------|-------------|---------|
| **实用知识 (fact)** | 稳定的项目知识、技术细节 | 问答对 / 知识陈述 | 小（rank 4-8） | 低 |
| **踩坑记录 (anti-pattern)** | 负面知识、不应做的事 | 反向指令（"避免..." / "不要..."） | 中（rank 8-16） | 中 |
| **工具调用习惯 (tool)** | MCP 工具使用偏好、参数习惯 | 工具调用模板、参数填充示例 | 中（rank 8-16） | 高 |
| **用户偏好 (preference)** | 编码风格、沟通风格、输出格式 | 风格示例对、系统提示变体 | 中（rank 8-16） | 中 |
| **工作流程 (workflow)** | 多步骤任务的标准执行路径 | 多轮对话流程、CoT 示例 | 大（rank 16-32） | 低 |
| **人物关系 (relation)** | 项目成员、职责、沟通方式 | 角色描述、上下文关联 | 小（rank 4-8） | 低 |

**差异化逻辑**：
- 工具调用和偏好需要更频繁地影响模型行为，可能需要更高的 rank 来捕捉细微差别
- 知识类和关系类更稳定，小 rank 即可内化
- 踩坑记录的训练数据需要特殊处理（反向示例、拒绝采样）

### 5.4 数据蒸馏的质量控制

从记忆到训练数据的关键风险：
1. **矛盾记忆**：两条记忆给出相反建议 → 需要 contradiction detection 和优先级裁决
2. **过时记忆**：lifecycle 为 stale/superseded 的记忆不应进入训练集
3. **低反馈记忆**：feedback 为 misleading/outdated 的记忆应降权或排除
4. **样本不平衡**：某类记忆过多（如全是 fact），导致 LoRA 偏科
5. **隐私敏感**：session scope 的记忆可能包含临时敏感信息，需要脱敏或排除

---

## 6. 本地可行性评估（M2 Ultra 64GB）

### 6.1 训练能力边界

基于 MLX QLoRA 的实测数据推算：

| 基座模型 | 量化精度 | 预估训练内存 | M2 Ultra 可行性 | 预估 1000 steps 时间 |
|---------|---------|------------|---------------|-------------------|
| Qwen2.5-7B | 4-bit | ~6-8 GB | ✅ 非常轻松 | ~10-15 分钟 |
| Qwen2.5-14B | 4-bit | ~10-14 GB | ✅ 轻松 | ~20-30 分钟 |
| Qwen2.5-32B | 4-bit | ~20-28 GB | ✅ 可行 | ~40-60 分钟 |
| Llama-3.3-70B | 4-bit | ~40-50 GB | ⚠️ 边缘 | ~2-3 小时 |
| 任何模型 FP16 | 16-bit | 2-4x 于上表 | ❌ 不可行 | — |

**结论**：64GB Unified Memory 在 4-bit QLoRA 下，可以舒适地训练 14B-32B 级别的模型。这是绝大多数高质量开源指令模型的区间。

### 6.2 推理能力边界

| 基座模型 | 量化精度 | 预估推理内存 | Ollama 加载可行性 |
|---------|---------|------------|-----------------|
| Qwen2.5-14B | 4-bit | ~8-10 GB | ✅ 轻松 |
| Qwen2.5-32B | 4-bit | ~18-22 GB | ✅ 可行 |
| Llama-3.3-70B | 4-bit | ~40-45 GB | ⚠️ 刚够，但留给 LoRA 的余地小 |

### 6.3 数据蒸馏能力

数据蒸馏（记忆 → 训练数据）主要是文本处理和本地小模型推理：
- **文本处理/清洗**：纯 CPU 即可，无压力
- **语义去重/聚类**：可用 Ollama 运行的本地 embedding 模型（如 Qwen3-embedding-0.6B），极快
- **对话式重构**：需要本地 LLM 生成数据。Qwen2.5-7B/14B 本地生成速度足够，或可使用 Apple 的 MLX 版小模型
- **质量评估**：可由规则引擎 + 本地小模型完成

**结论**：数据蒸馏全流程在 M2 Ultra 上完全可行，瓶颈不在硬件，而在算法设计。

---

## 7. 关键未解问题（待决策）

以下问题调研阶段无法给出标准答案，需要结合 mnemo 的产品演进和实际验证来决定：

### 7.1 训练框架选择
- **MLX** 还是 **PEFT+MPS**？或两者并存（MLX 为主，PEFT 为兼容性兜底）？

### 7.2 多 LoRA 策略
- **方案 A**：每类记忆训练独立 LoRA，使用前先 merge 成一个，Ollama 加载
- **方案 B**：所有记忆混合训练成一个「通用行为 LoRA」，简单但丧失独立更新能力
- **方案 C**：绕过 Ollama，自建 MLX 推理端，支持运行时多 LoRA compose（工程量大）

### 7.3 基座模型锁定
- mstudio 是绑定一个固定基座（如 Qwen2.5-14B），还是让用户/系统根据场景选择？
- 如果 mnemo 的 Agent 最终通过 Ollama 调用模型，基座模型的选择受 Ollama 生态限制

### 7.4 训练触发与生命周期
- 是用户手动触发「一键训练」，还是 mnemo 后台自动在 idle time 反刍？
- LoRA 的版本如何管理？旧 LoRA 是归档还是直接覆盖？

### 7.5 评估闭环
- 训练出的 LoRA 如何验证有效？是否需要本地 A/B 测试框架（训练前 vs 训练后回答对比）？

---

## 8. 参考文献与资料

1. **MLX LoRA Examples** — ml-explore/mlx-examples: https://github.com/ml-explore/mlx-examples/tree/main/lora
2. **MLX LM Fine-Tuning Guide** — Level Up Coding, 2025: https://levelup.gitconnected.com/fine-tuning-llms-locally-using-mlx-lm
3. **MLX LoRA on M2 Ultra Benchmark** — marknorgren.com, 2025: https://marknorgren.com/posts/mlx-lora/
4. **ANE LoRA Training (Apple Neural Engine)** — jmanhype/ane-lora-training, 2026: https://github.com/jmanhype/ane-lora-training
5. **Ollama Modelfile Reference** — docs.ollama.com, 2026: https://docs.ollama.com/modelfile
6. **Ollama Import Adapters** — docs.ollama.com/import: https://docs.ollama.com/import
7. **PEFT MPS Low Performance Issue** — huggingface/peft #2041: https://github.com/huggingface/peft/issues/2041
8. **LoRA-Apple-Silicon-Guide** — mattmireles/gemma-tuner-multimodal, 2026: https://github.com/mattmireles/gemma-tuner-multimodal/blob/main/README/guides/apple-silicon/LoRA-Apple-Silicon-Guide.md
9. **DLP-LoRA: Dynamic Multi-LoRA Fusion** — arXiv:2410.01497, 2025
10. **Multi-LoRA Composition (MultLFG)** — arXiv:2505.20525, 2025
11. **LoRA Skill Composition / LoRAHub** — arXiv:2508.11624, 2025
12. **LLM Agent Memory Survey** — preprints.org, 2026: https://www.preprints.org/manuscript/202603.0359/v1
13. **Me-Agent: Personalized Mobile Agent** — arXiv:2601.20162, 2026
14. **Fine-Tuning Small Language Models with Low-Rank Adapters** — arXiv:2509.09689, 2025
15. **vLLM Multi-LoRA Concurrent Decoding** — vLLM Forums, 2025
