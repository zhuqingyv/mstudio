# mstudio

AI 驱动的文本大模型 LoRA 生成器。将原始数据蒸馏为高质量训练数据，一键训练出可加载到本地 Ollama 的 LoRA adapter。

> **定位**：mnemo 的子模块，非独立产品。纯本地运行，零配置黑盒工具。

---

## 核心流程

mstudio 只做一件事：把用户本地的一堆原始数据，变成行为化的 LoRA 权重。

```
原始数据文件夹 → 蒸馏 Agent → 训练引擎 → LoRA adapter
```

### 1. 数据蒸馏（Distillation）

专用蒸馏 Agent（非通用 Agent）逐文件处理：

- **发现解析**：遍历文件夹，识别 `.txt` `.md` `.jsonl` `.json` `.csv` `.db` 等格式，提取文本
- **语义理解**：本地 LLM 推理每份文档的内容类型（多标签，如 `knowledge` + `anti_pattern`）
- **蒸馏重构**：按内容类型将原始文本改写为 instruction-following 训练样本
- **质量控制**：过滤低质量样本，保留有效数据（不去重——重复出现 = 信息重要）
- **训练准备**：输出标准 `.jsonl` 格式，供 MLX 训练引擎消费

Agent 不硬编码任何 schema，不感知上游数据结构，所有分类和理解由本地模型推理动态完成。

### 2. LoRA 训练（Training）

- **训练框架**：MLX / mlx-lm（Apple Silicon 原生，64GB 内存可训 14B-32B 模型）
- **输出格式**：Safetensors adapter，可直接被 Ollama 加载
- **超参策略**：内部根据数据量、样本长度自动推荐，用户不感知

### 3. 用户接口

```bash
mstudio train ./my_raw_data/
```

等待完成后，输出目录包含：

```
output/
├── adapter_model.safetensors
├── adapter_config.json
└── Modelfile
```

执行 `ollama create my-lora -f ./output/Modelfile`，即可在本地使用。

---

## 技术栈

- **训练**：MLX / mlx-lm
- **推理加载**：Ollama
- **数据蒸馏**：本地 LLM（Qwen2.5 / Mistral 等，通过 Ollama 或 MLX 加载）
- **运行环境**：Apple Silicon（M2 Ultra 64GB 为开发基准）

---

## 与 mnemo 的关系

mstudio 是 [mnemo](https://github.com/zhuqingyv/mnemo) 记忆内化流水线的「训练工厂」模块：

```
mnemo 记忆库 → 导出原始数据 → mstudio 蒸馏训练 → LoRA 加载回本地模型
```

mstudio 不感知 mnemo 的具体字段结构，只接收文件夹输入。

---

## 文档

- [`docs/TECHNICAL_DESIGN.md`](./docs/TECHNICAL_DESIGN.md) — 技术方案与架构设计
- [`docs/RESEARCH.md`](./docs/RESEARCH.md) — 训练框架、多 LoRA 策略、数据蒸馏调研

---

## 开发中

项目处于早期设计阶段，API 和接口随时可能变动。

## License

MIT
