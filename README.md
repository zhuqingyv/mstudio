# mstudio

AI 驱动的 LoRA 生成器。从原始输入到高质量训练数据，再到微调模型，全流程自动化。

---

## 核心流程

mstudio 将整个 LoRA 训练过程拆分为两个核心阶段：

### 1. 数据蒸馏（Data Distillation）

通过多 Agent 协作，将用户的原始输入（图片、文本描述、风格概念等）自动蒸馏为**高质量、结构化、分类清晰**的训练数据集。

- **输入解析**：支持图片、文本、风格参考等多种输入形式
- **智能分类**：根据内容特征自动划分数据类别（角色、风格、物品、概念等）
- **质量过滤**：Agent 自动评估并剔除低质量样本
- **标注生成**：自动生成精准的训练标注（caption、tag、触发词等）

### 2. 模型微调（LoRA Fine-tuning）

根据目标模型类型，自动选择最优配置，完成 LoRA 微调训练。

- **多基座支持**：兼容 SDXL、Flux、SD3、Pony 等主流模型架构
- **配置自适应**：根据数据类型和模型特性自动推荐训练参数
- **一键训练**：封装训练流水线，简化操作门槛
- **模型管理**：版本化管理生成的 LoRA 权重与训练日志

---

## 项目结构

```
mstudio/
├── data/               # 训练数据与蒸馏结果
├── agents/             # 数据蒸馏 Agent 模块
├── trainers/           # 微调训练器
├── configs/            # 模型配置文件
├── models/             # 输出模型与 LoRA 权重
├── scripts/            # 实用脚本
└── tests/              # 测试用例
```

---

## 快速开始

> 项目开发中，API 和接口可能随时变动。

```bash
# 克隆仓库
git clone https://github.com/zhuqingyv/mstudio.git
cd mstudio

# 安装依赖
# TODO

# 启动数据蒸馏
# TODO

# 开始 LoRA 训练
# TODO
```

---

## Roadmap

- [ ] 数据蒸馏 Agent 框架
- [ ] 多类型数据分类与标注
- [ ] SDXL / Flux LoRA 训练器
- [ ] Web UI 交互界面
- [ ] 模型版本管理与发布

---

## License

MIT
