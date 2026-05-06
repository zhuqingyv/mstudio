# mstudio 技术方案

> 版本：v0.1
> 日期：2026-05-06
> 状态：设计中

---

## 1. 项目概述

mstudio 是一个 AI 驱动的 LoRA 生成器，目标是通过自动化流水线降低 LoRA 训练门槛。系统分为两大核心阶段：

1. **数据蒸馏（Data Distillation）**：利用多 Agent 协作，将原始输入转化为高质量、结构化、可直接用于训练的数据集。
2. **LoRA 微调（LoRA Fine-tuning）**：基于目标模型架构，自动选择训练配置，执行微调并管理输出模型。

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        API Gateway                          │
│                     (FastAPI / REST)                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│   数据蒸馏    │ │   任务调度   │ │  LoRA 训练   │
│   引擎        │ │  (Celery)   │ │   引擎        │
└──────┬───────┘ └──────┬──────┘ └──────┬───────┘
       │                │               │
       ▼                ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                      存储与基础设施                          │
│  PostgreSQL  │  Redis  │  MinIO/S3  │  Docker / GPU Node   │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 架构原则

- **模块化**：蒸馏、训练、存储、调度完全解耦，可独立演进。
- **可扩展**：Agent 和训练后端均支持插件式扩展。
- **异步化**：长耗时任务（训练、大规模数据蒸馏）全部异步执行。
- **配置驱动**：训练参数、Agent 流程均通过配置管理，降低硬编码。

---

## 3. 核心模块设计

### 3.1 数据蒸馏引擎（Data Distillation Engine）

#### 3.1.1 输入网关（Input Gateway）

支持多种原始输入形式：

| 输入类型 | 处理方式 |
|---------|---------|
| 图片文件 / 图片文件夹 | 直接入库，提取基础元信息（分辨率、格式、色彩空间） |
| 文本描述 | 作为概念种子，后续可用于生成或检索参考图 |
| URL / 网页链接 | 爬虫模块抓取图片及关联文本 |
| 现有数据集 | 导入并重新进行质量评估与标注 |

#### 3.1.2 Agent 编排层（Agent Orchestration）

蒸馏流程采用**有向无环图（DAG）Pipeline** 编排，每个节点是一个 Agent。核心 Pipeline 如下：

```
[Input] → [Analyzer] → [Classifier] → [Filter] → [Annotator] → [Validator] → [Output]
              ↑            ↑            ↑           ↑             ↑
              └────────────┴────────────┴───────────┴─────────────┘
                         (反馈回路，支持人工干预节点)
```

#### 3.1.3 Agent 职责定义

| Agent | 职责 | 关键技术 |
|-------|------|---------|
| **Analyzer** | 分析输入内容，提取视觉特征、风格特征、主体特征 | CLIP 特征提取、图像哈希、色彩直方图 |
| **Classifier** | 将样本归类为：角色、风格、物品、概念、动作等 | 多模态分类模型 / 规则引擎 + LLM 辅助 |
| **Filter** | 质量评估（模糊、过曝、低分辨率）、去重、NSFW 检测 | 图像质量评估模型、感知哈希、pHash、CLIP 相似度去重 |
| **Annotator** | 生成训练标注：自然语言 Caption、Danbooru Tags、触发词 | WD14 Tagger、VLM（Qwen-VL / LLaVA）、LLM 触发词生成 |
| **Validator** | 最终数据集质量校验：标签一致性、样本均衡性、触发词有效性 | 规则校验 + 采样可视化 |

#### 3.1.4 数据输出格式

蒸馏完成的数据集导出为标准文件夹结构：

```
dataset/
├── train/
│   ├── 0001.jpg
│   ├── 0001.txt          # 对应图片的 caption / tags
│   ├── 0002.jpg
│   ├── 0002.txt
│   └── ...
├── meta.json             # 数据集元信息（类别、触发词、统计信息）
└── preview_grid.jpg      # 样本预览网格
```

### 3.2 LoRA 训练引擎（LoRA Training Engine）

#### 3.2.1 训练适配器层（Training Adapter）

为了兼容多种训练框架与模型架构，设计统一的训练适配器接口：

```python
class BaseTrainer(ABC):
    @abstractmethod
    def prepare(self, dataset_path: str, config: TrainingConfig) -> None: ...
    
    @abstractmethod
    def train(self) -> TrainingResult: ...
    
    @abstractmethod
    def export(self, output_dir: str) -> List[Path]: ...
```

| 适配器实现 | 适用模型 | 底层框架 |
|-----------|---------|---------|
| `KohyaSSAdapter` | SDXL、SD3、Pony | Kohya_ss / sd-scripts |
| `FluxToolkitAdapter` | Flux.1 [dev/schnell] | ai-toolkit |
| `SimpleTunerAdapter` | SDXL、Flux（轻量） | SimpleTuner |

#### 3.2.2 配置引擎（Config Engine）

系统内置**配置模板库**，根据以下维度自动推荐或生成训练参数：

- **模型类型**：SDXL（1024px）、Flux（1024px+）、SD3（1024px）
- **数据类型**：角色（高保真）、风格（泛化）、物品（细节）、概念（抽象）
- **数据规模**：样本数量决定学习率调度、训练步数
- **硬件约束**：显存大小决定 batch size、梯度累积、优化器选择（8G/16G/24G+）

配置引擎输出为标准训练配置（如 Kohya 的 `config.toml` 或 ai-toolkit 的 `config.yaml`），同时支持用户手动覆盖。

#### 3.2.3 训练任务生命周期

```
[创建任务] → [准备数据] → [拉取基座模型] → [生成配置] → [启动训练]
     ↑                                              ↓
     └──────── [完成] ← [导出模型] ← [生成样张] ← [训练中]
```

- **训练监控**：实时捕获 stdout/stderr，解析损失曲线、学习率变化。
- **样张生成**：按固定间隔生成验证样本图，便于观察收敛情况。
- **早停机制**：支持基于损失 plateau 或用户手动终止的早停。

### 3.3 任务调度与消息系统

- **队列**：Redis + Celery，支持优先级队列（高优先级任务优先调度）。
- **状态机**：每个任务具有明确状态：`pending` → `preparing` → `running` → `sampling` → `completed` / `failed`。
- **回调通知**：训练完成或失败时，通过 Webhook 或 SSE 向前端/用户推送事件。

---

## 4. 数据流设计

```
用户输入
   │
   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  输入网关    │────▶│  对象存储    │     │  PostgreSQL      │
│ (校验/转换)  │     │ (MinIO/S3)  │     │ (项目/任务/元信息)│
└─────────────┘     └─────────────┘     └─────────────────┘
   │
   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│ 蒸馏 Pipeline│────▶│ 数据集存储   │────▶│   LoRA 训练任务  │
│ (Agent DAG)  │     │ (标准格式)   │     │   (Celery Worker)│
└─────────────┘     └─────────────┘     └─────────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  模型仓库    │
                                        │(MinIO/S3)   │
                                        └─────────────┘
```

---

## 5. 技术选型

| 层级 | 选型 | 理由 |
|------|------|------|
| **后端语言** | Python 3.10+ | AI/ML 生态最成熟 |
| **Web 框架** | FastAPI | 高性能、异步原生、类型安全、自动生成文档 |
| **任务队列** | Celery + Redis | 成熟稳定，支持定时任务与优先级队列 |
| **数据库** | PostgreSQL 15+ | 关系型数据结构化存储，JSONB 支持灵活扩展 |
| **对象存储** | MinIO (本地) / S3 (生产) | 兼容 S3 API，存图片与大模型文件 |
| **容器化** | Docker + Docker Compose | 开发环境一致性，训练任务隔离 |
| **GPU 调度** | NVIDIA Docker Runtime | 训练 Worker 直接挂载 GPU |
| **前端（二期）** | React + TypeScript | 类型安全，生态丰富 |

---

## 6. 推荐目录结构

```
mstudio/
├── docs/                           # 技术文档与使用手册
├── src/
│   ├── api/                        # FastAPI 路由、依赖、中间件
│   │   ├── routers/
│   │   ├── schemas/
│   │   └── main.py
│   ├── core/                       # 业务核心
│   │   ├── distillation/           # 数据蒸馏引擎
│   │   │   ├── pipeline.py         # DAG 编排器
│   │   │   ├── agents/             # Agent 实现
│   │   │   └── exporters/          # 数据集导出器
│   │   ├── training/               # LoRA 训练引擎
│   │   │   ├── adapters/           # 训练框架适配器
│   │   │   ├── config_engine.py    # 配置生成引擎
│   │   │   └── monitor.py          # 训练监控
│   │   └── models/                 # 模型注册与管理
│   ├── agents/                     # Agent 具体实现（可独立部署）
│   │   ├── analyzer.py
│   │   ├── classifier.py
│   │   ├── filter.py
│   │   ├── annotator.py
│   │   └── validator.py
│   ├── storage/                    # 数据访问层
│   │   ├── db.py                   # PostgreSQL 连接
│   │   └── s3_client.py            # 对象存储客户端
│   ├── tasks/                      # Celery 任务定义
│   └── common/                     # 工具、配置、常量
│       ├── config.py
│       └── logger.py
├── configs/                        # 训练配置模板（按模型分类）
│   ├── sd_xl/
│   ├── flux/
│   └── sd3/
├── scripts/                        # 部署与维护脚本
├── tests/                          # 单元测试与集成测试
├── docker-compose.yml              # 本地全栈开发环境
├── Dockerfile
├── pyproject.toml                  # 依赖管理
└── README.md
```

---

## 7. 关键接口设计（API）

### 7.1 项目与数据蒸馏

```http
POST   /api/v1/projects                  # 创建项目
GET    /api/v1/projects/{id}             # 获取项目详情
POST   /api/v1/projects/{id}/upload      # 上传原始素材
POST   /api/v1/projects/{id}/distill     # 启动数据蒸馏任务
GET    /api/v1/tasks/{task_id}           # 查询任务状态与日志
GET    /api/v1/projects/{id}/dataset     # 下载/预览数据集
```

### 7.2 LoRA 训练

```http
POST   /api/v1/projects/{id}/train       # 启动训练任务（传入模型类型、覆盖参数）
GET    /api/v1/tasks/{task_id}/progress  # 获取训练进度（损失、步数、预计剩余时间）
GET    /api/v1/tasks/{task_id}/samples   # 获取训练过程中生成的验证样张
GET    /api/v1/projects/{id}/models      # 列出项目下所有输出模型
POST   /api/v1/models/{model_id}/export  # 导出 LoRA 权重（打包下载）
```

---

## 8. 部署方案

### 8.1 本地开发环境

```bash
docker-compose up -d
# 启动服务：API Server、PostgreSQL、Redis、MinIO
```

### 8.2 生产环境（单节点 GPU）

```
[Load Balancer]
      │
      ▼
[API Server Container] ──▶ [PostgreSQL + Redis]
      │
      ▼
[GPU Worker Node] ──▶ [NVIDIA Docker + Training Frameworks]
```

- API Server 与 GPU Worker 可分离部署，Worker 通过 Celery 消费任务。
- 模型权重与数据集统一存放于对象存储，Worker 按需拉取。

---

## 9. 风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|---------|
| 训练框架版本迭代快 | 适配器失效 | 抽象适配器接口，隔离底层变动；CI 集成框架冒烟测试 |
| Agent 生成标注质量不稳定 | 训练效果差 | Pipeline 中增加 Validator 节点；关键节点支持人工审核与编辑 |
| GPU 资源昂贵且紧张 | 成本高、排队久 | 支持任务优先级与队列限流；支持云 GPU 按需弹性扩容 |
| 大模型文件体积大 | 存储与传输瓶颈 | 对象存储 + CDN；Worker 本地缓存常用基座模型 |

---

## 10. 里程碑规划

| 阶段 | 目标 | 周期 |
|------|------|------|
| **MVP** | 完成数据蒸馏 Pipeline（Analyzer + Annotator）+ Kohya SDXL 训练适配器 | 4 周 |
| **v0.2** | 支持 Flux 训练、配置引擎自动推荐参数、基础 Web UI | +4 周 |
| **v0.3** | 多 Agent 全链路自动化、模型版本管理、公开 API | +4 周 |

---

## 附录：术语表

- **LoRA**：Low-Rank Adaptation，低秩适应，一种高效的大模型微调方法。
- **Agent**：在此语境下指具备特定能力（分析、分类、标注等）的自动化处理单元，可由传统算法或 LLM 驱动。
- **WD14 Tagger**：基于 Waifu Diffusion 训练的图像标签生成模型，广泛用于二次元图像标注。
- **VLM**：Vision Language Model，视觉语言模型，可同时理解图像与文本。
