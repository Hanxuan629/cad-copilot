# CAD-Copilot 周末项目执行计划

> 目标：一个周末做出一个「贴合 Mage-VL 组方向」的多模态项目，用于投简历（该组看简历不太看面试）。
> 定位：**既有真·多模态训练（碰模型内部），又有 Agent 编排（系统能力）**，避免被看成「只是 Agent 应用工程师」。

---

## 一、项目一句话

**CAD-Copilot**：程序化自动造数据 → LoRA 微调 Qwen3-VL-2B 作为「感知 tool」→ 用 Claude Code SDK 编排 Agent 完成「看 CAD 图 → 生成 CADQuery 建模脚本 → 渲染自检 → 迭代修正」的闭环。

## 二、为什么这个设计能打动 Mage-VL 组

| 他们的卖点（发布文案里） | 你项目里的对应点 |
|---|---|
| codec-native ViT / 流式感知 | 你 **LoRA 微调 Qwen3-VL-2B** 做 CAD 感知 tool（体现能碰模型内部，且直接用他们对标的 Qwen3-VL 系列） |
| AI4AI：AI-driven 数据处理 | 你用 **CADQuery 程序化免标注造数据**（呼应「750M 未标注数据」理念） |
| 多模态智能体 | **Claude Code SDK 编排** 感知+建模+自检 tools |
| 人机协作 | **迭代式建模**，人可介入修正（还呼应你自己的 step-size 实验课题） |

## 三、技术栈（40G 单卡可行）

- **数据生成**：`cadquery` 或 `build123d`（程序化生成 3D 模型 → 渲染 PNG → 自动导出结构化标注 JSON）
- **模型**：`Qwen3-VL-2B-Instruct` + LoRA（PEFT）。2B 在 40G 上跑 LoRA 非常宽裕；如时间足也可试 4B/8B。
- **训练框架**：`transformers` + `peft` + `trl`（SFTTrainer），或直接用 Qwen 官方微调脚本 / LLaMA-Factory（最省事）
- **Agent**：Claude Code SDK / Claude Agent SDK 编排 tools
- **渲染**：CADQuery 自带 或 `matplotlib`/`trimesh` 渲染

## 四、周末小时级时间表

### 🗓️ 周六（造数据 + 起训）

| 时段 | 任务 | 产出 |
|---|---|---|
| 上午 3h | 搭数据生成器：写参数化 CADQuery 模板（法兰、支架、轴、带孔板等 5~8 类），随机化参数 | `gen_data.py`，能批量产 (PNG, 标注JSON) |
| 午后 1h | 生成 2k~5k 样本，检查渲染图和标注质量 | `dataset/` 图 + 标注 |
| 午后 2h | 把数据转成 VLM 微调格式（对话式：图 + "描述这个零件的几何要素" → 结构化答案） | `train.jsonl` |
| 傍晚 2h | 起 LoRA 微调（Qwen3-VL-2B），先跑通 100 步确认 loss 下降 | 训练 pipeline 跑通 |
| 晚上 | 挂机训练（几个 epoch） | LoRA 权重 |

### 🗓️ 周日（Agent 编排 + 出 Demo + 写简历）

| 时段 | 任务 | 产出 |
|---|---|---|
| 上午 1h | 评测微调后的 VLM：几张没见过的 CAD 图，看感知准不准 | 前后对比（base vs LoRA） |
| 上午 2h | 把 VLM 封装成一个 tool（输入图 → 输出结构化几何描述） | `perception_tool.py` |
| 午后 3h | 用 Claude Code SDK 编排 Agent：看图(感知tool) → 生成CADQuery脚本(codegen) → 渲染 → 自检对比 → 迭代 | Agent 闭环 demo |
| 傍晚 2h | 录一个端到端 demo（gif/视频）：给一张 CAD 图，Agent 自动重建出模型 | demo 素材 |
| 晚上 2h | 整理 README + 简历话术 + GitHub 提交 | 可投的项目 |

> ⏱️ 若时间紧，砍掉「迭代自检」，保留「感知→生成→渲染」主线即可成 demo。数据和微调优先级最高。

## 五、避坑清单

- ⚠️ **数据质量 > 数据量**：先造 500 条跑通全流程，再放量。别一上来生成 5k 才发现标注格式错。
- ⚠️ **微调前先确认 base 模型能读图**：先用原始 Qwen2-VL 测一张图，确认环境 OK 再微调。
- ⚠️ **LoRA 目标 module 别配错**：VLM 微调通常只调 LLM 部分的注意力层，vision encoder 可先冻结。
- ⚠️ **CADQuery 脚本生成会大量报错**：Agent 里一定要有「执行报错 → 反馈给模型重生成」的循环，这本身就是亮点（自检）。
- ⚠️ **Claude Code SDK 的 tool 调用**：先跑通一个最简 tool，再接 VLM，别一次性全接。

## 六、简历话术（可直接改用）

> **CAD-Copilot：多模态 CAD 建模智能体**
> - 设计并实现「感知-生成-自检」多模态 Agent，用 Claude Agent SDK 编排自建 tools，实现从 CAD 图纸到可执行建模脚本的闭环重建。
> - **LoRA 微调 Qwen3-VL-2B** 作为几何感知 tool，在自建 CAD 数据集上使结构化几何识别准确率提升 XX%（填你的实测数）。
> - 构建 **程序化免标注数据生成 pipeline**（CADQuery），自动产出 (渲染图, 结构化标注) 训练对 Xk 条，无需人工标注。
> - 实现执行反馈驱动的迭代自检机制，Agent 根据渲染/报错结果自动修正建模脚本。

**面试/简历里要强调的关键词**（跟该组同频）：多模态微调、免标注数据 pipeline、AI-driven / Agent 编排、执行反馈闭环、人机协作迭代。

## 七、后续可延伸（如果想让项目更强）

- 把感知 tool 换成**你自己改的 vision encoder**（哪怕小改），直接对标他们「codec-native ViT」的叙事
- 做一个小 **ablation**：有/无微调、有/无自检循环的成功率对比表 —— 研究品味的体现
- 复现他们论文里的 **Zero-Vision SFT** 一个小实验，写进简历

---
*生成于 2026-07-31，配合 Mage-VL 组招聘方向定制。*
