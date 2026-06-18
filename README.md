# LongTS-Industrial

> 面向**视觉语言模型（VLM）长时序工业故障推理**能力的评测基准与配套工具链

LongTS-Industrial 将长时间序列的多通道工业监测数据可视化为图像，要求 VLM 通过「看图」完成从整体感知、异常定位、根因分析到运维决策的分层推理。本仓库提供 **数据合成、基准构建、模型评测** 的完整可复现代码；基准数据集发布在 HuggingFace。

- 📊 数据集（HuggingFace）：https://huggingface.co/datasets/ANTICH/LongTS-Industrial
- 📄 论文：（论文发表后补充）

---

## 任务分层（L1–L4）

| 层级 | 能力 | 典型问题 |
|------|------|----------|
| L1 | 整体描述 | 描述各传感器通道的趋势、波动与相互关系 |
| L2 | 异常定位 | 精确指出异常的时间区间（精确到步） |
| L3 | 根因分析 | 结合多通道推断故障根源及传播路径 |
| L4 | 运维决策 | 给出分级运维建议与优先级 |

## 覆盖设备域

四类工业设备，均使用 TimeBlender 物理建模合成多通道时序（热惯性 + 通道耦合 + 故障注入）：

- `coal_mill` 磨煤机
- `pump` 水泵
- `transformer` 变压器
- `wind_turbine` 风力发电机组

## 整体流程

```
真实数据观察 / 统计标定
        ↓
TimeBlender 物理建模合成多通道时序（热惯性 + 通道耦合 + 故障注入）
        ↓
可视化为 eval.png + 生成 metadata.json（故障类型 / 区间 / 因果链）
        ↓
Benchmark 打包（test.jsonl）
        ↓
多模型评估（HES 框架）
```

---

## 仓库结构

```
.
├── time_blender/            # 时序合成引擎（第三方，见 Acknowledgements）
├── MultiAgentTS/            # 各设备域的合成、故障注入与基准打包
│   ├── synth_tb_coal_mill.py, simulate_faults.py, ...   # 数据合成
│   └── benchmark_builder.py                              # 基准构建
├── generate_sft_data.py     # 基于 metadata 用 LLM 生成 L1–L4 参考答案
├── compute_data_profiles.py, sensor_groups.py           # 数据画像 / 通道分组
├── benchmark_eval.py        # HES 评测核心
├── run_benchmark_eval.py    # 评测入口（远程 API / 本地 vLLM）
├── semantic_similarity.py   # 语义相似度（embedding）
├── audit_*.py / validate_*.py / verify_qa_answers.py    # 质量校验工具
└── results/benchmark_eval_v2/domain_level_table.md      # 排行榜汇总
```

> 注：训练（SFT/GRPO）相关代码与数据不在本次发布范围内。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env` 并填入 Key（评测的 Judge 与 Embedding 走 DashScope）：

```bash
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
```

所有脚本均通过环境变量读取 Key，**不在代码中硬编码**。

### 3. 下载基准数据集

从 HuggingFace 下载并解压到 `Data/benchmark_tb/`：

```bash
huggingface-cli download <你的HF数据集ID> --repo-type dataset --local-dir Data/benchmark_tb
```

### 4. 在基准上评测某个 VLM

```bash
export DASHSCOPE_API_KEY="sk-..."
python run_benchmark_eval.py        # 具体参数见脚本内说明
```

---

## 评测指标：HES

```
HES = DA × (0.4 × SS + 0.6 × AQ)
```

- **DA**（Diagnostic Accuracy）：诊断方向是否正确，作为门控（错误直接归零）
- **SS**（Semantic Similarity）：与参考答案的语义相似度（`text-embedding-v3`）
- **AQ**（Answer Quality）：由 Judge 模型（`qwen-plus`）打分

按 `L1–L4 × 设备域 × 难度` 多维统计，排行榜见 `results/benchmark_eval_v2/domain_level_table.md`。

---

## 许可证

- 本仓库代码：见 [`LICENSE`](LICENSE)。
- `time_blender/` 为第三方组件，版权归原作者所有，详见 Acknowledgements。

## 引用

```bibtex
@misc{longts_industrial,
  title  = {LongTS-Industrial: A Benchmark for Long Time-Series Industrial Fault Reasoning with Vision-Language Models},
  author = {Chen, Tingan},
  year   = {2026},
  note   = {https://github.com/ctacyk/LongTS-Industrial}
}
```

## Acknowledgements

时序合成基于开源项目 **time_blender**。请在使用时遵循其原始许可证；本仓库 `time_blender/THIRD_PARTY_NOTICE.md` 保留其出处与许可说明。
