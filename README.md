# LongTS-Industrial

> A benchmark and toolchain for evaluating **long time-series industrial fault reasoning with Vision-Language Models (VLMs)**

**English** | **中文** ([README.zh-CN.md](README.zh-CN.md))

> Anonymized version for double-blind review. Author, repository, and dataset identifiers have been removed.

LongTS-Industrial visualizes long, multi-channel industrial monitoring time series as images and asks VLMs to perform hierarchical reasoning purely from the plots — spanning holistic perception, anomaly localization, root-cause analysis, and maintenance decision-making. This repository provides the complete, reproducible code for **data synthesis, benchmark construction, and model evaluation**; the benchmark dataset is released on HuggingFace.

- 📊 Dataset (HuggingFace): (anonymized for review)
- 📄 Paper: (under review)

---

## Task Hierarchy (L1–L4)

| Level | Capability | Typical Question |
|-------|------------|------------------|
| L1 | Holistic description | Describe the trends, fluctuations, and inter-relationships across sensor channels |
| L2 | Anomaly localization | Pinpoint the exact time interval of the anomaly (down to the step) |
| L3 | Root-cause analysis | Infer the fault origin and propagation path from multiple channels |
| L4 | Maintenance decision | Provide graded maintenance recommendations and priorities |

## Equipment Domains

Four classes of industrial equipment, all synthesized as multi-channel time series via TimeBlender physical modeling (thermal inertia + channel coupling + fault injection):

- `coal_mill` — coal pulverizer
- `pump` — water pump
- `transformer` — power transformer
- `wind_turbine` — wind turbine generator

## Overall Pipeline

```
Observation of real data / statistical calibration
        ↓
TimeBlender physical modeling of multi-channel series (thermal inertia + channel coupling + fault injection)
        ↓
Render to eval.png + generate metadata.json (fault type / interval / causal chain)
        ↓
Benchmark packaging (test.jsonl)
        ↓
Multi-model evaluation (HES framework)
```

---

## Repository Structure

```
.
├── time_blender/            # Time-series synthesis engine (third-party, see Acknowledgements)
├── MultiAgentTS/            # Per-domain synthesis, fault injection, and benchmark packaging
│   ├── synth_tb_coal_mill.py, simulate_faults.py, ...   # Data synthesis
│   └── benchmark_builder.py                              # Benchmark construction
├── compute_data_profiles.py, sensor_groups.py           # Data profiling / channel grouping
├── benchmark_eval.py        # HES evaluation core
├── run_benchmark_eval.py    # Evaluation entry point (remote API / local vLLM)
├── semantic_similarity.py   # Semantic similarity (embedding)
├── audit_*.py / validate_*.py / verify_qa_answers.py    # Quality-validation tools
└── results/benchmark_eval_v2/domain_level_table.md      # Leaderboard summary
```

> Note: Training (SFT/GRPO) code and data are out of scope for this release.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy `.env.example` to `.env` and fill in your keys (the evaluation Judge and Embedding use DashScope):

```bash
cp .env.example .env
# Edit .env and set DASHSCOPE_API_KEY
```

All scripts read keys from environment variables — **nothing is hard-coded**.

### 3. Download the benchmark dataset

Download from HuggingFace into `Data/benchmark_tb/`:

```bash
huggingface-cli download <DATASET_ID> --repo-type dataset --local-dir Data/benchmark_tb
```

### 4. Evaluate a VLM on the benchmark

```bash
export DASHSCOPE_API_KEY="sk-..."
python run_benchmark_eval.py        # see the script for arguments
```

---

## Evaluation Metric: HES

```
HES = DA × (0.4 × SS + 0.6 × AQ)
```

- **DA** (Diagnostic Accuracy): whether the diagnostic direction is correct, used as a gate (a wrong diagnosis zeroes the score)
- **SS** (Semantic Similarity): semantic similarity to the reference answer (`text-embedding-v3`)
- **AQ** (Answer Quality): scored by a Judge model (`qwen-plus`)

Results are aggregated across `L1–L4 × domain × difficulty`. See the leaderboard at `results/benchmark_eval_v2/domain_level_table.md`.

---

## License

- Code in this repository: see [`LICENSE`](LICENSE).
- `time_blender/` is a third-party component; copyright belongs to its original authors — see Acknowledgements.

## Citation

```bibtex
@misc{longts_industrial,
  title  = {LongTS-Industrial: A Benchmark for Long Time-Series Industrial Fault Reasoning with Vision-Language Models},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

## Acknowledgements

Time-series synthesis builds on the open-source project **time_blender**. Please comply with its original license when using it; this repository keeps its attribution and license notice in `time_blender/THIRD_PARTY_NOTICE.md`.
