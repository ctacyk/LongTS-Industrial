"""
LongTS-Industrial Benchmark 构建器

将合成数据 QA + EngineMT-QA 子集合并为最终 Benchmark，
并划分 train/val/test/test-needle 集合。
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path


def build_benchmark(synthetic_qa_path: str,
                     engineqa_path: str,
                     output_dir: str,
                     seed: int = 42):
    """
    构建最终 Benchmark

    划分策略：
    - train: ~65% (用于SFT训练)
    - val: ~10% (超参调优)
    - test: ~20% (最终评估)
    - test-needle: 从 test 中抽取所有 L=10000 且有故障的 L2 样本
    """
    rng = random.Random(seed)

    # === 加载数据 ===
    synthetic_records = []
    if os.path.exists(synthetic_qa_path):
        with open(synthetic_qa_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    synthetic_records.append(json.loads(line))
    print(f"Loaded {len(synthetic_records)} synthetic records")

    engineqa_records = []
    if os.path.exists(engineqa_path):
        with open(engineqa_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    engineqa_records.append(json.loads(line))
    print(f"Loaded {len(engineqa_records)} EngineMT-QA records")

    # === 合并 ===
    all_records = synthetic_records + engineqa_records
    rng.shuffle(all_records)
    print(f"Total records: {len(all_records)}")

    # === 划分 ===
    # 先抽出 test-needle: L=10000 且有故障且有 L2 QA
    needle_records = []
    remaining_records = []

    for r in all_records:
        is_needle = (
            r.get("source") == "synthetic" and
            r["metadata"].get("total_length", 0) == 10000 and
            len(r.get("ground_truth_events", [])) > 0
        )
        if is_needle:
            needle_records.append(r)
        else:
            remaining_records.append(r)

    # 从 remaining 划分 train/val/test
    n_total = len(remaining_records)
    n_train = int(n_total * 0.65)
    n_val = int(n_total * 0.10)
    n_test = n_total - n_train - n_val

    train_records = remaining_records[:n_train]
    val_records = remaining_records[n_train:n_train + n_val]
    test_records = remaining_records[n_train + n_val:]

    # === 统计 ===
    splits = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
        "test_needle": needle_records,
    }

    print(f"\n=== Benchmark Split ===")
    total_qa = 0
    split_stats = {}
    for split_name, records in splits.items():
        n_qa = sum(len(r["qa_pairs"]) for r in records)
        total_qa += n_qa

        # Domain distribution
        domains = defaultdict(int)
        levels = defaultdict(int)
        sources = defaultdict(int)
        difficulties = defaultdict(int)

        for r in records:
            domains[r["domain"]] += 1
            sources[r.get("source", "unknown")] += 1
            difficulties[r["metadata"].get("difficulty", "unknown")] += 1
            for qa in r["qa_pairs"]:
                levels[qa["level"]] += 1

        stats = {
            "n_samples": len(records),
            "n_qa": n_qa,
            "domains": dict(domains),
            "levels": dict(levels),
            "sources": dict(sources),
            "difficulties": dict(difficulties),
        }
        split_stats[split_name] = stats

        print(f"\n  {split_name}:")
        print(f"    Samples: {len(records)}, QA: {n_qa}")
        print(f"    Domains: {dict(domains)}")
        print(f"    Levels: {dict(levels)}")
        print(f"    Sources: {dict(sources)}")

    print(f"\n  TOTAL QA: {total_qa}")

    # === 保存 ===
    os.makedirs(output_dir, exist_ok=True)

    for split_name, records in splits.items():
        split_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(split_path, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"Saved: {split_path} ({len(records)} records)")

    # Save stats
    stats_path = os.path.join(output_dir, "benchmark_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(split_stats, f, ensure_ascii=False, indent=2)
    print(f"Stats saved: {stats_path}")

    # Save README
    readme = f"""# LongTS-Industrial Benchmark

## Overview

A benchmark for evaluating long time-series industrial reasoning capabilities of LLMs/VLMs.

## Statistics

| Split | Samples | QA Pairs | Description |
|-------|---------|----------|-------------|
| train | {split_stats['train']['n_samples']} | {split_stats['train']['n_qa']} | SFT training |
| val | {split_stats['val']['n_samples']} | {split_stats['val']['n_qa']} | Hyperparameter tuning |
| test | {split_stats['test']['n_samples']} | {split_stats['test']['n_qa']} | Final evaluation |
| test_needle | {split_stats['test_needle']['n_samples']} | {split_stats['test_needle']['n_qa']} | Needle-in-haystack (L=10000) |
| **Total** | **{sum(s['n_samples'] for s in split_stats.values())}** | **{total_qa}** | |

## Domains

- **coal_mill**: Coal mill (grinding equipment in power plants)
- **transformer**: Electricity transformer (ETT-inspired)
- **pump**: Water circulation pump system (SKAB-inspired)
- **aero_engine**: Aero engine (from EngineMT-QA)

## Reasoning Levels

- **L1 (Descriptive)**: Describe overall operating patterns
- **L2 (Localization)**: Find anomalies in long sequences (needle-in-haystack)
- **L3 (Causal)**: Analyze root causes and multi-variable relationships
- **L4 (Decision)**: Provide maintenance recommendations

## Data Format

Each record in JSONL files contains:
- `id`: Unique identifier
- `domain`: Industrial domain
- `source`: "synthetic" or "enginemt_qa"
- `metadata`: Length, channels, difficulty, needle_ratio
- `timeseries_path`: Path to CSV data file
- `ground_truth_events`: Precise fault locations and parameters
- `qa_pairs`: List of {{level, question, answer, reasoning_chain, required_interval}}

## Usage

```python
import json

with open('test.jsonl', 'r') as f:
    for line in f:
        sample = json.loads(line)
        for qa in sample['qa_pairs']:
            print(f"[{{qa['level']}}] {{qa['question'][:80]}}...")
```
"""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme)

    print(f"\nBenchmark built at: {output_dir}")
    return splits


if __name__ == "__main__":
    build_benchmark(
        synthetic_qa_path="Data/benchmark/synthetic_qa.jsonl",
        engineqa_path="Data/benchmark/engineqa_subset.jsonl",
        output_dir="Data/benchmark/LongTS_Industrial",
    )
