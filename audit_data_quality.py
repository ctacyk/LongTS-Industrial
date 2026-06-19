"""
严格的数据质量审计脚本

检查项：
1. CSV 数据的值域是否在物理合理范围
2. 故障注入位置是否与 metadata 一致（ground truth 对齐）
3. 通道间相关性是否符合物理规律
4. LLM QA 中的时间步引用是否与 metadata 匹配
5. 正常段 vs 异常段的统计差异是否显著
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, '.')

print("=" * 70)
print("LongTS-Industrial Benchmark 数据质量审计")
print("=" * 70)

# ====================================================================
# 审计1: CSV 数值范围与物理合理性
# ====================================================================
print("\n### 审计1: CSV 数值范围 vs 真实数据范围 ###\n")

# 真实磨煤机数据的参考范围（来自 merged.csv 分析）
real_ranges = {
    "coal_mill": {
        "motor_current": {"real_mean": 47.3, "real_std": 13.3, "real_min": -0.4, "real_max": 72.1},
        "winding_temp_1": {"real_mean": 50.4, "real_std": 7.1, "real_min": 28.5, "real_max": 73.0},
        "bearing_temp_drive": {"real_mean": 31.8, "real_std": 6.2, "real_min": 16.4, "real_max": 53.4},
        "inlet_temp": {"real_mean": 270.8, "real_std": 54.9, "real_min": 7.3, "real_max": 318.1},
    }
}

# 检查几个代表性的合成数据文件
test_files = [
    ("coal_mill/normal_steady_70_L5000_S42.csv", "正常运行"),
    ("coal_mill/fault_bearing_drive_moderate_L3000_S42.csv", "轴承故障"),
    ("coal_mill/fault_blockage_severe_L5000_S42.csv", "入口堵塞"),
    ("transformer/normal_summer_70_L3000_S42.csv", "变压器正常"),
    ("transformer/fault_overload_severe_L5000_S42.csv", "变压器过载"),
    ("pump/normal_steady_70_L3000_S42.csv", "水泵正常"),
    ("pump/fault_cavitation_moderate_L5000_S42.csv", "水泵气蚀"),
]

for rel_path, desc in test_files:
    full_path = f"Data/benchmark/{rel_path}"
    try:
        df = pd.read_csv(full_path)
        print(f"  [{desc}] {rel_path}")
        print(f"    shape: {df.shape}")
        for col in df.columns:
            vals = df[col].values
            print(f"    {col:30s}: mean={vals.mean():8.2f}, std={vals.std():7.2f}, "
                  f"min={vals.min():8.2f}, max={vals.max():8.2f}")
        print()
    except FileNotFoundError:
        print(f"  [MISSING] {full_path}")

# ====================================================================
# 审计2: Ground Truth 对齐检查
# ====================================================================
print("\n### 审计2: Ground Truth 对齐（metadata 时间步 vs CSV 实际变化）###\n")

# 加载一个有故障的样本，检查故障区间前后的统计差异
meta_path = "Data/benchmark/coal_mill/fault_bearing_drive_moderate_L3000_S42_metadata.json"
csv_path = "Data/benchmark/coal_mill/fault_bearing_drive_moderate_L3000_S42.csv"

with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)

df = pd.read_csv(csv_path)
event = meta["injected_events"][0]
fault_start, fault_end = event["time_range"]
primary_ch = event["affected_channels"][0]  # bearing_temp_drive

# 取故障前段和故障段对比
pre_fault = df[primary_ch].iloc[:fault_start]
fault_seg = df[primary_ch].iloc[fault_start:fault_end]
post_fault = df[primary_ch].iloc[fault_end:]

print(f"  文件: {csv_path}")
print(f"  故障通道: {primary_ch}")
print(f"  metadata 标注范围: [{fault_start}, {fault_end})")
print(f"  预期温升: {event['parameters']['delta_temp']}°C")
print()
print(f"  故障前段 [0, {fault_start}):   mean={pre_fault.mean():.2f}°C, std={pre_fault.std():.2f}")
print(f"  故障段   [{fault_start}, {fault_end}): mean={fault_seg.mean():.2f}°C, std={fault_seg.std():.2f}")
print(f"  故障后段 [{fault_end}, end):   mean={post_fault.mean():.2f}°C, std={post_fault.std():.2f}")
print(f"  实际变化: 故障段均值 - 故障前均值 = {fault_seg.mean() - pre_fault.mean():.2f}°C")
print(f"  实际最大值差: {fault_seg.max() - pre_fault.mean():.2f}°C")

# 检查故障后段是否保持了升高状态（inject_gradual_change 的 end 后保持效果）
if len(post_fault) > 0:
    print(f"  故障后段保持: {post_fault.mean() - pre_fault.mean():.2f}°C（应≈{event['parameters']['delta_temp']}°C）")

# ====================================================================
# 审计3: 通道间相关性是否符合物理规律
# ====================================================================
print("\n### 审计3: 通道间相关性 ###\n")

df_normal = pd.read_csv("Data/benchmark/coal_mill/normal_steady_70_L5000_S42.csv")
corr = df_normal.corr()

# 物理上应满足的相关性
checks = [
    ("winding_temp_1", "winding_temp_2", 0.9, "绕组双传感器应高度相关"),
    ("bearing_temp_nondrive", "bearing_temp_drive", 0.7, "同轴两端轴承应正相关"),
    ("inlet_temp", "inlet_valve_opening", 0.3, "入口温度与阀门应正相关"),
]

for ch1, ch2, min_corr, reason in checks:
    actual = corr.loc[ch1, ch2]
    status = "PASS" if actual >= min_corr else "FAIL"
    print(f"  [{status}] corr({ch1}, {ch2}) = {actual:.3f} (要求>={min_corr}): {reason}")

# ====================================================================
# 审计4: LLM QA 的时间步引用与 metadata 对齐
# ====================================================================
print("\n### 审计4: LLM QA 的 required_interval 与 metadata 对齐 ###\n")

llm_qa_path = "Data/benchmark/synthetic_qa_llm_dedup.jsonl"
mismatches = 0
checked = 0
with open(llm_qa_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        gt_events = record.get("ground_truth_events", [])
        if not gt_events:
            continue

        gt_range = gt_events[0]["range"]

        for qa in record.get("qa_pairs", []):
            ri = qa.get("required_interval")
            if ri is not None:
                checked += 1
                if ri != gt_range:
                    mismatches += 1
                    if mismatches <= 3:
                        print(f"  MISMATCH: {record['id']}")
                        print(f"    metadata GT: {gt_range}")
                        print(f"    QA interval: {ri}")

print(f"  Checked {checked} QA pairs with required_interval")
print(f"  Mismatches: {mismatches}")
if mismatches == 0:
    print(f"  [PASS] All LLM QA intervals match ground truth")

# ====================================================================
# 审计5: 数据多样性统计
# ====================================================================
print("\n### 审计5: 数据多样性 ###\n")

# 统计 benchmark 中的场景分布
from collections import Counter
scenarios = Counter()
lengths = Counter()
difficulties = Counter()

with open(llm_qa_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        scenarios[r["scenario_id"]] += 1
        lengths[r["metadata"]["total_length"]] += 1
        difficulties[r["metadata"]["difficulty"]] += 1

print(f"  Unique scenarios: {len(scenarios)}")
print(f"  Length distribution: {dict(sorted(lengths.items()))}")
print(f"  Difficulty distribution: {dict(sorted(difficulties.items()))}")
print(f"  Top 5 scenarios: {scenarios.most_common(5)}")

print("\n" + "=" * 70)
print("审计完成")
print("=" * 70)
