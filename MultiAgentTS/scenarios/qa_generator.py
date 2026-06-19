"""
QA 生成器：基于元数据模板化生成 L1-L4 开放式推理问题

设计原则：
- 所有 QA 的 ground truth 直接来自 GenerationMetadata（精确且无歧义）
- 问题模板多样化，避免重复
- 答案包含精确的时间步位置、数值变化、因果链
- 每条时序生成 5-6 个不同层级的问题

两种模式：
1. template_only=True: 纯模板生成（不调 LLM，速度快，适合批量）
2. template_only=False: 模板骨架 + LLM 润色（质量更高，需要 API）
"""

import json
import os
import random
from typing import List, Dict, Tuple
from pathlib import Path

from MultiAgentTS.scenarios.base import GenerationMetadata, EventMetadata


# ============================================================================
# 问题模板库
# ============================================================================

L1_TEMPLATES = [
    # 描述性推理：考察全局模式识别能力
    "请分析该{domain_display}在整个{duration_desc}监测周期内的运行状态，描述各传感器参数的整体趋势、波动特征和相互关系。",
    "观察该{domain_display}的全部{n_channels}个传感器通道数据（共{total_length}个时间步），概括其运行模式是否稳定，并说明你的判断依据。",
    "请描述该{domain_display}各传感器参数的统计特征（如均值水平、波动幅度、是否存在周期性），并分析通道间的关联关系。",
]

L2_TEMPLATES_FAULT = [
    # 定位推理（有故障）：大海捞针核心
    "该{domain_display}在{duration_desc}的运行数据中是否存在异常？如果有，请精确定位异常发生的时间区间，并描述异常在各传感器上的表现。",
    "请仔细检查该{domain_display}的{total_length}步运行数据，识别其中是否包含设备故障或异常状态。如果发现异常，请指出具体的时间步范围和涉及的传感器通道。",
    "分析该{domain_display}的运行数据，判断设备是否全程正常运行。如果存在偏离正常工况的区间，请定位并量化其偏离程度。",
]

L2_TEMPLATES_NORMAL = [
    # 定位推理（正常数据）：测试假阳性
    "该{domain_display}在{duration_desc}的运行数据中是否存在异常？请仔细检查各传感器参数，给出你的判断和依据。",
    "请分析该{domain_display}的运行数据，判断是否存在需要关注的异常区间或潜在故障征兆。",
]

L3_TEMPLATES = [
    # 因果推理：考察多变量关联分析能力
    "该{domain_display}在第{fault_start}步附近出现了异常变化。请分析可能的故障原因，并说明哪些传感器的变化模式支持你的推断。",
    "该设备数据显示{primary_channel}出现了{change_desc}。请结合其他传感器通道的变化，推断这一异常的根本原因和传播路径。",
    "请分析该{domain_display}异常事件中各传感器的变化先后顺序和因果关系，推断故障的起因和发展过程。",
]

L4_TEMPLATES = [
    # 决策推理：考察综合判断能力
    "基于该{domain_display}的全部运行数据和发现的异常情况，请给出运维建议：是否需要停机检修？优先处理哪些问题？建议的检修方案是什么？",
    "作为运维工程师，请根据该{domain_display}的监测数据评估设备的健康状态，给出风险等级判断和后续处置建议。",
    "该{domain_display}的运行数据反映了{severity_desc}的异常情况。请从安全性、经济性和可靠性角度，给出你的运维决策建议。",
]

# 正常数据也需要 L4
L4_TEMPLATES_NORMAL = [
    "该{domain_display}在监测期间运行正常。请评估其当前健康状态，并给出预防性维护建议。",
    "基于该{domain_display}的正常运行数据，请分析是否存在任何早期退化征兆，并建议下次计划性检修的时间。",
]


def _duration_desc(total_length: int, sampling_seconds: int) -> str:
    """将时间步数转为人类可读的时间描述"""
    total_seconds = total_length * sampling_seconds
    if total_seconds < 3600:
        return f"{total_seconds // 60}分钟"
    elif total_seconds < 86400:
        hours = total_seconds / 3600
        return f"{hours:.1f}小时"
    else:
        days = total_seconds / 86400
        return f"{days:.1f}天"


def _severity_desc(severity: str) -> str:
    return {"mild": "轻微", "moderate": "中等", "severe": "严重"}.get(severity, severity)


def _change_desc(event: EventMetadata) -> str:
    """从事件元数据生成变化描述"""
    params = event.parameters
    if "delta_temp" in params:
        return f"温度上升约{params['delta_temp']}°C"
    elif "amplitude" in params:
        return f"幅值变化约{params['amplitude']}"
    elif "overload_factor" in params:
        return f"负载升至额定的{params['overload_factor']:.0%}"
    elif "pressure_drop" in params:
        return f"压力下降约{params['pressure_drop']}Bar"
    elif "pressure_rise" in params:
        return f"压力升高约{params['pressure_rise']}Bar"
    elif "vib_delta" in params:
        return f"振动增大约{params['vib_delta']}g"
    elif "vib_increase" in params:
        return f"振动突增约{params['vib_increase']}g"
    else:
        return "参数异常变化"


# ============================================================================
# QA 生成核心逻辑
# ============================================================================

def generate_qa_for_sample(metadata: GenerationMetadata,
                            template_only: bool = True) -> List[Dict]:
    """
    为单条时序数据生成 5-6 个 QA 对

    Returns: list of {level, question, answer, reasoning_chain, required_interval}
    """
    rng = random.Random(hash(metadata.scenario_id) + metadata.total_length)
    qa_pairs = []

    has_fault = len(metadata.injected_events) > 0
    dur_desc = _duration_desc(metadata.total_length, metadata.sampling_seconds)

    fmt_kwargs = {
        "domain_display": metadata.domain_display,
        "total_length": metadata.total_length,
        "n_channels": len(metadata.channels),
        "duration_desc": dur_desc,
    }

    # === L1: 描述性推理 (1个) ===
    q_template = rng.choice(L1_TEMPLATES)
    question = q_template.format(**fmt_kwargs)

    if has_fault:
        ev = metadata.injected_events[0]
        answer = (
            f"{metadata.base_description} "
            f"但在第{ev.time_range[0]}-{ev.time_range[1]}步区间，"
            f"出现了{_severity_desc(ev.severity)}级{ev.fault_name}故障。"
            f"主要受影响通道：{', '.join(ev.affected_channels[:3])}。"
        )
    else:
        answer = metadata.base_description

    qa_pairs.append({
        "level": "L1",
        "question": question,
        "answer": answer,
        "reasoning_chain": ["观察各通道的全局趋势和波动水平",
                             "检查是否存在明显的偏离或异常区间",
                             "分析通道间的相关性是否符合物理规律"],
        "required_interval": None,
    })

    # === L2: 定位推理 (1-2个) ===
    if has_fault:
        ev = metadata.injected_events[0]
        q_template = rng.choice(L2_TEMPLATES_FAULT)
        question = q_template.format(**fmt_kwargs)

        answer = (
            f"是的，该设备存在异常。异常发生在第{ev.time_range[0]}步到第{ev.time_range[1]}步之间"
            f"（占总长度的{metadata.needle_ratio:.1%}）。"
            f"异常类型为{ev.fault_name}（{_severity_desc(ev.severity)}级）。"
            f"主要表现为：{ev.description}"
        )

        qa_pairs.append({
            "level": "L2",
            "question": question,
            "answer": answer,
            "reasoning_chain": [
                f"观察全局概览，前{ev.time_range[0]}步各参数运行平稳",
                f"在第{ev.time_range[0]}步附近，{ev.affected_channels[0]}出现明显变化",
                f"异常区间持续至第{ev.time_range[1]}步",
                f"异常占总监测时间的{metadata.needle_ratio:.1%}",
            ],
            "required_interval": list(ev.time_range),
        })
    else:
        q_template = rng.choice(L2_TEMPLATES_NORMAL)
        question = q_template.format(**fmt_kwargs)
        answer = (
            f"经过仔细检查全部{metadata.total_length}个时间步的"
            f"{len(metadata.channels)}个传感器通道，未发现明显异常。"
            f"各参数均在正常范围内波动，设备运行状态正常。"
        )
        qa_pairs.append({
            "level": "L2",
            "question": question,
            "answer": answer,
            "reasoning_chain": ["逐通道检查是否存在突变或趋势异常",
                                 "对比各通道的值域是否在正常范围内",
                                 "确认无需报告异常"],
            "required_interval": None,
        })

    # === L3: 因果推理 (1-2个, 仅有故障时) ===
    if has_fault:
        ev = metadata.injected_events[0]
        fmt_kwargs.update({
            "fault_start": ev.time_range[0],
            "primary_channel": ev.affected_channels[0] if ev.affected_channels else "未知",
            "change_desc": _change_desc(ev),
        })

        q_template = rng.choice(L3_TEMPLATES)
        question = q_template.format(**fmt_kwargs)

        causal_text = " → ".join(ev.causal_chain)
        answer = (
            f"故障的因果链为：{causal_text}。"
            f"受影响的传感器通道依次为：{', '.join(ev.affected_channels)}。"
            f"{'该故障是渐进式的，从趋势变化中可以识别。' if ev.event_type in ('fault',) and 'gradual' in ev.fault_name else ''}"
        )

        qa_pairs.append({
            "level": "L3",
            "question": question,
            "answer": answer,
            "reasoning_chain": ev.causal_chain,
            "required_interval": list(ev.time_range),
        })

        # 额外的 L3 问题：多变量对比
        if len(ev.affected_channels) >= 2:
            question2 = (
                f"在该{metadata.domain_display}的异常区间（第{ev.time_range[0]}-{ev.time_range[1]}步），"
                f"请对比{ev.affected_channels[0]}和{ev.affected_channels[1]}的变化模式，"
                f"分析它们的关联关系。"
            )
            answer2 = (
                f"{ev.affected_channels[0]}作为主要受影响通道，变化幅度更大且出现更早。"
                f"{ev.affected_channels[1]}的变化是由{ev.affected_channels[0]}通过物理耦合传导而来，"
                f"存在时间滞后和幅度衰减。"
                f"因果链：{causal_text}"
            )
            qa_pairs.append({
                "level": "L3",
                "question": question2,
                "answer": answer2,
                "reasoning_chain": ev.causal_chain,
                "required_interval": list(ev.time_range),
            })

    # === L4: 决策推理 (1个) ===
    if has_fault:
        ev = metadata.injected_events[0]
        fmt_kwargs["severity_desc"] = _severity_desc(ev.severity)

        q_template = rng.choice(L4_TEMPLATES)
        question = q_template.format(**fmt_kwargs)

        if ev.severity == "severe":
            urgency = "建议立即停机检修"
            risk = "高风险"
        elif ev.severity == "moderate":
            urgency = "建议在近期安排计划性停机检修"
            risk = "中等风险"
        else:
            urgency = "建议加强监测频率，在下次计划性检修时处理"
            risk = "低风险"

        answer = (
            f"设备健康状态评估：{risk}。"
            f"发现{_severity_desc(ev.severity)}级{ev.fault_name}故障，"
            f"影响了{', '.join(ev.affected_channels[:2])}等参数。"
            f"{urgency}。"
            f"检修重点：{ev.causal_chain[0]}。"
        )

        qa_pairs.append({
            "level": "L4",
            "question": question,
            "answer": answer,
            "reasoning_chain": [
                f"确认故障类型：{ev.fault_name}",
                f"评估严重程度：{ev.severity}",
                f"分析影响范围：{len(ev.affected_channels)}个通道受影响",
                f"给出处置建议：{urgency}",
            ],
            "required_interval": list(ev.time_range),
        })
    else:
        q_template = rng.choice(L4_TEMPLATES_NORMAL)
        question = q_template.format(**fmt_kwargs)
        answer = (
            f"设备当前运行正常，健康状态良好。各传感器参数均在正常范围内。"
            f"建议按照常规维护计划执行预防性检修，重点检查易损件的磨损情况。"
        )
        qa_pairs.append({
            "level": "L4",
            "question": question,
            "answer": answer,
            "reasoning_chain": ["确认所有通道在正常范围", "无异常需要处理",
                                 "给出预防性维护建议"],
            "required_interval": None,
        })

    return qa_pairs


# ============================================================================
# 批量 QA 生成
# ============================================================================

def batch_generate_qa(metadata_dir: str, output_path: str):
    """
    遍历目录下的所有 metadata.json，为每条生成 QA

    output_path: 输出的 JSONL 文件路径
    """
    metadata_files = []
    for root, dirs, files in os.walk(metadata_dir):
        for f in files:
            if f.endswith("_metadata.json"):
                metadata_files.append(os.path.join(root, f))

    print(f"Found {len(metadata_files)} metadata files")

    all_records = []
    for i, meta_path in enumerate(sorted(metadata_files)):
        try:
            metadata = GenerationMetadata.load(meta_path)
            qa_pairs = generate_qa_for_sample(metadata)

            # 关联到数据文件
            csv_path = meta_path.replace("_metadata.json", ".csv")
            relative_csv = os.path.relpath(csv_path, os.path.dirname(output_path))

            record = {
                "id": f"{metadata.domain}_{metadata.scenario_id}_L{metadata.total_length}_S{metadata.generation_seed}",
                "domain": metadata.domain,
                "source": "synthetic",
                "scenario_id": metadata.scenario_id,
                "metadata": {
                    "total_length": metadata.total_length,
                    "n_channels": len(metadata.channels),
                    "sampling_rate": metadata.sampling_rate,
                    "difficulty": metadata.difficulty,
                    "needle_ratio": metadata.needle_ratio,
                },
                "timeseries_path": relative_csv,
                "ground_truth_events": [
                    {
                        "type": ev.fault_name,
                        "range": list(ev.time_range),
                        "severity": ev.severity,
                        "affected_channels": ev.affected_channels,
                        "causal_chain": ev.causal_chain,
                    }
                    for ev in metadata.injected_events
                ],
                "qa_pairs": qa_pairs,
            }
            all_records.append(record)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(metadata_files)}")

        except Exception as e:
            print(f"  Error processing {meta_path}: {e}")

    # Save as JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    total_qa = sum(len(r["qa_pairs"]) for r in all_records)
    print(f"\nGenerated {total_qa} QA pairs for {len(all_records)} samples")
    print(f"Saved to: {output_path}")

    # Stats
    by_level = {}
    by_domain = {}
    for r in all_records:
        d = r["domain"]
        by_domain[d] = by_domain.get(d, 0) + len(r["qa_pairs"])
        for qa in r["qa_pairs"]:
            l = qa["level"]
            by_level[l] = by_level.get(l, 0) + 1

    print(f"\nBy domain: {by_domain}")
    print(f"By level: {by_level}")

    return all_records


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    batch_generate_qa(
        metadata_dir="Data/benchmark",
        output_path="Data/benchmark/synthetic_qa.jsonl"
    )
