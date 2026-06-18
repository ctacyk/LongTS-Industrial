# -*- coding: utf-8 -*-
"""
SFT Data Generator — 使用 Qwen3.5-Plus 根据 metadata 生成 L1-L4 训练数据
"""

import json
import os
import random
import time
import re
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ============================================================================
# 配置
# ============================================================================

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

SAMPLE_PER_DOMAIN = 75
NORMAL_PER_DOMAIN = 8
SEED = 42
MAX_WORKERS = 8
MAX_RETRIES = 3

OUTPUT_DIR = Path("Data/training/sft_v2")
TRAINING_SERIES_DIR = Path("Data/training_series")
GRPO_PARQUET = Path("Data/training/grpo_parquet/grpo_train_all.parquet")

DOMAIN_DISPLAY = {
    "coal_mill": "磨煤机",
    "pump": "水泵",
    "transformer": "变压器",
    "wind_turbine": "风力发电机组",
}

# ============================================================================
# L1-L4 User Question 模板（每个 Level 5 种变体）
# ============================================================================

L1_QUESTIONS = [
    "请描述该{device}在{length}个时间步内的整体运行状态，包括各传感器通道的主要变化趋势、波动范围和相互关系。",
    "观察该{device}的多通道传感器数据，描述各通道的运行状态和通道间的相互关系。",
    "请对该{device}的运行数据进行全面描述，分析各通道的波动特征和整体运行模式。",
    "该{device}在监测期间的各传感器通道表现如何？请综合描述各通道的数值范围和变化趋势。",
    "请分析该{device}各监测通道的数据特征，包括趋势、波动和通道间的耦合关系。",
]

L2_QUESTIONS = [
    "仔细分析全部{length}个时间步的数据，判断是否存在任何异常行为，并精确指出异常的时间区间（精确到步）。",
    "是否存在异常运行区间？如果存在，请精确指出异常开始和结束的时间步，并说明判断依据。",
    "请逐段检查该{device}的运行数据，是否能发现偏离正常工况的区间？如有请定位。",
    "该{device}的监测数据中是否隐藏着异常？请精确定位异常区间并说明识别依据。",
    "对该{device}的{length}步数据进行异常检测，判断设备是否全程正常运行。",
]

L3_QUESTIONS_FAULT = [
    "{channel_hint}出现异常变化，可能的原因是什么？请结合其他传感器数据推断故障根源及其传播路径。",
    "分析该{device}故障的根本原因，结合多通道数据说明因果关系链。",
    "请推断该{device}异常的根本原因，并描述故障从发生到影响各通道的传播路径。",
    "该{device}出现的异常背后，物理机制是什么？请从根因出发分析因果链。",
    "结合多通道数据的协同变化，推断该{device}故障的根源及其在各子系统间的传播过程。",
]

L3_QUESTIONS_NORMAL = [
    "该{device}各通道之间的协同关系是否正常？请分析各通道的物理耦合关系。",
    "请分析该{device}正常运行时各通道之间的相互影响关系。",
]

L4_QUESTIONS_FAULT = [
    "基于当前监测数据所反映的设备状态，作为运维工程师，应采取何种运维决策？是否需要停机？请给出具体建议及优先级。",
    "该{device}当前的运行状况是否需要干预？请给出分级运维建议。",
    "作为运维工程师，请根据该{device}的监测数据制定维护策略，包括是否需要停机和具体措施。",
    "请评估该{device}的当前状态风险等级，并给出运维决策建议和优先级排序。",
    "基于当前数据，该{device}应执行什么级别的维护？请给出具体的操作建议。",
]

L4_QUESTIONS_NORMAL = [
    "基于当前监测数据所反映的设备状态，作为运维工程师，应采取何种运维策略？",
    "该{device}当前运行正常，请给出预防性维护建议。",
]

# ============================================================================
# SFT 最终数据中的 System Prompt（简短）
# ============================================================================

SFT_SYSTEM_PROMPT = "你是一位工业设备时序数据分析专家。请仔细观察设备运行数据的可视化图像，进行专业的诊断分析。"

# ============================================================================
# Qwen3.5-Plus 的生成 System Prompt
# ============================================================================

GEN_SYSTEM_PROMPT = """你是一个工业时序数据诊断培训数据生成器。

你会收到一个设备的传感器通道定义和真实标注信息（包括故障类型、时间区间、因果链等）。
请你扮演一位正在分析该设备多通道时序监测图的高级工程师，根据给定的标注信息，生成一段专业的诊断分析回答。

## 要求
1. 回答要像真正在看多通道时序图一样，描述观察到的趋势、突变、漂移等视觉现象
2. 引用具体的时间步和通道名称
3. 回答直接给出分析内容，不要加任何前缀标签（不要<think>、<answer>等）
4. 回答长度适中（100-400字），自然流畅
5. 不要输出统计数据（均值、标准差、方差等）
6. 不要使用markdown标题格式（不要**粗体**、##标题等）
7. 针对不同类型的问题，调整回答风格：
   - 整体描述类：分段描述各通道状态
   - 异常定位类：明确指出时间区间和依据
   - 因果分析类：从根因出发描述传播路径
   - 运维建议类：给出分级的具体措施"""

# ============================================================================
# 工具函数
# ============================================================================

def load_metadata(meta_path: str) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_channels_desc(channels: list) -> str:
    lines = []
    for ch in channels:
        name = ch["name"]
        display = ch.get("display_name", name)
        unit = ch.get("unit", "")
        nr = ch.get("normal_range", [])
        alarm = ch.get("alarm_threshold", "")
        meaning = ch.get("physical_meaning", "")
        nr_str = f"正常范围{nr[0]}-{nr[1]}{unit}" if len(nr) == 2 else ""
        alarm_str = f"报警阈值{alarm}{unit}" if alarm else ""
        lines.append(f"- {name}({display}): {meaning}。{nr_str}，{alarm_str}")
    return "\n".join(lines)


def format_event_desc(event: dict) -> str:
    parts = []
    parts.append(f"故障类型: {event.get('fault_name', '未知')}")
    tr = event.get("time_range", [])
    if tr:
        parts.append(f"故障区间: 第{tr[0]}步 至 第{tr[1]}步")
    parts.append(f"严重程度: {event.get('severity', '未知')}")
    ac = event.get("affected_channels", [])
    if ac:
        parts.append(f"受影响通道: {', '.join(ac)}")
    cc = event.get("causal_chain", [])
    if cc:
        parts.append(f"因果链: {' → '.join(cc)}")
    desc = event.get("description", "")
    if desc:
        parts.append(f"描述: {desc}")
    return "\n".join(parts)


def build_gen_prompt(meta: dict, level: str, question: str) -> str:
    """构建给 Qwen3.5-Plus 的生成 prompt"""
    domain_display = meta.get("domain_display", meta.get("domain", "设备"))
    total_length = meta.get("total_length", 0)
    channels = meta.get("channels", [])
    events = meta.get("injected_events", [])

    ch_desc = format_channels_desc(channels)

    has_fault = any(e.get("event_type") == "fault" for e in events)
    fault_events = [e for e in events if e.get("event_type") == "fault"]

    prompt_parts = [
        f"设备类型: {domain_display}",
        f"数据长度: {total_length}步",
        f"\n通道定义:\n{ch_desc}",
    ]

    if has_fault and fault_events:
        prompt_parts.append("\n真实标注（故障信息）:")
        for ev in fault_events:
            prompt_parts.append(format_event_desc(ev))
    else:
        prompt_parts.append("\n真实标注: 设备全程正常运行，无任何故障。")

    prompt_parts.append(f"\n需要回答的问题（{level}级别）:\n{question}")
    prompt_parts.append("\n请根据以上信息，生成专业的诊断分析回答。")

    return "\n".join(prompt_parts)


def pick_question(level: str, meta: dict) -> str:
    """随机选择一个 user question 变体"""
    domain_display = meta.get("domain_display", meta.get("domain", "设备"))
    total_length = meta.get("total_length", 0)
    events = meta.get("injected_events", [])
    has_fault = any(e.get("event_type") == "fault" for e in events)
    fault_events = [e for e in events if e.get("event_type") == "fault"]

    fmt_kwargs = {"device": domain_display, "length": total_length}

    if level == "L1":
        template = random.choice(L1_QUESTIONS)
    elif level == "L2":
        template = random.choice(L2_QUESTIONS)
    elif level == "L3":
        if has_fault and fault_events:
            template = random.choice(L3_QUESTIONS_FAULT)
            ac = fault_events[0].get("affected_channels", [])
            if ac:
                ch_name = ac[0]
                for ch in meta.get("channels", []):
                    if ch["name"] == ch_name:
                        fmt_kwargs["channel_hint"] = ch.get("display_name", ch_name)
                        break
                else:
                    fmt_kwargs["channel_hint"] = ch_name
            else:
                fmt_kwargs["channel_hint"] = "部分通道"
        else:
            template = random.choice(L3_QUESTIONS_NORMAL)
    elif level == "L4":
        if has_fault:
            template = random.choice(L4_QUESTIONS_FAULT)
        else:
            template = random.choice(L4_QUESTIONS_NORMAL)
    else:
        template = "请分析该{device}的运行状态。"

    return template.format(**fmt_kwargs)


# ============================================================================
# API 调用
# ============================================================================

client = None

def get_client():
    global client
    if client is None:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return client


def call_llm(gen_prompt: str, retries: int = MAX_RETRIES) -> str:
    c = get_client()
    for attempt in range(retries):
        try:
            resp = c.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": gen_prompt},
                ],
                temperature=0.7,
                max_tokens=800,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [Retry {attempt+1}/{retries}] API error: {e}")
            time.sleep(2 ** attempt)
    return ""


# ============================================================================
# 质量检查
# ============================================================================

def quality_check(response: str, level: str, meta: dict) -> bool:
    if not response or len(response) < 30:
        return False
    if "<tool_call>" in response or "analyze_interval" in response:
        return False
    if "均值" in response and "标准差" in response:
        return False
    if "<think>" in response or "<answer>" in response:
        return False
    if response.startswith("```") or "```" in response:
        return False

    events = meta.get("injected_events", [])
    has_fault = any(e.get("event_type") == "fault" for e in events)

    if level == "L2":
        if has_fault:
            fault_events = [e for e in events if e.get("event_type") == "fault"]
            tr = fault_events[0].get("time_range", [])
            if tr:
                if str(tr[0]) not in response and str(tr[1]) not in response:
                    if "异常" not in response and "故障" not in response:
                        return False
        else:
            if "正常" not in response and "未发现" not in response and "无异常" not in response:
                return False

    return True


# ============================================================================
# 主流程
# ============================================================================

def select_images():
    """从 SFT-only 图片中采样 300 张"""
    import pandas as pd

    grpo_imgs = set()
    if GRPO_PARQUET.exists():
        df = pd.read_parquet(GRPO_PARQUET)
        for val in df["image_paths"]:
            parsed = json.loads(val) if isinstance(val, str) else val
            if isinstance(parsed, list):
                grpo_imgs.update(parsed)

    sft_only_by_domain = defaultdict(list)
    for domain in DOMAIN_DISPLAY:
        domain_dir = TRAINING_SERIES_DIR / domain
        if not domain_dir.exists():
            continue
        for f in sorted(domain_dir.iterdir()):
            if not f.name.endswith("_eval.png"):
                continue
            img_path = f"Data/training_series/{domain}/{f.name}"
            if img_path in grpo_imgs:
                continue
            meta_path = str(f).replace("_eval.png", "_metadata.json")
            if not os.path.exists(meta_path):
                continue
            sft_only_by_domain[domain].append((img_path, meta_path))

    random.seed(SEED)
    selected = []
    for domain in DOMAIN_DISPLAY:
        pool = sft_only_by_domain[domain]
        normal = [x for x in pool if "normal" in os.path.basename(x[0])]
        fault = [x for x in pool if "normal" not in os.path.basename(x[0])]

        n_normal = min(NORMAL_PER_DOMAIN, len(normal))
        n_fault = min(SAMPLE_PER_DOMAIN - n_normal, len(fault))

        chosen_normal = random.sample(normal, n_normal) if normal else []
        chosen_fault = random.sample(fault, n_fault) if fault else []
        selected.extend(chosen_normal + chosen_fault)
        print(f"  {domain}: {n_fault} fault + {n_normal} normal = {n_fault + n_normal}")

    print(f"  Total selected: {len(selected)}")
    return selected


def generate_one_sample(img_path, meta_path, level):
    """生成单条 SFT 样本"""
    meta = load_metadata(meta_path)
    question = pick_question(level, meta)
    gen_prompt = build_gen_prompt(meta, level, question)

    response = call_llm(gen_prompt)

    if not quality_check(response, level, meta):
        response = call_llm(gen_prompt)
        if not quality_check(response, level, meta):
            return None

    sft_record = {
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\n{question}"},
            {"role": "assistant", "content": response},
        ],
        "images": [img_path.replace("\\", "/")],
    }
    return sft_record


def main():
    print("=" * 60)
    print("SFT Data Generator v2")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Selecting images...")
    selected = select_images()

    print(f"\n[2/4] Generating {len(selected) * 4} SFT samples (4 levels × {len(selected)} images)...")
    print(f"  Model: {MODEL}, Workers: {MAX_WORKERS}")

    tasks = []
    for img_path, meta_path in selected:
        for level in ["L1", "L2", "L3", "L4"]:
            tasks.append((img_path, meta_path, level))

    random.shuffle(tasks)

    results = []
    failed = 0
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for img_path, meta_path, level in tasks:
            future = executor.submit(generate_one_sample, img_path, meta_path, level)
            futures[future] = (img_path, level)

        for future in as_completed(futures):
            done += 1
            img_path, level = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"  ERROR: {img_path} {level}: {e}")

            if done % 50 == 0:
                print(f"  Progress: {done}/{len(tasks)}, Success: {len(results)}, Failed: {failed}")

    print(f"\n  Done! Total: {len(results)} success, {failed} failed")

    print("\n[3/4] Splitting train/val (9:1)...")
    random.shuffle(results)
    split = int(len(results) * 0.9)
    train_data = results[:split]
    val_data = results[split:]

    train_path = OUTPUT_DIR / "sft_train_v2.jsonl"
    val_path = OUTPUT_DIR / "sft_val_v2.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for rec in train_data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for rec in val_data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Train: {len(train_data)} → {train_path}")
    print(f"  Val:   {len(val_data)} → {val_path}")

    print("\n[4/4] Statistics...")
    level_counts = defaultdict(int)
    domain_counts = defaultdict(int)
    for rec in results:
        img = rec["images"][0]
        for d in DOMAIN_DISPLAY:
            if d in img:
                domain_counts[d] += 1
                break
        q = rec["messages"][1]["content"]
        if "整体" in q or "描述" in q or "趋势" in q or "波动" in q:
            level_counts["L1"] += 1
        elif "异常" in q or "定位" in q or "偏离" in q:
            level_counts["L2"] += 1
        elif "原因" in q or "根因" in q or "因果" in q or "传播" in q or "协同" in q or "耦合" in q:
            level_counts["L3"] += 1
        elif "运维" in q or "决策" in q or "维护" in q or "停机" in q or "建议" in q:
            level_counts["L4"] += 1
        else:
            level_counts["other"] += 1

    print("  By domain:")
    for d, c in sorted(domain_counts.items()):
        print(f"    {d}: {c}")
    print("  By level:")
    for l, c in sorted(level_counts.items()):
        print(f"    {l}: {c}")

    avg_len = sum(len(r["messages"][2]["content"]) for r in results) / max(len(results), 1)
    print(f"  Avg response length: {avg_len:.0f} chars")

    print("\n" + "=" * 60)
    print(f"SFT v2 data generated successfully!")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
