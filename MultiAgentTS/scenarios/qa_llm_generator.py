"""
LLM 驱动的 QA 生成器

使用阿里 DashScope API (qwen3-max) 基于 metadata.json 生成自然语言 QA 对。
每条时序数据生成 5-6 个 L1-L4 层级的开放式推理问题和详细答案。

API 配置：
- base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
- model: qwen3-max
- API Key: 从 configs/config.yaml 读取
"""

import json
import os
import sys
import time
import traceback
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from MultiAgentTS.scenarios.base import GenerationMetadata


# ============================================================================
# LLM QA Generator
# ============================================================================

class LLMQAGenerator:
    """使用 LLM 生成高质量 QA 对"""

    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model: str = "qwen3-max", temperature: float = 0.7, max_tokens: int = 4096):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_qa(self, metadata: GenerationMetadata,
                    data_profile: Optional[Dict] = None) -> List[Dict]:
        """为单条时序数据生成 5-6 个 QA 对"""

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(metadata, data_profile)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content

            # 解析 JSON（LLM 可能在 ```json ``` 中返回）
            qa_pairs = self._parse_response(content)
            return qa_pairs

        except Exception as e:
            print(f"  LLM error for {metadata.scenario_id}: {e}")
            return []

    def _build_system_prompt(self) -> str:
        return """你是一位工业时序数据分析领域的资深专家，同时也是一位优秀的考试题目出题人。

你的任务是为工业设备的时序监测数据生成高质量的推理问答题（QA对），用于评测大型语言模型对工业时序数据的理解和推理能力。

你需要生成4个层级的问题，每个层级代表不同深度的推理能力：

**L1 - 描述性推理**：考察模型对时序数据整体模式的理解
- 要求模型描述设备运行的全局特征（趋势、波动、周期性、通道间关系）
- 问题应该自然、专业，像运维工程师的日常工作对话
- 答案应详细描述各通道的统计特征和运行模式

**L2 - 定位推理（大海捞针）**：考察模型在长序列中精确定位异常的能力
- 要求模型判断是否存在异常，并精确给出异常的时间步范围
- 这是最核心的能力：在几千甚至上万步的数据中找到占比很小的异常区间
- 答案必须包含精确的时间步范围和异常在各通道上的具体表现

**L3 - 因果推理**：考察模型分析故障根因和多变量关联的能力
- 要求模型分析异常的物理原因，解释传感器间的因果传播路径
- 问题应引导模型进行多步推理（如：温度升高 → 为什么？ → 哪些证据支持？）
- 答案应包含完整的因果链和支撑证据

**L4 - 决策推理**：考察模型给出运维决策建议的能力
- 要求模型基于数据分析结果给出具体的处置建议
- 应包含风险评估、优先级判断、具体措施
- 答案应体现工程师的专业判断

输出格式要求：严格按照JSON数组格式输出，不要包含任何其他内容。
每个QA对包含：level, question, answer, reasoning_chain(推理步骤列表), required_interval(异常区间[start,end]或null)

重要：
- 问题措辞要自然多样，不要使用模板化的句式
- 答案中的数值必须与"实际数据统计"中提供的真实统计量一致。当提供了正常/故障区间的 mean/std/min/max 等数据时，答案中引用的具体数值必须来自这些统计，不得凭空编造
- 因果链要符合物理规律
- 对于正常运行的数据（无故障），也要生成L1、L2（判断无异常）和L4（预防性建议）
- 不要使用英文变量名如 seal_leak，要用自然的中文描述如 密封泄漏
- L1 答案需描述各通道在正常区间和故障区间的统计变化（如"电流从均值50A升至67A"），确保数值准确
- L2 答案的时间步范围必须与注入事件的 time_range 一致
"""

    def _build_user_prompt(self, metadata: GenerationMetadata,
                           data_profile: Optional[Dict] = None) -> str:
        """构建包含完整元数据 + 实际数据统计的用户提示"""

        # 通道描述
        channel_desc = ""
        for ch in metadata.channels:
            channel_desc += (
                f"  - {ch.name}（{ch.display_name}）：{ch.physical_meaning}，"
                f"单位{ch.unit}，正常范围{ch.normal_range[0]}-{ch.normal_range[1]}{ch.unit}"
            )
            if ch.alarm_threshold:
                channel_desc += f"，报警阈值{ch.alarm_threshold}{ch.unit}"
            channel_desc += "\n"

        # 事件描述
        if metadata.injected_events:
            events_desc = ""
            for i, ev in enumerate(metadata.injected_events):
                events_desc += f"""
事件{i+1}：
  - 类型：{ev.fault_name}
  - 严重程度：{ev.severity}
  - 时间范围：第{ev.time_range[0]}步 到 第{ev.time_range[1]}步
  - 受影响通道：{', '.join(ev.affected_channels)}
  - 参数：{json.dumps(ev.parameters, ensure_ascii=False)}
  - 因果链：
    {'→ '.join(ev.causal_chain)}
  - 描述：{ev.description}
"""
        else:
            events_desc = "  无异常事件，设备全程正常运行。\n"

        # 实际数据统计（来自 data_profile）
        data_stats_section = ""
        if data_profile and "channels" in data_profile:
            data_stats_section = "\n## 实际数据统计（来自真实生成数据，答案数值必须基于以下统计量）\n\n"
            for ch_name, ch_data in data_profile["channels"].items():
                ch_display = ch_name
                for ch in metadata.channels:
                    if ch.name == ch_name:
                        ch_display = f"{ch_name}（{ch.display_name}）"
                        break

                line = f"**{ch_display}**：\n"

                if "normal" in ch_data:
                    ns = ch_data["normal"]
                    line += f"  - 正常区间：均值={ns['mean']}, 标准差={ns['std']}, 范围=[{ns['min']}, {ns['max']}]\n"

                if "fault" in ch_data:
                    fs = ch_data["fault"]
                    line += f"  - 故障区间：均值={fs['mean']}, 标准差={fs['std']}, 范围=[{fs['min']}, {fs['max']}]\n"

                if "change" in ch_data:
                    ch_change = ch_data["change"]
                    affected_tag = " [受影响通道]" if ch_change.get("is_affected") else ""
                    line += f"  - 变化趋势{affected_tag}：{ch_change['description']}\n"

                data_stats_section += line + "\n"

        # 持续时间
        total_seconds = metadata.total_length * metadata.sampling_seconds
        if total_seconds < 3600:
            dur = f"{total_seconds // 60}分钟"
        elif total_seconds < 86400:
            dur = f"{total_seconds / 3600:.1f}小时"
        else:
            dur = f"{total_seconds / 86400:.1f}天"

        prompt = f"""请为以下工业设备的时序监测数据生成高质量的QA对。

## 设备信息

- 设备类型：{metadata.domain_display}
- 场景名称：{metadata.scenario_name}
- 数据总长度：{metadata.total_length}个时间步（约{dur}）
- 采样间隔：{metadata.sampling_rate}
- 传感器通道（{len(metadata.channels)}个）：
{channel_desc}

## 基础运行状态

{metadata.base_description}

## 注入的异常事件

{events_desc}
{data_stats_section}
## 难度信息

- 难度等级：{metadata.difficulty}
- 异常占比：{metadata.needle_ratio:.1%}（异常区间长度 / 总长度）

## 要求

请生成{'5-6' if metadata.injected_events else '3'}个QA对：
- L1（描述性推理）：1个
- L2（定位推理）：1个{'（需要精确定位异常区间）' if metadata.injected_events else '（判断为正常）'}
- L3（因果推理）：{'1-2个（分析故障原因和传播路径）' if metadata.injected_events else '跳过（无异常事件）'}
- L4（决策推理）：1个

⚠️ 重要：答案中的所有数值（均值、变化量、范围等）必须直接来自上面"实际数据统计"中的数值，不得虚构。

请直接输出JSON数组，格式如下：
```json
[
  {{
    "level": "L1",
    "question": "...",
    "answer": "...",
    "reasoning_chain": ["步骤1", "步骤2", ...],
    "required_interval": null
  }},
  ...
]
```"""
        return prompt

    def _parse_response(self, content: str) -> List[Dict]:
        """解析 LLM 返回的 JSON"""
        # 移除 markdown 代码块标记
        content = content.strip()

        # 尝试提取 ```json ... ``` 中的内容
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()

        # 尝试找到 JSON 数组
        if not content.startswith("["):
            bracket_start = content.find("[")
            if bracket_start >= 0:
                content = content[bracket_start:]
            # 找到最后一个 ]
            bracket_end = content.rfind("]")
            if bracket_end >= 0:
                content = content[:bracket_end + 1]

        try:
            qa_pairs = json.loads(content)
            if isinstance(qa_pairs, list):
                # 验证格式
                valid_pairs = []
                for qa in qa_pairs:
                    if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                        # 确保有 level
                        if "level" not in qa:
                            qa["level"] = "L1"
                        if "reasoning_chain" not in qa:
                            qa["reasoning_chain"] = []
                        if "required_interval" not in qa:
                            qa["required_interval"] = None
                        valid_pairs.append(qa)
                return valid_pairs
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            print(f"  Content preview: {content[:200]}...")

        return []


# ============================================================================
# 批量生成
# ============================================================================

def batch_generate_qa_llm(metadata_dir: str, output_path: str,
                           api_key: str = None, model: str = "qwen3-max",
                           base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                           rpm_limit: int = 10,
                           skip_existing: bool = True):
    """
    使用 LLM 批量生成 QA

    Args:
        metadata_dir: metadata.json 文件所在目录
        output_path: 输出 JSONL 路径
        api_key: DashScope API Key
        model: 模型名称
        rpm_limit: 每分钟最大请求数
        skip_existing: 是否跳过已存在的输出
    """
    # 加载 API Key
    if api_key is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "configs", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            api_key = config.get("api", {}).get("api_key", "")
            print(f"Loaded API key from {config_path}")
        else:
            raise ValueError("No API key provided and config.yaml not found")

    generator = LLMQAGenerator(api_key=api_key, base_url=base_url, model=model)

    # 收集所有 metadata 文件
    metadata_files = []
    for root, dirs, files in os.walk(metadata_dir):
        for f in files:
            if f.endswith("_metadata.json"):
                metadata_files.append(os.path.join(root, f))

    metadata_files.sort()
    print(f"Found {len(metadata_files)} metadata files")
    print(f"Using model: {model}")
    print(f"RPM limit: {rpm_limit}")

    # 加载已有结果（用于断点续传）
    existing_ids = set()
    if skip_existing and os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        existing_ids.add(record.get("id", ""))
                    except:
                        pass
        print(f"Found {len(existing_ids)} existing records (will skip)")

    # 生成
    total = len(metadata_files)
    generated = 0
    skipped = 0
    errors = 0
    total_qa = 0
    start_time = time.time()
    request_times = []  # 用于 RPM 限制

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'a', encoding='utf-8') as out_f:
        for i, meta_path in enumerate(metadata_files):
            try:
                metadata = GenerationMetadata.load(meta_path)

                # 构建 ID
                record_id = (f"{metadata.domain}_{metadata.scenario_id}"
                            f"_L{metadata.total_length}_S{metadata.generation_seed}")

                # 跳过已存在
                if record_id in existing_ids:
                    skipped += 1
                    continue

                # RPM 限制
                now = time.time()
                request_times = [t for t in request_times if now - t < 60]
                if len(request_times) >= rpm_limit:
                    wait = 60 - (now - request_times[0]) + 0.5
                    print(f"  RPM limit reached, waiting {wait:.1f}s...")
                    time.sleep(wait)

                # 加载 data_profile
                profile_path = meta_path.replace("_metadata.json", "_data_profile.json")
                data_profile = None
                if os.path.exists(profile_path):
                    with open(profile_path, 'r', encoding='utf-8') as pf:
                        data_profile = json.load(pf)

                # 调用 LLM 生成 QA
                request_times.append(time.time())
                qa_pairs = generator.generate_qa(metadata, data_profile=data_profile)

                if not qa_pairs:
                    errors += 1
                    print(f"  [{i+1}/{total}] ERROR: No QA generated for {record_id}")
                    continue

                # 构建记录
                csv_path = meta_path.replace("_metadata.json", ".csv")
                relative_csv = os.path.relpath(csv_path, os.path.dirname(output_path))

                record = {
                    "id": record_id,
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

                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                out_f.flush()

                n_qa = len(qa_pairs)
                total_qa += n_qa
                generated += 1

                elapsed = time.time() - start_time
                rate = (generated + skipped) / elapsed if elapsed > 0 else 0

                if generated % 10 == 0 or generated <= 3:
                    print(f"  [{i+1}/{total}] {record_id}: {n_qa} QA | "
                          f"Total: {total_qa} QA | "
                          f"({rate:.1f}/s, ETA {(total - i - 1) / max(rate, 0.01):.0f}s)")

            except Exception as e:
                errors += 1
                print(f"  [{i+1}/{total}] ERROR: {meta_path}: {e}")
                traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n=== LLM QA Generation Done ===")
    print(f"Generated: {generated} records, {total_qa} QA pairs")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", default="Data/benchmark")
    parser.add_argument("--output", default="Data/benchmark/synthetic_qa_llm.jsonl")
    parser.add_argument("--model", default="qwen3-max")
    parser.add_argument("--rpm", type=int, default=10)
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    batch_generate_qa_llm(
        metadata_dir=args.metadata_dir,
        output_path=args.output,
        model=args.model,
        rpm_limit=args.rpm,
        skip_existing=not args.no_skip,
    )
