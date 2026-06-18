"""
LongTS-Industrial Benchmark: VLM + analyze_interval Tool 评测

在原 run_benchmark_eval.py 基础上增加 tool-calling 循环：
  1. 发送 eval.png + 问题 + tool_definitions 给 VLM
  2. 如果 VLM 请求 analyze_interval，从 CSV 计算多通道区间统计并返回
  3. 迭代最多 MAX_TOOL_ITERS 轮，直到 VLM 给出最终答案

结果保存到独立目录 (默认 qwen3-vl-32b-instruct-tool)，不覆盖原有结果。
"""

import os
import sys
import json
import base64
import time
import hashlib
import argparse
import threading
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ============================================================
# Configuration
# ============================================================
_DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
_DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

API_KEY = os.environ.get("VLM_API_KEY", _DASHSCOPE_KEY)
BASE_URL = os.environ.get("VLM_BASE_URL", _DASHSCOPE_URL)

MODEL = "qwen3-vl-32b-instruct"
OUTPUT_NAME = "qwen3-vl-32b-instruct-tool"

BENCHMARK_DIR = "Data/benchmark_tb/LongTS_Industrial"
DATA_DIR = "Data/benchmark_tb"
RESULTS_DIR = "results/benchmark_eval_v2"

RPM = 300
MAX_TOKENS = 2048
TEMPERATURE = 0.3
MAX_TOOL_ITERS = 2
MAX_WORKERS = 10

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_interval",
            "description": (
                "深入分析指定时序区间。返回该区间内所有通道的详细统计特征"
                "（均值、标准差、最小值、最大值、趋势斜率）。"
                "当你需要精确了解某个时间段各通道的数值特征时调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "integer",
                        "description": "区间起始索引（包含），从0开始"
                    },
                    "end": {
                        "type": "integer",
                        "description": "区间结束索引（不包含）"
                    },
                    "reason": {
                        "type": "string",
                        "description": "选择分析这个区间的原因"
                    }
                },
                "required": ["start", "end"]
            }
        }
    }
]

SYSTEM_PROMPT = (
    "你是一位工业设备时序数据分析专家。你将看到一张工业设备多通道传感器的时序数据可视化图。"
    "图中每个子图代表一个传感器通道，横轴是时间步，纵轴是传感器读数。"
    "请仔细观察所有通道的数据模式，然后根据用户的问题进行分析和回答。"
    "你的回答应该专业、具体，包含对时间步范围、数值变化、通道间关联的精确描述。"
    "\n\n你可以使用 analyze_interval 工具来深入分析感兴趣的时间区间。"
    "该工具会返回区间内所有通道的详细统计信息（均值、标准差、趋势等）。"
    "你可以调用此工具来获取精确数值，辅助你的分析判断。"
)


# ============================================================
# Utilities
# ============================================================
def _log(path, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def compute_interval_stats(df: pd.DataFrame, start: int, end: int) -> dict:
    """计算区间内所有通道的统计信息"""
    start = max(0, start)
    end = min(len(df), end)
    seg = df.iloc[start:end]
    stats = {}
    for ch in df.columns:
        arr = seg[ch].values
        if len(arr) == 0:
            continue
        slope = float(np.polyfit(np.arange(len(arr)), arr, 1)[0]) if len(arr) > 1 else 0.0
        stats[ch] = {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "slope": round(slope, 6),
        }
    return stats


def resume_key(rec: dict) -> str:
    qh = hashlib.md5(rec.get("question", "").encode()).hexdigest()[:8]
    return rec.get("id", "") + "_" + rec.get("level", "") + "_" + qh


# ============================================================
# Load benchmark samples
# ============================================================
def load_benchmark_samples(benchmark_dir: str, data_dir: str,
                           split: str = "test") -> list:
    jsonl_path = os.path.join(benchmark_dir, f"{split}.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return []

    samples = []
    missing = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("source") == "enginemt_qa":
                continue

            ts_path = record.get("timeseries_path", "")
            if not ts_path:
                continue
            ts_path = ts_path.replace("\\", "/")
            if ts_path.startswith("../"):
                ts_path = ts_path[3:]

            csv_path = os.path.join(data_dir, ts_path)
            eval_png = csv_path.replace(".csv", "_eval.png")

            if not os.path.exists(eval_png):
                missing += 1
                continue

            for qa in record.get("qa_pairs", []):
                samples.append({
                    "id": record["id"],
                    "level": qa.get("level", "?"),
                    "domain": record.get("domain", "?"),
                    "difficulty": record.get("metadata", {}).get("difficulty", "?"),
                    "question": qa.get("question", ""),
                    "answer_gt": qa.get("answer", ""),
                    "image_path": eval_png,
                    "csv_path": csv_path,
                })

    if missing > 0:
        print(f"Warning: {missing} records missing eval.png")
    print(f"Loaded {len(samples)} QA samples from {split}.jsonl")
    return samples


# ============================================================
# Tool-augmented inference for a single sample
# ============================================================
def infer_one_sample(client: OpenAI, sample: dict, log_path: str,
                     max_tool_iters: int = MAX_TOOL_ITERS) -> dict:
    """单样本推理：图片 + 问题 + tool-calling 循环"""
    image_b64 = encode_image(sample["image_path"])

    # 加载 CSV（缓存由调用者管理）
    df = pd.read_csv(sample["csv_path"])
    channels = list(df.columns)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": (
                f"数据包含 {len(channels)} 个通道: {', '.join(channels)}\n"
                f"总长度: {len(df)} 个时间步\n\n"
                f"问题: {sample['question']}"
            )},
        ]},
    ]

    tool_calls_log = []

    for iteration in range(max_tool_iters):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        choice = resp.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            return {
                "prediction": msg.content or "",
                "tool_calls": tool_calls_log,
                "iterations": iteration + 1,
            }

        # 处理 tool calls
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            tool_calls_log.append({"name": fn_name, "args": fn_args})

            if fn_name == "analyze_interval":
                start_idx = fn_args.get("start", 0)
                end_idx = fn_args.get("end", len(df))
                reason = fn_args.get("reason", "")
                ch_stats = compute_interval_stats(df, start_idx, end_idx)
                result_text = json.dumps({
                    "interval": [start_idx, end_idx],
                    "length": min(end_idx, len(df)) - max(start_idx, 0),
                    "reason": reason,
                    "channel_statistics": ch_stats,
                }, ensure_ascii=False)
            else:
                result_text = json.dumps({"error": f"Unknown tool: {fn_name}"})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

    # 达到最大迭代，强制无工具获取答案
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return {
        "prediction": resp.choices[0].message.content or "",
        "tool_calls": tool_calls_log,
        "iterations": max_tool_iters + 1,
    }


# ============================================================
# Batch inference with concurrency + checkpointing
# ============================================================
def run_inference(samples: list, output_dir: str, rpm: int, max_workers: int,
                  max_tool_iters: int = MAX_TOOL_ITERS):
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "predictions.jsonl")
    log_path = os.path.join(output_dir, "inference_log.txt")

    existing_ids = set()
    if os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        existing_ids.add(resume_key(r))
                    except:
                        pass

    pending = [s for s in samples if resume_key(s) not in existing_ids]

    _log(log_path, f"Model: {MODEL} + analyze_interval tool")
    _log(log_path, f"Total: {len(samples)}, Existing: {len(existing_ids)}, Pending: {len(pending)}")
    _log(log_path, f"Workers: {max_workers}, RPM: {rpm}, MaxToolIters: {max_tool_iters}")

    if not pending:
        _log(log_path, "All samples already completed. Skipping.")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    write_lock = threading.Lock()
    rate_lock = threading.Lock()
    last_req = [0.0]
    min_interval = 60.0 / rpm
    completed = [0]
    errors = [0]
    total_tool_calls = [0]
    start_time = time.time()

    def acquire_rate():
        with rate_lock:
            now = time.time()
            wait = last_req[0] + min_interval - now
            if wait > 0:
                time.sleep(wait)
            last_req[0] = time.time()

    def _process(sample):
        try:
            acquire_rate()
            result = infer_one_sample(client, sample, log_path, max_tool_iters)

            rec = {
                "id": sample["id"],
                "level": sample["level"],
                "domain": sample["domain"],
                "difficulty": sample["difficulty"],
                "question": sample["question"],
                "answer_gt": sample["answer_gt"],
                "prediction": result["prediction"],
                "model": MODEL,
                "tool_calls": result["tool_calls"],
                "iterations": result["iterations"],
            }

            with write_lock:
                with open(pred_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                completed[0] += 1
                total_tool_calls[0] += len(result["tool_calls"])

                if completed[0] % 20 == 0 or completed[0] <= 3:
                    elapsed = time.time() - start_time
                    rate = completed[0] / (elapsed / 60) if elapsed > 0 else 0
                    eta = (len(pending) - completed[0]) / max(rate, 0.01)
                    avg_tc = total_tool_calls[0] / completed[0]
                    _log(log_path,
                         f"  [{completed[0]}/{len(pending)}] "
                         f"{rate:.1f}/min | avgToolCalls={avg_tc:.1f} | "
                         f"ETA {eta:.0f}min | errors={errors[0]}")

        except Exception as e:
            errors[0] += 1
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(10)
            if errors[0] <= 10:
                _log(log_path, f"  [ERROR] {sample['id']}: {str(e)[:200]}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process, s) for s in pending]
        for f in as_completed(futures):
            f.result()

    elapsed = time.time() - start_time
    avg_tc = total_tool_calls[0] / max(completed[0], 1)
    _log(log_path,
         f"\nDone: completed={completed[0]}, errors={errors[0]}, "
         f"avgToolCalls={avg_tc:.1f}, time={elapsed / 60:.1f}min")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="LongTS-Industrial Benchmark: VLM + Tool Evaluation")
    parser.add_argument("--split", default="test")
    parser.add_argument("--benchmark-dir", default=BENCHMARK_DIR)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    parser.add_argument("--rpm", type=int, default=RPM)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-tool-iters", type=int, default=MAX_TOOL_ITERS)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip inference, only run judge evaluation")
    args = parser.parse_args()

    max_tool_iters = args.max_tool_iters
    output_dir = os.path.join(args.results_dir, args.output_name)

    print("=" * 70)
    print("LongTS-Industrial Benchmark: VLM + Tool Evaluation")
    print(f"  Model:         {MODEL}")
    print(f"  Output:        {output_dir}")
    print(f"  MaxToolIters:  {max_tool_iters}")
    print(f"  RPM:           {args.rpm}")
    print(f"  Workers:       {args.workers}")
    print("=" * 70)

    if not args.eval_only:
        samples = load_benchmark_samples(
            args.benchmark_dir, args.data_dir, args.split)
        if not samples:
            print("No samples loaded. Check paths.")
            return
        run_inference(samples, output_dir, args.rpm, args.workers, max_tool_iters)

    # Run judge evaluation
    pred_path = os.path.join(output_dir, "predictions.jsonl")
    if os.path.exists(pred_path):
        print("\n" + "=" * 70)
        print("Running Judge Evaluation...")
        print("=" * 70)
        from benchmark_eval import evaluate_model
        evaluate_model(pred_path, output_dir, f"{MODEL}+tool")

        # Tool usage stats
        tool_counts = []
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    tool_counts.append(len(r.get("tool_calls", [])))
        if tool_counts:
            tc = np.array(tool_counts)
            print(f"\n  Tool Usage Stats:")
            print(f"    Total samples: {len(tc)}")
            print(f"    Used tool:     {np.sum(tc > 0)} ({np.sum(tc > 0)/len(tc)*100:.1f}%)")
            print(f"    Avg calls:     {tc.mean():.2f}")
            print(f"    Max calls:     {tc.max()}")
    else:
        print(f"No predictions found at {pred_path}")

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
