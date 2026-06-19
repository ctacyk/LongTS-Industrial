"""
LongTS-Industrial Benchmark: VLM 评测

支持两种推理后端:
  1. 远程 API (DashScope): Qwen3-VL 系列
  2. 本地 vLLM:            InternVL3 / 任意 OpenAI-compatible 模型

环境变量:
  VLM_BASE_URL   推理 API 地址 (默认 DashScope)
  VLM_API_KEY    推理 API Key  (默认 DashScope key; 本地 vLLM 设 "EMPTY")
  JUDGE_API_KEY  评测用 qwen-plus/embedding API Key (默认同 VLM_API_KEY)

并发策略：各模型独立进程，模型内多线程 + RPM 速率控制。
"""

import os
import sys
import json
import base64
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, Queue
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

# ============================================================
# Configuration — 通过环境变量切换推理后端
# ============================================================
_DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
_DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

API_KEY = os.environ.get("VLM_API_KEY", _DASHSCOPE_KEY)
BASE_URL = os.environ.get("VLM_BASE_URL", _DASHSCOPE_URL)

QWEN_MODELS = [
    "qwen3-vl-32b-thinking",
    "qwen3-vl-32b-instruct",
    "qwen3-vl-30b-a3b-thinking",
    "qwen3-vl-30b-a3b-instruct",
    "qwen3-vl-8b-thinking",
    "qwen3-vl-8b-instruct",
]

RPM_PER_MODEL = 300
MAX_TOKENS = 2048
TEMPERATURE = 0.3

BENCHMARK_DIR = "Data/benchmark_tb/LongTS_Industrial"
DATA_DIR = "Data/benchmark_tb"
RESULTS_DIR = "results/benchmark_eval_v2"

THINKING_MODELS = {"qwen3-vl-32b-thinking", "qwen3-vl-30b-a3b-thinking", "qwen3-vl-8b-thinking"}


# ============================================================
# Utilities
# ============================================================
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_system_prompt() -> str:
    return (
        "你是一位工业设备时序数据分析专家。你将看到一张工业设备多通道传感器的时序数据可视化图。"
        "图中每个子图代表一个传感器通道，横轴是时间步，纵轴是传感器读数。"
        "请仔细观察所有通道的数据模式，然后根据用户的问题进行分析和回答。"
        "你的回答应该专业、具体，包含对时间步范围、数值变化、通道间关联的精确描述。"
    )


def build_user_content(question: str, image_base64: str) -> list:
    """构建多模态用户消息（图片 + 文本）"""
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}"
            }
        },
        {
            "type": "text",
            "text": question
        }
    ]


# ============================================================
# VLM Inference for a single model
# ============================================================
def run_model_inference(model_name: str, samples: list,
                         output_dir: str, rpm: int = RPM_PER_MODEL,
                         max_workers: int = 5):
    """
    对单个模型运行全部样本的推理（支持模型内多线程并发）

    samples: list of {id, level, question, answer_gt, domain, difficulty, image_path}
    output_dir: 输出目录
    max_workers: 模型内并发线程数
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "predictions.jsonl")
    log_path = os.path.join(output_dir, "inference_log.txt")

    # 加载已有预测（断点续传）— 用 id+level+question_hash 区分同 level 多 QA
    def _resume_key(rec):
        import hashlib
        qh = hashlib.md5(rec.get('question', '').encode()).hexdigest()[:8]
        return rec.get('id', '') + '_' + rec.get('level', '') + '_' + qh

    existing_ids = set()
    if os.path.exists(pred_path):
        with open(pred_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        existing_ids.add(_resume_key(r))
                    except:
                        pass

    pending = [s for s in samples if _resume_key(s) not in existing_ids]

    _log(log_path, f"Model: {model_name}")
    _log(log_path, f"Total: {len(samples)}, Existing: {len(existing_ids)}, Pending: {len(pending)}")
    _log(log_path, f"Workers: {max_workers}, RPM: {rpm}")

    if not pending:
        _log(log_path, "All samples already completed. Skipping.")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    is_thinking = model_name in THINKING_MODELS

    write_lock = threading.Lock()
    completed = [0]
    errors = [0]
    start_time = time.time()

    # Rate limiter: token bucket
    rate_lock = threading.Lock()
    last_request_time = [0.0]
    min_interval = 60.0 / rpm

    def acquire_rate():
        with rate_lock:
            now = time.time()
            wait = last_request_time[0] + min_interval - now
            if wait > 0:
                time.sleep(wait)
            last_request_time[0] = time.time()

    def process_sample(sample):
        try:
            image_path = sample['image_path']
            if not os.path.exists(image_path):
                errors[0] += 1
                return

            image_b64 = encode_image_to_base64(image_path)
            messages = [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_content(sample['question'], image_b64)},
            ]

            kwargs = {
                "model": model_name,
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            }
            if is_thinking:
                kwargs.pop("temperature")

            acquire_rate()
            response = client.chat.completions.create(**kwargs)

            prediction = ""
            choice = response.choices[0]
            thinking_content = ""

            if is_thinking:
                msg = choice.message
                if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                    thinking_content = msg.reasoning_content
                prediction = msg.content or ""
            else:
                prediction = choice.message.content or ""

            result = {
                "id": sample['id'],
                "level": sample['level'],
                "domain": sample['domain'],
                "difficulty": sample['difficulty'],
                "question": sample['question'],
                "answer_gt": sample['answer_gt'],
                "prediction": prediction,
                "model": model_name,
            }
            if is_thinking and thinking_content:
                result["thinking"] = thinking_content

            with write_lock:
                with open(pred_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                completed[0] += 1

                if completed[0] % 20 == 0 or completed[0] <= 3:
                    elapsed = time.time() - start_time
                    rate = completed[0] / (elapsed / 60) if elapsed > 0 else 0
                    eta = (len(pending) - completed[0]) / max(rate, 0.01)
                    _log(log_path, f"  [{completed[0]}/{len(pending)}] "
                                   f"{rate:.1f}/min | ETA {eta:.0f}min")

        except Exception as e:
            errors[0] += 1
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(10)
            if errors[0] <= 5:
                _log(log_path, f"  [ERROR] {sample['id']}: {str(e)[:150]}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, s) for s in pending]
        for f in as_completed(futures):
            f.result()

    elapsed = time.time() - start_time
    _log(log_path, f"\nDone: completed={completed[0]}, errors={errors[0]}, time={elapsed/60:.1f}min")


def _log(path, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ============================================================
# Load benchmark samples and match with images
# ============================================================
def load_benchmark_samples(benchmark_dir: str, data_dir: str,
                            split: str = "test") -> list:
    """
    加载 benchmark JSONL，将每个 QA 对展开为独立样本，
    并匹配对应的 eval.png 图片路径。
    """
    jsonl_path = os.path.join(benchmark_dir, f"{split}.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return []

    samples = []
    missing_images = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)

            # 确定图片路径
            source = record.get("source", "synthetic")
            if source == "enginemt_qa":
                # EngineMT-QA 没有我们的可视化图，跳过
                continue

            # 合成数据的 eval.png
            ts_path = record.get("timeseries_path", "")
            if ts_path:
                ts_path = ts_path.replace("\\", "/")
                if ts_path.startswith("../"):
                    ts_path = ts_path[3:]
                eval_png = os.path.join(data_dir,
                                         ts_path.replace(".csv", "_eval.png"))
            else:
                continue

            if not os.path.exists(eval_png):
                missing_images += 1
                continue

            # 展开 QA 对
            for qa in record.get("qa_pairs", []):
                samples.append({
                    "id": record["id"],
                    "level": qa.get("level", "?"),
                    "domain": record.get("domain", "?"),
                    "difficulty": record.get("metadata", {}).get("difficulty", "?"),
                    "question": qa.get("question", ""),
                    "answer_gt": qa.get("answer", ""),
                    "image_path": eval_png,
                })

    if missing_images > 0:
        print(f"Warning: {missing_images} records missing eval.png")

    print(f"Loaded {len(samples)} QA samples from {split}.jsonl (with images)")
    return samples


# ============================================================
# Process wrapper for multiprocessing
# ============================================================
def _model_process(model_name, samples, results_dir, rpm, max_workers=20):
    """单个模型的进程函数"""
    safe_name = model_name.replace("/", "--")
    output_dir = os.path.join(results_dir, safe_name)
    try:
        run_model_inference(model_name, samples, output_dir, rpm, max_workers)
    except Exception as e:
        print(f"[{model_name}] Fatal error: {e}")
        traceback.print_exc()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="LongTS-Industrial VLM Benchmark Evaluation")
    parser.add_argument("--models", nargs="*", default=QWEN_MODELS,
                        help="Models to evaluate (model names or HF paths)")
    parser.add_argument("--split", default="test",
                        help="Benchmark split: test or test_needle")
    parser.add_argument("--benchmark-dir", default=BENCHMARK_DIR)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--rpm", type=int, default=RPM_PER_MODEL)
    parser.add_argument("--workers", type=int, default=20,
                        help="Concurrent threads per model (default 20)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run models sequentially instead of parallel")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip inference, only run judge evaluation")
    args = parser.parse_args()

    print("=" * 70)
    print("LongTS-Industrial Benchmark VLM Evaluation")
    print(f"Models:   {args.models}")
    print(f"API URL:  {BASE_URL}")
    print(f"API Key:  {API_KEY[:8]}..." if len(API_KEY) > 8 else f"API Key:  {API_KEY}")
    print(f"Split:    {args.split}")
    print(f"RPM:      {args.rpm}")
    print(f"Workers:  {args.workers}")
    print(f"Parallel: {not args.sequential}")
    print("=" * 70)

    if not args.eval_only:
        # Load samples
        samples = load_benchmark_samples(args.benchmark_dir, args.data_dir, args.split)
        if not samples:
            print("No samples loaded. Check benchmark and image paths.")
            return

        # Run inference
        if args.sequential:
            for model in args.models:
                print(f"\n>>> Running {model}...")
                _model_process(model, samples, args.results_dir, args.rpm, args.workers)
        else:
            # Parallel: each model in its own process
            processes = []
            for model in args.models:
                p = Process(target=_model_process,
                           args=(model, samples, args.results_dir, args.rpm, args.workers))
                p.start()
                processes.append((model, p))
                print(f"  Started process for {model} (PID={p.pid})")
                time.sleep(2)  # 错开启动避免同时打开大量文件

            # Wait for all
            for model, p in processes:
                p.join()
                print(f"  {model} finished (exit code={p.exitcode})")

    # Run judge evaluation for each model
    print("\n" + "=" * 70)
    print("Running Judge Evaluation...")
    print("=" * 70)

    from benchmark_eval import evaluate_model, compare_models

    for model in args.models:
        safe_name = model.replace("/", "--")
        pred_path = os.path.join(args.results_dir, safe_name, "predictions.jsonl")
        if os.path.exists(pred_path):
            output_dir = os.path.join(args.results_dir, safe_name)
            evaluate_model(pred_path, output_dir, model)
        else:
            print(f"  No predictions found for {model}")

    # Comparison report: 自动发现所有已有 eval_results.json 的模型，生成完整对比
    compare_models_list = []
    if os.path.isdir(args.results_dir):
        compare_models_list = sorted(
            d for d in os.listdir(args.results_dir)
            if os.path.isfile(os.path.join(args.results_dir, d, "eval_results.json"))
        )
    if not compare_models_list:
        compare_models_list = args.models
    compare_models(args.results_dir, compare_models_list)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
