"""
LongTS-Industrial Benchmark 评估脚本

基于 htsr_eval.py 的 HES 框架，适配 LongTS-Industrial Benchmark：
- Judge prompt：中文通用工业领域
- 统计维度：按 L1/L2/L3/L4、domain、difficulty 分别统计
- HES = DA × (0.4 × SS + 0.6 × AQ)
- Judge 模型：qwen-plus（RPM 30000）
- Embedding 模型：text-embedding-v3

输入：某个模型的 predictions.jsonl（每行: id, level, question, answer_gt, prediction）
输出：eval_results.json（各维度的 HES/DA/SS/AQ 统计）
"""

import os
import json
import re
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from datetime import datetime
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
_DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
_DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.environ.get("JUDGE_API_KEY", os.environ.get("VLM_API_KEY", _DASHSCOPE_KEY))
BASE_URL = _DASHSCOPE_URL  # Judge/Embedding 始终走 DashScope

W_SS = 0.4
W_AQ = 0.6

JUDGE_MODEL = "qwen-plus"
EMBED_MODEL = "text-embedding-v3"

MAX_WORKERS = 50
RPM_LIMIT = 5000

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ============================================================
# Rate Limiter
# ============================================================
class RateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.last_call = 0

    def acquire(self):
        with self.lock:
            now = time.time()
            wait = self.last_call + self.interval - now
            if wait > 0:
                time.sleep(wait)
            self.last_call = time.time()


rate_limiter = RateLimiter(RPM_LIMIT)


# ============================================================
# 中文通用工业领域 Judge Prompt
# ============================================================
JUDGE_SYSTEM_PROMPT = """你是工业时序数据分析领域的资深评估专家。你的任务是对比模型的回答与标准答案，从两个维度进行评判。

## 维度1：方向对齐 (Direction Alignment)
模型的回答在核心结论/判断方向上是否与标准答案一致？

- "aligned"：核心结论完全一致（如都指出了异常、都定位到了正确的时间区间、都给出了相同方向的判断）
- "partial"：大方向正确但细节有偏差（如定位到了异常但时间范围不够精确，或原因分析方向正确但不完整）
- "wrong"：核心结论与标准答案矛盾（如标准答案说有异常但模型说正常，或判断方向完全相反）

## 维度2：答案质量 (1-5分)
综合评估模型回答的质量：

- 5分：覆盖了标准答案的所有关键信息，可能还补充了正确的额外细节。专业、完整、准确。
- 4分：覆盖了大部分关键信息，有少量遗漏但整体回答质量好。
- 3分：覆盖了核心信息，但缺少一些重要细节或数值不够精确。
- 2分：部分相关但遗漏了重要信息，或数值偏差较大。
- 1分：基本不相关或内容大部分错误。

重要规则：
- 如果模型的回答比标准答案更详细，但所有额外内容都是正确的，应给4-5分
- 关注语义等价而非字面匹配
- 时间步定位误差在10%以内视为可接受
- 温度/电流等数值误差在20%以内视为可接受

严格按以下JSON格式输出（不要包含任何其他文字）：
{"direction": "aligned|partial|wrong", "quality": 1-5, "reason": "一句话评判理由"}"""


def build_judge_prompt(question: str, ground_truth: str, prediction: str) -> str:
    return f"""问题：{question[:500]}

标准答案：
{ground_truth[:1000]}

模型回答：
{prediction[:1000]}

请评判模型回答与标准答案的一致性和质量。"""


# ============================================================
# Embedding
# ============================================================
def batch_get_embeddings(texts: list, batch_size: int = 25, max_workers: int = 10) -> list:
    """Parallel embedding with large batches for high throughput."""
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    results = [None] * len(batches)

    def _embed_batch(idx, batch):
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL, input=batch, encoding_format="float"
            )
            return idx, [d.embedding for d in response.data]
        except Exception as e:
            embs = []
            for text in batch:
                try:
                    resp = client.embeddings.create(model=EMBED_MODEL, input=text, encoding_format="float")
                    embs.append(resp.data[0].embedding)
                except:
                    embs.append(None)
            return idx, embs

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_embed_batch, idx, batch): idx for idx, batch in enumerate(batches)}
        with tqdm(total=len(batches), desc="Embedding") as pbar:
            for future in as_completed(futures):
                idx, embs = future.result()
                results[idx] = embs
                pbar.update(1)

    embeddings = []
    for batch_embs in results:
        embeddings.extend(batch_embs)
    return embeddings


def cosine_similarity(v1, v2) -> float:
    v1, v2 = np.array(v1), np.array(v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm)


# ============================================================
# LLM Judge
# ============================================================
def judge_single(question: str, ground_truth: str, prediction: str, retries: int = 3) -> dict:
    rate_limiter.acquire()
    user_prompt = build_judge_prompt(question, ground_truth, prediction)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            assert result['direction'] in ['aligned', 'partial', 'wrong']
            assert 1 <= result['quality'] <= 5
            return result
        except json.JSONDecodeError:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            return {"direction": "partial", "quality": 3, "reason": "JSON parse error"}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            return {"direction": "partial", "quality": 3, "reason": f"API error: {str(e)[:100]}"}


def batch_judge(samples: list, checkpoint_file: str = None) -> list:
    results = [None] * len(samples)
    completed_count = 0
    lock = threading.Lock()

    if checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        for idx, res in checkpoint.items():
            results[int(idx)] = res
        completed_count = sum(1 for r in results if r is not None)
        print(f"  Loaded checkpoint: {completed_count}/{len(samples)} already judged")

    def process(idx, sample):
        if results[idx] is not None:
            return idx, results[idx]
        result = judge_single(
            sample.get('question', ''),
            sample['ground_truth'],
            sample['prediction']
        )
        return idx, result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        pending = [(i, s) for i, s in enumerate(samples) if results[i] is None]
        futures = {executor.submit(process, i, s): i for i, s in pending}

        with tqdm(total=len(pending), desc="LLM Judge") as pbar:
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                    pbar.update(1)
                    with lock:
                        completed_count += 1
                        if checkpoint_file and completed_count % 50 == 0:
                            ckpt = {str(i): r for i, r in enumerate(results) if r is not None}
                            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                                json.dump(ckpt, f, ensure_ascii=False)
                except Exception as e:
                    idx = futures[future]
                    results[idx] = {"direction": "partial", "quality": 3, "reason": str(e)[:100]}
                    pbar.update(1)

    if checkpoint_file:
        ckpt = {str(i): r for i, r in enumerate(results) if r is not None}
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(ckpt, f, ensure_ascii=False)

    return results


# ============================================================
# Score Computation
# ============================================================
def direction_to_score(d: str) -> float:
    return {"aligned": 1.0, "partial": 0.5, "wrong": 0.0}.get(d, 0.5)

def quality_to_score(q: int) -> float:
    return (q - 1) / 4.0

def compute_hes(da, ss, aq):
    return da * (W_SS * ss + W_AQ * aq)


# ============================================================
# Main Evaluation
# ============================================================
def evaluate_model(predictions_path: str, output_dir: str, model_name: str = ""):
    """
    评估单个模型的预测结果

    predictions_path: JSONL, 每行 {id, level, question, answer_gt, prediction, domain, difficulty, ...}
    output_dir: 输出目录
    """
    print("=" * 70)
    print(f"LongTS-Industrial Benchmark Evaluation")
    print(f"Model: {model_name}")
    print(f"Judge: {JUDGE_MODEL} | Embed: {EMBED_MODEL}")
    print(f"Weights: SS={W_SS}, AQ={W_AQ}")
    print("=" * 70)

    # Load predictions
    samples = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"\nTotal samples: {len(samples)}")

    # Step 1: Semantic Similarity
    print("\n[Step 1/3] Computing Semantic Similarity...")
    emb_cache = os.path.join(output_dir, "emb_cache.json")
    if os.path.exists(emb_cache):
        with open(emb_cache, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        pred_embs = cache['pred']
        gt_embs = cache['gt']
        print(f"  Loaded cached embeddings")
    else:
        predictions = [s.get('prediction', '') for s in samples]
        ground_truths = [s.get('answer_gt', '') for s in samples]
        pred_embs = batch_get_embeddings(predictions)
        gt_embs = batch_get_embeddings(ground_truths)
        with open(emb_cache, 'w', encoding='utf-8') as f:
            json.dump({'pred': pred_embs, 'gt': gt_embs}, f)

    ss_scores = []
    for pe, ge in zip(pred_embs, gt_embs):
        if pe and ge:
            ss_scores.append(cosine_similarity(pe, ge))
        else:
            ss_scores.append(0.0)
    ss_scores = np.array(ss_scores)
    print(f"  SS mean={ss_scores.mean():.4f}")

    # Step 2: LLM Judge
    print("\n[Step 2/3] LLM Judge...")
    judge_ckpt = os.path.join(output_dir, "judge_checkpoint.json")
    judge_samples = [
        {"question": s.get("question", ""), "ground_truth": s.get("answer_gt", ""),
         "prediction": s.get("prediction", "")}
        for s in samples
    ]
    judge_results = batch_judge(judge_samples, checkpoint_file=judge_ckpt)

    da_scores = np.array([direction_to_score(jr['direction']) for jr in judge_results])
    aq_scores = np.array([quality_to_score(jr['quality']) for jr in judge_results])

    # Step 3: Compute HES
    print("\n[Step 3/3] Computing HES...")
    hes_scores = np.array([compute_hes(da, ss, aq) for da, ss, aq in
                            zip(da_scores, ss_scores, aq_scores)])

    # Attach scores
    for i, s in enumerate(samples):
        s['eval'] = {
            'ss': float(ss_scores[i]),
            'da': float(da_scores[i]),
            'aq': float(aq_scores[i]),
            'hes': float(hes_scores[i]),
            'direction': judge_results[i]['direction'],
            'quality': judge_results[i]['quality'],
            'reason': judge_results[i].get('reason', ''),
        }

    # ============ Multi-dimensional Statistics ============
    def compute_group_stats(indices):
        if len(indices) == 0:
            return {"n": 0, "hes": 0, "ss": 0, "da": 0, "aq": 0}
        idx = np.array(indices)
        return {
            "n": len(indices),
            "hes": float(hes_scores[idx].mean()),
            "ss": float(ss_scores[idx].mean()),
            "da": float(da_scores[idx].mean()),
            "aq": float(aq_scores[idx].mean()),
        }

    # Group by level
    level_groups = defaultdict(list)
    domain_groups = defaultdict(list)
    difficulty_groups = defaultdict(list)
    for i, s in enumerate(samples):
        level_groups[s.get('level', '?')].append(i)
        domain_groups[s.get('domain', '?')].append(i)
        difficulty_groups[s.get('difficulty', '?')].append(i)

    stats = {
        "model": model_name,
        "total_samples": len(samples),
        "overall": compute_group_stats(list(range(len(samples)))),
        "by_level": {k: compute_group_stats(v) for k, v in sorted(level_groups.items())},
        "by_domain": {k: compute_group_stats(v) for k, v in sorted(domain_groups.items())},
        "by_difficulty": {k: compute_group_stats(v) for k, v in sorted(difficulty_groups.items())},
        "direction_dist": {
            "aligned": int(np.sum(da_scores == 1.0)),
            "partial": int(np.sum(da_scores == 0.5)),
            "wrong": int(np.sum(da_scores == 0.0)),
        },
    }

    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*70}")
    print(f"\n  Overall ({len(samples)} samples):")
    print(f"    HES:  {stats['overall']['hes']*100:.2f}%")
    print(f"    DA:   {stats['overall']['da']*100:.2f}%")
    print(f"    SS:   {stats['overall']['ss']*100:.2f}%")
    print(f"    AQ:   {stats['overall']['aq']*100:.2f}%")

    print(f"\n  By Level:")
    for lvl, s in stats['by_level'].items():
        print(f"    {lvl} (n={s['n']}): HES={s['hes']*100:.1f}%  DA={s['da']*100:.1f}%  "
              f"SS={s['ss']*100:.1f}%  AQ={s['aq']*100:.1f}%")

    print(f"\n  By Domain:")
    for dom, s in stats['by_domain'].items():
        print(f"    {dom} (n={s['n']}): HES={s['hes']*100:.1f}%  DA={s['da']*100:.1f}%")

    print(f"\n  By Difficulty:")
    for diff, s in stats['by_difficulty'].items():
        print(f"    {diff} (n={s['n']}): HES={s['hes']*100:.1f}%  DA={s['da']*100:.1f}%")

    print(f"\n  Direction: aligned={stats['direction_dist']['aligned']}, "
          f"partial={stats['direction_dist']['partial']}, "
          f"wrong={stats['direction_dist']['wrong']}")

    # Save
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    detailed_path = os.path.join(output_dir, "eval_detailed.jsonl")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"\n  Results saved: {results_path}")
    print(f"  Detailed saved: {detailed_path}")

    return stats


def compare_models(results_dir: str, model_names: list):
    """生成多模型横向对比报告"""
    all_stats = {}
    for name in model_names:
        path = os.path.join(results_dir, name, "eval_results.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                all_stats[name] = json.load(f)

    if not all_stats:
        print("No evaluation results found")
        return

    print("\n" + "=" * 90)
    print("MODEL COMPARISON REPORT")
    print("=" * 90)

    # Overall
    header = f"{'Model':<35} {'HES':>8} {'DA':>8} {'SS':>8} {'AQ':>8} {'N':>6}"
    print(f"\n  Overall:\n  {header}")
    print(f"  {'-'*75}")
    for name in sorted(all_stats.keys()):
        s = all_stats[name]['overall']
        print(f"  {name:<35} {s['hes']*100:>7.2f}% {s['da']*100:>7.2f}% "
              f"{s['ss']*100:>7.2f}% {s['aq']*100:>7.2f}% {s['n']:>5}")

    # By Level
    levels = sorted(set(k for s in all_stats.values() for k in s.get('by_level', {})))
    for lvl in levels:
        print(f"\n  {lvl}:")
        print(f"  {'Model':<35} {'HES':>8} {'DA':>8} {'N':>6}")
        print(f"  {'-'*55}")
        for name in sorted(all_stats.keys()):
            s = all_stats[name].get('by_level', {}).get(lvl, {})
            if s:
                print(f"  {name:<35} {s['hes']*100:>7.2f}% {s['da']*100:>7.2f}% {s.get('n',0):>5}")

    # Save comparison
    report_path = os.path.join(results_dir, "comparison_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\n  Comparison saved: {report_path}")

    return all_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LongTS-Industrial Benchmark Evaluation")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model-name", default="", help="Model name for display")
    args = parser.parse_args()

    evaluate_model(args.predictions, args.output_dir, args.model_name)
