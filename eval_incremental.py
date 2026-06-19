"""Incremental evaluation: only evaluate wind_turbine predictions, merge with existing results."""
import os, sys, json, hashlib
import numpy as np
from collections import defaultdict

sys.path.insert(0, '.')
from benchmark_eval import (
    batch_get_embeddings, cosine_similarity, batch_judge,
    direction_to_score, quality_to_score, compute_hes
)

RESULTS_DIR = "results/benchmark_eval_v2"
MODELS = [
    "qwen3-vl-32b-thinking",
    "qwen3-vl-32b-instruct",
    "qwen3-vl-30b-a3b-thinking",
    "qwen3-vl-30b-a3b-instruct",
    "qwen3-vl-8b-thinking",
    "qwen3-vl-8b-instruct",
]


def _qhash(text):
    return hashlib.md5(text.encode()).hexdigest()[:12]


def evaluate_samples(samples, checkpoint_prefix):
    """Evaluate a list of samples, return list with 'eval' field attached."""
    predictions = [s.get("prediction", "") for s in samples]
    ground_truths = [s.get("answer_gt", "") for s in samples]

    print(f"  Computing embeddings for {len(samples)} samples...")
    pred_embs = batch_get_embeddings(predictions)
    gt_embs = batch_get_embeddings(ground_truths)

    ss_scores = []
    for pe, ge in zip(pred_embs, gt_embs):
        if pe and ge:
            ss_scores.append(cosine_similarity(pe, ge))
        else:
            ss_scores.append(0.0)
    ss_scores = np.array(ss_scores)
    print(f"  SS mean={ss_scores.mean():.4f}")

    print(f"  Running LLM Judge...")
    judge_ckpt = checkpoint_prefix + "_judge_ckpt.json"
    judge_samples = [
        {"question": s.get("question", ""), "ground_truth": s.get("answer_gt", ""),
         "prediction": s.get("prediction", "")}
        for s in samples
    ]
    judge_results = batch_judge(judge_samples, checkpoint_file=judge_ckpt)

    da_scores = np.array([direction_to_score(jr["direction"]) for jr in judge_results])
    aq_scores = np.array([quality_to_score(jr["quality"]) for jr in judge_results])
    hes_scores = np.array([compute_hes(da, ss, aq) for da, ss, aq in
                           zip(da_scores, ss_scores, aq_scores)])

    for i, s in enumerate(samples):
        s["eval"] = {
            "ss": float(ss_scores[i]),
            "da": float(da_scores[i]),
            "aq": float(aq_scores[i]),
            "hes": float(hes_scores[i]),
            "direction": judge_results[i]["direction"],
            "quality": judge_results[i]["quality"],
            "reason": judge_results[i].get("reason", ""),
        }
    return samples


def rebuild_eval_results(model_dir, model_name):
    """Regenerate eval_results.json from eval_detailed.jsonl."""
    detail_path = os.path.join(model_dir, "eval_detailed.jsonl")
    samples = []
    with open(detail_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    ss = np.array([s["eval"]["ss"] for s in samples])
    da = np.array([s["eval"]["da"] for s in samples])
    aq = np.array([s["eval"]["aq"] for s in samples])
    hes = np.array([s["eval"]["hes"] for s in samples])

    def group_stats(indices):
        if len(indices) == 0:
            return {"n": 0, "hes": 0, "ss": 0, "da": 0, "aq": 0}
        idx = np.array(indices)
        return {
            "n": len(indices),
            "hes": float(hes[idx].mean()),
            "ss": float(ss[idx].mean()),
            "da": float(da[idx].mean()),
            "aq": float(aq[idx].mean()),
        }

    by_level, by_domain, by_diff = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, s in enumerate(samples):
        by_level[s.get("level", "?")].append(i)
        by_domain[s.get("domain", "?")].append(i)
        by_diff[s.get("difficulty", "?")].append(i)

    result = {
        "model": model_name,
        "total_samples": len(samples),
        "overall": group_stats(list(range(len(samples)))),
        "by_level": {k: group_stats(v) for k, v in sorted(by_level.items())},
        "by_domain": {k: group_stats(v) for k, v in sorted(by_domain.items())},
        "by_difficulty": {k: group_stats(v) for k, v in sorted(by_diff.items())},
    }

    out_path = os.path.join(model_dir, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Saved eval_results.json: {len(samples)} samples")
    return result


def process_model(model_name):
    model_dir = os.path.join(RESULTS_DIR, model_name)
    pred_path = os.path.join(model_dir, "predictions.jsonl")
    detail_path = os.path.join(model_dir, "eval_detailed.jsonl")

    # Load predictions
    preds = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))

    # Load existing eval
    existing = []
    if os.path.exists(detail_path):
        with open(detail_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing.append(json.loads(line))

    # Build set of already-evaluated keys
    eval_keys = set()
    for e in existing:
        key = e.get("id", "") + "_" + e.get("level", "") + "_" + _qhash(e.get("question", ""))
        eval_keys.add(key)

    # Find predictions that need evaluation
    pending = []
    for p in preds:
        key = p.get("id", "") + "_" + p.get("level", "") + "_" + _qhash(p.get("question", ""))
        if key not in eval_keys:
            pending.append(p)

    # Also check if existing has MORE entries than predictions (cleanup needed)
    pred_keys = set()
    for p in preds:
        key = p.get("id", "") + "_" + p.get("level", "") + "_" + _qhash(p.get("question", ""))
        pred_keys.add(key)

    cleaned_existing = [e for e in existing
                        if (e.get("id", "") + "_" + e.get("level", "") + "_" + _qhash(e.get("question", ""))) in pred_keys]

    if len(cleaned_existing) < len(existing):
        print(f"  Cleaned eval_detailed: {len(existing)} -> {len(cleaned_existing)} (removed {len(existing) - len(cleaned_existing)} orphans)")
        existing = cleaned_existing

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"  Predictions: {len(preds)}")
    print(f"  Existing eval: {len(existing)}")
    print(f"  Pending eval: {len(pending)}")

    if len(pending) == 0 and len(existing) == len(preds):
        print(f"  Already complete! Rebuilding stats only.")
        # Just write cleaned existing if needed
        with open(detail_path, "w", encoding="utf-8") as f:
            for e in existing:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        rebuild_eval_results(model_dir, model_name)
        return

    if len(pending) > 0:
        ckpt_prefix = os.path.join(model_dir, "incremental_wt")
        evaluated = evaluate_samples(pending, ckpt_prefix)
        existing.extend(evaluated)

    # Write merged eval_detailed
    with open(detail_path, "w", encoding="utf-8") as f:
        for e in existing:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  Written eval_detailed: {len(existing)} entries")

    rebuild_eval_results(model_dir, model_name)


if __name__ == "__main__":
    for model in MODELS:
        process_model(model)
    print("\n" + "=" * 60)
    print("ALL DONE!")
