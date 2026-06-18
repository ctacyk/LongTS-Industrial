"""
Independent GPT-5.5 Judge for Human vs HES Agreement

Uses GPT-5.5 as an independent second evaluator to score 250 samples,
then computes inter-annotator agreement with the original HES (Qwen-Plus) judge.
"""

import json
import time
import sys
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# Config
# ============================================================================

API_KEY = os.environ.get("JUDGE_API_KEY", "")
BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://vip.auto-code.net/v1")
MODEL = "gpt-5.5"

INPUT_PATH = Path("D:/Paper_code/results/human_agreement/annotation_samples.jsonl")
OUTPUT_PATH = Path("D:/Paper_code/results/human_agreement/gpt55_scored.jsonl")
REPORT_PATH = Path("D:/Paper_code/results/human_agreement/agreement_report_gpt55.json")

SYSTEM_PROMPT = """You are an independent expert evaluator specializing in industrial equipment time-series analysis. You will evaluate a model's prediction against a reference answer for industrial monitoring QA tasks.

Evaluate on TWO dimensions:

1. **Direction Alignment** — Does the prediction's core conclusion match the reference?
   - "aligned": Core conclusion fully consistent with reference (same fault identified, approximate time range correct, same operating state assessment)
   - "partial": General direction correct but significant details differ (e.g., detects anomaly but wrong time range or imprecise; or correct fault type but missing key causal evidence; or partially correct but vague)
   - "wrong": Core conclusion contradicts reference (e.g., reference identifies specific fault but prediction says normal/uncertain/cannot determine; or prediction identifies completely wrong root cause that contradicts physical mechanism; or reverses causal direction)

2. **Quality Score** (1–5):
   - 5: Covers all key information, specific values match, professional and complete
   - 4: Covers most key information with minor omissions
   - 3: Covers core information but missing important details or values imprecise
   - 2: Partially relevant but missing critical information
   - 1: Mostly irrelevant, incorrect, or refuses to answer

**Scoring Rules:**
- Focus on semantic and factual equivalence, NOT literal word matching
- Time step localization error within ±10% of fault interval length → still "aligned"
- Physical quantity (temperature, current, pressure) error within ±20% → acceptable
- If model says "cannot determine" or "insufficient data" while reference provides clear answer → "wrong"
- A prediction that identifies the correct fault type but reverses the causal mechanism → "wrong"
- A prediction that describes the data correctly but fails to explicitly identify the anomaly mentioned in reference → at best "partial"
- Generic descriptions without addressing specific phenomena in reference → "wrong" or low "partial"

Output ONLY a valid JSON object (no markdown, no explanation outside JSON):
{"direction": "aligned" or "partial" or "wrong", "quality": 1-5, "reason": "one-sentence explanation in Chinese"}"""


# ============================================================================
# Scoring
# ============================================================================

def score_one(client: OpenAI, sample: dict) -> dict:
    """Score a single sample using GPT-5.5."""
    question = sample['question']
    gt = sample['answer_gt']
    pred = sample['prediction']

    # Truncate very long predictions to avoid token limits
    if len(pred) > 2000:
        pred = pred[:2000] + "...[truncated]"

    user_prompt = f"""Question: {question}

Reference Answer (Ground Truth): {gt}

Model Prediction to Evaluate: {pred}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON from response
        # Handle possible markdown wrapping
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        return {
            "direction": result.get("direction", "wrong"),
            "quality": int(result.get("quality", 1)),
            "reason": result.get("reason", ""),
        }

    except json.JSONDecodeError:
        # Try to extract from malformed response
        content_lower = content.lower() if content else ""
        if '"aligned"' in content_lower:
            direction = "aligned"
        elif '"partial"' in content_lower:
            direction = "partial"
        else:
            direction = "wrong"
        return {"direction": direction, "quality": 2, "reason": f"JSON parse failed: {content[:100]}"}

    except Exception as e:
        return {"direction": "error", "quality": 0, "reason": str(e)[:100]}


def batch_score():
    """Score all samples and compute agreement."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Load samples
    records = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Scoring {len(records)} samples with {MODEL}...")
    print(f"API: {BASE_URL}")
    print()

    # Check for checkpoint
    checkpoint = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    checkpoint[r['idx']] = r
        print(f"Loaded {len(checkpoint)} checkpointed results")

    # Score
    results = []
    errors = 0
    start_time = time.time()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
        for i, rec in enumerate(records):
            # Use checkpoint if available
            if rec['idx'] in checkpoint and checkpoint[rec['idx']].get('gpt_direction'):
                scored = checkpoint[rec['idx']]
                results.append(scored)
                out_f.write(json.dumps(scored, ensure_ascii=False) + '\n')
                continue

            # Score
            gpt_result = score_one(client, rec)

            if gpt_result['direction'] == "error":
                errors += 1
                if errors > 10:
                    print("Too many errors, stopping.")
                    break
                # Retry once after delay
                time.sleep(3)
                gpt_result = score_one(client, rec)

            rec['gpt_direction'] = gpt_result['direction']
            rec['gpt_quality'] = gpt_result['quality']
            rec['gpt_reason'] = gpt_result['reason']

            results.append(rec)
            out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out_f.flush()

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(records) - i - 1) / rate
                print(f"  [{i+1}/{len(records)}] {rate:.1f} samples/s, ETA {eta:.0f}s | "
                      f"last: dir={gpt_result['direction']}, q={gpt_result['quality']}")

            # Rate limiting: ~2 req/s to be safe
            time.sleep(0.5)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min), errors={errors}")

    # Filter valid results
    valid = [r for r in results if r.get('gpt_direction') not in (None, 'error')]
    print(f"Valid results: {len(valid)}/{len(records)}")

    if len(valid) < 50:
        print("Too few valid results for analysis")
        return

    # === Agreement Analysis ===
    compute_agreement(valid)


def compute_agreement(records: list):
    """Compute GPT-5.5 vs HES agreement."""
    n = len(records)
    print(f"\n{'='*60}")
    print(f"GPT-5.5 (Independent) vs HES (Qwen-Plus) AGREEMENT (N={n})")
    print(f"{'='*60}")

    dir_map = {"aligned": 2, "partial": 1, "wrong": 0}

    gpt_dirs = np.array([dir_map.get(r['gpt_direction'], 0) for r in records])
    hes_dirs = np.array([dir_map.get(r['hes_scores']['direction'], 0) for r in records])

    # Exact agreement
    exact = np.sum(gpt_dirs == hes_dirs) / n
    print(f"\n  Direction exact agreement: {exact:.3f} ({int(exact*n)}/{n})")

    # Within-1
    within1 = np.sum(np.abs(gpt_dirs - hes_dirs) <= 1) / n
    print(f"  Direction within-1 agreement: {within1:.3f}")

    # Weighted Kappa
    kappa = weighted_kappa(gpt_dirs, hes_dirs, 3)
    print(f"  Weighted Cohen's Kappa: {kappa:.3f}")

    # Confusion matrix
    print(f"\n  Confusion Matrix (GPT-5.5 rows × HES cols):")
    labels = ["wrong", "partial", "aligned"]
    cm = np.zeros((3, 3), dtype=int)
    for g, h in zip(gpt_dirs, hes_dirs):
        cm[g][h] += 1
    print(f"  {'':14s} {'HES-wrong':>10s} {'HES-partial':>12s} {'HES-aligned':>12s}")
    for i, label in enumerate(labels):
        print(f"  GPT-{label:9s} {cm[i][0]:>10d} {cm[i][1]:>12d} {cm[i][2]:>12d}")

    # Severe disagreements
    severe = int(np.sum(np.abs(gpt_dirs - hes_dirs) > 1))
    boundary = int(np.sum(np.abs(gpt_dirs - hes_dirs) == 1))
    print(f"\n  Disagreements: {n - int(exact*n)} total")
    print(f"    Boundary (±1): {boundary}")
    print(f"    Severe (wrong↔aligned): {severe}")

    # Quality correlation
    gpt_q = np.array([r['gpt_quality'] for r in records], dtype=float)
    hes_aq = np.array([r['hes_scores']['aq'] for r in records], dtype=float)
    hes_h = np.array([r['hes_scores']['hes'] for r in records], dtype=float)

    from scipy.stats import spearmanr, pearsonr
    sp_aq, _ = spearmanr(gpt_q, hes_aq)
    sp_hes, _ = spearmanr(gpt_q, hes_h)
    print(f"\n  Spearman(GPT_Quality, HES_AQ): {sp_aq:.3f}")
    print(f"  Spearman(GPT_Quality, HES):    {sp_hes:.3f}")

    # Normalize GPT quality to 0-1 for Pearson
    gpt_q_norm = (gpt_q - 1) / 4.0
    pr_aq, _ = pearsonr(gpt_q_norm, hes_aq)
    print(f"  Pearson(GPT_Quality_norm, HES_AQ): {pr_aq:.3f}")

    # By level
    print(f"\n  --- By Level ---")
    for level in ["L1", "L2", "L3", "L4"]:
        lr = [r for r in records if r['level'] == level]
        if not lr:
            continue
        n_l = len(lr)
        g_d = np.array([dir_map.get(r['gpt_direction'], 0) for r in lr])
        h_d = np.array([dir_map.get(r['hes_scores']['direction'], 0) for r in lr])
        ag = np.sum(g_d == h_d) / n_l
        g_q = np.array([r['gpt_quality'] for r in lr], dtype=float)
        h_q = np.array([r['hes_scores']['aq'] for r in lr], dtype=float)
        sp, _ = spearmanr(g_q, h_q)
        print(f"  {level}: DA_agree={ag:.3f} (n={n_l}), Spearman_AQ={sp:.3f}")

    # By domain
    print(f"\n  --- By Domain ---")
    for domain in ["coal_mill", "transformer", "pump", "wind_turbine"]:
        dr = [r for r in records if r['domain'] == domain]
        if not dr:
            continue
        n_d = len(dr)
        g_d = np.array([dir_map.get(r['gpt_direction'], 0) for r in dr])
        h_d = np.array([dir_map.get(r['hes_scores']['direction'], 0) for r in dr])
        ag = np.sum(g_d == h_d) / n_d
        print(f"  {domain:15s}: DA_agree={ag:.3f} (n={n_d})")

    # Distributions
    gpt_dist = defaultdict(int)
    hes_dist = defaultdict(int)
    for r in records:
        gpt_dist[r['gpt_direction']] += 1
        hes_dist[r['hes_scores']['direction']] += 1
    print(f"\n  GPT-5.5 direction dist: {dict(gpt_dist)}")
    print(f"  HES direction dist:     {dict(hes_dist)}")

    # Save report
    report = {
        "n_samples": n,
        "model_judge": MODEL,
        "direction_exact_agreement": round(float(exact), 4),
        "direction_within1_agreement": round(float(within1), 4),
        "weighted_cohens_kappa": round(float(kappa), 4),
        "spearman_quality_vs_aq": round(float(sp_aq), 4),
        "spearman_quality_vs_hes": round(float(sp_hes), 4),
        "pearson_quality_vs_aq": round(float(pr_aq), 4),
        "confusion_matrix": cm.tolist(),
        "n_severe_disagreements": severe,
        "n_boundary_disagreements": boundary,
        "gpt_direction_dist": dict(gpt_dist),
        "hes_direction_dist": dict(hes_dist),
    }
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved: {REPORT_PATH}")


def weighted_kappa(y1, y2, n_cat):
    n = len(y1)
    O = np.zeros((n_cat, n_cat))
    for a, b in zip(y1, y2):
        O[a][b] += 1
    O /= n
    rm = O.sum(axis=1)
    cm_arr = O.sum(axis=0)
    E = np.outer(rm, cm_arr)
    W = np.zeros((n_cat, n_cat))
    for i in range(n_cat):
        for j in range(n_cat):
            W[i][j] = (i - j) ** 2 / (n_cat - 1) ** 2
    num = np.sum(W * O)
    den = np.sum(W * E)
    return 1.0 - num / den if den > 0 else 1.0


if __name__ == "__main__":
    batch_score()
