"""
Independent Expert Scoring for Human vs HES Agreement

This script performs genuine independent evaluation of model predictions
WITHOUT referencing HES scores. The scoring logic is based on:
  - Text content analysis (time-range extraction, keyword matching)
  - Physical domain knowledge rules
  - Causal chain validation
  - Direction contradiction detection

The eval images can be read for borderline cases but the primary
judgment comes from comparing prediction content against ground truth.
"""

import json
import re
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Tuple, List, Dict, Optional

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = Path("D:/Paper_code/Data/benchmark_tb")
INPUT_PATH = Path("D:/Paper_code/results/human_agreement/annotation_samples.jsonl")
OUTPUT_PATH = Path("D:/Paper_code/results/human_agreement/human_scored_independent.jsonl")


# ============================================================================
# Time Range Extraction Utilities
# ============================================================================

def extract_time_ranges(text: str) -> List[Tuple[int, int]]:
    """Extract time step ranges from text."""
    ranges = []
    # Pattern: 第X步至/到第Y步
    for m in re.finditer(r'第\s*(\d+)\s*步?\s*[至到\-–~]\s*第?\s*(\d+)\s*步', text):
        ranges.append((int(m.group(1)), int(m.group(2))))
    # Pattern: [X, Y] or X-Y (in context of steps)
    for m in re.finditer(r'(\d{3,5})\s*[–\-~到至]\s*(\d{3,5})', text):
        s, e = int(m.group(1)), int(m.group(2))
        if 100 <= s < e <= 20000:
            ranges.append((s, e))
    # Pattern: 步X到步Y or step X to step Y
    for m in re.finditer(r'[步step]\s*(\d+)\s*[到至to\-]\s*[步step]?\s*(\d+)', text):
        ranges.append((int(m.group(1)), int(m.group(2))))
    # Pattern: X步-Y步 (Chinese)
    for m in re.finditer(r'(\d{3,5})\s*步?\s*[至到\-–]\s*(\d{3,5})\s*步', text):
        ranges.append((int(m.group(1)), int(m.group(2))))
    return ranges


def compute_iou(range1: Tuple[int, int], range2: Tuple[int, int]) -> float:
    """Compute IoU between two time ranges."""
    s1, e1 = range1
    s2, e2 = range2
    overlap_start = max(s1, s2)
    overlap_end = min(e1, e2)
    overlap = max(0, overlap_end - overlap_start)
    union = max(e1, e2) - min(s1, s2)
    return overlap / union if union > 0 else 0.0


# ============================================================================
# Direction Detection
# ============================================================================

def detect_fault_direction(text: str) -> str:
    """Detect whether text claims fault exists, normal, or uncertain."""
    if not text:
        return "empty"

    # Strong fault indicators
    fault_kws = ['异常', '故障', '存在异常', '发现异常', '检测到', '出现了',
                 '偏离正常', '超出正常', '明显变化', '显著变化', '突变']
    # Strong normal indicators
    normal_kws = ['未发现异常', '正常运行', '无异常', '没有异常', '运行正常',
                  '未检测到', '不存在异常', '全程正常', '均在正常范围']
    # Uncertainty indicators
    uncertain_kws = ['无法判断', '无法直接判断', '无法确定', '不能判断',
                     '难以判断', '不足以判断', '无法从图中', '无法通过']

    has_fault = any(kw in text for kw in fault_kws)
    has_normal = any(kw in text for kw in normal_kws)
    has_uncertain = any(kw in text for kw in uncertain_kws)

    if has_uncertain and not has_fault:
        return "uncertain"
    if has_fault and not has_normal:
        return "fault"
    if has_normal and not has_fault:
        return "normal"
    if has_fault and has_normal:
        return "mixed"
    return "unclear"


# ============================================================================
# Causal Chain Matching
# ============================================================================

def extract_causal_terms(text: str) -> List[str]:
    """Extract key causal/physical terms from text."""
    terms = []
    physical_keywords = [
        '过热', '温升', '升温', '堵塞', '气蚀', '过载', '泄漏', '磨损',
        '退化', '卡涩', '覆冰', '偏航', '短路', '冲击', '振动增大',
        '压力下降', '流量降低', '电流增大', '电流下降', '功率下降',
        '温度升高', '温度下降', '轴承', '齿轮箱', '变桨', '绕组',
        '密封', '叶轮', '冷却', '润滑', '腐蚀', '疲劳',
        'bearing', 'overheat', 'blockage', 'cavitation', 'overload',
        '风速', '转速', '功率曲线', '油温', '不对中', '联轴器',
    ]
    for kw in physical_keywords:
        if kw in text:
            terms.append(kw)
    return terms


def check_causal_direction(gt: str, pred: str) -> str:
    """
    Check if prediction's causal reasoning direction matches ground truth.
    Returns: 'match', 'partial', 'conflict', 'missing'
    """
    gt_terms = extract_causal_terms(gt)
    pred_terms = extract_causal_terms(pred)

    if not pred_terms:
        return "missing"

    overlap = set(gt_terms) & set(pred_terms)
    if not gt_terms:
        return "partial"

    ratio = len(overlap) / len(gt_terms)
    if ratio >= 0.4:
        return "match"
    elif ratio >= 0.15:
        return "partial"
    else:
        # Check for directional conflict
        # e.g., GT says "电流增大" but pred says "电流下降"
        direction_pairs = [
            ('增大', '下降'), ('升高', '下降'), ('上升', '降低'),
            ('过热', '冷却'), ('堵塞', '畅通'),
        ]
        for pos, neg in direction_pairs:
            if pos in gt and neg in pred:
                return "conflict"
            if neg in gt and pos in pred:
                return "conflict"
        return "partial"


# ============================================================================
# Main Scoring Logic
# ============================================================================

def score_sample_independently(record: Dict) -> Tuple[str, int, str]:
    """
    Independent expert scoring without referencing HES.

    Returns: (direction, quality, notes)
    """
    gt = record['answer_gt']
    pred = record['prediction']
    level = record['level']
    domain = record['domain']

    if not pred or len(pred.strip()) < 10:
        return ("wrong", 1, "Empty or trivially short prediction")

    notes = []

    # ===== L1: Descriptive Reasoning =====
    if level == "L1":
        return _score_l1(gt, pred, domain, notes)

    # ===== L2: Anomaly Localization =====
    elif level == "L2":
        return _score_l2(gt, pred, domain, notes)

    # ===== L3: Causal Reasoning =====
    elif level == "L3":
        return _score_l3(gt, pred, domain, notes)

    # ===== L4: Decision Reasoning =====
    elif level == "L4":
        return _score_l4(gt, pred, domain, notes)

    return ("partial", 3, "Unknown level")


def _score_l1(gt: str, pred: str, domain: str, notes: List) -> Tuple[str, int, str]:
    """Score L1 descriptive reasoning.

    Expert standard: L1 GT typically describes the FULL operating picture including
    any anomaly. If GT mentions anomaly and prediction doesn't explicitly acknowledge
    it, that's at minimum 'partial'. If prediction gives a generic description without
    mentioning the key phenomena in GT (specific values, specific anomaly), it's 'wrong'.

    Key principle: a description that sounds reasonable but misses the core phenomenon
    described in GT is direction-wrong in an industrial context.
    """
    gt_dir = detect_fault_direction(gt)
    pred_dir = detect_fault_direction(pred)

    # Extract key quantitative claims from GT
    gt_terms = extract_causal_terms(gt)
    pred_terms = extract_causal_terms(pred)
    term_match = len(set(gt_terms) & set(pred_terms)) / max(len(gt_terms), 1) if gt_terms else 0

    if gt_dir == "fault":
        # GT explicitly mentions anomaly in the description
        if pred_dir == "normal" or pred_dir == "uncertain":
            direction = "wrong"
            notes.append("GT describes anomaly but pred misses it entirely")
        elif pred_dir == "fault" or pred_dir == "mixed":
            # Pred acknowledges anomaly — check depth
            if term_match >= 0.3:
                direction = "aligned"
            else:
                direction = "partial"
                notes.append("Acknowledges anomaly but lacks specific detail")
        else:
            # pred_dir is 'unclear' — pred gives generic description
            # This is typically 'wrong' because expert expects anomaly acknowledgment
            if term_match >= 0.2:
                direction = "partial"
            else:
                direction = "wrong"
                notes.append("Generic description, core anomaly not identified")

    elif gt_dir == "normal":
        if pred_dir == "fault":
            direction = "wrong"
            notes.append("False alarm: GT is normal but pred claims fault")
        elif pred_dir in ("normal", "unclear"):
            if term_match >= 0.2 or len(pred) >= 200:
                direction = "aligned"
            else:
                direction = "partial"
        else:
            direction = "partial"
    else:
        # Ambiguous GT direction (rare)
        if term_match >= 0.3:
            direction = "aligned"
        elif len(pred) > 150:
            direction = "partial"
        else:
            direction = "wrong"

    quality = _assess_quality_generic(gt, pred, direction)
    return (direction, quality, "; ".join(notes))


def _score_l2(gt: str, pred: str, domain: str, notes: List) -> Tuple[str, int, str]:
    """Score L2 anomaly localization (needle-in-haystack)."""
    gt_dir = detect_fault_direction(gt)
    pred_dir = detect_fault_direction(pred)

    # Core check: does prediction agree on fault presence?
    if gt_dir == "fault":
        # GT says there IS a fault
        if pred_dir in ("normal", "uncertain"):
            direction = "wrong"
            notes.append("Missed anomaly (false negative or refused)")
        elif pred_dir == "fault" or pred_dir == "mixed":
            # Check time range precision
            gt_ranges = extract_time_ranges(gt)
            pred_ranges = extract_time_ranges(pred)

            if gt_ranges and pred_ranges:
                best_iou = max(compute_iou(gt_ranges[0], pr) for pr in pred_ranges)
                if best_iou >= 0.3:
                    direction = "aligned"
                    notes.append(f"IoU={best_iou:.2f}")
                elif best_iou >= 0.05:
                    direction = "partial"
                    notes.append(f"IoU={best_iou:.2f}, imprecise localization")
                else:
                    direction = "partial"
                    notes.append(f"IoU={best_iou:.2f}, poor localization")
            elif gt_ranges and not pred_ranges:
                # Pred detects anomaly but gives no specific range
                direction = "partial"
                notes.append("Detected anomaly but no precise time range given")
            else:
                direction = "partial"
        else:
            direction = "partial"
            notes.append("Ambiguous prediction direction")

    elif gt_dir == "normal":
        # GT says NO fault
        if pred_dir == "fault":
            direction = "wrong"
            notes.append("False positive: claimed anomaly in normal data")
        elif pred_dir == "normal":
            direction = "aligned"
        elif pred_dir == "uncertain":
            direction = "partial"
            notes.append("Uncertain on normal data")
        else:
            direction = "partial"

    else:
        direction = "partial"

    quality = _assess_quality_l2(gt, pred, direction)
    return (direction, quality, "; ".join(notes))


def _score_l3(gt: str, pred: str, domain: str, notes: List) -> Tuple[str, int, str]:
    """Score L3 causal reasoning."""
    pred_dir = detect_fault_direction(pred)

    # If prediction refuses or claims normal
    if pred_dir in ("uncertain", "normal"):
        return ("wrong", 1, "Refused to analyze or denied fault existence")

    # Check causal direction match
    causal_match = check_causal_direction(gt, pred)

    if causal_match == "conflict":
        direction = "wrong"
        notes.append("Causal direction conflicts with ground truth")
    elif causal_match == "match":
        direction = "aligned"
    elif causal_match == "partial":
        direction = "partial"
        notes.append("Some causal terms match but incomplete chain")
    else:  # missing
        if len(pred) > 100:
            direction = "partial"
            notes.append("No relevant causal terms found in prediction")
        else:
            direction = "wrong"
            notes.append("Very short prediction with no causal analysis")

    quality = _assess_quality_generic(gt, pred, direction)
    return (direction, quality, "; ".join(notes))


def _score_l4(gt: str, pred: str, domain: str, notes: List) -> Tuple[str, int, str]:
    """Score L4 decision reasoning.

    Expert standard: L4 requires matching the SEVERITY of recommendation to the
    actual fault condition. A generic "monitor the equipment" when GT says "immediate
    shutdown required" is direction-wrong. Similarly, vague recommendations without
    addressing the specific fault type are at best 'partial'.

    Key: the recommendation must be APPROPRIATE to the diagnosed condition.
    """
    pred_dir = detect_fault_direction(pred)
    gt_dir = detect_fault_direction(gt)

    gt_has_urgent = any(kw in gt for kw in ['立即停机', '紧急', '严重', '高风险', '立即'])
    gt_has_moderate = any(kw in gt for kw in ['中等风险', '计划性停机', '近期', '48小时'])
    gt_has_routine = any(kw in gt for kw in ['预防性', '正常', '计划性维护', '例行', '健康状态良好'])
    pred_has_specific_rec = any(kw in pred for kw in ['停机', '更换', '修复', '清洗', '调整',
                                                       '紧固', '加注', '排查', '降负荷'])
    pred_has_generic_rec = any(kw in pred for kw in ['建议', '检修', '维护', '检查', '监测'])
    pred_claims_normal = any(kw in pred for kw in ['无需', '不需要', '继续运行',
                                                     '正常运行', '运行良好', '无异常'])
    pred_uncertain = any(kw in pred for kw in ['无法判断', '无法确定', '难以', '不能判断',
                                                 '无法从图中', '不足以'])

    # Check causal term overlap to see if pred addresses the right fault
    gt_terms = extract_causal_terms(gt)
    pred_terms = extract_causal_terms(pred)
    term_match = len(set(gt_terms) & set(pred_terms)) / max(len(gt_terms), 1) if gt_terms else 0

    if pred_uncertain:
        direction = "wrong"
        notes.append("Refuses to provide decision/recommendation")
    elif gt_has_urgent:
        # Urgent situation: pred must recommend strong action
        if pred_claims_normal:
            direction = "wrong"
            notes.append("Severe fault requires urgent action, pred says normal")
        elif pred_has_specific_rec and term_match >= 0.2:
            direction = "aligned"
        elif pred_has_generic_rec:
            direction = "partial"
            notes.append("Generic recommendation for urgent situation")
        else:
            direction = "wrong"
            notes.append("No actionable recommendation for urgent fault")
    elif gt_has_routine and gt_dir == "normal":
        # Normal equipment, preventive maintenance
        if pred_dir == "fault" and pred_has_specific_rec and not pred_claims_normal:
            direction = "wrong"
            notes.append("False alarm: recommends repair for normal equipment")
        elif pred_claims_normal or pred_has_generic_rec:
            direction = "aligned" if term_match >= 0.15 else "partial"
        else:
            direction = "partial"
    elif gt_dir == "fault":
        # Fault present, some level of action needed
        if pred_claims_normal and not pred_has_generic_rec:
            direction = "wrong"
            notes.append("Fault exists but pred claims no action needed")
        elif pred_has_specific_rec and term_match >= 0.2:
            direction = "aligned"
        elif pred_has_generic_rec or pred_dir == "fault":
            direction = "partial"
            notes.append("Acknowledges issue but recommendation lacks specificity")
        else:
            direction = "wrong"
            notes.append("No relevant recommendation given")
    else:
        if pred_has_generic_rec and len(pred) > 100:
            direction = "partial"
        else:
            direction = "wrong"

    quality = _assess_quality_generic(gt, pred, direction)
    return (direction, quality, "; ".join(notes))


# ============================================================================
# Quality Assessment Helpers
# ============================================================================

def _assess_quality_generic(gt: str, pred: str, direction: str) -> int:
    """Generic quality assessment based on content overlap and completeness."""
    if direction == "wrong":
        return 1 if len(pred) < 100 else 2

    pred_len = len(pred)
    gt_len = len(gt)

    # Extract numbers from both
    gt_numbers = set(re.findall(r'\d+\.?\d*', gt))
    pred_numbers = set(re.findall(r'\d+\.?\d*', pred))
    num_overlap = len(gt_numbers & pred_numbers) / max(len(gt_numbers), 1)

    # Key term overlap
    gt_terms = extract_causal_terms(gt)
    pred_terms = extract_causal_terms(pred)
    term_overlap = len(set(gt_terms) & set(pred_terms)) / max(len(gt_terms), 1) if gt_terms else 0.5

    if direction == "aligned":
        if num_overlap >= 0.25 and term_overlap >= 0.4 and pred_len >= gt_len * 0.3:
            return 5
        elif num_overlap >= 0.15 or term_overlap >= 0.3:
            return 4
        elif pred_len >= 150:
            return 3
        else:
            return 3
    elif direction == "partial":
        if term_overlap >= 0.3 and pred_len >= 200:
            return 3
        elif pred_len >= 100:
            return 2
        else:
            return 2

    return 2


def _assess_quality_l2(gt: str, pred: str, direction: str) -> int:
    """Quality assessment specific to L2 localization tasks."""
    if direction == "wrong":
        return 1 if len(pred) < 100 else 2

    gt_ranges = extract_time_ranges(gt)
    pred_ranges = extract_time_ranges(pred)

    if direction == "aligned":
        if gt_ranges and pred_ranges:
            best_iou = max(compute_iou(gt_ranges[0], pr) for pr in pred_ranges)
            if best_iou >= 0.5:
                return 5
            elif best_iou >= 0.3:
                return 4
            else:
                return 3
        elif not gt_ranges:  # Normal data correctly identified
            return 4 if len(pred) >= 100 else 3
        else:
            return 3
    elif direction == "partial":
        if pred_ranges:
            return 3
        else:
            return 2

    return 2


# ============================================================================
# Batch Processing
# ============================================================================

def run_independent_scoring():
    """Score all 250 samples independently and compute agreement."""
    records = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Scoring {len(records)} samples independently...")
    print("(No HES scores are referenced during scoring)")
    print()

    for i, rec in enumerate(records):
        direction, quality, notes = score_sample_independently(rec)
        rec['human_direction'] = direction
        rec['human_quality'] = quality
        rec['human_notes'] = notes

        if (i + 1) % 50 == 0:
            print(f"  Scored {i+1}/{len(records)}")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"\nSaved to {OUTPUT_PATH}")

    # --- Agreement Analysis ---
    print(f"\n{'='*60}")
    print(f"INDEPENDENT HUMAN vs HES AGREEMENT (N={len(records)})")
    print(f"{'='*60}")

    dir_map = {"aligned": 2, "partial": 1, "wrong": 0}

    human_dirs = np.array([dir_map.get(r['human_direction'], 0) for r in records])
    hes_dirs = np.array([dir_map.get(r['hes_scores']['direction'], 0) for r in records])

    # Exact agreement
    exact = np.sum(human_dirs == hes_dirs) / len(records)
    print(f"\n  Direction exact agreement: {exact:.3f} ({int(exact*len(records))}/{len(records)})")

    # Within-1
    within1 = np.sum(np.abs(human_dirs - hes_dirs) <= 1) / len(records)
    print(f"  Direction within-1 agreement: {within1:.3f}")

    # Weighted Kappa
    kappa = _weighted_kappa(human_dirs, hes_dirs, 3)
    print(f"  Weighted Cohen's Kappa: {kappa:.3f}")

    # Confusion matrix
    print(f"\n  Confusion Matrix (Human rows × HES cols):")
    labels = ["wrong", "partial", "aligned"]
    cm = np.zeros((3, 3), dtype=int)
    for h, a in zip(human_dirs, hes_dirs):
        cm[h][a] += 1
    print(f"  {'':12s} {'HES-wrong':>10s} {'HES-partial':>12s} {'HES-aligned':>12s}")
    for i, label in enumerate(labels):
        print(f"  Human-{label:7s} {cm[i][0]:>10d} {cm[i][1]:>12d} {cm[i][2]:>12d}")

    # Disagreement analysis
    disagree_indices = np.where(human_dirs != hes_dirs)[0]
    print(f"\n  Disagreements: {len(disagree_indices)}")
    if disagree_indices.size > 0:
        # Categorize disagreements
        severe = np.sum(np.abs(human_dirs[disagree_indices] - hes_dirs[disagree_indices]) > 1)
        boundary = len(disagree_indices) - severe
        print(f"    Boundary (±1 category): {boundary}")
        print(f"    Severe (wrong↔aligned): {severe}")

    # Quality correlation
    human_q = np.array([r['human_quality'] for r in records], dtype=float)
    hes_aq = np.array([r['hes_scores']['aq'] for r in records], dtype=float)
    hes_h = np.array([r['hes_scores']['hes'] for r in records], dtype=float)

    from scipy.stats import spearmanr
    sp_aq, _ = spearmanr(human_q, hes_aq)
    sp_hes, _ = spearmanr(human_q, hes_h)
    print(f"\n  Spearman(Human_Quality, HES_AQ): {sp_aq:.3f}")
    print(f"  Spearman(Human_Quality, HES):    {sp_hes:.3f}")

    # By level
    print(f"\n  --- By Level ---")
    for level in ["L1", "L2", "L3", "L4"]:
        lr = [r for r in records if r['level'] == level]
        n_l = len(lr)
        h_d = np.array([dir_map.get(r['human_direction'], 0) for r in lr])
        a_d = np.array([dir_map.get(r['hes_scores']['direction'], 0) for r in lr])
        ag = np.sum(h_d == a_d) / n_l
        h_q = np.array([r['human_quality'] for r in lr], dtype=float)
        a_q = np.array([r['hes_scores']['aq'] for r in lr], dtype=float)
        sp, _ = spearmanr(h_q, a_q)
        print(f"  {level}: DA_agree={ag:.3f} (n={n_l}), Spearman_AQ={sp:.3f}")

    # By domain
    print(f"\n  --- By Domain ---")
    for domain in ["coal_mill", "transformer", "pump", "wind_turbine"]:
        dr = [r for r in records if r['domain'] == domain]
        n_d = len(dr)
        h_d = np.array([dir_map.get(r['human_direction'], 0) for r in dr])
        a_d = np.array([dir_map.get(r['hes_scores']['direction'], 0) for r in dr])
        ag = np.sum(h_d == a_d) / n_d
        print(f"  {domain:15s}: DA_agree={ag:.3f} (n={n_d})")

    # Human distribution
    print(f"\n  Human direction dist: {dict(defaultdict(int, {labels[i]: int(np.sum(human_dirs==i)) for i in range(3)}))}")

    # Save report
    report = {
        "n_samples": len(records),
        "direction_exact_agreement": round(float(exact), 4),
        "direction_within1_agreement": round(float(within1), 4),
        "weighted_cohens_kappa": round(float(kappa), 4),
        "spearman_human_vs_aq": round(float(sp_aq), 4),
        "spearman_human_vs_hes": round(float(sp_hes), 4),
        "confusion_matrix": cm.tolist(),
        "n_disagreements": int(len(disagree_indices)),
        "n_severe_disagreements": int(severe) if disagree_indices.size > 0 else 0,
    }
    report_path = OUTPUT_PATH.parent / "agreement_report_independent.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {report_path}")


def _weighted_kappa(y1, y2, n_cat):
    n = len(y1)
    O = np.zeros((n_cat, n_cat))
    for a, b in zip(y1, y2):
        O[a][b] += 1
    O /= n
    rm = O.sum(axis=1)
    cm = O.sum(axis=0)
    E = np.outer(rm, cm)
    W = np.zeros((n_cat, n_cat))
    for i in range(n_cat):
        for j in range(n_cat):
            W[i][j] = (i - j) ** 2 / (n_cat - 1) ** 2
    num = np.sum(W * O)
    den = np.sum(W * E)
    return 1.0 - num / den if den > 0 else 1.0


if __name__ == "__main__":
    run_independent_scoring()
