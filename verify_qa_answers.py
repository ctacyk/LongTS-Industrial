"""
Checkpoint 3.1: QA 答案程序化验证
=================================
验证维度：
  1. 数值验证: 答案中提及的数值是否与 data_profile 统计一致 (允许 15% 误差)
  2. 方向验证: 答案描述的变化方向是否与 data_profile 一致
  3. 时间步验证: L2 答案中的异常区间是否与 metadata 一致
  4. 因果链验证: L3 答案是否覆盖 metadata 中的因果链关键步骤
"""
import json
import os
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

QA_PATH = "Data/benchmark_tb/synthetic_qa_llm.jsonl"
DATA_DIR = "Data/benchmark_tb"

NUM_TOLERANCE = 0.15  # 数值允许 15% 相对误差
TIME_TOLERANCE = 5    # 时间步允许 ±5 步误差


def extract_numbers(text):
    """从文本中提取数字（含小数、负数）"""
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', text)]


def extract_time_ranges(text):
    """从 L2 答案中提取时间步范围 (start, end)"""
    patterns = [
        r'第\s*(\d+)\s*步.*?第\s*(\d+)\s*步',
        r'(\d+)\s*[-–到至]\s*(\d+)\s*步',
        r'时间步\s*(\d+)\s*[-–到至]\s*(\d+)',
    ]
    ranges = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            s, e = int(m.group(1)), int(m.group(2))
            if s < e and e - s > 5:
                ranges.append((s, e))
    return ranges


def check_numerical_accuracy(answer, profile):
    """检查答案中数值与 data_profile 是否一致"""
    if not profile or "channels" not in profile:
        return None, "no_profile"

    reference_vals = set()
    for ch_data in profile["channels"].values():
        for section in ["normal", "fault", "global"]:
            if section in ch_data:
                for v in ch_data[section].values():
                    if isinstance(v, (int, float)):
                        reference_vals.add(round(v, 2))

    answer_nums = extract_numbers(answer)
    if not answer_nums:
        return None, "no_numbers_in_answer"

    matched = 0
    total_checked = 0
    for num in answer_nums:
        if abs(num) < 0.5 or abs(num) > 100000:
            continue
        total_checked += 1
        for ref in reference_vals:
            if abs(ref) < 1e-9:
                continue
            if abs(num - ref) / abs(ref) < NUM_TOLERANCE:
                matched += 1
                break

    if total_checked == 0:
        return None, "no_relevant_numbers"
    return matched / total_checked, f"{matched}/{total_checked}"


def check_direction(answer, profile, metadata=None):
    """检查答案描述的变化方向是否与 data_profile 一致"""
    if not profile or "channels" not in profile:
        return None, "no_profile"

    correct = 0
    total = 0

    ch_display_map = {}
    if hasattr(metadata, 'get'):
        for ch in metadata.get("channels", []):
            if isinstance(ch, dict):
                ch_display_map[ch.get("name", "")] = ch.get("display_name", "")

    for ch_name, ch_data in profile["channels"].items():
        if "change" not in ch_data:
            continue
        change = ch_data["change"]
        if not change.get("is_affected"):
            continue

        direction = change["direction"]
        total += 1

        ch_display = ch_display_map.get(ch_name, ch_name)
        if ch_name in answer or ch_display in answer:
            if direction == "increase" and any(w in answer for w in ["上升", "升高", "增大", "增加", "升至", "升到", "增长"]):
                correct += 1
            elif direction == "decrease" and any(w in answer for w in ["下降", "降低", "减小", "减少", "降至", "降到"]):
                correct += 1
            elif direction == "stable" and any(w in answer for w in ["稳定", "不变", "基本不变", "无明显变化"]):
                correct += 1
            elif direction == "oscillation" and any(w in answer for w in ["振荡", "波动", "脉动"]):
                correct += 1

    if total == 0:
        return None, "no_affected_channels"
    return correct / total, f"{correct}/{total}"


def check_time_range(qa_pairs, metadata):
    """检查 L2 答案中的时间步范围"""
    events = metadata.get("injected_events", [])
    if not events or events[0].get("event_type") != "fault":
        return None, "normal_sample"

    true_range = tuple(events[0]["time_range"])
    true_start, true_end = true_range

    for qa in qa_pairs:
        if qa.get("level") != "L2":
            continue

        answer = qa.get("answer", "")
        extracted = extract_time_ranges(answer)

        if not extracted:
            return 0.0, "no_range_found"

        for (es, ee) in extracted:
            if abs(es - true_start) <= TIME_TOLERANCE and abs(ee - true_end) <= TIME_TOLERANCE:
                return 1.0, f"exact_match ({es}-{ee} vs {true_start}-{true_end})"

        best = extracted[0]
        start_err = abs(best[0] - true_start)
        end_err = abs(best[1] - true_end)
        window = true_end - true_start
        overlap_start = max(best[0], true_start)
        overlap_end = min(best[1], true_end)
        iou = max(0, overlap_end - overlap_start) / (max(best[1], true_end) - min(best[0], true_start))

        return iou, f"iou={iou:.2f} ({best[0]}-{best[1]} vs {true_start}-{true_end})"

    return None, "no_L2_qa"


def _extract_ngrams(text, n=3):
    """从中文文本提取 n-gram 片段"""
    chars = re.findall(r'[\u4e00-\u9fff]', text)
    return set(''.join(chars[i:i+n]) for i in range(len(chars) - n + 1))


def check_causal_chain(qa_pairs, metadata):
    """检查 L3 答案是否覆盖因果链关键步骤（n-gram 模糊匹配）"""
    events = metadata.get("injected_events", [])
    if not events or events[0].get("event_type") != "fault":
        return None, "normal_sample"

    causal_chain = events[0].get("causal_chain", [])
    if len(causal_chain) < 2:
        return None, "short_chain"

    l3_answers = " ".join(qa["answer"] for qa in qa_pairs if qa.get("level") == "L3")
    if not l3_answers:
        return None, "no_L3_qa"

    answer_trigrams = _extract_ngrams(l3_answers, 3)

    covered = 0
    for step in causal_chain:
        step_trigrams = _extract_ngrams(step, 3)
        if not step_trigrams:
            short_kws = re.findall(r'[\u4e00-\u9fff]{2,}', step)
            if any(kw in l3_answers for kw in short_kws):
                covered += 1
            continue
        overlap = len(step_trigrams & answer_trigrams)
        if overlap / len(step_trigrams) >= 0.4:
            covered += 1

    return covered / len(causal_chain), f"{covered}/{len(causal_chain)}"


def main():
    with open(QA_PATH, 'r', encoding='utf-8') as f:
        records = [json.loads(l) for l in f if l.strip()]

    print(f"Loaded {len(records)} QA records")

    # Build mapping: record_id -> (meta_path, profile_path) by scanning filesystem
    id_to_paths = {}
    for domain_dir in os.listdir(DATA_DIR):
        dpath = os.path.join(DATA_DIR, domain_dir)
        if not os.path.isdir(dpath):
            continue
        for fname in os.listdir(dpath):
            if fname.endswith("_metadata.json"):
                fpath = os.path.join(dpath, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as mf:
                        meta = json.load(mf)
                    rid_key = (f"{meta.get('domain', domain_dir)}_{meta.get('scenario_id', '')}"
                               f"_L{meta.get('total_length', 0)}_S{meta.get('generation_seed', 0)}")
                    pfpath = fpath.replace("_metadata.json", "_data_profile.json")
                    id_to_paths[rid_key] = (fpath, pfpath)
                except Exception:
                    pass

    print(f"Indexed {len(id_to_paths)} metadata files")

    results = {
        "numerical": {"pass": 0, "warn": 0, "fail": 0, "skip": 0, "scores": []},
        "direction": {"pass": 0, "warn": 0, "fail": 0, "skip": 0, "scores": []},
        "time_range": {"pass": 0, "warn": 0, "fail": 0, "skip": 0, "scores": []},
        "causal": {"pass": 0, "warn": 0, "fail": 0, "skip": 0, "scores": []},
    }
    domain_results = defaultdict(lambda: {"pass": 0, "warn": 0, "fail": 0})
    fail_examples = []

    unmatched = 0
    for r in records:
        rid = r["id"]
        domain = r["domain"]
        qa_pairs = r.get("qa_pairs", [])

        paths = id_to_paths.get(rid)
        if not paths:
            unmatched += 1
            continue
        meta_path, profile_path = paths

        meta = None
        profile = None
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        if os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)

        if not meta:
            continue

        all_answers = " ".join(qa["answer"] for qa in qa_pairs)
        sample_pass = True

        # 1. Numerical accuracy (check L1 answers)
        l1_answer = " ".join(qa["answer"] for qa in qa_pairs if qa.get("level") == "L1")
        score, detail = check_numerical_accuracy(l1_answer, profile)
        if score is not None:
            results["numerical"]["scores"].append(score)
            if score >= 0.5:
                results["numerical"]["pass"] += 1
            elif score >= 0.3:
                results["numerical"]["warn"] += 1
            else:
                results["numerical"]["fail"] += 1
                sample_pass = False
        else:
            results["numerical"]["skip"] += 1

        # 2. Direction check
        score, detail = check_direction(all_answers, profile, meta)
        if score is not None:
            results["direction"]["scores"].append(score)
            if score >= 0.5:
                results["direction"]["pass"] += 1
            elif score > 0:
                results["direction"]["warn"] += 1
            else:
                results["direction"]["fail"] += 1
                sample_pass = False
        else:
            results["direction"]["skip"] += 1

        # 3. Time range (L2)
        score, detail = check_time_range(qa_pairs, meta)
        if score is not None:
            results["time_range"]["scores"].append(score)
            if score >= 0.8:
                results["time_range"]["pass"] += 1
            elif score >= 0.5:
                results["time_range"]["warn"] += 1
            else:
                results["time_range"]["fail"] += 1
                if len(fail_examples) < 5:
                    fail_examples.append({"id": rid, "check": "time_range", "detail": detail})
        else:
            results["time_range"]["skip"] += 1

        # 4. Causal chain (L3)
        score, detail = check_causal_chain(qa_pairs, meta)
        if score is not None:
            results["causal"]["scores"].append(score)
            if score >= 0.5:
                results["causal"]["pass"] += 1
            elif score > 0:
                results["causal"]["warn"] += 1
            else:
                results["causal"]["fail"] += 1
                sample_pass = False
        else:
            results["causal"]["skip"] += 1

        if sample_pass:
            domain_results[domain]["pass"] += 1
        else:
            domain_results[domain]["fail"] += 1

    # Report
    print("\n" + "=" * 60)
    print("QA ANSWER VERIFICATION REPORT")
    print("=" * 60)

    for check_name, data in results.items():
        scores = data["scores"]
        avg = sum(scores) / len(scores) if scores else 0
        print(f"\n[{check_name.upper()}]")
        print(f"  PASS: {data['pass']}, WARN: {data['warn']}, FAIL: {data['fail']}, SKIP: {data['skip']}")
        print(f"  Avg score: {avg:.3f} (n={len(scores)})")

    print(f"\n[DOMAIN SUMMARY]")
    for domain, counts in sorted(domain_results.items()):
        total = counts["pass"] + counts["warn"] + counts["fail"]
        print(f"  {domain}: PASS={counts['pass']}/{total}")

    if fail_examples:
        print(f"\n[FAIL EXAMPLES]")
        for ex in fail_examples:
            print(f"  {ex['id']}: {ex['check']} - {ex['detail']}")

    # Save report
    report = {
        "summary": {k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in results.items()},
        "avg_scores": {k: (sum(v["scores"]) / len(v["scores"]) if v["scores"] else 0) for k, v in results.items()},
        "domain_summary": dict(domain_results),
        "fail_examples": fail_examples,
    }
    report_path = os.path.join("results", "qa_verification_report.json")
    os.makedirs("results", exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
