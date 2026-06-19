"""
Benchmark Ground Truth 数据一致性审计脚本
========================================
对所有 benchmark 样本检查:
1. 故障区间信号变化的信噪比 (SNR)
2. GT 答案中数值声明与实际 CSV 数据的一致性
3. 按 severity / domain / scenario 分组统计
"""
import json
import os
import re
import sys
import numpy as np
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_JSONL = os.path.join(BASE_DIR, "Data", "benchmark_tb",
                          "LongTS_Industrial", "test.jsonl")
DATA_DIR = os.path.join(BASE_DIR, "Data", "benchmark_tb")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "gt_audit_report.json")


def load_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
        headers = [h.strip() for h in header_line.split(",")]
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return headers, data


def extract_numeric_claims(answer_text, channel_name):
    """Try to extract 'from X to Y' style numeric claims for a channel."""
    cn_names = {
        "motor_current": ["电机电流", "motor_current", "电流"],
        "winding_temp_1": ["绕组温度1", "winding_temp_1", "绕组温度"],
        "winding_temp_2": ["绕组温度2", "winding_temp_2"],
        "bearing_temp_nondrive": ["非驱动侧轴承温度", "bearing_temp_nondrive"],
        "bearing_temp_drive": ["驱动侧轴承温度", "bearing_temp_drive", "轴承温度"],
        "inlet_temp": ["入口温度", "inlet_temp"],
        "inlet_valve_opening": ["入口调节阀开度", "inlet_valve_opening", "阀门开度"],
        "vibration_pump": ["泵侧振动", "vibration_pump"],
        "vibration_motor": ["电机侧振动", "vibration_motor"],
        "pressure": ["压力", "pressure", "出口压力"],
        "flow_rate": ["流量", "flow_rate"],
        "motor_temp": ["电机温度", "motor_temp"],
        "motor_voltage": ["电机电压", "motor_voltage", "电压"],
        "fluid_temp": ["流体温度", "fluid_temp", "循环液温度"],
        "HUFL": ["高压侧有功负载", "HUFL", "高压侧"],
        "MUFL": ["中压侧有功负载", "MUFL", "中压侧"],
        "LUFL": ["低压侧有功负载", "LUFL"],
        "HULL": ["高压侧无功", "HULL"],
        "MULL": ["中压侧无功", "MULL"],
        "LULL": ["低压侧无功", "LULL"],
        "OT": ["油温", "OT"],
        "wind_speed": ["风速", "wind_speed"],
        "rotor_speed": ["转子转速", "rotor_speed"],
        "generator_speed": ["发电机转速", "generator_speed"],
        "power_output": ["有功功率", "power_output", "功率"],
        "pitch_angle": ["桨距角", "pitch_angle"],
        "nacelle_temp": ["机舱温度", "nacelle_temp"],
        "gearbox_oil_temp": ["齿轮箱油温", "gearbox_oil_temp"],
        "gen_bearing_temp": ["发电机轴承温度", "gen_bearing_temp"],
    }
    aliases = cn_names.get(channel_name, [channel_name])
    claims = []
    for alias in aliases:
        patterns = [
            rf"{re.escape(alias)}[^。]*?从[约]?(\d+\.?\d*)\s*[A°%℃CmMkKgGRWBVLbar]*\s*[升上缓慢]*[升至到达]*[约]?\s*(\d+\.?\d*)",
            rf"{re.escape(alias)}[^。]*?(\d+\.?\d*)\s*[–\-~至到]\s*(\d+\.?\d*)",
            rf"{re.escape(alias)}[^。]*?维持在[约]?(\d+\.?\d*)\s*[–\-~至到]\s*(\d+\.?\d*)",
            rf"{re.escape(alias)}[^。]*?稳定在[约]?(\d+\.?\d*)\s*[–\-~至到]\s*(\d+\.?\d*)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, answer_text):
                try:
                    v1, v2 = float(m.group(1)), float(m.group(2))
                    claims.append((v1, v2))
                except (ValueError, IndexError):
                    pass
    return claims


def audit_sample(sample):
    """Audit a single sample. Returns a list of channel-level findings."""
    events = sample.get("ground_truth_events", [])
    if not events:
        return []

    ts_path_raw = sample["timeseries_path"].replace("\\", "/")
    domain = sample["domain"]
    csv_filename = os.path.basename(ts_path_raw)
    csv_path = os.path.join(DATA_DIR, domain, csv_filename)

    if not os.path.exists(csv_path):
        return [{"error": f"CSV not found: {csv_path}", "id": sample["id"]}]

    try:
        headers, data = load_csv(csv_path)
    except Exception as e:
        return [{"error": f"CSV load error: {e}", "id": sample["id"]}]

    total_len = data.shape[0]

    l1_answer = ""
    l2_answer = ""
    for qa in sample.get("qa_pairs", []):
        if qa["level"] == "L1":
            l1_answer = qa.get("answer", "")
        elif qa["level"] == "L2":
            l2_answer = qa.get("answer", "")

    combined_answer = l1_answer + " " + l2_answer

    findings = []
    for event in events:
        fault_start, fault_end = event["range"]
        affected = event.get("affected_channels", [])
        severity = event.get("severity", "unknown")
        fault_type = event.get("type", "unknown")

        if fault_start >= total_len:
            findings.append({
                "id": sample["id"], "domain": domain, "severity": severity,
                "fault_type": fault_type, "error": "fault_start >= data length"
            })
            continue

        fault_end_clip = min(fault_end + 1, total_len)

        normal_end = max(fault_start, 1)
        normal_data = data[:normal_end]
        fault_data = data[fault_start:fault_end_clip]

        if len(normal_data) < 10 or len(fault_data) < 5:
            continue

        for ch_name in affected:
            if ch_name not in headers:
                findings.append({
                    "id": sample["id"], "domain": domain, "severity": severity,
                    "fault_type": fault_type, "channel": ch_name,
                    "error": f"channel '{ch_name}' not in CSV headers"
                })
                continue

            ch_idx = headers.index(ch_name)
            n_vals = normal_data[:, ch_idx]
            f_vals = fault_data[:, ch_idx]

            n_mean = float(np.mean(n_vals))
            n_std = float(np.std(n_vals))
            n_min = float(np.min(n_vals))
            n_max = float(np.max(n_vals))
            f_mean = float(np.mean(f_vals))
            f_std = float(np.std(f_vals))
            f_min = float(np.min(f_vals))
            f_max = float(np.max(f_vals))

            delta = abs(f_mean - n_mean)
            snr = delta / n_std if n_std > 1e-9 else 0.0
            pct_change = (delta / abs(n_mean) * 100) if abs(n_mean) > 1e-9 else 0.0

            gt_claims = extract_numeric_claims(combined_answer, ch_name)
            claim_issues = []
            for (v1, v2) in gt_claims:
                lo, hi = min(v1, v2), max(v1, v2)
                actual_range_lo = min(n_min, f_min)
                actual_range_hi = max(n_max, f_max)
                if hi > actual_range_hi * 1.3 or lo < actual_range_lo * 0.7:
                    claim_issues.append({
                        "claimed_range": [v1, v2],
                        "actual_data_range": [round(actual_range_lo, 2),
                                              round(actual_range_hi, 2)],
                        "deviation": "GT claim outside actual data range (>30%)"
                    })
                elif abs(hi - actual_range_hi) / max(abs(actual_range_hi), 1e-9) > 0.2:
                    claim_issues.append({
                        "claimed_range": [v1, v2],
                        "actual_data_range": [round(actual_range_lo, 2),
                                              round(actual_range_hi, 2)],
                        "deviation": "GT claim >20% off from actual"
                    })

            finding = {
                "id": sample["id"],
                "domain": domain,
                "scenario_id": sample["scenario_id"],
                "severity": severity,
                "fault_type": fault_type,
                "channel": ch_name,
                "fault_range": [fault_start, fault_end],
                "normal_mean": round(n_mean, 3),
                "normal_std": round(n_std, 3),
                "fault_mean": round(f_mean, 3),
                "fault_max": round(f_max, 3),
                "fault_min": round(f_min, 3),
                "delta": round(delta, 3),
                "snr": round(snr, 3),
                "pct_change": round(pct_change, 2),
                "snr_category": (
                    "invisible" if snr < 1.0
                    else "weak" if snr < 2.0
                    else "marginal" if snr < 3.0
                    else "clear"
                ),
                "gt_claim_issues": claim_issues,
            }
            findings.append(finding)

    return findings


def main():
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"已加载 {len(samples)} 个样本")

    all_findings = []
    errors = []
    skipped_normal = 0
    processed = 0

    for i, sample in enumerate(samples):
        if not sample.get("ground_truth_events"):
            skipped_normal += 1
            continue
        results = audit_sample(sample)
        for r in results:
            if "error" in r:
                errors.append(r)
            else:
                all_findings.append(r)
        processed += 1
        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{len(samples)} ...")

    print(f"\n处理完成: {processed} 有故障样本, {skipped_normal} 正常样本跳过, "
          f"{len(errors)} 错误")
    print(f"共 {len(all_findings)} 条通道级检查结果\n")

    # ---- Aggregate statistics ----
    sev_stats = defaultdict(lambda: {"total": 0, "invisible": 0, "weak": 0,
                                      "marginal": 0, "clear": 0,
                                      "claim_issues": 0})
    domain_stats = defaultdict(lambda: {"total": 0, "invisible": 0, "weak": 0,
                                         "marginal": 0, "clear": 0,
                                         "claim_issues": 0})
    scenario_stats = defaultdict(lambda: {"total": 0, "invisible": 0, "weak": 0,
                                           "marginal": 0, "clear": 0,
                                           "claim_issues": 0})
    overall = {"total": 0, "invisible": 0, "weak": 0, "marginal": 0,
               "clear": 0, "claim_issues": 0}

    for f in all_findings:
        cat = f["snr_category"]
        sev = f["severity"]
        dom = f["domain"]
        scen = f"{dom}/{f['scenario_id']}"
        has_claim_issue = 1 if f["gt_claim_issues"] else 0

        for stats in [sev_stats[sev], domain_stats[dom],
                      scenario_stats[scen], overall]:
            stats["total"] += 1
            stats[cat] += 1
            stats["claim_issues"] += has_claim_issue

    def pct(n, total):
        return round(n / total * 100, 1) if total > 0 else 0

    # ---- Print summary ----
    print("=" * 70)
    print("总体统计")
    print("=" * 70)
    print(f"  通道检查总数: {overall['total']}")
    print(f"  invisible (SNR<1, 图上不可见): {overall['invisible']} "
          f"({pct(overall['invisible'], overall['total'])}%)")
    print(f"  weak (1≤SNR<2, 微弱):          {overall['weak']} "
          f"({pct(overall['weak'], overall['total'])}%)")
    print(f"  marginal (2≤SNR<3):             {overall['marginal']} "
          f"({pct(overall['marginal'], overall['total'])}%)")
    print(f"  clear (SNR≥3, 明显):            {overall['clear']} "
          f"({pct(overall['clear'], overall['total'])}%)")
    print(f"  GT数值声明偏差: {overall['claim_issues']} "
          f"({pct(overall['claim_issues'], overall['total'])}%)")

    print(f"\n{'=' * 70}")
    print("按 severity 分组")
    print("=" * 70)
    for sev in ["mild", "moderate", "severe"]:
        s = sev_stats.get(sev, {"total": 0})
        if s["total"] == 0:
            continue
        print(f"\n  [{sev.upper()}] (n={s['total']})")
        print(f"    invisible: {s['invisible']} ({pct(s['invisible'], s['total'])}%)")
        print(f"    weak:      {s['weak']} ({pct(s['weak'], s['total'])}%)")
        print(f"    marginal:  {s['marginal']} ({pct(s['marginal'], s['total'])}%)")
        print(f"    clear:     {s['clear']} ({pct(s['clear'], s['total'])}%)")
        print(f"    GT偏差:    {s['claim_issues']} ({pct(s['claim_issues'], s['total'])}%)")

    print(f"\n{'=' * 70}")
    print("按 domain 分组")
    print("=" * 70)
    for dom in sorted(domain_stats.keys()):
        s = domain_stats[dom]
        print(f"\n  [{dom}] (n={s['total']})")
        print(f"    invisible: {s['invisible']} ({pct(s['invisible'], s['total'])}%)")
        print(f"    weak:      {s['weak']} ({pct(s['weak'], s['total'])}%)")
        print(f"    clear:     {s['clear']} ({pct(s['clear'], s['total'])}%)")
        print(f"    GT偏差:    {s['claim_issues']} ({pct(s['claim_issues'], s['total'])}%)")

    print(f"\n{'=' * 70}")
    print("问题最严重的 scenario (invisible 比例最高)")
    print("=" * 70)
    ranked = sorted(scenario_stats.items(),
                    key=lambda x: x[1]["invisible"] / max(x[1]["total"], 1),
                    reverse=True)
    for scen, s in ranked[:20]:
        if s["total"] < 3:
            continue
        inv_pct = pct(s["invisible"], s["total"])
        print(f"  {scen:55s} invisible={s['invisible']}/{s['total']} ({inv_pct}%)")

    # ---- Detailed problem samples ----
    problem_samples = [f for f in all_findings if f["snr_category"] == "invisible"]
    problem_samples.sort(key=lambda x: x["snr"])

    print(f"\n{'=' * 70}")
    print(f"信号不可见 (SNR<1) 的详细列表 (前30条)")
    print("=" * 70)
    for f in problem_samples[:30]:
        print(f"  {f['id']}")
        print(f"    channel={f['channel']}, severity={f['severity']}")
        print(f"    normal: mean={f['normal_mean']}, std={f['normal_std']}")
        print(f"    fault:  mean={f['fault_mean']}, max={f['fault_max']}")
        print(f"    delta={f['delta']}, SNR={f['snr']}, pct_change={f['pct_change']}%")
        if f["gt_claim_issues"]:
            for ci in f["gt_claim_issues"]:
                print(f"    GT声称: {ci['claimed_range']} vs 实际: {ci['actual_data_range']}")
        print()

    # ---- Save report ----
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    report = {
        "overall": overall,
        "by_severity": dict(sev_stats),
        "by_domain": dict(domain_stats),
        "by_scenario": dict(scenario_stats),
        "invisible_samples": problem_samples,
        "all_findings_count": len(all_findings),
        "errors": errors,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n审计报告已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
