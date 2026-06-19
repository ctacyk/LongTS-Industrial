"""
Checkpoint 2: 全量 SNR 审计
============================
扫描 Data/benchmark_tb/ 下全部 metadata JSON，统计：
1. 每个故障类型的 min_snr 分布
2. 识别 invisible 样本 (min_snr < 0.15)
3. 输出报告 + invisible 列表
"""
import os, sys, json, glob
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = "Data/benchmark_tb"
INVISIBLE_THRESHOLD = 0.15  # primary channel SNR below this = invisible

HIGH_CV_CHANNELS = {"power_output", "rotor_speed", "generator_speed"}


def scan_metadata():
    results = []
    for meta_path in sorted(glob.glob(os.path.join(DATA_DIR, "**", "*_metadata.json"), recursive=True)):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        domain = meta.get("domain", "")
        scenario_id = meta.get("scenario_id", "")
        channel_snr = meta.get("channel_snr", {})
        min_snr = meta.get("min_snr")
        events = meta.get("injected_events", [])

        is_normal = not events or (len(events) == 1 and events[0].get("type") == "normal")

        results.append({
            "path": meta_path,
            "domain": domain,
            "scenario_id": scenario_id,
            "channel_snr": channel_snr,
            "min_snr": min_snr,
            "is_normal": is_normal,
            "total_length": meta.get("total_length", 0),
            "seed": meta.get("generation_seed", 0),
        })

    return results


def main():
    results = scan_metadata()
    print(f"Total metadata files: {len(results)}")

    fault_results = [r for r in results if not r["is_normal"]]
    normal_results = [r for r in results if r["is_normal"]]
    print(f"Fault samples: {len(fault_results)}, Normal samples: {len(normal_results)}")

    # Group by domain/fault_type
    by_fault = defaultdict(list)
    for r in fault_results:
        parts = r["scenario_id"].rsplit("_", 1)
        fault_type = parts[0] if len(parts) > 1 else r["scenario_id"]
        by_fault[(r["domain"], fault_type)].append(r)

    # Find invisible samples: ALL channels below threshold (not just min)
    invisible = []
    for r in fault_results:
        snr_vals = r["channel_snr"]
        if not snr_vals:
            continue
        max_snr = max(snr_vals.values())
        r["max_snr"] = max_snr
        if max_snr < INVISIBLE_THRESHOLD:
            invisible.append(r)

    # Summary by domain
    domain_stats = defaultdict(lambda: {"total": 0, "invisible": 0, "low_snr": 0})
    for r in fault_results:
        d = domain_stats[r["domain"]]
        d["total"] += 1
        max_snr = r.get("max_snr", r.get("min_snr", 999))
        if max_snr is not None:
            if max_snr < INVISIBLE_THRESHOLD:
                d["invisible"] += 1
            elif max_snr < 0.5:
                d["low_snr"] += 1

    print(f"\n{'='*60}")
    print("Domain Summary:")
    print(f"{'='*60}")
    for domain in sorted(domain_stats):
        s = domain_stats[domain]
        pct = s["invisible"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {domain}: {s['total']} fault samples, "
              f"{s['invisible']} invisible ({pct:.1f}%), {s['low_snr']} low_snr")

    # Detailed invisible list
    if invisible:
        print(f"\n{'='*60}")
        print(f"Invisible samples ({len(invisible)}):")
        print(f"{'='*60}")
        for r in sorted(invisible, key=lambda x: (x["domain"], x["scenario_id"])):
            snr_str = ", ".join(f"{k}={v:.2f}" for k, v in r["channel_snr"].items())
            basename = os.path.basename(r["path"]).replace("_metadata.json", "")
            print(f"  {basename}: min_snr={r['min_snr']:.3f} [{snr_str}]")

    # Per fault-type stats
    print(f"\n{'='*60}")
    print("Per fault-type min_snr stats:")
    print(f"{'='*60}")
    for (domain, fault_type), samples in sorted(by_fault.items()):
        min_snrs = [s["min_snr"] for s in samples if s["min_snr"] is not None]
        if not min_snrs:
            continue
        avg = sum(min_snrs) / len(min_snrs)
        worst = min(min_snrs)
        invis_count = sum(1 for x in min_snrs if x < INVISIBLE_THRESHOLD)
        flag = " **INVISIBLE**" if invis_count > 0 else ""
        print(f"  {domain}/{fault_type}: avg={avg:.2f}, worst={worst:.2f}, "
              f"n={len(min_snrs)}, invisible={invis_count}{flag}")

    # Overall stats
    total_fault = len(fault_results)
    total_invisible = len(invisible)
    pct = total_invisible / total_fault * 100 if total_fault > 0 else 0
    print(f"\n{'='*60}")
    print(f"OVERALL: {total_fault} fault samples, "
          f"{total_invisible} invisible ({pct:.1f}%)")
    target = "PASS" if pct < 5 else "FAIL"
    print(f"Target: invisible < 5% → {target}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
