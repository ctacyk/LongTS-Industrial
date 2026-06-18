"""
Phase 3: 为每个 CSV 生成 _data_profile.json
==============================================
统计内容：
  - 全局：每通道 mean/std/min/max/P5/P95
  - 正常区间：同上
  - 故障区间：同上 + 与正常区间的 delta 描述
  - 通道变化摘要（自然语言）
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = "Data/benchmark_tb"

DIRECTION_MAP = {
    "increase": "上升",
    "decrease": "下降",
    "stable": "基本稳定",
    "oscillation": "出现振荡",
}


def describe_change(n_mean, f_mean, n_std, direction_thresh=0.5):
    """Generate a natural language description of a channel's change."""
    if n_std < 1e-9:
        return "stable", "基本稳定"
    delta = f_mean - n_mean
    ratio = abs(delta) / n_std
    if ratio < direction_thresh:
        return "stable", f"基本稳定（均值 {f_mean:.2f}，变化 {delta:+.2f}）"
    direction = "increase" if delta > 0 else "decrease"
    magnitude = "轻微" if ratio < 1.5 else ("明显" if ratio < 4 else "大幅")
    cn_dir = "上升" if delta > 0 else "下降"
    return direction, f"{magnitude}{cn_dir}（{n_mean:.2f} → {f_mean:.2f}，变化 {delta:+.2f}）"


def compute_stats(arr):
    if len(arr) == 0:
        return {}
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "p5": round(float(np.percentile(arr, 5)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
    }


def compute_profile(csv_path, meta_path):
    df = pd.read_csv(csv_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    events = meta.get("injected_events", [])
    has_fault = events and events[0].get("event_type") == "fault"

    profile = {
        "scenario_id": meta.get("scenario_id", ""),
        "domain": meta.get("domain", ""),
        "total_length": len(df),
        "channels": {},
    }

    if has_fault:
        ev = events[0]
        fs, fe = ev["time_range"]
        affected = ev.get("affected_channels", [])
        profile["fault_type"] = ev.get("fault_name", "unknown")
        profile["severity"] = ev.get("severity", "")
        profile["fault_window"] = [fs, fe]
        profile["affected_channels"] = affected
        profile["causal_chain"] = ev.get("causal_chain", [])
    else:
        fs, fe = len(df), len(df)
        affected = []
        profile["fault_type"] = "normal"

    for col in df.columns:
        vals = df[col].values
        ch_profile = {"global": compute_stats(vals)}

        if has_fault and fs > 10:
            normal_vals = vals[:fs]
            fault_vals = vals[fs:fe] if fe > fs else np.array([])

            ch_profile["normal"] = compute_stats(normal_vals)
            if len(fault_vals) > 0:
                ch_profile["fault"] = compute_stats(fault_vals)

                n_mean = np.mean(normal_vals)
                f_mean = np.mean(fault_vals)
                n_std = np.std(normal_vals)
                direction, desc = describe_change(n_mean, f_mean, n_std)
                ch_profile["change"] = {
                    "direction": direction,
                    "description": desc,
                    "is_affected": col in affected,
                }

        profile["channels"][col] = ch_profile

    return profile


def main():
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True))
    print(f"Found {len(csv_files)} CSV files")

    generated = 0
    skipped = 0
    errors = 0

    for csv_path in csv_files:
        base = csv_path.replace(".csv", "")
        meta_path = f"{base}_metadata.json"
        profile_path = f"{base}_data_profile.json"

        if not os.path.exists(meta_path):
            continue

        try:
            profile = compute_profile(csv_path, meta_path)
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            generated += 1

            if generated % 200 == 0:
                print(f"  [{generated}/{len(csv_files)}] {os.path.basename(csv_path)}")

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ERROR: {os.path.basename(csv_path)}: {e}")

    print(f"\nDone: generated={generated}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    main()
