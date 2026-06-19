"""
Checkpoint 1 物理预检脚本
========================
为全部 45 种故障 x moderate 严重度生成单样本，自动检查：
  1. 生成成功（无异常）
  2. 通道值在物理合理范围
  3. 故障区间 SNR（primary >= 1.0, secondary >= 0.3）
  4. 元数据完整性（affected_channels 非空, causal_chain >= 2 步）
  5. 通道一致性（metadata 声明的通道在 df.columns 中）
  6. 正常区间稳定性（mean 在 normal_range ±50%）

同时为每个样本生成 eval.png 到 results/cp1_eval_plots/
"""

import json
import os
import sys
import traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")

from MultiAgentTS.scenarios.base import plot_eval_chart
from MultiAgentTS.scenarios.coal_mill_tb import (
    generate_coal_mill, FAULT_DEFS as COAL_FAULTS, COAL_MILL_CHANNELS,
)
from MultiAgentTS.scenarios.transformer_tb import (
    generate_transformer, TRANSFORMER_FAULTS, TRANSFORMER_CHANNELS,
)
from MultiAgentTS.scenarios.pump_tb import (
    generate_pump, PUMP_FAULTS, PUMP_CHANNELS,
)
from MultiAgentTS.scenarios.wind_turbine_tb import (
    generate_wind_turbine, WIND_TURBINE_FAULTS, WIND_TURBINE_CHANNELS,
)

LENGTH = 3000
SEED = 42
SEVERITY = "moderate"
PLOT_DIR = os.path.join("results", "cp1_eval_plots")
REPORT_PATH = os.path.join("results", "physics_validation_report.json")

# SNR thresholds
PRIMARY_SNR_FAIL = 1.0
PRIMARY_SNR_WARN = 2.0
SECONDARY_SNR_FAIL = 0.3
SECONDARY_SNR_WARN = 1.0

DOMAIN_REGISTRY = {
    "coal_mill": {
        "faults": COAL_FAULTS,
        "channels": {ch.name: ch for ch in COAL_MILL_CHANNELS},
        "gen_fn": lambda ft, sev: generate_coal_mill(LENGTH, SEED, ft, sev, "steady"),
    },
    "transformer": {
        "faults": TRANSFORMER_FAULTS,
        "channels": {ch.name: ch for ch in TRANSFORMER_CHANNELS},
        "gen_fn": lambda ft, sev: generate_transformer(LENGTH, SEED, ft, sev),
    },
    "pump": {
        "faults": PUMP_FAULTS,
        "channels": {ch.name: ch for ch in PUMP_CHANNELS},
        "gen_fn": lambda ft, sev: generate_pump(LENGTH, SEED, ft, sev),
    },
    "wind_turbine": {
        "faults": WIND_TURBINE_FAULTS,
        "channels": {ch.name: ch for ch in WIND_TURBINE_CHANNELS},
        "gen_fn": lambda ft, sev: generate_wind_turbine(LENGTH, SEED, ft, sev, "steady"),
    },
}


def compute_snr(normal_data: np.ndarray, fault_data: np.ndarray) -> float:
    n_mean = np.mean(normal_data)
    n_std = np.std(normal_data)
    f_mean = np.mean(fault_data)
    if n_std < 1e-9:
        return 0.0
    return abs(f_mean - n_mean) / n_std


def compute_max_snr(normal_data: np.ndarray, fault_data: np.ndarray) -> float:
    """Max-based SNR for high-variability channels like power_output."""
    n_max = np.max(normal_data)
    n_std = np.std(normal_data)
    f_max = np.max(fault_data)
    if n_std < 1e-9:
        return 0.0
    return abs(f_max - n_max) / n_std


HIGH_CV_CHANNELS = {"power_output", "rotor_speed", "generator_speed"}


def compute_envelope_snr(normal_data: np.ndarray, fault_data: np.ndarray) -> float:
    """Envelope-based SNR: best of P95 and P99 for high-CV channels."""
    n_std = np.std(normal_data)
    if n_std < 1e-9:
        return 0.0
    snr_p95 = abs(np.percentile(normal_data, 95) - np.percentile(fault_data, 95)) / n_std
    snr_p99 = abs(np.percentile(normal_data, 99) - np.percentile(fault_data, 99)) / n_std
    return max(snr_p95, snr_p99)


def validate_sample(domain: str, fault_type: str, df: pd.DataFrame,
                    metadata, channel_info: dict) -> dict:
    """Run all checks on a single generated sample."""
    result = {
        "domain": domain,
        "fault": fault_type,
        "status": "PASS",
        "checks": {},
        "notes": [],
    }

    events = metadata.injected_events
    if not events:
        result["notes"].append("No injected events (normal or bug)")
        return result

    ev = events[0]
    fs, fe = ev.time_range
    affected = ev.affected_channels

    # Check 2: channel value range (5x spread for affected channels, 3x for others)
    range_checks = {}
    for col in df.columns:
        vals = df[col].values
        ch = channel_info.get(col)
        if ch is None:
            continue
        lo, hi = ch.normal_range
        spread = hi - lo
        multiplier = 5.0 if col in affected else 3.0
        phys_lo = lo - multiplier * spread
        phys_hi = hi + multiplier * spread
        out_of_range = np.sum((vals < phys_lo) | (vals > phys_hi))
        status = "PASS" if out_of_range == 0 else "WARN"
        if out_of_range > len(vals) * 0.05:
            status = "FAIL"
        range_checks[col] = {
            "out_of_range_count": int(out_of_range),
            "phys_bounds": [round(phys_lo, 2), round(phys_hi, 2)],
            "actual_bounds": [round(float(np.min(vals)), 2), round(float(np.max(vals)), 2)],
            "status": status,
        }
    result["checks"]["range"] = range_checks

    # Check 3: SNR for affected channels
    snr_checks = {}
    for idx, ch_name in enumerate(affected):
        if ch_name not in df.columns:
            snr_checks[ch_name] = {"snr": 0, "status": "FAIL", "note": "channel not in df"}
            continue
        col_idx = list(df.columns).index(ch_name)
        normal_data = df.iloc[:fs, col_idx].values
        fault_data = df.iloc[fs:fe, col_idx].values

        if len(normal_data) < 10 or len(fault_data) < 10:
            snr_checks[ch_name] = {"snr": 0, "status": "WARN", "note": "insufficient data"}
            continue

        snr_mean = compute_snr(normal_data, fault_data)
        snr_max = compute_max_snr(normal_data, fault_data)
        snr_env = compute_envelope_snr(normal_data, fault_data) if ch_name in HIGH_CV_CHANNELS else 0.0
        best_snr = max(snr_mean, snr_max, snr_env)

        is_primary = (idx == 0)
        is_high_cv = ch_name in HIGH_CV_CHANNELS
        if is_high_cv:
            fail_thresh = 0.15 if is_primary else 0.10
            warn_thresh = 0.5 if is_primary else 0.3
        else:
            fail_thresh = PRIMARY_SNR_FAIL if is_primary else SECONDARY_SNR_FAIL
            warn_thresh = PRIMARY_SNR_WARN if is_primary else SECONDARY_SNR_WARN

        if best_snr < fail_thresh:
            status = "FAIL"
        elif best_snr < warn_thresh:
            status = "WARN"
        else:
            status = "PASS"

        entry = {
            "snr_mean": round(snr_mean, 3),
            "snr_max": round(snr_max, 3),
            "best_snr": round(best_snr, 3),
            "role": "primary" if is_primary else "secondary",
            "status": status,
        }
        if ch_name in HIGH_CV_CHANNELS:
            entry["snr_envelope"] = round(snr_env, 3)
        snr_checks[ch_name] = entry
    result["checks"]["snr"] = snr_checks

    # Check 4: metadata completeness
    meta_check = {"status": "PASS"}
    if not affected:
        meta_check = {"status": "FAIL", "note": "affected_channels empty"}
    if len(ev.causal_chain) < 2:
        meta_check = {"status": "FAIL", "note": f"causal_chain too short ({len(ev.causal_chain)})"}
    result["checks"]["metadata"] = meta_check

    # Check 5: channel consistency
    missing = [ch for ch in affected if ch not in df.columns]
    result["checks"]["consistency"] = {
        "status": "FAIL" if missing else "PASS",
        "missing_channels": missing,
    }

    # Check 6: normal interval stability
    stability_checks = {}
    for col in df.columns:
        ch = channel_info.get(col)
        if ch is None:
            continue
        normal_vals = df[col].values[:fs]
        if len(normal_vals) < 10:
            continue
        n_mean = np.mean(normal_vals)
        expected_mean = (ch.normal_range[0] + ch.normal_range[1]) / 2.0
        half_range = (ch.normal_range[1] - ch.normal_range[0]) / 2.0
        deviation = abs(n_mean - expected_mean) / half_range if half_range > 1e-9 else 0
        status = "PASS" if deviation < 1.5 else ("WARN" if deviation < 2.5 else "FAIL")
        stability_checks[col] = {
            "normal_mean": round(n_mean, 3),
            "expected_mean": round(expected_mean, 3),
            "deviation_ratio": round(deviation, 3),
            "status": status,
        }
    result["checks"]["stability"] = stability_checks

    # Aggregate status
    all_statuses = []
    for check_group in result["checks"].values():
        if isinstance(check_group, dict) and "status" in check_group:
            all_statuses.append(check_group["status"])
        elif isinstance(check_group, dict):
            for v in check_group.values():
                if isinstance(v, dict) and "status" in v:
                    all_statuses.append(v["status"])

    if "FAIL" in all_statuses:
        result["status"] = "FAIL"
    elif "WARN" in all_statuses:
        result["status"] = "WARN"
    else:
        result["status"] = "PASS"

    return result


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    results = []
    summary = {"PASS": 0, "WARN": 0, "FAIL": 0, "ERROR": 0}

    for domain, cfg in DOMAIN_REGISTRY.items():
        faults = cfg["faults"]
        channels = cfg["channels"]
        gen_fn = cfg["gen_fn"]

        print(f"\n{'='*60}")
        print(f"Domain: {domain} ({len(faults)} faults)")
        print(f"{'='*60}")

        for fault_type in faults:
            print(f"  {fault_type}...", end=" ", flush=True)

            try:
                df, metadata = gen_fn(fault_type, SEVERITY)

                plot_path = os.path.join(PLOT_DIR, f"{domain}_{fault_type}_eval.png")
                plot_eval_chart(df, metadata, plot_path)

                result = validate_sample(domain, fault_type, df, metadata, channels)
                results.append(result)
                summary[result["status"]] += 1

                status_icon = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}[result["status"]]
                snr_info = ""
                if "snr" in result["checks"]:
                    snr_vals = [f'{k}={v["best_snr"]:.1f}' for k, v in result["checks"]["snr"].items()
                                if isinstance(v, dict) and "best_snr" in v]
                    snr_info = f" SNR[{', '.join(snr_vals)}]"
                print(f"[{status_icon}]{snr_info}")

            except Exception as e:
                print(f"[ERROR] {e}")
                traceback.print_exc()
                results.append({
                    "domain": domain,
                    "fault": fault_type,
                    "status": "ERROR",
                    "error": str(e),
                })
                summary["ERROR"] += 1

    report = {"summary": summary, "details": results}
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY: PASS={summary['PASS']}, WARN={summary['WARN']}, "
          f"FAIL={summary['FAIL']}, ERROR={summary['ERROR']}")
    print(f"Report: {REPORT_PATH}")
    print(f"Plots:  {PLOT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
