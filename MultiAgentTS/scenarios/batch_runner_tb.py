"""
TimeBlender 版批量运行器 — 直接调用生成函数（不依赖 registry）
"""
import os, sys, time, json, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')

from MultiAgentTS.scenarios.base import (
    GenerationMetadata, plot_eval_chart, compute_difficulty,
)
from MultiAgentTS.scenarios.coal_mill_tb import generate_coal_mill, FAULT_DEFS as COAL_FAULTS
from MultiAgentTS.scenarios.transformer_tb import generate_transformer, TRANSFORMER_FAULTS
from MultiAgentTS.scenarios.pump_tb import generate_pump, PUMP_FAULTS
from MultiAgentTS.scenarios.wind_turbine_tb import generate_wind_turbine, WIND_TURBINE_FAULTS

HIGH_CV_CHANNELS = {"power_output", "rotor_speed", "generator_speed"}


def compute_sample_snr(df, metadata):
    """Compute per-channel SNR for fault samples. Returns dict {channel: best_snr}."""
    if not metadata.injected_events:
        return {}
    ev = metadata.injected_events[0]
    fs, fe = ev.time_range
    affected = ev.affected_channels
    snr_dict = {}
    for ch_name in affected:
        if ch_name not in df.columns:
            continue
        normal_data = df[ch_name].values[:fs]
        fault_data = df[ch_name].values[fs:fe]
        if len(normal_data) < 10 or len(fault_data) < 10:
            continue
        n_mean, n_std = np.mean(normal_data), np.std(normal_data)
        f_mean = np.mean(fault_data)
        if n_std < 1e-9:
            snr_dict[ch_name] = 0.0
            continue
        snr_mean = abs(f_mean - n_mean) / n_std
        snr_max = abs(np.max(fault_data) - np.max(normal_data)) / n_std
        snr_env = 0.0
        if ch_name in HIGH_CV_CHANNELS:
            snr_p95 = abs(np.percentile(normal_data, 95) - np.percentile(fault_data, 95)) / n_std
            snr_p99 = abs(np.percentile(normal_data, 99) - np.percentile(fault_data, 99)) / n_std
            snr_env = max(snr_p95, snr_p99)
        snr_dict[ch_name] = round(max(snr_mean, snr_max, snr_env), 3)
    return snr_dict

LENGTHS = [1000, 3000, 5000, 10000]
SEEDS = [42, 123, 789]
OUTPUT_BASE = "Data/benchmark_tb"


def build_scenario_list():
    """构建所有场景的 (domain, fault_type, severity, gen_fn) 列表"""
    scenarios = []

    # Coal Mill: 3 normal + 15 faults x 3 severity = 48
    for mode in ["steady", "varying", "frequent_startstop"]:
        ft = "normal" if mode != "frequent_startstop" else "frequent_startstop"
        scenarios.append(("coal_mill", f"normal_{mode}", "moderate",
                          lambda tl, s, _m=mode, _ft=ft: generate_coal_mill(tl, s, _ft, "moderate", _m)))

    for ft in COAL_FAULTS:
        for sev in ["mild", "moderate", "severe"]:
            scenarios.append(("coal_mill", f"{ft}_{sev}", sev,
                              lambda tl, s, _ft=ft, _sev=sev: generate_coal_mill(tl, s, _ft, _sev, "steady")))

    # Transformer: 1 normal + 10 faults x 3 severity = 31
    scenarios.append(("transformer", "normal", "moderate",
                      lambda tl, s: generate_transformer(tl, s, "normal")))
    for ft in TRANSFORMER_FAULTS:
        for sev in ["mild", "moderate", "severe"]:
            scenarios.append(("transformer", f"{ft}_{sev}", sev,
                              lambda tl, s, _ft=ft, _sev=sev: generate_transformer(tl, s, _ft, _sev)))

    # Pump: 1 normal + 10 faults x 3 severity = 31
    scenarios.append(("pump", "normal", "moderate",
                      lambda tl, s: generate_pump(tl, s, "normal")))
    for ft in PUMP_FAULTS:
        for sev in ["mild", "moderate", "severe"]:
            scenarios.append(("pump", f"{ft}_{sev}", sev,
                              lambda tl, s, _ft=ft, _sev=sev: generate_pump(tl, s, _ft, _sev)))

    # Wind Turbine: 3 normal + 10 faults x 3 severity = 33
    for mode in ["steady", "varying", "turbulent"]:
        scenarios.append(("wind_turbine", f"normal_{mode}", "moderate",
                          lambda tl, s, _m=mode: generate_wind_turbine(tl, s, "normal", "moderate", _m)))
    for ft in WIND_TURBINE_FAULTS:
        for sev in ["mild", "moderate", "severe"]:
            scenarios.append(("wind_turbine", f"{ft}_{sev}", sev,
                              lambda tl, s, _ft=ft, _sev=sev: generate_wind_turbine(tl, s, _ft, _sev, "steady")))

    return scenarios


def run_batch(output_base=None, lengths=None, seeds=None,
              generate_eval_plots=True, skip_existing=True):

    if output_base is None: output_base = OUTPUT_BASE
    if lengths is None: lengths = LENGTHS
    if seeds is None: seeds = SEEDS

    scenarios = build_scenario_list()
    total = len(scenarios) * len(lengths) * len(seeds)

    print(f"=== TB Batch Runner ===")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Lengths: {lengths}, Seeds: {seeds}")
    print(f"Total: {total}")

    generated = 0
    skipped = 0
    errors = []
    start_time = time.time()

    for domain, scenario_id, severity, gen_fn in scenarios:
        output_dir = os.path.join(output_base, domain)
        os.makedirs(output_dir, exist_ok=True)

        for length in lengths:
            for seed in seeds:
                base_name = f"{scenario_id}_L{length}_S{seed}"
                csv_path = os.path.join(output_dir, f"{base_name}.csv")

                if skip_existing and os.path.exists(csv_path):
                    skipped += 1
                    continue

                try:
                    df, metadata = gen_fn(length, seed)

                    # Save CSV
                    df.to_csv(csv_path, index=False)

                    # Compute and attach SNR
                    snr_dict = compute_sample_snr(df, metadata)
                    metadata.channel_snr = snr_dict
                    metadata.min_snr = min(snr_dict.values()) if snr_dict else None

                    # Save metadata
                    meta_path = os.path.join(output_dir, f"{base_name}_metadata.json")
                    metadata.scenario_id = scenario_id
                    metadata.total_length = len(df)
                    metadata.generation_seed = seed
                    metadata.save(meta_path)

                    # Eval chart
                    if generate_eval_plots:
                        eval_path = os.path.join(output_dir, f"{base_name}_eval.png")
                        plot_eval_chart(df, metadata, eval_path)

                    generated += 1

                    done = generated + skipped
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0

                    if generated % 50 == 0 or generated <= 3:
                        print(f"  [{done}/{total}] {base_name} ({rate:.1f}/s, ETA {eta:.0f}s)")

                except Exception as e:
                    errors.append((scenario_id, length, seed, str(e)))
                    if len(errors) <= 5:
                        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n=== Done ===")
    print(f"Generated: {generated}, Skipped: {skipped}, Errors: {len(errors)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    if errors:
        print(f"\nFirst 5 errors:")
        for sid, l, s, msg in errors[:5]:
            print(f"  {sid} L={l} S={s}: {msg}")

    manifest = {
        "generated": generated, "skipped": skipped, "errors": len(errors),
        "lengths": lengths, "seeds": seeds, "scenario_count": len(scenarios),
    }
    os.makedirs(output_base, exist_ok=True)
    with open(os.path.join(output_base, "generation_manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {os.path.join(output_base, 'generation_manifest.json')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_BASE)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    run_batch(output_base=args.output,
              generate_eval_plots=not args.no_plots,
              skip_existing=not args.no_skip)
