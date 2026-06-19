"""
批量运行器：遍历所有注册场景，生成全部时序数据 + 元数据

每个场景 × 4 个时间步长度 × 3 个随机种子 = 12 个变体
77 个场景 × 12 = 924 条时序数据

输出结构：
  Data/benchmark/{domain}/{scenario_id}_L{length}_S{seed}.csv
  Data/benchmark/{domain}/{scenario_id}_L{length}_S{seed}_metadata.json
  Data/benchmark/{domain}/{scenario_id}_L{length}_S{seed}_overview.png
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from MultiAgentTS.scenarios.base import get_all_scenarios, save_scenario_output

# Import domains to trigger registration
from MultiAgentTS.scenarios import coal_mill, transformer, pump


# ============================================================================
# 配置
# ============================================================================

LENGTHS = [1000, 3000, 5000, 10000]
SEEDS = [42, 123, 789]
OUTPUT_BASE = "Data/benchmark"
GENERATE_PLOTS = True  # 设为 False 可加速（不生成PNG）


def run_batch(domains=None, lengths=None, seeds=None,
              output_base=None, generate_plots=True,
              skip_existing=True):
    """
    批量执行所有场景

    Args:
        domains: 指定领域列表，None=全部
        lengths: 时间步长度列表
        seeds: 随机种子列表
        output_base: 输出根目录
        generate_plots: 是否生成概览图
        skip_existing: 是否跳过已存在的文件
    """
    if lengths is None:
        lengths = LENGTHS
    if seeds is None:
        seeds = SEEDS
    if output_base is None:
        output_base = OUTPUT_BASE

    all_scenarios = get_all_scenarios()

    # Filter by domain
    if domains:
        all_scenarios = {k: v for k, v in all_scenarios.items()
                         if v['domain'] in domains}

    total = len(all_scenarios) * len(lengths) * len(seeds)
    print(f"=== Batch Runner ===")
    print(f"Scenarios: {len(all_scenarios)}")
    print(f"Lengths: {lengths}")
    print(f"Seeds: {seeds}")
    print(f"Total to generate: {total}")
    print(f"Output: {output_base}")
    print()

    generated = 0
    skipped = 0
    errors = []
    start_time = time.time()

    for scenario_key in sorted(all_scenarios.keys()):
        scenario = all_scenarios[scenario_key]
        domain = scenario['domain']
        scenario_id = scenario['scenario_id']
        gen_fn = scenario['generate_fn']

        output_dir = os.path.join(output_base, domain)

        for length in lengths:
            for seed in seeds:
                base_name = f"{scenario_id}_L{length}_S{seed}"
                csv_path = os.path.join(output_dir, f"{base_name}.csv")

                # Skip if exists
                if skip_existing and os.path.exists(csv_path):
                    skipped += 1
                    continue

                try:
                    df, metadata = gen_fn(total_length=length, seed=seed)
                    save_scenario_output(df, metadata, output_dir,
                                         generate_plot=generate_plots)
                    generated += 1

                    # Progress
                    elapsed = time.time() - start_time
                    done = generated + skipped
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0

                    if generated % 50 == 0:
                        print(f"  [{done}/{total}] Generated {base_name} "
                              f"({rate:.1f}/s, ETA {eta:.0f}s)")

                except Exception as e:
                    errors.append((scenario_key, length, seed, str(e)))
                    traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n=== Done ===")
    print(f"Generated: {generated}")
    print(f"Skipped (existing): {skipped}")
    print(f"Errors: {len(errors)}")
    print(f"Time: {elapsed:.1f}s")

    if errors:
        print("\nErrors:")
        for key, l, s, msg in errors:
            print(f"  {key} L={l} S={s}: {msg}")

    # Save manifest
    manifest = {
        "total_generated": generated,
        "total_skipped": skipped,
        "total_errors": len(errors),
        "lengths": lengths,
        "seeds": seeds,
        "domains": list(set(v['domain'] for v in all_scenarios.values())),
        "scenario_count": len(all_scenarios),
        "errors": [{"key": k, "length": l, "seed": s, "error": m}
                   for k, l, s, m in errors],
    }
    manifest_path = os.path.join(output_base, "generation_manifest.json")
    os.makedirs(output_base, exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nManifest saved: {manifest_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch generate benchmark data")
    parser.add_argument("--domains", nargs="*", default=None,
                        help="Domains to generate (coal_mill, transformer, pump)")
    parser.add_argument("--lengths", nargs="*", type=int, default=LENGTHS)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--output", default=OUTPUT_BASE)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    run_batch(
        domains=args.domains,
        lengths=args.lengths,
        seeds=args.seeds,
        output_base=args.output,
        generate_plots=not args.no_plots,
        skip_existing=not args.no_skip,
    )
