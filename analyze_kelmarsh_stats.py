"""
Kelmarsh Wind Farm SCADA 数据统计分析 — 风电 TimeBlender 校准目标

数据源: Zenodo https://zenodo.org/record/5841834
机型: Senvion MM92, 2050 kW, 92m 转子直径, 78.5m 轮毂高度
采样: 10min SCADA 均值

分析目标 8 通道:
  Wind speed (m/s), Rotor speed (RPM), Generator RPM (RPM),
  Power (kW), Blade angle (pitch position) A,
  Nacelle temperature, Gear oil temperature,
  Generator bearing front temperature
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = "dataset/Kelmarsh"
TURBINE_FILE = "Turbine_Data_Kelmarsh_1_2019-01-01_-_2020-01-01_228.csv"

CHANNEL_MAP = {
    "Wind speed (m/s)": "wind_speed",
    "Rotor speed (RPM)": "rotor_speed",
    "Generator RPM (RPM)": "generator_speed",
    "Power (kW)": "power_output",
}

CHANNEL_MAP_FUZZY = {
    "Blade angle (pitch position) A": "pitch_angle",
    "Nacelle temperature": "nacelle_temp",
    "Gear oil temperature": "gearbox_oil_temp",
    "Generator bearing front temperature": "gen_bearing_temp",
}

RATED_POWER = 2050  # kW
CUT_IN_WIND = 3.0   # m/s
RATED_WIND = 12.5   # m/s (Senvion MM92 typical)
CUT_OUT_WIND = 25.0  # m/s


def load_data():
    path = os.path.join(DATA_DIR, TURBINE_FILE)
    df_raw = pd.read_csv(path, skiprows=9, parse_dates=["# Date and time"])
    df_raw = df_raw.rename(columns={"# Date and time": "timestamp"})

    cols_found = {}
    for orig, target in CHANNEL_MAP.items():
        if orig in df_raw.columns:
            cols_found[orig] = target

    for prefix, target in CHANNEL_MAP_FUZZY.items():
        matches = [c for c in df_raw.columns if c.startswith(prefix) and "Max" not in c
                    and "Min" not in c and "Standard" not in c and "Std" not in c]
        if matches:
            cols_found[matches[0]] = target

    rename = {orig: target for orig, target in cols_found.items()}
    df = df_raw[["timestamp"] + list(rename.keys())].copy()
    df = df.rename(columns=rename)

    print(f"Loaded: {len(df)} rows, columns: {list(df.columns)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def compute_stats(df):
    print("\n" + "=" * 80)
    print("KELMARSH SCADA STATISTICS — Calibration Targets for TimeBlender")
    print("=" * 80)

    channels = [c for c in df.columns if c != "timestamp"]
    stats = {}

    for ch in channels:
        s = df[ch].dropna()
        ac1 = s.autocorr(lag=1) if len(s) > 10 else np.nan
        ac6 = s.autocorr(lag=6) if len(s) > 10 else np.nan

        stats[ch] = {
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "q25": s.quantile(0.25),
            "q75": s.quantile(0.75),
            "AC1": ac1,
            "AC6": ac6,
            "n_valid": len(s),
            "pct_missing": (1 - len(s) / len(df)) * 100,
        }

        print(f"\n  {ch}:")
        print(f"    mean={stats[ch]['mean']:.2f}, std={stats[ch]['std']:.2f}")
        print(f"    range=[{stats[ch]['min']:.2f}, {stats[ch]['max']:.2f}]")
        print(f"    q25={stats[ch]['q25']:.2f}, q75={stats[ch]['q75']:.2f}")
        print(f"    AC1={stats[ch]['AC1']:.4f}, AC6={stats[ch]['AC6']:.4f}")
        print(f"    valid={stats[ch]['n_valid']}, missing={stats[ch]['pct_missing']:.1f}%")

    # Operational mode breakdown
    ws = df["wind_speed"].dropna()
    pw = df["power_output"].dropna()
    aligned = df[["wind_speed", "power_output"]].dropna()

    n_total = len(aligned)
    n_stopped = len(aligned[aligned["power_output"] <= 0])
    n_below_rated = len(aligned[(aligned["power_output"] > 0) & (aligned["wind_speed"] < RATED_WIND)])
    n_above_rated = len(aligned[(aligned["power_output"] > 0) & (aligned["wind_speed"] >= RATED_WIND)])

    print(f"\n  Operational modes (n={n_total}):")
    print(f"    Stopped (P<=0):      {n_stopped} ({100*n_stopped/n_total:.1f}%)")
    print(f"    Below rated (<{RATED_WIND}m/s): {n_below_rated} ({100*n_below_rated/n_total:.1f}%)")
    print(f"    Above rated (>={RATED_WIND}m/s): {n_above_rated} ({100*n_above_rated/n_total:.1f}%)")

    # Correlation matrix
    corr = df[channels].corr()
    print(f"\n  Correlation matrix:")
    print(corr.round(3).to_string())

    return stats, corr


def plot_overview(df, stats, corr):
    channels = [c for c in df.columns if c != "timestamp"]
    os.makedirs(os.path.join(DATA_DIR, "analysis"), exist_ok=True)

    # 1. Time series overview
    fig, axes = plt.subplots(len(channels), 1, figsize=(18, 3 * len(channels)), sharex=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for idx, ch in enumerate(channels):
        ax = axes[idx]
        vals = df[ch].values
        ax.plot(vals, color=colors[idx], linewidth=0.3, alpha=0.8)
        ax.set_ylabel(ch, fontsize=8, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        s = stats[ch]
        ax.set_title(f"{ch}: mean={s['mean']:.1f}, std={s['std']:.1f}, AC1={s['AC1']:.3f}",
                      fontsize=8, loc='left')

    axes[0].set_title("Kelmarsh Turbine 1 — 2019 SCADA (10min)", fontsize=12, fontweight='bold')
    axes[-1].set_xlabel("Time Step (10min intervals)")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "analysis", "kelmarsh_overview.png"), dpi=150)
    plt.close()
    print(f"\nSaved: {DATA_DIR}/analysis/kelmarsh_overview.png")

    # 2. Power curve
    aligned = df[["wind_speed", "power_output"]].dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(aligned["wind_speed"], aligned["power_output"],
               s=1, alpha=0.3, c='#1f77b4')
    ax.set_xlabel("Wind Speed (m/s)", fontsize=11)
    ax.set_ylabel("Power (kW)", fontsize=11)
    ax.set_title("Kelmarsh Turbine 1 — Power Curve (2019)", fontsize=13, fontweight='bold')
    ax.axhline(y=RATED_POWER, color='red', linestyle='--', alpha=0.5, label=f'Rated: {RATED_POWER} kW')
    ax.axvline(x=CUT_IN_WIND, color='green', linestyle='--', alpha=0.5, label=f'Cut-in: {CUT_IN_WIND} m/s')
    ax.axvline(x=RATED_WIND, color='orange', linestyle='--', alpha=0.5, label=f'Rated: {RATED_WIND} m/s')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "analysis", "kelmarsh_power_curve.png"), dpi=150)
    plt.close()
    print(f"Saved: {DATA_DIR}/analysis/kelmarsh_power_curve.png")

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(channels)))
    ax.set_yticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(channels, fontsize=8)
    for i in range(len(channels)):
        for j in range(len(channels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha='center', va='center', fontsize=7)
    plt.colorbar(im)
    ax.set_title("Channel Correlation Matrix", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "analysis", "kelmarsh_correlation.png"), dpi=150)
    plt.close()
    print(f"Saved: {DATA_DIR}/analysis/kelmarsh_correlation.png")

    # 4. Histograms
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, ch in enumerate(channels):
        ax = axes[idx]
        vals = df[ch].dropna().values
        ax.hist(vals, bins=80, color=colors[idx], alpha=0.7, edgecolor='none')
        ax.set_title(ch, fontsize=9, fontweight='bold')
        ax.axvline(stats[ch]['mean'], color='red', linewidth=1, linestyle='--')
        ax.grid(True, alpha=0.2)
    plt.suptitle("Channel Distributions", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "analysis", "kelmarsh_distributions.png"), dpi=150)
    plt.close()
    print(f"Saved: {DATA_DIR}/analysis/kelmarsh_distributions.png")


if __name__ == "__main__":
    df = load_data()
    stats, corr = compute_stats(df)
    plot_overview(df, stats, corr)
    print("\n=== Done ===")
