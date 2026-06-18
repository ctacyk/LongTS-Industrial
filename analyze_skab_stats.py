import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

skab_dir = 'MultiAgentTS/Dataset/SKAB'

# Load first anomaly-free file (should be 0.csv based on SKAB convention)
# Actually let's find the right one by checking content
csvs = sorted([f for f in os.listdir(skab_dir) if f.endswith('.csv')])
print(f"Found {len(csvs)} CSV files: {csvs[:5]}...")

# Load and analyze first file
df = pd.read_csv(os.path.join(skab_dir, csvs[0]), sep=';')
print(f"\nFile: {csvs[0]}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())

# Stats for sensor columns (exclude datetime and anomaly)
sensor_cols = [c for c in df.columns if c not in ['datetime', 'anomaly', 'changepoint']]
print(f"\n=== SKAB Sensor Stats ===")
print("{:<25} {:>8} {:>8} {:>8} {:>8} {:>8}".format("Channel", "Mean", "Std", "Min", "Max", "AC1"))
for c in sensor_cols:
    v = df[c].dropna().values
    if len(v) > 1:
        ac = np.corrcoef(v[:-1], v[1:])[0, 1]
    else:
        ac = 0
    print("{:<25} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.4f}".format(
        c, v.mean(), v.std(), v.min(), v.max(), ac))

# Correlation
print("\n=== Correlation Matrix ===")
print(df[sensor_cols].corr().round(3).to_string())

# Plot first file
fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(16, 2.5 * len(sensor_cols)), sharex=True)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
for idx, c in enumerate(sensor_cols):
    axes[idx].plot(df[c].values, color=colors[idx % len(colors)], linewidth=0.5)
    axes[idx].set_ylabel(c, fontsize=7)
    axes[idx].grid(True, alpha=0.2)
    # Mark anomaly regions if present
    if 'anomaly' in df.columns:
        anom = df['anomaly'].values
        for i in range(1, len(anom)):
            if anom[i] == 1 and anom[i-1] == 0:
                axes[idx].axvline(i, color='red', alpha=0.3, linewidth=0.5)
axes[0].set_title(f'SKAB: {csvs[0]} ({df.shape[0]} steps)', fontsize=10, fontweight='bold')
plt.tight_layout()
os.makedirs('MultiAgentTS/Dataset/SKAB/plots', exist_ok=True)
plt.savefig('MultiAgentTS/Dataset/SKAB/plots/sample_0.png', dpi=150)
plt.close()
print("\nSaved: MultiAgentTS/Dataset/SKAB/plots/sample_0.png")
