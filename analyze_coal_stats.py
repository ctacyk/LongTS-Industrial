import pandas as pd
import numpy as np

df = pd.read_csv('MultiAgentTS/Dataset/CoalMill/merged.csv')
cols = list(df.columns[1:])

print("=== Coal Mill Real Data Stats (85860 rows, 30s, ~30 days) ===")
header = "{:<35} {:>8} {:>8} {:>8} {:>8} {:>8}".format("Channel", "Mean", "Std", "Min", "Max", "AC1")
print(header)
for c in cols:
    v = df[c].values
    ac = np.corrcoef(v[:-1], v[1:])[0, 1]
    row = "{:<35} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.4f}".format(c, v.mean(), v.std(), v.min(), v.max(), ac)
    print(row)

print("\n=== Correlation Matrix ===")
print(df.iloc[:, 1:].corr().round(3).to_string())

# Shutdown analysis
current = df.iloc[:, 1].values
shutdown_mask = current < 5
transitions = np.diff(shutdown_mask.astype(int))
n_shutdowns = int(np.sum(transitions == 1))
shutdown_steps = int(np.sum(shutdown_mask))
print("\n=== Shutdown Analysis ===")
print("Shutdown events (current<5A): {} events".format(n_shutdowns))
print("Shutdown steps: {} ({:.1f}% of total)".format(shutdown_steps, shutdown_steps / len(current) * 100))
if n_shutdowns > 0:
    avg_dur = shutdown_steps / n_shutdowns
    print("Average shutdown duration: {:.0f} steps = {:.1f} hours".format(avg_dur, avg_dur * 30 / 3600))

# Shutdown interval analysis
shutdown_starts = np.where(transitions == 1)[0]
if len(shutdown_starts) > 1:
    intervals = np.diff(shutdown_starts)
    print("Shutdown intervals: mean={:.0f} steps ({:.1f}h), min={:.0f}, max={:.0f}".format(
        intervals.mean(), intervals.mean() * 30 / 3600, intervals.min(), intervals.max()))
