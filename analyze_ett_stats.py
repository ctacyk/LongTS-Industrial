import pandas as pd, numpy as np

df = pd.read_csv('MultiAgentTS/Dataset/ETT/ETTh2.csv')
cols = list(df.columns[1:])

print("=== ETT Real Data Stats (17420 rows, 1h, ~2 years) ===")
header = "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8}".format("Channel", "Mean", "Std", "Min", "Max", "AC1")
print(header)
for c in cols:
    v = df[c].values
    ac = np.corrcoef(v[:-1], v[1:])[0, 1]
    print("{:<12} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.4f}".format(c, v.mean(), v.std(), v.min(), v.max(), ac))

print("\n=== Correlation Matrix ===")
print(df.iloc[:, 1:].corr().round(3).to_string())

# Daily pattern analysis (first 30 days = 720 hours)
print("\n=== Daily Pattern (hour-of-day mean, first 30 days) ===")
df_30 = df.iloc[:720].copy()
df_30['hour'] = [i % 24 for i in range(len(df_30))]
for c in ['HUFL', 'OT']:
    hourly = df_30.groupby('hour')[c].mean()
    print("{}: peak_hour={}, peak_val={:.1f}, trough_hour={}, trough_val={:.1f}".format(
        c, hourly.idxmax(), hourly.max(), hourly.idxmin(), hourly.min()))
