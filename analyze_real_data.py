import pandas as pd
import numpy as np

df = pd.read_csv('MultiAgentTS/Dataset/CoalMill/merged.csv')
print('Shape:', df.shape)
print('\nColumns:', list(df.columns))
print('\nBasic stats:')
for col in df.columns[1:]:
    vals = df[col].values
    print(f'  {col}:')
    print(f'    mean={vals.mean():.3f}, std={vals.std():.3f}, min={vals.min():.3f}, max={vals.max():.3f}')
    ac1 = np.corrcoef(vals[:-1], vals[1:])[0,1]
    print(f'    autocorr(1)={ac1:.4f}')

print('\nCorrelation matrix:')
corr = df.iloc[:,1:].corr()
print(corr.round(3).to_string())

print('\nTotal duration:', (pd.to_datetime(df['timestamp'].iloc[-1]) - pd.to_datetime(df['timestamp'].iloc[0])))

# Also check ETT
df2 = pd.read_csv('MultiAgentTS/Dataset/ETT/ETTh2.csv')
print('\n\n=== ETT Data ===')
print('Shape:', df2.shape)
for col in df2.columns[1:]:
    vals = df2[col].values
    print(f'  {col}: mean={vals.mean():.3f}, std={vals.std():.3f}, min={vals.min():.3f}, max={vals.max():.3f}')
