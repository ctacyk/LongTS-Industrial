"""Pack final unified test set for LongTS-Industrial Benchmark v3
v3: 4 synthetic domains (coal_mill, transformer, pump, wind_turbine)
    Data profiles injected into QA prompt for numerical accuracy.
"""
import json, os, random
from collections import defaultdict

QA_PATH = 'Data/benchmark_tb/synthetic_qa_llm.jsonl'

records = []
with open(QA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

all_records = records
random.Random(42).shuffle(all_records)

print("Total records: {}, Unique IDs: {}".format(
    len(records), len(set(r['id'] for r in records))))

out_dir = 'Data/benchmark_tb/LongTS_Industrial'
os.makedirs(out_dir, exist_ok=True)

# Remove old splits
for old in ['test.jsonl', 'test_needle.jsonl', 'train.jsonl', 'val.jsonl']:
    p = os.path.join(out_dir, old)
    if os.path.exists(p):
        os.remove(p)

path = os.path.join(out_dir, 'test.jsonl')
with open(path, 'w', encoding='utf-8') as f:
    for r in all_records:
        if 'timeseries_path' in r and not r['timeseries_path'].startswith('..'):
            r['timeseries_path'] = os.path.join('..', r['timeseries_path'])
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# Stats
total_qa = sum(len(r['qa_pairs']) for r in all_records)
domains = defaultdict(int)
levels = defaultdict(int)
lengths = defaultdict(int)
difficulties = defaultdict(int)
sources = defaultdict(int)

for r in all_records:
    domains[r['domain']] += 1
    sources[r.get('source', 'unknown')] += 1
    tl = r['metadata'].get('total_length', 0)
    lengths[tl] += 1
    difficulties[r['metadata'].get('difficulty', '?')] += 1
    for qa in r['qa_pairs']:
        levels[qa['level']] += 1

print("\n=== LongTS-Industrial Benchmark v3 (Unified Test Set) ===")
print("  Samples: {}".format(len(all_records)))
print("  QA pairs: {}".format(total_qa))
print("  Domains: {}".format(dict(sorted(domains.items()))))
print("  Sources: {}".format(dict(sorted(sources.items()))))
print("  Levels: {}".format(dict(sorted(levels.items()))))
print("  Lengths: {}".format(dict(sorted(lengths.items()))))
print("  Difficulties: {}".format(dict(sorted(difficulties.items()))))

# Save stats
stats = {
    'total_samples': len(all_records),
    'total_qa': total_qa,
    'domains': dict(sorted(domains.items())),
    'sources': dict(sorted(sources.items())),
    'levels': dict(sorted(levels.items())),
    'lengths': {str(k): v for k, v in sorted(lengths.items())},
    'difficulties': dict(sorted(difficulties.items())),
}
stats_path = os.path.join(out_dir, 'benchmark_stats.json')
with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("\nSaved: {}".format(path))
print("Stats: {}".format(stats_path))
