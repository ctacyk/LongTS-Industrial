"""Deduplicate LLM QA JSONL and rebuild benchmark"""
import json
import sys
sys.path.insert(0, '.')

# 1. Deduplicate
input_path = "Data/benchmark/synthetic_qa_llm.jsonl"
dedup_path = "Data/benchmark/synthetic_qa_llm_dedup.jsonl"

seen_ids = {}
duplicates = 0
errors = 0

with open(input_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            rid = record.get("id", "")
            if rid in seen_ids:
                duplicates += 1
            else:
                seen_ids[rid] = record
        except json.JSONDecodeError:
            errors += 1

print(f"Input: {line_num} lines")
print(f"Unique records: {len(seen_ids)}")
print(f"Duplicates removed: {duplicates}")
print(f"Parse errors: {errors}")

# Save deduped
with open(dedup_path, 'w', encoding='utf-8') as f:
    for record in seen_ids.values():
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
print(f"Saved deduped to: {dedup_path}")

# Count QA stats
total_qa = sum(len(r.get("qa_pairs", [])) for r in seen_ids.values())
by_level = {}
for r in seen_ids.values():
    for qa in r.get("qa_pairs", []):
        l = qa.get("level", "?")
        by_level[l] = by_level.get(l, 0) + 1

print(f"\nTotal QA pairs: {total_qa}")
print(f"By level: {by_level}")

# 2. Rebuild benchmark
print("\n=== Rebuilding Benchmark ===")
from MultiAgentTS.benchmark_builder import build_benchmark

build_benchmark(
    synthetic_qa_path=dedup_path,
    engineqa_path="Data/benchmark/engineqa_subset.jsonl",
    output_dir="Data/benchmark/LongTS_Industrial",
)
