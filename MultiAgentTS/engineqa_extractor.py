"""
从 EngineMT-QA 数据集抽取 Stage 1 (Understanding) 和 Stage 4 (Decision-Making)
的开放式 QA 子集，转换为统一的 Benchmark 格式
"""
import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_engineqa_open_qa(input_path: str, output_path: str,
                              max_samples: int = 800, seed: int = 42):
    """
    从 EngineMT-QA 的 test_qa.jsonl 中抽取 Stage 1 和 Stage 4 的开放式 QA

    Stage 1 = Understanding (open-ended)
    Stage 4 = Decision-Making (open-ended)
    """
    rng = random.Random(seed)

    # Load all samples
    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from {input_path}")

    # Filter for Stage 1 and Stage 4 (open-ended)
    open_qa_samples = []
    for sample in samples:
        conversations = sample.get("conversations", [])
        if len(conversations) < 2:
            continue

        # Check if this is a Stage 1 or Stage 4 question
        human_msg = conversations[0]
        stage = human_msg.get("stage", "")
        attribute = human_msg.get("attribute", "")

        if stage in ("1", "4") and attribute == "open":
            # Extract question and answer
            question = human_msg.get("value", "")
            answer = conversations[1].get("value", "") if len(conversations) > 1 else ""

            if question and answer:
                # Map to our format
                level = "L1" if stage == "1" else "L4"

                record = {
                    "id": f"engineqa_{sample.get('id', ['unknown'])[0]}",
                    "domain": "aero_engine",
                    "source": "enginemt_qa",
                    "scenario_id": f"engineqa_stage{stage}",
                    "metadata": {
                        "total_length": 600,  # EngineMT-QA fixed length
                        "n_channels": 33,
                        "sampling_rate": "variable",
                        "difficulty": "medium",
                        "needle_ratio": 0.0,
                        "original_stage": int(stage),
                    },
                    "timeseries_path": f"enginemt_qa/{sample.get('name', ['unknown.csv'])[0]}",
                    "ground_truth_events": [],
                    "qa_pairs": [
                        {
                            "level": level,
                            "question": question,
                            "answer": answer,
                            "reasoning_chain": [],
                            "required_interval": None,
                        }
                    ],
                }
                open_qa_samples.append(record)

    print(f"Found {len(open_qa_samples)} open-ended QA (Stage 1 + 4)")

    # Sample if needed
    if len(open_qa_samples) > max_samples:
        open_qa_samples = rng.sample(open_qa_samples, max_samples)
        print(f"Sampled {max_samples} samples")

    # Count by stage
    stage_counts = {}
    for s in open_qa_samples:
        stage = s["metadata"].get("original_stage", 0)
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    print(f"Stage distribution: {stage_counts}")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in open_qa_samples:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    total_qa = sum(len(r["qa_pairs"]) for r in open_qa_samples)
    print(f"Saved {total_qa} QA pairs to {output_path}")

    return open_qa_samples


if __name__ == "__main__":
    input_path = "dataset/EngineMT-QA/test_qa.jsonl"
    output_path = "Data/benchmark/engineqa_subset.jsonl"

    if not os.path.exists(input_path):
        print(f"EngineMT-QA test file not found at {input_path}")
        print("Skipping EngineMT-QA extraction.")
    else:
        extract_engineqa_open_qa(input_path, output_path)
