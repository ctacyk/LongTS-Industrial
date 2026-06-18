"""
阶段 D: HES 三元组验证
======================
取 10 个样本，每个样本构造 golden/mediocre/wrong 三个答案，
验证 HES 能否正确区分答案质量：golden > mediocre > wrong。
"""
import json
import sys
import random
import re
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from benchmark_eval import (
    batch_get_embeddings, cosine_similarity, judge_single,
    direction_to_score, quality_to_score, compute_hes
)

QA_PATH = "Data/benchmark_tb/synthetic_qa_llm.jsonl"
random.seed(42)

# Load QA records (fault samples only)
with open(QA_PATH, 'r', encoding='utf-8') as f:
    all_records = [json.loads(l) for l in f if l.strip()]

fault_records = [r for r in all_records if r.get("ground_truth_events")]
selected = random.sample(fault_records, 10)


def make_mediocre(answer):
    """Mediocre: remove specific numbers, keep direction."""
    text = re.sub(r'\d+\.\d+', '某个数值', answer)
    text = re.sub(r'第\d+步', '某一时刻', text)
    text = text[:len(text)//2] + "...（分析到此为止）"
    return text


def make_wrong(answer, gt_events):
    """Wrong: flip direction and wrong fault type."""
    replacements = {
        "上升": "下降", "升高": "降低", "增大": "减小", "增加": "减少",
        "下降": "上升", "降低": "升高", "减小": "增大", "减少": "增加",
        "异常": "正常运行", "故障": "正常状态",
    }
    text = answer
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = "经过仔细分析，该设备处于正常运行状态，未发现任何异常。" + text[:200]
    return text


print(f"Selected {len(selected)} samples for HES triplet validation\n")

all_texts = []
triplets = []

for r in selected:
    l1_qa = next((qa for qa in r['qa_pairs'] if qa['level'] == 'L1'), None)
    if not l1_qa:
        continue

    question = l1_qa['question']
    golden = l1_qa['answer']
    mediocre = make_mediocre(golden)
    wrong = make_wrong(golden, r.get('ground_truth_events', []))

    triplets.append({
        'id': r['id'],
        'question': question,
        'golden': golden,
        'mediocre': mediocre,
        'wrong': wrong,
    })
    all_texts.extend([golden, mediocre, wrong, golden])

print(f"Getting embeddings for {len(all_texts)} texts...")
embeddings = batch_get_embeddings(all_texts)

results = []
idx = 0
for t in triplets:
    gt_emb = embeddings[idx + 3]  # golden as GT
    scores = {}

    for label, answer, emb_idx in [
        ('golden', t['golden'], idx),
        ('mediocre', t['mediocre'], idx + 1),
        ('wrong', t['wrong'], idx + 2)
    ]:
        pred_emb = embeddings[emb_idx]
        ss = cosine_similarity(pred_emb, gt_emb) if pred_emb and gt_emb else 0.0

        jr = judge_single(t['question'], t['golden'], answer)
        da = direction_to_score(jr.get('direction', 'partial'))
        aq = quality_to_score(jr.get('quality', 3))
        hes = compute_hes(da, ss, aq)

        scores[label] = {'da': da, 'ss': round(ss, 4), 'aq': aq, 'hes': round(hes, 4)}

    idx += 4

    ordered = scores['golden']['hes'] > scores['mediocre']['hes'] > scores['wrong']['hes']
    results.append({
        'id': t['id'],
        'scores': scores,
        'correct_order': ordered,
    })

    print(f"  {t['id'][:50]}...")
    for label in ['golden', 'mediocre', 'wrong']:
        s = scores[label]
        print(f"    {label:10s}: DA={s['da']:.1f}  SS={s['ss']:.3f}  AQ={s['aq']:.2f}  HES={s['hes']:.4f}")
    print(f"    Order correct: {ordered}")
    print()

correct = sum(1 for r in results if r['correct_order'])
print(f"\n{'='*50}")
print(f"HES TRIPLET VALIDATION: {correct}/{len(results)} correct ordering")
print(f"{'='*50}")

avg_golden = sum(r['scores']['golden']['hes'] for r in results) / len(results)
avg_mediocre = sum(r['scores']['mediocre']['hes'] for r in results) / len(results)
avg_wrong = sum(r['scores']['wrong']['hes'] for r in results) / len(results)
print(f"Avg HES: golden={avg_golden:.4f}, mediocre={avg_mediocre:.4f}, wrong={avg_wrong:.4f}")

report = {
    'total': len(results),
    'correct_order': correct,
    'avg_golden': round(avg_golden, 4),
    'avg_mediocre': round(avg_mediocre, 4),
    'avg_wrong': round(avg_wrong, 4),
    'details': results,
}
with open('results/hes_triplet_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"Report saved to results/hes_triplet_report.json")
