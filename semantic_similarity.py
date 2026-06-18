"""
Calculate semantic similarity between predictions and ground truth using Qwen embedding
"""
import os
import json
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

# 阿里云 DashScope API（通过环境变量 DASHSCOPE_API_KEY 配置）
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def get_embedding(text: str, model: str = "text-embedding-v3") -> list:
    """Get embedding for a single text"""
    try:
        response = client.embeddings.create(
            model=model,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def cosine_similarity(v1: list, v2: list) -> float:
    """Calculate cosine similarity between two vectors"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def batch_get_embeddings(texts: list, model: str = "text-embedding-v3", batch_size: int = 10) -> list:
    """Get embeddings for multiple texts in batches"""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
        batch = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                model=model,
                input=batch,
                encoding_format="float"
            )
            embeddings.extend([d.embedding for d in response.data])
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Fallback to individual requests
            for text in batch:
                emb = get_embedding(text, model)
                embeddings.append(emb if emb else [0] * 1024)
                time.sleep(0.1)
    return embeddings

def main():
    # Load results
    results_file = "results/htsr_vl_v2_20260203_044838.json"
    print(f"Loading results from {results_file}...")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    print(f"Total samples: {len(results)}")
    
    # Filter for open-ended questions (Stage 1 and 4) - these benefit most from semantic similarity
    # For Stage 2/3 (choice questions), exact match is more appropriate
    
    stage_data = {1: [], 2: [], 3: [], 4: []}
    for r in results:
        stage = r['stage']
        stage_data[stage].append(r)
    
    print(f"\nStage distribution:")
    for stage, samples in stage_data.items():
        print(f"  Stage {stage}: {len(samples)} samples")
    
    # Calculate semantic similarity for open-ended questions (Stage 1 and 4)
    print("\n" + "="*70)
    print("SEMANTIC SIMILARITY ANALYSIS")
    print("="*70)
    
    all_similarities = {}
    
    for stage in [1, 4]:  # Open-ended questions
        samples = stage_data[stage]
        if not samples:
            continue
            
        print(f"\n--- Stage {stage} ({len(samples)} samples) ---")
        
        # Prepare texts
        predictions = [s['prediction'] for s in samples]
        ground_truths = [s['ground_truth'] for s in samples]
        
        # Get embeddings
        print("Getting prediction embeddings...")
        pred_embeddings = batch_get_embeddings(predictions)
        
        print("Getting ground truth embeddings...")
        gt_embeddings = batch_get_embeddings(ground_truths)
        
        # Calculate similarities
        similarities = []
        for i, (pred_emb, gt_emb) in enumerate(zip(pred_embeddings, gt_embeddings)):
            if pred_emb and gt_emb:
                sim = cosine_similarity(pred_emb, gt_emb)
                similarities.append(sim)
                samples[i]['semantic_similarity'] = sim
            else:
                similarities.append(0)
                samples[i]['semantic_similarity'] = 0
        
        # Statistics
        similarities = np.array(similarities)
        print(f"\nSemantic Similarity Statistics:")
        print(f"  Mean:   {similarities.mean():.4f}")
        print(f"  Std:    {similarities.std():.4f}")
        print(f"  Min:    {similarities.min():.4f}")
        print(f"  Max:    {similarities.max():.4f}")
        print(f"  Median: {np.median(similarities):.4f}")
        
        # Distribution
        print(f"\n  Distribution:")
        print(f"    >= 0.9: {(similarities >= 0.9).sum()} ({(similarities >= 0.9).mean()*100:.1f}%)")
        print(f"    >= 0.8: {(similarities >= 0.8).sum()} ({(similarities >= 0.8).mean()*100:.1f}%)")
        print(f"    >= 0.7: {(similarities >= 0.7).sum()} ({(similarities >= 0.7).mean()*100:.1f}%)")
        print(f"    >= 0.6: {(similarities >= 0.6).sum()} ({(similarities >= 0.6).mean()*100:.1f}%)")
        print(f"    < 0.6:  {(similarities < 0.6).sum()} ({(similarities < 0.6).mean()*100:.1f}%)")
        
        all_similarities[f'stage{stage}'] = {
            'mean': float(similarities.mean()),
            'std': float(similarities.std()),
            'min': float(similarities.min()),
            'max': float(similarities.max()),
            'median': float(np.median(similarities)),
            'count': len(similarities)
        }
        
        # Show some examples
        print(f"\n  Top 3 highest similarity samples:")
        sorted_indices = np.argsort(similarities)[::-1]
        for idx in sorted_indices[:3]:
            s = samples[idx]
            print(f"    [{s['sample_id']}] Similarity: {similarities[idx]:.4f}")
            print(f"      GT:   {s['ground_truth'][:100]}...")
            print(f"      Pred: {s['prediction'][:100]}...")
        
        print(f"\n  Top 3 lowest similarity samples:")
        for idx in sorted_indices[-3:]:
            s = samples[idx]
            print(f"    [{s['sample_id']}] Similarity: {similarities[idx]:.4f}")
            print(f"      GT:   {s['ground_truth'][:100]}...")
            print(f"      Pred: {s['prediction'][:100]}...")
    
    # For Stage 2/3, calculate exact match accuracy
    print("\n" + "="*70)
    print("CHOICE QUESTION ACCURACY (Stage 2/3)")
    print("="*70)
    
    for stage in [2, 3]:
        samples = stage_data[stage]
        if not samples:
            continue
        
        correct = sum(1 for s in samples if s['prediction'].strip().lower() == s['ground_truth'].strip().lower())
        accuracy = correct / len(samples)
        
        print(f"\nStage {stage}: {correct}/{len(samples)} = {accuracy*100:.2f}% accuracy")
        all_similarities[f'stage{stage}'] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(samples)
        }
    
    # Save results
    output = {
        'source_file': results_file,
        'metrics': all_similarities,
        'total_samples': len(results)
    }
    
    output_file = 'results/semantic_similarity_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nResults saved to {output_file}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n  Stage 1 (Understanding) - Semantic Similarity: {:.2f}%".format(
        all_similarities.get('stage1', {}).get('mean', 0) * 100
    ))
    print("  Stage 2 (Perception) - Accuracy: {:.2f}%".format(
        all_similarities.get('stage2', {}).get('accuracy', 0) * 100
    ))
    print("  Stage 3 (Reasoning) - Accuracy: {:.2f}%".format(
        all_similarities.get('stage3', {}).get('accuracy', 0) * 100
    ))
    print("  Stage 4 (Decision) - Semantic Similarity: {:.2f}%".format(
        all_similarities.get('stage4', {}).get('mean', 0) * 100
    ))

if __name__ == '__main__':
    main()
