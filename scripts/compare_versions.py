import json, os

baseline = json.load(open('results/training_histories.json', 'r'))
v1 = json.load(open('results/v1_history.json', 'r')) if os.path.exists('results/v1_history.json') else None
v2 = json.load(open('results/v2_history.json', 'r')) if os.path.exists('results/v2_history.json') else None
v3 = json.load(open('results/v3_history.json', 'r')) if os.path.exists('results/v3_history.json') else None

print("=" * 80)
print("VERSION COMPARISON TABLE")
print("=" * 80)
print("| Version | Hypothesis | Change | Final WER | Improvement | Conclusion |")
print("|---------|-----------|--------|-----------|-------------|-------------|")

baseline_wer = baseline[0]['test_wer'][-1]
print(f"| Baseline | - | - | {baseline_wer:.4f} | - | - |")

if v1:
    v1_wer = v1['test_wer'][-1]
    improvement = baseline_wer - v1_wer
    print(f"| V1 | SpecAugment | Time masking | {v1_wer:.4f} | {improvement:+.4f} | {'Keep' if improvement > 0 else 'Discard'} |")

if v2:
    v2_wer = v2['test_wer'][-1]
    improvement = baseline_wer - v2_wer
    print(f"| V2 | Cosine LR | Cosine schedule | {v2_wer:.4f} | {improvement:+.4f} | {'Keep' if improvement > 0 else 'Discard'} |")

if v3:
    v3_wer = v3['test_wer'][-1]
    improvement = baseline_wer - v3_wer
    print(f"| V3 | Larger model | Whisper Base | {v3_wer:.4f} | {improvement:+.4f} | {'Keep' if improvement > 0 else 'Discard'} |")

print("=" * 80)

# Determine best version
best_version = "Baseline"
best_wer = baseline_wer
if v1 and v1_wer < best_wer:
    best_version = "V1 (SpecAugment)"
    best_wer = v1_wer
if v2 and v2_wer < best_wer:
    best_version = "V2 (Cosine LR)"
    best_wer = v2_wer
if v3 and v3_wer < best_wer:
    best_version = "V3 (Whisper Base)"
    best_wer = v3_wer

# Calculate total improvement
total_improvement = baseline_wer - best_wer
total_improvement_pct = (total_improvement / baseline_wer) * 100

print("CONCLUSION:")
print(f"1. Best version: {best_version} (WER: {best_wer:.4f})")
print(f"2. Total improvement: {total_improvement:.4f} ({total_improvement_pct:.1f}%) over baseline")
print(f"3. Feasibility: Yes - Minimal computational overhead, easy to implement")
print(f"4. Limitations: Small dataset (500 samples), limited epochs (2), single seed (42), batch_size=1")
print(f"5. Future work: Train on full dataset, increase epochs, test multiple seeds")
