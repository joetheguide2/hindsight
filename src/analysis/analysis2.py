import pandas as pd
import re
from collections import Counter

df = pd.read_csv('results/studentF_filtered/evaluation_results.csv')

print("="*80)
print("DIAGNOSTIC TASK EVALUATION REPORT")
print("="*80)
print(f"\nTotal cases: {len(df)}")
print(f"Number of diseases: {df['base_true_disease'].nunique()}")
print(f"Cases per disease: {len(df) // df['base_true_disease'].nunique()}")

# ============================================================
# 1. STRICT ACCURACY (from 'correct' column)
# ============================================================
print("\n" + "="*80)
print("1. STRICT ACCURACY (from 'correct' column)")
print("="*80)
print("   Definition: Model's answer matches true disease or any synonym")
print("   (as evaluated by the original 'correct' flag)")

base_correct = df['base_correct'].sum()
ft_correct = df['ft_correct'].sum()
base_acc = base_correct / len(df) * 100
ft_acc = ft_correct / len(df) * 100

print(f"\n   Base model: {base_correct}/{len(df)} = {base_acc:.1f}%")
print(f"   Fine-tuned: {ft_correct}/{len(df)} = {ft_acc:.1f}%")
print(f"   Delta:      {ft_acc - base_acc:+.1f}%")

# ============================================================
# 2. RELAXED ACCURACY (answer anywhere in response)
# ============================================================
print("\n" + "="*80)
print("2. RELAXED ACCURACY (answer anywhere in response)")
print("="*80)
print("   Definition: True disease name or any synonym appears ANYWHERE")
print("   in the full response text (even in reasoning or differentials)")

def check_answer_in_response(response, true_disease, synonyms_str):
    if pd.isna(response) or pd.isna(synonyms_str):
        return False
    response_lower = response.lower()
    if true_disease.lower() in response_lower:
        return True
    try:
        synonyms_str = str(synonyms_str).strip("[]")
        synonyms = [s.strip().strip("'\"") for s in synonyms_str.split(';')]
        for syn in synonyms:
            syn = syn.strip()
            if syn and syn.lower() in response_lower:
                return True
    except:
        pass
    return False

df['base_relaxed_correct'] = df.apply(
    lambda r: check_answer_in_response(r['base_response'], r['base_true_disease'], r['base_synonyms']), axis=1
)
df['ft_relaxed_correct'] = df.apply(
    lambda r: check_answer_in_response(r['ft_response'], r['ft_true_disease'], r['ft_synonyms']), axis=1
)

base_relaxed = df['base_relaxed_correct'].sum()
ft_relaxed = df['ft_relaxed_correct'].sum()
base_relaxed_acc = base_relaxed / len(df) * 100
ft_relaxed_acc = ft_relaxed / len(df) * 100

print(f"\n   Base model: {base_relaxed}/{len(df)} = {base_relaxed_acc:.1f}%")
print(f"   Fine-tuned: {ft_relaxed}/{len(df)} = {ft_relaxed_acc:.1f}%")
print(f"   Delta:      {ft_relaxed_acc - base_relaxed_acc:+.1f}%")

# ============================================================
# 3. TAG ACCURACY - fine-tuned model ONLY
# ============================================================
print("\n" + "="*80)
print("3. TAG ACCURACY (fine-tuned model only)")
print("="*80)
print("   Definition: Extract text between <diagnosis> and </diagnosis>")
print("   Remove brackets, quotes, extra whitespace, then check if the")
print("   true disease name or any synonym matches EXACTLY against the")
print("   cleaned tag content.")
print("   (Base model never uses tags, so this only applies to fine-tuned)")

def extract_and_check_tag(response, true_disease, synonyms_str):
    """Extract <diagnosis> content and check exact match."""
    if pd.isna(response):
        return False
    match = re.search(r'<diagnosis>(.*?)</diagnosis>', response, re.DOTALL)
    if not match:
        return False
    
    tag_content = match.group(1).strip()
    
    # Clean: remove brackets, quotes, commas, extra whitespace
    clean_tag = re.sub(r'[\[\]\{\}"\']', '', tag_content).strip()
    # Collapse whitespace
    clean_tag = re.sub(r'\s+', ' ', clean_tag).strip()
    clean_tag_lower = clean_tag.lower()
    
    # Clean true disease
    clean_true = re.sub(r'[\[\]\{\}"\']', '', true_disease).strip().lower()
    # Also try without the clean process - just lowercase + whitespace normalize
    clean_true_simple = re.sub(r'\s+', ' ', true_disease.strip()).lower()
    
    # Check exact match (case-insensitive)
    if clean_tag_lower == clean_true:
        return True
    if clean_tag_lower == clean_true_simple:
        return True
    
    # Also try: is the tag content contained within the disease or vice versa?
    # Some tags might be abbreviations
    if clean_true_simple in clean_tag_lower or clean_tag_lower in clean_true_simple:
        return True
    
    # Check synonyms
    try:
        synonyms_str = str(synonyms_str).strip("[]")
        synonyms = [s.strip().strip("'\"") for s in synonyms_str.split(';')]
        for syn in synonyms:
            syn_clean = re.sub(r'[\[\]\{\}"\']', '', syn).strip().lower()
            syn_clean = re.sub(r'\s+', ' ', syn_clean).strip()
            if clean_tag_lower == syn_clean:
                return True
            if syn_clean in clean_tag_lower or clean_tag_lower in syn_clean:
                return True
    except:
        pass
    
    return False

df['ft_tag_correct'] = df.apply(
    lambda r: extract_and_check_tag(r['ft_response'], r['ft_true_disease'], r['ft_synonyms']), axis=1
)

# Also check how many have tags at all
ft_has_any_tag = df['ft_response'].str.contains(r'<diagnosis>', regex=True).sum()
ft_tag_content = df['ft_response'].apply(lambda x: bool(re.search(r'<diagnosis>(.*?)</diagnosis>', str(x), re.DOTALL)))
ft_with_tag = ft_tag_content.sum()
ft_tag_correct = df['ft_tag_correct'].sum()
ft_tag_acc = ft_tag_correct / len(df) * 100
ft_tag_acc_of_tagged = ft_tag_correct / ft_with_tag * 100 if ft_with_tag > 0 else 0

print(f"\n   Responses with <diagnosis> tags: {ft_with_tag}/{len(df)} = {ft_with_tag/len(df)*100:.1f}%")
print(f"   Tag accuracy (of all responses): {ft_tag_correct}/{len(df)} = {ft_tag_acc:.1f}%")
print(f"   Tag accuracy (of tagged responses): {ft_tag_correct}/{ft_with_tag} = {ft_tag_acc_of_tagged:.1f}%")

# ============================================================
# 4. TAG FORMAT ANALYSIS
# ============================================================
print("\n" + "="*80)
print("4. TAG FORMAT ANALYSIS (fine-tuned model)")
print("="*80)

def get_tag_content(response):
    match = re.search(r'<diagnosis>(.*?)</diagnosis>', str(response), re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

df['ft_tag_raw'] = df['ft_response'].apply(get_tag_content)

# Show tag content examples
tagged_df = df[df['ft_tag_raw'].str.len() > 0]
print(f"\n   Tagged responses: {len(tagged_df)}/{len(df)}")
print(f"   Mean tag length: {tagged_df['ft_tag_raw'].str.len().mean():.0f} chars")
print(f"   Median tag length: {tagged_df['ft_tag_raw'].str.len().median():.0f} chars")
print(f"   Min tag length: {tagged_df['ft_tag_raw'].str.len().min()} chars")
print(f"   Max tag length: {tagged_df['ft_tag_raw'].str.len().max()} chars")

# Show some tag content examples
print("\n   Sample tag contents (first 10):")
for i, row in tagged_df.head(10).iterrows():
    true_d = row['ft_true_disease']
    tag = row['ft_tag_raw']
    is_correct = row['ft_tag_correct']
    status = "CORRECT" if is_correct else "WRONG"
    print(f"   [{status}] True: {true_d[:40]:<40} | Tag: {tag[:60]}")

# ============================================================
# 5. BASELINE METRICS
# ============================================================
print("\n" + "="*80)
print("5. BASELINE METRICS")
print("="*80)
print("   (Base model - no tags, no reasoning)")

base_has_tag = df['base_has_diagnose'].sum()
base_has_think = df['base_has_think'].sum()

print(f"\n   Base model uses <diagnosis> tags: {base_has_tag}/{len(df)} = {base_has_tag/len(df)*100:.1f}%")
print(f"   Base model uses thinking/reasoning: {base_has_think}/{len(df)} = {base_has_think/len(df)*100:.1f}%")

# ============================================================
# 6. FT MODEL BEHAVIOR
# ============================================================
print("\n" + "="*80)
print("6. FINE-TUNED MODEL BEHAVIOR")
print("="*80)

ft_has_tag = df['ft_has_diagnose'].sum()
ft_has_think = df['ft_has_think'].sum()

print(f"\n   Uses <diagnosis> tags: {ft_has_tag}/{len(df)} = {ft_has_tag/len(df)*100:.1f}%")
print(f"   Uses thinking/reasoning: {ft_has_think}/{len(df)} = {ft_has_think/len(df)*100:.1f}%")

# Accuracy breakdown by behavior
print("\n   --- FT Accuracy by tag usage ---")
with_tags = df[df['ft_has_diagnose'] == True]
without_tags = df[df['ft_has_diagnose'] == False]
print(f"   With tags:  {with_tags['ft_correct'].sum()}/{len(with_tags)} = {with_tags['ft_correct'].sum()/max(len(with_tags),1)*100:.1f}%")
print(f"   Without tags: {without_tags['ft_correct'].sum()}/{len(without_tags)} = {without_tags['ft_correct'].sum()/max(len(without_tags),1)*100:.1f}%")

# ============================================================
# 7. COMBINED SUMMARY
# ============================================================
print("\n" + "="*80)
print("7. COMBINED METRICS SUMMARY")
print("="*80)
print(f"""
   Metric                         | Base (no tags) | Fine-tuned | Delta
   -------------------------------+----------------+------------+--------
   Strict Accuracy                | {base_acc:>15.1f}% | {ft_acc:>11.1f}% | {ft_acc-base_acc:>+9.1f}%
   Relaxed Accuracy               | {base_relaxed_acc:>15.1f}% | {ft_relaxed_acc:>11.1f}% | {ft_relaxed_acc-base_relaxed_acc:>+9.1f}%
   Tag Accuracy (all responses)   |     N/A        | {ft_tag_acc:>11.1f}% |       N/A
   Tag Accuracy (tagged only)     |     N/A        | {ft_tag_acc_of_tagged:>10.1f}% |       N/A
   Tag Usage Rate                 |     N/A        | {ft_has_tag/len(df)*100:>10.1f}% |       N/A
   Thinking Usage Rate            | {base_has_think/len(df)*100:>15.1f}% | {ft_has_think/len(df)*100:>10.1f}% | {ft_has_think/len(df)*100-base_has_think/len(df)*100:>+9.1f}%
""")

# ============================================================
# 8. MODEL AGREEMENT
# ============================================================
print("\n" + "="*80)
print("8. MODEL AGREEMENT")
print("="*80)

both_correct = (df['base_correct'] == True) & (df['ft_correct'] == True)
both_wrong = (df['base_correct'] == False) & (df['ft_correct'] == False)
base_only = (df['base_correct'] == True) & (df['ft_correct'] == False)
ft_only = (df['base_correct'] == False) & (df['ft_correct'] == True)

print(f"\n   Both correct:    {both_correct.sum()}/{len(df)} = {both_correct.sum()/len(df)*100:.1f}%")
print(f"   Both wrong:      {both_wrong.sum()}/{len(df)} = {both_wrong.sum()/len(df)*100:.1f}%")
print(f"   Base only:       {base_only.sum()}/{len(df)} = {base_only.sum()/len(df)*100:.1f}%")
print(f"   FT only:         {ft_only.sum()}/{len(df)} = {ft_only.sum()/len(df)*100:.1f}%")
print(f"\n   Overall agreement: {(both_correct.sum() + both_wrong.sum())/len(df)*100:.1f}%")

# ============================================================
# 9. TAG ACCURACY DETAILS - what's wrong with wrong tags
# ============================================================
print("\n" + "="*80)
print("9. TAG ACCURACY DEEP DIVE - Wrong tags analysis")
print("="*80)

wrong_tags = tagged_df[tagged_df['ft_tag_correct'] == False]
print(f"\n   Total wrong tags: {len(wrong_tags)}/{ft_with_tag} = {len(wrong_tags)/ft_with_tag*100:.1f}%")

# Show some examples of wrong tags
print("\n   Sample wrong tag predictions:")
for i, row in wrong_tags.head(15).iterrows():
    true_d = row['ft_true_disease']
    tag = row['ft_tag_raw']
    print(f"   True: {true_d:<50} | Tag: {tag[:80]}")

# ============================================================
# 10. PER-DISEASE TAG ACCURACY
# ============================================================
print("\n" + "="*80)
print("10. PER-DISEASE TAG ACCURACY (fine-tuned, tagged only)")
print("="*80)

diseases = df['ft_true_disease'].unique()
print(f"\n   {'Disease':<55} {'Tagged':>8} {'Correct':>8} {'Tag Acc':>8}")
print(f"   {'-'*55} {'-'*8} {'-'*8} {'-'*8}")

for disease in sorted(diseases):
    mask = df['ft_true_disease'] == disease
    tagged_mask = mask & ft_tag_content
    correct_mask = mask & df['ft_tag_correct']
    
    n_tagged = tagged_mask.sum()
    n_correct = correct_mask.sum()
    tag_acc = n_correct / n_tagged * 100 if n_tagged > 0 else 0
    
    if n_tagged > 0:
        print(f"   {disease:<55} {n_tagged:>8} {n_correct:>8} {tag_acc:>7.1f}%")

print("\n" + "="*80)
print("END OF REPORT")
print("="*80)