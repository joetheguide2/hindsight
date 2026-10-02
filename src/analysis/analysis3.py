import pandas as pd
import re

df = pd.read_csv('results/studentF_filtered/evaluation_results.csv')

print("="*80)
print("DIAGNOSTIC TASK EVALUATION REPORT")
print("="*80)
print(f"\nTotal cases: {len(df)}")
print(f"Number of diseases: {df['base_true_disease'].nunique()}")
print(f"Cases per disease: {len(df) // df['base_true_disease'].nunique()}")

# ============================================================
# 1. RELAXED ACCURACY (from 'correct' column)
# ============================================================
print("\n" + "="*80)
print("1. RELAXED ACCURACY (from 'correct' column)")
print("="*80)
print("   Definition: True disease or synonym appears ANYWHERE in response")
print("   (from the 'correct' column)")

base_relaxed = df['base_correct'].sum()
ft_relaxed = df['ft_correct'].sum()
base_relaxed_acc = base_relaxed / len(df) * 100
ft_relaxed_acc = ft_relaxed / len(df) * 100

print(f"\n   Base model:  {base_relaxed}/{len(df)} = {base_relaxed_acc:.1f}%")
print(f"   Fine-tuned:  {ft_relaxed}/{len(df)} = {ft_relaxed_acc:.1f}%")
print(f"   Delta:       {ft_relaxed_acc - base_relaxed_acc:+.1f}%")

# ============================================================
# 2. STRICT ACCURACY (exact match inside <diagnosis> tags)
# ============================================================
print("\n" + "="*80)
print("2. STRICT ACCURACY (exact match inside <diagnosis> tags)")
print("="*80)
print("   Definition: Extract content between <diagnosis> and </diagnosis>,")
print("   clean it, and check if it EXACTLY matches the true disease")
print("   or any of its synonyms.")

def extract_and_check_strict(response, true_disease, synonyms_str):
    """Extract <diagnosis> content and check EXACT match."""
    if pd.isna(response):
        return False, "", False
    match = re.search(r'<diagnosis>(.*?)</diagnosis>', str(response), re.DOTALL)
    if not match:
        return False, "", False
    
    tag_content = match.group(1).strip()
    
    # Clean: remove brackets, quotes, commas, extra whitespace
    clean_tag = re.sub(r'[\[\]\{\}"\']', '', tag_content).strip()
    clean_tag = re.sub(r'\s+', ' ', clean_tag).strip()
    clean_tag_lower = clean_tag.lower()
    
    # Clean true disease for comparison
    clean_true = re.sub(r'\s+', ' ', true_disease.strip()).lower().strip()
    
    # Check exact match against true disease
    if clean_tag_lower == clean_true:
        return True, clean_tag, True
    
    # Check exact match against synonyms
    try:
        synonyms_str = str(synonyms_str).strip("[]")
        synonyms = [s.strip().strip("'\"") for s in synonyms_str.split(';')]
        for syn in synonyms:
            syn_clean = re.sub(r'\s+', ' ', syn.strip()).lower().strip()
            if clean_tag_lower == syn_clean:
                return True, clean_tag, True
    except:
        pass
    
    return False, clean_tag, False

df['base_strict_correct'] = False
df['base_tag_content'] = ''
base_tagged_count = 0
base_tagged_correct = 0
ft_tagged_count = 0
ft_tagged_correct = 0

for i, row in df.iterrows():
    # Base model
    correct, tag, _ = extract_and_check_strict(row['base_response'], row['base_true_disease'], row['base_synonyms'])
    df.loc[i, 'base_strict_correct'] = correct
    df.loc[i, 'base_tag_content'] = tag
    if tag:
        base_tagged_count += 1
        if correct:
            base_tagged_correct += 1
    
    # Fine-tuned model
    correct, tag, _ = extract_and_check_strict(row['ft_response'], row['ft_true_disease'], row['ft_synonyms'])
    df.loc[i, 'ft_strict_correct'] = correct
    df.loc[i, 'ft_tag_content'] = tag
    if tag:
        ft_tagged_count += 1
        if correct:
            ft_tagged_correct += 1

base_strict = df['base_strict_correct'].sum()
ft_strict = df['ft_strict_correct'].sum()
base_strict_acc = base_strict / len(df) * 100
ft_strict_acc = ft_strict / len(df) * 100

# Base model tag usage
base_has_diagnosis_tag = df['base_tag_content'].str.len() > 0
base_tagged_count = base_has_diagnosis_tag.sum()
base_tagged_correct = df.loc[base_has_diagnosis_tag, 'base_strict_correct'].sum()
base_tagged_acc = base_tagged_correct / base_tagged_count * 100 if base_tagged_count > 0 else 0

print(f"\n   --- Base model ---")
print(f"   Responses with <diagnosis> tags: {base_tagged_count}/{len(df)} = {base_tagged_count/len(df)*100:.1f}%")
if base_tagged_count > 0:
    print(f"   Strict accuracy (tagged only): {base_tagged_correct}/{base_tagged_count} = {base_tagged_acc:.1f}%")
print(f"   Strict accuracy (all responses): {base_strict}/{len(df)} = {base_strict_acc:.1f}%")

print(f"\n   --- Fine-tuned model ---")
print(f"   Responses with <diagnosis> tags: {ft_tagged_count}/{len(df)} = {ft_tagged_count/len(df)*100:.1f}%")
print(f"   Strict accuracy (tagged only): {ft_tagged_correct}/{ft_tagged_count} = {ft_tagged_correct/ft_tagged_count*100:.1f}%")
print(f"   Strict accuracy (all responses): {ft_strict}/{len(df)} = {ft_strict_acc:.1f}%")
print(f"   Delta (all responses): {ft_strict_acc - base_strict_acc:+.1f}%")

# ============================================================
# 3. TAG FORMAT ANALYSIS - fine-tuned
# ============================================================
print("\n" + "="*80)
print("3. TAG FORMAT ANALYSIS (fine-tuned model)")
print("="*80)

tagged_df = df[df['ft_tag_content'].str.len() > 0]
print(f"\n   Tagged responses: {len(tagged_df)}/{len(df)}")
if len(tagged_df) > 0:
    print(f"   Mean tag length: {tagged_df['ft_tag_content'].str.len().mean():.0f} chars")
    print(f"   Median tag length: {tagged_df['ft_tag_content'].str.len().median():.0f} chars")
    print(f"   Min tag length: {tagged_df['ft_tag_content'].str.len().min()} chars")
    print(f"   Max tag length: {tagged_df['ft_tag_content'].str.len().max()} chars")

    print("\n   Sample tag contents (first 10):")
    for i, row in tagged_df.head(10).iterrows():
        true_d = row['ft_true_disease']
        tag = row['ft_tag_content']
        is_correct = row['ft_strict_correct']
        status = "CORRECT" if is_correct else "WRONG"
        print(f"   [{status}] True: {true_d[:45]:<45} | Tag: {tag[:60]}")

# Show wrong tags
print("\n   Sample WRONG tag predictions:")
wrong = tagged_df[tagged_df['ft_strict_correct'] == False]
for i, row in wrong.head(10).iterrows():
    true_d = row['ft_true_disease']
    tag = row['ft_tag_content']
    print(f"   True: {true_d:<55} | Tag: {tag[:70]}")

# ============================================================
# 4. BASELINE METRICS
# ============================================================
print("\n" + "="*80)
print("4. BASELINE METRICS")
print("="*80)

base_has_think = df['base_has_think'].sum()
ft_has_think = df['ft_has_think'].sum()

print(f"\n   Base model uses thinking/reasoning: {base_has_think}/{len(df)} = {base_has_think/len(df)*100:.1f}%")
print(f"   Fine-tuned uses thinking/reasoning: {ft_has_think}/{len(df)} = {ft_has_think/len(df)*100:.1f}%")

# ============================================================
# 5. COMBINED SUMMARY
# ============================================================
print("\n" + "="*80)
print("5. COMBINED METRICS SUMMARY")
print("="*80)
print(f"""
   Metric                               | Base (no tags) | Fine-tuned | Delta
   -------------------------------------+----------------+------------+----------
   Relaxed Accuracy (correct column)    | {base_relaxed_acc:>15.1f}% | {ft_relaxed_acc:>11.1f}% | {ft_relaxed_acc-base_relaxed_acc:>+9.1f}%
   Strict Accuracy (all responses)      | {base_strict_acc:>15.1f}% | {ft_strict_acc:>11.1f}% | {ft_strict_acc-base_strict_acc:>+9.1f}%
   Strict Accuracy (tagged only)        |     N/A        | {ft_tagged_correct/ft_tagged_count*100 if ft_tagged_count > 0 else 0:>10.1f}% |       N/A
   Tag Usage Rate                       | {base_tagged_count/len(df)*100:>15.1f}% | {ft_tagged_count/len(df)*100:>10.1f}% | {ft_tagged_count/len(df)*100-base_tagged_count/len(df)*100:>+9.1f}%
   Thinking Usage Rate                  | {base_has_think/len(df)*100:>15.1f}% | {ft_has_think/len(df)*100:>10.1f}% | {ft_has_think/len(df)*100-base_has_think/len(df)*100:>+9.1f}%
""")

print("="*80)
print("END OF REPORT")
print("="*80)