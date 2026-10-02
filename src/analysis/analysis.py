import pandas as pd
import re
import json
from collections import Counter

# Load the combined evaluation file
df = pd.read_csv('results/studentF_filtered/evaluation_results.csv')

print("="*80)
print("DIAGNOSTIC TASK EVALUATION REPORT")
print("="*80)

# Basic stats
print(f"\nTotal cases: {len(df)}")
print(f"Teacher (base) model cases: {len(df)}")
print(f"Student (ft) model cases: {len(df)}")

# Disease distribution
print("\n--- Disease Distribution ---")
disease_counts = Counter(df['base_true_disease'].tolist() + df['ft_true_disease'].tolist())
for disease, count in disease_counts.most_common():
    print(f"  {disease}: {count}")

# ============================================================
# 1. STRICT ACCURACY (as recorded - correct column)
# ============================================================
print("\n" + "="*80)
print("1. STRICT ACCURACY (from 'correct' column)")
print("="*80)
print("   Definition: Model's answer matches true disease (exact or synonym match)")

base_correct = df['base_correct'].sum()
ft_correct = df['ft_correct'].sum()
base_acc = base_correct / len(df) * 100
ft_acc = ft_correct / len(df) * 100

print(f"\n   Teacher (base) correct: {base_correct}/{len(df)} = {base_acc:.1f}%")
print(f"   Student (ft) correct:   {ft_correct}/{len(df)} = {ft_acc:.1f}%")
print(f"   Improvement:            {ft_acc - base_acc:+.1f}%")

# ============================================================
# 2. RELAXED ACCURACY (answer anywhere in response)
# ============================================================
print("\n" + "="*80)
print("2. RELAXED ACCURACY (answer anywhere in response)")
print("="*80)
print("   Definition: True disease or any synonym appears anywhere in the response text")

def check_answer_in_response(response, true_disease, synonyms_str):
    """Check if true disease or any synonym appears anywhere in the response."""
    if pd.isna(response) or pd.isna(synonyms_str):
        return False
    
    response_lower = response.lower()
    
    # Check true disease
    if true_disease.lower() in response_lower:
        return True
    
    # Parse synonyms
    try:
        # Clean up the synonyms string - it's stored as a string representation of a list
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
    lambda r: check_answer_in_response(r['base_response'], r['base_true_disease'], r['base_synonyms']),
    axis=1
)
df['ft_relaxed_correct'] = df.apply(
    lambda r: check_answer_in_response(r['ft_response'], r['ft_true_disease'], r['ft_synonyms']),
    axis=1
)

base_relaxed = df['base_relaxed_correct'].sum()
ft_relaxed = df['ft_relaxed_correct'].sum()
base_relaxed_acc = base_relaxed / len(df) * 100
ft_relaxed_acc = ft_relaxed / len(df) * 100

print(f"\n   Teacher (base) relaxed: {base_relaxed}/{len(df)} = {base_relaxed_acc:.1f}%")
print(f"   Student (ft) relaxed:   {ft_relaxed}/{len(df)} = {ft_relaxed_acc:.1f}%")
print(f"   Improvement:            {ft_relaxed_acc - base_relaxed_acc:+.1f}%")

# ============================================================
# 3. STRICT TAG ACCURACY (answer ONLY inside <diagnosis> tags)
# ============================================================
print("\n" + "="*80)
print("3. STRICT TAG ACCURACY (answer only inside <diagnosis> tags)")
print("="*80)
print("   Definition: True disease or synonym appears ONLY inside <diagnosis> tags")

def extract_diagnosis_tag(response):
    """Extract content between <diagnosis> and </diagnosis> tags."""
    if pd.isna(response):
        return None
    match = re.search(r'<diagnosis>(.*?)</diagnosis>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def check_tag_answer(response, true_disease, synonyms_str):
    """Check if answer is correctly placed inside <diagnosis> tags."""
    tag_content = extract_diagnosis_tag(response)
    if not tag_content:
        return False
    
    tag_lower = tag_content.lower()
    
    # Clean tag content - remove brackets, quotes, etc.
    clean_tag = re.sub(r'[\[\]\'"\s]', '', tag_lower)
    
    # Check true disease (also cleaned)
    clean_true = re.sub(r'[\[\]\'"\s]', '', true_disease.lower())
    
    if clean_true in tag_lower or clean_true in clean_tag:
        return True
    
    # Check synonyms
    try:
        synonyms_str = str(synonyms_str).strip("[]")
        synonyms = [s.strip().strip("'\"") for s in synonyms_str.split(';')]
        for syn in synonyms:
            syn_clean = re.sub(r'[\[\]\'"\s]', '', syn.lower())
            if syn_clean in tag_lower or syn_clean in clean_tag:
                return True
    except:
        pass
    
    return False

df['base_tag_correct'] = df.apply(
    lambda r: check_tag_answer(r['base_response'], r['base_true_disease'], r['base_synonyms']),
    axis=1
)
df['ft_tag_correct'] = df.apply(
    lambda r: check_tag_answer(r['ft_response'], r['ft_true_disease'], r['ft_synonyms']),
    axis=1
)

base_tag = df['base_tag_correct'].sum()
ft_tag = df['ft_tag_correct'].sum()
base_tag_acc = base_tag / len(df) * 100
ft_tag_acc = ft_tag / len(df) * 100

print(f"\n   Teacher (base) tag: {base_tag}/{len(df)} = {base_tag_acc:.1f}%")
print(f"   Student (ft) tag:   {ft_tag}/{len(df)} = {ft_tag_acc:.1f}%")
print(f"   Improvement:        {ft_tag_acc - base_tag_acc:+.1f}%")

# ============================================================
# 4. TAG USAGE RATES
# ============================================================
print("\n" + "="*80)
print("4. TAG USAGE RATES")
print("="*80)

base_has_tag = df['base_has_diagnose'].sum()
ft_has_tag = df['ft_has_diagnose'].sum()

print(f"\n   Teacher (base) used <diagnosis> tags: {base_has_tag}/{len(df)} = {base_has_tag/len(df)*100:.1f}%")
print(f"   Student (ft) used <diagnosis> tags:   {ft_has_tag}/{len(df)} = {ft_has_tag/len(df)*100:.1f}%")

# ============================================================
# 5. THINKING USAGE RATES
# ============================================================
print("\n" + "="*80)
print("5. THINKING/REASONING USAGE RATES")
print("="*80)

base_has_think = df['base_has_think'].sum()
ft_has_think = df['ft_has_think'].sum()

print(f"\n   Teacher (base) used thinking: {base_has_think}/{len(df)} = {base_has_think/len(df)*100:.1f}%")
print(f"   Student (ft) used thinking:   {ft_has_think}/{len(df)} = {ft_has_think/len(df)*100:.1f}%")

# ============================================================
# 6. CROSS-TABULATION: Tag Usage vs Accuracy
# ============================================================
print("\n" + "="*80)
print("6. TAG USAGE IMPACT ON ACCURACY")
print("="*80)

# For teacher
base_with_tag_correct = df[df['base_has_diagnose'] == True]['base_correct'].sum()
base_with_tag_total = df[df['base_has_diagnose'] == True].shape[0]
base_without_tag_correct = df[df['base_has_diagnose'] == False]['base_correct'].sum()
base_without_tag_total = df[df['base_has_diagnose'] == False].shape[0]

# For student
ft_with_tag_correct = df[df['ft_has_diagnose'] == True]['ft_correct'].sum()
ft_with_tag_total = df[df['ft_has_diagnose'] == True].shape[0]
ft_without_tag_correct = df[df['ft_has_diagnose'] == False]['ft_correct'].sum()
ft_without_tag_total = df[df['ft_has_diagnose'] == False].shape[0]

print(f"\n   --- Teacher (base) ---")
print(f"   With tags:  {base_with_tag_correct}/{base_with_tag_total} = {base_with_tag_correct/max(base_with_tag_total,1)*100:.1f}%")
print(f"   Without tags: {base_without_tag_correct}/{base_without_tag_total} = {base_without_tag_correct/max(base_without_tag_total,1)*100:.1f}%")

print(f"\n   --- Student (ft) ---")
print(f"   With tags:  {ft_with_tag_correct}/{ft_with_tag_total} = {ft_with_tag_correct/max(ft_with_tag_total,1)*100:.1f}%")
print(f"   Without tags: {ft_without_tag_correct}/{ft_without_tag_total} = {ft_without_tag_correct/max(ft_without_tag_total,1)*100:.1f}%")

# ============================================================
# 7. ACCURACY BY DISEASE (per-class breakdown)
# ============================================================
print("\n" + "="*80)
print("7. ACCURACY BY DISEASE (per-class breakdown)")
print("="*80)

diseases = df['base_true_disease'].unique()
print(f"\n   {'Disease':<50} {'Teacher':>10} {'Student':>10} {'Diff':>10}")
print(f"   {'-'*50} {'-'*10} {'-'*10} {'-'*10}")

for disease in sorted(diseases):
    mask = df['base_true_disease'] == disease
    base_d = df[mask]['base_correct'].sum() / mask.sum() * 100
    ft_d = df[mask]['ft_correct'].sum() / mask.sum() * 100
    print(f"   {disease:<50} {base_d:>9.1f}% {ft_d:>9.1f}% {ft_d-base_d:>+9.1f}%")

# ============================================================
# 8. COMBINED METRICS
# ============================================================
print("\n" + "="*80)
print("8. COMBINED METRICS SUMMARY")
print("="*80)
print(f"""
   Metric                        | Teacher (base) | Student (ft) | Delta
   ------------------------------+----------------+--------------+--------
   Strict Accuracy               | {base_acc:>15.1f}% | {ft_acc:>13.1f}% | {ft_acc-base_acc:>+10.1f}%
   Relaxed Accuracy              | {base_relaxed_acc:>15.1f}% | {ft_relaxed_acc:>13.1f}% | {ft_relaxed_acc-base_relaxed_acc:>+10.1f}%
   Strict Tag Accuracy           | {base_tag_acc:>15.1f}% | {ft_tag_acc:>13.1f}% | {ft_tag_acc-base_tag_acc:>+10.1f}%
   Tag Usage Rate                | {base_has_tag/len(df)*100:>14.1f}% | {ft_has_tag/len(df)*100:>12.1f}% | {ft_has_tag/len(df)*100-base_has_tag/len(df)*100:>+10.1f}%
   Thinking Usage Rate           | {base_has_think/len(df)*100:>14.1f}% | {ft_has_think/len(df)*100:>12.1f}% | {ft_has_think/len(df)*100-base_has_think/len(df)*100:>+10.1f}%
""")

# ============================================================
# 9. TAG QUALITY METRICS
# ============================================================
print("="*80)
print("9. TAG QUALITY METRICS")
print("="*80)

# Extract tags and check quality
def extract_tag_content(response):
    match = re.search(r'<diagnosis>(.*?)</diagnosis>', str(response), re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

df['base_tag_content'] = df['base_response'].apply(extract_tag_content)
df['ft_tag_content'] = df['ft_response'].apply(extract_tag_content)

# Tag length stats
base_tag_lens = df[df['base_tag_content'].str.len() > 0]['base_tag_content'].str.len()
ft_tag_lens = df[df['ft_tag_content'].str.len() > 0]['ft_tag_content'].str.len()

print(f"\n   --- Teacher (base) tag content ---")
if len(base_tag_lens) > 0:
    print(f"   Non-empty tags: {len(base_tag_lens)}/{len(df)}")
    print(f"   Mean tag length: {base_tag_lens.mean():.0f} chars")
    print(f"   Median tag length: {base_tag_lens.median():.0f} chars")

print(f"\n   --- Student (ft) tag content ---")
if len(ft_tag_lens) > 0:
    print(f"   Non-empty tags: {len(ft_tag_lens)}/{len(df)}")
    print(f"   Mean tag length: {ft_tag_lens.mean():.0f} chars")
    print(f"   Median tag length: {ft_tag_lens.median():.0f} chars")

# ============================================================
# 10. AGREEMENT METRICS
# ============================================================
print("\n" + "="*80)
print("10. MODEL AGREEMENT")
print("="*80)

both_correct = (df['base_correct'] == True) & (df['ft_correct'] == True)
both_wrong = (df['base_correct'] == False) & (df['ft_correct'] == False)
base_only = (df['base_correct'] == True) & (df['ft_correct'] == False)
ft_only = (df['base_correct'] == False) & (df['ft_correct'] == True)

print(f"\n   Both correct:    {both_correct.sum()}/{len(df)} = {both_correct.sum()/len(df)*100:.1f}%")
print(f"   Both wrong:      {both_wrong.sum()}/{len(df)} = {both_wrong.sum()/len(df)*100:.1f}%")
print(f"   Teacher only:    {base_only.sum()}/{len(df)} = {base_only.sum()/len(df)*100:.1f}%")
print(f"   Student only:    {ft_only.sum()}/{len(df)} = {ft_only.sum()/len(df)*100:.1f}%")

# Kappa-like agreement
agreement = (both_correct.sum() + both_wrong.sum()) / len(df)
print(f"\n   Overall agreement: {agreement*100:.1f}%")

# ============================================================
# 11. CASE STUDIES - Student got it right but teacher got it wrong
# ============================================================
print("\n" + "="*80)
print("11. EXAMPLES: Student improved (got right, teacher wrong)")
print("="*80)

improved = df[(df['ft_correct'] == True) & (df['base_correct'] == False)]
print(f"\n   Total improved cases: {len(improved)}/{len(df)} = {len(improved)/len(df)*100:.1f}%")

for i, row in improved.head(5).iterrows():
    print(f"\n   Case {row['base_case_idx']} - True: {row['ft_true_disease']}")
    tag_ft = extract_diagnosis_tag(row['ft_response'])
    if tag_ft:
        print(f"   Student tag: {tag_ft[:100]}...")

# ============================================================
# 12. CASE STUDIES - Teacher got it right but student got it wrong
# ============================================================
print("\n" + "="*80)
print("12. EXAMPLES: Student regressed (got wrong, teacher right)")
print("="*80)

regressed = df[(df['ft_correct'] == False) & (df['base_correct'] == True)]
print(f"\n   Total regressed cases: {len(regressed)}/{len(df)} = {len(regressed)/len(df)*100:.1f}%")

for i, row in regressed.head(5).iterrows():
    print(f"\n   Case {row['base_case_idx']} - True: {row['ft_true_disease']}")
    tag_base = extract_diagnosis_tag(row['base_response'])
    if tag_base:
        print(f"   Teacher tag: {tag_base[:100]}...")

print("\n" + "="*80)
print("END OF REPORT")
print("="*80)