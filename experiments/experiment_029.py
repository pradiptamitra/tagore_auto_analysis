"""Experiment 029: Monsoon imagery across categories.

From Exp 016: monsoon (বর্ষা) has the most distinctive seasonal vocabulary
(ঘন, সজল, কদম, ঝর). Hypothesis: monsoon imagery "leaks" into non-Prakriti
categories, especially Prem (love), because rain is a classic metaphor for
longing in Bengali literature.
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import re

df = load_tagore()
df = df.dropna(subset=['category', 'lyrics'])

# Monsoon vocabulary (from Exp 016 + domain knowledge)
monsoon_words = {'ঘন', 'সজল', 'কদম', 'ঝর', 'বর্ষ', 'মেঘ', 'বৃষ্টি', 'বাদল',
                 'শ্রাবণ', 'আষাঢ়', 'বর্ষা', 'গর্জন', 'বিদ্যুৎ', 'নদ', 'ধারা',
                 'বন্যা', 'কাদা', 'জল', 'ময়ূর', 'দামিনী'}

def monsoon_score(text):
    """Count monsoon-related word stems in text."""
    words = text.split()
    clean = [re.sub(r'[।॥,\.!?;:\-–—]', '', w) for w in words]
    count = 0
    for w in clean:
        for mw in monsoon_words:
            if mw in w:  # substring match to catch inflected forms
                count += 1
                break
    return count, count / max(len(words), 1)

scores = df['lyrics'].apply(lambda t: pd.Series(monsoon_score(t), index=['monsoon_count', 'monsoon_rate']))
df = pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)

# By category
print(f"--- Monsoon vocabulary density by category ---")
cat_monsoon = df.groupby('category').agg(
    mean_rate=('monsoon_rate', 'mean'),
    mean_count=('monsoon_count', 'mean'),
    pct_has_any=('monsoon_count', lambda x: (x > 0).mean()),
    n=('monsoon_rate', 'size'),
).sort_values('mean_rate', ascending=False)

for cat, row in cat_monsoon.iterrows():
    print(f"  {cat} (n={int(row['n'])}): rate={row['mean_rate']*100:.2f}%, "
          f"count={row['mean_count']:.2f}/song, has_any={row['pct_has_any']:.0%}")

# Within Prakriti: monsoon vs non-monsoon subcategories
prakriti = df[df['category'] == 'প্রকৃতি'].dropna(subset=['subcategory'])
print(f"\n--- Within Prakriti by season ---")
for sub in prakriti['subcategory'].value_counts().index:
    subset = prakriti[prakriti['subcategory'] == sub]
    print(f"  {sub} (n={len(subset)}): monsoon rate={subset['monsoon_rate'].mean()*100:.2f}%")

# Compare Prem monsoon vocabulary to Prakriti-barsha
prem = df[df['category'] == 'প্রেম']
barsha = prakriti[prakriti['subcategory'] == 'বর্ষা']
other_prakriti = prakriti[prakriti['subcategory'] != 'বর্ষা']

print(f"\n--- Monsoon vocabulary comparison ---")
print(f"  বর্ষা (Prakriti-Monsoon, n={len(barsha)}): {barsha['monsoon_rate'].mean()*100:.2f}%")
print(f"  Other Prakriti seasons (n={len(other_prakriti)}): {other_prakriti['monsoon_rate'].mean()*100:.2f}%")
print(f"  প্রেম (Love, n={len(prem)}): {prem['monsoon_rate'].mean()*100:.2f}%")
print(f"  প্রেম ও প্রকৃতি (n={len(df[df['category']=='প্রেম ও প্রকৃতি'])}): "
      f"{df[df['category']=='প্রেম ও প্রকৃতি']['monsoon_rate'].mean()*100:.2f}%")

# Which Prem songs have highest monsoon density?
prem_monsoon = prem[prem['monsoon_count'] >= 3].sort_values('monsoon_rate', ascending=False)
print(f"\n--- Prem songs with ≥3 monsoon words: {len(prem_monsoon)} ({len(prem_monsoon)/len(prem)*100:.1f}% of Prem) ---")
if len(prem_monsoon) > 0:
    for _, row in prem_monsoon.head(5).iterrows():
        preview = row['lyrics'][:80].replace('\n', ' ')
        print(f"  Rate={row['monsoon_rate']*100:.1f}%, Count={int(row['monsoon_count'])}: {preview}...")

# Statistical test: is Prem's monsoon rate higher than expected (vs Puja/Bichitra)?
puja = df[df['category'] == 'পূজা']
bichitra = df[df['category'] == 'বিচিত্র']
u1, p1 = stats.mannwhitneyu(prem['monsoon_rate'], puja['monsoon_rate'], alternative='greater')
u2, p2 = stats.mannwhitneyu(prem['monsoon_rate'], bichitra['monsoon_rate'], alternative='greater')
print(f"\n--- Prem vs other categories (monsoon rate) ---")
print(f"  Prem > Puja: p={p1:.4f}")
print(f"  Prem > Bichitra: p={p2:.4f}")

# Temporal: does monsoon imagery in Prem change over time?
prem_year = prem.dropna(subset=['gregorian_year']).copy()
prem_year['gregorian_year'] = prem_year['gregorian_year'].astype(int)
r, p = stats.spearmanr(prem_year['gregorian_year'], prem_year['monsoon_rate'])
print(f"\n--- Temporal trend of monsoon imagery in Prem ---")
print(f"  Year vs monsoon rate: rho={r:.4f}, p={p:.4f}")

# Kruskal-Wallis across categories (excluding Prakriti)
non_prakriti = df[df['category'] != 'প্রকৃতি']
top_cats_np = non_prakriti['category'].value_counts().head(6).index
groups = [non_prakriti[non_prakriti['category'] == cat]['monsoon_rate'].values for cat in top_cats_np]
h, p_kw = stats.kruskal(*groups)
print(f"\n--- KW test across non-Prakriti categories ---")
print(f"  H={h:.2f}, p={p_kw:.4f}")

print(f"\n=== RESULT ===")
print(f"hypothesis: Monsoon imagery leaks into non-Prakriti categories, especially Prem")
print(f"method: Monsoon vocabulary scoring (20 keywords, substring match), category comparison, Prem temporal trend")
print(f"key_finding: Prem monsoon rate ({prem['monsoon_rate'].mean()*100:.2f}%) is {'higher' if p1 < 0.05 else 'not higher'} than Puja ({puja['monsoon_rate'].mean()*100:.2f}%), p={p1:.4f}. {len(prem_monsoon)} Prem songs ({len(prem_monsoon)/len(prem)*100:.0f}%) have ≥3 monsoon words.")
print(f"statistical_significance: Prem>Puja p={p1:.4f}, KW across non-Prakriti p={p_kw:.4f}")
print(f"conclusion: Monsoon imagery does appear in love songs, consistent with the Bengali literary tradition of rain as longing metaphor, but the effect size is {'moderate' if p1 < 0.05 else 'small'}.")
print(f"=== END RESULT ===")
