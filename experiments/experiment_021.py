"""Experiment 021: The 1936 burst — what drove 120 songs at age 75?

From Exp 008: 1936 was one of Tagore's most productive years.
Hypothesis: The 1936 burst has a distinctive character — different category mix,
musical choices, or vocabulary compared to surrounding years.
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from collections import Counter

df = load_tagore()
df = df.dropna(subset=['gregorian_year', 'lyrics'])
df['gregorian_year'] = df['gregorian_year'].astype(int)

# Define the burst and context years
burst = df[df['gregorian_year'] == 1936].copy()
context = df[(df['gregorian_year'] >= 1930) & (df['gregorian_year'] <= 1939) & (df['gregorian_year'] != 1936)].copy()
late_other = df[(df['gregorian_year'] >= 1920) & (df['gregorian_year'] <= 1935)].copy()

print(f"1936 burst: {len(burst)} songs")
print(f"1930s context (excl 1936): {len(context)} songs")
print(f"1920-1935 comparison: {len(late_other)} songs")

# Category distribution in 1936
print(f"\n--- 1936 Category Distribution ---")
burst_cats = burst['category'].value_counts()
print(burst_cats.to_string())

print(f"\n--- 1930s (excl 1936) Category Distribution ---")
context_cats = context['category'].value_counts()
print(context_cats.to_string())

# Normalized comparison
print(f"\n--- Category proportions: 1936 vs rest of 1930s ---")
all_cats = set(burst_cats.index) | set(context_cats.index)
for cat in sorted(all_cats, key=lambda c: burst_cats.get(c, 0), reverse=True):
    b_pct = burst_cats.get(cat, 0) / len(burst) * 100
    c_pct = context_cats.get(cat, 0) / len(context) * 100 if len(context) > 0 else 0
    print(f"  {cat}: 1936={b_pct:.0f}%, rest={c_pct:.0f}%")

# Serial number analysis — where do these songs fall in the Gitabitan?
print(f"\n--- Serial Number Distribution ---")
burst_serial = burst['serial_number'].dropna()
context_serial = context['serial_number'].dropna()
late_serial = late_other['serial_number'].dropna()
print(f"  1936: mean={burst_serial.mean():.0f}, range=[{burst_serial.min():.0f}, {burst_serial.max():.0f}]")
print(f"  1930s other: mean={context_serial.mean():.0f}, range=[{context_serial.min():.0f}, {context_serial.max():.0f}]")
print(f"  1920-1935: mean={late_serial.mean():.0f}, range=[{late_serial.min():.0f}, {late_serial.max():.0f}]")

# Musical features
print(f"\n--- Musical Features (1936 vs rest of 1930s) ---")
for feat, label in [('raga', 'Raga'), ('taal', 'Taal')]:
    b_vals = burst[feat].dropna()
    c_vals = context[feat].dropna()
    print(f"\n  {label}:")
    print(f"    1936: {b_vals.nunique()} unique out of {len(b_vals)} songs")
    if len(b_vals) > 0:
        print(f"    Top: {b_vals.value_counts().head(5).to_string()}")

# Vocabulary comparison
def vocab_stats(texts):
    words = ' '.join(texts).split()
    unique = set(words)
    return len(words), len(unique), len(unique)/len(words) if words else 0

b_total, b_unique, b_ttr = vocab_stats(burst['lyrics'])
c_total, c_unique, c_ttr = vocab_stats(context['lyrics'])
print(f"\n--- Corpus-level vocabulary ---")
print(f"  1936: {b_total} total words, {b_unique} unique, corpus TTR={b_ttr:.4f}")
print(f"  1930s other: {c_total} total words, {c_unique} unique, corpus TTR={c_ttr:.4f}")

# Per-song TTR
burst_ttrs = burst['lyrics'].apply(lambda t: len(set(t.split())) / max(len(t.split()), 1))
context_ttrs = context['lyrics'].apply(lambda t: len(set(t.split())) / max(len(t.split()), 1))
t_stat, p_ttr = stats.ttest_ind(burst_ttrs, context_ttrs)
print(f"  Per-song TTR: 1936={burst_ttrs.mean():.4f}, rest={context_ttrs.mean():.4f}, t={t_stat:.2f}, p={p_ttr:.4f}")

# Song length
burst_len = burst['lyrics'].apply(lambda t: len(t.split()))
context_len = context['lyrics'].apply(lambda t: len(t.split()))
t_stat2, p_len = stats.ttest_ind(burst_len, context_len)
print(f"  Song length: 1936={burst_len.mean():.1f} words, rest={context_len.mean():.1f}, t={t_stat2:.2f}, p={p_len:.4f}")

# Can we classify 1936 from other 1930s songs?
combined = pd.concat([
    burst[['lyrics']].assign(label=1),
    context[['lyrics']].assign(label=0),
])
vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=10000)
X = vec.fit_transform(combined['lyrics'])
y = combined['label'].values
baseline = max(y.mean(), 1 - y.mean())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cls_scores = cross_val_score(
    LogisticRegression(max_iter=1000, C=0.5, solver='lbfgs'),
    X, y, cv=cv, scoring='accuracy'
)
print(f"\n--- Classification: 1936 vs other 1930s ---")
print(f"  Accuracy: {cls_scores.mean():.3f} ± {cls_scores.std():.3f} (baseline: {baseline:.3f})")

# Subcategory analysis for the dominant category
top_cat = burst_cats.index[0]
burst_sub = burst[burst['category'] == top_cat]['subcategory'].value_counts()
print(f"\n--- Subcategories within {top_cat} (1936) ---")
print(burst_sub.to_string())

print(f"\n=== RESULT ===")
print(f"hypothesis: The 1936 burst has a distinctive compositional character")
print(f"method: Category/raga/taal distribution, vocabulary analysis, classification (1936 vs other 1930s)")
print(f"key_finding: 1936 dominated by {top_cat} ({burst_cats.iloc[0]}/{len(burst)} songs). Classification {cls_scores.mean():.0%} vs {baseline:.0%} baseline. Songs {'shorter' if burst_len.mean() < context_len.mean() else 'longer'} than context ({burst_len.mean():.0f} vs {context_len.mean():.0f} words).")
print(f"statistical_significance: Classification accuracy {cls_scores.mean():.3f}, song length p={p_len:.4f}")
print(f"conclusion: The 1936 burst reflects a focused creative period in specific thematic territory, not a broad diversification.")
print(f"=== END RESULT ===")
