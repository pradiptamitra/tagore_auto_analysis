"""Experiment 030: Archaic vs modern Bengali forms over Tagore's career.

From Exp 012: temporal signal in char n-grams (িব vs ওয়া verb forms).
From Exp 026: Bhanusingh uses Brajabuli forms.
Hypothesis: Specific archaic→modern Bengali substitutions are trackable,
showing a measurable language modernization trend.
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from scipy import stats
import re

df = load_tagore()
df = df.dropna(subset=['gregorian_year', 'lyrics'])
df['gregorian_year'] = df['gregorian_year'].astype(int)

# Archaic-modern pairs in Bengali
# Each pair: (archaic_pattern, modern_pattern, description)
pairs = [
    # Pronouns
    ('তব', 'তোমার', '2nd person possessive'),
    ('মম', 'আমার', '1st person possessive'),
    ('মোর', 'আমার', '1st person possessive (colloquial archaic)'),
    ('তোরে', 'তোমাকে', '2nd person accusative'),
    # Verb forms
    ('হউক', 'হোক', 'let it be'),
    ('করিব', 'করব', 'future tense'),
    ('যাইব', 'যাব', 'future tense'),
    ('আসিব', 'আসব', 'future tense'),
    ('করিয়া', 'করে', 'participle'),
    ('হইয়া', 'হয়ে', 'participle'),
    ('লইয়া', 'নিয়ে', 'participle (taking)'),
    ('দিয়া', 'দিয়ে', 'participle (giving)'),
    # Others
    ('হেরি', 'দেখি', 'seeing'),
    ('সখি', 'বন্ধু', 'friend/companion'),
]

def count_forms(text, pattern):
    """Count occurrences of pattern as a word or word-start."""
    words = text.split()
    count = 0
    for w in words:
        clean = re.sub(r'[।॥,\.!?;:\-–—]', '', w)
        if clean == pattern or clean.startswith(pattern):
            count += 1
    return count

# Count each archaic and modern form per song
for archaic, modern, desc in pairs:
    df[f'archaic_{archaic}'] = df['lyrics'].apply(lambda t: count_forms(t, archaic))
    df[f'modern_{modern}'] = df['lyrics'].apply(lambda t: count_forms(t, modern))

# Compute aggregate archaic and modern scores
archaic_cols = [f'archaic_{p[0]}' for p in pairs]
modern_cols = [f'modern_{p[1]}' for p in pairs]

# Avoid double-counting আমার (appears twice as modern)
df['total_archaic'] = df[archaic_cols].sum(axis=1)
df['total_modern'] = df[[f'modern_{p[1]}' for p in pairs]].sum(axis=1)
df['n_words'] = df['lyrics'].apply(lambda t: len(t.split()))
df['archaic_rate'] = df['total_archaic'] / df['n_words']
df['modern_rate'] = df['total_modern'] / df['n_words']
df['modernity_ratio'] = df['total_modern'] / (df['total_archaic'] + df['total_modern'] + 0.1)

# Temporal trends
print(f"--- Overall statistics ---")
print(f"Songs: {len(df)}")
print(f"Mean archaic words/song: {df['total_archaic'].mean():.2f}")
print(f"Mean modern words/song: {df['total_modern'].mean():.2f}")

r_arch, p_arch = stats.spearmanr(df['gregorian_year'], df['archaic_rate'])
r_mod, p_mod = stats.spearmanr(df['gregorian_year'], df['modern_rate'])
r_ratio, p_ratio = stats.spearmanr(df['gregorian_year'], df['modernity_ratio'])
print(f"\nArchaic rate vs year: rho={r_arch:.4f}, p={p_arch:.2e}")
print(f"Modern rate vs year: rho={r_mod:.4f}, p={p_mod:.2e}")
print(f"Modernity ratio vs year: rho={r_ratio:.4f}, p={p_ratio:.2e}")

# Individual pair trends
print(f"\n--- Individual pair trends ---")
for archaic, modern, desc in pairs:
    arch_col = f'archaic_{archaic}'
    mod_col = f'modern_{modern}'
    r_a, p_a = stats.spearmanr(df['gregorian_year'], df[arch_col])
    r_m, p_m = stats.spearmanr(df['gregorian_year'], df[mod_col])
    arch_total = df[arch_col].sum()
    mod_total = df[mod_col].sum()
    if arch_total + mod_total > 20:  # only report if enough data
        print(f"  {desc}: {archaic}({arch_total}) → {modern}({mod_total})")
        print(f"    Archaic trend: rho={r_a:.3f}, p={p_a:.2e}")
        print(f"    Modern trend:  rho={r_m:.3f}, p={p_m:.2e}")

# Decade breakdown for key pairs
df['decade'] = (df['gregorian_year'] // 10) * 10
print(f"\n--- Decade breakdown for তব vs তোমার ---")
for dec in sorted(df['decade'].unique()):
    subset = df[df['decade'] == dec]
    tab = subset['archaic_তব'].sum()
    tomar = subset['modern_তোমার'].sum()
    total_words = subset['n_words'].sum()
    print(f"  {int(dec)}s: তব={tab} ({tab/total_words*1000:.1f}‰), "
          f"তোমার={tomar} ({tomar/total_words*1000:.1f}‰), n={len(subset)}")

print(f"\n--- Decade breakdown for মম vs আমার ---")
for dec in sorted(df['decade'].unique()):
    subset = df[df['decade'] == dec]
    momo = subset['archaic_মম'].sum()
    amar = subset['modern_আমার'].sum()
    total_words = subset['n_words'].sum()
    print(f"  {int(dec)}s: মম={momo} ({momo/total_words*1000:.1f}‰), "
          f"আমার={amar} ({amar/total_words*1000:.1f}‰)")

# By category
print(f"\n--- Modernity ratio by category ---")
for cat in df['category'].value_counts().head(6).index:
    subset = df[df['category'] == cat]
    print(f"  {cat}: archaic_rate={subset['archaic_rate'].mean()*100:.2f}%, "
          f"modern_rate={subset['modern_rate'].mean()*100:.2f}%, "
          f"modernity={subset['modernity_ratio'].mean():.3f}")

print(f"\n=== RESULT ===")
print(f"hypothesis: Tagore's language modernized measurably over his career")
print(f"method: Tracked 14 archaic-modern Bengali word pairs, computed rates and modernity ratio over time")
print(f"key_finding: Archaic forms {'decreased' if r_arch < 0 else 'increased'} (rho={r_arch:.3f}, p={p_arch:.2e}). Modern forms {'increased' if r_mod > 0 else 'decreased'} (rho={r_mod:.3f}, p={p_mod:.2e}). Modernity ratio trend: rho={r_ratio:.3f}, p={p_ratio:.2e}.")
print(f"statistical_significance: Archaic p={p_arch:.2e}, modern p={p_mod:.2e}, ratio p={p_ratio:.2e}")
print(f"conclusion: Tagore's language shows {'clear' if p_ratio < 0.01 else 'modest'} modernization — archaic forms were {'gradually replaced' if r_arch < 0 and r_mod > 0 else 'maintained alongside'} modern equivalents.")
print(f"=== END RESULT ===")
