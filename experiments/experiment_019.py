"""Experiment 019: Rhyme patterns — line-ending analysis.

Hypothesis: Songs have measurable rhyme structure (matching line endings),
and rhyme patterns differ by category and period.
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
df = df.dropna(subset=['lyrics'])

def analyze_rhyme(text, suffix_len=3):
    """Analyze rhyme patterns by comparing line endings."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 2:
        return pd.Series({'rhyme_density': 0, 'n_rhyme_pairs': 0, 'n_lines': len(lines),
                          'dominant_ending': '', 'max_rhyme_group': 0})

    # Extract last N Bengali chars of each line (ignoring punctuation)
    endings = []
    for line in lines:
        # Strip punctuation and whitespace from end
        clean = re.sub(r'[।॥\s\.,!?;:–—\-\u2018\u2019\u201c\u201d]+$', '', line)
        bengali = [c for c in clean if '\u0980' <= c <= '\u09FF']
        if bengali:
            endings.append(''.join(bengali[-suffix_len:]))
        else:
            endings.append('')

    # Count matching ending pairs (consecutive or within a window)
    rhyme_pairs = 0
    # Check all pairs within a window of 4 lines
    for i in range(len(endings)):
        for j in range(i+1, min(i+4, len(endings))):
            if endings[i] and endings[j] and endings[i] == endings[j]:
                rhyme_pairs += 1

    # Rhyme density: rhyme pairs / possible pairs in window
    possible_pairs = sum(min(3, len(endings) - i - 1) for i in range(len(endings) - 1))
    rhyme_density = rhyme_pairs / possible_pairs if possible_pairs > 0 else 0

    # Dominant ending group
    ending_counts = Counter(e for e in endings if e)
    if ending_counts:
        dominant = ending_counts.most_common(1)[0]
        max_group = dominant[1]
        dominant_ending = dominant[0]
    else:
        max_group = 0
        dominant_ending = ''

    return pd.Series({
        'rhyme_density': rhyme_density,
        'n_rhyme_pairs': rhyme_pairs,
        'n_lines': len(lines),
        'dominant_ending': dominant_ending,
        'max_rhyme_group': max_group,
    })

# Try with 2-char and 3-char suffix
rhyme_feats = df['lyrics'].apply(lambda x: analyze_rhyme(x, suffix_len=2))
df = pd.concat([df.reset_index(drop=True), rhyme_feats.reset_index(drop=True)], axis=1)

print(f"Songs: {len(df)}")
print(f"\n--- Rhyme statistics (2-char endings) ---")
print(f"  Mean rhyme density: {df['rhyme_density'].mean():.4f}")
print(f"  Median rhyme density: {df['rhyme_density'].median():.4f}")
print(f"  Songs with rhyme_density > 0.1: {(df['rhyme_density'] > 0.1).sum()} ({(df['rhyme_density'] > 0.1).mean():.1%})")
print(f"  Songs with rhyme_density > 0.2: {(df['rhyme_density'] > 0.2).sum()} ({(df['rhyme_density'] > 0.2).mean():.1%})")

# By category
print(f"\n--- Rhyme density by category ---")
top_cats = df['category'].value_counts().head(8).index
cat_means = {}
for cat in top_cats:
    subset = df[df['category'] == cat]
    cat_means[cat] = subset['rhyme_density'].mean()
    print(f"  {cat}: mean={subset['rhyme_density'].mean():.4f}, "
          f"median={subset['rhyme_density'].median():.4f}, n={len(subset)}")

# Kruskal-Wallis
groups = [df[df['category'] == cat]['rhyme_density'].values for cat in top_cats]
h, p = stats.kruskal(*groups)
print(f"\n  Kruskal-Wallis: H={h:.2f}, p={p:.2e}")

# Temporal trend
df_year = df.dropna(subset=['gregorian_year']).copy()
df_year['gregorian_year'] = df_year['gregorian_year'].astype(int)
r, p_time = stats.spearmanr(df_year['gregorian_year'], df_year['rhyme_density'])
print(f"\n--- Temporal trend ---")
print(f"  Year vs rhyme density: rho={r:.4f}, p={p_time:.2e}")

df_year['decade'] = (df_year['gregorian_year'] // 10) * 10
decade_rhyme = df_year.groupby('decade')['rhyme_density'].agg(['mean', 'median', 'size'])
print(f"\n  By decade:")
for dec, row in decade_rhyme.iterrows():
    print(f"    {int(dec)}s: mean={row['mean']:.4f}, median={row['median']:.4f}, n={int(row['size'])}")

# Most common line endings across the corpus
all_endings = []
for text in df['lyrics']:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines:
        clean = re.sub(r'[।॥\s\.,!?;:–—\-]+$', '', line)
        bengali = [c for c in clean if '\u0980' <= c <= '\u09FF']
        if len(bengali) >= 2:
            all_endings.append(''.join(bengali[-2:]))

ending_counts = Counter(all_endings)
print(f"\n--- Most common line endings (last 2 chars) ---")
for ending, count in ending_counts.most_common(15):
    print(f"  '{ending}': {count} ({count/len(all_endings)*100:.1f}%)")

print(f"\n=== RESULT ===")
print(f"hypothesis: Songs have measurable rhyme structure that differs by category and evolves over time")
print(f"method: 2-character line-ending matching within 4-line windows, rhyme density metric")
print(f"key_finding: Mean rhyme density={df['rhyme_density'].mean():.4f}. {(df['rhyme_density'] > 0.1).mean():.0%} of songs have >10% rhyme density. Category differences: H={h:.1f}, p={p:.2e}. Temporal trend: rho={r:.4f}, p={p_time:.2e}.")
print(f"statistical_significance: Kruskal-Wallis p={p:.2e}, temporal correlation p={p_time:.2e}")
print(f"conclusion: Rhyme is present but modest in Bengali song lyrics as transcribed. {'Categories differ significantly in rhyme usage.' if p < 0.05 else 'Categories show similar rhyme patterns.'} {'Rhyme usage changed over time.' if p_time < 0.05 else 'Rhyme density was stable over time.'}")
print(f"=== END RESULT ===")
