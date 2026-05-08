"""Experiment 035: Refrain and repetition structure in songs.

Bengali songs traditionally have a sthāyī (refrain) that repeats.
Hypothesis: Repetition patterns differ by category (devotional songs more
repetitive for congregational singing) and changed over time (late period
more experimental with less repetition).
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

def repetition_features(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 2:
        return pd.Series({
            'n_lines': len(lines),
            'n_unique_lines': len(set(lines)),
            'line_repetition_rate': 0,
            'first_line_repeats': 0,
            'has_refrain': False,
            'max_line_repeats': 1,
            'word_repetition_rate': 0,
            'most_repeated_word_frac': 0,
        })

    n_lines = len(lines)
    n_unique = len(set(lines))
    line_rep_rate = 1 - (n_unique / n_lines)

    # Does the first line repeat? (proxy for sthāyī/refrain)
    first_line = lines[0]
    first_repeats = sum(1 for l in lines[1:] if l == first_line)

    # Most repeated line
    line_counts = Counter(lines)
    max_repeats = max(line_counts.values())

    # Partial line matching (first half of first line appears later)
    first_words = first_line.split()[:4]
    first_prefix = ' '.join(first_words) if len(first_words) >= 2 else first_line
    partial_repeats = sum(1 for l in lines[1:] if l.startswith(first_prefix))

    has_refrain = first_repeats >= 1 or partial_repeats >= 2

    # Word-level repetition
    words = [re.sub(r'[।॥,\.!?;:\-–—\d]', '', w) for w in text.split()]
    words = [w for w in words if len(w) >= 2]
    if words:
        word_counts = Counter(words)
        n_unique_words = len(word_counts)
        word_rep_rate = 1 - (n_unique_words / len(words))
        most_common_frac = word_counts.most_common(1)[0][1] / len(words)
    else:
        word_rep_rate = 0
        most_common_frac = 0

    return pd.Series({
        'n_lines': n_lines,
        'n_unique_lines': n_unique,
        'line_repetition_rate': line_rep_rate,
        'first_line_repeats': first_repeats + partial_repeats,
        'has_refrain': has_refrain,
        'max_line_repeats': max_repeats,
        'word_repetition_rate': word_rep_rate,
        'most_repeated_word_frac': most_common_frac,
    })

features = df['lyrics'].apply(repetition_features)
df = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

# Overall stats
print(f"--- Overall repetition statistics ---")
print(f"Songs: {len(df)}")
print(f"Has refrain: {df['has_refrain'].sum()} ({df['has_refrain'].mean()*100:.1f}%)")
print(f"Mean line repetition rate: {df['line_repetition_rate'].mean():.3f}")
print(f"Mean word repetition rate: {df['word_repetition_rate'].mean():.3f}")
print(f"Mean first-line repeats: {df['first_line_repeats'].mean():.2f}")

# By category
print(f"\n--- Repetition by category ---")
top_cats = df.dropna(subset=['category'])['category'].value_counts().head(8).index
for cat in top_cats:
    subset = df[df['category'] == cat]
    print(f"  {cat} (n={len(subset)}):")
    print(f"    Refrain: {subset['has_refrain'].mean()*100:.0f}%, "
          f"Line rep: {subset['line_repetition_rate'].mean():.3f}, "
          f"Word rep: {subset['word_repetition_rate'].mean():.3f}, "
          f"First-line repeats: {subset['first_line_repeats'].mean():.1f}")

# Statistical test: does Puja have more repetition?
puja = df[df['category'] == 'পূজা']
prem = df[df['category'] == 'প্রেম']
prakriti = df[df['category'] == 'প্রকৃতি']
drama = df[df['category'] == 'গীতিনাট্য ও নৃত্যনাট্য']

print(f"\n--- Pairwise comparisons (line repetition rate) ---")
comparisons = [
    ('Puja', puja, 'Prem', prem),
    ('Puja', puja, 'Prakriti', prakriti),
    ('Puja', puja, 'Drama', drama),
    ('Prem', prem, 'Prakriti', prakriti),
    ('Drama', drama, 'Prem', prem),
]
for name1, g1, name2, g2 in comparisons:
    u, p = stats.mannwhitneyu(g1['line_repetition_rate'], g2['line_repetition_rate'], alternative='two-sided')
    d = (g1['line_repetition_rate'].mean() - g2['line_repetition_rate'].mean()) / \
        np.sqrt((g1['line_repetition_rate'].var() + g2['line_repetition_rate'].var()) / 2)
    print(f"  {name1} vs {name2}: d={d:.3f}, p={p:.2e}")

# Kruskal-Wallis across categories
groups = [df[df['category'] == cat]['line_repetition_rate'].values for cat in top_cats]
h, p_kw = stats.kruskal(*groups)
print(f"\n  KW across 8 categories: H={h:.2f}, p={p_kw:.2e}")

# Temporal trend
df_year = df.dropna(subset=['gregorian_year']).copy()
df_year['gregorian_year'] = df_year['gregorian_year'].astype(int)

r1, p1 = stats.spearmanr(df_year['gregorian_year'], df_year['line_repetition_rate'])
r2, p2 = stats.spearmanr(df_year['gregorian_year'], df_year['has_refrain'].astype(int))
r3, p3 = stats.spearmanr(df_year['gregorian_year'], df_year['word_repetition_rate'])
print(f"\n--- Temporal trends ---")
print(f"  Line repetition vs year: rho={r1:.4f}, p={p1:.2e}")
print(f"  Has refrain vs year: rho={r2:.4f}, p={p2:.2e}")
print(f"  Word repetition vs year: rho={r3:.4f}, p={p3:.2e}")

# Decade breakdown
df_year['decade'] = (df_year['gregorian_year'] // 10) * 10
print(f"\n--- Decade breakdown ---")
for dec in sorted(df_year['decade'].unique()):
    subset = df_year[df_year['decade'] == dec]
    if len(subset) < 10:
        continue
    print(f"  {int(dec)}s (n={len(subset)}): "
          f"refrain={subset['has_refrain'].mean()*100:.0f}%, "
          f"line_rep={subset['line_repetition_rate'].mean():.3f}, "
          f"word_rep={subset['word_repetition_rate'].mean():.3f}")

# Repetition vs raga (do certain ragas have more repetitive songs?)
df_raga = df.dropna(subset=['raga']).copy()
top_ragas = df_raga['raga'].value_counts().head(10).index
print(f"\n--- Repetition by top ragas ---")
for raga in top_ragas:
    subset = df_raga[df_raga['raga'] == raga]
    print(f"  {raga} (n={len(subset)}): "
          f"refrain={subset['has_refrain'].mean()*100:.0f}%, "
          f"line_rep={subset['line_repetition_rate'].mean():.3f}")

# Repetition and taal
df_taal = df.dropna(subset=['taal']).copy()
top_taals = df_taal['taal'].value_counts().head(8).index
print(f"\n--- Repetition by top taals ---")
for taal in top_taals:
    subset = df_taal[df_taal['taal'] == taal]
    print(f"  {taal} (n={len(subset)}): "
          f"refrain={subset['has_refrain'].mean()*100:.0f}%, "
          f"line_rep={subset['line_repetition_rate'].mean():.3f}")

print(f"\n=== RESULT ===")
print(f"hypothesis: Repetition differs by category and decreases over time")
print(f"method: Line-level and word-level repetition metrics; refrain detection (first-line repeats + partial matching); KW test across categories; Spearman temporal correlation")
print(f"key_finding: {df['has_refrain'].mean()*100:.0f}% of songs have refrains. Category effect: KW H={h:.1f}, p={p_kw:.2e}. Temporal trend for line repetition: rho={r1:.3f}, p={p1:.2e}.")
print(f"statistical_significance: KW p={p_kw:.2e}, temporal rho p={p1:.2e}")
most_rep = max(top_cats, key=lambda c: df[df['category']==c]['line_repetition_rate'].mean())
least_rep = min(top_cats, key=lambda c: df[df['category']==c]['line_repetition_rate'].mean())
print(f"conclusion: Most repetitive category: {most_rep}. Least: {least_rep}. Repetition {'decreased' if r1 < 0 else 'increased'} over time, consistent with {'growing experimentalism' if r1 < 0 else 'tradition'}.")
print(f"=== END RESULT ===")
