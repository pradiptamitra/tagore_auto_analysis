"""Experiment 002: Temporal evolution of vocabulary richness across Tagore's career.

Hypothesis: Tagore's vocabulary richness changed systematically over his career,
with measurable shifts around key life events (wife's death 1902, Nobel 1913).
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from scipy import stats

df = load_tagore()
df = df.dropna(subset=['gregorian_year', 'lyrics'])
df['gregorian_year'] = df['gregorian_year'].astype(int)
print(f"Songs with year and lyrics: {len(df)}")
print(f"Year range: {df['gregorian_year'].min()} - {df['gregorian_year'].max()}")

# Compute vocabulary richness metrics per song
def vocab_metrics(text):
    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return pd.Series({'n_words': 0, 'n_unique': 0, 'ttr': 0, 'hapax_ratio': 0})
    unique = set(words)
    n_unique = len(unique)
    ttr = n_unique / n_words  # type-token ratio
    # Hapax legomena ratio (words appearing only once)
    from collections import Counter
    counts = Counter(words)
    hapax = sum(1 for w, c in counts.items() if c == 1)
    hapax_ratio = hapax / n_words
    return pd.Series({'n_words': n_words, 'n_unique': n_unique, 'ttr': ttr, 'hapax_ratio': hapax_ratio})

metrics = df['lyrics'].apply(vocab_metrics)
df = pd.concat([df, metrics], axis=1)

# Group by decade
df['decade'] = (df['gregorian_year'] // 10) * 10
decade_stats = df.groupby('decade').agg(
    count=('ttr', 'size'),
    mean_ttr=('ttr', 'mean'),
    std_ttr=('ttr', 'std'),
    mean_hapax=('hapax_ratio', 'mean'),
    mean_words=('n_words', 'mean'),
    mean_unique=('n_unique', 'mean'),
).reset_index()

print("\nDecade-level statistics:")
print(decade_stats.to_string(index=False))

# Correlation between year and TTR
r_ttr, p_ttr = stats.pearsonr(df['gregorian_year'], df['ttr'])
print(f"\nCorrelation year vs TTR: r={r_ttr:.4f}, p={p_ttr:.2e}")

r_hapax, p_hapax = stats.pearsonr(df['gregorian_year'], df['hapax_ratio'])
print(f"Correlation year vs hapax ratio: r={r_hapax:.4f}, p={p_hapax:.2e}")

r_words, p_words = stats.pearsonr(df['gregorian_year'], df['n_words'])
print(f"Correlation year vs word count: r={r_words:.4f}, p={p_words:.2e}")

# Test for structural break around 1902 (wife's death) and 1913 (Nobel)
for break_year, event in [(1902, "wife's death"), (1913, "Nobel Prize")]:
    before = df[df['gregorian_year'] < break_year]['ttr']
    after = df[(df['gregorian_year'] >= break_year) & (df['gregorian_year'] < break_year + 10)]['ttr']
    if len(before) > 10 and len(after) > 10:
        t_stat, p_val = stats.ttest_ind(before, after)
        print(f"\nTTR before vs after {event} ({break_year}):")
        print(f"  Before: mean={before.mean():.4f} (n={len(before)})")
        print(f"  After:  mean={after.mean():.4f} (n={len(after)})")
        print(f"  t={t_stat:.3f}, p={p_val:.4f}")

# Compare early (1873-1900), middle (1901-1920), late (1921-1941)
periods = {
    'Early (1873-1900)': df[(df['gregorian_year'] >= 1873) & (df['gregorian_year'] <= 1900)],
    'Middle (1901-1920)': df[(df['gregorian_year'] >= 1901) & (df['gregorian_year'] <= 1920)],
    'Late (1921-1941)': df[(df['gregorian_year'] >= 1921) & (df['gregorian_year'] <= 1941)],
}

print("\nPeriod comparison:")
for name, subset in periods.items():
    print(f"  {name}: n={len(subset)}, mean_ttr={subset['ttr'].mean():.4f}, mean_words={subset['n_words'].mean():.1f}")

# Kruskal-Wallis test across periods
period_groups = [subset['ttr'].values for subset in periods.values()]
h_stat, p_kw = stats.kruskal(*period_groups)
print(f"\nKruskal-Wallis test (TTR across periods): H={h_stat:.3f}, p={p_kw:.2e}")

# Spearman rank correlation (more robust)
rho, p_spear = stats.spearmanr(df['gregorian_year'], df['ttr'])
print(f"Spearman correlation year vs TTR: rho={rho:.4f}, p={p_spear:.2e}")

print(f"\n=== RESULT ===")
print(f"hypothesis: Tagore's vocabulary richness changed systematically over his career")
print(f"method: Type-token ratio and hapax ratio computed per song, correlated with year, compared across periods and around key events")
direction = "increased" if r_ttr > 0 else "decreased"
print(f"key_finding: TTR {direction} over career (Pearson r={r_ttr:.4f}, p={p_ttr:.2e}; Spearman rho={rho:.4f}). Kruskal-Wallis across 3 periods: H={h_stat:.3f}, p={p_kw:.2e}")
print(f"statistical_significance: Pearson p={p_ttr:.2e}, Spearman p={p_spear:.2e}, Kruskal-Wallis p={p_kw:.2e}")
sig = "strongly significant" if p_ttr < 0.001 else "significant" if p_ttr < 0.05 else "not significant"
print(f"conclusion: Vocabulary richness {direction} {sig}ly over Tagore's career. {'This suggests his later songs used more diverse/compressed vocabulary, possibly reflecting artistic maturity.' if r_ttr > 0 else 'This suggests his later songs became more repetitive/focused in word choice, possibly reflecting refined simplicity.'}")
print(f"=== END RESULT ===")
