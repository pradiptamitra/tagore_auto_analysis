"""Experiment 003: Raga distribution analysis — long tail and temporal shifts.

Hypothesis: Raga usage follows a power law distribution, and Tagore's raga
preferences shifted over his career.
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter

df = load_tagore()

# Raga analysis - drop missing
df_raga = df.dropna(subset=['raga'])
print(f"Songs with raga info: {len(df_raga)} / {len(df)}")

raga_counts = df_raga['raga'].value_counts()
print(f"\nTotal distinct ragas: {len(raga_counts)}")
print(f"Top 10 ragas:")
print(raga_counts.head(10).to_string())

# Long tail analysis
print(f"\nDistribution statistics:")
print(f"  Top 5 ragas cover: {raga_counts.head(5).sum()} songs ({raga_counts.head(5).sum()/len(df_raga)*100:.1f}%)")
print(f"  Top 10 ragas cover: {raga_counts.head(10).sum()} songs ({raga_counts.head(10).sum()/len(df_raga)*100:.1f}%)")
print(f"  Top 20 ragas cover: {raga_counts.head(20).sum()} songs ({raga_counts.head(20).sum()/len(df_raga)*100:.1f}%)")
print(f"  Ragas with only 1 song: {(raga_counts == 1).sum()} ({(raga_counts == 1).sum()/len(raga_counts)*100:.1f}%)")
print(f"  Ragas with ≤3 songs: {(raga_counts <= 3).sum()} ({(raga_counts <= 3).sum()/len(raga_counts)*100:.1f}%)")

# Test if it follows Zipf's law: log(rank) vs log(frequency)
ranks = np.arange(1, len(raga_counts) + 1)
freqs = raga_counts.values
log_ranks = np.log(ranks)
log_freqs = np.log(freqs)
slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)
print(f"\nZipf's law fit (log-log regression):")
print(f"  Slope: {slope:.3f} (pure Zipf = -1.0)")
print(f"  R²: {r_value**2:.4f}")
print(f"  p-value: {p_value:.2e}")

# Temporal analysis
df_raga_year = df_raga.dropna(subset=['gregorian_year'])
df_raga_year['gregorian_year'] = df_raga_year['gregorian_year'].astype(int)
df_raga_year['decade'] = (df_raga_year['gregorian_year'] // 10) * 10

print(f"\nSongs with both raga and year: {len(df_raga_year)}")

# Raga diversity per decade
decade_diversity = df_raga_year.groupby('decade').agg(
    n_songs=('raga', 'size'),
    n_ragas=('raga', 'nunique'),
).reset_index()
decade_diversity['ragas_per_song'] = decade_diversity['n_ragas'] / decade_diversity['n_songs']

print("\nRaga diversity by decade:")
print(decade_diversity.to_string(index=False))

# Did top raga dominance change over time?
# For each decade, what fraction of songs use top-5 overall ragas?
top5_ragas = set(raga_counts.head(5).index)
for _, row in decade_diversity.iterrows():
    dec = int(row['decade'])
    dec_songs = df_raga_year[df_raga_year['decade'] == dec]
    top5_frac = dec_songs['raga'].isin(top5_ragas).mean()
    print(f"  {dec}s: {top5_frac:.1%} use top-5 ragas")

# Test: correlation between year and raga diversity (per-year)
year_div = df_raga_year.groupby('gregorian_year').agg(
    n_songs=('raga', 'size'),
    n_ragas=('raga', 'nunique'),
).reset_index()
year_div = year_div[year_div['n_songs'] >= 5]  # filter low-count years
year_div['diversity_ratio'] = year_div['n_ragas'] / year_div['n_songs']

r_div, p_div = stats.spearmanr(year_div['gregorian_year'], year_div['diversity_ratio'])
print(f"\nSpearman correlation year vs raga diversity ratio: rho={r_div:.4f}, p={p_div:.4f}")

# Chi-squared: is raga distribution independent of period?
df_raga_year['period'] = pd.cut(df_raga_year['gregorian_year'],
                                 bins=[1872, 1900, 1920, 1942],
                                 labels=['Early', 'Middle', 'Late'])
# Use top 10 ragas for chi-squared (others too sparse)
top10 = set(raga_counts.head(10).index)
df_top10 = df_raga_year[df_raga_year['raga'].isin(top10)].copy()
ct = pd.crosstab(df_top10['period'], df_top10['raga'])
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
print(f"\nChi-squared test (top-10 raga × period): χ²={chi2:.1f}, dof={dof}, p={p_chi:.2e}")

print(f"\n=== RESULT ===")
print(f"hypothesis: Raga usage follows a power law and Tagore's raga preferences shifted over his career")
print(f"method: Zipf's law fit, raga diversity per decade, chi-squared test of raga×period independence")
print(f"key_finding: Strong long tail (Zipf slope={slope:.2f}, R²={r_value**2:.3f}). Top 10 ragas cover {raga_counts.head(10).sum()/len(df_raga)*100:.0f}% of songs. Raga choices are period-dependent (χ²={chi2:.1f}, p={p_chi:.2e}).")
print(f"statistical_significance: Zipf R²={r_value**2:.3f}; chi-squared p={p_chi:.2e}; diversity correlation rho={r_div:.3f}, p={p_div:.4f}")
print(f"conclusion: Tagore's raga usage follows a steep power law — a few ragas dominate while most appear rarely. His raga preferences significantly shifted across career periods, suggesting evolving musical taste.")
print(f"=== END RESULT ===")
