"""Experiment 010: Taal complexity over Tagore's career.

Hypothesis: Tagore shifted from complex taals to simpler ones over his career.

Taal beat counts (matras per cycle):
- দাদরা (Dadra): 6
- কাহারবা (Kaharba): 8
- ত্রিতাল (Tritaal/Teentaal): 16
- একতাল (Ektal): 12
- ঝাঁপতাল (Jhaptaal): 10
- তেওরা/তেওরা (Teora): 7
- চৌতাল (Chautal): 12
- ষষ্ঠী (Sashthi): 6
- খেমটা (Khemta): 6
- আড়াঠেকা (Addha Theka): 8
- রূপকড়া (Rupakra): 7
- যৎ (Yat): 8
- ধামার (Dhamar): 14
- ঝুমরা (Jhumra): 14
- সুরফাঁকতাল (Surfanktaal): 10
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from scipy import stats

df = load_tagore()
df = df.dropna(subset=['taal', 'gregorian_year'])
df['gregorian_year'] = df['gregorian_year'].astype(int)

# Map taals to beat counts
taal_beats = {
    'দাদরা': 6,
    'কাহারবা': 8,
    'ত্রিতাল': 16,
    'একতাল': 12,
    'ঝাঁপতাল': 10,
    'তেওরা': 7,
    ' তেওরা': 7,  # with leading space (data quirk from Exp 006)
    'চৌতাল': 12,
    'ষষ্ঠী': 6,
    'খেমটা': 6,
    'আড়াঠেকা': 8,
    'রূপকড়া': 7,
    'যৎ': 8,
    'ধামার': 14,
    'ঝুমরা': 14,
    'সুরফাঁকতাল': 10,
    'দাদরা/খেমটা': 6,  # both are 6-beat
    'নবতাল': 9,
    'একাদশী': 11,
    'ঝম্পক': 10,
}

df['beats'] = df['taal'].map(taal_beats)
mapped = df.dropna(subset=['beats'])
unmapped = df[df['beats'].isna()]

print(f"Songs with taal + year: {len(df)}")
print(f"Mapped to beat count: {len(mapped)} ({len(mapped)/len(df)*100:.1f}%)")
print(f"Unmapped taals: {unmapped['taal'].nunique()}")
print(f"  Top unmapped: {unmapped['taal'].value_counts().head(10).to_string()}")

# Map remaining taals by looking them up
# Add more mappings based on what's unmapped
extra_beats = {}
for taal in unmapped['taal'].value_counts().head(20).index:
    # Try to identify common patterns
    if 'দাদরা' in taal:
        extra_beats[taal] = 6
    elif 'কাহারবা' in taal:
        extra_beats[taal] = 8
    elif 'একতাল' in taal:
        extra_beats[taal] = 12
    elif 'ত্রিতাল' in taal:
        extra_beats[taal] = 16
    elif 'ঝাঁপতাল' in taal:
        extra_beats[taal] = 10

taal_beats.update(extra_beats)
df['beats'] = df['taal'].map(taal_beats)
mapped = df.dropna(subset=['beats'])
print(f"\nAfter extra mapping: {len(mapped)} songs ({len(mapped)/len(df)*100:.1f}%)")

# --- Analysis ---
print(f"\nBeat count distribution:")
print(mapped['beats'].value_counts().sort_index().to_string())

# Decade-level average beat count
mapped['decade'] = (mapped['gregorian_year'] // 10) * 10
decade_beats = mapped.groupby('decade').agg(
    n_songs=('beats', 'size'),
    mean_beats=('beats', 'mean'),
    median_beats=('beats', 'median'),
    pct_dadra=('taal', lambda x: (x == 'দাদরা').mean()),
    pct_kaharba=('taal', lambda x: (x == 'কাহারবা').mean()),
).reset_index()

print(f"\nBeat complexity by decade:")
for _, row in decade_beats.iterrows():
    print(f"  {int(row['decade'])}s: mean={row['mean_beats']:.2f}, median={row['median_beats']:.0f}, "
          f"n={int(row['n_songs'])}, Dadra={row['pct_dadra']:.0%}, Kaharba={row['pct_kaharba']:.0%}")

# Correlation: year vs beats
r, p = stats.spearmanr(mapped['gregorian_year'], mapped['beats'])
print(f"\nSpearman correlation (year vs beats): rho={r:.4f}, p={p:.2e}")

# Compare early vs late
early = mapped[mapped['gregorian_year'] <= 1905]
late = mapped[mapped['gregorian_year'] >= 1920]
u_stat, p_u = stats.mannwhitneyu(early['beats'], late['beats'], alternative='two-sided')
print(f"\nEarly (≤1905) vs Late (≥1920):")
print(f"  Early: mean={early['beats'].mean():.2f}, median={early['beats'].median():.0f}, n={len(early)}")
print(f"  Late:  mean={late['beats'].mean():.2f}, median={late['beats'].median():.0f}, n={len(late)}")
print(f"  Mann-Whitney U: p={p_u:.2e}")

# Proportion of "simple" taals (6 or 8 beats) over time
mapped['simple_taal'] = mapped['beats'] <= 8
decade_simple = mapped.groupby('decade')['simple_taal'].mean()
print(f"\nProportion of simple taals (≤8 beats) by decade:")
for dec, pct in decade_simple.items():
    print(f"  {int(dec)}s: {pct:.1%}")

r_simple, p_simple = stats.spearmanr(mapped['gregorian_year'], mapped['simple_taal'].astype(int))
print(f"\nSpearman (year vs simple taal): rho={r_simple:.4f}, p={p_simple:.2e}")

# Category and taal complexity interaction
print(f"\n--- Beat complexity by category ---")
cat_beats = mapped.groupby('category')['beats'].agg(['mean', 'median', 'size']).sort_values('size', ascending=False)
for cat, row in cat_beats.head(8).iterrows():
    print(f"  {cat}: mean={row['mean']:.2f}, median={row['median']:.0f}, n={int(row['size'])}")

print(f"\n=== RESULT ===")
print(f"hypothesis: Tagore shifted from complex taals to simpler ones over his career")
print(f"method: Mapped taals to beat counts (matras), tracked mean/median beats and simple taal proportion across decades")
direction = "toward simpler taals" if r < 0 else "toward more complex taals" if r > 0 else "no clear direction"
print(f"key_finding: Trend {direction} (Spearman rho={r:.4f}, p={p:.2e}). Simple taal (≤8 beats) proportion {'increased' if r_simple > 0 else 'decreased'} (rho={r_simple:.4f}, p={p_simple:.2e})")
print(f"statistical_significance: Year vs beats rho={r:.4f}, p={p:.2e}; year vs simple taal rho={r_simple:.4f}, p={p_simple:.2e}")
sig = "significant" if p < 0.05 else "not significant"
print(f"conclusion: The shift toward {'simpler' if r < 0 else 'more complex'} taals is {sig}. {'This supports the narrative of Tagore moving toward musical accessibility in his later years.' if r < 0 and p < 0.05 else 'Taal complexity remained relatively stable across Tagore career.'}")
print(f"=== END RESULT ===")
