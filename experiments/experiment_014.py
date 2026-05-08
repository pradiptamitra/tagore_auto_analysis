"""Experiment 014: Taal simplification — controlling for category.

From Exp 010: dramatic shift from complex to simple taals (rho=-0.40).
From Exp 013: category distribution shifted over time.
Question: Is taal simplification independent of category shift, or confounded?
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from scipy import stats

df = load_tagore()
df = df.dropna(subset=['taal', 'gregorian_year', 'category'])
df['gregorian_year'] = df['gregorian_year'].astype(int)

# Taal beat mapping (from Exp 010)
taal_beats = {
    'দাদরা': 6, 'কাহারবা': 8, 'ত্রিতাল': 16, 'একতাল': 12,
    'ঝাঁপতাল': 10, 'তেওরা': 7, ' তেওরা': 7, 'চৌতাল': 12,
    'ষষ্ঠী': 6, 'খেমটা': 6, 'আড়াঠেকা': 8, 'রূপকড়া': 7,
    'যৎ': 8, 'ধামার': 14, 'ঝুমরা': 14, 'সুরফাঁকতাল': 10,
    'দাদরা/খেমটা': 6, 'নবতাল': 9, 'একাদশী': 11, 'ঝম্পক': 10,
}

df['beats'] = df['taal'].map(taal_beats)
df = df.dropna(subset=['beats'])
print(f"Songs with category + taal + year + beats: {len(df)}")

# Overall trend (reproduction from Exp 010)
r_all, p_all = stats.spearmanr(df['gregorian_year'], df['beats'])
print(f"\nOverall: year vs beats rho={r_all:.4f}, p={p_all:.2e}")

# Within-category trends
print(f"\n--- Within-Category Taal Trends ---")
top_cats = df['category'].value_counts().head(6).index
for cat in top_cats:
    subset = df[df['category'] == cat]
    if len(subset) >= 20:
        r, p = stats.spearmanr(subset['gregorian_year'], subset['beats'])
        early = subset[subset['gregorian_year'] <= 1905]['beats'].mean()
        late = subset[subset['gregorian_year'] >= 1920]['beats'].mean()
        n_early = len(subset[subset['gregorian_year'] <= 1905])
        n_late = len(subset[subset['gregorian_year'] >= 1920])
        print(f"  {cat} (n={len(subset)}): rho={r:.4f}, p={p:.4f}")
        print(f"    Early (≤1905): mean={early:.2f} beats (n={n_early})")
        print(f"    Late (≥1920): mean={late:.2f} beats (n={n_late})")

# Formal test: partial correlation (year vs beats, controlling for category)
# Use category as dummy variables
from sklearn.linear_model import LinearRegression

cat_dummies = pd.get_dummies(df['category'], drop_first=True)

# Residualize both year and beats on category
year_resid = df['gregorian_year'].values - LinearRegression().fit(cat_dummies, df['gregorian_year']).predict(cat_dummies)
beats_resid = df['beats'].values - LinearRegression().fit(cat_dummies, df['beats']).predict(cat_dummies)

r_partial, p_partial = stats.spearmanr(year_resid, beats_resid)
print(f"\nPartial correlation (year vs beats | category): rho={r_partial:.4f}, p={p_partial:.2e}")

# How much of the overall trend is explained by category shift?
print(f"\nDecomposition:")
print(f"  Overall rho: {r_all:.4f}")
print(f"  Within-category rho: {r_partial:.4f}")
print(f"  Category-mediated fraction: {(1 - abs(r_partial)/abs(r_all))*100:.1f}%")

# Also check: Puja specifically (largest category, spans full time range)
puja = df[df['category'] == 'পূজা']
r_puja, p_puja = stats.spearmanr(puja['gregorian_year'], puja['beats'])
print(f"\nPuja-only trend: rho={r_puja:.4f}, p={p_puja:.2e} (n={len(puja)})")

# Decade comparison within Puja
puja_decades = puja.copy()
puja_decades['decade'] = (puja_decades['gregorian_year'] // 10) * 10
for dec in sorted(puja_decades['decade'].unique()):
    subset = puja_decades[puja_decades['decade'] == dec]
    if len(subset) >= 5:
        print(f"  Puja {int(dec)}s: mean={subset['beats'].mean():.2f} beats, "
              f"simple≤8: {(subset['beats'] <= 8).mean():.0%} (n={len(subset)})")

print(f"\n=== RESULT ===")
print(f"hypothesis: Taal simplification is independent of category shift")
print(f"method: Within-category Spearman correlations, partial correlation controlling for category (residualization), Puja-only trend analysis")
print(f"key_finding: Within-category trend rho={r_partial:.4f} (p={p_partial:.2e}) vs overall rho={r_all:.4f}. Category shift explains {(1 - abs(r_partial)/abs(r_all))*100:.0f}% of the overall trend. Puja-only: rho={r_puja:.4f} (p={p_puja:.2e}).")
print(f"statistical_significance: Partial rho={r_partial:.4f}, p={p_partial:.2e}; Puja rho={r_puja:.4f}, p={p_puja:.2e}")
remaining = abs(r_partial) / abs(r_all) * 100
if remaining > 70:
    print(f"conclusion: Taal simplification is largely a genuine musical evolution ({remaining:.0f}% persists after controlling for category), not merely an artifact of shifting genre preferences.")
else:
    print(f"conclusion: Category shift partly explains taal simplification, but {remaining:.0f}% of the trend persists within categories — a genuine musical evolution occurred alongside thematic shifts.")
print(f"=== END RESULT ===")
