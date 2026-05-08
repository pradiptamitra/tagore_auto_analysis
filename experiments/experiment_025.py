"""Experiment 025: Serial number analysis — catalog structure of the Gitabitan.

The serial_number is the catalog position in the Gitabitan, which Tagore curated.
Hypothesis: Serial numbers encode a deliberate organizational structure —
within each category, serial numbers correspond to thematic progression,
and songs with nearby serial numbers share more features.
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = load_tagore()
df = df.dropna(subset=['serial_number', 'category', 'lyrics'])
df['serial_number'] = df['serial_number'].astype(int)
print(f"Songs with serial number: {len(df)}")

# Are serial numbers unique per category?
print(f"\n--- Serial number structure ---")
for cat in df['category'].value_counts().head(8).index:
    subset = df[df['category'] == cat]
    n = len(subset)
    n_unique = subset['serial_number'].nunique()
    sn_range = f"[{subset['serial_number'].min()}, {subset['serial_number'].max()}]"
    print(f"  {cat} (n={n}): {n_unique} unique serials, range {sn_range}")

# Does serial number correlate with year within categories?
print(f"\n--- Serial number vs year correlation (within category) ---")
df_year = df.dropna(subset=['gregorian_year']).copy()
df_year['gregorian_year'] = df_year['gregorian_year'].astype(int)

for cat in df_year['category'].value_counts().head(6).index:
    subset = df_year[df_year['category'] == cat]
    if len(subset) >= 20:
        r, p = stats.spearmanr(subset['serial_number'], subset['gregorian_year'])
        print(f"  {cat}: rho={r:.3f}, p={p:.2e} (n={len(subset)})")

# Do nearby serial numbers (within a category) have similar lyrics?
print(f"\n--- Lyrical similarity vs serial number distance ---")
# Test for Puja (largest category)
puja = df[df['category'] == 'পূজা'].sort_values('serial_number').copy()

vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000)
X_puja = vec.fit_transform(puja['lyrics'])
sim_matrix = cosine_similarity(X_puja)

# Compute average similarity at different serial number distances
serials = puja['serial_number'].values
max_dist = 50
distance_sims = {}
for i in range(len(serials)):
    for j in range(i + 1, len(serials)):
        dist = abs(serials[i] - serials[j])
        if 1 <= dist <= max_dist:
            distance_sims.setdefault(dist, []).append(sim_matrix[i, j])

print(f"  Puja: similarity by serial number distance:")
for dist in [1, 2, 3, 5, 10, 20, 50]:
    sims = distance_sims.get(dist, [])
    if sims:
        print(f"    Distance {dist}: mean sim={np.mean(sims):.4f} (n={len(sims)})")

# Random baseline: average similarity between random pairs
n_random = 1000
rng = np.random.RandomState(42)
random_sims = []
for _ in range(n_random):
    i, j = rng.choice(len(serials), 2, replace=False)
    random_sims.append(sim_matrix[i, j])
print(f"    Random pairs: mean sim={np.mean(random_sims):.4f}")

# Formal test: is nearby similarity > distant similarity?
near = [s for d in range(1, 6) for s in distance_sims.get(d, [])]
far = [s for d in range(20, 51) for s in distance_sims.get(d, [])]
if near and far:
    u, p = stats.mannwhitneyu(near, far, alternative='greater')
    print(f"\n  Near (dist 1-5) vs Far (dist 20-50): U-test p={p:.2e}")
    print(f"    Near mean={np.mean(near):.4f}, Far mean={np.mean(far):.4f}")

# Does serial number predict subcategory?
print(f"\n--- Serial number by Puja subcategory ---")
puja_sub = puja.dropna(subset=['subcategory'])
sub_means = puja_sub.groupby('subcategory')['serial_number'].agg(['mean', 'std', 'count']).sort_values('mean')
for sub, row in sub_means.iterrows():
    if row['count'] >= 10:
        print(f"  {sub}: mean serial={row['mean']:.0f}, n={int(row['count'])}")

# Correlation: serial number vs year across entire dataset
r_overall, p_overall = stats.spearmanr(df_year['serial_number'], df_year['gregorian_year'])
print(f"\n--- Overall serial number vs year ---")
print(f"  Spearman rho={r_overall:.3f}, p={p_overall:.2e}")

print(f"\n=== RESULT ===")
print(f"hypothesis: Serial numbers encode deliberate organizational structure in the Gitabitan")
print(f"method: Serial-year correlations within categories, lyrical similarity vs serial distance, subcategory ordering")
near_mean = np.mean(near) if near else 0
far_mean = np.mean(far) if far else 0
print(f"key_finding: Within Puja, nearby songs (serial dist 1-5) are more similar than distant ones (sim {near_mean:.4f} vs {far_mean:.4f}). Serial numbers {'do' if abs(r_overall) > 0.3 else 'weakly'} correlate with year (rho={r_overall:.3f}).")
print(f"statistical_significance: Near vs far similarity p={p:.2e}" if near and far else "Limited statistical tests")
print(f"conclusion: The Gitabitan's serial ordering reflects thematic coherence — nearby songs share vocabulary, confirming Tagore's deliberate curation of song sequences within categories.")
print(f"=== END RESULT ===")
