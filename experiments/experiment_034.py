"""Experiment 034: Formal change-point detection in Tagore's style.

Multiple experiments found temporal shifts: Exp 010 (taal simplification),
Exp 019 (rhyme peaked 1900s), Exp 024 (inflection at 1913 Nobel),
Exp 027 (pronoun shift). But these used ad hoc decade bins.

Hypothesis: Formal change-point detection will identify specific years where
Tagore's style changed abruptly, and these will cluster around known
biographical events (1901 bereavements, 1913 Nobel, 1930s dance drama period).
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
df = df[(df['gregorian_year'] >= 1880) & (df['gregorian_year'] <= 1941)]

# Compute multiple style features per song
def compute_features(text):
    words = text.split()
    lines = text.split('\n')
    n_words = len(words)
    n_lines = max(len([l for l in lines if l.strip()]), 1)

    # Word length
    clean_words = [re.sub(r'[।॥,\.!?;:\-–—\d]', '', w) for w in words]
    clean_words = [w for w in clean_words if len(w) >= 1]
    avg_word_len = np.mean([len(w) for w in clean_words]) if clean_words else 0

    # Words per line
    words_per_line = n_words / n_lines

    # Unique word ratio (TTR proxy on first 50 words)
    first_50 = clean_words[:50]
    ttr_50 = len(set(first_50)) / max(len(first_50), 1)

    # Archaic form count (from Exp 030)
    archaic_forms = ['তব', 'মম', 'মোর', 'করিয়া', 'হইয়া', 'লইয়া', 'হেরি']
    archaic_count = sum(1 for w in clean_words if any(w == a or w.startswith(a) for a in archaic_forms))
    archaic_rate = archaic_count / max(n_words, 1)

    return pd.Series({
        'avg_word_len': avg_word_len,
        'words_per_line': words_per_line,
        'ttr_50': ttr_50,
        'archaic_rate': archaic_rate,
        'n_words': n_words,
    })

features = df['lyrics'].apply(compute_features)
df = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

# Change-point detection using PELT-like approach:
# For each candidate year, compute the difference in means before/after
# and assess with a two-sample test.

def find_changepoints(years, values, min_segment=30):
    """Find the best single change-point by maximizing log-likelihood ratio."""
    sorted_idx = np.argsort(years)
    y_sorted = years[sorted_idx]
    v_sorted = values[sorted_idx]

    n = len(v_sorted)
    overall_var = np.var(v_sorted)
    if overall_var == 0:
        return None

    best_score = -np.inf
    best_year = None

    unique_years = np.unique(y_sorted)

    for year in unique_years:
        left = v_sorted[y_sorted <= year]
        right = v_sorted[y_sorted > year]

        if len(left) < min_segment or len(right) < min_segment:
            continue

        # Log-likelihood ratio (assuming Gaussian)
        var_left = np.var(left) + 1e-10
        var_right = np.var(right) + 1e-10

        score = -(len(left) * np.log(var_left) + len(right) * np.log(var_right))
        score += n * np.log(overall_var + 1e-10)

        if score > best_score:
            best_score = score
            best_year = year

    if best_year is None:
        return None

    left = v_sorted[y_sorted <= best_year]
    right = v_sorted[y_sorted > best_year]
    u_stat, p_val = stats.mannwhitneyu(left, right, alternative='two-sided')

    return {
        'year': best_year,
        'score': best_score,
        'p_value': p_val,
        'left_mean': np.mean(left),
        'right_mean': np.mean(right),
        'left_n': len(left),
        'right_n': len(right),
        'effect_size': (np.mean(right) - np.mean(left)) / np.sqrt(overall_var + 1e-10),
    }

# Find two change-points by recursive splitting
def find_two_changepoints(years, values, min_segment=30):
    cp1 = find_changepoints(years, values, min_segment)
    if cp1 is None:
        return [None, None]

    # Split at first change-point, find second in larger segment
    mask_left = years <= cp1['year']
    mask_right = years > cp1['year']

    cp2_candidates = []
    if mask_left.sum() >= 2 * min_segment:
        cp_left = find_changepoints(years[mask_left], values[mask_left], min_segment)
        if cp_left:
            cp2_candidates.append(cp_left)
    if mask_right.sum() >= 2 * min_segment:
        cp_right = find_changepoints(years[mask_right], values[mask_right], min_segment)
        if cp_right:
            cp2_candidates.append(cp_right)

    cp2 = max(cp2_candidates, key=lambda x: x['score']) if cp2_candidates else None
    return [cp1, cp2]

# Run change-point detection on each feature
features_to_test = ['avg_word_len', 'words_per_line', 'ttr_50', 'archaic_rate', 'n_words']
feature_labels = ['Word length', 'Words/line', 'Lexical diversity (TTR-50)', 'Archaic form rate', 'Song length']

years = df['gregorian_year'].values

print(f"Songs analyzed: {len(df)}, Year range: {years.min()}-{years.max()}")
print(f"\n{'='*60}")
print(f"CHANGE-POINT DETECTION RESULTS")
print(f"{'='*60}")

all_changepoints = []
for feat, label in zip(features_to_test, feature_labels):
    values = df[feat].values
    cps = find_two_changepoints(years, values)

    print(f"\n--- {label} ({feat}) ---")
    for i, cp in enumerate(cps):
        if cp is None:
            continue
        direction = "↑" if cp['effect_size'] > 0 else "↓"
        sig = "***" if cp['p_value'] < 0.001 else "**" if cp['p_value'] < 0.01 else "*" if cp['p_value'] < 0.05 else "ns"
        print(f"  Change-point {i+1}: {int(cp['year'])} "
              f"({cp['left_mean']:.4f} → {cp['right_mean']:.4f}, {direction} d={abs(cp['effect_size']):.2f}, "
              f"p={cp['p_value']:.2e} {sig})")
        all_changepoints.append({
            'feature': label,
            'year': cp['year'],
            'p_value': cp['p_value'],
            'effect_size': cp['effect_size'],
        })

# Also test category-specific features
# Puja proportion per year
df_with_cat = df.dropna(subset=['category'])
year_counts = df_with_cat.groupby('gregorian_year').size()
puja_counts = df_with_cat[df_with_cat['category'] == 'পূজা'].groupby('gregorian_year').size()
year_puja_rate = (puja_counts / year_counts).dropna()
yr_vals = year_puja_rate.index.values
pr_vals = year_puja_rate.values

# Only years with ≥5 songs
mask = year_counts[year_counts >= 5].index
year_puja_filtered = year_puja_rate[year_puja_rate.index.isin(mask)]
if len(year_puja_filtered) >= 10:
    cp_puja = find_changepoints(year_puja_filtered.index.values, year_puja_filtered.values, min_segment=3)
    print(f"\n--- Puja proportion per year ---")
    if cp_puja:
        direction = "↑" if cp_puja['effect_size'] > 0 else "↓"
        sig = "***" if cp_puja['p_value'] < 0.001 else "**" if cp_puja['p_value'] < 0.01 else "*" if cp_puja['p_value'] < 0.05 else "ns"
        print(f"  Change-point: {int(cp_puja['year'])} "
              f"({cp_puja['left_mean']:.3f} → {cp_puja['right_mean']:.3f}, {direction} d={abs(cp_puja['effect_size']):.2f}, "
              f"p={cp_puja['p_value']:.2e} {sig})")

# Summary: cluster change-points
cp_df = pd.DataFrame(all_changepoints)
significant = cp_df[cp_df['p_value'] < 0.05]
print(f"\n{'='*60}")
print(f"SUMMARY: {len(significant)} significant change-points out of {len(cp_df)}")
print(f"{'='*60}")

if len(significant) > 0:
    print(f"\nSignificant change-point years:")
    for year in sorted(significant['year'].unique()):
        feats = significant[significant['year'] == year]['feature'].tolist()
        print(f"  {int(year)}: {', '.join(feats)}")

    # Do they cluster around known events?
    events = {
        1901: "Wife Mrinalini dies; son Rathindranath born",
        1902: "Daughter Renuka dies",
        1907: "Son Shamindranath dies",
        1913: "Nobel Prize for Gitanjali",
        1916: "First Japan/US tour",
        1924: "China tour, Visva-Bharati established",
        1930: "Dance drama period begins",
        1936: "Peak output year (Chitrangada, Shyama)",
    }

    print(f"\nBiographical events near change-points:")
    for year in sorted(significant['year'].unique()):
        near = [(y, e) for y, e in events.items() if abs(y - year) <= 3]
        if near:
            for y, e in near:
                print(f"  {int(year)} ≈ {y}: {e}")

# Permutation test for clustering
print(f"\n--- Permutation test: do change-points cluster more than chance? ---")
if len(significant) >= 2:
    observed_spread = significant['year'].std()
    n_cps = len(significant)
    year_range = (years.min(), years.max())

    n_perm = 10000
    random_spreads = []
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        random_years = rng.integers(year_range[0], year_range[1], size=n_cps)
        random_spreads.append(np.std(random_years))

    p_cluster = np.mean(np.array(random_spreads) <= observed_spread)
    print(f"  Observed spread (std): {observed_spread:.1f} years")
    print(f"  Expected random spread: {np.mean(random_spreads):.1f} years")
    print(f"  p(clustering ≤ observed) = {p_cluster:.4f}")

print(f"\n=== RESULT ===")
if len(significant) > 0:
    cp_years = sorted(significant['year'].unique())
    print(f"hypothesis: Style changes cluster around biographical events")
    print(f"method: Maximum likelihood single change-point detection with recursive splitting; Mann-Whitney significance test; permutation test for clustering")
    print(f"key_finding: {len(significant)} significant change-points at years {[int(y) for y in cp_years]}. Features: {', '.join(significant['feature'].unique())}.")
    print(f"statistical_significance: Individual change-points p<0.05; clustering test p={'<0.05 (clustered)' if p_cluster < 0.05 else '>0.05 (not significantly clustered)'}")
    print(f"conclusion: Tagore's style did not evolve smoothly — it shifted at identifiable moments, {'coinciding with' if any(abs(y - e) <= 3 for y in cp_years for e in events) else 'not clearly linked to'} major life events.")
else:
    print(f"hypothesis: Style changes cluster around biographical events")
    print(f"method: Maximum likelihood change-point detection")
    print(f"key_finding: No significant change-points detected")
    print(f"conclusion: Tagore's style evolved gradually without sharp transitions")
print(f"=== END RESULT ===")
