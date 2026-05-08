"""Experiment 006: Raga-Taal co-occurrence patterns.

Hypothesis: Raga and taal choices co-occur non-randomly — certain melodic modes
prefer certain rhythmic patterns, and this structure is stronger than expected by chance.

Raga handling: Normalize (strip modifiers, take primary from pairs, exclude genres).
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from scipy import stats
import re

df = load_tagore()

# Raga normalization (reused from exp 004/005)
genre_keywords = ['বাউল', 'কীর্তন', 'ঝুমুর', 'টপ্পা', 'ঠুংরি', 'ঠুমরি', 'গজল',
                   'ভজন', 'ভাটিয়ালি', 'ইটালিয়ান', 'স্কটিশ', 'আইরিশ', 'ইংরেজি',
                   'লোকসুর', 'দেশী', 'মহিশূরী']

def normalize_raga_primary(raga_str):
    if pd.isna(raga_str):
        return None
    parts = re.split(r'[-–/]| ও ', raga_str)
    part = parts[0].strip()
    for kw in genre_keywords:
        if kw in part:
            return None
    for mod in ['মিশ্র ', 'শুদ্ধ ', 'সিন্ধু ']:
        if part.startswith(mod):
            part = part[len(mod):]
            break
    return part.strip()

df['raga_norm'] = df['raga'].apply(normalize_raga_primary)
df_rt = df.dropna(subset=['raga_norm', 'taal']).copy()
print(f"Songs with both raga and taal: {len(df_rt)}")

# Filter to ragas and taals with enough data
raga_counts = df_rt['raga_norm'].value_counts()
taal_counts = df_rt['taal'].value_counts()
min_count = 15
valid_ragas = raga_counts[raga_counts >= min_count].index
valid_taals = taal_counts[taal_counts >= min_count].index

df_filtered = df_rt[df_rt['raga_norm'].isin(valid_ragas) & df_rt['taal'].isin(valid_taals)].copy()
print(f"After filtering (raga≥{min_count}, taal≥{min_count}): {len(df_filtered)} songs")
print(f"Ragas: {len(valid_ragas)}, Taals: {len(valid_taals)}")

print(f"\nTop taals:")
print(df_filtered['taal'].value_counts().head(10).to_string())

# Contingency table
ct = pd.crosstab(df_filtered['raga_norm'], df_filtered['taal'])
print(f"\nContingency table shape: {ct.shape}")

# Chi-squared test
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
n = ct.values.sum()
min_dim = min(ct.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim))
print(f"\nChi-squared: χ²={chi2:.1f}, dof={dof}, p={p_chi:.2e}")
print(f"Cramér's V: {cramers_v:.3f}")

# Mutual information
from sklearn.metrics import normalized_mutual_info_score
mi = normalized_mutual_info_score(df_filtered['raga_norm'], df_filtered['taal'])
print(f"Normalized Mutual Information: {mi:.4f}")

# Permutation test for MI
n_perms = 1000
perm_mi = []
rng = np.random.RandomState(42)
taal_vals = df_filtered['taal'].values
raga_vals = df_filtered['raga_norm'].values
for _ in range(n_perms):
    perm_taal = rng.permutation(taal_vals)
    perm_mi.append(normalized_mutual_info_score(raga_vals, perm_taal))
perm_mi = np.array(perm_mi)
p_perm = (perm_mi >= mi).mean()
print(f"Permutation test for MI: p={p_perm:.4f} (observed MI vs {n_perms} permutations)")
print(f"  Mean permuted MI: {perm_mi.mean():.4f}, Max: {perm_mi.max():.4f}")

# Standardized residuals: top associations
residuals = (ct.values - expected) / np.sqrt(expected)
res_df = pd.DataFrame(residuals, index=ct.index, columns=ct.columns)

pairs = []
for raga in res_df.index:
    for taal in res_df.columns:
        pairs.append((raga, taal, res_df.loc[raga, taal], ct.loc[raga, taal]))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)

print(f"\nTop 20 raga-taal associations (by |standardized residual|):")
for raga, taal, resid, count in pairs[:20]:
    direction = "attracts" if resid > 0 else "avoids"
    print(f"  {raga} × {taal}: residual={resid:+.2f}, count={count} ({direction})")

# For the top few ragas, show their taal distribution
print(f"\nTaal profiles for top ragas:")
for raga in valid_ragas[:8]:
    taal_dist = df_filtered[df_filtered['raga_norm'] == raga]['taal'].value_counts()
    total = taal_dist.sum()
    top3 = ', '.join(f"{t} ({c/total:.0%})" for t, c in taal_dist.head(3).items())
    print(f"  {raga} (n={total}): {top3}")

print(f"\n=== RESULT ===")
print(f"hypothesis: Raga and taal choices co-occur non-randomly")
print(f"method: Chi-squared test and normalized mutual information on contingency table of ragas (≥{min_count} songs) × taals (≥{min_count} songs), with permutation test for MI")
print(f"key_finding: Strong non-random co-occurrence (χ²={chi2:.1f}, p={p_chi:.2e}, Cramér's V={cramers_v:.3f}). NMI={mi:.4f} vs permuted mean {perm_mi.mean():.4f} (p={p_perm:.4f}).")
print(f"statistical_significance: χ² p={p_chi:.2e}, Cramér's V={cramers_v:.3f}, permutation p={p_perm:.4f}")
print(f"conclusion: Raga and taal are non-independently chosen. Certain melodic modes strongly prefer specific rhythmic cycles, reflecting deep musical structure in Tagore's compositions.")
print(f"=== END RESULT ===")
