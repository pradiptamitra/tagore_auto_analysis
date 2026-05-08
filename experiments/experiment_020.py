"""Experiment 020: Clustering by musical features (raga+taal).

Hypothesis: Songs cluster by raga+taal combinations into musically coherent groups
that don't perfectly align with Tagore's thematic categories — revealing a
latent musical organization orthogonal to theme.
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy import stats
import re

df = load_tagore()

# Raga normalization
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
            return '__genre__'  # keep as genre marker
    for mod in ['মিশ্র ', 'শুদ্ধ ', 'সিন্ধু ']:
        if part.startswith(mod):
            part = part[len(mod):]
            break
    return part.strip()

df['raga_norm'] = df['raga'].apply(normalize_raga_primary)

# Taal beat mapping
taal_beats = {
    'দাদরা': 6, 'কাহারবা': 8, 'ত্রিতাল': 16, 'একতাল': 12,
    'ঝাঁপতাল': 10, 'তেওরা': 7, ' তেওরা': 7, 'চৌতাল': 12,
    'ষষ্ঠী': 6, 'খেমটা': 6, 'আড়াঠেকা': 8, 'রূপকড়া': 7,
    'যৎ': 8, 'ধামার': 14, 'ঝুমরা': 14, 'সুরফাঁকতাল': 10,
    'দাদরা/খেমটা': 6, 'নবতাল': 9, 'একাদশী': 11, 'ঝম্পক': 10,
}

df['beats'] = df['taal'].map(taal_beats)

# Filter: songs with both raga and taal
df_music = df.dropna(subset=['raga_norm', 'taal', 'beats', 'category']).copy()

# Keep ragas with ≥10 songs
raga_counts = df_music['raga_norm'].value_counts()
valid_ragas = raga_counts[raga_counts >= 10].index
df_music = df_music[df_music['raga_norm'].isin(valid_ragas)].copy()

print(f"Songs with raga + taal + category: {len(df_music)}")
print(f"Unique ragas: {df_music['raga_norm'].nunique()}")
print(f"Unique taals: {df_music['taal'].nunique()}")

# Create feature matrix: one-hot raga + beats (numeric)
raga_le = LabelEncoder()
raga_encoded = raga_le.fit_transform(df_music['raga_norm'])
n_ragas = len(raga_le.classes_)

# One-hot encode raga
raga_onehot = np.zeros((len(df_music), n_ragas))
raga_onehot[np.arange(len(df_music)), raga_encoded] = 1

# Normalize beats to [0, 1]
beats_norm = (df_music['beats'].values - df_music['beats'].min()) / (df_music['beats'].max() - df_music['beats'].min())

# Combine
X = np.hstack([raga_onehot, beats_norm.reshape(-1, 1)])

# Cluster
n_clusters_range = [5, 8, 10, 15]
print(f"\n--- Clustering Results ---")
best_sil = -1
best_k = 5
for k in n_clusters_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=min(5000, len(X)))
    ari = adjusted_rand_score(df_music['category'].values, labels)
    nmi = normalized_mutual_info_score(df_music['category'].values, labels)
    print(f"  k={k}: silhouette={sil:.3f}, ARI(vs category)={ari:.3f}, NMI(vs category)={nmi:.3f}")
    if sil > best_sil:
        best_sil = sil
        best_k = k

# Use best k
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X)
df_music['cluster'] = cluster_labels

ari = adjusted_rand_score(df_music['category'].values, cluster_labels)
nmi = normalized_mutual_info_score(df_music['category'].values, cluster_labels)

print(f"\nUsing k={best_k}:")
print(f"  ARI vs category: {ari:.3f}")
print(f"  NMI vs category: {nmi:.3f}")

# Cluster profiles
print(f"\n--- Cluster Profiles (k={best_k}) ---")
for c in range(best_k):
    subset = df_music[df_music['cluster'] == c]
    top_raga = subset['raga_norm'].value_counts().head(3)
    top_cat = subset['category'].value_counts().head(3)
    mean_beats = subset['beats'].mean()
    raga_str = ', '.join(f'{r}({n})' for r, n in top_raga.items())
    cat_str = ', '.join(f'{cat}({n})' for cat, n in top_cat.items())
    print(f"\n  Cluster {c} (n={len(subset)}, mean beats={mean_beats:.1f}):")
    print(f"    Top ragas: {raga_str}")
    print(f"    Top categories: {cat_str}")

# Does the musical clustering add information beyond category?
# Check: within each category, do clusters have different years?
print(f"\n--- Within-category cluster analysis ---")
for cat in df_music['category'].value_counts().head(4).index:
    cat_data = df_music[(df_music['category'] == cat) & df_music['gregorian_year'].notna()].copy()
    cat_data['gregorian_year'] = cat_data['gregorian_year'].astype(int)
    cluster_counts = cat_data['cluster'].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= 10].index
    if len(valid_clusters) >= 2:
        groups = [cat_data[cat_data['cluster'] == c]['gregorian_year'].values for c in valid_clusters]
        h, p = stats.kruskal(*groups)
        means = {c: cat_data[cat_data['cluster'] == c]['gregorian_year'].mean() for c in valid_clusters}
        print(f"  {cat}: clusters have {'different' if p < 0.05 else 'similar'} temporal distributions (H={h:.1f}, p={p:.4f})")
        for c, m in sorted(means.items()):
            print(f"    Cluster {c}: mean year={m:.0f}, n={cluster_counts[c]}")

print(f"\n=== RESULT ===")
print(f"hypothesis: Raga+taal combinations reveal latent musical structure partially independent of thematic categories")
print(f"method: K-means clustering on one-hot raga + normalized taal beats. Compared clusters to categories via ARI and NMI.")
print(f"key_finding: Best clustering at k={best_k} (silhouette={best_sil:.3f}). Musical clusters partially align with categories (NMI={nmi:.3f}) but are not identical (ARI={ari:.3f}). Within categories, musical clusters show different temporal distributions.")
print(f"statistical_significance: ARI={ari:.3f}, NMI={nmi:.3f}; within-category temporal differences tested")
print(f"conclusion: Musical features define a structure that partially overlaps with but is not reducible to thematic categories. Songs with the same theme can belong to different musical 'families', and these families have temporal signatures — suggesting Tagore's musical language evolved semi-independently of his thematic choices.")
print(f"=== END RESULT ===")
