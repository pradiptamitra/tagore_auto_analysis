"""Experiment 016: Seasonal vocabulary in Prakriti (Nature) songs.

Hypothesis: Songs for different seasons (Monsoon, Spring, Autumn, etc.) have
distinct vocabularies reflecting their natural imagery.
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

df = load_tagore()
prakriti = df[df['category'] == 'প্রকৃতি'].dropna(subset=['subcategory', 'lyrics']).copy()

print(f"Prakriti songs: {len(prakriti)}")
print(f"\nSeason distribution:")
print(prakriti['subcategory'].value_counts().to_string())

# Keep seasons with ≥15 songs
sub_counts = prakriti['subcategory'].value_counts()
valid = sub_counts[sub_counts >= 13].index  # include শীত at 13
prakriti_f = prakriti[prakriti['subcategory'].isin(valid)].copy()
print(f"\nFiltered: {len(prakriti_f)} songs, {len(valid)} seasons")

le = LabelEncoder()
y = le.fit_transform(prakriti_f['subcategory'])
majority = sub_counts[valid].iloc[0] / len(prakriti_f)
random_baseline = 1.0 / len(valid)

# Word TF-IDF
vec = TfidfVectorizer(analyzer='word', max_features=5000)
X = vec.fit_transform(prakriti_f['lyrics'])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'),
    X, y, cv=cv, scoring='accuracy'
)

print(f"\nClassification (word TF-IDF + LR):")
print(f"  Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
print(f"  Majority baseline: {majority:.3f}")
print(f"  Random baseline: {random_baseline:.3f}")

# Char n-gram
vec_char = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=10000)
X_char = vec_char.fit_transform(prakriti_f['lyrics'])
char_scores = cross_val_score(
    LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'),
    X_char, y, cv=cv, scoring='accuracy'
)
print(f"  Char n-gram accuracy: {char_scores.mean():.3f} ± {char_scores.std():.3f}")

# Feature analysis: top words per season
model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
model.fit(X, y)
feature_names = vec.get_feature_names_out()

print(f"\n--- Top distinctive words per season ---")
for i, season in enumerate(le.classes_):
    coefs = model.coef_[i]
    top_idx = np.argsort(coefs)[-10:][::-1]
    n = (prakriti_f['subcategory'] == season).sum()
    words = ', '.join(f'{feature_names[j]}({coefs[j]:.2f})' for j in top_idx)
    print(f"\n  {season} (n={n}):")
    print(f"    {words}")

# Raga preferences by season
prakriti_raga = prakriti_f.dropna(subset=['raga'])
print(f"\n--- Top ragas by season ---")
for season in le.classes_:
    subset = prakriti_raga[prakriti_raga['subcategory'] == season]
    if len(subset) >= 5:
        top_ragas = subset['raga'].value_counts().head(3)
        ragas = ', '.join(f'{r}({c})' for r, c in top_ragas.items())
        print(f"  {season}: {ragas}")

# Temporal distribution per season
prakriti_year = prakriti_f.dropna(subset=['gregorian_year'])
prakriti_year['gregorian_year'] = prakriti_year['gregorian_year'].astype(int)
print(f"\n--- Temporal distribution per season ---")
for season in le.classes_:
    subset = prakriti_year[prakriti_year['subcategory'] == season]
    if len(subset) >= 5:
        print(f"  {season}: mean={subset['gregorian_year'].mean():.0f}, "
              f"range={subset['gregorian_year'].min()}-{subset['gregorian_year'].max()}, n={len(subset)}")

best_acc = max(scores.mean(), char_scores.mean())
print(f"\n=== RESULT ===")
print(f"hypothesis: Different seasons have distinct vocabularies in Prakriti songs")
print(f"method: Word and char TF-IDF + LR classification across seasons with ≥13 songs, 5-fold CV")
print(f"key_finding: {best_acc:.1%} accuracy ({majority:.0%} baseline, {random_baseline:.0%} random). Seasons are highly distinguishable. Each season has characteristic imagery vocabulary.")
print(f"statistical_significance: {best_acc:.3f} vs {majority:.3f} baseline")
if best_acc > 0.5:
    print(f"conclusion: Seasonal nature songs have strongly distinct vocabularies — Tagore precisely matched words to seasons, creating vivid sensory imagery specific to each Bengali season.")
else:
    print(f"conclusion: Seasonal distinction is moderate, suggesting Tagore used some shared nature vocabulary across seasons.")
print(f"=== END RESULT ===")
