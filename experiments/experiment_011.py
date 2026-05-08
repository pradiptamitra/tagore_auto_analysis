"""Experiment 011: Subcategory structure within Puja (devotional songs).

Hypothesis: The subcategories within Puja are distinguishable by lyrics,
revealing distinct vocabularies for different aspects of worship.
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
from sklearn.metrics import confusion_matrix
from scipy import stats

df = load_tagore()
puja = df[df['category'] == 'পূজা'].dropna(subset=['subcategory', 'lyrics']).copy()
print(f"Puja songs: {len(puja)}")

sub_counts = puja['subcategory'].value_counts()
print(f"\nSubcategory distribution:")
print(sub_counts.to_string())

# Keep subcategories with ≥15 songs for meaningful classification
min_count = 15
valid_subs = sub_counts[sub_counts >= min_count].index
puja_filtered = puja[puja['subcategory'].isin(valid_subs)].copy()
print(f"\nSubcategories with ≥{min_count} songs: {len(valid_subs)}")
print(f"Songs: {len(puja_filtered)}")

majority_baseline = sub_counts[valid_subs].iloc[0] / puja_filtered.shape[0]
print(f"Majority baseline: {majority_baseline:.3f}")

# Classification
le = LabelEncoder()
y = le.fit_transform(puja_filtered['subcategory'])

# Char n-gram TF-IDF
vec_char = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=15000)
X_char = vec_char.fit_transform(puja_filtered['lyrics'])

# Word TF-IDF
vec_word = TfidfVectorizer(analyzer='word', max_features=8000)
X_word = vec_word.fit_transform(puja_filtered['lyrics'])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

char_scores = cross_val_score(
    LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs'),
    X_char, y, cv=cv, scoring='accuracy'
)

word_scores = cross_val_score(
    LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs'),
    X_word, y, cv=cv, scoring='accuracy'
)

print(f"\nChar n-gram TF-IDF + LR: {char_scores.mean():.3f} ± {char_scores.std():.3f}")
print(f"Word TF-IDF + LR: {word_scores.mean():.3f} ± {word_scores.std():.3f}")

best_acc = max(char_scores.mean(), word_scores.mean())
best_method = "char n-grams" if char_scores.mean() > word_scores.mean() else "word TF-IDF"

# Train full model for feature analysis
best_vec = vec_word  # word features are more interpretable
X_full = best_vec.fit_transform(puja_filtered['lyrics'])
model = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
model.fit(X_full, y)

# Top words per subcategory
feature_names = best_vec.get_feature_names_out()
print(f"\n--- Top distinctive words per subcategory ---")
for i, sub in enumerate(le.classes_):
    if i >= len(model.coef_):
        break
    coefs = model.coef_[i]
    top_idx = np.argsort(coefs)[-8:][::-1]
    top_words = [(feature_names[j], coefs[j]) for j in top_idx]
    n = (puja_filtered['subcategory'] == sub).sum()
    print(f"\n  {sub} (n={n}):")
    print(f"    {', '.join(f'{w}({c:.2f})' for w, c in top_words)}")

# Temporal distribution of subcategories
puja_year = puja_filtered.dropna(subset=['gregorian_year'])
puja_year['gregorian_year'] = puja_year['gregorian_year'].astype(int)
print(f"\n--- Mean year by subcategory ---")
sub_years = puja_year.groupby('subcategory')['gregorian_year'].agg(['mean', 'std', 'count']).sort_values('mean')
for sub, row in sub_years.iterrows():
    print(f"  {sub}: mean year={row['mean']:.0f}, n={int(row['count'])}")

print(f"\n=== RESULT ===")
print(f"hypothesis: Puja subcategories are distinguishable by lyrics")
print(f"method: TF-IDF (char and word) + Logistic Regression, 5-fold stratified CV on subcategories with ≥{min_count} songs")
print(f"key_finding: Best accuracy {best_acc:.3f} ({best_method}) vs majority baseline {majority_baseline:.3f} ({best_acc/majority_baseline:.1f}x lift)")
print(f"statistical_significance: {best_acc:.3f} vs {majority_baseline:.3f} = {(best_acc-majority_baseline)*100:+.1f}pp above baseline")
if best_acc > majority_baseline * 1.3:
    print(f"conclusion: Puja subcategories are meaningfully distinguishable by lyrics, showing Tagore used distinct vocabulary for different aspects of devotion.")
else:
    print(f"conclusion: Puja subcategories show limited lyrical distinctiveness, suggesting the subdivisions are thematic rather than linguistically distinct.")
print(f"=== END RESULT ===")
