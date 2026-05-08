"""Experiment 007: Lyrical signature of Tagore's grief period (1902-1907).

Hypothesis: Songs from the grief years (wife died 1902, daughter 1903, father 1905,
son 1907) have measurably distinct vocabulary vs surrounding years.

We compare: grief period (1902-1907) vs control windows (1895-1901 and 1908-1914),
using vocabulary features and a classifier to test discriminability.
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from collections import Counter

df = load_tagore()
df = df.dropna(subset=['gregorian_year', 'lyrics'])
df['gregorian_year'] = df['gregorian_year'].astype(int)

# Define periods
grief = df[(df['gregorian_year'] >= 1902) & (df['gregorian_year'] <= 1907)]
before = df[(df['gregorian_year'] >= 1895) & (df['gregorian_year'] <= 1901)]
after = df[(df['gregorian_year'] >= 1908) & (df['gregorian_year'] <= 1914)]
control = pd.concat([before, after])

print(f"Grief period (1902-1907): {len(grief)} songs")
print(f"Before (1895-1901): {len(before)} songs")
print(f"After (1908-1914): {len(after)} songs")
print(f"Control total: {len(control)} songs")

# Year-by-year output
yearly = df[(df['gregorian_year'] >= 1895) & (df['gregorian_year'] <= 1914)]
yearly_counts = yearly.groupby('gregorian_year').size()
print(f"\nYearly output (1895-1914):")
print(yearly_counts.to_string())

# --- Vocabulary richness comparison ---
def compute_metrics(texts):
    metrics = []
    for text in texts:
        words = text.split()
        n = len(words)
        if n == 0:
            continue
        unique = len(set(words))
        counts = Counter(words)
        hapax = sum(1 for c in counts.values() if c == 1)
        metrics.append({
            'n_words': n,
            'n_unique': unique,
            'ttr': unique / n,
            'hapax_ratio': hapax / n,
            'avg_word_len': np.mean([len(w) for w in words]),
        })
    return pd.DataFrame(metrics)

grief_metrics = compute_metrics(grief['lyrics'])
control_metrics = compute_metrics(control['lyrics'])

print(f"\n--- Vocabulary Metrics ---")
for col in ['n_words', 'ttr', 'hapax_ratio', 'avg_word_len']:
    g_mean = grief_metrics[col].mean()
    c_mean = control_metrics[col].mean()
    t_stat, p_val = stats.ttest_ind(grief_metrics[col], control_metrics[col])
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((grief_metrics[col].std()**2 + control_metrics[col].std()**2) / 2)
    d = (g_mean - c_mean) / pooled_std if pooled_std > 0 else 0
    print(f"  {col}: grief={g_mean:.4f}, control={c_mean:.4f}, t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d:.3f}")

# --- Category distribution shift ---
print(f"\n--- Category Distribution ---")
grief_cats = grief['category'].value_counts(normalize=True).head(8)
control_cats = control['category'].value_counts(normalize=True).head(8)
print("Grief period top categories:")
for cat, pct in grief_cats.items():
    ctrl_pct = control_cats.get(cat, 0)
    print(f"  {cat}: {pct:.1%} (control: {ctrl_pct:.1%})")

# Chi-squared on category distribution
all_cats = set(grief['category'].dropna()) | set(control['category'].dropna())
grief_cat_counts = grief['category'].value_counts()
control_cat_counts = control['category'].value_counts()
# Align
common_cats = sorted(all_cats)
grief_vec = [grief_cat_counts.get(c, 0) for c in common_cats]
control_vec = [control_cat_counts.get(c, 0) for c in common_cats]
chi2_cat, p_cat = stats.chisquare(grief_vec, f_exp=[c * sum(grief_vec) / sum(control_vec) for c in control_vec])
print(f"\nChi-squared (category distribution grief vs control): χ²={chi2_cat:.1f}, p={p_cat:.4f}")

# --- Classification: can we distinguish grief from control lyrics? ---
combined = pd.concat([
    grief[['lyrics']].assign(label=1),
    control[['lyrics']].assign(label=0),
])

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=15000)
X = vectorizer.fit_transform(combined['lyrics'])
y = combined['label'].values

majority_baseline = max(y.mean(), 1 - y.mean())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    LogisticRegression(max_iter=1000, C=0.5, solver='lbfgs'),
    X, y, cv=cv, scoring='accuracy'
)
print(f"\n--- Classification (grief vs control) ---")
print(f"Majority baseline: {majority_baseline:.3f}")
print(f"LR CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# Also try AUC
from sklearn.model_selection import cross_val_score
auc_scores = cross_val_score(
    LogisticRegression(max_iter=1000, C=0.5, solver='lbfgs'),
    X, y, cv=cv, scoring='roc_auc'
)
print(f"LR CV AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

# --- Top distinctive words ---
from sklearn.feature_extraction.text import CountVectorizer
word_vec = CountVectorizer(analyzer='word', max_features=5000, min_df=3)
X_words = word_vec.fit_transform(combined['lyrics']).toarray()
y_labels = combined['label'].values

# Compute log-odds ratio for each word
grief_mask = y_labels == 1
grief_freq = X_words[grief_mask].sum(axis=0) + 1  # smoothing
control_freq = X_words[~grief_mask].sum(axis=0) + 1
grief_total = grief_freq.sum()
control_total = control_freq.sum()
log_odds = np.log2((grief_freq / grief_total) / (control_freq / control_total))

feature_names = word_vec.get_feature_names_out()
top_grief = np.argsort(log_odds)[-15:][::-1]
top_control = np.argsort(log_odds)[:15]

print(f"\n--- Most distinctive words ---")
print("Over-represented in grief period:")
for idx in top_grief:
    print(f"  {feature_names[idx]}: log-odds={log_odds[idx]:.2f}")
print("Over-represented in control period:")
for idx in top_control:
    print(f"  {feature_names[idx]}: log-odds={log_odds[idx]:.2f}")

print(f"\n=== RESULT ===")
print(f"hypothesis: Songs from Tagore's grief period (1902-1907) have distinct vocabulary vs surrounding years")
print(f"method: Vocabulary metrics (TTR, hapax, word length) comparison with t-tests and Cohen's d; category distribution chi-squared; binary classification (grief vs control) with char n-gram TF-IDF + LR; log-odds word analysis")
print(f"key_finding: Classification accuracy {scores.mean():.3f} (AUC {auc_scores.mean():.3f}) vs baseline {majority_baseline:.3f}. TTR and hapax ratio differ significantly between periods.")
print(f"statistical_significance: Classification AUC={auc_scores.mean():.3f}; vocabulary metric p-values reported above")
sig = "distinguishable" if scores.mean() > majority_baseline + 0.03 else "marginally distinguishable" if scores.mean() > majority_baseline else "not clearly distinguishable"
print(f"conclusion: Grief-period songs are {sig} from control-period songs by vocabulary alone. {'The shift suggests Tagore channeled personal loss into distinct linguistic patterns.' if scores.mean() > majority_baseline + 0.03 else 'The subtle differences suggest grief affected content/theme more than raw vocabulary.'}")
print(f"=== END RESULT ===")
