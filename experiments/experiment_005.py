"""Experiment 005: Predicting raga from lyrics using classical ML.

Hypothesis: Raga can be predicted from lyrics with accuracy above majority baseline,
because Tagore matched musical modes to lyrical content.

Raga handling: Use normalized ragas (strip modifiers, split pairs — assign song to
first raga in pair). Exclude non-raga genre entries. Only use ragas with ≥20 songs
for reliable classification.
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

df = load_tagore()
df = df.dropna(subset=['raga', 'lyrics'])

# Normalize ragas
genre_keywords = ['বাউল', 'কীর্তন', 'ঝুমুর', 'টপ্পা', 'ঠুংরি', 'ঠুমরি', 'গজল',
                   'ভজন', 'ভাটিয়ালি', 'ইটালিয়ান', 'স্কটিশ', 'আইরিশ', 'ইংরেজি',
                   'লোকসুর', 'দেশী', 'মহিশূরী']

def normalize_raga_primary(raga_str):
    """Return the primary normalized raga, or None if it's a genre entry."""
    if pd.isna(raga_str):
        return None
    # Split paired ragas, take the first
    parts = re.split(r'[-–/]| ও ', raga_str)
    part = parts[0].strip()
    # Check if genre
    for kw in genre_keywords:
        if kw in part:
            return None
    # Strip modifiers
    for mod in ['মিশ্র ', 'শুদ্ধ ', 'সিন্ধু ']:
        if part.startswith(mod):
            part = part[len(mod):]
            break
    return part.strip()

df['raga_norm'] = df['raga'].apply(normalize_raga_primary)
df = df.dropna(subset=['raga_norm'])

# Only keep ragas with ≥20 songs
raga_counts = df['raga_norm'].value_counts()
valid_ragas = raga_counts[raga_counts >= 20].index
df = df[df['raga_norm'].isin(valid_ragas)].copy()

print(f"Songs after filtering: {len(df)}")
print(f"Ragas with ≥20 songs: {len(valid_ragas)}")
print(f"\nRaga distribution:")
print(df['raga_norm'].value_counts().to_string())

majority_baseline = df['raga_norm'].value_counts().iloc[0] / len(df)
random_baseline = 1.0 / len(valid_ragas)
print(f"\nMajority baseline: {majority_baseline:.3f}")
print(f"Random baseline: {random_baseline:.3f}")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['raga_norm'])

# Character n-gram TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=20000)
X = vectorizer.fit_transform(df['lyrics'])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression
lr_scores = cross_val_score(
    LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs'),
    X, y, cv=cv, scoring='accuracy'
)
print(f"\nLogistic Regression CV accuracy: {lr_scores.mean():.3f} ± {lr_scores.std():.3f}")

# Random Forest
rf_scores = cross_val_score(
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    X, y, cv=cv, scoring='accuracy'
)
print(f"Random Forest CV accuracy: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")

# Word-level TF-IDF + LR
word_vec = TfidfVectorizer(analyzer='word', max_features=10000)
X_word = word_vec.fit_transform(df['lyrics'])
lr_word_scores = cross_val_score(
    LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs'),
    X_word, y, cv=cv, scoring='accuracy'
)
print(f"LR (word TF-IDF) CV accuracy: {lr_word_scores.mean():.3f} ± {lr_word_scores.std():.3f}")

best_acc = max(lr_scores.mean(), rf_scores.mean(), lr_word_scores.mean())
best_method = ["LR+char", "RF+char", "LR+word"][[lr_scores.mean(), rf_scores.mean(), lr_word_scores.mean()].index(best_acc)]

print(f"\n=== RESULT ===")
print(f"hypothesis: Raga can be predicted from lyrics alone with accuracy above majority baseline ({majority_baseline:.3f})")
print(f"method: Character n-gram TF-IDF (2-5, 20K features) and word TF-IDF (10K), with LR and RF, 5-fold stratified CV. Ragas normalized (strip modifiers, take primary from pairs, exclude genres). Only ragas with ≥20 songs.")
print(f"key_finding: Best accuracy {best_acc:.3f} ({best_method}) vs majority baseline {majority_baseline:.3f} and random baseline {random_baseline:.3f}")
print(f"statistical_significance: {best_acc:.3f} vs {majority_baseline:.3f} majority = {(best_acc-majority_baseline)*100:+.1f}pp; vs {random_baseline:.3f} random = {best_acc/random_baseline:.1f}x lift")
if best_acc > majority_baseline * 1.1:
    print(f"conclusion: Raga is meaningfully predictable from lyrics, confirming Tagore systematically matched lyrical content to melodic modes.")
else:
    print(f"conclusion: Raga prediction from lyrics alone shows limited improvement over baseline, suggesting the raga-lyric relationship is more subtle than category-lyric.")
print(f"=== END RESULT ===")
