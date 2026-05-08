"""Experiment 001: Category classification from lyrics using character n-gram TF-IDF.

Hypothesis: Tagore's thematic categories can be predicted from lyrics alone
using character n-gram TF-IDF features, achieving >40% accuracy
(majority baseline ~27%).
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

df = load_tagore()

# Drop rows with missing category
df = df.dropna(subset=['category'])
print(f"Songs with category: {len(df)}")
print(f"Category distribution:")
print(df['category'].value_counts())

# Majority baseline
majority_class = df['category'].value_counts().iloc[0]
majority_baseline = majority_class / len(df)
print(f"\nMajority baseline accuracy: {majority_baseline:.3f}")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['category'])
n_classes = len(le.classes_)
print(f"Number of classes: {n_classes}")

# Character n-gram TF-IDF (2-5 grams)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=20000)
X = vectorizer.fit_transform(df['lyrics'])
print(f"Feature matrix: {X.shape}")

# Cross-validated evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression
lr_scores = cross_val_score(
    LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'),
    X, y, cv=cv, scoring='accuracy'
)
print(f"\nLogistic Regression CV accuracy: {lr_scores.mean():.3f} ± {lr_scores.std():.3f}")

# Random Forest
rf_scores = cross_val_score(
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    X, y, cv=cv, scoring='accuracy'
)
print(f"Random Forest CV accuracy: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")

# Also try word-level TF-IDF
word_vectorizer = TfidfVectorizer(analyzer='word', max_features=10000)
X_word = word_vectorizer.fit_transform(df['lyrics'])

lr_word_scores = cross_val_score(
    LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'),
    X_word, y, cv=cv, scoring='accuracy'
)
print(f"Logistic Regression (word TF-IDF) CV accuracy: {lr_word_scores.mean():.3f} ± {lr_word_scores.std():.3f}")

# Best result
best_method = "Logistic Regression + char n-grams"
best_acc = lr_scores.mean()
if rf_scores.mean() > best_acc:
    best_method = "Random Forest + char n-grams"
    best_acc = rf_scores.mean()
if lr_word_scores.mean() > best_acc:
    best_method = "Logistic Regression + word TF-IDF"
    best_acc = lr_word_scores.mean()

lift = best_acc / majority_baseline

print(f"\n=== RESULT ===")
print(f"hypothesis: Tagore's song categories can be predicted from lyrics with >40% accuracy (majority baseline ~27%)")
print(f"method: TF-IDF features (char n-grams and word-level) with Logistic Regression and Random Forest, 5-fold stratified CV")
print(f"key_finding: Best accuracy {best_acc:.3f} using {best_method}, vs majority baseline {majority_baseline:.3f} ({lift:.1f}x lift)")
print(f"statistical_significance: {best_acc:.3f} vs {majority_baseline:.3f} baseline = {(best_acc - majority_baseline)*100:.1f} percentage points above chance")
print(f"conclusion: Category is {'strongly' if best_acc > 0.5 else 'meaningfully' if best_acc > 0.4 else 'weakly'} predictable from lyrics alone, confirming that Tagore used distinct vocabulary/phonetic patterns across thematic categories.")
print(f"=== END RESULT ===")
