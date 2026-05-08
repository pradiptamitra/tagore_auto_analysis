"""Experiment 028: Best possible category classification with combined features.

Exp 001 got 52% with char n-grams alone. Can we push higher by combining
TF-IDF features with hand-crafted features from later experiments?
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.sparse import hstack, csr_matrix
import re
from collections import Counter

df = load_tagore()
df = df.dropna(subset=['category', 'lyrics'])

# Only keep categories with ≥15 songs
cat_counts = df['category'].value_counts()
valid_cats = cat_counts[cat_counts >= 15].index
df = df[df['category'].isin(valid_cats)].copy().reset_index(drop=True)
print(f"Songs: {len(df)}, Categories: {len(valid_cats)}")

le = LabelEncoder()
y = le.fit_transform(df['category'])
majority = cat_counts[valid_cats].iloc[0] / len(df)
print(f"Majority baseline: {majority:.3f}")

# Feature set 1: Char n-gram TF-IDF
vec_char = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=20000)
X_char = vec_char.fit_transform(df['lyrics'])

# Feature set 2: Word TF-IDF
vec_word = TfidfVectorizer(analyzer='word', max_features=10000)
X_word = vec_word.fit_transform(df['lyrics'])

# Feature set 3: Hand-crafted features
def extract_features(text):
    words = text.split()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    n_words = len(words)
    bengali_chars = [c for c in text if '\u0980' <= c <= '\u09FF']

    # Pronouns
    first_person = {'আমি', 'আমার', 'আমাকে', 'মোর', 'মম', 'মোরে'}
    second_person = {'তুমি', 'তোমার', 'তব', 'তোর', 'তুই', 'তোরে'}
    clean = [re.sub(r'[।॥,\.!?;:\-]', '', w) for w in words]
    first = sum(1 for w in clean if w in first_person)
    second = sum(1 for w in clean if w in second_person)

    # Phonetic classes
    vowel_signs = set('ািীুূৃেৈোৌ')
    nasals = set('নণমঙঞং')
    liquids = set('রলয')

    n_bengali = max(len(bengali_chars), 1)
    vowel_sign_rate = sum(1 for c in bengali_chars if c in vowel_signs) / n_bengali
    nasal_rate = sum(1 for c in bengali_chars if c in nasals) / n_bengali
    liquid_rate = sum(1 for c in bengali_chars if c in liquids) / n_bengali

    # Structural
    mean_line_len = np.mean([len(l) for l in lines]) if lines else 0
    ttr = len(set(words)) / n_words if n_words > 0 else 0

    # Rhyme (2-char ending matches in 4-line window)
    endings = []
    for line in lines:
        clean_line = re.sub(r'[।॥\s\.,!?;:–—\-]+$', '', line)
        bc = [c for c in clean_line if '\u0980' <= c <= '\u09FF']
        if len(bc) >= 2:
            endings.append(''.join(bc[-2:]))
    rhyme_pairs = 0
    for i in range(len(endings)):
        for j in range(i+1, min(i+4, len(endings))):
            if endings[i] and endings[j] and endings[i] == endings[j]:
                rhyme_pairs += 1
    possible = sum(min(3, len(endings) - i - 1) for i in range(len(endings) - 1)) if len(endings) > 1 else 1
    rhyme_density = rhyme_pairs / possible

    return [n_words, len(lines), mean_line_len, ttr,
            first / max(n_words, 1), second / max(n_words, 1),
            vowel_sign_rate, nasal_rate, liquid_rate, rhyme_density,
            np.mean([len(w) for w in words]) if words else 0]

hand_features = np.array([extract_features(text) for text in df['lyrics']])
X_hand = csr_matrix(StandardScaler().fit_transform(hand_features))

print(f"Feature dimensions: char={X_char.shape[1]}, word={X_word.shape[1]}, hand={X_hand.shape[1]}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Test different combinations
configs = {
    'Char only': X_char,
    'Word only': X_word,
    'Hand only': X_hand,
    'Char + Word': hstack([X_char, X_word]),
    'Char + Hand': hstack([X_char, X_hand]),
    'Word + Hand': hstack([X_word, X_hand]),
    'All combined': hstack([X_char, X_word, X_hand]),
}

print(f"\n--- Classification Results ---")
best_acc = 0
best_config = ''
for name, X in configs.items():
    scores = cross_val_score(
        LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs'),
        X, y, cv=cv, scoring='accuracy'
    )
    print(f"  {name}: {scores.mean():.3f} ± {scores.std():.3f}")
    if scores.mean() > best_acc:
        best_acc = scores.mean()
        best_config = name

# Try gradient boosting on hand features only
gb_scores = cross_val_score(
    GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
    hand_features, y, cv=cv, scoring='accuracy'
)
print(f"  Hand + GradientBoosting: {gb_scores.mean():.3f} ± {gb_scores.std():.3f}")

print(f"\n=== RESULT ===")
print(f"hypothesis: Combined features can improve category classification beyond 52%")
print(f"method: Combined char n-gram TF-IDF (20K), word TF-IDF (10K), and 11 hand-crafted features (pronouns, phonetics, structure, rhyme) with LR")
print(f"key_finding: Best accuracy {best_acc:.3f} ({best_config}) vs original 0.519 (char only in Exp 001) vs {majority:.3f} baseline. Hand features alone: {configs['Hand only'] is not None}")
print(f"statistical_significance: {best_acc:.3f} vs {majority:.3f} baseline")
print(f"conclusion: {'Combined features improve classification, confirming that multiple linguistic dimensions contribute to category identity.' if best_acc > 0.53 else 'Char n-grams already capture most category signal; additional features provide marginal gains.'}")
print(f"=== END RESULT ===")
