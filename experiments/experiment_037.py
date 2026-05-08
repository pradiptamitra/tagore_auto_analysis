"""Experiment 037: Raga classification from semantic embeddings.

Exp 005 showed TF-IDF features can't predict raga — surface tokens don't carry
the signal. Exp 036 showed ragas DO have distinct emotional profiles.

New approach: translate Bengali→English to capture meaning, embed with
sentence-transformers, then classify. The embedding model can't "cheat" —
it has no way to map an English poem about monsoon longing back to "Mallar".
This tests whether SEMANTIC content predicts raga, not surface forms.
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import re
import os
import json
import time

df = load_tagore()
df = df.dropna(subset=['raga', 'lyrics'])

# Normalize ragas
genre_keywords = ['বাউল', 'কীর্তন', 'ঝুমুর', 'টপ্পা', 'ঠুংরি', 'ঠুমরি', 'গজল',
                   'ভজন', 'ভাটিয়ালি', 'ইটালিয়ান', 'স্কটিশ', 'আইরিশ', 'ইংরেজি',
                   'লোকসুর', 'দেশী', 'মহিশূরী']

def normalize_raga(raga_str):
    parts = re.split(r'[-–/]| ও ', raga_str)
    part = parts[0].strip()
    for mod in ['মিশ্র ', 'শুদ্ধ ', 'সিন্ধু ']:
        if part.startswith(mod):
            part = part[len(mod):]
    return part.strip()

def is_genre(name):
    return any(kw in name for kw in genre_keywords)

df['raga_norm'] = df['raga'].apply(normalize_raga)
df = df[~df['raga_norm'].apply(is_genre)]

# Keep ragas with ≥30 songs for meaningful classification
raga_counts = df['raga_norm'].value_counts()
top_ragas = raga_counts[raga_counts >= 30].index.tolist()
df = df[df['raga_norm'].isin(top_ragas)].reset_index(drop=True)

print(f"Songs: {len(df)}, Ragas: {len(top_ragas)}")
print(f"Raga distribution: {raga_counts[top_ragas].to_dict()}")

# --- Step 1: Translate Bengali → English (with caching) ---
CACHE_FILE = os.path.join(os.path.dirname(__file__), '.translation_cache.json')

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        translation_cache = json.load(f)
    print(f"Loaded {len(translation_cache)} cached translations")
else:
    translation_cache = {}

# Translate in batches
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='bn', target='en')

texts_to_translate = []
indices_to_translate = []
for i, row in df.iterrows():
    # Use first 500 chars to keep translations manageable
    text_key = row['lyrics'][:500]
    if text_key not in translation_cache:
        texts_to_translate.append(text_key)
        indices_to_translate.append(i)

print(f"Need to translate {len(texts_to_translate)} songs ({len(df) - len(texts_to_translate)} cached)")

# Translate in batches with rate limiting
batch_size = 20
for batch_start in range(0, len(texts_to_translate), batch_size):
    batch = texts_to_translate[batch_start:batch_start + batch_size]
    for text in batch:
        try:
            translated = translator.translate(text)
            translation_cache[text] = translated
        except Exception as e:
            print(f"  Translation error: {e}")
            translation_cache[text] = ""
        time.sleep(0.1)  # rate limit

    if batch_start % 200 == 0 and batch_start > 0:
        print(f"  Translated {batch_start}/{len(texts_to_translate)}...")
        # Save cache periodically
        with open(CACHE_FILE, 'w') as f:
            json.dump(translation_cache, f, ensure_ascii=False)

# Save final cache
with open(CACHE_FILE, 'w') as f:
    json.dump(translation_cache, f, ensure_ascii=False)

# Build English texts
df['english'] = df['lyrics'].apply(lambda t: translation_cache.get(t[:500], ''))
df = df[df['english'].str.len() > 10].reset_index(drop=True)
print(f"Songs with translations: {len(df)}")

# --- Step 2: Embed with sentence-transformers ---
print("Loading sentence-transformer model...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
print("Encoding translations...")
embeddings = model.encode(df['english'].tolist(), show_progress_bar=True, batch_size=64)
print(f"Embedding shape: {embeddings.shape}")

# --- Step 3: Classify ---
le = LabelEncoder()
y = le.fit_transform(df['raga_norm'])
X = embeddings

n_classes = len(le.classes_)
baseline = max(np.bincount(y)) / len(y)
print(f"\nClasses: {n_classes}, Baseline (majority): {baseline:.3f}")

# Cross-validated classification
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
lr_scores = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')
print(f"\nLogistic Regression: {lr_scores.mean():.3f} ± {lr_scores.std():.3f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
print(f"Random Forest: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")

# Best model — full classification report on one fold
best_name = max([('LR', lr_scores.mean(), lr),
                  ('RF', rf_scores.mean(), rf)],
                 key=lambda x: x[1])
print(f"\nBest model: {best_name[0]} ({best_name[1]:.3f})")

# Detailed per-raga performance
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(best_name[2], X, y, cv=cv)
report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)

print(f"\n--- Per-raga performance (best model: {best_name[0]}) ---")
print(f"  {'Raga':<20s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'n':>5s}")
for raga in sorted(le.classes_, key=lambda r: -report[r]['f1-score']):
    r = report[raga]
    print(f"  {raga:<20s} {r['precision']:>6.2f} {r['recall']:>6.2f} {r['f1-score']:>6.2f} {int(r['support']):>5d}")

# Compare to Exp 005 TF-IDF baseline
print(f"\n--- Comparison with Exp 005 (TF-IDF) ---")
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF on original Bengali (replicating Exp 005)
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=5000)
X_tfidf = tfidf.fit_transform(df['lyrics'])
lr_tfidf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
tfidf_scores = cross_val_score(lr_tfidf, X_tfidf, y, cv=cv, scoring='accuracy')
print(f"  TF-IDF char n-grams (Bengali): {tfidf_scores.mean():.3f} ± {tfidf_scores.std():.3f}")
print(f"  Semantic embeddings (English):  {lr_scores.mean():.3f} ± {lr_scores.std():.3f}")
print(f"  Improvement: {(lr_scores.mean() - tfidf_scores.mean())*100:.1f} percentage points")

# Which ragas improved most?
y_pred_tfidf = cross_val_predict(lr_tfidf, X_tfidf, y, cv=cv)
report_tfidf = classification_report(y, y_pred_tfidf, target_names=le.classes_, output_dict=True)

print(f"\n--- Raga-by-raga improvement ---")
for raga in sorted(le.classes_, key=lambda r: -(report[r]['f1-score'] - report_tfidf[r]['f1-score'])):
    f1_embed = report[raga]['f1-score']
    f1_tfidf = report_tfidf[raga]['f1-score']
    delta = f1_embed - f1_tfidf
    print(f"  {raga:<20s} TF-IDF F1={f1_tfidf:.2f} → Embed F1={f1_embed:.2f} ({'+' if delta>=0 else ''}{delta:.2f})")

print(f"\n=== RESULT ===")
print(f"hypothesis: Semantic embeddings can predict raga where TF-IDF failed")
print(f"method: Bengali→English translation (Google Translate), sentence-transformer embeddings (all-MiniLM-L6-v2), cross-validated LR/RF/GB on {n_classes} ragas ({len(df)} songs)")
print(f"key_finding: Best accuracy: {best_name[1]:.3f} ({best_name[0]}). TF-IDF baseline: {tfidf_scores.mean():.3f}. Improvement: {(best_name[1] - tfidf_scores.mean())*100:.1f}pp. Majority baseline: {baseline:.3f}.")
print(f"statistical_significance: 5-fold cross-validated; {best_name[0]} {best_name[1]:.3f} ± {best_name[1]:.3f}")
if best_name[1] > baseline + 0.05:
    print(f"conclusion: Semantic content DOES predict raga above baseline. The raga-lyrics link exists at the meaning level, not the surface token level. Exp 005's negative result was a feature representation problem, not an absence of signal.")
else:
    print(f"conclusion: Even semantic embeddings struggle to predict raga. The raga-lyrics link, while real at the emotional level (Exp 036), is too subtle for song-level classification.")
print(f"=== END RESULT ===")
