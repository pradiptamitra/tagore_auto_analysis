"""Experiment 036: Raga emotional profiles from lyrics.

Exp 005 showed raga isn't predictable from lyrics. But the reverse might work:
do ragas correlate with emotional vocabulary? In Indian music theory, ragas
evoke specific moods (rasa). Hypothesis: Songs in "dark" ragas (Bhairavi,
Malkauns) use more sorrow/longing words, while "bright" ragas (Bahar, Khamaj)
use more joy/nature words — revealing Tagore's intuitive rasa-raga mapping.
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import re

df = load_tagore()
df = df.dropna(subset=['raga', 'lyrics'])

# Normalize ragas (from prior experiments)
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

# Emotion word lists (Bengali — hand-curated from domain knowledge)
# Positive/bright: joy, beauty, light, celebration
positive_words = {
    'আনন্দ', 'সুখ', 'হাসি', 'উৎসব', 'খুশি', 'আলো', 'আলোক', 'উজ্জ্বল',
    'সুন্দর', 'মধুর', 'রঙ', 'ফুল', 'হর্ষ', 'আশা', 'মুক্তি', 'জয়',
    'প্রেম', 'গান', 'নাচ', 'খেলা', 'বসন্ত', 'পূর্ণ', 'আকাশ', 'তারা',
}
# Negative/dark: sorrow, longing, darkness, separation
negative_words = {
    'দুঃখ', 'কান্না', 'কাঁদ', 'অন্ধকার', 'আঁধার', 'বিরহ', 'বেদনা',
    'ব্যথা', 'যন্ত্রণা', 'শোক', 'মৃত্যু', 'নিশি', 'রাত', 'একলা',
    'নির্জন', 'হারা', 'ভয়', 'অশ্রু', 'ক্লান্ত', 'শূন্য', 'বিষাদ',
    'দীর্ঘ', 'নিরাশা', 'অবসান',
}

def emotion_scores(text):
    words = text.split()
    clean = [re.sub(r'[।॥,\.!?;:\-–—\d]', '', w) for w in words]
    n = max(len(clean), 1)
    pos = sum(1 for w in clean for pw in positive_words if pw in w)
    neg = sum(1 for w in clean for nw in negative_words if nw in w)
    return pos / n, neg / n, (pos - neg) / n

scores = df['lyrics'].apply(lambda t: pd.Series(emotion_scores(t),
    index=['pos_rate', 'neg_rate', 'valence']))
df = pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)

# Top ragas with enough songs
raga_counts = df['raga_norm'].value_counts()
top_ragas = raga_counts[raga_counts >= 15].index.tolist()

print(f"Songs: {len(df)}, Top ragas (≥15 songs): {len(top_ragas)}")

# Emotional profile per raga
print(f"\n--- Emotional profiles by raga ---")
raga_profiles = []
for raga in top_ragas:
    subset = df[df['raga_norm'] == raga]
    profile = {
        'raga': raga,
        'n': len(subset),
        'pos': subset['pos_rate'].mean(),
        'neg': subset['neg_rate'].mean(),
        'valence': subset['valence'].mean(),
        'pos_std': subset['pos_rate'].std(),
        'neg_std': subset['neg_rate'].std(),
    }
    raga_profiles.append(profile)

raga_df = pd.DataFrame(raga_profiles).sort_values('valence')

print(f"\n  {'Raga':<20s} {'n':>4s} {'Pos%':>7s} {'Neg%':>7s} {'Valence':>8s}")
for _, row in raga_df.iterrows():
    bar_pos = '+' * int(row['pos'] * 200)
    bar_neg = '-' * int(row['neg'] * 200)
    print(f"  {row['raga']:<20s} {int(row['n']):>4d} {row['pos']*100:>6.2f}% {row['neg']*100:>6.2f}% {row['valence']*100:>7.2f}%")

# Most negative (darkest) ragas
print(f"\n--- Darkest ragas (most negative valence) ---")
for _, row in raga_df.head(5).iterrows():
    print(f"  {row['raga']}: valence={row['valence']*100:.2f}% (pos={row['pos']*100:.1f}%, neg={row['neg']*100:.1f}%)")

print(f"\n--- Brightest ragas (most positive valence) ---")
for _, row in raga_df.tail(5).iterrows():
    print(f"  {row['raga']}: valence={row['valence']*100:.2f}% (pos={row['pos']*100:.1f}%, neg={row['neg']*100:.1f}%)")

# Kruskal-Wallis: does valence differ by raga?
groups = [df[df['raga_norm'] == r]['valence'].values for r in top_ragas]
h, p_kw = stats.kruskal(*groups)
print(f"\n--- Statistical test ---")
print(f"  KW test (valence across {len(top_ragas)} ragas): H={h:.2f}, p={p_kw:.4f}")

# Also test positive and negative separately
h_pos, p_pos = stats.kruskal(*[df[df['raga_norm']==r]['pos_rate'].values for r in top_ragas])
h_neg, p_neg = stats.kruskal(*[df[df['raga_norm']==r]['neg_rate'].values for r in top_ragas])
print(f"  KW test (positive across ragas): H={h_pos:.2f}, p={p_pos:.4f}")
print(f"  KW test (negative across ragas): H={h_neg:.2f}, p={p_neg:.4f}")

# Specific raga comparisons aligned with music theory:
# Bhairavi (melancholic) vs Khamaj (bright/romantic)
if 'ভৈরবী' in top_ragas and 'খাম্বাজ' in top_ragas:
    bhairavi = df[df['raga_norm'] == 'ভৈরবী']
    khamaj = df[df['raga_norm'] == 'খাম্বাজ']
    u, p = stats.mannwhitneyu(bhairavi['valence'], khamaj['valence'])
    d = (bhairavi['valence'].mean() - khamaj['valence'].mean()) / \
        np.sqrt((bhairavi['valence'].var() + khamaj['valence'].var()) / 2)
    print(f"\n  Bhairavi vs Khamaj valence: d={d:.3f}, p={p:.4f}")
    print(f"    Bhairavi: valence={bhairavi['valence'].mean()*100:.2f}%")
    print(f"    Khamaj: valence={khamaj['valence'].mean()*100:.2f}%")

# Bahar (spring/bright) vs Malhar (monsoon/intense)
if 'বাহার' in top_ragas:
    bahar = df[df['raga_norm'] == 'বাহার']
    print(f"\n  Bahar: valence={bahar['valence'].mean()*100:.2f}% (spring raga)")
if 'মল্লার' in top_ragas or 'মিয়াঁ' in top_ragas:
    mallar_mask = df['raga_norm'].str.contains('মল্লার')
    if mallar_mask.sum() >= 5:
        mallar = df[mallar_mask]
        print(f"  Mallar variants: valence={mallar['valence'].mean()*100:.2f}% (monsoon raga, n={len(mallar)})")

# Emotional profile by category (sanity check)
print(f"\n--- Valence by category (sanity check) ---")
df_cat = df.dropna(subset=['category'])
for cat in df_cat['category'].value_counts().head(6).index:
    subset = df_cat[df_cat['category'] == cat]
    print(f"  {cat}: valence={subset['valence'].mean()*100:.2f}% "
          f"(pos={subset['pos_rate'].mean()*100:.1f}%, neg={subset['neg_rate'].mean()*100:.1f}%)")

# Interaction: does raga valence vary by category?
# Within Puja, does Bhairavi have different valence than Khamaj?
puja = df[df['category'] == 'পূজা']
puja_ragas = puja['raga_norm'].value_counts()
puja_top = puja_ragas[puja_ragas >= 10].index
print(f"\n--- Raga valence within Puja ---")
for raga in puja_top:
    subset = puja[puja['raga_norm'] == raga]
    print(f"  {raga} (n={len(subset)}): valence={subset['valence'].mean()*100:.2f}%")

print(f"\n=== RESULT ===")
darkest = raga_df.iloc[0]['raga']
brightest = raga_df.iloc[-1]['raga']
print(f"hypothesis: Ragas have distinct emotional profiles matching music theory (dark ragas → sorrow words, bright ragas → joy words)")
print(f"method: Emotion word scoring (24 positive, 24 negative Bengali keywords), raga-level aggregation, KW test across {len(top_ragas)} ragas")
print(f"key_finding: Darkest raga: {darkest} (valence={raga_df.iloc[0]['valence']*100:.2f}%). Brightest: {brightest} (valence={raga_df.iloc[-1]['valence']*100:.2f}%). KW p={p_kw:.4f}.")
print(f"statistical_significance: KW valence p={p_kw:.4f}, positive p={p_pos:.4f}, negative p={p_neg:.4f}")
print(f"conclusion: {'Ragas show significantly different emotional profiles' if p_kw < 0.05 else 'Emotional differences across ragas are not significant'}, {'consistent with' if p_kw < 0.05 else 'not supporting'} the rasa-raga theory in Tagore's practice.")
print(f"=== END RESULT ===")
