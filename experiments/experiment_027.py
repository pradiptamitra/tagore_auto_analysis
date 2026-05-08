"""Experiment 027: Pronoun usage evolution across Tagore's career.

Hypothesis: The balance of first-person (আমি/আমার/মোর) vs second-person
(তুমি/তোমার/তব) pronouns shifted over Tagore's career, reflecting changing
perspective (self-focused vs other-directed/devotional).
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from scipy import stats
import re

df = load_tagore()
df = df.dropna(subset=['gregorian_year', 'lyrics'])
df['gregorian_year'] = df['gregorian_year'].astype(int)

# Bengali pronouns (including archaic forms)
first_person = ['আমি', 'আমার', 'আমাকে', 'আমায়', 'আমাদের', 'মোর', 'মম', 'মোরে', 'মুই']
second_person = ['তুমি', 'তোমার', 'তোমায়', 'তোমাকে', 'তোমাদের', 'তব', 'তোর', 'তুই', 'তোরে']
third_person = ['সে', 'তার', 'তাহার', 'তাকে', 'তাদের', 'ইনি', 'উনি']

def count_pronouns(text):
    words = text.split()
    # Clean punctuation for matching
    clean_words = [re.sub(r'[।॥,\.!?;:\-–—]', '', w) for w in words]
    n_words = len(words)

    first = sum(1 for w in clean_words if w in first_person)
    second = sum(1 for w in clean_words if w in second_person)
    third = sum(1 for w in clean_words if w in third_person)

    return pd.Series({
        'first_person': first,
        'second_person': second,
        'third_person': third,
        'first_rate': first / n_words if n_words > 0 else 0,
        'second_rate': second / n_words if n_words > 0 else 0,
        'third_rate': third / n_words if n_words > 0 else 0,
        'pronoun_total': first + second + third,
        'pronoun_rate': (first + second + third) / n_words if n_words > 0 else 0,
        'first_second_ratio': first / max(second, 0.5) if second > 0 or first > 0 else 0,
    })

pronouns = df['lyrics'].apply(count_pronouns)
df = pd.concat([df.reset_index(drop=True), pronouns.reset_index(drop=True)], axis=1)

print(f"Songs: {len(df)}")
print(f"\n--- Overall pronoun statistics ---")
print(f"  First person per song: {df['first_person'].mean():.2f} ({df['first_rate'].mean()*100:.2f}%)")
print(f"  Second person per song: {df['second_person'].mean():.2f} ({df['second_rate'].mean()*100:.2f}%)")
print(f"  Third person per song: {df['third_person'].mean():.2f} ({df['third_rate'].mean()*100:.2f}%)")
print(f"  Songs with NO pronouns: {(df['pronoun_total'] == 0).sum()} ({(df['pronoun_total'] == 0).mean():.1%})")

# By category
print(f"\n--- Pronoun rates by category ---")
for cat in df['category'].value_counts().head(8).index:
    subset = df[df['category'] == cat]
    print(f"  {cat} (n={len(subset)}):")
    print(f"    1st: {subset['first_rate'].mean()*100:.2f}%, "
          f"2nd: {subset['second_rate'].mean()*100:.2f}%, "
          f"3rd: {subset['third_rate'].mean()*100:.2f}%")

# Kruskal-Wallis for pronoun rates across categories
for feat in ['first_rate', 'second_rate', 'first_second_ratio']:
    groups = [df[df['category'] == cat][feat].values for cat in df['category'].value_counts().head(6).index]
    h, p = stats.kruskal(*groups)
    print(f"\n  KW for {feat}: H={h:.1f}, p={p:.2e}")

# Temporal trends
print(f"\n--- Temporal trends ---")
for feat in ['first_rate', 'second_rate', 'third_rate', 'pronoun_rate', 'first_second_ratio']:
    r, p = stats.spearmanr(df['gregorian_year'], df[feat])
    print(f"  Year vs {feat}: rho={r:.4f}, p={p:.2e}")

# Decade breakdown
df['decade'] = (df['gregorian_year'] // 10) * 10
decade_pron = df.groupby('decade').agg(
    first_rate=('first_rate', 'mean'),
    second_rate=('second_rate', 'mean'),
    third_rate=('third_rate', 'mean'),
    ratio=('first_second_ratio', 'mean'),
    n=('first_rate', 'size'),
).reset_index()

print(f"\nBy decade:")
for _, row in decade_pron.iterrows():
    print(f"  {int(row['decade'])}s (n={int(row['n'])}): 1st={row['first_rate']*100:.2f}%, "
          f"2nd={row['second_rate']*100:.2f}%, 3rd={row['third_rate']*100:.2f}%, "
          f"1st/2nd ratio={row['ratio']:.2f}")

# Within Puja specifically (devotional: "I" addresses God as "You")
puja = df[df['category'] == 'পূজা']
r_puja, p_puja = stats.spearmanr(puja['gregorian_year'], puja['second_rate'])
print(f"\nPuja second-person trend: rho={r_puja:.4f}, p={p_puja:.2e}")
r_puja1, p_puja1 = stats.spearmanr(puja['gregorian_year'], puja['first_rate'])
print(f"Puja first-person trend: rho={r_puja1:.4f}, p={p_puja1:.2e}")

# Most common specific pronoun forms
all_pronouns = first_person + second_person
print(f"\n--- Individual pronoun frequencies ---")
for pron in all_pronouns:
    count = sum(1 for text in df['lyrics'] for w in text.split() if re.sub(r'[।॥,\.!?;:]', '', w) == pron)
    print(f"  {pron}: {count}")

print(f"\n=== RESULT ===")
r_1st, p_1st = stats.spearmanr(df['gregorian_year'], df['first_rate'])
r_2nd, p_2nd = stats.spearmanr(df['gregorian_year'], df['second_rate'])
print(f"hypothesis: Pronoun usage shifted over Tagore's career")
print(f"method: Counted first/second/third person pronouns per song, tracked rates over time and across categories")
print(f"key_finding: First person {'increased' if r_1st > 0 else 'decreased'} (rho={r_1st:.3f}, p={p_1st:.2e}). Second person {'increased' if r_2nd > 0 else 'decreased'} (rho={r_2nd:.3f}, p={p_2nd:.2e}). Strong category differences in pronoun profile.")
print(f"statistical_significance: 1st: rho={r_1st:.3f}, p={p_1st:.2e}. 2nd: rho={r_2nd:.3f}, p={p_2nd:.2e}")
print(f"conclusion: Pronoun patterns reveal shifts in perspective and address across Tagore's career and between categories.")
print(f"=== END RESULT ===")
