"""Combined fork of Exp 010 + 014 + 034: taal SIMPLIFICATION re-tested
with a musicology-based complexity classification.

PROBLEM WITH PRIOR WORK
-----------------------
Exp 010/014/024/034 all operated on raw matra (beat) count, and Exp 010
defined "simple = <=8 beats". That proxy is musicologically wrong:

  * Tritaal/Teentaal has 16 matra but is the MOST versatile, symmetric
    (4+4+4+4) and accessible taal in Hindustani music — it is *simple*,
    yet the <=8 rule labels it "complex" (137+ songs mislabelled).
  * Jhaptaal has 10 matra but is an asymmetric (2+3+2+3) "heavy-beating"
    classical taal — it is *complex*, the <=8 rule got the spirit right
    only by accident of length.
  * Teora (7) is asymmetric (3+2+2) and classical — *complex* — but the
    <=8 rule calls it "simple".

geetabitan.com (authoritative Rabindra Sangeet source) frames the actual
evolution as: heavy-beating classical taals (Chautal, Jhaptaal, Dhamar,
Ektal) early -> short LIGHT taals (Dadra, Kaharba) late. The axis is
cultural register + vibhag symmetry, NOT cycle length.

CLASSIFICATION (1=simplest .. 5=most complex)
---------------------------------------------
Grounded in: (a) performance register — light/folk theka vs
semi-classical vs dhrupad/khayal "heavy" taal; (b) vibhag symmetry —
even/equal divisions are simpler than asymmetric ones; cycle length is
used only as a weak tiebreaker, never as the primary axis.
Sources: geetabitan.com taal pages; SwarGanga / Artium Academy / Kathak
World vibhag references (Teentaal 4+4+4+4 symmetric = simplest;
Jhaptaal 2+3+2+3 asymmetric = intermediate; Dhamar 5+2+3+4 = advanced).

This script reproduces, on the NEW scale, all three prior analyses:
  Exp 010  temporal trend + decade profile + early/late test
  Exp 014  category-controlled partial correlation + Puja-only
  Exp 034  formal change-point detection (+ permutation clustering)
and reports OLD (beat-rule) vs NEW side by side.
"""

import sys
sys.path.insert(0, '/Users/ppm/code/tagore_auto_analysis')

from dataset import load_tagore
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import re

# ----------------------------------------------------------------------
# 1. Musicology-based per-taal table.
#    score: 1..5 complexity.  beats: matra (for OLD-rule comparison).
#    Canonical Bengali keys; variants handled by normalisation below.
# ----------------------------------------------------------------------
TAAL_TABLE = {
    # --- Light / folk thekas, symmetric: SIMPLE (1) ---
    'দাদরা':        dict(beats=6,  score=1, note='light theka 3+3 symmetric'),
    'কাহারবা':      dict(beats=8,  score=1, note='light theka 4+4 symmetric'),
    'খেমটা':        dict(beats=6,  score=1, note='light/folk 3+3'),
    'ষষ্ঠী':        dict(beats=6,  score=1, note='Tagore light, even'),
    'বাউল':         dict(beats=8,  score=1, note='Baul folk pulse'),
    # --- Versatile / semi-classical, structurally regular: SIMPLE-ISH (2) ---
    'ত্রিতাল':      dict(beats=16, score=2, note='Teentaal 4+4+4+4 symmetric, most versatile'),
    'যৎ':           dict(beats=8,  score=2, note='even, light-classical'),
    'আড়াঠেকা':     dict(beats=8,  score=2, note='theka, medium register'),
    'মধ্যমান':      dict(beats=10, score=2, note='medium register'),
    # --- Asymmetric or odd-cycle, semi-classical / Tagore-coined: MEDIUM (3) ---
    'তেওরা':        dict(beats=7,  score=3, note='Tivra 3+2+2 asymmetric, classical'),
    'রূপক':         dict(beats=7,  score=3, note='Rupak 3+2+2 asymmetric'),
    'রূপকড়া':      dict(beats=7,  score=3, note='Tagore, derived from Rupak, asymmetric'),
    'নবতাল':        dict(beats=9,  score=3, note='Tagore, 9-beat odd cycle'),
    'অর্ধঝাঁপ':     dict(beats=5,  score=3, note='Tagore half-Jhaptaal, asymmetric'),
    'সুরফাঁকতাল':   dict(beats=10, score=3, note='Tagore creation, irregular'),
    'ঝম্পক':        dict(beats=10, score=3, note='Tagore creation'),
    'নবপঞ্চ':       dict(beats=9,  score=3, note='Tagore creation, odd'),
    # --- Heavy classical / khayal, longer cycles: COMPLEX (4) ---
    'ঝাঁপতাল':      dict(beats=10, score=4, note='heavy classical 2+3+2+3 asymmetric'),
    'একতাল':        dict(beats=12, score=4, note='khayal, slow elaboration, heavy'),
    'একাদশী':       dict(beats=11, score=4, note='Tagore, 11-beat odd, heavy'),
    'পঞ্চমসওয়ারি': dict(beats=15, score=4, note='15-beat sawari, heavy'),
    # --- Dhrupad / most asymmetric: MOST COMPLEX (5) ---
    'চৌতাল':        dict(beats=12, score=5, note='dhrupad, heavy beating'),
    'আড়া চৌতাল':   dict(beats=14, score=5, note='dhrupad ada-chautal, heavy'),
    'ধামার':        dict(beats=14, score=5, note='dhrupad 5+2+3+4 highly asymmetric'),
    'ঝুমরা':        dict(beats=14, score=5, note='vilambit khayal 3+4+3+4, heavy'),
}
# Longest keys first so 'আড়া চৌতাল' wins over 'চৌতাল', etc.
_KEYS_BY_LEN = sorted(TAAL_TABLE, key=len, reverse=True)

# Free-rhythm / non-metrical markers -> excluded (NaN), like Exp 010.
_FREE = ['মুক্ত', 'মুুক্ত', 'ছন্দ', 'মাত্রার তাল', 'মাত্রা', 'ঠুংরি']


def _strip(s):
    """Collapse repeated-vowel-sign typos and trim spaces."""
    s = s.strip()
    s = s.replace('সুুু', 'সু').replace('সুু', 'সু')        # সুুরফাঁকতাল
    s = s.replace('মুুক্ত', 'মুক্ত').replace('ঝাঁঁপ', 'ঝাঁপ')
    return s


def _match_one(token):
    """Map a single (already split) taal token to a complexity score."""
    t = _strip(token)
    if not t:
        return None
    for k in _KEYS_BY_LEN:
        if k in t:
            return TAAL_TABLE[k]['score']
    if any(f in t for f in _FREE):
        return None  # free meter — not a metrical taal
    return None      # unknown -> unmapped


def classify(raw):
    """Compound taals (A/B, A-B, A B) -> mean of component scores."""
    if not isinstance(raw, str):
        return np.nan
    parts = re.split(r'[/\-]| ও ', raw)
    scores = [s for s in (_match_one(p) for p in parts) if s is not None]
    return float(np.mean(scores)) if scores else np.nan


def old_beats(raw):
    """Reproduce Exp 010's beat lookup (first recognised component)."""
    if not isinstance(raw, str):
        return np.nan
    for p in re.split(r'[/\-]| ও ', raw):
        t = _strip(p)
        for k in _KEYS_BY_LEN:
            if k in t:
                return TAAL_TABLE[k]['beats']
    return np.nan


# ----------------------------------------------------------------------
# 2. Load + annotate
# ----------------------------------------------------------------------
df = load_tagore()
df = df.dropna(subset=['gregorian_year', 'taal']).copy()
df['gregorian_year'] = df['gregorian_year'].astype(int)
df = df[(df['gregorian_year'] >= 1880) & (df['gregorian_year'] <= 1941)]

df['cx'] = df['taal'].apply(classify)            # NEW complexity 1..5
df['beats'] = df['taal'].apply(old_beats)        # OLD matra count
m = df.dropna(subset=['cx']).copy()
mb = df.dropna(subset=['beats']).copy()

print(f"Songs with year+taal: {len(df)}")
print(f"  classified (new):   {len(m)} ({100*len(m)/len(df):.1f}%)")
print(f"  beat-mapped (old):  {len(mb)} ({100*len(mb)/len(df):.1f}%)")

# --- Reclassification audit: who flipped between OLD rule and NEW? ---
both = df.dropna(subset=['cx', 'beats']).copy()
both['old_simple'] = both['beats'] <= 8          # Exp 010 definition
both['new_simple'] = both['cx'] <= 2.0           # light/versatile-regular
flip = both[both['old_simple'] != both['new_simple']]
print(f"\nClass flips between OLD (<=8 beats) and NEW: {len(flip)} songs "
      f"({100*len(flip)/len(both):.1f}%)")
print("  Top flipped taals (old_simple -> new_simple):")
for taal, sub in flip.groupby('taal'):
    if len(sub) >= 5:
        o = bool(sub['old_simple'].iloc[0]); n = bool(sub['new_simple'].iloc[0])
        print(f"    {taal!r:>14}: n={len(sub):3d}  simple {o} -> {n}  (beats={int(sub['beats'].iloc[0])}, cx={sub['cx'].iloc[0]:.1f})")


def banner(t):
    print(f"\n{'='*66}\n{t}\n{'='*66}")


# ----------------------------------------------------------------------
# 3. Exp 010 redux — temporal trend (NEW vs OLD)
# ----------------------------------------------------------------------
banner("EXP 010 REDUX — TEMPORAL TREND")
r_new, p_new = stats.spearmanr(m['gregorian_year'], m['cx'])
r_old, p_old = stats.spearmanr(mb['gregorian_year'], mb['beats'])
print(f"NEW complexity vs year: rho={r_new:+.4f}, p={p_new:.2e}  (n={len(m)})")
print(f"OLD beats      vs year: rho={r_old:+.4f}, p={p_old:.2e}  (n={len(mb)})")

m['decade'] = (m['gregorian_year'] // 10) * 10
dec = m.groupby('decade').agg(n=('cx', 'size'), mean_cx=('cx', 'mean'),
                              pct_simple=('cx', lambda x: (x <= 2).mean()))
print("\nDecade profile (NEW):")
for d, row in dec.iterrows():
    print(f"  {int(d)}s: n={int(row['n']):4d}  mean_cx={row['mean_cx']:.2f}  "
          f"simple={row['pct_simple']*100:4.0f}%")

early = m[m['gregorian_year'] <= 1910]['cx']
late = m[m['gregorian_year'] >= 1920]['cx']
u, p_el = stats.mannwhitneyu(early, late, alternative='two-sided')
print(f"\nEarly(<=1910) mean_cx={early.mean():.2f} vs Late(>=1920) {late.mean():.2f} "
      f"-> Mann-Whitney p={p_el:.2e}")

# ----------------------------------------------------------------------
# 4. Exp 014 redux — control for category
# ----------------------------------------------------------------------
banner("EXP 014 REDUX — CATEGORY-CONTROLLED")
mc = m.dropna(subset=['category']).copy()
r_all, p_all = stats.spearmanr(mc['gregorian_year'], mc['cx'])
cat_d = pd.get_dummies(mc['category'], drop_first=True).astype(float)
yr_res = mc['gregorian_year'].values - LinearRegression().fit(cat_d, mc['gregorian_year']).predict(cat_d)
cx_res = mc['cx'].values - LinearRegression().fit(cat_d, mc['cx']).predict(cat_d)
r_par, p_par = stats.spearmanr(yr_res, cx_res)
print(f"Overall (with category) rho={r_all:+.4f}, p={p_all:.2e}  (n={len(mc)})")
print(f"Partial (| category)    rho={r_par:+.4f}, p={p_par:.2e}")
print(f"Category-mediated fraction: {(1-abs(r_par)/abs(r_all))*100:.1f}%")
for cat in mc['category'].value_counts().head(5).index:
    s = mc[mc['category'] == cat]
    rr, pp = stats.spearmanr(s['gregorian_year'], s['cx'])
    print(f"  {cat:>10}: rho={rr:+.3f}, p={pp:.1e}, n={len(s)}")

# ----------------------------------------------------------------------
# 5. Exp 034 redux — formal change-point detection
# ----------------------------------------------------------------------
banner("EXP 034 REDUX — CHANGE-POINT DETECTION")


def find_cp(years, values, min_seg=30):
    idx = np.argsort(years); y = years[idx]; v = values[idx]
    n = len(v); ov = np.var(v)
    if ov == 0:
        return None
    best, byr = -np.inf, None
    for yr in np.unique(y):
        L = v[y <= yr]; R = v[y > yr]
        if len(L) < min_seg or len(R) < min_seg:
            continue
        sc = -(len(L)*np.log(np.var(L)+1e-10) + len(R)*np.log(np.var(R)+1e-10))
        sc += n*np.log(ov+1e-10)
        if sc > best:
            best, byr = sc, yr
    if byr is None:
        return None
    L = v[y <= byr]; R = v[y > byr]
    _, pv = stats.mannwhitneyu(L, R, alternative='two-sided')
    return dict(year=int(byr), p=pv, lm=L.mean(), rm=R.mean(),
                d=(R.mean()-L.mean())/np.sqrt(ov+1e-10), score=best)


def two_cp(years, values, min_seg=30):
    c1 = find_cp(years, values, min_seg)
    if c1 is None:
        return []
    out = [c1]
    for mask in (years <= c1['year'], years > c1['year']):
        if mask.sum() >= 2*min_seg:
            c = find_cp(years[mask], values[mask], min_seg)
            if c:
                out.append(c)
    return out


events = {1901: "wife dies", 1902: "daughter dies", 1907: "son dies",
          1913: "Nobel Prize", 1919: "renounces knighthood",
          1921: "Visva-Bharati", 1930: "dance-drama period",
          1936: "peak output"}

for label, frame, col in [("NEW complexity", m, 'cx'), ("OLD beats", mb, 'beats')]:
    yrs = frame['gregorian_year'].values
    cps = two_cp(yrs, frame[col].values)
    print(f"\n--- {label} (n={len(frame)}) ---")
    for i, c in enumerate(cps):
        sig = "***" if c['p'] < .001 else "**" if c['p'] < .01 else "*" if c['p'] < .05 else "ns"
        arrow = "↓" if c['d'] < 0 else "↑"
        near = [f"{y}:{e}" for y, e in events.items() if abs(y-c['year']) <= 3]
        print(f"  CP{i+1}: {c['year']} ({c['lm']:.3f}→{c['rm']:.3f} {arrow} "
              f"|d|={abs(c['d']):.2f}, p={c['p']:.2e} {sig})"
              + (f"  ≈ {', '.join(near)}" if near else ""))

# permutation clustering on NEW change-points
cps_new = two_cp(m['gregorian_year'].values, m['cx'].values)
sig_years = [c['year'] for c in cps_new if c['p'] < 0.05]
if len(sig_years) >= 2:
    obs = np.std(sig_years)
    rng = np.random.default_rng(42)
    lo, hi = m['gregorian_year'].min(), m['gregorian_year'].max()
    rand = [np.std(rng.integers(lo, hi, size=len(sig_years))) for _ in range(10000)]
    p_clu = float(np.mean(np.array(rand) <= obs))
    print(f"\nClustering of NEW change-points: spread={obs:.1f}y "
          f"(random {np.mean(rand):.1f}y), p={p_clu:.4f}")
else:
    p_clu = float('nan')

# ----------------------------------------------------------------------
# 6. RESULT block
# ----------------------------------------------------------------------
banner("RESULT")
print("hypothesis: The 'great simplification' survives a musicology-based "
      "taal complexity scale (register + vibhag symmetry), not just raw beat count")
print("method: Reclassified taals on a 1-5 complexity scale; reran Exp 010 "
      "trend, Exp 014 category-partial correlation, Exp 034 change-point "
      "detection; OLD beat-rule kept as side-by-side baseline")
print(f"key_finding: NEW complexity vs year rho={r_new:+.3f} (p={p_new:.1e}) "
      f"vs OLD beats rho={r_old:+.3f} (p={p_old:.1e}); "
      f"{len(flip)} songs ({100*len(flip)/len(both):.0f}%) change simple/complex "
      f"label under the corrected scale; partial rho|category={r_par:+.3f} "
      f"({(1-abs(r_par)/abs(r_all))*100:.0f}% category-mediated); "
      f"NEW change-points at {sig_years}")
print(f"statistical_significance: trend p={p_new:.1e}; partial p={p_par:.1e}; "
      f"clustering p={p_clu}")
verdict = ("CONFIRMED — direction and significance hold under the corrected scale"
           if (np.sign(r_new) == np.sign(r_old) and p_new < 0.05)
           else "OVERTURNED — the corrected scale changes the conclusion")
print(f"conclusion: {verdict}. The beat-count proxy "
      f"{'overstated' if abs(r_old) > abs(r_new) else 'understated'} "
      f"the effect size (|rho| {abs(r_old):.2f} -> {abs(r_new):.2f}).")
print("=== END RESULT ===")
