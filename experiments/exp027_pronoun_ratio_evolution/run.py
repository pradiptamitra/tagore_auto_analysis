"""exp027 fork — pronoun ratio evolution with category decomposition.

See README.md in this directory for intent and methodology.
"""

import sys
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
HERE = Path(__file__).resolve().parent

from dataset import load_tagore  # noqa: E402

matplotlib.use("Agg")

FIRST_PERSON = ['আমি', 'আমার', 'আমাকে', 'আমায়', 'আমাদের', 'মোর', 'মম', 'মোরে', 'মুই']
SECOND_PERSON = ['তুমি', 'তোমার', 'তোমায়', 'তোমাকে', 'তোমাদের', 'তব', 'তোর', 'তুই', 'তোরে']
_PUNCT_RE = re.compile(r'[।॥,\.!?;:\-–—]')


def count_pronouns(text: str) -> tuple[int, int, int]:
    words = text.split()
    clean = [_PUNCT_RE.sub('', w) for w in words]
    first = sum(1 for w in clean if w in FIRST_PERSON)
    second = sum(1 for w in clean if w in SECOND_PERSON)
    return first, second, len(words)


def load_pronoun_df() -> pd.DataFrame:
    df = load_tagore().dropna(subset=['gregorian_year', 'lyrics']).copy()
    df['gregorian_year'] = df['gregorian_year'].astype(int)
    counts = df['lyrics'].apply(count_pronouns)
    df['first_count'] = [c[0] for c in counts]
    df['second_count'] = [c[1] for c in counts]
    df['n_words'] = [c[2] for c in counts]
    df['log_ratio'] = np.log((df['first_count'] + 0.5) / (df['second_count'] + 0.5))
    df['bin5'] = (df['gregorian_year'] // 5) * 5
    return df


def pooled_ratio(group: pd.DataFrame) -> float:
    s = group['second_count'].sum()
    f = group['first_count'].sum()
    if s == 0:
        return np.nan
    return f / s


def bootstrap_pooled_ratio(group: pd.DataFrame, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(group)
    if n == 0:
        return np.nan, np.nan
    first = group['first_count'].to_numpy()
    second = group['second_count'].to_numpy()
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        s_sum = second[idx].sum()
        samples[i] = first[idx].sum() / s_sum if s_sum > 0 else np.nan
    return np.nanpercentile(samples, 2.5), np.nanpercentile(samples, 97.5)


def bin_summary(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for bin5, group in df.groupby('bin5'):
        if len(group) < 10:
            continue
        ratio = pooled_ratio(group)
        lo, hi = bootstrap_pooled_ratio(group, n_boot=n_boot, seed=int(bin5))
        rows.append({
            'bin5': bin5,
            'n_songs': len(group),
            'first_total': int(group['first_count'].sum()),
            'second_total': int(group['second_count'].sum()),
            'pooled_ratio': ratio,
            'ci_lo': lo,
            'ci_hi': hi,
        })
    return pd.DataFrame(rows).sort_values('bin5').reset_index(drop=True)


def per_category_bins(df: pd.DataFrame, categories: list[str]) -> dict[str, pd.DataFrame]:
    return {cat: bin_summary(df[df['category'] == cat], n_boot=300) for cat in categories}


def plot_panels(overall: pd.DataFrame, per_cat: dict[str, pd.DataFrame], out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ax = axes[0]
    ax.errorbar(
        overall['bin5'], overall['pooled_ratio'],
        yerr=[overall['pooled_ratio'] - overall['ci_lo'], overall['ci_hi'] - overall['pooled_ratio']],
        fmt='o-', capsize=3, color='black', label='All songs (pooled 1st/2nd)',
    )
    ax.axhline(1.0, color='gray', ls='--', lw=0.8, label='Parity (1st = 2nd)')
    ax.set_ylabel('Pooled 1st/2nd pronoun ratio')
    ax.set_title("Tagore's pronoun ratio (1st / 2nd person) by 5-year interval")
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    for _, row in overall.iterrows():
        ax.annotate(f"n={row['n_songs']}", (row['bin5'], row['pooled_ratio']),
                    textcoords='offset points', xytext=(0, 8), fontsize=7, ha='center', color='gray')

    ax = axes[1]
    cmap = plt.colormaps['tab10']
    for i, (cat, table) in enumerate(per_cat.items()):
        if table.empty:
            continue
        ax.plot(table['bin5'], table['pooled_ratio'], 'o-',
                color=cmap(i), label=f'{cat} (n={table["n_songs"].sum()})')
    ax.axhline(1.0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('5-year bin (start year)')
    ax.set_ylabel('Pooled 1st/2nd ratio')
    ax.set_title('By major category')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def temporal_stats(df: pd.DataFrame, label: str) -> dict:
    rho, p = stats.spearmanr(df['gregorian_year'], df['log_ratio'])
    slope, intercept, r_val, p_val, _ = stats.linregress(df['gregorian_year'], df['log_ratio'])
    return {'label': label, 'n': len(df), 'rho': rho, 'rho_p': p,
            'ols_slope_per_year': slope, 'ols_p': p_val, 'pearson_r': r_val}


def partial_correlation_year_logratio_given_category(df: pd.DataFrame, categories: list[str]) -> dict:
    sub = df[df['category'].isin(categories)].copy()
    dummies = pd.get_dummies(sub['category'], drop_first=True).astype(float)

    def residualize(y: pd.Series) -> np.ndarray:
        X = np.column_stack([np.ones(len(sub)), dummies.to_numpy()])
        coef, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
        return y.to_numpy() - X @ coef

    res_year = residualize(sub['gregorian_year'].astype(float))
    res_lr = residualize(sub['log_ratio'])
    rho_partial, p_partial = stats.spearmanr(res_year, res_lr)
    rho_raw, p_raw = stats.spearmanr(sub['gregorian_year'], sub['log_ratio'])
    return {
        'n': len(sub),
        'raw_rho': rho_raw, 'raw_p': p_raw,
        'partial_rho': rho_partial, 'partial_p': p_partial,
        'attenuation_pct': (1 - abs(rho_partial) / abs(rho_raw)) * 100 if rho_raw else np.nan,
    }


def counterfactual_reweighting(df: pd.DataFrame, categories: list[str],
                               early_end: int = 1900, late_start: int = 1920) -> dict:
    sub = df[df['category'].isin(categories)].copy()
    early = sub[sub['gregorian_year'] < early_end]
    late = sub[sub['gregorian_year'] >= late_start]
    if len(early) == 0 or len(late) == 0:
        return {}

    early_mix = early['category'].value_counts(normalize=True)
    late_mix = late['category'].value_counts(normalize=True)

    def pooled_ratio_by_cat(frame: pd.DataFrame) -> dict[str, float]:
        out = {}
        for cat, g in frame.groupby('category'):
            s = g['second_count'].sum()
            out[cat] = (g['first_count'].sum() / s) if s > 0 else np.nan
        return out

    late_cat_ratio = pooled_ratio_by_cat(late)
    early_cat_ratio = pooled_ratio_by_cat(early)

    observed_early = pooled_ratio(early)
    observed_late = pooled_ratio(late)

    # Counterfactual: use late-period within-category ratios, weighted by early-period mix
    cf_late_with_early_mix = sum(early_mix.get(c, 0) * late_cat_ratio.get(c, np.nan)
                                 for c in late_cat_ratio if not np.isnan(late_cat_ratio.get(c, np.nan)))
    # Sanity counterfactual: early ratios weighted by late mix
    cf_early_with_late_mix = sum(late_mix.get(c, 0) * early_cat_ratio.get(c, np.nan)
                                 for c in early_cat_ratio if not np.isnan(early_cat_ratio.get(c, np.nan)))

    total_change = observed_late - observed_early
    within_cat_change = cf_late_with_early_mix - observed_early
    mix_change = observed_late - cf_late_with_early_mix
    pct_within = (within_cat_change / total_change * 100) if total_change else np.nan
    pct_mix = (mix_change / total_change * 100) if total_change else np.nan

    return {
        'early_window': f'<{early_end}',
        'late_window': f'>={late_start}',
        'n_early': len(early), 'n_late': len(late),
        'observed_early_ratio': observed_early,
        'observed_late_ratio': observed_late,
        'counterfactual_late_with_early_mix': cf_late_with_early_mix,
        'counterfactual_early_with_late_mix': cf_early_with_late_mix,
        'total_change': total_change,
        'within_category_component': within_cat_change,
        'mix_component': mix_change,
        'pct_within_category': pct_within,
        'pct_mix': pct_mix,
        'early_mix': early_mix.to_dict(),
        'late_mix': late_mix.to_dict(),
    }


def main():
    df = load_pronoun_df()
    print(f"Songs (with year + lyrics): {len(df)}")
    print(f"Year range: {df['gregorian_year'].min()}–{df['gregorian_year'].max()}")
    print(f"Songs with no 1st-person pronouns: {(df['first_count'] == 0).mean():.1%}")
    print(f"Songs with no 2nd-person pronouns: {(df['second_count'] == 0).mean():.1%}")

    # Pick the major categories: top 5 by song count
    cat_counts = df['category'].value_counts()
    print(f"\n--- Category counts (top 8) ---")
    print(cat_counts.head(8))
    major_cats = cat_counts.head(5).index.tolist()
    print(f"\nUsing major categories: {major_cats}")

    # Overall 5-year bin summary
    print("\n--- 5-year bin summary (overall) ---")
    overall = bin_summary(df, n_boot=1000)
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Per-category bin summary
    per_cat = per_category_bins(df, major_cats)
    print("\n--- 5-year bin summary by category ---")
    for cat, table in per_cat.items():
        print(f"\n  {cat}:")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Plot
    plot_path = HERE / 'pronoun_ratio_5yr.png'
    plot_panels(overall, per_cat, plot_path)
    print(f"\nPlot saved: {plot_path}")

    # Temporal trend stats — overall and per category
    print("\n--- Per-song temporal trend (Spearman year vs log-ratio) ---")
    overall_stats = temporal_stats(df, 'ALL')
    print(f"  ALL: n={overall_stats['n']}, rho={overall_stats['rho']:.4f} (p={overall_stats['rho_p']:.2e}), "
          f"OLS slope={overall_stats['ols_slope_per_year']:.5f}/yr (p={overall_stats['ols_p']:.2e})")
    cat_stats = []
    for cat in major_cats:
        s = temporal_stats(df[df['category'] == cat], cat)
        cat_stats.append(s)
        print(f"  {cat}: n={s['n']}, rho={s['rho']:.4f} (p={s['rho_p']:.2e}), "
              f"OLS slope={s['ols_slope_per_year']:.5f}/yr (p={s['ols_p']:.2e})")

    # Decomposition: partial correlation
    print("\n--- Decomposition: is the trend explained by category shift? ---")
    partial = partial_correlation_year_logratio_given_category(df, major_cats)
    print(f"  Subsample: top 5 categories, n={partial['n']}")
    print(f"  Raw Spearman year vs log-ratio:    rho={partial['raw_rho']:.4f} (p={partial['raw_p']:.2e})")
    print(f"  Partial (residualized on category): rho={partial['partial_rho']:.4f} (p={partial['partial_p']:.2e})")
    print(f"  Attenuation: {partial['attenuation_pct']:.1f}% of the |rho| absorbed by category")

    # Decomposition: counterfactual reweighting
    print("\n--- Counterfactual reweighting (early < 1900 vs late >= 1920) ---")
    cf = counterfactual_reweighting(df, major_cats)
    print(f"  Early ({cf['early_window']}, n={cf['n_early']}) observed pooled ratio: {cf['observed_early_ratio']:.3f}")
    print(f"  Late  ({cf['late_window']}, n={cf['n_late']}) observed pooled ratio: {cf['observed_late_ratio']:.3f}")
    print(f"  Counterfactual late-period ratio if early-period category mix held: {cf['counterfactual_late_with_early_mix']:.3f}")
    print(f"  Total change:               {cf['total_change']:+.3f}")
    print(f"  Within-category component:  {cf['within_category_component']:+.3f}  ({cf['pct_within_category']:.1f}% of total)")
    print(f"  Category-mix component:     {cf['mix_component']:+.3f}  ({cf['pct_mix']:.1f}% of total)")
    print(f"  Early category mix: " + ', '.join(f"{k}={v:.2f}" for k, v in cf['early_mix'].items()))
    print(f"  Late  category mix: " + ', '.join(f"{k}={v:.2f}" for k, v in cf['late_mix'].items()))

    # Persist a brief decomposition.txt for the agent's log
    decomp_path = HERE / 'decomposition.txt'
    with open(decomp_path, 'w', encoding='utf-8') as f:
        f.write("Pronoun ratio (1st/2nd) — temporal decomposition\n")
        f.write("=" * 60 + "\n")
        f.write(f"Overall raw Spearman: rho={overall_stats['rho']:.4f}, p={overall_stats['rho_p']:.2e}\n")
        f.write(f"Partial Spearman (category-controlled): rho={partial['partial_rho']:.4f}, "
                f"p={partial['partial_p']:.2e} (attenuation {partial['attenuation_pct']:.1f}%)\n\n")
        for s in cat_stats:
            f.write(f"  {s['label']}: rho={s['rho']:.4f} (p={s['rho_p']:.2e}), n={s['n']}\n")
        f.write("\nCounterfactual reweighting:\n")
        f.write(f"  observed_early={cf['observed_early_ratio']:.3f}\n")
        f.write(f"  observed_late={cf['observed_late_ratio']:.3f}\n")
        f.write(f"  cf_late_with_early_mix={cf['counterfactual_late_with_early_mix']:.3f}\n")
        f.write(f"  pct_within_category={cf['pct_within_category']:.1f}\n")
        f.write(f"  pct_mix={cf['pct_mix']:.1f}\n")
    print(f"\nDecomposition written: {decomp_path}")

    # RESULT block
    print("\n=== RESULT ===")
    print("hypothesis: The 1st/2nd pronoun ratio rose over Tagore's career; quantify temporal strength "
          "and how much is within-category drift vs category-mix shift.")
    print("method: 5-year pooled ratio bins with bootstrap CIs; per-song log-ratio Spearman/OLS overall and "
          "per major category; partial correlation residualizing log-ratio and year on category dummies; "
          "counterfactual reweighting (late-period within-category ratios held, early-period category mix applied).")
    print(f"key_finding: overall rho={overall_stats['rho']:.3f} (p={overall_stats['rho_p']:.2e}); "
          f"partial rho={partial['partial_rho']:.3f} after controlling for category "
          f"(attenuation {partial['attenuation_pct']:.1f}%); "
          f"observed pooled ratio early→late: {cf['observed_early_ratio']:.2f}→{cf['observed_late_ratio']:.2f}; "
          f"within-category component {cf['pct_within_category']:.0f}%, mix component {cf['pct_mix']:.0f}%.")
    print(f"statistical_significance: per-song rho p={overall_stats['rho_p']:.2e}; "
          f"partial p={partial['partial_p']:.2e}.")
    print("conclusion: see README.md hypotheses and the printed per-category trends; "
          "interpretation depends on the within-vs-mix split reported above.")
    print("=== END RESULT ===")


if __name__ == '__main__':
    main()
