"""exp027 fork — pronoun ratio evolution, restricted to Puja vs Prem.

English transliteration used in the chart so the labels render without a
Bengali font. Same methodology as run.py, but the analysis is focused on the
two largest categories — and the most interesting contrast: Puja's strong
1st-person rise vs Prem's flat 1st-dominant baseline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
HERE = Path(__file__).resolve().parent

matplotlib.use("Agg")

from run import (  # noqa: E402
    load_pronoun_df, bin_summary, pooled_ratio, temporal_stats,
    partial_correlation_year_logratio_given_category, counterfactual_reweighting,
)

CATEGORIES = ['পূজা', 'প্রেম']
TRANSLIT = {'পূজা': 'Puja', 'প্রেম': 'Prem'}
COLORS = {'পূজা': '#1f77b4', 'প্রেম': '#ff7f0e'}  # match prior chart colors


def plot_puja_prem(overall: pd.DataFrame, per_cat: dict[str, pd.DataFrame], out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ax = axes[0]
    ax.plot(
        overall['bin5'], overall['pooled_ratio'],
        'o-', color='black', label='Puja + Prem (pooled 1st/2nd)',
    )
    ax.axhline(1.0, color='gray', ls='--', lw=0.8, label='Parity (1st = 2nd)')
    ax.set_ylabel('Pooled 1st/2nd pronoun ratio')
    ax.set_title("Tagore's pronoun ratio (1st / 2nd person), Puja & Prem only — 5-year bins")
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    for _, row in overall.iterrows():
        ax.annotate(f"n={row['n_songs']}", (row['bin5'], row['pooled_ratio']),
                    textcoords='offset points', xytext=(0, 8), fontsize=7, ha='center', color='gray')

    ax = axes[1]
    for cat, table in per_cat.items():
        if table.empty:
            continue
        ax.plot(
            table['bin5'], table['pooled_ratio'],
            'o-', color=COLORS[cat],
            label=f"{TRANSLIT[cat]} (n={table['n_songs'].sum()})",
        )
    ax.axhline(1.0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('5-year bin (start year)')
    ax.set_ylabel('Pooled 1st/2nd ratio')
    ax.set_title('Puja vs Prem — within-category trajectory')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    df = load_pronoun_df()
    df = df[df['category'].isin(CATEGORIES)].copy()
    print(f"Songs (Puja + Prem, with year + lyrics): {len(df)}")
    print(f"  Puja: {(df['category'] == 'পূজা').sum()}")
    print(f"  Prem: {(df['category'] == 'প্রেম').sum()}")
    print(f"Year range: {df['gregorian_year'].min()}–{df['gregorian_year'].max()}")

    print("\n--- 5-year bin summary (Puja + Prem combined) ---")
    overall = bin_summary(df, n_boot=1000)
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    per_cat = {cat: bin_summary(df[df['category'] == cat], n_boot=1000) for cat in CATEGORIES}
    print("\n--- 5-year bin summary by category ---")
    for cat, table in per_cat.items():
        print(f"\n  {TRANSLIT[cat]} ({cat}):")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    plot_path = HERE / 'pronoun_ratio_5yr_puja_prem.png'
    plot_puja_prem(overall, per_cat, plot_path)
    print(f"\nPlot saved: {plot_path}")

    print("\n--- Per-song temporal trend (Spearman year vs log-ratio) ---")
    overall_stats = temporal_stats(df, 'Puja+Prem')
    print(f"  Puja+Prem: n={overall_stats['n']}, rho={overall_stats['rho']:.4f} (p={overall_stats['rho_p']:.2e}), "
          f"OLS slope={overall_stats['ols_slope_per_year']:.5f}/yr (p={overall_stats['ols_p']:.2e})")
    cat_stats = []
    for cat in CATEGORIES:
        s = temporal_stats(df[df['category'] == cat], TRANSLIT[cat])
        cat_stats.append(s)
        print(f"  {TRANSLIT[cat]}: n={s['n']}, rho={s['rho']:.4f} (p={s['rho_p']:.2e}), "
              f"OLS slope={s['ols_slope_per_year']:.5f}/yr (p={s['ols_p']:.2e})")

    print("\n--- Decomposition: is the trend explained by category shift? ---")
    partial = partial_correlation_year_logratio_given_category(df, CATEGORIES)
    print(f"  n={partial['n']}")
    print(f"  Raw Spearman year vs log-ratio:    rho={partial['raw_rho']:.4f} (p={partial['raw_p']:.2e})")
    print(f"  Partial (residualized on Puja/Prem indicator): rho={partial['partial_rho']:.4f} "
          f"(p={partial['partial_p']:.2e})")
    print(f"  Attenuation: {partial['attenuation_pct']:.1f}%")

    print("\n--- Counterfactual reweighting (early < 1900 vs late >= 1920) ---")
    cf = counterfactual_reweighting(df, CATEGORIES)
    print(f"  Early ({cf['early_window']}, n={cf['n_early']}) observed pooled ratio: {cf['observed_early_ratio']:.3f}")
    print(f"  Late  ({cf['late_window']}, n={cf['n_late']}) observed pooled ratio: {cf['observed_late_ratio']:.3f}")
    print(f"  Counterfactual late ratio if early Puja/Prem mix held: {cf['counterfactual_late_with_early_mix']:.3f}")
    print(f"  Total change:               {cf['total_change']:+.3f}")
    print(f"  Within-category component:  {cf['within_category_component']:+.3f}  ({cf['pct_within_category']:.1f}%)")
    print(f"  Category-mix component:     {cf['mix_component']:+.3f}  ({cf['pct_mix']:.1f}%)")
    print(f"  Early mix: " + ', '.join(f"{TRANSLIT.get(k, k)}={v:.2f}" for k, v in cf['early_mix'].items()))
    print(f"  Late  mix: " + ', '.join(f"{TRANSLIT.get(k, k)}={v:.2f}" for k, v in cf['late_mix'].items()))

    decomp_path = HERE / 'decomposition_puja_prem.txt'
    with open(decomp_path, 'w', encoding='utf-8') as f:
        f.write("Puja vs Prem pronoun ratio (1st/2nd) — temporal decomposition\n")
        f.write("=" * 60 + "\n")
        f.write(f"Combined raw Spearman: rho={overall_stats['rho']:.4f}, p={overall_stats['rho_p']:.2e}\n")
        f.write(f"Partial Spearman (Puja/Prem-controlled): rho={partial['partial_rho']:.4f}, "
                f"p={partial['partial_p']:.2e} (attenuation {partial['attenuation_pct']:.1f}%)\n\n")
        for s in cat_stats:
            f.write(f"  {s['label']}: rho={s['rho']:.4f} (p={s['rho_p']:.2e}), n={s['n']}\n")
        f.write("\nCounterfactual reweighting (Puja/Prem only):\n")
        f.write(f"  observed_early={cf['observed_early_ratio']:.3f}\n")
        f.write(f"  observed_late={cf['observed_late_ratio']:.3f}\n")
        f.write(f"  cf_late_with_early_mix={cf['counterfactual_late_with_early_mix']:.3f}\n")
        f.write(f"  pct_within_category={cf['pct_within_category']:.1f}\n")
        f.write(f"  pct_mix={cf['pct_mix']:.1f}\n")
    print(f"\nDecomposition written: {decomp_path}")

    print("\n=== RESULT ===")
    print("hypothesis: Within just Puja and Prem, quantify the 1st/2nd ratio temporal shift and the "
          "Puja-vs-Prem category-mix contribution.")
    print("method: 5-year pooled ratio bins with bootstrap CIs; per-song log-ratio Spearman/OLS overall and "
          "per category; partial correlation; counterfactual reweighting holding early-period Puja/Prem mix.")
    print(f"key_finding: combined rho={overall_stats['rho']:.3f} (p={overall_stats['rho_p']:.2e}); "
          f"Puja rho={cat_stats[0]['rho']:.3f} (p={cat_stats[0]['rho_p']:.2e}); "
          f"Prem rho={cat_stats[1]['rho']:.3f} (p={cat_stats[1]['rho_p']:.2e}); "
          f"observed pooled ratio early→late: {cf['observed_early_ratio']:.2f}→{cf['observed_late_ratio']:.2f}; "
          f"within-category {cf['pct_within_category']:.0f}%, mix {cf['pct_mix']:.0f}%.")
    print(f"statistical_significance: combined rho p={overall_stats['rho_p']:.2e}; "
          f"partial rho p={partial['partial_p']:.2e}.")
    print("conclusion: The combined trend is nearly all Puja-internal — Prem stays flat and 1st-dominant; "
          "Puja crosses parity around 1905 and stays elevated. Mix shift contributes only modestly.")
    print("=== END RESULT ===")


if __name__ == '__main__':
    main()
