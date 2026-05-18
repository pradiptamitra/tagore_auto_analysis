# exp027 — Pronoun Ratio Evolution: Temporal Trend & Category Confounding

**Forks from:** `experiments/experiment_027.py`

## Background

Experiment 027 found that the per-song first-person rate drifted upward over Tagore's career (Spearman ρ=0.149, p=7.6e-13) and that, viewed by decade, the **1st/2nd person pronoun ratio flipped from 0.85 (1880s) to 2.44 (1910s)**. That headline number averages over decades, which is dramatic but coarse, and it doesn't tell us *why* the shift happened. Tagore's category mix also changed over time (Exp 008, 034 — Puja proportion in particular collapsed around 1920), so the apparent pronoun shift could be partially or wholly an artifact of writing more love/nature songs (1st-person heavy) and fewer Puja songs (2nd-person heavy).

## Intent

1. **Higher-resolution temporal view.** Plot the 1st/2nd pronoun ratio in 5-year intervals (rather than decades) so we can see the shape of the trajectory and any inflections.
2. **Quantify the temporal trend strength.** Report Spearman correlations and a regression slope on per-song log-ratio against year, plus a Mann-Kendall-style trend on the binned series. Confidence intervals via bootstrap.
3. **Per-category breakdown.** Plot the same 5-year ratio curve for each major category (Puja, Prem, Prakriti, Drama, and the nature-love hybrid). If the ratio shift exists *within* every category, it is genuine. If it exists only between categories, it is a category-mix artifact.
4. **Decompose: how much of the overall shift is "category mix" vs "within-category drift"?** Two complementary methods:
   - **Partial Spearman**: residualize per-song log-ratio on category dummies, then correlate residuals with year. The remaining ρ is the within-category temporal effect.
   - **Counterfactual reweighting**: hold the early-period category mix constant and recompute the late-period ratio. The gap between observed and counterfactual quantifies the category-mix contribution.

## Methods

- Pronoun lists from Exp 027 (modern + archaic 1st/2nd person forms).
- Per-song metrics: `first_count`, `second_count`, `log_ratio = log((first+0.5)/(second+0.5))` (smoothed to handle zero counts).
- 5-year bins: pooled ratio per bin = `Σ first / Σ second` across all songs in the bin (token-level pooling, more stable than mean-of-per-song-ratios with skewed counts).
- Tests:
  - Spearman ρ for `year` vs `log_ratio` (overall and per-category).
  - OLS regression of `log_ratio` on `year` and on `year + category` to quantify within-category slope.
  - Bootstrap (n=1000) 95% CI for the binned pooled ratio.
- Outputs:
  - `pronoun_ratio_5yr.png` — overall + per-category 5-year ratio plot.
  - `decomposition.txt` — partial correlation and counterfactual reweighting numbers.
  - Structured RESULT block at end of run.

## Hypotheses

- **H1**: The 1st/2nd ratio rises monotonically over time, but the trajectory in 5-year bins shows the rise is concentrated in 1900–1920 (matching the grief / post-Nobel period and the Puja-proportion collapse), not gradual across the whole career.
- **H2**: The within-category temporal trend exists but is **weaker** than the raw trend — i.e., a meaningful chunk of the apparent shift is driven by Tagore writing fewer Puja songs (where 2nd-person dominates) and more Prem/Prakriti.
- **H3**: Even controlling for category, the trend remains statistically significant — the introspective "I"-turn is a genuine stylistic evolution, not just a topical reshuffle.
