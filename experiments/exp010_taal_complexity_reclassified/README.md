# exp010_taal_complexity_reclassified

**Forks:** Exp 010 (taal trend) + Exp 014 (category control) + Exp 034 (change-point), for the taal dimension only.

## Why

Exp 010/014/024/034 measured taal "simplification" using raw **matra (beat) count**, with Exp 010 defining `simple = ≤8 beats`. That proxy is musicologically wrong:

- **Tritaal / Teentaal** — 16 matra, but `4+4+4+4` symmetric and the single most versatile, accessible taal in Hindustani music. It is **simple**; the ≤8-beat rule called it "complex" (133 songs mislabelled).
- **Jhaptaal** — 10 matra, asymmetric `2+3+2+3`, a "heavy-beating" classical taal. Genuinely **complex**.
- **Teora (Tivra)** — 7 matra, asymmetric `3+2+2`, classical. **Complex**, but the ≤8 rule called it "simple" (91 songs).

`geetabitan.com` (authoritative Rabindra Sangeet source) frames the real evolution as *heavy-beating classical taals (Chautal, Jhaptaal, Dhamar, Ektal) early → short **light** taals (Dadra, Kaharba) late*. The axis is **performance register + vibhag symmetry**, not cycle length.

## Classification (1 = simplest … 5 = most complex)

Grounded in (a) register — light/folk theka vs semi-classical vs dhrupad/khayal "heavy" taal; (b) vibhag symmetry — even divisions simpler than asymmetric; cycle length only a weak tiebreaker. See `TAAL_TABLE` in `run.py` for per-taal scores and rationale. Compound taals → mean of component scores. Free-rhythm (মুক্তছন্দ) excluded, as in Exp 010. Coverage: 1535/1592 songs (96.4%).

Sources: geetabitan.com taal pages; SwarGanga / Artium Academy / Kathak World vibhag references (Teentaal 4+4+4+4 = simplest; Jhaptaal 2+3+2+3 = intermediate; Dhamar 5+2+3+4 = advanced).

## Result — finding CONFIRMED and strengthened

| Metric | OLD (beat count) | NEW (musicology scale) |
|---|---|---|
| Trend vs year (Spearman ρ) | −0.405 (p=9.6e-62) | **−0.527 (p=3.0e-110)** |
| Partial ρ \| category | −0.379 (Exp 014) | −0.439 (p=2.6e-73) |
| Category-mediated | ~7% (Exp 014) | 16.6% |
| Dominant change-point | 1920 / 1913 | **1918** (d=0.93, p=4e-82), 2nd at 1909 |

- **16.4% of songs (251) flip simple/complex label.** Biggest flips: Tritaal (133 songs) complex→simple; Teora (91) simple→complex. These pull in *opposite* temporal directions, yet the trend gets **stronger** — strong evidence the simplification is real, not a measurement artifact.
- Decade profile: mean complexity is **flat ~2.7–2.8 from the 1880s through the 1900s, then collapses sharply** (1.78 in 1910s → 1.20 in 1920s). There is no early-career "complexification" — this revises the Exp 024 narrative.
- **The inflection moves later.** Under the correct scale the dominant break is **~1918** (with a secondary post-bereavement step at 1909), not the 1913 Nobel year that Exp 024/034 emphasised. The break sits in the WWI / knighthood-renunciation (1919) / Visva-Bharati (1921) window.
- Within Puja alone ρ=−0.62 (p=1e-59) — the trend is, if anything, stronger in his most classical category. ~83% of it survives category control.

**Bottom line:** the "great simplification" is robust to a musicologically defensible definition — and is actually *understated* by the beat-count proxy (|ρ| 0.41→0.53). The one narrative correction: the turn is ~1918 and is not preceded by a complexity rise, so the "Nobel-Prize inflection / pre-Nobel complexification" framing of Exp 024 should be softened.

## Run

```bash
uv run python experiments/exp010_taal_complexity_reclassified/run.py
```
