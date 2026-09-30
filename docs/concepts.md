# Concepts

What the indicators mean, which one the Standard prefers, and where the package
goes beyond it. Aligned to the IAAO **Standard on Ratio Studies, September
2026** — a complete revision of the 2013 Standard.

---

## The ratio

$$\mathrm{ASR} = \frac{\text{assessed value}}{\text{time-adjusted sale price}}$$

Two things the Standard is emphatic about, and both constrain how results may
be used:

- **Market value cannot be observed directly** (§3.1). A single sale is one
  observation of it, not market value itself. Ratio study statistics describe a
  *population* and **cannot judge an individual property** — though they may be
  used to adjust an appealed assessment to the common level (§3.3).
- **Validity rests on representativeness** (§5.3). The sample must represent the
  population, and sold and unsold parcels must be valued the same way. Where
  they are not — *sales chasing* — the study reports uniformity that does not
  exist.

## The three questions

| Question | Concept | Indicators | Standard |
|---|---|---|---|
| Are we assessing at the right **level**? | Level of valuation | Median, Mean, Weighted Mean | §7 |
| Are similar properties treated **alike**? | Uniformity / horizontal equity | COD, COV, CoC, FSD | §8.1 |
| Are cheap and expensive properties treated **alike**? | Vertical equity | **VEI**, PRB, PRD | §8.2 |

### Order of operations

§8.1 and §8.2 are explicit: horizontal equity comes first. *"If equity is
achieved between key property criteria used in the appraisal process … significant
vertical inequities are highly unlikely."* Resolve inequity between
neighbourhoods, grades and age groups **before** running a formal vertical
equity analysis — and treat a vertical equity result as unreliable if the COD
standard has not been met, because wide dispersion widens confidence intervals
and masks inequity.

---

## Level of valuation

| Measure | Weights | Use |
|---|---|---|
| **Median** | each property equally | The preferred measure (§7.4): unaffected by extreme ratios |
| **Mean** | each property equally | Only after outliers have been properly trimmed |
| **Weighted mean** | each dollar equally | Indirect equalization (§13.4), where total value is the goal |

**Standard:** 0.90 to 1.10 for any class, target 1.00 (§11.1).

Between strata, two further tests apply (§11.2.1): each stratum within **5
percent** — not 5 percentage points — of the overall level, and classes not
provably differing from each other by more than **10 percent**.

## Uniformity

The **COD** is the recommended measure (§8.1.1): the average absolute percentage
deviation from the median ratio. It assumes nothing about the distribution,
which is why it survives the non-normality typical of ratio samples.

Table 11.1 gives ranges, not ceilings:

| Property category | Profile | COD range |
|---|---|---|
| Residential improved | general | 5.0 – 15.0 |
| | newer / homogeneous, condominiums | 4.0 – 10.0 |
| | rural, seasonal, manufactured homes | 10.0 – 20.0 |
| Multi-unit apartments | general | 7.0 – 20.0 |
| Commercial / industrial / retail | general | 10.0 – 25.0 |
| Vacant land | general | 10.0 – 20.0 |
| Agricultural land | | 10.0 – 25.0 |

Table 11.2 widens these substantially for blighted, distressed and emerging
markets.

> **A COD below the range is a warning, not a win** (§11.2.5). It points to
> sales chasing, model overfitting, over-trimming, or a non-representative
> sample. Screen before celebrating — see [sales chasing](#sales-chasing).

## Vertical equity

### The measurement problem

Vertical equity measures relate the ratio to market value — but market value is
unknown. Use **sale price** as the proxy and you create a spurious correlation:
the price is already the ratio's denominator, so a property that happens to sell
high gets both a raised price and a lowered ratio. That biases the analysis
**toward regressivity**. Substituting assessed value tilts it the other way.

The Standard's remedy, used by both the VEI and the PRB, weights the two
equally so the biases partly offset (Appendix E.1):

$$\text{proxy} = 0.5 \times SP + 0.5 \times \frac{AV}{\widetilde{ASR}}$$

`mkt_value_proxy_header` overrides this blend for the VEI; `prb_mvp_header`
overrides it for the PRB regression specifically. The two are independent —
a study can supply one, the other, both, or neither.

### The three measures with compliance ranges

| Measure | Range | Built-in test | Visualisation |
|---|---|---|---|
| **VEI** | −10% to +10% | yes | yes |
| **PRB** | −0.05 to +0.05 | yes | yes |
| **PRD** | 0.98 to 1.03 | no | no |

**The VEI is the preferred measure** (§8.2.1). It compares the median ratio of
the highest-value quantile group against the lowest, scaled by the overall
median. Group counts follow Table E.3.1 — halves for 20–50 sales, quartiles for
51–500, deciles for 501 or more; undefined below 20.

**Negative is regressive, positive is progressive.**

> **A note on the published Standard.** Appendix E.3, the worked examples in E.6
> and E.8, and the outcome table in E.7 all state that a *negative* VEI
> indicates regressivity. Table E.2.1 and the glossary state the opposite. The
> arithmetic settles it — E.3 is right, and that is the convention this package
> follows.

**The PRD is being retired.** §11.2.6 carries the footnote that this is the last
edition in which it counts for compliance. It has no built-in test, no
visualisation, is biased toward regressivity and is distorted by outliers. Keep
reporting it through the transition; stop relying on it.

### Appendix F: the supporting tests

None carries an IAAO compliance range, and the Standard recommends running
several because each has different blind spots.

| Test | What it adds |
|---|---|
| Modified Kakwani (MKI) / Kakwani | Gini-based: inequality of assessments vs inequality of prices. Equity at 1.00 |
| Reranking Index (RRI) | How much assessment *reorders* properties relative to price — horizontal, not vertical |
| Mann-Whitney | Two groups valued at equal percentages of market value? |
| Kruskal-Wallis | The same across three or more groups (Appendix E.4) |
| Spearman rank | Correlation of ratio rank with value rank |
| Instrumental variables | Clapp (1990); corrects measurement error directly |

The package implements all but the IV regression, and shows the first five in a
strip beneath the Vertical Equity Panel's gauges.

### Beyond the Standard

Two measures in this package are **not** IAAO measures, and are labelled as such
wherever they appear:

- **VEM** — a moving median of the ratio across the whole value range, with a
  confidence band. It exists because the VEI reads only the extremes and the PRB
  fits a single slope; neither describes what happens in between. Reports
  severity, prevalence and a direction.
- **ASR predictability** — fits a gradient-boosted model of the ratio on
  property characteristics. If the ratio were equitable, characteristics should
  not predict it; whatever the model *can* predict is inequity, and SHAP says
  which characteristics carry it. Follows Bidanset & Wehrli (2025) and Wehrli
  (2026).

---

## Confidence intervals

§4.2.1 requires every study to report them, and §7 makes the reason plain: a
point estimate is *an indication, not proof*. §11.1.2 asks for 90% two-tailed
(or 95% one-tailed) before concluding noncompliance.

The package offers two families:

| Mode | Method | Covers |
|---|---|---|
| `ci_mode='distribution'` (default) | Appendix D closed forms, plus PRB (regression) and PRD (delta method) | Median, Mean, Weighted Mean, COD, PRB, PRD — no minimum-count gate |
| `ci_mode='bootstrap'` | BCa or percentile resampling | Every supported indicator; needs 20+ sales |

Appendix D's estimators: Student's *t* for the mean (D.1), the rank formula for
the median (D.2), the ratio-estimator standard error for the weighted mean
(D.3), Clapp's binomial tables for a median on 30 sales or fewer (D.4), and
Bonett–Seier for the COD (D.6). D.8 adds a chi-square tolerance factor for
practitioners without bootstrap tooling. PRB and PRD aren't in Appendix D:
PRB keeps the interval its own slope regression always builds, and PRD (the
ratio of Mean to Weighted_Mean) gets a delta-method interval built from
those two formulas.

---

## Outlier trimming

Appendix C, in order: **identify**, **scrutinise and correct**, then **trim only
what would compromise representativeness**. Never trim to improve the numbers.

- Multipliers: **1.5 × IQR** (≈5% of ratios) or **3.0 × IQR** (≈1%). Which is
  right depends on how well the sales were screened before they reached you.
- Bounds are computed on **log(ASR)** by default, because raw-scale quartiles
  catch low outliers and miss high ones (C.1).
- Caps: no more than **5–10%** of a large sample; 10% (20% in extreme cases)
  below 30 sales. A **single pass**, never repeated rounds (C.5).
- Statistics from a trimmed distribution are **not comparable** with untrimmed
  ones, and the report must say what was removed and why (C.4, C.6).

---

## Sales chasing

Reappraising a property because it sold. It breaks the study in the most
dangerous direction: everything looks *better*. Appendix B asks for at least one
detection technique to be implemented.

| Technique | Needs | Compares |
|---|---|---|
| B.1 value change | prior + current value, sold flag | Year-over-year change, sold vs unsold |
| B.2 unit values | value, size, sold flag | Value per unit, sold vs unsold |
| B.3 split sample | sale date + cutoff | Sales the assessor could see vs could not |
| B.4 concentration | ratios only | Share within ±2% of centre vs Table B.4.1 |
| B.5 / B.6 | an independent model value | Assessment vs an independent estimate |

B.1 sets a tolerance on top of the p-value: on a large sample a trivial
difference is statistically significant, so both statistical *and* practical
significance must clear before anything is flagged.

If a screen fires, §5.3.1 gives the remedy — redo the study on a sample period
that precludes chasing, typically sales after the last date values could change,
time-adjusted back to the valuation date.

---

## Reporting

§10.1.2 lists what the exhibits must carry, per stratum: the tax year, parcel
and sale counts, the number trimmed, level, uniformity, price-related biases,
and confidence intervals. §4.7 adds the narrative — purpose, data sources,
methodology, sales period, adjustments, trimming rule, and the proper use of
results — and recommends the reader be able to *replicate* the study.

---

## References

- IAAO. 2026. *Standard on Ratio Studies.* Kansas City, MO: IAAO
- Gloudemans, R. & Almy, R. 2011. *Fundamentals of Mass Appraisal.* IAAO
- Quintos, C. 2020, 2021. *Journal of Property Tax Assessment & Administration* 17(2), 18(2)
- Bonett, D.G. & Seier, E. 2006. *Biometrical Journal* 48(1)
- Clapp, J.M. 1989, 1990. *Property Tax Journal* 8(3); *JREFE* 3
- Efron, B. & Tibshirani, R.J. 1993. *An Introduction to the Bootstrap*
- IAAO Statistical Tools and Measures Task Force. 2023. *JPTAA* 20(2)
