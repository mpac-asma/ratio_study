# Changelog

## [Unreleased]

### Fixed

* **pandas 3 compatibility.** Under pandas 3, string columns get their own
  `str` dtype, which refuses a float. `_add_comments` failed writing a
  de-marked PRB back into the (string) PRB column, which took `ratios` and
  `ratios_report` down with a `TypeError` whenever `descriptive=True` met a
  non-significant PRB; the column is now widened to `object` first, which
  is what older pandas already used. `compare_ratios` tested for
  `dtype == 'object'` to decide a PRB column needed its significance marker
  stripped, so under pandas 3 it silently skipped the step and produced no
  `PRB_diff`; it now asks whether the column is non-numeric. Both changes
  behave identically on pandas 1.x and 2.x.

* **`spatial.global_moran` no longer contradicts itself.** A Moran's I of
  0.006 with p = 0.019 was reported as "no spatial autocorrelation ...
  statistically insignificant" because significance required both
  |z| >= 2.58 and p <= 0.01 without saying so. The reading now judges
  the randomisation p-value at a stated 5 % level, reports magnitude
  (negligible / low / moderate / high) and significance separately, and
  drops the internal "contact your manager" wording for "Action required".
  The last printed line, mislabelled `z_norm`, is now `p_z_norm (one-sided)`.

* **`vem` documentation.** The docstring listed the no-inequity conclusion
  as `'no significant vertical inequity'`; the function returns
  `'acceptable'`. The docstring and the tests that still expected the old
  wording are corrected. (No behaviour change.)

* **Numpy 2.** Removed uses of `ndarray.ptp()` (gone in numpy 2) from the
  test-suite, and `test_prb_slope_is_invariant_to_the_weight_scale` now
  passes the median as `asr_md=` -- it had been passing it positionally
  into `market_value_proxy`.

### Changed

* The example notebook is split into five (`examples/01_basics` to
  `05_vertical_equity`) and runs entirely on the bundled sample dataset.
  The sample's latitude/longitude are random placeholders, and the docs say
  so. `docs/` is regenerated; `.gitlab-ci.yml`, `dist/` and `egg-info/` are
  no longer tracked.

## [0.4.5] – 2026-09-14

> This section is cumulative. Entries were accumulated under one
> `[Unreleased]` heading from 0.3.0 through 0.4.5, before per-version
> headings were kept, and are not split by release.

### Added

* **The market value proxy is one function now, and its blend is an
  argument.** The `0.5 * sale price + 0.5 * (assessment / median ASR)`
  expression was written out five times -- in `prb`, twice in `ratios`,
  once in `rts_calculator` and once in `vertical_equity_panel` -- each with
  its own way of resolving the median. All five now call
  `ratio_study.market_value_proxy()`, which is also exported for callers
  who want to build the proxy themselves. With it comes
  `mvp_sale_price_share` (default `0.5`, so nothing moves unless it is
  set), threaded through `prb`, `ratios`, `ratios_report`,
  `rts_calculator` and `vertical_equity_panel`, and exposed on the MCP
  server's `run_ratio_study`, `generate_ratio_report` and
  `generate_vertical_equity_panel` as the same argument name. Set it to
  `0.3` for a 30/70 proxy that leans onto the assessment where sale prices
  are the noisier of the two measures, `1.0` to regress PRB on the sale
  price alone, or pass it as a percentage (`30`) as the confidence-level
  arguments already allow. It reaches both places the proxy is used -- the
  PRB regression and the VEI grouping -- and is ignored for whichever of
  the two is supplied directly as a column (`prb_mvp_header` /
  `mkt_value_proxy_header`) or an array. Out-of-range values are rejected
  at the call site by `mvp_sale_price_share_as_fraction`, in front of
  `ratios`' error-collecting `try`, so a typo names itself instead of
  coming back as a failed study.

* **`ratio_map` takes a `colour_code_by` scheme, and every map takes
  `dot_size` and `opacity`.** `colour_code_by` picks how the five colours
  are assigned:

  * `'iqr'` (the new default) runs the package's own `iqr_filter` at both
    1.5 and 3 IQRs, so a point the study would trim at 3 IQRs is exactly
    the point the map paints as a far outlier -- grey inside the inner
    fence, light orange / light blue between the fences, red / dark blue
    outside the outer one. `iqr_min_cod` defaults to `0` rather than the
    filter's own `0.06`: that gate exists to stop a tight sample being
    stripped of legitimate sales, and colouring a point removes nothing,
    so left at `0.06` a low-dispersion roll would map entirely grey.
  * `'mad'` is the same shape measured in median absolute deviations
    (`mad_bands`, default `(3.5, 5.0)` -- wider than the IQR cutoffs
    because the MAD is a much smaller unit: on a roughly normal sample the
    IQR is about 2.7 MADs, so 1.5 MADs would band a fifth of an ordinary
    roll as drifting).
  * `'standard'` uses fixed bands as a share of the median
    (`standard_bands`, default `(0.10, 0.25)`), so grey within 10 % of the
    median, light orange / light blue out to 25 %, red / dark blue beyond
    -- the one scheme comparable between jurisdictions.
  * `'quantile'` is the previous scheme (fixed 80 %/120 % tails plus
    adaptive quantile bins), kept for maps already built against it.

  `iqr_bands`, `mad_bands` and `standard_bands` retune the cutoffs without
  a code change, and the legend now carries each band's count, share and
  observed ratio range, with the band name on the tooltip so a dot does
  not have to be matched to a swatch by eye.

  `dot_size` (default `40`, the `radius_scale` the maps always used) and
  `opacity` are both accepted by `ratio_map`, `vem_map` and `local_moran`.
  Raise `dot_size` for a sparse rural roll where the dots vanish at the
  fitted zoom, lower it for a dense urban one where they merge into a blob.
  `opacity` defaults to `0.4` rather than the solid fill the maps used to
  draw: on any real roll the dots overlap, and at 0.4 the stacking reads --
  a patch of two hundred sales is visibly darker than a patch of five, and
  the basemap stays legible underneath. Pass `opacity=1` for the old solid
  dots. Like `mvp_sale_price_share` it accepts a percentage (`40`) as well
  as a fraction, and is range-checked against the argument name rather than
  left to lonboard's trait, whose error names the deck.gl attribute.

  `local_moran` keeps its seven hot/cold-spot classes: those are z-score
  confidence bands (99/95/90 % each way), not a colour-binning choice, so
  cutting them to five would mean dropping the 90 % bands. It takes
  `dot_size` and `opacity` only -- and gained its first tests in the
  process, it had none.

### Changed

* **`local_moran`'s hot and cold spots were inverted, and its 99 % coldspot
  class was swallowing hotspots.** Two independent sign defects, both of
  which change published classifications:

  1. `Z_Value` was `gL.z_sim * (-1)`. `G_Local` reports a *positive* z for a
     point sitting in a cluster of high values, which for an ASR is a
     cluster of over-assessment -- a hotspot. The flip inverted that, so on
     a grid with a planted block of high ratios and a planted block of low
     ones, the high block came back labelled `coldspot 99% confidence` and
     the low block `hotspot 99% confidence`. The flip is gone; `Z_Value` now
     follows `G_Local`'s own convention, and the tooltip's z agrees with
     every textbook.
  2. The classifier's first branch read `z_value <= 2.58 and p_value <= 0.01`
     where it meant `z_value <= -2.58`. Every other cold branch used
     negative bounds, so the minus had simply been dropped. As written, any
     significant point with z anywhere below +2.58 was returned as
     `coldspot 99% confidence`, and because the chain was an if/elif, the
     `hotspot 90%` and `hotspot 95%` classes were unreachable whenever
     p <= 0.01.

  The if/elif chain is replaced by a `MORAN_BANDS` table --
  `(|z| threshold, max p, cold code, hot code)`, tested widest-first -- and
  a module-level `_moran_class()`. The bands are stated once on the
  *absolute* z with the sign applied afterwards, so the two sides cannot
  drift apart again, and the classifier is now reachable from a test with
  constructed `(z, p)` pairs rather than being nested inside the plotting
  function. Stating the bands symmetrically also closes gaps the old chain
  left: a point with, say, z = 3.0 but p = 0.03 matched no branch at all and
  fell to `Error classifying statistic`; it is now reported at the 95 %
  band it actually qualifies for. `Error classifying statistic` is still
  reached where the z and the p genuinely disagree -- the p-value is
  simulated and the z a normal approximation.

  `p_z_sim * 2` is left alone: `p_z_sim` is one-tailed, so doubling it is
  the correct two-tailed conversion and cannot exceed 1. There is now a
  test saying so.

  Also dropped `get_line_color` and `line_width_min_pixels` from the layer:
  it passes `stroked=False`, so no outline is drawn and both were inert.

* **`local_moran`'s "not significant" dots are grey, not black.** Black drew
  the eye to exactly the sales with no finding on them -- usually most of
  the roll. They are now mid grey `(128, 128, 128)`, and the legend swatch
  matches.

  The reason a first attempt at this recoloured nothing is now fixed
  underneath it: each class's label and colour were written out three
  separate times -- once in a `value_label` if-chain, once in the colormap,
  and once more in each hand-built legend button -- so editing one copy left
  the other two behind. (In that attempt the grey landed on the colormap's
  code `0`, the "Error classifying statistic" bucket that is effectively
  never populated, while the legend's change went to `text_color` rather
  than `button_color`, leaving a black swatch with grey-on-black text.) All
  three now derive from one `MORAN_CLASSES` table of `code -> (label, RGB)`,
  so recolouring a class moves the dots, the tooltip and the legend
  together. Legend text colour is derived from each fill's relative
  luminance by a shared `_style_legend_button`, replacing the hardcoded
  "these ones are dark" tuple that `ratio_map`'s legend also had to keep in
  step by hand.

* **Grey dots are drawn underneath the coloured ones.** Grey is almost
  always the bulk of a roll, and in plain draw order it buried the handful
  of dots the map exists to show -- one ordinary sale landing over a red
  neighbourhood hid the only finding on screen. The background bands
  (`BACKGROUND_BANDS`: grey, plus the white "no ratio" bucket, which
  carries no finding either) now go in their own `ScatterplotLayer` ahead
  of the four coloured bands, and deck.gl draws layers in the order the
  `Map` receives them. Two layers rather than sorting the frame, because
  draw order *within* a layer is row order -- an implementation detail, not
  a guarantee. Both carry the same `dot_size` and `opacity`, so the split
  is invisible except in what covers what, and a side with no rows is
  skipped rather than emitted as an empty layer. Under
  `colour_code_by='quantile'`, which has no single neutral colour to hold
  back, the graded middle goes underneath and the fixed 80 %/120 % tails on
  top -- restoring the z-order they had when they were triangles in their
  own layer.

* **`ratio_map` draws circles everywhere, in five colours.** The outlier
  tails were triangles in a separate `PolygonLayer`; they are circles now,
  so `dot_size` means the same thing across the whole map. `n_bins` drops
  from 7 to 5 and applies only to
  `colour_code_by='quantile'` -- the three statistical schemes are
  five-band by construction. The extra pair of shades the seven-bin scale
  spent on the central mass only ever split it finer than anyone reads,
  and made the two bands that matter harder to pick out.
  `spatial._triangle_polygon` is gone with them, and with it `ratio_map`'s
  only direct use of `shapely`.

* **`prb`'s market-value proxy is now overridable, end to end.** `prb`
  gained `market_value_proxy`, replacing its hardcoded 50/50 blend of
  `market_value` and the ASR-adjusted assessment when supplied. Wired
  through as `prb_mvp_header` on `ratios`, `ratios_report`,
  `rts_calculator` and `vertical_equity_panel` -- independent of
  `mkt_value_proxy_header`, which is the VEI's own proxy override and is
  unaffected by this one. On the MCP server, `run_ratio_study`,
  `generate_ratio_report` and `generate_vertical_equity_panel` all take
  the equivalent `prb_mvp_col`, alongside `mkt_value_proxy_col` (the VEI's
  own proxy override) which `generate_ratio_report` and
  `generate_vertical_equity_panel` were missing entirely. Three gaps in
  the initial wiring are fixed alongside it: `ratios()` dropped
  `prb_mvp_header` from the column list it keeps before handing sales off
  to the calculator, so passing it raised `KeyError` for the very column
  it named; both of `ratios()`'s calls into `vertical_equity_panel`
  (population and per-group) silently ignored it, so the panel's own PRB
  gauge never saw the override those callers already had in scope; and
  `vertical_equity_panel`'s own VEM computation looked up
  `mkt_value_proxy_header` as a column name in a two-column frame it had
  just built itself under the name `market_value_header` -- a column that
  is never actually called `mkt_value_proxy_header` -- so supplying that
  argument at all raised `Column '...' not found in DataFrame` regardless
  of `prb_mvp_header`. A fourth: the MCP `generate_vertical_equity_panel`
  builds its `verdicts` numbers from a second, separate `ratios()` call
  that never received either proxy, so the PNG's boxplot and gauges (built
  from the correctly-proxied `vertical_equity_panel()` call right above
  it) could disagree with the numbers reported alongside them -- a VEI of
  -12.7 in the image against -9.6 in `verdicts`, on the same request.

* **`ci_mode='distribution'` now covers `PRB` and `PRD`.** Previously only
  `Median`, `Mean`, `Weighted_Mean` and `CoD` had a closed-form interval,
  so requesting distribution-mode CIs for a study that only computes
  `PRB`/`PRD` (e.g. `indicators=['VEI', 'PRB', 'PRD']`) tripped "no
  indicator supports one" and switched confidence intervals off entirely.
  `PRB` now keeps the interval its own slope regression always builds
  (`prb`'s `ci_conf_level`-driven `beta_ci`), and `PRD` -- the ratio of
  `Mean` to `Weighted_Mean` -- gets a new delta-method interval,
  `ratio_study.confint.prd_ci`, built from the covariance of those two
  already-published formulas (D.1 and D.3). Both are on by default,
  alongside the original four, since neither costs more than a pass over
  the ratios.

* **`ratio_study.sales_chasing`: all six Appendix B sales-chasing detection
  techniques.** Sales chasing does not make a ratio study look bad, it
  makes it look *good* -- chased ratios collapse onto the centre, so CoD
  falls, the median moves to 1.00, and every uniformity statistic stops
  describing the unsold population it is meant to represent. None of the
  ordinary indicators can see it, hence a module of its own.

  Each function returns a dict of the raw statistics plus a boolean
  `flag` and a `conclusion` string, and each applies the appendix's
  **two gates**: a difference must clear both statistical significance
  *and* a practical tolerance (5 percentage points, tightened to 3 when
  the sold group's own change is under 5%), because at scale a
  meaningless difference is always significant.

  * `sc_value_change` (B.1) -- year-over-year percentage change in
    assessed value, sold against unsold. The strongest test in the
    appendix, and the only one that speaks to the mechanism rather than
    to a symptom: no ratio is involved, so it is immune to whatever the
    sale prices are, and it catches chasing on recoding as well as on
    value. Median by default, switching to the mean when a large share
    of unsold parcels had no change at all (their median is then just
    0). `exclude_col` drops the parcels the appendix insists come out
    first -- physical change, use change, appeal/stipulation,
    non-agency values -- since those bias the test *toward* a finding.
  * `sc_unit_values` (B.2) -- value per unit of size, sold against
    unsold. Needs one year of values rather than two, which is often
    the difference between a test that can be run and one that cannot.
  * `sc_split_sample` (B.3) -- two ratio studies either side of the
    date values were locked. Sales before it could have been chased,
    sales after could not. The only technique here an oversight agency
    can run on a sales file alone. Flags only when the early sample is
    directionally *better*, its interval is disjoint, and the gap
    clears tolerance; the conclusion states the appendix's asymmetric
    remedy (reject the early statistic, keep the late one).
  * `sc_concentration` (B.4) -- observed against expected concentration
    of ratios. `critical_concentration` reproduces Table B.4.1
    analytically rather than by lookup (`sigma = 1.25 x CoD`, critical
    percent = `floor(100 x (2 Phi(0.02/sigma) - 1)) + 1`); all ten
    published rows are pinned by tests.
  * `sc_cdf_gap` -- not one of the numbered techniques, but the
    sharpest signal for the crudest form of chasing: assessment set
    equal to sale price piles up *identical* ratios, which is a
    vertical jump in the ECDF. Returns the ratio value and the parcel
    count, so a flag names the records to pull.
  * `sc_mass_appraisal` (B.5) and `sc_independent_values` (B.6) --
    substitute a caller-supplied independent estimate of market value
    for the sale price in the denominator and compare the two studies'
    level and uniformity. **Nothing is modelled here.** B.5's premise
    is that the independent estimate is arrived at independently -- a
    different structure, ideally a different analyst -- which a model
    fitted inside this module from the same frame by the same process
    would satisfy none of. B.5 is one-sided (only a flattering *sales*
    study is evidence of sales chasing); B.6 is two-sided, because an
    oversight agency's proxy values can be chased too and the numbers
    cannot say which side moved. Optional `standards` reports each
    study's compliance, per the appendix's point that this technique
    yields measures that can be tested against the level standards.
  * `detect_sales_chasing` -- runs whichever techniques the supplied
    columns support, within strata when `group_by` is given, and
    returns one tidy frame. A technique that lacks its inputs appears
    as `not run: <reason>` rather than being dropped: "we did not test
    for it" and "we tested and found nothing" are different findings.

  Every test is gated at 30 observations (B.4's floor, applied
  throughout) and returns `flag=None` with a "Sample size too small"
  conclusion below it, rather than a number that cannot be defended.
  A flag is a screen, not proof -- the appendix is explicit that even an
  exceeded critical proportion warrants investigation before concluding
  that chasing occurred, and `detect_sales_chasing` warns in those terms.

  Four departures from `assesspy.is_sales_chased`, which covers part of
  the same ground, each driven by a case where it returns the wrong
  answer:

  * **The window follows the sample's own central tendency.** assesspy
    hardcodes `bounds=(0.98, 1.02)`, but B.4 says "within +-2% of *a
    measure of central tendency*". A stratum assessed at 0.85 with 10%
    of its sales pinned at 0.85 is invisible to it.
  * **The reference proportion comes from `min_achievable_cod`, not
    from the sample.** assesspy builds its reference normal from the
    sample's own standard deviation -- but chasing shrinks that
    standard deviation, so the benchmark moves toward the concentration
    it is meant to detect. B.4's argument is a floor supplied from
    outside.
  * **The concentration test is one-sided.** assesspy takes an absolute
    difference, so a sample *flatter* than its reference also fires.
    That is a dispersion problem, not a chasing problem.
  * **The CDF gap is the largest one *inside* the band.** assesspy
    takes the largest gap anywhere and then checks whether it happens
    to lie in the band, so a single far outlier owns the maximum and
    vetoes a genuine spike at 1.00.

  Uniformity comparisons in B.5/B.6 carry a caveat documented at the
  point of use: independent values have estimation error of their own,
  it lands in the denominator, and it inflates the independent study's
  CoD even when nothing was chased. The CoD gap is therefore measured
  in percentage points rather than relatively (9.3% against 11.6% is
  2.3 points of noise but a 20% relative gap, which would flag on the
  artefact), and a level finding is the more serious of the two.

* **`_add_comments` checks strata uniformity (IAAO 11.2.1) with the
  confidence-interval percent-change test, not a fixed band.** A
  stratum's `Median` used to be checked against the overall `Median`
  +/- 0.05 -- a percentage-point band, not the "5 percent, not 5
  percentage points" the standard specifies, and it never looked at the
  overall's own confidence interval. Now, when both the stratum's and
  the overall's `Median_ci` are available, the two intervals are
  compared at their nearest bounds and the gap is expressed as a
  percent of the overall's bound -- reproducing the standard's worked
  example exactly (a 0.85 median, CI 0.81-0.94, against a 1.01 overall,
  CI 0.98-1.05, is a compliant 4.08% change). Falls back to the fixed
  band when a usable `Median_ci` isn't available on either side.

* **`ci_mode` on `ratios` / `ratios_report`: confidence intervals now come
  in two families.** `ci_conf_level`, `confidence_intervals_aggs` and
  `ci_indicators` mean the same thing in both; only the estimator behind
  each `<indicator>_ci` changes.

  * `ci_mode='bootstrap'` (the default, so nothing moves) resamples, as
    before.
  * `ci_mode='distribution'` uses the closed-form formula the IAAO
    Exposure Draft (May 2026) Appendix D prescribes for each indicator,
    rather than one generic interval: Student's t about the mean (D.1),
    the large-sample rank formula for the median (D.2), the
    ratio-estimator standard error for the weighted mean (D.3), the exact
    binomial ("Clapp 1989") tables for a median on 30 sales or fewer
    (D.4), and Bonett–Seier for the CoD (D.6). Covers `Median`, `Mean`,
    `Weighted_Mean` and `CoD` — `DISTRIBUTION_CI_SUPPORTED_INDICATORS` —
    and nothing else, so `PRD` gets no column.

  Distribution mode carries no minimum-count gate: each formula is
  defined down to its own floor, and the binomial median tables reach 5
  sales (90 %) or 6 (95 %), well below the bootstrap's default threshold
  of 20. `bootstrap_method`, `bootstrap_resamplings`,
  `bootstrap_concurrency` and `ci_seed` are ignored there.

  The two worked examples the draft supplies are reproduced exactly and
  pinned by tests: the D.4 median interval (n=17 at 95 % →
  `(0.6952, 0.9326)`) and the D.7.1 CoD interval (n=54 at 90 % →
  `(0.21878, 0.28618)`).

  Also exported individually for direct use: `mean_ci`, `median_ci_rank`,
  `median_ci_clapp`, `median_ci_distribution`, `weighted_mean_ci`,
  `cod_ci`, `distribution_analysis`, plus
  `DISTRIBUTION_CI_SUPPORTED_INDICATORS`,
  `DISTRIBUTION_CI_DEFAULT_INDICATORS` and
  `MEDIAN_CI_SMALL_SAMPLE_MAX`.

  Three transcription problems in the source tables and formulas had to
  be resolved to make the published examples reproduce; each is
  documented at the point of use.

  * **D.6 step 18** is printed as `ln((n/(n-1)) × τ − (k × z × se2))`.
    The closing parenthesis is misplaced: Bonett–Seier is symmetric on
    the log scale, as the section's own preamble describes, so it must be
    `ln((n/(n-1)) × τ) ∓ (k × z × se2)`. Read literally the printed form
    gives L1 = −2.137 against the example's −1.62354; corrected it gives
    −1.62352.
  * **D.3 step 11** is printed as `(Ā/S̄²)(ΣS²)`. The exponent belongs to
    the whole ratio — `R²·ΣS²` — which is what makes step 12 equal
    `√Σ(A − R·S)²`.
  * **Table D.4.2** has two arithmetic slips: n=28's weights sum to
    .9950, and n=29's upper row prints .3062 against .3063 in its lower
    row. Only the lower row is stored and the upper bound derived as its
    mirror, which keeps the weights exactly complementary and repairs
    both.

* **`vem_conf_level` and `vem_ci_method` on `ratios` / `ratios_report`.**
  The VEM's confidence level was unreachable: `ratios` called `vem()`
  without `conf_level`, pinning it to that function's own 0.90 default,
  while `vertical_equity_panel` exposed `vem_conf_level` separately. A
  study could therefore label a band "90 % CI" in the panel and report
  indicator intervals at some other level, with no way to align them. The
  level is not decoration — it drives `ci_out_of_bounds` and so the VEM
  verdict, since a stretch of the value range only counts as inequitable
  if its whole local-median interval clears
  `vem_acceptable_inequity_level`. Both now thread through to `vem`, to
  the panel, and to the chart legend.

* **The MCP `run_ratio_study` tool exposes the confidence-interval
  options.** It previously had none at all — no `confidence_intervals_aggs`,
  `ci_conf_level`, `ci_indicators` or `ci_seed` — so an LLM driving the server
  could not ask for per-group intervals, change the level, or get
  reproducible ones. All four are now forwarded, along with the new
  `ci_mode` and `vem_conf_level`. Options that mean something specific
  when unset are only forwarded when actually supplied, so "not
  specified" stays distinguishable from an explicit `None`.

  The same tool also passed `ve_median_acceptable_inequity_level` to
  `ratios()`, which has no such parameter, so **every** invocation raised
  `TypeError` before reaching the study. Renamed to the keyword `ratios`
  takes, `vem_acceptable_inequity_level`, and the tool is now verified
  end to end.

* **`Weighted_Mean` is computed whenever it is requested.** It used to be
  set only as a by-product of the Gini block, so
  `indicators=['Weighted_Mean']` produced no column — and, with
  distribution-based CIs, a `Weighted_Mean_ci` with no point estimate to
  sit beside. It is a ratio of two totals, so it costs nothing to compute
  directly.

* **Spatial diagnostics (`ratio_study.spatial`), behind a new `spatial`
  extra: `pip install "ratio_study[spatial]"`.**

  * `spatial.ratio_map(df, column, groupby)` maps median/mean/min/max
    ratio and sale count by neighbourhood. The colour scale is now
    quantile bins fit to the actual distribution of `column` in the
    sample, rather than fixed cutoffs (0.85/0.90/.../1.15) -- a sample
    that skews high or low still gets a colour scale spread across its
    real range instead of one end lit up solid red or blue.
  * `spatial.global_moran(df, ratio)` and `spatial.local_moran(df, ratio)`
    port over unchanged, plus `moran_i()` (below) for the plain
    statistic without the mapping/plotting stack.
  * All three take `lat_col`/`long_col` (default `'latitude'`/
    `'longitude'`), and need a real pandas DataFrame with those columns
    plus geopandas/libpysal/esda/splot/lonboard/ipywidgets installed --
    calling one without the extra raises an `ImportError` naming the
    `pip install` that fixes it, same as the `ml` extra's diagnostics.

* **`moran_i(df, ratio, lat_col='latitude', long_col='longitude')`** in
  `indicators.py` (also exported from the package root) -- Global
  Moran's I as a numeric indicator, independent of the mapping stack in
  `spatial.py`. Requires the `spatial` extra for `libpysal`/`esda`
  (lazily imported, so plain `import ratio_study` never needs it), and
  raises `ValueError` up front if `df` is missing `ratio`, `lat_col`, or
  `long_col`.

* **The bundled sample dataset (`load_sample_dataset()`) gained
  `latitude`/`longitude` columns**, randomly placed around Ottawa with
  each neighbourhood clustered around its own centre, so the spatial
  functions above have something to plot out of the box.

* **`iqr_filter` (and `ratios` / `ratios_report`'s `iqr_trim_multiple`)
  gained `iqr_log_transform`, `iqr_group_by`, and `iqr_min_cod`.**

  ```python
  ratios(sales, 'assessment', 'sale_price', iqr_trim_multiple=3)          # log-scale bounds by default
  ratios(sales, ..., iqr_trim_multiple=3, iqr_group_by='nbhd')            # trim within each neighbourhood
  ratios(sales, ..., iqr_trim_multiple=3, iqr_group_by=[['county', 'nbhd']])  # combined hierarchy key
  ratios(sales, ..., iqr_trim_multiple=3, iqr_min_cod=0)                  # always trim, even tight samples
  ```

  * `iqr_log_transform` (default `True`) computes the trimming bounds on
    `log(ASR)` instead of the raw ratio, since ASR is right-skewed and
    raw-scale quartiles push the upper bound out, making high outliers
    easy to miss.
  * `iqr_group_by` computes the trim independently within groups instead
    of once over the whole sample, in the same shape as `group_by`: a
    column name or list of column names for independent groupings, or a
    list of lists to combine columns into one composite key.
  * `iqr_min_cod` (default `0.06`) skips trimming for a sample (or group)
    whose CoD is already below the threshold, since dispersion that low
    means trimming would only strip legitimate sales rather than genuine
    outliers -- most useful with `iqr_group_by`, where small, already-tight
    groups are the ones most at risk of over-trimming.

* **Kruskal-Wallis alongside the VEI, per Appendix E.4 ("Further
  Analysis") of the Standard on Ratio Studies.** The VEI itself only ever
  compares the first and last quantile groups; Kruskal-Wallis checks
  whether the level of assessment differs across *all* of them. It rides
  along on every VEI result rather than gating the VEI's own conclusion —
  the standard recommends it as supplementary, not as a pass/fail
  criterion.

  * `vei_analysis()` gains a `kruskal_wallis` key (`{'statistic': ...,
    'pvalue': ...}`), `None` below the VEI's own 20-sale minimum.
  * `ratios()` / `ratios_report()` / `rts_calculator()` gain
    `VEI_kw_statistic` and `VEI_kw_pvalue` columns whenever `'VEI'` is
    requested. They reach the CSV/Excel export by default, but not the
    PDF's ratios table — printed-page space is scarce, and
    `pdf_ratios_table_headers` is intersected with `indicators`, which
    these two column names are never a member of.

* **`percentile_method` / `ci_method` default to `'R7'` everywhere.**
  `ratios()` / `ratios_report()` already defaulted to `'R7'` and always
  passed it through explicitly; `rts_calculator()`, `vei_analysis()` and
  `_bootstrap_stacker()` each carried their own independent `'R6'`
  default, live only for a caller bypassing the public entry points. The
  standard's own software-defaults table (Appendix E, Section E.5.3)
  lists Python/numpy as R7 by default; this removes the inconsistency.

* **`ratios` / `ratios_report` gained `confidence_intervals_aggs`, `ci_conf_level`,
  `ci_indicators` and `ci_seed`.**  Asking for confidence intervals no
  longer means assembling a `bootstrap_args` dict:

  ```python
  ratios(sales, 'assessment', 'sale_price')                     # overall CIs, 95%
  ratios(sales, 'assessment', 'sale_price', confidence_intervals_aggs=False)
  ratios(sales, ..., group_by='nbhd', confidence_intervals_aggs='both', ci_conf_level=90)
  ratios(sales, ..., confidence_intervals_aggs=['Median', 'CoD'])
  ratios(sales, ..., ci_seed=7)                                 # reproducible
  ```

  `confidence_intervals_aggs` takes a bool, a scope name (`'none'`, `'overall'`,
  `'groups'`, `'both'`), a list of indicator names, or a `bootstrap_args`
  dict. `ci_conf_level` accepts either `0.9` or `90`. It is applied after
  `bootstrap_args`, so `confidence_intervals_aggs=False` is a reliable
  kill-switch. `bootstrap_args` keeps working unchanged for everything
  else.

* **`bootstrap_args` accepts short keys, and reports the ones it doesn't
  know.**  `{'groups': True, 'level': 0.9, 'resamplings': 500,
  'method': 'percentile', 'seed': 7}` is now equivalent to the
  fully-spelled-out form. An unrecognised key — the `'boostrap_indicators'`
  typo in the package's own usage examples, for one — raises a warning
  naming the closest match and lands in the study's error messages,
  instead of being silently ignored. Out-of-range values (a method that
  isn't BCa or percentile, a resampling count below 10, CIs for an
  indicator that has none, per-group CIs without a `group_by`) are
  reported and fall back rather than failing or producing nothing.

* **`bootstrap_seed` / `ci_seed`: reproducible intervals.**  The resampler
  took no seed, so two identical studies published slightly different
  intervals. It now accepts one, threaded from the public signature
  through `_bootstrap_analysis` to `_bootstrap_stacker`. Unset by default,
  so the previous behaviour is unchanged unless asked for.

* **`_add_comments` gained `add_binaries` (default True): one
  `'<indicator>_compliant'` column per assessment standard.**  For every
  standard naming a column in the ratios table (`'CoD'` ->
  `'CoD_compliant'`), a boolean column reports whether the value sits
  inside the band that applies to its row — the wider CoD band and the
  overall-median comparison for strata included.

  The flags read the point estimate only, so unlike the comments they
  ignore an overlapping confidence interval and are still filled in for
  samples too small to comment on; they are NA (not False) where the
  indicator or its standard is missing. Standards that name no column,
  or whose value is not a numeric pair (`'Mkt_desc'`), get no flag.
  These columns reach `ratios(descriptive=True)` and the CSV/Excel
  exports, but not the PDF table, which selects its own columns.

* **Vertical Equity Panel — new composite chart.**  A single figure that
  puts the four vertical-equity indicators side by side: a row of PRB,
  PRD, VEI and VEM gauges, then the VEI percentile-group boxplot, then
  the VEM moving-median chart, each spanning the full width.  The figure
  is laid out in letter-page proportions (portrait, `figsize=(8.5, 11)`)
  and saved without a tight bounding box, so it keeps those proportions
  and drops onto a report page as is.  The gauges have their own
  gridspec, so they stay centred on the page rather than inheriting the
  wider left margin the charts need for their y-axis labels.

  `include_vem=False` reports vertical equity without a VEM verdict: the
  VEM gauge is dropped and the remaining three are re-centred (at
  unchanged dial size), and the moving-median chart loses the
  acceptable-inequity lines and the green/red split of its scatter — one
  plain blue scatter, with the moving median and its CI band still
  drawn, so the chart still shows how the level of assessment moves
  across the value range.

  Every gauge shares one dial geometry, so the four are comparable at a
  glance: a 45° green wedge on top (within the acceptable band), a 135°
  red wedge on the right (progressivity), a 135° red wedge on the left
  (regressivity), and a 45° grey wedge at the bottom (inconclusive).  The
  value-to-angle scale is piecewise linear, so neutrality always points
  straight up and each band edge lands exactly on a green wedge edge —
  which is what lets PRD's asymmetric `(0.98, 1.03)` band share a dial
  with PRB's symmetric `(-0.10, 0.10)`.  PRD is reflected (a PRD below 1
  is progressivity), so "right means progressive" holds on all four.

  The needle shows where the point estimate falls, so it always agrees
  with the number printed under the dial; only inconclusiveness (a
  non-significant PRB, or a VEI/VEM whose own conclusion is inconclusive)
  moves it to the grey wedge.  Where an indicator's significance test
  reaches a verdict its point estimate does not support, that verdict is
  reported beneath the gauge (e.g. `VEI test: Acceptable` under a
  regressive needle) rather than silently repositioning the needle.

  * Available as `ratio_study.vertical_equity_panel`.  Returns a
    matplotlib `Figure` for inline use in a notebook, a
    `[caption, reportlab Image]` pair with `return_object=True`, or a
    saved filepath with `graph_filepath`.
  * Indicators not passed in are computed from the frame, so the panel
    works standalone; `ratios` passes the population PRD/PRB through so
    the gauges cannot drift from the ratios table.
  * Added to the PDF report with the new
    `pdf_args={'pdf_vertical_equity_panel': True}` option on
    `ratios_report` (also available as `graphs_args` on `ratios`, which
    additionally returns it under `'vertical_equity_panel'`).  Off by
    default.
  * Drawn entirely with matplotlib, so the report gains a static VEM
    chart without invoking Kaleido/Orca — no browser subprocess and no
    macOS default-browser prompt.  The moving-median chart is a static
    counterpart of the Plotly `ve_scat_median_plot`.
  * The market-value tick formatting used by `vei_boxplot` moved into a
    shared `_format_proxy_labels` helper, now used by both charts.

* **`vem` — new vertical-equity indicator (`VEM`).**  A
  band-free indicator built on the moving median of the median-centred
  ASR, replacing `ve_bands` (see **Removed**).  Any stretch of the
  market-value range whose local median sits further than
  `acceptable_inequity_level` from the jurisdiction-wide median counts as
  vertical inequity; the index is the mean absolute moving median over
  those rows, and the direction (Progressivity / Regressivity /
  Inconclusive) is decided by comparing the two halves of the value
  range.

  * Available as `ratio_study.vem`, as the `'VEM'` indicator in
    `ratios` / `rts_calculator` (adding `Inequity_conclusion`,
    `Inequity_severity` and `Inequity_prevalence` columns), and via the
    `run_ratio_study` MCP tool.  `'VEM'` is now part of the default
    `indicators` list.
  * New `ratios` / `rts_calculator` arguments:
    `ve_median_acceptable_inequity_level`, `ve_median_plot`,
    `ve_median_title`; the plot reference label is `ve_plot_ref`.
  * New chart `ratio_study.graphs.ve_scat_median_plot`.

### Removed

* **`ve_bands` and its chart have been removed — breaking change.**  The
  moving-median indicator (`VEM` / `ve_median`) and the Vertical Equity
  Panel supersede them, so the band machinery no longer earns its keep.

  * Gone from the public API: `ratio_study.ve_bands`,
    `ratio_study.indicators.ve_bands` and `ve_scat_bands_plot` (both the
    `graphs` chart and the unused `reporting` copy).  Removed from
    `__all__`.
  * Gone from `graphs.py`: the internal helpers `_compute_bands`,
    `_select_masked` and `_plotly_to_mpl`, which had no other callers.
    Dropping `_plotly_to_mpl` also drops `reporting.py`'s last
    `plotly.graph_objects` dependency.
  * The `'VE_Bands'` indicator is no longer accepted by `ratios` /
    `rts_calculator`, and the `VE_Bands_index` / `VE_Bands_direction`
    result columns are gone.
  * Removed `ratios` / `rts_calculator` arguments:
    `ve_bands_acceptable_inequity_level`, `ve_bands_plot` and
    `ve_bands_title`.
  * Removed the `ve_bands_acceptable_inequity_level` argument from the
    `run_ratio_study` MCP tool, which had continued to forward it to
    `ratios` and so failed with `TypeError` on every call.

  **Migrating:** request `'VEM'` instead of `'VE_Bands'`, read
  `Inequity_conclusion` / `Inequity_severity` / `Inequity_prevalence`
  instead of `VE_Bands_direction` / `VE_Bands_index`, pass
  `ve_median_acceptable_inequity_level` instead of
  `ve_bands_acceptable_inequity_level`, and use `ve_scat_median_plot` or
  `vertical_equity_panel` in place of `ve_scat_bands_plot`.

### Fixed

* **`presets.KNOWN_INDICATORS` spelled `Weighted_Mean` in lower case and
  omitted `VEM`,** so `RatioStudyConfig(indicators=['Weighted_Mean'])` --
  the name `ratios` actually reads -- was rejected as unknown, as was the
  package's own `DEFAULT_INDICATORS`.

* **Every confidence level now goes through one gate, and a significance
  level passed by mistake is rejected instead of honoured.**
  `confidence_level_as_fraction` (the old private
  `_ci_level_as_fraction`, still aliased) gained a floor of
  `CI_LEVEL_FLOOR = 0.5`: `ci_conf_level=0.05` used to build a 5 % interval in
  silence, which is indistinguishable in the output from a very precise
  study — and `req_significance=0.05` sits in the same call signature.
  The error names the argument, the value they probably meant, and where
  `0.05` does belong.

  The gate is now the only validator. `ci_conf_level=90` worked while
  `vei_conf_level=90` raised, because the vertical-equity levels checked
  `0 < x < 1` directly; `compute_median_ci` and `log_median_diff_ci`
  never validated at all, so a percentage produced a negative alpha and a
  `NaN` crash. All of them route through it, as do
  `BootstrapConfig.confint_level`, `RatioStudyConfig.vei_conf_level` and
  the new `vem_conf_level`, so the fraction and percentage spellings are
  interchangeable everywhere a level is accepted.

  `ratios` validates its levels *before* its try block, alongside
  `normalize_ci_args`. Downstream they were swallowed into an error dict,
  so a mistyped level came back as a failed study rather than a message
  saying what was wrong. `req_significance` is untouched — it is a
  significance level and keeps its own `(0, 1)` bound.

* **`RatioStudyConfig` can now express the settings it was missing:**
  `vem_conf_level`, `vem_ci_method` and `vei_percentile_method`. The
  model is `extra="forbid"`, so a preset previously could not carry them
  at all. `BootstrapConfig.min_count_threshold` also has its floor raised
  from 1 to 4, matching what `normalize_ci_args` actually enforces — a
  validated 1 was silently becoming 4.

* **`Inequity_severity` and `Inequity_prevalence` now measure the
  *confirmed* windows.** Severity is the average percent deviation of the
  moving-median ASR from the overall median ASR, over the stretches where
  the moving median **and its whole confidence band** sit outside the
  acceptable band. It previously used the point estimate alone, so a
  window whose band still overlapped the acceptable range contributed to
  the reported severity even though it could not be demonstrated.

  Prevalence moved onto the same mask. The two are read together — "44 %
  of the value range, averaging 15.8 points" — so measuring them over
  different sets of rows made that sentence untrue.

  The verdict states stay distinct, which the redefinition alone would
  have collapsed: nothing outside the band at all is "no significant
  vertical inequity", outside on the point estimate but never confirmed is
  'Inconclusive' (severity 0), and confirmed above the threshold names a
  direction.

  `Inequity_ci_severity` is gone — `Inequity_severity` now *is* that
  statistic, and publishing both was duplication. Nothing outside
  `indicators.py` read it.

  `max_acceptable_severity` now defaults to None, meaning the acceptable
  band itself (`acceptable_inequity_level * 100`), so out of the box any
  confirmed excursion is reported. Raising it above the band is what makes
  the gate bind: a 5 % band with `max_acceptable_severity=7` reports only
  once the confirmed excursions average more than 7 points.

* **`mean_ci` and `weighted_mean_ci` clamp their lower bound at zero.** An
  ASR is a ratio of positive amounts, so it cannot be negative — but both
  of these are symmetric in value, unlike the order-statistic median
  intervals whose bounds are drawn from the sample. On a small, sharply
  right-skewed sample the margin can exceed the estimate and publish an
  impossible negative ratio. The upper bound is left alone: an ASR has no
  ceiling.

* **Comments now name the level they were decided at, and say when a
  breach was left unconfirmed.** `_add_comments` takes `ci_conf_level` and uses
  it to label every comment settled by an interval rather than by a
  p-value; `req_significance` still labels the significance tests, and the
  two are no longer conflated in the output.

  More importantly, a value outside its standard whose interval still
  reached back inside produced **no comment at all** — which in a report
  reads as a pass. A median of 0.8446 against a (0.90, 1.10) standard came
  back silent. It now says so explicitly: *"Median is outside assessment
  standards (0.9, 1.1), but its 95% confidence interval still reaches
  them, so the breach is not confirmed"*. The compliance marker is still
  withheld, since the breach genuinely is unconfirmed; only the silence is
  fixed. A confirmed breach likewise now states that its interval is
  entirely outside the standard.

* **The VEM's severity, prevalence and CI-severity indices were on two
  different scales, and the gate between them on a third.**
  `severity_index` was multiplied by 100 in the banded branch and not in
  the other, `ci_severity_index` was left as a fraction, and
  `max_acceptable_severity` was written as `0.01` — a fraction — while
  being compared against a severity in percentage points. The threshold
  was therefore never binding. All four are now in percentage points of
  the ASR, with the default restated as `1.0`.

  `prevalence_index` was also only assigned inside
  `if acceptable_inequity_level`, so a falsy band raised `NameError` on
  the way to the return dict. It is now defined in both branches.

  One caveat is documented rather than silently changed: because
  `severity_index` averages |moving median| over rows that are *already*
  outside the band, it is at least `acceptable_inequity_level * 100`
  whenever it is non-zero, so any threshold below that is equivalent to
  "are there any out-of-bounds windows at all" and the CI check does the
  real discrimination. Making the gate bind on its own would mean
  measuring the mean *excess beyond* the band instead, which changes what
  the published number means.

* **A dead `confidence_level_tail` in `_bootstrap_analysis`** suggested
  the level was applied at that layer; `_bca_ci` and `_percentile_ci`
  each derive their own tail. Removed.

* **`compute_median_ci(method='R6'|'R7')` returned the spread of the
  sample, not a confidence interval for its median.** The branch passed
  `alpha/2` into the Hyndman-Fan position formula, so a 90 % "interval"
  was the 5th-to-95th percentile of the ratios. Two symptoms: the width
  did not shrink as the sample grew (~0.61 at n=100 and ~0.66 at n=1600
  on the same distribution, against a true median-CI width falling from
  0.080 to 0.021), and `achieved_coverage` came back as 1.0 whatever
  `conf_level` asked for.

  A median CI puts its bounds a rank distance of `z·√n/2` either side of
  the middle of the sample; as percentile ranks of the data that is
  `p = 0.5 ∓ z/(2·√n)`, and R6/R7 are only the rule turning `p` into a
  position. Fixed to use those ranks, with the value at a fractional
  position interpolated between the two order statistics bracketing it —
  which is also what makes R6 and R7 genuinely distinct rather than two
  names for the same flooring. Now: width falls 0.095 → 0.020 from n=100
  to n=1600 and matches the asymptotic `2·z·√(π/2)·s/√n` to within 5 %;
  empirical coverage 0.890–0.961 across n=25–200 at nominal 0.90/0.95.

  `achieved_coverage` was also over-reported by the half-unit continuity
  correction. Re-centring the normal approximation on `(n+1)/2` cancels
  both shifts, so the bounds go in unadjusted: 0.886 against an exact
  binomial 0.892 at n=25, 0.898 against 0.896 at n=200.

  This branch is what `vei_analysis` uses, following its
  `percentile_method`. `ci_mode='distribution'` is unaffected — it uses
  `median_ci_rank` (Appendix D.2's conservative integer-rank form) above
  30 sales and `median_ci_clapp` below.

* **`prb`'s confidence interval was pinned to `1 - req_significance`.**
  The significance threshold and the interval width are two different
  questions, but one number set both, so a study asking for 90 %
  intervals got a 95 % `PRB_ci` — which is why the column was hidden
  whenever the bootstrap was not filling it. `prb` gained `ci_conf_level`
  (default None = the old behaviour, so nothing moves at the default
  `ci_conf_level=0.95` / `req_significance=0.05`), `ratios` passes the study's
  level down, and the column is now shown whenever the bootstrap is not
  already filling it. The p-value test and the non-significance marker
  are untouched by `ci_conf_level`.

* **The VEI confidence level fell back to `0.1` — a 10 % interval — in
  the bootstrap path.** `_ratios_calculator_wrap` defaulted
  `vei_conf_level` to `0.1` when the key was absent from its input dict,
  and the docstring documented that value, so the mistake had been
  propagated rather than caught. It is a confidence level, not a
  significance level; every other default in the package spells it
  `0.9`. Now `0.9`, and pinned by a test that checks the default across
  `ratios`, `ratios_report`, `rts_calculator` and
  `RatioStudyConfig`.

* **`compute_median_ci(method='binom')` under-reported its own
  coverage.** The interval returned is `[x_(j), x_(k)]` for 1-indexed
  ranks j and k, whose exact coverage is `P(j <= B <= k-1)` for
  `B ~ Bin(n, 0.5)`. `achieved_coverage` was computed from `k_hi`
  instead of from the rank actually handed back — one rank narrower than
  the interval — so a genuinely 96.5 % interval at n=15 was reported as
  92.3 % (n=31: 93.0 % reported as 89.0 %; n=101: 92.9 % as 90.8 %).
  Now derived from the returned ranks, and checked both against the
  closed form and against a simulated coverage experiment.

* **`ratios_report(pdf_args={'gen_pdf_report': False})` generated a PDF
  anyway.** The gate checked `'gen_pdf_report' in pdf_args` — key
  *presence* — but the key is always present after the defaults merge
  (`std_pdf_args = {'gen_pdf_report': True, ...}`), so a report was
  written and added to `exported_files` regardless of what the flag was
  set to. Now checks the value.

* **A compliant VEI was reported as non-compliant on every row.**
  `_add_comments` tested the verdict with `value != 'Compliant'`, but
  `vei_analysis` reports `'Acceptable'`, `'Regressivity'` or
  `'Progressivity'` — never `'Compliant'`. So every ratio study with a
  compliant VEI carried the self-contradicting comment *"Vertical Equity
  Indicator is not acceptable: Acceptable."*, with the VEI value marked
  non-compliant, on every row. Only the two inequity verdicts are findings
  now; `'Sample size too small'` is the absence of a verdict and is left to
  the small-sample comment.

  The same fictional word appeared twice more in `graphs.py`. The Vertical
  Equity Panel's VEI gauge prints the test's verdict underneath only when it
  *disagrees* with where the needle sits, and compared against
  `'Compliant'` — so every compliant panel carried a redundant
  *"VEI test: Acceptable"* note. And `vei_boxplot`'s short title for a
  compliant result could never fire.

  The verdict vocabulary is now published as `VEI_ACCEPTABLE`,
  `VEI_INEQUITY_CONCLUSIONS`, `VEI_SMALL_SAMPLE` and `VEI_CONCLUSIONS` in
  `indicators.py`, and the reporting test asserts producer and consumer
  agree across the whole set — the original bug survived because each side
  was only ever tested alone.

* **The VEI called a large, precisely estimated inequity 'Acceptable'.**
  The verdict needs a point estimate past `acceptable_range` *and* a
  confidence interval that stays past it. The second half compared
  `vei_sig` — which held `vei_ci_upper - vei_ci_lower`, the interval's
  *width* — against the threshold, so the test effectively read "a wide
  interval means inequity": the more precisely an inequity was measured, the
  more likely it was passed.

  A VEI of -45.8% with a 90% interval of (-49.7, -41.8) — as unambiguous as
  regressivity gets — was concluded `'Acceptable'`, because that interval is
  only 7.9 points wide against a 10-point threshold. Meanwhile a noisy
  estimate straddling zero could be flagged.

  `vei_sig` now holds the smallest inequity the interval supports (the bound
  nearest zero, or zero when it straddles), so
  `vei_sig > acceptable_range` asks whether the whole interval clears the
  threshold. That is the same rule `log_median_diff_ci` already reported as
  `flag_min_effect` and nothing read; a test pins the two implementations to
  each other across 1080 tilt/sample-size combinations. **Published VEI
  verdicts will change** — real inequities that were passing are now named.

* **BCa intervals bias-corrected a negative PRB against a positive point
  estimate.**  An insignificant PRB reaches `_bootstrap_analysis` as a
  marked string (`'-0.05896*'`), and the parse that stripped the marker
  used `re.findall(r'\d+\.\d*', ...)`, which dropped the minus sign. The
  BCa bias correction then counted resamples above `+0.05896` for a
  statistic centred on `-0.05896`, pushing the interval far off. Same
  class of bug as the `compare_ratios` one below, in the other direction.

* **`compare_ratios` dropped the sign of a formatted PRB.**  Cleaning
  the significance marker off a value like `'-0.0500*'` used
  `re.sub(r'[^\d.%]', '')`, which deleted the minus sign along with the
  marker, so a PRB moving from -0.05 to -0.10 -- regressivity getting
  worse -- was reported as `0.05↑`, an improvement. It now reads the
  number with the shared marker parser: the marker still goes (it has to,
  for the value to be differenced) but the sign and any percent
  formatting survive, and an unreadable value becomes NaN instead of
  raising.

* **The weighted median returned the lower of two central values.**  On
  an even-length sample an exact tie in cumulative weight was resolved by
  whichever side `searchsorted` happened to land on, so uniform weights
  did not reproduce `numpy.median`. Ties at the midpoint now average the
  straddling pair.

* **Weighted `mae` and `mape` divided by the sample size instead of the
  weight total**, so both were correct only when the weights happened to
  average 1 -- true for `ratios`, which rescales, but not for
  `rts_calculator` or for a bootstrap resample of an already-rescaled
  column.

* `weights` is now keyword-only on the indicators that take a positional
  slot before it (`cod`, `prb`, `rmse`, ...): `cod(asr, weights)` used to
  bind the weight array to `asr_md` and return a plausible wrong number
  in silence, and now raises.

* **`_add_comments` crashed on any row it flagged for significance.**
  The list of flagged columns was written to its working column as a
  list while the marker pass split it as a string, so a row with a
  p-value above `req_significance`, or an indicator without a standard
  whose bootstrap CI spanned zero, died with `AttributeError: 'list'
  object has no attribute 'split'` — reachable with
  `descriptive=True` plus gini p-values or bootstrap CIs.

* **`_add_comments` never reported a non-significant PRB.**  The check
  compared the last *two* characters of the value against a
  single-character symbol, and against the marker column rather than
  the marker `prb` actually emits (`'*'`), so the branch could not fire
  — and because it sat in an `elif` chain, a PRB arriving as a marked
  string was not standards-checked either. A marked PRB now gets its
  'not significant at 95%' comment, is restated as a number so it
  formats like the other indicators, and carries the non-compliance
  symbol.

* **`_add_comments` never compared a stratum's median with the overall
  median.**  The band was built from `df.loc[df['Level'] == 'Overall',
  'Median']` — a Series, not a scalar — so every comparison raised
  internally and was swallowed as 'could not check compliance'. Strata
  whose level of assessment sits more than 5% from the overall level
  are now flagged, as the IAAO guidance the code cites intends.

* `_add_comments` appended the non-compliance symbol twice to some
  values: a duplicate guard tested membership of the significance list
  before appending to the standards list.

* The PDF export filled missing values with `'-'` across every column,
  which a boolean column cannot hold; it now leaves the
  `'<indicator>_compliant'` flags alone.

* `ratios` referenced `ve_headers` before assignment whenever `'VEM'`
  was absent from `indicators`, so every such call died with
  `UnboundLocalError` — including `ratios_report` run with its own
  default `indicators` list, which does not contain `'VEM'`.

* `ve_median`'s small-sample guard returned `VE_Median_index` /
  `VE_Median_direction` / `data` while all of its callers read
  `Inequity_severity` / `Inequity_conclusion` / `dataframe`, so
  requesting the `VEM` indicator for any group with fewer than 30 sales
  raised `KeyError` and aborted the whole ratio study.

* **narwhals backend support.**  The package is now backend-agnostic
  across pandas, polars, and PyArrow thanks to
  [narwhals](https://github.com/narwhals-dev/narwhals).

  * Public functions accept any of: `pandas.DataFrame`,
    `polars.DataFrame`, `pyarrow.Table`, a narwhals-wrapped frame,
    or a 1D value (numpy / pandas / polars / narwhals Series, or a
    plain list).
  * Results from `ratios`, `rts_calculator`, `vei_analysis`,
    `ve_median`, and the public top-level scalar indicators
    (`asr_median`, `cod`, `prd`, …) are returned in the **same
    backend** as the input where feasible.
  * `load_sample_dataset()` now accepts a ``backend`` argument:
    ``"pandas"`` (default), ``"polars"``, or ``"pyarrow"``.

* New `ratio_study.nw_utils` submodule exposing the boundary helpers
  used internally: `to_numpy_1d`, `to_narwhals`, `to_native`,
  `BackendTracker`, `require_pandas`, `frame_columns`, `has_columns`,
  `get_series`, `select_columns`, `native_backend`, plus the
  `narwhals` symbol re-exported for convenience.

* **`import ratio_study` used to silently kill inline plotting in
  Jupyter.** three modules (`gini.py`, `reporting.py` and another) each called
  `matplotlib.use("Agg")` at module import time, and `__init__` imports
  all three. So a notebook that ran `%matplotlib inline`, then
  `import ratio_study`, then plotted with its own `matplotlib` code, got
  every subsequent `plt.show()` turned into a no-op — the figure was
  built but never displayed, with no error. The three modules only ever
  needed Agg for their own headless PNG/PDF export, not as the process
  backend forced on every caller.

  Replaced with `ratio_study.helpers._use_agg_if_headless`, called
  locally inside the functions that actually render a chart to disk
  (matching the pattern already used in `mcp_server.py`). It switches to
  Agg only when nothing display-capable is active yet — checking for a
  running IPython/Jupyter kernel and for an already-selected headless or
  inline backend — so it stays out of the way of a notebook's own
  backend while still guaranteeing headless CI, script and MCP-server use
  can render without a display.

### Changed

* **`ci_conf_level` now defaults to 90 %, two-tailed, instead of 95 %.**
  90 % is the level the Standard on Ratio Studies uses to decide
  compliance (s.11.1.2, s.13.6.1.3) and the level its own demonstration
  report is printed at (Table 10.2). It is also what `vei_conf_level` and
  `vem_conf_level` already defaulted to, so a default study used to
  publish a 90 % VEI interval beside 95 % intervals on everything else --
  reproducing Table 10.2 meant knowing to pass `ci_conf_level=90`. The
  same default now applies to `BootstrapConfig.confint_level` in
  `presets`. The standalone helpers (`mean_ci`, `median_ci_rank`,
  `weighted_mean_ci`, `cod_ci`, `prd_ci`, ...) keep their own documented
  `conf_level=0.95` default; pass the level explicitly when comparing
  them against a study.

* **All three measures of valuation level are in the default indicator
  set.** `DEFAULT_INDICATORS` gained `Mean` and `Weighted_Mean`, which the
  Standard asks a published study to report along with their confidence
  intervals (s.10.1.2). Because the CI indicator list is intersected with
  the study's own indicators, leaving them out also cost their intervals:
  a default distribution-mode study had no `Mean_ci` or
  `Weighted_Mean_ci` despite Appendix D giving a formula for each. A plain
  `ratios()` call now reproduces all six intervals of Table 10.2.

  Relatedly, when the caller names `ci_indicators` explicitly in
  distribution mode, an indicator the study does not compute is now added
  to the study rather than silently losing its interval -- each closed
  form is one pass over the ratios. Bootstrap mode still drops it with a
  message, and a *default* CI indicator list is still intersected, so a
  deliberately narrow `indicators` list keeps the results table narrow.

* **`prd_ci` is exported from the package root**, alongside the other
  Appendix D interval helpers.

* **`ci_level` renamed to `ci_conf_level`, and `confidence_intervals` to
  `confidence_intervals_aggs`.** Same behaviour, clearer names — the old
  ones read like they set a significance level and a computation
  respectively, not a confidence level and an aggregation switch. Applies
  everywhere they appeared: `ratios`, `ratios_report`, `prb`,
  `normalize_ci_args`, and the MCP `run_ratio_study` /
  `generate_ratio_report` tools. `confidence_intervals_aggs` also accepts
  `'all'` as a synonym for `'both'`.

* **`ci_mode` now defaults to `'distribution'` instead of `'bootstrap'`.**
  A plain `ratios()` call now builds `Median_ci` and `CoD_ci` from the
  IAAO Exposure Draft's closed-form formulas rather than by resampling,
  and carries no minimum-sample gate. `PRD` and `PRB` do not have a
  distribution-mode formula, so their `_ci` columns are only populated
  under `ci_mode='bootstrap'`. Pass `ci_mode='bootstrap'` to keep the old
  default behaviour.

* **Overall bootstrap confidence intervals are on by default, for Median,
  CoD, PRD and PRB.**  `bootstrap_cis_for_population` now defaults to True
  and `bootstrap_indicators` to `['Median', 'CoD', 'PRD', 'PRB']` (PRB is
  new to that list), so a plain `ratios()` call returns `Median_ci`,
  `CoD_ci`, `PRD_ci` and `PRB_ci` alongside the point estimates. The list
  is intersected with the study's own `indicators`, so a narrower study
  does not pay for intervals it cannot show.

  `descriptive=True` reads those intervals: a point estimate outside an
  assessment standard whose interval still reaches the standard is no
  longer flagged, which is the check the IAAO standard actually asks for.

  It is not free — roughly 0.16 s on 600 sales and 1.5 s on 20 000 on top
  of the point estimates, spread fairly evenly across the four indicators
  now that PRB has been sped up. `confidence_intervals_aggs=False` restores the
  old speed.

* **Only the bootstrapped indicators get a `_ci` column.**  The column
  list was previously every indicator that *could* have an interval.
  `PRB_ci` was the one that showed up regardless, filled from the PRB
  slope regression at `req_significance` rather than at the bootstrap's
  level — so asking for 90 % intervals put a 95 % `PRB_ci` next to them.

* **Both panel charts are titled with their own result.**  The VEI
  boxplot reads `Vertical Equity Indicator: -35.0%, Regressivity` and the
  VEM chart `Vertical Equity: Regressivity, severity: 15.8%, prevalence:
  44.0%`, so each chart states its verdict rather than describing its
  construction. Whichever part is unavailable is dropped, and a VEI that
  bailed out on sample size reports that instead of its placeholder 0.

  A compliant VEM carries its conclusion alone (`Vertical Equity: no
  significant vertical inequity`): severity and prevalence are nil by
  construction there, so printing them said nothing.

  With `include_vem=False` — no VEM verdict anywhere on the panel — the
  VEM chart is titled `Moving Median of ASR` instead. Either way the
  smoothing is named on the moving-median legend entry (`Moving median
  ASR (window 120)`) rather than in the title, since it describes that
  line and not the chart as a whole.

* The vertical equity panel leaves more room between the VEI and VEM
  charts, so the VEI tick labels and axis title clear the VEM title.

* The VEM chart's legend is drawn at 60% of the tick size, with padding
  and spacing scaled to match: its six entries covered about a quarter
  of the plot at full size, and now take roughly 6%. The size is
  overridable through the new `legend_fontsize` argument on the axes
  helper.

* **`_add_comments` rewritten.**  Same three passes (check, format,
  mark) and the same comment wording, but the eight-deep `if` tree
  became one early-`continue` branch per rule, and the standards
  lookup, marker parsing, small-sample text, direction reading and
  compliance flags moved into documented helpers
  (`_standard_for`, `_standard_band`, `_split_marker`,
  `_small_sample_comment`, `_vertical_equity_direction`,
  `_compliance_binaries`). Statement count in the function dropped from
  106 to 82 while gaining `add_binaries`; nesting depth went from 18 to
  8. The `temp_sig` / `temp_standards` / `temp_small_sample` working
  columns are gone, the input frame is no longer mutated, and row access
  is by index label rather than assuming a 0..n-1 index.

* **The VEM charts now read in ASR units instead of around zero.**  `vem`
  still centres the ASR on the jurisdiction-wide median to compute
  `Inequity_severity`, `Inequity_prevalence` and `Inequity_conclusion` —
  that part is unchanged — but the moving median, its CI band and the
  acceptable-inequity bands are shifted back by the median before they
  are drawn.  Both `ve_scat_median_plot` and the panel's VEM axes
  therefore show the ASR around its median (typically, but not
  necessarily, around 1) rather than around 0, which puts the VEM chart
  on the same scale as the VEI boxplot above it in the panel.

  * `vem` returns the offset it used as a new `'asr_center'` key, and
    both charts take an optional `center` argument.  When it is omitted
    the offset is recovered from the frame's `asr_raw` column, so
    passing a `vem` frame straight to either chart also gives ASR units.
  * The reference line sits at the median ASR and is labelled with its
    value; the red dashed bands sit at the median ± the acceptable
    inequity level; the y-axis is labelled `ASR`.
  * `zoom_y` / `vem_zoom_y` is now interpreted in ASR units:
    `True` zooms to the median ± 0.3, and an explicit `(low, high)` pair
    is read as ASR values rather than as offsets from zero.

* **New `smart_zoom` argument on the VEM charts, on by default.**  Sale
  prices are heavily right-skewed and the ratios have outliers at both
  ends, so plotting the full data range spent most of the plot area on a
  handful of points.  With `smart_zoom=True` the x-axis is framed on the
  1st–99th percentile of the market value proxy and the y-axis on 95% of
  the ASR values inside that window — in practice a large zoom on x and a
  small one on y.  The moving median, its CI band and the
  acceptable-inequity lines are always kept in view, so the zoom can
  never hide the statistic the chart is about.

  * Available as `smart_zoom` on `vem` and `graphs.ve_scat_median_plot`,
    and as `vem_smart_zoom` on `vertical_equity_panel`.  Set it False to
    plot the full range and fall back to `zoom_y`.
  * An explicit `zoom_y=(low, high)` still wins; otherwise `smart_zoom`
    takes precedence over `zoom_y=True`.
  * The VEM x-axis is now labelled `Value (log scale)` (`Value` on a
    linear axis) instead of `Market Value Proxy`.

* **Panel layout.**  The two charts are stacked (VEI, then VEM) instead
  of sitting side by side, each spanning the full width, and the figure
  is letter-proportioned in portrait (`figsize=(8.5, 11)`, default was
  `(11.5, 8.8)`).  The panel image is no longer saved with a tight
  bounding box, so it keeps those proportions.

* The panel no longer draws the wedge-colour legend row under the
  gauges: the dial colours repeat on all four gauges and each
  conclusion is spelled out in words beneath its own dial.

* **Gauge read-out spacing.**  The dial, the value, the conclusion and
  the detail lines (VEM's prevalence, VEI's differing verdict and CoD)
  each got their own band of clear space; the lines previously ran
  together as one block.  There is also more room between the panel
  title and the gauge row.  A gauge can now carry two detail lines, so
  `_ve_gauge_specs` returns `'detail'` as a tuple of lines (empty rather
  than None when there are none).

* **The CoD is reported under the VEI gauge**, as `COD: 8.3%`, beneath
  the conclusion (and beneath the differing-verdict note when there is
  one).  It is computed from the ratios the panel was handed, or passed
  in via the new `cod_value` argument -- which `ratios` now does from its
  population figures, so the read-out cannot drift from the ratios table.

* **More value reference points on the VEM chart's x-axis.**  The log
  locator went from 3 to 5 subdivisions per decade (1, 2, 3, 5, 7), so a
  smart-zoomed axis -- typically about one decade wide -- carries 4-5
  labelled points instead of 2-3.

* The panel's VEM chart is now titled `Moving Median (<window>) of ASR`
  — or `Moving Median (<n> quantiles) of ASR` — instead of `VEM – Moving
  Median of the ASR`, so the smoothing behind the line is on the chart.
  `vem` returns the smoothing it used under the new `'window_size'` and
  `'quantiles'` keys (one of the two is None).

* `pyproject.toml`: `narwhals>=1.13` and `polars>=0.20` are now
  declared runtime dependencies.
* `ratio_study/_data.py::load_sample_dataset` gained an optional
  ``backend`` keyword (``"pandas"`` / ``"polars"`` / ``"pyarrow"``).
* `ratio_study/mcp_server.py::_to_jsonable` now serialises polars
  frames, pyarrow tables, and narwhals-wrapped frames in addition to
  pandas.

### Implementation notes

* The package follows a "narwhals-wrapped boundary" approach:
  internal computation is rewritten where it makes sense
  (`nw_utils.to_numpy_1d`, `np.unique`-based one-hot in
  `_simple_regression`, etc.) and a pandas conversion is performed
  at the public boundary of functions that need libraries that
  are not narwhals-aware (seaborn, xlsxwriter, statsmodels
  formula API, statsmodels.weightstats.DescrStatsW).
* `statsmodels.api.OLS` and `scipy.stats` continue to work on
  numpy arrays — no changes needed there.
* No public API renames; legacy pandas-only call sites
  continue to work unchanged.

### Pre-existing caveat

* `gini_decomp` has a pre-existing quirk where it indexes
  ``population_ratios[0]['...']`` on what the calling code in
  `ratio_study.py` actually passes as a flat ``dict``.  This
  behaviour is unchanged by the narwhals migration.
