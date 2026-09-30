"""Tests for the Appendix B sales-chasing screens.

The tests are built around synthetic samples whose chasing status is known
by construction, because that is the only way to check a detector: a screen
that fires on a chased sample is worth nothing unless it stays quiet on a
clean one, and most of the failure modes here are false positives rather
than misses.

``ratio_study.sales_chasing`` needs only numpy/pandas/scipy/statsmodels,
all core dependencies, so nothing is skipped conditionally.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from ratio_study import sales_chasing as sc


# --------------------------------------------------------------------------
# Fixtures: samples whose chasing status is known by construction
# --------------------------------------------------------------------------

@pytest.fixture
def clean_ratios():
    """1,000 honest ratios: centred on 1.00, CoD about 12%."""
    return np.random.default_rng(101).normal(1.0, 0.15, 1000)


@pytest.fixture
def chased_ratios():
    """900 honest ratios plus 100 pinned at exactly 1.00."""
    rng = np.random.default_rng(102)
    return np.concatenate([rng.normal(1.0, 0.15, 900), np.ones(100)])


@pytest.fixture
def roll():
    """A parcel roll with sold/unsold parcels, values, sizes and dates.

    Nothing is chased: sold and unsold parcels are valued the same way, so
    every screen should stay quiet on it. Individual tests mutate a copy to
    introduce one specific distortion at a time.
    """
    rng = np.random.default_rng(103)
    n, n_sold = 1000, 250
    market_value = rng.lognormal(12.6, 0.5, n)
    sold = np.concatenate([np.ones(n_sold, bool), np.zeros(n - n_sold, bool)])
    return pd.DataFrame({
        'market_value': market_value,
        'assessment': market_value * rng.normal(1.0, 0.15, n),
        'sale_price': np.where(sold, market_value * rng.normal(1.0, 0.05, n), np.nan),
        'independent_value': np.where(~sold, market_value * rng.normal(1.0, 0.05, n), np.nan),
        'prior_value': market_value * 0.90,
        'sqft': rng.uniform(900, 2500, n),
        'sold': sold,
        'excluded': False,
        'stratum': rng.choice(['A', 'B'], n),
        'sale_date': pd.to_datetime('2025-01-01') + pd.to_timedelta(rng.integers(0, 720, n), 'D'),
    })


# --------------------------------------------------------------------------
# B.4: Table B.4.1 is reproduced, not approximated
# --------------------------------------------------------------------------

# Every row of Appendix B, Table B.4.1: (minimum achievable CoD, implied
# standard deviation, critical proportion of ratios in percent).
TABLE_B41 = [
    (0.016, 0.0200, 69),
    (0.040, 0.0500, 32),
    (0.050, 0.0625, 26),
    (0.060, 0.0750, 22),
    (0.070, 0.0875, 19),
    (0.080, 0.1000, 16),
    (0.100, 0.1250, 13),
    (0.120, 0.1500, 11),
    (0.140, 0.1750, 10),
    (0.160, 0.2000, 8),
]


@pytest.mark.parametrize('cod, sd, critical', TABLE_B41)
def test_critical_concentration_reproduces_table_b41(cod, sd, critical):
    result = sc.critical_concentration(cod)
    assert result['sd'] == pytest.approx(sd)
    assert result['critical_proportion'] == pytest.approx(critical / 100.0)


@pytest.mark.parametrize('cod, below_098, below_102', [
    (0.016, 0.1587, 0.8413),
    (0.040, 0.3446, 0.6554),
    (0.050, 0.3745, 0.6255),
    (0.080, 0.4207, 0.5793),
    (0.160, 0.4602, 0.5398),
])
def test_table_b41_normal_tail_probabilities(cod, below_098, below_102):
    # Table B.4.1's "expected proportion of ratios below 0.98 / 1.02"
    # columns, which is where the critical proportion comes from.
    from scipy.stats import norm
    sd = sc.critical_concentration(cod)['sd']
    assert norm.cdf((0.98 - 1.0) / sd) == pytest.approx(below_098, abs=5e-5)
    assert norm.cdf((1.02 - 1.0) / sd) == pytest.approx(below_102, abs=5e-5)


def test_critical_concentration_rejects_nonsense_inputs():
    with pytest.raises(ValueError):
        sc.critical_concentration(0)
    with pytest.raises(ValueError):
        sc.critical_concentration(0.04, band=0)


# --------------------------------------------------------------------------
# B.4: sc_concentration
# --------------------------------------------------------------------------

def test_concentration_quiet_on_clean_sample(clean_ratios):
    result = sc.sc_concentration(clean_ratios)
    assert result['flag'] is False
    assert result['observed_proportion'] < result['critical_proportion']


def test_concentration_flags_overfit_uniformity():
    # A CoD of 3% is below anything the standard treats as achievable, so
    # the in-band share has to exceed the 32% critical proportion.
    ratios = np.random.default_rng(104).normal(1.0, 0.04, 1000)
    result = sc.sc_concentration(ratios)
    assert result['flag'] is True
    assert result['observed_proportion'] > result['critical_proportion']
    assert result['p_value'] < 0.05


def test_concentration_is_one_sided_not_two():
    # A wildly dispersed sample is a uniformity problem, not a chasing one.
    # assesspy's `dist` method takes an absolute difference and so fires
    # here; this must not.
    ratios = np.random.default_rng(105).normal(1.0, 0.60, 1000)
    result = sc.sc_concentration(ratios)
    assert result['flag'] is False
    assert result['observed_proportion'] < result['expected_proportion']


def test_concentration_window_follows_the_level_of_assessment():
    # A stratum assessed at 0.85. The window must sit on the sample's own
    # centre, not on 1.00, or the test has nothing to measure.
    ratios = np.random.default_rng(106).normal(0.85, 0.04, 1000)
    result = sc.sc_concentration(ratios, center='median')
    assert result['center_value'] == pytest.approx(0.85, abs=0.01)
    assert result['flag'] is True

    # Pinned to unity, the same sample looks all but empty -- 0.98-1.02 is
    # nearly four standard deviations from where these ratios actually sit.
    at_unity = sc.sc_concentration(ratios, center='unity')
    assert at_unity['observed_proportion'] < 0.01
    assert at_unity['flag'] is False


def test_concentration_mode_scan_sees_a_secondary_pile():
    # Honest ratios at 0.90 with a chased pile at 1.00: the centre looks
    # fine, but the fullest window in the sample is not at the centre.
    rng = np.random.default_rng(107)
    ratios = np.concatenate([rng.normal(0.90, 0.10, 600), np.full(400, 1.0)])
    result = sc.sc_concentration(ratios, scan_modes=True)
    assert result['max_window_center'] == pytest.approx(1.0, abs=0.02)
    assert result['max_window_proportion'] > result['observed_proportion']


def test_concentration_respects_the_thirty_sale_floor():
    result = sc.sc_concentration(np.random.default_rng(108).normal(1.0, 0.15, 20))
    assert result['flag'] is None
    assert result['conclusion'] == 'Sample size too small'


def test_concentration_drops_non_finite_values():
    ratios = np.concatenate([np.random.default_rng(109).normal(1.0, 0.15, 100),
                             [np.nan, np.inf, -np.inf]])
    result = sc.sc_concentration(ratios)
    assert result['n'] == 100
    assert result['n_dropped'] == 3


def test_concentration_raising_min_achievable_cod_lowers_the_threshold():
    # Table B.4.1's point: a stratum that could never reach 4% is judged
    # against a lower critical proportion, making the test more sensitive.
    strict = sc.sc_concentration(np.random.default_rng(110).normal(1.0, 0.15, 500),
                                 min_achievable_cod=0.04)
    lenient = sc.sc_concentration(np.random.default_rng(110).normal(1.0, 0.15, 500),
                                  min_achievable_cod=0.10)
    assert lenient['critical_proportion'] < strict['critical_proportion']


# --------------------------------------------------------------------------
# CDF discontinuity screen
# --------------------------------------------------------------------------

def test_cdf_gap_finds_the_pinned_ratio_and_counts_it(chased_ratios):
    result = sc.sc_cdf_gap(chased_ratios)
    assert result['flag'] is True
    assert result['max_gap_at'] == pytest.approx(1.0)
    assert result['max_gap_n'] == 100
    assert result['max_gap'] == pytest.approx(0.10)


def test_cdf_gap_quiet_on_clean_sample(clean_ratios):
    assert sc.sc_cdf_gap(clean_ratios)['flag'] is False


def test_cdf_gap_survives_a_far_outlier():
    # The failure mode of taking the global largest gap and then checking
    # whether it lies in the band: one extreme ratio owns the largest gap
    # and vetoes the real finding.
    rng = np.random.default_rng(111)
    ratios = np.concatenate([rng.normal(1.0, 0.15, 900), np.ones(100), [8.0]])
    result = sc.sc_cdf_gap(ratios)
    assert result['flag'] is True
    assert result['max_gap_at'] == pytest.approx(1.0)


def test_cdf_gap_finds_a_pile_away_from_the_contaminated_median():
    # Honest ratios at 0.90 drag the median toward the pile at 1.00, which
    # lands outside a band centred on the median alone. The union with a
    # band at unity is what saves it.
    rng = np.random.default_rng(112)
    ratios = np.concatenate([rng.normal(0.90, 0.10, 800), np.full(200, 1.0)])
    assert sc.sc_cdf_gap(ratios)['flag'] is True
    assert sc.sc_cdf_gap(ratios, center='median')['flag'] is False


def test_cdf_gap_reports_top_gaps_in_descending_order():
    rng = np.random.default_rng(113)
    ratios = np.concatenate([rng.normal(1.0, 0.15, 800), np.full(120, 1.0), np.full(80, 1.01)])
    gaps = [gap for _, gap, _ in sc.sc_cdf_gap(ratios, top_k=3)['top_gaps']]
    assert gaps == sorted(gaps, reverse=True)


def test_cdf_gap_respects_the_thirty_sale_floor():
    assert sc.sc_cdf_gap(np.ones(20))['flag'] is None


# --------------------------------------------------------------------------
# B.1: sc_value_change
# --------------------------------------------------------------------------

def _with_value_change(roll, sold_change, unsold_change, noise=0.03, seed=200):
    rng = np.random.default_rng(seed)
    change = np.where(roll['sold'],
                      rng.normal(sold_change, noise, len(roll)),
                      rng.normal(unsold_change, noise, len(roll)))
    return roll.assign(current_value=roll['prior_value'] * (1 + change))


def test_value_change_flags_a_gap_beyond_tolerance(roll):
    # The appendix's own example: 10% on sold, 4% on unsold.
    frame = _with_value_change(roll, 0.10, 0.04)
    result = sc.sc_value_change(frame, 'sold', 'prior_value', 'current_value')
    assert result['flag'] is True
    assert result['statistic'] == 'median'
    assert result['difference'] == pytest.approx(0.06, abs=0.01)
    assert result['tolerance'] == 0.05


def test_value_change_significant_but_within_tolerance_is_not_a_finding(roll):
    # 5.0% vs 4.5% on a tight distribution: comfortably significant, and
    # meaningless. This is the case the appendix warns about explicitly.
    frame = _with_value_change(roll, 0.050, 0.045, noise=0.01)
    result = sc.sc_value_change(frame, 'sold', 'prior_value', 'current_value')
    assert result['statistically_significant'] is True
    assert result['practically_significant'] is False
    assert result['flag'] is False


def test_value_change_tolerance_tightens_below_five_percent(roll):
    small = _with_value_change(roll, 0.04, 0.005)
    assert sc.sc_value_change(small, 'sold', 'prior_value', 'current_value')['tolerance'] == 0.03
    large = _with_value_change(roll, 0.10, 0.04)
    assert sc.sc_value_change(large, 'sold', 'prior_value', 'current_value')['tolerance'] == 0.05


def test_value_change_switches_to_the_mean_when_unsold_parcels_are_frozen(roll):
    # Every unsold parcel keeps last year's value, so its median change is
    # 0 and tells us nothing about the distribution.
    rng = np.random.default_rng(201)
    frame = roll.assign(current_value=np.where(
        roll['sold'], roll['prior_value'] * (1 + rng.normal(0.06, 0.03, len(roll))),
        roll['prior_value']))
    result = sc.sc_value_change(frame, 'sold', 'prior_value', 'current_value')
    assert result['unsold_pct_no_change'] == pytest.approx(1.0)
    assert result['statistic'] == 'mean'
    assert result['flag'] is True


def test_value_change_honours_the_exclusion_column(roll):
    # Parcels with a physical change bias the test toward finding chasing,
    # which is why the appendix insists they come out first.
    frame = _with_value_change(roll, 0.04, 0.04)
    rebuilt = frame['sold'] & (np.arange(len(frame)) % 2 == 0)
    frame = frame.assign(
        current_value=np.where(rebuilt, frame['prior_value'] * 2.0, frame['current_value']),
        excluded=rebuilt,
    )
    without = sc.sc_value_change(frame, 'sold', 'prior_value', 'current_value')
    with_exclusion = sc.sc_value_change(frame, 'sold', 'prior_value', 'current_value',
                                        exclude_col='excluded')
    assert without['flag'] is True
    assert with_exclusion['flag'] is False
    assert with_exclusion['n_excluded'] == int(rebuilt.sum())


def test_value_change_respects_the_thirty_parcel_floor(roll):
    frame = _with_value_change(roll, 0.10, 0.04).head(40)
    assert sc.sc_value_change(frame, 'sold', 'prior_value', 'current_value')['flag'] is None


def test_value_change_rejects_a_missing_column(roll):
    with pytest.raises(RuntimeError):
        sc.sc_value_change(roll, 'sold', 'prior_value', 'no_such_column')


# --------------------------------------------------------------------------
# B.2: sc_unit_values
# --------------------------------------------------------------------------

def test_unit_values_flags_a_relative_gap_beyond_tolerance(roll):
    frame = roll.assign(assessment=np.where(roll['sold'],
                                            roll['sqft'] * 220 * 1.12,
                                            roll['sqft'] * 220))
    result = sc.sc_unit_values(frame, 'sold', 'assessment', 'sqft')
    assert result['flag'] is True
    assert result['relative_difference'] == pytest.approx(0.12, abs=0.005)


def test_unit_values_tolerance_is_relative_not_in_dollars(roll):
    # The tolerance is a percentage; unit values are in dollars. A 2% gap
    # on a $220/sqft base is $4.40 -- inside tolerance, and it must be
    # judged as 2%, not as 4.4 against 0.05.
    frame = roll.assign(assessment=np.where(roll['sold'],
                                            roll['sqft'] * 220 * 1.02,
                                            roll['sqft'] * 220))
    result = sc.sc_unit_values(frame, 'sold', 'assessment', 'sqft')
    assert result['relative_difference'] == pytest.approx(0.02, abs=0.005)
    assert result['tolerance'] == 0.03
    assert result['flag'] is False


def test_unit_values_drops_non_positive_sizes(roll):
    frame = roll.copy()
    frame.loc[frame.index[:100], 'sqft'] = 0
    result = sc.sc_unit_values(frame, 'sold', 'assessment', 'sqft')
    assert result['n_sold'] + result['n_unsold'] == len(frame) - 100


# --------------------------------------------------------------------------
# B.3: sc_split_sample
# --------------------------------------------------------------------------

def test_split_sample_flags_a_tighter_pre_cutoff_cod(roll):
    # Values could still be changed up to 2025-12-31, and were: sales that
    # closed before it have a CoD of 3%, after it 15%.
    rng = np.random.default_rng(300)
    early = (roll['sale_date'] <= pd.Timestamp('2025-12-31')).to_numpy()
    price = roll['market_value'].to_numpy()
    frame = roll.assign(
        sale_price=price,
        assessment=np.where(early, price * rng.normal(1, 0.03, len(roll)),
                            price * rng.normal(1, 0.15, len(roll))),
    )
    result = sc.sc_split_sample(frame, 'assessment', 'sale_price', 'sale_date', '2025-12-31')
    assert result['flag'] is True
    assert result['cod_known'] < result['cod_unknown']
    assert result['cod_ci_disjoint'] is True
    assert 'post-cutoff sample is the valid one' in result['conclusion']


def test_split_sample_quiet_when_both_halves_match(roll):
    rng = np.random.default_rng(301)
    price = roll['market_value'].to_numpy()
    frame = roll.assign(sale_price=price,
                        assessment=price * rng.normal(1, 0.12, len(roll)))
    assert sc.sc_split_sample(frame, 'assessment', 'sale_price', 'sale_date',
                              '2025-12-31')['flag'] is False


def test_split_sample_does_not_flag_a_worse_early_half(roll):
    # The test is directional: a *looser* pre-cutoff CoD is not chasing.
    rng = np.random.default_rng(302)
    early = (roll['sale_date'] <= pd.Timestamp('2025-12-31')).to_numpy()
    price = roll['market_value'].to_numpy()
    frame = roll.assign(
        sale_price=price,
        assessment=np.where(early, price * rng.normal(1, 0.20, len(roll)),
                            price * rng.normal(1, 0.05, len(roll))),
    )
    assert sc.sc_split_sample(frame, 'assessment', 'sale_price', 'sale_date',
                              '2025-12-31')['flag'] is False


# --------------------------------------------------------------------------
# B.5: sc_mass_appraisal
# --------------------------------------------------------------------------

def test_mass_appraisal_flags_assessments_set_to_the_sale_price(roll):
    # The extreme case: every sold parcel's assessment is its sale price,
    # so the sales study has a CoD of 0 and no usable CoD interval. The
    # dispersion test has to carry the finding here.
    frame = roll.assign(assessment=np.where(roll['sold'], roll['sale_price'],
                                            roll['assessment']))
    result = sc.sc_mass_appraisal(frame, 'assessment', 'sale_price',
                                  'independent_value', sold_col='sold')
    assert result['flag'] is True
    assert result['cod_sales'] == pytest.approx(0.0, abs=1e-9)
    assert not np.isfinite(result['cod_ci_sales']).all()
    assert result['dispersion_p'] < 0.05
    # Location is untouched by chasing to a level that was already 1.00,
    # so the level test must stay quiet and the CoD leg must not need it.
    assert result['mannwhitney_p'] > 0.05


def test_mass_appraisal_quiet_when_sold_and_unsold_are_valued_alike(roll):
    result = sc.sc_mass_appraisal(roll, 'assessment', 'sale_price',
                                  'independent_value', sold_col='sold')
    assert result['flag'] is False
    assert result['conclusion'] == 'No evidence of sales chasing'


def test_mass_appraisal_is_one_sided(roll):
    # A sales study that looks *worse* than the independent one is not
    # sales chasing, however large the gap.
    rng = np.random.default_rng(400)
    frame = roll.assign(assessment=np.where(
        roll['sold'], roll['market_value'] * rng.normal(1, 0.30, len(roll)),
        roll['market_value'] * rng.normal(1, 0.10, len(roll))))
    result = sc.sc_mass_appraisal(frame, 'assessment', 'sale_price',
                                  'independent_value', sold_col='sold')
    assert result['cod_sales'] > result['cod_independent']
    assert result['flag'] is False


def test_mass_appraisal_detects_partial_chasing(roll):
    rng = np.random.default_rng(401)
    chased = roll['sold'].to_numpy() & (rng.random(len(roll)) < 0.7)
    frame = roll.assign(assessment=np.where(chased, roll['sale_price'], roll['assessment']))
    result = sc.sc_mass_appraisal(frame, 'assessment', 'sale_price',
                                  'independent_value', sold_col='sold')
    assert result['flag'] is True
    assert abs(result['cod_difference']) > 0.05


def test_mass_appraisal_tolerance_is_what_gates_a_borderline_cod_gap(roll):
    # Half the sold parcels chased puts the CoD gap at 4.9 percentage
    # points -- just inside the appendix's 5-point tolerance, so the
    # default stays quiet. That is the tolerance working, not a miss, and
    # it is the caller's dial: at 3 points the same data is a finding.
    rng = np.random.default_rng(402)
    chased = roll['sold'].to_numpy() & (rng.random(len(roll)) < 0.5)
    frame = roll.assign(assessment=np.where(chased, roll['sale_price'], roll['assessment']))
    kwargs = dict(sold_col='sold')
    default = sc.sc_mass_appraisal(frame, 'assessment', 'sale_price',
                                   'independent_value', **kwargs)
    tightened = sc.sc_mass_appraisal(frame, 'assessment', 'sale_price',
                                     'independent_value', tolerance=0.03, **kwargs)
    assert 0.03 < abs(default['cod_difference']) < 0.05
    assert default['cod_ci_disjoint'] is True
    assert default['flag'] is False
    assert tightened['flag'] is True


def test_mass_appraisal_falls_back_to_sale_price_as_sold(roll):
    # Without `sold_col`, a usable sale price means sold, and the two
    # samples stay disjoint.
    with_col = sc.sc_mass_appraisal(roll, 'assessment', 'sale_price',
                                    'independent_value', sold_col='sold')
    without_col = sc.sc_mass_appraisal(roll, 'assessment', 'sale_price',
                                       'independent_value')
    assert without_col['n_sales'] == with_col['n_sales']
    assert without_col['n_independent'] == with_col['n_independent']


def test_mass_appraisal_reports_acceptance_against_standards(roll):
    standards = {'Median': (0.90, 1.10), 'CoD': (0.05, 0.15)}
    result = sc.sc_mass_appraisal(roll, 'assessment', 'sale_price',
                                  'independent_value', sold_col='sold',
                                  standards=standards)
    assert result['sales_median_acceptable'] is True
    assert result['independent_median_acceptable'] is True
    assert result['cod_standard'] == (0.05, 0.15)


def test_mass_appraisal_respects_the_thirty_parcel_floor(roll):
    result = sc.sc_mass_appraisal(roll.head(40), 'assessment', 'sale_price',
                                  'independent_value', sold_col='sold')
    assert result['flag'] is None
    assert result['conclusion'] == 'Sample size too small'


def test_mass_appraisal_does_not_model_anything(roll):
    # The independent values are the caller's. Blank the column and the
    # function must decline, not substitute an estimate of its own.
    frame = roll.assign(independent_value=np.nan)
    result = sc.sc_mass_appraisal(frame, 'assessment', 'sale_price',
                                  'independent_value', sold_col='sold')
    assert result['n_independent'] == 0
    assert result['flag'] is None


# --------------------------------------------------------------------------
# B.6: sc_independent_values
# --------------------------------------------------------------------------

def _proxy_roll(roll, proxy_level, seed=500):
    """Independent values implying a ratio level of ``proxy_level``."""
    rng = np.random.default_rng(seed)
    n = len(roll)
    covered = rng.random(n) < 0.4
    market_value = roll['market_value'].to_numpy()
    return roll.assign(
        assessment=market_value * rng.normal(1, 0.10, n),
        sale_price=market_value * rng.normal(1, 0.05, n),
        independent_value=np.where(
            covered, market_value / proxy_level * rng.normal(1, 0.10, n), np.nan),
    )


@pytest.mark.parametrize('proxy_level', [0.85, 1.15])
def test_independent_values_is_two_sided(roll, proxy_level):
    # Either side can be the distorted one, so a material gap in either
    # direction is a finding.
    result = sc.sc_independent_values(_proxy_roll(roll, proxy_level),
                                      'assessment', 'sale_price', 'independent_value')
    assert result['flag'] is True
    assert 'investigation is needed to establish which' in result['conclusion']


def test_independent_values_quiet_when_the_two_sources_agree(roll):
    result = sc.sc_independent_values(_proxy_roll(roll, 1.0),
                                      'assessment', 'sale_price', 'independent_value')
    assert result['flag'] is False


def test_independent_values_does_not_flag_a_small_cod_gap(roll):
    # Independent values carry estimation error that lands in the
    # denominator and inflates their study's CoD. A couple of percentage
    # points of that artefact must not read as chasing -- which it would
    # under a *relative* CoD tolerance, where 9% vs 11% is a 20% gap.
    result = sc.sc_independent_values(_proxy_roll(roll, 1.0),
                                      'assessment', 'sale_price', 'independent_value')
    assert abs(result['cod_difference']) < 0.05
    assert abs(result['cod_relative_gap']) > 0.05
    assert result['flag'] is False


def test_independent_values_reports_overlap(roll):
    # Unlike B.5 the two groups may share parcels, which is the caveat on
    # the p-value and so has to be visible.
    frame = _proxy_roll(roll, 1.0)
    result = sc.sc_independent_values(frame, 'assessment', 'sale_price', 'independent_value')
    assert result['n_overlap'] == result['n_independent']


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _full_roll(roll):
    return _with_value_change(roll, 0.10, 0.04).assign(
        sale_price=roll['market_value'] * 1.0,
    )


def test_detect_runs_every_technique_when_the_data_supports_it(roll):
    frame = _full_roll(roll)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = sc.detect_sales_chasing(
            frame, assessment_col='assessment', sale_price_col='sale_price',
            sold_col='sold', prior_value_col='prior_value',
            current_value_col='current_value', size_col='sqft',
            sale_date_col='sale_date', cutoff_date='2025-12-31',
            independent_value_col='independent_value', independent_technique='both',
        )
    assert set(out['reference']) == {'B.1', 'B.2', 'B.3', 'B.4', 'B.5', 'B.6', '--'}
    assert not out['conclusion'].str.startswith('not run').any()


def test_detect_names_what_it_could_not_run():
    # "We did not test for it" and "we tested and found nothing" are
    # different findings and must not look alike in the output.
    out = sc.detect_sales_chasing(asr=np.random.default_rng(600).normal(1, 0.15, 500))
    ran = out.loc[~out['conclusion'].str.startswith('not run'), 'reference']
    skipped = out.loc[out['conclusion'].str.startswith('not run'), 'reference']
    assert set(ran) == {'B.4', '--'}
    assert set(skipped) == {'B.1', 'B.2', 'B.3', 'B.5 / B.6'}
    assert out.loc[out['conclusion'].str.startswith('not run'), 'flag'].isna().all()


def test_detect_stratifies_every_technique(roll):
    frame = _full_roll(roll)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = sc.detect_sales_chasing(
            frame, assessment_col='assessment', sale_price_col='sale_price',
            sold_col='sold', prior_value_col='prior_value',
            current_value_col='current_value', group_by='stratum',
        )
    assert set(out['scope']) == {'overall', 'A', 'B'}
    # Each technique that ran did so on all three scopes.
    ran = out.loc[~out['conclusion'].str.startswith('not run')]
    assert (ran.groupby('technique')['scope'].nunique() == 3).all()


def test_detect_auto_picks_b5_with_sold_col_and_b6_without(roll):
    frame = _full_roll(roll)
    kwargs = dict(assessment_col='assessment', sale_price_col='sale_price',
                  independent_value_col='independent_value')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with_sold = sc.detect_sales_chasing(frame, sold_col='sold', **kwargs)
        without_sold = sc.detect_sales_chasing(frame, **kwargs)
    assert 'B.5' in set(with_sold['reference']) and 'B.6' not in set(with_sold['reference'])
    assert 'B.6' in set(without_sold['reference']) and 'B.5' not in set(without_sold['reference'])


def test_detect_warns_when_a_screen_flags(chased_ratios):
    with pytest.warns(UserWarning, match='sales-chasing screen'):
        sc.detect_sales_chasing(asr=chased_ratios)


def test_detect_rejects_group_by_without_a_frame():
    with pytest.raises(RuntimeError):
        sc.detect_sales_chasing(asr=np.ones(50), group_by='stratum')


def test_detect_rejects_an_unknown_independent_technique(roll):
    with pytest.raises(RuntimeError):
        sc.detect_sales_chasing(roll, assessment_col='assessment',
                                sale_price_col='sale_price',
                                independent_value_col='independent_value',
                                independent_technique='B.7')


def test_detect_accepts_a_polars_frame(roll):
    polars = pytest.importorskip('polars')
    frame = polars.from_pandas(_full_roll(roll))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = sc.detect_sales_chasing(frame, assessment_col='assessment',
                                      sale_price_col='sale_price', sold_col='sold')
    assert isinstance(out, pd.DataFrame)
    assert 'B.4' in set(out['reference'])


def test_public_api_exports_the_screens():
    import ratio_study

    for name in ('detect_sales_chasing', 'sc_concentration', 'sc_cdf_gap',
                 'sc_value_change', 'sc_unit_values', 'sc_split_sample',
                 'sc_mass_appraisal', 'sc_independent_values',
                 'critical_concentration', 'SC_MIN_N'):
        assert name in ratio_study.__all__
        assert hasattr(ratio_study, name)
