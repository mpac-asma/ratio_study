"""Shared fixtures for ratio_study tests."""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_prices():
    """100 synthetic sale prices drawn from a log-normal distribution."""
    rng = np.random.default_rng(42)
    return rng.lognormal(mean=12.5, sigma=0.6, size=100)


@pytest.fixture
def sample_assessments(sample_prices):
    """Assessments ≈ prices × uniform(0.90, 1.10) — slight noise."""
    rng = np.random.default_rng(99)
    return sample_prices * rng.uniform(0.90, 1.10, size=len(sample_prices))


@pytest.fixture
def sample_asr(sample_prices, sample_assessments):
    """Assessment-to-Sale Ratio array."""
    return sample_assessments / sample_prices


@pytest.fixture
def sample_weights(sample_prices):
    """Uniform weights with mean ≈ 1."""
    w = np.ones(len(sample_prices))
    return w / w.mean()


@pytest.fixture
def sample_df(sample_prices, sample_assessments):
    """DataFrame with sale_amt, assessment, and a grouping column."""
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "sale_amt": sample_prices,
        "assessment": sample_assessments,
        "propcode": rng.choice(["RES", "COM", "IND"], size=len(sample_prices)),
    })