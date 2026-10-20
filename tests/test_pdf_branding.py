"""Optional PDF branding: logo, watermark, page numbers, cover and index."""
import re

import pytest
from PIL import Image

from ratio_study.reporting import _normalize_pdf_branding, _pdf_output


@pytest.fixture
def images(tmp_path):
    logo = tmp_path / 'logo.png'
    mark = tmp_path / 'mark.png'
    Image.new('RGBA', (300, 100), (200, 30, 30, 255)).save(logo)
    Image.new('RGBA', (400, 300), (30, 30, 200, 255)).save(mark)
    return logo, mark


def _pages(path):
    return int(re.search(rb'/Count (\d+)', path.read_bytes()).group(1))


def _build(tmp_path, name='r', **kw):
    import pandas as pd
    out = tmp_path / f'{name}.pdf'
    table = pd.DataFrame({'Count': [10], 'Median': ['1.00']})
    _pdf_output(str(out), {'Overall': table}, 'm', **kw)
    return out


def test_defaults_are_off():
    assert _normalize_pdf_branding() == {k: None for k in ('logo', 'watermark', 'page_numbers', 'cover', 'index')}


def test_shorthands_and_aliases(images):
    logo, mark = images
    b = _normalize_pdf_branding(str(logo), {'path': str(mark), 'opacity': 0.3}, {'position': 'right', 'format': 'page_x_of_n'}, True, True)
    assert b['logo']['position'] == 'top_left' and b['logo']['pages'] == 'first'
    assert b['watermark']['position'] == 'fill' and b['watermark']['opacity'] == 0.3
    assert b['page_numbers']['position'] == 'bottom_right' and b['page_numbers']['format'] == 'page_of_n'
    assert b['index'] == {'title': None}


@pytest.mark.parametrize('kwargs', [
    {'logo': 'missing.png'},
    {'watermark': {'path': __file__}},
    {'page_numbers': {'position': 'top_left'}},
])
def test_bad_input_raises(kwargs):
    with pytest.raises((ValueError, FileNotFoundError)):
        _normalize_pdf_branding(**kwargs)


def test_watermark_opacity_range(images):
    with pytest.raises(ValueError):
        _normalize_pdf_branding(watermark={'path': str(images[1]), 'opacity': 2})


def test_index_without_cover_is_ignored(images):
    with pytest.warns(UserWarning):
        assert _normalize_pdf_branding(index=True)['index'] is None


def test_full_report_builds_and_numbers_exclude_cover(tmp_path, images):
    logo, mark = images
    out = _build(tmp_path, logo={'path': str(logo), 'pages': 'all', 'position': 'top_center'}
                 , watermark={'path': str(mark), 'position': 'center'}
                 , page_numbers={'format': 'page_of_n'}
                 , cover={'title': 'T', 'authors': 'A', 'logo': str(logo)}, index=True)
    assert _pages(out) == 3          # cover, index, one report page


def test_cover_alone_adds_one_page(tmp_path):
    plain = _build(tmp_path)
    with_cover = _build(tmp_path, 'c', cover={'title': 'T'})
    assert _pages(with_cover) == _pages(plain) + 1
